import os
import re
import shutil
import json
import numpy as np
import yaml
from scipy.ndimage import median_filter

import massibo_ana.utils.htype as htype
import massibo_ana.utils.custom_exceptions as cuex
from massibo_ana.preprocess.DataPreprocessor import DataPreprocessor
from massibo_ana.preprocess.NumpyDataPreprocessor import NumpyDataPreprocessor


class DewarDataPreprocessor:
    """This class is aimed at handling and preprocessing the gain and dark-noise
    raw data taken with Massibo v2 (i.e. using the cryogenic dewar). In this
    measurement protocol, three motherboards are used per measurement, so that
    up to 9 strips (54 SiPMs) are measured simultaneously. Each set consists
    of up to 3 measurements: the first two contain 9 strips each and the
    optional third one contains the remaining strips. Additionally, only one
    gain measurement is performed per SiPM (at the same operation voltage as
    the dark-noise measurement).

    The input data is organised in a SET folder containing:
        - A 'Set_data.txt' file with the strip-ID-to-channel/socket mapping.
        - MEAS_<i> subfolders, each containing '.npy' files whose filenames
          encode the measurement type (darknoise '_W' or gain '_G'), the
          channel (CH0/CH1/CH2, i.e. the motherboard index) and the SiPM
          index within that channel (0-17, where 0-5 correspond to Socket_1,
          6-11 to Socket_2 and 12-17 to Socket_3)."""

    SIPMS_PER_STRIP = 6
    SOCKETS_PER_CHANNEL = 3
    SIPMS_PER_CHANNEL = SIPMS_PER_STRIP * SOCKETS_PER_CHANNEL  # 18

    # Vertical order of the temperature-sensor columns of the temperature
    # '.npy' file, from the deepest sensor to the shallowest one. This is a
    # property of how the sensor string is wired in the setup, not of the
    # data, so it must not be re-inferred from each file: a badly formed
    # curve yields a meaningless immersion time, and therefore a meaningless
    # ordering. It was calibrated against the 21 well-formed temperature
    # files available, whose measured immersion-time ranking is (1, 3, 2, 4)
    # without exception.
    T_SENSORS_VERTICAL_ORDER = (1, 3, 2, 4)

    # Regex for .npy filenames produced by the Massibo v2 DAQ.
    # Groups: (set_no, meas_no, type_indicator, channel, idx, type_indicator2)
    # The pattern accepts both formats:
    #   - set<N>_meas<N>_..._<type>_CH<ch>_<idx>_<type>.npy (SET_7 format)
    #   - set<N>_<extra_N>_meas<N>_..._<type>_CH<ch>_<idx>_<type>.npy (SET_4 format)
    _FILENAME_REGEX = re.compile(
        r'^set(\d+)(?:_\d+)?_meas(\d+)_.*_(W|G)_CH(\d+)_(\d+)_(W|G)\.npy$',
        re.IGNORECASE
    )

    def __init__(
        self,
        set_folderpath,
        verbose=True,
    ):
        """This class is aimed at handling and preprocessing the gain and
        dark-noise raw data taken with Massibo v2 (i.e. using the cryogenic
        dewar) for one set of SiPMs. This initializer gets the following
        mandatory positional argument

        - set_folderpath (string): Path to the SET folder (e.g.
        'input_data/SET_6'). This folder must contain a 'Set_data.txt'
        file and one or more MEAS_<i> subfolders.

        And the following optional keyword argument:

        - verbose (boolean): Whether to print functioning-related messages.

        This initializer

            1) Parses 'Set_data.txt' to build the mapping
               (meas, channel, socket) -> strip_ID.
            2) Scans every MEAS_<i> subfolder for '.npy' files and
               classifies them into gain and dark-noise candidates
               keyed by (meas, channel, idx).
            3) Cross-references the parsed mappings with the files
               actually found and applies robustness rules so that
               inconsistencies (duplicate strip IDs, missing files,
               unexpected files) are logged as warnings but do not
               abort execution."""

        htype.check_type(
            set_folderpath,
            str,
            exception_message=htype.generate_exception_message(
                "DewarDataPreprocessor.__init__", 90001
            ),
        )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "DewarDataPreprocessor.__init__", 90002
            ),
        )

        if not os.path.isdir(set_folderpath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.__init__",
                    90003,
                    extra_info=f"The folder '{set_folderpath}' does not exist."
                )
            )

        self.__set_folderpath = set_folderpath
        self.__warnings = []

        # ---- 0. Parse set_no from folder name ----
        folder_basename = os.path.basename(os.path.normpath(set_folderpath))
        set_folder_match = re.search(
            r'(^|_)set[_\s]*(\d+)($|_)', folder_basename, re.IGNORECASE
        )
        if set_folder_match:
            self.__set_no = int(set_folder_match.group(2))
        else:
            raise cuex.MalFunction(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.__init__",
                    90005,
                    extra_info=(
                        f"Cannot parse a set number from the folder name "
                        f"'{folder_basename}'. Expected a name matching "
                        f"'SET_<N>' or containing '_SET_<N>' "
                        f"(case-insensitive)."
                    )
                )
            )

        # ---- 1. Parse Set_data.txt ----
        set_data_path = os.path.join(set_folderpath, "Set_data.txt")
        if not os.path.isfile(set_data_path):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.__init__",
                    90004,
                    extra_info=(
                        f"'Set_data.txt' not found in '{set_folderpath}'."
                    )
                )
            )

        (
            _,  # set_no from Set_data.txt is ignored; we use the folder name
            self.__meas_strip_map,
            self.__meas_vop_map
        ) = DewarDataPreprocessor._parse_set_data(set_data_path)

        if verbose:
            print(
                f"Parsed Set_data.txt: set_no={self.__set_no} "
                f"(from folder name), "
                f"{len(self.__meas_strip_map)} measurement block(s) found."
            )

        # ---- 2. Detect duplicate strip IDs ----
        # We distinguish two cases:
        #   a) Intra-meas duplicate: the same strip ID appears twice in
        #      the *same* measurement. This is physically impossible
        #      (one strip cannot be on two sockets at once), so we
        #      invalidate *all* intra-meas occurrences.
        #   b) Inter-meas duplicate: a strip was re-measured in a later
        #      measurement. We keep only the data from the latest
        #      measurement (highest meas number) and discard the
        #      earlier one(s), under the assumption that a measurement
        #      was repeated because the previous one was defective.

        # Group all positions by strip_id
        strip_id_locations = {}  # strip_id -> list of (meas, channel, socket)
        for (meas, channel, socket), strip_id in self.__meas_strip_map.items():
            strip_id_locations.setdefault(strip_id, []).append(
                (meas, channel, socket)
            )

        self.__invalid_positions = set()
        self.__superseded_positions = set()  # earlier inter-meas occurrences

        for strip_id, locations in strip_id_locations.items():
            if len(locations) <= 1:
                continue

            # Group by meas number to separate intra- vs inter-meas
            by_meas = {}  # meas -> list of (meas, channel, socket)
            for loc in locations:
                by_meas.setdefault(loc[0], []).append(loc)

            # Check for intra-meas duplicates (same meas, >1 position)
            has_intra_dup = False
            for meas_no, locs_in_meas in by_meas.items():
                if len(locs_in_meas) > 1:
                    has_intra_dup = True
                    warning = (
                        f"Intra-meas duplicate: strip ID {strip_id} "
                        f"appears {len(locs_in_meas)} times in "
                        f"Meas_{meas_no} at positions {locs_in_meas}. "
                        f"All intra-meas occurrences will be "
                        f"invalidated."
                    )
                    self.__warnings.append(warning)
                    for loc in locs_in_meas:
                        self.__invalid_positions.add(loc)

            if has_intra_dup:
                # If any meas has an intra-meas duplicate for this
                # strip, invalidate ALL occurrences of this strip
                # (we cannot trust any of them).
                for loc in locations:
                    self.__invalid_positions.add(loc)
                continue

            # Pure inter-meas duplicate: keep only the latest meas
            if len(by_meas) > 1:
                sorted_meas = sorted(by_meas.keys())
                latest_meas = sorted_meas[-1]
                earlier_meas = sorted_meas[:-1]
                warning = (
                    f"Inter-meas duplicate: strip ID {strip_id} was "
                    f"measured in Meas {sorted_meas}. Keeping only "
                    f"the data from Meas_{latest_meas} (the latest)."
                )
                self.__warnings.append(warning)
                for m in earlier_meas:
                    for loc in by_meas[m]:
                        self.__superseded_positions.add(loc)

        # Build the valid strip map (excluding intra-meas duplicates
        # and superseded inter-meas positions)
        excluded = self.__invalid_positions | self.__superseded_positions
        self.__valid_strip_map = {
            k: v
            for k, v in self.__meas_strip_map.items()
            if k not in excluded
        }

        # ---- 3. Scan MEAS_<i> subfolders for .npy files ----
        # gain_candidates[(meas, channel, idx)] = filepath
        # darknoise_candidates[(meas, channel, idx)] = filepath
        self.__gain_candidates = {}
        self.__darknoise_candidates = {}
        self.__unmatched_npy_files = []

        meas_folders = sorted([
            d for d in os.listdir(set_folderpath)
            if os.path.isdir(os.path.join(set_folderpath, d))
            and re.match(r'^MEAS_\d+$', d, re.IGNORECASE)
        ])

        for meas_folder in meas_folders:
            meas_path = os.path.join(set_folderpath, meas_folder)
            for filename in os.listdir(meas_path):
                filepath = os.path.join(meas_path, filename)
                if not os.path.isfile(filepath):
                    continue

                if not filename.lower().endswith('.npy'):
                    continue

                match = DewarDataPreprocessor._FILENAME_REGEX.match(filename)
                if match is None:
                    self.__unmatched_npy_files.append(filename)
                    continue

                _, meas_str, type1, ch_str, idx_str, type2 = match.groups()
                meas_no = int(meas_str)
                channel = int(ch_str)
                idx = int(idx_str)

                if type1 != type2:
                    warning = (
                        f"Filename '{filename}' has inconsistent type "
                        f"indicators ('{type1}' vs '{type2}'). Skipping."
                    )
                    self.__warnings.append(warning)
                    continue

                key = (meas_no, channel, idx)
                if type1.upper() == 'G':
                    self.__gain_candidates[key] = filepath
                else:
                    self.__darknoise_candidates[key] = filepath

        if verbose:
            print(
                f"Found {len(self.__gain_candidates)} gain candidate(s) "
                f"and {len(self.__darknoise_candidates)} dark-noise "
                f"candidate(s)."
            )

        # ---- 4. Cross-reference files with Set_data.txt ----
        # Build the set of expected (meas, channel, idx) tuples from
        # the *valid* strip map. For each valid (meas, channel, socket),
        # the expected SiPM indices are
        #     (socket - 1) * SIPMS_PER_STRIP ... socket * SIPMS_PER_STRIP - 1
        self.__expected_keys = set()
        for (meas, channel, socket) in self.__valid_strip_map:
            start_idx = (socket - 1) * DewarDataPreprocessor.SIPMS_PER_STRIP
            end_idx = socket * DewarDataPreprocessor.SIPMS_PER_STRIP
            for idx in range(start_idx, end_idx):
                self.__expected_keys.add((meas, channel, idx))

        # Rule 5.2: listed in Set_data.txt but no files found — warn
        # and remove from valid candidates.
        self.__missing_data_positions = set()
        for key in sorted(self.__expected_keys):
            gain_exists = key in self.__gain_candidates
            darknoise_exists = key in self.__darknoise_candidates
            if not gain_exists and not darknoise_exists:
                meas, channel, idx = key
                socket = idx // DewarDataPreprocessor.SIPMS_PER_STRIP + 1
                strip_id = self.__valid_strip_map.get(
                    (meas, channel, socket), "?"
                )
                warning = (
                    f"No data files found for Meas_{meas}, CH{channel}, "
                    f"idx={idx} (Socket_{socket}, strip_ID={strip_id}). "
                    f"This SiPM will be skipped."
                )
                self.__warnings.append(warning)
                self.__missing_data_positions.add(key)
            elif not gain_exists:
                meas, channel, idx = key
                socket = idx // DewarDataPreprocessor.SIPMS_PER_STRIP + 1
                strip_id = self.__valid_strip_map.get(
                    (meas, channel, socket), "?"
                )
                warning = (
                    f"Gain data missing for Meas_{meas}, CH{channel}, "
                    f"idx={idx} (Socket_{socket}, strip_ID={strip_id}). "
                    f"Only dark-noise will be processed for this SiPM."
                )
                self.__warnings.append(warning)
            elif not darknoise_exists:
                meas, channel, idx = key
                socket = idx // DewarDataPreprocessor.SIPMS_PER_STRIP + 1
                strip_id = self.__valid_strip_map.get(
                    (meas, channel, socket), "?"
                )
                warning = (
                    f"Dark-noise data missing for Meas_{meas}, CH{channel}, "
                    f"idx={idx} (Socket_{socket}, strip_ID={strip_id}). "
                    f"Only gain will be processed for this SiPM."
                )
                self.__warnings.append(warning)

        # Rule 5.3: files exist but position not in Set_data.txt — warn
        all_file_keys = (
            set(self.__gain_candidates.keys())
            | set(self.__darknoise_candidates.keys())
        )
        unexpected_keys = all_file_keys - self.__expected_keys
        # Also exclude keys coming from invalidated or superseded positions
        excluded_positions = (
            self.__invalid_positions | self.__superseded_positions
        )
        invalid_keys = set()
        for (meas, channel, socket) in excluded_positions:
            start_idx = (socket - 1) * DewarDataPreprocessor.SIPMS_PER_STRIP
            end_idx = socket * DewarDataPreprocessor.SIPMS_PER_STRIP
            for idx in range(start_idx, end_idx):
                invalid_keys.add((meas, channel, idx))

        for key in sorted(unexpected_keys | invalid_keys):
            meas, channel, idx = key
            socket = idx // DewarDataPreprocessor.SIPMS_PER_STRIP + 1
            if key in invalid_keys:
                # Distinguish reason for user clarity
                is_intra = any(
                    (meas, channel, s) in self.__invalid_positions
                    for s in range(1, DewarDataPreprocessor.SOCKETS_PER_CHANNEL + 1)
                    if (socket - 1) * DewarDataPreprocessor.SIPMS_PER_STRIP
                    <= idx < socket * DewarDataPreprocessor.SIPMS_PER_STRIP
                    and (meas, channel, s) == (meas, channel, socket)
                )
                is_superseded = any(
                    (meas, channel, s) in self.__superseded_positions
                    for s in range(1, DewarDataPreprocessor.SOCKETS_PER_CHANNEL + 1)
                    if (meas, channel, s) == (meas, channel, socket)
                )
                if is_intra:
                    reason = "associated strip ID has an intra-meas duplicate"
                elif is_superseded:
                    reason = (
                        "strip was re-measured in a later measurement "
                        "(keeping only the latest)"
                    )
                else:
                    reason = "associated strip ID is duplicated"
            else:
                reason = (
                    "no strip ID is assigned to this position in Set_data.txt"
                )
            files_found = []
            if key in self.__gain_candidates:
                files_found.append(
                    os.path.basename(self.__gain_candidates[key])
                )
            if key in self.__darknoise_candidates:
                files_found.append(
                    os.path.basename(self.__darknoise_candidates[key])
                )
            if files_found:
                warning = (
                    f"Unexpected files for Meas_{meas}, CH{channel}, "
                    f"idx={idx} (Socket_{socket}): {files_found}. "
                    f"Reason: {reason}. These files will be ignored."
                )
                self.__warnings.append(warning)

        # Build the definitive set of processable keys
        self.__processable_keys = (
            self.__expected_keys
            - self.__missing_data_positions
            - invalid_keys
        )

        if verbose:
            print(
                f"{len(self.__processable_keys)} processable SiPM "
                f"position(s) identified."
            )
            if self.__warnings:
                print(f"{len(self.__warnings)} warning(s) generated.")

    @property
    def SetFolderpath(self):
        return self.__set_folderpath

    @property
    def SetNo(self):
        return self.__set_no

    @property
    def MeasStripMap(self):
        return self.__meas_strip_map

    @property
    def ValidStripMap(self):
        return self.__valid_strip_map

    @property
    def GainCandidates(self):
        return self.__gain_candidates

    @property
    def DarkNoiseCandidates(self):
        return self.__darknoise_candidates

    @property
    def ProcessableKeys(self):
        return self.__processable_keys

    @property
    def UnmatchedNpyFiles(self):
        """Returns list of .npy filenames that did not match the expected
        filename pattern. This usually indicates either:
        - A typo in the filename
        - A DAQ firmware/naming convention change
        - Temporary/debug files that should not be processed."""
        return self.__unmatched_npy_files

    @property
    def Warnings(self):
        return self.__warnings

    def generate_meas_config_files(
        self,
        root_directory,
        load_folderpath,
        aux_folderpath,
        data_folderpath,
        packing_version=2,
        median_filter_window_size=5.0,
        temperature_fluctuations_threshold=0.45,
        max_elapsed_cryo_time_min=600.0,
        verbose=True,
    ):
        """This method generates the intermediate-format configuration files
        (i.e. the load, aux and data files) for the SiPMs whose data was
        successfully validated during initialisation. This method gets the
        following mandatory positional arguments:

        - root_directory (string): Root directory of the workspace. All
        output paths written into JSON files are relative to this directory.

        - load_folderpath (string): Destination folder for the
        '*_gainmeas.json' and '*_darknoisemeas.json' files. Must be
        contained within root_directory.

        - aux_folderpath (string): Destination folder for the
        '*_wvf.json' and '*_wvfset.json' files. Must be contained
        within root_directory.

        - data_folderpath (string): Destination folder where the raw
        '.npy' data files will be moved. Must be contained within
        root_directory.

        This method also gets the following optional keyword arguments:

        - packing_version (int): The packing version used to interpret
        the binary metadata header of the '.npy' files. Defaults to 2.

        - median_filter_window_size (positive float): Window size used by
        scipy.ndimage.median_filter() to test whether each temperature
        curve is well formed. Defaults to 5.0.

        - temperature_fluctuations_threshold (semipositive float): Maximum
        allowed fluctuation energy for a temperature curve to be considered
        well formed. The fluctuation energy of a curve is the mean absolute
        deviation of the curve with respect to its median-filtered version,
        i.e. it is normalised by the number of samples, so that this
        threshold does not depend on how long the LN2 fill took. Defaults
        to 0.45, which was calibrated so that it tells apart the well-formed
        curves (up to ~0.39) from those showing a temperature rebound at the
        immersion moment (from ~0.53 on).

        - max_elapsed_cryo_time_min (semipositive float): Maximum elapsed
        cryogenic time, in minutes, which is considered plausible for a
        measurement. It is used to check that the temperature data found in
        a MEAS folder actually belong to the cryogenic bath of that
        measurement: if the immersion times they imply are not compatible
        with the epoch timestamp of the waveforms of that measurement, the
        temperature data are rejected. Defaults to 600.0.

        - verbose (boolean): Whether to print functioning-related messages.

        The JSON file containing the default values for all the metadata
        fields that are not available in the new input format (e.g.
        'operation_voltage_V', 'overvoltage_V', 'thermal_cycle', 'date',
        'delivery_no', 'set_no', etc.) is auto-discovered inside the SET
        folder (self.__set_folderpath). There must be exactly one '.json'
        file in that folder. Unlike the other preprocessors, this method
        does NOT prompt the user for any interactive input: all metadata
        is either extracted from the binary file headers, inferred from
        Set_data.txt, or read from this JSON defaults file.

        For the dark-noise trigger threshold, the defaults file may
        provide either 'threshold_mV' (single value) or the three
        per-board keys 'threshold_Ch0', 'threshold_Ch1' and
        'threshold_Ch2'. When both formats are present, the per-board
        values are preferred and a warning is emitted. If only a subset
        of the 'threshold_Ch*' keys is present, preprocessing aborts.

        For every processable SiPM, this method generates at most 6
        output files (3 for gain, 3 for dark-noise):

            - data/{base}_raw_gain.npy
            - aux/{base}_gain_wvf.json
            - aux/{base}_gain_wvfset.json
            - load/{base}_gainmeas.json
            - data/{base}_raw_darknoise.npy
            - aux/{base}_darknoise_wvf.json
            - aux/{base}_darknoise_wvfset.json
            - load/{base}_darknoisemeas.json

        where {base} follows the naming convention
        {strip_ID}-{sipm_location}-{thermal_cycle}-OV{ov}dV-{date}.

        Any inconsistencies found during processing are appended to
        self.Warnings and printed at the end if verbose is True."""

        # ---- Validate folder paths ----

        DataPreprocessor.check_well_formedness_of_input_folderpath(
            root_directory
        )
        DataPreprocessor.check_well_formedness_of_input_folderpath(
            load_folderpath,
            container_folderpath=root_directory
        )
        DataPreprocessor.check_well_formedness_of_input_folderpath(
            aux_folderpath,
            container_folderpath=root_directory
        )
        DataPreprocessor.check_well_formedness_of_input_folderpath(
            data_folderpath,
            container_folderpath=root_directory
        )

        htype.check_type(
            packing_version,
            int,
            exception_message=htype.generate_exception_message(
                "DewarDataPreprocessor.generate_meas_config_files", 90012
            ),
        )

        if packing_version < 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.generate_meas_config_files",
                    90013,
                    extra_info=(
                        f"packing_version must be a non-negative integer, "
                        f"but {packing_version} was given."
                    )
                )
            )

        htype.check_type(
            median_filter_window_size,
            int,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "DewarDataPreprocessor.generate_meas_config_files", 90032
            ),
        )

        if median_filter_window_size <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.generate_meas_config_files",
                    90033,
                    extra_info=(
                        "median_filter_window_size must be positive, but "
                        f"{median_filter_window_size} was given."
                    )
                )
            )

        htype.check_type(
            temperature_fluctuations_threshold,
            int,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "DewarDataPreprocessor.generate_meas_config_files", 90034
            ),
        )

        if temperature_fluctuations_threshold < 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.generate_meas_config_files",
                    90035,
                    extra_info=(
                        "temperature_fluctuations_threshold must be "
                        "semipositive, but "
                        f"{temperature_fluctuations_threshold} was given."
                    )
                )
            )

        htype.check_type(
            max_elapsed_cryo_time_min,
            int,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "DewarDataPreprocessor.generate_meas_config_files", 90047
            ),
        )

        if max_elapsed_cryo_time_min < 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.generate_meas_config_files",
                    90048,
                    extra_info=(
                        "max_elapsed_cryo_time_min must be semipositive, "
                        f"but {max_elapsed_cryo_time_min} was given."
                    )
                )
            )

        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "DewarDataPreprocessor.generate_meas_config_files", 90014
            ),
        )

        # ---- Auto-discover the JSON defaults file in the SET folder ----

        json_files = [
            f for f in os.listdir(self.__set_folderpath)
            if f.lower().endswith('.json') and os.path.isfile(
                os.path.join(self.__set_folderpath, f)
            )
        ]

        if len(json_files) == 0:
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.generate_meas_config_files",
                    90010,
                    extra_info=(
                        f"No JSON defaults file found in the SET folder "
                        f"'{self.__set_folderpath}'. There must be exactly "
                        f"one '.json' file in the SET folder."
                    )
                )
            )
        elif len(json_files) > 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.generate_meas_config_files",
                    90011,
                    extra_info=(
                        f"Found {len(json_files)} JSON files in the SET "
                        f"folder '{self.__set_folderpath}': {json_files}. "
                        f"There must be exactly one '.json' file in the "
                        f"SET folder."
                    )
                )
            )

        path_to_json_default_values = os.path.join(
            self.__set_folderpath, json_files[0]
        )

        if verbose:
            print(
                f"Using JSON defaults file: {path_to_json_default_values}"
            )

        # ---- Load the JSON defaults file ----

        with open(path_to_json_default_values, "r", encoding="utf-8") as f:
            try:
                defaults = json.load(f)
            except json.JSONDecodeError as e:
                f.seek(0)
                defaults = yaml.safe_load(f)
                warning = (
                    f"The JSON defaults file {path_to_json_default_values} "
                    f"could not be parsed with json.load(): {e}. It was "
                    "loaded with yaml.safe_load() instead."
                )
                self.__warnings.append(warning)
                if verbose:
                    print(warning)

        # The metadata translator maps the keys returned by
        # NumpyDataPreprocessor.get_metadata() to the field names
        # used in the output JSON files.

        translator = {
            "Horizontal Units": "time_unit",
            "Vertical Units": "signal_unit",
            "Sample Interval": "time_resolution",
            "Record Length": "points_per_wvf",
            "FastFrame Count": "wvfs_to_read",
        }

        # Fields expected in the defaults JSON. We split them by
        # destination file so that the generated output matches
        # exactly the structure produced by the other preprocessors.

        sipmmeas_field_names = [
            "delivery_no",
            "meas_ID",
            "date",
            "location",
            "operator",
            "setup_ID",
            "system_characteristics",
            "thermal_cycle",
            "cover_type",
            "operation_voltage_V",
            "overvoltage_V",
            "PDE",
            "status",
        ]

        gainmeas_extra_field_names = [
            "LED_voltage_V",
            "LED_frequency_kHz",
            "LED_pulse_shape",
            "LED_high_width_ns",
        ]

        threshold_mV_key = "threshold_mV"
        threshold_ch_keys = [
            "threshold_Ch0",
            "threshold_Ch1",
            "threshold_Ch2",
        ]

        wvfset_field_names = [
            "set_name",
            "creation_dt_offset_min",
        ]

        # Fields whose values are hardcoded and must NOT come from the
        # JSON defaults file. If they are present there, a warning is
        # emitted and the value in the file is ignored.
        ignored_if_present = [
            "signal_magnitude",
            "tray_no",
            "electronic_board_location",
            "packing_version",
            "set_no",
        ]
        for field in ignored_if_present:
            if field in defaults:
                warning = (
                    f"The field '{field}' was found in the JSON defaults "
                    f"file but will be ignored (its value is determined "
                    f"automatically)."
                )
                self.__warnings.append(warning)

        # Validate that all required keys are present in the defaults

        base_required_keys = (
            sipmmeas_field_names
            + gainmeas_extra_field_names
            + wvfset_field_names
        )

        missing_keys = [k for k in base_required_keys if k not in defaults]
        if missing_keys:
            raise cuex.InsufficientParameters(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.generate_meas_config_files",
                    90015,
                    extra_info=(
                        f"The following keys are missing from the JSON "
                        f"defaults file: {missing_keys}"
                    )
                )
            )

        has_threshold_mV = threshold_mV_key in defaults
        has_threshold_ch_any = any(
            key in defaults for key in threshold_ch_keys
        )
        has_threshold_ch_all = all(
            key in defaults for key in threshold_ch_keys
        )

        if has_threshold_ch_any and not has_threshold_ch_all:
            missing_threshold_ch = [
                key for key in threshold_ch_keys if key not in defaults
            ]
            raise cuex.InsufficientParameters(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.generate_meas_config_files",
                    90042,
                    extra_info=(
                        "The following threshold_Ch* keys are missing from "
                        f"the JSON defaults file: {missing_threshold_ch}. "
                        "Either provide all of them or use 'threshold_mV'."
                    )
                )
            )

        if not has_threshold_mV and not has_threshold_ch_all:
            raise cuex.InsufficientParameters(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.generate_meas_config_files",
                    90043,
                    extra_info=(
                        "No dark-noise threshold was found in the JSON "
                        "defaults file. Provide either 'threshold_mV' or "
                        "all of 'threshold_Ch0', 'threshold_Ch1' and "
                        "'threshold_Ch2'."
                    )
                )
            )

        if has_threshold_mV and has_threshold_ch_all:
            warning = (
                "Both 'threshold_mV' and 'threshold_Ch0/Ch1/Ch2' were "
                "found in the JSON defaults file. The per-board "
                "thresholds will be used."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)

        # Build the base metadata dictionaries from the defaults

        sipmmeas_base = {k: defaults[k] for k in sipmmeas_field_names}
        # Hardcoded fields
        sipmmeas_base["set_no"] = self.__set_no
        sipmmeas_base["electronic_board_location"] = (
            "Infer from electronic_board_number"
        )
        sipmmeas_base["tray_no"] = "Untracked"

        gainmeas_extra = {k: defaults[k] for k in gainmeas_extra_field_names}
        if has_threshold_ch_all:
            threshold_by_board = {
                1: defaults["threshold_Ch0"],
                2: defaults["threshold_Ch1"],
                3: defaults["threshold_Ch2"],
            }
        else:
            threshold_by_board = {
                1: defaults[threshold_mV_key],
                2: defaults[threshold_mV_key],
                3: defaults[threshold_mV_key],
            }
        wvf_base = {"signal_magnitude": "Voltage"}
        wvfset_base = {k: defaults[k] for k in wvfset_field_names}

        meas_folderpaths = {}
        for folder_name in os.listdir(self.__set_folderpath):
            folderpath = os.path.join(self.__set_folderpath, folder_name)
            if not os.path.isdir(folderpath):
                continue
            match = re.match(r'^MEAS_(\d+)$', folder_name, re.IGNORECASE)
            if match:
                meas_folderpaths[int(match.group(1))] = folderpath

        thermal_model_by_meas = {}

        # ---- Process every processable SiPM ----

        processed_gain = 0
        processed_darknoise = 0

        for key in sorted(self.__processable_keys):

            meas, channel, idx = key
            socket = idx // DewarDataPreprocessor.SIPMS_PER_STRIP + 1
            sipm_location = idx % DewarDataPreprocessor.SIPMS_PER_STRIP + 1
            strip_id = self.__valid_strip_map[(meas, channel, socket)]

            # The date follows the format 'YYYY-MM-DD HH:MM:SS'.
            # Thus, sipmmeas_base['date'].split()[0], gives 'YYYY-MM-DD'.
            output_filepath_base = (
                f"{strip_id}-{sipm_location}"
                f"-{sipmmeas_base['thermal_cycle']}"
                f"-OV{round(10. * sipmmeas_base['overvoltage_V'])}dV"
                f"-{sipmmeas_base['date'].split()[0]}"
            )

            electronic_board_number = channel + 1

            if meas not in thermal_model_by_meas:
                meas_folderpath = meas_folderpaths.get(meas)
                if meas_folderpath is None:
                    warning = (
                        f"No MEAS_{meas} folder was found in the SET folder "
                        f"'{self.__set_folderpath}'. Elapsed cryogenic times "
                        "will not be available for this measurement."
                    )
                    self.__warnings.append(warning)
                    if verbose:
                        print(warning)
                    thermal_model_by_meas[meas] = None
                else:
                    reference_filepath, reference_sipm_location = (
                        self._find_reference_waveform_file(meas)
                    )
                    thermal_model_by_meas[meas] = self._build_temperature_model(
                        defaults,
                        median_filter_window_size,
                        temperature_fluctuations_threshold,
                        max_elapsed_cryo_time_min,
                        reference_filepath,
                        reference_sipm_location,
                        temperature_folderpath=meas_folderpath,
                        verbose=verbose
                    )

            thermal_model = thermal_model_by_meas[meas]

            # -- Process gain --
            if key in self.__gain_candidates:
                try:
                    self._process_single_file(
                        filepath=self.__gain_candidates[key],
                        meas_type="gain",
                        output_filepath_base=output_filepath_base,
                        root_directory=root_directory,
                        load_folderpath=load_folderpath,
                        aux_folderpath=aux_folderpath,
                        data_folderpath=data_folderpath,
                        packing_version=packing_version,
                        translator=translator,
                        sipmmeas_base=sipmmeas_base,
                        type_extra_fields=gainmeas_extra,
                        wvf_base=wvf_base,
                        wvfset_base=wvfset_base,
                        thermal_model=thermal_model,
                        meas_no=meas,
                        strip_id=strip_id,
                        sipm_location=sipm_location,
                        electronic_board_socket=socket,
                        electronic_board_number=electronic_board_number,
                        verbose=verbose,
                    )
                    processed_gain += 1
                except Exception as e:
                    warning = (
                        f"Error processing gain file "
                        f"'{self.__gain_candidates[key]}': {e}. Skipping."
                    )
                    self.__warnings.append(warning)
                    if verbose:
                        print(warning)

            # -- Process dark-noise --
            if key in self.__darknoise_candidates:
                try:
                    darknoisemeas_extra = {
                        "threshold_mV": threshold_by_board[
                            electronic_board_number
                        ]
                    }
                    self._process_single_file(
                        filepath=self.__darknoise_candidates[key],
                        meas_type="darknoise",
                        output_filepath_base=output_filepath_base,
                        root_directory=root_directory,
                        load_folderpath=load_folderpath,
                        aux_folderpath=aux_folderpath,
                        data_folderpath=data_folderpath,
                        packing_version=packing_version,
                        translator=translator,
                        sipmmeas_base=sipmmeas_base,
                        type_extra_fields=darknoisemeas_extra,
                        wvf_base=wvf_base,
                        wvfset_base=wvfset_base,
                        thermal_model=thermal_model,
                        meas_no=meas,
                        strip_id=strip_id,
                        sipm_location=sipm_location,
                        electronic_board_socket=socket,
                        electronic_board_number=electronic_board_number,
                        verbose=verbose,
                    )
                    processed_darknoise += 1
                except Exception as e:
                    warning = (
                        f"Error processing dark-noise file "
                        f"'{self.__darknoise_candidates[key]}': {e}. "
                        f"Skipping."
                    )
                    self.__warnings.append(warning)
                    if verbose:
                        print(warning)

        if verbose:
            print(
                f"\nProcessing complete: {processed_gain} gain file(s) "
                f"and {processed_darknoise} dark-noise file(s) processed."
            )
            if self.__unmatched_npy_files:
                print(
                    f"\n[WARNING] {len(self.__unmatched_npy_files)} .npy file(s) "
                    f"did not match the expected filename pattern and were ignored:"
                )
                for fname in self.__unmatched_npy_files[:5]:
                    print(f"  - {fname}")
                if len(self.__unmatched_npy_files) > 5:
                    print(
                        f"  ... and {len(self.__unmatched_npy_files) - 5} more"
                    )
                print(
                    f"\nExpected pattern: set<N>[_<extra_number>]_meas<N>_..._"
                    f"<W|G>_CH<ch>_<idx>_<W|G>.npy"
                )
            if self.__warnings:
                print(f"\n--- {len(self.__warnings)} warning(s) ---")
                for w in self.__warnings:
                    print(f"  [WARNING] {w}")

        return

    def _process_single_file(
        self,
        filepath,
        meas_type,
        output_filepath_base,
        root_directory,
        load_folderpath,
        aux_folderpath,
        data_folderpath,
        packing_version,
        translator,
        sipmmeas_base,
        type_extra_fields,
        wvf_base,
        wvfset_base,
        thermal_model,
        meas_no,
        strip_id,
        sipm_location,
        electronic_board_socket,
        electronic_board_number,
        verbose,
    ):
        """Helper method that processes a single .npy file (either gain or
        dark-noise) and generates the corresponding output files in the
        intermediate format. This method is not intended to be called
        directly by the user.

        - filepath (string): Path to the .npy file to process.
        - meas_type (string): Either 'gain' or 'darknoise'.
        - output_filepath_base (string): The base string for output
          filenames (e.g. '7281-1-2-OV30dV-2025-05-28').
        - root_directory (string): Workspace root directory.
        - load_folderpath (string): Path to the load/ directory.
        - aux_folderpath (string): Path to the aux/ directory.
        - data_folderpath (string): Path to the data/ directory.
        - packing_version (int): Packing version for metadata extraction.
        - translator (dict): Maps metadata keys to output field names.
        - sipmmeas_base (dict): Common SiPM measurement metadata.
        - type_extra_fields (dict): Type-specific metadata (LED params
          for gain, threshold for dark-noise).
        - wvf_base (dict): Waveform-level metadata.
        - wvfset_base (dict): WaveformSet-level metadata.
        - thermal_model (None or dict): Optional temperature-derived model
          to compute elapsed cryogenic times.
        - meas_no (int): Measurement number.
        - strip_id (int): Strip ID.
        - sipm_location (int): SiPM position within the strip (1-6).
        - electronic_board_socket (int): Socket index (1-3).
        - electronic_board_number (int): Board/channel number (1-3).
        - verbose (bool): Whether to print progress messages."""

        if meas_type == "darknoise" and packing_version in (2, 3):
            self._drop_first_waveform_if_zero_timestamp(
                filepath,
                packing_version,
                verbose=verbose
            )

        effective_packing_version = self._resolve_effective_packing_version(
            filepath,
            packing_version
        )

        if effective_packing_version != packing_version:
            warning = (
                f"The file '{filepath}' was requested with packing_version="
                f"{packing_version}, but its second column was not identified "
                "as an epoch timestamp in milliseconds. Falling back to "
                f"packing_version={effective_packing_version} for this file."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)

        elapsed_cryo_time_min = self._compute_elapsed_cryo_time_min(
            filepath,
            sipm_location,
            thermal_model,
            effective_packing_version,
            verbose=verbose
        )

        # 1. Extract metadata from the binary file header
        metadata = NumpyDataPreprocessor.get_metadata(
            filepath,
            packing_version=effective_packing_version,
            get_creation_date=False,
            verbose=verbose
        )

        # 2. Build the wvf dictionary (waveform-level: units and magnitudes)
        wvf_dict = dict(wvf_base)
        wvf_dict[translator["Horizontal Units"]] = metadata["Horizontal Units"]
        wvf_dict[translator["Vertical Units"]] = metadata["Vertical Units"]

        # 3. Build the wvfset dictionary (waveform-set-level)
        wvfset_dict = dict(wvfset_base)
        wvfset_dict[translator["Sample Interval"]] = metadata["Sample Interval"]
        wvfset_dict[translator["Record Length"]] = metadata["Record Length"]
        wvfset_dict[translator["FastFrame Count"]] = metadata["FastFrame Count"]
        wvfset_dict["packing_version"] = effective_packing_version

        # 4. Move the raw data file
        _, extension = os.path.splitext(filepath)
        new_raw_filepath = os.path.join(
            data_folderpath,
            output_filepath_base + f"_raw_{meas_type}" + extension
        )
        shutil.move(filepath, new_raw_filepath)

        wvfset_dict["wvf_filepath"] = os.path.relpath(
            new_raw_filepath,
            start=root_directory
        )

        # 5. Generate the wvf JSON file
        wvf_output_filepath = os.path.join(
            aux_folderpath,
            output_filepath_base + f"_{meas_type}_wvf.json"
        )
        # Waveform.Signs.setter inputs must be lists
        DataPreprocessor.generate_json_file(
            {k: [v] for (k, v) in wvf_dict.items()},
            wvf_output_filepath
        )

        wvfset_dict["wvf_extra_info"] = os.path.relpath(
            wvf_output_filepath,
            start=root_directory
        )

        # 6. Generate the wvfset JSON file
        wvfset_output_filepath = os.path.join(
            aux_folderpath,
            output_filepath_base + f"_{meas_type}_wvfset.json"
        )
        DataPreprocessor.generate_json_file(
            wvfset_dict,
            wvfset_output_filepath
        )

        # 7. Build the measurement-level dictionary
        if meas_type == "gain":
            meas_suffix = "gainmeas"
        else:
            meas_suffix = "darknoisemeas"

        meas_dict = dict(type_extra_fields)
        meas_dict.update(sipmmeas_base)
        meas_dict["meas_no"] = meas_no
        meas_dict["electronic_board_socket"] = electronic_board_socket
        meas_dict["electronic_board_number"] = electronic_board_number
        meas_dict["sipm_location"] = sipm_location
        meas_dict["strip_ID"] = strip_id
        meas_dict["elapsed_cryo_time_min"] = elapsed_cryo_time_min
        meas_dict["wvfset_json_filepath"] = os.path.relpath(
            wvfset_output_filepath,
            start=root_directory
        )

        # 8. Generate the measurement JSON file
        meas_output_filepath = os.path.join(
            load_folderpath,
            output_filepath_base + f"_{meas_suffix}.json"
        )
        DataPreprocessor.generate_json_file(
            meas_dict,
            meas_output_filepath
        )

        return

    def _drop_first_waveform_if_zero_timestamp(
        self,
        filepath,
        packing_version,
        verbose=True
    ):
        """Helper method that checks whether the first FPGA-timestamp entry
        is null and, if so, rewrites the input file without the first row.

        - filepath (string): Path to the .npy file to inspect.
        - packing_version (int): Packing version requested for this file.
        - verbose (boolean): Whether to print progress messages.

        This method only applies to packing_version 2 or 3. It does not
        raise if the file cannot be inspected or rewritten; instead, it
        logs a warning and returns."""

        if packing_version not in (2, 3):
            return

        try:
            data = np.load(filepath)
        except Exception as e:
            warning = (
                "Could not inspect the FPGA timestamp of "
                f"'{filepath}': {e}. The file will be processed as-is."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return

        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
            warning = (
                "Could not validate the FPGA timestamp of "
                f"'{filepath}': unexpected array shape {data.shape}. "
                "The file will be processed as-is."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return

        if data[0, 0] != 0:
            return

        try:
            trimmed = np.array(data[1:, :])
            np.save(filepath, trimmed)
        except Exception as e:
            warning = (
                "Detected a null FPGA timestamp in the first row of "
                f"'{filepath}', but the file could not be rewritten: {e}. "
                "The file will be processed as-is."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return

        warning = (
            "Detected a null FPGA timestamp in the first row of "
            f"'{filepath}'. The first waveform was dropped and the file "
            "was rewritten."
        )
        self.__warnings.append(warning)
        if verbose:
            print(warning)

    def _resolve_effective_packing_version(
        self,
        filepath,
        requested_packing_version
    ):
        """Helper method that returns the packing version to store for a
        single file. If packing_version=3 was requested but the second
        column is not an epoch timestamp, it returns 2."""

        if requested_packing_version != 3:
            return requested_packing_version

        if NumpyDataPreprocessor.has_epoch_timestamp_column(filepath):
            return 3

        return 2

    def _find_reference_waveform_file(self, meas):
        """Helper method that returns a (filepath, sipm_location) tuple for
        one arbitrary-but-deterministic waveform file of the given
        measurement, or (None, None) if the measurement has no candidate.
        It is used to check that the temperature data found in a MEAS folder
        actually belong to the cryogenic bath of that measurement. Any of
        its files would do: the elapsed cryogenic times of the files of one
        measurement span a few hours at most, which is well within the
        tolerance of that check."""

        for key in sorted(self.__processable_keys):
            if key[0] != meas:
                continue

            filepath = self.__darknoise_candidates.get(
                key,
                self.__gain_candidates.get(key)
            )
            if filepath is not None:
                sipm_location = (
                    key[2] % DewarDataPreprocessor.SIPMS_PER_STRIP + 1
                )
                return filepath, sipm_location

        return None, None

    def _build_temperature_model(
        self,
        defaults,
        median_filter_window_size,
        temperature_fluctuations_threshold,
        max_elapsed_cryo_time_min,
        reference_waveform_filepath,
        reference_sipm_location,
        temperature_folderpath=None,
        verbose=True
    ):
        """Helper method that builds a linear LN2 immersion-time model
        from the optional temperature-sensor data in the given folder.
        It returns None when the required information is not available or
        is not well formed."""

        temperature_filepath = self._find_temperature_data_filepath(
            temperature_folderpath
        )
        if temperature_filepath is None:
            return None

        geometry_keys = [
            "deepest_SiPM_vpos_wrt_deepest_T_sensor_cm",
            "T_sensors_separation_cm",
            "SiPMs_separation_cm",
        ]
        missing_geometry_keys = [
            key for key in geometry_keys if key not in defaults
        ]
        if missing_geometry_keys:
            warning = (
                "Temperature data were found, but elapsed cryogenic times "
                "cannot be computed because the following keys are missing "
                f"from the JSON defaults file: {missing_geometry_keys}."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return None

        try:
            temperature_data = self._load_valid_temperature_data(
                temperature_filepath,
                median_filter_window_size,
                temperature_fluctuations_threshold,
                verbose=verbose
            )
        except Exception as e:
            warning = (
                f"Temperature data file '{temperature_filepath}' is invalid: "
                f"{e}. Elapsed cryogenic times will not be available."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return None
        
        well_formed_sensor_ids = sorted(
            temperature_data["immersion_times_s"].keys()
        )

        try:
            t_sensors_separation_cm = float(
                defaults["T_sensors_separation_cm"]
            )
            sipms_separation_cm = float(defaults["SiPMs_separation_cm"])
            deepest_sipm_vpos_cm = float(
                defaults["deepest_SiPM_vpos_wrt_deepest_T_sensor_cm"]
            )
        except Exception as e:
            warning = (
                "Temperature data were found, but elapsed cryogenic times "
                "cannot be computed because at least one geometry parameter "
                f"could not be converted to float: {e}."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return None

        if t_sensors_separation_cm <= 0.0 or sipms_separation_cm <= 0.0:
            warning = (
                "Temperature data were found, but elapsed cryogenic times "
                "cannot be computed because T_sensors_separation_cm and "
                "SiPMs_separation_cm must be positive."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return None

        # The vertical position of each sensor is fixed by how the sensor
        # string is wired in the setup, which is a constant of the setup.
        # It must not be re-inferred from the immersion times of each file:
        # a badly formed curve gives a meaningless immersion time, and so a
        # meaningless ordering.
        sensor_positions_cm = {
            sensor_id: (
                DewarDataPreprocessor.T_SENSORS_VERTICAL_ORDER.index(sensor_id)
                * t_sensors_separation_cm
            )
            for sensor_id in DewarDataPreprocessor.T_SENSORS_VERTICAL_ORDER
        }

        # When every curve is well formed, the measured immersion times are
        # trustworthy, so they can be used to cross-check the assumed wiring.
        all_immersion_times_s = temperature_data["all_immersion_times_s"]
        if len(well_formed_sensor_ids) == 4:
            measured_order = tuple(
                sorted(
                    all_immersion_times_s,
                    key=lambda sensor_id: all_immersion_times_s[sensor_id]
                )
            )
            if measured_order != DewarDataPreprocessor.T_SENSORS_VERTICAL_ORDER:
                warning = (
                    "The temperature sensors of "
                    f"'{temperature_filepath}' got immersed in the order "
                    f"{measured_order}, which does not match the assumed "
                    "wiring order (deepest to shallowest) "
                    f"{DewarDataPreprocessor.T_SENSORS_VERTICAL_ORDER}. The "
                    "assumed order will be used anyway, but the elapsed "
                    "cryogenic times of this measurement may be wrong."
                )
                self.__warnings.append(warning)
                if verbose:
                    print(warning)

        # Least-squares fit of the immersion time vs. the vertical position
        # over every well-formed sensor. With exactly two of them, this is
        # the straight line through both points.
        z_values = np.array(
            [sensor_positions_cm[sensor_id] for sensor_id in well_formed_sensor_ids]
        )
        time_values = np.array(
            [
                temperature_data["immersion_times_s"][sensor_id]
                for sensor_id in well_formed_sensor_ids
            ]
        )

        if np.all(z_values == z_values[0]):
            warning = (
                "Temperature data were found, but elapsed cryogenic times "
                "cannot be computed because every well-formed temperature "
                "sensor has the same vertical position."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return None

        slope_s_per_cm, intercept_s = np.polyfit(z_values, time_values, 1)

        # The LN2 level can only rise, so a deeper sensor must get immersed
        # before a shallower one. A non-positive slope means that the model
        # is inconsistent.
        if slope_s_per_cm <= 0.0:
            warning = (
                "Temperature data were found in "
                f"'{temperature_filepath}', but they imply a non-positive "
                f"LN2 rise time of {slope_s_per_cm} s/cm, which is "
                "unphysical. Elapsed cryogenic times will not be available "
                "for this measurement."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return None

        thermal_model = {
            "slope_s_per_cm": float(slope_s_per_cm),
            "intercept_s": float(intercept_s),
            "deepest_sipm_vpos_cm": deepest_sipm_vpos_cm,
            "sipms_separation_cm": sipms_separation_cm,
        }

        if not self._temperature_data_belong_to_meas(
            thermal_model,
            temperature_filepath,
            max_elapsed_cryo_time_min,
            reference_waveform_filepath,
            reference_sipm_location,
            verbose=verbose
        ):
            return None

        if verbose:
            if temperature_folderpath is None:
                temperature_folderpath = self.__set_folderpath
            print(
                f"Temperature data accepted in '{temperature_folderpath}'. "
                f"Fitted the LN2 rise over sensors {well_formed_sensor_ids} "
                f"as {slope_s_per_cm:.2f} s/cm."
            )

        return thermal_model

    def _temperature_data_belong_to_meas(
        self,
        thermal_model,
        temperature_filepath,
        max_elapsed_cryo_time_min,
        reference_waveform_filepath,
        reference_sipm_location,
        verbose=True
    ):
        """Helper method that checks that the given thermal model, built out
        of the temperature data of a MEAS folder, is compatible with the
        waveforms of that same measurement. It returns False when the
        elapsed cryogenic time implied for the reference waveform file falls
        outside of [0, max_elapsed_cryo_time_min], which means that the
        temperature file does not describe the cryogenic bath of this
        measurement. It returns True when the check passes, and also when it
        cannot be carried out."""

        if reference_waveform_filepath is None:
            return True

        first_waveform_epoch_s = (
            NumpyDataPreprocessor.get_first_epoch_timestamp_s(
                reference_waveform_filepath
            )
        )
        if first_waveform_epoch_s is None:
            # Without an epoch timestamp no elapsed cryogenic time can be
            # computed anyway, so there is nothing to check here.
            return True

        elapsed_cryo_time_min = (
            DewarDataPreprocessor._elapsed_cryo_time_min_from_epoch(
                thermal_model,
                reference_sipm_location,
                first_waveform_epoch_s
            )
        )

        if 0.0 <= elapsed_cryo_time_min <= max_elapsed_cryo_time_min:
            return True

        warning = (
            f"The temperature data in '{temperature_filepath}' were "
            "rejected: the LN2 fill they describe implies an elapsed "
            f"cryogenic time of {elapsed_cryo_time_min:.1f} min for "
            f"'{reference_waveform_filepath}', which falls outside of the "
            f"allowed [0.0, {max_elapsed_cryo_time_min}] min range. Most "
            "likely, this temperature file does not belong to this "
            "measurement, i.e. it was recorded during a different cryogenic "
            "bath. Elapsed cryogenic times will not be available for this "
            "measurement."
        )
        self.__warnings.append(warning)
        if verbose:
            print(warning)

        return False

    @staticmethod
    def _elapsed_cryo_time_min_from_epoch(
        thermal_model,
        sipm_location,
        first_waveform_epoch_s
    ):
        """Helper static method that returns the elapsed cryogenic time, in
        minutes, of a SiPM whose first waveform was acquired at the given
        epoch timestamp. The returned value may be negative, meaning that
        the given thermal model places the immersion of this SiPM after its
        first waveform."""

        sipm_vpos_cm = (
            thermal_model["deepest_sipm_vpos_cm"]
            + (DewarDataPreprocessor.SIPMS_PER_STRIP - sipm_location)
            * thermal_model["sipms_separation_cm"]
        )
        immersion_epoch_s = (
            thermal_model["intercept_s"]
            + thermal_model["slope_s_per_cm"] * sipm_vpos_cm
        )

        return (first_waveform_epoch_s - immersion_epoch_s) / 60.0

    def _find_temperature_data_filepath(self, folderpath=None):
        """Helper method that returns the unique temperature-data filepath
        found in the provided folder, or None if no usable unique candidate
        exists."""

        if folderpath is None:
            folderpath = self.__set_folderpath

        candidates = []
        for filename in os.listdir(folderpath):
            filepath = os.path.join(folderpath, filename)
            filename_lower = filename.lower()
            if (
                os.path.isfile(filepath)
                and filename_lower.endswith(".npy")
                and (
                    "sensor" in filename_lower
                    or "temperatur" in filename_lower
                )
            ):
                candidates.append(filepath)

        if len(candidates) == 0:
            warning = (
                "No temperature data candidate was found in the folder "
                f"'{folderpath}'. Elapsed cryogenic times will not be "
                "available."
            )
            self.__warnings.append(warning)
            return None

        if len(candidates) > 1:
            warning = (
                f"Found {len(candidates)} temperature data candidates in the "
                f"folder '{folderpath}': {candidates}. The candidate is "
                "ambiguous, so elapsed cryogenic times will not be "
                "available."
            )
            self.__warnings.append(warning)
            return None

        return candidates[0]

    def _load_valid_temperature_data(
        self,
        filepath,
        median_filter_window_size,
        temperature_fluctuations_threshold,
        verbose=True
    ):
        """Helper method that validates the temperature data format and
        returns immersion times for the well-formed sensor curves."""

        try:
            temperature_data = np.load(filepath)
        except Exception as e:
            raise cuex.BinReadException(
                htype.generate_exception_message(
                    "DewarDataPreprocessor._load_valid_temperature_data",
                    90036,
                    extra_info=(
                        f"Caught the following error while reading {filepath}: "
                        f"{e}"
                    )
                )
            )

        if temperature_data.ndim != 2:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DewarDataPreprocessor._load_valid_temperature_data",
                    90037,
                    extra_info="The temperature array must be bidimensional."
                )
            )

        if (temperature_data.shape[0] == 5) == (temperature_data.shape[1] == 5):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DewarDataPreprocessor._load_valid_temperature_data",
                    90038,
                    extra_info=(
                        "Exactly one temperature-array dimension must be "
                        f"equal to 5, but shape {temperature_data.shape} was "
                        "found."
                    )
                )
            )

        if temperature_data.shape[0] == 5:
            temperature_data = temperature_data.T

        time_data = temperature_data[:, 0]
        if not np.all(np.diff(time_data) > 0.0):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DewarDataPreprocessor._load_valid_temperature_data",
                    90039,
                    extra_info=(
                        "The timestamp sequence in the temperature array "
                        "must be strictly increasing."
                    )
                )
            )

        filter_window_size = int(median_filter_window_size)
        if filter_window_size < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DewarDataPreprocessor._load_valid_temperature_data",
                    90041,
                    extra_info=(
                        "The integer median-filter window size must be at "
                        f"least 1, but {filter_window_size} was obtained."
                    )
                )
            )
        well_formed_sensors = {}
        defective_sensors = {}
        all_immersion_times_s = {}

        for sensor_id in range(1, 5):
            temperature_curve = temperature_data[:, sensor_id]

            # Mean absolute deviation of the curve with respect to its
            # median-filtered version. Both the absolute value and the
            # normalisation matter: summing the signed deviations would let
            # the positive and the negative ones cancel out, and not
            # normalising would couple this figure to the duration of the
            # LN2 fill.
            fluctuation_energy = np.mean(
                np.abs(
                    median_filter(
                        temperature_curve,
                        size=filter_window_size
                    ) - temperature_curve
                )
            )

            # The immersion moment is the steepest temperature drop.
            derivative = np.diff(temperature_curve)
            immersion_idx = int(np.argmin(derivative)) + 1
            all_immersion_times_s[sensor_id] = float(time_data[immersion_idx])

            if fluctuation_energy > temperature_fluctuations_threshold:
                defective_sensors[sensor_id] = float(fluctuation_energy)
                continue

            well_formed_sensors[sensor_id] = all_immersion_times_s[sensor_id]

        if len(well_formed_sensors) < 2:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DewarDataPreprocessor._load_valid_temperature_data",
                    90040,
                    extra_info=(
                        "At least two temperature sensor curves must be "
                        "well formed, but only "
                        f"{len(well_formed_sensors)} were accepted. "
                        f"Defective sensors: {defective_sensors}."
                    )
                )
            )

        if verbose:
            print(
                "Temperature well-formed sensors: "
                f"{sorted(well_formed_sensors.keys())}; defective sensors: "
                f"{defective_sensors}."
            )

        return {
            "immersion_times_s": well_formed_sensors,
            "all_immersion_times_s": all_immersion_times_s,
            "defective_sensors": defective_sensors,
        }

    def _compute_elapsed_cryo_time_min(
        self,
        filepath,
        sipm_location,
        thermal_model,
        effective_packing_version,
        verbose=True
    ):
        """Helper method that computes elapsed cryogenic time for one SiPM
        measurement. It returns None if any required information is missing
        or if the computed elapsed time is negative."""

        if thermal_model is None:
            return None

        if effective_packing_version != 3:
            warning = (
                f"Elapsed cryogenic time cannot be computed for '{filepath}' "
                "because no valid epoch timestamp column was found for the "
                "waveforms."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return None

        first_waveform_epoch_s = (
            NumpyDataPreprocessor.get_first_epoch_timestamp_s(filepath)
        )
        if first_waveform_epoch_s is None:
            warning = (
                f"Elapsed cryogenic time cannot be computed for '{filepath}' "
                "because no valid epoch timestamp column was found for the "
                "waveforms."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return None

        elapsed_cryo_time_min = (
            DewarDataPreprocessor._elapsed_cryo_time_min_from_epoch(
                thermal_model,
                sipm_location,
                first_waveform_epoch_s
            )
        )

        if elapsed_cryo_time_min < 0.0:
            warning = (
                f"Elapsed cryogenic time for '{filepath}' was computed as "
                f"{elapsed_cryo_time_min} min, which is negative. The value "
                "will be stored as None."
            )
            self.__warnings.append(warning)
            if verbose:
                print(warning)
            return None

        return elapsed_cryo_time_min

    @staticmethod
    def _parse_set_data(filepath):
        """This static method parses a 'Set_data.txt' file and returns
        the set number, the strip-ID mapping, and the Vop mapping.

        - filepath (string): Path to the 'Set_data.txt' file.

        Returns a tuple (set_no, strip_map, vop_map) where:

        - set_no (int): The set number parsed from the first line.
        - strip_map (dict): Maps (meas_no, channel, socket) -> strip_ID
          (int). channel is 0-based (0, 1, 2), socket is 1-based (1, 2, 3).
        - vop_map (dict): Maps meas_no -> vop_string (e.g. '44_5')."""

        htype.check_type(
            filepath,
            str,
            exception_message=htype.generate_exception_message(
                "DewarDataPreprocessor._parse_set_data", 90020
            ),
        )

        set_no = None
        strip_map = {}
        vop_map = {}

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_meas = None
        current_channels = []  # ordered list of channel indices

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                continue

            # First line: Set_<N>
            set_match = re.match(r'^Set[_\s]*(\d+)$', line, re.IGNORECASE)
            if set_match:
                set_no = int(set_match.group(1))
                continue

            # Meas line: Meas_<N> Vop:<vop>
            meas_match = re.match(
                r'^Meas[_\s]*(\d+)\s+Vop[:\s]*(.+)$',
                line,
                re.IGNORECASE
            )
            if meas_match:
                current_meas = int(meas_match.group(1))
                vop_map[current_meas] = meas_match.group(2).strip()
                current_channels = []
                continue

            # Header line: Ch_0  Ch_1  Ch_2
            ch_match = re.findall(r'Ch[_\s]*(\d+)', line, re.IGNORECASE)
            if ch_match and current_meas is not None:
                current_channels = [int(c) for c in ch_match]
                continue

            # Socket line: Socket_<N>  <id1>  <id2>  [<id3>]
            socket_match = re.match(
                r'^Socket[_\s]*(\d+)\s+(.*)',
                line,
                re.IGNORECASE
            )
            if socket_match and current_meas is not None:
                socket = int(socket_match.group(1))
                values = socket_match.group(2).split()

                for ch_idx, val in enumerate(values):
                    if ch_idx >= len(current_channels):
                        break
                    val = val.strip()
                    if not val:
                        continue
                    try:
                        strip_id = int(val)
                    except ValueError:
                        continue
                    channel = current_channels[ch_idx]
                    strip_map[(current_meas, channel, socket)] = strip_id

                continue

        if set_no is None:
            raise cuex.MalFunction(
                htype.generate_exception_message(
                    "DewarDataPreprocessor._parse_set_data",
                    90021,
                    extra_info=(
                        "Could not parse the set number from "
                        f"'{filepath}'."
                    )
                )
            )

        return set_no, strip_map, vop_map

    @staticmethod
    def find_set_folders(input_folderpath, verbose=True):
        """This static method scans the given folder for subfolders
        matching the pattern 'SET_<N>' (case-insensitive) and returns
        them sorted by set number. This is intended as a convenience
        method for use in the preprocessor notebook.

        - input_folderpath (string): Path to the directory to scan.

        - verbose (boolean): Whether to print functioning-related messages.

        Returns a list of absolute paths to SET folders, sorted by set
        number."""

        htype.check_type(
            input_folderpath,
            str,
            exception_message=htype.generate_exception_message(
                "DewarDataPreprocessor.find_set_folders", 90030
            ),
        )

        if not os.path.isdir(input_folderpath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DewarDataPreprocessor.find_set_folders",
                    90031,
                    extra_info=(
                        f"The folder '{input_folderpath}' does not exist."
                    )
                )
            )

        set_folders = []
        pattern = re.compile(
            r'SET_(\d+)',
            re.IGNORECASE
        )
        for name in os.listdir(input_folderpath):
            full_path = os.path.join(input_folderpath, name)
            if os.path.isdir(full_path):
                match = pattern.search(name)
                if match:
                    set_folders.append((int(match.group(1)), full_path))

        set_folders.sort(key=lambda x: x[0])
        result = [path for _, path in set_folders]

        if verbose:
            print(
                f"Found {len(result)} SET folder(s) in "
                f"'{input_folderpath}': "
                f"{[os.path.basename(p) for p in result]}"
            )

        return result
