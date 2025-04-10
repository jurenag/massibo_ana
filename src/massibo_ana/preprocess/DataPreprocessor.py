import os
import json
import numpy as np
import datetime
import shutil
import struct

import massibo_ana.utils.htype as htype
import massibo_ana.utils.custom_exceptions as cuex


class DataPreprocessor:

    def __init__(
        self,
        input_folderpath,
        gain_base="gain",
        darknoise_base="darknoise",
        timestamp_prefix="ts_",
        binary_extension="wfm",
        key_separator="_",
        verbose=True,
    ):
        """This class is aimed at handling and preprocessing the gain and
        dark-noise raw data that we get out of the LabView application for
        one thermal-cycle, i.e. for one cryogenic immersion of the whole
        setup. This initializer gets the following mandatory positional
        argument

        - input_folderpath (string): Path to the folder where the input
        data is hosted.

        And the following optional keyword arguments:

        - gain_base (string): Every file which contains this string
        in its filename, but do not contain darknoise_base, will be
        considered the output of a gain measurement. If such file
        extension (resp. does not) matches binary_extension, then its
        format is considered (not) to be the Tektronix .wfm file format.
        - darknoise_base (string): Every file which:
            1) contains this string in its filename,
            2) does not contain gain_base in its filename and
            3) its file name does not start with timestamp_prefix
        will be considered the output of a dark noise measurement.
        Such file format is considered to be ASCII or the Tektronix
        .wfm file format using the same criterion that is applied to
        gain measurements.
        - timestamp_prefix (string): For every file which is considered
        to be the output of a dark noise measurement in ASCII format,
        this initializer will look for a file with the same file name
        but with a prefix equal to timestamp_prefix. If such file exist,
        then it is considered to be the time stamp of such dark noise
        measurement.
        - binary_extension (string): It is used to interpret whether the
        format of a measurement file is ASCII or the Tektronix .wfm file
        format.
        - key_separator (string): All of the filepaths that are definitely
        considered a measurement candidate must contain at least two
        occurrences of key_separator after its base. For gain (resp. dark
        noise) measurements, its base is gain_base (resp. darknoise_base).
        The substring that takes place in between such two occurrences,
        is casted to integer by DataPreprocessor.find_integer_after_base,
        and later used as a key for dictionary population.
        - verbose (boolean): Whether to print functioning-related messages.

        Based on the criteria explained above, this initializer populates
        the attribute self.__bin_gain_candidates (resp.
        self.__ascii_gain_candidates) with the filepaths to the files within
        the provided input folder which are considered to be the output of a
        gain measurement in the Tektronix .wfm file format (resp. ASCII
        format). The same goes for the dark noise measurements and the
        attributes self.__bin_darknoise_candidates and
        self.__ascii_darknoise_candidates. For every member in
        self.__ascii_darknoise_candidates, the file path to its time stamp
        will be saved to self.__timestamp_candidates, if it exists. The key
        for every value that is added to such dictionary attributes is
        extracted from the corresponding filepath by
        DataPreprocessor.find_integer_after_base. Check the key_separator
        parameter documentation or the DataPreprocessor.find_integer_after_base
        docstring for more information.
        """

        htype.check_type(
            input_folderpath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.__init__", 89176
            ),
        )
        htype.check_type(
            gain_base,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.__init__", 89911
            ),
        )
        htype.check_type(
            darknoise_base,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.__init__", 40891
            ),
        )
        htype.check_type(
            timestamp_prefix,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.__init__", 31061
            ),
        )
        htype.check_type(
            binary_extension,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.__init__", 31061
            ),
        )
        htype.check_type(
            key_separator,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.__init__", 47299
            ),
        )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.__init__", 79488
            ),
        )

        self.__input_folderpath = input_folderpath
        self.__gain_base = gain_base
        self.__darknoise_base = darknoise_base
        self.__timestamp_prefix = timestamp_prefix

        self.__ascii_gain_candidates = {}
        self.__ascii_darknoise_candidates = {}
        self.__bin_gain_candidates = {}
        self.__bin_darknoise_candidates = {}

        self.__timestamp_candidates = {}

        for filename in os.listdir(self.__input_folderpath):
            filepath = os.path.join(self.__input_folderpath, filename)
            if os.path.isfile(filepath):
                if (
                    self.__gain_base in filename
                    and self.__darknoise_base not in filename
                ):

                    # Possible failures are
                    # anticipated within
                    # find_integer_after_base
                    aux = DataPreprocessor.find_integer_after_base(
                        filename, self.__gain_base, separator=key_separator
                    )

                    DataPreprocessor.bin_ascii_splitter(
                        self.__bin_gain_candidates,
                        self.__ascii_gain_candidates,
                        aux,
                        filepath,
                        binary_extension=binary_extension,
                    )

                elif (
                    self.__gain_base not in filename
                    and self.__darknoise_base in filename
                    and not filename.startswith(self.__timestamp_prefix)
                ):

                    # Possible failures are
                    # anticipated within
                    # find_integer_after_base
                    aux = DataPreprocessor.find_integer_after_base(
                        filename, self.__darknoise_base, separator=key_separator
                    )

                    DataPreprocessor.bin_ascii_splitter(
                        self.__bin_darknoise_candidates,
                        self.__ascii_darknoise_candidates,
                        aux,
                        filepath,
                        binary_extension=binary_extension,
                    )

        for key in self.__ascii_darknoise_candidates.keys():
            for filename in os.listdir(self.__input_folderpath):
                filepath = os.path.join(self.__input_folderpath, filename)
                if os.path.isfile(filepath):
                    if (
                        filename
                        == self.__timestamp_prefix
                        + os.path.split(self.__ascii_darknoise_candidates[key])[1]
                    ):
                        self.__timestamp_candidates[key] = filepath
                        break

        DataPreprocessor.print_dictionary_info(
            self.__ascii_gain_candidates,
            "gain",
            candidate_type="ASCII",
            verbose=verbose,
        )

        DataPreprocessor.print_dictionary_info(
            self.__ascii_darknoise_candidates,
            "dark noise",
            candidate_type="ASCII",
            verbose=verbose,
        )

        DataPreprocessor.print_dictionary_info(
            self.__timestamp_candidates,
            "dark noise",
            candidate_type="complete ASCII",
            verbose=verbose,
        )

        DataPreprocessor.print_dictionary_info(
            self.__bin_gain_candidates, "gain", candidate_type="binary", verbose=verbose
        )

        DataPreprocessor.print_dictionary_info(
            self.__bin_darknoise_candidates,
            "dark noise",
            candidate_type="binary",
            verbose=verbose,
        )

        # By construction, if a key exists in self.__timestamp_candidates.keys(),
        # then it exists in self.__ascii_darknoise_candidates.keys()
        aux = len(self.__ascii_darknoise_candidates) - len(self.__timestamp_candidates)
        if aux > 0:
            print(
                f"----> There are {aux} ASCII dark noise candidates with no matching time stamps!"
            )

        return

    @property
    def InputFolderpath(self):
        return self.__input_folderpath

    @property
    def GainBase(self):
        return self.__gain_base

    @property
    def DarknoiseBase(self):
        return self.__darknoise_base

    @property
    def TimestampPrefix(self):
        return self.__timestamp_prefix

    @property
    def ASCIIGainCandidates(self):
        return self.__ascii_gain_candidates

    @property
    def ASCIIDarkNoiseCandidates(self):
        return self.__ascii_darknoise_candidates

    @property
    def TimeStampCandidates(self):
        return self.__timestamp_candidates

    @property
    def BinaryGainCandidates(self):
        return self.__bin_gain_candidates

    @property
    def BinaryDarkNoiseCandidates(self):
        return self.__bin_darknoise_candidates

    @staticmethod
    def check_well_formedness_of_input_folderpath(
        folderpath, 
        container_folderpath=None
    ):
        """This helper method gets the following positional argument:
        
        - folderpath (string): The folderpath to be checked
        
        And the following optional keyword argument:
        
        - container_folderpath (string): If it is not None, then folder
        pointed to by folderpath is checked to be contained within the
        folder pointed to by container_folderpath.

        This method checks that folderpath is a string and that it points 
        to an existing directory. Additionally, if container_folderpath 
        is not None, then it checks that the folder pointed to by folderpath 
        is contained within the folder pointed to by container_folderpath."""

        htype.check_type(
            folderpath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.check_well_formedness_of_input_folderpath", 
                45200
            ),
        )

        if not os.path.isdir(folderpath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DataPreprocessor.check_well_formedness_of_input_folderpath",
                    91371,
                    extra_info=f"Path {folderpath} does not exist or is not a directory.",
                )
            )

        # In this case, the well-formedness of container_folderpath is
        # checked by DataPreprocessor.path_is_contained_in_dir()
        if container_folderpath is not None:
            if not DataPreprocessor.path_is_contained_in_dir(
                folderpath, container_folderpath
            ):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "DataPreprocessor.check_well_formedness_of_input_folderpath",
                        86735,
                        extra_info=f"{folderpath} is not contained within {container_folderpath}.",
                    )
                )
        return

    def generate_meas_config_files(
        self,
        root_directory,
        load_folderpath,
        aux_folderpath,
        data_folderpath,
        wvf_skiprows_identifier="TIME,",
        path_to_json_default_values=None,
        sipms_per_strip=None,
        strips_ids=None,
        ask_for_inference_confirmation=True,
        verbose=True,
    ):
        """This method gets the following mandatory positional arguments:

        - root_directory (string): Path which points to an existing directory,
        which is considered to be the root directory. Every path which is
        written by this function is relative to this root directory.
        - load_folderpath (string): Path to a folder where the DarkNoiseMeas and
        GainMeas configuration json files will be saved. It must be contained,
        at an arbitrary depth, within the root directory.
        - aux_folderpath (string): Path to a folder where the WaveformSet and
        Waveform json configuration files will be saved. It must be contained,
        at an arbitrary depth, within the root directory.
        - data_folderpath (string): Raw data files regardless its original 
        format, will be saved in this folder. Time stamp data files, if applicable, 
        will be also saved in this folder. It must be contained, at an arbitrary 
        depth, within the root directory.

        This method gets the following optional keyword arguments:

        - wvf_skiprows_identifier (string): This parameter only makes a difference
        for ASCII input files. It is given to DataPreprocessor.get_metadata()
        as skiprows_identifier for the case where files hosting ASCII waveform
        sets are processed. Check DataPreprocessor.get_metadata() docstring for
        more information on this parameter.
        - path_to_json_default_values (string): If it is not none, it should be
        the path to a json file from which some default values may be read.
        - sipms_per_strip (positive integer): The number of SiPMs per strip. If
        it is not None, the electronic_board_socket and sipm_location fields will
        be inferred. To do so, for each type of measurement (namely ascii gain,
        ascii dark noise, binary gain and binary dark noise), each candidate is
        assigned an electronic_board_socket (resp. sipm_location) value which is
        computed as (i//sipms_per_strip)+1 (resp. (i%sipms_per_strip)+1), where i
        is the key of such candidate within the corresponding dictionary, i.e.
        self.__ascii_gain_candidates for ASCII gain measurements, 
        self.__bin_gain_candidates for binary gain measurements and so on.
        - strips_ids (dictionary): Its keys and values must be integers. The
        value for this parameter only makes a difference if sipms_per_strip
        is defined. In such case (and if it is defined), then for each type
        of measurement, every measurement whose key takes a value from
        (i-1)*sipms_per_strip to (i*sipms_per_strip)-1 (both inclusive), is
        assumed to belong to the strip with ID strips_ids[i], i.e. the strip_ID
        field for such measurement will be set to strips_ids[i]. To this end,
        it is required that all of the measurement keys belong to the union of
        the sets U_i = {(i-1)*sipms_per_strip, ..., (i*sipms_per_strip)-1} for
        every i in the set of keys of the strips_ids dictionary. If this is
        not the case, then an exception is raised.
        - ask_for_inference_confirmation (boolean): This parameter only makes
        a difference if sipms_per_strip is not None. In that case, then this
        parameter determines whether the user is asked for confirmation before
        the fields 'electronic_board_socket' and 'sipm_location' are inferred.
        This is also applied to the 'strip_ID' field if the strips_ids parameter
        is also defined.
        - verbose (boolean): Whether to print functioning-related messages.

        This method iterates over self.__ascii_gain_candidates,
        self.__ascii_darknoise_candidates, self.__bin_gain_candidates and
        self.__bin_darknoise_candidates.

        For each filepath in self.__ascii_gain_candidates or
        self.__bin_gain_candidates (resp. self.__ascii_darknoise_candidates or
        self.__bin_darknoise_candidates), it generates a json file which contains
        all of the information needed to create a GainMeas (DarkNoiseMeas) object
        using the GainMeas.from_json_file (DarkNoiseMeas.from_json_file)
        initializer class method. To do so, some information is taken from the
        input files themselves, such as the fields 'time_unit', 'signal_unit',
        'time_resolution', 'points_per_wvf' or 'wvfs_to_read', among others.
        The remaining necessary information, namely

            - signal_magnitude (str),
            - set_name (str),
            - creation_dt_offset_min (float),
            - delivery_no (int),
            - set_no (int),
            - meas_no (int),
            - strip_ID (int),
            - meas_ID (str),
            - date (str, in the format 'YYYY-MM-DD HH:MM:SS'),
            - location (str),
            - operator (str),
            - setup_ID (str),
            - system_characteristics (str),
            - thermal_cycle (int),
            - electronic_board_number (int),
            - electronic_board_location (str),
            - electronic_board_socket (int),
            - sipm_location (int),
            - cover_type (str),
            - operation_voltage_V (float),
            - overvoltage_V (float),
            - PDE (float),
            - status (str),
            - LED_voltage_V (float),
            - LED_frequency_kHz (float),
            - LED_pulse_shape (str),
            - LED_high_width_ns (float) and
            - threshold_mV (float),

        is taken from the json file given to path_to_json_default_values, if
        it is available there and the values comply with the expected types.
        The user is interactively asked for the fields which could not be 
        retrieved from the given json file."""

        DataPreprocessor.check_well_formedness_of_input_folderpath(
            root_directory,
            container_folderpath=None
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
            wvf_skiprows_identifier,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.generate_meas_config_files", 42451
            ),
        )
        fReadDefaultsFromFile = False
        if path_to_json_default_values is not None:
            htype.check_type(
                path_to_json_default_values,
                str,
                exception_message=htype.generate_exception_message(
                    "DataPreprocessor.generate_meas_config_files", 91126
                ),
            )
            fReadDefaultsFromFile = True

        fInferrFields = False
        if sipms_per_strip is not None:
            htype.check_type(
                sipms_per_strip,
                int,
                np.int64,
                exception_message=htype.generate_exception_message(
                    "DataPreprocessor.generate_meas_config_files", 89701
                ),
            )
            if sipms_per_strip < 1:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "DataPreprocessor.generate_meas_config_files", 12924
                    )
                )
            fInferrFields = True
            fAssignStripID = False

            if (
                strips_ids is not None
            ):  # Yes, only check strips_ids if sipms_per_strip is defined
                htype.check_type(
                    strips_ids,
                    dict,
                    exception_message=htype.generate_exception_message(
                        "DataPreprocessor.generate_meas_config_files", 42323
                    ),
                )
                for key in strips_ids.keys():
                    htype.check_type(
                        key,
                        int,
                        np.int64,
                        exception_message=htype.generate_exception_message(
                            "DataPreprocessor.generate_meas_config_files", 61191
                        ),
                    )

                    htype.check_type(
                        strips_ids[key],
                        int,
                        np.int64,
                        exception_message=htype.generate_exception_message(
                            "DataPreprocessor.generate_meas_config_files", 10370
                        ),
                    )

                allowed_measurement_keys = []
                for key in strips_ids.keys():
                    allowed_measurement_keys += list(
                        range(
                            (key - 1) * sipms_per_strip,
                            key * sipms_per_strip
                        )
                    )

                not_allowed_found_keys = set()

                for key in self.ASCIIGainCandidates.keys():
                    if key not in allowed_measurement_keys:
                        not_allowed_found_keys.add(key)

                for key in self.ASCIIDarkNoiseCandidates.keys():
                    if key not in allowed_measurement_keys:
                        not_allowed_found_keys.add(key)

                for key in self.BinaryGainCandidates.keys():
                    if key not in allowed_measurement_keys:
                        not_allowed_found_keys.add(key)

                for key in self.BinaryDarkNoiseCandidates.keys():
                    if key not in allowed_measurement_keys:
                        not_allowed_found_keys.add(key)

                if len(not_allowed_found_keys) > 0:
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message(
                            "DataPreprocessor.generate_meas_config_files",
                            39450,
                            extra_info=f"Found the following not-allowed "
                            f"measurement keys in the candidates: {list(not_allowed_found_keys)}."
                            " Note that the measurement keys must belong "
                            f"to the following set: {allowed_measurement_keys}.",
                        )
                    )
                
                fAssignStripID = True

        htype.check_type(
            ask_for_inference_confirmation,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.generate_meas_config_files", 67213
            ),
        )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.generate_meas_config_files", 92127
            ),
        )
        for key in self.ASCIIDarkNoiseCandidates.keys():
            if key not in self.TimeStampCandidates.keys():
                raise cuex.NoAvailableData(
                    htype.generate_exception_message(
                        "DataPreprocessor.generate_meas_config_files",
                        44613,
                        extra_info=f"ASCII Dark noise measurement candidate with key={key} lacks a time stamp.",
                    )
                )
                # As of this point, every key which belongs to self.ASCIIDarkNoiseCandidates,
                # is also present in self.TimeStampCandidates

        queried_wvf_fields = {"signal_magnitude": str}
        read_wvf_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_wvf_fields_from_file, queried_wvf_fields = (
                DataPreprocessor.try_grabbing_from_json(
                    queried_wvf_fields, path_to_json_default_values, verbose=verbose
                )
            )
        queried_once_wvf_fields = {}
        if bool(queried_wvf_fields):  # True if queried_wvfs_fields is not empty
            queried_once_wvf_fields, queried_wvf_fields = (
                DataPreprocessor.query_dictionary_splitting(queried_wvf_fields)
            )

        # Deprecation note: The 'separator' field is not needed 
        # anymore, and so, its support in the WaveformSet reading
        # methods has been removed. The 'separator' field was used
        # in the past to separate the last signal sample of the i-th
        # waveform from the first signal sample of the (i+1)-th 
        # waveform. In the case of ASCII files, our oscilloscope 
        # always outputs the number of samples, i.e. points_per_wvf, 
        # so we won't need a separator in such case. For the case of 
        # binary files, the read process does not need a separator either.
        queried_wvfset_fields = {  #'separator':str,
            "set_name": str,
            "creation_dt_offset_min": float,
        }

        read_wvfset_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_wvfset_fields_from_file, queried_wvfset_fields = (
                DataPreprocessor.try_grabbing_from_json(
                    queried_wvfset_fields, path_to_json_default_values, verbose=verbose
                )
            )
        queried_once_wvfset_fields = {}
        if bool(queried_wvfset_fields):  # True if queried_wvfset_fields is not empty
            queried_once_wvfset_fields, queried_wvfset_fields = (
                DataPreprocessor.query_dictionary_splitting(queried_wvfset_fields)
            )

        queried_sipmmeas_fields = {
            "delivery_no": int,
            "set_no": int,
            "meas_no": int,
            "strip_ID": int,
            "meas_ID": str,
            "date": str,  # It must be in the format 'YYYY-MM-DD HH:MM:SS'
            "location": str,
            "operator": str,
            "setup_ID": str,
            "system_characteristics": str,
            "thermal_cycle": int,
            "electronic_board_number": int,
            "electronic_board_location": str,
            "electronic_board_socket": int,
            "sipm_location": int,
            "cover_type": str,
            "operation_voltage_V": float,
            "overvoltage_V": float,
            "PDE": float,
            "status": str,
        }

        inferred_sipmmeas_fields = {}
        if fInferrFields:
            inferred_sipmmeas_fields = {
                "electronic_board_socket": queried_sipmmeas_fields[
                    "electronic_board_socket"
                ],
                "sipm_location": queried_sipmmeas_fields["sipm_location"],
            }
            del queried_sipmmeas_fields["electronic_board_socket"]
            del queried_sipmmeas_fields["sipm_location"]

            if fAssignStripID:
                inferred_sipmmeas_fields["strip_ID"] = queried_sipmmeas_fields[
                    "strip_ID"
                ]
                del queried_sipmmeas_fields["strip_ID"]

            if not fAssignStripID:
                if ask_for_inference_confirmation:
                    if not DataPreprocessor.yes_no_translator(
                        input(
                            f"In function DataPreprocessor.generate_meas_config_files(): The values for the fields 'electronic_board_socket' and 'sipm_location' will be inferred according to the candidates keys and the value given to the 'sipms_per_strip' ({sipms_per_strip}) parameter. Do you want to continue? (y/n)"
                        )
                    ):
                        return
                else:
                    print(f"In function DataPreprocessor.generate_meas_config_files(): The values for the fields 'electronic_board_socket' and 'sipm_location' will be inferred according to the candidates keys and the value given to the 'sipms_per_strip' ({sipms_per_strip}) parameter.")
            else:
                if ask_for_inference_confirmation:
                    if not DataPreprocessor.yes_no_translator(
                        input(
                            f"In function DataPreprocessor.generate_meas_config_files(): The values for the fields 'electronic_board_socket', 'sipm_location' and 'strip_ID', will be inferred according to the candidates keys and the values given to the 'sipms_per_strip' ({sipms_per_strip}) and the 'strips_ids' ({strips_ids}) parameters. Do you want to continue? (y/n)"
                        )
                    ):
                        return
                else:
                    print(f"In function DataPreprocessor.generate_meas_config_files(): The values for the fields 'electronic_board_socket', 'sipm_location' and 'strip_ID', will be inferred according to the candidates keys and the values given to the 'sipms_per_strip' ({sipms_per_strip}) and the 'strips_ids' ({strips_ids}) parameters.")

        read_sipmmeas_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_sipmmeas_fields_from_file, queried_sipmmeas_fields = (
                DataPreprocessor.try_grabbing_from_json(
                    queried_sipmmeas_fields,
                    path_to_json_default_values,
                    verbose=verbose,
                )
            )
        queried_once_sipmmeas_fields = {}
        if bool(
            queried_sipmmeas_fields
        ):  # True if queried_sipmmeas_fields is not empty
            queried_once_sipmmeas_fields, queried_sipmmeas_fields = (
                DataPreprocessor.query_dictionary_splitting(queried_sipmmeas_fields)
            )

        queried_gainmeas_fields = {
            "LED_voltage_V": float,
            "LED_frequency_kHz": float,
            "LED_pulse_shape": str,
            "LED_high_width_ns": float,
        }

        inferred_gainmeas_fields = {}
        if fInferrFields:
            inferred_gainmeas_fields.update(inferred_sipmmeas_fields)

        read_gainmeas_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_gainmeas_fields_from_file, queried_gainmeas_fields = (
                DataPreprocessor.try_grabbing_from_json(
                    queried_gainmeas_fields,
                    path_to_json_default_values,
                    verbose=verbose,
                )
            )
        queried_once_gainmeas_fields = {}
        if bool(
            queried_gainmeas_fields
        ):  # True if queried_gainmeas_fields is not empty
            print("The following only applies to gain measurements: ", end="")
            queried_once_gainmeas_fields, queried_gainmeas_fields = (
                DataPreprocessor.query_dictionary_splitting(queried_gainmeas_fields)
            )
        queried_gainmeas_fields.update(queried_sipmmeas_fields)

        # The acquisition time is not queried because
        # it is computed from the time stamp data
        queried_darknoisemeas_fields = {"threshold_mV": float}

        inferred_darknoisemeas_fields = {}
        if fInferrFields:
            inferred_darknoisemeas_fields.update(inferred_sipmmeas_fields)

        read_darknoisemeas_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_darknoisemeas_fields_from_file, queried_darknoisemeas_fields = (
                DataPreprocessor.try_grabbing_from_json(
                    queried_darknoisemeas_fields,
                    path_to_json_default_values,
                    verbose=verbose,
                )
            )
        queried_once_darknoisemeas_fields = {}
        if bool(
            queried_darknoisemeas_fields
        ):  # True if queried_darknoisemeas_fields is not empty
            print("The following only applies to dark noise measurements: ", end="")
            queried_once_darknoisemeas_fields, queried_darknoisemeas_fields = (
                DataPreprocessor.query_dictionary_splitting(
                    queried_darknoisemeas_fields
                )
            )
        queried_darknoisemeas_fields.update(queried_sipmmeas_fields)

        translator = {
            "Horizontal Units": [str, "time_unit"],
            "Vertical Units": [str, "signal_unit"],
            "Sample Interval": [float, "time_resolution"],
            "Record Length": [int, "points_per_wvf"],
            "FastFrame Count": [int, "wvfs_to_read"],
            # The casuistry for the following one is as follows:
            # - ASCII gain: There's no timestamp from which to compute this, so
            #               this value may be computed from LED_frequency_kHz
            # - ASCII dark noise: The value is computed from the input timestamp
            # - Binary gain:    For our particular case, the DAQ oscilloscope is giving an
            #                   empty timestamp, since the trigger in this case is external
            #                   (it comes from the LED voltage source). The code should
            #                   try to compute 'delta_t_wf' from the timestamp, but if it
            #                   results in 0.0, the code should alternatively compute it
            #                   using LED_frequency_kHz.
            # - Binary dark noise: The value is computed from the input timestamp
            "average_delta_t_wf": [float, "delta_t_wf"],
            "acquisition_time": [float, "acquisition_time_min"],
        }

        # Query unique-query data and add update them with the default values gotten from the json file
        if (
            queried_once_wvf_fields
            or queried_once_wvfset_fields
            or queried_once_sipmmeas_fields
            or queried_once_gainmeas_fields
            or queried_darknoisemeas_fields
        ):
            print(
                "Let us retrieve the unique-query fields. These fields will apply for every measurement in this DataPreprocessor instance."
            )

        aux_wvf_dict = DataPreprocessor.query_fields_in_dictionary(
            queried_once_wvf_fields, default_dict=None
        )
        aux_wvf_dict.update(read_wvf_fields_from_file)
        aux_wvfset_dict = DataPreprocessor.query_fields_in_dictionary(
            queried_once_wvfset_fields, default_dict=None
        )
        aux_wvfset_dict.update(read_wvfset_fields_from_file)
        aux_sipmmeas_dict = DataPreprocessor.query_fields_in_dictionary(
            queried_once_sipmmeas_fields, default_dict=None
        )
        aux_sipmmeas_dict.update(read_sipmmeas_fields_from_file)
        aux_gainmeas_dict = DataPreprocessor.query_fields_in_dictionary(
            queried_once_gainmeas_fields, default_dict=None
        )
        aux_gainmeas_dict.update(read_gainmeas_fields_from_file)
        aux_darknoisemeas_dict = DataPreprocessor.query_fields_in_dictionary(
            queried_once_darknoisemeas_fields, default_dict=None
        )
        aux_darknoisemeas_dict.update(read_darknoisemeas_fields_from_file)

        aux_gainmeas_dict.update(aux_sipmmeas_dict)
        aux_darknoisemeas_dict.update(aux_sipmmeas_dict)

        for i, key in enumerate(sorted(self.ASCIIGainCandidates.keys())):

            aux = DataPreprocessor.get_metadata(
                self.ASCIIGainCandidates[key],
                *translator.keys(),
                get_creation_date=False,
                verbose=verbose,
                is_ASCII=True,
                skiprows_identifier=wvf_skiprows_identifier,
                parameters_delimiter=",",
                casting_functions=tuple(
                    [translator[key][0] for key in translator.keys()]
                )
            )

            print(
                f"Let us retrieve some information for the waveform set in {self.ASCIIGainCandidates[key]}"
            )

            aux_wvf_dict.update(
                {
                    translator["Horizontal Units"][1]: aux["Horizontal Units"],
                    translator["Vertical Units"][1]: aux["Vertical Units"],
                }
            )
            aux_wvf_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvf_fields, default_dict=aux_wvf_dict
                )
            )

            aux_wvfset_dict.update(
                {
                    translator["Sample Interval"][1]: aux["Sample Interval"],
                    translator["Record Length"][1]: aux["Record Length"],
                    translator["FastFrame Count"][1]: aux["FastFrame Count"],
                }
            )
            aux_wvfset_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvfset_fields, default_dict=aux_wvfset_dict
                )
            )

            if fInferrFields:
                aux_gainmeas_dict.update(
                    {
                        "electronic_board_socket": (key // sipms_per_strip) + 1,
                        "sipm_location": (key % sipms_per_strip) + 1,
                    }
                )
                if fAssignStripID:
                    aux_gainmeas_dict.update(
                        {"strip_ID": strips_ids[
                            aux_gainmeas_dict["electronic_board_socket"]
                            ]
                        }
                    )

            aux_gainmeas_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_gainmeas_fields, default_dict=aux_gainmeas_dict
                )
            )

            # The date follows the format 'YYYY-MM-DD HH:MM:SS'. Thus,
            # aux_gainmeas_dict['date'][:10], gives 'YYYY-MM-DD'.
            output_filepath_base = f"{aux_gainmeas_dict['strip_ID']}-{aux_gainmeas_dict['sipm_location']}-{aux_gainmeas_dict['thermal_cycle']}-OV{round(10.*aux_gainmeas_dict['overvoltage_V'])}dV-{aux_gainmeas_dict['date'][:10]}"

            _, extension = os.path.splitext(self.ASCIIGainCandidates[key])

            new_raw_filepath = os.path.join(
                data_folderpath, 
                output_filepath_base + "_raw_gain" + extension)
            
            shutil.move(
                self.ASCIIGainCandidates[key], 
                new_raw_filepath
            )

            aux_wvfset_dict.update(
                {
                    "wvf_filepath": os.path.relpath(
                        new_raw_filepath, start=root_directory
                    )
                }
            )

            wvf_output_filepath = os.path.join(
                aux_folderpath, output_filepath_base + "_gain_wvf.json"
            )

            DataPreprocessor.generate_json_file(
                # Waveform.Signs.setter inputs must be lists
                {key: [value] for (key, value) in aux_wvf_dict.items()}, 
                wvf_output_filepath)

            aux_wvfset_dict["wvf_extra_info"] = os.path.relpath(
                wvf_output_filepath, start=root_directory
            )

            wvfset_output_filepath = os.path.join(
                aux_folderpath, output_filepath_base + "_gain_wvfset.json"
            )

            DataPreprocessor.generate_json_file(aux_wvfset_dict, wvfset_output_filepath)

            aux_gainmeas_dict["wvfset_json_filepath"] = os.path.relpath(
                wvfset_output_filepath, start=root_directory
            )

            gainmeas_output_filepath = os.path.join(
                load_folderpath, output_filepath_base + "_gainmeas.json"
            )

            DataPreprocessor.generate_json_file(
                aux_gainmeas_dict, gainmeas_output_filepath
            )

        for i, key in enumerate(sorted(self.ASCIIDarkNoiseCandidates.keys())):

            aux = DataPreprocessor.get_metadata(
                self.ASCIIDarkNoiseCandidates[key],
                *translator.keys(),
                get_creation_date=False,
                verbose=verbose,
                is_ASCII=True,
                skiprows_identifier=wvf_skiprows_identifier,
                parameters_delimiter=",",
                casting_functions=tuple(
                    [translator[key][0] for key in translator.keys()]
                )
            )

            print(
                f"Let us retrieve some information for the waveform set in {self.ASCIIDarkNoiseCandidates[key]}"
            )

            aux_wvf_dict.update(
                {
                    translator["Horizontal Units"][1]: aux["Horizontal Units"],
                    translator["Vertical Units"][1]: aux["Vertical Units"],
                }
            )

            aux_wvf_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvf_fields, default_dict=aux_wvf_dict
                )
            )

            aux_wvfset_dict.update(
                {
                    translator["Sample Interval"][1]: aux["Sample Interval"],
                    translator["Record Length"][1]: aux["Record Length"],
                    # Extracting this value, although it is not
                    # strictly necessary for the Dark Noise case.
                    translator["FastFrame Count"][1]: aux["FastFrame Count"]
                }
            )  

            aux_wvfset_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvfset_fields, default_dict=aux_wvfset_dict
                )
            )

            if fInferrFields:
                aux_darknoisemeas_dict.update(
                    {
                        "electronic_board_socket": (key // sipms_per_strip) + 1,
                        "sipm_location": (key % sipms_per_strip) + 1,
                    }
                )
                if fAssignStripID:
                    aux_darknoisemeas_dict.update(
                        {"strip_ID": strips_ids[
                            aux_darknoisemeas_dict["electronic_board_socket"]
                            ]
                        }
                    )

            aux_darknoisemeas_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_darknoisemeas_fields, default_dict=aux_darknoisemeas_dict
                )
            )

            # The date follows the format 'YYYY-MM-DD HH:MM:SS'. Thus,
            # aux_darknoisemeas_dict['date'][:10], gives 'YYYY-MM-DD'.
            output_filepath_base = f"{aux_darknoisemeas_dict['strip_ID']}-{aux_darknoisemeas_dict['sipm_location']}-{aux_darknoisemeas_dict['thermal_cycle']}-OV{round(10.*aux_darknoisemeas_dict['overvoltage_V'])}dV-{aux_darknoisemeas_dict['date'][:10]}"

            _, extension = os.path.splitext(self.ASCIIDarkNoiseCandidates[key])
            _, ts_extension = os.path.splitext(self.TimeStampCandidates[key])

            new_raw_filepath = os.path.join(
                data_folderpath, 
                output_filepath_base + "_raw_darknoise" + extension)
            
            new_raw_ts_filepath = os.path.join(
                data_folderpath, 
                output_filepath_base + "_raw_ts_darknoise" + ts_extension)  

            shutil.move(
                self.ASCIIDarkNoiseCandidates[key], 
                new_raw_filepath
            )

            shutil.move(
                self.TimeStampCandidates[key], 
                new_raw_ts_filepath
            )

            aux_wvfset_dict.update(
                {
                    "wvf_filepath": os.path.relpath(
                        new_raw_filepath, start=root_directory
                    ),
                    "timestamp_filepath": os.path.relpath(
                        new_raw_ts_filepath, start=root_directory
                    ),
                }
            )

            wvf_output_filepath = os.path.join(
                aux_folderpath, output_filepath_base + "_darknoise_wvf.json"
            )

            DataPreprocessor.generate_json_file(
                # Waveform.Signs.setter inputs must be lists
                {key: [value] for (key, value) in aux_wvf_dict.items()}, 
                wvf_output_filepath)

            aux_wvfset_dict["wvf_extra_info"] = os.path.relpath(
                wvf_output_filepath, start=root_directory
            )
            wvfset_output_filepath = os.path.join(
                aux_folderpath, output_filepath_base + "_darknoise_wvfset.json"
            )
            DataPreprocessor.generate_json_file(aux_wvfset_dict, wvfset_output_filepath)

            aux_darknoisemeas_dict["wvfset_json_filepath"] = os.path.relpath(
                wvfset_output_filepath, start=root_directory
            )
            darknoisemeas_output_filepath = os.path.join(
                load_folderpath, output_filepath_base + "_darknoisemeas.json"
            )
            DataPreprocessor.generate_json_file(
                aux_darknoisemeas_dict, darknoisemeas_output_filepath
            )

        try:
            # Binary Gain measurements won't overwrite
            # the 'timestamp_filepath' entry of wvfset_dict
            # via dictionary update. We need to manually
            # remove it, otherwise all of our binary gain
            # measurements will count on a fixed erroneous
            # 'timestamp_filepath' entry.
            del aux_wvfset_dict["timestamp_filepath"]

        # This exception can happen if no ASCII dark noise
        # measurement was processed. In such case, there's
        # no entry within aux_wvfset_dict under the
        # 'timestamp_filepath' key.
        except KeyError:
            pass

        for i, key in enumerate(sorted(self.BinaryGainCandidates.keys())):

            aux = DataPreprocessor.get_metadata(
                self.BinaryGainCandidates[key],
                get_creation_date=False,
                verbose=verbose,
                is_ASCII=False
            )

            print(
                f"Let us retrieve some information for the waveform set in {self.BinaryGainCandidates[key]}"
            )

            aux_wvf_dict.update(
                {
                    translator["Horizontal Units"][1]: aux["Horizontal Units"],
                    translator["Vertical Units"][1]: aux["Vertical Units"],
                }
            )

            aux_wvf_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvf_fields, default_dict=aux_wvf_dict
                )
            )

            aux_wvfset_dict.update(
                {
                    translator["Sample Interval"][1]: aux["Sample Interval"],
                    translator["Record Length"][1]: aux["Record Length"],
                    translator["FastFrame Count"][1]: aux["FastFrame Count"]
                }
            )

            aux_wvfset_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvfset_fields, default_dict=aux_wvfset_dict
                )
            )

            if fInferrFields:
                aux_gainmeas_dict.update(
                    {
                        "electronic_board_socket": (key // sipms_per_strip) + 1,
                        "sipm_location": (key % sipms_per_strip) + 1,
                    }
                )
                if fAssignStripID:
                    aux_gainmeas_dict.update(
                        {"strip_ID": strips_ids[
                            aux_gainmeas_dict["electronic_board_socket"]
                            ]
                        }
                    )

            aux_gainmeas_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_gainmeas_fields, default_dict=aux_gainmeas_dict
                )
            )

            # The date follows the format 'YYYY-MM-DD HH:MM:SS'. Thus,
            # aux_gainmeas_dict['date'][:10], gives 'YYYY-MM-DD'.
            output_filepath_base = f"{aux_gainmeas_dict['strip_ID']}-{aux_gainmeas_dict['sipm_location']}-{aux_gainmeas_dict['thermal_cycle']}-OV{round(10.*aux_gainmeas_dict['overvoltage_V'])}dV-{aux_gainmeas_dict['date'][:10]}"

            _, extension = os.path.splitext(self.BinaryGainCandidates[key])

            new_raw_filepath = os.path.join(
                data_folderpath, 
                output_filepath_base + "_raw_gain" + extension)
            
            shutil.move(
                self.BinaryGainCandidates[key], 
                new_raw_filepath
            )

            aux_wvfset_dict.update(
                {
                    "wvf_filepath": os.path.relpath(
                        new_raw_filepath, start=root_directory
                    )
                }
            )

            wvf_output_filepath = os.path.join(
                aux_folderpath, output_filepath_base + "_gain_wvf.json"
            )

            DataPreprocessor.generate_json_file(
                # Waveform.Signs.setter inputs must be lists
                {key: [value] for (key, value) in aux_wvf_dict.items()}, 
                wvf_output_filepath)

            aux_wvfset_dict["wvf_extra_info"] = os.path.relpath(
                wvf_output_filepath, start=root_directory
            )

            wvfset_output_filepath = os.path.join(
                aux_folderpath, output_filepath_base + "_gain_wvfset.json"
            )
            DataPreprocessor.generate_json_file(aux_wvfset_dict, wvfset_output_filepath)

            aux_gainmeas_dict["wvfset_json_filepath"] = os.path.relpath(
                wvfset_output_filepath, start=root_directory
            )
            gainmeas_output_filepath = os.path.join(
                load_folderpath, output_filepath_base + "_gainmeas.json"
            )
            DataPreprocessor.generate_json_file(
                aux_gainmeas_dict, gainmeas_output_filepath
            )

        for i, key in enumerate(sorted(self.BinaryDarkNoiseCandidates.keys())):

            aux = DataPreprocessor.get_metadata(
                self.BinaryDarkNoiseCandidates[key],
                get_creation_date=False,
                verbose=verbose,
                is_ASCII=False
            )

            print(
                f"Let us retrieve some information for the waveform set in {self.BinaryDarkNoiseCandidates[key]}"
            )

            aux_wvf_dict.update(
                {
                    translator["Horizontal Units"][1]: aux["Horizontal Units"],
                    translator["Vertical Units"][1]: aux["Vertical Units"],
                }
            )

            aux_wvf_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvf_fields, default_dict=aux_wvf_dict
                )
            )

            aux_wvfset_dict.update(
                {
                    translator["Sample Interval"][1]: aux["Sample Interval"],
                    translator["Record Length"][1]: aux["Record Length"],
                    translator["FastFrame Count"][1]: aux["FastFrame Count"],
                }
            )

            aux_wvfset_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvfset_fields, default_dict=aux_wvfset_dict
                )
            )

            if fInferrFields:
                aux_darknoisemeas_dict.update(
                    {
                        "electronic_board_socket": (key // sipms_per_strip) + 1,
                        "sipm_location": (key % sipms_per_strip) + 1,
                    }
                )
                if fAssignStripID:
                    aux_darknoisemeas_dict.update(
                        {"strip_ID": strips_ids[
                            aux_darknoisemeas_dict["electronic_board_socket"]
                            ]
                        }
                    )

            aux_darknoisemeas_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_darknoisemeas_fields, default_dict=aux_darknoisemeas_dict
                )
            )

            output_filepath_base = f"{aux_darknoisemeas_dict['strip_ID']}-{aux_darknoisemeas_dict['sipm_location']}-{aux_darknoisemeas_dict['thermal_cycle']}-OV{round(10.*aux_darknoisemeas_dict['overvoltage_V'])}dV-{aux_darknoisemeas_dict['date'][:10]}"
            # The date follows the format 'YYYY-MM-DD HH:MM:SS'. Thus,
            # aux_darknoisemeas_dict['date'][:10], gives 'YYYY-MM-DD'.

            _, extension = os.path.splitext(self.BinaryDarkNoiseCandidates[key])

            new_raw_filepath = os.path.join(
                data_folderpath, 
                output_filepath_base + "_raw_darknoise" + extension)
            
            shutil.move(
                self.BinaryDarkNoiseCandidates[key], 
                new_raw_filepath
            )

            aux_wvfset_dict.update(
                {
                    "wvf_filepath": os.path.relpath(
                        new_raw_filepath, start=root_directory
                    )
                }
            )

            wvf_output_filepath = os.path.join(
                aux_folderpath, output_filepath_base + "_darknoise_wvf.json"
            )

            DataPreprocessor.generate_json_file(
                # Waveform.Signs.setter inputs must be lists
                {key: [value] for (key, value) in aux_wvf_dict.items()}, 
                wvf_output_filepath)

            aux_wvfset_dict["wvf_extra_info"] = os.path.relpath(
                wvf_output_filepath, start=root_directory
            )

            wvfset_output_filepath = os.path.join(
                aux_folderpath, output_filepath_base + "_darknoise_wvfset.json"
            )

            DataPreprocessor.generate_json_file(aux_wvfset_dict, wvfset_output_filepath)

            aux_darknoisemeas_dict["wvfset_json_filepath"] = os.path.relpath(
                wvfset_output_filepath, start=root_directory
            )

            darknoisemeas_output_filepath = os.path.join(
                load_folderpath, output_filepath_base + "_darknoisemeas.json"
            )
            
            DataPreprocessor.generate_json_file(
                aux_darknoisemeas_dict, darknoisemeas_output_filepath
            )

        return

    @staticmethod
    def integer_generator():
        integer = -1
        while True:
            integer += 1
            yield integer

    @staticmethod
    def find_integer_after_base(input_string, base, separator="_"):
        """This static method gets the following mandatory poisitional
        argument:

        - input_string (string): Must contain at least one occurrence
        of base. It must also contain at least two occurrences of separator
        after the first occurrence of base.
        - base (string)

        And the following optional keyword argument:

        - separator (string):

        This method searches for the first occurrence of base in
        input_string. Then takes the substring that goes after such
        occurrence, and looks for the first two occurrences of separator.
        This method takes what's in between such occurrences of separator
        and tries to cast it to integer. This function returns the result
        of the casting process if it is successful. It raises an
        InvalidParameterDefinition exception otherwise."""

        htype.check_type(
            input_string,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.find_integer_after_base", 17544
            ),
        )
        htype.check_type(
            base,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.find_integer_after_base", 79240
            ),
        )
        htype.check_type(
            separator,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.find_integer_after_base", 87101
            ),
        )
        if DataPreprocessor.count_occurrences(input_string, base) < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.find_integer_after_base",
                    96145,
                    extra_info=f"There's not a single occurence of {base} in {input_string}.",
                )
            )

        # Take what's after the
        # first occurrence of base
        idx = input_string.find(base, 0) + len(base)
        aux = input_string[idx:]

        if DataPreprocessor.count_occurrences(aux, separator) < 2:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.find_integer_after_base",
                    42225,
                    extra_info=f"There must be at least two occurrences of the separator, {separator}, in {aux}.",
                )
            )

        # Take what's in between
        # both occurrences of separator
        idx = aux.find(separator, 0) + len(separator)
        aux = aux[idx:]
        idx = aux.find(separator, 0)
        aux = aux[:idx]

        try:
            return int(aux)
        except ValueError:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.find_integer_after_base",
                    57274,
                    extra_info=f"{aux} is not broadcastable to integer.",
                )
            )

    @staticmethod
    def count_occurrences(string, substring):
        """This static method takes two mandatory positional arguments:

        - string (string)
        - substring (string)

        This static method returns an integer which matches the number
        of occurrences of substring within string."""

        htype.check_type(
            string,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.count_occurrences", 97796
            ),
        )
        htype.check_type(
            substring,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.count_occurrences", 43079
            ),
        )
        result = 0
        idx = 0
        while True:
            idx = string.find(substring, idx)
            if idx == -1:
                break
            result += 1
            idx += len(substring)
        return result

    @staticmethod
    def bin_ascii_splitter(bin_dict, ascii_dict, key, filepath, binary_extension="wfm"):
        """This static method gets the following mandatory positional arguments:

        - bin_dict (dictionary)
        - ascii_dict (dictionary)
        - key (object)
        - filepath (string)

        This static method gets the following optional keyword arguments:

        - binary_extension (string)

        This method computes the extension of the given filepath as
        os.path.splitext(filepath)[1][1:], which gives the substring of the filepath
        after the period. If such extension (resp. does not) matches binary_extension,
        then filepath is added to bin_dict (resp. ascii_dict) under a key equal to the
        given key.
        """

        htype.check_type(
            bin_dict,
            dict,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.bin_ascii_splitter", 48102
            ),
        )
        htype.check_type(
            ascii_dict,
            dict,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.bin_ascii_splitter", 89915
            ),
        )
        htype.check_type(
            filepath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.bin_ascii_splitter", 47181
            ),
        )
        htype.check_type(
            binary_extension,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.bin_ascii_splitter", 38311
            ),
        )
        extension = os.path.splitext(filepath)[1][1:]
        if extension == binary_extension:
            bin_dict[key] = filepath
        else:
            ascii_dict[key] = filepath
        return

    @staticmethod
    def print_dictionary_info(dictionary, meas_type, candidate_type="", verbose=False):

        htype.check_type(
            dictionary,
            dict,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.print_dictionary_info", 33719
            ),
        )
        htype.check_type(
            meas_type,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.print_dictionary_info", 71881
            ),
        )
        htype.check_type(
            candidate_type,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.print_dictionary_info", 38122
            ),
        )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.print_dictionary_info", 44422
            ),
        )

        print(
            f"----> Found {len(dictionary)} {candidate_type} candidates to {meas_type} measurements, with keys={list(dictionary.keys())}"
        )
        if verbose:
            for key in dictionary.keys():
                print(f"Key={key}, Filename={dictionary[key]}")
        return

    @staticmethod
    def find_skiprows(input_filepath, identifier):
        """This static method gets:

        - input filepath (string): Path to the file which will be parsed.
        - identifier (string): String used to identify the line which immediately
        precedes the data columns in input_filepath. Let us call this the immediate
        header. identifier must be defined so that the immediate header starts with
        identifier, i.e. identifier==header[0:N] must evaluate to True, for some
        1<=N<=len(header).

        This file returns the number of rows, skiprows, that should be skipped when
        reading the data in input_filepath, in order to directly access the numerical
        data. This function computes skiprows by finding the first line within
        input_filepath which starts with identifier. Thus, it is recommended that
        identifier is as long as possible, so that mis-identification of the immediate
        header won't occur. For example, if we specify a very short identifier, it is
        more likely that some line before the actual immediate header starts with
        identifier. In such case, this function will consider this line to be the
        immediate header. If no line within input_filepath starts with identifier,
        then -1 is returned."""

        htype.check_type(
            input_filepath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.find_skiprows", 90123
            ),
        )
        htype.check_type(
            identifier,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.find_skiprows", 12767
            ),
        )
        found = False
        skiprows = 1
        with open(input_filepath, "r") as file:
            for line in file:
                if line.startswith(identifier):
                    found = True
                    break
                skiprows += 1
        if found:
            return skiprows
        else:
            return -1

    @staticmethod
    def _parse_headers(
        filepath,
        *identifiers,
        identifier_delimiter=",",
        casting_functions=None,
        headers_end_identifier=None,
        return_skiprows=True,
    ):
        """This static method is a helper method which must only be called
        by DataPreprocessor.get_metadata(), where the well-formedness checks
        of the input parameters have been performed. This method gets the 
        following compulsory positional arguments:

        - filepath (string): Path to the file which will be parsed.

        This function also gets the following optional positional arguments:

        - identifiers (tuple of strings): Each one is considered to be the string
        which precedes the value of a parameter of interest within the input file
        headers.

        And the following keyword arguments:

        - identifier_delimiter (string): String used to separate the identifier
        from its value.
        - casting_functions (tuple/list of functions): The i-th function within
        casting_functions will be used to transform the string read from
        the input file for the i-th identifier.
        - headers_end_identifier (string): This string is given to
        DataPreprocessor.find_skiprows() as identifier. The return value of
        DataPreprocessor.find_skiprows() is used, in this case, to find the
        number of the headers end line, say L. Then, this function will parse
        the input file for ocurrences of the provided identifiers from the first
        line through the L-th line. If it is not set, then L is set to the last
        line of the input file.

        For every string within identifiers, say, the i-th identifier, this
        function will parse the lines in filepath, from the first line
        to the L-th line, to find any line which starts with such string.
        If it is found, then ocurrences of identifier_delimiter are searched
        through such line. The line is split into sublines using the ocurrences
        of identifier_delimiter. The last subline will be transformed under
        casting_functions[i] and added to a dictionary 'result', with key equal
        to identifier. If for some identifier there's no line within
        filepath which starts with it, or there's some line that starts
        with it but there's no ocurrences of identifier_delimiter in it, then
        no entry is added to result with such key. This function returns the
        'result' dictionary.
        """

        headers_endline = -1
        if headers_end_identifier is not None:
            htype.check_type(
                headers_end_identifier,
                str,
                exception_message=htype.generate_exception_message(
                    "DataPreprocessor._parse_headers", 17819
                ),
            )
            headers_endline = DataPreprocessor.find_skiprows(
                filepath, headers_end_identifier
            )
        # If headers_endline ends up being -1, then this function will search
        # through the whole input file to find ocurrences of any given identifier

        result = {}
        for i in range(len(identifiers)):
            with open(filepath, "r") as file:
                cont = 0
                line = file.readline()
                while cont != headers_endline and line:
                    line = file.readline()
                    cont += 1
                    if line.startswith(identifiers[i]):
                        aux = line.strip().split(identifier_delimiter)[-1]
                        if aux != "":
                            result[identifiers[i]] = casting_functions[i](aux)
                            # Stop the search (the while loop)
                            # only if a value was successfully
                            # added to result
                            break

        if return_skiprows:
            return result, headers_endline
        else:
            return result

    @staticmethod
    def remove_non_alpha_bytes(input):
        """This static method gets the following mandatory positional argument:

        - byte_data (bytes): Chain of bytes

        This method returns an string which is the result of removing every
        non-alphabetic character from the given bytes chain and join the
        rest of them to create an string. To do so, this method converts the
        chain of bytes to a chain of characters, using the built-in method chr()
        (which uses utf-8 encoding) and then uses the method isalpha() to decide
        whether each character is alphabetic or not."""

        htype.check_type(
            input,
            bytes,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.remove_non_alpha_bytes", 47131
            ),
        )
        filtered_alphabetic_characters = list(
            filter(lambda b: b.isalpha(), [chr(b) for b in input])
        )
        return "".join(filtered_alphabetic_characters)

    @staticmethod
    def _extract_tek_wfm_metadata(filepath):
        """This helper static method has a fixed purposed and is not tunable via 
        input parameters. This method gets the following mandatory positional 
        argument:

        - filepath (str): Path to a binary file which is interpreted to have the
        Tektronix WFM file format. Its extension must be equal to '.wfm'.

        This method reads the first 838 bytes of the input WFM file. These bytes
        contain the metadata for the stored fast frames by the Tektronix oscilloscope
        (See the Tektronix Reference Waveform File Format Instructions). This method
        performs some consistency checks and then returns two dictionaries which host
        meta-data for the FastFrame set which is stored in the provided filepath. The
        first returned dictionary comprises the following key-value pairs:

            - 'Horizontal Units' (resp. 'Vertical Units') (string): Indicates the
            horizontal (vertical) units of the waveforms which are stored in the
            provided filepath.
            - 'Sample Interval' (scalar float): Indicates the horizontal resolution
            of the waveforms in the provided filepath.
            - 'Record Length' (scalar integer): Indicates the number of points
            per waveform for each waveform in the provided filepath.
            - 'FastFrame Count' (scalar integer): Indicates the number of waveforms
            that are stored in the provided filepath.

        The second returned dictionary, comprises the following key-value pairs:

            - 'curve_buffer_offset' (scalar integer): The number of bytes from the
            beginning of the file to the start of the curve buffer.
            - 'vscale' (scalar float): This one matches the number of units (p.e.
            number of volts) per least significant bit (LSB).
            - 'voffset' (scalar float): This value matches the distance in vertical
            units from the dimension zero value to the true zero value.
            - 'tstart' (scalar float): Trigger position.
            - 'tfrac[0]' (scalar float): This value matches the fraction of the
            sample time from the trigger time stamp to the next sample, for the first
            frame of the FastFrame set.
            - 'tdatefrac[0]' (scalar float): The fraction of the second when the
            trigger occurred.
            - 'tdate[0]' (scalar float): GMT (in seconds from the epoch) when the
            trigger occurred.
            - 'samples_datatype' (type): Data type of the samples in the curves.
            - 'samples_no' (scalar integer): Number of samples in each (whole) curve.
            - 'pre-values_no' (scalar inteber): Number of pre-appended interpolation samples.
            - 'postvalues_no' (scalar integer): Number of post-appended interpolation samples.
        """

        htype.check_type(
            filepath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor._extract_tek_wfm_metadata", 48219
            ),
        )
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DataPreprocessor._extract_tek_wfm_metadata",
                    11539,
                    extra_info=f"Path {filepath} does not exist or is not a file.",
                )
            )
        else:
            _, extension = os.path.splitext(filepath)
            if extension != ".wfm":
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "DataPreprocessor._extract_tek_wfm_metadata",
                        42881,
                        extra_info=f"The extension of the input file must match '.wfm'.",
                    )
                )
        with open(filepath, "rb") as file:  # Binary-reading mode
            header_bytes = file.read(838)

        if len(header_bytes) != 838:
            raise cuex.WfmReadException(
                htype.generate_exception_message(
                    "DataPreprocessor._extract_tek_wfm_metadata",
                    47199,
                    extra_info="WFM header comprise 838 bytes. A different value was given.",
                )
            )
        data_buffer = {}
        data_buffer["byte_order"] = struct.unpack_from("H", header_bytes, offset=0)[
            0
        ]  # 'H' for Unsigned short (2 bytes)

        # Little-endian: Less significant bytes
        # are stored in lower memory addresses
        if data_buffer["byte_order"] == 0x0F0F:
            bo = "<"
        # Big-endian: More significant bytes
        # belong to lower memory addresses
        else:
            bo = ">"
        del data_buffer["byte_order"]

        # '8s' for an 8-bytes string
        # Also, adding here a first
        # character, either '<' or '>'
        # indicating the byte order.
        data_buffer["version"] = struct.unpack_from(bo + "8s", header_bytes, offset=2)[
            0
        ]

        # In the tekwfm proof of concept, they check that the version is equal to 3
        # (otherwise an error is raised). However, they make no use of any of the
        # memory addresses which vary from version 2 to version 3 (you can check the
        # changes from version 2 to version 3 in page 15 of the waveform file
        # reference manual). That's why I see no reason to raise an error based on
        # the version.
        del data_buffer["version"]

        data_buffer["record_type"] = struct.unpack_from(
            bo + "I", header_bytes, offset=122
        )[
            0
        ]  # 'I' for a 4-bytes unsigned int.
        if data_buffer["record_type"] != 2:
            raise cuex.WfmReadException(
                htype.generate_exception_message(
                    "DataPreprocessor._extract_tek_wfm_metadata",
                    67112,
                    extra_info="For normal YT waveforms, the record type must be 2.",
                )
            )
        del data_buffer["record_type"]

        data_buffer["imp_dim_count"] = struct.unpack_from(
            bo + "I", header_bytes, offset=114
        )[0]
        if data_buffer["imp_dim_count"] != 1:
            raise cuex.WfmReadException(
                htype.generate_exception_message(
                    "DataPreprocessor._extract_tek_wfm_metadata",
                    10925,
                    extra_info="For normal YT waveforms, the number of implicit dimensions must be 1.",
                )
            )
        del data_buffer["imp_dim_count"]

        data_buffer["exp_dim_count"] = struct.unpack_from(
            bo + "I", header_bytes, offset=118
        )[0]
        if data_buffer["exp_dim_count"] != 1:
            raise cuex.WfmReadException(
                htype.generate_exception_message(
                    "DataPreprocessor._extract_tek_wfm_metadata",
                    48108,
                    extra_info="For normal YT waveforms, the number of explicit dimensions must be 1.",
                )
            )
        del data_buffer["exp_dim_count"]

        data_buffer["exp_dim_1_type"] = struct.unpack_from(
            bo + "I", header_bytes, offset=244
        )[0]
        if data_buffer["exp_dim_1_type"] != 0:
            raise cuex.WfmReadException(
                htype.generate_exception_message(
                    "DataPreprocessor._extract_tek_wfm_metadata",
                    48108,
                    extra_info="This value must be 0, which matches the case of 'EXPLICIT_SAMPLE'.",
                )
            )
        del data_buffer["exp_dim_1_type"]

        data_buffer["time_base_1"] = struct.unpack_from(
            bo + "I", header_bytes, offset=768
        )[0]
        if data_buffer["time_base_1"] != 0:
            raise cuex.WfmReadException(
                htype.generate_exception_message(
                    "DataPreprocessor._extract_tek_wfm_metadata",
                    48108,
                    extra_info="This value must be 0, which matches the case of 'BASE_TIME'.",
                )
            )
        del data_buffer["time_base_1"]

        data_buffer["fastframe"] = struct.unpack_from(
            bo + "I", header_bytes, offset=78
        )[0]
        if data_buffer["fastframe"] != 1:
            raise cuex.WfmReadException(
                htype.generate_exception_message(
                    "DataPreprocessor._extract_tek_wfm_metadata",
                    30182,
                    extra_info="This value must be 1, which matches the case of a FastFrame set.",
                )
            )
        del data_buffer["fastframe"]

        data_buffer["imp_dim_units"] = struct.unpack_from(
            bo + "20s", header_bytes, offset=508
        )[
            0
        ]  # As an example, this value might evaluate to 's'
        data_buffer["imp_dim_units"] = DataPreprocessor.remove_non_alpha_bytes(
            data_buffer["imp_dim_units"]
        )

        data_buffer["exp_dim_units"] = struct.unpack_from(
            bo + "20s", header_bytes, offset=188
        )[
            0
        ]  # As an example, this value might evaluate to 'V'
        data_buffer["exp_dim_units"] = DataPreprocessor.remove_non_alpha_bytes(
            data_buffer["exp_dim_units"]
        )

        # The extracted value matches the number
        # number of fast frames minus 1. Adding 1
        # gives the number of fast frames in our
        # FastFrame set.
        data_buffer["Frames"] = (
            1 + struct.unpack_from(bo + "I", header_bytes, offset=72)[0]
        )

        # 'I' stands for a 4-bytes long int. This
        # one matches the number of bytes from the
        # beginning of the file to the start of the
        # curve buffer.
        # 'vscale' for Vertical Scale
        data_buffer["curve_buffer_offset"] = struct.unpack_from(
            bo + "I", header_bytes, offset=16
        )[0]

        # 'd' stands for an 8-bytes double. This
        # one matches the number of units (p.e.
        # number of volts) per least significant
        # bit (LSB).
        # 'voffset' for Vertical Offset
        data_buffer["vscale"] = struct.unpack_from(bo + "d", header_bytes, offset=168)[
            0
        ]

        # This value matches the distance in vertical
        # units from the dimension zero value to
        # the true zero value. Eventually, the
        # vertical-magnitude values are computed as
        # the curve data times this vscale plus voffset.
        # 'hscale' for Horizontal Scale
        data_buffer["voffset"] = struct.unpack_from(bo + "d", header_bytes, offset=176)[
            0
        ]

        # This value matches the sample interval
        # for the horizontal implicit dimension,
        # p.e. time per point
        data_buffer["hscale"] = struct.unpack_from(bo + "d", header_bytes, offset=488)[
            0
        ]

        data_buffer["tstart"] = struct.unpack_from(bo + "d", header_bytes, offset=496)[
            0
        ]  # Trigger position

        # Every waveform in our FastFrame set will have a 'Waveform Update Specification' object, which, in turn, comprises
        #   1) a 4-bytes unsigned long int
        #   2) an 8-bytes double
        #   3) an 8-bytes double
        #   4) an 4-bytes long
        # The following three pieces of information that we extract: 'tfrac', 'tdatefrac' and 'tdate' match the last three
        # pieces of data we enumerated, respectively, for the first fast frame in the FastFrame set (index 0). Later on,
        # somewhere else in the code, we will need to extract such information for the rest of fast frames.

        # The time from the point the trigger occurred
        # to the next data point in the waveform
        # record. This value represents the fraction
        # of the sample time (hscale) from the trigger
        # time stamp to the next sample, i.e. the
        # first point of the curve.
        data_buffer["tfrac[0]"] = struct.unpack_from(
            bo + "d", header_bytes, offset=788
        )[0]

        # The fraction of the second when the
        # trigger occurred. This one should be
        # used in combination with 'tdate'.
        data_buffer["tdatefrac[0]"] = struct.unpack_from(
            bo + "d", header_bytes, offset=796
        )[0]

        # GMT (in seconds from the epoch) when the trigger occurred.
        data_buffer["tdate[0]"] = struct.unpack_from(
            bo + "I", header_bytes, offset=804
        )[0]

        # What we encounter from offset 808 to offset 838 (in bytes) is information regarding
        # the format of the first waveform curve object. Since every waveform curve object has
        # the same size, the information we extract from this is applicable to every waveform
        # curve object that we will encounter in the curve buffer (i.e. in our FastFrame set).

        # The byte offset from the beginning of the curve buffer to the
        # first point of the record available to the oscilloscope user.
        # Every curve in the curve buffer has a (fixed) number of artifact-
        # points which are placed at the beginning of the curve, and which
        # serve for oscilloscope interpolation purposes. The same goes
        # for the end of the curve, where a number of artifact-poins are
        # appended. If we take out such pre and post artifact-points, what
        # we are left with is what we are going to call the user-accesible
        # part of the curve.
        dpre = struct.unpack_from(bo + "I", header_bytes, offset=822)[0]

        # The byte offset to the point right after the last user-accesible
        # point in the curve.
        dpost = struct.unpack_from(bo + "I", header_bytes, offset=826)[0]

        readbytes = (
            dpost - dpre
        )  # Number of bytes per (the user-accesible part of the) curve
        allbytes = struct.unpack_from(bo + "I", header_bytes, offset=830)[
            0
        ]  # Bytes per curve (including the interpolation points)

        # A code (scalar integer) which indicates the data type of the
        # values stored in the curve buffer.
        dt_code = struct.unpack_from(bo + "i", header_bytes, offset=240)[0]

        # Number of bytes per curve data point. Stands for Bytes Per Sample.
        bps = struct.unpack_from(bo + "b", header_bytes, offset=15)[0]

        if dt_code == 0 and bps == 2:
            samples_datatype = np.int16
            useable_samples_no = readbytes // 2
        elif dt_code == 1 and bps == 4:
            samples_datatype = np.int32
            useable_samples_no = readbytes // 2
        elif dt_code == 2 and bps == 4:
            samples_datatype = np.uint32
            useable_samples_no = readbytes // 4
        elif dt_code == 3 and bps == 8:
            samples_datatype = np.uint64
            useable_samples_no = readbytes // 8
        elif dt_code == 4 and bps == 4:
            samples_datatype = np.float32
            useable_samples_no = readbytes // 4
        elif dt_code == 5 and bps == 8:
            samples_datatype = np.float64
            useable_samples_no = readbytes // 8
        else:
            raise cuex.WfmReadException(
                htype.generate_exception_message(
                    "DataPreprocessor._extract_tek_wfm_metadata",
                    21230,
                    extra_info="The data-type code is not consistent with the read bytes-per-sample.",
                )
            )

        data_buffer["samples_datatype"] = (
            samples_datatype  # Data type of the samples in the curves
        )
        data_buffer["useable_samples_no"] = (
            useable_samples_no  # Number of samples in the user-accesible part of each curve
        )
        data_buffer["samples_no"] = (
            allbytes // bps
        )  # Number of samples in each (whole) curve
        data_buffer["pre-values_no"] = (
            dpre // bps
        )  # Number of pre-appended interpolation samples
        data_buffer["post-values_no"] = (
            allbytes - dpost
        ) // bps  # Number of post-appended interpolation samples

        main_extraction = {}
        main_extraction["Horizontal Units"] = data_buffer["imp_dim_units"]
        main_extraction["Vertical Units"] = data_buffer["exp_dim_units"]
        main_extraction["Sample Interval"] = data_buffer["hscale"]
        main_extraction["Record Length"] = data_buffer["useable_samples_no"]
        main_extraction["FastFrame Count"] = data_buffer["Frames"]

        supplementary_extraction = {}
        supplementary_extraction["curve_buffer_offset"] = data_buffer[
            "curve_buffer_offset"
        ]
        supplementary_extraction["vscale"] = data_buffer["vscale"]
        supplementary_extraction["voffset"] = data_buffer["voffset"]
        supplementary_extraction["tstart"] = data_buffer["tstart"]
        supplementary_extraction["tfrac[0]"] = data_buffer["tfrac[0]"]
        supplementary_extraction["tdatefrac[0]"] = data_buffer["tdatefrac[0]"]
        supplementary_extraction["tdate[0]"] = data_buffer["tdate[0]"]
        supplementary_extraction["samples_datatype"] = data_buffer["samples_datatype"]
        supplementary_extraction["samples_no"] = data_buffer["samples_no"]
        supplementary_extraction["pre-values_no"] = data_buffer["pre-values_no"]
        supplementary_extraction["post-values_no"] = data_buffer["post-values_no"]

        # For now, the only reason for this dictionary splitting is simply to make the first returned
        # dictionary resemble that of the ASCII case, i.e. the one returned by
        # DataPreprocessor._parse_headers(). This might not be the optimal way and may vary in the future.

        return main_extraction, supplementary_extraction

    @staticmethod
    def get_metadata(
        filepath,
        *parameters_identifiers,
        get_creation_date=False,
        verbose=True,
        is_ASCII=True,
        skiprows_identifier="TIME,",
        parameters_delimiter=",",
        casting_functions=None
    ):
        """This static method gets the following mandatory positional argument:

        - filepath (string): Path to the file whose meta-data will be retrieved.

        This function gets the following optional positional arguments:

        - parameters_identifiers (tuple of strings): These parameters only make
        a difference if is_ASCII is True. In such case, they are given to
        DataPreprocessor._parse_headers() as identifiers. Each one is considered
        to be the string which precedes the value of a parameter of interest
        within the input file headers.

        This function also gets the following optional keyword arguments:

        - get_creation_date (bool): If True, the creation date of the input file
        is added to the resulting dictionary under the key 'creation_date'. The
        associated value is an string which follows the format 'YYYY-MM-DD HH:MM:SS'.
        If False, no extra entry is added to the resulting dictionary.

        - verbose (boolean): Whether to print functioning-related messages.

        - is_ASCII (bool): Indicates whether the input file should be interpreted
        as an ASCII file, or as a binary file (in the Tektronix .WFM file format).
        This parameter determines whether this function delegates the meta-data
        extraction to DataPreprocessor._parse_headers() (is_ASCII is True) or
        to DataPreprocessor._extract_tek_wfm_metadata() (is_ASCII is False).

        - skiprows_identifier (string): This parameter only makes a difference
        if is_ASCII is True. In such case, it is passed to DataPreprocessor._parse_headers()
        as headers_end_identifier, which, in turn, passes it to
        DataPreprocessor.find_skiprows() as identifier. This string is used to
        identify the line which immediately precedes the data columns in an
        ASCII data file, say the L-th line. identifier must be defined so
        that the L-th line starts with identifier, i.e. identifier==header[0:N]
        must evaluate to True, for some 1<=N<=len(header). This function will
        also parse the input file for occurrences of the provided,
        parameters_identifiers, from the first line through the L-th line.

        - parameters_delimiter (string): This parameter only makes a difference if
        is_ASCII is True. In such case, it is given to DataPreprocessor._parse_headers()
        as identifier_delimiter. This string is used to separate each identifier
        from its value.

        - casting_functions (tuple/list of functions): This parameter only makes a
        difference if is_ASCII is True. In such case, it is given to
        DataPreprocessor._parse_headers() as casting_functions. The i-th function
        within casting_functions will be used to transform the string read from the
        input file for the i-th parameter identifier.

        This static method

            - receives a path to an input file, which could be either ASCII or binary
            (in the Tektronix .WFM file format),
            - extracts some meta-data from it which is partially returned by this method
            as a dictionary. Such extraction is carried out by
            DataPreprocessor._parse_headers() or DataPreprocessor._extract_tek_wfm_metadata()
            for the case where is_ASCII is True or False, respectively.
            - optionally, if get_creation_date is True, the creation date of the input
            file is also retrieved and added to the resulting dictionary under the key
            'creation_date'.

        For DataPreprocessor.generate_meas_config_files() to work properly, the dictionary
        returned by this method must, at least, contain the following keys: 'Horizontal Units',
        'Vertical Units', 'Sample Interval', 'Record Length', 'FastFrame Count' and 
        'creation_date'.
        """

        htype.check_type(
            filepath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.get_metadata", 17219
            ),
        )
        for i in range(len(parameters_identifiers)):
            htype.check_type(
                parameters_identifiers[i],
                str,
                exception_message=htype.generate_exception_message(
                    "DataPreprocessor.get_metadata", 43177
                ),
            )

        htype.check_type(
            get_creation_date,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.get_metadata", 11280
            ),
        )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.get_metadata", 56912
            ),
        )
        htype.check_type(
            is_ASCII,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.get_metadata", 67189
            ),
        )
        htype.check_type(
            skiprows_identifier,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.get_metadata", 99170
            ),
        )
        htype.check_type(
            parameters_delimiter,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.get_metadata", 12851
            ),
        )
        casting_functions_ = [lambda x: x for y in parameters_identifiers]
        if casting_functions is not None:
            htype.check_type(
                casting_functions,
                tuple,
                list,
                exception_message=htype.generate_exception_message(
                    "DataPreprocessor.get_metadata", 41470
                ),
            )
            for i in range(len(casting_functions)):
                if not callable(casting_functions[i]):
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message(
                            "DataPreprocessor.get_metadata", 55415
                        )
                    )
            if len(casting_functions) != len(parameters_identifiers):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "DataPreprocessor.get_metadata", 99167
                    )
                )
            casting_functions_ = casting_functions

        result = {}
        if is_ASCII:
            parameters, _ = DataPreprocessor._parse_headers(
                filepath,
                *parameters_identifiers,
                identifier_delimiter=parameters_delimiter,
                casting_functions=casting_functions_,
                headers_end_identifier=skiprows_identifier,
                return_skiprows=True,
            )
        else:
            parameters, _ = (
                DataPreprocessor._extract_tek_wfm_metadata(filepath)
            )
        result.update(parameters)

        if get_creation_date:
            result["creation_date"] = DataPreprocessor.get_str_creation_date(filepath)

        if verbose:
            print(
                f"In function DataPreprocessor.get_metadata(): Succesfully processed {filepath}"
            )

        return result

    @staticmethod
    def rename_file(filepath, new_filename, overwrite=False, verbose=False):
        """This static method gets the following mandatory positional arguments:

        - filepath (string): Full path to the file which should be renamed.
        - new_filename (string): New file name.

        This method also gets the following optional keyword argument:

        - overwrite (bool)
        - verbose (boolean): Whether to print functioning-related messages.

        If overwrite is False, and there's already a file whose name matches
        new_filename within the same directory as filepath, then this method
        does nothing. If overwrite is True, then the file whose path matches
        filepath is renamed to new_filename. In this case, overwritting
        may occur. This method returns the path to the renamed file."""

        htype.check_type(
            filepath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.rename_file", 35673
            ),
        )
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DataPreprocessor.rename_file",
                    46110,
                    extra_info=f"Path {filepath} does not exist or is not a file.",
                )
            )
        htype.check_type(
            new_filename,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.rename_file", 88007
            ),
        )
        htype.check_type(
            overwrite,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.rename_file", 12432
            ),
        )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.rename_file", 71900
            ),
        )

        folderpath = os.path.split(filepath)[0]
        new_filepath = os.path.join(folderpath, new_filename)

        if os.path.isfile(new_filepath):
            if overwrite:
                os.rename(filepath, new_filepath)
                if verbose:
                    print(
                        htype.generate_exception_message(
                            "DataPreprocessor.rename_file",
                            47189,
                            extra_info=f"File {new_filepath} already exists. It has been overwritten.",
                        )
                    )
            else:
                raise FileExistsError(
                    htype.generate_exception_message(
                        "DataPreprocessor.rename_file",
                        95995,
                        extra_info=f"File {new_filepath} already exists. It cannot be overwritten. Renaming was not performed.",
                    )
                )
        else:
            os.rename(filepath, new_filepath)
        return new_filepath

    @staticmethod
    def query_fields_in_dictionary(input_dict, default_dict=None):
        """This static method get the following mandatory positional argument:

        - input_dict (dictionary): Its keys must be strings and its values must be
        callable.
        - default_dict (None or dictionary)

        This method returns a dictionary, say result, whose keys match the keys of
        input_dict. Let us refer to the string keys of input_dict as fields. For a
        given field 'field', this static method asks the user to input an answer.
        If such answer is emtpy, i.e. '', and default_dict does not contain an entry
        for the key 'field', then an exception is raised. If such answer matches ''
        and default_dict contains an entry for the key 'field', then
        default_dict['field'] is added to result under the key 'field'. If such
        answer do not match '', then it is passed to the callable which is associated
        to the given field, up to the input_dict. The result of such callable is
        added to result, under the key 'field'. After iterating over every field
        in the input dictionary, this static method returns the dictionary 'result'."""

        htype.check_type(
            input_dict,
            dict,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.query_fields_in_dictionary", 52810
            ),
        )
        for key in input_dict.keys():
            htype.check_type(
                key,
                str,
                exception_message=htype.generate_exception_message(
                    "DataPreprocessor.query_fields_in_dictionary", 46180
                ),
            )
            if not callable(input_dict[key]):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "DataPreprocessor.query_fields_in_dictionary", 55832
                    )
                )
        default_dict_ = {key: None for (key, value) in input_dict.items()}
        if default_dict is not None:
            htype.check_type(
                default_dict,
                dict,
                exception_message=htype.generate_exception_message(
                    "DataPreprocessor.query_fields_in_dictionary", 47224
                ),
            )
            default_dict_.update(default_dict)

        result = {}
        for key in input_dict.keys():
            query_ok = False
            while not query_ok:
                try:
                    result[key] = DataPreprocessor.query_field(
                        key,
                        casting_function=input_dict[key],
                        substitution_trigger="",
                        substitute=default_dict_[key],
                    )
                    query_ok = True
                except cuex.NoAvailableData:
                    print(
                        htype.generate_exception_message(
                            "DataPreprocessor.query_fields_in_dictionary",
                            47188,
                            extra_info=f"There's no default value for {key}. Please, provide one.",
                        )
                    )
        return result

    @staticmethod
    def query_field(
        field_name, casting_function=str, substitution_trigger="", substitute=None
    ):
        """This static method gets the following mandatory positional argument:

         - field_name (string)

        And the following optional keyword argument:

        - casting_function (callable):
        - substitution_trigger (string):
        - substitute (None or object*): If it is not None, its type must match
        the one that is induced by the specified casting function, i.e.
        casting_function(substitute)==substitute must evaluate to True.

        This static method queries the user for the value associated to the name
        field_name. If the string given by the user matches substitution_trigger
        and substitute is None, then this method raises an exception. If the string
        given by the user matches substitution_trigger and substitute is defined,
        then this method returns substitute. If the string given by the user does
        not match substitution_trigger, then the string given by the user is
        transformed under casting_function, and returned by this method."""

        htype.check_type(
            field_name,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.query_field", 89813
            ),
        )
        if not callable(casting_function):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("DataPreprocessor.query_field", 91631)
            )
        htype.check_type(
            substitution_trigger,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.query_field", 36199
            ),
        )
        if substitute is not None:
            try:
                if casting_function(substitute) != substitute:
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message(
                            "DataPreprocessor.query_field", 91631
                        )
                    )
            except ValueError:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "DataPreprocessor.query_field", 77549
                    )
                )

        keep_asking = True
        while keep_asking:
            input_is_ok = False
            while not input_is_ok:
                user_input = input(f"Enter a value for '{field_name}':")
                if user_input == substitution_trigger:
                    if substitute is None:
                        raise cuex.NoAvailableData(
                            htype.generate_exception_message(
                                "DataPreprocessor.query_field",
                                23120,
                                extra_info="Default value was not specified.",
                            )
                        )
                    else:
                        user_input = substitute
                        result = substitute
                        input_is_ok = True
                else:
                    try:
                        result = casting_function(user_input)
                        input_is_ok = True
                    except ValueError:
                        print(
                            htype.generate_exception_message(
                                "DataPreprocessor.query_field",
                                19532,
                                extra_info=f"The specified value do not comply with the expected type ({casting_function})",
                            )
                        )
            keep_asking = not DataPreprocessor.yes_no_translator(
                input(f"The entered value is {user_input}. ¿Is it OK? (y/n)")
            )
        return result

    @staticmethod
    def yes_no_translator(user_input):
        """This static method gets the following mandatory positional argument:

        - user_input (string)

        This method returns True if user_input is equal to any value within
        ('y', 'Y', 'ye', 'Ye', 'yE', 'YE', 'yes', 'Yes', 'yEs', 'yeS', 'YEs', 'YeS', 'yES', 'YES').
        It returns False otherwise."""

        htype.check_type(
            user_input,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.yes_no_translator", 89813
            ),
        )
        if user_input in (
            "y",
            "Y",
            "ye",
            "Ye",
            "yE",
            "YE",
            "yes",
            "Yes",
            "yEs",
            "yeS",
            "YEs",
            "YeS",
            "yES",
            "YES",
        ):
            return True
        else:
            return False

    @staticmethod
    def generate_json_file(input_dictionary, output_filepath):
        """This method takes the following mandatory positional arguments:

        - input_dictionary (dictionary)
        - output_filepath (string)

        This static method writes the given dictionary, input_dictionary, to a
        json file whose filepath matches output_filepath. If the specified
        output filepath do not match an already existing file, then such file
        is created. If the specified output filepath matches an already existing
        file, then such file is overwritten."""

        htype.check_type(
            input_dictionary,
            dict,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.generate_json_file", 31965
            ),
        )
        htype.check_type(
            output_filepath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.generate_json_file", 31900
            ),
        )
        with open(output_filepath, "w") as file:
            json.dump(input_dictionary, file, indent="\t")
        return

    @staticmethod
    def get_str_creation_date(filepath):
        """This static method gets the following mandatory positional argument:

        - filepath (string)

        This static method returns the creation date of the file whose filepath
        matches the given filepath, as a string in the format YYYY-MM-DD hh:mm:ss."""

        htype.check_type(
            filepath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.get_str_creation_date", 46223
            ),
        )

        if os.path.exists(filepath):
            return datetime.datetime.fromtimestamp(os.path.getctime(filepath)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DataPreprocessor.get_str_creation_date", 62018
                )
            )

    @staticmethod
    def ask_for_unique_query_splitting(func):
        def wrapper(*args):
            print(
                "Are there any fields within the following dictionary whose value for all of the measurements is the same?"
            )
            print("If so, mark them so that they will be queried just once.")
            return func(*args)

        return wrapper

    @staticmethod
    def query_tuple_of_semipositive_integers(n_max):
        """This static method gets the following mandatory positional argument:

        - n_max (positive integer)

        This method iteratively asks the user for a semipositive integer and
        whether she/he wants to provide another semipositive integer. Eventually,
        the user could provide a total of n semipositive integers, with n<=n_max."""

        htype.check_type(
            n_max,
            int,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.query_tuple_of_semipositive_integers", 51812
            ),
        )
        if n_max < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.query_tuple_of_semipositive_integers",
                    11012,
                    extra_info="One integer must be queried at least.",
                )
            )

        keep_asking = True
        while keep_asking:
            user_input = input(
                "Provide a comma-separated tuple of semipositive integer values. Any value which cannot be casted to integer will be ignored. The absolute value of every integer-casted input will be computed."
            )
            result = []
            for sample in user_input.split(","):
                try:
                    result.append(abs(int(sample)))
                except Exception:
                    pass
            result = tuple(
                result[:n_max]
            )  # Comply with the provided maximum number of samples
            keep_asking = not DataPreprocessor.yes_no_translator(
                input(f"The processed input is {result}. Is it OK? (y/n)")
            )
        return result

    @staticmethod
    @ask_for_unique_query_splitting
    def query_dictionary_splitting(input_dict):
        """This static method has the following mandatory positional argument:

        - input_dict (dictionary)

        This method queries the user for a subset of the keys of the input
        dictionary. This method returns two dictionaries, so that the union
        of both is equal to the input dictionary. The first returned dictionary
        includes the key-value pairs of the input dictionary whose keys belong
        to the keys subset provided by the user.
        """

        htype.check_type(
            input_dict,
            dict,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.query_dictionary_splitting", 37119
            ),
        )
        i_gen = DataPreprocessor.integer_generator()
        ordered_dict = {next(i_gen): key for (key, value) in input_dict.items()}

        print("The input dictionary is:")
        for key in ordered_dict.keys():
            print(f"{key}: ({ordered_dict[key]}, {input_dict[ordered_dict[key]]})")

        print(
            "Input the integer-labels of the entries which you would like to split apart."
        )
        splitted_keys = set(
            DataPreprocessor.query_tuple_of_semipositive_integers(len(input_dict))
        )

        aux = {}
        for key in ordered_dict.keys():
            if key in splitted_keys:
                aux[key] = ordered_dict[key]

        first_split = {value: input_dict[value] for (key, value) in aux.items()}
        second_split = input_dict
        for key in first_split.keys():
            del second_split[key]

        return first_split, second_split

    @staticmethod
    def path_is_contained_in_dir(path, dir_path):
        """This function gets the following positional arguments:

        - path (string): It must be a path which exists.
        I.e. os.path.exists(path) must evaluate to True.
        - dir_path (string): It must be a path to an existing
        directory. I.e. os.path.isdir(dir_path) must evaluate
        to True.

        This function returns True if the file/directory pointed
        to by path is contained, at an arbitrary depth, within
        the directory pointed to by dir_path"""

        htype.check_type(
            path,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.path_is_contained_in_dir", 25500
            ),
        )
        htype.check_type(
            dir_path,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.path_is_contained_in_dir", 39223
            ),
        )
        if not os.path.exists(path):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.path_is_contained_in_dir", 23589
                )
            )
        if not os.path.isdir(dir_path):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.path_is_contained_in_dir", 49032
                )
            )

        # If instead of os.path.samefile() we use simple
        # string comparation, paths which point to the same
        # directory or file may appear as different paths.
        # For example, while os.path.samefile('/a/b/c', '/a/b/c/')
        # evaluates to True, '/a/b/c'=='/a/b/c/' does not.
        # The only difference is the last slash.
        return os.path.samefile(
            os.path.commonpath([dir_path, path]), dir_path
        )  

    @staticmethod
    def try_grabbing_from_json(catalog, path_to_json_file, verbose=False):
        """This function gets the following positional arguments:

        - catalog (dictionary): Its keys must be strings and the type of its values
        must be 'type'. P.e. type(int)==type evaluates to True.
        - path_to_json_file (string): Absolute path which must point to a file whose
        extension matches '.json'.

        This function gets the following keyword arguments:

        - verbose (boolean): Whether to print functioning-related messages.

        For each key in catalog, this function checks whether the key is present in
        the dictionary which is loaded from the json file pointed by path_to_json_file.
        If it is present, and the type of its value matches the one signalled by the
        catalog, the key-value pair is added to the output dictionary, and the key
        is removed from the catalog. If the key is not present in the loaded dictionary,
        or it is present but its type is not suitable according to catalog, then the key
        remains in the catalog and nothing is added to the output dictionary. There is
        one exception to these rules: if the key is present in the loaded dictionary but
        its value is of type int (resp. float) when the expected type is float (resp. int),
        then the value is casted to the expected type and added to the output dictionary.
        This function returns the output dictionary and the remaining catalog, in such
        order.
        """

        htype.check_type(
            catalog,
            dict,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.try_grabbing_from_json", 48501
            ),
        )
        for key in catalog.keys():

            htype.check_type(
                key,
                str,
                exception_message=htype.generate_exception_message(
                    "DataPreprocessor.try_grabbing_from_json", 29757
                ),
            )
            htype.check_type(
                catalog[key],
                type,
                exception_message=htype.generate_exception_message(
                    "DataPreprocessor.try_grabbing_from_json", 44642
                ),
            )
        htype.check_type(
            path_to_json_file,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.try_grabbing_from_json", 62236
            ),
        )
        if not os.path.isfile(path_to_json_file):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DataPreprocessor.try_grabbing_from_json",
                    37189,
                    extra_info=f"The path {path_to_json_file} does not exist or is not a file.",
                )
            )
        elif not path_to_json_file.endswith(".json"):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.try_grabbing_from_json",
                    37756,
                    extra_info=f"File {path_to_json_file} must end with '.json' extension.",
                )
            )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.try_grabbing_from_json", 88675
            ),
        )

        output_dictionary = {}
        remaining_catalog = catalog.copy()  # Otherwise we'll get a view to catalog

        with open(path_to_json_file, "r") as file:
            try:
                input_dictionary = json.load(file)
            except json.JSONDecodeError:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "DataPreprocessor.try_grabbing_from_json",
                        97781,
                        extra_info=f"File {path_to_json_file} must contain a valid json object.",
                    )
                )
        for key in catalog.keys():
            # This key of catalog was
            # found in the input dictionary
            if key in input_dictionary.keys():

                c1 = catalog[key] == int and type(input_dictionary[key]) == float
                c2 = catalog[key] == float and type(input_dictionary[key]) == int

                # Implement the int<-->float exception
                if c1 or c2:

                    # Cast the value to the expected type
                    output_dictionary[key] = catalog[key](input_dictionary[key])
                    del remaining_catalog[key]

                    if verbose:
                        print(
                            htype.generate_exception_message(
                                "DataPreprocessor.try_grabbing_from_json",
                                42481,
                                extra_info=f"A candidate for key '{key}' was found with type ({type(input_dictionary[key])}). The candidate has been casted to the expected type ({catalog[key]}).",
                            )
                        )

                else:
                    try:
                        htype.check_type(
                            input_dictionary[key],
                            catalog[key],
                            exception_message=htype.generate_exception_message(
                                "DataPreprocessor.try_grabbing_from_json", 35881
                            ),
                        )
                    # Although the key was found, its
                    # value does not have a suitable type
                    # cuex.TypeException is the exception raised
                    # by htype.check_type if types do not match.
                    except cuex.TypeException:
                        if verbose:
                            print(
                                htype.generate_exception_message(
                                    "DataPreprocessor.try_grabbing_from_json",
                                    58548,
                                    extra_info=f"A candidate for key '{key}' was found but its type does not match the required one. The candidate has been ignored.",
                                )
                            )
                        continue

                    # The found candidate has a suitable type
                    # Actually add it to the output dictionary
                    else:
                        output_dictionary[key] = input_dictionary[key]
                        del remaining_catalog[key]

        return output_dictionary, remaining_catalog

    @staticmethod
    def count_folders(
        input_folderpath, ignore_hidden_folders=True, return_foldernames=False
    ):
        """This function gets the following positional argument:

        - input_folderpath (string): Path to an existing folder

        This function gets the following keyword argument:

        - ignore_hidden_folders (boolean): Whether to ignore hidden
        folders, i.e. folders whose name starts with a dot ('.').
        - return_foldernames (boolean): Whether to return the names
        of the folders, in addition to the number of folders.

        This function gets the path to a folder, and returns the
        number of sub-folders within that folder. If return_foldernames
        is set to True, then, in addition to the number of sub-folders,
        this function also returns the names of such sub-folders, i.e.
        a list of strings."""

        htype.check_type(
            input_folderpath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.count_folders", 27138
            ),
        )
        htype.check_type(
            ignore_hidden_folders,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.count_folders", 32748
            ),
        )
        htype.check_type(
            return_foldernames,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.count_folders", 98900
            ),
        )
        if not os.path.isdir(input_folderpath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DataPreprocessor.count_folders", 12972
                )
            )
        if ignore_hidden_folders:
            aux = [
                name
                for name in os.listdir(input_folderpath)
                if os.path.isdir(os.path.join(input_folderpath, name))
                and not name.startswith(".")
            ]
        else:
            aux = [
                name
                for name in os.listdir(input_folderpath)
                if os.path.isdir(os.path.join(input_folderpath, name))
            ]

        if return_foldernames:
            return len(aux), aux
        else:
            return len(aux)

    @staticmethod
    def count_files_by_extension_in_folder(
        input_folderpath, 
        extension,
        count_matches=True,
        ignore_hidden_files=True, 
        return_filenames=False
    ):
        """This function gets the following positional arguments:

        - input_folderpath (string): Path to an existing folder.
        - extension (string): Extension of the files which will
        contribute to the count, or which will be excluded from 
        it, depending on the value given to the 'count_matches' 
        parameter. 
        - count_matches (bool): Whether to count the files whose
        extension matches the given one. If False, then the
        files which contribute to the count are those whose 
        extension does not match the given one. I.e. if 
        count_matches is True (resp. False), the file names, 
        say x, that will contribute to the count are those for 
        which x.endswith('.'+extension) evaluates to True 
        (resp. False).

        This function gets the following keyword arguments:

        - ignore_hidden_files (boolean): Whether to ignore hidden
        files, i.e. files whose name starts with a dot ('.').
        - return_filenames (boolean): Whether to return the names
        of the files which contributed to the count, in addition 
        to the number of files.

        This function gets the path to a folder, and returns the
        number of files whose extension matches, or not, the given 
        one, up to the count_matches parameter. If return_filenames 
        is True, then this function also returns the names of such 
        files, i.e. a list of strings."""

        htype.check_type(
            input_folderpath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.count_files_by_extension_in_folder", 67390
            ),
        )
        if not os.path.isdir(input_folderpath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DataPreprocessor.count_files_by_extension_in_folder", 66829
                )
            )
        htype.check_type(
            extension,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.count_files_by_extension_in_folder", 78573
            ),
        )
        htype.check_type(
            count_matches,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.count_files_by_extension_in_folder", 46527
            ),
        )
        htype.check_type(
            ignore_hidden_files,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.count_files_by_extension_in_folder", 23241
            ),
        )
        htype.check_type(
            return_filenames,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.count_files_by_extension_in_folder", 13523
            ),
        )

        if ignore_hidden_files:
            candidates = [
                filename
                for filename in os.listdir(input_folderpath)
                if not filename.startswith(".") and 
                os.path.isfile(os.path.join(input_folderpath, filename))
            ]
        else:
            candidates = [
                filename
                for filename in os.listdir(input_folderpath)
                if os.path.isfile(os.path.join(input_folderpath, filename))
            ]

        if count_matches:
            aux = [
                filename
                for filename in candidates
                if filename.endswith("." + extension)
            ]
        else:
            aux = [
                filename
                for filename in candidates
                if not filename.endswith("." + extension)
            ]

        if return_filenames:
            return len(aux), aux
        else:
            return len(aux)

    @staticmethod
    def check_structure_of_input_folder(
        input_folderpath,
        subfolders_no=7,
        json_files_no_at_2nd_level=1,
        json_files_no_at_3rd_level=1,
        non_json_files_no_at_2nd_and_3rd_level=18,
    ):
        """This function gets the following positional argument:

        - input_folderpath (string): Path to the folder where the input
        data is hosted.

        This function also gets the following keyword arguments:

        - subfolders_no (positive integer): It must match the number
        of folders within the given input folder. On top of that, the
        names of these sub-folders must be set so that for each integer
        in [1, ..., subfolders_no], say i, there is at least one sub-
        folder that contains the string str(i).
        - json_files_no_at_2nd_level (positive integer): It must match
        the number of json files at every second level. I.e. any folder
        within the given input folder must contain exactly this number
        of json files.
        - json_files_no_at_3rd_level (positive integer): It must match
        the number of json files at every third level. I.e. any folder
        within any folder within the given input folder must contain
        exactly this number of json files.
        - non_json_files_no_at_2nd_and_3rd_level (positive integer): It 
        must match the number of non-json files at every second or 
        third level. I.e. any folder within the given input folder, 
        or any folder within any folder within the given input folder, 
        must contain exactly this number of non-json files.

        This function gets the path to a folder, and checks that the
        file structure within that folder follows the expected pattern.
        This function returns two objects. The first one is a list of 
        strings, each of which is a warning, i.e. a description of 
        a deviation from the expected file structure. If the file 
        structure is well-formed, then this returned list is empty.
        The second returned object is also a list of strings. In this
        case, each string is a path to a folder where the number of
        json files does match the expected number. These folders are
        considered to be analysable."""

        htype.check_type(
            input_folderpath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.check_structure_of_input_folder", 74528
            ),
        )
        if not os.path.isdir(input_folderpath):

            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DataPreprocessor.check_structure_of_input_folder", 22434
                )
            )
        htype.check_type(
            subfolders_no,
            int,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.check_structure_of_input_folder", 23452
            ),
        )
        if subfolders_no < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.check_structure_of_input_folder", 45244
                )
            )
        htype.check_type(
            json_files_no_at_2nd_level,
            int,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.check_structure_of_input_folder", 95821
            ),
        )
        if json_files_no_at_2nd_level < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.check_structure_of_input_folder", 36144
                )
            )
        htype.check_type(
            json_files_no_at_3rd_level,
            int,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.check_structure_of_input_folder", 47829
            ),
        )
        if json_files_no_at_3rd_level < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.check_structure_of_input_folder", 12311
                )
            )
        htype.check_type(
            non_json_files_no_at_2nd_and_3rd_level,
            int,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.check_structure_of_input_folder", 45829
            ),
        )
        if non_json_files_no_at_2nd_and_3rd_level < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.check_structure_of_input_folder", 58821
                )
            )
        
        warnings = []
        analysable_folderpaths = []

        folders_no, folders_names = DataPreprocessor.count_folders(
            input_folderpath, ignore_hidden_folders=True, return_foldernames=True
        )
        folders_names.sort()

        if folders_no != subfolders_no:
            warnings.append(
                f"Expected {subfolders_no} sub-folder(s) in {input_folderpath}, but {folders_no} were found."
            )

        folders_names_copy = folders_names.copy()

        # In reversed order because we'll be
        # deleting elements from the list
        for i in reversed(range(len(folders_names_copy))):
            for j in range(1, subfolders_no + 1):

                if str(j) in folders_names_copy[i]:
                    del folders_names_copy[i]
                    break

        # Means that there is one or more
        # sub-folders which do not contain
        # one of the expected sub-strings
        # (str(i), for some i=1,...,subfolders_no)
        if len(folders_names_copy) != 0:
            warnings.append(
                f"The following sub-folder(s) of the given root folder are not integer-labelled: {folders_names_copy}"
            )

        for folder_name in folders_names:
            aux_warnings, aux_analysable_folderpaths = DataPreprocessor.__check_well_formedness_of_subfolder(
                os.path.join(input_folderpath, folder_name),
                json_files_no_at_1st_level=json_files_no_at_2nd_level,
                json_files_no_at_2nd_level=json_files_no_at_3rd_level,
                non_json_files_no_at_1st_and_2nd_level=non_json_files_no_at_2nd_and_3rd_level,
            )
            warnings += aux_warnings
            analysable_folderpaths += aux_analysable_folderpaths
        
        return warnings, analysable_folderpaths

    @staticmethod
    def __check_well_formedness_of_subfolder(
        input_folderpath,
        json_files_no_at_1st_level=1,
        json_files_no_at_2nd_level=1,
        non_json_files_no_at_1st_and_2nd_level=18,
    ):
        """This static method is a helper method which should only be
        called by the method check_structure_of_input_folder().
        No type-checking is performed on the arguments of this method.
        This method gets the following positional argument:

        - input_folderpath (string): Path to an existing folder.

        This function also gets the following keyword arguments:

        - json_files_no_at_1st_level (positive integer): It must match
        the number of json files at the first level. I.e. the given
        input folder must contain exactly this number of json files.
        - json_files_no_at_2nd_level (positive integer): It must match
        the number of json files at every second level. I.e. any folder
        within the given input folder must contain exactly this number
        of json files.
        - non_json_files_no_at_1st_and_2nd_level (positive integer): It
        must match the number of non-json files at the first level and
        every second level. I.e. the given folder and any folder within
        it, must contain exactly this number of non-json files.

        This function gets the path to a folder, and checks that the
        file structure within that folder follows the expected pattern.
        This function returns two objects. The first one is a list of 
        strings, each of which is a warning, i.e. a description of 
        a deviation from the expected file structure. If the file 
        structure is well-formed, then this returned list is empty.
        The second returned object is also a list of strings. In this
        case, each string is a path to a folder where the number of
        json files does match the expected number. These folders are
        considered to be analysable."""

        warnings = []
        analysable_folderpaths = []

        aux_json_files_no = DataPreprocessor.count_files_by_extension_in_folder(
            input_folderpath, 
            "json",
            count_matches=True,
            ignore_hidden_files=True, 
            return_filenames=False
        )
        if aux_json_files_no != json_files_no_at_1st_level:
            warnings.append(
                f"Expected {json_files_no_at_1st_level} json file(s) in {input_folderpath}, but {aux_json_files_no} were found."
            )
        else:
            analysable_folderpaths.append(input_folderpath)

        aux_non_json_files_no = DataPreprocessor.count_files_by_extension_in_folder(
            input_folderpath, 
            "json", 
            count_matches=False,
            ignore_hidden_files=True, 
            return_filenames=False
        )
        if aux_non_json_files_no != non_json_files_no_at_1st_and_2nd_level:
            warnings.append(
                f"Expected {non_json_files_no_at_1st_and_2nd_level} non-json file(s) in {input_folderpath}, but {aux_non_json_files_no} were found."
            )

        _, folders_names = DataPreprocessor.count_folders(
            input_folderpath, 
            ignore_hidden_folders=True, 
            return_foldernames=True
        )
        folders_names.sort()

        for folder_name in folders_names:

            subfolder_path = os.path.join(input_folderpath, folder_name)

            aux_json_files_no = DataPreprocessor.count_files_by_extension_in_folder(
                subfolder_path, 
                "json", 
                count_matches=True,
                ignore_hidden_files=True, 
                return_filenames=False
            )
            if aux_json_files_no != json_files_no_at_2nd_level:
                warnings.append(
                    f"Expected {json_files_no_at_2nd_level} json file(s) in {subfolder_path}, but {aux_json_files_no} were found."
                )
            else:
                analysable_folderpaths.append(subfolder_path)

            aux_non_json_files_no = DataPreprocessor.count_files_by_extension_in_folder(
                subfolder_path, 
                "json",
                count_matches=False,
                ignore_hidden_files=True, 
                return_filenames=False
            )
            if aux_non_json_files_no != non_json_files_no_at_1st_and_2nd_level:
                warnings.append(
                    f"Expected {non_json_files_no_at_1st_and_2nd_level} non-json file(s) in {subfolder_path}, but {aux_non_json_files_no} were found."
                )

        return warnings, analysable_folderpaths
    
    @staticmethod
    def hosts_gain_data(folderpath):
        """This static method gets the following mandatory positional 
        argument:

        - folderpath (string): Path to a folder which contains the data
        to be processed.

        This method returns a boolean, which is True (resp. False) if 
        this function inferred that the given folder contains gain 
        (resp. darknoise) data. This inference is done based on the
        first json file found. If the name of such json file contains
        the 'gain' substring, then this function returns True. If such
        json file does not contain the 'gain' substring, and it
        contains the 'darknoise' substring, then this function returns
        False. If no json file at all was found, or if it was found
        but it does not contain neither the 'gain' nor the 'darknoise' 
        substrings in its name, then an exception is raised."""

        htype.check_type(
            folderpath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.hosts_gain_data", 24194
            ),
        )
        if not os.path.isdir(folderpath):

            raise Exception(
                htype.generate_exception_message(
                    "DataPreprocessor.hosts_gain_data", 20859
                )
            )
        
        aux, filenames = DataPreprocessor.count_files_by_extension_in_folder(
            folderpath, 
            "json",
            count_matches=True,
            ignore_hidden_files=True, 
            return_filenames=True
        )
        
        if aux == 0:
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DataPreprocessor.hosts_gain_data", 
                    57289,
                    extra_info=f"Not a single JSON file was found in {folderpath}"
                )
            )
        else:
            if 'gain' in filenames[0]:
                return True
            elif 'darknoise' in filenames[0]:
                return False
            else:
                raise Exception(
                    htype.generate_exception_message(
                        "DataPreprocessor.hosts_gain_data", 
                        45231,
                        extra_info=f"The inspected json file ({filenames[0]}) should contain either the 'gain' or the 'darknoise' substring."
                    )
                )
            
    @staticmethod
    def grab_strip_IDs(
        json_filepath,
        max_strip_id_no=3,
        verbose=True
    ):
        """This static method gets the following positional arguments:
        
        - json_filepath (string): Path to a json file. It must exist
        and it must end with the '.json' substring.
        - max_strip_id_no (integer): The maximum number of strip IDs
        to read from the json file. It must be a positive integer.
        - verbose (boolean): Whether to print functioning-related
        messages.

        This method loads the contents of the given json file to
        a dictionary. Then, for every i in {1, ..., max_strip_id_no},
        it looks for the value whose key matches
        f"socket_{i}_strip_ID". If such key is found, but its value
        cannot be casted to an integer, an exception is raised.
        If such key is found and its value can be casted to an
        integer, then it is casted and it is added to the output
        dictionary under a key which matches the i value. The return
        type is a dictionary whose keys, if any, are integers in the
        range [1, max_strip_id_no], and their values are the matching
        strip IDs, up to the content loaded from the input jsonf file.
        """

        htype.check_type(
            json_filepath,
            str,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.grab_strip_IDs", 40893
            ),
        )
        if not os.path.isfile(json_filepath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "DataPreprocessor.grab_strip_IDs", 47742
                )
            )
        if not json_filepath.endswith(".json"):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.grab_strip_IDs", 98490
                )
            )
        htype.check_type(
            max_strip_id_no,
            int,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.grab_strip_IDs", 13296
            ),
        )
        if max_strip_id_no < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "DataPreprocessor.grab_strip_IDs", 33351
                )
            )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "DataPreprocessor.grab_strip_IDs", 45289
            ),
        )
        
        with open(json_filepath, 'r') as file:
            data = json.load(file)

        strips_ids = {}
        for i in range(1, max_strip_id_no+1):
            try:
                aux = data[f"socket_{i}_strip_ID"]
            except KeyError:
                if verbose:
                    print(
                        "In function DataPreprocessor.grab_strip_IDs(): "
                        f"No strip ID was found for socket {i} in {json_filepath}."
                    )
                continue
            
            try:
                strips_ids[i] = int(aux)
            except ValueError:
                raise Exception(
                    "In function DataPreprocessor.grab_strip_IDs(): "
                    f"The key 'socket_{i}_strip_ID' was found in "
                    f"{json_filepath}, but its value ({aux}), of type "
                    f"{type(aux)}, cannot be casted to an integer."
                )

        return strips_ids