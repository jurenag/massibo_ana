import os
import numpy as np
import shutil
import struct

import massibo_ana.utils.htype as htype
import massibo_ana.utils.custom_exceptions as cuex
from massibo_ana.preprocess.DataPreprocessor import DataPreprocessor


class NumpyDataPreprocessor:

    def __init__(
        self,
        input_folderpath,
        gain_base="gain",
        darknoise_base="darknoise",
        key_separator="_",
        verbose=True,
    ):
        """This class is aimed at handling and preprocessing the gain and
        dark-noise raw data, typically measured with Daphne and packed into
        a numpy array in a binary format, for one thermal-cycle, i.e. for one
        cryogenic immersion of the whole setup. This initializer gets the
        following mandatory positional argument

        - input_folderpath (string): Path to the folder where the input
        data is hosted.

        And the following optional keyword arguments:

        - gain_base (string): Every file which contains this string
        in its filename and does not contain the darknoise_base string,
        will be considered the output of a gain measurement.
        - darknoise_base (string): Every file which contains this
        string in its filename and does not contain the gain_base
        string, will be considered the output of a dark noise measurement.
        - key_separator (string): All of the filepaths that are definitely
        considered a measurement candidate may contain an occurrence of
        key_separator after its base and before its extension. For gain
        (resp. dark noise) measurements, its base is gain_base (resp.
        darknoise_base). The substring that takes place somewhere after
        the first occurrence of the base and before the extension of
        the file (for more information check the documentation string of
        the DataPreprocessor.find_integer_after_base static method) is
        casted to an integer by DataPreprocessor.find_integer_after_base,
        and later used as a key for dictionary population.
        - verbose (boolean): Whether to print functioning-related messages.

        Based on the criteria explained above, this initializer populates
        the attribute self.__gain_candidates (resp.
        self.__darknoise_candidates) with the filepaths to the files within
        the provided input folder which are considered to be the output of a
        gain (resp. dark noise) measurement. The key for every value that is
        added to such dictionary attributes is extracted from the corresponding
        filepath by DataPreprocessor.find_integer_after_base. Check the
        key_separator parameter documentation of the
        DataPreprocessor.find_integer_after_base docstring for more information.
        """

        htype.check_type(
            input_folderpath,
            str,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.__init__", 89176
            ),
        )
        htype.check_type(
            gain_base,
            str,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.__init__", 89911
            ),
        )
        htype.check_type(
            darknoise_base,
            str,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.__init__", 40891
            ),
        )
        htype.check_type(
            key_separator,
            str,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.__init__", 47299
            ),
        )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.__init__", 79488
            ),
        )

        self.__input_folderpath = input_folderpath
        self.__gain_base = gain_base
        self.__darknoise_base = darknoise_base

        self.__gain_candidates = {}
        self.__darknoise_candidates = {}

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
                        os.path.splitext(filename)[0],
                        self.__gain_base,
                        separator=key_separator
                    )

                    self.__gain_candidates[aux] = filepath

                elif (
                    self.__gain_base not in filename
                    and self.__darknoise_base in filename
                ):

                    # Possible failures are
                    # anticipated within
                    # find_integer_after_base
                    aux = DataPreprocessor.find_integer_after_base(
                        os.path.splitext(filename)[0],
                        self.__darknoise_base,
                        separator=key_separator
                    )

                    self.__darknoise_candidates[aux] = filepath

        DataPreprocessor.print_dictionary_info(
            self.__gain_candidates,
            "gain",
            candidate_type="binary-numpy",
            verbose=verbose,
        )

        DataPreprocessor.print_dictionary_info(
            self.__darknoise_candidates,
            "dark noise",
            candidate_type="binary-numpy",
            verbose=verbose,
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
    def GainCandidates(self):
        return self.__gain_candidates

    @property
    def DarkNoiseCandidates(self):
        return self.__darknoise_candidates

    def generate_meas_config_files(
        self,
        root_directory,
        load_folderpath,
        aux_folderpath,
        data_folderpath,
        packing_version=0,
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
        format, will be saved in this folder. It must be contained, at an
        arbitrary depth, within the root directory.

        This method gets the following optional keyword arguments:

        - packing_version (integer): It must be a semipositive integer. It is
        eventually given to the packing_version parameter of the
        NumpyDataPreprocessor.get_metadata() method. It refers to the version of
        the procedure which was used to pack the data read by Daphne into a binary
        numpy file. I.e. this version determines how the meta-data was packed, and
        so, how it should be retrieved. One could argue that this parameter should
        be retrieved from the default-values JSON file. However, the 
        'path_to_json_default_values' parameter is not mandatory, and requiring
        another input (but mandatory) JSON file to retrieve the packing_version
        parameter is as unpractical as just asking for packing_version as a
        parameter of this method. Note that the packing_version parameter is
        necessary to retrieve the metadata from the binary input files, so it
        should be necessarily defined in order to access other parameters such
        as 'time_resolution' or 'points_per_wvf'.
        - path_to_json_default_values (string): If it is not none, it should be
        the path to a json file from which some default values may be read.
        - sipms_per_strip (positive integer): The number of SiPMs per strip. If
        it is not None, the electronic_board_socket and sipm_location fields will
        be inferred. To do so, each gain (resp. dark noise) measurement candidate
        is assigned an electronic_board_socket and a sipm_location value which
        are computed as (i//sipms_per_strip)+1 and (i%sipms_per_strip)+1) respectively,
        where i is the key of such candidate within the self.__gain_candidates
        (resp. self.__darknoise_candidates) dictionary.
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

        For each filepath in self.__gain_candidates (resp.
        self.__darknoise_candidates), this method generates a json file which
        contains all of the information needed to create a GainMeas (resp.
        DarkNoiseMeas) object using the GainMeas.from_json_file (resp.
        DarkNoiseMeas.from_json_file) initializer class method. To do so, some
        information is taken from the input files themselves, such as the fields
        'time_unit', 'signal_unit', 'time_resolution', 'points_per_wvf' or
        'wvfs_to_read', among others. The remaining necessary information, namely

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
            packing_version,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.generate_meas_config_files", 24710
            ),
        )
        if packing_version < 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.generate_meas_config_files", 57283
                )
            )

        fReadDefaultsFromFile = False
        if path_to_json_default_values is not None:
            htype.check_type(
                path_to_json_default_values,
                str,
                exception_message=htype.generate_exception_message(
                    "NumpyDataPreprocessor.generate_meas_config_files", 91126
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
                    "NumpyDataPreprocessor.generate_meas_config_files", 89701
                ),
            )
            if sipms_per_strip < 1:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "NumpyDataPreprocessor.generate_meas_config_files", 12924
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
                        "NumpyDataPreprocessor.generate_meas_config_files", 42323
                    ),
                )
                for key in strips_ids.keys():
                    htype.check_type(
                        key,
                        int,
                        np.int64,
                        exception_message=htype.generate_exception_message(
                            "NumpyDataPreprocessor.generate_meas_config_files", 61191
                        ),
                    )

                    htype.check_type(
                        strips_ids[key],
                        int,
                        np.int64,
                        exception_message=htype.generate_exception_message(
                            "NumpyDataPreprocessor.generate_meas_config_files", 10370
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

                for key in self.GainCandidates.keys():
                    if key not in allowed_measurement_keys:
                        not_allowed_found_keys.add(key)

                for key in self.DarkNoiseCandidates.keys():
                    if key not in allowed_measurement_keys:
                        not_allowed_found_keys.add(key)

                if len(not_allowed_found_keys) > 0:
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message(
                            "NumpyDataPreprocessor.generate_meas_config_files",
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
                "NumpyDataPreprocessor.generate_meas_config_files", 67213
            ),
        )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.generate_meas_config_files", 92127
            ),
        )

        queried_wvf_fields = {"signal_magnitude": str}
        read_wvf_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_wvf_fields_from_file, queried_wvf_fields = (
                DataPreprocessor.try_grabbing_from_json(
                    queried_wvf_fields,
                    path_to_json_default_values,
                    verbose=verbose
                )
            )
        queried_once_wvf_fields = {}
        if bool(queried_wvf_fields):  # True if queried_wvfs_fields is not empty
            queried_once_wvf_fields, queried_wvf_fields = (
                DataPreprocessor.query_dictionary_splitting(queried_wvf_fields)
            )

        queried_wvfset_fields = {
            "set_name": str,
            "creation_dt_offset_min": float,
        }

        read_wvfset_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_wvfset_fields_from_file, queried_wvfset_fields = (
                DataPreprocessor.try_grabbing_from_json(
                    queried_wvfset_fields,
                    path_to_json_default_values,
                    verbose=verbose
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
                            f"In function NumpyDataPreprocessor.generate_meas_config_files(): The values for the fields 'electronic_board_socket' and 'sipm_location' will be inferred according to the candidates keys and the value given to the 'sipms_per_strip' ({sipms_per_strip}) parameter. Do you want to continue? (y/n)"
                        )
                    ):
                        return
                else:
                    print(f"In function NumpyDataPreprocessor.generate_meas_config_files(): The values for the fields 'electronic_board_socket' and 'sipm_location' will be inferred according to the candidates keys and the value given to the 'sipms_per_strip' ({sipms_per_strip}) parameter.")
            else:
                if ask_for_inference_confirmation:
                    if not DataPreprocessor.yes_no_translator(
                        input(
                            f"In function NumpyDataPreprocessor.generate_meas_config_files(): The values for the fields 'electronic_board_socket', 'sipm_location' and 'strip_ID', will be inferred according to the candidates keys and the values given to the 'sipms_per_strip' ({sipms_per_strip}) and the 'strips_ids' ({strips_ids}) parameters. Do you want to continue? (y/n)"
                        )
                    ):
                        return
                else:
                    print(f"In function NumpyDataPreprocessor.generate_meas_config_files(): The values for the fields 'electronic_board_socket', 'sipm_location' and 'strip_ID', will be inferred according to the candidates keys and the values given to the 'sipms_per_strip' ({sipms_per_strip}) and the 'strips_ids' ({strips_ids}) parameters.")

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
            "FastFrame Count": [int, "wvfs_to_read"]
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
                "Let us retrieve the unique-query fields. These fields will apply for every measurement in this NumpyDataPreprocessor instance."
            )

        aux_wvf_dict = DataPreprocessor.query_fields_in_dictionary(
            queried_once_wvf_fields,
            default_dict=None
        )
        aux_wvf_dict.update(read_wvf_fields_from_file)
        aux_wvfset_dict = DataPreprocessor.query_fields_in_dictionary(
            queried_once_wvfset_fields,
            default_dict=None
        )
        aux_wvfset_dict.update(read_wvfset_fields_from_file)
        aux_sipmmeas_dict = DataPreprocessor.query_fields_in_dictionary(
            queried_once_sipmmeas_fields,
            default_dict=None
        )
        aux_sipmmeas_dict.update(read_sipmmeas_fields_from_file)
        aux_gainmeas_dict = DataPreprocessor.query_fields_in_dictionary(
            queried_once_gainmeas_fields,
            default_dict=None
        )
        aux_gainmeas_dict.update(read_gainmeas_fields_from_file)
        aux_darknoisemeas_dict = DataPreprocessor.query_fields_in_dictionary(
            queried_once_darknoisemeas_fields,
            default_dict=None
        )
        aux_darknoisemeas_dict.update(read_darknoisemeas_fields_from_file)

        aux_gainmeas_dict.update(aux_sipmmeas_dict)
        aux_darknoisemeas_dict.update(aux_sipmmeas_dict)

        for i, key in enumerate(sorted(self.GainCandidates.keys())):

            aux = NumpyDataPreprocessor.get_metadata(
                self.GainCandidates[key],
                packing_version=packing_version,
                get_creation_date=False,
                verbose=verbose
            )

            print(
                f"Let us retrieve some information for the waveform set in {self.GainCandidates[key]}"
            )

            aux_wvf_dict.update(
                {
                    translator["Horizontal Units"][1]: aux["Horizontal Units"],
                    translator["Vertical Units"][1]: aux["Vertical Units"],
                }
            )

            aux_wvf_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvf_fields,
                    default_dict=aux_wvf_dict
                )
            )

            aux_wvfset_dict.update(
                {
                    translator["Sample Interval"][1]: aux["Sample Interval"],
                    translator["Record Length"][1]: aux["Record Length"],
                    translator["FastFrame Count"][1]: aux["FastFrame Count"],
                    "packing_version": packing_version,
                }
            )

            aux_wvfset_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvfset_fields,
                    default_dict=aux_wvfset_dict
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
                    queried_gainmeas_fields,
                    default_dict=aux_gainmeas_dict
                )
            )

            # The date follows the format 'YYYY-MM-DD HH:MM:SS'. Thus,
            # aux_gainmeas_dict['date'][:10], gives 'YYYY-MM-DD'.
            output_filepath_base = f"{aux_gainmeas_dict['strip_ID']}-{aux_gainmeas_dict['sipm_location']}-{aux_gainmeas_dict['thermal_cycle']}-OV{round(10.*aux_gainmeas_dict['overvoltage_V'])}dV-{aux_gainmeas_dict['date'][:10]}"

            _, extension = os.path.splitext(self.GainCandidates[key])

            new_raw_filepath = os.path.join(
                data_folderpath, 
                output_filepath_base + "_raw_gain" + extension
            )
            
            shutil.move(
                self.GainCandidates[key], 
                new_raw_filepath
            )

            aux_wvfset_dict.update(
                {
                    "wvf_filepath": os.path.relpath(
                        new_raw_filepath,
                        start=root_directory
                    )
                }
            )

            wvf_output_filepath = os.path.join(
                aux_folderpath,
                output_filepath_base + "_gain_wvf.json"
            )

            DataPreprocessor.generate_json_file(
                # Waveform.Signs.setter inputs must be lists
                {key: [value] for (key, value) in aux_wvf_dict.items()}, 
                wvf_output_filepath)

            aux_wvfset_dict["wvf_extra_info"] = os.path.relpath(
                wvf_output_filepath,
                start=root_directory
            )

            wvfset_output_filepath = os.path.join(
                aux_folderpath,
                output_filepath_base + "_gain_wvfset.json"
            )
            DataPreprocessor.generate_json_file(
                aux_wvfset_dict,
                wvfset_output_filepath)

            aux_gainmeas_dict["wvfset_json_filepath"] = os.path.relpath(
                wvfset_output_filepath,
                start=root_directory
            )
            gainmeas_output_filepath = os.path.join(
                load_folderpath,
                output_filepath_base + "_gainmeas.json"
            )
            DataPreprocessor.generate_json_file(
                aux_gainmeas_dict,
                gainmeas_output_filepath
            )

        for i, key in enumerate(sorted(self.DarkNoiseCandidates.keys())):

            aux = NumpyDataPreprocessor.get_metadata(
                self.DarkNoiseCandidates[key],
                packing_version=packing_version,
                get_creation_date=False,
                verbose=verbose
            )

            print(
                f"Let us retrieve some information for the waveform set in {self.DarkNoiseCandidates[key]}"
            )

            aux_wvf_dict.update(
                {
                    translator["Horizontal Units"][1]: aux["Horizontal Units"],
                    translator["Vertical Units"][1]: aux["Vertical Units"],
                }
            )

            aux_wvf_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvf_fields,
                    default_dict=aux_wvf_dict
                )
            )

            aux_wvfset_dict.update(
                {
                    translator["Sample Interval"][1]: aux["Sample Interval"],
                    translator["Record Length"][1]: aux["Record Length"],
                    translator["FastFrame Count"][1]: aux["FastFrame Count"],
                    "packing_version": packing_version,
                }
            )

            aux_wvfset_dict.update(
                DataPreprocessor.query_fields_in_dictionary(
                    queried_wvfset_fields,
                    default_dict=aux_wvfset_dict
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
                    queried_darknoisemeas_fields,
                    default_dict=aux_darknoisemeas_dict
                )
            )

            output_filepath_base = f"{aux_darknoisemeas_dict['strip_ID']}-{aux_darknoisemeas_dict['sipm_location']}-{aux_darknoisemeas_dict['thermal_cycle']}-OV{round(10.*aux_darknoisemeas_dict['overvoltage_V'])}dV-{aux_darknoisemeas_dict['date'][:10]}"
            # The date follows the format 'YYYY-MM-DD HH:MM:SS'. Thus,
            # aux_darknoisemeas_dict['date'][:10], gives 'YYYY-MM-DD'.

            _, extension = os.path.splitext(self.DarkNoiseCandidates[key])

            new_raw_filepath = os.path.join(
                data_folderpath, 
                output_filepath_base + "_raw_darknoise" + extension)
            
            shutil.move(
                self.DarkNoiseCandidates[key], 
                new_raw_filepath
            )

            aux_wvfset_dict.update(
                {
                    "wvf_filepath": os.path.relpath(
                        new_raw_filepath,
                        start=root_directory
                    )
                }
            )

            wvf_output_filepath = os.path.join(
                aux_folderpath,
                output_filepath_base + "_darknoise_wvf.json"
            )

            DataPreprocessor.generate_json_file(
                # Waveform.Signs.setter inputs must be lists
                {key: [value] for (key, value) in aux_wvf_dict.items()}, 
                wvf_output_filepath)

            aux_wvfset_dict["wvf_extra_info"] = os.path.relpath(
                wvf_output_filepath,
                start=root_directory
            )

            wvfset_output_filepath = os.path.join(
                aux_folderpath,
                output_filepath_base + "_darknoise_wvfset.json"
            )

            DataPreprocessor.generate_json_file(
                aux_wvfset_dict,
                wvfset_output_filepath
            )

            aux_darknoisemeas_dict["wvfset_json_filepath"] = os.path.relpath(
                wvfset_output_filepath,
                start=root_directory
            )

            darknoisemeas_output_filepath = os.path.join(
                load_folderpath,
                output_filepath_base + "_darknoisemeas.json"
            )
            
            DataPreprocessor.generate_json_file(
                aux_darknoisemeas_dict,
                darknoisemeas_output_filepath
            )

        return
    
    @staticmethod
    def get_metadata(
        filepath,
        packing_version=0,
        get_creation_date=False,
        verbose=True
    ):
        """This static method gets the following mandatory positional argument:

        - filepath (string): Path to the file whose meta-data will be retrieved.

        This function also gets the following optional keyword arguments:

        - packing_version (int): It must be a semipositive integer. No well-formedness
        checks for this parameter are performed here. The caller is responsible for
        this. It refers to the version of the procedure which was used to pack the data
        read by Daphne into a binary numpy file. I.e. this version determines how the
        meta-data was packed, and so, how it should be retrieved.

        - get_creation_date (bool): If True, the creation date of the input file
        is added to the resulting dictionary under the key 'creation_date'. The
        associated value is an string which follows the format 'YYYY-MM-DD HH:MM:SS'.
        If False, no extra entry is added to the resulting dictionary.

        - verbose (boolean): Whether to print functioning-related messages.

        This static method

            1) receives a path to an input file, which should be a binary file which
            combines some metadata in a header plus a numpy array, and
            2) extracts some meta-data from it which is partially returned by this method
            as a dictionary.
            3) Optionally, if get_creation_date is True, the creation date of the input
            file is also retrieved and added to the resulting dictionary under the key
            'creation_date'.

        For NumpyDataPreprocessor.generate_meas_config_files() to work properly, the dictionary
        returned by this method must, at least, contain the following keys: 'Horizontal Units',
        'Vertical Units', 'Sample Interval', 'Record Length', 'FastFrame Count' and
        'creation_date'. The 'Timestamp Bytes' key may be also required by the corresponding
        core-data extractor."""

        htype.check_type(
            filepath,
            str,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.get_metadata", 17219
            ),
        )
        htype.check_type(
            get_creation_date,
            bool,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.get_metadata", 11280
            ),
        )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.get_metadata", 56912
            ),
        )

        result = {}

        if packing_version == 0:
            result.update(
                NumpyDataPreprocessor.__get_metadata_v0(
                    filepath
                )
            )

        elif packing_version == 1:
            result.update(
                NumpyDataPreprocessor.__get_metadata_v1(
                    filepath
                )
            )

        else:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.get_metadata",
                    31801,
                    extra_info=f"The packing version {packing_version} is not supported."
                )
            )

        if get_creation_date:
            result["creation_date"] = DataPreprocessor.get_str_creation_date(filepath)

        if verbose:
            print(
                f"In function NumpyDataPreprocessor.get_metadata(): Succesfully processed {filepath}"
            )

        return result
    
    @staticmethod
    def __get_metadata_v0(
        filepath
    ):
        """This is a helper method which should only be called by the
        NumpyDataPreprocessor.get_metadata() method i.e. it is not intended to be
        called directly by the user. No type checks are done here. Its purpose is
        to carry out the second bullet of the NumpyDataPreprocessor.get_metadata()
        docstring, for the particular case when the version parameter is equal to 0.
        """

        shape, _ = NumpyDataPreprocessor.__get_npy_file_shape_and_dtype(filepath)

        return {
            'Horizontal Units': 'ns',
            'Vertical Units': 'ADU',
            'Sample Interval': 1000/41.66,
            # The first column is typically the timestamp,
            # so the number of points per waveform is
            # actually the number of columns minus one
            'Record Length': shape[1] - 1,
            'FastFrame Count': shape[0],
            'PreTrigger length': 64,
            'Timestamp Bytes': 4
        }
    
    @staticmethod
    def __get_metadata_v1(
        filepath
    ):
        """This is a helper method which should only be called by the
        NumpyDataPreprocessor.get_metadata() method i.e. it is not intended to be
        called directly by the user. No type checks are done here. Its purpose is
        to carry out the second bullet of the NumpyDataPreprocessor.get_metadata()
        docstring, for the particular case when the version parameter is equal to 1.
        """

        with open(filepath, 'rb') as file:
            
            metadata_length_bytes = struct.unpack(
                '<H',
                # Note that the file pointer is forwarded by 2 bytes
                # and is not reset to the beginning
                file.read(2)
            )[0]

            metadata = file.read(metadata_length_bytes)

        # This offset is referred to metadata
        offset = 0
    
        # Read the number of bytes used
        # to store the VERTICAL_UNITS string
        VERTICAL_UNITS_length_bytes = struct.unpack_from(
            # 2 bytes
            '<H',
            metadata,
            offset
        )[0]
        offset += 2

        # Read the VERTICAL_UNITS string
        VERTICAL_UNITS = metadata[
            offset:offset + VERTICAL_UNITS_length_bytes
        ].decode('utf-8')
        offset += VERTICAL_UNITS_length_bytes

        # Read the number of bytes used
        # to store the HORIZONTAL_UNITS string
        HORIZONTAL_UNITS_length_bytes = struct.unpack_from(
            # 2 bytes
            '<H',
            metadata,
            offset
        )[0]
        offset += 2

        # Read the HORIZONTAL_UNITS string
        HORIZONTAL_UNITS = metadata[
            offset:offset + HORIZONTAL_UNITS_length_bytes
        ].decode('utf-8')
        offset += HORIZONTAL_UNITS_length_bytes

        SAMPLE_INTERVAL = struct.unpack_from(
            # 4 bytes
            '<f',
            metadata,
            offset
        )[0]
        offset += 4

        WAVEFORM_LENGTH = struct.unpack_from(
            # 4 bytes
            '<I',
            metadata,
            offset
        )[0]
        offset += 4

        NUMBER_OF_WAVEFORMS = struct.unpack_from(
            '<I',
            metadata,
            offset
        )[0]
        offset += 4

        PRETRIGGER_LENGTH = struct.unpack_from(
            '<I',
            metadata,
            offset
        )[0]
        offset += 4

        TIMESTAMP_BYTES = struct.unpack_from(
            '<H',
            metadata,
            offset
        )[0]
        offset += 2

        return {
            'Horizontal Units': HORIZONTAL_UNITS,
            'Vertical Units': VERTICAL_UNITS,
            'Sample Interval': SAMPLE_INTERVAL,
            'Record Length': WAVEFORM_LENGTH,
            'FastFrame Count': NUMBER_OF_WAVEFORMS,
            'PreTrigger length': PRETRIGGER_LENGTH,
            'Timestamp Bytes': TIMESTAMP_BYTES,
        }
    
    @staticmethod
    def __get_npy_file_shape_and_dtype(filepath):
        """This is a helper method which should only be called by the
        NumpyDataPreprocessor.__get_metadata_v0() method i.e. it is not intended to
        be called directly by the user. No type checks are done here. This static
        method gets the following mandatory positional argument:

        - filepath (string): Path to a numpy file which was created using the
        numpy.save() function.
        
        This function returns two values:

        - shape (tuple): Its length matches 2. It is the shape of the numpy array
        stored in the input file.
        - dtype (numpy.dtype): The data type of the numpy array stored in the input
        file.
        
        This function is particularly useful for cases when the stored array is very
        big. This function is able to retrieve the desired information (shape and dtype)
        without loading the entire array into memory. This is done by reading just the
        header of the file.
        """

        with open(
            filepath,
            'rb') as file:

            aux = file.read(6)
            if aux != b'\x93NUMPY':            
                raise ValueError(
                    htype.generate_exception_message(
                        "NumpyDataPreprocessor.__get_npy_file_shape_and_dtype",
                        46912,
                        extra_info=f"The file {filepath} is not a valid numpy file."
                    )
                )

            version = tuple(file.read(2))
            if version == (1, 0):
                header = np.lib.format.read_array_header_1_0(file)

            elif version == (2, 0):
                header = np.lib.format.read_array_header_2_0(file)

            else:
                raise ValueError(
                    htype.generate_exception_message(
                        "NumpyDataPreprocessor.__get_npy_file_shape_and_dtype",
                        23568,
                        extra_info=f"Version {version} is not supported."
                    )
                )

            return header[0], header[2]
        
    @staticmethod
    def extract_homemade_bin_coredata(
            filepath,
            packing_version=0,
            tolerance=0,
            verbose=True,
        ):
        """This static method gets the following mandatory positional arguments:

        - filepath (string): Path to the homemade binary file (.npy or .bin),
        whose core data should be extracted, i.e. its waveform and timestamp data.
        - packing_version (int): It must be a semipositive integer. No well-
        formedness checks for this parameter are performed here. The caller is
        responsible for this. It refers to the version of the procedure which was
        used to pack the data read by Daphne into a binary numpy file. I.e. this
        version determines how the meta-data and the core-data was packed, and so,
        how it should be retrieved.
        - tolerance (int): It must be a semipositive integer. It is given to the
        tolerance parameter of
        NumpyDataPreprocessor.fix_timestamp_overflow(). For more information,
        check the docstring of such method.
        - verbose (bool): Whether to print functioning related messages or not.

        This method returns two arrays. The first one is an unidimensional array
        of length M, which stores timestamp information. The second one is a
        bidimensional array which stores the waveforms. Say such array has
        shape NxM: then N is the number of points per waveform, while M is the
        number of waveforms. The waveform entries in such array are already
        expressed in the vertical units which are extracted to the key 'Vertical
        Units' by NumpyDataPreprocessor.get_metadata(). In this context, the i-th
        entry of the first array returned by this function gives the time
        difference, in seconds, between the trigger of the i-th waveform and
        the trigger of the (i-1)-th waveform. The first entry, which is undefined
        up to the given definition, is manually set to zero."""

        htype.check_type(
            filepath,
            str,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.extract_homemade_bin_coredata",
                59755
            ),
        )
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.extract_homemade_bin_coredata",
                    83081,
                    extra_info=f"Path {filepath} does not exist or is not a file.",
                )
            )
        else:
            _, extension = os.path.splitext(filepath)
            if extension not in (".npy", ".bin"):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "NumpyDataPreprocessor.extract_homemade_bin_coredata",
                        89792,
                        extra_info=f"The extension of the input file ({extension}) must match '.npy' or '.bin'.",
                    )
                )
            
        htype.check_type(
            tolerance,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.extract_homemade_bin_coredata",
                83499
            ),
        )
        if tolerance < 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.extract_homemade_bin_coredata",
                    24243
                )
            )
        
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.extract_homemade_bin_coredata",
                48211
            ),
        )

        try:
            metadata = NumpyDataPreprocessor.get_metadata(
                filepath,
                packing_version=packing_version,
                get_creation_date=False,
                verbose=verbose
            )
        except Exception as e:
            raise Exception(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.extract_homemade_bin_coredata",
                    22831,
                    extra_info=f"The following error (maybe due to an erroneous "
                    f"packing_version, which was set to {packing_version}) ocurred"
                    f" while trying to get the metadata: {e}"
                )
            )
            
        # For packing versions in (0,1), if an erroneous packing_version was defined, then the
        # NumpyDataPreprocessor.get_metadata() will have raised an exception. I.e. there is no
        # need to handle exceptions in the calls to NumpyDataPreprocessor.__get_coredata_vi().
        # This should be revised when a new packing version is added.

        if packing_version == 0:
            timestamp, waveforms = NumpyDataPreprocessor.__get_coredata_v0(
                    filepath
                )
            
        elif packing_version == 1:
            timestamp, waveforms = NumpyDataPreprocessor.__get_coredata_v1(
                    filepath,
                    verbose=verbose
                )
        else:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.extract_homemade_bin_coredata",
                    87937,
                    extra_info=f"The packing version {packing_version} is not supported."
                )
            )
        
        timestamp_bytes = NumpyDataPreprocessor.infer_bytes_number(
            timestamp
        )
        
        # The operations that happen as of this point and until the
        # return statement are common to all the packing versions. 
        timestamp = NumpyDataPreprocessor.fix_timestamp_overflow(
                timestamp,
                # Using the inferred bytes number instead of
                # metadata['Timestamp Bytes'] because the metadata
                # provided by the operator is not reliable and prone to errors
                timestamp_bytes,
                tolerance=tolerance,
                check_upper_bound=True,
                return_overflow_idcs=False
            )

        sample_interval_s = metadata['Sample Interval'] * \
            NumpyDataPreprocessor.interpret_time_unit_in_seconds(
                metadata['Horizontal Units']
            )

        # N.B. 1: An UFuncTypeError is raised if no previous casting is
        # done here
        # N.B. 2: Casting 8-bytes integers (np.uint64) to 8-bytes floats
        # (np.float64) introduces a rounding error only if the integers
        # are bigger than 2**53, which won't happen for our case where
        # (at ~24 ns of sampling rate), 2**53 corrresponds to a ~6.9 years
        # non-stop data-taking.
        timestamp = sample_interval_s * timestamp.astype(np.float64)

        # N.B. 1: The following concatenation fixes the fact that the timestamp
        # definition in the docstring of this function is different from that of
        # the definition of the numpy.diff function.
        # N.B. 2: One could think that computing the np.diff() here is just a
        # way of unifying the computation pipeline with the Tektronix ASCII
        # case (where the timestamp contains time increments by default, and
        # a cumulative sum needs to be done later in WaveformSet.read_wvfs()).
        # However, this is not the only reason. The other (and more important)
        # reason is that the timestamps stored in the homemade-binary files have
        # an arbitrary time origin. This time origin has to do with the
        # Daphne timestamp counter, which is NOT reset everytime a new SiPM
        # measurement is started. Therefore, although it potentially makes us 
        # lose the time lapse between the start of the measurement and the
        # first trigger, the np.diff() operation is necessary.
        timestamp = np.concatenate((np.array([0.0]), np.diff(timestamp)), axis=0)

        return timestamp, waveforms
        
    @staticmethod
    def __get_coredata_v0(
        filepath
    ):
        """This is a helper method which should only be called by the
        NumpyDataPreprocessor.extract_homemade_bin_coredata() method i.e. it is
        not intended to be called directly by the user. No type checks are done
        here. Its purpose is to carry out the extraction of the core/bulk data of
        the given homemade binary file, for the particular case when its packing
        version is equal to 0.
        """

        try:
            aux = np.load(filepath)

        except Exception as e:
            raise cuex.BinReadException(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.__get_coredata_v0",
                    48019,
                    extra_info="Caught the following error while trying "
                    f"to read the core data of the file {filepath}: {e}"
                )
            )

        timestamp = aux[:,0]
        waveforms = np.transpose(aux[:,1:])

        return timestamp, waveforms

    @staticmethod
    def __get_coredata_v1(
        filepath,
        verbose=True
    ):
        """This is a helper method which should only be called by the
        NumpyDataPreprocessor.extract_homemade_bin_coredata() method i.e. it is
        not intended to be called directly by the user. No type checks are done
        here. Its purpose is to carry out the extraction of the core/bulk data of
        the given homemade binary file, for the particular case when its packing
        version is equal to 1.
        """

        # Read the metadata because we need to know the number of
        # waveforms and the waveform length for the core-data extraction
        metadata = NumpyDataPreprocessor.get_metadata(
            filepath,
            packing_version=1,
            get_creation_date=False,
            verbose=verbose
        )

        with open(filepath, 'rb') as file:
            
            metadata_length_bytes = struct.unpack(
                '<H',
                # Note that the file pointer is forwarded by 2 bytes
                # and is not reset to the beginning
                file.read(2)
            )[0]

            # Dump the metadata
            _ = file.read(metadata_length_bytes)

            try:
                data = np.frombuffer(
                    file.read(
                        # Plus one because the first entry is the timestamp
                        metadata['FastFrame Count'] * (metadata['Record Length'] + 1) * 4
                    ),
                    dtype=np.uint32
                )  # 4 bytes por uint32

                data = data.reshape(
                    (metadata['FastFrame Count'], metadata['Record Length'] + 1)
                )

            except Exception as e:
                raise cuex.BinReadException(
                    htype.generate_exception_message(
                        "NumpyDataPreprocessor.__get_coredata_v1",
                        93004,
                        extra_info="Caught the following error while trying "
                        f"to read the core data of the file {filepath}: {e}"
                    )
                )

        timestamp = data[:,0]
        waveforms = np.transpose(data[:,1:])

        return timestamp, waveforms

    @staticmethod  
    def fix_timestamp_overflow(
            timestamp,
            timestamp_bytes,
            tolerance=0,
            check_upper_bound=False,
            return_overflow_idcs=False
        ):
        """
        This static method gets the following positional arguments:

        - timestamp (np.ndarray): 1D array of integer timestamp values potentially
        affected by overflows.
        - timestamp_bytes (integer): Number of bytes used to store each timestamp in
        the original source (e.g., 4 for a 32-bit register). Must be at least 1.

        This static method gets the following keyword arguments: 

        - tolerance (integer): A non-negative tolerance threshold for detecting
        overflows. timestamp[i] is considered to have overflown with respect to
        timestamp[i-1] if the following is true: timestamp[i-1] - tolerance > timestamp[i]
        Default is 0, in which case, an overflow is detected if
        timestamp[i-1] > timestamp[i].
        - check_upper_bound (bool): If True, checks whether any timestamp value
        exceeds the maximum possible value for the given byte size. Raises an error
        if such values are found. Default is False.
        - return_overflow_idcs (bool): If True, returns a list of indices where
        overflows were detected in addition to the corrected timestamp array. Default
        is False.

        This method fixes timestamp overflows caused by limited register size. It
        processes an array of integer timestamps and corrects overflow errors that
        occur when the timestamp counter exceeds its maximum value. It assumes that
        overflows can be detected by a drop in the timestamp values and sums the
        appropriate correction to subsequent timestamps to restore their monotonicity."""

        htype.check_type(
            timestamp,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.fix_timestamp_overflow",
                59960
            ),
        )

        htype.check_type(
            timestamp_bytes,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.fix_timestamp_overflow",
                52220
            ),
        )
        if timestamp_bytes < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.fix_timestamp_overflow",
                    48947
                )
            )

        htype.check_type(
            tolerance,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.fix_timestamp_overflow",
                34915
            ),
        )
        if tolerance < 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.fix_timestamp_overflow",
                    37137
                )
            )
        
        htype.check_type(
            check_upper_bound,
            bool,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.fix_timestamp_overflow",
                21030
            ),
        )

        htype.check_type(
            return_overflow_idcs,
            bool,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.fix_timestamp_overflow",
                83745
            ),
        )
        
        if timestamp_bytes <= 4:
            # Worst-case scenario is we need to fix timestamp overflows even having
            # a 4 bytes register for the timestamp, in which case, we will prepare
            # a np.uint64 array, for which each entry is 8 bytes (i.e. can represent
            # up to 2**64). I.e. the maximum for each new entry is 2**32 times
            # larger than the previous one, which was 4 bytes (i.e. 2**32).
            aux = np.uint64
        else:
            raise ValueError(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.fix_timestamp_overflow",
                    67568,
                    extra_info="We probably don't need to fix timestamp overflows "
                    "if the timestamp register is larger than 4 bytes. P.e. at the "
                    "typical sampling rate of ~24 ns, 4 bytes can represent up to "
                    "~103 seconds, while 5 bytes can represent up to ~7.5 hours. "
                    "If our measurement did not exceed 7 hours, (which it probably"
                    " did not), we do not need to fix any timestamp overflow. Make"
                    " sure not to call this function in this case."
                )
            )
        
        fixed_timestamp = np.empty(
            np.shape(timestamp),
            dtype=aux
        )
        
        correction = 0
        correction_step = 2 ** (8 * timestamp_bytes)

        if check_upper_bound:
            for i in range(len(timestamp)):
                if timestamp[i] >= correction_step:
                    raise ValueError(
                        htype.generate_exception_message(
                            "NumpyDataPreprocessor.fix_timestamp_overflow",
                            71013,
                            extra_info=f"The {i}-th entry of the given timestamp"
                            f" ({timestamp[i]}) cannot be larger or equal to"
                            f" the maximum value ({correction_step}) for a "
                            f"{timestamp_bytes}-byte(s) register."
                        )
                    ) 

        overflown_idcs = []

        # Assuming that the first entry of the timestamp is not overflown
        fixed_timestamp[0] = timestamp[0]

        for i in range(1, len(timestamp)):
            if timestamp[i-1] - tolerance > timestamp[i]:
                correction += correction_step
                overflown_idcs.append(i)

            fixed_timestamp[i] = timestamp[i]
            fixed_timestamp[i] += correction

        if not return_overflow_idcs:
            return fixed_timestamp
        else:
            return fixed_timestamp, overflown_idcs
        
    @staticmethod
    def interpret_time_unit_in_seconds(time_unit):
        """
        This function gets the following positional argument:

        - time_unit (string): It represents a time unit which is
        a decimal multiple of a second. It can take a value in 
        ['s', 'ms', 'us', 'ns', 'ps'].
        
        This function returns the numerical value of the given
        time unit in seconds. For example, if the input is 'ms',
        the output will be 1e-3."""

        htype.check_type(
            time_unit,
            str,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.interpret_time_unit_in_seconds",
                82593
            ),
        )

        if time_unit == 's':
            return 1.
        elif time_unit == 'ms':
            return 1.e-3
        elif time_unit == 'us':
            return 1.e-6
        elif time_unit == 'ns':
            return 1.e-9
        elif time_unit == 'ps':
            return 1.e-12
        else:
            raise ValueError(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.interpret_time_unit_in_seconds",
                    96926,
                    extra_info=f"An unknown time unit ({time_unit}) was given."
                )
            )

    @staticmethod
    def infer_bytes_number(
            input
        ): 

        """This static method gets the following mandatory positional argument:

        - input (np.ndarray): Unidimensional array of integers, which stores raw
        values as given by an N-bytes register. I.e. these values are supposed to
        be integer values coming from a counter which can take values from 0 to
        (2^N)-1, both included.

        This method returns a positive integer, which is the inferred number of
        bytes used to represent the values which belong to the given input. To
        do such inference, this method checks the maximum value of input, say
        i_max. Then, the inferred number of bytes is given by the smallest
        integer N such that (2^N)-1 >= i_max."""

        htype.check_type(
            input,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "NumpyDataPreprocessor.infer_bytes_number",
                23818
            ),
        )

        if np.ndim(input) != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.infer_bytes_number",
                    69278,
                    extra_info="The given array must be unidimensional.",
                )
            )
        
        maximum_value = np.max(input)
        fInferred = False

        # Iterate from 1 to 16, so that we find the minimum
        # number of bytes that can store the maximum value
        for bytes_number in range(1, 17):
            if maximum_value <= (2 ** (bytes_number * 8)) - 1:
                fInferred = True
                break

        if not fInferred:
            raise Exception(
                htype.generate_exception_message(
                    "NumpyDataPreprocessor.infer_bytes_number",
                    12152,
                    extra_info="The inferred number of bytes used to store "
                    "the timestamp information is bigger than 16 bytes, which "
                    "is very unlikely. Please check the input timestamp array."
                )
            )

        return bytes_number