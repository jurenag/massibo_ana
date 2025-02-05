import numpy as np
import typing

import massibo_ana.utils.htype as htype
import massibo_ana.utils.custom_exceptions as cuex
from massibo_ana.custom_types.IdentifiedDict import IdentifiedDict


class SiPMMeasSummary(IdentifiedDict):

    def __init__(self, input_data, ignored_fields=[]):
        """This class, which inherits from IdentifiedDict, aims to model a
        a summary of a SiPMMeas object, meaning the result of the analysis
        of such SiPMMeas object, which can be labeled with an identifier.

        This class ultimately inherits from python's built-in dictionary, so
        the data that it stores is completely flexible, giving room to store
        analysis results from measurements performed at different sites
        with different data-formats.

        I.e. this class does not provide almost any SiPMMeas related
        but it offers some tools to identify each object according to an
        unified format, which can be helpful to match SiPMMeasSummary
        objects which arise from the same physical SiPMs but have been
        measured in different sites, and so, have different formats.
        This matching is needed prior to comparing the results of such
        measurements.

        The parameters received by this initializer are the same as the
        ones received by the IdentifiedDict initializer, and they are
        passed to it in the same order. For more information on such
        parameters, check the IdentifiedDict.__init__ documentation."""

        super().__init__(input_data, ignored_fields=ignored_fields)

    def standard_identifier(
        self,
        add_one_to_sipm_location=False,
        strip_id_key_candidates=[],
        sipm_location_key_candidates=[],
    ) -> str:
        """This method gets the following keyword arguments:

        - add_one_to_sipm_location (bool): If True, then the sipm-location value
        is increased by one. Note that this addition only affects self.ID, but
        the spotted sipm-location value in the underlying IdentifiedDict remains
        unchanged. This is useful when the sipm-location is zero-indexed and we
        eventually want to compare this measurement against a one-indexed one.
        By default, this parameter is False.

        - strip_id_key_candidates (resp. sipm_location_key_candidates) (tuple/list
        of strings): The first string within this tuple/list which matches one key
        in self, except for lower- or upper-case conversion, gives the key within
        self to the strip-identifier (resp. sipm-location) value of this SiPMMeasSummary
        object. If no such key is found, SiPMMeasSummary.try_retrieving_a_value_from_dict
        will raise a cuex.NoAvailableData exception. By default, this method considers
        a number of strip-identifier (resp. sipm-location) key candidates, but the
        user-given ones are considered in the first place.

        If this method succeeds to spot the strip-identifier and the sipm-location
        in self, then this method returns an string with the following format
        <strip-identifier>-<sipm-location>."""

        htype.check_type(
            add_one_to_sipm_location,
            bool,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.standard_identifier", 50221
            ),
        )
        htype.check_type(
            strip_id_key_candidates,
            tuple,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.standard_identifier", 59288
            ),
        )
        htype.check_type(
            sipm_location_key_candidates,
            tuple,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.standard_identifier", 64548
            ),
        )
        for x in strip_id_key_candidates + sipm_location_key_candidates:
            htype.check_type(
                x,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeasSummary.standard_identifier", 76611
                ),
            )

        # Avoid creating a view of the keyword argument, otherwise
        # modifications to the list would affect the default value
        # of the keyword argument in subsequent calls to this method
        strip_id_key_candidates_ = strip_id_key_candidates.copy()

        sipm_location_key_candidates_ = sipm_location_key_candidates.copy()

        # Add some key candidates default values
        strip_id_key_candidates_.extend(
            [
                "sipm_strip_id",
                "sipmstrip_id",
                "sipm_stripid",
                "sipmstripid",
                "strip_id",
                "stripid",
                "strip",
                "strip_number",
                "stripnumber",
                "stripnum",
                "board_id",
                "boardid",
                "board",
                "board_number",
                "boardnumber",
                "boardnum",
            ]
        )

        sipm_location_key_candidates_.extend(
            [
                "sipm_location",
                "sipmlocation",
                "sipmloc",
                "sipm",
                "sipmnumber",
                "sipm_number",
            ]
        )

        # Add first, the default key candidates, then remove
        # the redundancy, just in case the user gave a key
        # candidate which is already considered by default
        # If that was the case, since IdentifiedDict.lowercase_and_remove_redundancy
        # preserves the ordering, the user-given key candidates
        # will still be considered prior to the default ones

        strip_id_key_candidates_ = IdentifiedDict.lowercase_and_remove_redundancy(
            strip_id_key_candidates_
        )
        sipm_location_key_candidates_ = IdentifiedDict.lowercase_and_remove_redundancy(
            sipm_location_key_candidates_
        )

        strip_ID = SiPMMeasSummary.try_retrieving_a_value_from_dict(
            self, key_candidates=strip_id_key_candidates_, return_successful_key=False
        )

        sipm_location = SiPMMeasSummary.try_retrieving_a_value_from_dict(
            self,
            key_candidates=sipm_location_key_candidates_,
            return_successful_key=False,
        )

        if not add_one_to_sipm_location:
            return f"{int(strip_ID)}-{int(sipm_location)}"
        else:
            return f"{int(strip_ID)}-{int(sipm_location)+1}"

    @classmethod
    def from_json_file(cls, json_filepath, ignored_fields=[]):
        """This class method is meant to be an alternative initializer.
        This class method lets you create a SiPMMeasSummary object from a
        json file. This method calls IdentifiedDict.from_json_file, passing
        to it the given arguments, which match one-to-one to the arguments
        given to IdentifiedDict.from_json_file. For more information on
        such arguments, check the IdentifiedDict.from_json_file documentation."""

        return super().from_json_file(json_filepath, ignored_fields=ignored_fields)

    # Not including SiPMMeasSummary here because in that case I would need
    # to use quotes (that's known as a forward declaration, and it is needed
    # since I am referencing the class from within the class. For more information,
    # check https://peps.python.org/pep-0484/#forward-references ), otherwise this
    # will yield a NameError. I could use a forward reference, but inspecting the
    # signature for the type-hints is not straightforward, because what you get at
    # the inspection phase is actually a forward-reference, not its associated type.
    @staticmethod
    def find_and_translate_dcr_key(
        input_dict: typing.Union[dict, IdentifiedDict], dcr_key_candidates=[]
    ):
        """This method gets the following positional argument:

        - input_dict (dictionary, IdentifiedDict or SiPMMeasSummary): This dictionary
        is expected to contain the dark-counts rate in milli-Hertz per square millimeter.

        This method gets the following keyword argument:

        - dcr_key_candidates (tuple/list of strings): The first string within this
        tuple/list which matches one key in input_dict, except for lower- or upper-case
        conversion, gives the key within input_dict to the dark-counts rate in milli-Hertz
        per square millimeter value of the given dictionary. By default, this method
        considers a number of dark-counts rate key candidates, but the user-given ones
        are considered in the first place.

        If spotted, this method changes the key of the dark-counts rate in milli-Hertz
        per square millimeter to 'DCR_mHz_per_mm2'. If not spotted, this method raises
        a cuex.NoAvailableData exception."""

        htype.check_type(
            input_dict,
            dict,
            IdentifiedDict,
            SiPMMeasSummary,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.find_and_translate_dcr_key", 16657
            ),
        )
        htype.check_type(
            dcr_key_candidates,
            tuple,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.find_and_translate_dcr_key", 85356
            ),
        )
        for x in dcr_key_candidates:
            htype.check_type(
                x,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeasSummary.find_and_translate_dcr_key", 31884
                ),
            )
        # If the key is already there do nothing
        if "DCR_mHz_per_mm2" in input_dict.keys():
            return

        # Avoid creating a view of the keyword argument, otherwise
        # modifications to the list would affect the default value
        # of the keyword argument in subsequent calls to this method
        dcr_key_candidates_ = dcr_key_candidates.copy()

        dcr_key_candidates_.extend(
            [
                "dcr",
                "dcr (mhz/mm2)",
                "dcr_mhz_mm2",
                "dcr_mhz_per_mm2",
                "dcr (mhz/mm^2)",
                "dcr_mhz_mm^2",
                "dcr_mhz_per_mm^2",
                "dcrate",
                "dcrate (mhz/mm2)",
                "dcrate_mhz_mm2",
                "dcr_mhz_per_mm2",
                "dcrate (mhz/mm^2)",
                "dcrate_mhz_mm^2",
                "dcrate_mhz_per_mm^2",
                "dc_rate",
                "dc_rate (mhz/mm2)",
                "dc_rate_mhz_mm2",
                "dc_rate_mhz_per_mm2",
                "dc_rate (mhz/mm^2)",
                "dc_rate_mhz_mm^2",
                "dc_rate_mhz_per_mm^2",
                "darkcountrate",
                "darkcountrate (mhz/mm2)",
                "darkcountrate_mhz_mm2",
                "darkcountrate_mhz_per_mm2",
                "darkcountrate (mhz/mm^2)",
                "darkcountrate_mhz_mm^2",
                "darkcountrate_mhz_per_mm^2",
                "darkcount_rate" "darkcount_rate (mhz/mm2)",
                "darkcount_rate_mhz_mm2",
                "darkcount_rate_mhz_per_mm2",
                "darkcount_rate (mhz/mm^2)",
                "darkcount_rate_mhz_mm^2",
                "darkcount_rate_mhz_per_mm^2",
                "dark_count_rate",
                "dark_count_rate (mhz/mm2)",
                "dark_count_rate_mhz_mm2",
                "dark_count_rate_mhz_per_mm2",
                "dark_count_rate (mhz/mm^2)",
                "dark_count_rate_mhz_mm^2",
                "dark_count_rate_mhz_per_mm^2",
            ]
        )

        # Add first, the default key candidates, then remove
        # the redundancy, just in case the user gave a key
        # candidate which is already considered by default
        # If that was the case, since IdentifiedDict.lowercase_and_remove_redundancy
        # preserves the ordering, the user-given key candidates
        # will still be considered prior to the default ones

        dcr_key_candidates_ = IdentifiedDict.lowercase_and_remove_redundancy(
            dcr_key_candidates_
        )

        _, dcr_key = SiPMMeasSummary.try_retrieving_a_value_from_dict(
            input_dict, key_candidates=dcr_key_candidates_, return_successful_key=True
        )
        # Reaching this point, means that the key was found, because
        # SiPMMeasSummary.try_retrieving_a_value_from_dict did not raise a cuex.NoAvailableData

        input_dict["DCR_mHz_per_mm2"] = input_dict[
            dcr_key
        ]  # Add the DCR under the proper key
        del input_dict[dcr_key]  # Delete the old key

        return

    @staticmethod
    def try_computing_dcr_in_dict(
        input_dict: typing.Union[dict, IdentifiedDict],
        sipm_surface_mm2,
        acquisition_time_key_candidates=[],
        dc_number_key_candidates=[],
    ):
        """This method gets the following positional arguments:

        - input_dict (dictionary, IdentifiedDict or SiPMMeasSummary): This dictionary
        is expected to contain the acquisition-time in seconds and the dark-counts
        number of a SiPMMeasSummary.

        - sipm_surface_mm2 (float): The surface, in square millimeters, of the SiPM
        whose superficial density of DCR is should be computed by this static method.

        This method gets the following keyword arguments:

        - acquisition_time_key_candidates (resp. dc_counts_key_candidates) (tuple/list
        of strings): The first string within this tuple/list which matches one key
        in input_dict, except for lower- or upper-case conversion, gives the key within
        input_dict to the acquisition-time in seconds (resp. dark-counts number) value
        of the given dictionary. By default, this method considers a number of
        acquisition-time (resp. dark-counts number) key candidates, but the user-given
        ones are considered in the first place.

        This method adds the DCR_mHz_per_mm2 key to input_dict. Its value is computed
        as the dark-counts number divided by the acquisition time (which is assumed to
        be in seconds) and the SiPM surface in square millimeters. To do so, this
        method tries to infer the acquisition-time and the dark-counts number from
        the input_dict, using the given key candidates. If this method succeeds to
        find those, and input_dict already contains the DCR_mHz_per_mm2 key, then
        it is overwritten. If this method fails to find any of those, and the
        DCR_mHz_per_mm2 key is already in input_dict, then this method leaves
        input_dict as it is. If this method fails to find any of those, and the
        DCR_mHz_per_mm2 key is not in input_dict, then this method raises a
        cuex.NoAvailableData exception."""

        htype.check_type(
            sipm_surface_mm2,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.try_computing_dcr_in_dict", 28546
            ),
        )
        htype.check_type(
            input_dict,
            dict,
            IdentifiedDict,
            SiPMMeasSummary,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.try_computing_dcr_in_dict", 40583
            ),
        )
        htype.check_type(
            acquisition_time_key_candidates,
            tuple,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.try_computing_dcr_in_dict", 99501
            ),
        )
        htype.check_type(
            dc_number_key_candidates,
            tuple,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.try_computing_dcr_in_dict", 67381
            ),
        )
        for x in acquisition_time_key_candidates + dc_number_key_candidates:
            htype.check_type(
                x,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeasSummary.try_computing_dcr_in_dict", 28102
                ),
            )

        # Avoid creating a view of the keyword argument, otherwise modifications to
        # the list would affect the default value of the keyword argument in subsequent
        # calls to this method
        acquisition_time_key_candidates_ = acquisition_time_key_candidates.copy()

        dc_number_key_candidates_ = dc_number_key_candidates.copy()

        acquisition_time_key_candidates_.extend(
            [
                "acquisition_time",
                "acquisition_time_s",
                "acquisition_time_sec",
                "acquisition_time_secs",
                "acquisition_time_seconds",
            ]
        )

        dc_number_key_candidates_.extend(
            [
                "counts",
                "counts_num",
                "counts_number",
                "dark_counts",
                "dark_counts_num",
                "dark_counts_number",
                "dc_counts",
                "dc_counts_num",
                "dc_counts_number",
                "dc",
                "dc_num",
                "dc_number",
            ]
        )

        # Add first, the default key candidates, then remove
        # the redundancy, just in case the user gave a key
        # candidate which is already considered by default
        # If that was the case, since IdentifiedDict.lowercase_and_remove_redundancy
        # preserves the ordering, the user-given key candidates
        # will still be considered prior to the default ones

        acquisition_time_key_candidates_ = (
            IdentifiedDict.lowercase_and_remove_redundancy(
                acquisition_time_key_candidates_
            )
        )
        dc_number_key_candidates_ = IdentifiedDict.lowercase_and_remove_redundancy(
            dc_number_key_candidates_
        )

        fDataNotAvailable = False
        try:
            acquisition_time = SiPMMeasSummary.try_retrieving_a_value_from_dict(
                input_dict,
                key_candidates=acquisition_time_key_candidates_,
                return_successful_key=False,
            )

            dc_number = SiPMMeasSummary.try_retrieving_a_value_from_dict(
                input_dict,
                key_candidates=dc_number_key_candidates_,
                return_successful_key=False,
            )
        except cuex.NoAvailableData:
            fDataNotAvailable = True

        if fDataNotAvailable:
            if "DCR_mHz_per_mm2" in input_dict.keys():
                return  # Leave the input_dict as it is
            else:
                raise cuex.NoAvailableData(
                    htype.generate_exception_message(
                        "SiPMMeasSummary.try_computing_dcr_in_dict",
                        49988,
                        extra_info=f"The acquisition-time or the dark-counts number could not be found in the given data. This is the available data: {input_dict}.",
                    )
                )
        else:
            input_dict["DCR_mHz_per_mm2"] = (
                1000.0 * dc_number / (acquisition_time * sipm_surface_mm2)
            )
            return  # Add the key to input_dict and return

    @staticmethod
    def try_retrieving_a_value_from_dict(
        input_dict, key_candidates=[], return_successful_key=False
    ):
        """This static method gets the following positional argument:

        - input_dict (dictionary, IdentifiedDict or SiPMMeasSummary): The dictionary
        from which the desired value should be retrieved.

        - key_candidates (tuple/list of strings): The first string within this
        tuple/list which matches one key in input_dict, except for lower- or upper-case
        conversion, gives the key within input_dict to the desired value.

        - return_successful_key (bool): If True, then, in addition to the desired
        value, this method also returns the key within input_dict to such value.

        This method iterates through the given key_candidates. For each string in such
        sequence, this method looks for a key in input_dict.keys() which matches such
        string, except for lower- or upper-case conversion. The value of the first key
        which is found to match, is returned by this method. The value of the successful
        key may be also returned, up to return_successful_key. If the search does not
        yield any match, then this method raises a cuex.NoAvailableData exception."""

        htype.check_type(
            input_dict,
            dict,
            IdentifiedDict,
            SiPMMeasSummary,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.try_retrieving_a_value_from_dict", 60230
            ),
        )
        htype.check_type(
            key_candidates,
            tuple,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeasSummary.try_retrieving_a_value_from_dict", 35664
            ),
        )
        for x in key_candidates:
            htype.check_type(
                x,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeasSummary.try_retrieving_a_value_from_dict", 89936
                ),
            )
        desired_value = None
        fBreak = False  # This flag is used to break the outer loop
        for key_candidate in key_candidates:  # Iterate in order
            if fBreak:
                break
            else:
                for key in input_dict.keys():
                    if key_candidate.lower() == key.lower():
                        successful_key = key
                        desired_value = input_dict[key]
                        fBreak = True
                        break

        if desired_value is None:
            raise cuex.NoAvailableData(
                htype.generate_exception_message(
                    "SiPMMeasSummary.try_retrieving_a_value_from_dict",
                    17561,
                    extra_info=f"The desired value could not be found in the given data. This is the available data: {input_dict}.",
                )
            )
        else:
            if not return_successful_key:
                return desired_value
            else:
                return desired_value, successful_key
            