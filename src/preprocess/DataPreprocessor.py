#######################################################################################
#
# Bugs
#
#   1. In DataPreprocessor.generate_meas_config_files():
#       1.1)    At some point the "date" field that is read from the json file, is
#               overwritten by the creation date of the file.
#
#######################################################################################

import os
import json
import numpy as np
import datetime
import shutil
import struct

import src.utils.htype as htype
import src.utils.custom_exceptions as cuex

class DataPreprocessor:

    def __init__(self,  input_folderpath,
                        gain_base='gain',
                        darknoise_base='darknoise',
                        timestamp_prefix='ts_',
                        binary_extension='wfm',
                        key_separator='_',
                        verbose=True):

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

        htype.check_type(   input_folderpath, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.__init__", 
                                                                                89176))
        htype.check_type(   gain_base, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.__init__", 
                                                                                89911))
        htype.check_type(   darknoise_base, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.__init__", 
                                                                                40891))
        htype.check_type(   timestamp_prefix, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.__init__", 
                                                                                31061))
        htype.check_type(   binary_extension, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.__init__", 
                                                                                31061))
        htype.check_type(   key_separator, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.__init__", 
                                                                                47299))        
        htype.check_type(   verbose, bool,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.__init__", 
                                                                                79488))

        self.__input_folderpath = input_folderpath
        self.__gain_base = gain_base
        self.__darknoise_base = darknoise_base
        self.__timestamp_prefix = timestamp_prefix

        self.__ascii_gain_candidates = {}
        self.__ascii_darknoise_candidates = {}
        self.__bin_gain_candidates = {}
        self.__bin_darknoise_candidates = {}

        self.__timestamp_candidates = {}
        
        self.__num_files = 0    # Number of files in the provided folderpath

        for filename in os.listdir(self.__input_folderpath):
            filepath = os.path.join(self.__input_folderpath, filename)
            if os.path.isfile(filepath):
                self.__num_files += 1
                if self.__gain_base in filename and self.__darknoise_base not in filename:
                    aux = DataPreprocessor.find_integer_after_base( filename,               # Possible failures are
                                                                    self.__gain_base,       # anticipated within    
                                                                    separator=key_separator)# find_integer_after_base
                    
                    DataPreprocessor.bin_ascii_splitter(self.__bin_gain_candidates,
                                                        self.__ascii_gain_candidates,
                                                        aux, filepath, binary_extension=binary_extension)

                elif self.__gain_base not in filename and self.__darknoise_base in filename and not filename.startswith(self.__timestamp_prefix):
                    aux = DataPreprocessor.find_integer_after_base( filename,               # Possible failures are
                                                                    self.__darknoise_base,  # anticipated within    
                                                                    separator=key_separator)# find_integer_after_base
                    
                    DataPreprocessor.bin_ascii_splitter(self.__bin_darknoise_candidates,
                                                        self.__ascii_darknoise_candidates,
                                                        aux, filepath, binary_extension=binary_extension)

        for key in self.__ascii_darknoise_candidates.keys():
            for filename in os.listdir(self.__input_folderpath):
                filepath = os.path.join(self.__input_folderpath, filename)
                if os.path.isfile(filepath):
                    if filename==self.__timestamp_prefix+os.path.split(self.__ascii_darknoise_candidates[key])[1]:
                        self.__timestamp_candidates[key]=filepath
                        break

        DataPreprocessor.print_dictionary_info( self.__ascii_gain_candidates, 
                                                'gain', candidate_type='ASCII', verbose=verbose)
        
        DataPreprocessor.print_dictionary_info( self.__ascii_darknoise_candidates, 
                                                'dark noise', candidate_type='ASCII', verbose=verbose)
        
        DataPreprocessor.print_dictionary_info( self.__timestamp_candidates, 
                                                'dark noise', candidate_type='complete ASCII', verbose=verbose)

        DataPreprocessor.print_dictionary_info( self.__bin_gain_candidates, 
                                                'gain', candidate_type='binary', verbose=verbose)
        
        DataPreprocessor.print_dictionary_info( self.__bin_darknoise_candidates, 
                                                'dark noise', candidate_type='binary', verbose=verbose)
        
        aux = len(self.__ascii_darknoise_candidates)-len(self.__timestamp_candidates)   # By construction, if a key exists in self.__timestamp_candidates.keys(),   
        if aux>0:                                                                       # then it exists in self.__ascii_darknoise_candidates.keys()
            print(f"----> There are {aux} ASCII dark noise candidates with no matching time stamps!")

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
    
    @property
    def NumFiles(self):
        return self.__num_files    
    
    def generate_meas_config_files(self,    root_directory,
                                            load_folderpath, 
                                            aux_folderpath, 
                                            backup_folderpath, 
                                            data_folderpath,
                                            wvf_skiprows_identifier='TIME,',
                                            ts_skiprows_identifier='X:',
                                            data_delimiter=',',
                                            path_to_json_default_values=None,
                                            sipms_per_strip=None,
                                            strips_ids=None,
                                            verbose=True):
        
        """This method gets the following mandatory positional arguments:
         
        - root_directory (string): Path which points to an existing directory,
        which is consireded to be the root directory. Every path which is
        written by this function is relative to this root directory.
        - load_folderpath (string): Path to a folder where the DarkNoiseMeas and
        GainMeas configuration json files will be saved. It must be contained,
        at an arbitrary depth, within the root directory.
        - aux_folderpath (string): Path to a folder where the WaveformSet and 
        Waveform json configuration files will be saved. It must be contained,
        at an arbitrary depth, within the root directory.
        - backup_folderpath (string): A backup of the raw input data, regardless
        if it's a binary or an ASCII measurement file, will be saved in this 
        folder. It must be contained, at an arbitrary depth, within the root 
        directory.
        - data_folderpath (string): Clean data files (following the unified format
        of one column with no headers), regardless its original format, will be 
        saved in this folder. Time stamp data files, if applicable, will be also 
        saved in this folder, following the same format.  It must be contained,
        at an arbitrary depth, within the root directory.

        This method gets the following optional keyword arguments:

        - wvf_skiprows_identifier (string): This parameter only makes a difference
        for ASCII input files. It is given to DataPreprocessor.process_file()
        as skiprows_identifier for the case where files hosting ASCII waveform 
        sets are processed. Check DataPreprocessor.process_file docstring for 
        more information on this parameter.
        - ts_skiprows_identifier (string): This parameter only makes a difference 
        for dark noise ASCII measurements. It is given to 
        DataPreprocessor.process_file() as skiprows_identifier for the case where 
        files hosting ASCII time stamps are processed. Check 
        DataPreprocessor.process_file docstring for more information on this 
        parameter.
        - data_delimiter (string): This parameter only makes a difference for 
        ASCII measurements. It is given to DataPreprocessor.process_file() as 
        data_delimiter. It is used to separate entries of the different columns 
        of the ASCII input data files.
        - path_to_json_default_values (string): If it is not none, it should be
        the path to a json file from which some default values may be read.
        - sipms_per_strip (positive integer): The number of SiPMs per strip. If 
        it is not None, the electronic_board_socket and sipm_location fields will 
        be inferred. To do so, for each type of measurement (namely ascii gain,
        ascii dark noise, binary gain and binary dark noise), the candidates
        are sorted according to its keys (p.e. for ascii gain measurements,
        they are sorted according to the keys of self.__ascii_gain_candidates) 
        associating an iterator value i>=0 to each candidate, so that the 
        electronic_board_socket value is inferred as (i//sipms_per_strip)+1 and 
        the sipm_location value is inferred as (i%sipms_per_strip)+1.
        - strips_ids (list of integers): Its value only makes a difference if
        sipms_per_strip is defined. In such case (and if it is defined), then for
        each type of measurement, strips_ids[i] is assumed to be the strip_ID
        field for the k-th measurement candidate, where k takes values from
        i*sipms_per_strip to ((i+1)*sipms_per_strip)-1. To this end, it is required
        that the number of candidates for whichever type of measurement is a 
        multiple of sipms_per_strip.
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
        'time_resolution', 'points_per_wvf', 'wvfs_to_read' or 'date', among 
        others. The remaining necessary information, namely

            - signal_magnitude (str),
            - delta_t_wf (float),
            - set_name (str),
            - creation_dt_offset_min (float),
            - delivery_no (int),
            - set_no (int),
            - tray_no (int),
            - meas_no (int),
            - strip_ID (int),
            - meas_ID (str),
            - date (str),
            - location (str),
            - operator (str),
            - setup_ID (str),
            - system_characteristics (str),
            - thermal_cycle (int),
            - elapsed_cryo_time_min (float),
            - electronic_board_number (int),
            - electronic_board_location (str),
            - electronic_board_socket (int),
            - sipm_location (int),
            - overvoltage_V (float),
            - PDE (float),
            - status (str),
            - LED_voltage_V (float) and
            - threshold_mV (float),

        is taken from the json file given to path_to_json_default_values, if 
        it is available there and the values comply with the expected types. 
        The user is interactively a interactively asked for the remaining 
        necessary information which could not be retrieved from the given json 
        file."""
        
        htype.check_type(   root_directory, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                44391))
        if not os.path.isdir(root_directory):
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.generate_meas_config_files", 
                                                                        61665,
                                                                        extra_info=f"Path {root_directory} does not exist or is not a directory."))
        htype.check_type(   load_folderpath, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                47131))
        if not os.path.isdir(load_folderpath):
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.generate_meas_config_files", 
                                                                        44692,
                                                                        extra_info=f"Path {load_folderpath} does not exist or is not a directory."))
        
        if not DataPreprocessor.path_is_contained_in_dir(load_folderpath, root_directory):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.find_integer_after_base", 
                                                                                    24323,
                                                                                    extra_info=f"{load_folderpath} is not contained within {root_directory}."))
        htype.check_type(   aux_folderpath, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                96001))
        if not os.path.isdir(aux_folderpath):
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.generate_meas_config_files", 
                                                                        54053,
                                                                        extra_info=f"Path {aux_folderpath} does not exist or is not a directory."))
    
        if not DataPreprocessor.path_is_contained_in_dir(aux_folderpath, root_directory):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.find_integer_after_base", 
                                                                                    79077,
                                                                                    extra_info=f"{aux_folderpath} is not contained within {root_directory}."))
        htype.check_type(   backup_folderpath, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                31414))
        if not os.path.isdir(backup_folderpath):
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.generate_meas_config_files", 
                                                                        74660,
                                                                        extra_info=f"Path {backup_folderpath} does not exist or is not a directory."))
        
        if not DataPreprocessor.path_is_contained_in_dir(backup_folderpath, root_directory):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.find_integer_after_base", 
                                                                                    74514,
                                                                                    extra_info=f"{backup_folderpath} is not contained within {root_directory}."))
        htype.check_type(   data_folderpath, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                46722))
        if not os.path.isdir(data_folderpath):
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.generate_meas_config_files", 
                                                                        21355,
                                                                        extra_info=f"Path {data_folderpath} does not exist or is not a directory."))
        
        if not DataPreprocessor.path_is_contained_in_dir(data_folderpath, root_directory):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.find_integer_after_base", 
                                                                                    84451,
                                                                                    extra_info=f"{data_folderpath} is not contained within {root_directory}."))
        htype.check_type(   wvf_skiprows_identifier, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                42451))
        htype.check_type(   ts_skiprows_identifier, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                42451))
        htype.check_type(   data_delimiter, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                79439))
        fReadDefaultsFromFile = False
        if path_to_json_default_values is not None:
            htype.check_type(   path_to_json_default_values, str,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                    91126))
            fReadDefaultsFromFile = True

        fInferrFields = False
        if sipms_per_strip is not None:
            htype.check_type(   sipms_per_strip, int, np.int64,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                    89701))
            if sipms_per_strip<1:
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                        12924))
            fInferrFields = True
            fAssignStripID = False

            if strips_ids is not None:  # Yes, only check strips_ids if sipms_per_strip is defined
                htype.check_type(   strips_ids, list,
                                    exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                        42323))
                for elem in strips_ids:
                    htype.check_type(   elem, int, np.int64,
                                        exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                            61191))
                p1 = len(self.ASCIIGainCandidates)%sipms_per_strip!=0
                p2 = len(self.ASCIIDarkNoiseCandidates)%sipms_per_strip!=0
                p3 = len(self.BinaryGainCandidates)%sipms_per_strip!=0
                p4 = len(self.BinaryDarkNoiseCandidates)%sipms_per_strip!=0

                if p1 or p2 or p3 or p4:
                    raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                            39450,
                                                                                            extra_info=f"The number of candidates for at least one type of measurement is not a multiple of {sipms_per_strip}. The provided strip IDs cannot be automatically assigned to the candidates."))
                max_candidates = max(   len(self.ASCIIGainCandidates), 
                                        len(self.ASCIIDarkNoiseCandidates), 
                                        len(self.BinaryGainCandidates), 
                                        len(self.BinaryDarkNoiseCandidates))
                
                if len(strips_ids)<(max_candidates/sipms_per_strip):    # max_candidates is a multiple of sipms_per_strip
                    raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files",
                                                                                            55152,
                                                                                            extra_info=f"For at least one type of measurement, the number of provided strip IDs is not enough for all of the given candidates. The provided strip IDs cannot be automatically assigned to the candidates."))
                fAssignStripID = True

        htype.check_type(   verbose, bool,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_meas_config_files", 
                                                                                92127))
        for key in self.ASCIIDarkNoiseCandidates.keys():
            if key not in self.TimeStampCandidates.keys():
                raise cuex.NoAvailableData(htype.generate_exception_message(    "DataPreprocessor.generate_meas_config_files", 
                                                                                44613,
                                                                                extra_info=f"ASCII Dark noise measurement candidate with key={key} lacks a time stamp."))
                                                                                # As of this point, every key which belongs to self.ASCIIDarkNoiseCandidates,
                                                                                # is also present in self.TimeStampCandidates
        queried_wvf_fields = {'signal_magnitude':str}
        read_wvf_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_wvf_fields_from_file, queried_wvf_fields = DataPreprocessor.try_grabbing_from_json(queried_wvf_fields, 
                                                                                                    path_to_json_default_values,
                                                                                                    verbose=verbose)
        queried_once_wvf_fields = {}
        if bool(queried_wvf_fields):    # True if queried_wvfs_fields is not empty
            queried_once_wvf_fields, queried_wvf_fields = DataPreprocessor.query_dictionary_splitting(queried_wvf_fields)

        queried_wvfset_fields = {   #'separator':str,   # In the case of ASCII files, our oscilloscope always outputs 
                                                        # the number of samples, i.e. points_per_wvf, so we won't need 
                                                        # a separator in such case. For the case of binary files, the
                                                        # read process does not need a separator either.
                                    'delta_t_wf':float, # Only queried for ASCII gain meas. For the rest 
                                                        # of cases, it is computed from the input data.
                                    'set_name':str,
                                    'creation_dt_offset_min':float}
        read_wvfset_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_wvfset_fields_from_file, queried_wvfset_fields = DataPreprocessor.try_grabbing_from_json(  queried_wvfset_fields, 
                                                                                                            path_to_json_default_values,
                                                                                                            verbose=verbose)
        queried_once_wvfset_fields = {}
        if bool(queried_wvfset_fields): # True if queried_wvfset_fields is not empty
            queried_once_wvfset_fields, queried_wvfset_fields = DataPreprocessor.query_dictionary_splitting(queried_wvfset_fields)

        queried_sipmmeas_fields = { 'delivery_no':int,
                                    'set_no':int,
                                    'tray_no':int,
                                    'meas_no':int,
                                    'strip_ID':int,
                                    'meas_ID':str,
                                    'date':str,         # The 'date' field will be queried, although the creation
                                    'location':str,     # date of the file will be available as default value
                                    'operator':str,
                                    'setup_ID':str,
                                    'system_characteristics':str,
                                    'thermal_cycle':int,
                                    'elapsed_cryo_time_min':float,
                                    'electronic_board_number':int,
                                    'electronic_board_location':str,
                                    'electronic_board_socket':int,
                                    'sipm_location':int,
                                    'overvoltage_V':float,
                                    'PDE':float,
                                    'status':str}
        
        inferred_sipmmeas_fields = {} 
        if fInferrFields:
            inferred_sipmmeas_fields = {'electronic_board_socket':queried_sipmmeas_fields['electronic_board_socket'],
                                        'sipm_location':queried_sipmmeas_fields['sipm_location']}
            del queried_sipmmeas_fields['electronic_board_socket']
            del queried_sipmmeas_fields['sipm_location']

            if fAssignStripID:
                inferred_sipmmeas_fields['strip_ID'] = queried_sipmmeas_fields['strip_ID']
                del queried_sipmmeas_fields['strip_ID']

            if not fAssignStripID:
                if not DataPreprocessor.yes_no_translator(input(f"In function DataPreprocessor.generate_meas_config_files(): The values for the fields 'electronic_board_socket' and 'sipm_location' will be inferred according to the filepaths ordering and the value given to the 'sipms_per_strip' ({sipms_per_strip}) parameter. Do you want to continue? (y/n)")):
                    return
            else:
                if not DataPreprocessor.yes_no_translator(input(f"In function DataPreprocessor.generate_meas_config_files(): The values for the fields 'electronic_board_socket', 'sipm_location' and 'strip_ID', will be inferred according to the filepaths ordering and the values given to the 'sipms_per_strip' ({sipms_per_strip}) and the 'strips_ids' ({strips_ids}) parameters. Do you want to continue? (y/n)")):
                    return         

        read_sipmmeas_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_sipmmeas_fields_from_file, queried_sipmmeas_fields = DataPreprocessor.try_grabbing_from_json(  queried_sipmmeas_fields, 
                                                                                                                path_to_json_default_values,
                                                                                                                verbose=verbose)
        queried_once_sipmmeas_fields = {}
        if bool(queried_sipmmeas_fields): # True if queried_sipmmeas_fields is not empty
            queried_once_sipmmeas_fields, queried_sipmmeas_fields = DataPreprocessor.query_dictionary_splitting(queried_sipmmeas_fields)

        queried_gainmeas_fields = { 'LED_voltage_V':float}

        inferred_gainmeas_fields = {}
        if fInferrFields:
            inferred_gainmeas_fields.update(inferred_sipmmeas_fields)

        read_gainmeas_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_gainmeas_fields_from_file, queried_gainmeas_fields = DataPreprocessor.try_grabbing_from_json(  queried_gainmeas_fields, 
                                                                                                                path_to_json_default_values,
                                                                                                                verbose=verbose)
        queried_once_gainmeas_fields = {}
        if bool(queried_gainmeas_fields): # True if queried_gainmeas_fields is not empty
            queried_once_gainmeas_fields, queried_gainmeas_fields = DataPreprocessor.query_dictionary_splitting(queried_gainmeas_fields)
        queried_gainmeas_fields.update(queried_sipmmeas_fields)
        
        queried_darknoisemeas_fields = {'threshold_mV':float}       # The acquisition time is not queried because 
                                                                    # it is computed from the time stamp data
        
        inferred_darknoisemeas_fields = {}
        if fInferrFields:
            inferred_darknoisemeas_fields.update(inferred_sipmmeas_fields)

        read_darknoisemeas_fields_from_file = {}
        if fReadDefaultsFromFile:
            read_darknoisemeas_fields_from_file, queried_darknoisemeas_fields = DataPreprocessor.try_grabbing_from_json(    queried_darknoisemeas_fields, 
                                                                                                                            path_to_json_default_values,
                                                                                                                            verbose=verbose)
        queried_once_darknoisemeas_fields = {}
        if bool(queried_darknoisemeas_fields): # True if queried_darknoisemeas_fields is not empty
            queried_once_darknoisemeas_fields, queried_darknoisemeas_fields = DataPreprocessor.query_dictionary_splitting(queried_darknoisemeas_fields)
        queried_darknoisemeas_fields.update(queried_sipmmeas_fields)

        translator = {  'Horizontal Units':[str, 'time_unit'],
                        'Vertical Units':[str, 'signal_unit'],
                        'Sample Interval':[float, 'time_resolution'],
                        'Record Length':[int, 'points_per_wvf'],
                        'FastFrame Count':[int, 'wvfs_to_read'],
                        'creation_date':[str, 'date'],
                        'average_delta_t_wf':[float, 'delta_t_wf'],             # Can be used in every case but ASCII Gain - in such
                                                                                # case the user needs to manually input this value
                        'acquisition_time':[float, 'acquisition_time_min']}

        # Query unique-query data and add update them with the default values gotten from the json file
        print("Let us retrieve the unique-query fields. These fields will apply for every measurement in this DataPreprocessor instance.")
        aux_wvf_dict            = DataPreprocessor.query_fields_in_dictionary(queried_once_wvf_fields,          default_dict=None)
        aux_wvf_dict            .update(read_wvf_fields_from_file)
        aux_wvfset_dict         = DataPreprocessor.query_fields_in_dictionary(queried_once_wvfset_fields,       default_dict=None)
        aux_wvfset_dict         .update(read_wvfset_fields_from_file)
        aux_sipmmeas_dict       = DataPreprocessor.query_fields_in_dictionary(queried_once_sipmmeas_fields,     default_dict=None)
        aux_sipmmeas_dict       .update(read_sipmmeas_fields_from_file)
        aux_gainmeas_dict       = DataPreprocessor.query_fields_in_dictionary(queried_once_gainmeas_fields,     default_dict=None)
        aux_gainmeas_dict       .update(read_gainmeas_fields_from_file)
        aux_darknoisemeas_dict  = DataPreprocessor.query_fields_in_dictionary(queried_once_darknoisemeas_fields,default_dict=None)
        aux_darknoisemeas_dict  .update(read_darknoisemeas_fields_from_file)

        aux_gainmeas_dict.update(aux_sipmmeas_dict)
        aux_darknoisemeas_dict.update(aux_sipmmeas_dict)

        for i, key in enumerate(sorted(self.ASCIIGainCandidates.keys())):

            aux = DataPreprocessor.process_file(self.ASCIIGainCandidates[key],
                                                *translator.keys(),
                                                destination_folderpath=data_folderpath,
                                                backup_folderpath=backup_folderpath,
                                                get_creation_date=True,                 # Extracts the creation date to aux['creation_date']
                                                overwrite_files=False,
                                                ndecimals=18,
                                                verbose=verbose,
                                                is_ASCII=True,
                                                contains_timestamp=False,
                                                skiprows_identifier=wvf_skiprows_identifier,
                                                parameters_delimiter=',',
                                                data_delimiter=data_delimiter,
                                                casting_functions=tuple([translator[key][0] for key in translator.keys()]))
            
            print(f"Let us retrieve some information for the waveform set in {self.ASCIIGainCandidates[key]}")

            aux_wvf_dict.update({translator['Horizontal Units'][1]:aux['Horizontal Units'],
                                translator['Vertical Units'][1]:aux['Vertical Units']})
            aux_wvf_dict.update(DataPreprocessor.query_fields_in_dictionary(queried_wvf_fields, 
                                                                            default_dict=aux_wvf_dict))
            
            aux_wvfset_dict.update({'wvf_filepath':aux['processed_filepath'],
                                    translator['Sample Interval'][1]:aux['Sample Interval'],
                                    translator['Record Length'][1]:aux['Record Length'],
                                    translator['FastFrame Count'][1]:aux['FastFrame Count']})
            aux_wvfset_dict.update(DataPreprocessor.query_fields_in_dictionary( queried_wvfset_fields,
                                                                                default_dict=aux_wvfset_dict))

            if fInferrFields:
                aux_gainmeas_dict.update({  'electronic_board_socket':(i//sipms_per_strip)+1,
                                            'sipm_location':(i%sipms_per_strip)+1})
                if fAssignStripID:
                    aux_gainmeas_dict.update({  'strip_ID':strips_ids[i//sipms_per_strip]})

            aux_gainmeas_dict.update({translator['creation_date'][1]:aux['creation_date']})                             # The 'date' field will be queried,
            aux_gainmeas_dict.update(DataPreprocessor.query_fields_in_dictionary(   queried_gainmeas_fields,            # although the creation date of the
                                                                                    default_dict=aux_gainmeas_dict))    # file will be available as a 
                                                                                                                        # default value
            output_filepath_base = f"{aux_gainmeas_dict['strip_ID']}-{aux_gainmeas_dict['sipm_location']}-{aux_gainmeas_dict[translator['creation_date'][1]][:10]}"
            # The creation date extracted by process follows the format 'YYYY-MM-DD HH:MM:SS'. 
            # Therefore, aux_gainmeas_dict[translator['creation_date'][1]][:10], gives 'YYYY-MM-DD'.

            _, extension = os.path.splitext(aux['raw_filepath'])        # Preserve the original extension
            new_raw_filename = output_filepath_base+'_raw_gain'+extension
            _ = DataPreprocessor.rename_file(   aux['raw_filepath'], 
                                                new_raw_filename, 
                                                overwrite=True, verbose=verbose)

            _, extension = os.path.splitext(aux['processed_filepath'])  # Preserve the original extension
            new_processed_filename = output_filepath_base+'_processed_gain'+extension
            new_processed_filepath = DataPreprocessor.rename_file(  aux['processed_filepath'], 
                                                                    new_processed_filename, 
                                                                    overwrite=True, verbose=verbose)
            aux_wvfset_dict.update({'wvf_filepath': os.path.relpath(new_processed_filepath,     # The name of the processed filepath has changed,
                                                                    start=root_directory)})     # so we must correct it in aux_wvfset_dict

            wvf_output_filepath = os.path.join(aux_folderpath, output_filepath_base+'_gain_wvf.json')
            aux_wvf_dict = {key:[value] for (key,value) in aux_wvf_dict.items()}    # Waveform.Signs.setter inputs must be lists
            DataPreprocessor.generate_json_file(aux_wvf_dict, wvf_output_filepath)
            aux_wvf_dict = {key:value[0] for (key,value) in aux_wvf_dict.items()}   # Transform back to scalars to preserve
                                                                                    # proper functioning of this method
            aux_wvfset_dict['wvf_extra_info'] = os.path.relpath(wvf_output_filepath,
                                                                start=root_directory)
            wvfset_output_filepath = os.path.join(aux_folderpath, output_filepath_base+'_gain_wvfset.json')
            DataPreprocessor.generate_json_file(aux_wvfset_dict, wvfset_output_filepath)

            aux_gainmeas_dict['wvfset_json_filepath'] = os.path.relpath(wvfset_output_filepath, 
                                                                        start=root_directory)
            gainmeas_output_filepath = os.path.join(load_folderpath, output_filepath_base+'_gainmeas.json')
            DataPreprocessor.generate_json_file(aux_gainmeas_dict, gainmeas_output_filepath)
        
        try:                                    # This field should only be queried for ASCII gain measurements,
            del aux_wvfset_dict['delta_t_wf']   # i.e. we should delete it from aux_wvfset_dict so it is not
        except KeyError:                        # queried again for further measurements. However, if this
            pass                                # parameter was not set as a query-once parameter AND there 
                                                # are no ASCII Gain measurements, then there's no entry in 
                                                # aux_wvfset_dict under the 'delta_t_wf' key so far. Handle
                                                # this case. 
        try:                                                    # The 'delta_t_wf' field might have been moved
            del queried_wvfset_fields['delta_t_wf']             # to queried_once_wvfset_fields. If that's the case
        except KeyError:                                        # case, it won't be queried again, so everything's ok.
            pass                                    

        for i, key in enumerate(sorted(self.ASCIIDarkNoiseCandidates.keys())):

            aux = DataPreprocessor.process_file(self.ASCIIDarkNoiseCandidates[key],
                                                *translator.keys(),
                                                destination_folderpath=data_folderpath,
                                                backup_folderpath=backup_folderpath,
                                                get_creation_date=True,
                                                overwrite_files=False,
                                                ndecimals=18,
                                                verbose=verbose,
                                                is_ASCII=True,
                                                contains_timestamp=False,
                                                skiprows_identifier=wvf_skiprows_identifier,
                                                parameters_delimiter=',',
                                                data_delimiter=data_delimiter,
                                                casting_functions=tuple([translator[key][0] for key in translator.keys()]))
            
            aux_2 = DataPreprocessor.process_file(  self.TimeStampCandidates[key],      # Process the time stamps as well
                                                    destination_folderpath=data_folderpath,
                                                    backup_folderpath=backup_folderpath,
                                                    get_creation_date=False,
                                                    overwrite_files=False,
                                                    ndecimals=10,
                                                    verbose=verbose,
                                                    is_ASCII=True,
                                                    contains_timestamp=True,            # Extracts the acquisition time to aux_2['acquisition_time']
                                                    skiprows_identifier=ts_skiprows_identifier,
                                                    data_delimiter=data_delimiter)
            
            print(f"Let us retrieve some information for the waveform set in {self.ASCIIDarkNoiseCandidates[key]}")

            aux_wvf_dict.update({translator['Horizontal Units'][1]:aux['Horizontal Units'],
                                translator['Vertical Units'][1]:aux['Vertical Units']})
            aux_wvf_dict.update(DataPreprocessor.query_fields_in_dictionary(queried_wvf_fields, 
                                                                            default_dict=aux_wvf_dict))

            aux_wvfset_dict.update({'wvf_filepath':aux['processed_filepath'],
                                    translator['Sample Interval'][1]:aux['Sample Interval'],
                                    translator['Record Length'][1]:aux['Record Length'],
                                    translator['FastFrame Count'][1]:aux['FastFrame Count'],
                                    'timestamp_filepath':aux_2['processed_filepath'],
                                    translator['average_delta_t_wf'][1]:aux_2['average_delta_t_wf']})   # Extracting this value is not necessary
                                                                                                        # for the Dark Noise case.
            aux_wvfset_dict.update(DataPreprocessor.query_fields_in_dictionary( queried_wvfset_fields,
                                                                                default_dict=aux_wvfset_dict))
            
            if fInferrFields:
                aux_darknoisemeas_dict.update({ 'electronic_board_socket':(i//sipms_per_strip)+1,
                                                'sipm_location':(i%sipms_per_strip)+1})
                if fAssignStripID:
                    aux_darknoisemeas_dict.update({ 'strip_ID':strips_ids[i//sipms_per_strip]})

            aux_darknoisemeas_dict.update({translator['acquisition_time'][1]:aux_2['acquisition_time']/60.})    # The acquisition time is not queried.
                                                                                                                # Here, I am assuming that the time 
                                                                                                                # stamp unit is the second.
                                                                                                                                # It is computed from the time stamp.
            aux_darknoisemeas_dict.update({translator['creation_date'][1]:aux['creation_date']})                                # The 'date' field will be queried,
            aux_darknoisemeas_dict.update(DataPreprocessor.query_fields_in_dictionary(  queried_darknoisemeas_fields,           # although the creation date of the
                                                                                        default_dict=aux_darknoisemeas_dict))   # file will be available as a 
                                                                                                                                # default value
            output_filepath_base = f"{aux_darknoisemeas_dict['strip_ID']}-{aux_darknoisemeas_dict['sipm_location']}-{aux_darknoisemeas_dict[translator['creation_date'][1]][:10]}"    
            # The creation date extracted by process follows the format 'YYYY-MM-DD HH:MM:SS'. 
            # Therefore, aux_darknoisemeas_dict[translator['creation_date'][1]][:10], gives 'YYYY-MM-DD'.

            _, extension = os.path.splitext(aux['raw_filepath'])
            new_raw_filename = output_filepath_base+'_raw_darknoise'+extension
            _ = DataPreprocessor.rename_file(   aux['raw_filepath'], 
                                                new_raw_filename, 
                                                overwrite=True, verbose=verbose)

            _, extension = os.path.splitext(aux['processed_filepath'])
            new_processed_filename = output_filepath_base+'_processed_darknoise'+extension
            new_processed_filepath = DataPreprocessor.rename_file(  aux['processed_filepath'], 
                                                                    new_processed_filename, 
                                                                    overwrite=True, verbose=verbose)
            aux_wvfset_dict.update({'wvf_filepath':os.path.relpath( new_processed_filepath,
                                                                    start=root_directory)})
            
            _, extension = os.path.splitext(aux_2['processed_filepath'])
            new_processed_filename = output_filepath_base+'_processed_ts_darknoise'+extension
            new_processed_filepath = DataPreprocessor.rename_file(  aux_2['processed_filepath'], 
                                                                    new_processed_filename, 
                                                                    overwrite=True, verbose=verbose)
            aux_wvfset_dict.update({'timestamp_filepath':os.path.relpath(   new_processed_filepath,     # The name of the processed timestamp filepath has changed,
                                                                            start=root_directory)})     # so we must correct it in aux_wvfset_dict

            wvf_output_filepath = os.path.join(aux_folderpath, output_filepath_base+'_darknoise_wvf.json')
            aux_wvf_dict = {key:[value] for (key,value) in aux_wvf_dict.items()}    # Waveform.Signs.setter inputs must be lists
            DataPreprocessor.generate_json_file(aux_wvf_dict, wvf_output_filepath)
            aux_wvf_dict = {key:value[0] for (key,value) in aux_wvf_dict.items()}   # Transform back to scalars to preserve
                                                                                    # proper functioning of this method
            aux_wvfset_dict['wvf_extra_info'] = os.path.relpath(wvf_output_filepath,
                                                                start=root_directory)
            wvfset_output_filepath = os.path.join(aux_folderpath, output_filepath_base+'_darknoise_wvfset.json')
            DataPreprocessor.generate_json_file(aux_wvfset_dict, wvfset_output_filepath)

            aux_darknoisemeas_dict['wvfset_json_filepath'] = os.path.relpath(   wvfset_output_filepath,
                                                                                start=root_directory)
            darknoisemeas_output_filepath = os.path.join(load_folderpath, output_filepath_base+'_darknoisemeas.json')
            DataPreprocessor.generate_json_file(aux_darknoisemeas_dict, darknoisemeas_output_filepath)

        try: 
            del aux_wvfset_dict['timestamp_filepath']   # Binary Gain measurements won't overwrite
                                                        # the 'timestamp_filepath' entry of wvfset_dict
                                                        # via dictionary update. We need to manually
                                                        # remove it, otherwise all of our binary gain 
                                                        # measurements will count on a fixed erroneous
                                                        # 'timestamp_filepath' entry.
        except KeyError:    # This exception can happen if no ASCII dark noise 
                            # measurement was processed. In such case, there's 
                            # no entry within aux_wvfset_dict under the
                            # 'timestamp_filepath' key.
            pass 

        for i, key in enumerate(sorted(self.BinaryGainCandidates.keys())):

            aux = DataPreprocessor.process_file(self.BinaryGainCandidates[key],
                                                destination_folderpath=data_folderpath,
                                                backup_folderpath=backup_folderpath,
                                                get_creation_date=True,                 # Extracts the creation date to aux['creation_date']
                                                overwrite_files=False,
                                                ndecimals=18,
                                                verbose=verbose,
                                                is_ASCII=False,
                                                contains_timestamp=True)    # Even though this is a gain-case,
                                                                            # we need to process the time stamp
                                                                            # to extract the 'delta_t_wf' field.
            os.remove(aux['processed_ts_filepath'])     # We do not want to keep the processed timestamp file though
            
            print(f"Let us retrieve some information for the waveform set in {self.BinaryGainCandidates[key]}")

            aux_wvf_dict.update({translator['Horizontal Units'][1]:aux['Horizontal Units'],
                                translator['Vertical Units'][1]:aux['Vertical Units']})
            aux_wvf_dict.update(DataPreprocessor.query_fields_in_dictionary(queried_wvf_fields, 
                                                                            default_dict=aux_wvf_dict))
            
            aux_wvfset_dict.update({'wvf_filepath':aux['processed_filepath'],
                                    translator['Sample Interval'][1]:aux['Sample Interval'],
                                    translator['Record Length'][1]:aux['Record Length'],
                                    translator['FastFrame Count'][1]:aux['FastFrame Count'],
                                    translator['average_delta_t_wf'][1]:aux['average_delta_t_wf']})
            aux_wvfset_dict.update(DataPreprocessor.query_fields_in_dictionary( queried_wvfset_fields,
                                                                                default_dict=aux_wvfset_dict))

            if fInferrFields:
                aux_gainmeas_dict.update({  'electronic_board_socket':(i//sipms_per_strip)+1,
                                            'sipm_location':(i%sipms_per_strip)+1})
                if fAssignStripID:
                    aux_gainmeas_dict.update({  'strip_ID':strips_ids[i//sipms_per_strip]})

            aux_gainmeas_dict.update({translator['creation_date'][1]:aux['creation_date']})                             # The 'date' field will be queried,
            aux_gainmeas_dict.update(DataPreprocessor.query_fields_in_dictionary(   queried_gainmeas_fields,            # although the creation date of the
                                                                                    default_dict=aux_gainmeas_dict))    # file will be available as a 
                                                                                                                        # default value
            output_filepath_base = f"{aux_gainmeas_dict['strip_ID']}-{aux_gainmeas_dict['sipm_location']}-{aux_gainmeas_dict[translator['creation_date'][1]][:10]}"
            # The creation date extracted by process follows the format 'YYYY-MM-DD HH:MM:SS'. 
            # Therefore, aux_gainmeas_dict[translator['creation_date'][1]][:10], gives 'YYYY-MM-DD'.
            
            _, extension = os.path.splitext(aux['raw_filepath'])
            new_raw_filename = output_filepath_base+'_raw_gain'+extension
            _ = DataPreprocessor.rename_file(   aux['raw_filepath'], 
                                                new_raw_filename, 
                                                overwrite=True, verbose=verbose)

            _, extension = os.path.splitext(aux['processed_filepath'])
            new_processed_filename = output_filepath_base+'_processed_gain'+extension
            new_processed_filepath = DataPreprocessor.rename_file(  aux['processed_filepath'], 
                                                                    new_processed_filename, 
                                                                    overwrite=True, verbose=verbose)
            aux_wvfset_dict.update({'wvf_filepath':os.path.relpath( new_processed_filepath,
                                                                    start=root_directory)})

            wvf_output_filepath = os.path.join(aux_folderpath, output_filepath_base+'_gain_wvf.json')
            aux_wvf_dict = {key:[value] for (key,value) in aux_wvf_dict.items()}    # Waveform.Signs.setter inputs must be lists
            DataPreprocessor.generate_json_file(aux_wvf_dict, wvf_output_filepath)
            aux_wvf_dict = {key:value[0] for (key,value) in aux_wvf_dict.items()}   # Transform back to scalars to preserve
                                                                                    # proper functioning of this method
            aux_wvfset_dict['wvf_extra_info'] = os.path.relpath(wvf_output_filepath,
                                                                start=root_directory)
            wvfset_output_filepath = os.path.join(aux_folderpath, output_filepath_base+'_gain_wvfset.json')
            DataPreprocessor.generate_json_file(aux_wvfset_dict, wvfset_output_filepath)

            aux_gainmeas_dict['wvfset_json_filepath'] = os.path.relpath(wvfset_output_filepath,
                                                                        start=root_directory)
            gainmeas_output_filepath = os.path.join(load_folderpath, output_filepath_base+'_gainmeas.json')
            DataPreprocessor.generate_json_file(aux_gainmeas_dict, gainmeas_output_filepath)

        for i, key in enumerate(sorted(self.BinaryDarkNoiseCandidates.keys())):

            aux = DataPreprocessor.process_file(self.BinaryDarkNoiseCandidates[key],
                                                destination_folderpath=data_folderpath,
                                                backup_folderpath=backup_folderpath,
                                                get_creation_date=True,                 # Extracts the creation date to aux['creation_date']
                                                overwrite_files=False,
                                                ndecimals=18,
                                                verbose=verbose,
                                                is_ASCII=False,
                                                contains_timestamp=True)
            
            print(f"Let us retrieve some information for the waveform set in {self.BinaryDarkNoiseCandidates[key]}")

            aux_wvf_dict.update({translator['Horizontal Units'][1]:aux['Horizontal Units'],
                                translator['Vertical Units'][1]:aux['Vertical Units']})
            aux_wvf_dict.update(DataPreprocessor.query_fields_in_dictionary(queried_wvf_fields, 
                                                                            default_dict=aux_wvf_dict))
            
            aux_wvfset_dict.update({'wvf_filepath':aux['processed_filepath'],
                                    translator['Sample Interval'][1]:aux['Sample Interval'],
                                    translator['Record Length'][1]:aux['Record Length'],
                                    translator['FastFrame Count'][1]:aux['FastFrame Count'],
                                    'timestamp_filepath':aux['processed_ts_filepath'],
                                    translator['average_delta_t_wf'][1]:aux['average_delta_t_wf']}) # Extracting this value is not necessary
                                                                                                    # for the Dark Noise case.
            aux_wvfset_dict.update(DataPreprocessor.query_fields_in_dictionary( queried_wvfset_fields,
                                                                                default_dict=aux_wvfset_dict))

            if fInferrFields:
                aux_darknoisemeas_dict.update({ 'electronic_board_socket':(i//sipms_per_strip)+1,
                                                'sipm_location':(i%sipms_per_strip)+1})
                if fAssignStripID:
                    aux_darknoisemeas_dict.update({ 'strip_ID':strips_ids[i//sipms_per_strip]})

            aux_darknoisemeas_dict.update({translator['acquisition_time'][1]:aux['acquisition_time']/60.})  # The acquisition time is not queried.
                                                                                                            # Here, I am assuming that the time 
                                                                                                            # stamp unit is the second.
                                                                                                                                # It is computed from the time stamp.
            aux_darknoisemeas_dict.update({translator['creation_date'][1]:aux['creation_date']})                                # The 'date' field will be queried,
            aux_darknoisemeas_dict.update(DataPreprocessor.query_fields_in_dictionary(  queried_darknoisemeas_fields,           # although the creation date of the
                                                                                        default_dict=aux_darknoisemeas_dict))   # file will be available as a 
                                                                                                                                # default value

            output_filepath_base = f"{aux_darknoisemeas_dict['strip_ID']}-{aux_darknoisemeas_dict['sipm_location']}-{aux_darknoisemeas_dict[translator['creation_date'][1]][:10]}"    
            # The creation date extracted by process follows the format 'YYYY-MM-DD HH:MM:SS'. 
            # Therefore, aux_darknoisemeas_dict[translator['creation_date'][1]][:10], gives 'YYYY-MM-DD'.
             
            _, extension = os.path.splitext(aux['raw_filepath'])
            new_raw_filename = output_filepath_base+'_raw_darknoise'+extension
            _ = DataPreprocessor.rename_file(   aux['raw_filepath'], 
                                                new_raw_filename, 
                                                overwrite=True, verbose=verbose)

            _, extension = os.path.splitext(aux['processed_filepath'])
            new_processed_filename = output_filepath_base+'_processed_darknoise'+extension
            new_processed_filepath = DataPreprocessor.rename_file(  aux['processed_filepath'], 
                                                                    new_processed_filename, 
                                                                    overwrite=True, verbose=verbose)
            aux_wvfset_dict.update({'wvf_filepath':os.path.relpath( new_processed_filepath,
                                                                    start=root_directory)})

            _, extension = os.path.splitext(aux['processed_ts_filepath'])
            new_processed_filename = output_filepath_base+'_processed_ts_darknoise'+extension
            new_processed_filepath = DataPreprocessor.rename_file(  aux['processed_ts_filepath'], 
                                                                    new_processed_filename, 
                                                                    overwrite=True, verbose=verbose)
            aux_wvfset_dict.update({'timestamp_filepath':os.path.relpath(   new_processed_filepath,
                                                                            start=root_directory)})

            wvf_output_filepath = os.path.join(aux_folderpath, output_filepath_base+'_darknoise_wvf.json')
            aux_wvf_dict = {key:[value] for (key,value) in aux_wvf_dict.items()}    # Waveform.Signs.setter inputs must be lists
            DataPreprocessor.generate_json_file(aux_wvf_dict, wvf_output_filepath)
            aux_wvf_dict = {key:value[0] for (key,value) in aux_wvf_dict.items()}   # Transform back to scalars to preserve
                                                                                    # proper functioning of this method
            aux_wvfset_dict['wvf_extra_info'] = os.path.relpath(wvf_output_filepath,
                                                                start=root_directory)
            wvfset_output_filepath = os.path.join(aux_folderpath, output_filepath_base+'_darknoise_wvfset.json')
            DataPreprocessor.generate_json_file(aux_wvfset_dict, wvfset_output_filepath)

            aux_darknoisemeas_dict['wvfset_json_filepath'] = os.path.relpath(   wvfset_output_filepath,
                                                                                start=root_directory)
            darknoisemeas_output_filepath = os.path.join(load_folderpath, output_filepath_base+'_darknoisemeas.json')
            DataPreprocessor.generate_json_file(aux_darknoisemeas_dict, darknoisemeas_output_filepath)

        return

    @staticmethod
    def integer_generator():
        integer = -1
        while True:
            integer += 1
            yield integer

    @staticmethod
    def find_integer_after_base(input_string, base, separator='_'):

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

        htype.check_type(   input_string, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.find_integer_after_base", 
                                                                                17544))
        htype.check_type(   base, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.find_integer_after_base", 
                                                                                79240))
        htype.check_type(   separator, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.find_integer_after_base", 
                                                                                87101))
        if DataPreprocessor.count_occurrences(input_string, base)<1:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.find_integer_after_base", 
                                                                                    96145,
                                                                                    extra_info=f"There's not a single occurence of {base} in {input_string}."))

        idx = input_string.find(base, 0)+len(base)      # Take what's after the
        aux = input_string[idx:]                        # first occurrence of base

        if DataPreprocessor.count_occurrences(aux, separator)<2:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.find_integer_after_base", 
                                                                                    42225,
                                                                                    extra_info=f"There must be at least two occurrences of the separator, {separator}, in {aux}."))
        idx = aux.find(separator, 0)+len(separator)     # Take what's
        aux = aux[idx:]                                 # in between
        idx = aux.find(separator, 0)                    # both occurrences
        aux = aux[:idx]                                 # of separator

        try:
            return int(aux)
        except ValueError:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.find_integer_after_base", 
                                                                                    57274,
                                                                                    extra_info=f"{aux} is not broadcastable to integer."))

    @staticmethod
    def count_occurrences(string, substring):

        """This static method takes two mandatory positional arguments:
        
        - string (string)
        - substring (string)
        
        This static method returns an integer which matches the number
        of occurrences of substring within string."""

        htype.check_type(   string, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.count_occurrences", 
                                                                                97796))
        htype.check_type(   substring, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.count_occurrences", 
                                                                                43079))        
        result = 0
        idx = 0
        while True:
            idx = string.find(substring, idx)
            if idx==-1:
                break
            result += 1
            idx += len(substring)
        return result    
    
    @staticmethod
    def bin_ascii_splitter(bin_dict, ascii_dict, key, filepath, binary_extension='wfm'):

        """ This static method gets the following mandatory positional arguments:

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

        htype.check_type(   bin_dict, dict, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.bin_ascii_splitter", 
                                                                                48102))
        htype.check_type(   ascii_dict, dict, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.bin_ascii_splitter", 
                                                                                89915))
        htype.check_type(   filepath, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.bin_ascii_splitter", 
                                                                                47181))
        htype.check_type(   binary_extension, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.bin_ascii_splitter", 
                                                                                38311))
        extension = os.path.splitext(filepath)[1][1:]
        if extension==binary_extension:
            bin_dict[key]=filepath
        else:
            ascii_dict[key]=filepath
        return
    
    @staticmethod
    def print_dictionary_info(dictionary, meas_type, candidate_type='', verbose=False):

        htype.check_type(   dictionary, dict,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.print_dictionary_info", 
                                                                                33719))
        htype.check_type(   meas_type, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.print_dictionary_info", 
                                                                                71881))
        htype.check_type(   candidate_type, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.print_dictionary_info", 
                                                                                38122))
        htype.check_type(   verbose, bool,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.print_dictionary_info", 
                                                                                44422))
        
        print(f"----> Found {len(dictionary)} {candidate_type} candidates to {meas_type} measurements, with keys={list(dictionary.keys())}")
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

        htype.check_type(   input_filepath, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.find_skiprows", 
                                                                                90123))
        htype.check_type(   identifier, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.find_skiprows", 
                                                                                12767))
        found=False
        skiprows=1
        with open(input_filepath, "r") as file:
            for line in file:
                if line.startswith(identifier):
                    found=True
                    break
                skiprows+=1
        if found:
            return skiprows
        else:
            return -1

    @staticmethod
    def parse_headers(  filepath, *identifiers, identifier_delimiter=',', 
                        casting_functions=None, headers_end_identifier=None,
                        return_skiprows=True):
        
        """This static method gets the following compulsory positional arguments:

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

        htype.check_type(   filepath, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.parse_headers", 
                                                                                41251))
        for i in range(len(identifiers)):
            htype.check_type(   identifiers[i], str,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.parse_headers", 
                                                                                    27001))
        htype.check_type(   identifier_delimiter, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.parse_headers", 
                                                                                76281))
        casting_functions_ = [ lambda x : x for y in identifiers]
        if casting_functions is not None:
            htype.check_type(   casting_functions, tuple, list,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.parse_headers", 
                                                                                    38100))
            for i in range(len(casting_functions)):
                if not callable(casting_functions[i]):
                    raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.parse_headers", 
                                                                                            13721))
            if len(casting_functions)!=len(identifiers):
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.parse_headers", 
                                                                                        38293))
            casting_functions_ = casting_functions

        headers_endline = -1
        if headers_end_identifier is not None:
            htype.check_type(   headers_end_identifier, str,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.parse_headers", 
                                                                                    17819))
            headers_endline = DataPreprocessor.find_skiprows(filepath, headers_end_identifier)
        # If headers_endline ends up being -1, then this function will search 
        # through the whole input file to find ocurrences of any given identifier
            
        result = {}
        for i in range(len(identifiers)):
            with open(filepath, "r") as file:
                cont = 0
                line = file.readline()
                while cont!=headers_endline and line:
                    line = file.readline()
                    cont += 1
                    if line.startswith(identifiers[i]):
                        aux = line.strip().split(identifier_delimiter)[-1]
                        if aux!='':
                            result[identifiers[i]]=casting_functions_[i](aux)
                            break   # Stop the search (the while loop)
                                    # only if a value was successfully 
                                    # added to result
        if return_skiprows:
            return result, headers_endline
        else:
            return result

    @staticmethod 
    def remove_non_alpha_bytes(input):

        '''This static method gets the following mandatory positional argument:
        
        - byte_data (bytes): Chain of bytes
        
        This method returns an string which is the result of removing every 
        non-alphabetic character from the given bytes chain and join the
        rest of them to create an string. To do so, this method converts the 
        chain of bytes to a chain of characters, using the built-in method chr() 
        (which uses utf-8 encoding) and then uses the method isalpha() to decide
        whether each character is alphabetic or not.'''

        htype.check_type(   input, bytes,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.remove_non_alpha_bytes", 
                                                                                47131))
        filtered_alphabetic_characters = list(filter(   lambda b: b.isalpha(), 
                                                        [chr(b) for b in input]))
        return ''.join(filtered_alphabetic_characters)

    @staticmethod
    def extract_tek_wfm_metadata(filepath):
        
        """This static method has a fixed purposed and is not tunable via input 
        parameters. This method gets the following mandatory positional argument:
        
        - filepath (str): Path to a binary file which is interpreted to have the
        Tektronix WFM file format. Its extension must be equal to '.wfm'.
        
        This method reads the first 838 bytes of the input WFM file. These bytes
        contain the metadata for the stored fast frames by the Tektronix oscilloscope
        (See the Tektronix Reference Waveform File Format Instructions). This method 
        performs some consistency checks and then returns two dictionaries which host
        meta-data for the FastFrame set which is stored in the provided filepaht. The 
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

        htype.check_type(   filepath, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.extract_tek_wfm_metadata", 
                                                                                48219))
        if not os.path.isfile(filepath):
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.extract_tek_wfm_metadata", 
                                                                        11539,
                                                                        extra_info=f"Path {filepath} does not exist or is not a file."))
        else:
            _, extension = os.path.splitext(filepath)
            if extension!='.wfm':
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.extract_tek_wfm_metadata",
                                                                                        42881,
                                                                                        extra_info=f"The extension of the input file must match '.wfm'."))
        with open(filepath, 'rb') as file:  #Binary-reading mode
            header_bytes = file.read(838)    

        if len(header_bytes)!=838:
            raise cuex.WfmReadException(htype.generate_exception_message(   'DataPreprocessor.extract_tek_wfm_metadata', 
                                                                            47199,
                                                                            extra_info='WFM header comprise 838 bytes. A different value was given.'))
        data_buffer = {}
        data_buffer['byte_order'] = struct.unpack_from('H', header_bytes, offset=0)[0]  # 'H' for Unsigned short (2 bytes)
        if data_buffer['byte_order']==0x0f0f:   # Little-endian: Less significant bytes 
            bo='<'                              # are stored in lower memory addresses
        else:       # Big-endian: More significant bytes 
            bo='>'  # belong to lower memory addresses
        del data_buffer['byte_order']
                    
        data_buffer['version'] = struct.unpack_from(bo+'8s', header_bytes, offset=2)[0]     # '8s' for an 8-bytes string
        # In the tekwfm proof of concept, they check that the version is equal to 3         # Also, adding here a first
        # (otherwise an error is raised). However, they make no use of any of the           # character, either '<' or '>'
        # memory addresses which vary from version 2 to version 3 (you can check the        # indicating the byte order.
        # changes from version 2 to version 3 in page 15 of the waveform file 
        # reference manual). That's why I see no reason to raise an error based on 
        # the version.
        del data_buffer['version']

        data_buffer['record_type'] = struct.unpack_from(bo+'I', header_bytes, offset=122)[0]    # 'I' for a 4-bytes
        if data_buffer['record_type']!=2:                                                       # unsigned int.
            raise cuex.WfmReadException(htype.generate_exception_message(   'DataPreprocessor.extract_tek_wfm_metadata', 
                                                                            67112,
                                                                            extra_info='For normal YT waveforms, the record type must be 2.'))
        del data_buffer['record_type']

        data_buffer['imp_dim_count'] = struct.unpack_from(bo+'I', header_bytes, offset=114)[0]
        if data_buffer['imp_dim_count']!=1:
            raise cuex.WfmReadException(htype.generate_exception_message(   'DataPreprocessor.extract_tek_wfm_metadata', 
                                                                            10925,
                                                                            extra_info='For normal YT waveforms, the number of implicit dimensions must be 1.'))
        del data_buffer['imp_dim_count']

        data_buffer['exp_dim_count'] = struct.unpack_from(bo+'I', header_bytes, offset=118)[0]
        if data_buffer['exp_dim_count']!=1:
            raise cuex.WfmReadException(htype.generate_exception_message(   'DataPreprocessor.extract_tek_wfm_metadata', 
                                                                            48108,
                                                                            extra_info='For normal YT waveforms, the number of explicit dimensions must be 1.'))
        del data_buffer['exp_dim_count']

        data_buffer['exp_dim_1_type'] = struct.unpack_from(bo+'I', header_bytes, offset=244)[0]
        if data_buffer['exp_dim_1_type']!=0:
            raise cuex.WfmReadException(htype.generate_exception_message(   'DataPreprocessor.extract_tek_wfm_metadata', 
                                                                            48108,
                                                                            extra_info="This value must be 0, which matches the case of 'EXPLICIT_SAMPLE'."))
        del data_buffer['exp_dim_1_type']

        data_buffer['time_base_1'] = struct.unpack_from(bo+'I', header_bytes, offset=768)[0]
        if data_buffer['time_base_1']!=0:
            raise cuex.WfmReadException(htype.generate_exception_message(   'DataPreprocessor.extract_tek_wfm_metadata', 
                                                                            48108,
                                                                            extra_info="This value must be 0, which matches the case of 'BASE_TIME'."))
        del data_buffer['time_base_1']

        data_buffer['fastframe'] = struct.unpack_from(bo+'I', header_bytes, offset=78)[0]
        if data_buffer['fastframe']!=1:
            print(f"In extract_tek_wfm_metadata(), data_buffer['fastframe']={data_buffer['fastframe']}")
            print(f"In extract_tek_wfm_metadata(), filepath={filepath}")
            raise cuex.WfmReadException(htype.generate_exception_message(   'DataPreprocessor.extract_tek_wfm_metadata', 
                                                                            30182,
                                                                            extra_info="This value must be 1, which matches the case of a FastFrame set."))
        del data_buffer['fastframe']

        data_buffer['imp_dim_units'] = struct.unpack_from(bo+'20s', header_bytes, offset=508)[0]    # As an example, this value might
        data_buffer['imp_dim_units'] = DataPreprocessor.remove_non_alpha_bytes(data_buffer['imp_dim_units'])         # evaluate to 's'

        data_buffer['exp_dim_units'] = struct.unpack_from(bo+'20s', header_bytes, offset=188)[0]    # As an example, this value might
        data_buffer['exp_dim_units'] = DataPreprocessor.remove_non_alpha_bytes(data_buffer['exp_dim_units'])         # evaluate to 'V'

        data_buffer['Frames'] = 1+struct.unpack_from(bo+'I', header_bytes, offset=72)[0]        # The extracted value matches the number
                                                                                                # number of fast frames minus 1. Adding 1
                                                                                                # gives the number of fast frames in our
                                                                                                # FastFrame set.

        data_buffer['curve_buffer_offset'] = struct.unpack_from(bo+'I', header_bytes, offset=16)[0] # 'I' stands for a 4-bytes long int. This
                                                                                                    # one matches the number of bytes from the
                                                                                                    # beginning of the file to the start of the 
                                                                                                    # curve buffer.
        # 'vscale' for Vertical Scale
        data_buffer['vscale'] = struct.unpack_from(bo+'d', header_bytes, offset=168)[0]         # 'd' stands for an 8-bytes double. This 
                                                                                                # one matches the number of units (p.e. 
                                                                                                # number of volts) per least significant 
                                                                                                # bit (LSB).                                  
        # 'voffset' for Vertical Offset
        data_buffer['voffset'] = struct.unpack_from(bo+'d', header_bytes, offset=176)[0]        # This value matches the distance in vertical 
                                                                                                # units from the dimension zero value to 
                                                                                                # the true zero value. Eventually, the 
                                                                                                # vertical-magnitude values are computed as 
                                                                                                # the curve data times this vscale plus voffset. 
        # 'hscale' for Horizontal Scale
        data_buffer['hscale'] = struct.unpack_from(bo+'d', header_bytes, offset=488)[0]         # This value matches the sample interval 
                                                                                                # for the horizontal implicit dimension, 
                                                                                                # p.e. time per point

        data_buffer['tstart'] = struct.unpack_from(bo+'d', header_bytes, offset=496)[0]         # Trigger position

        # Every waveform in our FastFrame set will have a 'Waveform Update Specification' object, which, in turn, comprises 
        #   1) a 4-bytes unsigned long int
        #   2) an 8-bytes double
        #   3) an 8-bytes double
        #   4) an 4-bytes long
        # The following three pieces of information that we extract: 'tfrac', 'tdatefrac' and 'tdate' match the last three
        # pieces of data we enumerated, respectively, for the first fast frame in the FastFrame set (index 0). Later on,
        # somewhere else in the code, we will need to extract such information for the rest of fast frames.
        
        data_buffer['tfrac[0]'] = struct.unpack_from(bo+'d', header_bytes, offset=788)[0]   # The time from the point the trigger occurred
                                                                                            # to the next data point in the waveform 
                                                                                            # record. This value represents the fraction 
                                                                                            # of the sample time (hscale) from the trigger
                                                                                            # time stamp to the next sample, i.e. the 
                                                                                            # first point of the curve.

        data_buffer['tdatefrac[0]'] = struct.unpack_from(bo+'d', header_bytes, offset=796)[0]   # The fraction of the second when the 
                                                                                                # trigger occurred. This one should be 
                                                                                                # used in combination with 'tdate'.
        data_buffer['tdate[0]'] = struct.unpack_from(bo+'I', header_bytes, offset=804)[0]   # GMT (in seconds from the epoch) when the 
                                                                                            # trigger occurred.
        
        # What we encounter from offset 808 to offset 838 (in bytes) is information regarding the format of the first waveform curve object. 
        # Since every waveform curve object has the same size, the information we extract from this is applicable to every waveform curve 
        # object that we will encounter in the curve buffer (i.e. in our FastFrame set).

        dpre = struct.unpack_from(bo+'I', header_bytes, offset=822)[0]  # The byte offset from the beginning of the curve buffer to the
                                                                        # first point of the record available to the oscilloscope user.
                                                                        # Every curve in the curve buffer has a (fixed) number of artifact-
                                                                        # points which are placed at the beginning of the curve, and which 
                                                                        # serve for oscilloscope interpolation purposes. The same goes 
                                                                        # for the end of the curve, where a number of artifact-poins are 
                                                                        # appended. If we take out such pre and post artifact-points, what 
                                                                        # we are left with is what we are going to call the user-accesible 
                                                                        # part of the curve.
        dpost = struct.unpack_from(bo+'I', header_bytes, offset=826)[0] # The byte offset to the point right after the last user-accesible
                                                                        # point in the curve.
        readbytes = dpost - dpre                                        # Number of bytes per (the user-accesible part of the) curve   
        allbytes = struct.unpack_from(bo+'I', header_bytes, offset=830)[0]  # Bytes per curve (including the interpolation points)
        dt_code = struct.unpack_from(bo+'i', header_bytes, offset=240)[0]   # A code (scalar integer) which indicates the data type of the 
                                                                            # values stored in the curve buffer.
        bps = struct.unpack_from(bo+'b', header_bytes, offset=15)[0]    # Number of bytes per curve data point. Stands for Bytes Per Sample.

        if dt_code==0 and bps==2:
            samples_datatype = np.int16
            useable_samples_no = readbytes//2
        elif dt_code==1 and bps==4:
            samples_datatype = np.int32
            useable_samples_no = readbytes//2
        elif dt_code==2 and bps==4:
            samples_datatype = np.uint32
            useable_samples_no = readbytes//4
        elif dt_code==3 and bps==8:
            samples_datatype = np.uint64
            useable_samples_no = readbytes//8
        elif dt_code==4 and bps==4:
            samples_datatype = np.float32
            useable_samples_no = readbytes//4
        elif dt_code==5 and bps==8:
            samples_datatype = np.float64
            useable_samples_no = readbytes//8
        else:
            raise cuex.WfmReadException(htype.generate_exception_message(   'DataPreprocessor.extract_tek_wfm_metadata', 
                                                                            21230,
                                                                            extra_info="The data-type code is not consistent with the read bytes-per-sample."))
        
        data_buffer['samples_datatype'] = samples_datatype      # Data type of the samples in the curves
        data_buffer['useable_samples_no'] = useable_samples_no  # Number of samples in the user-accesible part of each curve
        data_buffer['samples_no'] = allbytes//bps               # Number of samples in each (whole) curve
        data_buffer['pre-values_no'] = dpre//bps                # Number of pre-appended interpolation samples
        data_buffer['post-values_no'] = (allbytes - dpost)//bps # Number of post-appended interpolation samples

        main_extraction = {}
        main_extraction['Horizontal Units'] = data_buffer['imp_dim_units']
        main_extraction['Vertical Units']   = data_buffer['exp_dim_units']
        main_extraction['Sample Interval']  = data_buffer['hscale']
        main_extraction['Record Length']    = data_buffer['useable_samples_no']
        main_extraction['FastFrame Count']  = data_buffer['Frames']

        supplementary_extraction = {}
        supplementary_extraction['curve_buffer_offset']     = data_buffer['curve_buffer_offset']
        supplementary_extraction['vscale']                  = data_buffer['vscale']
        supplementary_extraction['voffset']                 = data_buffer['voffset']
        supplementary_extraction['tstart']                  = data_buffer['tstart']
        supplementary_extraction['tfrac[0]']                = data_buffer['tfrac[0]']
        supplementary_extraction['tdatefrac[0]']            = data_buffer['tdatefrac[0]']
        supplementary_extraction['tdate[0]']                = data_buffer['tdate[0]']
        supplementary_extraction['samples_datatype']        = data_buffer['samples_datatype']
        supplementary_extraction['samples_no']              = data_buffer['samples_no']
        supplementary_extraction['pre-values_no']           = data_buffer['pre-values_no']
        supplementary_extraction['post-values_no']          = data_buffer['post-values_no']

        # For now, the only reason for this dictionary splitting is simply to make the first returned
        # dictionary resemble that of the ASCII case, i.e. the one returned by 
        # DataPreprocessor.parse_headers(). This might not be the optimal way and may vary in the future.
        
        return main_extraction, supplementary_extraction

    @staticmethod
    def get_raw_and_processed_filepaths(    input_filepath, 
                                            raw_prelabel='raw_', 
                                            processed_prelabel='processed_',
                                            raw_folderpath=None,
                                            processed_folderpath=None):

        """This static methods the following mandatory positional argument:

        - input_filepath (string)

        It also gets the following optional keyword arguments:

        - raw_prelabel (string)
        - processed_prelabel (string)
        - raw_folderpath (None or string)
        - processed_folderpath (None or string)
        
        This static method gets the path to a certain input file, input_filepath, 
        and returns two strings, say raw_filepath and processed_filepath. 
        raw_filepath (resp. processed_filepath) is the path to a file whose name
        matches raw_prelabel (processed_prelabel) plus the name of the input file. 
        If raw_folderpath (processed_folderpath) is None, then the folder path of 
        raw_filepath (processed_filepath) matches that of the input file. If 
        raw_folderpath (processed_folderpath) is defined, then the folder path of 
        raw_filepath (processed_filepath) matches raw_folderpath (processed_folderpath). 
        Also, if the file name of the input file already starts with raw_prelabel, 
        then raw_prelabel is not added to raw_filepath."""

        htype.check_type(   input_filepath, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.get_raw_and_processed_filepaths", 
                                                                                68321))
        htype.check_type(   raw_prelabel, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.get_raw_and_processed_filepaths", 
                                                                                16667))
        htype.check_type(   processed_prelabel, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.get_raw_and_processed_filepaths", 
                                                                                18435)) 
        fMoveRawFile = False    
        if raw_folderpath is not None:
            htype.check_type(   raw_folderpath, str, 
                                exception_message=htype.generate_exception_message( "DataPreprocessor.get_raw_and_processed_filepaths", 
                                                                                    91793))
            if not os.path.isdir(raw_folderpath):
                raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.get_raw_and_processed_filepaths", 
                                                                            93171,
                                                                            extra_info=f"Path {raw_folderpath} does not exist or is not a directory."))
            fMoveRawFile = True

        fMoveProcessedFile = False
        if processed_folderpath is not None:
            htype.check_type(   processed_folderpath, str, 
                                exception_message=htype.generate_exception_message( "DataPreprocessor.get_raw_and_processed_filepaths", 
                                                                                    99171))
            if not os.path.isdir(processed_folderpath):
                raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.get_raw_and_processed_filepaths", 
                                                                            90490,
                                                                            extra_info=f"Path {processed_folderpath} does not exist or is not a directory."))
            fMoveProcessedFile = True

        folderpath, filename = os.path.split(input_filepath)

        raw_folderpath_         =  raw_folderpath       if fMoveRawFile         else folderpath
        processed_folderpath_   =  processed_folderpath if fMoveProcessedFile   else folderpath

        raw_filename = filename if filename.startswith(raw_prelabel) else raw_prelabel+filename
        processed_filename = processed_prelabel+filename

        raw_filepath = os.path.join(raw_folderpath_, raw_filename)
        processed_filepath = os.path.join(processed_folderpath_, processed_filename)

        return raw_filepath, processed_filepath

    @staticmethod
    def process_file(   filepath, 
                        *parameters_identifiers, 
                        destination_folderpath=None,
                        backup_folderpath=None,
                        get_creation_date=False,
                        overwrite_files=False,
                        ndecimals=18,
                        verbose=True,
                        is_ASCII=True,
                        contains_timestamp=False,
                        skiprows_identifier='TIME,',
                        parameters_delimiter=',',
                        data_delimiter=',',
                        casting_functions=None):
        
        """This static method gets the following mandatory positional argument:
        
        - filepath (string): Path to the file which will be processed.

        This function gets the following optional positional arguments:

        - parameters_identifiers (tuple of strings): These parameters only make
        a difference if is_ASCII is True. In such case, they are given to 
        DataPreprocessor.parse_headers() as identifiers. Each one is considered 
        to be the string which precedes the value of a parameter of interest 
        within the input file headers.

        This function also gets the following optional keyword arguments:

        - destination_folderpath (None or string): If it is defined, then the 
        processed file(s) are removed from the input-filepath folder and moved 
        to the folder whose path matches destination_folderpath.

        - backup_folderpath (None or string): If it is defined, then the backup
        of the input file is moved to the folder whose path matches backup_folderpath.

        - get_creation_date (bool): If True, the creation date of the input file 
        is added to the resulting dictionary under the key 'creation_date'. The
        associated value is an string which follows the format 'YYYY-MM-DD HH:MM:SS'.
        If False, no extra entry is added to the resulting dictionary.

        - overwrite_files (boolean): If it set to True (resp. False), the goal files, 
        i.e. the raw and processed files, will (resp. won't) be overwritten if they 
        already exist.

        - ndecimals (int): Number of decimals to use to save the real values to the
        processed file(s). Scientific notation is assumed.

        - verbose (boolean): Whether to print functioning-related messages. 

        - is_ASCII (bool): Indicates whether the input file should be interpreted
        as an ASCII file, or as a binary file (in the Tektronix .WFM file format).
        This parameter, together with contains_timestamp, determines how the input
        file needs to be processed.

        - contains_timestamp (bool): If contains_timestamp is True, then two 
        additional entries are added to the resulting dictionary. First, the average
        of the time difference between consecutive triggers is added under the key
        'average_delta_t_wf'. Second, the time difference between the last trigger 
        and the first trigger of the waveforms dataset stored in the provided 
        filepath, which is added under the key 'acquisition_time'. In such case, 
        the way of computing such entry is different depending on is_ASCII. If 
        is_ASCII is True, then it is assumed that the provided filepath points 
        to an ASCII timestamp file, whose second column hosts the time stamp of 
        the set of triggers of the waveform set, where the i-th time stamp should 
        be understood as a time increment with respect to the (i-1)-th time stamp. 
        In this context, the acquistion time is computed as the sum of all of the 
        entries in the second column of the file. If is_ASCII is False, then it 
        is understood that the provided filepath points to a binary WFM file whose 
        time stamp needs to be extracted and then the acquisition time is computed 
        as the time difference between the last-trigger time stamp and the first-trigger.

        - skiprows_identifier (string): This parameter only makes a difference 
        if is_ASCII is True. In such case, it is passed to DataPreprocessor.parse_headers() 
        as headers_end_identifier, which, in turn, passes it to 
        DataPreprocessor.find_skiprows() as identifier. This string is used to 
        identify the line which immediately precedes the data columns in an
        ASCII data file, say the L-th line. identifier must be defined so 
        that the L-th line starts with identifier, i.e. identifier==header[0:N] 
        must evaluate to True, for some 1<=N<=len(header). This function will 
        also parse the input file for occurrences of the provided,
        parameters_identifiers, from the first line through the L-th line.

        - parameters_delimiter (string): This parameter only makes a difference if
        is_ASCII is True. In such case, it is given to DataPreprocessor.parse_headers() 
        as identifier_delimiter. This string is used to separate each identifier 
        from its value.

        - data_delimiter (string): This parameter only makes a difference if is_ASCII
        is True. In such case, it is given to DataPreprocessor.process_core_data(), 
        which in turn passes it to numpy.loadtxt() as delimiter. It is used to separate 
        entries of the different columns of the input file.

        - casting_functions (tuple/list of functions): This parameter only makes a 
        difference if is_ASCII is True. In such case, it is given to 
        DataPreprocessor.parse_headers() as casting_functions. The i-th function 
        within casting_functions will be used to transform the string read from the 
        input file for the i-th parameter identifier.

        This static method

            - receives a path to an input file, which could be either ASCII or binary 
            (in the Tektronix .WFM file format), 
            - extracts some meta-data from it which is partially returned by this method 
            as a dictionary. Such extraction is carried out by 
            DataPreprocessor.parse_headers() or DataPreprocessor.extract_tek_wfm_metadata()
            for the case where is_ASCII is True or False, respectively.
            - Then, backs it up with a modified name (the backup filename is formed by 
            pre-appending the 'raw_' string to the original filename) to a location which 
            depends on the backup_folderpath parameter (overwritting may occur, up to 
            overwrite_files),
            - crafts one* ASCII file in a unified format which comprises just one 
            column of float values with no headers, whose filename is crafted out of 
            the original filename by pre-appending the 'processed_'* string,
            - saves it to a tunable location (up to the destination_folderpath parameter, 
            overwritting may occur),
            - and adds the raw-and-processed filepaths to the returned dictionary under 
            the keys 'raw_filepath' and 'processed_filepath'*, respectively.

        *In the particular case when is_ASCII is False and contains_timestamp is True, 
        then two processed files are crafted: a processed waveform-set file and a processed 
        timestamp. In such case, the pre-appended strings to the processed filenames 
        are 'processed_' and 'processed_ts_' respectively. Both resulting filepaths are 
        added to the resulting dictionary under the keys 'processed_filepath' and 
        'processed_ts_filepath' respectively.

        Some additional entries could be added to the resulting dictionary, up to the boolean 
        values given to get_creation_date and contains_timestamp. Check such parameters
        documentation for more information. 
        
        For DataPreprocessor.generate_meas_config_files() to work properly, the dictionary 
        returned by this method must, at least, contain the following keys: 'Horizontal Units', 
        'Vertical Units', 'Sample Interval', 'Record Length', 'FastFrame Count', 'creation_date'. 
        If contains_timestamp is True, then the returned dictionary must also contain the 
        following keys: 'average_delta_t_wf' and 'acquisition_time'. If get_creation_date is
        True, then the returned dictionary must also contain the 'creation_date' key."""

        htype.check_type(   filepath, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                17219))
        for i in range(len(parameters_identifiers)):
            htype.check_type(   parameters_identifiers[i], str,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                    43177))
        if destination_folderpath is not None:
            htype.check_type(   destination_folderpath, str, 
                                exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                    85829))
            if not os.path.isdir(destination_folderpath):
                raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.process_file", 
                                                                            21570,
                                                                            extra_info=f"Path {destination_folderpath} does not exist or is not a directory."))
        if backup_folderpath is not None:
            htype.check_type(   backup_folderpath, str, 
                                exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                    32598))
            if not os.path.isdir(backup_folderpath):
                raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.process_file", 
                                                                            47109,
                                                                            extra_info=f"Path {backup_folderpath} does not exist or is not a directory."))
        htype.check_type(   get_creation_date, bool, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                11280))
        htype.check_type(   overwrite_files, bool, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                44161))
        htype.check_type(   ndecimals, int, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                57103))
        htype.check_type(   verbose, bool, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                56912))
        htype.check_type(   is_ASCII, bool, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                67189))
        htype.check_type(   contains_timestamp, bool, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                19270))
        htype.check_type(   skiprows_identifier, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                99170))
        htype.check_type(   parameters_delimiter, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                12851))
        htype.check_type(   data_delimiter, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                64539))
        casting_functions_ = [ lambda x : x for y in parameters_identifiers]
        if casting_functions is not None:
            htype.check_type(   casting_functions, tuple, list,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                    41470))
            for i in range(len(casting_functions)):
                if not callable(casting_functions[i]):
                    raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                            55415))
            if len(casting_functions)!=len(parameters_identifiers):
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.process_file", 
                                                                                        99167))
            casting_functions_ = casting_functions

        result = {}
        tek_wfm_metadata = {}
        if is_ASCII:
            parameters, skiprows = DataPreprocessor.parse_headers(  filepath, 
                                                                    *parameters_identifiers, 
                                                                    identifier_delimiter=parameters_delimiter, 
                                                                    casting_functions=casting_functions_, 
                                                                    headers_end_identifier=skiprows_identifier,
                                                                    return_skiprows=True)
        else:
            parameters, supplementary_extraction = DataPreprocessor.extract_tek_wfm_metadata(filepath)
            tek_wfm_metadata = parameters | supplementary_extraction
        result.update(parameters)

        if get_creation_date:
            result['creation_date'] = DataPreprocessor.get_str_creation_date(filepath)
        
        if is_ASCII:
            file_type_code = 0 if not contains_timestamp else 1
        else:
            file_type_code = 2 if not contains_timestamp else 3
        
        filepaths_dict = DataPreprocessor.process_core_data(filepath,
                                                            file_type_code,
                                                            destination_folderpath=destination_folderpath,
                                                            backup_folderpath=backup_folderpath,
                                                            overwrite_files=overwrite_files,
                                                            skiprows=skiprows if is_ASCII else 0,
                                                            data_delimiter=data_delimiter,
                                                            ndecimals=ndecimals,
                                                            tek_wfm_metadata=tek_wfm_metadata)
        result.update(filepaths_dict)   # If applicable, filepaths_dict will also include
                                        # an entry with the key 'acquisition_time' and 
                                        # another one with the key 'average_delta_t_wf'. 
                                        # We are adding them to the resulting dictionary here.
        if verbose:
            print(f"In function DataPreprocessor.clean_file(): Succesfully processed {filepath}")
            
        return result
    
    @staticmethod
    def process_core_data(  filepath,
                            file_type_code,
                            destination_folderpath=None,
                            backup_folderpath=None,
                            overwrite_files=True,
                            skiprows=0,
                            data_delimiter=',',
                            ndecimals=18,
                            tek_wfm_metadata=None):
        
        """This static method gets the following mandatory positional argument:
        
        - filepath (string): Path to the file whose data will be processed.

        - file_type_code (scalar integer): It must be either 0, 1, 2 or 3. 
        This integer indicates the type of file which should be processed.
        0 matches an ASCII waveform dataset and 1 matches an ASCII timestamp.
        2 matches a binary (Tektronix WFM file format) file whose timestamp
        should not be extracted, while 3 matches a binary file whose 
        timestamp should be extracted.
        
        This function also gets the following optional keyword arguments:
        
        - destination_folderpath (None or string): If it is defined, then 
        the processed file(s) are removed from the input file folder and 
        moved to the folder whose path matches destination_folderpath.

        - backup_folderpath (None or string): If it is defined, then the 
        backup of the input file is moved to the folder whose path matches 
        backup_folderpath.

        - overwrite_files (boolean): If it set to True (resp. False), the 
        resulting files, i.e. the raw and processed files, will (resp. won't) 
        be overwritten if they already exist.

        - skiprows (integer scalar): This parameter is only used for the 
        case of ASCII input files, i.e. either file_type_code is 0 or 1. 
        It indicates the number of rows to skip in the input file in order 
        to access the core data. 

        - data_delimiter (string): This parameter is only used for the case 
        of ASCII input files, i.e. either file_type_code is 0 or 1. In such 
        case, it is given to numpy.loadtxt() as delimiter, which in turn, 
        uses it to separate entries of the different columns of the input file.

        - ndecimals (int): Number of decimals to use to save the real values 
        to the processed file(s). Scientific notation is assumed.

        - tek_wfm_metadata (None or dictionary): This parameter is only used 
        for the case of binary input files, i.e. either file_type_code is 2 
        or 3. In such case, it must be defined. It is a dictionary containing 
        the metadata of the provided input file. It must be the union of the 
        two dictionaries returned by DataPreprocessor.extract_tek_wfm_metadata(). 
        For more information on the keys which these dictionaries should contain, 
        check such method documentation.

        This static method returns a dictionary. To do so, it does the following:
            - Backs up the input file to a filepath whose file name is crafted
            by pre-appending the 'raw_' string to the input file name.
            - Then, if file_type_code is 0, 1 or 2,
                - it creates one ASCII file in a unified format which comprises 
                just one column of float values, which is the result of 
                removing the headers and the first column from the input file, 
                and whose filename is crafted out of the original filename by 
                pre-appending the 'processed_' string. Such file contains the 
                formatted waveform dataset information (file_type_code==0, 2) 
                or the formatted timestamp information (file_type_code==1) and 
                is given a file name which is crafted by pre-appending the 
                'processed_' string to the input file name.
            - Else, if file_type_code is 3, then
                - it creates two ASCII files following the unified format 
                specified before. The first (resp. second) one contains the 
                waveform dataset (resp. the timestamp). The filename of the 
                first (resp. second) is crafted out of the original filename 
                by pre-appending the 'processed_' (resp. 'processed_ts_') 
                string. For both generated ASCII files, the data cleaning goes 
                as in the previous case.
            - Then, saves the processed file(s) to a tunable location (up 
            to the destination_folderpath parameter, overwritting may occur).
            - Then, if file_type_code is 0, 1 or 2, this method
                - adds the raw-and-processed filepaths to the returned 
                dictionary under the keys 'raw_filepath' and 'processed_filepath', 
                respectively.
            - Else, if file_type_code is 3, then
                - adds the raw filepath, the processed-waveforms filepath 
                and the processed-timestamp filepath to the returned dictionary 
                under the keys 'raw_filepath', 'processed_filepath' and 
                'processed_ts_filepath'.    
            - To end with, if file_type_code is 1 or 3, then two additional 
            quantities are computed and saved to the returned dictionary. The 
            first one is saved under the key 'average_delta_t_wf' and is the 
            average of the time differences between consecutive triggers in the 
            waveforms dataset. The second one is saved under the key 
            'acquisition_time' and is the acquisition time of the waveform 
            dataset which is stored in the provided input file. By acquisition 
            time we refer to the time difference between the last trigger and 
            the first trigger of the waveforms dataset stored in the provided 
            filepath. It is assumed that the i-th entry of the time stamp is
            time increment of the i-th waveform trigger with respect to the 
            (i-1)-th waveform trigger. Therefore, the acquisition time is 
            computed as the sum of all of the entries of the time stamp."""

        htype.check_type(   filepath, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_core_data", 
                                                                                46221))
        if not os.path.isfile(filepath):
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.process_core_data", 
                                                                        49000,
                                                                        extra_info=f"Path {filepath} does not exist or is not a file."))
        else:
            _, extension = os.path.splitext(filepath)

        htype.check_type(   file_type_code, int, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_core_data", 
                                                                                79829))
        if file_type_code<0 or file_type_code>3:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.process_core_data", 
                                                                                    66937))
        elif file_type_code<2 and extension not in ('.csv', '.txt', '.dat'):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.process_core_data",
                                                                                    12823,
                                                                                    extra_info=f"Not allowed extension for an ASCII input file."))
        elif file_type_code>1 and extension!='.wfm':
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.process_core_data",
                                                                                    47001,
                                                                                    extra_info=f"Binary input files must be WFM files."))
        if destination_folderpath is not None:
            htype.check_type(   destination_folderpath, str, 
                                exception_message=htype.generate_exception_message( "DataPreprocessor.process_core_data", 
                                                                                    19713))
            if not os.path.isdir(destination_folderpath):
                raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.process_core_data", 
                                                                            24040,
                                                                            extra_info=f"Path {destination_folderpath} does not exist or is not a directory."))
        if backup_folderpath is not None:
            htype.check_type(   backup_folderpath, str, 
                                exception_message=htype.generate_exception_message( "DataPreprocessor.process_core_data", 
                                                                                    79090))
            if not os.path.isdir(backup_folderpath):
                raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.process_core_data", 
                                                                            34147,
                                                                            extra_info=f"Path {backup_folderpath} does not exist or is not a directory."))
        htype.check_type(   overwrite_files, bool, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_core_data", 
                                                                                60125))
        htype.check_type(   skiprows, int, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_core_data", 
                                                                                96245))
        htype.check_type(   data_delimiter, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_core_data", 
                                                                                68852))
        htype.check_type(   ndecimals, int, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.process_core_data", 
                                                                                72220))
        if file_type_code>1:
            htype.check_type(   tek_wfm_metadata, dict, 
                                exception_message=htype.generate_exception_message( "DataPreprocessor.process_core_data", 
                                                                                    72345))
        result = {}
        raw_filepath, processed_filepath = DataPreprocessor.get_raw_and_processed_filepaths(filepath, 
                                                                                            raw_prelabel='raw_', 
                                                                                            processed_prelabel='processed_',
                                                                                            raw_folderpath=backup_folderpath,
                                                                                            processed_folderpath=destination_folderpath)
        result['raw_filepath'], result['processed_filepath'] = raw_filepath, processed_filepath
        if file_type_code==3:
            _, processed_ts_filepath = DataPreprocessor.get_raw_and_processed_filepaths(filepath, 
                                                                                        raw_prelabel='raw_', 
                                                                                        processed_prelabel='processed_ts_',
                                                                                        raw_folderpath=backup_folderpath,
                                                                                        processed_folderpath=destination_folderpath)
            result['processed_ts_filepath'] = processed_ts_filepath

        if file_type_code<2:    # ASCII input
            data = np.loadtxt(filepath, delimiter=data_delimiter, skiprows=skiprows)
            if np.ndim(data)<2 or np.shape(data)[1]<2:
                raise cuex.NoAvailableData(htype.generate_exception_message("DataPreprocessor.process_core_data", 
                                                                            41984,
                                                                            extra_info="Input ASCII data must have at least two columns."))
            data = data[:,1]    # Either voltage entries or a timestamp,
                                # we remove the rest of the data and
                                # preserve the second column
            np.savetxt(processed_filepath, data, fmt=f"%.{ndecimals}e", delimiter=',')

            if file_type_code==1:
                result['acquisition_time']=np.sum(data)         # Assuming that the acquisition time of the
                                                                # last waveform is negligible with respect 
                                                                # to the time difference between triggers
                result['average_delta_t_wf']=result['acquisition_time']/(data.shape[0]-1)
                
        else:    # Binary input
            timestamp, waveforms = DataPreprocessor.extract_tek_wfm_coredata(   filepath,
                                                                                tek_wfm_metadata)
            
            waveforms = waveforms.flatten(order='F')  # Concatenate waveforms in a 1D-array
            np.savetxt(processed_filepath, waveforms, fmt=f"%.{ndecimals}e", delimiter=',')

            if file_type_code==3:
                result['acquisition_time']=np.sum(timestamp)    # Assuming that the acquisition time of the     # The time stamp, as returned by
                                                                # last waveform is negligible with respect      # DataPreprocessor.extract_tek_wfm_coredata(),
                                                                # to the time difference between triggers       # contains as many entries as waveforms in 
                result['average_delta_t_wf']=result['acquisition_time']/(timestamp.shape[0]-1)                  # in the FastFrame set. The first one is null.
                np.savetxt(processed_ts_filepath, timestamp, fmt=f"%.{ndecimals}e", delimiter=',')

        shutil.move(filepath, raw_filepath) # Backup
        return result

    @staticmethod
    def extract_tek_wfm_coredata(filepath, metadata):

        """This static method gets the following mandatory positional arguments:
        
        - filepath (string): Path to the binary file (Tektronix WFM file format),
        which must host a FastFrame set and whose core data should be extracted.
        DataPreprocessor.extract_tek_wfm_metadata() should have previously checked
        that, indeed, the input file hosts a FastFrame set. It is a check based
        on the 4-bytes integer which you can find at offset 78 of the WFM file.
        - metadata (dictionary): It is a dictionary which contains meta-data of
        the input file which is necessary to extract the core data. It should
        contain the union of the two dictionaries returned by 
        DataPreprocessor.extract_tek_wfm_metadata(). For more information on 
        the data contained in such dictionaries, check such method documentation.
        
        This method returns two arrays. The first one is an unidimensional array
        of length M, which stores timestamp information. The second one is a 
        bidimensional array which stores the waveforms of the FastFrame set of 
        the given input file. Say such array has shape NxM: then N is the number 
        of (user-accesible) points per waveform, while M is the number of 
        waveforms. The waveform entries in such array are already expressed in 
        the vertical units which are extracted to the key 'Vertical Units' by 
        DataPreprocessor.extract_tek_wfm_metadata(). In this context, the i-th
        entry of the first array returned by this function gives the time
        difference, in seconds, between the trigger of the i-th waveform and
        the trigger of the (i-1)-th waveform. The first entry, which is undefined
        up to the given definition, is manually set to zero."""

        htype.check_type(   filepath, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.extract_tek_wfm_coredata", 
                                                                                82855))
        if not os.path.isfile(filepath):
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.extract_tek_wfm_coredata", 
                                                                        58749,
                                                                        extra_info=f"Path {filepath} does not exist or is not a file."))
        else:
            _, extension = os.path.splitext(filepath)
            if extension!='.wfm':
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.extract_tek_wfm_coredata",
                                                                                        21667,
                                                                                        extra_info=f"The extension of the input file must match '.wfm'."))
        htype.check_type(   metadata, dict, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.extract_tek_wfm_coredata", 
                                                                                35772))

        first_sample_delay = np.empty((metadata['FastFrame Count'],), dtype=np.double)  # Fraction of the sample time 
                                                                                        # from the trigger time stamp 
                                                                                        # to the next sample.
        triggers_second_fractions = np.empty((metadata['FastFrame Count'],), dtype=np.double)   # The fraction of the second
                                                                                                # when the trigger occurred.
        gmt_in_seconds = np.empty((metadata['FastFrame Count'],), dtype=np.double)  # GMT (in seconds from the epoch) 
                                                                                    # when the trigger occurred.
        first_sample_delay[0] = metadata['tfrac[0]']                # Add info of the first frame
        triggers_second_fractions[0] = metadata['tdatefrac[0]']
        gmt_in_seconds[0] = metadata['tdate[0]']

        # For FastFrame, we've got a chunk of metadata['FastFrame Count']*54 bytes which store WfmUpdateSpec and 
        # WfmCurveSpec objects, containing data on the timestamp and the number of points of each frame.

        with open(filepath, 'rb') as file:  # Binary read mode
            _ = file.read(838)              # Throw away the header bytes (838 bytes)

            # WUS stands for Waveform Update Specification. WUS objects count on a 4 bytes 
            # unsigned long, a 8 bytes double, another 8 bytes double and a 4 bytes long.

            dtype = [   ('_',                       'i4'),  # Structure of the output array of np.fromfile
                        ('first_sample_delay',      'f8'),  # The first element of each tuple is the name
                        ('trigger_second_fraction', 'f8'),  # of the field, whereas the second element is the
                        ('gmt_in_seconds',          'i4')]  # data type of each field                             

            data = np.fromfile( file, 
                                dtype=dtype, 
                                count=(metadata['FastFrame Count']-1))  # Within the same 'with' context, 
                                                                        # np.fromfile continues the reading
                                                                        # process as of the already-read
                                                                        # 838 bytes. Also, we are taking into
                                                                        # account that the time information
                                                                        # of the first frame was already read.

            first_sample_delay[1:]          = data['first_sample_delay']        # Merge first frame trigger 
            triggers_second_fractions[1:]   = data['trigger_second_fraction']   # info. with info. from the
            gmt_in_seconds[1:]              = data['gmt_in_seconds']            # the rest of the frames.

            # Read waveforms
            waveforms = np.memmap(  file,
                                    dtype = metadata['samples_datatype'],
                                    mode = 'r',
                                    offset = metadata['curve_buffer_offset'],
                                    shape = (metadata['samples_no'], metadata['FastFrame Count']),  # Shape of the returned array
                                    order = 'F')                                                    # Running along second dimension 
                                                                                                    # gives different waveforms
        seconds_from_first_trigger = gmt_in_seconds - gmt_in_seconds[0]     # While the numbers in gmt_in_seconds are O(9)
        timestamp = seconds_from_first_trigger+triggers_second_fractions    # The fractions of seconds are O(-1). Summing
                                                                            # the fractions of the second to the GMT could
                                                                            # result in losing the second fraction info. due
                                                                            # to rounding error. It's better to shift the
                                                                            # time origin to the first trigger, then add the
                                                                            # seconds fractions.
        timestamp = np.concatenate((np.array([0.0]), np.diff(timestamp)), axis=0)
        waveforms = waveforms[metadata['pre-values_no']:metadata['samples_no']-metadata['post-values_no'],:]    # Filter out the oscilloscope
                                                                                                                # interpolation samples
        waveforms = (waveforms*metadata['vscale'])+metadata['voffset']  # 2D array of waveforms, in vertical units
        return timestamp, waveforms

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

        htype.check_type(   filepath, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.rename_file", 
                                                                                35673))
        if not os.path.isfile(filepath):
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.rename_file",
                                                                        46110,
                                                                        extra_info=f"Path {filepath} does not exist or is not a file."))
        htype.check_type(   new_filename, str, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.rename_file", 
                                                                                88007))
        htype.check_type(   overwrite, bool, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.rename_file", 
                                                                                12432))        
        htype.check_type(   verbose, bool, 
                            exception_message=htype.generate_exception_message( "DataPreprocessor.rename_file", 
                                                                                71900))

        folderpath = os.path.split(filepath)[0]
        new_filepath = os.path.join(folderpath, new_filename)

        if os.path.isfile(new_filepath):    
            if overwrite:
                os.rename(filepath, new_filepath)
                if verbose:
                    print(htype.generate_exception_message( "DataPreprocessor.rename_file", 
                                                            47189, 
                                                            extra_info=f"File {new_filepath} already exists. It has been overwritten."))
            else:
                raise FileExistsError(htype.generate_exception_message( "DataPreprocessor.rename_file", 
                                                                        95995, 
                                                                        extra_info=f"File {new_filepath} already exists. It cannot be overwritten. Renaming was not performed."))
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

        htype.check_type(   input_dict, dict,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.query_fields_in_dictionary", 
                                                                                52810))
        for key in input_dict.keys():
            htype.check_type(   key, str,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.query_fields_in_dictionary", 
                                                                                    46180))
            if not callable(input_dict[key]):
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.query_fields_in_dictionary", 
                                                                                        55832))  
        default_dict_ = {key:None for (key, value) in input_dict.items()}
        if default_dict is not None:
            htype.check_type(   default_dict, dict,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.query_fields_in_dictionary", 
                                                                                    47224))
            default_dict_.update(default_dict)

        result = {}
        for key in input_dict.keys():
            query_ok = False
            while not query_ok:
                try:
                    result[key] = DataPreprocessor.query_field( key, 
                                                                casting_function=input_dict[key], 
                                                                substitution_trigger='',
                                                                substitute=default_dict_[key])
                    query_ok = True
                except cuex.NoAvailableData:
                    print(htype.generate_exception_message( "DataPreprocessor.query_fields_in_dictionary", 
                                                            47188,
                                                            extra_info=f"There's no default value for {key}. Please, provide one."))
        return result

    @staticmethod
    def query_field(field_name, casting_function=str, substitution_trigger='', substitute=None):

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

        htype.check_type(   field_name, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.query_field", 
                                                                                89813))        
        if not callable(casting_function):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.query_field", 
                                                                                    91631))
        htype.check_type(   substitution_trigger, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.query_field", 
                                                                                36199))        
        if substitute is not None:
            try:
                if casting_function(substitute)!=substitute:
                    raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.query_field", 
                                                                                            91631))      
            except ValueError:
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.query_field", 
                                                                                        77549))  
                  
        keep_asking = True
        while keep_asking:
            input_is_ok = False
            while not input_is_ok:
                user_input = input(f"Enter a value for '{field_name}':" )
                if user_input==substitution_trigger:
                    if substitute is None:
                        raise cuex.NoAvailableData(htype.generate_exception_message("DataPreprocessor.query_field", 
                                                                                    23120,
                                                                                    extra_info="Default value was not specified."))
                    else:
                        user_input = substitute
                        result = substitute
                        input_is_ok = True
                else:
                    try:
                        result = casting_function(user_input)
                        input_is_ok = True
                    except ValueError:
                        print(htype.generate_exception_message( "DataPreprocessor.query_field", 
                                                                19532,
                                                                extra_info=f"The specified value do not comply with the expected type ({casting_function})"))
            keep_asking = not DataPreprocessor.yes_no_translator(input(f"The entered value is {user_input}. ¿Is it OK? (y/n)"))
        return result

    @staticmethod
    def yes_no_translator(user_input):

        """This static method gets the following mandatory positional argument:
        
        - user_input (string)
        
        This method returns True if user_input is equal to any value within 
        ('y', 'Y', 'ye', 'Ye', 'yE', 'YE', 'yes', 'Yes', 'yEs', 'yeS', 'YEs', 'YeS', 'yES', 'YES'). 
        It returns False otherwise."""

        htype.check_type(   user_input, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.yes_no_translator", 
                                                                                89813))  
        if user_input in ('y', 'Y', 'ye', 'Ye', 'yE', 'YE', 'yes', 'Yes', 'yEs', 'yeS', 'YEs', 'YeS', 'yES', 'YES'):
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
        
        htype.check_type(   input_dictionary, dict,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_json_file", 
                                                                                31965))
        htype.check_type(   output_filepath, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.generate_json_file", 
                                                                                31900))
        with open(output_filepath, "w") as file:
            json.dump(input_dictionary, file, indent='\t')
        return
    
    @staticmethod
    def get_str_creation_date(filepath):

        """This static method gets the following mandatory positional argument:
        
        - filepath (string)
        
        This static method returns the creation date of the file whose filepath
        matches the given filepath, as a string in the format YYYY-MM-DD hh:mm:ss."""

        htype.check_type(   filepath, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.get_str_creation_date", 
                                                                                46223))

        if os.path.exists(filepath):
            return datetime.datetime.fromtimestamp(os.path.getctime(filepath)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.get_str_creation_date",
                                                                        62018))

    @staticmethod
    def ask_for_unique_query_splitting(func):
        def wrapper(*args):
            print("Are there any fields within the following dictionary whose value for all of the measurements is the same?")
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

        htype.check_type(   n_max, int,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.query_tuple_of_semipositive_integers", 
                                                                                51812))
        if n_max<1:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.query_tuple_of_semipositive_integers", 
                                                                                    11012,
                                                                                    extra_info="One integer must be queried at least."))
        
        keep_asking=True
        while keep_asking:
            user_input = input("Provide a comma-separated tuple of semipositive integer values. Any value which cannot be casted to integer will be ignored. The absolute value of every integer-casted input will be computed.")
            result = []
            for sample in user_input.split(','):
                try:
                    result.append(abs(int(sample)))
                except Exception:
                    pass
            result = tuple(result[:n_max])    # Comply with the provided maximum number of samples
            keep_asking = not DataPreprocessor.yes_no_translator(input(f"The processed input is {result}. Is it OK? (y/n)"))
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

        htype.check_type(   input_dict, dict,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.query_dictionary_splitting", 
                                                                                37119))
        i_gen = DataPreprocessor.integer_generator()
        ordered_dict = {next(i_gen):key for (key, value) in input_dict.items()}

        print("The input dictionary is:")
        for key in ordered_dict.keys():
            print(f"{key}: ({ordered_dict[key]}, {input_dict[ordered_dict[key]]})")

        print("Input the integer-labels of the entries which you would like to split apart.")
        splitted_keys = set(DataPreprocessor.query_tuple_of_semipositive_integers(len(input_dict)))
        
        aux = {}
        for key in ordered_dict.keys():
            if key in splitted_keys:
                aux[key] = ordered_dict[key]

        first_split = {value:input_dict[value] for (key, value) in aux.items()}
        second_split = input_dict
        for key in first_split.keys():
            del second_split[key]

        return first_split, second_split
    
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
        
        htype.check_type(   path, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.path_is_contained_in_dir", 
                                                                                25500))
        htype.check_type(   dir_path, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.path_is_contained_in_dir", 
                                                                                39223))
        if not os.path.exists(path):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.path_is_contained_in_dir", 
                                                                                    23589))
        if not os.path.isdir(dir_path):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.path_is_contained_in_dir", 
                                                                                    49032))

        return os.path.samefile(os.path.commonpath([dir_path, path]), dir_path) # If instead of os.path.samefile() we use simple
                                                                                # string comparation, paths which point to the same
                                                                                # directory or file may appear as different paths.
                                                                                # For example, while os.path.samefile('/a/b/c', '/a/b/c/')
                                                                                # evaluates to True, '/a/b/c'=='/a/b/c/' does not.
                                                                                # The only difference is the last slash.
    
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
        remains in the catalog and nothing is added to the output dictionary. This
        function returns the output dictionary and the remaining catalog, in such order."""

        htype.check_type(   catalog, dict,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.try_grabbing_from_json", 
                                                                                48501))
        for key in catalog.keys():

            htype.check_type(   key, str,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.try_grabbing_from_json", 
                                                                                    29757))
            htype.check_type(   catalog[key], type,
                                exception_message=htype.generate_exception_message( "DataPreprocessor.try_grabbing_from_json", 
                                                                                    44642))
        htype.check_type(   path_to_json_file, str,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.try_grabbing_from_json", 
                                                                                62236))
        if not os.path.isfile(path_to_json_file):
            raise FileNotFoundError(htype.generate_exception_message(   "DataPreprocessor.try_grabbing_from_json", 
                                                                        37189,
                                                                        extra_info=f"The path {path_to_json_file} does not exist or is not a file."))
        elif not path_to_json_file.endswith(".json"):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.try_grabbing_from_json", 
                                                                                    37756,
                                                                                    extra_info=f"File {path_to_json_file} must end with '.json' extension.")) 
        htype.check_type(   verbose, bool,
                            exception_message=htype.generate_exception_message( "DataPreprocessor.try_grabbing_from_json", 
                                                                                88675))

        output_dictionary = {}
        remaining_catalog = catalog.copy()  # Otherwise we'll get a view to catalog
        
        with open(path_to_json_file, "r") as file:
            try:
                input_dictionary = json.load(file)
            except json.JSONDecodeError:
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DataPreprocessor.try_grabbing_from_json", 
                                                                                        97781,
                                                                                        extra_info=f"File {path_to_json_file} must contain a valid json object."))
        for key in catalog.keys():
            if key in input_dictionary.keys():  # This key of catalog was 
                                                # found in the input dictionary
                try:
                    htype.check_type(   input_dictionary[key], catalog[key],
                                        exception_message=htype.generate_exception_message( "DataPreprocessor.try_grabbing_from_json", 
                                                                                            35881))
                except cuex.TypeException:  # Although the key was found, its 
                                            # value does not have a suitable type
                                            # cuex.TypeException is the exception raised
                                            # by htype.check_type if types do not match.
                    if verbose: 
                        print(htype.generate_exception_message( "DataPreprocessor.try_grabbing_from_json", 
                                                                58548,
                                                                extra_info=f"A candidate for key '{key}' was found but its type does not match the required one. The candidate has been ignored."))
                    continue

                else:           # The found candidate has a suitable type
                                # Actually add it to the output dictionary
                    
                    output_dictionary[key] = input_dictionary[key]
                    del remaining_catalog[key]

        return output_dictionary, remaining_catalog