import os
import json
import math
from datetime import datetime
from abc import ABC
import numpy as np
from scipy import signal as spsi
from scipy import constants as spcon
from scipy import optimize as spopt

import src.utils.htype as htype
import src.utils.custom_exceptions as cuex
from src.custom_types.RigidKeyDictionary import RigidKeyDictionary
from src.core.WaveformSet import WaveformSet


class SiPMMeas(ABC):

    def __init__(
        self,
        *args,
        delivery_no=None,
        set_no=None,
        meas_no=None,
        strip_ID=None,
        meas_ID=None,
        date=None,
        location=None,
        operator=None,
        setup_ID=None,
        system_characteristics=None,
        thermal_cycle=None,
        electronic_board_number=None,
        electronic_board_location=None,
        electronic_board_socket=None,
        sipm_location=None,
        sampling_ns=None,
        cover_type=None,
        operation_voltage_V=None,
        overvoltage_V=None,
        PDE=None,
        status=None,
        **kwargs,
    ):
        """This class aims to model a base class from which to derive certain SiPM measurements.
        For example, a class which implements a gain measurement of a SiPM, say SiPMMeas, could
        inherit from SiPMMeas. The same goes for a measurement of the dark noise of a SiPM. In line
        with Waveform and WaveformSet classes, the assumed time unit is the second, unless otherwise
        indicated. This initializer gets the following positional arguments:

        - args: These positional arguments are given to WaveformSet.from_files. They must be
        two positional arguments: input_filepath (string) and time_resolution (positive float),
        in such order. For more information on these arguments, please refer to
        WaveformSet.from_files docstring. Particularly, time_resolution is assumed to be expressed
        in seconds. If applicable (i.e. if the given sampling_ns is None), its value is converted
        to nanosecons and assigned to the self.__sampling_ns attribute.

        This initializer gets the following keyword arguments:

        - delivery_no (semipositive integer): Integer which identifies the delivery where the
        measured SiPM was included. For DUNE's particular case, this number identifies the
        manufacturer's delivery to the DUNE collaboration.
        - set_no (semipositive integer): Integer which identifies the set where the measured
        SiPM was included. For DUNE's particular case, this number identifies the internal
        delivery which we receive from another DUNE institution.
        - meas_no (semipositive integer): Integer which identifies the measurement within the
        set where the measured SiPM was included.
        - strip_ID (int): Integer which identifies the SiPM strip which hosts the measured SiPM.
        - meas_ID (string): String which identifies this measurement.
        - date (string): Date of the measurement. This string must follow the following format:
        'YYYY-MM-DD hh:mm:ss'.
        - location (string): ¿Where was the measurement carried out?
        - operator (string): ¿Who operated the setup during this measurement?
        - setup_ID (string): String which identifies the used setup.
        - system_characteristics (string): Any extra information on the setup for this measurement.
        - thermal_cycle (semipositive integer): ¿How many thermal cycles have this SiPM undergone
        by the end of this measurement?
        - electronic_board_number (semipositive integer): Number which identifies the electronic
        board where the flex board was mounted on.
        - electronic_board_location (string): String which identifies the location of the used
        electronic board within the cold box.
        - electronic_board_socket (integer in (1,2,3): Number which identifies the socket of the
        electronic board where the SiPM flex board was connected to. Each electronic board
        counts on 3 sockets.
        - sipm_location (integer in (1,2,...,6)): Number which identifies the SiPM location within
        its flex board. Each flex board hosts 6 SiPMs, so SiPMs are numbered from 1 to 6 with
        respect to some fixed orientation.
        - sampling_ns (positive float): Time resolution, in nanoseconds, for the acquired waveforms.
        I.e. the time delta between one time point and the following one. If it is not provided
        (i.e. if it is None), it is taken from the second positional argument, which is assumed to
        be the time resolution in seconds.
        - cover_type (string): String which identifies the type of cover used to optically-isolate
        the SiPM from the bulk of the cryogenic bath and from the rest of the SiPMs.
        - operation_voltage_V (semipositive float): Feeding voltage given to the measured SiPM.
        - overvoltage_V (semipositive float): Feeding voltage given to the measured SiPM, measured
        with respect to the breakdown voltage.
        - PDE (semipositive float): Photon detection efficiency of the measured SiPM.
        - status (string): String which identifies the status of the measured SiPM.
        - kwargs: These keyword arguments are given to WaveformSet.from_files. The expected keywords
        are points_per_wvf (int), wvfs_to_read (int), timestamp_filepath (string),
        delta_t_wf (float), set_name (string), creation_dt_offset_min (float) and
        wvf_extra_info (string). To understand these arguments, please refer to the
        WaveformSet.from_files docstring.

        All of the keyword arguments, except for **kwargs, are loaded into object-attributes whose
        name matches the keyword of the kwarg except for two previous underscores, which are added
        to the attribute name. None of these arguments are positional arguments, so none of them are
        mandatory. If a certain keyword argument is not set in the instance initialization, then its
        associated attribute is set to None within this initializer, meaning that its information is
        not available. Thus, when requesting any attribute of this class via a getter, one should be
        prepared to handle a None value and interpret it as the unavailability of such data.
        """

        if len(args) != 2:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SiPMMeas.__init__", 47901)
            )
        else:
            htype.check_type(
                args[0],
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 48729
                ),
            )

            htype.check_type(
                args[1],
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 12090
                ),
            )
            if args[1] <= 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 57900)
                )

        self.__delivery_no = None
        if delivery_no is not None:
            htype.check_type(
                delivery_no,
                int,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 57294
                ),
            )
            if delivery_no < 0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 68001)
                )
            self.__delivery_no = delivery_no

        self.__set_no = None
        if set_no is not None:
            htype.check_type(
                set_no,
                int,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 45625
                ),
            )
            if set_no < 0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 11234)
                )
            self.__set_no = set_no

        self.__meas_no = None
        if meas_no is not None:
            htype.check_type(
                meas_no,
                int,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 91887
                ),
            )
            if meas_no < 0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 44252)
                )
            self.__meas_no = meas_no

        self.__strip_ID = None
        if strip_ID is not None:
            htype.check_type(
                strip_ID,
                int,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 97522
                ),
            )
            self.__strip_ID = strip_ID

        self.__meas_ID = None
        if meas_ID is not None:
            htype.check_type(
                meas_ID,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 60773
                ),
            )
            self.__meas_ID = meas_ID

        self.__date = None
        if date is not None:
            htype.check_type(
                date,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 36955
                ),
            )
            self.__date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

        self.__location = None
        if location is not None:
            htype.check_type(
                location,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 80056
                ),
            )
            self.__location = location

        self.__operator = None
        if operator is not None:
            htype.check_type(
                operator,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 85904
                ),
            )
            self.__operator = operator

        self.__setup_ID = None
        if setup_ID is not None:
            htype.check_type(
                setup_ID,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 47564
                ),
            )
            self.__setup_ID = setup_ID

        self.__system_characteristics = None
        if system_characteristics is not None:
            htype.check_type(
                system_characteristics,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 35647
                ),
            )
            self.__system_characteristics = system_characteristics

        self.__thermal_cycle = None
        if thermal_cycle is not None:
            htype.check_type(
                thermal_cycle,
                int,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 43662
                ),
            )
            if thermal_cycle < 0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 89778)
                )
            self.__thermal_cycle = thermal_cycle

        self.__electronic_board_number = None
        if electronic_board_number is not None:
            htype.check_type(
                electronic_board_number,
                int,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 77685
                ),
            )
            if electronic_board_number < 0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 86100)
                )
            self.__electronic_board_number = electronic_board_number

        self.__electronic_board_location = None
        if electronic_board_location is not None:
            htype.check_type(
                electronic_board_location,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 64793
                ),
            )
            self.__electronic_board_location = electronic_board_location

        self.__electronic_board_socket = None
        if electronic_board_location is not None:
            if electronic_board_socket not in (1, 2, 3):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 12009)
                )
            self.__electronic_board_socket = electronic_board_socket

        self.__sipm_location = None
        if sipm_location is not None:
            if sipm_location not in (1, 2, 3, 4, 5, 6):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 62580)
                )
            self.__sipm_location = sipm_location

        self.__sampling_ns = None
        if sampling_ns is not None:
            htype.check_type(
                sampling_ns,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 58458
                ),
            )
            if sampling_ns <= 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 11055)
                )
            self.__sampling_ns = sampling_ns
        else:
            self.__sampling_ns = 1e9 * args[1]

        self.__cover_type = None
        if cover_type is not None:
            htype.check_type(
                cover_type,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 66836
                ),
            )
            self.__cover_type = cover_type

        self.__operation_voltage_V = None
        if operation_voltage_V is not None:
            htype.check_type(
                operation_voltage_V,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 62399
                ),
            )
            if operation_voltage_V < 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 41690)
                )
            self.__operation_voltage_V = operation_voltage_V

        self.__overvoltage_V = None
        if overvoltage_V is not None:
            htype.check_type(
                overvoltage_V,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 32027
                ),
            )
            if overvoltage_V < 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 49182)
                )
            self.__overvoltage_V = overvoltage_V

        self.__PDE = None
        if PDE is not None:
            htype.check_type(
                PDE,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 86273
                ),
            )
            if PDE < 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.__init__", 93806)
                )
            self.__PDE = PDE

        self.__status = None
        if status is not None:
            htype.check_type(
                status,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.__init__", 36997
                ),
            )
            self.__status = status
        
        # points_per_wvf must be available
        points_per_wvf = SiPMMeas.get_value_from_dict(
            kwargs, "points_per_wvf", none_fallback=False)
        
        timestamp_filepath = SiPMMeas.get_value_from_dict(
            kwargs, "timestamp_filepath", none_fallback=True)
        
        delta_t_wf = SiPMMeas.get_value_from_dict(
            kwargs, "delta_t_wf", none_fallback=True)
        
        creation_dt_offset_min = SiPMMeas.get_value_from_dict(
            kwargs, "creation_dt_offset_min", none_fallback=True)
        
        wvf_extra_info = SiPMMeas.get_value_from_dict(
            kwargs, "wvf_extra_info", none_fallback=True)
            
        # N.B.: This attribute is meant to be the time duration 
        # of the measurement, in minutes. It is not assigned via 
        # an input parameter, but computed out of the WaveformSet 
        # core data (the waveforms time stamp) which is processed 
        # by the WaveformSet.from_files class method.
        self.__acquisition_time_min = None
        
        # N.B.: WaveformSet.from_files() takes a couple of keyword
        # arguments (headers_end_identifier and data_delimiter)
        # whose default values are "TIME," and ",", respectively.
        # These work for the current data format and I do not 
        # expect the data format to change. If it does, it will 
        # be convenient to root those parameters here and up to 
        # the full call chain, i.e. 
        # GainMeas/DarkNoiseMeas.from_json_file() -> ...
        # GainMeas/DarkNoiseMeas.__init__() -> ...
        # SiPMMeas.__init__() -> WaveformSet.from_files().
        self.__waveforms, self.__acquisition_time_min = WaveformSet.from_files(
            *args,
            points_per_wvf,
            timestamp_filepath=timestamp_filepath,
            delta_t_wf=delta_t_wf,
            ref_datetime=self.__date,
            creation_dt_offset_min=creation_dt_offset_min,
            wvf_extra_info=wvf_extra_info
        )

        htype.check_type(
            self.__acquisition_time_min,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.__init__", 58815
            ),
        )

        if self.__acquisition_time_min < 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SiPMMeas.__init__", 77855)
            )

        # The WaveformSet date is set to match the SiPMMeas date
        # Up to this point, self.__date is either None or a datetime

        if self.__waveforms.check_homogeneity_of_sign_through_set("signal_unit"):
            # Note that this attribute has a different data flow compared to
            # the rest of attributes. It is not explictly given to this
            # initializer, but implictly given to WaveformSet.from_files. Then,
            # we recover it from the WaveformSet object.
            self.__signal_unit = self.__waveforms[0].Signs["signal_unit"][0]

        else:
            raise cuex.InconsistentParametersDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.__init__",
                    84774,
                    extra_info="The signal unit must be the same for every waveform in this waveform set.",
                )
            )
        self.__N_events = len(self.__waveforms)

        # Remember that the absolute time for Waveform objects is
        # recovered as wvf.Time +wvf.T0, i.e. wvf.Time[0]==0.0, and so,
        # wvf.Time[-1]  gives the width of the time window in seconds.
        self.__waveform_window_mus = 1e6 * np.mean(
            [wvf.Time[-1] for wvf in self.__waveforms]
        )
        return

    # Getters
    @property
    def DeliveryNo(self):
        return self.__delivery_no

    @property
    def SetNo(self):
        return self.__set_no

    @property
    def MeasNo(self):
        return self.__meas_no

    @property
    def StripID(self):
        return self.__strip_ID

    @property
    def MeasID(self):
        return self.__meas_ID

    @property
    def Date(self):
        return self.__date

    @property
    def Location(self):
        return self.__location

    @property
    def Operator(self):
        return self.__operator

    @property
    def SetupID(self):
        return self.__setup_ID

    @property
    def SystemCharacteristics(self):
        return self.__system_characteristics

    @property
    def ThermalCycle(self):
        return self.__thermal_cycle

    @property
    def ElectronicBoardNumber(self):
        return self.__electronic_board_number

    @property
    def ElectronicBoardLocation(self):
        return self.__electronic_board_location

    @property
    def ElectronicBoardSocket(self):
        return self.__electronic_board_socket

    @property
    def SiPMLocation(self):
        return self.__sipm_location

    @property
    def Sampling_ns(self):
        return self.__sampling_ns

    @property
    def WaveformWindow_mus(self):
        """This attribute, together with self.__N_events, is the only one
        that is not assigned from a parameter given to the initializer, but
        it is computed out of the rest of information given to the initializer.
        self.__waveform_window_mus is meant to be a semipositive float which
        gives the time width, in micro seconds, of the time window used to
        acquire each waveform."""

        return self.__waveform_window_mus

    @property
    def CoverType(self):
        return self.__cover_type

    @property
    def OperationVoltage_V(self):
        return self.__operation_voltage_V

    @property
    def Overvoltage_V(self):
        return self.__overvoltage_V

    @property
    def PDE(self):
        return self.__PDE

    @property
    def NEvents(self):
        """This attribute, together with self.__waveform_window_mus, is the
        only one that is not assigned from a parameter given to the initializer,
        but it is computed out of the rest of information given to the
        initializer. self.__N_events is meant to be a semipositive integer
        which gives the number of waveforms contained in self.__waveforms."""

        return self.__N_events

    @property
    def SignalUnit(self):
        return self.__signal_unit

    @property
    def Status(self):
        return self.__status
    
    @property
    def AcquisitionTime_min(self):
        return self.__acquisition_time_min

    @property
    def Waveforms(self):
        return self.__waveforms

    @staticmethod
    def fit_piecewise_gaussians_to_the_n_highest_peaks(
        samples,
        peaks_to_detect=2,
        peaks_to_fit=None,
        bins_no=125,
        starting_fraction=0.0,
        step_fraction=0.01,
        minimal_prominence_wrt_max=0.0,
        minimal_width_in_samples=0,
        std_no=3.0,
        fit_to_density=True,
    ):
        """This static method gets the following optional keyword arguments:

        - samples (unidimensional float numpy array): The samples within this
        array are used to build an histogram. Such histogram is, then,
        piecewise fit to gaussian functions.
        - peaks_to_detect (scalar integer): It must be positive (>0). Number of
        peaks which will be detected to start with. A subset of the detected
        peaks will be fit, up to the peaks_to_fit argument.
        - peaks_to_fit (None or tuple): If None, then it is assumed that all of
        the detected peaks should be fit. If it is a tuple, then it must
        contain integers. Its length must comply with
        0<=len(peaks_to_fit)<=peaks_to_detect. Every entry must belong to
        the interval [0, peaks_to_detect-1]. Let us sort the peaks_to_detect
        detected peaks according to the iterator value for samples where they
        occur. Then if i belongs to peaks_to_fit, the i-th detected peak will
        be fit.
        - bins_no (scalar integer): It must be positive (>0). It is the number
        of bins which are used to histogram the given samples.
        - starting_fraction (scalar float): It must be semipositive (>=0.0)
        and smaller or equal to 1 (<=1.0). It is given to the static method
        SiPMMeas.tune_peak_height(). Check its docstring for more information.
        - step_fraction (scalar float): It must be positive (>0.0) and smaller
        or equal to 1 (<=1.0). It is given to the static method
        SiPMMeas.tune_peak_height(). Check its docstring for more information.
        - minimal_prominence_wrt_max (scalar float): It must be semipositive
        (>=0) and smaller or equal than 1.0 (<=1.0). It is understood as a
        fraction of the maximum value of the histogram of samples. It is given
        to the 'minimal_prominence_wrt_max' keyword argument of
        SiPMMeas.tune_peak_height(). It sets a minimal prominence for a peak
        to be detected, based on a fraction of the maximum value within the
        samples histogram. I.e. the only detected peaks are those whose
        prominence is bigger or equal to a fraction of the samples histogram
        maximum. For more information check the SiPMMeas.tune_peak_height()
        docstring.
        - minimal_width_in_samples (scalar integer): It must be a semipositive
        (>=0) integer. It is understood as the required width of a peak (in
        samples), for it to be detected as a peak. It is given to the
        'minimal_width_in_samples' keyword argument of SiPMMeas.tune_peak_height().
        For more information check the SiPMMeas.tune_peak_height() docstring.
        - std_no (scalar float): It must be positive (>0.0). This parameter is
        given to the std_no keyword argument of
        SiPMMeas.piecewise_gaussian_fits(). Check its docstring for more
        information.
        - fit_to_density (scalar boolean): It is given to the density parameter
        of the numpy.histogram method when histogramming the input samples.

        This method fits one gaussian to each peak within a subset of the
        peaks_to_detect highest peaks of the histogram of samples. Such
        subset is defined via peaks_to_fit. This method returns the optimal
        values for the fitting parameters, as well as the covariance matrix.
        To do so, this method

        1) generates an histogram using samples entries,
        2) if fit_to_density, then it generates a probability distribution
        function (pdf) out of such histogram. If not fit_to_density, then
        the function considers the hits histogram of the samples entries,
        3) then detects the peaks_to_detect highest peaks of such histogram,
        via SiPMMeas.tune_peak_height() and scipy.signal.find_peaks(),
        4) and targets the specified subset of the peaks_to_detect highest
        peaks of such histogram (up to peaks_to_fit).
        5) Then, it uses the output of SiPMMeas.tune_peak_height() to give
        accurate seeds to SiPMMeas.piecewise_gaussian_fits(), which, in turn,
        gives them to scipy.optimize.curve_fit(),
        6) fits one gaussian function to each one of the targeted peaks
        7) and returns the output of SiPMMeas.piecewise_gaussian_fits(), which
        is made up of two lists, say popt and pcov, so that popt[i] (resp.
        pcov[i]) is the set of optimal values (resp. covariance matrix) for the
        fit of the i-th fit peak. For more information on such output, check
        the SiPMMeas.piecewise_gaussian_fits() docstring."""

        htype.check_type(
            samples,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 36790
            ),
        )
        if np.ndim(samples) != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 82069
                )
            )
        if samples.dtype != np.float64:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 57236
                )
            )
        htype.check_type(
            peaks_to_detect,
            int,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 79312
            ),
        )
        if peaks_to_detect < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 11025
                )
            )
        peaks_to_fit_ = tuple(range(0, peaks_to_detect))
        if peaks_to_fit is not None:

            htype.check_type(
                peaks_to_fit,
                tuple,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 11257
                ),
            )
            if len(peaks_to_fit) > peaks_to_detect:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 93702
                    )
                )
            for elem in peaks_to_fit:
                htype.check_type(
                    elem,
                    int,
                    np.int64,
                    exception_message=htype.generate_exception_message(
                        "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 26318
                    ),
                )
                if elem < 0 or elem > (peaks_to_detect - 1):
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message(
                            "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks",
                            47109,
                        )
                    )
            peaks_to_fit_ = peaks_to_fit

        # We need at least 3 samples per gaussian
        # fit (3 free parameters per gaussian)
        if len(samples) < (3 * len(peaks_to_fit_)):
            raise cuex.NoAvailableData(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks",
                    34723,
                    extra_info=f"The samples array does not have samples enough ({len(samples)}) to fit {len(peaks_to_fit_)} gaussians with 3 free parameters each.",
                )
            )
        htype.check_type(
            bins_no,
            int,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 67585
            ),
        )
        if bins_no < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 11025
                )
            )
        htype.check_type(
            starting_fraction,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 55892
            ),
        )
        if starting_fraction < 0.0 or starting_fraction > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 36107
                )
            )
        htype.check_type(
            step_fraction,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 84304
            ),
        )
        if step_fraction <= 0.0 or step_fraction > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 25201
                )
            )
        htype.check_type(
            minimal_prominence_wrt_max,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 47190
            ),
        )
        if minimal_prominence_wrt_max < 0.0 or minimal_prominence_wrt_max > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 12314
                )
            )
        htype.check_type(
            minimal_width_in_samples,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 13817
            ),
        )
        if minimal_width_in_samples < 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 22781
                )
            )
        htype.check_type(
            std_no,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 69757
            ),
        )
        if std_no <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 23163
                )
            )
        htype.check_type(
            fit_to_density,
            bool,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks", 47288
            ),
        )
        y_values, bin_edges = np.histogram(
            samples, bins=bins_no, density=fit_to_density
        )

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        resolution = np.mean(np.diff(bin_centers))

        peak_height = SiPMMeas.tune_peak_height(
            y_values,
            peaks_to_detect,
            starting_fraction=starting_fraction,
            step_fraction=step_fraction,
            minimal_prominence_wrt_max=minimal_prominence_wrt_max,
            minimal_width_in_samples=minimal_width_in_samples,
        )
        if peak_height is None:
            raise RuntimeError(
                htype.generate_exception_message(
                    "SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks",  # SiPMMeas.tune_peak_height() might not find a suitable peak height
                    21319,
                    extra_info=f"SiPMMeas.tune_peak_height() failed to find a suitable peak height. Please check the data in 'y_values' and 'bin_centers' and the values of the arguments given to SiPMMeas.tune_peak_height().",
                )
            )
        peaks_idx, peaks_properties = spsi.find_peaks(
            y_values,
            # Tuning the height from SiPMMeas.tune_peak_height() so
            # that only peaks_to_detect peaks are found. Also, we are
            # adding a dummy width just so scipy.signal.find_peaks()
            # returns the width information for the detected peaks.
            height=peak_height,
            width=0.0,
            prominence=minimal_prominence_wrt_max * np.max(y_values),
        )

        fit_peaks_idx = [
            peaks_idx[i] for i in peaks_to_fit_
        ]  # Filter out non-fit peaks
        fit_peaks_properties = {
            key: np.array([value[i] for i in peaks_to_fit_])
            for key, value in peaks_properties.items()
        }

        # We are going to fit gaussian functions to the pieces of data which match each of the detected peaks. To do so,
        # the output from scipy.signal.find_peaks() contains valuable information for the seeds of the fit parameters.

        mean_seeds = [bin_centers[fit_peaks_idx[i]] for i in range(len(fit_peaks_idx))]

        # The width calculated by scipy.signal.find_peaks
        # is the peak FWHM in samples. You can check so
        # in scipy documentation on how the peak width
        # and the peak prominence is computed by find_peaks().
        std_seeds = [
            fit_peaks_properties["widths"][i] * resolution / 2.355
            for i in range(len(fit_peaks_properties["widths"]))
        ]

        scaling_seeds = [y_values[fit_peaks_idx[i]] for i in range(len(fit_peaks_idx))]

        popt, pcov = SiPMMeas.piecewise_gaussian_fits(
            bin_centers,
            y_values,
            mean_seeds,
            std_seeds,
            scaling_seeds=scaling_seeds,
            std_no=std_no,
        )
        return popt, pcov

    @staticmethod
    def tune_peak_height(
        signal,
        peaks_to_detect,
        starting_fraction=0.0,
        step_fraction=0.05,
        minimal_prominence_wrt_max=0.0,
        minimal_width_in_samples=0,
    ):
        """This static method gets the following mandatory positional arguments:

        - signal (unidimensional numpy array, int or float)
        - peaks_to_detect (scalar integer)

        This static method gets the following optional keyword argument:

        - starting_fraction (scalar float): It must be semipositive (>=0.0)
        and smaller or equal to 1 (<=1.0).
        - step_fraction (scalar float): It must be positive (>0.0) and
        smaller or equal to 1 (<=1.0).
        - minimal_prominence_wrt_max (scalar float): It must be semipositive
        (>=0) and smaller or equal than 1.0 (<=1.0). It is understood as a
        fraction of the maximum value of the signal, i.e. np.max(signal).
        It is used to feed the prominence keyword argument of
        scipy.signal.find_peaks(). For more information on the prominence
        parameter, check scipy.signal.peak_prominences documentation.
        - minimal_width_in_samples (scalar integer): It must be a semipositive
        (>=0) integer. It is understood as the required width of a peak (in
        samples), for it to be detected as a peak. It is given to the 'width'
        keyword argument of scipy.signal.find_peaks().

        If successful, this method returns an scalar float in the range
        [np.min(signal), np.max(signal)], so that if such value is fed to
        the keyword argument 'height' of
        scipy.signal.find_peaks(signal, prominence=minimal_prominence,
        width=minimal_width_in_samples), where minimal_prominence matches
        minimal_prominence_wrt_max*np.max(signal), the number of detected
        peaks equals peaks_to_detect. If it fails to find such value, this
        method returns None.

        To search the desired value, this method iteratively calls
        scipy.signal.find_peaks(signal, height=input[i], prominence=aux,
        width=minimal_width_in_samples), where aux matches
        minimal_prominence_wrt_max*np.max(signal) and input[i] is the
        value which is used in the i-th iteration. In the 0-th iteration,
        input[0] is computed as

        input[0] = np.min(signal)+(signal_range*starting_fraction)      (0)

        Where

        signal_range = np.max(signal)-np-min(signal)                    (1)

        The update rule for input is the following one:

        input[i+1] = input[i]+(signal_range*step_fraction)              (2)

        If, at any iteration, input[j] happens to yield the desired number of
        detected peaks, namely peaks_to_detect, then the recursive process is
        killed and input[j] is returned.

        If, at any iteration, upon computation of input[k] via (0) or (2),
        input[k] happens to be greater or equal to np.max(signal), then
        scipy.signal.find_peaks(signal, height=np.max(signal), prominence=aux,
        width=minimal_width_in_samples) is evaluated, and the iterative process
        ends, whatever the output. If the last call of scipy.signal.find_peaks
        did (resp. did not) yield the desired number of peaks, then
        np.max(signal) (resp. None) is returned."""

        htype.check_type(
            signal,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.tune_peak_height", 47109
            ),
        )
        if signal.ndim != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SiPMMeas.tune_peak_height", 38978)
            )
        htype.check_type(
            peaks_to_detect,
            int,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.tune_peak_height", 41241
            ),
        )
        htype.check_type(
            starting_fraction,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.tune_peak_height", 38041
            ),
        )
        if starting_fraction < 0.0 or starting_fraction > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SiPMMeas.tune_peak_height", 99213)
            )
        htype.check_type(
            step_fraction,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.tune_peak_height", 41341
            ),
        )
        if step_fraction <= 0.0 or step_fraction > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SiPMMeas.tune_peak_height", 40663)
            )
        htype.check_type(
            minimal_prominence_wrt_max,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.tune_peak_height", 15511
            ),
        )
        if minimal_prominence_wrt_max < 0.0 or minimal_prominence_wrt_max > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SiPMMeas.tune_peak_height", 45453)
            )
        htype.check_type(
            minimal_width_in_samples,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.tune_peak_height", 18481
            ),
        )
        if minimal_width_in_samples < 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SiPMMeas.tune_peak_height", 41122)
            )
        aux = np.max(signal) - np.min(signal)  # Signal range
        height_candidate = np.min(signal) + (aux * starting_fraction)  # Starting height
        keep_on_iterating = True
        while keep_on_iterating:
            # If we have reached the maximum height,
            # then make this iteration the last one
            if height_candidate >= np.max(signal):
                height_candidate = np.max(signal)
                keep_on_iterating = False
            detected_peaks_no = len(
                spsi.find_peaks(
                    signal,
                    height=height_candidate,
                    width=minimal_width_in_samples,
                    prominence=minimal_prominence_wrt_max * np.max(signal),
                )[0]
            )
            if detected_peaks_no == peaks_to_detect:
                return height_candidate
            height_candidate += aux * step_fraction
        # Reaching this line means
        # that no solution was found.
        return None

    @staticmethod
    def piecewise_gaussian_fits(
        x, y, mean_seeds, std_seeds, scaling_seeds=None, std_no=3.0
    ):
        """This static method gets the following mandatory positional arguments:

        - x (unidimensional float numpy array)
        - y (unidimensional numpy array, int or float): Its length must match
        that of x.
        - mean_seeds (resp. std_seeds) (list of floats): mean_seeds[i] (resp.
        std_seeds[i]) is the seed for the gaussian mean (resp. standard
        deviation) of the i-th fit. The length of mean_seeds must match that of
        std_seeds.

        This static method gets the following optional keyword arguments:

        - scaling_seeds (None or list of floats/ints): If defined, then its
        length must match that of mean_seeds, and the gaussian functions which
        are piecewise fitted are not normalized, but the exponential term is
        scaled by a certain factor. In such case, scaling_seeds[i] is the seed
        for such scale factor in the i-th fit.
        - std_no (scalar float): It must be positive (>0.0). This number determines
        the x-range of the input data which is used for each fit. Namely, the x-y
        points which are used for the i-th fit are those which fall within the
        range [mean_seeds[i]-(std_no*std_seeds[i]),
        mean_seeds[i]+(std_no*std_seeds[i])].

        This static method performs N gaussian fits, where N is the length of the
        provided mean_seeds. For each fit, only a piece of the input x-y is used,
        up to the given mean_seeds, std_no and std_seeds. This function returns
        two lists, say popt and pcov, such that len(popt) and len(pcov) matches N,
        and popt[i] (resp. pcov[i]) is the list of optimal parameters (resp. the
        covariance matrix) for the i-th fit, as returned by
        scipy.optimize.curve_fit(). In any case, popt[i][0] (resp. popt[i][1]) is
        the optimal value for the mean (resp. the standard deviation) of the i-th
        gaussian fit. In addition, if scaling_seeds is suitably defined, then
        popt[i][2] is the optimal value for the scaling of the i-th gaussian fit."""

        htype.check_type(
            x,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.piecewise_gaussian_fits", 94064
            ),
        )
        htype.check_type(
            y,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.piecewise_gaussian_fits", 79213
            ),
        )
        if np.ndim(x) != 1 or np.ndim(y) != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.piecewise_gaussian_fits", 30079
                )
            )
        if x.dtype != np.float64:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.piecewise_gaussian_fits", 87459
                )
            )
        if len(x) != len(y):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.piecewise_gaussian_fits", 43730
                )
            )
        htype.check_type(
            mean_seeds,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.piecewise_gaussian_fits", 54380
            ),
        )
        for aux in mean_seeds:
            htype.check_type(
                aux,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.piecewise_gaussian_fits", 80462
                ),
            )
        htype.check_type(
            std_seeds,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.piecewise_gaussian_fits", 50925
            ),
        )
        for aux in std_seeds:
            htype.check_type(
                aux,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.piecewise_gaussian_fits", 71267
                ),
            )
        if len(mean_seeds) != len(std_seeds):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.piecewise_gaussian_fits", 39804
                )
            )
        fWithScaling = False
        scaling_seeds_ = [None for aux in mean_seeds]
        if scaling_seeds is not None:
            htype.check_type(
                scaling_seeds,
                list,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.piecewise_gaussian_fits", 25034
                ),
            )
            for aux in scaling_seeds:
                htype.check_type(
                    aux,
                    float,
                    np.float64,
                    int,
                    np.int64,
                    exception_message=htype.generate_exception_message(
                        "SiPMMeas.piecewise_gaussian_fits", 42173
                    ),
                )
            if len(scaling_seeds) != len(mean_seeds):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SiPMMeas.piecewise_gaussian_fits", 94710
                    )
                )
            fWithScaling = True
            scaling_seeds_ = scaling_seeds

        htype.check_type(
            std_no,
            float,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.piecewise_gaussian_fits", 53503
            ),
        )
        if std_no <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.piecewise_gaussian_fits", 43210
                )
            )
        if fWithScaling:
            gaussian = lambda z, mean, std, scaling: scaling * math.exp(
                -0.5 * (z - mean) * (z - mean) / (std * std)
            )
        else:

            def gaussian(z, mean, std):
                return (1.0 / (std * math.sqrt(2.0 * spcon.pi))) * math.exp(
                    -0.5 * (z - mean) * (z - mean) / (std * std)
                )

            gaussian = lambda z, mean, std: (
                1.0 / (std * math.sqrt(2.0 * spcon.pi))
            ) * math.exp(-0.5 * (z - mean) * (z - mean) / (std * std))

        # Although it is not specified in the scipy.optimize.curve_fit
        # f parameter, it seems that giving a function whose first
        # parameter is not vectorized results in an execution error
        gaussian = np.vectorize(
            gaussian, excluded=(1, 2, 3) if fWithScaling else (1, 2)
        )

        popt, pcov = [], []
        for i in range(len(mean_seeds)):
            mask = x >= (mean_seeds[i] - (std_no * std_seeds[i]))
            mask *= x <= (mean_seeds[i] + (std_no * std_seeds[i]))
            fit_x, fit_y = x[mask], y[mask]
            seeds_package = [mean_seeds[i], std_seeds[i], scaling_seeds_[i]]
            p0 = seeds_package if fWithScaling else seeds_package[:-1]
            aux_popt, aux_pcov = spopt.curve_fit(
                gaussian,
                fit_x,
                fit_y,
                p0=p0,
                # **kwargs are pased to leastsq for method='lm' or least_squares otherwise
                # Got this information from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
                maxfev=2000,
            )

            popt.append(aux_popt)
            pcov.append(aux_pcov)
        return popt, pcov

    @classmethod
    def from_json_file(cls, sipmmeas_config_json):
        """This class method is meant to be an alternative initializer
        for SiPMMeas. This class method gets the following mandatory
        positional argument:

        - sipmmeas_config_json (string): Path to a json file which
        hosts all of the necessary information to define the SiPMMeas
        object.

        This method creates and returns a SiPMMeas object that is
        crafted out of the given json file. To do so, first, this
        method populates two RigidKeyDictionary's. The first one,
        say RKD1, which concerns the SiPMMeas attributes, has the
        following potential keys:

        "delivery_no", "set_no", "meas_no", "strip_ID", 
        "meas_ID", "date", "location", "operator", "setup_ID", 
        "system_characteristics", "thermal_cycle", 
        "electronic_board_number", "electronic_board_location", 
        "electronic_board_socket", "sipm_location", "sampling_ns", 
        "cover_type", "operation_voltage_V", "overvoltage_V", 
        "PDE", "status" and "wvfset_json_filepath".

        Although "sampling_ns" appears here, it is not meant to be
        read from sipmmeas_config_json. The value for
        self.__sampling_ns will be duplicated from the value given to
        "time_resolution" in the file given to wvfset_json_filepath.

        The second one, say RKD2, concerns the WaveformSet.from_files
        parameters, and is populated out of the file whose path is
        given to the wvfset_json_filepath key of RKD1. It has the following
        potential keys:

        "wvf_filepath", "time_resolution", "points_per_wvf",
        "wvfs_to_read", "timestamp_filepath", "delta_t_wf", 
        "set_name", "creation_dt_offset_min" and
        "wvf_extra_info".

        Here, we do not expect a date because the date information
        is taken from the SiPMMeas json file.

        These potential keys are typed according to the
        SiPMMeas.__init__ docstring. To populate RKD1 and RKD2, this
        method uses the dictionaries which are loaded from the
        specified json files. Namely, every entry that belongs to
        one of the two json dictionaries and is suitably formatted, up
        to its corresponding RigidKeyDictionary rules, is added to
        its RigidKeyDictionary.

        Once both RigidKeyDictionary's have been populated, a SiPMMeas
        object is created by calling the class initializer using the
        key-value pairs of RKD1 and RKD2 as the kwarg-value pairs for
        the initializer call, in that order. I.e. the class initializer
        is called with **RKD1, **RKD2. The only exceptions are the
        values given to "wvf_filepath" and "time_resolution" in RKD1,
        which are passed as positional arguments, in such order, to
        the class initializer. "wvfset_json_filepath" is also an exception,
        since it is used to populate RKD2, and it's deleted afterwards.
        """

        htype.check_type(
            sipmmeas_config_json,
            str,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.from_json_file", 67497
            ),
        )

        # These are used to configure the SiPMMea attributes
        pks1 = {
            "delivery_no": int,
            "set_no": int,
            "meas_no": int,
            "strip_ID": int,
            "meas_ID": str,
            "date": str,
            "location": str,
            "operator": str,
            "setup_ID": str,
            "system_characteristics": str,
            "thermal_cycle": int,
            "electronic_board_number": int,
            "electronic_board_location": str,
            "electronic_board_socket": int,
            "sipm_location": int,
            "sampling_ns": float,
            "cover_type": str,
            "operation_voltage_V": float,
            "overvoltage_V": float,
            "PDE": float,
            "status": str,
            "wvfset_json_filepath": str,
        }

        # These are used to configure the WaveformSet
        pks2 = {
            "wvf_filepath": str,
            "time_resolution": float,
            "points_per_wvf": int,
            "wvfs_to_read": int,
            "timestamp_filepath": str,
            "delta_t_wf": float,
            "set_name": str,
            "creation_dt_offset_min": float,
            "wvf_extra_info": str,
        }

        RKD1 = RigidKeyDictionary(
            list(pks1.keys()), is_typed=True, values_types=list(pks1.values())
        )

        RKD2 = RigidKeyDictionary(
            list(pks2.keys()), is_typed=True, values_types=list(pks2.values())
        )

        with open(sipmmeas_config_json, "r") as file:
            input_data = json.load(file)
        RKD1.update(input_data)

        try:
            wvfset_json_filepath = RKD1["wvfset_json_filepath"]
        except KeyError:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.from_json_file",
                    10135,
                    extra_info="No filepath to the waveform set was provided.",
                )
            )
        del RKD1["wvfset_json_filepath"]

        with open(wvfset_json_filepath, "r") as file:
            input_data = json.load(file)
        RKD2.update(input_data)

        # Unless otherwise stated, all of the time values are given in seconds.
        # However, as its name indicates, sampling_ns is expressed in nanoseconds.
        # Thus, here I am converting time_resolution, which is given in seconds, to nanoseconds.
        RKD1["sampling_ns"] = 1e9 * RKD2["time_resolution"]

        input_filepath = RKD2["wvf_filepath"]
        del RKD2["wvf_filepath"]

        time_resolution = RKD2["time_resolution"]
        del RKD2["time_resolution"]

        return cls(input_filepath, time_resolution, **RKD1, **RKD2)

    def output_summary(
        self,
        additional_entries={},
        folderpath=None,
        filename=None,
        overwrite=False,
        indent=None,
        verbose=False
    ):
        """This method gets the following keyword arguments:

        - additional_entries (dictionary): The output summary (a dictionary)
        is updated with this dictionary, additional_entries, right before
        being returned, or loaded to the output json file, up to the value
        given to the 'folderpath' parameter. This update is done via the
        'update' method of dictionaries. Hence, note that if any of the keys
        within additional_entries.keys() already exists in the output
        dictionary, it will be overwritten. Below, you can consult the keys
        that will be part of the output dictionary by default.
        - folderpath (string): If it is defined, it must be a path
        which points to an existing folder, where an output json file
        will be saved.
        - filename (None or string): This parameter only makes a difference
        if the 'folderpath' parameter is defined. In such case, and if the
        'filename' parameter is defined, this is the name of the output json
        file. If it is not defined, then the output json file will be named
        f"{self.__strip_ID}-{self.__sipm_location}-{self.__thermal_cycle}-OV{round(10.*self.__overvoltage_V)}dV-{self.__date.strftime('%Y-%m-%d')}.json".
        - overwrite (bool): This parameter only makes a difference if
        the 'folderpath' parameter is defined and if there is already
        a file in the given folder path whose name matches the output
        json file name, up to the value given to the 'filename' parameter.
        In such case, and if overwrite is True, then this method overwrites
        such file with a new json file. In such case, and if overwrite is
        False, then this method does not generate any json file.
        - indent (None, non-negative integer or string): This parameter only
        makes a difference if the 'folderpath' parameter is defined. It 
        controls the indentation with which the json summary-file is generated.
        It is passed to the 'indent' parameter of json.dump. If indent is None,
        then the most compact representation is used. If indent is a non-negative
        integer, then one new line is added per each key-value pair, and indent
        is the number of spaces that are added at the very beginning of each
        new line. If indent is a string, then one new line is added per each
        key-value pair, and indent is the string that is added at the very
        beginning of each new line. P.e. if indent is a string (such as "\t"),
        each key-value pair is preceded by a tabulator in its own line.
        - verbose (bool): Whether to print functioning-related messages.

        The goal of this method is to produce a summary dictionary of this
        SiPMMeas object. Additionally, this method can serialize this dictionary
        to an output json file if the 'folderpath' parameter is defined. This
        dictionary has as many fields as objects of interest which should be
        summarized. For SiPMMeas objects, these fields are:

        - "delivery_no": Contains self.__delivery_no
        - "set_no": Contains self.__set_no
        - "meas_no": Contains self.__meas_no
        - "strip_ID": Contains self.__strip_ID
        - "meas_ID": Contains self.__meas_ID
        - "date": Contains self.__date
        - "location": Contains self.__location
        - "operator": Contains self.__operator
        - "setup_ID": Contains self.__setup_ID
        - "system_characteristics": Contains self.__system_characteristics
        - "thermal_cycle": Contains self.__thermal_cycle
        - "electronic_board_number": Contains self.__electronic_board_number
        - "electronic_board_location": Contains self.__electronic_board_location
        - "electronic_board_socket": Contains self.__electronic_board_socket
        - "sipm_location": Contains self.__sipm_location
        - "sampling_ns": Contains self.__sampling_ns
        - "waveform_window_mus": Contains self.__waveform_window_mus
        - "cover_type": Contains self.__cover_type
        - "operation_voltage_V": Contains self.__operation_voltage_V
        - "overvoltage_V": Contains self.__overvoltage_V
        - "PDE": Contains self.__PDE
        - "N_events": Contains self.__N_events
        - "signal_unit": Contains self.__signal_unit
        - "status": Contains self.__status
        - "acquisition_time_min": Contains self.__acquisition_time_min

        This method returns a summary dictionary of the SiPMMeas object.
        If a folder path is given, then the output dictionary is additionally
        serialized to a json file in the specified folder with the specified
        file name. If it was not specified, then the default file name is

        f"{self.__strip_ID}-{self.__sipm_location}-{self.__thermal_cycle}-OV{round(10.*self.__overvoltage_V)}dV-{self.__date.strftime('%Y-%m-%d')}.json"
        """

        htype.check_type(
            additional_entries,
            dict,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.output_summary", 15693
            ),
        )
        
        fOutputJSON = False

        if folderpath is not None:

            htype.check_type(
                folderpath,
                str,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.output_summary", 84477
                ),
            )

            if not os.path.exists(folderpath):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.output_summary", 62875)
                )
            elif not os.path.isdir(folderpath):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SiPMMeas.output_summary", 64055)
                )
            
            fOutputJSON = True

            if filename is not None:

                htype.check_type(
                    filename,
                    str,
                    exception_message=htype.generate_exception_message(
                        "SiPMMeas.output_summary", 84055
                    ),
                )

                output_filepath = os.path.join(folderpath, filename)

            else:

                aux = f"{self.__strip_ID}-{self.__sipm_location}-"
                f"{self.__thermal_cycle}-OV{round(10.*self.__overvoltage_V)}dV"
                f"-{self.__date.strftime('%Y-%m-%d')}.json"

                output_filepath = os.path.join(folderpath, aux)
        
            htype.check_type(
                overwrite,
                bool,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.output_summary", 99583
                ),
            )

            if indent is not None:

                htype.check_type(
                    indent,
                    int,
                    np.int64,
                    str,
                    exception_message=htype.generate_exception_message(
                        "SiPMMeas.output_summary", 87057
                    ),
                )

                if isinstance(indent, int) or isinstance(indent, np.int64):
                    if indent < 0:
                        raise cuex.InvalidParameterDefinition(
                            htype.generate_exception_message(
                                "SiPMMeas.output_summary", 68241
                            )
                        )

        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.output_summary", 40366
            ),
        )

        output = {
            "delivery_no": self.__delivery_no,
            "set_no": self.__set_no,
            "meas_no": self.__meas_no,
            "strip_ID": self.__strip_ID,
            "meas_ID": self.__meas_ID,
            # Object of type datetime is not
            # JSON serializable, but strings are
            "date": self.Date.strftime("%Y-%m-%d %H:%M:%S"),
            "location": self.__location,
            "operator": self.__operator,
            "setup_ID": self.__setup_ID,
            "system_characteristics": self.__system_characteristics,
            "thermal_cycle": self.__thermal_cycle,
            "electronic_board_number": self.__electronic_board_number,
            "electronic_board_location": self.__electronic_board_location,
            "electronic_board_socket": self.__electronic_board_socket,
            "sipm_location": self.__sipm_location,
            "sampling_ns": self.__sampling_ns,
            "waveform_window_mus": self.__waveform_window_mus,
            "cover_type": self.__cover_type,
            "operation_voltage_V": self.__operation_voltage_V,
            "overvoltage_V": self.__overvoltage_V,
            "PDE": self.__PDE,
            "N_events": self.__N_events,
            "signal_unit": self.__signal_unit,
            "status": self.__status,
            "acquisition_time_min": self.__acquisition_time_min
        }

        output.update(additional_entries)

        if fOutputJSON:

            if os.path.exists(output_filepath):
                if not overwrite:
                    fOutputJSON = False
                    if verbose:
                        print(
                            f"In function SiPMMeas.output_summary(): "
                            f"{output_filepath} already exists. It won't be overwritten."
                        )
                else:
                    if verbose:
                        print(
                            f"In function SiPMMeas.output_summary(): "
                            f"{output_filepath} already exists. It will be overwritten."
                        )

        # The value of fOutputJSON might have changed
        # in the previous if-block of code. This is
        # why we need to re-check it here
        if fOutputJSON:

            with open(output_filepath, "w") as file:
                json.dump(output, file, indent=indent)

            if verbose:
                print(
                    f"In function SiPMMeas.output_summary(): The output file has been written to {output_filepath}."
                )

        return output

    @staticmethod
    def get_value_from_dict(dictionary, key, none_fallback=False):
        """This static method gets the following arguments:

        - dictionary (dict)
        - key (object)
        - none_fallback (bool): This parameter only makes a 
        difference if key does not match any of the keys in
        the given dictionary. If that's the case, and if
        none_fallback is True, then this method returns None.
        If none_fallback is False, then this method raises a
        KeyError.

        This method returns dictionary[key] if the given key
        exists in the given dictionary. If it does not, the
        behaviour depends on the value of none_fallback."""

        htype.check_type(
            dictionary,
            dict,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.get_value_from_dict", 57329
            ),
        )

        htype.check_type(
            none_fallback,
            bool,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.get_value_from_dict", 46243
            ),
        )

        try:
            return dictionary[key]
        except KeyError:
            if none_fallback:
                return None
            else:
                raise KeyError(
                    f"In function SiPMMeas.get_value_from_dict(): "
                    "The key {key} does not exist in the given dictionary."
                )