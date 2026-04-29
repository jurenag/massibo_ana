import os
import json
import math
import copy
from datetime import datetime
from abc import ABC
import numpy as np
from scipy import signal as spsi
from scipy import constants as spcon
from scipy import optimize as spopt
from scipy import stats as spsta
from iminuit import Minuit
from iminuit.cost import BinnedNLL

import massibo_ana.utils.htype as htype
import massibo_ana.utils.custom_exceptions as cuex
from massibo_ana.custom_types.RigidKeyDictionary import RigidKeyDictionary
from massibo_ana.preprocess.DataPreprocessor import DataPreprocessor
from massibo_ana.preprocess.NumpyDataPreprocessor import NumpyDataPreprocessor
from massibo_ana.core.WaveformSet import WaveformSet


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
        verbose=True,
        **kwargs,
    ):
        """This class aims to model a base class from which to derive certain SiPM measurements.
        For example, a class which implements a gain measurement of a SiPM, say SiPMMeas, could
        inherit from SiPMMeas. The same goes for a measurement of the dark noise of a SiPM. In line
        with Waveform and WaveformSet classes, the assumed time unit is the second, unless otherwise
        indicated. This initializer gets the following positional arguments:

        - args: These positional arguments are given to WaveformSet.from_files. They must be
        two positional arguments: input_filepath (string) and time_resolution_s (positive float),
        in such order. For more information on these arguments, please refer to
        WaveformSet.from_files docstring. Particularly, time_resolution_s is assumed to be expressed
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
        - verbose (boolean): Whether to print functioning related messages.
        - kwargs: These keyword arguments are given to WaveformSet.from_files. The expected keywords
        are points_per_wvf (int), wvfs_to_read (int), timestamp_filepath (string),
        delta_t_wf (float), packing_version (int), set_name (string), creation_dt_offset_min (float)
        and wvf_extra_info (string). To understand these arguments, please refer to the
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

        packing_version = SiPMMeas.get_value_from_dict(
            kwargs, "packing_version", none_fallback=True)

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
            packing_version=packing_version,
            ref_datetime=self.__date,
            creation_dt_offset_min=creation_dt_offset_min,
            wvf_extra_info=wvf_extra_info,
            verbose=verbose
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

    @AcquisitionTime_min.setter
    def AcquisitionTime_min(self, value):
        htype.check_type(
            value,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.AcquisitionTime_min.setter",
                31832
            ),
        )

        if value <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.AcquisitionTime_min.setter",
                    84211
                )
            )

        self.__acquisition_time_min = value

    @property
    def Waveforms(self):
        return self.__waveforms

    @staticmethod
    def fit_gaussians_to_the_n_highest_peaks(
        samples,
        peaks_to_detect=2,
        peaks_to_fit=None,
        bins_no=125,
        histogram_range=None,
        starting_fraction=0.0,
        step_fraction=0.01,
        minimal_prominence_wrt_max=0.0,
        fit_parameters_bounds=None,
        std_no=3.0,
        peak_distance_in_bins=5,
        use_correlated_gaussians=False,
        optimize_poisson_likelihood=False
    ):
        """This static method gets the following optional keyword arguments:

        - samples (unidimensional float numpy array): The samples within this
        array are used to build an histogram. Such histogram is, then,
        fit to gaussian functions.
        - peaks_to_detect (scalar integer): It must be positive (>0). Number of
        peaks which will be detected to start with. A subset of the detected
        peaks will be fit, up to the peaks_to_fit argument.
        - peaks_to_fit (None or tuple): This parameter only makes a difference
        if use_correlated_gaussians is False. In that case, if it is None, then
        it is assumed that all of the detected peaks should be fit. If it is a
        tuple, then it must contain integers. Its length must comply with
        0<=len(peaks_to_fit)<=peaks_to_detect. Every entry must belong to
        the interval [0, peaks_to_detect-1]. Let us sort the peaks_to_detect
        detected peaks according to the iterator value for the fit histogram
        where they occur. Then if i belongs to peaks_to_fit, the i-th detected
        peak will be fit. Otherwise, if use_correlated_gaussians is True, then
        all detected peaks are used for the correlated fit.
        - bins_no (scalar integer): It must be positive (>0). It is the number
        of bins which are used to histogram the given samples.
        - histogram_range (None or tuple): If None, then numpy.histogram
        automatically sets the histogram range to the minimum and maximum values
        of the samples array. Otherwise, it should be a tuple of two elements,
        say (min, max), that defines the lower and upper range of the histogram.
        - starting_fraction (scalar float): It must be semipositive (>=0.0)
        and smaller or equal to 1 (<=1.0). It is given to the 'initial_percentage'
        parameter of the static method
        SiPMMeas.__spot_first_peaks_in_CalibrationHistogram(). Check its
        docstring for more information.
        - step_fraction (scalar float): It must be positive (>0.0) and smaller
        or equal to 1 (<=1.0). It is given to the 'percentage_step' parameter of
        the static method SiPMMeas.__spot_first_peaks_in_CalibrationHistogram().
        Check its docstring for more information.
        - minimal_prominence_wrt_amp (scalar float): It must be semipositive
        (>=0) and smaller or equal than 1.0 (<=1.0). It is given to the
        'prominence' parameter of the static method
        SiPMMeas.__spot_first_peaks_in_CalibrationHistogram().
        It sets a minimal prominence for a peak to be detected, based on a
        fraction of the amplitude of the samples histogram. I.e. the only
        detected peaks are those whose prominence is bigger or equal to a
        fraction of the samples histogram amplitude. For more information
        check the SiPMMeas.__spot_first_peaks_in_CalibrationHistogram()
        docstring.
        - fit_parameters_bounds (None or 2-tuple of array-like): It sets
        the lower and upper bounds on the gaussian fit parameters. The
        structure of this parameter depends on the value of
        use_correlated_gaussians. When False (independent fits), the bounds
        apply to mean, std, and scaling if applicable, for each Gaussian.
        When True (correlated fit), the bounds apply to [center_0,
        mean_increment, std_0, std_increment, S_0, S_1, ...]. If None,
        default bounds are used.
        - std_no (scalar float): It must be positive (>0.0). This parameter
        is given to the std_no keyword argument of
        SiPMMeas.piecewise_gaussian_fits() or SiPMMeas.correlated_gaussians_fit().
        Check their docstrings for more information.
        - peak_distance_in_bins (positive integer): The minimum distance
        (in histogram bins) between detected peaks. It is given to the
        'distance' parameter of scipy.signal.find_peaks() via
        SiPMMeas.__spot_first_peaks_in_CalibrationHistogram().
        - use_correlated_gaussians (scalar boolean): If False (default),
        independent Gaussian fits are performed for each peak via
        SiPMMeas.piecewise_gaussian_fits(). If True, a single fit using
        correlated Gaussians is performed via SiPMMeas.correlated_gaussians_fit(),
        where the peak means follow mu_i = center_0 + i*gain and the standard
        deviations follow sigma_i = sqrt(std_0^2 + i*std_increment^2).
        - optimize_poisson_likelihood (scalar boolean): If False (default),
        scipy.optimize.curve_fit() is used in the selected fitting method.
        If True, iminuit.BinnedNLL is used to maximize a Poisson likelihood
        for the histogram counts.

        This method fits Gaussian functions to peaks within the histogram
        of samples. By default (use_correlated_gaussians=False), it fits one
        independent Gaussian to each peak. When use_correlated_gaussians=True,
        it fits a sum of correlated Gaussians where the peak positions and
        widths follow a parametric relationship.

        The method returns a tuple (popt, pcov, fit_function) where the structure
        depends on use_correlated_gaussians:

        When use_correlated_gaussians=False:
          - popt: list of arrays, popt[i] = [mean, std, scaling] for i-th peak
          - pcov: list of 2D arrays, pcov[i] is covariance matrix for i-th peak
          - fit_function: callable evaluating the sum of independent Gaussians

        When use_correlated_gaussians=True:
          - popt: single array [center_0, mean_increment, std_0, std_increment,
            S_0, S_1, ...], where mean_increment is the gain
          - pcov: single 2D covariance matrix for all parameters
          - fit_function: callable evaluating the correlated Gaussian sum

        The method proceeds as follows:
        1) Generates a histogram using samples entries
        2) Detects the peaks_to_detect first peaks of such histogram
           via SiPMMeas.__spot_first_peaks_in_CalibrationHistogram()
        3) When use_correlated_gaussians=False, targets the specified subset
           of peaks (up to peaks_to_fit) and fits independent Gaussians
        4) When use_correlated_gaussians=True, uses all detected peaks to
           fit a sum of correlated Gaussians"""

        htype.check_type(
            samples,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                36790
            ),
        )
        if np.ndim(samples) != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    82069
                )
            )
        if samples.dtype != np.float64:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    57236
                )
            )
        htype.check_type(
            peaks_to_detect,
            int,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                79312
            ),
        )
        if peaks_to_detect < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    58294
                )
            )
        peaks_to_fit_ = tuple(range(0, peaks_to_detect))
        if peaks_to_fit is not None:

            htype.check_type(
                peaks_to_fit,
                tuple,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    11257
                ),
            )
            if len(peaks_to_fit) > peaks_to_detect:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                        93702
                    )
                )
            for elem in peaks_to_fit:
                htype.check_type(
                    elem,
                    int,
                    np.int64,
                    exception_message=htype.generate_exception_message(
                        "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                        26318
                    ),
                )
                if elem < 0 or elem > (peaks_to_detect - 1):
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message(
                            "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                            90280,
                        )
                    )
            peaks_to_fit_ = peaks_to_fit

        htype.check_type(
            bins_no,
            int,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                67585
            ),
        )
        if bins_no < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    11025
                )
            )

        if histogram_range is not None:
            htype.check_type(
                histogram_range,
                tuple,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    47289
                ),
            )
            if len(histogram_range) != 2:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                        29866
                    )
                )
            for elem in histogram_range:
                htype.check_type(
                    elem,
                    int,
                    float,
                    np.int64,
                    np.float64,
                    exception_message=htype.generate_exception_message(
                        "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                        19276
                    ),
                )

        htype.check_type(
            starting_fraction,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                55892
            ),
        )
        if starting_fraction < 0.0 or starting_fraction > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    36107
                )
            )
        htype.check_type(
            step_fraction,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                84304
            ),
        )
        if step_fraction <= 0.0 or step_fraction > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    25201
                )
            )
        htype.check_type(
            minimal_prominence_wrt_max,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                47190
            ),
        )
        if minimal_prominence_wrt_max < 0.0 or minimal_prominence_wrt_max > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    12314
                )
            )
        htype.check_type(
            std_no,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                69757
            ),
        )
        if std_no <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    23163
                )
            )
        htype.check_type(
            optimize_poisson_likelihood,
            bool,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                92831
            ),
        )
        y_values, bin_edges = np.histogram(
            samples,
            bins=bins_no,
            range=histogram_range,
        )

        if not use_correlated_gaussians:
            # We need at least 3 points per gaussian
            # fit (3 free parameters per gaussian)
            if len(y_values) < (3 * len(peaks_to_fit_)):
                raise cuex.NoAvailableData(
                    htype.generate_exception_message(
                        "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                        34723,
                        extra_info=f"The y_values array does not contain samples "
                        f"enough ({len(samples)}) to fit {len(peaks_to_fit_)} "
                        "gaussians with 3 free parameters each.",
                    )
                )

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        resolution = np.mean(np.diff(bin_centers))

        found_requested_peaks, spsi_output = \
            SiPMMeas.__spot_first_peaks_in_CalibrationHistogram(
                y_values,
                peaks_to_detect,
                minimal_prominence_wrt_max,
                initial_percentage=starting_fraction,
                percentage_step=step_fraction,
                peak_distance_in_bins=peak_distance_in_bins
            )

        if not found_requested_peaks:
            raise cuex.RequestedPeaksNotFound(
                htype.generate_exception_message(
                    "SiPMMeas.fit_gaussians_to_the_n_highest_peaks",
                    21318,
                    extra_info=f"SiPMMeas.fit_gaussians_to_the_n_highest_peaks()"
                    " failed to find the requested number of peaks. Please check the data"
                    " in 'y_values' and 'bin_centers' and the values of the arguments "
                    "given to SiPMMeas.__spot_first_peaks_in_CalibrationHistogram().",
                )
            )

        # If we reached this point it means that peaks_to_detect peaks were
        # found, so peaks_to_fit_ is well-formed with respect to spsi_output.
        # Also, we are using the call to SiPMMeas.__spot_first_peaks_in_CalibrationHistogram()
        # to unpack the output of scipy.signal.find_peaks().

        targeted_peaks_idx, targeted_peaks_properties = \
            SiPMMeas.__select_peaks_from_spsi_find_peaks_output(
                spsi_output,
                tuple(range(peaks_to_detect)) if use_correlated_gaussians\
                    else peaks_to_fit_
            )

        # Use the output of scipy.signal.find_peaks() to get
        # seeds for the fit parameters.
        mean_seeds = [
            bin_centers[targeted_peaks_idx[i]]
            for i in range(len(targeted_peaks_idx))
        ]

        # The width calculated by scipy.signal.find_peaks
        # is the peak FWHM in samples. You can check so
        # in scipy documentation on how the peak width
        # and the peak prominence is computed by find_peaks().
        # Also, it is worth noting here that
        # SiPMMeas.__spot_first_peaks_in_CalibrationHistogram()
        # calls scipy.signal.find_peaks(*args, width=0,
        # rel_height=0.5, **kwargs), so we are safe asking
        # for the peak width property here, and interpreting
        # it as a width at half height.
        std_seeds = [
            targeted_peaks_properties["widths"][i] * resolution / 2.355
            for i in range(len(targeted_peaks_properties["widths"]))
        ]

        scaling_seeds = [
            y_values[targeted_peaks_idx[i]]
            for i in range(len(targeted_peaks_idx))
        ]

        aux_function = SiPMMeas.correlated_gaussians_fit \
            if use_correlated_gaussians else SiPMMeas.piecewise_gaussian_fits

        return aux_function(
            bin_centers,
            y_values,
            mean_seeds,
            std_seeds,
            bin_edges=bin_edges,
            scaling_seeds=scaling_seeds,
            fit_parameters_bounds=fit_parameters_bounds,
            std_no=std_no,
            optimize_poisson_likelihood=optimize_poisson_likelihood
        )

    @staticmethod
    def __spot_first_peaks_in_CalibrationHistogram(
        y_values,
        max_peaks: int,
        prominence: float,
        initial_percentage=0.1,
        percentage_step=0.1,
        peak_distance_in_bins=5
    ):
        """This helper method gets the positional argument:

        - y_values (unidimensional numpy array, int or float):
        The values to spot peaks on
        - max_peaks (int): The maximum number of peaks to spot.
        It must be a positive integer. This is not checked here,
        it is the caller's responsibility to ensure this.
        - prominence (float): The prominence parameter to pass
        to the scipy.signal.find_peaks() function. Since the
        signal is normalized, this prominence can be understood
        as a fraction of the total amplitude of the signal. P.e.
        setting prominence to 0.5, will prevent scipy.signal.find_peaks()
        from spotting peaks whose prominence is less than half
        of the total amplitude of the signal.

        This helper method gets the following keyword arguments:

        - initial_percentage (float): The initial percentage
        of the y_values array to consider. It must be greater
        than 0.0 and smaller than 1.0.
        - percentage_step (float): The percentage step to
        increase the signal to consider in successive calls
        of scipy.signal.find_peaks(). It must be greater than
        0.0 and smaller than 1.0.
        - peak_distance_in_bins (positive integer): The minimum
        distance (in bins) between detected peaks. It is given
        to the 'distance' parameter of scipy.signal.find_peaks().

        This helper method is not intended for user usage.
        It must be only called by
        fit_gaussians_to_the_n_highest_peaks(),
        where the well-formedness checks of the input
        parameters have been performed. This function tries
        to find peaks over the signal which is computed as

            signal = (y_values - np.min(y_values))/np.max(y_values)

        This function iteratively calls

            scipy.signal.find_peaks(signal[0:points],
                                    prominence = prominence)

        to spot, at most, max_peaks peaks. To do so, at the
        first iteration, points is computed as
        math.floor(initial_percentage * len(signal)). If the
        number of spotted peaks is less than max_peaks, then
        points is increased by
        math.ceil(percentage_step * len(signal)) and the
        scipy.signal.find_peaks() function is called again.
        This process is repeated until the number of spotted peaks
        is equal to max_peaks, or until the number of points
        reaches len(signal). If the number of points reaches
        len(signal), then scipy.signal.find_peaks() is called
        one last time as

            scipy.signal.find_peaks(signal,
                                    prominence = prominence)

        If the last call found a number of peaks smaller than
        max_peaks, then this function returns (False, peaks),
        where peaks is the output of the last call to
        scipy.signal.find_peaks(). If the last call found a
        number of peaks greater than or equal to max_peaks,
        then the function returns (True, peaks), where peaks
        is the output of scipy.signal.find_peaks() but
        truncated to the first max_peaks found peaks. For
        more information on the second object of the returned
        tuple, check the scipy.signal.find_peaks() documentation.
        """

        signal = (y_values - np.min(y_values)) / np.max(y_values)

        fFoundMax = False
        fReachedEnd = False
        points = math.floor(initial_percentage * len(signal))

        while not fFoundMax and not fReachedEnd:

            points = min(points, len(signal))

            # Adding a minimal 0 width, which constraints nothing,
            # but which makes scipy.signal.find_peaks() return
            # information about each peak-width at half its height.

            spsi_output = spsi.find_peaks(
                signal[0:points],
                prominence=prominence,
                width=0,
                rel_height=0.5,
                distance=peak_distance_in_bins
            )

            if len(spsi_output[0]) >= max_peaks:

                # Using __select_peaks_from_spsi_find_peaks_output()
                # to truncate the output of scipy.signal.find_peaks()
                spsi_output = \
                    SiPMMeas.__select_peaks_from_spsi_find_peaks_output(
                        spsi_output,
                        tuple(range(0, max_peaks))
                )
                fFoundMax = True

            if points == len(signal):
                fReachedEnd = True

            points += math.ceil(percentage_step * len(signal))

        if fFoundMax:
            return (True, spsi_output)
        else:
            return (False, spsi_output)

    @staticmethod
    def __filter_out_same_height_peaks_from_spsi_find_peaks_output(
        spsi_output,
        signal,
    ):
        """
        DEPRECATED: This function was called by
        SiPMMeas.__spot_first_peaks_in_CalibrationHistogram() to
        filter out the doubly spotted peaks. I.e. signals which reach
        the exact same top value twice due to a fluctuation right
        on the top of the peak. However, this function also discards
        some peaks which are not doubly spotted, but which reach the
        same top value as another peak. This is why this function is
        no longer used. The doubly spotted peaks, though, will be
        addressed using the `distance` parameter of
        scipy.signal.find_peaks().

        This helper method gets the following positional
        arguments:

        - spsi_output (tuple of (np.ndarray, dict,)): The output
        of a call to scipy.signal.find_peaks(). No checks are
        performed here regarding the well-formedness of this input.
        - signal (unidimensional numpy array, int or float): The
        first positional argument that was given to
        scipy.signal.find_peaks() when generating spsi_output.
        The caller is responsible for ensuring the well-formedness
        of this input.

        This helper method should only be called by
        SiPMMeas.__spot_first_peaks_in_CalibrationHistogram().
        This function gets the output of a certain call to
        scipy.signal.find_peaks() and, for any group of peaks
        which reach the exact same height, it keeps only the
        first one, i.e. the one which occurs for the smallest
        index in the original signal. The rest of them are discarded.
        To this end, this function also gets the original signal,
        which is used to evaluate the height of each spotted peak.
        """

        original_peaks_idx = spsi_output[0]
        original_peaks_properties = spsi_output[1]

        if len(original_peaks_idx) < 2:
            return spsi_output

        already_seen_peaks_top = set()
        filtered_peaks_idx = []
        filtered_peaks_properties = {}

        for i in range(len(original_peaks_idx)):

            if signal[original_peaks_idx[i]] not in already_seen_peaks_top:
                already_seen_peaks_top.add(signal[original_peaks_idx[i]])
                filtered_peaks_idx.append(original_peaks_idx[i])

                for key, value in original_peaks_properties.items():
                    if key not in filtered_peaks_properties:
                        filtered_peaks_properties[key] = []
                    filtered_peaks_properties[key].append(value[i])

        return (np.array(filtered_peaks_idx), filtered_peaks_properties)

    @staticmethod
    def __select_peaks_from_spsi_find_peaks_output(
        spsi_output,
        peaks_to_select
    ):
        """This helper method gets the following positional
        arguments:

        - spsi_output (tuple of (np.ndarray, dict,)): The output
        of a call to scipy.signal.find_peaks(). No checks are
        performed here regarding the well-formedness of this input.
        - peaks_to_select (tuple of int): Its length must comply
        with 0<=len(peaks_to_select)<=len(spsi_output[0]). Every
        entry must belong to the interval [0, len(spsi_output[0])-1].
        These checks are not performed here.

        This helper method should only be called by the
        SiPMMeas.fit_gaussians_to_the_n_highest_peaks()
        static method, where the well-formedness checks of the
        input parameters have been performed. This function
        gets the output of a certain call to
        scipy.signal.find_peaks(), selects some entries from it,
        up to the 'peaks_to_select' parameter, and returns the
        list of selected peaks, following the same format as
        the scipy.signal.find_peaks() output. I.e. the returned
        output is a tuple of two elements. The first element
        is an unidimensional numpy array which contains the
        selected elements of the first element of the given
        spsi_output. The second element is a dictionary which
        contains the same keys as the second element of the
        given spsi_output, but the values (which are
        unidimensional numpy arrays) contain only the selected
        elements of the values of the second element of the
        given spsi_output.
        """

        aux = set(peaks_to_select)

        first_output = [
            spsi_output[0][i] for i in aux
        ]

        second_output = {
            key: np.array([value[i] for i in aux])
            for key, value in spsi_output[1].items()
        }

        return (first_output, second_output)

    @staticmethod
    def tune_peak_height(
        signal,
        peaks_to_detect,
        starting_fraction=0.0,
        step_fraction=0.05,
        minimal_prominence_wrt_max=0.0,
        minimal_width_in_samples=0,
    ):
        """Note: This function is currently not used. It is kept here
        just in case it becomes useful again at some point.

        This static method gets the following mandatory positional arguments:

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
        x,
        y,
        mean_seeds,
        std_seeds,
        bin_edges=None,
        scaling_seeds=None,
        fit_parameters_bounds=None,
        std_no=3.0,
        optimize_poisson_likelihood=False
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

        - bin_edges (None or unidimensional float numpy array): It is ignored
        if optimize_poisson_likelihood is False. Otherwise, it must define
        the histogram bin edges used to produce y, so len(bin_edges)=len(y)+1.
        - scaling_seeds (None or list of floats/ints): If defined, then its
        length must match that of mean_seeds, and the gaussian functions which
        are piecewise fitted are not normalized, but the exponential term is
        scaled by a certain factor. In such case, scaling_seeds[i] is the seed
        for such scale factor in the i-th fit.
        - fit_parameters_bounds (None or 2-tuple of array-like): It sets the
        lower and upper bounds on the gaussian fit parameters, i.e. for the
        gaussian mean and standard deviation. Note that, additionally, if
        the scaling_seeds are defined, then the bounds for the scaling should
        also be set. If it is None, then, then only the [0, np.inf] bounds
        are used for the scaling factor (if applicable) while the rest of the
        parameters are unbounded (-np.inf, np.inf). Otherwise, if this parameter
        is defined, then both elements of the tuple must contain as many entries
        as fit parameters. The i-th entry of the first (resp. second) element
        of the tuple contains the lower (resp. upper) bound for the i-th fit
        parameter.
        - std_no (scalar float): It must be positive (>0.0). This number determines
        the x-range of the input data which is used for each fit. Namely, the x-y
        points which are used for the i-th fit are those which fall within the
        range [mean_seeds[i]-(std_no*std_seeds[i]),
        mean_seeds[i]+(std_no*std_seeds[i])].
        - optimize_poisson_likelihood (scalar boolean): If False (default),
        scipy.optimize.curve_fit() is used. If True, a Poisson negative
        likelihood maximization is performed via iminuit.BinnedNLL. In this
        case, the scaling seeds are ignored. For each peak, the normalized
        histogram is fitted to a normalized gaussian, and the optimal scaling
        factor is computed as the integral of the fitted histogram times the
        inverse of the normal distribution normalization factor. This is done
        so that the fit scale is interpreted as a factor that is applied
        directly to the exponential term of the gaussian, and not the normalized
        normal distribution.

        This static method performs N gaussian fits, where N is the length of the
        provided mean_seeds. For each fit, only a piece of the input x-y is used,
        up to the given mean_seeds, std_no and std_seeds. This function returns
        two lists, say popt and pcov, such that len(popt) and len(pcov) matches N,
        and popt[i] (resp. pcov[i]) is the list of optimal parameters (resp. the
        covariance matrix) for the i-th fit, as returned by
        scipy.optimize.curve_fit(). In any case, popt[i][0] (resp. popt[i][1]) is
        the optimal value for the mean (resp. the standard deviation) of the i-th
        gaussian fit. In addition, if scaling_seeds is suitably defined, then
        popt[i][2] is the optimal value for the scaling of the i-th gaussian fit.
        If optimize_poisson_likelihood is True, then the returned covariance
        matrices are 2x2 (they do not include scaling information) regardless
        whether scaling_seeds is defined or not. In addition to those two lists,
        this function returns a callable, say fit_functions_sum, which evaluates
        the sum of the piecewise gaussian fits."""

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

        htype.check_type(
            optimize_poisson_likelihood,
            bool,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.piecewise_gaussian_fits", 68954
            ),
        )

        if optimize_poisson_likelihood:
            htype.check_type(
                bin_edges,
                np.ndarray,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.piecewise_gaussian_fits", 95012
                ),
            )
            if np.ndim(bin_edges) != 1 or len(bin_edges) != (len(y) + 1):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SiPMMeas.piecewise_gaussian_fits", 23411
                    )
                )
            if bin_edges.dtype != np.float64:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SiPMMeas.piecewise_gaussian_fits", 49092
                    )
                )

        if fWithScaling:
            # Compensate the normalization factor of the
            # normal distribution given by spsta.norm.pdf()
            gaussian = lambda z, mean, std, scaling: \
                scaling * \
                    std * math.sqrt(2. * spcon.pi) * \
                        spsta.norm.pdf(
                            z,
                            loc=mean,
                            scale=std
                        )
        else:
            gaussian = lambda z, mean, std: \
                spsta.norm.pdf(
                    z,
                    loc=mean,
                    scale=std
                )

        # Although it is not specified in the scipy.optimize.curve_fit
        # f parameter, it seems that giving a function whose first
        # parameter is not vectorized results in an execution error
        gaussian = np.vectorize(
            gaussian,
            excluded=(1, 2, 3) if fWithScaling else (1, 2)
        )

        # Note that, in both cases, the std parameter is unbounded.
        # In principle, we could set a lower bound of 0 for the std parameter.
        # However, we have seen in previous executions of the gain analysis
        # that std may be of the order of 1e-10. For such an small value,
        # setting a lower bound of 0 (close to 1e-10) may make the 'trf'
        # method unstable, p.e. causing a reflection in this 0 boundary.
        if fit_parameters_bounds is None:
            if fWithScaling:
                fit_parameters_bounds_ = (
                    np.array([-np.inf, -np.inf, 0.0]),
                    np.array([np.inf, np.inf, np.inf])
                )
            else:
                fit_parameters_bounds_ = (
                    np.array([-np.inf, -np.inf]),
                    np.array([np.inf, np.inf])
                )
        else:
            fit_parameters_bounds_ = fit_parameters_bounds

        popt, pcov = [], []
        for i in range(len(mean_seeds)):

            seeds_package = [
                mean_seeds[i], std_seeds[i], scaling_seeds_[i]
            ]

            # Use copy to avoid modifying the
            # original seeds_package when scaling p0
            p0 = seeds_package if fWithScaling \
                else seeds_package[:-1]

            for j in range(len(p0)):
                try:
                    if p0[j] < fit_parameters_bounds_[0][j] or \
                        p0[j] > fit_parameters_bounds_[1][j]:

                        aux = np.clip(
                            p0[j],
                            fit_parameters_bounds_[0][j],
                            fit_parameters_bounds_[1][j]
                        )

                        print(
                            "In function SiPMMeas.piecewise_gaussian_fits(): "
                            f"The seed value {p0[j]} for the {j}-th fit parameter"
                            " is not within the used bounds "
                            f"({fit_parameters_bounds_[0][j]}, {fit_parameters_bounds_[1][j]})."
                            f" This seed will be replaced by the closest bound ({aux})."
                        )

                        p0[j] = aux

                except IndexError:
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message(
                            "SiPMMeas.piecewise_gaussian_fits",
                            54013,
                            extra_info=f"The fit_parameters_bounds_ parameter is ill-formed."
                        )
                    )

            # Scale and shift the input data to make it
            # (approximatedly) distributed as a gaussian
            # with mean ~0.0 and std ~1.0, avoiding
            # extreme orders of magnitude and making the
            # fit more numerically stable.
            horizontal_axis_shift_for_fit = mean_seeds[i]
            horizontal_axis_scaling_for_fit = std_seeds[i]

            # Re-set the seeds accordingly
            p0[0] = 0.
            p0[1] = 1.

            # Note 1: This transformation preserves the
            # clipped values, since we will transform the
            # bounds are scaled as well
            # Note 2: The seeds and bounds are only used
            # for the optimization process, i.e. they are
            # not returned as part of the output of this
            # function, so they don't need to be rescaled
            # back at any point later
            # Note 3: The seeds (p0) are defined within
            # this loop (current value of i), so they are
            # are scaled for every iteration, but the
            # bounds (fit_parameters_bounds_) are defined
            # outside this loop, so they are only scaled
            # in the first iteration of this loop.
            if i == 0:
                for j in (0, 1):
                    # Scale the mean bounds
                    fit_parameters_bounds_[j][0] = \
                        (fit_parameters_bounds_[j][0] - horizontal_axis_shift_for_fit) \
                        / horizontal_axis_scaling_for_fit

                    # Scale the std bounds
                    fit_parameters_bounds_[j][1] /= horizontal_axis_scaling_for_fit

            # Select the data range for fitting
            x_min = mean_seeds[i] - (std_no * std_seeds[i])
            x_max = mean_seeds[i] + (std_no * std_seeds[i])

            if optimize_poisson_likelihood:

                # In this case, ignore the scaling seeds
                p0 = p0[:2]
                fit_parameters_bounds_ = fit_parameters_bounds_[:2]

                mask_edges = (
                    (bin_edges[:-1] >= x_min)
                    & (bin_edges[1:] <= x_max)
                )
                if not np.any(mask_edges):
                    raise cuex.NotEnoughFitSamples(
                        htype.generate_exception_message(
                            "SiPMMeas.piecewise_gaussian_fits",
                            68959,
                            extra_info="The selected histogram interval does not "
                            "contain bins for the requested fit."
                        )
                    )

                fit_counts = y[mask_edges]
                fit_edges = copy.deepcopy(
                    np.concatenate(
                        (
                            bin_edges[:-1][mask_edges],
                            # Right edge of the last bin
                            np.array(
                                [bin_edges[1:][mask_edges][-1]],
                                dtype=np.float64
                            ),
                        )
                    )
                )

                if len(fit_counts) < 2:
                    raise cuex.NotEnoughFitSamples(
                        htype.generate_exception_message(
                            "SiPMMeas.piecewise_gaussian_fits",
                            68958,
                            extra_info=f"The selected histogram interval does not "
                            f"contain enough bins ({len(fit_counts)}) to fit "
                            f"2 parameters."
                        )
                    )
                
                cumulative_model = lambda edges, mean_, std_: \
                    spsta.norm.cdf(
                        edges,
                        loc=mean_,
                        scale=std_
                    )

                # Compute the normalization factor (the area) of the
                # histogram in the selected interval, which will be used
                # to scale the fitted normalized gaussian, so that it
                # matches the scale of the histogram. It does not matter
                # if this is done before of after scaling the edges, but
                # the resulting scaling factors are computed using the
                # scaled or re-scaled result of the fit sigma parameter
                # depending on the order of these two operations. I.e.
                # if we compute the area before scaling the edges, then the
                # the scaling factor is computed using the resulting sigma
                # which has been re-scaled back to the original scale.
                aux_area = np.sum(
                    fit_counts * np.diff(fit_edges)
                )

                # Transform the edges for the optimization process
                fit_edges = \
                    (fit_edges - horizontal_axis_shift_for_fit) \
                    / horizontal_axis_scaling_for_fit

                # Define the cost function
                cost = BinnedNLL(
                    fit_counts,
                    fit_edges,
                    cumulative_model
                )

                param_names = (
                    "mean",
                    "std",
                )

                m = Minuit(
                    cost,
                    *p0,
                    name=param_names
                )

                # Set the limits
                for j in range(len(p0)):
                    lower = fit_parameters_bounds_[0][j]
                    upper = fit_parameters_bounds_[1][j]

                    m.limits[param_names[j]] = (
                        None if np.isneginf(lower) else lower,
                        None if np.isposinf(upper) else upper
                    )

                # Run the optimization
                m.migrad()

                # Compute the covariance matrix at the minimum
                m.hesse()

                if not m.valid:
                    raise RuntimeError(
                        "In function SiPMMeas.piecewise_gaussian_fits(): "
                        "iminuit failed to converge to a valid minimum."
                    )

                aux_popt = np.array(m.values)

                if m.covariance is None:
                    raise RuntimeError(
                        "In function SiPMMeas.piecewise_gaussian_fits(): "
                        "iminuit could not compute a covariance matrix."
                    )

                aux_pcov = np.array(m.covariance)

            else:

                mask = (x >= x_min) & (x <= x_max)
                fit_centers, fit_counts = x[mask], y[mask]

                if len(fit_centers) < (2 if not fWithScaling else 3):
                    raise cuex.NotEnoughFitSamples(
                        htype.generate_exception_message(
                            "SiPMMeas.piecewise_gaussian_fits",
                            83500,
                            extra_info=f"The fit_centers array does not contain "
                            f"samples enough ({len(fit_centers)}) to fit a "
                            f"gaussian with {2 if not fWithScaling else 3} free "
                            "parameters.",
                        )
                    )
                
                # Scale the centers for the optimization process
                fit_centers = \
                    (fit_centers - horizontal_axis_shift_for_fit) \
                    / horizontal_axis_scaling_for_fit

                aux_popt, aux_pcov = spopt.curve_fit(
                    gaussian,
                    fit_centers,
                    fit_counts,
                    p0=p0,
                    bounds=fit_parameters_bounds_,
                    method='trf',
                    sigma=np.array([
                        math.sqrt(val) if val >= 1 else 1.0
                        for val in fit_counts
                    ]),
                    absolute_sigma=True
                )

            # Reescale back the optimal parameters
            aux_popt[0] = (aux_popt[0] * horizontal_axis_scaling_for_fit) \
                + horizontal_axis_shift_for_fit

            aux_popt[1] *= horizontal_axis_scaling_for_fit

            # Reescaling of the covariance matrix
            # mean variance, std variance and mean-std covariance
            aux_pcov[0:2, 0:2] *= (horizontal_axis_scaling_for_fit ** 2)

            if optimize_poisson_likelihood:
                # In this case, the scaling factor was not a fit parameter,
                # so we need to compute it and add it to the list of optimal
                # parameters. Since BinnedNLL fits a PDF (automatically
                # normalizes the given counts), scale the fitted normalized
                # gaussian by the area of the histogram in the selected
                # interval. Also, add the inverse of the normalization factor
                # of the normal distribution given by spsta.norm.pdf() to
                # compensate it, so that the computed optimal scaling is
                # understood as an scaling factor which is directly applied
                # to exp(...), and not the whole normalized gaussian.
                aux_popt = np.append(
                    aux_popt,
                    # The area of the histogram in the selected interval gives
                    # the scaling factor for the normalized gaussian fit
                    aux_area / (aux_popt[1] * math.sqrt(2.*spcon.pi)),
                )
            else:
                # If not optimize_poisson_likelihood, then the covariance
                # matrix is 3x3, and we should re-escale the covariances
                # between the mean and the scaling factor, and between the
                # std and the scaling factor
                for j in (0, 1):
                    aux_pcov[j, 2] *= horizontal_axis_scaling_for_fit
                    aux_pcov[2, j] *= horizontal_axis_scaling_for_fit

            popt.append(aux_popt)
            pcov.append(aux_pcov)

        def fit_functions_sum(x):
            y = 0.

            for i in range(len(mean_seeds)):
                y += gaussian(
                    x,
                    *popt[i]
                )

            return y

        return popt, pcov, fit_functions_sum

    @staticmethod
    def sum_of_correlated_gaussians(
        x,
        gaussians_number,
        center_0,
        mean_increment,
        std_0,
        std_increment,
        *scaling_factors
    ):
        """This static method evaluates a sum of N Gaussian functions at
        the given x values. The Gaussians are 'correlated' in the sense that
        their means and standard deviations follow a parametric relationship:

            f(x) = sum_{i=0}^{N-1} S_i * gauss(x, mu_i, sigma_i)

        where:
            mu_i = center_0 + (i * mean_increment)
            sigma_i = sqrt(std_0^2 + (i * std_increment^2))

        This function gets the following arguments:

        - x (scalar float, or unidimensional float numpy array): The x
          values at which to evaluate the function.
        - gaussians_number (positive integer): The number of Gaussian
          functions to sum (N in the formula above).
        - center_0 (scalar float): The mean of the first (i=0) Gaussian.
        - mean_increment (scalar float): The increment in the mean for
          each successive Gaussian. For gain analysis, this parameter
          represents the gain (charge per photoelectron).
        - std_0 (scalar float): The standard deviation of the first
          (i=0) Gaussian. It must be positive.
        - std_increment (scalar float): A parameter controlling how the
          standard deviation grows for successive Gaussians. It must be
          semipositive (>=0). The i-th Gaussian has standard deviation
          sqrt(std_0^2 + i * std_increment^2).
        - *scaling_factors (floats): The scaling factors S_i for each
          Gaussian. The number of scaling factors must match
          gaussians_number.

        This method returns the evaluated sum of Gaussians at x.
        """

        # This function may be evaluated lots of times during the fit,
        # so we are not performing any checks here. P.e. we are trusting
        # that len(scaling_factors) matches gaussians_number.

        result = 0.0
        for i in range(gaussians_number):
            mu_i = center_0 + (i * mean_increment)

            # sigma_i = sqrt(std_0^2 + i * std_increment^2)
            sigma_i = math.sqrt(
                (std_0 * std_0) + \
                    (i * std_increment * std_increment)
            )

            # Compensate the normalization factor of the
            # normal distribution given by spsta.norm.pdf()
            result += \
                scaling_factors[i] * \
                    sigma_i * math.sqrt(2. * spcon.pi) * \
                        spsta.norm.pdf(
                            x,
                            loc=mu_i,
                            scale=sigma_i
                        )
 
        return result

    @staticmethod
    def correlated_gaussians_fit(
        x,
        y,
        mean_seeds,
        std_seeds,
        scaling_seeds,
        bin_edges=None,
        fit_parameters_bounds=None,
        std_no=3.0,
        optimize_poisson_likelihood=False
    ):
        """This static method fits a sum of correlated Gaussian functions
        to the given x-y data. The Gaussians are 'correlated' in the sense
        that their means and standard deviations follow a parametric
        relationship (see SiPMMeas.sum_of_correlated_gaussians() docstring).

        This method gets the following mandatory positional arguments:

        - x (unidimensional float numpy array): The x values of the data
          to fit.
        - y (unidimensional numpy array, int or float): The y values of
          the data to fit. Its length must match that of x.
        - mean_seeds (list of floats): mean_seeds[i] is the seed for the
          mean of the i-th Gaussian. These are used to compute the initial
          seeds for center_0 and mean_increment.
        - std_seeds (list of floats): std_seeds[i] is the seed for the
          standard deviation of the i-th Gaussian. These are used to
          compute the initial seeds for std_0 and std_increment.
        - scaling_seeds (list of floats/ints): scaling_seeds[i] is the
          seed for the scaling factor of the i-th Gaussian.

        This method gets the following optional keyword arguments:

        - bin_edges (None or unidimensional float numpy array): It is
          ignored if optimize_poisson_likelihood is False. Otherwise, it
          must define the histogram bin edges used to produce y, so
          len(bin_edges)=len(y)+1.
        - fit_parameters_bounds (None or 2-tuple of array-like): Bounds
          for the fit parameters. The parameters are ordered as:
          [center_0, mean_increment, std_0, std_increment, S_0, S_1, ...],
          where S_i is the scaling factor for the i-th Gaussian.
          If None, default bounds are used: center_0, std_0 and
          std_increment are unbounded. mean_increment and the scaling
          factors have lower bound 0.
        - std_no (scalar float): It must be positive (>0.0). This number
          determines the x-range of the input data which is used for the
          fit. Namely, the x-y points used are those which fall within
          [mean_seeds[0] - std_no*std_seeds[0],
           mean_seeds[-1] + std_no*std_seeds[-1]].
        - optimize_poisson_likelihood (scalar boolean): If False (default),
          scipy.optimize.curve_fit() is used. If True, a Poisson likelihood
          maximization is performed via iminuit.BinnedNLL.

        This method returns a tuple (popt, pcov, fit_function) where:

        - popt (numpy array): The optimal values for the fit parameters,
          ordered as [center_0, mean_increment, std_0, std_increment,
          S_0, S_1, ...].
        - pcov (2D numpy array): The covariance matrix of the fit
          parameters.
        - fit_function (callable): A function that evaluates the fitted
          sum of correlated Gaussians at any x value.

        The gain is directly available as popt[1] (mean_increment), and
        its error is sqrt(pcov[1, 1]).
        """

        # Type checks
        htype.check_type(
            x,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.correlated_gaussians_fit", 48249
            ),
        )
        htype.check_type(
            y,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.correlated_gaussians_fit", 20371
            ),
        )
        if np.ndim(x) != 1 or np.ndim(y) != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.correlated_gaussians_fit", 46173
                )
            )
        if x.dtype != np.float64:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.correlated_gaussians_fit", 12312
                )
            )
        if len(x) != len(y):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.correlated_gaussians_fit", 94090
                )
            )

        htype.check_type(
            mean_seeds,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.correlated_gaussians_fit", 38294
            ),
        )
        htype.check_type(
            std_seeds,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.correlated_gaussians_fit", 88298
            ),
        )
        htype.check_type(
            scaling_seeds,
            list,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.correlated_gaussians_fit", 19093
            ),
        )

        if len(mean_seeds) != len(std_seeds) or len(mean_seeds) != len(scaling_seeds):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.correlated_gaussians_fit",
                    83749,
                    extra_info="mean_seeds, std_seeds and "\
                    "scaling_seeds must have the same length."
                )
            )

        if len(mean_seeds) < 2:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.correlated_gaussians_fit",
                    83750,
                    extra_info="At least 2 peaks are required "\
                    "for correlated Gaussian fit."
                )
            )

        htype.check_type(
            std_no,
            float,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.correlated_gaussians_fit", 38113
            ),
        )
        if std_no <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SiPMMeas.correlated_gaussians_fit", 48222
                )
            )
        htype.check_type(
            optimize_poisson_likelihood,
            bool,
            exception_message=htype.generate_exception_message(
                "SiPMMeas.correlated_gaussians_fit", 48223
            ),
        )

        if optimize_poisson_likelihood:
            htype.check_type(
                bin_edges,
                np.ndarray,
                exception_message=htype.generate_exception_message(
                    "SiPMMeas.correlated_gaussians_fit", 48224
                ),
            )
            if np.ndim(bin_edges) != 1 or len(bin_edges) != (len(y) + 1):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SiPMMeas.correlated_gaussians_fit", 48225
                    )
                )
            if bin_edges.dtype != np.float64:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SiPMMeas.correlated_gaussians_fit", 48226
                    )
                )

        gaussians_number = len(mean_seeds)

        # Compute initial seeds for the correlated
        # parameters
        center_0_seed = mean_seeds[0]

        # Compute a seed for mean_increment as the
        # average of differences between consecutive
        # means
        mean_increment_seed = np.mean(np.diff(mean_seeds))

        std_0_seed = std_seeds[0]

        # Since by definition
        #
        #   sigma_i^2 = sigma_0^2 + i * delta_sigma^2,
        #
        # for every i > 0 we can get one estimation of
        # delta_sigma as sqrt((sigma_i^2 - sigma_0^2) / i).
        # Hence, we can estimate delta_sigma as the average
        # of these estimates.
        std_increment_squared_estimates = []

        sigma_0_sq = std_seeds[0] ** 2
        for i in range(1, len(std_seeds)):
            sigma_i_sq = std_seeds[i] ** 2

            if sigma_i_sq > sigma_0_sq:
                std_increment_squared_estimates.append(
                    (sigma_i_sq - sigma_0_sq) / i
                )
            else:
                std_increment_squared_estimates.append(0.0)

        if std_increment_squared_estimates: # i.e. if len(std_seeds) >= 2
            std_increment_seed = math.sqrt(
                np.mean(std_increment_squared_estimates)
            )
        else:
            std_increment_seed = 0.0

        # Build the initial parameter vector:
        # [center_0, mean_increment, std_0, std_increment, S_0, S_1, ...]
        p0 = [
            center_0_seed,
            mean_increment_seed,
            std_0_seed,
            std_increment_seed
        ]
        p0.extend(scaling_seeds)

        # Build bounds
        if fit_parameters_bounds is None:
            # Similarly to SiPMMeas.piecewise_gaussian_fits(), we are not
            # setting a lower bound of 0 for std_0 and std_increment, because
            # we have seen in previous executions of the gain analysis that
            # std (or std_increment) may be of the order of 1e-10. For such
            # an small value, setting a lower bound of 0 (close to 1e-10) may
            # make the 'trf' method unstable, p.e. causing a reflection in
            # such 0 boundary.

            lower_bounds = [
                -np.inf, # center_0
                0.0, # mean_increment
                -np.inf, # std_0
                -np.inf # std_increment
            ]
            upper_bounds = [
                np.inf, # center_0
                np.inf, # mean_increment
                np.inf, # std_0
                np.inf # std_increment
            ]

            # Bound scaling factors to be non-negative
            for _ in range(gaussians_number):
                lower_bounds.append(0.0)
                upper_bounds.append(np.inf)

            fit_parameters_bounds_ = (
                np.array(lower_bounds),
                np.array(upper_bounds)
            )

        else:
            fit_parameters_bounds_ = fit_parameters_bounds

        # Clip seeds to bounds if necessary
        for i in range(len(p0)):
            if p0[i] < fit_parameters_bounds_[0][i]:
                p0[i] = fit_parameters_bounds_[0][i]
            elif p0[i] > fit_parameters_bounds_[1][i]:
                p0[i] = fit_parameters_bounds_[1][i]

        # Scale and shift the input data to make the
        # first gaussian (approximatedly) distributed as
        # a gaussian with mean ~0.0 and std ~1.0,
        # avoiding extreme orders of magnitude and
        # making the fit more numerically stable. First,
        # define the parameters of the transformation.
        # At some point below, we will apply it.
        horizontal_axis_shift_for_fit = center_0_seed
        horizontal_axis_scaling_for_fit = std_0_seed

        # Re-set the seeds accordingly
        # center_0
        p0[0] = 0.
        # mean_increment
        p0[1] = p0[1] / horizontal_axis_scaling_for_fit
        # std_0
        p0[2] = 1.
        # std_increment
        p0[3] = p0[3] / horizontal_axis_scaling_for_fit

        # Note 1: This transformation preserves the
        # clipped values, since we will transform the
        # bounds as well
        # Note 2: The seeds and bounds are only used
        # for the optimization process, i.e. they are
        # not returned as part of the output of this
        # function, so they don't need to be rescaled
        # back at any point later
        for i in (0, 1):
            # Scale the center_0 bounds
            fit_parameters_bounds_[i][0] = \
                (fit_parameters_bounds_[i][0] - horizontal_axis_shift_for_fit) \
                / horizontal_axis_scaling_for_fit
            
            # Scale the mean_increment, std_0 and std_increment bounds
            for j in (1, 2, 3):
                fit_parameters_bounds_[i][j] /= horizontal_axis_scaling_for_fit

        # Select the data range for fitting
        x_min = mean_seeds[0] - (std_no * std_seeds[0])
        x_max = mean_seeds[-1] + (std_no * std_seeds[-1])

        if optimize_poisson_likelihood:

            # In this case, since we are fitting a PDF to a
            # normalized histogram via BinnedNLL, we should
            # drop (not fit) one of the scaling factors to
            # avoid a degenerate fit. I.e. since the PDF is
            # normalized, the scaling factors are exactly
            # correlated by the normalization constraint.
            # Although the result of the fitting parameters
            # may still be correct, the covariance matrix
            # would probably overestimate the errors for
            # the scaling factors. Particularly, we will
            # drop the last scaling factor, by fixing it to
            # 1.0 in the definition of the cumulative_model
            # function, and by dropping the last entry of
            # the seeds and the bounds.

            # N.B: The case of gaussians_number scaling
            # factors was already tried and, although the
            # optimal parameters were correct, the relative
            # error for the scaling factors was of the order
            # of 200%.

            p0 = p0[:-1]
            fit_parameters_bounds_ = (
                fit_parameters_bounds_[0][:-1],
                fit_parameters_bounds_[1][:-1]
            )

            mask_edges = (
                (bin_edges[:-1] >= x_min)
                & (bin_edges[1:] <= x_max)
            )
            if not np.any(mask_edges):
                raise cuex.NotEnoughFitSamples(
                    htype.generate_exception_message(
                        "SiPMMeas.correlated_gaussians_fit",
                        48228,
                        extra_info="The selected histogram interval does not "
                        "contain bins for the requested fit."
                    )
                )

            fit_counts = y[mask_edges]
            fit_edges = copy.deepcopy(
                np.concatenate(
                    (
                        bin_edges[:-1][mask_edges],
                        np.array(
                            [bin_edges[1:][mask_edges][-1]],
                            dtype=np.float64),
                    )
                )
            )

            if len(fit_counts) < len(p0):
                raise cuex.NotEnoughFitSamples(
                    htype.generate_exception_message(
                        "SiPMMeas.correlated_gaussians_fit",
                        48227,
                        extra_info=f"The selected histogram interval does not "
                        f"contain enough bins ({len(fit_counts)}) to fit "
                        f"{len(p0)} parameters."
                    )
                )
            
            # Define the (normalized) cumulative model for
            # iminuit.cost.BinnedNLL. This definition is, indeed,
            # a CDF, since it comes from integrating a normalized
            # sum of gaussian functions. I.e. the integral of such
            # a sum evaluates to one. Particularly, the used PDF is
            #
            #   pdf(x) = \frac{
            #       gauss(x, \mu _{N-1}, \sigma _{N-1}) + \sum _{i=0} ^{N-2} s_i\cdot gauss(x,\mu _i, \sigma _i)
            #   }{
            #       \sqrt{2 \pi}\cdot (\sigma _{N-1}+\sum_{i=0}^{N-2} s_i \cdot \sigma _i)
            #   }
            #
            # where gauss(x, \mu, \sigma) = exp( -(x-\mu)^2 / (2\sigma^2)).
            # You can check that the integral of this PDF is, indeed, 1.

            def cumulative_model(
                    edges,
                    center_0,
                    mean_increment,
                    std_0,
                    std_increment,
                    *scales
                ):

                result = np.zeros_like(
                    edges,
                    dtype=np.float64
                )

                # Assume that len(scales) == gaussians_number - 1,
                # so add the fixed one. In this way, we shall not
                # separate the last iteration of the loop below,
                # making the code more compact.
                scales = np.array(
                    scales,
                    dtype=np.float64
                )

                scales = np.append(
                    scales,
                    1.
                )

                normalization = 0.

                for idx in range(gaussians_number):
                    mu_i = center_0 + (idx * mean_increment)
                    sigma_i = math.sqrt(
                        (std_0 * std_0) + (idx * std_increment * std_increment)
                    )
                    result += scales[idx] * sigma_i * \
                        spsta.norm.cdf(
                            edges,
                            loc=mu_i,
                            scale=sigma_i
                        )   
                    normalization += (scales[idx] * sigma_i)

                return result / normalization

            # Compute the normalization factor (the area) of the
            # histogram in the selected interval, which will be used
            # to scale the fit function as a whole, so that its
            # integral eventually matches the integral of the
            # histogram in the selected interval. For clarification,
            # the workflow we are following is:
            #
            #   1)  Define (`def cumulative_model()`) a CDF which
            #       comes from a normalized a sum of N correlated
            #       normal distributions with N-1 free scaling factors.
            #       This sum is a properly normalized PDF.
            # 
            #   2)  Fit it to a part of the histogram via BinnedNLL,
            #       which internally normalizes the given counts, so
            #       that the fit is performed with a normalized PDF to
            #       a normalized histogram.
            # 
            #   3)  Scale the fitted normalized PDF, as a whole, by
            #       the area of the fit histogram, so that the integral
            #       of the fitted function matches the integral of the
            #       histogram in the selected interval. This scaling
            #       is done further below in the code.
            #  
            # To this end, we need to compute such an area, and it does
            # not matter if this is done before or after scaling the
            # edges, but the resulting scaling factors are eventually
            # re-scaled using the scaled or re-scaled result of the fit
            # sigmas depending on the order of these two operations. I.e.
            # if we compute the area before scaling the edges, then the
            # the scaling factor is computed using the resulting sigmas
            # which has been re-scaled back to the original scale.

            aux_area = np.sum(
                fit_counts * np.diff(fit_edges)
            )
            
            # Transform the edges for the optimization process
            fit_edges = \
                (fit_edges - horizontal_axis_shift_for_fit) \
                / horizontal_axis_scaling_for_fit

            # Define the cost function
            cost = BinnedNLL(
                fit_counts,
                fit_edges,
                cumulative_model
            )

            param_names = (
                "center_0",
                "mean_increment",
                "std_0",
                "std_increment",
            # Don't add a name for the last scaling
            # factor, since it is not a fit parameter
            ) + tuple([f"S_{i}" for i in range(gaussians_number - 1)])

            m = Minuit(
                cost,
                *p0,
                name=param_names
            )

            # Set the limits
            for i in range(len(p0)):
                lower = fit_parameters_bounds_[0][i]
                upper = fit_parameters_bounds_[1][i]

                m.limits[param_names[i]] = (
                    None if np.isneginf(lower) else lower,
                    None if np.isposinf(upper) else upper
                )

            # Run the optimization
            m.migrad()

            # Compute the covariance matrix at the minimum
            m.hesse()

            if not m.valid:
                raise RuntimeError(
                    "In function SiPMMeas.correlated_gaussians_fit(): "
                    "iminuit failed to converge to a valid minimum."
                )

            popt = np.array(m.values)

            if m.covariance is None:
                raise RuntimeError(
                    "In function SiPMMeas.correlated_gaussians_fit(): "
                    "iminuit could not compute a covariance matrix."
                )

            pcov = np.array(m.covariance)

        else:
            # Define the fit function for scipy.optmize.curve_fit
            def fit_model(
                    x_data,
                    center_0,
                    mean_increment,
                    std_0,
                    std_increment,
                    *scales
                ):

                return SiPMMeas.sum_of_correlated_gaussians(
                    x_data,
                    gaussians_number,
                    center_0,
                    mean_increment,
                    std_0,
                    std_increment,
                    *scales
                )
            
            mask = (x >= x_min) & (x <= x_max)
            fit_centers, fit_counts = x[mask], y[mask]

            if len(fit_centers) < len(p0):
                raise cuex.NotEnoughFitSamples(
                    htype.generate_exception_message(
                        "SiPMMeas.correlated_gaussians_fit",
                        83753,
                        extra_info=f"The fit_centers array does not contain samples "
                        f"enough ({len(fit_centers)}) to fit {len(p0)} parameters."
                    )
                )
            
            # Scale the centers for the optimization process
            fit_centers = \
                (fit_centers - horizontal_axis_shift_for_fit) \
                / horizontal_axis_scaling_for_fit

            popt, pcov = spopt.curve_fit(
                fit_model,
                fit_centers,
                fit_counts,
                p0=p0,
                bounds=fit_parameters_bounds_,
                method='trf',
                sigma=np.array([
                    math.sqrt(val) if val >= 1 else 1.0
                    for val in fit_counts
                ]),
                absolute_sigma=True
            )

        # Reescale back the optimal parameters
        # center_0
        popt[0] = (popt[0] * horizontal_axis_scaling_for_fit) \
            + horizontal_axis_shift_for_fit
        
        # mean_increment, std_0 and std_increment
        for i in (1, 2, 3):
            popt[i] *= horizontal_axis_scaling_for_fit

        # Reescale the covariance matrix
        # The part of the matrix which corresponds to the mean
        # and std parameters (variances and covariances) is
        # rescaled by the square of the horizontal_axis_scaling_for_fit,
        # since each one of these parameters are scaled
        # by horizontal_axis_scaling_for_fit
        pcov[0:4, 0:4] *= (horizontal_axis_scaling_for_fit ** 2)

        # The off-diagonal matrices which correspond to the
        # covariance between the mean/std parameters and
        # the scaling factors are rescaled only by
        # horizontal_axis_scaling_for_fit, since the scaling
        # factors were not scaled for the optimization process
        pcov[0:4, 4:] *= horizontal_axis_scaling_for_fit
        pcov[4:, 0:4] *= horizontal_axis_scaling_for_fit

        if optimize_poisson_likelihood:

            # In this case, note that the last scaling factor was not
            # a fit parameter, but fixed to 1.0. We will artificially
            # append it to the list of optimal parameters, before the
            # normalization rescaling
            popt = np.append(
                popt,
                1.
            )

            # Now we need to rescale the scaling factors so that the
            # integral of the fit function matches the integral of the
            # histogram in the selected interval. Since the integral
            # of the fit pdf(x) is one, the integral of k*pdf(x) is k.
            # We also need to take into account the fact that, although
            # the fit scaling factors are defined in a normalized pdf,
            # they are later used as scaling factors for a non-normalized
            # sum of gaussians, so we need to identify the proper
            # rescaling by comparing
            #
            # pdf(x) -> k*pdf(x) = \sum _{i=0}^{N-1} s'_i * exp(...)
            #
            # where pdf(x) is the normalized sum of correlated gaussians
            # which depend on the computed s_i, and whose CDF is given by
            # the cumulative_model() defined above. Using the definition
            # of pdf(x) above and solving for s'_i, we get
            #
            #  s'_i = (k * s_i) / ((sqrt(2 * pi) * \sum _{j=0}^{N-1} s_j * sigma_j))
            #
            # where s_{N-1} = 1.0, and k is the `aux_area` computed above.
              
            reconstructed_sigmas = np.array([
                math.sqrt((popt[2] * popt[2]) + (i * popt[3] * popt[3]))
                for i in range(gaussians_number)
            ])

            aux_norm = math.sqrt(2. * spcon.pi) * \
                    np.sum(popt[4:] * reconstructed_sigmas)

            aux_rescaling_factor = aux_area / aux_norm

            popt[4:] = aux_rescaling_factor * popt[4:]

            # Also note that, since k (aux_area) was computed before
            # re-scaling fit_edges, the sigmas which we are now using
            # are those which have been re-scaled back to the original
            # scale.

            # The scaling-factors part of the covariance matrix
            # is rescaled by the square of aux_rescaling_factor
            pcov[4:, 4:] *= (aux_rescaling_factor ** 2)
            
            # The off-diagonal matrices which correspond to the
            # covariance between the mean/std parameters and
            # the scaling factors are rescaled by just
            # aux_rescaling_factor
            pcov[0:4, 4:] *= aux_rescaling_factor
            pcov[4:, 0:4] *= aux_rescaling_factor

        # Build the fit function for return
        def fit_function(x_eval):
            return SiPMMeas.sum_of_correlated_gaussians(
                x_eval,
                gaussians_number,
                popt[0],  # center_0
                popt[1],  # mean_increment
                popt[2],  # std_0
                popt[3],  # std_increment
                *popt[4:]  # scaling_factors
            )

        return popt, pcov, fit_function

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
        self.__sampling_ns will be computed from the value given to
        "time_resolution" in the file given to wvfset_json_filepath.

        The second one, say RKD2, concerns the WaveformSet.from_files
        parameters, and is populated out of the file whose path is
        given to the wvfset_json_filepath key of RKD1. It has the following
        potential keys:

        "wvf_filepath", "time_resolution", "points_per_wvf",
        "wvfs_to_read", "timestamp_filepath", "delta_t_wf",
        "packing_version",  "set_name", "creation_dt_offset_min"
        and "wvf_extra_info".

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
        the class initializer. Particularly, the "time_resolution"
        parameter is converted to seconds using the time unit
        information which is read from the file whose path is given by
        the "wvf_extra_info" entry of RKD2. "wvfset_json_filepath" is
        also an exception, since it is used to populate RKD2, and it's
        deleted afterwards.
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
            "packing_version": int,
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

        if "wvf_extra_info" not in RKD2.keys():
            raise cuex.NoAvailableData(
                htype.generate_exception_message(
                    "SiPMMeas.from_json_file",
                    29564,
                    extra_info="No filepath to the the waveform-extra-info"
                    " file was provided.",
                )
            )
        else:
            aux, _ = DataPreprocessor.try_grabbing_from_json(
                # Assuming that the 'time_unit' entry of the wvf_extra_info
                # file is a list which contains one string, i.e. the time unit
                {'time_unit': list},
                RKD2['wvf_extra_info'],
                verbose=False
            )

            # NumpyDataPreprocessor.interpret_time_unit_in_seconds() takes
            # care of type-checking aux[0]
            time_unit_in_s = NumpyDataPreprocessor.interpret_time_unit_in_seconds(
                aux[0]
            )

            time_resolution_s = RKD2["time_resolution"] * time_unit_in_s
            del RKD2["time_resolution"]

        # Unless otherwise stated, all of the time values are given in seconds.
        # However, as its name indicates, sampling_ns is expressed in nanoseconds.
        # Thus, here I am converting time_resolution_s, which is given in seconds, to nanoseconds.
        RKD1["sampling_ns"] = 1e9 * time_resolution_s

        input_filepath = RKD2["wvf_filepath"]
        del RKD2["wvf_filepath"]

        return cls(input_filepath, time_resolution_s, **RKD1, **RKD2)

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
                    "In function SiPMMeas.get_value_from_dict(): "
                    f"The key {key} does not exist in the given dictionary."
                )

    # Despite not following the same conventions as the rest of the class,
    # for the methods written below this point I am using the typing module
    # and not calling the htype.check_type(). The reason for this is that
    # at some point I plan to deprecate htype.check_type() and replace it
    # with the usage of the typing module. This will make the code more
    # efficient, standard and readable.
    def rebin(
            self,
            group: int,
            verbose: bool = False
        ) -> None:
        """This function gets the following positional arguments:

        - group (integer): It must be positive and smaller or equal to
        half of the length of every waveform in the __waveforms attribute
        of this SiPMMeas object. The second condition grants that, for
        every Waveform object, there is at least two entries in the
        resulting Waveform.
        - verbose (bool): Whether to print functioning related messages.

        This function rebins every Waveform object in the __waveforms
        attribute of this SiPMMeas object. To do so, this method calls the
        WaveformSet.rebin() method of the __waveforms WaveformSet. For
        more information on the rebinning process, check the documentation
        of such WaveformSet method. Note that this method modifies (inplace)
        the values of the NPoints, Time and Signal attributes of every
        considered Waveform object. Also, the __sampling_ns attribute of
        this SiPMMeas object is updated to the new sampling time of the
        rebinned waveforms."""

        self.__waveforms.rebin(group, verbose)
        self.__sampling_ns *= group

        return

    def get_title(
            self,
            abbreviate: bool = False
        ):
        """This method gets the following keyword argument:

        - abbreviate (bool): If it is True (resp. False), then
        the output string is abbreviated (resp. not abbreviated).

        This method returns an string which could serve as a title
        for this SiPMMeas object. Such title contains information
        on the ElectronicBoardSocket, StripID, SiPMLocation,
        ThermalCycle and Date attributes of this SiPMMeas object."""

        if not abbreviate:
            return f"Socket {self.ElectronicBoardSocket}, SiPM {self.StripID}-{self.SiPMLocation}, T.C. {self.ThermalCycle}, {self.Date.strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            return f"S. {self.ElectronicBoardSocket}, {self.StripID}-{self.SiPMLocation}, T.C. {self.ThermalCycle}, {self.Date.strftime('%Y-%m-%d')}"