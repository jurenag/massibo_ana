import os
import json
import math
import numpy as np
import matplotlib
from scipy import stats as spsta

import massibo_ana.utils.htype as htype
import massibo_ana.utils.custom_exceptions as cuex
from massibo_ana.custom_types.RigidKeyDictionary import RigidKeyDictionary
from massibo_ana.core.Waveform import Waveform
from massibo_ana.core.SiPMMeas import SiPMMeas
from massibo_ana.preprocess.DataPreprocessor import DataPreprocessor
from massibo_ana.preprocess.NumpyDataPreprocessor import NumpyDataPreprocessor


class GainMeas(SiPMMeas):

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
        LED_voltage_V=None,
        LED_frequency_kHz=None,
        LED_pulse_shape=None,
        LED_high_width_ns=None,
        verbose=True,
        **kwargs,
    ):
        """This class, which derives from SiPMMeas class, aims to implement a SiPM gain
        measurement. In line with Waveform, WaveformSet and SiPMMeas classes, the assumed time unit
        is the second, unless otherwise indicated. This initializer gets the following positional
        argument:

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
        - LED_voltage_V (positive float): Amplitude of the voltage (pulsed) signal that feeds the LED,
        in volts. It is loaded into the object-attribute self.__LED_voltage_V.
        - LED_frequency_kHz (positive float): Frequency of the pulsed signal that feeds the LED,
        in kilohertzs . It is loaded into the object-attribute self.__LED_frequency_kHz.
        - LED_pulse_shape (string): Shape of the pulsed signal that feeds the LED. It is loaded
        into the object-attribute self.__LED_pulse_shape.
        - LED_high_width_ns (positive float): Width of the high level of the pulsed signal that feeds
        the LED, in nanoseconds. It is loaded into the object-attribute self.__LED_high_width_ns.
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

        self.__LED_voltage_V = None
        if LED_voltage_V is not None:
            htype.check_type(
                LED_voltage_V,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "GainMeas.__init__", 52239
                ),
            )
            if LED_voltage_V <= 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("GainMeas.__init__", 43748)
                )
            self.__LED_voltage_V = LED_voltage_V

        self.__LED_frequency_kHz = None
        if LED_frequency_kHz is not None:
            htype.check_type(
                LED_frequency_kHz,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "GainMeas.__init__", 21085
                ),
            )
            if LED_frequency_kHz <= 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("GainMeas.__init__", 99855)
                )
            self.__LED_frequency_kHz = LED_frequency_kHz

        self.__LED_pulse_shape = None
        if LED_pulse_shape is not None:
            htype.check_type(
                LED_pulse_shape,
                str,
                exception_message=htype.generate_exception_message(
                    "GainMeas.__init__", 28436
                ),
            )
            self.__LED_pulse_shape = LED_pulse_shape

        self.__LED_high_width_ns = None
        if LED_high_width_ns is not None:
            htype.check_type(
                LED_high_width_ns,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "GainMeas.__init__", 53321
                ),
            )
            if LED_high_width_ns <= 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("GainMeas.__init__", 57786)
                )
            self.__LED_high_width_ns = LED_high_width_ns

        # The rest of the arguments are handled by the base class initializer

        # self.fit_chi2 = None
        # self.fit_parameters = None
        # self.fit_parameters_erros = None
        # self.gain = None
        # self.gain_error = None

        self.__charge_entries = None

        # N.B.: acquisition_time_min does not appear in
        # this list because it comes from the WaveformSet
        # core data. It is computed by WaveformSet.from_files().
        super().__init__(
            *args,
            delivery_no=delivery_no,
            set_no=set_no,
            meas_no=meas_no,
            strip_ID=strip_ID,
            meas_ID=meas_ID,
            date=date,
            location=location,
            operator=operator,
            setup_ID=setup_ID,
            system_characteristics=system_characteristics,
            thermal_cycle=thermal_cycle,
            electronic_board_number=electronic_board_number,
            electronic_board_location=electronic_board_location,
            electronic_board_socket=electronic_board_socket,
            sipm_location=sipm_location,
            sampling_ns=sampling_ns,
            cover_type=cover_type,
            operation_voltage_V=operation_voltage_V,
            overvoltage_V=overvoltage_V,
            PDE=PDE,
            status=status,
            verbose=verbose,
            **kwargs,
        )

    @property
    def LEDVoltage_V(self):
        return self.__LED_voltage_V

    @property
    def LEDFrequency_kHz(self):
        return self.__LED_frequency_kHz

    @property
    def LEDPulseShape(self):
        return self.__LED_pulse_shape

    @property
    def LEDHighWidth_ns(self):
        return self.__LED_high_width_ns

    @property
    def ChargeEntries(self):
        return self.__charge_entries

    def integrate(
        self,
        input_resistance_in_ohms=50.0,
        system_amplification_factor=1.0,
        integration_lower_lim=None,
        integration_upper_lim=None,
    ):
        """This method analyzes the underlying WaveformSet: it integrates every
        waveform in such set, and populates self.__charge_entries with the results
        gotten from such integrals, divided by the input resistance of the DAQ device
        which was used to measure the waveforms and the amplification factor of the
        overall readout system. I.e. each charge entry is computed as the integral
        of one waveform divided by the input resistance and the system amplification
        factor. Integration is performed via the trapezoid method. Namely, this method
        calls self.Waveforms.integrate(), which in turn calls numpy.trapz. When this
        method is called, it overrides the current value of self.ChargeEntries. This
        method gets the following optional keyword arguments:

        - input_resistance_in_ohms (float): Electrical resistance in ohms. It is
        set to 50 ohms by default, which is a pretty standard value for oscilloscopes

        - system_amplification_factor (float): It must be positive (>0.0).

        - integration_lower_lim (float): Inclusive upper bound to the lower limit
        of the integration. In order to apply the trapezoid rule, the minimum point
        (i.e. the point whose x-value is the minimum) which is used is the point
        with the maximum x-value which still meets x<=integration_lower_lim.
        integration_lower_lim must belong to [min, max], where min (resp. max)
        is the smallest (biggest) x-value for which the integrated waveform is
        defined. If this parameter is not set or integration_lower_lim<min, then
        integration_lower_lim is set to min. If integration_lower_lim>max, then
        the integration lower limit is set to max.

        - integration_upper_lim (float): Inclusive lower bound to the upper limit
        of the integration. In order to apply the trapezoid rule, the maximum point
        (i.e. the point whose x-value is the maximum) which is used is the point
        with the minimum x-value which still meets x>=integration_upper_lim.
        integration_upper_lim must belong to (integration_lower_lim, max], where
        max is the biggest x-value for which the integrated waveform is defined.
        If this parameter is not set or integration_upper_lim>max, then
        integration_upper_lim is set to max. If integration_upper_lim is smaller
        or equal to integration_lower_lim, then the trapezoid rule is applied by
        using just the two nearest points (i.e. one trapezoid) to
        integration_lower_lim."""

        htype.check_type(
            input_resistance_in_ohms,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "GainMeas.integrate", 67343
            ),
        )
        htype.check_type(
            system_amplification_factor,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "GainMeas.integrate", 47221
            ),
        )
        if system_amplification_factor <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("GainMeas.integrate", 94103)
            )

        # integration_lower_lim and integration_upper_lim
        # are checked and handled by WaveformSet.integrate().

        result = self.Waveforms.integrate(
            input_resistance_in_ohms=input_resistance_in_ohms,
            system_amplification_factor=system_amplification_factor,
            filter_out_infs_and_nans=True,
            integration_lower_lim=integration_lower_lim,
            integration_upper_lim=integration_upper_lim,
        )

        # Filtering out entries which equate to float('inf'),
        # float('-inf') or float('nan'), which resulted from
        # integrating a defective waveform.
        self.__charge_entries = Waveform.filter_infs_and_nans(result, get_mask=False)
        return

    def fit_peaks_histogram(
        self,
        peaks_to_detect=3,
        peaks_to_fit=None,
        bins_no=200,
        histogram_range=None,
        starting_fraction=0.0,
        step_fraction=0.01,
        minimal_prominence_wrt_max=0.0,
        std_no=3.0,
        plot_axes=None,
        logarithmic_plot=False,
        axes_title=None,
        gaussian_plot_npoints=100,
        plot_charge_range=None,
        plot_fit=True,
    ):
        """This method gets the following optional keyword arguments:

        - peaks_to_detect (scalar integer): It must be positive (>0). It is
        given to the 'peaks_to_detect' keyword argument of
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(). It is the
        number of peaks which will be detected to start with. If
        peaks_to_detect matches N, then the N highest peaks of the histogram
        of self.__charge_entries will be detected. A subset of the detected
        peaks will be fit, up to the peaks_to_fit argument.
        - peaks_to_fit (None or tuple): It is given to the 'peaks_to_fit'
        keyword argument of
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(). If None,
        then it is assumed that all of the detected peaks should be fit.
        If it is a tuple, then it must contain integers. Its length must
        comply with 0<=len(peaks_to_fit)<=peaks_to_detect. Every entry must
        belong to the interval [0, peaks_to_detect-1]. Let us sort the
        peaks_to_detect detected peaks according to the iterator value for
        self.__charge_entries where they occur. Then if i belongs to
        peaks_to_fit, the i-th detected peak will be fit.
        - bins_no (scalar integer): It must be positive (>0). It is eventually
        given to the 'bins' parameter of matplotlib.pyplot.hist(). It is the
        number of bins which are used to histogram the self.__charge_entries.
        - histogram_range (None or tuple): This parameter is given to the
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks() static
        method. If None, then numpy.histogram automatically sets the charge
        histogram range to the minimum and maximum values of the samples array,
        which in this case is self.__charge_entries. Otherwise, it should be
        a tuple of two elements, say (min, max), that defines the lower and
        upper range of the charge histogram to fit. This function does not
        check the well-formedness of this parameter. The called static method
        does.
        - starting_fraction (scalar float): It must be semipositive (>=0.0)
        and smaller or equal to 1 (<=1.0). It is given to the static method
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(), which in
        turn, gives it to SiPMMeas.__spot_first_peaks_in_CalibrationHistogram().
        Check its docstrings for more information.
        - step_fraction (scalar float): It must be positive (>0.0) and smaller
        or equal to 1 (<=1.0). It is given to the static method
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(), which in
        turn, gives it to SiPMMeas.__spot_first_peaks_in_CalibrationHistogram().
        Check its docstrings for more information.
        - minimal_prominence_wrt_max (scalar float): It must be semipositive
        (>=0) and smaller or equal than 1.0 (<=1.0). It is understood as a
        fraction of the maximum value of the histogram of self.ChargeEntries.
        It is given to the minimal_prominence_wrt_max keyword argument of
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(). It sets
        a minimal prominence for a peak to be fit, based on a fraction of
        the maximum value within the self.ChargeEntries histogram. I.e. the
        only considered peaks are those whose prominence is bigger or equal
        to a fraction of the self.ChargeEntries histogram maximum. For more
        information check the
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks() docstring.
        - std_no (scalar float): It must be positive (>0.0). It is given to
        the static method SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(),
        which in turn, gives it to SiPMMeas.piecewise_gaussian_fits(). Check
        its docstrings for more information. Check its docstrings for more
        information.
        - plot_axes (None or matplotlib.axes.Axes object): If it is not
        defined, then no plot is done. If it is defined, then the fit
        histogram is plotted in the given axes.
        - logarithmic_plot (scalar boolean): This parameter only makes a
        difference if plot_axes is defined. In such case, it means whether
        the plotted histogram, and potentially the plotted gaussian fits,
        are in logarithmic scale or not.
        - axes_title (None or string): This parameter only makes a difference
        if plot_axes is defined. In such case, it is the title of the given
        axes.
        - gaussian_plot_npoints (scalar integer): It must be positive (>0).
        This parameter only makes a difference if plot_axes is defined. It
        matches the number of points which are plotted for each gaussian fit.
        - plot_charge_range (None or list of two floats): This parameter
        only makes a difference if plot_axes is suitably defined. In such
        case, it is given to plot_axes.set_xlim().
        - plot_fit (scalar boolean): This parameter only makes a difference
        if plot_axes is defined. In such case, it means whether to plot
        the fit functions together with the plotted histogram. Note that,
        if plot_fit is True, then the fit plots are also affected by the
        logarithmic_plot parameter.

        This method histograms self.__charge_entries and fits one gaussian
        to a subset of the peaks_to_detect highest peaks which comply with
        the specified prominence requirement. Such subset is specified via
        the peaks_to_fit keyword argument. To perform the fits, this method
        calls SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(). In
        addition, if plot_axes is provided, then the resulting histogram is
        plotted in the given axes. Furthermore, if plot_fit is True, then the
        fit functions are plotted together with the histogram. To end with,
        this method returns the output of
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(), which are
        two lists, say popt and pcov, so that popt[i] (resp. pcov[i]) is the
        set of optimal values (resp. covariance matrix) for the fit of the
        i-th peak. For more information, check
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks() docstring."""

        htype.check_type(
            peaks_to_detect,
            int,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_peaks_histogram", 67343
            ),
        )
        if peaks_to_detect < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("GainMeas.fit_peaks_histogram", 92127)
            )

        # peaks_to_fit is handled by SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks()

        htype.check_type(
            bins_no,
            int,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_peaks_histogram", 28150
            ),
        )
        if bins_no < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("GainMeas.fit_peaks_histogram", 96724)
            )
        htype.check_type(
            starting_fraction,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_peaks_histogram", 25946
            ),
        )
        if starting_fraction < 0.0 or starting_fraction > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("GainMeas.fit_peaks_histogram", 91126)
            )
        htype.check_type(
            step_fraction,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_peaks_histogram", 42480
            ),
        )
        if step_fraction <= 0.0 or step_fraction > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("GainMeas.fit_peaks_histogram", 79370)
            )
        htype.check_type(
            minimal_prominence_wrt_max,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_peaks_histogram", 42990
            ),
        )
        if minimal_prominence_wrt_max < 0.0 or minimal_prominence_wrt_max > 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("GainMeas.fit_peaks_histogram", 12397)
            )
        htype.check_type(
            std_no,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_peaks_histogram", 44358
            ),
        )
        if std_no <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("GainMeas.fit_peaks_histogram", 24400)
            )
        fPlot = False
        if plot_axes is not None:
            htype.check_type(
                plot_axes,
                matplotlib.axes._axes.Axes,
                exception_message=htype.generate_exception_message(
                    "GainMeas.fit_peaks_histogram", 71025
                ),
            )
            fPlot = True

        htype.check_type(
            logarithmic_plot,
            bool,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_peaks_histogram", 92611
            ),
        )
        if axes_title is not None:
            htype.check_type(
                axes_title,
                str,
                exception_message=htype.generate_exception_message(
                    "GainMeas.fit_peaks_histogram", 47180
                ),
            )
        htype.check_type(
            gaussian_plot_npoints,
            int,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_peaks_histogram", 54408
            ),
        )
        if gaussian_plot_npoints < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("GainMeas.fit_peaks_histogram", 13908)
            )
        if plot_charge_range is not None:
            htype.check_type(
                plot_charge_range,
                list,
                exception_message=htype.generate_exception_message(
                    "GainMeas.fit_peaks_histogram", 22037
                ),
            )
            if len(plot_charge_range) != 2:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "GainMeas.fit_peaks_histogram", 37188
                    )
                )
            for element in plot_charge_range:
                htype.check_type(
                    element,
                    float,
                    np.float64,
                    int,
                    np.int64,
                    exception_message=htype.generate_exception_message(
                        "GainMeas.fit_peaks_histogram", 32461
                    ),
                )
        htype.check_type(
            plot_fit,
            bool,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_peaks_histogram", 47189
            ),
        )
        if self.__charge_entries is None:
            raise cuex.NoAvailableData(
                htype.generate_exception_message(
                    "GainMeas.fit_peaks_histogram",
                    99872,
                    extra_info="For GainMeas.fit_peaks_histogram() to fit the charge histogram, the charge entries must have been previously computed, p.e. via GainMeas.integrate().",
                )
            )

        # Assuming popt[i][0], popt[i][1] and popt[i][2]
        # to be the optimal value for the gaussian mean,
        # standard deviation and scaling, respectively.
        popt, pcov = SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(
            self.__charge_entries,
            peaks_to_detect=peaks_to_detect,
            peaks_to_fit=peaks_to_fit,
            bins_no=bins_no,
            histogram_range=histogram_range,
            starting_fraction=starting_fraction,
            step_fraction=step_fraction,
            minimal_prominence_wrt_max=minimal_prominence_wrt_max,
            std_no=std_no
        )
        if fPlot:
            _, _, _ = plot_axes.hist(
                self.__charge_entries,
                bins_no,
                log=logarithmic_plot,
                histtype="step")
            
            # Do not extend the vertical axis down to order of
            # magnitudes which are senseless for a hits-histogram.
            if logarithmic_plot:
                plot_axes.set_ylim(10**-1)

            if plot_fit:

                # SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks() gives
                # scaling seeds to SiPMMeas.piecewise_gaussian_fits(),
                # therefore this is the fitting function.
                gaussian = lambda z, mean, std, scaling: scaling * math.exp(
                    -0.5 * (z - mean) * (z - mean) / (std * std)
                )
                gaussian = np.vectorize(gaussian, excluded=(1, 2, 3))

                # G. mean minus std_no times the g. std.    # G. mean plus std_no times the g. std.
                piecewise_xs = [
                    np.linspace(
                        popt[i][0] - (popt[i][1] * std_no),
                        popt[i][0] + (popt[i][1] * std_no),
                        num=gaussian_plot_npoints,
                    )
                    for i in range(len(popt))
                ]
                piecewise_ys = [
                    gaussian(piecewise_xs[i], popt[i][0], popt[i][1], popt[i][2])
                    for i in range(len(popt))
                ]

                aux_func = plot_axes.semilogy if logarithmic_plot else plot_axes.plot

                for i in range(len(piecewise_xs)):
                    aux_func(
                        piecewise_xs[i],
                        piecewise_ys[i],
                        linestyle="--",
                        color="black"
                    )

            # Add some text giving the number
            # of fitted peaks in the histogram
            plot_axes.text(
                .99,
                .98,
                f"{len(popt)} f. p.",
                # Make the coordinates relative to the axes system
                transform=plot_axes.transAxes,
                verticalalignment='top',
                horizontalalignment='right'
            )

            plot_axes.set_xlabel(f"Charge (C)")
            plot_axes.set_ylabel(f"Hits")
            plot_axes.set_xlim(plot_charge_range)
            plot_axes.set_title(axes_title)
            plot_axes.grid()

        return popt, pcov

    def fit_gain(
        self,
        has_pedestal=True,
        peaks_to_detect=3,
        peaks_to_fit=None,
        bins_no=200,
        starting_fraction=0.0,
        step_fraction=0.01,
        minimal_prominence_wrt_max=0.0,
        std_no=3.0,
        gain_fit_axes=None,
        errorbars_scaling=1.0,
        histogram_fit_axes=None,
        logarithmic_plot=False,
        histogram_axes_title=None,
        gaussian_plot_npoints=100,
        plot_charge_range=None,
        plot_histogram_fit=True,
    ):
        """This method gets the following optional keyword arguments:

        - has_pedestal (scalar boolean): Whether to assume that the entries
        within self.__charge_entries yield a peaks-histogram with a pedestal.
        - peaks_to_detect (scalar integer): It must be positive (>0). It is
        given to the 'peaks_to_detect' keyword argument of
        GainMeas.fit_peaks_histogram(). It is the number of peaks which will
        be detected to start with. If peaks_to_detect matches N, then the N
        highest peaks of the histogram of self.__charge_entries will be
        detected. A subset of the detected peaks will be fit, up to the
        peaks_to_fit argument.
        - peaks_to_fit (None or tuple): It is given to the 'peaks_to_fit'
        keyword argument of GainMeas.fit_peaks_histogram(). If None,
        then it is assumed that all of the detected peaks should be fit.
        If it is a tuple, then it must contain integers. Its length must
        comply with 0<=len(peaks_to_fit)<=peaks_to_detect. Every entry must
        belong to the interval [0, peaks_to_detect-1]. Let us sort the
        peaks_to_detect detected peaks according to the iterator value for
        self.__charge_entries where they occur. Then if i belongs to
        peaks_to_fit, the i-th detected peak will be fit.
        - bins_no (scalar integer): It must be positive (>0). It is eventually
        given to the 'bins' parameter of matplotlib.pyplot.hist(). It is the
        number of bins which are used to histogram the self.__charge_entries.
        - starting_fraction (scalar float): It must be semipositive (>=0.0)
        and smaller or equal to 1 (<=1.0). It is given to the
        GainMeas.fit_peaks_histogram() method, which in turn gives it to the
        static method SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(),
        which in turn, gives it to
        SiPMMeas.__spot_first_peaks_in_CalibrationHistogram(). Check its
        docstrings for more information.
        - step_fraction (scalar float): It must be positive (>0.0) and smaller
        or equal to 1 (<=1.0). It is given to the
        GainMeas.fit_peaks_histogram() method, which in turn gives it to the
        static method SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(),
        which in turn, gives it to
        SiPMMeas.__spot_first_peaks_in_CalibrationHistogram(). Check its
        docstrings for more information.
        - minimal_prominence_wrt_max (scalar float): It must be semipositive
        (>=0) and smaller or equal than 1.0 (<=1.0). It is understood as a
        fraction of the maximum value of the histogram of self.ChargeEntries.
        It is given to the minimal_prominence_wrt_max keyword argument of
        SiPMMeas.fit_peaks_histogram(). It sets a minimal prominence for a
        peak (of the charge histogram) to be fit, based on a fraction of
        the maximum value within the self.ChargeEntries histogram. I.e. the
        peaks from the charge histogram which are considered are only those
        whose prominence is bigger or equal to a fraction
        np.max(self.ChargeEntries) histogram maximum. For more information
        check the SiPMMeas.fit_peaks_histogram() docstring.
        - std_no (scalar float): It must be positive (>0.0). It is given to the
        GainMeas.fit_peaks_histogram() method, which in turn gives it to the
        static method SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(),
        which in turn, gives it to SiPMMeas.piecewise_gaussian_fits(). Check its
        docstrings for more information.
        - gain_fit_axes (None or matplotlib.axes.Axes object): If it is not
        defined, then gain fit plot is done. If it is defined, then the gain
        fit is plotted in the given axes.
        - errorbars_scaling (scalar float): This parameter only makes a
        difference if gain_fit_axes is suitably defined. In that case, the
        error bars of the gain-fit plot, are scaled by errorbars_scaling.
        - histogram_fit_axes (None or matplotlib.axes.Axes object): If it is
        not defined, then the histogram fit is not done. If it is defined,
        then the fit histogram is plotted in the given axes.
        - logarithmic_plot (scalar boolean): It is given to the 'logarithmic_plot'
        keyword argument of GainMeas.fit_peaks_histogram(). This parameter
        only makes a difference if histogram_fit_axes is defined. In such
        case, it means whether the plotted histogram, and potentially the
        plotted gaussian fits, are in logarithmic scale or not.
        - histogram_axes_title (None or string): This parameter only makes
        a difference if histogram_fit_axes is defined. It is given to the
        'axes_title' keyword argument of GainMeas.fit_peaks_histogram(). It
        is used as the title of the axes given by histogram_fit_axes.
        - gaussian_plot_npoints (scalar integer): It must be positive (>0).
        This parameter only makes a difference if histogram_fit_axes is
        defined. It matches the number of points which are plotted for each
        gaussian fit.
        - plot_charge_range (None or list of two floats): This parameter
        only makes a difference if histogram_fit_axes is suitably defined.
        In such case, it is given to self.fit_peaks_histogram(), which in
        turn gives it to plot_axes.set_xlim().
        - plot_histogram_fit (scalar boolean): This parameter only makes a
        difference if histogram_fit_axes is defined. In such case, it means
        whether to plot the fit functions together with the plotted histogram.
        Note that, if plot_histogram_fit is True, then the fit plots are also
        affected by the logarithmic_plot parameter.

        This method calls self.fit_peaks_histogram(), which fits a gaussian
        function to a subset of the peaks_to_detect highest peaks of the
        charge histogram which meet the specified prominence requirement.
        Such subset is specified via the peaks_to_fit keyword argument. For
        more information, check the GainMeas.fit_peaks_histogram() docstring.
        The mean of the i-th fit gaussian is used as the charge-value for
        the i-photoelectrons point. Furthermore, the standard deviation of
        such charge-value, computed as the square root of the i-th diagonal
        element of the fit covariance matrix (see
        GainMeas.fit_peaks_histogram() docstring for more information), is
        assumed to be the error of such charge-value. If has_pedestal is
        True, then the first fit peak is assumed to be the pedestal, i.e.
        the 0-photoelectrons case. If else, then the first fit peak is
        assumed to match the 1-photoelectrons case. As a result, this
        method computes a set of len(peaks_to_fit) points, which is then
        fit to a linear function. Furthermore, if gain_fit_axes is suitably
        defined, then the resulting points, together with the linear fit,
        are plotted in the gain_fit_axes axes, with y-errorbars which
        match the charge values errors, but scaled by errorbars_scaling.
        To end with, this method returns four lists, in the following order:

            - gain_popt: Optimal values for the gain-fit parameters.
            gain_popt[0] (resp. gain_popt[1]) matches the optimal slope (resp.
            intercept) resulting from the gain fit.

            - gain_stderr: Standard error of the estimated parameters, as
            computed by scipy.stats.linregress() under the assumption of
            residual normality. gain_stderr[0] (resp. gain_stderr[1]) is
            the standard error for the slope (resp. intercept).

            - histogram_popt, histogramp_pcov: These lists match the output
            of self.fit_peaks_histogram(). histogram_popt[i] (resp.
            histogram_pcov[i]) is the set of optimal values (resp. covariance
            matrix) for the fit of the i-th peak. For more information, check
            GainMeas.fit_peaks_histogram() docstring."""

        htype.check_type(
            has_pedestal,
            bool,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_gain", 37199
            ),
        )
        htype.check_type(
            peaks_to_detect,
            int,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_gain", 12275
            ),
        )
        if peaks_to_detect < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("GainMeas.fit_gain", 83758)
            )
        peaks_to_fit_ = tuple(range(0, peaks_to_detect))
        if peaks_to_fit is not None:

            htype.check_type(
                peaks_to_fit,
                tuple,
                exception_message=htype.generate_exception_message(
                    "GainMeas.fit_gain", 36637
                ),
            )
            if len(peaks_to_fit) > peaks_to_detect:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("GainMeas.fit_gain", 67688)
                )
            for elem in peaks_to_fit:
                htype.check_type(
                    elem,
                    int,
                    np.int64,
                    exception_message=htype.generate_exception_message(
                        "GainMeas.fit_gain", 18999
                    ),
                )
                if elem < 0 or elem > (peaks_to_detect - 1):
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message("GainMeas.fit_gain", 29834)
                    )
            peaks_to_fit_ = peaks_to_fit

        fPlotGainFit = False
        if gain_fit_axes is not None:
            htype.check_type(
                gain_fit_axes,
                matplotlib.axes._axes.Axes,
                exception_message=htype.generate_exception_message(
                    "GainMeas.fit_gain", 67687
                ),
            )
            fPlotGainFit = True

        htype.check_type(
            errorbars_scaling,
            float,
            exception_message=htype.generate_exception_message(
                "GainMeas.fit_gain", 12478
            ),
        )

        # bins_no, starting_fraction, step_fraction, minimal_prominence_wrt_max,
        # std_no, histogram_fit_axes, logarithmic_plot, histogram_axes_title,
        # gaussian_plot_npoints, plot_charge_range and plot_histogram_fit
        # are handled to GainMeas.fit_peaks_histogram() and never used
        # again in this method. Well-formedness checks of these parameters
        # are done by GainMeas.fit_peaks_histogram().

        histogram_popt, histogram_pcov = self.fit_peaks_histogram(
            peaks_to_detect=peaks_to_detect,
            peaks_to_fit=peaks_to_fit_,
            bins_no=bins_no,
            starting_fraction=starting_fraction,
            step_fraction=step_fraction,
            minimal_prominence_wrt_max=minimal_prominence_wrt_max,
            std_no=std_no,
            plot_axes=histogram_fit_axes,
            logarithmic_plot=logarithmic_plot,
            axes_title=histogram_axes_title,
            gaussian_plot_npoints=gaussian_plot_npoints,
            plot_charge_range=plot_charge_range,
            plot_fit=plot_histogram_fit,
        )

        photoelectrons_no = (
            range(0, len(peaks_to_fit_))
            if has_pedestal
            else range(1, len(peaks_to_fit_) + 1)
        )
        photoelectrons_no = np.array(photoelectrons_no)

        photoelectrons_charge = np.array(
            [histogram_popt[i][0] for i in range(len(histogram_popt))]
        )
        photoelectrons_charge_errors = [
            math.sqrt(histogram_pcov[i][0, 0]) for i in range(len(histogram_pcov))
        ]

        aux = spsta.linregress(photoelectrons_no, y=photoelectrons_charge)
        gain_popt = [aux.slope, aux.intercept]
        gain_stderr = [aux.stderr, aux.intercept_stderr]

        fit_function = lambda x: (gain_popt[0] * x) + gain_popt[1]

        if fPlotGainFit:
            gain_fit_axes.errorbar(
                photoelectrons_no,
                photoelectrons_charge,
                # The given value is considered to be a
                # symmetric error. I.e. the error bar
                # length matches this value times two.
                yerr=errorbars_scaling * np.array(photoelectrons_charge_errors) / 2.0,
                marker="s",
                markersize=4,
                linestyle="none",
                color="black",
            )

            gain_fit_axes.plot(
                [-0.5, photoelectrons_no[-1] + 0.5],
                # Plotting the linear fit down to -0.5
                # PE, so that we can check whether the
                # intercept is near to zero, as it should
                # be if the waveforms baseline matches zero
                [fit_function(-0.5), fit_function(photoelectrons_no[-1] + 0.5)],
                linestyle="--",
                color="black",
            )
            gain_fit_axes.set_xlabel(f"# Photoelectrons")
            gain_fit_axes.set_ylabel(f"Charge (C)")
            gain_fit_axes.set_xlim([-0.5, photoelectrons_no[-1] + 0.5])
            gain_fit_axes.grid()

        return gain_popt, gain_stderr, histogram_popt, histogram_pcov

    @classmethod
    def from_json_file(cls, gainmeas_config_json):
        """This class method is meant to be an alternative initializer
        for GainMeas. This class method gets the following mandatory
        positional argument:

        - gainmeas_config_json (string): Path to a json file which
        hosts all of the necessary information to define the GainMeas
        object.

        This method creates and returns a GainMeas object that is
        crafted out of the given json file. To do so, first, this
        method populates two RigidKeyDictionary's. The first one,
        say RKD1, which concerns the GainMeas attributes, has the
        following potential keys:

        "delivery_no", "set_no", "meas_no", "strip_ID", 
        "meas_ID", "date", "location", "operator", "setup_ID",
        "system_characteristics", "thermal_cycle",
        "electronic_board_number", "electronic_board_location", 
        "electronic_board_socket", "sipm_location", "sampling_ns", 
        "cover_type", "operation_voltage_V", "overvoltage_V", 
        "PDE", "status", "LED_voltage_V", "LED_frequency_kHz",
        "LED_pulse_shape", "LED_high_width_ns" and
        "wvfset_json_filepath".

        Although "sampling_ns" appears here, it is not meant to be
        read from gainmeas_config_json. The value for
        self.__sampling_ns will be computed from the value given to
        "time_resolution" in the file given to wvfset_json_filepath.

        The second one, say RKD2, concerns the WaveformSet.from_files
        parameters, and is populated out of the file whose path is
        given to the wvfset_json_filepath key of RKD1. It has the following
        potential keys:

        "wvf_filepath", "time_resolution", "points_per_wvf",
        "wvfs_to_read", "timestamp_filepath", "delta_t_wf",
        "packing_version", "set_name", "creation_dt_offset_min"
        and "wvf_extra_info".

        Here, we do not expect a date because the date information
        is taken from the GainMeas json file.

        These potential keys are typed according to the
        GainMeas.__init__ docstring. To populate RKD1 and RK2,
        this method uses the dictionaries which are loaded from the
        specified json files. Namely, every entry that belongs to one of
        the two json dictionaries and is suitably formatted, up to its
        corresponding RigidKeyDictionary rules, is added to its
        RigidKeyDictionary.

        Once both RigidKeyDictionary's have been populated, a
        GainMeas object is created by calling the class initializer
        using the key-value pairs of RKD1 and RKD2 as the kwarg-value
        pairs for the initializer call, in that order. I.e. the class
        initializer is called with **RKD1, **RKD2. The only exceptions
        are the values given to "wvf_filepath" and "time_resolution" in
        RKD1, which are passed as positional arguments, in such order,
        to the class initializer. Particularly, the "time_resolution"
        parameter is converted to seconds using the time unit information
        which is read from the file whose path is given by the
        "wvf_extra_info" entry of RKD2. "wvfset_json_filepath" is also
        an exception, since it is used to populate RKD2, and it's
        deleted afterwards."""

        htype.check_type(
            gainmeas_config_json,
            str,
            exception_message=htype.generate_exception_message(
                "GainMeas.from_json_file", 67497
            ),
        )

        # These are used to configure the SiPMMeas attributes
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
            "LED_voltage_V": float,
            "LED_frequency_kHz": float,
            "LED_pulse_shape": str,
            "LED_high_width_ns": float,
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

        with open(gainmeas_config_json, "r") as file:
            input_data = json.load(file)
        RKD1.update(input_data)

        try:
            wvfset_json_filepath = RKD1["wvfset_json_filepath"]
        except KeyError:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "GainMeas.from_json_file",
                    64191,
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
                    "GainMeas.from_json_file",
                    59683,
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
            # care of type-checking aux['time_unit'][0]
            time_unit_in_s = NumpyDataPreprocessor.interpret_time_unit_in_seconds(
                aux['time_unit'][0]
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

    # Overrides SiPMMeas.output_summary()
    def output_summary(
        self,
        additional_entries={},
        folderpath=None,
        include_analysis_results=True,
        overwrite=False,
        indent=None,
        verbose=False,
    ):
        """
        This method gets the following keyword arguments:

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
        - include_analysis_results (bool): If this parameter is True, then 
        the self.__charge_entries value is included in the output
        dictionary under the key "charge_entries". If this parameter
        is False, then this key is still included in the output dictionary,
        but its value is set to float('nan').
        - overwrite (bool): This parameter only makes a difference if
        the 'folderpath' parameter is defined and if there is already
        a file in the given folder path whose name matches
        f"G-{self.StripID}-{self.SiPMLocation}-{self.ThermalCycle}-OV{round(10.*self.Overvoltage_V)}dV-{self.Date.strftime('%Y-%m-%d')}.json".
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
        GainMeas object. Additionally, this method can serialize this dictionary
        to an output json file if the 'folderpath' parameter is defined. This
        dictionary has as many fields as objects of interest which should be
        summarized. For GainMeas objects, these fields are:

        - "delivery_no": Contains self.__delivery_no
        - "set_no": Contains self.__set_no
        - "meas_no": Contains self.__meas_no
        - "strip_ID": Contains self.StripID
        - "meas_ID": Contains self.MeasID
        - "date": Contains self.Date
        - "location": Contains self.Location
        - "operator": Contains self.Operator
        - "setup_ID": Contains self.SetupID
        - "system_characteristics": Contains self.SystemCharacteristics
        - "thermal_cycle": Contains self.ThermalCycle
        - "electronic_board_number": Contains self.ElectronicBoardNumber
        - "electronic_board_location": Contains self.ElectronicBoardLocation
        - "electronic_board_socket": Contains self.ElectronicBoardSocket
        - "sipm_location": Contains self.SiPMLocation
        - "sampling_ns": Contains self.Sampling_ns
        - "waveform_window_mus": Contains self.WaveformWindow_mus
        - "cover_type": Contains self.CoverType
        - "operation_voltage_V": Contains self.OperationVoltage_V
        - "overvoltage_V": Contains self.Overvoltage_V
        - "PDE": Contains self.PDE
        - "N_events": Contains self.NEvents
        - "signal_unit": Contains self.SignalUnit
        - "status": Contains self.Status
        - "acquisition_time_min": Contains self.AcquisitionTime_min,

        - "LED_voltage_V": Contains self.__LED_voltage_V,
        - "LED_frequency_kHz": Contains self.__LED_frequency_kHz,
        - "LED_pulse_shape": Contains self.__LED_pulse_shape,
        - "LED_high_width_ns": Contains self.__LED_high_width_ns,
        - "charge_entries": Contains self.__charge_entries if
        include_analysis_results and float('nan') otherwise

        Note that the output does not contain the gain of this GainMeas. To do
        so, we could allow the use of kwargs and pass them onto GainMeas.fit_gain().
        For now, this functionality is not enabled, so that the functioning
        of GainMeas.fit_gain() is separated from GainMeas.output_summary().
        I.e. GainMeas.output_summary() does not need to call GainMeas.fit_gain().
        Still, if one wants to include the gain in the output dictionary,
        it can be computed externally and passed to the 'additional_entries'
        parameter.

        This method returns a summary dictionary of the GainMeas object.
        If a folder path is given, then the output dictionary is additionally
        serialized to a json file in the specified folder. The file name
        matches the following formatted string:

        f"G-{self.StripID}-{self.SiPMLocation}-{self.ThermalCycle}-OV{round(10.*self.Overvoltage_V)}dV-{self.Date.strftime('%Y-%m-%d')}.json"
        """

        htype.check_type(
            additional_entries,
            dict,
            exception_message=htype.generate_exception_message(
                "GainMeas.output_summary", 55230
            ),
        )

        htype.check_type(
            include_analysis_results,
            bool,
            exception_message=htype.generate_exception_message(
                "GainMeas.output_summary", 60135
            ),
        )

        # The only parameters we are checking are additional_entries
        # and include_analysis_results, because we are making use of
        # them in the body of this function. The rest of them are
        # only used by the overriden base class, so they are type-
        # and well-formedness- checked there.

        gainmeas_additional_output = {
            "LED_voltage_V": self.__LED_voltage_V,
            "LED_frequency_kHz": self.__LED_frequency_kHz,
            "LED_pulse_shape": self.__LED_pulse_shape,
            "LED_high_width_ns": self.__LED_high_width_ns
        }

        if include_analysis_results:
            analysis_results = {
                # Object of type numpy.ndarray is not
                # JSON serializable, but lists are
                "charge_entries": list(self.__charge_entries)
            }
        else:
            analysis_results = {
                "charge_entries": float('nan')
            }

        gainmeas_additional_output.update(analysis_results)
        gainmeas_additional_output.update(additional_entries)

        return super().output_summary(
            additional_entries=gainmeas_additional_output,
            folderpath=folderpath,
            filename=f"G-{self.StripID}-{self.SiPMLocation}-"
            f"{self.ThermalCycle}-OV{round(10.*self.Overvoltage_V)}dV-"
            f"{self.Date.strftime('%Y-%m-%d')}.json",
            overwrite=overwrite,
            indent=indent,
            verbose=verbose
        )
    
    # Overrides SiPMMeas.get_title(), simply because this
    # implementation includes information about the 
    # overvoltage, which is a more relevant parameter in
    # the case of gain measurements
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

        return super().get_title(abbreviate=abbreviate) + f", OV {round(10.*self.Overvoltage_V)} dV"