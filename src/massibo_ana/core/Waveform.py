import math
import numpy as np

import massibo_ana.utils.htype as htype
import massibo_ana.utils.custom_exceptions as cuex
from massibo_ana.custom_types.TypedList import TypedList
from massibo_ana.custom_types.ListsRKD import ListsRKD


class Waveform:

    def __init__(
        self,
        t0,
        signal,
        t_step=None,
        time=None,
        infer_t0=False,
        signs=None
    ):
        """This class aims to model a certain waveform (a curve in a 2D plane) as a function
        of the time. In practice, waveforms start at t=0. To recover its time position with
        respect to some absolute time scale, one should refer to the t0 attribute. Unless
        otherwise stated, all of the time values are expressed in seconds. This is the
        initializer for such class. This initializer gets two mandatory positional arguments:

        - t0 (float scalar): Time, with respect to some time absolute scale, of the first
        data point of the waveform.
        - signal (unidimensional float numpy array): For a waveform of a certain magnitude,
        say V, signal hosts the V values of the waveform datapoints. If its type does not
        match np.float64, it will be casted to np.float64.

        This initializer gets the following optional keyword arguments:

        - t_step (scalar float): Time step between two adjacent data points of the waveform.
        't_step' and 'time' are alternative ways to give the minimal required information
        about the time entries for the waveform. One of them must be appropriately defined.
        If both are defined, t_step is used by default.
        - time (unidimensional float numpy array): Hosts the time entries of the waveform
        datapoints.
        - infer_t0 (scalar boolean): If t_step is not None, then this parameter makes no
        difference. Else, and if time is not None, infer_t0 has the following behaviour. If
        infer_t0 is False, then the initial time information (self.t0) is taken from the t0
        argument. Else, if infer_t0 is True and the smallest entry in time (assume it is
        time[0] after a proper sort) is different from 0, then self.t0 is taken to be time[0].
        If infer_t0 is True, but time[0] is 0, then self.t0 is set to the t0 argument.
        - signs (dictionary): This dictionary is meant to host any extra information about the
        waveform. Apart from the storage purpose, this information may come in handy for data
        visualization, p.e. plotting the waveform and the predicted position for peaks in it,
        all at the same time, plotting the waveform together with the lower and upper limits
        for some associated integration operation, or plotting the waveform together with the
        its predicted baseline. Ideally, the argument passed to signs is loaded to the attribute
        self.__signs, which is a ListsRKD (i.e. a RigidKeyDictionary whose values are either
        lists or TypedList objects). Addition to self.__signs is handled by its setter
        (Signs.setter), which forces the compliance of the given information with certain
        format: The provided key must belong to the class variable "nonstackable_aks" or the
        class variable "stackable_aks". Also, depending on the specified key, the provided
        value must comply with some type format. For the time being, the allowed keys and its
        associated-value types are:

            -> "time_unit":                 The type of its value must be a list with one
                                            string. It represents the time unit of the
                                            waveform datapoints. (non stackable)
            -> "signal_magnitude":          The type of its value must be a list with one
                                            string. It represents the name of the signal
                                            magnitude, p.e. "voltage". (non stackable)
            -> "signal_unit":               The type of its value must be a list with one
                                            string. It represents the signal-magnitude unit.
                                            (non stackable)
            -> "integration_ll":            Its value must be a list with one float. It
                                            represents the lower limit for some associated
                                            integration of the waveform. (non stackable)
            -> "integration_ul":            Its value must be a list with one float. It
                                            represents the upper limit for some associated
                                            integration of the waveform. (non stackable)
            -> "first_peak_baseline":       Its value must be a list with one float. It
                                            represents the baseline, in signal units, of
                                            the first peak (in time) within this waveform.
                                            This value may not match the physical baseline
                                            of any other secondary peak within the
                                            acquisition window of this waveform.
                                            (non stackable)
            -> "median_cutoff":             Its value must be a list with one float. The
                                            first-peak baseline is computed as the median
                                            of the signal points which happen below a
                                            certain cutoff-time. That cutoff-time is what
                                            we call the median cutoff. (non stackable)
            -> "peaks_pos":                 Its value must be a list whose elements have
                                            type float. They represent the time position
                                            where a peak has been detected. (stackable)
            -> "peaks_top":                 Its value must be a list whose elements have
                                            type float. They match the signal value at the
                                            detected peak. (stackable)

        For more information on this parameter and the setting process, see Signs.setter docstring.
        """

        htype.check_type(
            t0,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "Waveform.__init__", 1
            ),
        )
        htype.check_type(
            signal,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "Waveform.__init__", 2
            ),
        )
        if np.ndim(signal) != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("Waveform.__init__", 3)
            )
        
        if signal.dtype != np.float64:
            signal = signal.astype(np.float64)

        if t_step is None and time is None:
            raise cuex.InsufficientParameters(
                htype.generate_exception_message("Waveform.__init__", 4)
            )
        elif t_step is not None:
            # If both, time and t_step are defined, t_step configuration is used by default
            htype.check_type(
                t_step,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "Waveform.__init__", 5
                ),
            )
            if t_step <= 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("Waveform.__init__", 6)
                )
            fUseTStep = True
        else:
            # Reaching this block is only logically possible if t_step is None and time is not None
            htype.check_type(
                time,
                np.ndarray,
                exception_message=htype.generate_exception_message(
                    "Waveform.__init__", 7
                ),
            )
            if np.ndim(time) != 1:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("Waveform.__init__", 8)
                )
            htype.check_type(
                time[0],
                np.float64,
                exception_message=htype.generate_exception_message(
                    "Waveform.__init__", 9
                ),
            )
            if np.shape(time) != np.shape(signal):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("Waveform.__init__", 10)
                )
            fUseTStep = False

        if infer_t0 is not None:
            htype.check_type(
                infer_t0,
                bool,
                exception_message=htype.generate_exception_message(
                    "Waveform.__init__", 11
                ),
            )

        self.__signs = ListsRKD(
            Waveform.nonstackable_aks + Waveform.stackable_aks,  # potential_keys
            is_subtyped=True,  # is_subtyped=False
            values_subtypes=[str, str, str, float, float, float, float, float, float],
        )  # values_subtypes=None
        if signs is not None:
            htype.check_type(
                signs,
                dict,
                exception_message=htype.generate_exception_message(
                    "Waveform.__init__", 12
                ),
            )
            for key in signs.keys():

                # The self.__signs setter (Waveform.Signs.setter) will handle
                # the key and values type-checks as well as the proper addition
                # to self.__signs. Also, since in this case we are calling the
                # setter from the initializer, we can set overwrite to True
                # without worrying abour overwritting previous information.
                self.Signs = (key, signs[key], True)  # (key, value, overwrite)

        self.__npoints = np.shape(signal)[0]

        if fUseTStep:
            self.__signal = signal
            self.__time = np.linspace(
                0, ((self.__npoints - 1) * t_step), self.__npoints
            )
            self.__t0 = t0
        else:
            # We never checked that time was increasingly sorted
            self.__signal = signal[np.argsort(time)]
            self.__time = time[np.argsort(time)]
            if infer_t0 and self.__time[0] != 0.0:
                self.__t0 = self.__time[0]
            else:
                self.__t0 = t0
            self.__time -= self.__time[0]

        # This attribute is not computed by default.
        # It is only computed by self.integrate().
        self.__integral = None
        return

    # Class variables
    nonstackable_aks = [
        "time_unit",  # aks stands for Allowed KeyS
        "signal_magnitude",
        "signal_unit",
        "integration_ll",
        "integration_ul",
        "first_peak_baseline",
        "median_cutoff",
    ]

    stackable_aks = ["peaks_pos", "peaks_top"]

    # Getters
    @property
    def T0(self):
        return self.__t0

    @property
    def Npoints(self):
        return self.__npoints

    @property
    def Signal(self):
        return self.__signal

    @property
    def Time(self):
        return self.__time

    @property
    def Integral(self):
        return self.__integral

    @property
    def Signs(self):
        return self.__signs

    def get_absolute_time(self):
        return self.__t0 + self.__time

    def compute_first_peak_baseline(self, signal_fraction_for_median_cutoff=0.2):
        """This method gets the following optional keyword argument:

        - signal_fraction_for_median_cutoff (scalar float): It must belong to the
        interval [0.0, 1.0]. This value represents the fraction of the signal
        which is used to compute the baseline. P.e. 0.2 means that only the first
        20% (in time) of the signal is used to compute the baseline.

        This method computes the baseline of the first peak within the waveform.
        This value may not match the physical baseline for any other secondary peak
        within the acquisition window of this waveform. This method adds the
        computed baseline to the self.__signs dictionary under the key
        'first_peak_baseline', overwritting any previous value for such key,
        if applicable.

        This method computes the baseline in the following way. First, it computes
        the time cutoff below which the signal is considered. Then it filters out
        infinite or undefined points from such considered piece of the signal, using
        Waveform.filter_infs_and_nans(). The baseline is computed as the median of
        the remaining points. Check the signal_fraction_for_median_cutoff parameter
        documentation for more information. The maximum time below which the signal
        points are used to compute the baseline, is also added by this method to the
        self.__signs dictionary under the key 'median_cutoff', overwritting any
        previous value for such key, if applicable. The reason why only an initial
        (in time) fraction of the signal is used to compute the baseline is that the
        rest of the signal may be affected by the undershoot of the first peak. In
        this context, the median of a signal which is affected by a deep long
        undershoot would result in a baseline which is biased towards smaller values.
        """

        htype.check_type(
            signal_fraction_for_median_cutoff,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "Waveform.compute_first_peak_baseline", 1
            ),
        )
        if (
            signal_fraction_for_median_cutoff < 0.0
            or signal_fraction_for_median_cutoff > 1.0
        ):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "Waveform.compute_first_peak_baseline", 2
                )
            )
        cutoff_idx = round(signal_fraction_for_median_cutoff * len(self.__signal))
        signal_chunk = Waveform.filter_infs_and_nans(
            self.__signal[:cutoff_idx], get_mask=False
        )

        # Happens if the signal_fraction_for_median_cutoff
        # initial fraction of self.__signal contains nothing
        # but infs and nans. It makes no sense to compute
        # the median of such infinite/undefined values.
        if len(signal_chunk) == 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "Waveform.compute_first_peak_baseline",
                    3,
                    extra_info=f"The {signal_fraction_for_median_cutoff} initial fraction of the signal is either infinite or undefined. A baseline cannot be computed.",
                )
            )

        baseline = np.median(signal_chunk)
        self.Signs = ("first_peak_baseline", [baseline], True)
        self.Signs = ("median_cutoff", [self.__time[cutoff_idx]], True)
        return

    def plot(
        self, ax, xlim=None, ylim=None, plot_peaks=True, wvf_linewidth=1.0, x0=[], y0=[]
    ):
        """Still under development"""

        htype.check_type(
            plot_peaks,
            bool,
            exception_message=htype.generate_exception_message("Waveform.plot", 1),
        )
        htype.check_type(
            x0,
            list,
            exception_message=htype.generate_exception_message("Waveform.plot", 2),
        )
        for elem in x0:
            htype.check_type(
                elem,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "Waveform.plot", 3
                ),
            )
        htype.check_type(
            y0,
            list,
            exception_message=htype.generate_exception_message("Waveform.plot", 4),
        )
        for elem in y0:
            htype.check_type(
                elem,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "Waveform.plot", 5
                ),
            )
        # ax.set_title('Temporary title')
        if "time_unit" in self.__signs.keys():
            ax.set_xlabel(f"Time ({self.__signs['time_unit'][0]})")
        else:
            ax.set_xlabel("Time (a.u.)")
        if "signal_magnitude" in self.__signs.keys():
            if "signal_unit" in self.__signs.keys():
                ax.set_ylabel(
                    f"{self.__signs['signal_magnitude'][0]} ({self.__signs['signal_unit'][0]})"
                )
            else:
                ax.set_ylabel(f"{self.__signs['signal_magnitude'][0]} (a.u.)")
        if "integration_ll" in self.__signs.keys():
            ax.axvline(
                x=self.__signs["integration_ll"],
                linestyle="-",
                linewidth=0.5,
                color="red",
                label="Int. LL",
            )
        if "integration_ul" in self.__signs.keys():
            ax.axvline(
                x=self.__signs["integration_ul"],
                linestyle="-",
                linewidth=0.5,
                color="red",
                label="Int. UL",
            )
        if "first_peak_baseline" in self.__signs.keys():
            ax.axhline(
                y=self.__signs["first_peak_baseline"],
                linestyle="-",
                linewidth=0.5,
                color="blue",
                label="T.P. Baseline",
            )
        if "median_cutoff" in self.__signs.keys():
            ax.axvline(
                x=self.__signs["median_cutoff"],
                linestyle="-",
                linewidth=0.5,
                color="blue",
                label="Median cutoff",
            )
        if plot_peaks:
            if "peaks_pos" in self.__signs.keys():

                # If computing this amplitude at this level (plotting a single 
                # waveform) becomes an efficiency issue at some point, we can change
                # the way of computing the marker position for a less fancy one.
                aux_amplitude = np.max(self.__signal)-np.min(self.__signal)

                # Add some text giving the number
                # of spotted peaks in this waveform
                ax.text(
                    .99,
                    .98,
                    f"{len(self.__signs["peaks_pos"])} p.",
                    # Make the coordinates relative to the axes system
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    color='black'
                    if len(self.__signs["peaks_pos"]) == 1
                    else 'red'
                )

                # Assuming that peaks_pos and peaks_top have the same length
                for i in range(len(self.__signs["peaks_pos"])):

                    ax.plot(
                        self.__signs["peaks_pos"][i],
                        self.__signs["peaks_top"][i] + (0.15 * aux_amplitude),
                        marker='v',
                        color='red',
                        markersize=5
                    )
                    ax.axvline(
                        x=self.__signs["peaks_pos"][i],
                        linestyle=":",
                        linewidth=0.5,
                        color="black"
                    )
                    ax.axhline(
                        y=self.__signs["peaks_top"][i],
                        linestyle=":",
                        linewidth=0.5,
                        color="black"
                    )
        for x in x0:
            ax.axvline(x=x, linestyle="-", linewidth=0.5, color="green")
        for y in y0:
            ax.axhline(y=y, linestyle="-", linewidth=0.5, color="green")

        ax.plot(
            self.__time, self.__signal, linewidth=wvf_linewidth, color="black"
        )  # , label="Signal")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # ax.legend()
        return

    # Setters
    @T0.setter
    def T0(self, input):
        self.__t0 = input
        return

    @Signs.setter
    def Signs(self, pack):
        """Method for adding information to self.__signs, complying with the format that
        is specified below. This setter gets one positional argument, pack, which must be
        a tuple or list. It must be made up of three arguments, (key, value, overwrite),
        in this order. These arguments, in turn, must comply with the following format:

        - key (string): Key of the dictionary entry which will be created or updated. The
        provided key must belong to the class variable "nonstackable_aks" or the class
        variable "stackable_aks". Also, depending on the specified key, the provided
        value must comply with some type format. For the time being, the allowed keys and
        its associated-value types are:

            -> "time_unit":                 The type of its value must be a list with one
                                            string. It represents the time unit of the
                                            waveform datapoints. (non stackable)
            -> "signal_magnitude":          The type of its value must be a list with one
                                            string. It represents the name of the signal
                                            magnitude, p.e. "voltage". (non stackable)
            -> "signal_unit":               The type of its value must be a list with one
                                            string. It represents the signal-magnitude unit.
                                            (non stackable)
            -> "integration_ll":            Its value must be a list with one float. It
                                            represents the lower limit for some associated
                                            integration of the waveform. (non stackable)
            -> "integration_ul":            Its value must be a list with one float. It
                                            represents the upper limit for some associated
                                            integration of the waveform. (non stackable)
            -> "first_peak_baseline":       Its value must be a list with one float. It
                                            represents the baseline, in signal units, of
                                            the first peak (in time) within this waveform.
                                            This value may not match the physical baseline
                                            of any other secondary peak within the
                                            acquisition window of this waveform.
                                            (non stackable)
            -> "median_cutoff":             Its value must be a list with one float. The
                                            first-peak baseline is computed as the median
                                            of the signal points which happen below a
                                            certain cutoff-time. That cutoff-time is what
                                            we call the median cutoff. (non stackable)
            -> "peaks_pos":                 Its value must be a list whose elements have
                                            type float. They represent the time position
                                            where a peak has been detected. (stackable)
            -> "peaks_top":                 Its value must be a list whose elements have
                                            type float. They match the signal value at the
                                            detected peak. (stackable)

        - value: For the case of non-stackable keys, value must be a list with length equal to
        one. For the case of stackable keys, value must be a list with whichever length. The
        exact meaning of value depends on the overwrite parameter. See the 'overwrite' parameter
        documentation for more information.

        - overwrite (bool): This parameter entails different behaviour depending on whether we
        are trying to set data for a non-stackable or an stackable key.

            - For a non-stackable key: If true, this method adds an entry to the self.__signs,
            whose key is equal to key, and whose value is equal to value. Overwritting may occur.
            If False, this method searches for the given key in self.__signs. If it is found,
            it is left as it is, i.e. this function changes nothing. If it is not found, it is
            added. I.e. the entry (key:value) is added to self.__signs.

            - For an stackable key: For this case, if overwrite is True, this method adds the
            following entry to the self.__signs: (key:value). Overwritting may occur. If False,
            this function searches whether the given key is already present in self.__signs.
            If it is already there, say (key, previous_value) (where previous_value is a list),
            then this function appends the entries of value to previous_value. If it is not there,
            the function adds the entry (key:value) to self.__signs.

        This setter takes care of ensuring that the entries that are added to self.__signs comply
        with the information above. If the provided key of do not belong to nonstackable_aks nor to
        stackable_aks, or its value do not comply with the expected type, this entry will not be
        added to the dictionary.
        """

        htype.check_type(
            pack,
            tuple,
            list,
            exception_message=htype.generate_exception_message(
                "Waveform.Signs.setter", 1
            ),
        )
        if len(pack) != 3:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("Waveform.Signs.setter", 2)
            )
        key, value, overwrite = pack  # Unpack the arguments
        htype.check_type(
            value,
            list,
            exception_message=htype.generate_exception_message(
                "Waveform.Signs.setter", 3
            ),
        )
        htype.check_type(
            overwrite,
            bool,
            exception_message=htype.generate_exception_message(
                "Waveform.Signs.setter", 4
            ),
        )

        # Key-wise type check
        fIsStackable = False
        if key in ["time_unit", "signal_magnitude", "signal_unit"]:
            for element in value:
                htype.check_type(
                    element,
                    str,
                    exception_message=htype.generate_exception_message(
                        "Waveform.Signs.setter", 5
                    ),
                )
            fIsStackable = False
        elif key in [
            "integration_ll",
            "integration_ul",
            "first_peak_baseline",
            "median_cutoff",
        ]:
            for element in value:
                htype.check_type(
                    element,
                    float,
                    np.float64,
                    exception_message=htype.generate_exception_message(
                        "Waveform.Signs.setter", 6
                    ),
                )
            fIsStackable = False
        elif key in ["peaks_pos", "peaks_top"]:
            for element in value:
                htype.check_type(
                    element,
                    float,
                    np.float64,
                    exception_message=htype.generate_exception_message(
                        "Waveform.Signs.setter", 7
                    ),
                )
            fIsStackable = True
        else:
            # If the key is not recognised, no addition to self.__signs is done
            # This optional addition could have been handle
            return

        if not fIsStackable and len(value) != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("Waveform.Signs.setter", 8)
            )
        if not overwrite and not fIsStackable:
            if key in self.Signs.keys():
                # This is the only case in which
                # nothing is added to self.Signs
                return

        self.Signs.__setitem__(
            key, TypedList(init_members=value), append=not overwrite and fIsStackable
        )
        return

    def integrate(self, integration_lower_lim=None, integration_upper_lim=None):
        """This method integrates the signal of this Waveform object minus its
        baseline, which is considered to match self.__signs['first_peak_baseline'].
        I.e. the waveform baseline must have been computed before calling this
        method, p.e. by calling the Waveform.compute_first_peak_baseline() method.
        The integral is performed using the trapezoid method, and the resulting
        integral is loaded into the self.__integral attribute. The trapezoid
        integration is performed by numpy.trapz. This method gets the following
        optional keyword arguments:

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

        # The input parameters are checked and handled by Waveform.adjust_integration_limits().

        i_low, i_up = Waveform.adjust_integration_limits(
            self.__time,
            ill_candidate=integration_lower_lim,
            iul_candidate=integration_upper_lim,
        )

        self.Signs = ("integration_ll", [self.__time[i_low]], True)
        self.Signs = ("integration_ul", [self.__time[i_up]], True)

        try:
            self.__integral = np.trapz(
                self.__signal[i_low : i_up + 1] - self.Signs["first_peak_baseline"],
                x=self.__time[i_low : i_up + 1],
            )
        except KeyError:
            raise cuex.NoAvailableData(
                htype.generate_exception_message(
                    "Waveform.integrate",
                    1,
                    extra_info="The baseline of the first peak must "
                    "have been computed before calling this method.",
                )
            )

        return self.__integral

    def find_beginning_of_rise(self, tolerance=0.05, return_iterator=True):
        """This method gets the following keyword argument:

        - tolerance (scalar float): It must be positive (>0.0) and
        smaller than 1 (<1.0).
        - return_iterator (scalar boolean): If True, this method
        returns the iterator value, say idx, where the rise begins.
        If False, this method returns self.__time[idx].

        This method iteratively parses every entry in self.__signal
        in increasing order, i.e. from smaller iterator values to
        bigger iterator values. When it encounters a value in
        self.__signal which is bigger than the first-peak-baseline
        up to some tolerance, it stops the iterative process and
        returns the immediately previous iterator value (or the
        self.__time entry for such iterator value, up to
        return_iterator), which points to the first entry within
        self.__signal which does not exceed the first-peak-baseline.

        The tolerance parameter is interpreted as a fraction of
        signal_maximum-self.__signs['first_peak_baseline'][0],
        where signal_maximum is equal to
        np.max(Waveform.filter_infs_and_nans(self.__signal, get_mask=False)),
        which should match the signal value at the peak whose baseline is
        stored at self.__signs['first_peak_baseline'][0]. In this way,
        this function returns the iterator value (or the self.__time
        entry for such iterator value, up to return_iterator) for the
        first point in self.__signal which exceeds
        self.__signs['first_peak_baseline'][0] + ...
        + tolerance*(signal_maximum-self.__signs['first_peak_baseline'][0]).
        Note that the waveform baseline must have been computed
        before calling this method, p.e. by calling the
        Waveform.compute_first_peak_baseline() method.

        As a matter of fact, the result of this method is only
        well-defined for well-defined waveforms, meaning those
        waveforms which are few-photo-electrons waveforms and have
        a signal to noise ratio (SNR) which is smaller than tolerance
        (up to a proper definition of SNR)."""

        htype.check_type(
            tolerance,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "Waveform.find_beginning_of_rise", 1
            ),
        )
        if tolerance <= 0.0 or tolerance >= 1.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "Waveform.find_beginning_of_rise", 2
                )
            )
        htype.check_type(
            return_iterator,
            bool,
            exception_message=htype.generate_exception_message(
                "Waveform.find_beginning_of_rise", 3
            ),
        )

        signal_maximum = np.max(
            Waveform.filter_infs_and_nans(self.__signal, get_mask=False)
        )

        try:
            aux = signal_maximum - self.__signs["first_peak_baseline"][0]
        except KeyError:
            raise cuex.NoAvailableData(
                htype.generate_exception_message(
                    "Waveform.find_beginning_of_rise",
                    4,
                    extra_info="The baseline of the first peak must "
                    "have been computed before calling this method.",
                )
            )

        # This is a cross-check. Up to the computation of
        # first_peak_baseline, aux should be positive.
        # If it is negative, then self is a too-ill-formed
        # waveform and trying to find the rise index of
        # the first peak makes no sense.
        if aux <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "Waveform.find_beginning_of_rise",
                    5,
                    extra_info="This waveform is too ill-formed. Trying to find the rise index of its first peak makes no sense.",
                )
            )
        
        # If the code execution made it to this point, then
        # self.__signs["first_peak_baseline"][0] is defined
        threshold = self.__signs["first_peak_baseline"][0] + (tolerance * aux)

        result_iterator = -2
        for i in range(len(self.__signal)):
            if self.__signal[i] > threshold:
                result_iterator = i - 1
                break

        # If the first point of self.__signal exceeded
        # the threshold, then result_iterator is -1.
        # In this case, return 0 instead of of -1.
        if result_iterator == -1:
            result_iterator = 0

        # If no point surpassing threshold
        # was found in in self.__signal,
        # then result_iterator is still -2.
        if result_iterator == -2:

            # Since threshold belongs to (0.0, 1.0), it is clear that
            # threshold belongs to (first peak baseline, np.max(self.__signal)).
            # Therefore, the previous loop would ultimately result in returning
            # j, where j is the iterator index for the maximum of self.__signal.
            # Hence, if the execution reached this point, then it's because
            # something is not working as expected.
            raise cuex.MalFunction(
                htype.generate_exception_message(
                    "Waveform.find_beginning_of_rise",
                    6,
                    extra_info="Something is not working as expected.",
                )
            )
        else:
            if return_iterator:
                return result_iterator
            else:
                return self.__time[result_iterator]

    @staticmethod
    def adjust_integration_limits(timearray, ill_candidate=None, iul_candidate=None):
        """This static method returns two integer values, say i_low and i_up, so
        that i_low (resp. i_up) is the iterator value for the lower (upper)
        integration limit for a signal whose x-values are contained within
        timearray. I.e. the signal will be potentially integrated in the range
        [timearray[ilow], timearray[iup]]. This static method gets the following
        mandatory positional argument:

        - timearray (sorted unidimensional numpy float array): X-values of the
        signal whose integration limits we want to set.

        This static method also gets two optional keyword arguments:

        - ill_candidate (float): Inclusive upper bound to the lower limit
        of the integration. In order to eventually integrate, the minimum point
        (i.e. the point whose x-value is the minimum) which is used is the point
        with the maximum x-value which still meets x<=ill_candidate.
        ill_candidate must belong to [min, max], where min (resp. max) is the
        smallest (biggest) x-value for which the integrated waveform is defined,
        i.e. the minimum (maximum) value within timearray. If this parameter is
        not set or ill_candidate<min, then the integration lower limit is set to
        min. If ill_candidate>max, then the integration lower limit is set to max.

        - iul_candidate (float): Inclusive lower bound to the upper limit
        of the integration. In order to eventually integrate, the maximum point
        (i.e. the point whose x-value is the maximum) which is used is the point
        with the minimum x-value which still meets x>=iul_candidate.
        iul_candidate must belong to (ill_candidate, max], where max is the
        biggest x-value for which the integrated waveform is defined, i.e.
        the maximum value within timearray. If this parameter is not set or
        iul_candidate>max, then iul_candidate is set to max. If iul_candidate is
        smaller or equal to ill_candidate, then the integration limits returned
        by this method match the two nearest points to ill_candidate within
        timearray.
        """

        htype.check_type(
            timearray,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "Waveform.adjust_integration_limits", 1
            ),
        )
        if np.ndim(timearray) != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "Waveform.adjust_integration_limits", 2
                )
            )
        htype.check_type(
            timearray[0],
            np.float64,
            exception_message=htype.generate_exception_message(
                "Waveform.adjust_integration_limits", 3
            ),
        )
        # This condition is met if the array is not sorted
        if not bool(
            np.prod(np.argsort(timearray) == np.arange(np.shape(timearray)[0]))
        ):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "Waveform.adjust_integration_limits", 4
                )
            )

        if ill_candidate is not None:
            htype.check_type(
                ill_candidate,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "Waveform.adjust_integration_limits", 5
                ),
            )
            ill_candidate_ = Waveform.force_to_range(
                ill_candidate, min=timearray[0], max=timearray[-1]
            )
            ill_idx = (np.abs(timearray - ill_candidate_)).argmin()
        else:
            ill_idx = 0

        if iul_candidate is not None:
            htype.check_type(
                iul_candidate,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "Waveform.adjust_integration_limits", 6
                ),
            )
            iul_candidate_ = Waveform.force_to_range(
                iul_candidate, min=timearray[ill_idx], max=timearray[-1]
            )
            iul_idx = (np.abs(timearray - iul_candidate_)).argmin()
        else:
            iul_idx = np.shape(timearray)[0] - 1

        # Number of points which will be potentially used for integration
        integrated_points = (iul_idx - ill_idx) + 1

        # By definition, integrated_points>=1.
        # If integrated_points==1, then iul_idx==ill_idx
        if integrated_points == 1:

            # Find the entry which is the nearest one to timearray[ill_idx]
            aux = np.delete(timearray, ill_idx)
            aux_idx = (np.abs(aux - timearray[ill_idx])).argmin()

            # The new index, aux_idx, is referred to aux,
            # which is timearray without timearray[ill_idx].
            # We need to correct aux_idx to take this into
            # account, so that it points to the correct entry
            # in the original timearray.
            if aux_idx >= ill_idx:
                iul_idx = aux_idx + 1

            # The second closest entry, pointed to by aux_idx,
            # could have been smaller than timearray[ill_idx]
            else:
                ill_idx, iul_idx = aux_idx, ill_idx
        return ill_idx, iul_idx

    @staticmethod
    def force_to_range(x, min=0.0, max=1.0):
        """This static method gets the following mandatory positional argument:

        - x (int or float): Value to force to the range [min, max].

        This method gets the following optional keyword arguments:

        - min (int or float): Lower limit of the range.
        - max (int or float): Upper limit of the range. Must be greater or equal to min.

        This function returns min if x<=min, it returns x if min<x<max, and it
        returns max if x>=max."""

        htype.check_type(
            x,
            int,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "Waveform.force_to_range", 1
            ),
        )
        htype.check_type(
            min,
            int,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "Waveform.force_to_range", 2
            ),
        )
        htype.check_type(
            max,
            int,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "Waveform.force_to_range", 3
            ),
        )
        if min > max:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("Waveform.force_to_range", 4)
            )
        if x < min:
            return float(min)
        elif x < max:
            return float(x)
        else:
            return float(max)

    @staticmethod
    def filter_infs_and_nans(input, get_mask=False):
        """This static method gets the following mandatory positional
        argument:

        - input (unidimensional float numpy array)

        This static method gets the following optional keyword argument:

        - get_mask (scalar boolean)

        This function retuns an unidimensional numpy array which
        contains all of the entries within input which are not
        equal to either float('nan'), float('inf') or float('-inf').
        If get_mask is True, then this function additionally returns
        a boolean list whose length matches that of input, so that
        masking the input with the specified boolean list gives the
        first output of this method."""

        htype.check_type(
            input,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "Waveform.filter_infs_and_nans", 1
            ),
        )
        if np.ndim(input) != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("Waveform.filter_infs_and_nans", 2)
            )
        if input.dtype != np.float64:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("Waveform.filter_infs_and_nans", 3)
            )
        htype.check_type(
            get_mask,
            bool,
            exception_message=htype.generate_exception_message(
                "Waveform.filter_infs_and_nans", 4
            ),
        )

        mask = [
            not (math.isnan(input[i]) or math.isinf(input[i]))
            for i in range(len(input))
        ]

        if not get_mask:
            return input[mask]
        else:
            return input[mask], mask
        
    # Despite not following the same conventions as the rest of the class,
    # for the methods written below this point I am using the typing module
    # and not calling the htype.check_type(). The reason for this is that
    # at some point I plan to deprecate htype.check_type() and replace it
    # with the usage of the typing module. This will make the code more
    # efficient, standard and readable.
    @staticmethod
    def rebin_array(
            samples: np.ndarray, 
            group: int,
            verbose: bool = False
        ) -> np.ndarray:
        """This function gets the following positional arguments:

        - samples (unidimensional numpy array): The array to re-bin.
        The type of its entries must be so that the np.mean() operation
        is well-defined.
        - group (integer): It must be positive and smaller or equal to
        half of the length of samples. The second condition grants that
        there is at least two entries in the output array.
        - verbose (bool): Whether to print functioning related messages.
        
        This function returns an unidimensional numpy array which is
        computed as follows. To start with, the input array samples is
        trimmed until its length is divisible by 'group'. The division,
        say n = len(trimmed_samples) / group, matches the number of
        entries in the output (re-binned) array. Adjacent entries of
        the trimmed array are grouped up into sets of 'group' entries.
        The i-th entry of the output (re-binned) array is computed as
        the mean of the entries in the i-th group of the trimmed array."""

        if samples.ndim != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    'Waveform.rebin()',
                    92989,
                    extra_info="The given array must have one dimension, "
                    "but an array with a number of dimensions equal to "
                    f"{samples.ndim} was given."
                )
            )

        # Make sure that there is at least
        # 2 points in the resulting array
        if group < 1 or group > len(samples) / 2.:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    'Waveform.rebin()',
                    21628,
                    extra_info=f"The given 'group' ({group}) must be positive "
                    "and smaller or equal to half of the length of the given "
                    f"array (<= {len(samples) / 2.}). The second condition is "
                    "required to make sure that there is at least two "
                    "points in the output arrays."
                )
            )
            
        trimmed_length = len(samples)
        fTrimmed = False
        while trimmed_length % group != 0:
            trimmed_length -= 1
            fTrimmed = True

        if fTrimmed:
            if verbose: 
                print(
                    f"In function Waveform.rebin(): The last "
                    f"{len(samples) - trimmed_length} points were trimmed in "
                    f"order to rebin the input array into groups of {group} "
                    "entries."
                )

            samples_ = samples[:trimmed_length]
        else:
            samples_ = samples

        return np.mean(
            samples_.reshape(
                -1,
                group
            ),
            axis=1
        )
    
    def rebin(
            self,
            group: int,
            verbose: bool = False
        ) -> None:
        """This function gets the following positional arguments:

        - group (integer): It must be positive and smaller or equal to
        half of the length of this waveform, i.e. len(self.__signal) / 2.
        The second condition grants that there is at least two entries
        in the resulting Waveform.
        - verbose (bool): Whether to print functioning related messages.
        
        This function rebins this Waveform object, by calling the static
        method Waveform.rebin_array() two times. First using the attribute
        self.__signal as its first argument, and then using the attribute
        self.__time as its first argument. The second argument of both
        calls is the group parameter. The resulting arrays are stored
        back into self.__signal and self.__time, respectively. For more
        information on the rebinning process, check the documentation of
        Waveform.rebin_array().

        Note that this function affects the following attributes of self:
        self.__npoints, self.__signal and self.__time. Also, contrary to
        what one could think, this function does not affect self.__integral
        nor self.__signs. P.e. one could think that the integration limits
        or the peaks positions stored in self.__signs are iterator values
        referred to self.__time, i.e. referred to the previous binning.
        However, they typically are (or they should be) time values, not
        iterator values. Therefore, they are well-defined even after
        the rebinning."""

        # Well-formedness checks are handled by Waveform.rebin_array()
        self.__time = Waveform.rebin_array(
            self.__time,
            group,
            verbose
        )
        self.__signal = Waveform.rebin_array(
            self.__signal,
            group,
            verbose
        )
        self.__npoints = len(self.__time)

        return