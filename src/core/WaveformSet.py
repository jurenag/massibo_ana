import os
import math
import copy
import datetime as dt
import random
import json
import inspect
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as spsi

import src.utils.htype as htype
import src.utils.search as search
import src.utils.custom_exceptions as cuex
from src.custom_types.OneTypeRTL import OneTypeRTL
from src.core.Waveform import Waveform
from src.preprocess.DataPreprocessor import DataPreprocessor


class WaveformSet(OneTypeRTL):

    def __init__(self, *waveforms, set_name=None, ref_datetime=None):
        """This class aims to model a set of waveforms. It is basically a list of Waveform
        objects with a certain name and reference datetime. This class inherits from
        OneTypeRTL. Check such class documentation for more information. This initializer
        gets the following optional positional arguments:

        - waveforms (an unpacked list of Waveform objects): It must contain at least one
        Waveform object. These are the waveforms that will belong to the initialized instance
        of WaveformSet.

        This initializer gets the following keyword arguments:

        - set_name (string): Name of the set of waveforms. If not provided, the set name
        defaults to "Default".
        - ref_datetime (datetime object): Represents the time reference from which the initial
        time of each waveform is measured (see Waveform.T0). If provided, the datetime is
        appended to the set name.
        """

        self.__set_name_is_available = False
        self.__ref_datetime_is_available = False

        if len(waveforms) < 1:
            raise cuex.NoAvailableData(
                htype.generate_exception_message("WaveformSet.__init__", 10001)
            )

        for i in range(len(waveforms)):
            htype.check_type(
                waveforms[i],
                Waveform,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.__init__", 10002
                ),
            )
        if set_name is not None:
            htype.check_type(
                set_name,
                str,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.__init__", 10003
                ),
            )
            self.__set_name_is_available = True
        if ref_datetime is not None:
            htype.check_type(
                ref_datetime,
                dt.datetime,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.__init__", 10004
                ),
            )
            self.__ref_datetime_is_available = True

        if self.__set_name_is_available:
            aux = set_name
        else:
            aux = "Default"

        if self.__ref_datetime_is_available:
            self.__ref_datetime = ref_datetime
            aux = f"{aux}, on {ref_datetime}"

        super().__init__(Waveform, list_name=aux, init_members=list(waveforms))

        # The set window time-width matches
        # the T0 of the last waveform plus
        # the last-waveform window time-width
        self.__set_time_window = self[-1].T0 + self[-1].Time[-1]

        return

    @property
    def RefDatetime(self):
        if self.__ref_datetime_is_available:
            return self.__ref_datetime
        else:
            raise cuex.NoAvailableData(
                htype.generate_exception_message("WaveformSet.RefDatetime", 20001)
            )

    @property
    def SetTimeWindow(self):
        return self.__set_time_window

    def recompute_first_peak_baseline_of_the_whole_wvfset(
        self, signal_fraction_for_median_cutoff=0.2
    ):
        """This method gets the following optional keyword argument:

        - signal_fraction_for_median_cutoff (scalar float): It must belong to the
        interval [0.0, 1.0]. For each waveform within this WaveformSet object,
        this value represents the fraction of the signal which is used to compute
        the baseline. P.e. 0.2 means that, for each waveform, only its first 20%
        (in time) of the signal is used to compute the baseline.

        This method computes the baseline of the first peak within the waveform
        for each waveform within this WaveformSet object. To do so, this method
        iterates over the whole waveform set and, for each waveform, it populates
        the 'first_peak_baseline' and 'median_cutoff' keys of its self.__signs
        dictionary by calling its Waveform.compute_first_peak_baseline() method,
        overwritting any previous value for such keys, if applicable. For each
        waveform, the computed value may not match the physical baseline for any
        other secondary peak within the acquisition window of this waveform.
        For more information, check the Waveform.compute_first_peak_baseline()
        docstring.

        The reason why only an initial (in time) fraction of the signal is used to
        compute the baseline is that the rest of the signal may be affected by the
        undershoot of the first peak. In this context, the median of a signal which
        is affected by a deep long undershoot would result in a baseline which
        is biased towards smaller values."""

        htype.check_type(
            signal_fraction_for_median_cutoff,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.recompute_first_peak_baseline_of_the_whole_wvfset", 30001
            ),
        )
        if (
            signal_fraction_for_median_cutoff < 0.0
            or signal_fraction_for_median_cutoff > 1.0
        ):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.recompute_first_peak_baseline_of_the_whole_wvfset",
                    30002,
                )
            )
        for wvf in self:
            wvf.compute_first_peak_baseline(
                signal_fraction_for_median_cutoff=signal_fraction_for_median_cutoff
            )
        return

    def mean_waveform(self, amplitude_range=None, integral_range=None):
        """This method gets the following optional keyword arguments:

        - amplitude_range (resp. integral_range) (list of two floats):
        amplitude_range[0] (resp. integral_range[0]) must be smaller
        than amplitude_range[1] (resp. integral_range[1]). The amplitude
        is computed as the maximum value of the signal minus its
        baseline, i.e. for a waveform wvf, its
        wvf.Signs['first_peak_baseline'][0].

        Either amplitude_range or integral_range must be defined. If
        both are defined, then integral_range is ignored. If
        amplitude_range is defined, then amplitude_range[0]
        (resp. amplitude_range[1]) is interpreted as the minimum
        (resp. maximum) value of the amplitude range. If integral_range
        is defined and amplitude_range is not, then integral_range[0]
        (resp. integral_range[1]) is interpreted as the minimum (resp.
        maximum) value of the integral range.

        Once an amplitude (resp. integral) range is defined, the mean
        waveform of all of the waveforms whose amplitude (resp. integral)
        fall into such range is computed. A necessary requirement for
        this computation to be performed is that all of the waveforms
        within this WaveformSet object have matching x-values, i.e.
        that self.__time is the same for all of the waveforms.

        When calling this method, this waveform set must contain at least
        one waveform. If not, this method raises a cuex.NoAvailableData
        exception."""

        if len(self) == 0:
            raise cuex.NoAvailableData(
                htype.generate_exception_message(
                    "WaveformSet.mean_waveform",
                    40001,
                    extra_info="There must be at least one waveform in this waveform set.",
                )
            )
        fAmplitudeIsDefined = False
        if amplitude_range is not None:
            htype.check_type(
                amplitude_range,
                list,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.mean_waveform", 40002
                ),
            )
            if len(amplitude_range) != 2:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.mean_waveform", 40003)
                )
            for i in range(len(amplitude_range)):
                htype.check_type(
                    amplitude_range[i],
                    float,
                    np.float64,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.mean_waveform", 40004
                    ),
                )
            if amplitude_range[0] >= amplitude_range[1]:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.mean_waveform", 40005)
                )
            fAmplitudeIsDefined = True

        if not fAmplitudeIsDefined:
            if integral_range is not None:
                htype.check_type(
                    integral_range,
                    list,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.mean_waveform", 40006
                    ),
                )
                if len(integral_range) != 2:
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message(
                            "WaveformSet.mean_waveform", 40007
                        )
                    )
                for i in range(len(integral_range)):
                    htype.check_type(
                        integral_range[i],
                        float,
                        np.float64,
                        exception_message=htype.generate_exception_message(
                            "WaveformSet.mean_waveform", 40008
                        ),
                    )
                if integral_range[0] >= integral_range[1]:
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message(
                            "WaveformSet.mean_waveform", 40009
                        )
                    )
            else:
                raise cuex.NoAvailableData(
                    htype.generate_exception_message(
                        "WaveformSet.mean_waveform",
                        40010,
                        extra_info=f"Either amplitude_range or integral_range must be defined.",
                    )
                )
        if len(self) < 1:
            raise cuex.NoAvailableData(
                htype.generate_exception_message(
                    "WaveformSet.mean_waveform",
                    40011,
                    extra_info=f"There are no waveforms in this waveform set.",
                )
            )
        aux = self[0].Time
        for i in range(1, len(self)):
            if not np.array_equal(aux, self[i].Time):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "WaveformSet.mean_waveform",
                        40012,
                        extra_info=f"The {i}-th waveform time array does not match that of the 0-th waveform within the waveform set.",
                    )
                )
        filtered_wvfs_idx = []

        if fAmplitudeIsDefined:
            for i in range(len(self)):
                ith_amplitude = (
                    np.max(self[i].Signal) - self[i].Signs["first_peak_baseline"][0]
                )
                if (
                    ith_amplitude >= amplitude_range[0]
                    and ith_amplitude <= amplitude_range[1]
                ):
                    filtered_wvfs_idx.append(i)
        else:
            for i in range(len(self)):
                # If self[i] had not been integrated previously,
                # then self[i].Integral will be None
                ith_integral = self[i].Integral

                try:
                    if (
                        ith_integral >= integral_range[0]
                        and ith_integral <= integral_range[1]
                    ):
                        filtered_wvfs_idx.append(i)

                # This is what happens if you try to evaluate if None
                # is smaller, equal or bigger than an scalar float.
                except TypeError:
                    raise cuex.NoAvailableData(
                        htype.generate_exception_message(
                            "WaveformSet.mean_waveform",
                            40013,
                            extra_info=f"The integral of the {i}-th waveform could not be retrieved.",
                        )
                    )
        if len(filtered_wvfs_idx) == 0:
            raise cuex.NoAvailableData(
                htype.generate_exception_message(
                    "WaveformSet.mean_waveform",
                    40014,
                    extra_info=f"There are no waveforms which comply with the given {'amplitude' if fAmplitudeIsDefined else 'integral'} range.",
                )
            )

        # We have made sure that there's at
        # least one waveform in this waveform set
        sum = copy.copy(self[filtered_wvfs_idx[0]].Signal)
        for i in range(1, len(filtered_wvfs_idx)):
            sum += self[filtered_wvfs_idx[i]].Signal
        return sum / len(filtered_wvfs_idx)

    def plot(
        self,
        wvfs_to_plot=None,
        plot_peaks=True,
        randomize=False,
        fig_title=None,
        mode="grid",
        nrows=2,
        ncols=2,
        xlim=None,
        ylim=None,
        wvf_linewidth=1.0,
        x0=[],
        y0=[],
    ):
        """This method gets the following keyword arguments:

        - wvfs_to_plot (None, int or list of integers): If this parameter
        is None, this function plots every waveform in the dataset. If it
        is an integer value, say N, this function plots N waveforms in the
        set. If this parameter is a list of integers, this function plots
        the waveforms whose iterator value within the set matches any
        integer within wvfs_to_plot.
        - plot_peaks (boolean): For each waveform which should be plotted,
        up to wvfs_to_plot, this parameter is given to the 'plot_peaks'
        parameter of its Waveform.plot() method. If True, and there are
        any peaks position available within its Waveform.Signs attribute
        (under the 'peaks_pos' and 'peaks_top' keys), then such positions
        are plotted. If else, no peaks positions are plotted.
        - randomize (boolean): This parameter only makes a difference
        if wvfs_to_plot is an integer. If so, and randomize is False, then
        this function plots the first wvfs_to_plot waveforms from the set,
        following the order they have within the set. In this case, but
        with randomize set to True, then this function plots wfts_to_plot
        randomly chosen waveforms from the set.
        - fig_title (string): The title of the figure.
        - mode (string): It must be either 'grid' or 'superposition'. Any
        other input will be understood as 'grid'. If it matches 'grid',
        then each waveform is plot in an exlcusive pair of axes, i.e. one
        waveform per pair of axes. Those axes make up a 2D-grid of
        nrows*ncols axes, where there are nrows rows and ncols columns.
        If mode matches 'superposition', then each waveform is plotted
        in the same pair of axes.
        - nrows (resp. ncols) (integer): Number of rows (resp. cols) in the
        plot grid. It must be greater or equal to two.
        - xlim (resp. ylim) (None or a tuple of two floats): For each
        selected waveform, xlim (resp. ylim) is given to the 'xlim' (resp.
        'ylim') parameter of its Waveform.plot method. If it is not None,
        then its first element gives the minimum x-value (resp. y-value)
        displayed in the plot, while the second element gives the maximum
        x-value (resp. y-value) displayed in the plot. The first element
        must be bigger than the second one.
        - wvf_linewidth (positive float): It is given to the 'wvf_linewidth'
        parameter of the Waveform.plot() method in every one of its calls.
        It is eventually passed to the 'linewidth' parameter of
        matplotlib.axes.Axes.plot when plotting the waveform. It tunes the
        width of the plot line.
        - x0 (resp. y0) (list of floats): It is given to the x0 (resp. y0)
        keyword argument of Waveform.plot(). Each selected waveform is
        plotted along with a set of vertical (resp. horizontal) lines at
        the positions given by the entries in this list."""

        htype.check_type(
            plot_peaks,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.plot", 50001
            ),
        )
        htype.check_type(
            randomize,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.plot", 50002
            ),
        )
        if wvfs_to_plot is None:
            wvfs_indices = tuple(range(0, self.__len__()))
        elif type(wvfs_to_plot) == int:
            if wvfs_to_plot < 1 or wvfs_to_plot > self.__len__():
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.plot", 50003)
                )
            if randomize:
                wvfs_indices = tuple(
                    WaveformSet.get_random_indices(self.__len__(), wvfs_to_plot)
                )
            else:
                wvfs_indices = tuple(range(0, wvfs_to_plot))
        elif type(wvfs_to_plot) == list:
            for i in range(len(wvfs_to_plot)):
                htype.check_type(
                    wvfs_to_plot[i],
                    int,
                    np.int64,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.plot", 50004
                    ),
                )
                if wvfs_to_plot[i] < 0 or wvfs_to_plot[i] >= self.__len__():
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message("WaveformSet.plot", 50005)
                    )
            wvfs_indices = tuple(set(wvfs_to_plot))  # Purge matching entries
        else:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.plot", 50006)
            )
        if fig_title is not None:
            htype.check_type(
                fig_title,
                str,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.plot", 50007
                ),
            )
        htype.check_type(
            mode,
            str,
            exception_message=htype.generate_exception_message(
                "WaveformSet.plot", 50008
            ),
        )
        htype.check_type(
            nrows,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.plot", 50009
            ),
        )
        if nrows < 2:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.plot", 50010)
            )
        htype.check_type(
            ncols,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.plot", 50011
            ),
        )
        if ncols < 2:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.plot", 50012)
            )
        if xlim is not None:
            htype.check_type(
                xlim,
                tuple,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.plot", 50013
                ),
            )
            if len(xlim) != 2:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.plot", 50014)
                )
            for aux in xlim:
                htype.check_type(
                    aux,
                    float,
                    np.float64,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.plot", 50015
                    ),
                )
            if xlim[0] >= xlim[1]:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.plot", 50016)
                )
        if ylim is not None:
            htype.check_type(
                ylim,
                tuple,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.plot", 50017
                ),
            )
            if len(ylim) != 2:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.plot", 50018)
                )
            for aux in ylim:
                htype.check_type(
                    aux,
                    float,
                    np.float64,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.plot", 50019
                    ),
                )
            if ylim[0] >= ylim[1]:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.plot", 50020)
                )
        htype.check_type(
            wvf_linewidth,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.plot", 50021
            ),
        )
        if wvf_linewidth <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.plot", 50022)
            )
        htype.check_type(
            x0,
            list,
            exception_message=htype.generate_exception_message(
                "WaveformSet.plot", 50023
            ),
        )
        for elem in x0:
            htype.check_type(
                elem,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.plot", 50024
                ),
            )
        htype.check_type(
            y0,
            list,
            exception_message=htype.generate_exception_message(
                "WaveformSet.plot", 50025
            ),
        )
        for elem in y0:
            htype.check_type(
                elem,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.plot", 50026
                ),
            )

        if mode != "superposition":  # Grid plot is default
            counter = 0
            how_many_canvases = int(math.ceil(len(wvfs_indices) / (nrows * ncols)))
            for i in range(how_many_canvases):
                index_generator = WaveformSet.index_generator(nrows, ncols)
                fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
                for j in range(nrows * ncols):
                    iterator = next(index_generator)
                    try:
                        self.Members[wvfs_indices[counter]].plot(
                            axs[iterator],
                            xlim=xlim,
                            ylim=ylim,
                            plot_peaks=plot_peaks,
                            wvf_linewidth=wvf_linewidth,
                            x0=x0,
                            y0=y0,
                        )
                        axs[iterator].set_title(f"Nº {wvfs_indices[counter]}")
                        WaveformSet.set_custom_labels_visibility(
                            axs[iterator], iterator[0], iterator[1], nrows
                        )

                    # If the number of waveforms to plot is not a multiple
                    # of nrows*ncols, in the last canvas (last value for i
                    # in the outer loop), an IndexError will raise when
                    # there are no more waveforms to plot, but there are
                    # some free spots in the plot-grid yet. Just handle it
                    # by breaking out of the inner loop, and let the outer
                    # loop naturally finish.
                    except IndexError:
                        break
                    counter += 1
                fig.suptitle(fig_title)
                fig.tight_layout()
                plt.show(block=False)
                input("Press any key to iterate to next canvas...")
                plt.close()
        else:
            fig, ax = plt.subplots()
            for i in wvfs_indices:
                self.Members[i].plot(
                    ax,
                    xlim=xlim,
                    ylim=ylim,
                    plot_peaks=plot_peaks,
                    wvf_linewidth=wvf_linewidth,
                    x0=x0,
                    y0=y0,
                )
            fig.suptitle(fig_title)
            plt.show(block=False)
            input("Press any key to exit")
            plt.close()
        return

    @staticmethod
    def set_custom_labels_visibility(ax, ax_i, ax_j, nrows):
        """This static method gets the following mandatory positional arguments:

        - ax (matplotlib.axes.Axes object)
        - nrows (integer): Number of axes along the vertical direction of the axes
        grid which the given axis (ax) belongs to.
        - ax_i (resp. ax_j) (integer): First (second) coordinate of the position
        of the given axis (ax) within the axes grid which ax belongs to.

        For every given axis, ax, the x-ticks and y-ticks labels visibility is set
        to True by this method. For the x-axis and y-axis labels, this method sets
        the visibility to True or False depending on its position within the axes
        grid. Such visibility is set to True if the given axis, ax, belongs to the
        first column or the last row of the axes grid. It is set to False otherwise.
        *Note that this method is similar to matplotlib.axes.Axes.label_outer.
        Indeed, as far as x-axis and y-axis labels are concerned, the behaviour
        of this static method and matplotlib.axes.Axes.label_outer is the same.
        However, matplotlib.axes.Axes.label_outer also makes the ticks labels
        invisible for inner axes. However, we would like to see ticks labels on
        every axis within the axes plot, though."""

        htype.check_type(
            ax_i,
            int,
            exception_message=htype.generate_exception_message(
                "WaveformSet.set_custom_labels_visibility", 60001
            ),
        )
        htype.check_type(
            ax_j,
            int,
            exception_message=htype.generate_exception_message(
                "WaveformSet.set_custom_labels_visibility", 60002
            ),
        )
        htype.check_type(
            nrows,
            int,
            exception_message=htype.generate_exception_message(
                "WaveformSet.set_custom_labels_visibility", 60003
            ),
        )
        if nrows < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.set_custom_labels_visibility", 60004
                )
            )
        if ax_i < 0 or ax_i >= nrows:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.set_custom_labels_visibility", 60005
                )
            )
        if ax_j < 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.set_custom_labels_visibility", 60006
                )
            )
        if ax_j > 0:
            ax.yaxis.set_label_text("")
        if ax_i < nrows - 1:
            ax.xaxis.set_label_text("")
        return

    @staticmethod
    def get_random_indices(i_max, how_many_samples):
        """This function gets the following positional arguments:

        - i_max (integer): Upper bound to the integer which might be randomly
        sampled. This upper limit is exclusive, i.e. all of the samples will
        belong to [0, i_max).
        - how_many_samples (integer): Number of samples. This integer must
        belong to [0, i_max].

        This function returns a list of how_many_samples integers. The entries
        within this list belong to [0, i_max) and are not repeated. I.e. any
        entry within the returned list is unique. This is why how_many_samples
        cannot exceed i_max.
        """

        htype.check_type(
            i_max,
            int,
            exception_message=htype.generate_exception_message(
                "WaveformSet.get_random_indices", 70001
            ),
        )
        if i_max < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.get_random_indices", 70002
                )
            )
        htype.check_type(
            how_many_samples,
            int,
            exception_message=htype.generate_exception_message(
                "WaveformSet.get_random_indices", 70003
            ),
        )
        if how_many_samples < 0 or how_many_samples > i_max:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.get_random_indices", 70004
                )
            )

        result = []
        sample_space = list(range(0, i_max))
        for i in range(how_many_samples):
            sampled_index = random.randint(0, len(sample_space) - 1)
            result.append(sample_space.pop(sampled_index))
        return result

    @staticmethod
    def index_generator(i_max, j_max):
        """This static method gets the following positional arguments:

        - i_max (int): Integer value in [1, +\infty)
        - j_max (int): Integer value in [1, +\infty]

        This function returns a generator. This object generates iterator
        values (tuples) for unidimensional or bidimensional arrays. If i_max
        (resp. j_max) is equal to one, an unidimensional-iterator generator is
        returned. Such generator runs from 0 to j_max-1 (resp. i_max-1). If
        none of them are equal to one, a bidimensional-iterator generator is
        returned, which takes values in [0, i_max-1]x[0, j_max-1]."""

        htype.check_type(
            i_max,
            int,
            exception_message=htype.generate_exception_message(
                "WaveformSet.index_generator", 80001
            ),
        )
        if i_max < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.index_generator", 80002)
            )
        htype.check_type(
            i_max,
            int,
            exception_message=htype.generate_exception_message(
                "WaveformSet.index_generator", 80003
            ),
        )
        if j_max < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.index_generator", 80004)
            )
        if i_max == 1:
            return WaveformSet.unidimensional_index_generator(j_max)
        if j_max == 1:
            return WaveformSet.unidimensional_index_generator(i_max)
        return WaveformSet.bidimensional_index_generator(i_max, j_max)

    @staticmethod
    def unidimensional_index_generator(k_max):
        for k in range(k_max):
            yield k

    @staticmethod
    def bidimensional_index_generator(i_max, j_max):
        for j in range(j_max):
            for i in range(i_max):
                yield (i, j)

    @classmethod
    def from_files(
        cls,
        input_filepath,
        time_resolution,
        points_per_wvf,
        timestamp_filepath=None,
        headers_end_identifier="TIME,",
        data_delimiter=",",
        delta_t_wf=None,
        set_name=None,
        ref_datetime=None,
        creation_dt_offset_min=None,
        wvf_extra_info=None,
    ):
        """This class method is meant to be an alternative initializer.
        It creates a WaveformSet out of an ASCII or binary (in the 
        Tektronix WFM format) input file. Optionally, in the case of 
        an ASCII input file, this initializer could make use of another 
        file which stores the timestamp of the waveforms. This class 
        method gets the following mandatory positional arguments:

        - input_filepath (string): File from which to read the waveforms.
        It is given to the WaveformSet.read_wvfs() method. Its extension
        must match either '.csv', '.txt', '.dat' or '.wfm'. For more
        information, check the WaveformSet.read_wvfs() method documentation.
        - time_resolution (float): It is interpreted as the time step
        between two consecutive points of a waveform in seconds.
        It must be a positive float.
        - points_per_wvf (int): The expected number of points per 
        waveform in the input filepath. It must be a positive integer.
        It is given to the 'points_per_wvf' positional argument of the
        WaveformSet.read_wvfs() method.

        This method also takes the following optional keyword arguments:

        - timestamp_filepath (string): It is given to the 'timestamp_filepath'
        parameter of WaveformSet.read_wvfs(). This parameter only makes a 
        difference if the input file is ASCII. It is the file path to the 
        file which hosts an ASCII time stamp (its extension must match either 
        '.txt', '.csv' or '.dat') of the waveforms which are hosted in 
        the file whose path is given by input_filepath. The i-th entry of 
        this file is considered to be the initial time of ocurrence of the 
        i-th waveform measured with respect to the initial time of ocurrence 
        of the (i-1)-th waveform. It is assumed that all of the entries in 
        the timestamp are broadcastable to an strictly positive float.
        - headers_end_identifier (string): This parameter is given to the 
        'headers_end_identifier' parameter of the WaveformSet.read_wvfs()
        method. It is a string which is used to find the end of the headers
        in the input file, in case an ASCII file is provided. For more 
        information, check the documentation of the 'headers_end_identifier'
        parameter of the WaveformSet.read_wvfs().
        - data_delimiter (string): This parameter is given to the 
        'data_delimiter' parameter of the WaveformSet.read_wvfs() method.
        For more information, check such parameter documentation in the 
        WaveformSet.read_wvfs() docstring.
        - delta_t_wf (float): It is given to the delta_t_wf parameter of
        WaveformSet.read_wvfs(). This parameter makes a difference only 
        if the input file is ASCII and the timestamp filepath is not available. 
        In this case, this is considered to be the time increment in 
        between the triggers of two consecutive waveforms. This is helpful 
        for reading waveform sets which were measured using a trigger on 
        a periodic external signal. Then, delta_t_wf can be set to the period 
        of such external signal without needing to provide a complete text 
        file with a time stamp. It must be a positive float.
        - set_name (string): It is passed to cls.__init__ as set_name.
        - ref_datetime (datetime): It is passed to cls.__init__ as
        ref_datetime. This parameter is interpreted as the reference
        time from which the waveforms initial time are measured. If it 
        is not provided, it is set to the creation datetime of the file 
        whose path is input_filepath.
        - creation_dt_offset_min (float): It is assumed to be a quantity
        in minutes. If ref_datetime is None (resp. not None), then the 
        ref_datetime of the created WaveformSet is the creation datetime 
        of the input file (resp. the given reference datetime) plus the 
        provided creation_df_offset.
        - wvf_extra_info (string): Path to a json file. This file is read to 
        a dictionary, which is then passed to the __signs attribute of every 
        Waveform object in the WaveformSet. For more information on which keys 
        should you use when building the json file, see the Waveform class 
        documentation.

        For the case of an ASCII input file, at least one of [timestamp_filepath, 
        delta_t_wf] must be different from None. In other case, there's not 
        enough information to write the keys of the goal dictionary.
        This class method returns a WaveformSet and the acquisition time of
        such waveform set, in minutes. To do so, it assumes that the acquisition
        time gotten from WaveformSet.read_wvfs() is expressed in seconds.
        """

        # Although this parameter is later passed to the
        # read_wvfs() method, it is better checked here
        # because it will be used by os.path.getctime()
        # if ref_datetime is None.
        htype.check_type(
            input_filepath,
            str,
            exception_message=htype.generate_exception_message(
                "WaveformSet.from_files", 1
            ),
        )
        if not os.path.isfile(input_filepath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "WaveformSet.from_files",
                    2,
                    extra_info=f"Path {input_filepath} does not exist or is not a file.",
                )
            )
        htype.check_type(
            time_resolution,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.from_files", 3
            ),
        )
        if time_resolution <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.from_files", 4)
            )
        if set_name is not None:
            htype.check_type(
                set_name,
                str,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.from_files", 5
                ),
            )
        if ref_datetime is not None:
            htype.check_type(
                ref_datetime,
                dt.datetime,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.from_files", 6
                ),
            )
            ref_datetime_ = ref_datetime
        else:
            # Seconds passed between the epoch
            # and the creation of the input file
            ref_datetime_ = os.path.getctime(input_filepath)
            ref_datetime_ = dt.datetime.fromtimestamp(ref_datetime_)

        if creation_dt_offset_min is not None:
            htype.check_type(
                creation_dt_offset_min,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.from_files", 7
                ),
            )
            # Offset creation datetime
            minutes_offset = int(creation_dt_offset_min)
            seconds_offset = int((creation_dt_offset_min - minutes_offset) * 60)
            ref_datetime_ = ref_datetime_ + dt.timedelta(
                minutes=minutes_offset, seconds=seconds_offset
            )

        if wvf_extra_info is not None:
            htype.check_type(
                wvf_extra_info,
                str,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.from_files", 8
                ),
            )
            with open(wvf_extra_info, "r") as file:
                extra_info_ = json.load(file)
        else:
            extra_info_ = None

        timestamps, waveforms, additional_dict = WaveformSet.read_wvfs(
            input_filepath,
            points_per_wvf,
            timestamp_filepath=timestamp_filepath,
            headers_end_identifier=headers_end_identifier,
            data_delimiter=data_delimiter,
            delta_t_wf=delta_t_wf,
        )

        waveforms_pack = []

        for i in range(len(timestamps)):
            signal_holder = waveforms[i * points_per_wvf:(i + 1) * points_per_wvf]
            waveform_holder = Waveform(
                timestamps[i],
                signal_holder,
                t_step=time_resolution,
                signs=extra_info_,
            )
            waveforms_pack.append(waveform_holder)

        # Assuming that acquisition_time is expressed in seconds
        return cls(
            *waveforms_pack, 
            set_name=set_name, 
            ref_datetime=ref_datetime_), additional_dict["acquisition_time"] / 60.

    @staticmethod
    def read_wvfs(
        input_filepath,
        points_per_wvf,
        timestamp_filepath=None,
        headers_end_identifier="TIME,",
        data_delimiter=",",
        delta_t_wf=None,
    ):
        """This method takes the following mandatory positional argument:

        -  input_filepath (string): File from which to read the waveforms.
        If its extension matches either '.txt', '.csv' or '.dat', it will 
        be interpreted as an ASCII file. If its extension matches '.wfm',
        it will be interpreted as a binary file in the Tektronix WFM format.
        If any other extension is found, an exception is raised. 
        - points_per_wvf (int): The expected number of points per 
        waveform in the input filepath. It must be a positive integer.

        This method also takes the following optional keyword arguments:

        - timestamp_filepath (string): This parameter only makes a difference
        if the input file is ASCII. File path to the file which hosts a
        an ASCII time stamp (its extension must match either '.txt', '.csv' 
        or '.dat') of the waveforms which are hosted in input_filepath. 
        The i-th entry of this file is considered to be the initial time of
        ocurrence of the i-th waveform measured with respect to the initial
        time of ocurrence of the (i-1)-th waveform. It is assumed that all 
        of the entries in the timestamp are broadcastable to an strictly 
        positive float. For more information on the expected format for this
        file, check the WaveformSet._extract_core_data() method documentation
        for the case of file_type_code equal to 1.
        - headers_end_identifier (string): This parameter only makes a
        difference if the input file is ASCII. In such case, this parameter 
        is given to the 'identifier' parameter of the 
        DataPreprocessor.find_skiprows() method. This method is used to
        find the line in which the headers end in the input file, i.e. the
        number of rows which should be skipped by the 
        WaveformSet._extract_core_data() method. Additionally, if the
        timestamp_filepath parameter is defined, then headers_end_identifier 
        is also used by DataPreprocessor.find_skiprows() to find a suitable 
        skip-rows value for the time stamp file.
        - data_delimiter (string): This parameter only makes a
        difference if the input file is ASCII. This parameter is given to 
        the 'data_delimiter' parameter of the WaveformSet._extract_core_data()
        method for the case of ASCII input files (waveforms input file and
        time stamp file, if applicable). In turn, such method gives it to
        to numpy.loadtxt() as delimiter. It is used to separate entries of 
        the different columns of the input files.
        - delta_t_wf (float): This parameter makes a difference only if the
        input file is ASCII and the timestamp filepath is not available. In 
        this case, this is considered to be the time increment in between 
        the triggers of two consecutive waveforms. This is helpful for 
        reading waveform sets which were measured using a trigger on a 
        periodic external signal. Then, delta_t_wf can be set to the period 
        of such external signal without needing to provide one time stamp 
        per waveform. It must be a positive float.

        For the case of an ASCII input file, at least one of [timestamp_filepath, 
        delta_t_wf] must be different from None. In other case, there's not 
        enough information to write the keys of the goal dictionary.
        This class method returns a WaveformSet.

        This function returns three parameters, in the following order:

        - An unidimensional numpy array with the timestamps of the waveforms,
        where the i-th entry is the time increment of the i-th waveform trigger
        with respect to the first waveform trigger.
        - An unidimensional numpy array with the concatenation of the
        waveforms
        - A dictionary which contains two entries:
            - 'average_delta_t_wf': The average of the time differences 
            between consecutive triggers in the waveforms dataset.
            - 'acquisition_time': The time difference between the last 
            trigger and the first trigger of the waveforms dataset.
        """

        # input_filepath is checked in the WaveformSet.from_files() method

        _, extension = os.path.splitext(input_filepath)

        if extension not in (".txt", ".csv", ".dat", ".wfm"):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.read_wvfs", 1)
            )

        fIsBinary = False
        if extension == ".wfm":
            fIsBinary = True

        htype.check_type(
            points_per_wvf,
            int,
            exception_message=htype.generate_exception_message(
                "WaveformSet.read_wvfs", 2
            ),
        )
        if points_per_wvf < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.read_wvfs", 3)
            )

        fUseTStamp = False
        if not fIsBinary:
            if timestamp_filepath is not None:
                htype.check_type(
                    timestamp_filepath,
                    str,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.read_wvfs", 4
                    ),
                )
                if not os.path.isfile(timestamp_filepath):
                    raise FileNotFoundError(
                        htype.generate_exception_message(
                            "WaveformSet.read_wvfs",
                            5,
                            extra_info=f"Path {timestamp_filepath} does not exist or is not a file.",
                        )
                    )

                _, ts_extension = os.path.splitext(timestamp_filepath)

                if ts_extension not in (".txt", ".csv", ".dat"):
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message("WaveformSet.read_wvfs", 6)
                    )

                fUseTStamp = True

            if not fUseTStamp:  # In this case, delta_t_wf must be available
                if delta_t_wf is None:
                    raise cuex.InsufficientParameters(
                        htype.generate_exception_message("WaveformSet.read_wvfs", 7)
                    )
                htype.check_type(
                    delta_t_wf,
                    float,
                    np.float64,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.read_wvfs", 8
                    ),
                )
                if delta_t_wf <= 0.0:
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message("WaveformSet.read_wvfs", 9)
                    )

        # These parameters are only used in the ASCII case
        if not fIsBinary:   

            htype.check_type(
                headers_end_identifier,
                str,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.read_wvfs", 10
                ),
            )
            htype.check_type(
                data_delimiter,
                str,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.read_wvfs", 11
                ),
            )

        if not fIsBinary:

            headers_endline = DataPreprocessor.find_skiprows(
                input_filepath, 
                headers_end_identifier,
            )

            waveforms = WaveformSet._extract_core_data(
                input_filepath,
                0,
                skiprows=headers_endline,
                data_delimiter=data_delimiter,
            )

            if len(waveforms) % points_per_wvf != 0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "WaveformSet.read_wvfs", 
                        12,
                        extra_info=f"The number of data-points in the concatenation "
                        f"of waveforms ({len(waveforms)}) must be a multiple of "
                        f"points_per_wvf ({points_per_wvf}).",
                    )
                )
            
            waveforms_no = int(len(waveforms) // points_per_wvf)

            if fUseTStamp:

                headers_endline = DataPreprocessor.find_skiprows(
                    timestamp_filepath,
                    headers_end_identifier,
                )

                timestamps, additional_dict = \
                    WaveformSet._extract_core_data(
                        timestamp_filepath,
                        1,
                        skiprows=headers_endline,
                        data_delimiter=data_delimiter,
                    )

                # In WaveformSet._extract_core_data(), it is
                # assumed that the i-th entry of the time stamp
                # is the time increment of the i-th waveform trigger
                # with respect to the (i-1)-th waveform trigger.
                # However, what WaveformSet.read_wvfs() should
                # return is the time increment of the i-th waveform
                # trigger with respect to the first waveform trigger.
                timestamps = np.cumsum(timestamps)

            else:
                timestamps = np.arange(
                    0., 
                    waveforms_no * delta_t_wf,
                    delta_t_wf,
                    )
                
                additional_dict = {
                    "average_delta_t_wf": delta_t_wf,
                    "acquisition_time": waveforms_no * delta_t_wf,
                }
        else:

            parameters, supplementary_extraction = (
                DataPreprocessor._extract_tek_wfm_metadata(input_filepath)
            )
            tek_wfm_metadata = parameters | supplementary_extraction

            timestamps, waveforms, additional_dict = \
                WaveformSet._extract_core_data(
                    input_filepath,
                    2,
                    tek_wfm_metadata=tek_wfm_metadata,
                )
            
            timestamps = np.cumsum(timestamps)

        return timestamps, waveforms, additional_dict

    @staticmethod
    def _extract_core_data(
        filepath,
        file_type_code,
        skiprows=0,
        data_delimiter=",",
        tek_wfm_metadata=None,
    ):
        """This static method is a helper method which should only be called 
        by the WaveformSet.read_wvfs() method. It gets the following mandatory 
        positional argument:

        - filepath (string): Path to the file whose data will be extracted.

        - file_type_code (scalar integer): It must be either 0, 1 or 2.
        This integer indicates the type of file which should be processed.
        0 matches an ASCII waveform dataset, 1 matches an ASCII timestamp
        and 2 matches a binary (Tektronix WFM file format).

        This function also gets the following optional keyword arguments:

        - skiprows (integer scalar): This parameter is only used for the
        case of ASCII input files, i.e. either file_type_code is 0 or 1.
        It indicates the number of rows to skip in the input file in order
        to access the core data. It must be a semi-positive integer.

        - data_delimiter (string): This parameter is only used for the case
        of ASCII input files, i.e. either file_type_code is 0 or 1. In such
        case, it is given to numpy.loadtxt() as delimiter, which in turn,
        uses it to separate entries of the different columns of the input file.

        - tek_wfm_metadata (None or dictionary): This parameter is only used
        for the case of binary input files, i.e. file_type_code is 2. In such 
        case, it must be defined. It is a dictionary containing the metadata 
        of the provided input file. It must be the union of the two dictionaries 
        returned by DataPreprocessor._extract_tek_wfm_metadata(). For more 
        information on the keys which these dictionaries should contain, check 
        such method documentation.

        If file_type_code is 0, then this method returns an unidimensional
        numpy array which contains the waveform dataset. I.e. this array is
        the result of concatenating all of the waveforms in the input file.
        If file_type_code is 1, then this method returns a unidimensional 
        numpy array which contains the time stamp of the waveforms. If 
        file_type_code is 2, then this method returns two unidimensional 
        numpy arrays. The first one contains the time stamp of the waveforms. 
        The second one contains the concatenation of the waveforms. If 
        file_type_code is 1 or 2, then this method returns an additional 
        dictionary which contains two entries. The first one is saved under 
        the key 'average_delta_t_wf' and is the average of the time differences 
        between consecutive triggers in the waveforms dataset. The second one 
        is saved under the key 'acquisition_time' and is the acquisition time 
        of the waveform dataset which is stored in the provided input file. 
        By acquisition time we refer to the time difference between the last 
        trigger and the first trigger of the waveforms dataset stored in the 
        provided filepath. It is assumed that the i-th entry of the time 
        stamp is time increment of the i-th waveform trigger with respect to 
        the (i-1)-th waveform trigger. Therefore, the acquisition time is 
        computed as the sum of all of the entries of the time stamp."""

        # filepath and data_delimiter are checked in the WaveformSet.read_wvfs() method

        htype.check_type(
            file_type_code,
            int,
            exception_message=htype.generate_exception_message(
                "WaveformSet._extract_core_data", 1
            ),
        )
        if file_type_code < 0 or file_type_code > 2:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet._extract_core_data", 2
                )
            )
        htype.check_type(
            skiprows,
            int,
            exception_message=htype.generate_exception_message(
                "WaveformSet._extract_core_data", 3
            ),
        )
        if skiprows < 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet._extract_core_data", 4
                )
            )
        if file_type_code > 1:
            htype.check_type(
                tek_wfm_metadata,
                dict,
                exception_message=htype.generate_exception_message(
                    "WaveformSet._extract_core_data", 5
                ),
            )

        result = None
        additional_dict = {}

        if file_type_code < 2:  # ASCII input

            data = np.loadtxt(
                filepath, 
                delimiter=data_delimiter, 
                skiprows=skiprows)

            if np.ndim(data) < 2 or np.shape(data)[1] < 2:
                raise cuex.NoAvailableData(
                    htype.generate_exception_message(
                        "WaveformSet._extract_core_data",
                        6,
                        extra_info="Input ASCII data must have at least two columns.",
                    )
                )
            # Either voltage entries or a timestamp,
            # we remove the rest of the data and
            # preserve the second column
            data = data[:, 1]

            if file_type_code == 0:
                result = (data,)

            else:  # file_type_code == 1
                # Assuming that the acquisition time of the
                # last waveform is negligible with respect
                # to the time difference between triggers
                additional_dict["acquisition_time"] = np.sum(data)
                additional_dict["average_delta_t_wf"] = additional_dict["acquisition_time"] / (
                    data.shape[0] - 1
                )

                result = (data, additional_dict)

        else:  # Binary input

            timestamp, waveforms = WaveformSet.extract_tek_wfm_coredata(
                filepath, tek_wfm_metadata
            )

            # Concatenate waveforms in a 1D-array
            waveforms = waveforms.flatten(
                order="F"
            )

            # Assuming that the acquisition time of the
            # last waveform is negligible with respect
            # to the time difference between triggers
            additional_dict["acquisition_time"] = np.sum(timestamp)

            # The time stamp, as returned by
            # WaveformSet.extract_tek_wfm_coredata(),
            # contains as many entries as waveforms in
            # in the FastFrame set. The first one is null.
            additional_dict["average_delta_t_wf"] = additional_dict["acquisition_time"] / (
                timestamp.shape[0] - 1
            )

            result = (timestamp, waveforms, additional_dict)

        return result

    @staticmethod
    def extract_tek_wfm_coredata(filepath, metadata):
        """This static method gets the following mandatory positional arguments:

        - filepath (string): Path to the binary file (Tektronix WFM file format),
        which must host a FastFrame set and whose core data should be extracted.
        DataPreprocessor._extract_tek_wfm_metadata() should have previously checked
        that, indeed, the input file hosts a FastFrame set. It is a check based
        on the 4-bytes integer which you can find at offset 78 of the WFM file.
        - metadata (dictionary): It is a dictionary which contains meta-data of
        the input file which is necessary to extract the core data. It should
        contain the union of the two dictionaries returned by
        DataPreprocessor._extract_tek_wfm_metadata(). For more information on
        the data contained in such dictionaries, check such method documentation.

        This method returns two arrays. The first one is an unidimensional array
        of length M, which stores timestamp information. The second one is a
        bidimensional array which stores the waveforms of the FastFrame set of
        the given input file. Say such array has shape NxM: then N is the number
        of (user-accesible) points per waveform, while M is the number of
        waveforms. The waveform entries in such array are already expressed in
        the vertical units which are extracted to the key 'Vertical Units' by
        DataPreprocessor._extract_tek_wfm_metadata(). In this context, the i-th
        entry of the first array returned by this function gives the time
        difference, in seconds, between the trigger of the i-th waveform and
        the trigger of the (i-1)-th waveform. The first entry, which is undefined
        up to the given definition, is manually set to zero."""

        htype.check_type(
            filepath,
            str,
            exception_message=htype.generate_exception_message(
                "WaveformSet.extract_tek_wfm_coredata", 82855
            ),
        )
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                htype.generate_exception_message(
                    "WaveformSet.extract_tek_wfm_coredata",
                    58749,
                    extra_info=f"Path {filepath} does not exist or is not a file.",
                )
            )
        else:
            _, extension = os.path.splitext(filepath)
            if extension != ".wfm":
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "WaveformSet.extract_tek_wfm_coredata",
                        21667,
                        extra_info=f"The extension of the input file must match '.wfm'.",
                    )
                )
        htype.check_type(
            metadata,
            dict,
            exception_message=htype.generate_exception_message(
                "WaveformSet.extract_tek_wfm_coredata", 35772
            ),
        )

        # Fraction of the sample time
        # from the trigger time stamp
        # to the next sample.
        first_sample_delay = np.empty((metadata["FastFrame Count"],), dtype=np.double)

        # The fraction of the second
        # when the trigger occurred.
        triggers_second_fractions = np.empty(
            (metadata["FastFrame Count"],), dtype=np.double
        )

        # GMT (in seconds from the epoch)
        # when the trigger occurred.
        gmt_in_seconds = np.empty((metadata["FastFrame Count"],), dtype=np.double)

        first_sample_delay[0] = metadata["tfrac[0]"]  # Add info of the first frame
        triggers_second_fractions[0] = metadata["tdatefrac[0]"]
        gmt_in_seconds[0] = metadata["tdate[0]"]

        # For FastFrame, we've got a chunk of metadata['FastFrame Count']*54
        # bytes which store WfmUpdateSpec and WfmCurveSpec objects, containing
        # data on the timestamp and the number of points of each frame.

        with open(filepath, "rb") as file:  # Binary read mode
            _ = file.read(838)  # Throw away the header bytes (838 bytes)

            # WUS stands for Waveform Update Specification. WUS objects count on a 4 bytes
            # unsigned long, a 8 bytes double, another 8 bytes double and a 4 bytes long.

            # Structure of the output array of np.fromfile
            # The first element of each tuple is the name
            # of the field, whereas the second element is the
            # data type of each field
            dtype = [
                ("_", "i4"),
                ("first_sample_delay", "f8"),
                ("trigger_second_fraction", "f8"),
                ("gmt_in_seconds", "i4"),
            ]

            # Within the same 'with' context,
            # np.fromfile continues the reading
            # process as of the already-read
            # 838 bytes. Also, we are taking into
            # account that the time information
            # of the first frame was already read.
            data = np.fromfile(
                file, dtype=dtype, count=(metadata["FastFrame Count"] - 1)
            )

            # Merge first frame trigger
            # info. with info. from the
            # the rest of the frames.
            first_sample_delay[1:] = data["first_sample_delay"]
            triggers_second_fractions[1:] = data["trigger_second_fraction"]
            gmt_in_seconds[1:] = data["gmt_in_seconds"]

            # N.B. For binary gain measurements (with external trigger),
            # it was observed that all of the entries of
            # triggers_second_fractions, and gmt_in_seconds are null at this point.

            # Read waveforms
            waveforms = np.memmap(
                file,
                dtype=metadata["samples_datatype"],
                mode="r",
                offset=metadata["curve_buffer_offset"],
                # Shape of the returned array
                # Running along second dimension
                # gives different waveforms
                shape=(metadata["samples_no"], metadata["FastFrame Count"]),
                order="F",
            )

        # While the numbers in gmt_in_seconds are O(9)
        # The fractions of seconds are O(-1). Summing
        # the fractions of the second to the GMT could
        # result in losing the second fraction info. due
        # to rounding error. It's better to shift the
        # time origin to the first trigger, then add the
        # seconds fractions.
        seconds_from_first_trigger = gmt_in_seconds - gmt_in_seconds[0]
        timestamp = seconds_from_first_trigger + triggers_second_fractions

        timestamp = np.concatenate((np.array([0.0]), np.diff(timestamp)), axis=0)

        # Filter out the oscilloscope interpolation samples
        waveforms = waveforms[
            metadata["pre-values_no"] : metadata["samples_no"]
            - metadata["post-values_no"],
            :,
        ]

        # 2D array of waveforms, in vertical units
        waveforms = (waveforms * metadata["vscale"]) + metadata["voffset"]
        return timestamp, waveforms

    @staticmethod
    def swap(x, y):
        return y, x

    @staticmethod
    def bubble_sort(list_of_scalars, sort_increasingly=True):
        """This method gets the following positional argument:

        - list_of_scalars (list): Its elements must be either integer
        or float.

        This method gets the following keyword argument:

        - sort_increasingly (bool): If True (resp. False), the returned
        list is increasingly (resp. decreasingly) sorted.

        This method implements the bubble sort algorithm to sort the
        given list of scalars, list_of_scalars."""

        htype.check_type(
            list_of_scalars,
            list,
            exception_message=htype.generate_exception_message(
                "WaveformSet.bubble_sort", 160001
            ),
        )
        for elem in list_of_scalars:
            htype.check_type(
                elem,
                int,
                np.int64,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.bubble_sort", 160002
                ),
            )
        htype.check_type(
            sort_increasingly,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.bubble_sort", 160003
            ),
        )
        samples_ = list_of_scalars
        for i in reversed(range(1, len(samples_))):
            for j in range(0, i):
                if samples_[j + 1] < samples_[j]:
                    samples_[j], samples_[j + 1] = WaveformSet.swap(
                        samples_[j], samples_[j + 1]
                    )
        if sort_increasingly:
            return samples_
        else:
            samples_.reverse()
            return samples_

    def purge(self, wvfs_to_erase=[], ask_for_confirmation=True):
        """This method gets the following optional keyword arguments:

        - wvfs_to_erase (list of semipositive integers): This list
        contains the iterator values (with respect to self) of the
        waveforms within this WaveformSet object which should be erased.

        - ask_for_confirmation (bool): Whether to ask the user for
        confirmation before the deletion.

        This method erases the waveforms within this Waveform set
        object whose iterator values initially belong to the given
        wvfs_to_erase."""

        htype.check_type(
            wvfs_to_erase,
            list,
            exception_message=htype.generate_exception_message(
                "WaveformSet.purge", 170001
            ),
        )
        for elem in wvfs_to_erase:
            htype.check_type(
                elem,
                int,
                np.int64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.purge", 170002
                ),
            )
            if elem < 0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.plot", 170003)
                )
        htype.check_type(
            ask_for_confirmation,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.purge", 170004
            ),
        )
        wvfs_to_erase_ = list(set(wvfs_to_erase))  # Remove redundancy
        wvfs_to_erase_ = WaveformSet.bubble_sort(
            wvfs_to_erase_, sort_increasingly=True
        )  # Sort increasingly
        fErase = True
        if ask_for_confirmation:
            fErase = DataPreprocessor.yes_no_translator(
                input(
                    f"In WaveformSet.purge(): Do you want to proceed to the deletion of the waveforms with the following indices? \n {wvfs_to_erase_}"
                )
            )

        if fErase:
            # Sort decreasingly
            # wvfs_to_erase_ must be sorted decreasingly
            wvfs_to_erase_.reverse()
            for idx in wvfs_to_erase_:
                try:
                    del self[idx]
                # Happens if the user gave some iterator
                # value which exceeds len(self). Ignore it.
                except IndexError:
                    continue
        return

    def filter(
        self,
        filter_function,
        *args,
        return_idcs=False,
        purge=False,
        ask_for_confirmation=True,
        **kwargs,
    ):
        """This method gets the following positional argument:

        - filter_function (function): The signature of this function,
        which must have at least one input parameter, must meet two
        requirements:

            1)  Its first parameter must be a positional argument
                called 'waveform', whose type must be Waveform.
            2)  The return type of this function must be specified
                and must be boolean.

        The boolean output of this filter function is interpreted in
        the following way. True (resp. False) means that the provided
        Waveform object does (resp. does not) pass the implemented filter.

        - args: These positional arguments are given to filter_function
        in every one of its calls. It is the user responsibility to
        give as many positional arguments as the filter_function
        needs.

        This method gets the following keyword argument:

        - return_idcs (bool): If False, this function returns
        a list of N booleans, say result, where N matches len(self)
        before any purge. In this case, the i-th entry of the
        returned list is True (resp. False) if the i-th member of
        self passed (resp. failed to pass) the filter function,
        i.e. result[i] is equal to
        filter_function(self[i], *args, **kwargs). If True, then
        this function returns a list of integers, which contains
        the iterator value, with respect to the unpurged self,
        for every waveform which has failed to pass filter function.

        - purge (bool): If False, this function does not delete
        any member of self. If True, then this function delete
        every member of self (i.e. every waveform of this
        waveform set) which fails to pass the given filter, i.e.
        every member m for which
        filter_function(m, *args, **kwargs) is False.

        - ask_for_confirmation (bool): This parameter only makes a
        difference if purge is True. In that case, this parameter
        is given to the ask_for_confirmation keyword argument of
        WaveformSet.purge(). It determines whether the used should
        be asked for confirmation before actually purging this
        waveform set.

        - kwargs: These keyword arguments are given to filter_function
        in every one of its calls. It is the user responsibility to
        give keyword arguments which are actually defined within
        the signature of filter_function.

        This function returns a list of booleans or integers, up to
        the 'return_idcs' parameter. Check its documentation for
        more information. Additionally, the members of self which
        fail to pass the given filter, filter_function, can be
        erased from self, up to the input given to the 'purge'
        parameter. Check its documentation for more information."""

        if not callable(filter_function):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.filter", 180001)
            )
        signature = inspect.signature(filter_function)
        if len(signature.parameters) < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.filter",
                    180002,
                    extra_info="The signature of the given filter must have one argument at least.",
                )
            )
        if list(signature.parameters.keys())[0] != "waveform":
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.filter",
                    180003,
                    extra_info="The name of the first parameter of the signature of the given filter must be 'waveform'.",
                )
            )
        if signature.parameters["waveform"].annotation != Waveform:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.filter",
                    180004,
                    extra_info="The type of the first parameter of the signature of the given filter must be hinted as Waveform.",
                )
            )
        if signature.return_annotation != bool:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.filter",
                    180005,
                    extra_info="The return type of the given filter must be hinted as bool.",
                )
            )
        htype.check_type(
            return_idcs,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.filter", 180006
            ),
        )
        htype.check_type(
            purge,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.filter", 180007
            ),
        )
        htype.check_type(
            ask_for_confirmation,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.filter", 180008
            ),
        )
        mask, wvfs_to_erase = [], []

        for i in range(len(self)):
            try:
                fPassed = filter_function(self[i], *args, **kwargs)
                htype.check_type(
                    fPassed,
                    bool,
                    np.bool_,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.filter",
                        180009,
                        extra_info="The filter function is not behaving as expected.",
                    ),
                )
                mask.append(fPassed)
                if not fPassed:
                    wvfs_to_erase.append(i)

            # Happens if the given keyword
            # arguments are not defined in
            # the signature of filter_function.
            except TypeError:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "WaveformSet.filter",
                        180010,
                        extra_info="Either the number of provided positional arguments do not match the expected one, or you gave some keyword argument which was not defined in the signature of the given filter_function.",
                    )
                )
        if purge:
            self.purge(
                wvfs_to_erase=wvfs_to_erase, ask_for_confirmation=ask_for_confirmation
            )
        if not return_idcs:
            return mask
        else:
            return wvfs_to_erase

    @staticmethod
    def random_filter(waveform: Waveform) -> bool:
        """A dummy template for filter functions which
        may be passed to the filter_function argument
        of WaveformSet.filter()."""

        if np.random.uniform() <= 0.5:
            return True
        else:
            return False

    @staticmethod
    def threshold_filter(
        waveform: Waveform,
        threshold,
        threshold_from_baseline=False,
        rise=True,
        filter_out=False,
        i_subrange=None,
        subrange=None,
    ) -> bool:
        """This method gets the following positional arguments:

        - waveform (Waveform): Waveform object which will be
        filtered.

        - threshold (float): For more information, check the
        documentation below.

        This method also gets the following keyword arguments:

        - threshold_from_baseline (bool): If False, then the
        given threshold is understood as an absolute threshold.
        If else, then the given threshold is understood as an
        offset of wvf.Signs['first_peak_baseline'][0].

        - rise (bool): For more information, check the
        documentation below.

        - filter_out (bool): For more information, check the
        documentation below.

        - i_subrange (tuple of two integers): i_subrange[0]
        (resp. i_subrange[1]) is the iterator value, with
        respect to waveform.Time, of the inclusive lower
        (resp. upper) limit of the subrange of Waveform.Time
        which will be parsed. i_subrange must meet the following
        conditions:
        0<=i_subrange[0]<i_subrange[1]<=len(waveform.Time)-1.
        If this parameters is defined, then the subrange parameter
        is ignored.

        - subrange (tuple of two floats): This parameter only
        makes a diference if i_subrange is None. subrange[0]
        (resp. subrange[1]) is the lower (resp. upper) limit
        of the subrange of waveform.Time which will be parsed.
        subrange must meet the following conditions:
        np.min(waveform.Time)<=subrange[0]<subrange[1]<=np.max(waveform.Time)

        If i_subrange nor subrange are defined, then the whole
        available range of waveform.Time is parsed, i.e.
        [np.min(waveform.Time), np.max(waveform.Time)].

        If rise is True, then this method evaluates whether the
        maximum value of waveform.Signal in the specified
        subrange is bigger than or equal to the set threshold, up
        to the input given to 'threshold' and 'threshold_from_baseline'.
        If this condition is met (resp. not met) and filter_out is
        False, then this function returns True (resp. False). The
        output is inverted if filter_out is True.

        If rise is False, then this method evaluates whether the
        minimum value of waveform.Signal in the specified subrange
        is smaller than or equal to the set threshold, up to the
        input given to 'threshold' and 'threshold_from_baseline'.
        If this condition is met (resp. not met) and filter_out is
        False, then this function returns True (resp. False). The
        output is inverted if filter_out is True."""

        htype.check_type(
            waveform,
            Waveform,
            exception_message=htype.generate_exception_message(
                "WaveformSet.threshold_filter", 190001
            ),
        )
        htype.check_type(
            threshold,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.threshold_filter", 190002
            ),
        )
        htype.check_type(
            threshold_from_baseline,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.threshold_filter", 190003
            ),
        )
        htype.check_type(
            rise,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.threshold_filter", 190004
            ),
        )
        htype.check_type(
            filter_out,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.threshold_filter", 190005
            ),
        )
        fUseISubrange = False
        if i_subrange is not None:
            htype.check_type(
                i_subrange,
                tuple,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.threshold_filter", 190006
                ),
            )
            if len(i_subrange) != 2:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.filter", 190007)
                )
            htype.check_type(
                i_subrange[0],
                int,
                np.int64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.threshold_filter", 190008
                ),
            )
            htype.check_type(
                i_subrange[1],
                int,
                np.int64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.threshold_filter", 190009
                ),
            )

            if (
                i_subrange[0] < 0
                or i_subrange[0] > len(waveform.Time) - 1
                or i_subrange[1] < 0
                or i_subrange[1] > len(waveform.Time) - 1
            ):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.filter", 190010)
                )
            if i_subrange[0] >= i_subrange[1]:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("WaveformSet.filter", 190011)
                )
            fUseISubrange = True

        fUseSubrange = False
        if (
            not fUseISubrange
        ):  # Only pay attention to subrange if i_subrange is not defined
            if subrange is not None:
                htype.check_type(
                    subrange,
                    tuple,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.threshold_filter", 190012
                    ),
                )
                if len(subrange) != 2:
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message("WaveformSet.filter", 190013)
                    )
                htype.check_type(
                    subrange[0],
                    float,
                    np.float64,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.threshold_filter", 190014
                    ),
                )
                htype.check_type(
                    subrange[1],
                    float,
                    np.float64,
                    exception_message=htype.generate_exception_message(
                        "WaveformSet.threshold_filter", 190015
                    ),
                )

                aux_min, aux_max = np.min(waveform.Time), np.max(waveform.Time)

                if (
                    subrange[0] < aux_min
                    or subrange[0] > aux_max
                    or subrange[1] < aux_min
                    or subrange[1] > aux_max
                ):
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message("WaveformSet.filter", 190016)
                    )
                if subrange[0] >= subrange[1]:
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message("WaveformSet.filter", 190017)
                    )
                # Use subrange only if not fUseISubrange
                # AND subrange is properly defined
                fUseSubrange = True

        if not fUseISubrange and not fUseSubrange:
            i_low, i_up = 0, len(waveform.Time) - 1
        elif fUseISubrange:
            i_low, i_up = i_subrange[0], i_subrange[1]

        # Note that fUseISubrange and fUseSubrange
        # cannot be simultaneously True
        else:
            i_low, _ = search.find_nearest_neighbour(waveform.Time, subrange[0])
            i_up, _ = search.find_nearest_neighbour(waveform.Time, subrange[1])

        if i_low == i_up:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.filter",
                    190018,
                    extra_info="The waveform signal resolution is not fine enough. Please provide a wider range for parsing.",
                )
            )
        threshold_ = threshold
        if threshold_from_baseline:
            threshold_ += waveform.Signs["first_peak_baseline"][0]

        if rise:
            result = (
                np.max(waveform.Signal[i_low : i_up + 1]) >= threshold_
            )  # Make the upper limit inclusive
        else:
            result = (
                np.min(waveform.Signal[i_low : i_up + 1]) <= threshold_
            )  # Make the upper limit inclusive

        if not filter_out:
            return result
        else:
            return not result

    @staticmethod
    def correlation_filter(
        waveform: Waveform,
        threshold,
        model_y,
        i0=0,
        delta_t=None,
        also_return_correlation=False,
    ) -> bool:
        """This method gets the following positional arguments:

        - waveform (Waveform): Waveform object which will be
        filtered based on its correlation with a model signal.

        - threshold (float): The correlation result, say x, is
        compared to this value, threshold. If x>=threshold, then
        this filter returns True. It returns false if else.

        - model_y (unidimensional float numpy array): model_y[i]
        is the i-th y-value of the curve whose correlation
        with waveform.Signal is studied.Its length must match
        that of waveform.Time. In that case, it is assumed that
        the time value for model_y[i] matches that of
        waveform.Signal[i], i.e. waveform.Time[i].

        This method also gets the following keyword arguments:

        - i0 (scalar integer): It must be semipositive (>=0).
        It is the iterator value of the first point to consider
        for the computation. This is applied to both, waveform.Signal
        and model_y.

        - delta_t (scalar float): It must be positive. This
        parameter controls the last time point which is considered
        for the computation. Particularly, that point is taken to
        be the nearest one to waveform.Time[i0]+delta_t among
        waveform.Time.

        - also_return_correlation (bool): If False, this method
        only returns a boolean value, indicating whether waveform
        has passed the filter. If True, this method returns the
        result of the filter as the first parameter, but it also
        returns the correlation value as a second parameter.

        This method computes the correlation between waveform.Signal
        and model_y in the selected time window (up to the input
        given to i0 and delta_t), meaning its simultaneous behaviour
        about its half-amplitude. This means that if both signals
        cross their respective half-amplitude level, become bigger
        (or smaller) simultaneously, then this measurement will be
        big. The concept is similar to the covariance of two random
        variables, although in this case the reference value for
        each signal is taken to be its half-amplitude, and not its
        mean value, as it is taken for the covariance.

        This method computes such correlation in the following way:

            - it filters both, waveform.Time and model_y, to get rid
            of the points which fall out of the selected time window,
            - then each signal is offseted by its minimum and scaled
            by its maximum. This normalization make each signal belong
            to the [0.0,1.0] interval.
            - The signal is then offseted by -1/2, so that each signal
            belongs to the [-0.5,0.5] interval.
            - The correlation is then computed as the integral of the
            product of both normalized signals.
            - This correlation measurement is normalized by K, which
            is computed as the integral of the product of the normalized
            model_y times 0.5*sign(model_y), where sign(x) is the sign
            function which evaluates to 1.0 (resp. -1.0) if x>=0 (x<0).

        Note that g(t)=0.5*sign(model_y)(t) gives the bigger
        'correlation' against model_y. I.e. g(t) is the signal which,
        complying with g(t)\in[-0.5,0.5], gives the biggest correlation
        measurement. This implies the following:

            - This measurement is not a 'good' correlation measurement,
            since there are functions f(t) which, although resembling
            model_y to a lesser extent than model_y itself, give
            a bigger correlation measurement against model_y. g(t)
            is an example."""

        htype.check_type(
            waveform,
            Waveform,
            exception_message=htype.generate_exception_message(
                "WaveformSet.correlation_filter", 200001
            ),
        )
        htype.check_type(
            threshold,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.correlation_filter", 200002
            ),
        )
        htype.check_type(
            model_y,
            np.ndarray,
            exception_message=htype.generate_exception_message(
                "WaveformSet.correlation_filter", 200003
            ),
        )
        if model_y.dtype != np.float64:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.correlation_filter", 200004
                )
            )
        if model_y.ndim != 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.correlation_filter", 200005
                )
            )
        if len(model_y) != len(waveform.Time):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.correlation_filter", 200006
                )
            )
        htype.check_type(
            i0,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.correlation_filter", 200007
            ),
        )
        if i0 < 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.correlation_filter", 200008
                )
            )
        fUseDeltaT = False
        if delta_t is not None:
            htype.check_type(
                delta_t,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.correlation_filter", 200009
                ),
            )
            if delta_t <= 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "WaveformSet.correlation_filter", 200010
                    )
                )
            fUseDeltaT = True

        htype.check_type(
            also_return_correlation,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.correlation_filter", 200011
            ),
        )
        if fUseDeltaT:
            iN, _ = search.find_nearest_neighbour(
                waveform.Time, waveform.Time[i0] + delta_t
            )
            if iN <= i0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "WaveformSet.correlation_filter",
                        200012,
                        extra_info="The given delta_t is not big enough (compared to the waveform time resolution) so as to give at least two different points for the integral.",
                    )
                )
            time = waveform.Time[i0 : iN + 1]
            signal_1 = waveform.Signal[i0 : iN + 1]
            signal_2 = model_y[i0 : iN + 1]
        else:
            time = waveform.Time[i0:]
            signal_1 = waveform.Signal[i0:]
            signal_2 = model_y[i0:]

        shifter = lambda array_1d: array_1d - np.min(array_1d)
        scaler = lambda array_1d: array_1d / np.max(array_1d)

        signal_1 = scaler(shifter(signal_1)) - 0.5
        signal_2 = scaler(shifter(signal_2)) - 0.5

        normalization = np.trapz(signal_2 * (0.5 * np.sign(signal_2)), x=time)
        correlation = np.trapz(signal_1 * signal_2, x=time) / normalization

        if not also_return_correlation:
            return correlation >= threshold
        else:
            return correlation >= threshold, correlation

    @staticmethod
    def integral_filter(
        waveform: Waveform, threshold, i0=0, delta_t=None, also_return_integral=False
    ) -> bool:
        """This method gets the following positional arguments:

        - waveform (Waveform): Waveform object which will be
        filtered based on its integral.

        - threshold (float): The integral, say x, is compared
        to this value, threshold. If x>=threshold, then this
        filter returns True. It returns false if else.

        This method also gets the following keyword arguments:

        - i0 (scalar integer): It must be semipositive (>=0).
        It is the iterator value of the first point to consider
        for the integral.

        - delta_t (scalar float): It must be positive. This
        parameter controls the last time point which is considered
        for the integral. Particularly, that point is taken to
        be the nearest one to waveform.Time[i0]+delta_t among
        waveform.Time.

        - also_return_integral (bool): If False, this method
        only returns a boolean value, indicating whether waveform
        has passed the filter. If True, this method returns the
        result of the filter as the first parameter, but it also
        returns the integral as a second parameter.

        This method computes the integral of the signal of waveform
        in the selected time window, up to the input given to i0
        and delta_t, and returns True (resp. False) if such integral
        is equal or bigger (resp. smaller) than threshold."""

        htype.check_type(
            waveform,
            Waveform,
            exception_message=htype.generate_exception_message(
                "WaveformSet.integral_filter", 210001
            ),
        )
        htype.check_type(
            threshold,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.integral_filter", 210002
            ),
        )
        htype.check_type(
            i0,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.integral_filter", 210003
            ),
        )
        if i0 < 0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.integral_filter", 210004)
            )
        fUseDeltaT = False
        if delta_t is not None:
            htype.check_type(
                delta_t,
                float,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.integral_filter", 210005
                ),
            )
            if delta_t <= 0.0:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "WaveformSet.integral_filter", 210006
                    )
                )
            fUseDeltaT = True

        htype.check_type(
            also_return_integral,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.integral_filter", 210007
            ),
        )
        if not fUseDeltaT:
            result = waveform.integrate(
                integration_lower_lim=waveform.Time[i0],
                integration_upper_lim=waveform.Time[-1],
            )
        else:
            result = waveform.integrate(
                integration_lower_lim=waveform.Time[i0],
                integration_upper_lim=waveform.Time[i0] + delta_t,
            )
        if not also_return_integral:
            return result >= threshold
        else:
            return result >= threshold, result

    @staticmethod
    def peaks_no_filter(waveform: Waveform, n_peaks, filter_out_below=True) -> bool:
        """This method gets the following positional arguments:

        - waveform (Waveform): Waveform object which will be
        filtered based on the number of peaks that have been
        already spotted in it.

        - n_peaks (scalar integer): Exclusive threshold for
        the number of peaks. Its meaning depends on the value
        given to the 'filter_out_below' parameter.

        This method also gets the following keyword argument:

        - filter_out_below (bool): If True (resp. False), the
        waveforms that are considered not to pass the filter are
        those for which the number of already-spotted peaks is
        smaller (resp. bigger) n_peaks.

        For the given waveform, this method computes the number of
        already-spotted peaks as len(waveform.Signs['peaks_pos']).
        Comparing it to n_peaks and taking into account
        filter_out_below, this method returns True if the waveform
        meet the filter requirement and False if else."""

        htype.check_type(
            waveform,
            Waveform,
            exception_message=htype.generate_exception_message(
                "WaveformSet.peaks_no_filter", 220001
            ),
        )
        htype.check_type(
            n_peaks,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.peaks_no_filter", 220002
            ),
        )
        htype.check_type(
            filter_out_below,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.peaks_no_filter", 220003
            ),
        )
        if filter_out_below:
            if len(waveform.Signs["peaks_pos"]) < n_peaks:
                return False
            else:
                return True
        else:
            if len(waveform.Signs["peaks_pos"]) > n_peaks:
                return False
            else:
                return True

    def exclude_these_indices(self, idcs_to_exclude):
        """This method gets the following positional argument:

        - idcs_to_exclude (list of integers): Every integer in
        this list must be a semipositive integer (>=0) and must
        be smaller than len(self).

        This method returns a list of integers, which is
        basically list(range(0,len(self))) after having erased
        the integers which are contained in idcs_to_exclude."""

        htype.check_type(
            idcs_to_exclude,
            list,
            exception_message=htype.generate_exception_message(
                "WaveformSet.exclude_these_indices", 230001
            ),
        )
        for elem in idcs_to_exclude:
            htype.check_type(
                elem,
                int,
                np.int64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.exclude_these_indices", 230002
                ),
            )
            if elem < 0 or elem >= len(self):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "WaveformSet.exclude_these_indices", 230003
                    )
                )
        idcs_to_exclude_ = list(set(idcs_to_exclude))  # Remove redundancy
        idcs_to_exclude_ = WaveformSet.bubble_sort(
            idcs_to_exclude_, sort_increasingly=True
        )  # Sort increasingly
        result = []
        pointer = 0
        for i in range(0, len(self)):

            # I can use this method to populate result
            # because I have removed the redundancy and
            # I have ordered idcs_to_exclude_ increasingly
            if pointer < len(idcs_to_exclude_):
                if i != idcs_to_exclude_[pointer]:
                    result.append(i)
                else:
                    pointer += 1

            # At this point, we have
            # exhausted idcs_to_exclude,
            # so add the remaining indices.
            else:
                result += list(range(i, len(self)))
                break

        return result

    def get_average_baseline(self, get_std=False, get_dispersion=False):
        """This method gets the following keyword arguments:

        - get_std (bool)
        - get_dispersion (bool)

        If get_std and get_dispersion are equal to False, then
        this function returns the average of the waveform
        baselines, i.e. the average of
        wvf.Signs.['first_peak_baseline'][0] for every waveform
        wvf in self. If get_std is True or get_dispersion is
        True, then this function returns a dictionary, with
        two keys at least, among which there's the 'average'
        key, whose value is the average of the waveforms
        baselines. If get_std is True, then the returned
        dictionary contains the 'std' key, whose value is the
        standard deviation of the waveform baselines. If
        get_dispersion is True, then the returned dictionary
        contains the 'dispersion' key, whose value is the
        difference between the maximal baseline and the
        minimal baseline."""

        htype.check_type(
            get_std,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.get_average_baseline", 240001
            ),
        )
        htype.check_type(
            get_dispersion,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.get_average_baseline", 240002
            ),
        )

        baselines = np.array([wvf.Signs["first_peak_baseline"][0] for wvf in self])

        if get_std == False and get_dispersion == False:
            return np.mean(baselines)
        else:
            if len(baselines) < 2:
                raise cuex.NoAvailableData(
                    htype.generate_exception_message(
                        "WaveformSet.get_average_baseline",
                        240003,
                        extra_info=f"There are not enough waveforms in this waveform set to compute the standard deviation or the disperion of its baselines.",
                    )
                )
            result = {"average": np.mean(baselines)}
            if get_std:
                result["std"] = np.std(baselines)
            if get_dispersion:
                result["dispersion"] = np.max(baselines) - np.min(baselines)

            return result

    def find_peaks(self, return_peak_properties=False, **kwargs):
        """This method gets the following optional keyword argument:

        - return_peak_properties (bool): If True, this method returns a list of
        dictionaries, say result, where result[i] is the dictionary which hosts
        the properties of the peaks spotted in the i-th waveform of this waveform
        set, as given by the second parameter returned by scipy.signal.find_peaks.

        - kwargs: Every keyword argument, except for the 'height' one (if defined),
        are given to scipy.signal.find_peaks as they are. For the case of the
        'height' keyword argument, if it is defined, then it is checked to be 
        a scalar integer/float. In that case, for each waveform in this waveform
        set, say wvf, the 'height' keyword argument of 
        scipy.signal.find_peaks(wvf.Signal, **kwargs) is set to
        wvf.Signs['first_peak_baseline'][0] + h, where h is the value of the
        'height' keyword argument given to this method. This means that the
        minimal height for a peak to be considered so is measured with respect
        to the baseline of each waveform.

        This method analyzes this waveform set: it uses scipy.signal.find_peaks to
        spot the peaks in every waveform in the WaveformSet. For each waveform, wvf,
        the peaks that have been spotted in it are added to the 'peaks_pos' and
        'peaks_top' entries of the wvf.Signs dictionary. When this method is called,
        the information contained in wvf.Signs['peaks_pos'] and wvf.Signs['peaks_top'],
        for every wvf, is overriden. The return value of this method may vary depending
        on the value given to the return_peak_properties parameter."""

        htype.check_type(
            return_peak_properties,
            bool,
            exception_message=htype.generate_exception_message(
                "WaveformSet.find_peaks", 250001
            ),
        )

        fUseHeight = False
        if "height" in kwargs:
            htype.check_type(
                kwargs["height"],
                int,
                float,
                np.int64,
                np.float64,
                exception_message=htype.generate_exception_message(
                    "WaveformSet.find_peaks", 250002
                ),
            )
            fUseHeight = True
            height = kwargs["height"]
            del kwargs["height"]

        if not return_peak_properties:
            for wvf in self:

                wvf.Signs = ("peaks_pos", [], True)  # Erasing previous info.
                wvf.Signs = ("peaks_top", [], True)  # Erasing previous info.

                peaks_idx, _ = spsi.find_peaks(
                    wvf.Signal,
                    height=None if not fUseHeight 
                        else wvf.Signs["first_peak_baseline"][0] + height,
                    **kwargs
                )  # Peak finding algorithm

                for idx in peaks_idx:
                    wvf.Signs = (
                        "peaks_pos",
                        [wvf.Time[idx]],
                        False,
                    )  # Add peak info. to waveforms
                    wvf.Signs = ("peaks_top", [wvf.Signal[idx]], False)

            return

        # Duplicating the code so that the return_peak_properties
        # condition is not checked at every iteration
        else:
            result = []
            for wvf in self:

                wvf.Signs = ("peaks_pos", [], True)  # Erasing previous info.
                wvf.Signs = ("peaks_top", [], True)  # Erasing previous info.

                peaks_idx, properties = spsi.find_peaks(
                    wvf.Signal, **kwargs
                )  # Peak finding algorithm
                result.append(properties)

                for idx in peaks_idx:
                    wvf.Signs = (
                        "peaks_pos",
                        [wvf.Time[idx]],
                        False,
                    )  # Add peak info. to waveforms
                    wvf.Signs = ("peaks_top", [wvf.Signal[idx]], False)

            return result

    def integrate(
        self,
        input_resistance_in_ohms=50.0,
        system_amplification_factor=1.0,
        filter_out_infs_and_nans=True,
        integration_lower_lim=None,
        integration_upper_lim=None,
    ):
        """This method integrates every waveform in this waveform set and returns
        an unidimensional numpy array of floats, so that the i-th entry of such
        array is the integral of the i-th waveform in the set divided by the input
        resistance of the DAQ device which was used to measure the waveforms and
        the amplification factor of the overall readout system. In this way, the
        magnitude of the returned entries is electrical charge. Integration is
        performed via the trapezoid method. Namely, this method calls numpy.trapz.
        This method gets the following optional keyword arguments:

        - input_resistance_in_ohms (float): Electrical resistance in ohms. It is
        set to 50 ohms by default, which is a pretty standard value for oscilloscopes.

        - system_amplification_factor (float): It must be positive (>0.0).

        - filter_out_infs_and_nans (bool): Whether to filter out the result entries
        which equate to float('inf'), float('-inf') or float('nan').

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
                "WaveformSet.integrate", 260001
            ),
        )
        htype.check_type(
            system_amplification_factor,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "WaveformSet.integrate", 260002
            ),
        )
        if system_amplification_factor <= 0.0:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("WaveformSet.integrate", 260003)
            )

        # integration_lower_lim and integration_upper_lim
        # are checked and handled by Waveform.integrate().
        result = []
        for wvf in self:
            result.append(
                wvf.integrate(
                    integration_lower_lim=integration_lower_lim,
                    integration_upper_lim=integration_upper_lim,
                )
                / (input_resistance_in_ohms * system_amplification_factor)
            )
        if filter_out_infs_and_nans:

            # Filtering out entries which equate to float('inf'),
            # float('-inf') or float('nan'), which resulted from
            # integrating a defective waveform.
            return Waveform.filter_infs_and_nans(np.array(result), get_mask=False)

        else:
            return np.array(result)

    def check_homogeneity_of_sign_through_set(self, sign):
        """This method gets the following positional argument:

        - sign (string): It must belong to any of the two following Waveform class
        variables: Waveform.stackable_aks or Waveform.nonstackable_aks. For more
        information, check the definition of those class variables in the Waveform
        class definition.

        This method returns True if, for every waveform wvf in this waveform set,
        wvf.Signs[sign] is the same."""

        htype.check_type(
            sign,
            str,
            exception_message=htype.generate_exception_message(
                "WaveformSet.check_homogeneity_of_sign_through_set", 52547
            ),
        )
        if not sign in Waveform.stackable_aks + Waveform.nonstackable_aks:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "WaveformSet.check_homogeneity_of_sign_through_set", 87272
                )
            )
        result = True

        try:
            standard = self[0].Signs[sign]
            for wvf in self[1:]:
                if wvf.Signs[sign] != standard:
                    result = False
                    break

        # Even if the given sign belongs to Waveform.stackable_aks
        # or Waveform.nonstackable_aks, it may not be defined as
        # a key in the Signs attribute. An example, is the
        # 'peaks_pos' key for waveform sets which are meant to
        # belong to a GainMeas object.
        except KeyError:
            raise cuex.NoAvailableData(
                htype.generate_exception_message(
                    "WaveformSet.check_homogeneity_of_sign_through_set",
                    86306,
                    extra_info=f"The given sign ({sign}) is not defined in the Signs dictionary of some waveform in this waveform set.",
                )
            )
        return result