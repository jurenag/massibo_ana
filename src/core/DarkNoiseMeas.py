import os
import json
import copy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import src.utils.htype as htype
import src.utils.custom_exceptions as cuex
from src.custom_types.RigidKeyDictionary import RigidKeyDictionary
from src.core.Waveform import Waveform
from src.core.SiPMMeas import SiPMMeas

class DarkNoiseMeas(SiPMMeas):
    
    def __init__(self,  *args,
                        strip_ID=None, meas_ID=None, date=None, location=None, operator=None, 
                        setup_ID=None, system_characteristics=None, thermal_cycle=None,
                        elapsed_cryo_time_min=None, electronic_board_number=None, 
                        electronic_board_location=None, electronic_board_socket=None, 
                        sipm_location=None, sampling_ns=None, overvoltage_V=None, PDE=None, 
                        status=None, acquisition_time_min=None, threshold_mV=None, **kwargs):

        """This class, which derives from SiPMMeas class, aims to implement a SiPM dark
        noise measurement. In line with Waveform, WaveformSet and SiPMMeas classes, the 
        assumed time unit is the second, unless otherwise indicated. This initializer gets 
        the following positional argument:

        - args: These positional arguments are given to WaveformSet.from_files. They must be
        two positional arguments: input_filepath (string) and time_resolution (positive float), 
        in such order. For more information on these arguments, please refer to 
        WaveformSet.from_files docstring. Particularly, time_resolution is assumed to be expressed 
        in seconds. If applicable (i.e. if the given sampling_ns is None), its value is converted 
        to nanosecons and assigned to the self.__sampling_ns attribute.
        
        This initializer gets the following keyword arguments:

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
        - elapsed_cryo_time_min (semipositive float): Elapsed time, in minutes, in cryogenic 
        conditions for this SiPM (in the current cryogenic bath) when this measurement started.
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
        - overvoltage_V (semipositive float): Feeding voltage given to the measured SiPM, measured 
        with respect to the breakdown voltage.
        - PDE (semipositive float): Photon detection efficiency of the measured SiPM.
        - status (string): String which identifies the status of the measured SiPM.
        - acquisition_time_min (semipositive float): Time duration of the dark noise measurement.
        - threshold_mV (float): Trigger threshold which was used for acquiring the waveforms.
        - kwargs: These keyword arguments are given to WaveformSet.from_files. The expected keywords 
        are points_per_wvf (int), wvfs_to_read (int), separator (string), timestamp_filepath (string), 
        delta_t_wf (float), set_name (string), creation_dt_offset_min (float) and 
        wvf_extra_info (string). To understand these arguments, please refer to the 
        WaveformSet.from_files docstring.

        All of the keyword arguments, except for **kwargs, are loaded into object-attributes whose 
        name matches the keyword of the kwarg except for two previous underscores, which are added 
        to the attribute name. None of these arguments are positional arguments, so none of them are 
        mandatory. If a certain keyword argument is not set in the instance initialization, then its 
        associated attribute is set to None within this initializer, meaning that its information is 
        not available. Thus, when requesting any attribute of this class via a getter, one should be 
        prepared to handle a None value and interpret it as the unavailability of such data."""


        self.__acquisition_time_min = None
        if acquisition_time_min is not None:
            htype.check_type(   acquisition_time_min, float, np.float64,
                                exception_message=htype.generate_exception_message( "DarkNoiseMeas.__init__", 
                                                                                    58815))
            if acquisition_time_min<0.:
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.__init__", 
                                                                                        77855))
            self.__acquisition_time_min = acquisition_time_min

        self.__threshold_mV = None
        if threshold_mV is not None:
            htype.check_type(   threshold_mV, float, np.float64,
                                exception_message=htype.generate_exception_message( "DarkNoiseMeas.__init__", 
                                                                                    37454))
            self.__threshold_mV = threshold_mV

        # The rest of the arguments are handled by the base class initializer

        # Are you going to implement these?
        
		#self.UDCN_pa       # Uncorrelated DCN (Peaks Algorithm)
		#self.UDCR_pa       # (PA)
		#self.UDCN_fit      # Uncorrelated DCN (Fit)
		#self.UDCR_fit      # Uncorrelated DCR (Fit)
		#self.fit_chi2
		#self.fit_parameters
		#self.fit_parameters_errors

        self.__timedelay = None         # In order to populate self.__timedelay, self.__amplitude and self.__frame_idx 
        self.__amplitude = None         # the user needs to call self.analyze(). However, the analyze() method relies 
                                        # on the peaks detection, which should have been performed previously. I.e. 
                                        # analyze() won't do so. To detect the peaks in the underlying waveform set, 
                                        # the user can call self.Waveforms.find_peaks(). The call of such method is not 
                                        # automatised within this initializer because it requires some keyword arguments 
                                        # (kwargs) which are passed to the peak-finding algorithm (scipy.signal.find_peaks()). 
                                        # It is convenient to keep those kwargs apart from the kwargs of this initializer.
        self.__frame_idx = None         # This is intended to be a one dimensional float numpy array, whose length
                                        # matches that of self.__timedelay and self.__amplitude. self.__frame_idx[i]
                                        # is meant to be iterator value, within self.Waveforms, for the waveform where
                                        # the peak whose time delay with respect the previous one is self.__timedelay[i]
                                        # and whose amplitude is self.__amplitude[i].
        self.__half_a_pe = None         # In order to compute both, self.__half_a_pe and self.__one_and_a_half_pe,
        self.__one_and_a_half_pe = None # the user needs to call self.compute_amplitude_levels(), which relies on
                                        # self.__amplitude. Therefore, such array should have been computed prior to
                                        # self.compute_amplitude_levels() calling, p.e. via self.analyze().

        super().__init__(   *args,
                            strip_ID=strip_ID, meas_ID=meas_ID, date=date, location=location, 
                            operator=operator, setup_ID=setup_ID, 
                            system_characteristics=system_characteristics, 
                            thermal_cycle=thermal_cycle, 
                            elapsed_cryo_time_min=elapsed_cryo_time_min,
                            electronic_board_number=electronic_board_number, 
                            electronic_board_location=electronic_board_location,
                            electronic_board_socket=electronic_board_socket, 
                            sipm_location=sipm_location, sampling_ns=sampling_ns,
                            overvoltage_V=overvoltage_V, PDE=PDE, status=status, 
                            **kwargs)

    @property
    def AcquisitionTime_min(self):
        return self.__acquisition_time_min
    
    @property
    def Threshold_mV(self):
        return self.__threshold_mV
    
    @property
    def TimeDelay(self):
        return self.__timedelay
    
    @property
    def Amplitude(self):
        return self.__amplitude

    @property
    def FrameIDX(self):
        return self.__frame_idx

    @TimeDelay.setter
    def TimeDelay(self, input):

        """This setter gets an unidimensional float numpy array
        and sets it to self.__timedelay."""

        htype.check_type(   input, np.ndarray,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.TimeDelay.Setter", 
                                                                                85070))
        if input.ndim!=1:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.TimeDelay.Setter", 
                                                                                    81275)) 
        if input.dtype!=np.float64:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.TimeDelay.Setter", 
                                                                                    83452)) 
        self.__timedelay=input
        return

    @Amplitude.setter
    def Amplitude(self, input):

        """This setter gets an unidimensional float numpy array
        and sets it to self.__amplitude."""

        htype.check_type(   input, np.ndarray,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.Amplitude.Setter", 
                                                                                22280))
        if input.ndim!=1:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.Amplitude.Setter", 
                                                                                    54375)) 
        if input.dtype!=np.float64:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.Amplitude.Setter", 
                                                                                    19977)) 
        self.__amplitude=input
        return

    @FrameIDX.setter
    def FrameIDX(self, input):

        """This setter gets an unidimensional integer numpy array
        and sets it to self.__frame_idx."""

        htype.check_type(   input, np.ndarray,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.FrameIDX.Setter", 
                                                                                22280))
        if input.ndim!=1:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.FrameIDX.Setter", 
                                                                                    54375)) 
        if input.dtype!=np.int64:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.FrameIDX.Setter", 
                                                                                    19977)) 
        self.__frame_idx=input
        return

    @property
    def HalfAPE(self):
        """Voltage amplitude of 0.5 photoelectrons."""
        return self.__half_a_pe
    
    @property
    def OneAndAHalfPE(self):
        """Voltage amplitude of 1.5 photoelectrons."""
        return self.__one_and_a_half_pe
    

    def filter_out_peaks_based_on_prominence_wrt_baseline(self, prominennce_wrt_baseline):

        """This tool might be useful. Scipy.signal.find_peaks() computation of prominence
        does not give a good estimation of the prominence of a peak with respect to its
        baseline when sinusoidal noises (which go under the actual signal baseline) are
        present. Based on such computation, thresholds on prominence are not effective
        to rule out noise-induced peaks, because the prominence can be arbitrarily big,
        based on the minimum value of the sinusoidal noise. However, externally to
        scipy.signal.find_peaks, I can compute my custom prominence, as the difference
        between the maximum height of the peak and the baseline of the signal (which I
        computed) and set a threshold on such custom prominence, to rule out whichever 
        peak."""
        
        pass


    def construct_absolute_time_peaks_map(self):

        """This method takes no parameter, and returns three lists 
        containing floats: 'times', 'amplitudes' and 'indices'. times[i] 
        gives the time where the i-th peak in the waveform set occurred. 
        Note that each waveform set matches the readout from one SiPM, 
        so two waveforms cannot occur at the same time. Therefore, by 
        'i-th' peak, I mean the peak which occurred after i-1 peaks have 
        occurred. Thus, 'times' is an ordered array of the times of 
        occurrence of the peaks in the underlying waveform set. 
        amplitudes[i] gives the amplitude of the i-th peak, which, for 
        a given waveform, is computed as the signal value at the peak 
        minus the baseline of the waveform. For a waveform wvf, its 
        baseline is considered to match wvf.Signs['first_peak_baseline'].
        indices[i] gives the iterator value, within self.Waveforms, for
        the waveform whose occurrence time and amplitude are have been
        saved to times[i] and amplitudes[i], respectively.
        """
        
        t0s = np.array([self.Waveforms[i].T0 for i in range(len(self.Waveforms))])
        ordering_wvfs_idx = np.argsort(t0s)
        times, amplitudes, indices = [], [], []
                                        # WARNING: The 'amplitudes' that this method compute
                                        # are not valid for every peak, but just for the first
                                        # peak within the FastFrame window. Indeed, this method 
                                        # computes the amplitudes as the difference between  
                                        # the value of the signal at each peak and the baseline 
                                        # of first peak within the FastFrame window (check 
                                        # Waveform.nonstackable_aks.first_peak_baseline). 
                                        # The amplitudes of the secondary peaks (secondary
                                        # in time) are not correctly evaluated.

        for i in range(np.shape(ordering_wvfs_idx)[0]):                                 
            aux_t = np.array(self.Waveforms[ordering_wvfs_idx[i]].Signs['peaks_pos'])   # 'peaks_pos' and 'peaks_top' are always defined as a key 
            aux_t += self.Waveforms[ordering_wvfs_idx[i]].T0                            # of a Waveform object Signs attribute, even if its associated
            times += list(aux_t)                                                        # value is an empty list (case of no found peaks). That's why
                                                                                        # there's no need to try to handle a KeyError here.
            aux_baseline = self.Waveforms[ordering_wvfs_idx[i]].Signs['first_peak_baseline'][0]
            aux_a = [self.Waveforms[ordering_wvfs_idx[i]].Signs['peaks_top'][j]-aux_baseline for j in range(len(self.Waveforms[ordering_wvfs_idx[i]].Signs['peaks_top']))]
            amplitudes += aux_a

            how_many_peaks = len(self.Waveforms[ordering_wvfs_idx[i]].Signs['peaks_pos'])
            aux_i = [ordering_wvfs_idx[i] for aux in range(how_many_peaks)]
            indices += aux_i

        return times, amplitudes, indices                                            

    def analyze(self, **kwargs):

        """The optional keyword arguments given to this method are handled
        to self.compute_amplitude_levels(). These are numerical parameters 
        that tune the algorithm that computes the 0.5-photoelectrons (PE) 
        and the 1.5-PE voltage amplitude.
        
        This method populates self.__timedelay, self.__amplitude and 
        self.__frame_idx, based on the peaks that are already spotted in 
        the underlying WaveformSet, p.e. via self.Waveforms.find_peaks(). 
        After calling such method, self.__timedelay[i] contains the time 
        difference between the (i+1)-th and the i-th peak in the waveform 
        set. By 'i-th' peak, I mean the peak which occurred after i-1 
        peaks have occurred. Secondly, self.__amplitude[i] 
        contains the amplitude of the (i+1)-th peak that occurred in the 
        waveform set, which accounts for baseline-correction (check 
        DarkNoiseMeas.construct_absolute_time_peaks_map() docstring for
        more information). Thirdly, self.__frame_idx[i] contains the 
        iterator value, within self.Waveforms, for the waveform where the
        (i+1)-th peak occurred. This method also computes the attributes 
        self.__half_a_pe and self.__one_and_a_half_pe via 
        self.compute_amplitude_levels(). For more information on the 
        computing-algorithm of such attributes, check the docstring of 
        DarkNoiseMeas.compute_amplitude_levels()."""

        aux_t, aux_a, aux_i = self.construct_absolute_time_peaks_map()

        if aux_t==[]:                                                                               # If self.construct_absolute_time_peaks_map() returned an empty
            raise cuex.NoAvailableData(htype.generate_exception_message("DarkNoiseMeas.analyze",    # times list, it means that no peaks had previously been spotted
                                                                        84301))                     # in the underlying waveform set, i.e. no peaks analysis had been
                                                                                                    # run before (p.e. WaveformSet.find_peaks()), and so, there's 
                                                                                                    # no available data to populate the self.__timedelay, self.__amplitude
                                                                                                    # and self.__frame_idx attributes
        aux_t, aux_a, aux_i = np.array(aux_t), np.array(aux_a), np.array(aux_i)

        self.__timedelay = np.diff(aux_t)
        self.__amplitude = aux_a[1:]
        self.__frame_idx = aux_i[1:]

        self.compute_amplitude_levels(**kwargs)
        return
    
    def compute_amplitude_levels(self,  peaks_to_detect=3,
                                        bins_no=125, 
                                        starting_fraction=0.0, 
                                        step_fraction=0.01,
                                        minimal_prominence_wrt_max=0.0,
                                        minimal_width_in_samples=0,
                                        std_no=3.,
                                        timedelay_cut=0.0):
        
        """ This method gets the following optional keyword arguments:

       - peaks_to_detect (scalar integer): It must be positive (>0). It is
        given to the 'peaks_to_detect' keyword argument of 
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(). It matches 
        the number of peaks of the self.__amplitude histogram, which will be 
        detected to start with. However, only the first and the second detected 
        peaks will be fit (those which occur for the smallest x-values of the
        self.__amplitude histogram). In order to compute the one-and-a-half 
        PE amplitude, we need to fit the 1-PE and the 2-PE peaks. For ideal 
        datasets, where the 1-PE and 2-PE peaks of the self.__amplitude are 
        the highest ones, it suffices to set peaks_to_detect to 2. However, 
        for less ideal cases, where there are >2-PE peaks which are actually 
        higher than those of <3-PE, manual peak discrimination should be done. 
        An strategy to work around these cases, where there are N >2-PE peaks 
        which are higher than those of <3-PE, is to set peaks_to_detect to 2+N 
        and peaks_to_fit to (0,1).
        - bins_no (scalar integer): It must be positive (>0). It is the number 
        of bins which are used to histogram the amplitudes of the peaks in 
        the underlying waveform set.
        - starting_fraction (scalar float): It must be semipositive (>=0.0) 
        and smaller or equal to 1 (<=1.0). It is given to the static method
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(), which in
        turn, gives it to SiPMMeas.tune_peak_height(). Check its docstrings 
        for more information.
        - step_fraction (scalar float): It must be positive (>0.0) and smaller 
        or equal to 1 (<=1.0). It is given to the static method
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(), which in
        turn, gives it to SiPMMeas.tune_peak_height(). Check its docstrings 
        for more information.
        - minimal_prominence_wrt_max (scalar float): It must be semipositive
        (>=0) and smaller or equal than 1.0 (<=1.0). It is understood as a
        fraction of the maximum value of the histogram of self.__amplitude. 
        It is given to the 'minimal_prominence_wrt_max' keyword argument of 
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(). It sets 
        a minimal prominence for a peak to be detected, based on a fraction 
        of the maximum value within the specified histogram. I.e. the only 
        considered peaks are those whose prominence is bigger or equal to a 
        fraction of the histogram maximum. For more information check the 
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks() docstring.
        - minimal_width_in_samples (scalar integer): It must be a semipositive
        (>=0) integer. It is understood as the required width of a peak (in 
        samples), for it to be detected as a peak. It is given to the 
        'minimal_width_in_samples' keyword argument of 
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(). For more
        information, check its docstring.
        - std_no (scalar float): It must be positive (>0.0). This parameter is
        given to the std_no keyword argument of the static method
        SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(), which in
        turn, gives it to SiPMMeas.piecewise_gaussian_fits(). Check its 
        docstrings for more information.
        - timedelay_cut (scalar float): It must be semipositive (>=0.0). It 
        is used as an inclusive lower bound for the time-delay value of the
        peaks that have been spotted within the underlying WaveformSet object.
        I.e. the entries within self.__amplitude which are histogrammed in 
        order to fit the 1-PE and 2-PE peaks, are those whose matching 
        time-delay value is bigger or equal to timedelay_cut.

        This method computes the voltage amplitudes matching 0.5 and 1.5 
        photoelectrons. Those values are stored into the self.__half_a_pe and
        self.__one_and_a_half_pe attributes, respectively. To do so, this 
        method 
        
        1) filters out the entries within self.__amplitude whose matching entry
        within self.__timedelay is lower than timedelay_cut,
        2) then calls SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(),
        which generates an histogram of the filtered self.__amplitude entries,
        3) then generates a probability distribution function (pdf) out of such
        histogram,
        4) then targets the two highest peaks of such pdf which have the lowest
        amplitude value, via SiPMMeas.tune_peak_height() and scipy.signal.find_peaks(),
        5) then fits one gaussian function to each one of these two peaks
        6) and uses the fit mean of both gaussian functions to compute the desired
        attributes. Say that the fit mean of the gaussian function which fits to the
        1-PE (resp. 2-PE) peak is \mu_1 (resp. \mu_2), then the attributes are 
        computed in the following way:
            6.1) The 0.5-PE voltage amplitude is computed as \mu_1 -((\mu_2-\mu_1)/2)
            6.2) The 1.5-PE voltage amplitude is computed as (\mu_1+\mu_2)/2"""

        htype.check_type(   bins_no, int,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                67585))   
        if bins_no<1:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                    11025)) 
        htype.check_type(   starting_fraction, float, np.float64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                55892))
        if starting_fraction<0.0 or starting_fraction>1.0:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                    36107)) 
        htype.check_type(   step_fraction, float, np.float64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                84304))
        if step_fraction<=0.0 or step_fraction>1.0:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                    25201))
        htype.check_type(   minimal_prominence_wrt_max, float, np.float64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                20629))
        if minimal_prominence_wrt_max<0.0 or minimal_prominence_wrt_max>1.0:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                    41035))     
        htype.check_type(   minimal_width_in_samples, int, np.int64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                15669))
        if minimal_width_in_samples<0:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                    43512))
        htype.check_type(   std_no, float, np.float64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                69757))
        if std_no<=0.0:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                    23163))
        htype.check_type(   timedelay_cut, float, np.float64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.compute_amplitude_levels", 
                                                                                21739))   
        
        samples = Waveform.filter_infs_and_nans(self.__amplitude[self.__timedelay>=timedelay_cut], get_mask=False)  # Applying time-delay cut and
                                                                                                                    # filtering out infs and nans
        popt, _ = SiPMMeas.fit_piecewise_gaussians_to_the_n_highest_peaks(  samples,                                            # Note that we filtered out 'inf' and
                                                                            peaks_to_detect=peaks_to_detect,                    # 'nan' entries JUST FOR 1.5 PE amplitude 
                                                                            peaks_to_fit=(0,1),                                 # computation. Such entries may still
                                                                            bins_no=bins_no,                                    # be contained wihtin self.__amplitude.
                                                                            starting_fraction=starting_fraction,
                                                                            step_fraction=step_fraction,
                                                                            minimal_prominence_wrt_max=minimal_prominence_wrt_max,
                                                                            minimal_width_in_samples=minimal_width_in_samples,
                                                                            std_no=std_no,
                                                                            fit_to_density=True)

        # Assess which fit matches which peak
        amplitudes_means = (popt[0][0], popt[1][0]) if popt[0][0]<popt[1][0] else (popt[1][0], popt[0][0])

        # Compute the desired amplitude levels
        self.__half_a_pe = amplitudes_means[0]-((amplitudes_means[1]-amplitudes_means[0])/2.)
        self.__one_and_a_half_pe = (amplitudes_means[0]+amplitudes_means[1])/2.
        return
    
    def get_dark_counts_number(self):

        """This method returns an integer scalar which is the number of peaks which
        which were spotted by the last peaks-analysis (p.e. via WaveformSet.find_peaks()) 
        whose amplitude is bigger than the 0.5-PE amplitude, i.e. self.__half_a_pe.
        If no peaks-analysis has been run, i.e. if self.__amplitude is None, then this 
        function raises a cuex.NoAvailableData exception."""
        
        if self.__amplitude is None:
            raise cuex.NoAvailableData(htype.generate_exception_message("DarkNoiseMeas.get_dark_counts_number", 
                                                                        84301))

        return int(np.sum(self.__amplitude>self.__half_a_pe))   # Introducing a +/- 1 error in the number of dark counts,
                                                                # since we are not taking into account the first spotted
                                                                # peak, whose time delay is not defined and thus, its 
                                                                # amplitude is thrown away by DarkNoiseMeas.analyze()

    def get_dark_count_rate_in_mHz_per_mm2(self, sipm_sensitive_area_in_mm2):

        """This method gets the following mandatory positional argument:

        - sipm_sensitive_area_in_mm2 (float scalar): Area of the sensitive surface
        of the SiPM whose dark noise measurement matches self. This input must be
        expressed in square millimeters.
        
        This method returns a float scalar which matches the superficial density
        of dark count rate of this dark noise measurement. Such result is the number 
        of dark counts divided by the sensitive surface of the sipm and the acquisition 
        time. The returned magnitude is expressed in millihertz per square millimeters."""

        return 1000.*(self.get_dark_counts_number()/(sipm_sensitive_area_in_mm2*self.AcquisitionTime_min*60.))
    
    def get_cross_talk_number(self):

        """This method returns an integer scalar which is the number of cross-talk 
        events which were spotted in this dark noise measurement. If no peaks-analysis
        has been run (i.e. self.__timedelay is None) or self has not been analyzed
        (i.e. self.__one_and_a_half_pe is None), then this method raises a 
        cuex.NoAvailableData exception."""

        if self.__timedelay is None:
            raise cuex.NoAvailableData(htype.generate_exception_message("DarkNoiseMeas.get_cross_talk_number", 
                                                                        46111))
        elif self.__one_and_a_half_pe is None:
            raise cuex.NoAvailableData(htype.generate_exception_message("DarkNoiseMeas.get_cross_talk_number", 
                                                                        27383))
        return int(np.sum(self.__amplitude>self.__one_and_a_half_pe))
    
    def get_cross_talk_probability(self):

        """This method returns a float scalar which is the cross-talk event 
        probability. Such quantity is computed as the number of cross talk events, 
        as returned by self.get_cross_talk_number(), divided by the number of dark
        counts as returned by self.get_dark_counts_number()."""

        return self.get_cross_talk_number()/self.get_dark_counts_number()
    
    def get_after_pulse_number(self, afterpulse_threshold_in_s=5e-6):               # The after pulse threshold which was used by
                                                                                    # M. A. García et al is 5 microseconds.
        """This method gets the following optional keyword argument:
        
        - afterpulse_threshold_in_s (scalar float): This input is interpreted
        as a time value in seconds. Any two consecutive peaks which are less
        than afterpulse_threshold_in_s seconds apart, contribute as one
        after-pulse event.
        
        This method returns an integer scalar which is the number of after-pulse 
        events which were spotted in this dark noise measurement. If no peaks-analysis
        has been run (i.e. self.__timedelay is None), then this method raises a 
        cuex.NoAvailableData exception."""
         
        if self.__timedelay is None:
            raise cuex.NoAvailableData(htype.generate_exception_message("DarkNoiseMeas.get_after_pulse_number", 
                                                                        25329))
        return int(np.sum(self.__timedelay<afterpulse_threshold_in_s))
    
    def get_after_pulse_probability(self):

        """This method returns a float scalar which is the after-pulse event 
        probability. Such quantity is computed as the number of after pulse events, 
        as returned by self.get_after_pulse_number(), divided by the number of dark
        counts as returned by self.get_dark_counts_number()."""    

        return self.get_after_pulse_number()/self.get_dark_counts_number()

    def identify_bursts(self,   min_events_no=5,                                # The default values for the parameters
                                timedelay_threshold_in_s=0.1):      #100 ms     # given in M. García et al paper.
                                
        """This method gets the following optional keyword arguments:

        - min_events_no (scalar integer): Exclusive lower bound to the
        number of peaks in a consecutive-peaks group, for it to be 
        considered a burst.
        - timedelay_threshold_in_s (scalar float): Exclusive upper 
        bound to the time delay between every two time-adjacent peaks 
        in a consecutive-peaks group, for such group to be considered 
        a burst.

        The functioning of this method relies on the concept of burst.

        - Every group of time-consecutive peaks within the underlying
        WaveformSet which contains more than min_events_no peaks,
        meeting the requirement that the time delay between every two
        time-adjacent peaks is smaller than timedelay_threshold_in_s 
        seconds, is a burst.

        This method returns four lists of integers. The first two,
        say bursts_init_frame and bursts_end_frame, match in length. 
        bursts_init_frame[i] (resp. bursts_end_frame[i]) matches the 
        iterator value of the fast frame, with respect to 
        self.Waveforms, where the i-th identified burst started (resp. 
        ended). The third and fourth returned list, say 
        bursts_init_peak and bursts_end_peak, match in length. They
        are analogous to the first two lists, but their entries are
        iterator values for the spotted peaks in the WaveformSet,
        rather than iterator values for the frames. I.e. 
        bursts_init_peak[i] (resp. bursts_end_peak[i]) matches the 
        iterator value, with respect to self.__timedelay, of the first 
        (resp. last) peak of the i-th identified burst."""

        htype.check_type(   min_events_no, int, np.int64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.identify_bursts", 
                                                                                57281))
        htype.check_type(   timedelay_threshold_in_s, float, np.float64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.identify_bursts", 
                                                                                23134))
        bursts_init_frame,  bursts_end_frame    = [], []
        bursts_init_peak,   bursts_end_peak     = [], []

        i=0
        while i<len(self.__timedelay)-min_events_no-1:  # Setting this upper bound to i, so that in
                                                        # the next if statement, at most we will query
                                                        # self.__timedelay[len(self.__timedelay)-min_events_no-2:len(self.__timedelay)-1]
            if np.max(self.__timedelay[i:i+min_events_no+1])<timedelay_threshold_in_s:      # Condition to start 
                bursts_init_frame.append(self.__frame_idx[i])                               # considering a burst
                bursts_init_peak.append(i)

                j = i+min_events_no+1   # j is the iterator with which we will
                                        # look for new members of this burst.
                                        # I set its starting value to i+min_events_no+1, 
                                        # since this one is actually excluded from 
                                        # timedelay[i:i+min_events_no+1]
                try:
                    while self.__timedelay[j]<timedelay_threshold_in_s:     # Look for new members of this burst
                        j += 1
                except IndexError:  # This is what happens when j reaches len(timedelay)
                                                                    # If the condition timedelay[j]<timedelay_threshold,
                                                                    # (with j==len(timedelay)) was evaluated, it's because
                    bursts_end_frame.append(self.__frame_idx[j-1])  # timedelay[len(timedelay)-1] was smaller than 
                    bursts_end_peak.append(j-1)                     # timedelay_threshold, then add it to the burst.
                    break   # Break out of while
                
                bursts_end_frame.append(self.__frame_idx[j-1])  # Adding the iterator value for
                                                                # the fast frame where the last
                                                                # last peak of the burst occurred
                bursts_end_peak.append(j-1)

                i = j+1 # Currently, j points to a peak which does
                        # not meet timedelay[j]<timedelay_threshold.
                        # Let i hop directly to j+1
            else:
                i += 1
        return bursts_init_frame, bursts_end_frame, bursts_init_peak, bursts_end_peak

    def plot_timedelay_vs_amplitude(self,   axes, 
                                            mode='2dhist', 
                                            nbins=50, 
                                            axes_title=None,
                                            plot_half_a_pe_level=False):

        """This method gets the following mandatory positional argument:
        
        - axes (matplotlib.axes._axes.Axes)
        
        This method also gets the following optional keyword argument:
        
        - mode (str): It defaults to '2dhist'. In such case, the plot
        is a 2D-histogram. If 'scatter' is given to this parameter,
        then the plot is an scatter plot. Any other input is interpreted
        as '2dhist'.

        - nbins (scalar integer): This parameter only makes a difference 
        for the 2D-histogram case. In such case, it is interpreted as the
        number of bins for both, the time and amplitude dimensions.
        It is given to the 'bins' parameter of axes.hist2d.

        - axes_title (None or str): This parameter will be passed to 
        axes.set_title(). It matches the title of the axes.

        - plot_half_a_pe_level (scalar boolean): If this parameter is 
        true and the attribute self.__half_a_pe is defined, then it is 
        plotted as an horizontal dotted line.
        
        This method considers the set of 2D-points, so that the first
        (resp. second) coordinate of the i-th point is given by 
        self.__timedelay[i] (resp. self.__amplitude[i]). The resulting 
        set of points represented either as a 2D-histogram or an scatter 
        plot, up to the value given to the 'mode' parameter."""

        htype.check_type(   axes, matplotlib.axes._axes.Axes,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.plot_timedelay_vs_amplitude", 
                                                                                18281))
        htype.check_type(   mode, str,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.plot_timedelay_vs_amplitude", 
                                                                                31738))
        htype.check_type(   nbins, int, np.int64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.plot_timedelay_vs_amplitude", 
                                                                                31771))
        if axes_title is not None:
            htype.check_type(   axes_title, str,
                                exception_message=htype.generate_exception_message( "DarkNoiseMeas.plot_timedelay_vs_amplitude", 
                                                                                    94709))
        htype.check_type(   plot_half_a_pe_level, bool,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.plot_timedelay_vs_amplitude", 
                                                                                35129))
        y, mask = Waveform.filter_infs_and_nans(self.__amplitude, 
                                                get_mask=True)
        x = self.__timedelay[mask]

        axes.set_title(axes_title)

        if 'signal_magnitude' in self.Waveforms[0].Signs.keys():
            if 'signal_unit' in self.Waveforms[0].Signs.keys():
                axes.set_ylabel(f"{self.Waveforms[0].Signs['signal_magnitude'][0]} amplitude ({self.Waveforms[0].Signs['signal_unit'][0]})")
            else:
                axes.set_ylabel(f"{self.Waveforms[0].Signs['signal_magnitude'][0]} amplitude (a.u.)")

        if plot_half_a_pe_level:
            if self.__half_a_pe is not None:
                axes.axhline( y=self.__half_a_pe, 
                            linestyle='dotted', 
                            linewidth=1.0, 
                            color='black',
                            label='0.5 PE')

        if self.__one_and_a_half_pe is not None:
            axes.axhline( y=self.__one_and_a_half_pe, 
                        linestyle='dashed', 
                        linewidth=1.0, 
                        color='black',
                        label='1.5 PE')

        if mode=='scatter':
            if 'time_unit' in self.Waveforms[0].Signs.keys():
                axes.set_xlabel(f"Time delay ({self.Waveforms[0].Signs['time_unit'][0]})")    # Taking time unit from the first 
                                                                                        # waveform in the waveform set 
            else:
                axes.set_xlabel("Time (a.u.)")

            axes.axvline( x=self.WaveformWindow_mus*1e-6, 
                        linestyle='-', 
                        linewidth=1.0, 
                        color='red',
                        label='FastFrame window limit')

            axes.set_xscale('log')
            axes.scatter(   x, y,
                            marker='x')
        else:
            if 'time_unit' in self.Waveforms[0].Signs.keys():
                axes.set_xlabel(r"log$_{10}$("+f"time delay) ({self.Waveforms[0].Signs['time_unit'][0]})")    # Taking time unit from the first 
                                                                                        # waveform in the waveform set 
            else:
                axes.set_xlabel("Time (a.u.)")

            axes.axvline( x=np.log10(self.WaveformWindow_mus*1e-6), 
                        linestyle='-', 
                        linewidth=1.0, 
                        color='red',
                        label='FastFrame window limit')

            hist = axes.hist2d( np.log10(x),
                                y, 
                                bins=nbins,
                                cmap='inferno',
                                cmin=1)
            plt.colorbar(   hist[3], 
                            orientation='vertical', ax=axes)

        if plot_half_a_pe_level:
            if self.__half_a_pe is not None:
                axes.set_ylim(bottom=self.__half_a_pe -0.005)

        axes.grid()
        axes.legend()
        return
    
    @classmethod
    def from_json_file(cls, darknoisemeas_config_json):

        """This class method is meant to be an alternative initializer
        for DarkNoiseMeas. This class method gets the following mandatory
        positional argument:
        
        - darknoisemeas_config_json (string): Path to a json file which
        hosts all of the necessary information to define the DarkNoiseMeas
        object.

        This method creates and returns a SiPMMeas object that is
        crafted out of the given json file. To do so, first, this 
        method populates two RigidKeyDictionary's. The first one, 
        say RKD1, which concerns the DarkNoiseMeas attributes, has the
        following potential keys:

        "strip_ID", "meas_ID", "date", "location", "operator", 
        "setup_ID", "system_characteristics", "thermal_cycle",
        "elapsed_cryo_time_min", "electronic_board_number", 
        "electronic_board_location", "electronic_board_socket", 
        "sipm_location", "sampling_ns", "overvoltage_V", "PDE", 
        "status", "acquisition_time_min", "threshold_mV" and 
        "wvfset_json_filepath".

        Although "sampling_ns" appears here, it's is not meant to be
        read from darknoisemeas_config_json. The value for 
        self.__sampling_ns will be duplicated from the value given to 
        "time_resolution" in the file given to wvfset_json_filepath.

        The second one, say RKD2, concerns the WaveformSet.from_files
        parameters, and is populated out of the file whose path is
        given to the wvfset_json_filepath key of RKD1. It has the following
        potential keys:

        "wvf_filepath", "time_resolution", "points_per_wvf", 
        "wvfs_to_read", "separator", "timestamp_filepath", 
        "delta_t_wf", "set_name", "creation_dt_offset_min" and
        "wvf_extra_info".

        Here, we do not expect a date because the date information 
        is taken from the DarkNoiseMeas json file.
 
        These potential keys are typed according to the 
        DarkNoiseMeas.__init__ docstring. To populate RKD1 and RK2, 
        this method uses the dictionaries which are loaded from the 
        specified json files. Namely, every entry that belongs to one of 
        the two json dictionaries and is suitably formatted, up to its 
        corresponding RigidKeyDictionary rules, is added to its 
        RigidKeyDictionary. 
        
        Once both RigidKeyDictionary's have been populated, a 
        DarkNoiseMeas object is created by calling the class initializer 
        using the key-value pairs of RKD1 and RKD2 as the kwarg-value 
        pairs for the initializer call, in that order. I.e. the class 
        initializer is called with **RKD1, **RKD2. The only exceptions 
        are the values given to "wvf_filepath" and "time_resolution" in 
        RKD1, which are passed as positional arguments, in such order, 
        to the class initializer. "wvfset_json_filepath" is also an 
        exception, since it is used to populate RKD2, and it's deleted 
        afterwards."""
        
        htype.check_type(   darknoisemeas_config_json, str,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.from_json_file", 
                                                                                67497))
        
        pks1 = {'strip_ID':int, 'meas_ID':str, 'date':str,          # These are used
                'location':str, 'operator':str,  'setup_ID':str,    # to configure
                'system_characteristics':str, 'thermal_cycle':int,  # the DarkNoiseMeas
                'elapsed_cryo_time_min':float,                      # attributes
                'electronic_board_number':int,
                'electronic_board_location':str, 
                'electronic_board_socket':int, 
                'sipm_location':int, 'sampling_ns':float, 
                'overvoltage_V':float, 'PDE':float, 'status':str,
                'acquisition_time_min':float, 'threshold_mV':float,
                'wvfset_json_filepath':str}

        pks2 = {'wvf_filepath':str, 'time_resolution':float,        # These are used to
                'points_per_wvf':int, 'wvfs_to_read':int,           # configure the 
                'separator':str, 'timestamp_filepath':str,          # WaveformSet
                'delta_t_wf':float, 'set_name':str, 
                'creation_dt_offset_min':float,  'wvf_extra_info':str}

        RKD1 = RigidKeyDictionary(  list(pks1.keys()), 
                                    is_typed=True, 
                                    values_types=list(pks1.values()))
        
        RKD2 = RigidKeyDictionary(  list(pks2.keys()), 
                                    is_typed=True, 
                                    values_types=list(pks2.values()))

        with open(darknoisemeas_config_json, "r") as file:
            input_data = json.load(file)
        RKD1.update(input_data)

        try:
            wvfset_json_filepath = RKD1['wvfset_json_filepath']
        except KeyError:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.from_json_file", 
                                                                                    45581,
                                                                                    extra_info="No filepath to the waveform set was provided."))
        del RKD1['wvfset_json_filepath']

        with open(wvfset_json_filepath, "r") as file:
            input_data = json.load(file)
        RKD2.update(input_data)

        RKD1['sampling_ns'] = 1e+9*RKD2['time_resolution']  # Unless otherwise stated, all of 
                                                            # the time values are given in seconds.
                                                            # However, as its name indicates, 
                                                            # sampling_ns is expressed in nanoseconds.
                                                            # Thus, here I am converting time_resolution,
                                                            # which is given in seconds, to nanoseconds.
        input_filepath = RKD2['wvf_filepath']
        del RKD2['wvf_filepath']

        time_resolution = RKD2['time_resolution']
        del RKD2['time_resolution']

        return cls(input_filepath, time_resolution, **RKD1, **RKD2)

    @classmethod
    def purge_bursts(cls,   darknoisemeas_to_purge,
                            min_events_no=5,                            # The default values for these
                            timedelay_threshold_in_s=0.1):  # 100 ms    # parameters are given in M. García
                                                                        # et al paper.

        """This class method gets the following mandatory positional 
        argument:
        
        - darknoisemeas_to_purge (DarkNoiseMeas): It must have been 
        previously analyzed, p.e. via DarkNoiseMeas.analyze(). I.e. 
        darknoisemeas_to_purge.__amplitude, 
        darknoisemeas_to_purge.__timedelay 
        and darknoise_to_purge.__frame_idx must be defined. The bursts
        purge will be performed on a copy of this object.

        This class method gets the following optional keyword arguments:

        - min_events_no (scalar integer): Exclusive lower bound to the
        number of peaks in a consecutive-peaks group, for it to be 
        considered a burst.
        - timedelay_threshold_in_s (scalar float): Exclusive upper 
        bound to the time delay between every two time-adjacent peaks 
        in a consecutive-peaks group, for such group to be considered 
        a burst.

        This class method returns a DarkNoiseMeas object. Such object
        is a modified copy of darknoisemeas_to_purge. The modification
        relies on the concept of burst, which is defined as follows:

        - Every group of time-consecutive peaks within the underlying
        WaveformSet which contains more than min_events_no peaks,
        meeting the requirement that the time delay between every two
        time-adjacent peaks is smaller than timedelay_threshold_in_s 
        seconds, is a burst.

        This class method spots every burst within the given DarkNoiseMeas
        object, via its identify_bursts() method, up the given parameters. 
        Then, for every burst, every waveform which contains at least one 
        peak which belongs to the burst, is removed from its underlying 
        WaveformSet object. The self.__timedelay, self.__amplitude and 
        self.__frame_idx attributes are also purged, meaning that the 
        entries (and only those entries) which match any peak belonging
        to a burst, are also removed from the three attributes at once.
        In this way, the analysis that is inherited from 
        darknoisemeas_to_purge is not destroyed, but filtered."""

        # I doubted whether I should implement this tool as a class method 
        # or as a regular method. However, since the purpose of this tool 
        # is not to modify a given DarkNoiseMeas object, but to modify a 
        # copy of it, I guess it's more (conceptually) natural to implement 
        # this as a class method.

        htype.check_type(   darknoisemeas_to_purge, DarkNoiseMeas,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.purge_bursts", 
                                                                                47193))
        htype.check_type(   min_events_no, int, np.int64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.purge_bursts", 
                                                                                19581))
        htype.check_type(   timedelay_threshold_in_s, float, np.float64,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.purge_bursts", 
                                                                                45153))
        p = darknoisemeas_to_purge.TimeDelay is None
        q = darknoisemeas_to_purge.Amplitude is None
        r = darknoisemeas_to_purge.FrameIDX is None

        if p or q or r:
            raise RuntimeError(htype.generate_exception_message("DarkNoiseMeas.purge_bursts", 
                                                                47100,
                                                                extra_info="The provided DarkNoiseMeas object must have been already analyzed."))
        purged_copy = copy.deepcopy(darknoisemeas_to_purge)
        bursts_init_frame, bursts_end_frame, bursts_init_peak, bursts_end_peak = darknoisemeas_to_purge.identify_bursts(min_events_no=min_events_no,
                                                                                                                        timedelay_threshold_in_s=timedelay_threshold_in_s)
        p = DarkNoiseMeas.lists_are_intertwined(bursts_init_frame, bursts_end_frame)
        q = DarkNoiseMeas.lists_are_intertwined(bursts_init_peak, bursts_end_peak)

        if not p or not q:
            raise cuex.MalFunction(htype.generate_exception_message("DarkNoiseMeas.purge_bursts", 
                                                                    89264,
                                                                    extra_info="DarkNoiseMeas.identify_bursts is not working. Its output is not consistent."))
        for i in reversed(range(len(bursts_init_frame))):                               # Removal from WaveformSet 
            for j in reversed(range(bursts_init_frame[i], 1+bursts_end_frame[i])):      # (list) should be done from 
                purged_copy.Waveforms.remove_member_by_index(j)                         # bigger to smaller iterator values

        timedelay, amplitude, frame_idx = list(purged_copy.TimeDelay), list(purged_copy.Amplitude), list(purged_copy.FrameIDX)

        for i in reversed(range(len(bursts_init_peak))):                            # Converting the attributes to lists,
            for j in reversed(range(bursts_init_peak[i], 1+bursts_end_peak[i])):    # so that removal here needs to happen
                del timedelay[j]                                                    # from bigger to smaller iterator values
                del amplitude[j]
                del frame_idx[j]                

        purged_copy.TimeDelay   = np.array(timedelay)
        purged_copy.Amplitude   = np.array(amplitude)
        purged_copy.FrameIDX    = np.array(frame_idx)

        return purged_copy


    @staticmethod
    def lists_are_intertwined(a, b):
        
        """This static method gets two lists of integers, whose lengths
        must match and be bigger or equal to 2. For every i in the interval 
        [0,len(a)], this method checks that a[i]<=b[i]. Also, for every i 
        in [0, len(a)-1], this function checks that b[i]<a[i+1]. If one or
        more checks result in False, then this function returns False. This 
        function returns True if else."""

        htype.check_type(   a, list,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.lists_are_intertwined", 
                                                                                29141))
        htype.check_type(   b, list,
                            exception_message=htype.generate_exception_message( "DarkNoiseMeas.lists_are_intertwined", 
                                                                                33897))
        if len(a)!=len(b):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.lists_are_intertwined", 
                                                                                    24930))
        if len(a)<2:
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.lists_are_intertwined", 
                                                                                    29830))
        for i in range(len(a)):
            htype.check_type(   a[i], int, np.int64,
                                exception_message=htype.generate_exception_message( "DarkNoiseMeas.lists_are_intertwined", 
                                                                                    13200))
            htype.check_type(   b[i], int, np.int64,
                                exception_message=htype.generate_exception_message( "DarkNoiseMeas.lists_are_intertwined", 
                                                                                    21868))
        for i in range(len(a)-1):
            p = a[i]<=b[i]
            q = b[i]<a[i+1]
            if not p or not q:
                return False
        last_p = a[-1]<=b[-1]
        if not last_p:
            return False
        return True
    
    def output_summary(self,    folderpath,
                                *args,
                                overwrite=False,
                                additional_entries={}, 
                                verbose=False,

                                **kwargs):

        """This method gets the following positional argument:

        - folderpath (string): Path which must point to an existing folder.
        It is the folder where the output json file will be saved.
        - args: Extra positional arguments which are given to
        self.get_dark_count_rate_in_mHz_per_mm2.

        This method gets the following keyword arguments:

        - overwrite (bool): This parameter only makes a difference if there
        is already a file in the given folder path whose name matches 
        f"DN-{self.StripID}-{self.SiPMLocation}-{self.ThermalCycle}-{self.Date.strftime('%Y-%m-%d')}.json".
        If that is the case, and overwrite is False, then this method does
        not generate any json file. In any other case, this method generates
        a new json file with the previously specified name in the given
        folder. In this case, overwriting may occur.
        - additional_entries (dictionary): The output dictionary, i.e. the
        dictionary which is loaded into the output json file, is updated
        with this dictionary, additional_entries, right before being loaded
        to the output json file. This update is done via the 'update' method
        of dictionaries. Hence, note that if any of the keys within
        additional_entries.keys() already exists in the output dictionary,
        it will be overwritten. Below, you can consult the keys that will 
        be part of the output dictionary by default.
        - verbose (bool): Whether to print functioning-related messages.
        - kwargs: Included so that this signature matches that of the
        overrided method. It is not used, although it may be used in the
        future.
        
        The goal of this method is to produce a summary of this DarkNoiseMeas 
        object, in the form of a json file. This json file has as many fields 
        as objects of interest which should be summarized. These fields are:

        - "strip_ID": Contains self.StripID
        - "meas_ID": Contains self.MeasID
        - "date": Contains self.Date
        - "location": Contains self.Location
        - "operator": Contains self.Operator
        - "setup_ID": Contains self.SetupID
        - "system_characteristics": Contains self.SystemCharacteristics
        - "thermal_cycle": Contains self.ThermalCycle
        - "elapsed_cryo_time_min": Contains self.ElapsedCryoTimeMin
        - "electronic_board_number": Contains self.ElectronicBoardNumber
        - "electronic_board_location": Contains self.ElectronicBoardLocation
        - "electronic_board_socket": Contains self.ElectronicBoardSocket
        - "sipm_location": Contains self.SiPMLocation
        - "sampling_ns": Contains self.Sampling_ns
        - "waveform_window_mus": Contains self.WaveformWindow_mus
        - "overvoltage_V": Contains self.Overvoltage_V
        - "PDE": Contains self.PDE
        - "N_events": Contains self.NEvents
        - "status": Contains self.Status

        - "acquisition_time_min": Contains self.__acquisition_time_min,
        - "threshold_mV": Contains self.__threshold_mV,
        - "timedelay": Contains self.__timedelay,
        - "amplitude": Contains self.__amplitude,
        - "frame_idx": Contains self.__frame_idx,
        - "half_a_pe": Contains self.__half_a_pe,
        - "one_and_a_half_pe": Contains self.__one_and_a_half_pe,
        - "DC#" Contains :self.get_dark_counts_number(),
        - "DCR_mHz_per_mm2": Contains self.get_dark_count_rate_in_mHz_per_mm2(*args),
        - "XTP": Contains self.get_cross_talk_probability(),
        - "APP": Contains self.get_after_pulse_probability()

        The summary json file is saved within the given folder, up to folderpath.
        Its name matches the following formatted string:

        f"DN-{self.StripID}-{self.SiPMLocation}-{self.ThermalCycle}-{self.Date.strftime('%Y-%m-%d')}.json"
        """
        
        htype.check_type(   folderpath, str,
                            exception_message=htype.generate_exception_message("DarkNoiseMeas.output_summary", 75697))
        
        if not os.path.exists(folderpath):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.output_summary", 
                                                                                    22733))
        elif not os.path.isdir(folderpath):
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "DarkNoiseMeas.output_summary", 
                                                                                    72052))
        htype.check_type(   overwrite, bool,
                            exception_message=htype.generate_exception_message("DarkNoiseMeas.output_summary", 92284))
        
        htype.check_type(   additional_entries, dict,
                            exception_message=htype.generate_exception_message("DarkNoiseMeas.output_summary", 71007))
        
        htype.check_type(   verbose, bool,
                            exception_message=htype.generate_exception_message("DarkNoiseMeas.output_summary", 79529))
        
        output_filename = f"DN-{self.StripID}-{self.SiPMLocation}-{self.ThermalCycle}-{self.Date.strftime('%Y-%m-%d')}.json"
        output_filepath = os.path.join(folderpath, output_filename)

        if os.path.exists(output_filepath):
            if not overwrite:   # No need to assemble the ouptut dictionary if the output
                                # filepath already exists and we are not allowed to overwrite it
                if verbose: 
                    print(f"In function DarkNoiseMeas.output_summary(): {output_filepath} already exists. It won't be overwritten.")
                return
            else:
                if verbose: 
                    print(f"In function DarkNoiseMeas.output_summary(): {output_filepath} already exists. It will be overwritten.")

        output = {  "strip_ID": self.StripID, 
                    "meas_ID": self.MeasID,
                    "date": self.Date.strftime("%Y-%m-%d %H:%M:%S"),    # Object of type datetime is not
                                                                        # JSON serializable, but strings are
                    "location": self.Location,
                    "operator": self.Operator,
                    "setup_ID": self.SetupID,
                    "system_characteristics": self.SystemCharacteristics,
                    "thermal_cycle": self.ThermalCycle,
                    "elapsed_cryo_time_min": self.ElapsedCryoTimeMin,
                    "electronic_board_number": self.ElectronicBoardNumber,
                    "electronic_board_location": self.ElectronicBoardLocation,
                    "electronic_board_socket": self.ElectronicBoardSocket,
                    "sipm_location": self.SiPMLocation,
                    "sampling_ns": self.Sampling_ns,
                    "waveform_window_mus": self.WaveformWindow_mus,
                    "overvoltage_V": self.Overvoltage_V,
                    "PDE": self.PDE,
                    "N_events": self.NEvents,
                    "status": self.Status,

                    "acquisition_time_min":self.__acquisition_time_min,
                    "threshold_mV":self.__threshold_mV,
                    "timedelay":list(self.__timedelay),     # Object of type numpy.ndarray is not 
                    "amplitude":list(self.__amplitude),     # JSON serializable, but lists are     
                    "frame_idx":[int(aux) for aux in self.__frame_idx],     # Need the casting to python-built-in int type,
                                                                            # because np.int64 is not JSON serializable.
                                                                            # This is actually a python open issue:
                                                                            # https://bugs.python.org/issue24313
                    "half_a_pe":self.__half_a_pe,
                    "one_and_a_half_pe":self.__one_and_a_half_pe,
                    "DC#":self.get_dark_counts_number(),
                    "DCR_mHz_per_mm2":self.get_dark_count_rate_in_mHz_per_mm2(*args),
                    "XTP":self.get_cross_talk_probability(),
                    "APP":self.get_after_pulse_probability()}
        
        output.update(additional_entries)
        
        with open(output_filepath, 'w') as file:
            json.dump(output, file)

        if verbose:
            print(f"In function DarkNoiseMeas.output_summary(): The output file has been written to {output_filepath}.")

        return