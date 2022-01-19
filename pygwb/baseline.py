import warnings

import numpy as np
from bilby.core.utils import create_frequency_series

from .notch import StochNotchList
from .orfs import calc_orf
from .spectral import coarse_grain_spectrogram, cross_spectral_density


class Baseline(object):
    def __init__(
        self,
        name,
        interferometer_1,
        interferometer_2,
        duration=None,
        frequencies=None,
        calibration_epsilon=0,
        notch_list=None,
        do_overlap=False,
        overlap_factor=0.5,
        zeropad_psd=False,
        zeropad_csd=True,
        window_fftgram="hann",
        do_overlap_welch_psd=True,
        welch_psd_duration=32.0,
    ):
        """
        Parameters
        ----------
        name: str
            Name for the baseline, e.g H1H2
        interferometer_1/2: bilby Interferometer object
            the two detectors spanning the baseline
        duration: float, optional
            the duration in seconds of each data segment in the interferometers. None by default, in which case duration is inherited from the interferometers.
        frequencies: array_like, optional
            the frequency array for the Baseline and
            interferometers
        calibration_epsilon: float, optional
            calibration uncertainty for this baseline
        notch_list: str, optional
            filename of the baseline notch list
        do_overlap: bool, optional
            if True, implements overlaps in the psd and csd estimation from the data with the factor declared by overlap_factor
        overlap_factor: float, optional
            factor by which to overlap the segments in the psd and csd estimation
        zeropad_psd: bool, optional
            if True, applies zeropadding in the psd estimation. False by default.
        zeropad_csd: bool, optional
            if True, applies zeropadding in the csd estimation. True by default.
        window_fftgram: str, optional
            what type of window to use to produce the fftgrams
        do_overlap_welch_psd: bool, optional
            if True, implements a 50% overlap in the welch calculation of each segment psd
        welch_psd_duration: float, optional
            duration of each psd used in the welch estimation of the average psd
        """
        self.name = name
        self.interferometer_1 = interferometer_1
        self.interferometer_2 = interferometer_2
        self.calibration_epsilon = calibration_epsilon
        self.notch_list = notch_list
        self.do_overlap = do_overlap
        self.overlap_factor = overlap_factor
        self.zeropad_psd = zeropad_psd
        self.zeropad_csd = zeropad_csd
        self.window_fftgram = window_fftgram
        self.do_overlap_welch_psd = do_overlap_welch_psd
        self.welch_psd_duration = welch_psd_duration
        self._tensor_orf_calculated = False
        self._vector_orf_calculated = False
        self._scalar_orf_calculated = False
        self._gamma_v_calculated = False
        self.set_duration(duration)
        self.set_frequencies(frequencies)
        self.minimum_frequency = max(
            interferometer_1.minimum_frequency, interferometer_2.minimum_frequency
        )
        self.maximum_frequency = min(
            interferometer_1.maximum_frequency, interferometer_2.maximum_frequency
        )

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        else:
            return all(
                [
                    getattr(self, key) == getattr(other, key)
                    for key in [
                        "name",
                        "interferometer_1",
                        "interferometer_2",
                        "calibration_epsilon",
                        "duration",
                        "frequencies",
                    ]
                ]
            )

    @property
    def tensor_overlap_reduction_function(self):
        if not self._tensor_orf_calculated:
            self._tensor_orf = self.calc_baseline_orf("tensor")
            self._tensor_orf_calculated = True
        return self._tensor_orf

    @property
    def overlap_reduction_function(self):
        return self.tensor_overlap_reduction_function

    @property
    def vector_overlap_reduction_function(self):
        if not self._vector_orf_calculated:
            self._vector_orf = self.calc_baseline_orf("vector")
            self._vector_orf_calculated = True
        return self._vector_orf

    @property
    def scalar_overlap_reduction_function(self):
        if not self._scalar_orf_calculated:
            self._scalar_orf = self.calc_baseline_orf("scalar")
            self._scalar_orf_calculated = True
        return self._scalar_orf

    def set_frequency_mask(self, notch_list):
        mask = (self.frequencies >= self.minimum_frequency) & (
            self.frequencies <= self.maximum_frequency
        )
        if notch_list is not None:
            notch_list = StochNotchList.load_from_file(notch_list)
            _, notch_mask = notch_list.get_idxs(self.frequencies)
            mask = np.logical_and(mask, notch_mask)
        return mask

    @property
    def gamma_v(self):
        if not self._gamma_v_calculated:
            self._gamma_v = self.calc_baseline_orf("right_left")
            self._gamma_v_calculated = True
        return self._gamma_v

    def set_duration(self, duration):
        """Sets the duration for the Baseline and interferometers

        If `duration` is passed, check that it matches the `duration`
        in the interferometers, if present.
        If not passed, check that the durations in the interferometers
        match each other, if present, and set the Baseline duration from
        the interferometer durations.
        If not passed and only one of the interferometers has the duration
        set, set the Baseline duration and duration for the other
        interferometer from that.
        Requires that either `duration` is not None or at least one of the
        interferometers has the `duration` set.

        Parameters
        ==========
        duration: float, optional
            The duration to set for the Baseline and interferometers
        """
        if duration is not None:
            self.check_durations_match_baseline_ifos(duration)
            self.duration = duration
            if not self.interferometer_1.duration:
                self.interferometer_1.duration = duration
            if not self.interferometer_2.duration:
                self.interferometer_2.duration = duration
        elif self.interferometer_1.duration and self.interferometer_2.duration:
            self.check_ifo_durations_match()
            self.duration = self.interferometer_1.duration
        elif self.interferometer_1.duration:
            self.duration = self.interferometer_1.duration
            self.interferometer_2.duration = self.interferometer_1.duration
        elif self.interferometer_2.duration:
            self.duration = self.interferometer_2.duration
            self.interferometer_1.duration = self.interferometer_2.duration
        else:
            warnings.warn("Neither baseline nor interferometer duration is set.")
            self.duration = duration

    def set_frequencies(self, frequencies):
        """Sets the frequencies array in the baseline.

        Parameters
        ==========
        frequencies: array_like, optional
        """
        if frequencies is None:
            warnings.warn("baseline frequencies have not been set.")
        self.frequencies = frequencies

    def check_durations_match_baseline_ifos(self, duration):
        if self.interferometer_1.duration and self.interferometer_2.duration:
            self.check_ifo_durations_match()
            if not duration == self.interferometer_1.duration:
                raise AssertionError(
                    "Interferometer durations do not match given Baseline duration!"
                )
        elif self.interferometer_1.duration:
            if not duration == self.interferometer_1.duration:
                raise AssertionError(
                    "Interferometer_1 duration does not match given Baseline duration!"
                )
        elif self.interferometer_2.duration:
            if not duration == self.interferometer_2.duration:
                raise AssertionError(
                    "Interferometer_2 duration does not match given Baseline duration!"
                )

    def check_ifo_durations_match(self):
        if not (self.interferometer_1.duration == self.interferometer_2.duration):
            raise AssertionError("Interferometer durations do not match each other!")

    def set_sampling_frequency(self, sampling_frequency):
        """Sets the sampling_frequency for the Baseline and interferometers

        If `sampling_frequency` is passed, check that it matches the `sampling_frequency`
        in the interferometers, if present.
        If not passed, check that the sampling_frequencies in the interferometers
        match each other, if present, and set the Baseline sampling_frequency from
        the sampling_frequencies.
        If not passed and only one of the interferometers has the sampling_frequency
        set, set the Baseline sampling_frequency and sampling_frequency for the other
        interferometer from that.
        Requires that either `sampling_frequency` is not None or at least one of the
        interferometers has the `sampling_frequency` set.

        Parameters
        ==========
        sampling_frequency: float, optional
            The sampling_frequency to set for the Baseline and interferometers
        """
        if sampling_frequency is not None:
            self.check_sampling_frequencies_match_baseline_ifos(sampling_frequency)
            self.sampling_frequency = sampling_frequency
            if not self.interferometer_1.sampling_frequency:
                self.interferometer_1.sampling_frequency = sampling_frequency
            if not self.interferometer_2.sampling_frequency:
                self.interferometer_2.sampling_frequency = sampling_frequency
        elif (
            self.interferometer_1.sampling_frequency
            and self.interferometer_2.sampling_frequency
        ):
            self.check_ifo_sampling_frequencies_match()
            self.sampling_frequency = self.interferometer_1.sampling_frequency
        elif self.interferometer_1.sampling_frequency:
            self.sampling_frequency = self.interferometer_1.sampling_frequency
            self.interferometer_2.sampling_frequency = (
                self.interferometer_1.sampling_frequency
            )
        elif self.interferometer_2.sampling_frequency:
            self.sampling_frequency = self.interferometer_2.sampling_frequency
            self.interferometer_1.sampling_frequency = (
                self.interferometer_2.sampling_frequency
            )
        else:
            raise AttributeError(
                "Need either interferometer sampling_frequency or sampling_frequency passed to __init__!"
            )

    def check_sampling_frequencies_match_baseline_ifos(self, sampling_frequency):
        if (
            self.interferometer_1.sampling_frequency
            and self.interferometer_2.sampling_frequency
        ):
            self.check_ifo_sampling_frequencies_match()
            if not sampling_frequency == self.interferometer_1.sampling_frequency:
                raise AssertionError(
                    "Interferometer sampling_frequencies do not match given Baseline sampling_frequency!"
                )
        elif self.interferometer_1.sampling_frequency:
            if not sampling_frequency == self.interferometer_1.sampling_frequency:
                raise AssertionError(
                    "Interferometer_1 sampling_frequency does not match given Baseline sampling_frequency!"
                )
        elif self.interferometer_2.sampling_frequency:
            if not sampling_frequency == self.interferometer_2.sampling_frequency:
                raise AssertionError(
                    "Interferometer_2 sampling_frequency does not match given Baseline sampling_frequency!"
                )

    def check_ifo_sampling_frequencies_match(self):
        if not (
            self.interferometer_1.sampling_frequency
            == self.interferometer_2.sampling_frequency
        ):
            raise AssertionError(
                "Interferometer sampling_frequencies do not match each other!"
            )

    def calc_baseline_orf(self, polarization):
        return calc_orf(
            self.frequencies,
            self.interferometer_1.vertex,
            self.interferometer_2.vertex,
            self.interferometer_1.x,
            self.interferometer_2.x,
            self.interferometer_1.y,
            self.interferometer_2.y,
            polarization,
        )

    @classmethod
    def from_interferometers(
        cls,
        interferometers,
        duration=None,
        calibration_epsilon=0,
    ):
        name = "".join([ifo.name for ifo in interferometers])
        return cls(
            name=name,
            interferometer_1=interferometers[0],
            interferometer_2=interferometers[1],
            duration=duration,
            calibration_epsilon=calibration_epsilon,
        )

    @classmethod
    def from_parameters(
        cls,
        interferometer_1,
        interferometer_2,
        parameters,
        frequencies=None,
        notch_list=None,
    ):
        name = interferometer_1.name + interferometer_2.name
        return cls(
            name=name,
            interferometer_1=interferometer_1,
            interferometer_2=interferometer_2,
            duration=parameters.segment_duration,
            calibration_epsilon=parameters.calibration_epsilon,
            frequencies=frequencies,
            notch_list=notch_list,
            do_overlap=parameters.do_overlap,
            overlap_factor=parameters.overlap_factor,
            zeropad_psd=parameters.zeropad_psd,
            zeropad_csd=parameters.zeropad_csd,
            window_fftgram=parameters.window_fftgram,
            do_overlap_welch_psd=parameters.do_overlap_welch_psd,
            welch_psd_duration=parameters.welch_psd_duration,
        )

    def set_cross_and_power_spectral_density(self, frequency_resolution):
        """Sets the power spectral density in each interferometer
        and the cross spectral density for the baseline object when data are available

        Parameters
        ==========
        frequency_resolution: float
            the frequency resolution at which the cross and power spectral densities are calculated.
        """
        try:
            self.interferometer_1.set_psd_spectrogram(
                frequency_resolution,
                do_overlap=self.do_overlap,
                overlap_factor=self.overlap_factor,
                zeropad=self.zeropad_psd,
                window_fftgram=self.window_fftgram,
                do_overlap_welch_psd=self.do_overlap_welch_psd,
                welch_psd_duration=self.welch_psd_duration,
            )
        except AttributeError:
            raise AssertionError(
                "Interferometer {self.interferometer_1.name} has no timeseries data! Need to set timeseries data in the interferometer first."
            )
        try:
            self.interferometer_2.set_psd_spectrogram(
                frequency_resolution,
                do_overlap=self.do_overlap,
                overlap_factor=self.overlap_factor,
                zeropad=self.zeropad_psd,
                window_fftgram=self.window_fftgram,
                do_overlap_welch_psd=self.do_overlap_welch_psd,
                welch_psd_duration=self.welch_psd_duration,
            )
        except AttributeError:
            raise AssertionError(
                "Interferometer {self.interferometer_2.name} has no timeseries data! Need to set timeseries data in the interferometer first."
            )
        self.csd = cross_spectral_density(
            self.interferometer_1.timeseries,
            self.interferometer_2.timeseries,
            self.duration,
            frequency_resolution,
            do_overlap=self.do_overlap,
            overlap_factor=self.overlap_factor,
            zeropad=self.zeropad_csd,
            window_fftgram=self.window_fftgram,
        )

    def set_average_power_spectral_densities(self):
        """If psds have been calculated, sets the average psd in each ifo"""
        try:
            self.interferometer_1.set_average_psd(self.welch_psd_duration)
            self.interferometer_2.set_average_psd(self.welch_psd_duration)
        except AttributeError:
            print(
                "PSDs have not been calculated yet! Need to set_cross_and_power_spectral_density first."
            )

    def set_average_cross_spectral_density(self):
        """If csd has been calculated, sets the average csd for the baseline"""
        stride = self.duration * (1 - self.overlap_factor)
        csd_segment_offset = int(np.ceil(self.duration / stride))
        try:
            self.average_csd = coarse_grain_spectrogram(self.csd)[
                csd_segment_offset : -(csd_segment_offset + 1) + 1
            ]
        except AttributeError:
            print(
                "CSD has not been calculated yet! Need to set_cross_and_power_spectral_density first."
            )
    
    def save_data(self, save_data_type, filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new):    
        """Saves the overall point estimate Y_pygwb_new, its error bar sigma_pyGWB_new, 
        the frequency-dependent estimates and variances and the corresponding frequencies in the required save_data_type, which can be npz, pickle or json.
        You can call upon this data afterwards when loaoding in using the ['key'] dictionary format.
        
        Parameters
        ==========
        save_data_type: str
            The required type of data file where the information will be stored
        filename: str
            the path/name of the file in which you want to save
        freqs: array
            Array of frequencies that correspond with the frequency-dependent estimates
        Y_f_new: array
            The frequency-dependent estimates
        var_f_new: array 
            The frequency-dependent variances on the estimates
        Y_pyGWB_new: float
            The overall point estimate
        sigma_pyGWB_new: float
            The errorbar of the overall point estimate
        
        """
        
        if save_data_type == 'pickle': 
            filename = filename + '.p'
            self.pickle_save(filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new)
        
        elif save_data_type == 'npz': 
            self.save_data_to_file(filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new)
            
        elif save_data_type == 'json':
            filename = filename + '.json'
            self.json_save(filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new)
        
        else:
            raise ValueError("The provided data type is not supported, try using 'pickle', 'npz' or 'json' instead.")
    
    def save_data_to_file(self, filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new):
        np.savez(filename, freqs = freqs, Y_f_new = Y_f_new, var_f_new = var_f_new, Y_pyGWB_new = Y_pyGWB_new, sigma_pyGWB_new = sigma_pyGWB_new)
        
        
    def pickle_save(self, filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new):
        #saveObject = (freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new)
        
        save_dictionary = {'freqs' : freqs, 'Y_f_new' : Y_f_new, 'var_f_new': var_f_new, 'Y_pyGWB': Y_pyGWB_new, 'sigma_pyGWB' : sigma_pyGWB_new}
        
        
        #with open(filename, "wb") as f:
         #   pickle.dump(saveObject, f)
            
        with open(filename, "wb") as f:
            pickle.dump(save_dictionary, f)
            
    def json_save(self, filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new):
        list_freqs = freqs.tolist()
        list_Yf = Y_f_new.tolist()
        list_varf = var_f_new.tolist()
        
        save_dictionary = {'freqs' : list_freqs, 'Y_f_new' : list_Yf, 'var_f_new': list_varf, 'Y_pyGWB': Y_pyGWB_new, 'sigma_pyGWB' : sigma_pyGWB_new}
        
        with open(filename, "w") as outfile:
            json.dump(save_dictionary, outfile)
            
    
    def save_data_csd(self, save_data_type, filename, freqs, csd, psd_1, psd_2):    
        """
        Saves the computed csd and average pds together with their corresponding frequencies in the required save_data_type, which can be npz, pickle or json.
        You can call upon this data afterwards when loaoding in using the ['key'] dictionary format.
        
        Parameters
        ----------
        
        save_data_type: str
            The required type of data file where the information will be stored
        filename: str
            The path/name of the file in which you want to save
        freqs: array_like
            The corresponding frequencies of the csd and psds
        csd: spectrogram
            The computed CSD as a spectrogram, hence with corresponding times and frequencies
        psd_1, psd_2: spectrogram
            The computed with before_and_after_average psds
        """
        
        if save_data_type == 'pickle': 
            filename = filename + '.p'
            self.pickle_save_csd(filename, freqs, csd, psd_1, psd_2)
        
        elif save_data_type == 'npz': 
            self.save_data_to_file_csd(filename, freqs, csd, psd_1, psd_2)
            
        elif save_data_type == 'json':
            filename = filename + '.json'
            self.json_save_csd(filename, freqs, csd, psd_1, psd_2)
        
        else:
            raise ValueError("The provided data type is not supported, try using 'pickle', 'npz' or 'json' instead.")      
            
            
    def save_data_to_file_csd(self, filename, freqs, csd, avg_psd_1, avg_psd_2):
        np.savez(filename, freqs = freqs, csd = csd, avg_psd_1 = avg_psd_1, avg_psd_2 = avg_psd_2)
        
        
    def pickle_save_csd(self, filename, freqs, csd, psd_1, psd_2):
        #saveObject = (freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new)
        
        save_dictionary = {'freqs' : freqs, 'csd' : csd, 'avg_psd_1' : psd_1, 'avg_psd_2' : psd_2}
        
        
        #with open(filename, "wb") as f:
         #   pickle.dump(saveObject, f)
            
        with open(filename, "wb") as f:
            pickle.dump(save_dictionary, f)
            
    def json_save_csd(self, filename, freqs, csd, psd_1, psd_2):
        """
        It seems that saving spectrograms in json does not work, hence everything is converted into a list and saved that way in the json file.
        A second issue is that json does not seem to recognise complex values, hence the csd is split up into a real and imaginary part. 
        When loading in this json file, one needs to 'reconstruct' the csd as a spectrogram using these two lists and the times and frequencies. 
        """
        list_freqs = freqs.tolist()
        list_csd = csd.value.tolist()
        real_csd = np.zeros(np.shape(list_csd))
        imag_csd = np.zeros(np.shape(list_csd))
        for index, row in enumerate(list_csd):
            for j,elem in enumerate(row):
                real_csd[index,j] = elem.real
                imag_csd[index,j] = elem.imag
        real_csd_list = real_csd.tolist()
        imag_csd_list = imag_csd.tolist()
        csd_times = csd.times.value.tolist()
        list_psd_1 = psd_1.value.tolist()
        psd_times = psd_1.times.value.tolist()
        list_psd_2 = psd_2.value.tolist()
        psd_2_times = psd_2.times.value.tolist()

        save_dictionary = {'freqs' : list_freqs, 'csd_real' : real_csd_list, 'csd_imag' : imag_csd_list , 'csd_times' : csd_times ,  'avg_psd_1' : list_psd_1, 'psd_1_times' : psd_times, 'avg_psd_2' : list_psd_2, 'psd_2_times' : psd_2_times}
        
        with open(filename, "w") as outfile:
            json.dump(save_dictionary, outfile)
