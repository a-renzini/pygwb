import json
import logging
import warnings

import gwpy.frequencyseries
import gwpy.spectrogram
import numpy as np
from bilby.core.utils import create_frequency_series

from .notch import StochNotchList
from .orfs import calc_orf
from .postprocessing import postprocess_Y_sigma
from .spectral import coarse_grain_spectrogram, cross_spectral_density
from .util import calc_Y_sigma_from_Yf_varf, calculate_point_estimate_sigma_spectrogram


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
        overlap_factor=0.5,
        zeropad_csd=True,
        window_fftgram="hann",
        overlap_factor_welch_psd=0,
        N_average_segments_welch_psd=2,
        sampling_frequency=None
    ):
        """
        Parameters
        ----------
        name: str
            Name for the baseline, e.g H1H2
        interferometer_1/2: bilby Interferometer object
            the two detectors spanning the baseline
        duration: float, optional
            the duration in seconds of each data segment in the interferometers.
            None by default, in which case duration is inherited from the interferometers.
        frequencies: array_like, optional
            the frequency array for the Baseline and
            interferometers
        calibration_epsilon: float, optional
            calibration uncertainty for this baseline
        notch_list: str, optional
            filename of the baseline notch list
        overlap_factor: float, optional
            factor by which to overlap the segments in the psd and csd estimation.
            Default is 1/2, if set to 0 no overlap is performed.
        zeropad_csd: bool, optional
            if True, applies zeropadding in the csd estimation. True by default.
        window_fftgram: str, optional
            what type of window to use to produce the fftgrams
        overlap_factor_welch_psd: float, optional
            Amount of overlap between data blocks used in pwelch method (range between 0 and 1)
            (default 0, no overlap)
        N_average_segments_welch_psd: int, optional
            Number of segments used for PSD averaging (from both sides of the segment of interest)
            N_avg_segs should be even and >= 2
        """
        self.name = name
        self.interferometer_1 = interferometer_1
        self.interferometer_2 = interferometer_2
        self.calibration_epsilon = calibration_epsilon
        self.notch_list = notch_list
        self.overlap_factor = overlap_factor
        self.zeropad_csd = zeropad_csd
        self.window_fftgram = window_fftgram
        self.overlap_factor_welch_psd = overlap_factor_welch_psd
        self.N_average_segments_welch_psd = N_average_segments_welch_psd
        self._tensor_orf_calculated = False
        self._vector_orf_calculated = False
        self._scalar_orf_calculated = False
        self._gamma_v_calculated = False
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.frequencies = frequencies
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

    @property
    def duration(self):
        if self._duration_set:
            return self._duration
        else:
            raise ValueError('Duration not yet set')

    @duration.setter
    def duration(self, dur):
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
        if dur is not None:
            self.check_durations_match_baseline_ifos(dur)
            self._duration = dur
            if not self.interferometer_1.duration:
                self.interferometer_1.duration = dur
            if not self.interferometer_2.duration:
                self.interferometer_2.duration = dur
            self._duration_set = True
        elif self.interferometer_1.duration and self.interferometer_2.duration:
            self.check_ifo_durations_match()
            self.duration = self.interferometer_1.duration
            self._duration_set = True
        elif self.interferometer_1.duration:
            self.duration = self.interferometer_1.duration
            self.interferometer_2.duration = self.interferometer_1.duration
            self._duration_set = True
        elif self.interferometer_2.duration:
            self.duration = self.interferometer_2.duration
            self.interferometer_1.duration = self.interferometer_2.duration
            self._duration_set = True
        else:
            warnings.warn("Neither baseline nor interferometer duration is set.")
            self._duration = dur
            self._duration_set = True

    @property
    def frequencies(self):
        if self._frequencies_set:
            return self._frequencies
        else:
            raise ValueError('frequencies have not yet been set')

    @frequencies.setter
    def frequencies(self, freqs):
        self._frequencies = freqs
        self._frequencies_set = True
        # delete the orfs, set the calculated flag to zero
        if self._tensor_orf_calculated:
            delattr(self, '_tensor_orf')
            self._tensor_orf_calculated = False
        if self._scalar_orf_calculated:
            delattr(self, '_scalar_orf')
            self._scalar_orf_calculated = False
        if self._vector_orf_calculated:
            delattr(self, '_vector_orf')
            self._vector_orf_calculated = False

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

    @property
    def sampling_frequency(self):
        if hasattr(self, '_sampling_frequency'):
            return self._sampling_frequency
        else:
            raise ValueError('sampling frequency not set')

    @sampling_frequency.setter
    def sampling_frequency(self, sampling_frequency):
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
            self._sampling_frequency = sampling_frequency
            if not self.interferometer_1.sampling_frequency:
                self.interferometer_1.sampling_frequency = sampling_frequency
            if not self.interferometer_2.sampling_frequency:
                self.interferometer_2.sampling_frequency = sampling_frequency
            self._sampling_frequency_set = True
        elif (
            self.interferometer_1.sampling_frequency
            and self.interferometer_2.sampling_frequency
        ):
            self.check_ifo_sampling_frequencies_match()
            self._sampling_frequency = self.interferometer_1.sampling_frequency
            self._sampling_frequency_set = True
        elif self.interferometer_1.sampling_frequency:
            self.sampling_frequency = self.interferometer_1.sampling_frequency
            self.interferometer_2.sampling_frequency = (
                self.interferometer_1.sampling_frequency
            )
            self._sampling_frequency_set = True
        elif self.interferometer_2.sampling_frequency:
            self._sampling_frequency = self.interferometer_2.sampling_frequency
            self.interferometer_1.sampling_frequency = (
                self.interferometer_2.sampling_frequency
            )
            self._sampling_frequency_set = True
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
            overlap_factor=parameters.overlap_factor,
            zeropad_csd=parameters.zeropad_csd,
            window_fftgram=parameters.window_fftgram,
            overlap_factor_welch_psd=parameters.overlap_factor_welch_psd,
            N_average_segments_welch_psd=parameters.N_average_segments_welch_psd,
            sampling_frequency=parameters.new_sample_rate
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
                overlap_factor=self.overlap_factor,
                window_fftgram=self.window_fftgram,
                overlap_factor_welch_psd=self.overlap_factor_welch_psd,
                N_average_segments_welch_psd=self.N_average_segments_welch_psd,
            )
        except AttributeError:
            raise AssertionError(
                "Interferometer {self.interferometer_1.name} has no timeseries data! Need to set timeseries data in the interferometer first."
            )
        try:
            self.interferometer_2.set_psd_spectrogram(
                frequency_resolution,
                overlap_factor=self.overlap_factor,
                window_fftgram=self.window_fftgram,
                overlap_factor_welch_psd=self.overlap_factor_welch_psd,
                N_average_segments_welch_psd=self.N_average_segments_welch_psd,
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
            overlap_factor=self.overlap_factor,
            zeropad=self.zeropad_csd,
            window_fftgram=self.window_fftgram,
        )

        # TODO: make this less fragile.
        # For now, reset frequencies,
        # recalculate ORF in case frequencies have changed.
        self._tensor_orf_calculated = False
        self.frequencies = self.csd.frequencies.value

    def set_average_power_spectral_densities(self):
        """If psds have been calculated, sets the average psd in each ifo"""
        try:
            self.interferometer_1.set_average_psd(self.N_average_segments_welch_psd)
            self.interferometer_2.set_average_psd(self.N_average_segments_welch_psd)
        except AttributeError:
            print(
                "PSDs have not been calculated yet! Need to set_cross_and_power_spectral_density first."
            )

        # TODO: make this less fragile.
        # For now, recalculate ORF in case frequencies have changed.
        self._tensor_orf_calculated = False
        self.frequencies = self.csd.frequencies.value

    def set_average_cross_spectral_density(self):
        """If csd has been calculated, sets the average csd for the baseline"""
        stride = self.duration * (1 - self.overlap_factor)
        csd_segment_offset = int(np.ceil(self.duration / stride))
        try:
            self.average_csd = coarse_grain_spectrogram(self.csd)[
                csd_segment_offset: -(csd_segment_offset + 1) + 1
            ]
        except AttributeError:
            print(
                "CSD has not been calculated yet! Need to set_cross_and_power_spectral_density first."
            )
        # TODO: make this less fragile.
        # For now, recalculate ORF in case frequencies have changed.
        self._tensor_orf_calculated = False

    def crop_frequencies_average_psd_csd(self, flow, fhigh):
        """crop frequencies of average PSDs and CSDS. Done in place. This is not completely implemented yet.

        Parameters:
        ===========
            flow (float)
                low frequency
            fhigh (float)
                high frequency
        """
        deltaF = self.frequencies[1] - self.frequencies[0]
        indexes = (self.frequencies >= flow) * (self.frequencies <= fhigh)
        # reset frequencies
        self.frequencies = self.frequencies[indexes]

        if hasattr(self.interferometer_1, 'average_psd'):
            self.interferometer_1.average_psd = self.interferometer_1.average_psd.crop_frequencies(flow, fhigh + deltaF)
        if hasattr(self.interferometer_2, 'average_psd'):
            self.interferometer_2.average_psd = self.interferometer_2.average_psd.crop_frequencies(flow, fhigh + deltaF)
        if hasattr(self, 'average_csd'):
            self.average_csd = self.average_csd.crop_frequencies(flow, fhigh + deltaF)

    def set_point_estimate_sigma_spectrogram(self, params):
        """Set point estimate and sigma spectrogram. Resulting spectrogram
        *does not include frequency weighting for alpha*.
        """
        # set CSD if not set
        # self.set_average_cross_spectral_density()

        # set PSDs if not set
        # self.set_average_power_spectral_densities()

        self.crop_frequencies_average_psd_csd(params.flow, params.fhigh)

        # don't get rid of information unless we need to.
        Y_fs, var_fs = calculate_point_estimate_sigma_spectrogram(self.frequencies,
                                                                  self.average_csd,
                                                                  self.interferometer_1.average_psd,
                                                                  self.interferometer_2.average_psd,
                                                                  self.overlap_reduction_function,
                                                                  self.duration,
                                                                  self.sampling_frequency
                                                                  )
        self.point_estimate_spectrogram = gwpy.spectrogram.Spectrogram(Y_fs, times=self.average_csd.times,
                                                                       frequencies=self.average_csd.frequencies,
                                                                       name=self.name +
                                                                       ' unweighted point estimate spectrogram')
        self.sigma_spectrogram = gwpy.spectrogram.Spectrogram(np.sqrt(var_fs), times=self.average_csd.times,
                                                              frequencies=self.average_csd.frequencies,
                                                              name=self.name + ' sigma spectrogram')

    def set_point_estimate_per_segment(self, params, lines_object=None):
        if hasattr(self, 'point_estimate_spectrogram'):
            self.set_point_estimate_sigma_spectrogram(params)
        # TODO : combine over frequency for every segment

    def set_point_estimate_sigma_spectrum(self, params, badtimes=np.array([]), lines_object=None):
        """Sets time-integrated point estimate spectrum and variance in each frequency bin.
        Point estimate is *unweighted* by alpha.
        """

        # set unweighted point estimate and sigma spectrograms
        if not hasattr(self, 'point_estimate_spectrogram'):
            logging.info('Point estimate and sigma spectrograms have set yet. setting now...')
            self.set_point_estimate_sigma_spectrogram(params)
        deltaF = self.frequencies[1] - self.frequencies[0]

        if lines_object is None:
            notches = np.array([], dtype=int)
        else:
            notches, _ = lines_object.get_idxs(self.frequencies)

        # should be True for each bad time
        bad_times_indexes = np.array([np.any(t == badtimes) for t in self.point_estimate_spectrogram.times.value])

        logging.info(f'{np.sum(bad_times_indexes)} bad segments removed.')

        # start time, for metadata
        epoch = self.point_estimate_spectrogram.times[0]

        self.point_estimate_spectrogram[bad_times_indexes, :] = 0
        self.sigma_spectrogram[bad_times_indexes, :] = np.inf

        # Post process. Last argument is frequency notches. Do not include these yet.
        # Leave that for when we combine over freqs.
        point_estimate, sigma = postprocess_Y_sigma(self.point_estimate_spectrogram.value,
                                                    self.sigma_spectrogram.value**2,
                                                    self.duration,
                                                    deltaF,
                                                    self.sampling_frequency,
                                                    notches)

        self.point_estimate_spectrum = gwpy.frequencyseries.FrequencySeries(point_estimate,
                                                               frequencies=self.frequencies,
                                                               name=self.name + 'unweighted point estimate spectrum',
                                                               epoch=epoch)
        self.sigma_spectrum = gwpy.frequencyseries.FrequencySeries(np.sqrt(sigma),
                                                                   frequencies=self.frequencies,
                                                                   name=self.name + 'unweighted sigma spectrum',
                                                                   epoch=epoch)

    def set_point_estimate_sigma(self, params, lines_object=None, apply_weighting=True,
                                 badtimes=np.array([], dtype=int)):
        """Set point estimate sigma based on a set of parameters.
        """
        # set point estimate and sigma spectrum
        # this is estimate of omega_gw in each frequency bin

        # TODO: Add check if badtimes is apssed and point estimate spectrum
        # already exists...
        if not hasattr(self, 'point_estimate_spectrum'):
            logging.info('Point estimate and sigma spectra have not been set before. Setting it now...')
            self.set_point_estimate_sigma_spectrum(params, badtimes=badtimes, lines_object=lines_object)

        # crop frequencies according to params before combining over them
        deltaF = self.frequencies[1] - self.frequencies[0]
        Y_spec = self.point_estimate_spectrum.crop(params.flow, params.fhigh + deltaF)
        sigma_spec = self.sigma_spectrum.crop(params.flow, params.fhigh + deltaF)
        freq_band_cut = (self.frequencies >= params.flow) & (self.frequencies <= params.fhigh)
        self.frequencies = self.frequencies[freq_band_cut]

        # check notch list
        # TODO: make this less fragile...at the moment these indexes
        # must agree with those after cropping, so the notches must agree with the params
        # struct in some way. Seems dangerous
        if lines_object is None:
            notch_indexes = np.arange(Y_spec.size)
        else:
            _, notch_indexes = lines_object.get_idxs(Y_spec.frequencies.value)
        # get Y, sigma
        if apply_weighting:
            Y, sigma = calc_Y_sigma_from_Yf_varf(Y_spec.value[notch_indexes],
                                                 sigma_spec.value[notch_indexes]**2,
                                                 freqs=self.frequencies[notch_indexes],
                                                 alpha=params.alpha,
                                                 fref=params.fref)
        else:
            print('Be careful, in general weighting is not applied until this point')
            Y, sigma = calc_Y_sigma_from_Yf_varf(self.point_estimate_spectrum.value,
                                                 self.sigma_spectrogram.value**2)

        self.point_estimate = Y
        self.sigma = sigma

    def save_data(
        self,
        save_data_type,
        filename,
        freqs,
        Y_f_new,
        var_f_new,
        Y_pyGWB_new,
        sigma_pyGWB_new,
    ):
        """Saves the overall point estimate Y_pygwb_new, its error bar sigma_pyGWB_new,
        the frequency-dependent estimates and variances and the corresponding frequencies
        in the required save_data_type,which can be npz, pickle or json.
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

        if save_data_type == "pickle":
            filename = filename + ".p"
            self.pickle_save(
                filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new
            )

        elif save_data_type == "npz":
            self.save_data_to_file(
                filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new
            )

        elif save_data_type == "json":
            filename = filename + ".json"
            self.json_save(
                filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new
            )

        else:
            raise ValueError(
                "The provided data type is not supported, try using 'pickle', 'npz' or 'json' instead."
            )

    def save_data_to_file(
        self, filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new
    ):
        np.savez(
            filename,
            freqs=freqs,
            Y_f_new=Y_f_new,
            var_f_new=var_f_new,
            Y_pyGWB_new=Y_pyGWB_new,
            sigma_pyGWB_new=sigma_pyGWB_new,
        )

    def pickle_save(
        self, filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new
    ):
        # saveObject = (freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new)

        save_dictionary = {
            "freqs": freqs,
            "Y_f_new": Y_f_new,
            "var_f_new": var_f_new,
            "Y_pyGWB": Y_pyGWB_new,
            "sigma_pyGWB": sigma_pyGWB_new,
        }

        # with open(filename, "wb") as f:
        #   pickle.dump(saveObject, f)

        with open(filename, "wb") as f:
            pickle.dump(save_dictionary, f)

    def json_save(
        self, filename, freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new
    ):
        list_freqs = freqs.tolist()
        list_Yf = Y_f_new.tolist()
        list_varf = var_f_new.tolist()

        save_dictionary = {
            "freqs": list_freqs,
            "Y_f_new": list_Yf,
            "var_f_new": list_varf,
            "Y_pyGWB": Y_pyGWB_new,
            "sigma_pyGWB": sigma_pyGWB_new,
        }

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

        if save_data_type == "pickle":
            filename = filename + ".p"
            self.pickle_save_csd(filename, freqs, csd, psd_1, psd_2)

        elif save_data_type == "npz":
            self.save_data_to_file_csd(filename, freqs, csd, psd_1, psd_2)

        elif save_data_type == "json":
            filename = filename + ".json"
            self.json_save_csd(filename, freqs, csd, psd_1, psd_2)

        else:
            raise ValueError(
                "The provided data type is not supported, try using 'pickle', 'npz' or 'json' instead."
            )

    def save_data_to_file_csd(self, filename, freqs, csd, avg_psd_1, avg_psd_2):
        np.savez(
            filename, freqs=freqs, csd=csd, avg_psd_1=avg_psd_1, avg_psd_2=avg_psd_2
        )

    def pickle_save_csd(self, filename, freqs, csd, psd_1, psd_2):
        # saveObject = (freqs, Y_f_new, var_f_new, Y_pyGWB_new, sigma_pyGWB_new)

        save_dictionary = {
            "freqs": freqs,
            "csd": csd,
            "avg_psd_1": psd_1,
            "avg_psd_2": psd_2,
        }

        # with open(filename, "wb") as f:
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
            for j, elem in enumerate(row):
                real_csd[index, j] = elem.real
                imag_csd[index, j] = elem.imag
        real_csd_list = real_csd.tolist()
        imag_csd_list = imag_csd.tolist()
        csd_times = csd.times.value.tolist()
        list_psd_1 = psd_1.value.tolist()
        psd_times = psd_1.times.value.tolist()
        list_psd_2 = psd_2.value.tolist()
        psd_2_times = psd_2.times.value.tolist()

        save_dictionary = {
            "freqs": list_freqs,
            "csd_real": real_csd_list,
            "csd_imag": imag_csd_list,
            "csd_times": csd_times,
            "avg_psd_1": list_psd_1,
            "psd_1_times": psd_times,
            "avg_psd_2": list_psd_2,
            "psd_2_times": psd_2_times,
        }

        with open(filename, "w") as outfile:
            json.dump(save_dictionary, outfile)

def get_baselines(interferometers, frequencies=None):
    """
    Parameters
    ==========
    interferometers: list of bilby interferometer objects
    """
    Nd = len(interferometers)

    combo_tuples = []
    for j in range(1, Nd):
        for k in range(j):
            combo_tuples.append((k, j))

    baselines = []
    for i, j in combo_tuples:
        base_name = f"{interferometers[i].name} - {interferometers[j].name}"
        baselines.append(
            Baseline(
                base_name,
                interferometers[i],
                interferometers[j],
                frequencies=frequencies,
            )
        )
    return baselines
