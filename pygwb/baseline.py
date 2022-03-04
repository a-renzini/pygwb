import json
import pickle
import warnings

import gwpy.frequencyseries
import gwpy.spectrogram
import numpy as np
from bilby.core.utils import create_frequency_series
from loguru import logger

from .delta_sigma_cut import run_dsc
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
        notch_list_path=None,
        overlap_factor=0.5,
        zeropad_csd=True,
        window_fftgram="hann",
        overlap_factor_welch_psd=0,
        N_average_segments_welch_psd=2,
        sampling_frequency=None,
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
        notch_list_path: str, optional
            file path of the baseline notch list
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
        self.notch_list_path = notch_list_path
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

    def set_frequency_mask(self, notch_list_path):
        mask = (self.frequencies >= self.minimum_frequency) & (
            self.frequencies <= self.maximum_frequency
        )
        if notch_list_path is not None:
            notch_list = StochNotchList.load_from_file(notch_list_path)
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
            raise ValueError("Duration not yet set")

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
            raise ValueError("frequencies have not yet been set")

    @frequencies.setter
    def frequencies(self, freqs):
        self._frequencies = freqs
        self._frequencies_set = True
        # delete the orfs, set the calculated flag to zero
        if self._tensor_orf_calculated:
            delattr(self, "_tensor_orf")
            self._tensor_orf_calculated = False
        if self._scalar_orf_calculated:
            delattr(self, "_scalar_orf")
            self._scalar_orf_calculated = False
        if self._vector_orf_calculated:
            delattr(self, "_vector_orf")
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
        if hasattr(self, "_sampling_frequency"):
            return self._sampling_frequency
        else:
            raise ValueError("sampling frequency not set")

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
            warnings.warn(
                "Neither baseline nor interferometer sampling_frequency is set."
            )
            self._sampling_frequency = sampling_frequency
            self._sampling_frequency_set = True

    @property
    def badGPStimes(self):
        if hasattr(self, "_badGPStimes"):
            return self._badGPStimes
        else:
            raise ValueError(
                "bad GPS times are not set - need to run delta_sigma_cut first."
            )

    @badGPStimes.setter
    def badGPStimes(self, badGPStimes):
        self._badGPStimes = badGPStimes

    @property
    def delta_sigmas(self):
        if hasattr(self, "delta_sigmas"):
            return self._delta_sigmas
        else:
            raise ValueError(
                "delta_sigmas are not set - need to run delta_sigma_cut first."
            )

    @delta_sigmas.setter
    def delta_sigmas(self, delta_sigmas):
        self._delta_sigmas = delta_sigmas

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
    ):
        name = interferometer_1.name + interferometer_2.name
        return cls(
            name=name,
            interferometer_1=interferometer_1,
            interferometer_2=interferometer_2,
            duration=parameters.segment_duration,
            calibration_epsilon=parameters.calibration_epsilon,
            frequencies=frequencies,
            notch_list_path=parameters.notch_list_path,
            overlap_factor=parameters.overlap_factor,
            zeropad_csd=parameters.zeropad_csd,
            window_fftgram=parameters.window_fftgram,
            overlap_factor_welch_psd=parameters.overlap_factor_welch_psd,
            N_average_segments_welch_psd=parameters.N_average_segments_welch_psd,
            sampling_frequency=parameters.new_sample_rate,
        )

    @classmethod
    def load_from_pickle(cls, filename):
        """Loads entire baseline object from pickle file"""
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save_to_pickle(self, filename):
        """Saves entire baseline object to pickle file"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

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
                csd_segment_offset : -(csd_segment_offset + 1) + 1
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
            flow: float
                low frequency
            fhigh: float
                high frequency
        """
        deltaF = self.frequencies[1] - self.frequencies[0]
        indexes = (self.frequencies >= flow) * (self.frequencies <= fhigh)
        # reset frequencies
        self.frequencies = self.frequencies[indexes]

        if hasattr(self.interferometer_1, "average_psd"):
            self.interferometer_1.average_psd = (
                self.interferometer_1.average_psd.crop_frequencies(flow, fhigh + deltaF)
            )
        if hasattr(self.interferometer_2, "average_psd"):
            self.interferometer_2.average_psd = (
                self.interferometer_2.average_psd.crop_frequencies(flow, fhigh + deltaF)
            )
        if hasattr(self, "average_csd"):
            self.average_csd = self.average_csd.crop_frequencies(flow, fhigh + deltaF)

    def set_point_estimate_sigma_spectrogram(
        self, weight_spectrogram=False, alpha=0, fref=1, flow=20, fhigh=1726
    ):
        """Set point estimate and sigma spectrogram. Resulting spectrogram
        *does not include frequency weighting for alpha*.
        """
        # set CSD if not set
        # self.set_average_cross_spectral_density()

        # set PSDs if not set
        # self.set_average_power_spectral_densities()

        self.crop_frequencies_average_psd_csd(flow, fhigh)

        # don't get rid of information unless we need to.
        Y_fs, var_fs = calculate_point_estimate_sigma_spectrogram(
            self.frequencies,
            self.average_csd,
            self.interferometer_1.average_psd,
            self.interferometer_2.average_psd,
            self.overlap_reduction_function,
            self.sampling_frequency,
            self.duration,
            weight_spectrogram=weight_spectrogram,
            fref=fref,
            alpha=alpha,
        )

        if weight_spectrogram:
            self.spectrogram_alpha_weight = alpha
        else:
            self.spectrogram_alpha_weight = 0

        sigma_name = (
            self.name + f" sigma spectrogram alpha={self.spectrogram_alpha_weight}"
        )
        self.point_estimate_spectrogram = gwpy.spectrogram.Spectrogram(
            Y_fs,
            times=self.average_csd.times,
            frequencies=self.average_csd.frequencies,
            name=self.name + f" with alpha={self.spectrogram_alpha_weight}",
        )
        self.sigma_spectrogram = gwpy.spectrogram.Spectrogram(
            np.sqrt(var_fs),
            times=self.average_csd.times,
            frequencies=self.average_csd.frequencies,
            name=sigma_name,
        )

    def set_point_estimate_sigma_spectrum(
        self,
        badtimes=None,
        weight_spectrogram=False,
        alpha=0,
        fref=25,
        flow=20,
        fhigh=1726,
        notch_list_path=None,
    ):
        """Sets time-integrated point estimate spectrum and variance in each frequency bin.
        Point estimate is *unweighted* by alpha.

        Parameters
        ==========
        badtimes: np.array, optional
            array of times to exclude from point estimate/sigma calculation. If no times are passed, none will be excluded.
        weight_spectrogram: bool, optional
            weight spectrogram flag; if True, the spectrogram will be re-weighted using the alpha passed here.Default is False.
        alpha: float, optional
            spectral index to use in the re-weighting. Default is 0.
        fref: float, optional
            reference frequency to use in the re-weighting. Default is 25.
        flow: float, optional
            low frequency. Default is 20 Hz.
        fhigh: float, optional
            high frequency. Default is 1726 Hz.
        notch_list_path: str, optional
            path to the notch list to use in the spectrum; if the notch_list isn't set in the baseline, user can pass it directly here. If it is not set and if none is passed no notches will be applied.
        """

        # set unweighted point estimate and sigma spectrograms
        if not hasattr(self, "point_estimate_spectrogram"):
            logger.info(
                "Point estimate and sigma spectrograms are not set yet. setting now..."
            )
            self.set_point_estimate_sigma_spectrogram(
                weight_spectrogram=weight_spectrogram,
                alpha=alpha,
                fref=fref,
                flow=flow,
                fhigh=fhigh,
            )
        deltaF = self.frequencies[1] - self.frequencies[0]

        if self.notch_list_path is not None:
            lines_object = StochNotchList.load_from_file(self.notch_list_path)
            notches, _ = lines_object.get_idxs(self.frequencies)
        elif notch_list_path is not None:
            lines_object = StochNotchList.load_from_file(notch_list_path)
            notches, _ = lines_object.get_idxs(self.frequencies)
        else:
            notches = np.array([], dtype=int)

        if badtimes is None:
            if hasattr(self, "badGPStimes"):
                badtimes = self.badGPStimes
            else:
                badtimes = np.array([])

        # should be True for each bad time
        bad_times_indexes = np.array(
            [np.any(t == badtimes) for t in self.point_estimate_spectrogram.times.value]
        )

        logger.info(f"{np.sum(bad_times_indexes)} bad segments removed.")

        # start time, for metadata
        epoch = self.point_estimate_spectrogram.times[0]

        self.point_estimate_spectrogram[bad_times_indexes, :] = 0
        self.sigma_spectrogram[bad_times_indexes, :] = np.inf

        # Post process. Last argument is frequency notches. Do not include these yet.
        # Leave that for when we combine over freqs.
        if self.sampling_frequency is None:
            raise ValueError(
                "the sampling frequency is not set! Cannot proceed with spectrum calculation."
            )
        point_estimate, sigma = postprocess_Y_sigma(
            self.point_estimate_spectrogram.value,
            self.sigma_spectrogram.value ** 2,
            self.duration,
            deltaF,
            self.sampling_frequency,
            notches,
        )

        # REWEIGHT FUNCTION, self.spectrogram_alpha_weight is old weight, supplied alpha is new weight.

        self.point_estimate_spectrum = gwpy.frequencyseries.FrequencySeries(
            point_estimate,
            frequencies=self.frequencies,
            name=self.name + "unweighted point estimate spectrum",
            epoch=epoch,
        )
        self.sigma_spectrum = gwpy.frequencyseries.FrequencySeries(
            np.sqrt(sigma),
            frequencies=self.frequencies,
            name=self.name + "unweighted sigma spectrum",
            epoch=epoch,
        )
        self.point_estimate_alpha = 0

    def set_point_estimate_sigma(
        self,
        badtimes=None,
        apply_weighting=True,
        alpha=0,
        fref=1,
        flow=20,
        fhigh=1726,
        notch_list_path=None,
    ):
        """Set point estimate sigma based on a set of parameters. This is estimate of omega_gw in each frequency bin.

        Parameters
        ==========
        badtimes: np.array, optional
            array of times to exclude from point estimate/sigma calculation. If no times are passed, none will be excluded.
        apply_weighting: bool, optional
            apply weighting flag; if True, the point estimate and sigma will be weighted using the alpha passed here. Default is True.
        alpha: float, optional
            spectral index to use in the re-weighting. Default is 0.
        fref: float, optional
            reference frequency to use in the re-weighting. Default is 25.
        flow: float, optional
            low frequency. Default is 20 Hz.
        fhigh: float, optional
            high frequency. Default is 1726 Hz.
        notch_list_path: str, optional
            path to the notch list to use in the spectrum; if the notch_list isn't set in the baseline, user can pass it directly here. If it is not set and if none is passed no notches will be applied.
        """
        # TODO: Add check if badtimes is passed and point estimate spectrum
        # already exists...
        if not hasattr(self, "point_estimate_spectrum"):
            logger.info(
                "Point estimate and sigma spectra have not been set before. Setting it now..."
            )
            logger.debug(
                "No weighting supplied in setting of spectrum. Supplied when combining for final sigma"
            )
            self.set_point_estimate_sigma_spectrum(
                badtimes=badtimes,
                notch_list_path=notch_list_path,
                weight_spectrogram=False,
                alpha=alpha,
                fref=fref,
                flow=flow,
                fhigh=fhigh,
            )

        # crop frequencies according to params before combining over them
        deltaF = self.frequencies[1] - self.frequencies[0]
        Y_spec = self.point_estimate_spectrum.crop(flow, fhigh + deltaF)
        sigma_spec = self.sigma_spectrum.crop(flow, fhigh + deltaF)
        freq_band_cut = (self.frequencies >= flow) & (self.frequencies <= fhigh)
        self.frequencies = self.frequencies[freq_band_cut]

        # check notch list
        # TODO: make this less fragile...at the moment these indexes
        # must agree with those after cropping, so the notches must agree with the params
        # struct in some way. Seems dangerous
        if notch_list_path is not None:
            print("hello!")
            exit()
            lines_object = StochNotchList.load_from_file(notch_list_path)
            _, notch_indexes = lines_object.get_idxs(Y_spec.frequencies.value)
        else:
            notch_indexes = np.arange(Y_spec.size)

        # get Y, sigma
        if apply_weighting:
            Y, sigma = calc_Y_sigma_from_Yf_varf(
                Y_spec.value[notch_indexes],
                sigma_spec.value[notch_indexes] ** 2,
                freqs=self.frequencies[notch_indexes],
                alpha=alpha,
                fref=fref,
            )
        else:
            logger.info(
                "Be careful, in general weighting is not applied until this point"
            )
            Y, sigma = calc_Y_sigma_from_Yf_varf(
                self.point_estimate_spectrum.value, self.sigma_spectrogram.value ** 2
            )

        self.point_estimate = Y
        self.sigma = sigma

    def calculate_delta_sigma_cut(
        self,
        delta_sigma_cut,
        alphas,
        flow=20,
        fhigh=1726,
    ):
        """Calculates the delta sigma cut using the naive and average psds, if set in the baseline.

        Parameters
        ==========
        delta_sigma_cut: float
            the cutoff to implement in the delta sigma cut.
        alphas: list
            set of spectral indices to use in the delta sigma cut calculation.
        flow: float, optional
            low frequency. Default is 20 Hz.
        fhigh: float, optional
            high frequency. Default is 1726 Hz.
        """

        deltaF = self.frequencies[1] - self.frequencies[0]
        self.crop_frequencies_average_psd_csd(flow=flow, fhigh=fhigh)
        naive_psd_1_cropped = self.interferometer_1.psd_spectrogram.crop_frequencies(
            flow, fhigh + deltaF
        )
        naive_psd_2_cropped = self.interferometer_2.psd_spectrogram.crop_frequencies(
            flow, fhigh + deltaF
        )

        badGPStimes, delta_sigmas = run_dsc(
            delta_sigma_cut,
            self.duration,
            self.sampling_frequency,
            naive_psd_1_cropped,
            naive_psd_2_cropped,
            self.interferometer_1.average_psd,
            self.interferometer_2.average_psd,
            alphas,
            self.notch_list_path,
        )
        self.badGPStimes = badGPStimes
        self.delta_sigmas = delta_sigmas

    def save_data(
        self,
        save_data_type,
        filename,
    ):
        """Saves the overall point estimate Y, its error bar sigma,
        the frequency-dependent estimates and variances and the corresponding frequencies
        in the required save_data_type, which can be npz, pickle, json or hdf5.
        You can call upon this data afterwards when loaoding in using the ['key'] dictionary format.

        Parameters
        ==========
        save_data_type: str
            The required type of data file where the information will be stored
        filename: str
            the path/name of the file in which you want to save

        """

        if save_data_type == "pickle":
            save = self.pickle_save
            save_csd = self.pickle_save_csd
            ext = ".p"

        elif save_data_type == "npz":
            save = self.npz_save
            save_csd = self.npz_save_csd
            ext = ".npz"

        elif save_data_type == "json":
            save = self.json_save
            save_csd = self.json_save_csd
            ext = ".json"

        elif save_data_type == "hdf5":
            save = self.hdf5_save
            save_csd = self.hdf5_save_csd
            ext = ".h5"

        else:
            raise ValueError(
                "The provided data type is not supported, try using 'pickle', 'npz', 'json' or 'hdf5' instead."
            )

        save(
            f"{filename}{ext}",
            self.frequencies,
            self.point_estimate_spectrum,
            self.sigma_spectrum,
            self.point_estimate,
            self.sigma,
            self.point_estimate_spectrogram,
            self.sigma_spectrogram,
        )
        save_csd(
            f"psds_csds_{filename}{ext}",
            self.frequencies,
            self.average_csd,
            self.interferometer_1.average_psd,
            self.interferometer_2.average_psd,
        )

    def npz_save(
        self,
        filename,
        frequencies,
        point_estimate_spectrum,
        sigma_spectrum,
        point_estimate,
        sigma,
        point_estimate_spectrogram,
        sigma_spectrogram,
    ):
        np.savez(
            filename,
            frequencies=frequencies,
            point_estimate_spectrum=point_estimate_spectrum,
            sigma_spectrum=sigma_spectrum,
            point_estimate=point_estimate,
            sigma=sigma,
            point_estimate_spectrogram=point_estimate_spectrogram,
            sigma_spectrogram=sigma_spectrogram,
        )

    def pickle_save(
        self,
        filename,
        frequencies,
        point_estimate_spectrum,
        sigma_spectrum,
        point_estimate,
        sigma,
        point_estimate_spectrogram,
        sigma_spectrogram,
    ):
        save_dictionary = {
            "frequencies": frequencies,
            "point_estimate_spectrum": point_estimate_spectrum,
            "sigma_spectrum": sigma_spectrum,
            "point_estimate": point_estimate,
            "sigma": sigma,
            "point_estimate_spectrogram": point_estimate_spectrogram,
            "sigma_spectrogram": sigma_spectrogram,
        }

        with open(filename, "wb") as f:
            pickle.dump(save_dictionary, f)

    def json_save(
        self,
        filename,
        frequencies,
        point_estimate_spectrum,
        sigma_spectrum,
        point_estimate,
        sigma,
        point_estimate_spectrogram,
        sigma_spectrogram,
    ):
        list_freqs = frequencies.tolist()
        list_point_estimate_spectrum = point_estimate_spectrum.tolist()
        list_sigma_spectrum = sigma_spectrum.tolist()

        list_point_estimate_segment = point_estimate_spectrogram.value.tolist()
        point_estimate_segment_times = point_estimate_spectrogram.times.value.tolist()

        list_sigma_segment = sigma_spectrogram.value.tolist()
        sigma_segment_times = sigma_spectrogram.times.value.tolist()

        save_dictionary = {
            "frequencies": list_freqs,
            "point_estimate_spectrum": list_point_estimate_spectrum,
            "sigma_spectrum": list_sigma_spectrum,
            "point_estimate": point_estimate,
            "sigma": sigma,
            "point_estimate_spectrogram": list_point_estimate_segment,
            "point_estimate_spectrogram_times": point_estimate_segment_times,
            "sigma_spectrogram": list_sigma_segment,
            "sigma_spectrogram_times": sigma_segment_times,
        }

        with open(filename, "w") as outfile:
            json.dump(save_dictionary, outfile)

    def hdf5_save(
        self,
        filename,
        frequencies,
        point_estimate_spectrum,
        sigma_spectrum,
        point_estimate,
        sigma,
        point_estimate_spectrogram,
        sigma_spectrogram,
    ):
        hf = h5py.File(filename, "w")

        hf.create_dataset("freqs", data=frequencies)
        hf.create_dataset("point_estimate_spectrum", data=point_estimate_spectrum)
        hf.create_dataset("sigma_spectrum", data=sigma_spectrum)
        hf.create_dataset("point_estimate", data=point_estimate)
        hf.create_dataset("sigma", data=sigma)
        hf.create_dataset(
            "point_estimate_spectrogram", data=point_estimate_spectrogram
        ),
        hf.create_dataset("sigma_spectrogram", data=sigma_spectrogram)

        hf.close()

    def npz_save_csd(self, filename, freqs, csd, avg_psd_1, avg_psd_2):
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

    def hdf5_save_csd(self, filename, freqs, csd, psd_1, psd_2):
        hf = h5py.File(filename, "w")

        csd_times = csd.times.value
        psd_1_times = psd_1.times.value
        psd_2_times = psd_2.times.value

        hf.create_dataset("freqs", data=freqs)

        csd_group = hf.create_group("csd_group")

        csd_group.create_dataset("csd", data=csd)
        csd_group.create_dataset("csd_times", data=csd_times)

        psd_group = hf.create_group("psds_group")

        psd_1_group = hf.create_group("psds_group/psd_1")
        psd_1_group.create_dataset("psd_1", data=psd_1)
        psd_1_group.create_dataset("psd_1_times", data=psd_1_times)

        psd_2_group = hf.create_group("psds_group/psd_2")
        psd_2_group.create_dataset("psd_2", data=psd_2)
        psd_2_group.create_dataset("psd_2_times", data=psd_2_times)

        hf.close()


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
