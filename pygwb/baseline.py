"""
The ``Baseline`` module is by design the core of the pygwb stochastic analysis. Its main role is to manage the cross-
correlation between ``Interferometer`` data products, combine these into a single cross-spectrum, which represents the
point estimate of the analysis, and calculate the associated error.

The ``Baseline`` object relies on the ``pygwb.spectral`` module to calculate cross-correlations between the data
streams. Similarly, it relies on the ``pygwb.postprocessing`` module to obtain the point estimate and its variance.
Calculating these, as well as performing parameter estimation on the gravitational-wave background (GWB) spectrum, requires the two-detector
overlap reduction function (ORF). The ORF is calculated using the ``pygwb.orfs`` module at ``Baseline`` object 
initialization, then stored as an attribute.

Examples
--------

To show how a ``Baseline`` object can be instantiated, we start by importing the relevant
packages:

>>> import numpy as np
>>> from pygwb.detector import Interferometer
>>> from pygwb.baseline import Baseline

For concreteness, we work with the LIGO Hanford and Livingston detectors, which we 
instantiate through:

>>> H1 = Interferometer.get_empty_interferometer("H1")
>>> L1 = Interferometer.get_empty_interferometer("L1")

The standard initialization of a ``Baseline`` object then simply requires a pair of 
``Interferometer`` objects:

>>> H1L1_baseline = baseline.Baseline("H1-L1", H1, L1)
"""

import json
import pickle
import warnings

import h5py
import numpy as np
from gwpy.frequencyseries import FrequencySeries
from loguru import logger

from .coherence import calculate_coherence
from .constants import h0
from .delta_sigma_cut import run_dsc
from .notch import StochNotchList
from .omega_spectra import OmegaSpectrogram, OmegaSpectrum
from .orfs import calc_orf
from .postprocessing import (
    calc_Y_sigma_from_Yf_sigmaf,
    calculate_point_estimate_sigma_spectra,
    postprocess_Y_sigma,
)
from .spectral import cross_spectral_density


class Baseline:
    def __init__(
        self,
        name,
        interferometer_1,
        interferometer_2,
        duration=None,
        frequencies=None,
        calibration_epsilon=0,
        notch_list_path="",
        coarse_grain_psd=False,
        coarse_grain_csd=True,
        overlap_factor_welch=0.5,
        overlap_factor=0.5,
        window_fftgram_dict={"window_fftgram": "hann"},
        window_fftgram_dict_welch={"window_fftgram": "hann"},
        N_average_segments_psd=2,
        sampling_frequency=None,
    ):
        """
        Instantiate a Baseline.

        Parameters
        =======

        name: ``str``
            Name for the baseline, e.g H1H2
        interferometer_1/2: ``bilby.Interferometer`` object
            The two detectors spanning the baseline.
        duration: ``float``, optional
            The duration in seconds of each data segment in the interferometers.
            None by default, in which case duration is inherited from the interferometers.
        frequencies: ``array_like``, optional
            The frequency array for the Baseline and
            interferometers.
        calibration_epsilon: ``float``, optional
            Calibration uncertainty for this baseline.
        notch_list_path: ``str``, optional
            File path of the baseline notch list.
        coarse_grain_psd: ``bool``, optional
            Whether to apply coarse graining to obtain PSD spectra. Default is False.
        coarse_grain_csd: ``bool``, optional
            Whether to apply coarse graining to obtain CSD spectra. Default is True.
        overlap_factor_welch: ``float``, optional
            Overlap factor to use when if using Welch's method to estimate spectra (NOT coarsegraining). 
            For \"hann\" window use 0.5 overlap_factor and for \"boxcar"\ window use 0 overlap_factor. Default is 0.5 (50% overlap), 
            which is optimal when using Welch's method with a \"hann\" window.
        overlap_factor: ``float``, optional
            Factor by which to overlap the segments in the psd and csd estimation.
            Default is 1/2, if set to 0 no overlap is performed.
        window_fftgram_dict: ``dict``, optional
            Dictionary containing name and parameters describing which window to use when producing fftgrams for csds (and psds if these are coarse-grained). Default is \"hann\".
        window_fftgram_dict_welch: ``dict``, optional
            Dictionary containing name and parameters describing which window to use when producing fftgrams with Welch's method. Default is \"hann\".
        N_average_segments_psd: ``int``, optional
            Number of segments used for PSD averaging (from both sides of the segment of interest)
            N_avg_segs should be even and >= 2.
        """
        self.name = name
        self.interferometer_1 = interferometer_1
        self.interferometer_2 = interferometer_2
        self.calibration_epsilon = calibration_epsilon
        self.notch_list_path = notch_list_path
        self.coarse_grain_psd = coarse_grain_psd
        self.coarse_grain_csd = coarse_grain_csd
        self.overlap_factor_welch = overlap_factor_welch
        self.overlap_factor = overlap_factor
        self.window_fftgram_dict = window_fftgram_dict
        self.window_fftgram_dict_welch = window_fftgram_dict_welch
        self.N_average_segments_psd = N_average_segments_psd
        self._tensor_orf_calculated = False
        self._vector_orf_calculated = False
        self._scalar_orf_calculated = False
        self._gamma_v_calculated = False
        self._orf_polarization_set = False
        self._point_estimate_spectrogram_set = False
        self._point_estimate_spectrum_set = False
        self._point_estimate_set = False
        self._sigma_spectrogram_set = False
        self._sigma_spectrum_set = False
        self._sigma_set = False
        self._coherence_spectrum_set = False
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.frequencies = frequencies
        self.minimum_frequency = max(
            interferometer_1.minimum_frequency, interferometer_2.minimum_frequency
        )
        self.maximum_frequency = min(
            interferometer_1.maximum_frequency, interferometer_2.maximum_frequency
        )
        # if CSD is estimated by coarse-graining, it must be zeropaded. 
        self.zeropad_csd = self.coarse_grain_csd
        # if PSD is estimated by coarse-graining, no overlap is used between PSD estimates. This is required for the bias factor calculation.
        if self.coarse_grain_psd:
            self.overlap_factor_psd = 0.0
            self.window_fftgram_dict_psd = self.window_fftgram_dict
            self.window_fftgram_dict_for_bias_factors = {"window_fftgram": "boxcar"}
        else:
            self.overlap_factor_psd = self.overlap_factor_welch
            self.window_fftgram_dict_psd = self.window_fftgram_dict_welch
            self.window_fftgram_dict_for_bias_factors = self.window_fftgram_dict_psd
        if self.coarse_grain_csd:
            self.window_fftgram_dict_csd = self.window_fftgram_dict
        else:
            self.window_fftgram_dict_csd = self.window_fftgram_dict_welch
        # throw a warning if overlap factors are unsupported
        if self.overlap_factor>0.5:
            warnings.warn("Overlap factor not fully supported. Overlap factor should be overlap_factor <= 0.5.")
        if self.overlap_factor_welch>0.5:
            warnings.warn("Overlap factor for spectral estimation using Welch's method not fully supported. Overlap factor should be overlap_factor_welch <= 0.5.")

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
    def overlap_reduction_function(self):
        """Overlap reduction function associated to this baseline, calculated for the requested polarization."""
        if self._orf_polarization == "tensor":
            return self.tensor_overlap_reduction_function
        elif self._orf_polarization == "vector":
            return self.vector_overlap_reduction_function
        elif self._orf_polarization == "scalar":
            return self.scalar_overlap_reduction_function
        elif self._orf_polarization == "right_left":
            return self.gamma_v
        else:
            raise AttributeError(
                "Overlap reduction function to be used has not yet been set. To set it, set the orf_polarization property."
            )

    @property
    def orf_polarization(self):
        """Overlap reduction function polarization"""
        if self._orf_polarization_set:
            return self._orf_polarization
        else:
            raise AttributeError(
                "Overlap reduction function polarization han not yet been set."
            )

    @orf_polarization.setter
    def orf_polarization(self, pol):
        self._orf_polarization = pol
        self._orf_polarization_set = True

    @property
    def tensor_overlap_reduction_function(self):
        """Overlap reduction function calculated for tensor polarization."""
        if not self._tensor_orf_calculated:
            self._tensor_orf = self.calc_baseline_orf(
                polarization="tensor", frequencies=self.frequencies
            )
            self._tensor_orf_calculated = True
        return self._tensor_orf

    @property
    def vector_overlap_reduction_function(self):
        """Overlap reduction function calculated for vector polarization."""
        if not self._vector_orf_calculated:
            self._vector_orf = self.calc_baseline_orf(polarization="vector")
            self._vector_orf_calculated = True
        return self._vector_orf

    @property
    def scalar_overlap_reduction_function(self):
        """Overlap reduction function calculated for scalar polarization."""
        if not self._scalar_orf_calculated:
            self._scalar_orf = self.calc_baseline_orf(polarization="scalar")
            self._scalar_orf_calculated = True
        return self._scalar_orf

    def set_frequency_mask(self, notch_list_path="", apply_notches=True):
        """
        Set frequency mask to frequencies attribute.

        Parameters
        =======

        notch_list_path: ``str``, optional
            Path to notch list to apply to frequency array. If not 
            provided, no notching will be applied at this point.

        apply_notches: ``bool``, optional
            If True, apply frequency notches. Default is True.

        See also
        --------
        pygwb.notch.StochNotchList : Used to read in the frequency notches.
        """
        mask = (self.frequencies >= self.minimum_frequency) & (
            self.frequencies <= self.maximum_frequency
        )
        if apply_notches:
            if notch_list_path:
                self.notch_list_path = notch_list_path
            if self.notch_list_path:
                logger.debug("loading notches from " + str(self.notch_list_path))
                notch_list = StochNotchList.load_from_file(self.notch_list_path)
                notch_mask = notch_list.get_notch_mask(self.frequencies)
                mask = np.logical_and(mask, notch_mask)
            else:
                logger.debug("no notching will be applied at this point.")

        self.frequency_mask = mask

    @property
    def gamma_v(self, frequencies=None):
        """
        Overlap reduction function for asymmetrically polarised backgrounds,
        as described in https://arxiv.org/pdf/0707.0535.pdf
        """
        if not self._gamma_v_calculated:
            self._gamma_v = self.calc_baseline_orf(polarization="right_left")
            self._gamma_v_calculated = True
        return self._gamma_v

    @property
    def duration(self):
        """Duration in seconds of a unit segment of data stored in the baseline detectors."""
        if self._duration_set:
            return self._duration
        else:
            raise AttributeError("Duration not yet set.")

    @duration.setter
    def duration(self, dur):
        """
        Sets the duration for the Baseline and interferometers.

        Parameters
        =======
        dur: ``float``
            The duration to set for the Baseline and interferometers.

        Notes
        -----

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
        """
        if dur is not None:
            self._check_durations_match_baseline_ifos(dur)
            self._duration = dur
            if not self.interferometer_1.duration:
                self.interferometer_1.duration = dur
            if not self.interferometer_2.duration:
                self.interferometer_2.duration = dur
            self._duration_set = True
        elif self.interferometer_1.duration and self.interferometer_2.duration:
            self._check_ifo_durations_match()
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
    def csd_segment_offset(self):
        """CSD segment offset to use for this baseline."""
        if self._duration_set:
            stride = self.duration * (1 - self.overlap_factor)
            return int(np.ceil(self.duration / stride)) * int(self.N_average_segments_psd/2)
        else:
            raise ValueError("Trying to calculate CSD segment offset before setting duration. Need to set duration before attempting this.")

    @property
    def frequencies(self):
        """Frequency array associated to this baseline."""
        if self._frequencies_set:
            return self._frequencies
        else:
            raise AttributeError("frequencies have not yet been set.")

    @frequencies.setter
    def frequencies(self, freqs):
        self._frequencies = freqs
        self._frequencies_set = True
        self.frequency_mask = None
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

    @property
    def point_estimate_spectrogram(self):
        """Point estimate spectrogram (in Omega units) calculated using data in this baseline."""
        if self._point_estimate_spectrogram_set:
            return self._point_estimate_spectrogram
        else:
            raise AttributeError(
                "Omega point estimate spectrogram not yet set. To set it, use `set_point_estimate_sigma_spectrogram` method."
            )

    @point_estimate_spectrogram.setter
    def point_estimate_spectrogram(self, pt_est):
        self._point_estimate_spectrogram = pt_est
        self._point_estimate_spectrogram_set = True

    @property
    def sigma_spectrogram(self):
        """Sigma spectrogram (in Omega units) calculated using data in this baseline."""
        if self._sigma_spectrogram_set:
            return self._sigma_spectrogram
        else:
            raise AttributeError(
                "Omega sigma spectrogram not yet set. To set it, use `set_point_estimate_sigma_spectrogram` method."
            )

    @sigma_spectrogram.setter
    def sigma_spectrogram(self, sig):
        self._sigma_spectrogram = sig
        self._sigma_spectrogram_set = True

    @property
    def point_estimate_spectrum(self):
        """Point estimate spectrum (in Omega units) calculated using data in this baseline."""
        if self._point_estimate_spectrum_set:
            return self._point_estimate_spectrum
        else:
            raise AttributeError(
                "Omega point estimate spectrum not yet set. To set it, use `set_point_estimate_sigma_spectrum` method."
            )

    @point_estimate_spectrum.setter
    def point_estimate_spectrum(self, pt_est):
        self._point_estimate_spectrum = pt_est
        self._point_estimate_spectrum_set = True

    @property
    def sigma_spectrum(self):
        """Sigma spectrum (in Omega units) calculated using data in this baseline."""
        if self._sigma_spectrum_set:
            return self._sigma_spectrum
        else:
            raise AttributeError(
                "Omega sigma spectrum not yet set. To set it, use `set_point_estimate_sigma_spectrum` method."
            )

    @sigma_spectrum.setter
    def sigma_spectrum(self, sig):
        self._sigma_spectrum = sig
        self._sigma_spectrum_set = True

    @property
    def point_estimate(self):
        """Point estimate (in Omega units) calculated using data in this baseline."""
        if self._point_estimate_set:
            return self._point_estimate
        else:
            raise AttributeError(
                "Omega point estimate not yet set. To set it, use `set_point_estimate_sigma` method."
            )

    @point_estimate.setter
    def point_estimate(self, pt_est):
        self._point_estimate = pt_est
        self._point_estimate_set = True

    @property
    def sigma(self):
        """Sigma (in Omega units) calculated using data in this baseline."""
        if self._sigma_set:
            return self._sigma
        else:
            raise AttributeError(
                "Omega sigma not yet set. To set it, use `set_point_estimate_sigma` method."
            )

    @sigma.setter
    def sigma(self, sig):
        self._sigma = sig
        self._sigma_set = True

    @property
    def coherence_spectrum(self):
        """Coherence spectrum calculated using data in this baseline."""
        if self._coherence_spectrum_set:
            return self._coherence_spectrum
        else:
            raise AttributeError(
                "Coherence spectrum not yet set. To set it, use `set_coherence_spectrum` method."
            )

    @coherence_spectrum.setter
    def coherence_spectrum(self, coh):
        self._coherence_spectrum = coh
        self._coherence_spectrum_set = True

    @property
    def coherence_dict(self):
        """Coherence dictionary based on data in this baseline."""
        if self._coherence_spectrum_set:
            return self._coherence_dict
        else:
            raise AttributeError(
                "Coherence spectrum not yet set. To set it, use `set_coherence_spectrum` method."
            )

    @coherence_dict.setter
    def coherence_dict(self, cohdict):
        self._coherence_dict = cohdict

    def _check_durations_match_baseline_ifos(self, duration):
        """
        Checks whether the baseline duration matches the duration set in 
        the interferometers of the baseline.

        Parameters
        =======

        duration: ``float``
            Duration of the baseline.
        """
        if self.interferometer_1.duration and self.interferometer_2.duration:
            self._check_ifo_durations_match()
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

    def _check_ifo_durations_match(self):
        """
        Checks whether the duration in both interferometers agree.
        """
        if not (self.interferometer_1.duration == self.interferometer_2.duration):
            raise AssertionError("Interferometer durations do not match each other!")

    @property
    def sampling_frequency(self):
        """Sampling frequency of the data stored in this baseline. This must match the
        sampling frequency stored in this baseline's interferometers."""
        if hasattr(self, "_sampling_frequency"):
            return self._sampling_frequency
        else:
            raise AttributeError("sampling frequency not set.")

    @sampling_frequency.setter
    def sampling_frequency(self, sampling_frequency):
        """Sets the sampling_frequency for the Baseline and interferometers

        Parameters
        =======

        sampling_frequency: ``float``, optional
            The sampling frequency to set for the Baseline and interferometers

        Warning
        -------

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
            self._check_ifo_sampling_frequencies_match()
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
        """GPS times flagged by delta sigma cut."""
        if hasattr(self, "_badGPStimes"):
            return self._badGPStimes
        else:
            raise AttributeError(
                "bad GPS times are not set - need to run delta_sigma_cut first."
            )

    @badGPStimes.setter
    def badGPStimes(self, badGPStimes):
        self._badGPStimes = badGPStimes

    @property
    def delta_sigmas(self):
        """Values of delta sigmas for data segments in the baseline."""
        if hasattr(self, "_delta_sigmas"):
            return self._delta_sigmas
        else:
            raise AttributeError(
                "delta_sigmas are not set - need to run delta_sigma_cut first."
            )

    @delta_sigmas.setter
    def delta_sigmas(self, delta_sigmas):
        self._delta_sigmas = delta_sigmas

    def check_sampling_frequencies_match_baseline_ifos(self, sampling_frequency):
        """Check that the sampling frequency of the two interferometers in this Baseline match the Baseline sampling frequency.

        Parameters
        =======

        sampling_frequency: ``float``
            The sampling frequency that is being set for the Baseline.

        Notes
        -----

        If the sampling frequency passed is `None`, the Baseline sampling frequency will be set to that of the interferometers, if these
        match. If these don't match, an error will be raised. If the sampling frequency of the interferometers is also `None`, then no
        sampling frequency will be set, and the user can set it at a later time.
        """
        if (
            self.interferometer_1.sampling_frequency
            and self.interferometer_2.sampling_frequency
        ):
            self._check_ifo_sampling_frequencies_match()
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

    def _check_ifo_sampling_frequencies_match(self):
        """
        Check whether the sampling frequencies of the interferometers match.
        """
        if not (
            self.interferometer_1.sampling_frequency
            == self.interferometer_2.sampling_frequency
        ):
            raise AssertionError(
                "Interferometer sampling_frequencies do not match each other!"
            )

    def calc_baseline_orf(self, polarization="tensor", frequencies=None):
        """
        Calculate the overlap reduction function for this baseline.
        Wraps the ORF module.

        Parameters
        =======

        polarization: ``str``, optional
            Polarization of the signal to consider (scalar, vector, tensor) for the ORF calculation.
            Default is tensor.

        frequencies: ``array_like``, optional
            Frequency array to use in the calculation of the ORF. By default, self.frequencies is used.

        Returns
        =======

        orf: ``array_like``
            Overlap reduction function for the required polarization.

        See also
        --------
        pygwb.orfs.calc_orf
            Method to compute the overlap reduction function.
        """
        if frequencies is not None:
            return calc_orf(
                frequencies,
                self.interferometer_1.vertex,
                self.interferometer_2.vertex,
                self.interferometer_1.x,
                self.interferometer_2.x,
                self.interferometer_1.y,
                self.interferometer_2.y,
                polarization,
            )
        elif self.frequencies is not None:
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
        else:
            raise ValueError(
                "Frequencies have not been provided for the orf calculation; user should either pass frequencies in or set them for this Baseline."
            )

    @classmethod
    def from_interferometers(
        cls,
        interferometers,
        duration=None,
        calibration_epsilon=0,
    ):
        """
        Load a Baseline from a list of interferometers.

        Parameters
        =======

        interferometers: ``list``
            List of two bilby Interferometer objects.

        duration: ``float``, optional
            Segment duration in seconds. Default is None.

        calibration_epsilon: ``float``, optional
            Calibration uncertainty for this baseline. Default is 0.

        Returns
        =======

        Baseline: cls
            Baseline class
        """
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
        """
        Load a Baseline from a Parameters object.

        Parameters
        =======

        interferometer_1/2: ``bilby.Interferometer`` object
            The two detectors spanning this baseline.

        parameters: ``pygwb.parameters`` object
            Parameters object containing necessary parameters for
            the instantiation of the baseline, and subsequent
            analyses.

        frequencies: ``array_like``, optional
            Frequency array to use in the instantiation of this baseline.
            Default is None.

        Returns
        =======

        Baseline: cls
            Baseline class
        """
        name = interferometer_1.name + interferometer_2.name
        return cls(
            name=name,
            interferometer_1=interferometer_1,
            interferometer_2=interferometer_2,
            duration=parameters.segment_duration,
            calibration_epsilon=parameters.calibration_epsilon,
            frequencies=frequencies,
            notch_list_path=parameters.notch_list_path,
            coarse_grain_psd=parameters.coarse_grain_psd,
            coarse_grain_csd=parameters.coarse_grain_csd,
            overlap_factor_welch=parameters.overlap_factor_welch,
            overlap_factor=parameters.overlap_factor,
            window_fftgram_dict=parameters.window_fft_dict,
            window_fftgram_dict_welch=parameters.window_fft_dict_welch,
            N_average_segments_psd=parameters.N_average_segments_psd,
            sampling_frequency=parameters.new_sample_rate,
        )

    @classmethod
    def load_from_pickle(cls, filename):
        """
        Load baseline object from pickle file.

        Parameters
        =======

        filename: ``str``
            Filename (inclusive of path) to load the pickled baseline from.

        Returns
        =======

        Baseline: ``cls``
            Baseline class.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save_to_pickle(self, filename, wipe=True):
        """
        Save baseline object to pickle file.

        Parameters
        =======

        filename: ``str``
            Filename (inclusive of path) to save the pickled baseline to.
        """
        if wipe:
            self.interferometer_1.timeseries = None
            self.interferometer_2.timeseries = None
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def set_cross_and_power_spectral_density(self, frequency_resolution):
        """
        Set the power spectral density in each interferometer
        and the cross spectral density for the baseline object when data are available.

        Parameters
        =======

        frequency_resolution: ``float``
            The frequency resolution at which the cross and power spectral densities are calculated.

        See also
        --------
        pygwb.spectral.cross_spectral_density
        """
        try:
            self.interferometer_1.set_psd_spectrogram(
                frequency_resolution,
                coarse_grain=self.coarse_grain_psd,
                overlap_factor=self.overlap_factor,
                window_fftgram_dict=self.window_fftgram_dict_psd,
                overlap_factor_welch=self.overlap_factor_welch,
            )
        except AttributeError:
            raise AssertionError(
                "Interferometer {self.interferometer_1.name} has no timeseries data! Need to set timeseries data in the interferometer first."
            )
        try:
            self.interferometer_2.set_psd_spectrogram(
                frequency_resolution,
                coarse_grain=self.coarse_grain_psd,
                overlap_factor=self.overlap_factor,
                window_fftgram_dict=self.window_fftgram_dict_psd,
                overlap_factor_welch=self.overlap_factor_welch,
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
            coarse_grain=self.coarse_grain_csd,
            overlap_factor=self.overlap_factor,
            zeropad=self.zeropad_csd,
            window_fftgram_dict=self.window_fftgram_dict_csd,
            overlap_factor_welch=self.overlap_factor_welch,
        )

        # TODO: make this less fragile.
        # For now, reset frequencies,
        # recalculate ORF in case frequencies have changed.
        self._tensor_orf_calculated = False
        self.frequencies = self.csd.frequencies.value

    def set_average_power_spectral_densities(self):
        """If psds have been calculated, sets the average psd in each ifo."""
        try:
            self.interferometer_1.set_average_psd(self.N_average_segments_psd)
            self.interferometer_2.set_average_psd(self.N_average_segments_psd)
        except AttributeError:
            print(
                "PSDs have not been calculated yet! Need to set_cross_and_power_spectral_density first."
            )

        # TODO: make this less fragile.
        # For now, recalculate ORF in case frequencies have changed.
        # self._tensor_orf_calculated = False
        # self.frequencies = self.csd.frequencies.value

    def set_average_cross_spectral_density(self):
        """If csd has been calculated, sets the average csd for the baseline."""
        try:
            self.average_csd = self.csd[
                self.csd_segment_offset : -(self.csd_segment_offset + 1) + 1
            ]
        except AttributeError:
            print(
                "CSD has not been calculated yet! Need to set_cross_and_power_spectral_density first."
            )
        # TODO: make this less fragile.
        # For now, recalculate ORF in case frequencies have changed.
        self._tensor_orf_calculated = False
        self.frequencies = self.csd.frequencies.value

    def crop_frequencies_average_psd_csd(self, flow, fhigh):
        """
        Crop frequencies of average PSDs and CSDS. Done in place. This is not completely implemented yet.

        Parameters
        =======

        flow: ``float``
            Low frequency to crop.

        fhigh: ``float``
            High frequency to crop.
        """
        deltaF = self.frequencies[1] - self.frequencies[0]
        # reset frequencies using the same calculation as in crop_frequencies so we get
        # consistent frequency ranges
        idx0 = int(float(flow - self.frequencies[0]) // deltaF)
        idx1 = int(float(fhigh + deltaF - self.frequencies[0]) // deltaF)
        self.frequencies = self.frequencies[idx0:idx1]

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
        if self._point_estimate_spectrogram_set:
            self.point_estimate_spectrogram = (
                self.point_estimate_spectrogram.crop_frequencies(flow, fhigh + deltaF)
            )
        if self._sigma_spectrogram_set:
            self.sigma_spectrogram = self.sigma_spectrogram.crop_frequencies(
                flow, fhigh + deltaF
            )
        if self._point_estimate_spectrum_set:
            self.point_estimate_spectrum = self.point_estimate_spectrum.crop(
                flow, fhigh + deltaF
            )
        if self._sigma_spectrum_set:
            self.sigma_spectrum = self.sigma_spectrum.crop(flow, fhigh + deltaF)
        if self._point_estimate_set:
            self.set_point_estimate_sigma()
        if self._coherence_spectrum_set:
            self.coherence_spectrum = self.coherence_spectrum.crop(flow, fhigh + deltaF)

    def set_point_estimate_sigma_spectrogram(
        self, alpha=0.0, fref=25, flow=20, fhigh=1726, polarization="tensor"
    ):
        """
        Set point estimate and sigma spectrogram. Resulting spectrogram
        *does not include frequency weighting for alpha*.

        Parameters
        =======

        alpha: ``float``, optional
            Spectral index to use in the weighting. Default is 0.

        fref: ``float``, optional
            Reference frequency to use in the weighting calculation.
            Final result refers to this frequency. Default is 25 Hz.

        flow: ``float``, optional
            Lowest frequency to consider. Default is 20 Hz.

        fhigh: ``float``, optional
            Highest frequency to consider. Default is 1726 Hz.

        polarization: ``str``, optional
            Polarization of the signal to consider (scalar, vector, tensor) for the ORF calculation.
            Default is tensor.

        See also
        --------
        pygwb.omega_spectra.OmegaSpectrogram

        pygwb.postprocessing.calculate_point_estimate_sigma_spectra
        """
        # set CSD if not set
        # self.set_average_cross_spectral_density()

        # set PSDs if not set
        # self.set_average_power_spectral_densities()

        self.crop_frequencies_average_psd_csd(flow, fhigh)

        if not self._orf_polarization_set:
            self.orf_polarization = polarization

        Y_fs, var_fs = calculate_point_estimate_sigma_spectra(
            freqs=self.frequencies,
            csd=self.average_csd,
            avg_psd_1=self.interferometer_1.average_psd,
            avg_psd_2=self.interferometer_2.average_psd,
            orf=self.overlap_reduction_function,
            sample_rate=self.sampling_frequency,
            segment_duration=self.duration,
            fref=fref,
            alpha=alpha,
            overlap_factor=self.overlap_factor,
            window_fftgram_dict=self.window_fftgram_dict_csd
        ) 

        sigma_name = f"{self.name} sigma spectrogram alpha={alpha}"
        self.point_estimate_spectrogram = OmegaSpectrogram(
            Y_fs,
            times=self.average_csd.times,
            frequencies=self.average_csd.frequencies,
            name=self.name + f" with alpha={alpha}",
            alpha=alpha,
            fref=fref,
            h0=h0,
        )

        self.sigma_spectrogram = OmegaSpectrogram(
            np.sqrt(var_fs),
            times=self.average_csd.times,
            frequencies=self.average_csd.frequencies,
            name=sigma_name,
            alpha=alpha,
            fref=fref,
            h0=h0,
        )

    def set_point_estimate_sigma_spectrum(
        self,
        badtimes=None,
        alpha=0.0,
        fref=25,
        flow=20,
        fhigh=1726,
        notch_list_path="",
        polarization="tensor",
        apply_dsc=True,
        apply_notches=True,
    ):
        """
        Set time-integrated point estimate spectrum and variance in each frequency bin.

        Parameters
        =======

        badtimes: ``array_like``, optional
            Array of times to exclude from point estimate/sigma calculation.
            If no times are passed, none will be excluded.

        alpha: ``float``, optional
            Spectral index to use in the re-weighting. Default is 0.

        fref: ``float``, optional
            Reference frequency to use in the re-weighting. Default is 25 Hz.

        flow: ``float``, optional
            Low frequency. Default is 20 Hz.

        fhigh: ``float``, optional
            High frequency. Default is 1726 Hz.

        notch_list_path: ``str``, optional
            Path to the notch list to use in the spectrum. Default is empty string.

        polarization: ``str``, optional
            Polarization of the signal to consider (scalar, vector, tensor) for the orf calculation.
            Default is tensor.

        apply_dsc: ``bool``, optional
            Apply delta sigma cut flag; if True, removes the badGPStimes from the spectra calculations.
            Default is True.

        apply_notches: ``bool``, optional
            Apply spectral notches flag; if True, remove the notches specified in the notch_list from the spectra calculations.
            Default is True.

        See also
        --------
        pygwb.postprocessing.postprocess_Y_sigma

        pygwb.omega_spectra.OmegaSpectrum
        """
        if apply_dsc:
            if not hasattr(self, "badGPStimes"):
                warnings.warn(
                    "Delta sigma cut has not been calculated yet, hence no delta sigma cut will be applied... If this is a mistake, please run `calculate_delta_sigma_cut` first, then re-calculate point estimate/sigma spectra."
                )

        if self._point_estimate_spectrogram_set:
            # reweight based on alpha that has been supplied
            self.point_estimate_spectrogram.reweight(new_alpha=alpha, new_fref=fref)
            self.sigma_spectrogram.reweight(new_alpha=alpha, new_fref=fref)
        else:
            logger.info(
                "Point estimate and sigma spectrograms are not set yet. setting now..."
            )
            self.set_point_estimate_sigma_spectrogram(
                alpha=alpha,
                fref=fref,
                flow=flow,
                fhigh=fhigh,
                polarization=polarization,
            )
        deltaF = self.frequencies[1] - self.frequencies[0]

        # should be True for each bad time
        bad_times_indexes = self._get_bad_times_indexes(times=self.point_estimate_spectrogram.times.value, badtimes=badtimes, apply_dsc=apply_dsc)

        logger.info(f"{np.sum(bad_times_indexes)} bad segments removed.")

        # start time, for metadata
        epoch = self.point_estimate_spectrogram.times[0]

        if self.sampling_frequency is None:
            raise AttributeError(
                "the sampling frequency is not set! Cannot proceed with spectrum calculation."
            )

        if apply_dsc:
            if len(self.delta_sigmas["times"]) == len(self.badGPStimes):
                warnings.warn(
                    "The delta sigma cut has flagged all times in this dataset. No point estimate/sigma values can be calculated."
                )
                self.point_estimate_spectrum =  np.zeros(len(self.frequencies))
                self.sigma_spectrum = np.inf * np.ones(len(self.frequencies))
                return 
            
        # setting the frequency mask for the before/after calculation
        self.set_frequency_mask(
            notch_list_path=notch_list_path, apply_notches=apply_notches
        )

        point_estimate, sigma = postprocess_Y_sigma(
            self.point_estimate_spectrogram.value,
            self.sigma_spectrogram.value ** 2,
            self.duration,
            deltaF,
            self.sampling_frequency,
            frequency_mask=self.frequency_mask,
            window_fftgram_dict=self.window_fftgram_dict,
            window_fftgram_dict_welch=self.window_fftgram_dict_for_bias_factors,
            badtimes_mask=bad_times_indexes,
            overlap_factor=self.overlap_factor,
            overlap_factor_welch=self.overlap_factor_psd,
            N_avg_segs=self.N_average_segments_psd,
        )

        self.point_estimate_spectrum = OmegaSpectrum(
            point_estimate,
            frequencies=self.frequencies,
            name=self.name + " point estimate spectrum",
            epoch=epoch,
            alpha=alpha,
            fref=fref,
            h0=h0,
        )
        self.sigma_spectrum = OmegaSpectrum(
            sigma,
            frequencies=self.frequencies,
            name=self.name + " sigma spectrum",
            epoch=epoch,
            alpha=alpha,
            fref=fref,
            h0=h0,
        )
        self.point_estimate_alpha = alpha

    def set_point_estimate_sigma(
        self,
        badtimes=None,
        alpha=0.0,
        fref=25,
        flow=20,
        fhigh=1726,
        notch_list_path="",
        polarization="tensor",
        apply_dsc=True,
        apply_notches=True,
    ):
        """
        Set point estimate sigma based on a set of parameters. This is estimate of omega_gw in each frequency bin.

        Parameters
        =======

        badtimes: ``array_like``, optional
            Array of times to exclude from point estimate/sigma calculation.
            If no times are passed, none will be excluded.

        alpha: ``float``, optional
            Spectral index to use in the re-weighting. Default is 0.

        fref: ``float``, optional
            Reference frequency to use in the re-weighting. Default is 25.

        flow: ``float``, optional
            Low frequency. Default is 20 Hz.

        fhigh: ``float``, optional
            High frequency. Default is 1726 Hz.

        notch_list_path: ``str``, optional
            Path to the notch list to use in the spectrum; if the notch_list isn't set in the baseline,
            user can pass it directly here. If it is not set and if none is passed no notches will be applied.

        polarization: ``str``, optional
            Polarization of the signal to consider (scalar, vector, tensor) for the orf calculation.
            Default is Tensor.

        apply_dsc: ``bool``, optional
            Apply delta sigma cut flag; if True, removes the badGPStimes from the spectra calculations.
            Default is True.

        apply_notches: ``bool``, optional
            Apply spectral notches flag; if True, remove the notches specified in the notch_list from the spectra calculations.
            Default is True.

        See also
        --------
        pygwb.postprocessing.calc_Y_sigma_from_Yf_sigmaf
        """
        if self._point_estimate_spectrum_set:
            self.point_estimate_spectrum.reweight(new_alpha=alpha, new_fref=fref)
            self.sigma_spectrum.reweight(new_alpha=alpha, new_fref=fref)
            self.point_estimate_spectrogram.reweight(new_alpha=alpha, new_fref=fref)
            self.sigma_spectrogram.reweight(new_alpha=alpha, new_fref=fref)
        else:
            logger.info(
                "Point estimate and sigma spectra have not been set before. Setting it now..."
            )
            logger.debug(
                "No weighting supplied in setting of spectrum. Supplied when combining for final sigma"
            )
            self.set_point_estimate_sigma_spectrum(
                badtimes=badtimes,
                alpha=alpha,
                fref=fref,
                flow=flow,
                fhigh=fhigh,
                polarization=polarization,
                apply_dsc=apply_dsc,
                notch_list_path=notch_list_path,
            )

        self.set_frequency_mask(
            notch_list_path=notch_list_path, apply_notches=apply_notches
        )

        if self.sigma_spectrum[0] == np.inf:
                self.sigma = np.inf
                self.point_estimate = 0
                return 

        Y, sigma = calc_Y_sigma_from_Yf_sigmaf(
            self.point_estimate_spectrum,
            self.sigma_spectrum,
            frequency_mask=self.frequency_mask,
            alpha=alpha,
            fref=fref,
        )

        self.point_estimate = Y
        self.sigma = sigma

    def reweight(self, new_alpha=None, new_fref=None):
        """Reweight all the frequency-weighted attributes of this Baseline, if these are set.

        Parameters
        =======

        new_alpha: ``float``, optional
            New alpha to weight the spectra to. Default is None.

        new_fref: ``float``, optional
            New reference frequency to refer the spectra to. Default is None.
        """
        self.set_point_estimate_sigma(alpha=new_alpha, fref=new_fref)

    def calculate_delta_sigma_cut(
        self,
        delta_sigma_cut,
        alphas,
        fref,
        flow=20,
        fhigh=1726,
        notch_list_path="",
        polarization="tensor",
        return_naive_and_averaged_sigmas: bool = False,
    ):
        """
        Calculate the delta sigma cut using the naive and average psds, if set in the baseline.

        Parameters
        =======

        delta_sigma_cut: ``float``
            The cutoff to implement in the delta sigma cut.

        alphas: ``list``
            Set of spectral indices to use in the delta sigma cut calculation.

        flow: ``float``, optional
            Low frequency. Default is 20 Hz.

        fhigh: ``float``, optional
            High frequency. Default is 1726 Hz.

        notch_list_path: ``str``, optional
            File path of the baseline notch list.

        fref: ``int``
            Reference frequency (Hz).

        return_naive_and_averaged_sigmas: ``bool``, optional
            Option to return naive and sliding sigmas. Default is False.

        polarization: ``str``, optional
            Polarization of the signal to consider (scalar, vector, tensor) for the orf calculation.
            Default is tensor.

        See also
        --------
        pygwb.delta_sigma_cut.run_dsc
            Function used to run the delta sigma cut.
        """
        if not self._orf_polarization_set:
            self.orf_polarization = polarization

        deltaF = self.frequencies[1] - self.frequencies[0]
        self.crop_frequencies_average_psd_csd(flow=flow, fhigh=fhigh)

        naive_psd_1 = self.interferometer_1.psd_spectrogram[
            self.csd_segment_offset:-self.csd_segment_offset
        ]
        naive_psd_2 = self.interferometer_2.psd_spectrogram[
            self.csd_segment_offset:-self.csd_segment_offset
        ]

        naive_psd_1_cropped = naive_psd_1.crop_frequencies(flow, fhigh + deltaF)
        naive_psd_2_cropped = naive_psd_2.crop_frequencies(flow, fhigh + deltaF)
        
        if notch_list_path:
            self.notch_list_path = notch_list_path
        if self.notch_list_path:
            logger.debug(
                "loading notches for delta sigma cut from " + self.notch_list_path
            )
            self.set_frequency_mask(self.notch_list_path)
        else:
            self.set_frequency_mask()

        badGPStimes, delta_sigmas = run_dsc(
            dsc=delta_sigma_cut,
            segment_duration=self.duration,
            psd1_naive=naive_psd_1_cropped,
            psd2_naive=naive_psd_2_cropped,
            psd1_slide=self.interferometer_1.average_psd,
            psd2_slide=self.interferometer_2.average_psd,
            alphas=alphas,
            sample_rate=self.sampling_frequency,
            orf=self.overlap_reduction_function,
            fref=fref,
            frequency_mask=self.frequency_mask,
            window_fftgram_dict=self.window_fftgram_dict_for_bias_factors,
            overlap_factor=self.overlap_factor_psd, 
            N_average_segments_psd = self.N_average_segments_psd,
            return_naive_and_averaged_sigmas=return_naive_and_averaged_sigmas,
        )

        self.badGPStimes = badGPStimes
        self.delta_sigmas = delta_sigmas

    def set_coherence_spectrum(self, flow=20, fhigh=1726, badtimes=None, apply_dsc=True):
        """
        Set the coherence spectrum between detectors, averaged over all data in the baseline.

        Parameters
        =======

        flow: ``float``, optional
            Low frequency. Default is 20 Hz.

        fhigh: ``float``, optional
            High frequency. Default is 1726 Hz.

        badtimes: ``array_like``, optional
            Array of times to exclude from the coherence calculation.
            Default is None.

        apply_dsc: ``bool``, optional
            Apply delta sigma cut flag; if True, removes the badGPStimes from the spectra calculations.
            Default is True.

        Notes
        -----

        The coherence calculation uses averaged naive PSD estimates as the coherence is calculated using CSD and PSD estimates of each individual segment, calculated \"on shell\".
        
        See also
        --------
        pygwb.coherence.calculate_coherence
        """
        bad_times_indexes = self._get_bad_times_indexes(times=self.interferometer_1.psd_spectrogram.times.value, apply_dsc=apply_dsc)

        deltaF = self.frequencies[1] - self.frequencies[0]
        n_segs = len(self.interferometer_1.psd_spectrogram[~bad_times_indexes])

        psd_1_average = np.mean(self.interferometer_1.psd_spectrogram[~bad_times_indexes].crop_frequencies(flow, fhigh + deltaF), axis=0)
        psd_2_average = np.mean(self.interferometer_2.psd_spectrogram[~bad_times_indexes].crop_frequencies(flow, fhigh + deltaF), axis=0)
        csd_average = np.mean(self.csd[~bad_times_indexes].crop_frequencies(flow, fhigh + deltaF), axis=0)

        coherence = calculate_coherence(
            psd_1_average,
            psd_2_average,
            csd_average,
        )

        epoch = self.csd.times[0]

        self.coherence_spectrum = FrequencySeries(
            coherence,
            frequencies=self.frequencies,
            name=self.name + " coherence spectrum",
            epoch=epoch,
        )

        self.coherence_dict = {}
        self.coherence_dict['psd_1_average'] = psd_1_average
        self.coherence_dict['psd_2_average'] = psd_2_average
        self.coherence_dict['csd_average']= csd_average
        self.coherence_dict['n_segs']= n_segs
        self.coherence_dict['coherence']= coherence
        self.coherence_dict['frequencies']= self.frequencies
        self.coherence_dict['epoch']= epoch

    def _get_bad_times_indexes(self, times, badtimes=None, apply_dsc=False):
        """
        Get indices for segments with bad GPS times, to be removed from analysis.

        Parameters
        =======

        badtimes: ``array_like``, optional
            Array of times to exclude from further calculation.
            Default is None.

        apply_dsc: ``bool``
            If True, calculates the indexes of the segments with a bad GPS time, according to the delta sigma cut. If False, returns None.
        """
        if apply_dsc:
            if not hasattr(self, "badGPStimes"):
                warnings.warn(
                    "Delta sigma cut has not been calculated yet, hence no delta sigma cut will be applied... If this is a mistake, please run `calculate_delta_sigma_cut` first, then re-calculate point estimate/sigma spectra."
                )
            if badtimes is not None:
                if hasattr(self, "badGPStimes"):
                    badtimes = np.append(badtimes, self.badGPStimes)
                self.badGPStimes = badtimes
            else:
                badtimes = self.badGPStimes
        else:
            badtimes = np.array([])

        bad_times_indexes = np.array(
            [np.any(t == badtimes) for t in times]
        )
        return bad_times_indexes

    def save_point_estimate_spectra(
        self,
        save_data_type,
        filename,
    ):
        """
        Save the overall point estimate Y, its error bar sigma,
        the frequency-dependent estimates and variances and the corresponding frequencies
        in the required save_data_type, which can be npz, pickle, json or hdf5.
        You can call upon this data afterwards when loaoding in using the ['key'] dictionary format.

        Parameters
        =======
        
        save_data_type: ``str``
            The required type of data file where the information will be stored.

        filename: ``str``
            The path/name of the file in which you want to save.
        """
        if save_data_type == "pickle":
            save = self._pickle_save
            ext = ".p"

        elif save_data_type == "npz":
            save = self._npz_save
            ext = ".npz"

        elif save_data_type == "json":
            save = self._json_save
            ext = ".json"

        elif save_data_type == "hdf5" or save_data_type == "h5":
            save = self._hdf5_save
            ext = ".h5"

        else:
            raise ValueError(
                "The provided data type is not supported, try using 'pickle', 'npz', 'json' or 'hdf5' instead."
            )

        save(
            f"{filename}{ext}",
            self.frequencies,
            self.frequency_mask,
            self.point_estimate_spectrum,
            self.sigma_spectrum,
            self.point_estimate,
            self.sigma,
            self.point_estimate_spectrogram,
            self.sigma_spectrogram,
            self.badGPStimes,
            self.delta_sigmas,
            self.interferometer_1.gates,
            self.interferometer_1.gate_pad,
            self.interferometer_2.gates,
            self.interferometer_2.gate_pad,
        )

    def save_psds_csds(
        self,
        save_data_type,
        filename,
    ):
        """
        Save the average and naive psds and csds and the corresponding frequencies
        in the required save_data_type, which can be npz, pickle, json or hdf5.
        One can call upon the data afterwards when loading in using the ['key'] dictionary format.

        Parameters
        =======

        save_data_type: ``str``
            The required type of data file where the information will be stored.

        filename: ``str``
            The path/name of the file in which you want to save.
        """
        if save_data_type == "pickle":
            save_csd = self._pickle_save_csd
            ext = ".p"

        elif save_data_type == "npz":
            save_csd = self._npz_save_csd
            ext = ".npz"

        elif save_data_type == "json":
            save_csd = self._json_save_csd
            ext = ".json"

        elif save_data_type == "hdf5" or save_data_type == "h5":
            save_csd = self._hdf5_save_csd
            ext = ".h5"

        else:
            raise ValueError(
                "The provided data type is not supported, try using 'pickle', 'npz', 'json' or 'hdf5' instead."
            )

        try:
            coherence = self.coherence_spectrum
            psd_1_coh = self.coherence_dict['psd_1_average']
            psd_2_coh = self.coherence_dict['psd_2_average']
            csd_coh = self.coherence_dict['csd_average']
            n_segs_coh = self.coherence_dict['n_segs']

        except AttributeError:
            coherence = None
            psd_1_coh = None
            psd_2_coh = None
            csd_coh = None
            n_segs_coh = None

        save_csd(
            f"{filename}{ext}",
            self.csd.frequencies.value,
            self.average_csd.frequencies.value,
            self.csd,
            self.average_csd,
            self.interferometer_1.psd_spectrogram,
            self.interferometer_2.psd_spectrogram,
            self.interferometer_1.average_psd,
            self.interferometer_2.average_psd,
            coherence,
            psd_1_coh,
            psd_2_coh,
            csd_coh,
            n_segs_coh,
        )

    def _npz_save(
        self,
        filename,
        frequencies,
        frequency_mask,
        point_estimate_spectrum,
        sigma_spectrum,
        point_estimate,
        sigma,
        point_estimate_spectrogram,
        sigma_spectrogram,
        badGPStimes,
        delta_sigmas,
        ifo_1_gates,
        ifo_1_gate_pad,
        ifo_2_gates,
        ifo_2_gate_pad,
    ):
        try:
            naive_sigma_values = delta_sigmas["naive_sigmas"]
            slide_sigma_values = delta_sigmas["slide_sigmas"]
        except KeyError:
            naive_sigma_values = None
            slide_sigma_values = None

        np.savez(
            filename,
            frequencies=frequencies,
            frequency_mask=frequency_mask,
            point_estimate_spectrum=point_estimate_spectrum,
            sigma_spectrum=sigma_spectrum,
            point_estimate=point_estimate,
            sigma=sigma,
            point_estimate_spectrogram=point_estimate_spectrogram,
            sigma_spectrogram=sigma_spectrogram,
            badGPStimes=badGPStimes,
            delta_sigma_alphas=delta_sigmas["alphas"],
            delta_sigma_times=delta_sigmas["times"],
            delta_sigma_values=delta_sigmas["values"],
            naive_sigma_values=naive_sigma_values,
            slide_sigma_values=slide_sigma_values,
            ifo_1_gates=ifo_1_gates,
            ifo_1_gate_pad=ifo_1_gate_pad,
            ifo_2_gates=ifo_2_gates,
            ifo_2_gate_pad=ifo_2_gate_pad,
        )

    def _pickle_save(
        self,
        filename,
        frequencies,
        frequency_mask,
        point_estimate_spectrum,
        sigma_spectrum,
        point_estimate,
        sigma,
        point_estimate_spectrogram,
        sigma_spectrogram,
        badGPStimes,
        delta_sigmas,
        ifo_1_gates,
        ifo_1_gate_pad,
        ifo_2_gates,
        ifo_2_gate_pad,
    ):
        save_dictionary = {
            "frequencies": frequencies,
            "frequency_mask": frequency_mask,
            "point_estimate_spectrum": point_estimate_spectrum,
            "sigma_spectrum": sigma_spectrum,
            "point_estimate": point_estimate,
            "sigma": sigma,
            "point_estimate_spectrogram": point_estimate_spectrogram,
            "sigma_spectrogram": sigma_spectrogram,
            "badGPStimes": badGPStimes,
            "delta_sigmas": list(delta_sigmas.items()),
            "ifo_1_gates": ifo_1_gates,
            "ifo_1_gate_pad": ifo_1_gate_pad,
            "ifo_2_gates": ifo_2_gates,
            "ifo_2_gate_pad": ifo_2_gate_pad,
        }

        with open(filename, "wb") as f:
            pickle.dump(save_dictionary, f)

    def _json_save(
        self,
        filename,
        frequencies,
        frequency_mask,
        point_estimate_spectrum,
        sigma_spectrum,
        point_estimate,
        sigma,
        point_estimate_spectrogram,
        sigma_spectrogram,
        badGPStimes,
        delta_sigmas,
        ifo_1_gates,
        ifo_1_gate_pad,
        ifo_2_gates,
        ifo_2_gate_pad,
    ):
        list_freqs = frequencies.tolist()
        list_freqs_mask = frequency_mask.tolist()
        list_point_estimate_spectrum_r = np.real(point_estimate_spectrum.value).tolist()
        list_point_estimate_spectrum_i = np.imag(point_estimate_spectrum.value).tolist()
        list_sigma_spectrum = sigma_spectrum.value.tolist()

        list_point_estimate_segment_r = np.real(
            point_estimate_spectrogram.value
        ).tolist()
        list_point_estimate_segment_i = np.imag(
            point_estimate_spectrogram.value
        ).tolist()
        point_estimate_segment_times = point_estimate_spectrogram.times.value.tolist()

        list_sigma_segment = sigma_spectrogram.value.tolist()
        sigma_segment_times = sigma_spectrogram.times.value.tolist()

        badGPStimes_list = badGPStimes.tolist()
        
        save_dictionary = {
            "frequencies": list_freqs,
            "frequency_mask": list_freqs_mask,
            "point_estimate_spectrum_real": list_point_estimate_spectrum_r,
            "point_estimate_spectrum_imag": list_point_estimate_spectrum_i,
            "sigma_spectrum": list_sigma_spectrum,
            "point_estimate": point_estimate,
            "sigma": sigma,
            "point_estimate_spectrogram_real": list_point_estimate_segment_r,
            "point_estimate_spectrogram_imag": list_point_estimate_segment_i,
            "point_estimate_spectrogram_times": point_estimate_segment_times,
            "sigma_spectrogram": list_sigma_segment,
            "sigma_spectrogram_times": sigma_segment_times,
            "badGPStimes": badGPStimes_list,
            "delta_sigma_alphas": delta_sigmas["alphas"],
            "delta_sigma_values": delta_sigmas["values"].tolist(),
            "ifo_1_gates": ifo_1_gates,
            "ifo_1_gate_pad": ifo_1_gate_pad,
            "ifo_2_gates": ifo_2_gates,
            "ifo_2_gate_pad": ifo_2_gate_pad,
        }
        try:
            save_dictionary["naive_sigma_values"] = delta_sigmas[
                "naive_sigmas"
            ].tolist()
            save_dictionary["slide_sigma_values"] = delta_sigmas[
                "slide_sigmas"
            ].tolist()
        except KeyError:
            pass

        with open(filename, "w") as outfile:
            json.dump(save_dictionary, outfile)

    def _hdf5_save(
        self,
        filename,
        frequencies,
        frequency_mask,
        point_estimate_spectrum,
        sigma_spectrum,
        point_estimate,
        sigma,
        point_estimate_spectrogram,
        sigma_spectrogram,
        badGPStimes,
        delta_sigmas,
        ifo_1_gates,
        ifo_1_gate_pad,
        ifo_2_gates,
        ifo_2_gate_pad,
        compress=False,
    ):
        hf = h5py.File(filename, "w")

        if compress:
            compression = "gzip"
            logger.info("Data will be compressed without loss of data")
        else:
            compression = None

        hf.create_dataset("freqs", data=frequencies, compression=compression)
        hf.create_dataset("freqs_mask", data=frequency_mask, compression=compression)
        hf.create_dataset(
            "point_estimate_spectrum",
            data=point_estimate_spectrum,
            compression=compression,
        )
        hf.create_dataset(
            "sigma_spectrum", data=sigma_spectrum, compression=compression
        )
        hf.create_dataset(
            "point_estimate",
            data=point_estimate,
            dtype="float",
            compression=compression,
        )
        hf.create_dataset("sigma", data=sigma, dtype="float", compression=compression)
        hf.create_dataset(
            "point_estimate_spectrogram",
            data=point_estimate_spectrogram,
            compression=compression,
        ),
        hf.create_dataset(
            "sigma_spectrogram", data=sigma_spectrogram, compression=compression
        )
        hf.create_dataset("badGPStimes", data=badGPStimes, compression=compression)
        delta_sigmas_group = hf.create_group("delta_sigmas")
        delta_sigmas_group.create_dataset(
            "delta_sigma_alphas", data=delta_sigmas["alphas"], compression=compression
        )
        delta_sigmas_group.create_dataset(
            "delta_sigma_times", data=delta_sigmas["times"], compression=compression
        )
        delta_sigmas_group.create_dataset(
            "delta_sigma_values", data=delta_sigmas["values"], compression=compression
        )
        try:
            delta_sigmas_group.create_dataset(
                "naive_sigma_values",
                data=delta_sigmas["naive_sigmas"],
                compression=compression,
            )
            delta_sigmas_group.create_dataset(
                "slide_sigma_values",
                data=delta_sigmas["slide_sigmas"],
                compression=compression,
            )
        except KeyError:
            pass

        gates_group = hf.create_group("Gated_Times")
        gates_group.create_dataset(
            "ifo_1_gates", data=ifo_1_gates, compression=compression
        )
        gates_group.create_dataset(
            "ifo_2_gates", data=ifo_2_gates, compression=compression
        )

        hf.close()

    def _npz_save_csd(
        self,
        filename,
        freqs,
        avg_freqs,
        csd,
        avg_csd,
        psd_1,
        psd_2,
        avg_psd_1,
        avg_psd_2,
        coherence,
        psd_1_coh,
        psd_2_coh,
        csd_coh,
        n_segs_coh,
    ):
        np.savez(
            filename,
            freqs=freqs,
            avg_freqs=avg_freqs,
            csd=csd,
            avg_csd=avg_csd,
            psd_1=psd_1,
            psd_2=psd_2,
            avg_psd_1=avg_psd_1,
            avg_psd_2=avg_psd_2,
            csd_times=csd.times.value,
            avg_csd_times=avg_csd.times.value,
            psd_times=psd_1.times.value,
            avg_psd_times=avg_psd_1.times.value,
            coherence=coherence,
            psd_1_coh=psd_1_coh,
            psd_2_coh=psd_2_coh,
            csd_coh=csd_coh,
            n_segs_coh=n_segs_coh,
        )

    def _pickle_save_csd(
        self,
        filename,
        freqs,
        avg_freqs,
        csd,
        avg_csd,
        psd_1,
        psd_2,
        avg_psd_1,
        avg_psd_2,
        coherence,
        psd_1_coh,
        psd_2_coh,
        csd_coh,
        n_segs_coh,
    ):

        save_dictionary = {
            "freqs": freqs,
            "avg_freqs": avg_freqs,
            "csd": csd,
            "avg_csd": avg_csd,
            "psd_1": psd_1,
            "psd_2": psd_2,
            "avg_psd_1": avg_psd_1,
            "avg_psd_2": avg_psd_2,
            "coherence": coherence,
            "psd_1_coh": psd_1_coh,
            "psd_2_coh": psd_2_coh,
            "csd_coh": csd_coh,
            "n_segs_coh": n_segs_coh,
        }

        # with open(filename, "wb") as f:
        #   pickle.dump(saveObject, f)

        with open(filename, "wb") as f:
            pickle.dump(save_dictionary, f)

    def _json_save_csd(
        self,
        filename,
        freqs,
        avg_freqs,
        csd,
        avg_csd,
        psd_1,
        psd_2,
        avg_psd_1,
        avg_psd_2,
        coherence,
        psd_1_coh,
        psd_2_coh,
        csd_coh,
        n_segs_coh,
    ):
        """
        It seems that saving spectrograms in json does not work, hence everything is converted into a list and saved that way in the json file.
        A second issue is that json does not seem to recognise complex values, hence the csd is split up into a real and imaginary part.
        When loading in this json file, one needs to 'reconstruct' the csd as a spectrogram using these two lists and the times and frequencies.
        """
        list_freqs = freqs.tolist()
        list_avg_freqs = avg_freqs.tolist()
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
        list_avg_csd = avg_csd.value.tolist()
        real_avg_csd = np.zeros(np.shape(list_avg_csd))
        imag_avg_csd = np.zeros(np.shape(list_avg_csd))
        for index, row in enumerate(list_avg_csd):
            for j, elem in enumerate(row):
                real_avg_csd[index, j] = elem.real
                imag_avg_csd[index, j] = elem.imag
        real_avg_csd_list = real_avg_csd.tolist()
        imag_avg_csd_list = imag_avg_csd.tolist()
        avg_csd_times = avg_csd.times.value.tolist()
        list_psd_1 = psd_1.value.tolist()
        psd_times = psd_1.times.value.tolist()
        list_psd_2 = psd_2.value.tolist()
        psd_2_times = psd_2.times.value.tolist()
        list_avg_psd_1 = avg_psd_1.value.tolist()
        avg_psd_times = avg_psd_1.times.value.tolist()
        list_avg_psd_2 = avg_psd_2.value.tolist()
        avg_psd_2_times = avg_psd_2.times.value.tolist()
        if coherence:
            list_coherence = coherence.value.tolist()
            list_psd_1_coh = psd_1_coh.value.tolist()
            list_psd_2_coh = psd_2_coh.value.tolist()
            list_csd_coh = csd_coh.value.tolist()
            real_csd_coh = np.zeros(np.shape(list_csd_coh))
            imag_csd_coh = np.zeros(np.shape(list_csd_coh))
            for ix, elem in enumerate(list_csd_coh):
                    real_csd_coh[ix] = elem.real
                    imag_csd_coh[ix] = elem.imag
            real_csd_coh_list = real_csd_coh.tolist()
            imag_csd_coh_list = imag_csd_coh.tolist()
            list_n_segs_coh = n_segs_coh
        else:
            list_coherence = None
            list_psd_1_coh = None
            list_psd_2_coh = None
            list_csd_coh = None
            real_csd_coh_list = None
            imag_csd_coh_list = None
            list_n_segs_coh = None

        save_dictionary = {
            "frequencies": list_freqs,
            "avg_frequencies": list_avg_freqs,
            "csd_real": real_csd_list,
            "csd_imag": imag_csd_list,
            "csd_times": csd_times,
            "avg_csd_real": real_avg_csd_list,
            "avg_csd_imag": imag_avg_csd_list,
            "avg_csd_times": avg_csd_times,
            "psd_1": list_psd_1,
            "psd_1_times": psd_times,
            "psd_2": list_psd_2,
            "psd_2_times": psd_2_times,
            "avg_psd_1": list_avg_psd_1,
            "avg_psd_1_times": avg_psd_times,
            "avg_psd_2": list_avg_psd_2,
            "avg_psd_2_times": avg_psd_2_times,
            "coherence": list_coherence,
            "psd_1_coh": list_psd_1_coh,
            "psd_2_coh": list_psd_2_coh,
            "csd_coh_real": real_csd_coh_list,
            "csd_coh_imag": real_csd_coh_list,
            "n_segs_coh": list_n_segs_coh,
        }

        with open(filename, "w") as outfile:
            json.dump(save_dictionary, outfile)

    def _hdf5_save_csd(
        self,
        filename,
        freqs,
        avg_freqs,
        csd,
        avg_csd,
        psd_1,
        psd_2,
        avg_psd_1,
        avg_psd_2,
        coherence,
        psd_1_coh,
        psd_2_coh,
        csd_coh,
        n_segs_coh,
        compress=False,
    ):
        hf = h5py.File(filename, "w")

        csd_times = csd.times.value
        psd_1_times = psd_1.times.value
        psd_2_times = psd_2.times.value
        avg_csd_times = avg_csd.times.value
        avg_psd_1_times = avg_psd_1.times.value
        avg_psd_2_times = avg_psd_2.times.value

        if compress:
            compression = "gzip"
        else:
            compression = None

        hf.create_dataset("freqs", data=freqs, compression=compression)
        hf.create_dataset("avg_freqs", data=avg_freqs, compression=compression)

        csd_group = hf.create_group("csd_group")
        csd_group.create_dataset("csd", data=csd, compression=compression)
        csd_group.create_dataset("csd_times", data=csd_times, compression=compression)

        avg_csd_group = hf.create_group("avg_csd_group")
        avg_csd_group.create_dataset("avg_csd", data=avg_csd, compression=compression)
        avg_csd_group.create_dataset(
            "avg_csd_times", data=avg_csd_times, compression=compression
        )

        psd_group = hf.create_group("psds_group")

        psd_1_group = hf.create_group("psds_group/psd_1")
        psd_1_group.create_dataset("psd_1", data=psd_1, compression=compression)
        psd_1_group.create_dataset(
            "psd_1_times", data=psd_1_times, compression=compression
        )

        psd_2_group = hf.create_group("psds_group/psd_2")
        psd_2_group.create_dataset("psd_2", data=psd_2, compression=compression)
        psd_2_group.create_dataset(
            "psd_2_times", data=psd_2_times, compression=compression
        )

        avg_psd_group = hf.create_group("avg_psds_group")

        avg_psd_1_group = hf.create_group("avg_psds_group/avg_psd_1")
        avg_psd_1_group.create_dataset(
            "avg_psd_1", data=avg_psd_1, compression=compression
        )
        avg_psd_1_group.create_dataset(
            "avg_psd_1_times", data=avg_psd_1_times, compression=compression
        )

        avg_psd_2_group = hf.create_group("avg_psds_group/avg_psd_2")
        avg_psd_2_group.create_dataset(
            "avg_psd_2", data=avg_psd_2, compression=compression
        )
        avg_psd_2_group.create_dataset(
            "avg_psd_2_times", data=avg_psd_2_times, compression=compression
        )

        if coherence:
            hf.create_dataset("coherence", data=coherence, compression=compression)
            hf.create_dataset("psd_1_coherence", data=psd_1_coh, compression=compression)
            hf.create_dataset("psd_2_coherence", data=psd_2_coh, compression=compression)
            hf.create_dataset("csd_coherence", data=csd_coh, compression=compression)
            hf.create_dataset("n_segs_coherence", data=n_segs_coh, compression=compression)

        hf.close()

def get_baselines(interferometers, frequencies=None):
    """
    Get set of Baseline objects given a list of interferometers.

    Parameters
    =======

    interferometers: ``list``
        List of bilby.interferometer objects.
    
    frequencies: ``array_like``, optional
        Frequencies to construct the baseline with. Defaults to None.
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
