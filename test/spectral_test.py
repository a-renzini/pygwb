import unittest

import numpy as np
from gwpy import timeseries
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram
import gwpy.testing.utils

from pygwb.spectral import (
    before_after_average,
    coarse_grain,
    coarse_grain_exact,
    cross_spectral_density,
    power_spectral_density,
    reweight_spectral_object,
)


class TestSpectralReweighting(unittest.TestCase):
    def setUp(self) -> None:
        self.freqs = np.array([1.12, 33.333, 17.324, 18.2345])
        # try not to use 1 anywhere...special number
        self.scale_val = 34.23456
        self.original_spectrum = FrequencySeries(self.scale_val * np.ones(self.freqs.size), frequencies=self.freqs)
        self.original_specgram = Spectrogram(self.scale_val * np.ones((self.freqs.size, 3)),
                                             frequencies=self.freqs,
                                             times=np.arange(3))

    def tearDown(self) -> None:
        del self.freqs
        del self.scale_val
        del self.original_specgram
        del self.original_spectrum

    def test_reweight_spectrum(self):
        """Apply weights to a spectrum, check against simple
        implementation. Undo the weighting, check against original
        """
        alpha_new = 3.1234
        fref_new = 101.678
        alpha_old = 0
        fref_old = 203.521

        # perform reweighting
        new_spec = reweight_spectral_object(self.original_spectrum, self.freqs,
                                            alpha_new, fref_new, alpha_old, fref_old)

        # compare to simple formula to reweight spec with alpha=0 before
        gwpy.testing.utils.assert_quantity_sub_equal(
            new_spec, self.original_spectrum * (self.freqs / fref_new)**alpha_new, almost_equal=True
        )

        # reweight back to original spectrum
        old_again_spec = reweight_spectral_object(new_spec, self.freqs, alpha_old, fref_old, alpha_new, fref_new)

        gwpy.testing.utils.assert_quantity_sub_equal(
            old_again_spec, self.original_spectrum, almost_equal=True
        )

    def test_reweight_spectrogram(self):
        """Apply weights to a spectrogram, check it against simple
        implementation. Undo the weighting, check against original
        """
        alpha_new = 3.1234
        fref_new = 101.678
        alpha_old = 0
        fref_old = 203.521

        # reweight our spectrogram object
        new_specgram = reweight_spectral_object(self.original_specgram, self.freqs,
                                                alpha_new, fref_new, alpha_old, fref_old)

        # check column by column (time by time), applying the simple formula to each column specifically
        for ii in range(self.original_specgram.times.size):
            gwpy.testing.utils.assert_quantity_sub_equal(
                new_specgram[:, ii],
                self.original_specgram[:, ii] * (self.freqs / fref_new)**alpha_new, almost_equal=True
            )

        old_again_specgram = reweight_spectral_object(new_specgram, self.freqs,
                                                      alpha_old, fref_old, alpha_new, fref_new)
        for ii in range(self.original_specgram.times.size):
            gwpy.testing.utils.assert_quantity_sub_equal(
                self.original_specgram[:, ii], old_again_specgram[:, ii], almost_equal=True
            )


class TestSpectralDensities(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 256
        self.segment_duration = 60
        self.no_of_segments = 4
        N = self.segment_duration * self.sample_rate * self.no_of_segments
        # random time series
        self.mu, self.sigma = 0, 1  # mean and standard deviation
        data = np.random.default_rng().normal(self.mu, self.sigma, N)
        data_start_time = 0
        self.time_series_data = timeseries.TimeSeries(
            data, t0=data_start_time, sample_rate=self.sample_rate
        )

    def tearDown(self) -> None:
        pass

    def test_csd(self):
        frequency_resolution = 0.25
        overlap_factor = 0.5
        csd = cross_spectral_density(
            self.time_series_data,
            self.time_series_data,
            self.segment_duration,
            frequency_resolution,
            overlap_factor=overlap_factor,
            zeropad=True,
            window_fftgram="hann",
        )

        # Check that the recovered value is within 10% of the expected, usually it is much smaller
        sigma_est = np.sqrt(np.sum(csd[0].value) * frequency_resolution)
        self.assertAlmostEqual(
            self.sigma,
            sigma_est,
            delta=0.1,
            msg="Injected and recovered parameters differ by more than 10%",
        )

        # Check that the number of arrays is as expected
        expected_no_of_arrays = (
            2 * self.no_of_segments - 1
        )  # only works for 50% overlapping
        self.assertEqual(
            expected_no_of_arrays, len(csd), msg="The csd array sizes do not match"
        )

    def test_psd(self):
        frequency_resolution = 0.25
        overlap_factor = 0.5
        psd = power_spectral_density(
            self.time_series_data,
            self.segment_duration,
            frequency_resolution,
            overlap_factor=overlap_factor,
            window_fftgram="hann",
        )
        N_avg_segs = 2
        avg_psd = before_after_average(psd, self.segment_duration, N_avg_segs)

        # Check that the value is within 10% of the expected, usually it is much smaller
        sigma_est = np.sqrt(np.sum(psd[0].value) * frequency_resolution)
        self.assertAlmostEqual(
            self.sigma,
            sigma_est,
            delta=0.10,
            msg="Injected and recovered parameters differ by more than 10%",
        )

        # Check that the number of arrays is as expected
        expected_no_of_arrays = (
            2 * self.no_of_segments - 1
        )  # only works for 50% overlapping
        self.assertEqual(
            expected_no_of_arrays, len(psd), msg="The psd array sizes do not match"
        )

        # Check whether the average PSD arrays is as expected
        expected_no_of_arrays = (
            2 * (self.no_of_segments - 2) - 1
        )  # only works for 50% overlapping
        self.assertEqual(
            expected_no_of_arrays,
            len(avg_psd),
            msg="The average psd array sizes do not match",
        )


class TestCoarseGrain(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_coarse_grain_matches_coarse_grain_exact_even(self):
        self.assertTrue(
            np.array_equal(
                coarse_grain(np.arange(17, dtype=float), 4),
                coarse_grain_exact(np.arange(17, dtype=float), 4),
            )
        )

    def test_coarse_grain_matches_coarse_grain_exact_odd(self):
        self.assertTrue(
            np.array_equal(
                coarse_grain(np.arange(25, dtype=float), 3),
                coarse_grain_exact(np.arange(25, dtype=float), 3),
            )
        )

    def test_coarse_grain_matches_coarse_grain_exact_float(self):
        self.assertTrue(
            np.array_equal(
                coarse_grain(np.arange(25, dtype=float), 3.3),
                coarse_grain_exact(np.arange(25, dtype=float), 3.3),
            )
        )

    def test_coarse_grain_frequencies_matches_expected(self):
        self.assertTrue(
            np.array_equal(
                np.linspace(0, 1, 33)[2:-2:2], coarse_grain(np.linspace(0, 1, 33), 2)
            )
        )


if __name__ == "__main__":
    unittest.main()
