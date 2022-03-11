import unittest

import numpy as np

from pygwb.spectral import coarse_grain, coarse_grain_exact, reweight_spectral_object
from gwpy.spectrogram import Spectrogram
from gwpy.frequencyseries import FrequencySeries


class TestReweightFunction(unittest.TestCase):
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

    def test_reweight_spectral_object(self):
        alpha_new = 3
        fref_new = 100
        alpha_old = 0
        fref_old = 100

        new_spec = reweight_spectral_object(self.original_spectrum, self.freqs,
                                            alpha_new, fref_new, alpha_old, fref_old)
        # formula to reweight spec with alpha=0 before
        self.assertTrue(np.array_equal(new_spec, self.original_spectrum * (self.freqs / fref_new)**alpha_new))

        # reweight back now
        old_again_spec = reweight_spectral_object(new_spec, self.freqs, alpha_old, fref_old, alpha_new, fref_new)
        self.assertTrue(np.array_equal(self.original_spectrum, old_again_spec))

        new_specgram = reweight_spectral_object(self.original_specgram, self.freqs,
                                                alpha_new, fref_new, alpha_old, fref_old)

        # check column by column
        for ii in range(self.original_specgram.times.size):
            self.assertTrue(np.array_equal(new_specgram[:, ii], self.original_specgram[:, ii] * (self.freqs / fref_new)**alpha_new))

        old_again_specgram = reweight_spectral_object(new_specgram, self.freqs,
                                                      alpha_old, fref_old, alpha_new, fref_new)
        for ii in range(self.original_specgram.times.size):
            self.assertTrue(np.array_equal(self.original_specgram[:, ii], old_again_specgram[:, ii]))


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
