import unittest

import gwpy.testing.utils
import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram

from pygwb.omega_spectra import reweight_spectral_object


class TestSpectralReweighting(unittest.TestCase):
    def setUp(self) -> None:
        self.freqs = np.array([1.12, 33.333, 17.324, 18.2345])
        # try not to use 1 anywhere...special number
        self.scale_val = 34.23456
        self.original_spectrum = FrequencySeries(
            self.scale_val * np.ones(self.freqs.size), frequencies=self.freqs
        )
        self.original_specgram = Spectrogram(
            self.scale_val * np.ones((self.freqs.size, 3)),
            frequencies=self.freqs,
            times=np.arange(3),
        )

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
        new_spec = reweight_spectral_object(
            self.original_spectrum, self.freqs, alpha_new, fref_new, alpha_old, fref_old
        )

        # compare to simple formula to reweight spec with alpha=0 before
        gwpy.testing.utils.assert_quantity_sub_equal(
            new_spec,
            self.original_spectrum * (self.freqs / fref_new) ** alpha_new,
            almost_equal=True,
        )

        # reweight back to original spectrum
        old_again_spec = reweight_spectral_object(
            new_spec, self.freqs, alpha_old, fref_old, alpha_new, fref_new
        )

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
        new_specgram = reweight_spectral_object(
            self.original_specgram, self.freqs, alpha_new, fref_new, alpha_old, fref_old
        )

        # check column by column (time by time), applying the simple formula to each column specifically
        for ii in range(self.original_specgram.times.size):
            gwpy.testing.utils.assert_quantity_sub_equal(
                new_specgram[:, ii],
                self.original_specgram[:, ii] * (self.freqs / fref_new) ** alpha_new,
                almost_equal=True,
            )

        old_again_specgram = reweight_spectral_object(
            new_specgram, self.freqs, alpha_old, fref_old, alpha_new, fref_new
        )
        for ii in range(self.original_specgram.times.size):
            gwpy.testing.utils.assert_quantity_sub_equal(
                self.original_specgram[:, ii],
                old_again_specgram[:, ii],
                almost_equal=True,
            )


if __name__ == "__main__":
    unittest.main()
