import copy
import os
import tempfile
import unittest
from pathlib import Path

import gwpy.testing.utils
import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram
from numpy.testing import assert_allclose, assert_array_equal

from pygwb.omega_spectra import (
    OmegaSpectrogram,
    OmegaSpectrum,
    reweight_spectral_object,
)


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
            self.original_spectrum / (self.freqs / fref_new) ** alpha_new,
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
                self.original_specgram[:, ii] / (self.freqs / fref_new) ** alpha_new,
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


class TestOmegaSpectrum(unittest.TestCase):
    def setUp(self) -> None:
        self.freqs = np.array([0, 1, 2, 3])
        self.scale_val = 34.2
        self.gwpy_spectrum = FrequencySeries(
            self.scale_val * np.ones(self.freqs.size), frequencies=self.freqs
        )
        self.alpha = 0.0
        self.fref = 25.0
        self.h0 = 1.0
        self.omega_spectrum = OmegaSpectrum(
            self.gwpy_spectrum, alpha=self.alpha, fref=self.fref, h0=self.h0
        )

    def tearDown(self) -> None:
        del self.freqs
        del self.scale_val
        del self.gwpy_spectrum
        del self.omega_spectrum
        del self.alpha
        del self.fref
        del self.h0

    def test_alpha_property(self):
        self.assertEqual(self.omega_spectrum.alpha, self.alpha)

    def test_fref_property(self):
        self.assertEqual(self.omega_spectrum.fref, self.fref)

    def test_h0_property(self):
        self.assertEqual(self.omega_spectrum.h0, self.h0)

    def test_reset_h0(self):
        new_h0 = 0.7
        omega_spec_reset = copy.copy(self.omega_spectrum)
        omega_spec_reset.reset_h0(new_h0=new_h0)
        self.assertEqual(omega_spec_reset.value[0]/self.omega_spectrum.value[0], (self.h0/new_h0)**2)

    def test_read_write(self):
        self.omega_spectrum.name = "test_omega_spectrum"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / "test.hdf5"
            self.omega_spectrum.write(tmp)
            new = OmegaSpectrum.read(tmp)
            gwpy.testing.utils.assert_quantity_sub_equal(new, self.omega_spectrum)

    def test_save_to_pickle(self):
        self.omega_spectrum.save_to_pickle("omega_spectrum_test.pickle")
        os.remove("omega_spectrum_test.pickle")

    def test_load_from_pickle(self):
        omega_sgram = OmegaSpectrum.load_from_pickle(
            "test/test_data/omega_spectrum_test.pickle"
        )


class TestOmegaSpectrogram(unittest.TestCase):
    def setUp(self) -> None:
        self.freqs = np.array([0, 0.1, 0.2, 0.3])
        self.scale_val = 34.2
        self.gwpy_specgram = Spectrogram(
            self.scale_val * np.ones((2, self.freqs.size)),
            frequencies=self.freqs,
            times=np.arange(2),
        )

        self.alpha = 0.0
        self.fref = 25.0
        self.h0 = 1.0
        self.omega_spectrogram = OmegaSpectrogram(
            self.gwpy_specgram, alpha=self.alpha, fref=self.fref, h0=self.h0
        )

    def tearDown(self) -> None:
        del self.freqs
        del self.scale_val
        del self.gwpy_specgram
        del self.omega_spectrogram
        del self.alpha
        del self.fref
        del self.h0

    def test_alpha_property(self):
        self.assertEqual(self.omega_spectrogram.alpha, self.alpha)

    def test_fref_property(self):
        self.assertEqual(self.omega_spectrogram.fref, self.fref)

    def test_h0_property(self):
        self.assertEqual(self.omega_spectrogram.h0, self.h0)

    def test_reset_h0(self):
        new_h0 = 0.7
        omega_spec_reset = copy.copy(self.omega_spectrogram)
        omega_spec_reset.reset_h0(new_h0=new_h0)
        self.assertEqual(omega_spec_reset.value[0,0]/self.omega_spectrogram.value[0,0], (self.h0/new_h0)**2)


    def test_read_write(self):
        self.omega_spectrogram.name = "test_omega_spectrogram"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / "test.hdf5"
            self.omega_spectrogram.write(tmp)
            new = OmegaSpectrogram.read(tmp)
            # Due to numerical precision the yindex comparison is failing and
            # there seems no way to pass almost equal option for this
            # attributes (attributes in general), so that attribute is 
            # compared separately with almost equal option
            gwpy.testing.utils.assert_quantity_sub_equal(new, self.omega_spectrogram, exclude=['yindex',])
            assert_allclose(new.yindex, self.omega_spectrogram.yindex)

    def test_save_to_pickle(self):
        self.omega_spectrogram.save_to_pickle("omega_spectrogram_test.pickle")
        os.remove("omega_spectrogram_test.pickle")

    def test_load_from_pickle(self):
        omega_sgram = OmegaSpectrogram.load_from_pickle(
            "test/test_data/omega_spectrogram_test.pickle"
        )


if __name__ == "__main__":
    unittest.main()
