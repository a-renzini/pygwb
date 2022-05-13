import unittest

import numpy as np

from pygwb.postprocessing import postprocess_Y_sigma, reweight_spectral_object

# simple examples

NTIMES = 3
NFREQS = 2
Y = np.ones((NTIMES, NFREQS))
SIGMA = np.ones((NTIMES, NFREQS))
SIGMA[0, 0] = 2
SIGMA[1, 0] = 3
TIMES = np.array([1, 2, 3])
SEGDUR = 1
FREQS = np.array([1, 2])
DELTAF = 1
SAMPLE_RATE = 128

TESTMATFILE = "test/pproc/stoch.job1.mat"


# test object
class PostProcessingTest(unittest.TestCase):
    """
    test the base class
    """

    def test_average_over_times(self):
        newY1, newvar1 = postprocess_Y_sigma(Y, SIGMA, SEGDUR, DELTAF, SAMPLE_RATE)

        # check size
        self.assertTrue(np.size(newY1) == NFREQS)
        self.assertTrue(np.size(newvar1) == NFREQS)


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


class IsotropicJobTest(unittest.TestCase):
    def test_combined_segment_stats(self):
        """
        Test that combining over frequencies
        does what we want.
        """
        newY, newvar = postprocess_Y_sigma(Y, SIGMA, SEGDUR, DELTAF, SAMPLE_RATE)
        newsigma = np.sqrt(newvar)
        # test __repr__
        # combine over frequencies
        self.assertTrue(np.size(newY) == NFREQS)
        self.assertTrue(np.size(newsigma) == NFREQS)
        # check that theset wo guys run as well
        # combine over just time
        # check that combined sigmas are correct based on the way its
        # constructed
        self.assertAlmostEqual(newsigma[0], 1.59438329)
        self.assertAlmostEqual(newsigma[1], 1.25042364)


if __name__ == "__main__":
    unittest.main()
