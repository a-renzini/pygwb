import unittest

import numpy as np

from pygwb.postprocessing import postprocess_Y_sigma

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


class IsotropicJobTest(unittest.TestCase):
    def test_combined_segment_stats(self):
        """
        Test that combining over frequencies
        does what we want.
        """
        newY, newsigma = postprocess_Y_sigma(Y, SIGMA, SEGDUR, DELTAF, SAMPLE_RATE)
        #newsigma = np.sqrt(newvar)

        # test __repr__
        # combine over frequencies
        self.assertTrue(np.size(newY) == NFREQS)
        self.assertTrue(np.size(newsigma) == NFREQS)
        # check that theset wo guys run as well
        # combine over just time
        # check that combined sigmas are correct based on the way its
        # constructed
        self.assertAlmostEqual(newsigma[0], 0.755622856)
        self.assertAlmostEqual(newsigma[1], 0.592610753)


if __name__ == "__main__":
    unittest.main()
