import unittest

import numpy as np

from pygwb.postprocessing import IsotropicJob, SingleStochasticJob

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
SAMPLE_RATE = 128

TESTMATFILE = "test/pproc/stoch.job1.mat"


# test object
class SingleStochJobTest(unittest.TestCase):
    """
    test the base class
    """

    def test_consistency_check(self):
        """
        consistency check
        """
        test_obj = SingleStochasticJob(
            Y, SIGMA, TIMES, SEGDUR, SAMPLE_RATE, frequencies=FREQS
        )
        test_obj.consistency_check()
        # make it fail
        test_obj.times = np.array([0])
        try:
            test_obj.consistency_check()
            # if we got to this point...it's wrong
            self.assertTrue(False)
        except ValueError as e:
            if e == "Shape of Y is incorrect (may be transposed?)":
                pass

    def test_apply_bad_gps_times(self):
        test_obj = SingleStochasticJob(
            Y, SIGMA, TIMES, SEGDUR, SAMPLE_RATE, frequencies=FREQS
        )
        test_obj.apply_bad_gps_times([1])
        self.assertTrue(int(np.size(test_obj.times)) == int(NTIMES - 1))
        self.assertTrue(np.shape(test_obj.Y) == (NTIMES - 1, NFREQS))
        self.assertTrue(np.shape(test_obj.sigma) == (NTIMES - 1, NFREQS))
        self.assertTrue(test_obj.ntimes_removed == 1)
        # apply it a second time with a time that is already gone
        # in the past this accidentally reset ntimes_removed.
        # it should no longer do this.
        test_obj.apply_bad_gps_times([1])
        self.assertTrue(test_obj.ntimes_removed == 1)
        test_obj.apply_bad_gps_times([2])
        self.assertTrue(test_obj.ntimes_removed == 2)

    def test_average_over_times(self):
        test_obj = SingleStochasticJob(
            Y, SIGMA, TIMES, SEGDUR, SAMPLE_RATE, frequencies=FREQS
        )
        # just test that it runs for now
        newY1, newsigma1 = test_obj._combine_non_time_dimension_even_odd()

        # check size
        self.assertTrue(np.size(newY1) == NFREQS)
        self.assertTrue(np.size(newsigma1) == NFREQS)


class IsotropicJobTest(unittest.TestCase):
    def test_combined_segment_stats(self):
        """
        Test that combining over frequencies
        does what we want.
        """
        test_obj = IsotropicJob(Y, SIGMA, TIMES, SEGDUR, SAMPLE_RATE, frequencies=FREQS)
        # test __repr__
        print(test_obj)
        # combine over frequencies
        newY, newsigma = test_obj.calculate_segment_by_segment_broadband_statistics(0)
        self.assertTrue(np.size(newY) == NTIMES)
        self.assertTrue(np.size(newsigma) == NTIMES)
        # check that theset wo guys run as well
        # combine over time and freq
        final_Y, finalSigma = test_obj.calculate_broadband_statistics(0)
        # combine over just time
        final_Y, finalSigma = (
            test_obj.combined_Y_spectrum,
            test_obj.combined_sigma_spectrum,
        )
        # check that combined sigmas are correct based on the way its
        # constructed
        self.assertAlmostEqual(newsigma[-1], 1 / np.sqrt(2))
        self.assertAlmostEqual(newsigma[0], 1 / np.sqrt(1 + 1.0 / 4))
        self.assertAlmostEqual(newsigma[1], 1 / np.sqrt(1 + 1.0 / 9))

    def test_read_matfile(self):
        # just make sure reading still works
        newjob = IsotropicJob.from_matlab_file(TESTMATFILE)
        self.assertTrue(newjob.segdur == 60)
        self.assertTrue(newjob.sample_rate == 128)
        self.assertTrue(newjob.df == 0.25)

if __name__ == "__main__":
    unittest.main()
