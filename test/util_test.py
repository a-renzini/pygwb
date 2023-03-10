import unittest

import gwpy.testing.utils
import numpy as np
from gwpy.frequencyseries import FrequencySeries

from pygwb import util


class WindowTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_window_factors(self):
        window_check = tuple((1.0, 1.0, 0.0, 0.0))
        self.assertEqual(util.window_factors(1), window_check)

    def test_calc_rho1(self):
        self.assertEqual(util.calc_rho1(100000), 0.027775555605193483)

    def test_calc_rho(self):
        # Simple case, boxcar window of length 4, since it's easy to calculate exact values of rho
        self.assertEqual(util.calc_rho(4, 1, window_tuple="boxcar", overlap_factor=0.0), 0)
        self.assertEqual(util.calc_rho(4, 1, window_tuple="boxcar", overlap_factor=1/4), (1/4)**2)
        self.assertEqual(util.calc_rho(4, 1, window_tuple="boxcar", overlap_factor=1/2), (2/4)**2)
        self.assertEqual(util.calc_rho(4, 1, window_tuple="boxcar", overlap_factor=3/4), (3/4)**2)
        self.assertEqual(util.calc_rho(4, 1, window_tuple="boxcar", overlap_factor=1.0), 1)

    def test_effective_welch_averages(self):
        # Simple case, 8 samples, boxcar window of length 4
        self.assertEqual(util.effective_welch_averages(8, 4, window_tuple="boxcar", overlap_factor=0.0), 2)
        self.assertEqual(util.effective_welch_averages(8, 4, window_tuple="boxcar", overlap_factor=0.5), 2.25)

        # Effective Welch averages for 60s segments, 4s FFTs @ 4096 Hz
        self.assertAlmostEqual(util.effective_welch_averages(60*4096, 4*4096, window_tuple="hann", overlap_factor=0.5), 27.52432046866242)

        # Effective Welch averages for 192s segments, 32s FFTs @ 4096 Hz
        self.assertAlmostEqual(util.effective_welch_averages(192*4096, 32*4096, window_tuple="hann", overlap_factor=0.5), 10.47118457209110)

    def test_calc_bias(self):
        self.assertAlmostEqual(util.calc_bias(60, 1./4, 1./4096), 1.018501852824747)
        self.assertAlmostEqual(util.calc_bias(192, 1./32, 1./4096), 1.050144493503758)

    def test_omega_to_power(self):
        frequencies = np.arange(1.0, 100.0)
        omega = frequencies ** 3
        omega_check = FrequencySeries(
            1.46145149e-37 * np.ones(len(frequencies)), frequencies=frequencies
        )
        gwpy.testing.utils.assert_quantity_sub_equal(
            util.omega_to_power(omega, frequencies), omega_check, almost_equal=True
        )



if __name__ == "__main__":
    unittest.main()
