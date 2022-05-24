import unittest

from gwpy.frequencyseries import FrequencySeries
import gwpy.testing.utils
import numpy as np

from pygwb import util


class WindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.window_fft_dict={'window_fftgram': 'tukey', 'alpha': 0.6}
        self.window_fft_dict_2={'window_fftgram': 'tukey', 'alpha': 0.6, 'sym': 'True'}

    def tearDown(self) -> None:
        del self.window_fft_dict

    def test_window_factors(self):
        window_check = tuple((1.0, 1.0, 0.0, 0.0))
        self.assertEqual(util.window_factors(1), window_check)

    def test_get_window_tuple(self):
        self.assertEqual(util.get_window_tuple(), tuple(['hann']))
        self.assertEqual(util.get_window_tuple(self.window_fft_dict), tuple(['tukey', 0.6]))
        self.assertEqual(util.get_window_tuple(self.window_fft_dict_2), tuple(['tukey', 0.6, 'True']))

class UtilTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_calc_rho1(self):
        self.assertEqual(util.calc_rho1(100000), 0.027775555605193483)

    def test_calc_bias(self):
        self.assertEqual(util.calc_bias(192, 1 / 32, 32), 1.0480235570801641)

    def test_omega_to_power(self):
        frequencies = np.arange(1.0, 100.0)
        omega = frequencies ** 3
        omega_check = FrequencySeries(
            3.19242291e-37 * np.ones(len(frequencies)), frequencies=frequencies
        )
        gwpy.testing.utils.assert_quantity_sub_equal(
            util.omega_to_power(omega, frequencies), omega_check, almost_equal=True
        )




if __name__ == "__main__":
    unittest.main()
