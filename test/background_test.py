import json
import unittest
from test.conftest import testdir

import gwpy.testing.utils
import numpy as np
from gwpy.frequencyseries import FrequencySeries

from pygwb import background


class BackgroundTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_compute_Omega_from_CBC_dictionary(self):
        test_file=f"{testdir}/test_data/test_omega_gw_calculation.json"
        with open(test_file, 'r') as fp:
            test_data = json.load(fp)
        injection_dict = test_data['injections']
        sampling_frequency = test_data['sampling_frequency']
        T_obs = test_data['T_obs']
        freqs, omega_gw = background.compute_Omega_from_CBC_dictionary(injection_dict, sampling_frequency, T_obs, waveform_minimum_frequency=10)
        self.assertEqual(
            freqs.tolist(), test_data['freqs']
        )
        self.assertEqual(
            omega_gw.tolist(), test_data['omega_gw']
        )

if __name__ == "__main__":
    unittest.main()

