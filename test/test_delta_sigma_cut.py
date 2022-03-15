import pickle
import unittest
from pathlib import Path

import numpy as np

from pygwb.delta_sigma_cut import run_dsc


def read_notch_list(notch_file):
    f_min_tmp, f_max_tmp, _ = np.loadtxt(
        notch_file, dtype="str", unpack=True, delimiter=","
    )

    f_min = []
    f_max = []

    for i in range(0, len(f_min_tmp)):
        f_min.append(float(f_min_tmp[i]))
        f_max.append(float(f_max_tmp[i]))

    output = [np.array(f_min), np.array(f_max)]

    return np.transpose(output)


class Test(unittest.TestCase):
    def setUp(self) -> None:

        return None

    def test_delta_sigma_cut(self):

        segment_duration = 192  # also fftlength in pre-processing
        sampling_frequency = 4096
        dsc = 0.2
        alphas = [-5, 0, 3]
        notch_file = "test/test_data/Official_O3_HL_notchlist.txt"

        test = Path(__file__).parent.resolve()
        pickle_path = test/'test_data/naive_and_sliding_psds.pickle'
        print(pickle_path)

        with open(pickle_path, 'rb') as handle:
            pickle_loaded = pickle.load(handle)

        naive_psd_1 = pickle_loaded['naive_psd_1']
        naive_psd_2 = pickle_loaded['naive_psd_2']
        avg_psd_1 = pickle_loaded['avg_psd_1']
        avg_psd_2 = pickle_loaded['avg_psd_2']

        badGPStimes, _ = run_dsc(
            dsc=dsc,
            segment_duration=segment_duration,
            sampling_frequency=sampling_frequency,
            psd1_naive=naive_psd_1,
            psd2_naive=naive_psd_2,
            psd1_slide=avg_psd_1,
            psd2_slide=avg_psd_2,
            alphas=alphas,
            notch_path=notch_file,
            orf=np.array([1])
        )
        
        self.assertTrue(badGPStimes[0], 1.24764440e09)
        self.assertTrue(badGPStimes[1], 1.24764449e09)
        self.assertTrue(badGPStimes[2], 1.24764459e09)

if __name__ == "__main__":
    unittest.main()
