import pickle
import unittest
from pathlib import Path

import numpy as np

from pygwb import baseline
from pygwb.delta_sigma_cut import run_dsc
from pygwb.notch import StochNotchList


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

        flow = 50
        fhigh = 500
        segment_duration = 192  # also fftlength in pre-processing
        overlap = 0.5
        segment_duration = 192  # also fftlength in pre-processing
        sampling_frequency = 4096
        dsc = 0.2
        alphas = [-5, 0, 3]
        notch_file = "test/test_data/Official_O3_HL_notchlist.txt"

        # test = Path(__file__).parent.resolve()
        # pickle_path = test / "test_data/naive_and_sliding_psds.pickle"

        # with open(pickle_path, "rb") as handle:
        #    pickle_loaded = pickle.load(handle)

        # naive_psd_1 = pickle_loaded["naive_psd_1"]
        # naive_psd_2 = pickle_loaded["naive_psd_2"]
        # avg_psd_1 = pickle_loaded["avg_psd_1"]
        # avg_psd_2 = pickle_loaded["avg_psd_2"]

        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        pickled_ifo_1 = pickled_base.interferometer_1
        pickled_ifo_2 = pickled_base.interferometer_2
        naive_psd_1 = pickled_ifo_1.psd_spectrogram
        naive_psd_2 = pickled_ifo_2.psd_spectrogram
        avg_psd_1 = pickled_ifo_1.average_psd
        avg_psd_2 = pickled_ifo_2.average_psd

        dF = avg_psd_1.frequencies.value[1] - avg_psd_1.frequencies.value[0]
        naive_psd_1 = naive_psd_1.crop_frequencies(flow, fhigh + dF)
        naive_psd_2 = naive_psd_2.crop_frequencies(flow, fhigh + dF)
        avg_psd_1 = avg_psd_1.crop_frequencies(flow, fhigh + dF)
        avg_psd_2 = avg_psd_2.crop_frequencies(flow, fhigh + dF)

        # calculate CSD
        stride = segment_duration - overlap
        csd_segment_offset = int(np.ceil(segment_duration / stride))

        # also remove naive psds from edge segments
        naive_psd_1 = naive_psd_1[csd_segment_offset : -(csd_segment_offset + 1) + 1]
        naive_psd_2 = naive_psd_2[csd_segment_offset : -(csd_segment_offset + 1) + 1]

        badGPStimes, _ = run_dsc(
            dsc=dsc,
            segment_duration=segment_duration,
            sampling_frequency=sampling_frequency,
            psd1_naive=naive_psd_1,
            psd2_naive=naive_psd_2,
            psd1_slide=avg_psd_1,
            psd2_slide=avg_psd_2,
            sample_rate=sampling_frequency,
            alphas=alphas,
            orf=np.array([1]),
            notch_list_path=notch_file,
        )

        self.assertTrue(badGPStimes[0], 1.24764440e09)
        self.assertTrue(badGPStimes[1], 1.24764449e09)
        self.assertTrue(badGPStimes[2], 1.24764459e09)


if __name__ == "__main__":
    unittest.main()
