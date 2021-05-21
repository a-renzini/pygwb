import bilby
from pygwb import baseline
import numpy as np
import unittest
import copy
import os


class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.interferometer_1 = bilby.gw.detector.get_empty_interferometer("H1")
        self.interferometer_2 = bilby.gw.detector.get_empty_interferometer("H1")
        self.interferometer_2.name = "H2"
        self.interferometer_3 = bilby.gw.detector.get_empty_interferometer("L1")
        self.frequencies = np.arange(0, 1024 + 0.25, 0.25)

    def tearDown(self):
        del self.interferometer_1
        del self.interferometer_2
        del self.interferometer_3
        del self.frequencies

    def test_no_frequencies(self):
        with self.assertRaises(AttributeError):
            base = baseline.Baseline(
                "H1H2", self.interferometer_1, self.interferometer_2
            )

    def test_set_frequencies_no_interferomter_freqs(self):
        base = baseline.Baseline(
            "H1H2", self.interferometer_1, self.interferometer_2, self.frequencies
        )
        self.assertTrue(np.array_equal(base.frequencies, self.frequencies))
        self.assertTrue(
            np.array_equal(base.interferometer_1.frequency_array, self.frequencies)
        )
        self.assertTrue(
            np.array_equal(base.interferometer_2.frequency_array, self.frequencies)
        )

    def test_set_frequencies_from_ifo1(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 4.0
        ifo1.sampling_frequency = 2048.0
        base = baseline.Baseline("H1H2", ifo1, self.interferometer_2)
        self.assertTrue(np.array_equal(base.frequencies, ifo1.frequency_array))
        self.assertTrue(
            np.array_equal(base.interferometer_2.frequency_array, ifo1.frequency_array)
        )

    def test_set_frequencies_from_ifo2(self):
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 4.0
        ifo2.sampling_frequency = 2048.0
        base = baseline.Baseline("H1H2", self.interferometer_1, ifo2)
        self.assertTrue(np.array_equal(base.frequencies, ifo2.frequency_array))
        self.assertTrue(
            np.array_equal(base.interferometer_1.frequency_array, ifo2.frequency_array)
        )

    def test_interferometer_frequencies_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 8.0
        ifo1.sampling_frequency = 2048.0
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 4.0
        ifo2.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline("H1H2", ifo1, ifo2)

    def test_passed_frequencies_ifo_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 8.0
        ifo1.sampling_frequency = 2048.0
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 8.0
        ifo2.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline("H1H2", ifo1, ifo2, self.frequencies)

    def test_passed_frequencies_ifo1_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 8.0
        ifo1.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline(
                "H1H2", ifo1, self.interferometer_2, self.frequencies
            )

    def test_passed_frequencies_ifo2_mismatch(self):
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 8.0
        ifo2.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline(
                "H1H2", self.interferometer_1, ifo2, self.frequencies
            )

    def test_H1H2_orf(self):
        base = baseline.Baseline(
            "H1H2", self.interferometer_1, self.interferometer_2, self.frequencies
        )
        self.assertTrue(
            np.allclose(base.overlap_reduction_function, np.ones(len(self.frequencies)))
        )

    def test_H1L1_orf(self):
        orf_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "orfs/ORF_HL.dat"
        )
        freqs, orf_from_file = np.loadtxt(orf_file, unpack=True)
        base = baseline.Baseline(
            "H1L1", self.interferometer_1, self.interferometer_3, freqs
        )
        self.assertTrue(
            np.allclose(base.overlap_reduction_function, orf_from_file, atol=3e-4)
        )


if __name__ == "__main__":
    unittest.main()
