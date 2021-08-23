import copy
import os
import unittest

import bilby
import numpy as np

from pygwb import baseline


class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.interferometer_1 = bilby.gw.detector.get_empty_interferometer("H1")
        self.interferometer_2 = bilby.gw.detector.get_empty_interferometer("H1")
        self.interferometer_2.name = "H2"
        self.interferometer_3 = bilby.gw.detector.get_empty_interferometer("L1")
        self.duration = 4.0
        self.sampling_frequency = 2048.0

    def tearDown(self):
        del self.interferometer_1
        del self.interferometer_2
        del self.interferometer_3
        del self.duration
        del self.sampling_frequency

    def test_no_duration(self):
        with self.assertRaises(AttributeError):
            base = baseline.Baseline(
                "H1H2", self.interferometer_1, self.interferometer_2
            )

    def test_duration_no_sampling_frequency(self):
        with self.assertRaises(AttributeError):
            base = baseline.Baseline(
                "H1H2", self.interferometer_1, self.interferometer_2, self.duration
            )

    def test_set_duration_sampling_frequency_not_from_interferomters(self):
        base = baseline.Baseline(
            "H1H2",
            self.interferometer_1,
            self.interferometer_2,
            self.duration,
            self.sampling_frequency,
        )
        self.assertTrue(base.duration, self.duration)
        self.assertTrue(base.interferometer_1.duration, self.duration)
        self.assertTrue(base.interferometer_2.duration, self.duration)
        self.assertTrue(base.sampling_frequency, self.sampling_frequency)
        self.assertTrue(
            base.interferometer_1.sampling_frequency, self.sampling_frequency
        )
        self.assertTrue(
            base.interferometer_2.sampling_frequency, self.sampling_frequency
        )

    def test_set_duration_sampling_frequency_from_ifo1(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 4.0
        ifo1.sampling_frequency = 2048.0
        base = baseline.Baseline("H1H2", ifo1, self.interferometer_2)
        self.assertTrue(np.array_equal(base.frequencies, ifo1.frequency_array))
        self.assertTrue(
            np.array_equal(base.interferometer_2.frequency_array, ifo1.frequency_array)
        )

    def test_set_duration_sampling_frequency_from_ifo2(self):
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 4.0
        ifo2.sampling_frequency = 2048.0
        base = baseline.Baseline("H1H2", self.interferometer_1, ifo2)
        self.assertTrue(np.array_equal(base.frequencies, ifo2.frequency_array))
        self.assertTrue(
            np.array_equal(base.interferometer_1.frequency_array, ifo2.frequency_array)
        )

    def test_interferometer_duration_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 8.0
        ifo1.sampling_frequency = 2048.0
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 4.0
        ifo2.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline("H1H2", ifo1, ifo2)

    def test_interferometer_sampling_frequency_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 4.0
        ifo1.sampling_frequency = 1024.0
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 4.0
        ifo2.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline("H1H2", ifo1, ifo2)

    def test_passed_duration_ifo_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 8.0
        ifo1.sampling_frequency = 2048.0
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 8.0
        ifo2.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline(
                "H1H2", ifo1, ifo2, self.duration, self.sampling_frequency
            )

    def test_passed_sampling_frequency_ifo_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 4.0
        ifo1.sampling_frequency = 1024.0
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 4.0
        ifo2.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline(
                "H1H2", ifo1, ifo2, self.duration, self.sampling_frequency
            )

    def test_passed_duration_ifo1_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 8.0
        ifo1.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline(
                "H1H2",
                ifo1,
                self.interferometer_2,
                self.duration,
                self.sampling_frequency,
            )

    def test_passed_sampling_frequency_ifo1_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 4.0
        ifo1.sampling_frequency = 1024.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline(
                "H1H2",
                ifo1,
                self.interferometer_2,
                self.duration,
                self.sampling_frequency,
            )

    def test_passed_duration_ifo2_mismatch(self):
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 8.0
        ifo2.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline(
                "H1H2",
                self.interferometer_1,
                ifo2,
                self.duration,
                self.sampling_frequency,
            )

    def test_passed_sampling_frequency_ifo2_mismatch(self):
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 4.0
        ifo2.sampling_frequency = 1024.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline(
                "H1H2",
                self.interferometer_1,
                ifo2,
                self.duration,
                self.sampling_frequency,
            )

    def test_H1H2_orf(self):
        base = baseline.Baseline(
            "H1H2",
            self.interferometer_1,
            self.interferometer_2,
            self.duration,
            self.sampling_frequency,
        )
        self.assertTrue(
            np.allclose(base.overlap_reduction_function, np.ones(len(base.frequencies)))
        )

    def test_H1L1_orf(self):
        orf_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "orfs/ORF_HL.dat"
        )
        freqs, orf_from_file = np.loadtxt(orf_file, unpack=True)
        duration = 1.0 / (freqs[1] - freqs[0])
        sampling_frequency = 2 * freqs[-1]
        base = baseline.Baseline(
            "H1L1",
            self.interferometer_1,
            self.interferometer_3,
            duration,
            sampling_frequency,
        )
        self.assertTrue(
            np.allclose(
                base.overlap_reduction_function[int(10 * duration) :],
                orf_from_file,
                atol=3e-4,
            )
        )

    def test_from_interferometers_equivalent(self):
        self.interferometer_1.set_strain_data_from_zero_noise(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.interferometer_3.set_strain_data_from_zero_noise(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.assertEqual(
            baseline.Baseline("H1L1", self.interferometer_1, self.interferometer_3),
            baseline.Baseline.from_interferometers(
                [self.interferometer_1, self.interferometer_3]
            ),
        )


if __name__ == "__main__":
    unittest.main()
