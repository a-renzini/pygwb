import copy
import os
import unittest

import bilby
import gwpy.testing.utils
import numpy as np
import pytest

from pygwb import baseline, parameters


class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.interferometer_1 = bilby.gw.detector.get_empty_interferometer("H1")
        self.interferometer_2 = bilby.gw.detector.get_empty_interferometer("H1")
        self.interferometer_2.name = "H2"
        self.interferometer_3 = bilby.gw.detector.get_empty_interferometer("L1")
        self.duration = 4.0
        self.sampling_frequency = 2048.0
        self.frequencies = np.arange(100)

    def tearDown(self):
        del self.interferometer_1
        del self.interferometer_2
        del self.interferometer_3
        del self.duration
        del self.frequencies

    def test_set_duration_frequencies_not_from_interferomters(self):
        base = baseline.Baseline(
            "H1H2",
            self.interferometer_1,
            self.interferometer_2,
            self.duration,
            self.frequencies,
        )
        self.assertTrue(base.duration, self.duration)
        self.assertTrue(base.interferometer_1.duration, self.duration)
        self.assertTrue(base.interferometer_2.duration, self.duration)
        self.assertTrue(np.all(base.frequencies == self.frequencies), True)

    def test_set_duration_from_ifo1(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 4.0
        base = baseline.Baseline("H1H2", ifo1, self.interferometer_2)
        # self.assertTrue(np.array_equal(base.frequencies, ifo1.frequency_array))
        # self.assertTrue(
        #    np.array_equal(base.interferometer_2.frequency_array, ifo1.frequency_array)
        # )

    def test_set_duration_from_ifo2(self):
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 4.0
        base = baseline.Baseline("H1H2", self.interferometer_1, ifo2)
        # self.assertTrue(np.array_equal(base.frequencies, ifo2.frequency_array))
        # self.assertTrue(
        #    np.array_equal(base.interferometer_1.frequency_array, ifo2.frequency_array)
        # )

    def test_interferometer_duration_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 8.0
        ifo1.sampling_frequency = 2048.0
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 4.0
        ifo2.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline("H1H2", ifo1, ifo2)

    # def test_interferometer_sampling_frequency_mismatch(self):
    #    ifo1 = copy.deepcopy(self.interferometer_1)
    #    ifo1.duration = 4.0
    #    ifo1.sampling_frequency = 1024.0
    #    ifo2 = copy.deepcopy(self.interferometer_2)
    #    ifo2.duration = 4.0
    #    ifo2.sampling_frequency = 2048.0
    #    with self.assertRaises(AssertionError):
    #        base = baseline.Baseline("H1H2", ifo1, ifo2)

    def test_passed_duration_ifo_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 8.0
        ifo1.sampling_frequency = 2048.0
        ifo2 = copy.deepcopy(self.interferometer_2)
        ifo2.duration = 8.0
        ifo2.sampling_frequency = 2048.0
        with self.assertRaises(AssertionError):
            base = baseline.Baseline("H1H2", ifo1, ifo2, self.duration)

    # def test_passed_sampling_frequency_ifo_mismatch(self):
    #    ifo1 = copy.deepcopy(self.interferometer_1)
    #    ifo1.duration = 4.0
    #    ifo1.sampling_frequency = 1024.0
    #    ifo2 = copy.deepcopy(self.interferometer_2)
    #    ifo2.duration = 4.0
    #    ifo2.sampling_frequency = 2048.0
    #    with self.assertRaises(AssertionError):
    #        base = baseline.Baseline(
    #            "H1H2", ifo1, ifo2, self.duration
    #        )

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
            )

    # def test_passed_sampling_frequency_ifo1_mismatch(self):
    #    ifo1 = copy.deepcopy(self.interferometer_1)
    #    ifo1.duration = 4.0
    #    ifo1.sampling_frequency = 1024.0
    #    with self.assertRaises(AssertionError):
    #        base = baseline.Baseline(
    #            "H1H2",
    #            ifo1,
    #            self.interferometer_2,
    #            self.duration,
    #        )

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
            )

    # def test_passed_sampling_frequency_ifo2_mismatch(self):
    #    ifo2 = copy.deepcopy(self.interferometer_2)
    #    ifo2.duration = 4.0
    #    ifo2.sampling_frequency = 1024.0
    #    with self.assertRaises(AssertionError):
    #        base = baseline.Baseline(
    #            "H1H2",
    #            self.interferometer_1,
    #            ifo2,
    #            self.duration,
    #        )

    def test_H1H2_orf(self):
        base = baseline.Baseline(
            "H1H2",
            self.interferometer_1,
            self.interferometer_2,
            self.duration,
            self.frequencies,
        )
        self.assertTrue(
            np.allclose(base.overlap_reduction_function, np.ones(len(base.frequencies)))
        )

    def test_H1L1_orf(self):
        orf_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "orfs/ORF_HL.dat"
        )
        freqs, orf_from_file = np.loadtxt(orf_file, unpack=True)
        base = baseline.Baseline(
            "H1L1",
            self.interferometer_1,
            self.interferometer_3,
            frequencies=freqs,
        )
        self.assertTrue(
            np.allclose(
                base.overlap_reduction_function,
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

    def test_save_to_pickle(self):
        base = baseline.Baseline(
            "H1H2",
            self.interferometer_1,
            self.interferometer_2,
        )
        base.save_to_pickle("test/test_data/baseline_pickle_test.pickle")

    def test_load_from_pickle(self):
        base = baseline.Baseline.load_from_pickle(
            "test/test_data/baseline_pickle_test.pickle"
        )

    def test_from_parameters(self):
        param_file = "test/test_data/parameters_baseline_test.ini"
        params = parameters.Parameters.from_file(param_file)
        base = baseline.Baseline.from_parameters(
            self.interferometer_1, self.interferometer_2, params
        )

    def test_set_cross_and_power_spectral_density(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        pickled_ifo_1 = pickled_base.interferometer_1
        pickled_ifo_2 = pickled_base.interferometer_2
        PSD_1_test = pickled_ifo_1.psd_spectrogram
        PSD_2_test = pickled_ifo_2.psd_spectrogram
        CSD_test = pickled_base.csd

        ifo_1 = copy.deepcopy(pickled_ifo_1)
        ifo_2 = copy.deepcopy(pickled_ifo_2)
        ifo_1.psd_spectrogram = None
        ifo_2.psd_spectrogram = None
        frequency_resolution = PSD_1_test.df.value
        base = baseline.Baseline.from_interferometers([ifo_1, ifo_2])
        base.set_cross_and_power_spectral_density(frequency_resolution)
        gwpy.testing.utils.assert_quantity_sub_equal(
            PSD_1_test, base.interferometer_1.psd_spectrogram, almost_equal=True
        )
        gwpy.testing.utils.assert_quantity_sub_equal(
            PSD_2_test, base.interferometer_2.psd_spectrogram, almost_equal=True
        )
        gwpy.testing.utils.assert_quantity_sub_equal(
            CSD_test, base.csd, almost_equal=True
        )

    def test_set_average_power_spectral_densities(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        pickled_base.crop_frequencies_average_psd_csd(50, 500)
        pickled_ifo_1 = pickled_base.interferometer_1
        pickled_ifo_2 = pickled_base.interferometer_2
        PSD_1_test = pickled_ifo_1.average_psd
        PSD_2_test = pickled_ifo_2.average_psd

        ifo_1 = copy.deepcopy(pickled_ifo_1)
        ifo_2 = copy.deepcopy(pickled_ifo_2)
        ifo_1.average_psd = None
        ifo_2.average_psd = None
        base = baseline.Baseline.from_interferometers([ifo_1, ifo_2])
        base.set_average_power_spectral_densities()
        base.frequencies = pickled_base.frequencies
        base.crop_frequencies_average_psd_csd(50, 500)
        gwpy.testing.utils.assert_quantity_sub_equal(
            PSD_1_test, base.interferometer_1.average_psd, almost_equal=True
        )
        gwpy.testing.utils.assert_quantity_sub_equal(
            PSD_2_test, base.interferometer_2.average_psd, almost_equal=True
        )

    def test_set_average_cross_spectral_density(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        pickled_base.crop_frequencies_average_psd_csd(50, 500)
        pickled_ifo_1 = pickled_base.interferometer_1
        pickled_ifo_2 = pickled_base.interferometer_2
        CSD_test = pickled_base.average_csd

        base = copy.deepcopy(pickled_base)
        base.average_csd = None
        base.set_average_cross_spectral_density()
        base.crop_frequencies_average_psd_csd(50, 500)
        gwpy.testing.utils.assert_quantity_sub_equal(
            CSD_test, base.average_csd, almost_equal=True
        )

    def test_calculate_delta_sigma_cut(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        badGPStimes_test = pickled_base.badGPStimes
        dsc_test = pickled_base.delta_sigmas
        base = copy.deepcopy(pickled_base)
        base.badGPStimes = None
        base.delta_sigmas = None
        notch_file = "test/test_data/Official_O3_HL_notchlist.txt"
        base.calculate_delta_sigma_cut(
            delta_sigma_cut=0.2, alphas=[-5, 0, 3], notch_list_path=notch_file
        )
        self.assertEqual(badGPStimes_test.tolist(), base.badGPStimes.tolist())

    def test_set_point_estimate_sigma(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        # save these for later...
        pickled_ifo_1 = pickled_base.interferometer_1
        pickled_ifo_2 = pickled_base.interferometer_2
        PSD_1_test = pickled_ifo_1.average_psd
        PSD_2_test = pickled_ifo_2.average_psd
        CSD_test = pickled_base.average_csd
        point_estimate_test = pickled_base.point_estimate
        sigma_test = pickled_base.sigma

        point_estimate_spectrum_test = pickled_base.point_estimate_spectrum
        sigma_spectrum_test = pickled_base.sigma_spectrum

        point_estimate_spectrogram_test = pickled_base.point_estimate_spectrogram
        sigma_spectrogram_test = pickled_base.sigma_spectrogram

        # get rid of a few things...
        ifo_1 = copy.deepcopy(pickled_ifo_1)
        ifo_2 = copy.deepcopy(pickled_ifo_2)
        ifo_1.psd_spectrogram = None
        ifo_2.psd_spectrogram = None
        pickled_ifo_1.average_psd = None
        pickled_ifo_2.average_psd = None
        base = copy.deepcopy(pickled_base)
        frequency_resolution = PSD_1_test.df.value

        # create new baseline from ifo's
        base = baseline.Baseline.from_interferometers([ifo_1, ifo_2])
        base.set_cross_and_power_spectral_density(frequency_resolution)
        base.set_average_cross_spectral_density()
        base.crop_frequencies_average_psd_csd(
            pickled_base.frequencies[0], pickled_base.frequencies[-1]
        )

        # run dsc
        base.calculate_delta_sigma_cut(
            delta_sigma_cut=np.inf,
            alphas=[-5, 0, 3],
            notch_list_path="test/test_data/Official_O3_HL_notchlist.txt",
        )

        # set point estimate, sigma with notch list
        base.set_point_estimate_sigma(
            notch_list_path="test/test_data/Official_O3_HL_notchlist.txt"
        )

        # check point estimate, sigma spectrum
        gwpy.testing.utils.assert_quantity_sub_equal(
            point_estimate_spectrum_test,
            base.point_estimate_spectrum,
            almost_equal=True,
        )

        gwpy.testing.utils.assert_quantity_sub_equal(
            sigma_spectrum_test, base.sigma_spectrum, almost_equal=True
        )

        # check point estimate, sigma spectrograms
        gwpy.testing.utils.assert_quantity_sub_equal(
            sigma_spectrogram_test, base.sigma_spectrogram, almost_equal=True
        )

        gwpy.testing.utils.assert_quantity_sub_equal(
            point_estimate_spectrogram_test,
            base.point_estimate_spectrogram,
            almost_equal=True,
        )
        # check final point estimate and sigma values
        self.assertAlmostEqual(sigma_test, base.sigma)
        self.assertAlmostEqual(point_estimate_test, base.point_estimate)


if __name__ == "__main__":
    unittest.main()
