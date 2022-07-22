import copy
import json
import os
import pickle
import sys
import unittest

import bilby
import gwpy.testing.utils
import h5py
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

    def test_interferometer_duration_mismatch(self):
        ifo1 = copy.deepcopy(self.interferometer_1)
        ifo1.duration = 8.0
        ifo1.sampling_frequency = 2048.0
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
            base = baseline.Baseline("H1H2", ifo1, ifo2, self.duration)

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

    def test_H1H2_orf(self):
        base = baseline.Baseline(
            "H1H2",
            self.interferometer_1,
            self.interferometer_2,
            self.duration,
            self.frequencies,
        )
        base.orf_polarization='tensor'
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
        base.orf_polarization='tensor'
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
        params = parameters.Parameters()
        params.update_from_file(param_file)
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

        base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
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
        dsc_test = pickled_base.delta_sigmas['values']
        base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        base.badGPStimes = None
        base.delta_sigmas = None
        notch_file = "test/test_data/Official_O3_HL_notchlist.txt"
        print(base.frequency_mask)
        base.calculate_delta_sigma_cut(
            delta_sigma_cut=0.2, alphas=[-5, 0, 3], fref= 25
        )
        self.assertTrue(np.array_equal(badGPStimes_test, base.badGPStimes))

    def test_set_point_estimate_sigma(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
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

        ifo_1 = copy.deepcopy(pickled_ifo_1)
        ifo_2 = copy.deepcopy(pickled_ifo_2)
        ifo_1.psd_spectrogram = None
        ifo_2.psd_spectrogram = None
        ifo_1.average_psd = None
        ifo_2.average_psd = None
        frequency_resolution = PSD_1_test.df.value

        base = baseline.Baseline.from_interferometers([ifo_1, ifo_2])
        base.set_cross_and_power_spectral_density(frequency_resolution)
        base.set_average_power_spectral_densities()
        base.set_average_cross_spectral_density()
        base.crop_frequencies_average_psd_csd(
            pickled_base.frequencies[0], pickled_base.frequencies[-1]
        )

        # run dsc
        base.calculate_delta_sigma_cut(
            delta_sigma_cut=np.inf,
            alphas=[-5, 0, 3],
            fref=25,
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

    def test_save_point_estimate_spectra_npz(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        save_data_type = "npz"
        filename = "test/test_data/testing_save_function_of_baseline"
        pickled_base.save_point_estimate_spectra(save_data_type, filename)
        Loaded_npzfile = np.load(filename + ".npz")
        loaded_frequencies = Loaded_npzfile["frequencies"]
        test_frequencies = pickled_base.frequencies
        self.assertEqual(loaded_frequencies.tolist(), test_frequencies.tolist())

        loaded_point_estimate = Loaded_npzfile["point_estimate"]
        test_point_estimate = pickled_base.point_estimate
        self.assertEqual(loaded_point_estimate, test_point_estimate)

        loaded_sigma = Loaded_npzfile["sigma"]
        test_sigma = pickled_base.sigma
        self.assertEqual(loaded_sigma, test_sigma)

        loaded_point_estimate_spectrum = Loaded_npzfile["point_estimate_spectrum"]
        test_spectrum = pickled_base.point_estimate_spectrum
        self.assertEqual(
            loaded_point_estimate_spectrum.tolist(), test_spectrum.value.tolist()
        )

        loaded_sigma_spectrum = Loaded_npzfile["sigma_spectrum"]
        test_sigma_spectrum = pickled_base.sigma_spectrum
        self.assertEqual(
            loaded_sigma_spectrum.tolist(), test_sigma_spectrum.value.tolist()
        )

        loaded_spectrogram = Loaded_npzfile["point_estimate_spectrogram"]
        test_spectrogram = pickled_base.point_estimate_spectrogram
        self.assertEqual(loaded_spectrogram.tolist(), test_spectrogram.value.tolist())

        loaded_sigma_spectrogram = Loaded_npzfile["sigma_spectrogram"]
        test_sigma_spectrogram = pickled_base.sigma_spectrogram
        self.assertEqual(
            loaded_sigma_spectrogram.tolist(), test_sigma_spectrogram.value.tolist()
        )

        badGPStimes_loaded = Loaded_npzfile["badGPStimes"]
        test_badGPStimes = pickled_base.badGPStimes
        self.assertTrue(np.array_equal(badGPStimes_loaded, test_badGPStimes))

        loaded_delta_sigmas = Loaded_npzfile["delta_sigma_values"]
        test_delta_sigmas = pickled_base.delta_sigmas['values']
        self.assertTrue(np.array_equal(loaded_delta_sigmas, test_delta_sigmas))

    def test_save_psds_csd_npz(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        save_data_type = "npz"
        filename = "psds_csds_4138-5038"
        pickled_base.save_psds_csds(save_data_type, filename)
        Loading_npzfile = np.load(f"{filename}.npz")

        loaded_frequencies = Loading_npzfile["freqs"]
        test_frequencies = pickled_base.frequencies
        self.assertEqual(loaded_frequencies.tolist(), test_frequencies.tolist())

        loaded_avg_frequencies = Loading_npzfile["avg_freqs"]
        test_avg_frequencies = pickled_base.average_csd.frequencies
        self.assertEqual(
            loaded_avg_frequencies.tolist(), test_avg_frequencies.value.tolist()
        )

        loaded_naive_psd_1 = Loading_npzfile["psd_1"]
        test_naive_psd_1 = pickled_base.interferometer_1.psd_spectrogram
        self.assertEqual(loaded_naive_psd_1.tolist(), test_naive_psd_1.value.tolist())

        loaded_naive_psd_2 = Loading_npzfile["psd_2"]
        test_naive_psd_2 = pickled_base.interferometer_2.psd_spectrogram
        self.assertEqual(loaded_naive_psd_2.tolist(), test_naive_psd_2.value.tolist())

        loaded_avg_psd_1 = Loading_npzfile["avg_psd_1"]
        test_avg_psd_1 = pickled_base.interferometer_1.average_psd
        self.assertEqual(loaded_avg_psd_1.tolist(), test_avg_psd_1.value.tolist())

        loaded_avg_psd_2 = Loading_npzfile["avg_psd_2"]
        test_avg_psd_2 = pickled_base.interferometer_2.average_psd
        self.assertEqual(loaded_avg_psd_2.tolist(), test_avg_psd_2.value.tolist())

        loaded_csd = Loading_npzfile["csd"]
        test_csd = pickled_base.csd
        self.assertEqual(loaded_csd.tolist(), test_csd.value.tolist())

        loaded_avg_csd = Loading_npzfile["avg_csd"]
        test_avg_csd = pickled_base.average_csd
        self.assertEqual(loaded_avg_csd.tolist(), test_avg_csd.value.tolist())

    def test_save_point_estimate_spectra_pickle(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        save_data_type = "pickle"
        filename = "test/test_data/testing_save_function_of_baseline"
        pickled_base.save_point_estimate_spectra(save_data_type, filename)
        with open(filename + ".p", "rb") as f:
            Loaded_picklefile = pickle.load(f)

        loaded_frequencies = Loaded_picklefile["frequencies"]
        test_frequencies = pickled_base.frequencies
        self.assertEqual(loaded_frequencies.tolist(), test_frequencies.tolist())

        loaded_point_estimate = Loaded_picklefile["point_estimate"]
        test_point_estimate = pickled_base.point_estimate
        self.assertEqual(loaded_point_estimate, test_point_estimate)

        loaded_sigma = Loaded_picklefile["sigma"]
        test_sigma = pickled_base.sigma
        self.assertEqual(loaded_sigma, test_sigma)

        loaded_point_estimate_spectrum = Loaded_picklefile["point_estimate_spectrum"]
        test_spectrum = pickled_base.point_estimate_spectrum
        self.assertEqual(
            loaded_point_estimate_spectrum.value.tolist(), test_spectrum.value.tolist()
        )

        loaded_sigma_spectrum = Loaded_picklefile["sigma_spectrum"]
        test_sigma_spectrum = pickled_base.sigma_spectrum
        self.assertEqual(
            loaded_sigma_spectrum.value.tolist(), test_sigma_spectrum.value.tolist()
        )

        loaded_spectrogram = Loaded_picklefile["point_estimate_spectrogram"]
        test_spectrogram = pickled_base.point_estimate_spectrogram
        self.assertEqual(
            loaded_spectrogram.value.tolist(), test_spectrogram.value.tolist()
        )

        loaded_sigma_spectrogram = Loaded_picklefile["sigma_spectrogram"]
        test_sigma_spectrogram = pickled_base.sigma_spectrogram
        self.assertEqual(
            loaded_sigma_spectrogram.value.tolist(),
            test_sigma_spectrogram.value.tolist(),
        )

        badGPStimes_loaded = Loaded_picklefile["badGPStimes"]
        test_badGPStimes = pickled_base.badGPStimes
        self.assertTrue(np.array_equal(badGPStimes_loaded, test_badGPStimes))

        loaded_delta_sigmas = dict(Loaded_picklefile["delta_sigmas"])['values']
        test_delta_sigmas = pickled_base.delta_sigmas['values']
        self.assertTrue(np.array_equal(loaded_delta_sigmas, test_delta_sigmas))

    def test_save_psds_csd_pickle(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        save_data_type = "pickle"
        filename = "psds_csds_4138-5038"
        pickled_base.save_psds_csds(save_data_type, filename)
        with open(f"{filename}.p", "rb") as f:
            Loading_picklefile = pickle.load(f)

        loaded_frequencies = Loading_picklefile["freqs"]
        test_frequencies = pickled_base.frequencies
        self.assertEqual(loaded_frequencies.tolist(), test_frequencies.tolist())

        loaded_avg_frequencies = Loading_picklefile["avg_freqs"]
        test_avg_frequencies = pickled_base.average_csd.frequencies
        self.assertEqual(
            loaded_avg_frequencies.tolist(), test_avg_frequencies.value.tolist()
        )

        loaded_naive_psd_1 = Loading_picklefile["psd_1"]
        test_naive_psd_1 = pickled_base.interferometer_1.psd_spectrogram
        self.assertEqual(
            loaded_naive_psd_1.value.tolist(), test_naive_psd_1.value.tolist()
        )

        loaded_naive_psd_2 = Loading_picklefile["psd_2"]
        test_naive_psd_2 = pickled_base.interferometer_2.psd_spectrogram
        self.assertEqual(
            loaded_naive_psd_2.value.tolist(), test_naive_psd_2.value.tolist()
        )

        loaded_avg_psd_1 = Loading_picklefile["avg_psd_1"]
        test_avg_psd_1 = pickled_base.interferometer_1.average_psd
        self.assertEqual(loaded_avg_psd_1.value.tolist(), test_avg_psd_1.value.tolist())

        loaded_avg_psd_2 = Loading_picklefile["avg_psd_2"]
        test_avg_psd_2 = pickled_base.interferometer_2.average_psd
        self.assertEqual(loaded_avg_psd_2.value.tolist(), test_avg_psd_2.value.tolist())

        loaded_csd = Loading_picklefile["csd"]
        test_csd = pickled_base.csd
        self.assertEqual(loaded_csd.value.tolist(), test_csd.value.tolist())

        loaded_avg_csd = Loading_picklefile["avg_csd"]
        test_avg_csd = pickled_base.average_csd
        self.assertEqual(loaded_avg_csd.value.tolist(), test_avg_csd.value.tolist())

    def test_save_point_estimate_spectra_json(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        save_data_type = "json"
        filename = "test/test_data/testing_save_function_of_baseline"
        pickled_base.save_point_estimate_spectra(save_data_type, filename)
        with open(filename + ".json", "r") as j:
            Loaded_jsonfile = json.loads(j.read())

        loaded_frequencies = Loaded_jsonfile["frequencies"]
        test_frequencies = pickled_base.frequencies
        self.assertEqual(loaded_frequencies, test_frequencies.tolist())

        loaded_point_estimate = Loaded_jsonfile["point_estimate"]
        test_point_estimate = pickled_base.point_estimate
        self.assertEqual(loaded_point_estimate, test_point_estimate)

        loaded_sigma = Loaded_jsonfile["sigma"]
        test_sigma = pickled_base.sigma
        self.assertEqual(loaded_sigma, test_sigma)

        loaded_point_estimate_spectrum = Loaded_jsonfile["point_estimate_spectrum_real"]
        test_spectrum = pickled_base.point_estimate_spectrum
        self.assertEqual(loaded_point_estimate_spectrum, np.real(test_spectrum.value).tolist())

        loaded_sigma_spectrum = Loaded_jsonfile["sigma_spectrum"]
        test_sigma_spectrum = pickled_base.sigma_spectrum
        self.assertEqual(loaded_sigma_spectrum, test_sigma_spectrum.value.tolist())

        loaded_spectrogram = Loaded_jsonfile["point_estimate_spectrogram_real"]
        test_spectrogram = pickled_base.point_estimate_spectrogram
        self.assertEqual(loaded_spectrogram, np.real(test_spectrogram.value).tolist())

        loaded_sigma_spectrogram = Loaded_jsonfile["sigma_spectrogram"]
        test_sigma_spectrogram = pickled_base.sigma_spectrogram
        self.assertEqual(
            loaded_sigma_spectrogram, test_sigma_spectrogram.value.tolist()
        )

        badGPStimes_loaded = Loaded_jsonfile["badGPStimes"]
        test_badGPStimes = pickled_base.badGPStimes
        self.assertTrue(np.array_equal(badGPStimes_loaded, test_badGPStimes))

        loaded_delta_sigmas = np.array(Loaded_jsonfile["delta_sigma_values"])
        test_delta_sigmas = pickled_base.delta_sigmas['values']
        self.assertTrue(np.array_equal(loaded_delta_sigmas, test_delta_sigmas))

    def test_save_psds_csd_json(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        save_data_type = "json"
        filename = "psds_csds_4138-5038"
        pickled_base.save_psds_csds(save_data_type, filename)
        with open(f"{filename}.json", "r") as j:
            Loading_jsonfile = json.loads(j.read())

        loaded_frequencies = Loading_jsonfile["frequencies"]
        test_frequencies = pickled_base.frequencies
        self.assertEqual(loaded_frequencies, test_frequencies.tolist())

        loaded_avg_frequencies = Loading_jsonfile["avg_frequencies"]
        test_avg_frequencies = pickled_base.average_csd.frequencies
        self.assertEqual(loaded_avg_frequencies, test_avg_frequencies.value.tolist())

        loaded_naive_psd_1 = Loading_jsonfile["psd_1"]
        test_naive_psd_1 = pickled_base.interferometer_1.psd_spectrogram
        self.assertEqual(loaded_naive_psd_1, test_naive_psd_1.value.tolist())

        loaded_naive_psd_2 = Loading_jsonfile["psd_2"]
        test_naive_psd_2 = pickled_base.interferometer_2.psd_spectrogram
        self.assertEqual(loaded_naive_psd_2, test_naive_psd_2.value.tolist())

        loaded_avg_psd_1 = Loading_jsonfile["avg_psd_1"]
        test_avg_psd_1 = pickled_base.interferometer_1.average_psd
        self.assertEqual(loaded_avg_psd_1, test_avg_psd_1.value.tolist())

        loaded_avg_psd_2 = Loading_jsonfile["avg_psd_2"]
        test_avg_psd_2 = pickled_base.interferometer_2.average_psd
        self.assertEqual(loaded_avg_psd_2, test_avg_psd_2.value.tolist())

        loaded_csd_real = Loading_jsonfile["csd_real"]
        loaded_csd_imag = Loading_jsonfile["csd_imag"]
        loaded_csd = [
            complex(real, imag)
            for row, row_2 in zip(loaded_csd_real, loaded_csd_imag)
            for real, imag in zip(row, row_2)
        ]
        test_csd = pickled_base.csd
        # self.assertEqual(loaded_csd, test_csd.value.tolist())

        loaded_avg_csd_real = Loading_jsonfile["avg_csd_real"]
        loaded_avg_csd_imag = Loading_jsonfile["avg_csd_imag"]
        loaded_avg_csd = [
            complex(real, imag)
            for row, row_2 in zip(loaded_avg_csd_real, loaded_avg_csd_imag)
            for real, imag in zip(row, row_2)
        ]
        test_avg_csd = pickled_base.average_csd
        # self.assertEqual(loaded_avg_csd, test_avg_csd.value.tolist())

    def test_save_point_estimate_spectra_hdf5(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        save_data_type = "hdf5"
        filename = "test/test_data/testing_save_function_of_baseline"
        pickled_base.save_point_estimate_spectra(save_data_type, filename)

        hf = h5py.File(f"{filename}.h5", "r")
        loaded_freqs = np.array(hf.get("freqs"))
        test_frequencies = pickled_base.frequencies
        self.assertEqual(loaded_freqs.tolist(), test_frequencies.tolist())

        loaded_point_estimate = hf.get("point_estimate")
        test_point_estimate = pickled_base.point_estimate
        self.assertEqual(loaded_point_estimate, test_point_estimate)

        loaded_sigma = hf.get("sigma")
        test_sigma = pickled_base.sigma
        self.assertEqual(loaded_sigma, test_sigma)

        loaded_point_estimate_spectrum = list(hf.get("point_estimate_spectrum"))
        test_point_estimate_spectrum = pickled_base.point_estimate_spectrum
        self.assertEqual(
            loaded_point_estimate_spectrum, test_point_estimate_spectrum.value.tolist()
        )

        loaded_sigma_spectrum = list(hf.get("sigma_spectrum"))
        test_sigma_spectrum = pickled_base.sigma_spectrum
        self.assertEqual(loaded_sigma_spectrum, test_sigma_spectrum.value.tolist())

        loaded_point_estimate_spectrogram = list(hf.get("point_estimate_spectrogram"))
        test_point_estimate_spectrogram = pickled_base.point_estimate_spectrogram
        for ele, ele_2 in zip(
            loaded_point_estimate_spectrogram,
            test_point_estimate_spectrogram.value.tolist(),
        ):
            self.assertEqual(list(ele), ele_2)
        # self.assertListEqual(loaded_point_estimate_spectrogram, test_point_estimate_spectrogram.value.tolist())

        loaded_sigma_spectrogram = list(hf.get("sigma_spectrogram"))
        test_sigma_spectrogram = pickled_base.sigma_spectrogram
        for row, row_2 in zip(
            loaded_sigma_spectrogram, test_sigma_spectrogram.value.tolist()
        ):
            self.assertEqual(list(row), row_2)
        # self.assertEqual(loaded_sigma_spectrogram, test_sigma_spectrogram.value.tolist())

        loaded_badGPStimes = list(hf.get("badGPStimes"))
        test_badGPStimes = pickled_base.badGPStimes
        self.assertTrue(np.array_equal(loaded_badGPStimes, test_badGPStimes))

        loaded_delta_sigmas_group = hf.get("delta_sigmas")
        loaded_delta_sigmas = loaded_delta_sigmas_group['delta_sigma_values']
        test_delta_sigmas = pickled_base.delta_sigmas['values']
        self.assertTrue(np.array_equal(loaded_delta_sigmas, test_delta_sigmas))

    def test_save_psds_csd_hdf5(self):
        pickled_base = baseline.Baseline.load_from_pickle(
            "test/test_data/H1L1_1247644138-1247645038.pickle"
        )
        save_data_type = "hdf5"
        filename = "psds_csds_4138-5038"
        pickled_base.save_psds_csds(save_data_type, filename)
        hf = h5py.File(f"{filename}.h5", "r")

        loaded_frequencies = list(hf.get("freqs"))
        test_frequencies = pickled_base.frequencies
        self.assertEqual(loaded_frequencies, test_frequencies.tolist())

        loaded_avg_frequencies = list(hf.get("avg_freqs"))
        test_avg_frequencies = pickled_base.average_csd.frequencies
        self.assertEqual(loaded_avg_frequencies, test_avg_frequencies.value.tolist())

        loaded_naive_psd_1_group = hf.get("psds_group/psd_1")

        loaded_naive_psd_1 = list(loaded_naive_psd_1_group["psd_1"])
        test_naive_psd_1 = pickled_base.interferometer_1.psd_spectrogram
        list_test_naive_psd_1 = test_naive_psd_1.value.tolist()
        for row, row_2 in zip(loaded_naive_psd_1, list_test_naive_psd_1):
            self.assertEqual(list(row), row_2)
        # self.assertEqual(loaded_naive_psd_1, list_test_naive_psd_1)

        loaded_naive_psd_1_times = list(loaded_naive_psd_1_group["psd_1_times"])
        test_psd_1_times = pickled_base.interferometer_1.psd_spectrogram.times
        self.assertListEqual(loaded_naive_psd_1_times, test_psd_1_times.value.tolist())

        loaded_naive_psd_2_group = hf.get("psds_group/psd_2")

        loaded_naive_psd_2 = list(loaded_naive_psd_2_group["psd_2"])
        test_naive_psd_2 = pickled_base.interferometer_2.psd_spectrogram
        for row, row_2 in zip(loaded_naive_psd_2, test_naive_psd_2.value.tolist()):
            self.assertEqual(list(row), row_2)
        # self.assertListEqual(loaded_naive_psd_2, test_naive_psd_2.value.tolist())

        loaded_naive_psd_2_times = list(loaded_naive_psd_2_group["psd_2_times"])
        test_psd_2_times = pickled_base.interferometer_2.psd_spectrogram.times
        self.assertEqual(loaded_naive_psd_2_times, test_psd_2_times.value.tolist())

        loaded_avg_psd_1_group = hf.get("avg_psds_group/avg_psd_1")

        loaded_avg_psd_1 = list(loaded_avg_psd_1_group["avg_psd_1"])
        test_avg_psd_1 = pickled_base.interferometer_1.average_psd
        for array, list_1 in zip(loaded_avg_psd_1, test_avg_psd_1.value.tolist()):
            self.assertEqual(list(array), list_1)
        # self.assertEqual(loaded_avg_psd_1, test_avg_psd_1.value.tolist())

        loaded_avg_psd_1_times = list(loaded_avg_psd_1_group["avg_psd_1_times"])
        test_psd_1_times = pickled_base.interferometer_1.average_psd.times
        self.assertEqual(loaded_avg_psd_1_times, test_psd_1_times.value.tolist())

        loaded_avg_psd_2_group = hf.get("avg_psds_group/avg_psd_2")

        loaded_avg_psd_2 = list(loaded_avg_psd_2_group["avg_psd_2"])
        test_avg_psd_2 = pickled_base.interferometer_2.average_psd
        for array, list_2 in zip(loaded_avg_psd_2, test_avg_psd_2.value.tolist()):
            self.assertEqual(list(array), list_2)
        # self.assertEqual(loaded_avg_psd_2, test_avg_psd_2.value.tolist())

        loaded_avg_psd_2_times = list(loaded_avg_psd_2_group["avg_psd_2_times"])
        test_psd_2_times = pickled_base.interferometer_2.average_psd.times
        self.assertEqual(loaded_avg_psd_2_times, test_psd_2_times.value.tolist())

        loaded_csd_group = hf.get("csd_group")

        loaded_csd = list(loaded_csd_group["csd"])
        test_csd = pickled_base.csd
        for ele, ele_2 in zip(loaded_csd, test_csd.value.tolist()):
            self.assertEqual(list(ele), ele_2)
        # self.assertEqual(loaded_csd, test_csd.value.tolist())

        loaded_csd_times = list(loaded_csd_group["csd_times"])
        test_csd_times = pickled_base.csd.times
        self.assertEqual(loaded_csd_times, test_csd_times.value.tolist())

        loaded_avg_csd_group = hf.get("avg_csd_group")

        loaded_avg_csd = list(loaded_avg_csd_group["avg_csd"])
        test_avg_csd = pickled_base.average_csd
        for row, row_2 in zip(loaded_avg_csd, test_avg_csd.value.tolist()):
            self.assertEqual(list(row), row_2)
        # self.assertEqual(loaded_avg_csd, test_avg_csd.value.tolist())

        loaded_avg_csd_times = list(loaded_avg_csd_group["avg_csd_times"])
        test_avg_csd_times = pickled_base.average_csd.times
        self.assertEqual(loaded_avg_csd_times, test_avg_csd_times.value.tolist())


if __name__ == "__main__":
    unittest.main()
