import copy
import os
import unittest

import bilby
import numpy as np

from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

import pygwb
from pygwb import simulator


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.interferometer_1 = bilby.gw.detector.get_empty_interferometer("H1")
        self.interferometer_2 = bilby.gw.detector.get_empty_interferometer("L1")
        self.interferometer_3 = bilby.gw.detector.get_empty_interferometer("V1")
        self.intensity_GW = FrequencySeries.read('intensity_GW_file.txt')
        self.N_segments = 10
        self.duration = 60.0
        self.sampling_frequency = 2048.0

    def tearDown(self):
        del self.interferometer_1
        del self.interferometer_2
        del self.interferometer_3
        del self.intensity_GW
        del self.N_segments
        del self.duration
        del self.sampling_frequency

    def test_get_frequencies(self):
        ifo_list = [self.interferometer_1, self.interferometer_2]
        for ifo in ifo_list:
            ifo.duration = self.duration
            ifo.sampling_frequency = self.sampling_frequency

        simulator_1 = simulator.Simulator(
            ifo_list, self.intensity_GW, self.N_segments, self.duration, self.sampling_frequency
        )
        frequencies = simulator_1.frequencies

        maximum_frequency = self.sampling_frequency//2
        frequency_length = maximum_frequency//self.duration

        self.assertTrue(np.max(frequencies), maximum_frequency)
        self.assertTrue(len(frequencies), frequency_length)


    def test_get_noise_PSD_array(self):
        ifo_list = [self.interferometer_1, self.interferometer_2]
        for ifo in ifo_list:
            ifo.duration = self.duration
            ifo.sampling_frequency = self.sampling_frequency
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(ifo.frequency_array, np.nan_to_num(ifo.power_spectral_density_array, posinf=1.e-41))

        simulator_1 = simulator.Simulator(
            ifo_list, self.intensity_GW, self.N_segments, self.duration, self.sampling_frequency
        )
        noise_PSD_array_1 = simulator_1.noise_PSD_array

        self.assertTrue(len(noise_PSD_array_1), len(ifo_list))
        self.assertTrue(isinstance(noise_PSD_array_1, np.ndarray))
        self.assertTrue([isinstance(array_1, np.ndarray) for array_1 in noise_PSD_array_1])
        

    def test_baselines(self):
        ifo_list = [self.interferometer_1, self.interferometer_2, self.interferometer_3]
        for ifo in ifo_list:
            ifo.duration = self.duration
            ifo.sampling_frequency = self.sampling_frequency
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(ifo.frequency_array, np.nan_to_num(ifo.power_spectral_density_array, posinf=1.e-41))
        simulator_1 = simulator.Simulator(
            ifo_list, self.intensity_GW, self.N_segments, self.duration, self.sampling_frequency
        )
        baselines = simulator_1.baselines
        Nd = len(ifo_list)

        self.assertTrue(len(baselines), Nd*(Nd-1)//2)
        self.assertTrue(isinstance(baselines, list))
        self.assertTrue([isinstance(base, pygwb.baseline.Baseline) for base in baselines])


    def test_get_orf(self):
        ifo_list = [self.interferometer_1, self.interferometer_2, self.interferometer_3]
        for ifo in ifo_list:
            ifo.duration = self.duration
            ifo.sampling_frequency = self.sampling_frequency
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(ifo.frequency_array, np.nan_to_num(ifo.power_spectral_density_array, posinf=1.e-41))
        simulator_1 = simulator.Simulator(
            ifo_list, self.intensity_GW, self.N_segments, self.duration, self.sampling_frequency
        )
        baselines = simulator_1.baselines
        Nb = len(baselines)
        overlap_reduction_function = simulator_1.orf

        self.assertTrue(len(overlap_reduction_function), Nb)
        self.assertTrue(isinstance(overlap_reduction_function, list))
        self.assertTrue([isinstance(orf, np.ndarray) for orf in overlap_reduction_function])

    
        simulator_1 = simulator.Simulator(
            [self.interferometer_1, self.interferometer_2], self.intensity_GW, self.N_segments, self.duration, self.sampling_frequency
        )
        self.assertTrue(len(simulator_1.intensity_GW), len(simulator_1.frequencies))


    def test_generate_data(self):
        ifo_list = [self.interferometer_1, self.interferometer_2]
        for ifo in ifo_list:
            ifo.duration = self.duration
            ifo.sampling_frequency = self.sampling_frequency
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(ifo.frequency_array, np.nan_to_num(ifo.power_spectral_density_array, posinf=1.e-41))
        simulator_1 = simulator.Simulator(
            ifo_list, self.intensity_GW, self.N_segments, self.duration, self.sampling_frequency
        )
        data = simulator_1.generate_data()
        self.assertTrue([isinstance(dat, TimeSeries) for dat in data])


if __name__ == "__main__":
    unittest.main()
