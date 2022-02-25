import pickle
import unittest

import numpy as np

from pygwb import detector, parameters, pre_processing


class TestInterferometer(unittest.TestCase):
    def setUp(self):
        self.ifo = "H1"
        param_file = "./pygwb/parameters.ini"
        self.parameters = parameters.Parameters.from_file(param_file)
        self.kwargs={param : getattr(self.parameters, param)
             for param in ["channel",
                           "t0",
                           "tf",
                           "segment_duration",
                           "number_cropped_seconds",
                           "data_type",
                           "cutoff_frequency",
                           "new_sample_rate",
                           "window_downsampling",
                           "ftype",
                           "frequency_resolution",
                           "overlap_factor",
                           "overlap_factor_welch_psd",
                           "N_average_segments_welch_psd"
                          ]}
        with open("./test/detector_testdata_H1.pickle", "rb") as pkl:
            self.testdata = pickle.load(pkl)

    def tearDown(self):
        del self.ifo
        del self.parameters
        del self.kwargs

    def test_get_empty_interferometer(self):
        ifo = detector.Interferometer.get_empty_interferometer(self.ifo)
        self.assertTrue(ifo.name, self.ifo)
        return ifo

    def test_set_timeseries_from_channel_name(self):
        ifo = self.test_get_empty_interferometer()
        ifo.set_timeseries_from_channel_name(**self.kwargs)
        self.assertTrue(np.all(abs(ifo.timeseries.value - self.testdata["filtered_timeseries"].value) < 1e-10), True)

    def test_set_timeseries_from_timeseries_array(self):
        timeseries_array = self.testdata["original_timeseries"].value
        sample_rate = self.testdata["original_timeseries"].sample_rate.value
        ifo = self.test_get_empty_interferometer()
        ifo.set_timeseries_from_timeseries_array(timeseries_array, sample_rate, **self.kwargs)
        self.assertTrue(np.all(abs(ifo.timeseries.value - self.testdata["filtered_timeseries"].value) < 1e-10), True)

    def test_gwpy_timeseries_spectrogram_average_psd(self):
        gwpy_timeseries = self.testdata["original_timeseries"]
        ifo = self.test_get_empty_interferometer()
        ifo.set_timeseries_from_gwpy_timeseries(gwpy_timeseries=gwpy_timeseries, **self.kwargs)
        self.assertTrue(np.all(abs(ifo.timeseries.value - self.testdata["filtered_timeseries"].value) < 1e-10), True)
        ifo.set_psd_spectrogram(frequency_resolution=self.parameters.frequency_resolution,
                               overlap_factor=self.parameters.overlap_factor,
                               overlap_factor_welch_psd=self.parameters.overlap_factor_welch_psd,
                               window_fftgram=self.parameters.window_fftgram
                               )
        self.assertTrue(np.all(abs(ifo.psd_spectrogram.value - self.testdata["psd_spectrogram"].value) < 1e-10), True)
        ifo.set_average_psd(self.parameters.N_average_segments_welch_psd)
        self.assertTrue(np.all(abs(ifo.average_psd.value - self.testdata["average_psd"].value) < 1e-10), True)


if __name__ == "__main__":
    unittest.main()
