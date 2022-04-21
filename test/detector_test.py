import pickle
import unittest

import numpy as np

from pygwb import detector, parameters


class TestInterferometer(unittest.TestCase):
    def setUp(self):
        self.ifo = "H1"
        param_file = "./test/test_data/parameters_detector_test.ini"
        self.parameters = parameters.Parameters()
        self.parameters.update_from_file(param_file)
        self.kwargs = {
            param: getattr(self.parameters, param)
            for param in [
                "channel",
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
                "N_average_segments_welch_psd",
                "time_shift",
            ]
        }
        self.kwargs["local_data_path"] = ""
        with open("./test/test_data/detector_testdata_H1.pickle", "rb") as pkl:
            self.testdata = pickle.load(pkl)

    def tearDown(self):
        del self.ifo
        del self.parameters
        del self.kwargs

    def test_from_parameters(self):
        ifo = detector.Interferometer.from_parameters(self.ifo, self.parameters)
        self.assertTrue(ifo.name, self.ifo)
        self.assertTrue(ifo.timeseries._name,"Strain")
        self.assertTrue(ifo.timeseries._channel, f'{self.ifo}:{self.kwargs["channel"]}')
        self.assertTrue(ifo.timeseries._x0.value, f'{self.kwargs["t0"]}')
        self.assertTrue(ifo.timeseries._dx.value, f'{self.kwargs["tf"]}-{self.kwargs["t0"]}')

    def test_get_empty_interferometer(self):
        ifo = detector.Interferometer.get_empty_interferometer(self.ifo)
        self.assertTrue(ifo.name, self.ifo)
        self.assertTrue(ifo.length, 4.0)
        self.assertTrue(ifo.latitude, 46.45514666666667)
        self.assertTrue(ifo.longitude, -119.4076571388889)

    def test_set_timeseries_from_channel_name(self):
        ifo = detector.Interferometer.get_empty_interferometer(self.ifo)
        ifo.set_timeseries_from_channel_name(**self.kwargs)
        self.assertTrue(
            np.all(
                abs(ifo.timeseries.value - self.testdata["filtered_timeseries"].value)
                < 1e-10
            ),
            True,
        )

    def test_set_timeseries_from_timeseries_array(self):
        timeseries_array = self.testdata["original_timeseries"].value
        sample_rate = self.testdata["original_timeseries"].sample_rate.value
        ifo = detector.Interferometer.get_empty_interferometer(self.ifo)
        ifo.set_timeseries_from_timeseries_array(
            timeseries_array, sample_rate, **self.kwargs
        )
        self.assertTrue(
            np.all(
                abs(ifo.timeseries.value - self.testdata["filtered_timeseries"].value)
                < 1e-10
            ),
            True,
        )

    def test_gwpy_timeseries_spectrogram_average_psd(self):
        gwpy_timeseries = self.testdata["original_timeseries"]
        ifo = detector.Interferometer.get_empty_interferometer(self.ifo)
        ifo.set_timeseries_from_gwpy_timeseries(
            gwpy_timeseries=gwpy_timeseries, **self.kwargs
        )
        self.assertTrue(
            np.all(
                abs(ifo.timeseries.value - self.testdata["filtered_timeseries"].value)
                < 1e-10
            ),
            True,
        )
        ifo.set_psd_spectrogram(
            frequency_resolution=self.parameters.frequency_resolution,
            overlap_factor=self.parameters.overlap_factor,
            window_fftgram=self.parameters.window_fftgram,
        )
        self.assertTrue(
            np.all(
                abs(ifo.psd_spectrogram.value - self.testdata["psd_spectrogram"].value)
                < 1e-10
            ),
            True,
        )
        ifo.set_average_psd(self.parameters.N_average_segments_welch_psd)
        self.assertTrue(
            np.all(
                abs(ifo.average_psd.value - self.testdata["average_psd"].value) < 1e-10
            ),
            True,
        )

    def test_gwpy_timeseries_gating(self):
        gwpy_timeseries = self.testdata["original_timeseries"]
        ifo = detector.Interferometer.get_empty_interferometer(self.ifo)
        ifo.set_timeseries_from_gwpy_timeseries(
            gwpy_timeseries=gwpy_timeseries, **self.kwargs
        )
        gate_tzero = 1.0
        gate_tpad = 0.5
        gate_threshold = 50.0
        cluster_window = 0.5
        gate_whiten = True
        _known_gate = 1247644447 
        ifo.gate_data_apply(
            gate_tzero = gate_tzero, gate_tpad = gate_tpad,
            gate_threshold = gate_threshold, cluster_window = cluster_window,
            gate_whiten = gate_whiten,
        )
        self.assertTrue(max(ifo.timeseries.whiten().value) < gate_threshold, True)
        self.assertTrue((_known_gate in ifo.gates), True)
        self.assertTrue(abs(ifo.gates), 2*gate_tzero)
        self.assertTrue(ifo.gate_pad, gate_tpad)

if __name__ == "__main__":
    unittest.main()
