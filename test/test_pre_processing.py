import unittest

import numpy as np
from gwpy import timeseries

from pygwb import pre_processing


class Test(unittest.TestCase):
    def setUp(self) -> None:
        # Initialize array
        channel = "L1:GDS-CALIB_STRAIN"
        t0 = 1238183936
        tf = t0 + 500
        IFO = "H1"
        segment_duration = 192
        number_cropped_seconds = 2
        data_type = "public"
        data_start_time = pre_processing.set_start_time(
            t0, tf, number_cropped_seconds, segment_duration
        )
        self.timeseries_data = pre_processing.read_data(
            IFO, data_type, channel, data_start_time - number_cropped_seconds, tf
        )
        self.timeseries_array = np.array(self.timeseries_data.value)
        self.a = timeseries.TimeSeries(
            self.timeseries_array,
            t0=data_start_time,
            sample_rate=1.0 / self.timeseries_data.dt,
        )
        return None

    def tearDown(self) -> None:

        pass

    def test_pre_processing(self):
        """
        Test1: we make sure the output of each pre_processing function has 3 psds
        Test2: we make sure the output of each pre_processing function has a sampling frequency of 1/192.0 Hz
        Test3: we test the different outputs of set_start_time
        """
        channel = "L1:'GWOSC-16KHZ_R1_STRAIN'"
        t0 = 1238183936
        tf = t0 + 500
        IFO = "H1"
        data_type = "public"
        new_sample_rate = 4096
        cutoff_frequency = 11
        segment_duration = 192
        print(len(self.timeseries_data))
        print(len(self.timeseries_array))

        timeseries_output1 = pre_processing.preprocessing_data_channel_name(
            IFO=IFO,
            t0=t0,
            tf=tf,
            data_type=data_type,
            channel=channel,
            new_sample_rate=new_sample_rate,
            cutoff_frequency=cutoff_frequency,
            segment_duration=segment_duration,
            number_cropped_seconds=2,
            window_downsampling="hamming",
            ftype="fir",
        )

        self.assertEqual(len(timeseries_output1), 1802240)
        self.assertEqual(timeseries_output1.sample_rate.value, 4096.0)

        timeseries_output2 = pre_processing.preprocessing_data_timeseries_array(
            t0=t0,
            tf=tf,
            IFO=IFO,
            array=self.timeseries_array,
            new_sample_rate=new_sample_rate,
            cutoff_frequency=cutoff_frequency,
            segment_duration=segment_duration,
            sample_rate=1.0 / self.timeseries_data.dt,
            number_cropped_seconds=2,
            window_downsampling="hamming",
            ftype="fir",
        )

        self.assertEqual(len(timeseries_output2), 1802240)
        self.assertEqual(timeseries_output2.sample_rate.value, 4096.0)

        timeseries_output3 = pre_processing.preprocessing_data_gwpy_timeseries(
            IFO=IFO,
            gwpy_timeseries=self.a,
            new_sample_rate=new_sample_rate,
            cutoff_frequency=cutoff_frequency,
            number_cropped_seconds=2,
            window_downsampling="hamming",
            ftype="fir",
        )

        self.assertEqual(len(timeseries_output3), 1802240)
        self.assertEqual(timeseries_output1.sample_rate.value, 4096.0)

        self.assertEqual(
            pre_processing.set_start_time(t0, tf, 2, segment_duration, False),
            1238183994.0,
        )
        self.assertEqual(
            pre_processing.set_start_time(t0, tf, 2, segment_duration, True),
            1238184444.0,
        )
