import pickle
import unittest
from random import sample

import numpy as np
from gwpy import timeseries

from pygwb import preprocessing


class Test(unittest.TestCase):
    def setUp(self) -> None:
        # Initialize array
        self.channel = "L1:GDS-CALIB_STRAIN"
        self.t0 = 1238183936
        self.tf = self.t0 + 500
        self.IFO = "H1"
        self.segment_duration = 192
        self.number_cropped_seconds = 2
        self.data_type = "public"
        self.time_shift = 0
        self.sample_rate = 4096
        self.local_data_path = './test/test_data/data_gwf_preproc_testing.gwf'
        self.cutoff_frequency = 11

        data_start_time = preprocessing.set_start_time(
            self.t0, self.tf, self.number_cropped_seconds, self.segment_duration
        )
        self.data_start_time = data_start_time
        self.timeseries_data = timeseries.TimeSeries(np.random.normal(0, 1, int((self.tf-self.t0)*self.sample_rate)), t0=data_start_time-self.number_cropped_seconds, dt=1 / self.sample_rate, channel="my_channel")

        self.timeseries_array = np.array(self.timeseries_data.value)

        self.a = timeseries.TimeSeries(
            self.timeseries_array,
            t0=data_start_time,
            sample_rate=1.0 / self.timeseries_data.dt,
        )

        self.timeseries_data.write(self.local_data_path)

        return None

    def tearDown(self) -> None:

        pass

    def test_preprocessing(self):
        """
        Test1: we make sure the output of each preprocessing function has 3 psds
        Test2: we make sure the output of each preprocessing function has a sampling frequency of 1/192.0 Hz
        Test3: we test the different outputs of set_start_time
        """

        timeseries_output1 = preprocessing.preprocessing_data_channel_name(
            IFO=self.IFO,
            t0=self.t0,
            tf=self.tf,
            data_type="local",
            channel="my_channel",
            new_sample_rate=self.sample_rate,
            cutoff_frequency=self.cutoff_frequency,
            segment_duration=self.segment_duration,
            number_cropped_seconds=2,
            window_downsampling="hamming",
            ftype="fir",
            time_shift=self.time_shift,
            local_data_path=self.local_data_path,
        )

        self.assertEqual(len(timeseries_output1), 1802240)
        self.assertEqual(timeseries_output1.sample_rate.value, 4096.0)

        timeseries_output2 = preprocessing.preprocessing_data_timeseries_array(
            t0=self.t0,
            tf=self.tf,
            IFO=self.IFO,
            array=self.timeseries_array,
            new_sample_rate=self.sample_rate,
            cutoff_frequency=self.cutoff_frequency,
            segment_duration=self.segment_duration,
            sample_rate=1.0 / self.timeseries_data.dt,
            number_cropped_seconds=2,
            window_downsampling="hamming",
            ftype="fir",
            time_shift=self.time_shift,
        )
        self.assertEqual(len(timeseries_output2), 2031616)
        self.assertEqual(timeseries_output2.sample_rate.value, 4096.0) 

        timeseries_output3 = preprocessing.preprocessing_data_gwpy_timeseries(
            IFO=self.IFO,
            gwpy_timeseries=self.timeseries_data,
            new_sample_rate=self.sample_rate,
            cutoff_frequency=self.cutoff_frequency,
            number_cropped_seconds=2,
            window_downsampling="hamming",
            ftype="fir",
            time_shift=self.time_shift,
        )

        self.assertEqual(len(timeseries_output3), 2031616)
        self.assertEqual(timeseries_output3.sample_rate.value, 4096.0)

        self.assertEqual(
            preprocessing.set_start_time(self.t0, self.tf, 2, self.segment_duration, False),
            1238183994.0,
        )
        self.assertEqual(
            preprocessing.set_start_time(self.t0, self.tf, 2, self.segment_duration, True),
            1238184444.0,
        )
        time_shifted_data = preprocessing.shift_timeseries(time_series_data = self.timeseries_data, time_shift = 1)
        self.assertEqual(
            self.timeseries_data.value[0],
            time_shifted_data.value[1],
        )
        


if __name__ == "__main__":
    unittest.main()
