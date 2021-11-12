import unittest

import numpy as np
from gwpy import timeseries

from pygwb import pre_processing, spectral, delta_sigma_cut


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

        # Analysis parameters
        IFO1 = "H1"
        IFO2 = "L1"
        t0 = 1247644138  # start GPS time
        tf = 1247645038  # end GPS time
        data_type = "public"  # private -> running on LIGO data grid
        channel_suffix = "GWOSC-16KHZ_R1_STRAIN"  # detector name will be added later
        new_sample_rate = 4096  # sampled rate after resampling
        cutoff_frequency = 11  # high pass filter cutoff frequency
        segment_duration = 192  # also fftlength in pre-processing
        frequency_resolution = 1.0 / 32  # final frequency resolution of CSD and PSD
        overlap = segment_duration / 2  # overlapping between segments
        fftlength = 192
        dsc = 0.2
        alphas = [-5, 0, 3]
        notch_file = "test/test_data/Official_O3_HL_notchlist.txt"
        lines = read_notch_list(notch_file)
        badGPStimes_matlab = [1247644396, 1247644492, 1247644588]

        ifo1_filtered = pre_processing.preprocessing_data_channel_name(
            IFO=IFO1,
            t0=t0,
            tf=tf,
            data_type=data_type,
            channel=IFO1 + ":" + channel_suffix,
            new_sample_rate=new_sample_rate,
            cutoff_frequency=cutoff_frequency,
            segment_duration=segment_duration,
            number_cropped_seconds=2,
            window_downsampling="hamming",
            ftype="fir",
        )

        ifo2_filtered = pre_processing.preprocessing_data_channel_name(
            IFO=IFO2,
            t0=t0,
            tf=tf,
            data_type=data_type,
            channel=IFO2 + ":" + channel_suffix,
            new_sample_rate=new_sample_rate,
            cutoff_frequency=cutoff_frequency,
            segment_duration=segment_duration,
            number_cropped_seconds=2,
            window_downsampling="hamming",
            ftype="fir",
        )

        naive_psd_1 = spectral.power_spectral_density(
            ifo1_filtered, segment_duration, frequency_resolution, do_overlap=True
        )
        naive_psd_2 = spectral.power_spectral_density(
            ifo2_filtered, segment_duration, frequency_resolution, do_overlap=True
        )

        # adjacent averated PSDs (detector 1) for each possible CSD
        avg_psd_1 = spectral.before_after_average(
            naive_psd_1, segment_duration, segment_duration
        )

        # adjacent averated PSDs (detector 2) for each possible CSD
        avg_psd_2 = spectral.before_after_average(
            naive_psd_2, segment_duration, segment_duration
        )

        # calcaulate CSD
        stride = segment_duration - overlap
        csd_segment_offset = int(np.ceil(segment_duration / stride))

        # also remove naive psds from edge segments
        naive_psd_1 = naive_psd_1[csd_segment_offset : -(csd_segment_offset + 1) + 1]
        naive_psd_2 = naive_psd_2[csd_segment_offset : -(csd_segment_offset + 1) + 1]

        badGPStimes = delta_sigma_cut.run_dsc(
            dsc, naive_psd_1, naive_psd_2, avg_psd_1, avg_psd_2, alphas, lines
        )
        self.assertTrue(
            np.allclose(badGPStimes - badGPStimes_matlab, [0., 0., 0.])
        )