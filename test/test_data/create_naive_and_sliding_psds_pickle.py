import pickle
from pathlib import Path

import numpy as np
from gwpy import timeseries

from pygwb import pre_processing, spectral


def create_psd_data():
    # Analysis parameters
    ifos = ["H1", "L1"]
    t0 = 1247644138  # start GPS time
    tf = 1247645038  # end GPS time
    data_type = "public"  # private -> running on LIGO data grid
    channel_suffix = "GWOSC-16KHZ_R1_STRAIN"  # detector name will be added later
    new_sample_rate = 4096  # sampled rate after resampling
    cutoff_frequency = 11  # high pass filter cutoff frequency
    segment_duration = 192  # also fftlength in pre-processing
    frequency_resolution = 1.0 / 32  # final frequency resolution of CSD and PSD
    flow = 20
    fhigh = 1726
    overlap = segment_duration / 2  # overlapping between segments

    segment_duration = 192  # also fftlength in pre-processing
    dsc = 0.2
    alphas = [-5, 0, 3]
    file_directory = Path(__file__).parent.resolve()
    notch_file = file_directory / "Official_O3_HL_notchlist.txt"

    stride = segment_duration - overlap
    csd_segment_offset = int(np.ceil(segment_duration / stride))

    data = dict()

    for ii, ifo in enumerate(ifos):
        ifo_filtered = pre_processing.preprocessing_data_channel_name(
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

        naive_psd = spectral.power_spectral_density(
            ifo_filtered,
            segment_duration,
            frequency_resolution,
            overlap_factor=0.5,
            overlap_factor_welch_psd=0.5,
            window_fftgram="hann",
        )

        # adjacent averated PSDs (detector 1) for each possible CSD
        avg_psd = spectral.before_after_average(naive_psd, segment_duration, 2)

        dF = avg_psd.frequencies.value[1] - avg_psd.frequencies.value[0]
        naive_psd = naive_psd.crop_frequencies(flow, fhigh + dF)
        avg_psd = avg_psd.crop_frequencies(flow, fhigh + dF)

        naive_psd = naive_psd[csd_segment_offset : -(csd_segment_offset + 1) + 1]
        data[f"naive_psd_{ii}"] = naive_psd
        data[f"avg_psd_{ii}"] = avg_psd

    return data


if __name__ == "__main__":
    file_directory = Path(__file__).parent.resolve()
    pickle_path = file_directory / "naive_and_sliding_psds.pickle"
    print(pickle_path)
    data = create_psd_data()
    with open(pickle_path, 'wb') as handle:
        pickle.dump(data, handle)
