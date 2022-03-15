import pickle
from pathlib import Path

import numpy as np
from gwpy import timeseries

from pygwb import preprocessing, spectral

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
flow = 20
fhigh = 1726
overlap = segment_duration / 2  # overlapping between segments

segment_duration = 192  # also fftlength in pre-processing
dsc = 0.2
alphas = [-5, 0, 3]
notch_file = "test/test_data/Official_O3_HL_notchlist.txt"

ifo1_filtered = preprocessing.preprocessing_data_channel_name(
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

ifo2_filtered = preprocessing.preprocessing_data_channel_name(
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
    ifo1_filtered,
    segment_duration,
    frequency_resolution,
    overlap_factor=0.5,
    overlap_factor_welch_psd=0.5,
    window_fftgram="hann",
)
naive_psd_2 = spectral.power_spectral_density(
    ifo2_filtered,
    segment_duration,
    frequency_resolution,
    overlap_factor=0.5,
    overlap_factor_welch_psd=0.5,
    window_fftgram="hann",
)

# adjacent averated PSDs (detector 1) for each possible CSD
avg_psd_1 = spectral.before_after_average(naive_psd_1, segment_duration, 2)

# adjacent averated PSDs (detector 2) for each possible CSD
avg_psd_2 = spectral.before_after_average(naive_psd_2, segment_duration, 2)

dF = avg_psd_1.frequencies.value[1] - avg_psd_1.frequencies.value[0]
naive_psd_1 = naive_psd_1.crop_frequencies(flow, fhigh + dF)
naive_psd_2 = naive_psd_2.crop_frequencies(flow, fhigh + dF)
avg_psd_1 = avg_psd_1.crop_frequencies(flow, fhigh + dF)
avg_psd_2 = avg_psd_2.crop_frequencies(flow, fhigh + dF)

# calcaulate CSD
stride = segment_duration - overlap
csd_segment_offset = int(np.ceil(segment_duration / stride))

# also remove naive psds from edge segments
naive_psd_1 = naive_psd_1[csd_segment_offset : -(csd_segment_offset + 1) + 1]
naive_psd_2 = naive_psd_2[csd_segment_offset : -(csd_segment_offset + 1) + 1]

my_saved_output = {
    "naive_psd_1": naive_psd_1,
    "naive_psd_2": naive_psd_2,
    "avg_psd_1": avg_psd_1,
    "avg_psd_2": avg_psd_2,
}

test = Path(__file__).parent.resolve()
pickle_path = "naive_and_sliding_psds.pickle"
print(pickle_path)

with open(pickle_path, "wb") as handle:
    pickle.dump(my_saved_output, handle)
