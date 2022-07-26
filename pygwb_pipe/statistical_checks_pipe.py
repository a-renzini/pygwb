import argparse
import sys
from os import listdir
from os.path import isfile, join

import numpy as np

sys.path.insert(0,'..')

from pygwb.baseline import Baseline
from pygwb.detector import Interferometer
from pygwb.omega_spectra import OmegaSpectrogram
from pygwb.parameters import Parameters
from pygwb.postprocessing import (
    calc_Y_sigma_from_Yf_sigmaf,
    calculate_point_estimate_sigma_spectra,
)
from pygwb.statistical_checks import StatisticalChecks


def sortingFunctionCSD(item):
    return np.float64(item[10:].partition('-')[0])  

def sortingFunctionSpectrum(item):
    return np.float64(item[21:].partition('-')[0])

def loading_function_file(path, file):
    path_to_load = path + file
    loaded_file = np.load(path_to_load, mmap_mode='r')
    return loaded_file 

def sortingFunction(item):
    return np.float64(item[5:].partition('-')[0])  

def loading_function(path, file):
    loader = path + file
    baseload = Baseline.load_from_pickle(loader)
    return baseload 

def run_statistical_checks_from_file(combine_file_path, dsc_file_path, plot_dir, param_file):
    """
    Assumes files are in npz for now. Will generalize later.
    """
    params = Parameters()
    params.update_from_file(param_file)
    
    spectra_file = np.load(combine_file_path)
    dsc_file = np.load(dsc_file_path)
    
    badGPStimes = dsc_file['badGPStimes']
    delta_sigmas = dsc_file['delta_sigmas']
    sliding_times = dsc_file['times']
    naive_sigma_all = dsc_file['naive_sigmas']

    sliding_omega_all, sliding_sigmas_all = spectra_file['point_estimates_seg_UW'], spectra_file['sigmas_seg_UW']
    
    freqs = np.arange(0,params.new_sample_rate/2.+params.frequency_resolution,params.frequency_resolution)
    cut=(params.fhigh>=freqs)&(freqs>=params.flow)
    
    spectrum_file = np.load(combine_file_path, mmap_mode='r')
    
    point_estimate_spectrum = spectrum_file['point_estimate_spectrum']
    sigma_spectrum = spectrum_file['sigma_spectrum']

    baseline_name = params.interferometer_list[0]+params.interferometer_list[1]

    # select alpha for statistical checks
    delta_sigmas_sel = delta_sigmas.T[1]
    naive_sigmas_sel = naive_sigma_all.T[1]

    return StatisticalChecks(sliding_times, sliding_omega_all, sliding_sigmas_all, naive_sigmas_sel, point_estimate_spectrum, sigma_spectrum, freqs[cut], badGPStimes, delta_sigmas_sel, plot_dir, baseline_name, param_file)

def run_statistical_checks_baseline_pickle(baseline_directory, combine_file_path, plot_dir, param_file):
    params = Parameters()
    params.update_from_file(param_file)
    
    baseline_list = [f for f in listdir(baseline_directory) if isfile(join(baseline_directory, f)) if f.startswith("H1")]
    baseline_list.sort(key=sortingFunction)
    
    baseline_list = np.array(baseline_list)
    
    file_0 = baseline_directory + baseline_list[0]
    baseline_0 = Baseline.load_from_pickle(file_0)

    freqs = baseline_0.frequencies
    baseline_name = baseline_0.name
    
    bad_GPS_times = np.array([])
    
    for idx, baseline in enumerate(baseline_list):
        filename = baseline_directory + baseline
        base = Baseline.load_from_pickle(filename) 
        bad_GPS_times = np.append(bad_GPS_times, base.badGPStimes)
    
    sigma_spectrograms = [loading_function(baseline_directory, file).sigma_spectrogram for file in baseline_list]
    final_sigma_spectrogram = sigma_spectrograms[0]
    sigma_iterator = sigma_spectrograms[1:]

    for spectrogram in sigma_iterator:
        final_sigma_spectrogram = final_sigma_spectrogram.append(spectrogram, inplace = False, gap = 'ignore')
    
    point_estimate_spectrograms = [loading_function(pabaseline_directoryth, file).point_estimate_spectrogram for file in baseline_list]
    final_point_estimate_spectrogram = point_estimate_spectrograms[0]
    point_estimate_iterator = point_estimate_spectrograms[1:]

    for spectrogram in point_estimate_iterator:
        final_point_estimate_spectrogram = final_point_estimate_spectrogram.append(spectroogram, inplace = False, gap = 'ignore')

    naive_psd_1 = [loading_function(baseline_directory, file).interferometer_1.psd_spectrogram for file in baseline_list]
    naive_psd_2 = [loading_function(baseline_directory, file).interferometer_2.psd_spectrogram for file in baseline_list]
    final_naive_psd_1_spectorgram = naive_psd_1[0]
    final_naive_psd_2_spectorgram = naive_psd_2[0]
    psd_1_iterator = naive_psd_1[1:]
    psd_2_iterator = naive_psd_2[1:]
    
    for naive_psd_1_spectrogram in psd_1_iterator:
        final_naive_psd_1_spectorgram = final_naive_psd_1_spectorgram.append(naive_psd_1_spectrogram, inplace=False, gap = 'ignore')
    
    for naive_psd_2_spectrogram in psd_2_iterator:
        final_naive_psd_2_spectorgram = final_naive_psd_2_spectorgram.append(naive_psd_2_spectrogram, inplace=False, gap = 'ignore')
    
    delta_sigma = [list(loading_function(baseline_directory, file).delta_sigmas[1]) for file in baseline_list]
    delta_list_final = [item for listt in delta_sigma for item in listt]
    delta_sigmas = np.array(delta_list_final)

    sliding_times_all = final_point_estimate_spectrogram.times.value
    
    sliding_omega_all, sliding_sigmas_all = calc_Y_sigma_from_Yf_sigmaf(final_point_estimate_spectrogram.value, final_sigma_spectrogram.value, freqs = final_point_estimate_spectrogram.frequencies.value, alpha = 0, fref=25, weight_spectrum=False)
    
    deltaF = freqs[1] - freqs[0]
    
    naive_psd_1_cropped = final_naive_psd_1_spectorgram.crop_frequencies(flow, fhigh + deltaF)
    naive_psd_2_cropped = final_naive_psd_1_spectorgram.crop_frequencies(flow, fhigh + deltaF)
    
    times = np.array(naive_psd_1_cropped.times)
    freqs_naive_psd = np.array(naive_psd_1_cropped.frequencies)
    
    for time in range(len(times)):
        naive_sigma_with_Hf = calculate_point_estimate_sigma_spectra(freqs=freqs_naive_psd, avg_psd_1=naive_psd_1_cropped[time,:], avg_psd_2=naive_psd_1_cropped[time,:], orf=baseline_0.overlap_reduction_function, sample_rate=params.new_sample_rate, window_fftgram_dict=params.window_fft_dict, segment_duration=baseline_0.duration, csd = None, fref=25, alpha=0)

        naive_sensitivity_integrand_with_Hf = 1./naive_sigma_with_Hf
        
        naive_sigma_all[time] = np.sqrt(1 / np.sum(naive_sensitivity_integrand_with_Hf))
        
    spectrum_file = np.load(combine_file_path, mmap_mode='r')
    
    point_estimate_spectrum = spectrum_file['point_estimate_spectrum']
    sigma_spectrum = spectrum_file['sigma_spectrum']
        
    return StatisticalChecks(sliding_times_all, sliding_omega_all, sliding_sigmas_all, naive_sigma_all, point_estimate_spectrum, sigma_spectrum, freqs, bad_GPS_times, delta_sigmas, plot_dir, baseline_name, param_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-baseline_directory',help="Baseline directory",action="store", type=str)
    parser.add_argument('-combine_file_path',help="Combined file containing spectra",action="store", type=str)
    parser.add_argument('-plot_dir',help="Directory where plots should be saved",action="store", type=str)
    parser.add_argument('-param_file',help="Parameter file used during analysis",action="store", type=str)
    
    args = parser.parse_args()

    test = run_statistical_checks_from_file(args.baseline_directory, args.combine_file_path, args.plot_dir, args.param_file)
    
    test.generate_all_plots()
