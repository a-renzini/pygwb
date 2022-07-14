import argparse
import numpy as np
from os import listdir
from os.path import isfile, join

import sys
sys.path.insert(0,'..')

from pygwb.statistical_checks import StatisticalChecks
from pygwb.baseline import Baseline
from pygwb.parameters import Parameters
from pygwb.postprocessing import calculate_point_estimate_sigma_spectra, calc_Y_sigma_from_Yf_sigmaf
from pygwb.detector import Interferometer
from pygwb.omega_spectra import OmegaSpectrogram 

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

def run_statistical_checks_from_file(file_directory, combine_file_path, plot_dir, param_file):
    """
    Assumes files are in npz for now. Will generalize later.
    """
    params = Parameters()
    params.update_from_file(param_file)
    
    csds_psds_list = [f for f in listdir(file_directory) if isfile(join(file_directory, f)) if f.startswith("csds")]
    csds_psds_list.sort(key=sortingFunctionCSD)
    csds_psds_list = np.array(csds_psds_list)
    
    spectrum_list = [f for f in listdir(file_directory) if isfile(join(file_directory, f)) if f.startswith("point")]
    spectrum_list.sort(key=sortingFunctionSpectrum)
    spectrum_list = np.array(spectrum_list)
    
    bad_GPS_times = np.array([])
    delta_sigmas = np.array([])
    sliding_times = np.array([])

    for idx,file in enumerate(spectrum_list):
        if idx==20:
            break
        filename = file_directory + file
        file_loaded = np.load(filename, mmap_mode='r') 
        bad_GPS_times = np.append(bad_GPS_times, file_loaded['badGPStimes'])
        delta_sigmas = np.append(delta_sigmas, file_loaded['delta_sigma_values'][1])
        sliding_times = np.append(sliding_times, file_loaded['delta_sigma_times'])
        if idx==0:
            point_estimate_spectrogram=file_loaded['point_estimate_spectrogram']
            sigma_spectrogram=file_loaded['sigma_spectrogram']
            spectrogram_freqs=file_loaded['frequencies']
        else:
            point_estimate_spectrogram=np.append(point_estimate_spectrogram, file_loaded['point_estimate_spectrogram'], axis=0)
            sigma_spectrogram=np.append(sigma_spectrogram, file_loaded['sigma_spectrogram'], axis=0)
        
    point_estimate_spectrogram_omegaGW = OmegaSpectrogram(point_estimate_spectrogram,
            times=sliding_times,
            frequencies=spectrogram_freqs,
            name="pt_est_spectrogram" + f" with alpha={params.alpha}",
            alpha=params.alpha,
            fref=params.fref,
            h0=1,
        )
    
    sigma_spectrogram_omegaGW = OmegaSpectrogram(sigma_spectrogram,
            times=sliding_times,
            frequencies=spectrogram_freqs,
            name="pt_est_spectrogram" + f" with alpha={params.alpha}",
            alpha=params.alpha,
            fref=params.fref,
            h0=1,
        )

    sliding_omega_all, sliding_sigmas_all = calc_Y_sigma_from_Yf_sigmaf(point_estimate_spectrogram_omegaGW, sigma_spectrogram_omegaGW)
    print("Done sliding")
    
    for idx,file in enumerate(csds_psds_list):
        if idx==20:
            break
        filename = file_directory + file
        file_loaded = np.load(filename, mmap_mode='r') 
        
        stride = params.segment_duration * (1 - params.overlap_factor)
        csd_segment_offset = int(np.ceil(params.segment_duration / stride))
        
        if idx==0:
            naive_psd_1_spectrogram_temp = file_loaded['psd_1']
            naive_psd_1_spectrogram = naive_psd_1_spectrogram_temp[csd_segment_offset:-csd_segment_offset]
            naive_psd_2_spectrogram_temp = file_loaded['psd_2']
            naive_psd_2_spectrogram = naive_psd_2_spectrogram_temp[csd_segment_offset:-csd_segment_offset]
        else:
            naive_psd_1_spectrogram_temp = file_loaded['psd_1']
            naive_psd_1_spectrogram = np.append(naive_psd_1_spectrogram,naive_psd_1_spectrogram_temp[csd_segment_offset:-csd_segment_offset],axis=0)
            naive_psd_2_spectrogram_temp = file_loaded['psd_2']
            naive_psd_2_spectrogram = np.append(naive_psd_2_spectrogram,naive_psd_2_spectrogram_temp[csd_segment_offset:-csd_segment_offset],axis=0)
    print("Done reading naive")
    freqs = np.arange(0,params.new_sample_rate/2.+params.frequency_resolution,params.frequency_resolution)
    cut=(params.fhigh>=freqs)&(freqs>=params.flow)
    
    naive_psd_1_spectrogram=naive_psd_1_spectrogram[:,cut]
    naive_psd_2_spectrogram=naive_psd_2_spectrogram[:,cut]

    ifo1 = Interferometer.get_empty_interferometer(params.interferometer_list[0])
    ifo2 = Interferometer.get_empty_interferometer(params.interferometer_list[1])
    ifo_baseline = Baseline.from_parameters(ifo1, ifo2, params)
    ifo_baseline.orf_polarization="tensor"
    ifo_baseline.frequencies=freqs[cut]
    orf = ifo_baseline.overlap_reduction_function
    
    naive_sigma_all=np.zeros(len(sliding_times))
    
    for time in range(len(sliding_times)):
        naive_sigma_with_Hf = calculate_point_estimate_sigma_spectra(freqs=freqs[cut], avg_psd_1=naive_psd_1_spectrogram[time,:], avg_psd_2=naive_psd_2_spectrogram[time,:], orf=orf, sample_rate=params.new_sample_rate, window_fftgram_dict=params.window_fft_dict, segment_duration=params.segment_duration, csd = None, fref=25, alpha=0)

        naive_sensitivity_integrand_with_Hf = 1./naive_sigma_with_Hf
        
        naive_sigma_all[time] = np.sqrt(1 / np.sum(naive_sensitivity_integrand_with_Hf))
    print("Done naive sigma computation")
    spectrum_file = np.load(combine_file_path, mmap_mode='r')
    
    point_estimate_spectrum = spectrum_file['point_estimate_spectrum']
    sigma_spectrum = spectrum_file['sigma_spectrum']
    baseline_name = params.interferometer_list[0]+params.interferometer_list[1]
    return StatisticalChecks(sliding_times, sliding_omega_all, sliding_sigmas_all, naive_sigma_all, point_estimate_spectrum, sigma_spectrum, freqs[cut], bad_GPS_times, delta_sigmas, plot_dir, baseline_name, param_file)

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
    naive_sigma_all = np.zeros(len(times))
    
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