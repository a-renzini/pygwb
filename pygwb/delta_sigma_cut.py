import numpy as np
from loguru import logger

from pygwb.constants import H0
from pygwb.notch import StochNotch, StochNotchList
from pygwb.postprocessing import calculate_point_estimate_sigma_spectra
from pygwb.util import calc_bias


def dsc_cut(
    naive_sigma: np.ndarray,
    slide_sigma: np.ndarray,
    dsc: float = 0.2,
    bf_ss: float = 1,
    bf_ns: float = 1,
):
    """
    Function that performs the delta sigma cut, a veto that marks certain GPS times as unusable if the estimations of
    the PSD in the naive (estimating sigma in bin J) and sliding (estimating sigma in bins J \pm 1) differ by more than
    a certain factor (default: dsc=0.2)

    Parameters
    ==========
    naive_sigma: array
        Naive sigma

    slide_sigma: array
        Sliding sigma

    dsc: float
        Threshold to perform the delta sigma cut

    bf_ss: float
        Sliding bias factor

    bf_ns: float
        Naive bias factor

    Returns
    =======
    dsigma >= dsc: bool
        True: the segment's delta sigma exceeds the threshold value, thus making its corresponding GPStime BAD
        False:  the segment's delta sigma is less than the threshold value, thus making its corresponding GPStime GOOD
    dsigma: np.array
        values of the difference between sliding sigma and naive sigma, i.e.: real value of the delta sigma cut per segment
    """

    dsigma = np.abs(slide_sigma * bf_ss - naive_sigma * bf_ns) / (slide_sigma * bf_ss)

    return dsigma >= dsc, dsigma


def veto_lines(freqs: np.ndarray, lines: np.ndarray, df: float = 0):

    """
    Function that vetos noise lines

    Parameters
    ==========
    freqs: array
        An array of frequencies from a PSD

    lines: array
        a matrix of form [fmin,fmax] that gives the frequency range of the line

    df: float
        the frequency bin, used if you want to veto frequencies with a frequency bin of the line

    Returns
    =======
    veto: bool
        True: this frequency is contaminated by a noise line
        False: this frequency is fine to use
    """
    nbins = len(freqs)
    veto = np.zeros((nbins, 1), dtype="bool")

    if not len(lines):
        return veto

    fmins = lines[:, 0]
    fmaxs = lines[:, 1]
    for fbin in range(len(freqs)):
        freq = freqs[fbin]
        index = np.argwhere((freq >= (fmins - df)) & (freq <= fmaxs + df))
        if index.size != 0:
            veto[fbin] = True
    return veto


def run_dsc(
    dsc: float,
    segment_duration: int,
    sampling_frequency: int,
    psd1_naive: np.ndarray,
    psd2_naive: np.ndarray,
    psd1_slide: np.ndarray,
    psd2_slide: np.ndarray,
    alphas: np.ndarray,
    sample_rate: np.int,
    orf: np.array,
    fref: np.int,
    notch_list_path: str = "",
    window_fftgram_dict: dict = {"window_fftgram": "hann"},
    return_naive_and_averaged_sigmas: np.bool = False,
):

    """
    Function that runs the delta sigma cut

    Parameters
    ==========
    dsc: float
        The value of the delta sigma cut to use

    segment_duration: int
        Duration of each segment

    psd1_naive; psd2_naive: np.array
        an FFTgram of the PSD computed naively, as in in the particular bin J for detector #1 and #2

    psd1_slide, psd2_slide: np.array
        an FFTgram of the PSD computed by considering the noise in adjacent bins to the bin J, i.e. J-1, J+1 for
        detectors #1 and #2

    alphas: np.array
        the spectral indices to use; the code combines the BadGPStimes from each alpha

    sample_rate: np.int
        sampling rate (Hz)

    notch_list_path: np.array
        path to the notch list file
    
    window_fftgram_dict: dictionary, optional
        Dictionary with window characteristics. Default is `(window_fftgram_dict={"window_fftgram": "hann"}`

    orf: array
        the overlap reduction function as a function of frequency that quantifies the overlap of a detector baseline,
        which depends on the detector locations, relative orientations, etc.
        
    fref: int 
        reference frequency (Hz)
    
    return_naive_and_averaged_sigmas: bool
        option to return naive and sliding sigmas

    Returns
    =======
    BadGPStimes: np.array
        an array of the GPS times to not be considered based on the chosen value of the delta sigma cut
    """
    if notch_list_path:
        lines_stochnotch = StochNotchList.load_from_file(f"{notch_list_path}")
        lines = np.zeros((len(lines_stochnotch), 2))

        for index, notch in enumerate(lines_stochnotch):
            lines[index, 0] = lines_stochnotch[index].minimum_frequency
            lines[index, 1] = lines_stochnotch[index].maximum_frequency
    else:
        lines = np.zeros((0, 2))

    logger.info("Running delta sigma cut")
    nalphas = len(alphas)
    times = np.array(psd1_naive.times)
    ntimes = len(times)
    df = psd1_naive.df.value
    dt = psd1_naive.df.value ** (-1)
    # Naive estimate
    bf_ns = calc_bias(
        segmentDuration=segment_duration,
        deltaF=df,
        deltaT=dt,
        N_avg_segs=1,
        window_fftgram_dict=window_fftgram_dict,
    )  
    # Sliding estimate
    bf_ss = calc_bias(
        segmentDuration=segment_duration,
        deltaF=df,
        deltaT=dt,
        N_avg_segs=2,
        window_fftgram_dict=window_fftgram_dict,
    )  
    freqs = np.array(psd1_naive.frequencies)
    overall_cut = np.zeros((ntimes, 1), dtype="bool")
    cuts = np.zeros((nalphas, ntimes), dtype="bool")
    dsigmas = np.zeros((nalphas, ntimes))
    veto = veto_lines(freqs=freqs, lines=lines)
    keep = np.squeeze(~veto)

    for alpha in range(nalphas):
        Hf = (freqs / fref) ** alphas[alpha]
        cut = np.zeros((ntimes, 1), dtype="bool")
        dsigma = np.zeros((ntimes, 1), dtype="float")
        for time in range(len(times)):
            psd1_naive_time = psd1_naive[time, :]
            psd1_slide_time = psd1_slide[time, :]
            psd2_naive_time = psd2_naive[time, :]
            psd2_slide_time = psd2_slide[time, :]

            naive_sigma_with_Hf = calculate_point_estimate_sigma_spectra(
                    freqs=freqs,
                    avg_psd_1=psd1_naive_time,
                    avg_psd_2=psd2_naive_time,
                    orf=orf,
                    sample_rate=sample_rate,
                    window_fftgram_dict=window_fftgram_dict,
                    segment_duration=segment_duration,
                    csd = None,
                    fref=1,
                    alpha=0,
                )
            
            slide_sigma_with_Hf = calculate_point_estimate_sigma_spectra(
                    freqs=freqs,
                    avg_psd_1=psd1_slide_time,
                    avg_psd_2=psd2_slide_time,
                    orf=orf,
                    sample_rate=sample_rate,
                    window_fftgram_dict=window_fftgram_dict,
                    segment_duration=segment_duration,
                    csd = None,
                    fref=1,
                    alpha=0,
                )

            naive_sensitivity_integrand_with_Hf = 1./naive_sigma_with_Hf
            slide_sensitivity_integrand_with_Hf = 1./slide_sigma_with_Hf

            naive_sigma_alpha = np.sqrt(1 / np.sum(naive_sensitivity_integrand_with_Hf[keep]))
            slide_sigma_alpha = np.sqrt(1 / np.sum(slide_sensitivity_integrand_with_Hf[keep]))

            cut[time], dsigma[time] = dsc_cut(
                naive_sigma=naive_sigma_alpha,
                slide_sigma=slide_sigma_alpha,
                dsc=dsc,
                bf_ss=bf_ss,
                bf_ns=bf_ns,
            )

        cuts[alpha, :] = np.squeeze(cut)
        dsigmas[alpha, :] = np.squeeze(dsigma)

    for time in range(len(times)):
        overall_cut[time] = any(cuts[:, time])

    BadGPStimes = times[np.squeeze(overall_cut)]

    if return_naive_and_averaged_sigmas==True:
        dsigmas_dict = {}
        dsigmas_dict['alphas'] = alphas
        dsigmas_dict['times'] = times
        dsigmas_dict['values'] = dsigmas
        dsigmas_dict['slide_sigma'] = slide_sigma_alpha * bf_ss
        dsigmas_dict['naive_sigma'] = naive_sigma_alpha * bf_ns 
    else:
        dsigmas_dict = {}
        dsigmas_dict['alphas'] = alphas
        dsigmas_dict['times'] = times
        dsigmas_dict['values'] = dsigmas

    return BadGPStimes, dsigmas_dict
