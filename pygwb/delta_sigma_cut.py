import numpy as np
from loguru import logger

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


def run_dsc(
    dsc: float,
    segment_duration: int,
    psd1_naive: np.ndarray,
    psd2_naive: np.ndarray,
    psd1_slide: np.ndarray,
    psd2_slide: np.ndarray,
    alphas: np.ndarray,
    sample_rate: np.int,
    orf: np.array,
    fref: np.int,
    frequency_mask: np.array = True,
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

    notch_list_path: str
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

    logger.info("Running delta sigma cut")
    nalphas = len(alphas)
    times = np.array(psd1_naive.times)
    ntimes = len(times)
    df = psd1_naive.df.value
    dt = 1 / sample_rate  # psd1_naive.df.value ** (-1)
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
    naive_sigmas = np.zeros((nalphas, ntimes))
    slide_sigmas = np.zeros((nalphas, ntimes))
    dsigmas = np.zeros((nalphas, ntimes))

    for idx, alpha in enumerate(alphas):
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
                csd=None,
                fref=fref,
                alpha=alpha,
            )

            slide_sigma_with_Hf = calculate_point_estimate_sigma_spectra(
                freqs=freqs,
                avg_psd_1=psd1_slide_time,
                avg_psd_2=psd2_slide_time,
                orf=orf,
                sample_rate=sample_rate,
                window_fftgram_dict=window_fftgram_dict,
                segment_duration=segment_duration,
                csd=None,
                fref=fref,
                alpha=alpha,
            )

            naive_sensitivity_integrand_with_Hf = 1.0 / naive_sigma_with_Hf
            slide_sensitivity_integrand_with_Hf = 1.0 / slide_sigma_with_Hf

            naive_sigma_alpha = np.sqrt(
                1 / np.sum(naive_sensitivity_integrand_with_Hf[frequency_mask])
            )
            slide_sigma_alpha = np.sqrt(
                1 / np.sum(slide_sensitivity_integrand_with_Hf[frequency_mask])
            )
            naive_sigmas[idx, time] = naive_sigma_alpha
            slide_sigmas[idx, time] = slide_sigma_alpha

            cut[time], dsigma[time] = dsc_cut(
                naive_sigma=naive_sigma_alpha,
                slide_sigma=slide_sigma_alpha,
                dsc=dsc,
                bf_ss=bf_ss,
                bf_ns=bf_ns,
            )

        cuts[idx, :] = np.squeeze(cut)
        dsigmas[idx, :] = np.squeeze(dsigma)

    for time in range(len(times)):
        overall_cut[time] = any(cuts[:, time])

    BadGPStimes = times[np.squeeze(overall_cut)]

    dsigmas_dict = {}
    dsigmas_dict["alphas"] = alphas
    dsigmas_dict["times"] = times
    dsigmas_dict["values"] = dsigmas

    if return_naive_and_averaged_sigmas == True:
        dsigmas_dict["slide_sigmas"] = slide_sigmas * bf_ss
        dsigmas_dict["naive_sigmas"] = naive_sigmas * bf_ns

    return BadGPStimes, dsigmas_dict
