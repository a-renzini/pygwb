import numpy as np
from loguru import logger

from pygwb.notch import StochNotch, StochNotchList
from pygwb.util import calc_bias, window_factors


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



def calc_sigma_square_avg(
    freq: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    delta_f: float,
    orf: np.array,
    N: int,
    T: int = 32,
    H0: float = 67.9e3 / 3.086e22,
):

    """
    Function that calculates the sensitivity integrand in
    https://git.ligo.org/stochastic_lite/stochastic_lite/-/issues/11#note_242064
    Implicilty for \alpha=0

    Parameters
    ==========
    freq: array
        An array of frequencies from a PSD

    P1: array
        the PSD of detector #1; size should equal size of freq

    P2: array
        the PSD of detector #2; size should equal size of freq

    window1: array
        typically Hann window of size np.hanning(4096*192)

    window2: array
        typically Hann window of size np.hanning(4096*192)

    delta_f: float
        frequency resolution (Hz)

    N: int 
        number of samples??

    T: int
        coherence time (s)

    orf: array
        the overlap reduction function as a function of frequency that quantifies the overlap of a detector baseline,
        which depends on the detector locations, relative orientations, etc.

    H0: float
        the Hubble constant

    Returns
    =======
    sigma_square_avg: array
        sigma square average
    """

    w1w2bar, w1w2squaredbar,_ ,_ = window_factors(N)
    S_alpha = 3 * H0 ** 2 / (10 * np.pi ** 2) * 1.0 / freq ** 3
    sigma_square_avg = (
        (w1w2squaredbar / w1w2bar ** 2)
        * 1/ (2 * T * delta_f)* P1
        * P2/ (np.power(orf, 2.0) * S_alpha ** 2)
    )

    return sigma_square_avg



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
    orf: np.array,
    N: np.int,
    notch_list_path: str = "",
    window_fftgram_dict: dict = {"window_fftgram": "hann"},
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

    notch_list_path: np.array
        path to the notch list file

    orf: array
        the overlap reduction function as a function of frequency that quantifies the overlap of a detector baseline,
        which depends on the detector locations, relative orientations, etc.
    N: int 
        number of samples??

    fref: int
        reference frequency

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
    )  # Naive estimate
    bf_ss = calc_bias(
        segmentDuration=segment_duration,
        deltaF=df,
        deltaT=dt,
        N_avg_segs=2,
        window_fftgram_dict=window_fftgram_dict,
    )  # Sliding estimate
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

            naive_sigma_with_Hf = (
                calc_sigma_square_avg(
                    freq=freqs,
                    P1=psd1_naive_time,
                    P2=psd2_naive_time,
                    delta_f=df,
                    orf=orf,
                    N = N,
                    T=dt,
                )/ Hf ** 2
            )
            slide_sigma_with_Hf = (
                calc_sigma_square_avg(
                    freq=freqs,
                    P1=psd1_slide_time,
                    P2=psd2_slide_time,
                    delta_f=df,
                    orf=orf,
                    N = N,
                    T=dt,
                )/ Hf ** 2
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

    return BadGPStimes, dsigmas
