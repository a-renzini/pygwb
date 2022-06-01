import numpy as np
from loguru import logger

from pygwb.constants import H0
from pygwb.notch import StochNotch, StochNotchList

from .util import calc_bias


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

    dsigma = np.abs(slide_sigma * bf_ss - naive_sigma * bf_ns) / slide_sigma * bf_ss

    return dsigma >= dsc, dsigma


def calc_Hf(freqs: np.ndarray, alpha: float = 0, fref: int = 20):

    """
    Function that calculates the H(f) power law

    Parameters
    ==========
    freqs: array
        An array of frequencies from an FFT

    alpha: float
        spectral index

    fref: int
        reference frequency

    Returns
    =======
    H(f): array
        H(f) power law frequencies weighted by alpha
    """

    Hf = (freqs / fref) ** alpha
    return Hf  # do for different power laws , take all badgps times from all alphas, multiple calls in main func


def calc_sigma_alpha(sensitivity_integrand_with_Hf: np.ndarray):

    """
    Function that calculates the sliding or naive sigma by integrating over the sensitivity integrand

    Parameters
    ==========
    sensitivity_integrand_with_Hf: array
        An array that has been calculated with calc_sens_integrand(), given by S_\alpha (f) in
        https://git.ligo.org/stochastic_lite/stochastic_lite/-/issues/11#note_242064


    Returns
    =======
    sigma_alpha: float
        value of sigma for a particular alpha, PSD, and delta_f; can be naive or sliding depending on PSD used
    """

    sigma_alpha = np.sqrt(1 / np.sum(sensitivity_integrand_with_Hf))

    return sigma_alpha


def calc_sens_integrand(
    freq: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    window1: np.ndarray,
    window2: np.ndarray,
    delta_f: float,
    orf: np.array,
    T: int = 32,
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

    T: int
        coherence time (s)

    orf: array
        the overlap reduction function as a function of frequency that quantifies the overlap of a detector baseline,
        which depends on the detector locations, relative orientations, etc.

    Returns
    =======
    sens_integrand: array
        the sensitivity integrand
    """

    w1w2bar, w1w2squaredbar, oo = WindowFactors(window1=window1, window2=window2)
    S_alpha = 3 * H0.si.value ** 2 / (10 * np.pi ** 2) * 1.0 / freq ** 3
    sigma_square_avg = (
        (w1w2squaredbar / w1w2bar ** 2)
        * 1
        / (2 * T * delta_f)
        * P1
        * P2
        / (np.power(orf, 2.0) * S_alpha ** 2)
    )

    return sigma_square_avg


def WindowFactors(window1: np.ndarray, window2: np.ndarray):

    """
    Function that calculates the necessary window factors in line 24 of
    https://git.ligo.org/stochastic-public/stochastic_cleaned/-/blob/master/CrossCorr/src_cc/normalization.m

    Parameters
    ==========
    window1: array
        typically Hann window of size np.hanning(4096*192)

    window2: array
        typically Hann window of size np.hanning(4096*192)

    Returns
    =======
    w1w2bar: array
        Average of the product of the two windows

    w1w2squaredbar: array
        average of the product of the squares of the two windows

    w1w2ovlsquaredbar: array
        average of the product of the first half times second half of each window
    """

    N1 = len(window1)
    N2 = len(window2)
    Nred = np.gcd(N1, N2).astype(int)
    indices1 = (np.array(range(0, Nred, 1)) * N1 / Nred).astype(int)
    indices2 = (np.array(range(0, Nred, 1)) * N2 / Nred).astype(int)
    window1red = window1[indices1]
    window2red = window2[indices2]

    # extract 1st and 2nd half of windows

    cut = int(np.floor(Nred / 2))

    firsthalf1 = window1red[0:cut]
    secondhalf1 = window1red[cut:Nred]

    firsthalf2 = window2red[0:cut]
    secondhalf2 = window2red[cut:Nred]

    # calculate window factors
    w1w2bar = np.mean(window1red * window2red)
    w1w2squaredbar = np.mean((window1red ** 2) * (window2red ** 2))
    w1w2ovlsquaredbar = np.mean((firsthalf1 * secondhalf1) * (firsthalf2 * secondhalf2))

    return w1w2bar, w1w2squaredbar, w1w2ovlsquaredbar


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
    notch_list_path: str = "",
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
    bf_ns = calc_bias(
        segmentDuration=segment_duration, deltaF=df, deltaT=dt, N_avg_segs=1
    )  # Naive estimate
    bf_ss = calc_bias(
        segmentDuration=segment_duration, deltaF=df, deltaT=dt, N_avg_segs=2
    )  # Sliding estimate
    freqs = np.array(psd1_naive.frequencies)
    overall_cut = np.zeros((ntimes, 1), dtype="bool")
    cuts = np.zeros((nalphas, ntimes), dtype="bool")
    dsigmas = np.zeros((nalphas, ntimes))
    veto = veto_lines(freqs=freqs, lines=lines)
    keep = np.squeeze(~veto)

    window1 = np.hanning(segment_duration * sampling_frequency)
    window2 = window1
    for alpha in range(nalphas):
        Hf = calc_Hf(freqs=freqs, alpha=alphas[alpha])
        cut = np.zeros((ntimes, 1), dtype="bool")
        dsigma = np.zeros((ntimes, 1), dtype="float")
        for time in range(len(times)):
            psd1_naive_time = psd1_naive[time, :]
            psd1_slide_time = psd1_slide[time, :]
            psd2_naive_time = psd2_naive[time, :]
            psd2_slide_time = psd2_slide[time, :]

            naive_sensitivity_integrand_with_Hf = (
                calc_sens_integrand(
                    freq=freqs,
                    P1=psd1_naive_time,
                    P2=psd2_naive_time,
                    window1=window1,
                    window2=window2,
                    delta_f=df,
                    orf=orf,
                    T=dt,
                )
                / Hf ** 2
            )
            slide_sensitivity_integrand_with_Hf = (
                calc_sens_integrand(
                    freq=freqs,
                    P1=psd1_slide_time,
                    P2=psd2_slide_time,
                    window1=window1,
                    window2=window2,
                    delta_f=df,
                    orf=orf,
                    T=dt,
                )
                / Hf ** 2
            )
            naive_sigma_alpha = calc_sigma_alpha(
                sensitivity_integrand_with_Hf=naive_sensitivity_integrand_with_Hf[keep]
            )
            slide_sigma_alpha = calc_sigma_alpha(
                sensitivity_integrand_with_Hf=slide_sensitivity_integrand_with_Hf[keep]
            )
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
