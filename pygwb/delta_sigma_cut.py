import numpy as np


def dsc_cut(naive_sigma, slide_sigma, dsc=0.2, bf_ss=1, bf_ns=1):
    """Function that performs the delta sigma cut, a veto that marks certain GPS times as unusable if the estimations of
    the PSD in the naive (estimating sigma in bin J) and sliding (estimating sigma in bins J \pm 1) differ by more than
    a certain factor (default: dsc=0.2)

       Parameters
       ==========
       naive_sigma: array_like
           An array of input data to feed to Andrew's function.

       slide_sigma: array_like
           Interferometer from which to retrieve the data

       dsc: number
           Name of the channel (e.g.: "L1:GWOSC-4KHZ_R1_STRAIN")

       bf_ss: number
           GPS time of the start of the data taking

       bf_ns: number
           GPS time of the end of the data taking

       Returns
       =======
       dsigma >= dsc: boolean_like
           1: the segment's delta sigma exceeds the threshold value, thus making its corresponding GPStime BAD
           0:  the segment's delta sigma is less than the threshold value, thus making its corresponding GPStime GOOD
    """

    dsigma = np.abs(slide_sigma * bf_ss - naive_sigma * bf_ns) / slide_sigma * bf_ss

    return dsigma >= dsc


def calc_bias_facts(
    seg_dur, delta_f
):  # bias from estimting sigma in neighboring segment

    """Function that performs the delta sigma cut, a veto that marks certain GPS times as unusable if the estimations of
    the PSD in the naive (estimating sigma in bin J) and sliding (estimating sigma in bins J \pm 1) differ by more than
    a certain factor (default: dsc=0.2)

       Parameters
       ==========
       seg_dur: array_like
           A number that specifies how long the FFTs are (in seconds)

       delta_f: array_like
           The frequency resolution of the spectrogram


       Returns
       =======
       bf_ns: array_like
           bias factor for the naive sigma case
      bf_ss: array_like
           bias factor for the sliding sigma case
    """

    segs = seg_dur * delta_f * 2 - 1
    nn = 2 * 9 / 11 * segs  # 9/11 for Welch factor
    bf_ss = nn / (nn - 1)  # sliding sigma bias factor
    nn = 9 / 11 * segs
    bf_ns = nn / (nn - 1)  # naive sigma bias factor
    return bf_ns, bf_ss


def calc_Hf(freqs, alpha=0, fref=20):

    """Function that calculates the H(f) power law

    Parameters
    ==========
    freqs: array_like
        An array of frequencies from an FFT

    alpha: number
        spectral index

    fref: number
        reference frequency

    Returns
    =======
    H(f): array_like
        H(f) power law frequencies weighted by alpha
    """

    Hf = (freqs / fref) ** alpha
    return Hf  # do for different power laws , take all badgps times from all alphas, multiple calls in main func


def calc_sigma_alpha(sensitivity_integrand_with_Hf):

    """Function that calculates the sliding or naive sigma by integrating over the sensitivity integrand

    Parameters
    ==========
    sensitivity_integrand_with_Hf: array_like
        An array that has been calculated with calc_sens_integrand(), given by S_\alpha (f) in
        https://git.ligo.org/stochastic_lite/stochastic_lite/-/issues/11#note_242064


    Returns
    =======
    sigma_alpha: number
        value of sigma for a particular alpha, PSD, and delta_f; can be naive or sliding depending on PSD used
    """

    sigma_alpha = np.sqrt(1 / np.sum(sensitivity_integrand_with_Hf))

    return sigma_alpha


def calc_sens_integrand(
    freq, P1, P2, window1, window2, delta_f, T=32, orf=1, H0=67.9e3 / 3.086e22
):

    """Function that calculates the sensitivity integrand in
    https://git.ligo.org/stochastic_lite/stochastic_lite/-/issues/11#note_242064
    Implicilty for \alpha=0

       Parameters
       ==========
       freq: array_like
           An array of frequencies from a PSD

       P1: array_like
           the PSD of detector #1; size should equal size off freq

       P2: array_like
           the PSD of detector #2; size should equal size of freq

       window1: array-like
            typically Hann window of size np.hanning(4096*192)

       window2: array-like
            typically Hann window of size np.hanning(4096*192)

       delta_f: number
            frequency resolution (Hz)

       T: number
            coherence time (s)

       orf: array_like
           the overlap reduction function as a function of frequency that quantifies the overlap of a detector baseline,
           which depends on the detector locations, relative orientations, etc.

       H0: number
           the Hubble constant



       Returns
       =======
       sens_integrand: array_like
           the sensitivity integrand
    """

    w1w2bar, w1w2squaredbar, oo = WindowFactors(window1, window2)
    S_alpha = 3 * H0 ** 2 / (10 * np.pi ** 2) * 1.0 / freq ** 3
    sigma_square_avg = (
        (w1w2squaredbar / w1w2bar ** 2)
        * 1
        / (2 * T * delta_f)
        * P1
        * P2
        / (orf ** 2.0 * S_alpha ** 2)
    )

    return sigma_square_avg


def WindowFactors(window1, window2):

    """Function that calculates the necessary window factors in line 24 of
    https://git.ligo.org/stochastic-public/stochastic_cleaned/-/blob/master/CrossCorr/src_cc/normalization.m

       Parameters
       ==========
       window1: array-like
            typically Hann window of size np.hanning(4096*192)

       window2: array-like
            typically Hann window of size np.hanning(4096*192)

       Returns
       =======
       w1w2bar: array-like
            Average of the product of the two windows

       w1w2squaredbar: array-like
           average of the product of the squares of the two windows

       w1w2ovlsquaredbar: array-like
            average of the product of the first half times second half of each window
    """

    N1 = len(window1)
    N2 = len(window2)
    Nred = np.gcd(N1, N2).astype(int)
    # if Nred == 1:
    # os.error('size mismatch\n')
    # If window lengths are different, select reduced windows
    # (points at corresponding times)
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


def veto_lines(freqs, lines, df=0):

    """Function that vetos noise lines

    Parameters
    ==========
    freqs: array_like
        An array of frequencies from a PSD

    lines: array_like
        a matrix of form [fmin,fmax] that gives the frequency range of the line

    df: number
        the frequency bin, used if you want to veto frequencies with a frequency bin of the line

    Returns
    =======
    veto: boolean_like
        1: this frequency is contaminated by a noise line
        0: this frequency is fine to use
    """

    fmins = lines[:, 0]
    fmaxs = lines[:, 1]
    nbins = len(freqs)
    veto = np.zeros((nbins, 1), dtype="bool")
    for fbin in range(len(freqs)):
        freq = freqs[fbin]
        index = np.argwhere((freq >= (fmins - df)) & (freq <= fmaxs + df))
        if index.size != 0:
            veto[fbin] = True
    return veto


def run_dsc(dsc, psd1_naive, psd2_naive, psd1_slide, psd2_slide, alphas, lines):

    """Function that runs the delta sigma cut

    Parameters
    ==========
    dsc: number
        The value of the delta sigma cut to use

    psd1_naive; psd2_naive: array_like
        an FFTgram of the PSD computed naively, as in in the particular bin J for detector #1 and #2

    psd1_slide, psd2_slide: array_like
        an FFTgram of the PSD computed by considering the noise in adjacent bins to the bin J, i.e. J-1, J+1 for
        detectors #1 and #2

    alphas: array_like
        the spectral indices to use; the code combines the BadGPStimes from each alpha

    lines: array_like
        a matrix of the form [fmin,fmax] that describes known noise lines

    Returns
    =======
    BadGPStimes: array_like
        an array of the GPS times to not be considered based on the chosen value of the delta sigma cut
    """

    print("running dsc")
    nalphas = len(alphas)
    times = np.array(psd1_naive.times)
    ntimes = len(times)
    df = psd1_naive.df
    Tcoh = 1 / df
    bf_ns, bf_ss = calc_bias_facts(Tcoh, df)
    freqs = np.array(psd1_naive.frequencies)
    overall_cut = np.zeros((ntimes, 1), dtype="bool")
    cuts = np.zeros((nalphas, ntimes), dtype="bool")

    window1 = np.hanning(4096 * 192)
    window2 = window1
    for alpha in range(nalphas):
        Hf = calc_Hf(freqs, alphas[alpha])
        cut = np.zeros((ntimes, 1), dtype="bool")
        for time in range(len(times)):
            psd1_naive_time = psd1_naive[time, :]
            psd1_slide_time = psd1_slide[time, :]
            psd2_naive_time = psd2_naive[time, :]
            psd2_slide_time = psd2_slide[time, :]
            naive_sensitivity_integrand_with_Hf = (
                calc_sens_integrand(
                    freqs, psd1_naive_time, psd2_naive_time, window1, window2, df, Tcoh
                )
                / Hf ** 2
            )
            slide_sensitivity_integrand_with_Hf = (
                calc_sens_integrand(
                    freqs, psd1_slide_time, psd2_slide_time, window1, window2, df, Tcoh
                )
                / Hf ** 2
            )
            veto = veto_lines(freqs, lines)
            keep = np.squeeze(~veto)
            naive_sigma_alpha = calc_sigma_alpha(
                naive_sensitivity_integrand_with_Hf[keep]
            )
            slide_sigma_alpha = calc_sigma_alpha(
                slide_sensitivity_integrand_with_Hf[keep]
            )
            cut[time] = dsc_cut(naive_sigma_alpha, slide_sigma_alpha, dsc, bf_ss, bf_ns)

        cuts[alpha, :] = np.squeeze(cut)

    for time in range(len(times)):
        overall_cut[time] = any(cuts[:, time])

    BadGPStimes = times[np.squeeze(overall_cut)]

    return BadGPStimes


# def run_dsc_again(dsc, naive_sensitivity_integrand, times, psd1_slide, psd2_slide, alphas, lines=0):
#
#
#     nalphas = len(alphas)
#     # times = np.array(psd1_slide.times)
#     ntimes = len(times)
#     df = np.squeeze(psd1_slide[0, 0].deltaF)
#     flow = np.squeeze(psd1_slide[0, 0].flow)
#     nfreqs = np.squeeze(len(psd1_slide[0, 0].data))
#     freqs = np.arange(flow, nfreqs*df+flow, df)
#     Tcoh = np.abs(times[1]-times[0])
#     bf_ns, bf_ss = calc_bias_facts(Tcoh, df)
#     # print("bias ns")
#     # print(bf_ns)
#     # print("bias ss")
#     # print(bf_ss)
#
#     # freqs = np.array(psd1_slide.frequencies)
#     overall_cut = np.zeros((ntimes, 1), dtype='bool')
#     cuts = np.zeros((nalphas, ntimes), dtype='bool')
#     window1 = np.hanning(4096*192)
#     window2 = window1
#     print("starting...")
#     for alpha in range(nalphas):
#         Hf = calc_Hf(freqs, alphas[alpha])
#         cut = np.zeros((ntimes, 1), dtype='bool')
#         for time in range(len(times)):
#             # psd1_naive_time = psd1_naive[time, :]
#             psd1_slide_time = np.squeeze(psd1_slide[time, 0].data)
#             # psd2_naive_time = psd2_naive[time, :]
#             psd2_slide_time = np.squeeze(psd2_slide[time, 0].data)
#             nai_sens_integr = np.squeeze(naive_sensitivity_integrand[time, 0].data)
#             slide_sensitivity_integrand_with_Hf = calc_sens_integrand(freqs, psd1_slide_time, psd2_slide_time, window1, window2, Tcoh) / Hf ** 2
#             naive_sensitivity_integrand_with_Hf = nai_sens_integr / Hf ** 2
#
#             print(slide_sensitivity_integrand_with_Hf)
#             if lines == 0:
#                 keep = np.squeeze(np.ones((nfreqs, 1), dtype='bool'))
#             else:
#                 veto = veto_lines(freqs, lines)
#                 keep = np.squeeze(~veto)
#             naive_sigma_alpha = calc_sigma_alpha(naive_sensitivity_integrand_with_Hf[keep])
#             slide_sigma_alpha = calc_sigma_alpha(slide_sensitivity_integrand_with_Hf[keep])
#             cut[time] = dsc_cut(naive_sigma_alpha, slide_sigma_alpha, dsc, bf_ss, bf_ns)
#             # print("finished one time")
#         cuts[alpha, :] = np.squeeze(cut)
#         # print("finished one alpha")
#
#     for time in range(len(times)):
#         overall_cut[time] = any(cuts[:, time])
#         # print("finished one overall time")
#
#     BadGPStimes = times[np.squeeze(overall_cut)]
#
#     return BadGPStimes
