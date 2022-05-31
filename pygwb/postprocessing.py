import pickle

import h5py
import numpy as np
from loguru import logger
from tqdm import tqdm

from pygwb.constants import H0

from .util import calc_bias, window_factors


def postprocess_Y_sigma(Y_fs, var_fs, segment_duration, deltaF, new_sample_rate):
    '''Run postprocessing of point estimate and sigma spectrograms, combining even and 
    odd segments. For more details see - '''
    size = np.size(Y_fs, axis=0)
    _, w1w2squaredbar, _, w1w2squaredovlbar = window_factors(
        segment_duration * new_sample_rate
    )
    k = w1w2squaredovlbar / w1w2squaredbar

    # even/odd indices
    evens = np.arange(0, size, 2)
    odds = np.arange(1, size, 2)

    X_even = np.nansum(Y_fs[evens] / var_fs[evens], axis=0)
    GAMMA_even = np.nansum(var_fs[evens] ** -1, axis=0)
    X_odd = np.nansum(Y_fs[odds] / var_fs[odds], axis=0)
    GAMMA_odd = np.nansum(var_fs[odds] ** -1, axis=0)
    sigma2_oo = 1 / np.nansum(GAMMA_odd)
    sigma2_ee = 1 / np.nansum(GAMMA_even)
    sigma2_1 = 1 / np.nansum(var_fs[0, :] ** -1)
    sigma2_N = 1 / np.nansum(var_fs[-1, :] ** -1)
    sigma2IJ = 1 / sigma2_oo + 1 / sigma2_ee - (1 / 2) * (1 / sigma2_1 + 1 / sigma2_N)

    Y_f_new = (
        X_odd * (1 - (k / 2) * sigma2_oo * sigma2IJ)
        + X_even * (1 - (k / 2) * sigma2_ee * sigma2IJ)
    ) / (
        GAMMA_even
        + GAMMA_odd
        - k
        * (GAMMA_even + GAMMA_odd - (1 / 2) * (1 / var_fs[0, :] + 1 / var_fs[-1, :]))
    )

    inv_var_f_new = (
        GAMMA_odd
        + GAMMA_even
        - k
        * (GAMMA_odd + GAMMA_even - (1 / 2) * (1 / var_fs[0, :] + 1 / var_fs[-1, :]))
    ) / (1 - (k ** 2 / 4) * sigma2_oo * sigma2_ee * sigma2IJ ** 2)
    bias = calc_bias(segment_duration, deltaF, 1 / new_sample_rate, N_avg_segs=2)
    logger.debug(f"Bias factor: {bias}")
    var_f_new = (1 / inv_var_f_new) * bias ** 2

    return Y_f_new, var_f_new


def calc_Y_sigma_from_Yf_sigmaf(Y_f, sigma_f, frequency_mask=True, alpha=None, fref=None):
    """
    Calculate the omega point estimate and sigma from their respective spectra,
    or spectrograms, taking into account the desired spectral weighting.
    To apply weighting, the frequency array associated to the spectra must be supplied.

    Parameters
    ==========
    Y_f: `pygwb.omega_spectrogram.OmegaSpectrogram`
        Point estimate spectrum
    var_f: `pygwb.omega_spectrogram.OmegaSpectrogram`
        Sigma spectrum

    Note
    ====
    If passing in spectrograms, the point estimate and sigma will be calculated per
    spectrum, without any time-averaging applied.

    """
    # Reweight in case one wants to pass it.
    Y_f.reweight(new_alpha=alpha, new_fref=fref)
    sigma_f.reweight(new_alpha=alpha, new_fref=fref)

    # now just strip off what we need...
    Y_f = Y_f.value
    var_f = sigma_f.value**2

    var = 1 / np.sum(var_f[frequency_mask] ** (-1), axis=-1)

    if len(Y_f.shape) == 1:
        Y = np.nansum(Y_f[frequency_mask] * (var / var_f[frequency_mask]), axis=-1)
    # need to make this nan-safe
    elif len(Y_f.shape) == 2:
        Y = np.einsum("tf, f -> t", (Y_f[frequency_mask] / var_f[frequency_mask])) * var
    else:
        raise ValueError("The input is neither a spectrum nor a spectrogram.")

    sigma = np.sqrt(var)

    return Y, sigma


def calculate_point_estimate_sigma_spectrogram(
    freqs,
    csd,
    avg_psd_1,
    avg_psd_2,
    orf,
    sample_rate,
    segment_duration,
    fref=1,
    alpha=0,
):
    """
    Calculate the Omega point estimate and associated sigma spectrograms,
    given a set of cross-spectral and power-spectral density spectrograms.

    Parameters
    ==========
    freqs: array_like
        Frequencies associated to the spectrograms.
    csd: gwpy Spectrogram
        CSD spectrogram for detectors 1 and 2.
    avg_psd_1: gwpy Spectrogram
        Spectrogram of averaged PSDs for detector 1.
    avg_psd_2: gwpy Spectrogram
        Spectrogram of averaged PSDs for detector 2.
    orf: array_like
        Overlap reduction function.
    sample_rate: float
        Sampling rate of the data.
    segment_duration: float
        Duration of each segment in seconds.
    fref: float, optional
        Reference frequency to use in the weighting calculation.
        Final result refers to this frequency.
    alpha: float, optional
        Spectral index to use in the weighting.
    weight_spectrogram: bool, optional
        Flag to apply spectral weighting, True by default.
    """
    S_alpha = 3 * H0.si.value ** 2 / (10 * np.pi ** 2) / freqs ** 3
    S_alpha *= (freqs / fref) ** alpha
    Y_fs = np.real(csd) / (orf * S_alpha)
    var_fs = (
        1
        / (2 * segment_duration * (freqs[1] - freqs[0]))
        * avg_psd_1
        * avg_psd_2
        / (orf ** 2 * S_alpha ** 2)
    )

    w1w2bar, w1w2squaredbar, _, _ = window_factors(sample_rate * segment_duration)

    var_fs = var_fs * w1w2squaredbar / w1w2bar ** 2
    return Y_fs, var_fs


def calculate_point_estimate_sigma_integrand(
    freqs,
    csd,
    avg_psd_1,
    avg_psd_2,
    orf,
    sample_rate,
    segment_duration,
    fref=1,
    alpha=0,
):
    """
    Calculate the Omega point estimate and associated sigma integrand,
    given a set of cross-spectral and power-spectral density spectrograms.
    This is particularly useful for statistical checks.

    Parameters
    ==========
    freqs: array_like
        Frequencies associated to the spectrograms.
    csd: gwpy Spectrogram
        CSD spectrogram for detectors 1 and 2.
    avg_psd_1: gwpy Spectrogram
        Spectrogram of averaged PSDs for detector 1.
    avg_psd_2: gwpy Spectrogram
        Spectrogram of averaged PSDs for detector 2.
    orf: array_like
        Overlap reduction function.
    sample_rate: float
        Sampling rate of the data.
    segment_duration: float
        Duration of each segment in seconds.
    fref: float, optional
        Reference frequency to use in the weighting calculation.
        Final result refers to this frequency.
    alpha: float, optional
        Spectral index to use in the weighting.
    weight_spectrogram: bool, optional
        Flag to apply spectral weighting, True by default.
    """
    S_alpha = 3 * H0.si.value ** 2 / (10 * np.pi ** 2) / freqs ** 3
    S_alpha *= (freqs / fref) ** alpha
    Y_fs = csd / (orf * S_alpha)
    var_fs = (
        1
        / (2 * segment_duration * (freqs[1] - freqs[0]))
        * avg_psd_1
        * avg_psd_2
        / (orf ** 2 * S_alpha ** 2)
    )

    w1w2bar, w1w2squaredbar, _, _ = window_factors(sample_rate * segment_duration)

    var_fs = var_fs * w1w2squaredbar / w1w2bar ** 2
    return Y_fs, var_fs

def combine_spectra_with_sigma_weights(main_spectra, weights_spectra):
    """
    Combine different statistically independent spectra :math: `S_i(f)` using spectral weights :math: `w_i(f)`, as

    .. math::
    S(f) = \frac{\sum_i \frac{S_i(f)}{w^2_i(f)}}{\sum_i \frac{1}{w^2_i(f)}}, \qquad \sigma = \sqrt{\frac{1}{\sum_i \frac{1}{w^2_i(f)}}}.

    Parameters
    =========
    main_spectra: list 
        List of arrays or FrequencySeries or OmegaSpectrum objects to be combined.
    weights_spectra: list
        List of arrays or FrequencySeries or OmegaSpectrum objects to use as weights.

    Returns
    =======
    combined_weighted_spectrum: array_like
        Final spectrum obtained combining the original spectra with given weights.
    combined_weights_spectrum: array_like
        Variance associated to the final spectrum obtained combining the given weights.
    """
    res_1 = 1 / np.sum(1 / weights_spectra ** 2, axis=0)
    combined_weights_spectrum = np.sqrt(res_1)
    combined_weighted_spectrum = np.sum(
        main_spectra / weights_spectra ** 2
    , axis = 0) * res_1
    return combined_weighted_spectrum, combined_weights_spectrum
