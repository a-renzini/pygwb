"""The postprocessing module combines all methods which are useful in the end stages of the analysis,
more specifically when combining spectrograms into spectra, and spectra into one overall point estimate for the gravitational-wave background.

This module contains methods such as ``postprocess_Y_sigma``, which combines point estimate and sigma spectrograms
into spectra using a method where odd and even segments are treated differently if the data are overlapping.
To account for the overlap, it uses another function of the postprocessing module, ``odd_even_segment_postprocessing``. 
Additional information about this procedure can be found `here <https://arxiv.org/pdf/2303.15696.pdf>`_.

These spectrograms are computed from the cross spectral density (CSD) and the power spectral density (PSD).
The postprocessing module also contains the method which takes care of the above, namely
``calculate_point_estimate_sigma_spectra``.
Starting from a set of CSD and PSD spectrograms, one can compute the point estimate and sigma spectrograms,
objects that contain both frequency and segment data information.
Then, the spectrograms are combined into spectra with these methods.
These spectra then have to be combined into one single point estimate and its variance, which is achieved by calling ``calc_Y_sigma_from_Yf_sigmaf``.

Examples
--------

Starting from averaged PSDs and the CSD of a baseline,
we can compute the overall point estimate for the gravitational-wave background.

Assuming we already computed the CSDs and PSDs, see for example in :doc:`pygwb.spectral`, 
we also need the overlap reduction function, see :doc:`pygwb.orfs`.
With these at hand, we can compute the point estimate spectrogram
and its variance.

>>> Y_spectrogram, var_spectrogram = calculate_point_estimate_sigma_spectra(
        CSD_baseline.frequencies.value,
        CSD_baseline[2:-2],
        ifo_1.average_psd.crop_frequencies(3.12500000e-02,2.048e+03),
        ifo_2.average_psd.crop_frequencies(3.12500000e-02,2.048e+03),
        orf,
        sample_rate=4096,
        segment_duration=192,
        window_fftgram_dict={"window_fftgram": "hann"},
        overlap_factor=0.5,
        fref=25.0,
        alpha=0.0,
    )

For this example, we used some pre-computed CSD and PSDs, for some baseline, together with its overlap reduction function.
The above returns the point estimate and variance as a function of frequency and time, i.e., for each segment.
These can be combined over all analysis segments into a single point estimate and sigma spectrum, i.e. as a function of frequency only.

>>> Y_spectrum, var_spectrum = postprocess_Y_sigma(
        Y_spectrogram.value,
        var_spectrogram.value,
        segment_duration=192,
        deltaF=1/32.,
        new_sample_rate=4096,
        frequency_mask=True,
        badtimes_mask=None,
        window_fftgram_dict={"window_fftgram": "hann"},
        window_fftgram_dict_welch={"window_fftgram": "hann"},
        overlap_factor=0.5,
        overlap_factor_welch=0.5,
        N_avg_segs=2,
    )

To compute a single point estimate and its variance for the magnitude of the GWB, one uses the frequency 
spectra computed above and relies on the following method:

>>> Y, sigma = postpp.calc_Y_sigma_from_Yf_sigmaf(
        Y_spectrum, np.sqrt(var_spectrum), frequency_mask=True, alpha=None, fref=None
    )

The result is an overall point estimate and standard deviation. Additional information on the various methods outlined above
can be found in the following dedicated API documentation of the module.
"""
import numpy as np
import scipy.ndimage as ndi
from loguru import logger

from pygwb.constants import H0
from pygwb.omega_spectra import OmegaSpectrum

from .util import _check_omegaspectra, calc_bias, window_factors


def postprocess_Y_sigma(
    Y_fs,
    var_fs,
    segment_duration,
    deltaF,
    new_sample_rate,
    frequency_mask=True,
    badtimes_mask=None,
    window_fftgram_dict={"window_fftgram": "hann"},
    window_fftgram_dict_welch={"window_fftgram": "hann"},
    overlap_factor=0.5,
    overlap_factor_welch=0.5,
    N_avg_segs=2,
):
    """Run postprocessing of point estimate and sigma spectrograms, combining even and
    odd segments in the case of overlapping data.
    For more details see - https://dcc.ligo.org/public/0027/T040089/000/T040089-00.pdf

    Parameters
    =======
    Y_fs: ``array-like``
        2D array of point estimates with Ntimes x Nfreqs with overlapping segments.
    var_fs: ``array-like``
        2D array of variances or 2D with dimensions Ntimes x Nfreqs with overlapping time segments.
    segment_duration: ``float``
        Duration of each time segment.
    deltaF: ``float``
        Frequency resolution.
    new_sample_rate: ``float``
        Sample rate of timeseries after resampling.
    frequency_mask: ``array-like``, optional
        Boolean mask to apply to frequencies for the calculation.
        Defaults to True which includes all frequencies in the analysis.
    badtimes_mask: ``array-like``, optional
        Boolean mask to apply to GPStimes in the calculation. Defaults to None such that all times are included.
    window_fftgram_dict: ``dictionary``, optional
        Dictionary with window characteristics used in PSD estimation.
        Default is ``window_fftgram_dict={"window_fftgram": "hann"}``
    overlap_factor: ``float``, optional
        Overlap factor used in PSD estimation. Default is 0.5.
    N_avg_segs: ``int``, optional
        Number of segments over which the average is performed. 
        This is useful for computing the bias, nothing more. Default is 2.

    Returns
    =======
    Y_f_new: ``array-like``
        1D point estimate spectrum.
    sigma_f_few: ``array-like``
        1D sigma spectrum.

    See also
    --------
    pygwb.util.calc_bias
    """
    if badtimes_mask is None:
        badtimes_mask = np.zeros(len(Y_fs), dtype=bool)

    goodtimes_mask = ~badtimes_mask
    labels, n_labels = ndi.label(goodtimes_mask)

    Y_fs_sliced = []
    var_fs_sliced = []

    for sli in ndi.find_objects(labels):
        Y = Y_fs[sli]
        var = var_fs[sli]

        if len(Y) == 1:
            Y_fs_sliced.append(Y[0])
            var_fs_sliced.append(var[0])
        else:
            Y_red, var_red = odd_even_segment_postprocessing(
                Y,
                var,
                segment_duration,
                new_sample_rate,
                frequency_mask=frequency_mask,
                window_fftgram_dict=window_fftgram_dict,
                overlap_factor=overlap_factor,
                N_avg_segs=N_avg_segs,
            )
            Y_fs_sliced.append(Y_red)
            var_fs_sliced.append(var_red)

    Y_fs_sliced = np.array(Y_fs_sliced)
    var_fs_sliced = np.array(var_fs_sliced)

    Y_f_new, sigma_f_new = combine_spectra_with_sigma_weights(
        Y_fs_sliced, np.sqrt(var_fs_sliced)
    )

    bias = calc_bias(
        segment_duration,
        deltaF,
        1 / new_sample_rate,
        N_avg_segs=N_avg_segs,
        window_fftgram_dict=window_fftgram_dict_welch,
        overlap_factor=overlap_factor_welch,
    )
    logger.debug(f"Bias factor: {bias}")
    sigma_f_new *= bias

    return Y_f_new, sigma_f_new

def odd_even_segment_postprocessing(
    Y_fs,
    var_fs,
    segment_duration,
    new_sample_rate,
    frequency_mask=True,
    window_fftgram_dict={"window_fftgram": "hann"},
    overlap_factor=0.5,
    N_avg_segs=2,
):
    """Perform averaging which combines even and odd segments for overlapping data. 

    Parameters
    =======
    Y_fs: ``array-like``
        2D array of point estimates with Ntimes x Nfreqs with overlapping segments.
    var_fs: ``array-like``
        2D array of variances or 2D with dimensions Ntimes x Nfreqs with overlapping time segments.
    segment_duration: ``float``
        Duration of each time segment.
    new_sample_rate: ``float``
        Sample rate of timeseries after resampling.
    frequency_mask: ``array-like``, optional
        Boolean mask to apply to frequencies for the calculation.
    window_fftgram_dict: ``dictionary``, optional
        Dictionary with window characteristics used in PSD estimation.
        Default is ``window_fftgram_dict={"window_fftgram": "hann"}``.
    overlap_factor: ``float``, optional
        Defines the overlap between consecutive data chunks used in the calculation. Default is 0.5.
    
    Returns
    =======
    Y_f_new: ``array-like``
        1D point estimate spectrum.
    var_f_few: ``array-like``
        1D sigma spectrum.

    See also
    --------
    pygwb.util.window_factors
    """
    _, w1w2squaredbar, _, w1w2squaredovlbar = window_factors(
        int(segment_duration * new_sample_rate), window_fftgram_dict, overlap_factor=overlap_factor
    )
    k = w1w2squaredovlbar / w1w2squaredbar

    size = np.size(Y_fs, axis=0)
    # even/odd indices
    evens = np.arange(0, size, 2)
    odds = np.arange(1, size, 2)

    X_even = np.nansum(Y_fs[evens] / var_fs[evens], axis=0)
    GAMMA_even = np.nansum(var_fs[evens] ** -1, axis=0)
    X_odd = np.nansum(Y_fs[odds] / var_fs[odds], axis=0)
    GAMMA_odd = np.nansum(var_fs[odds] ** -1, axis=0)

    sigma2_oo = 1 / np.nansum(GAMMA_odd[frequency_mask])
    sigma2_ee = 1 / np.nansum(GAMMA_even[frequency_mask])
    sigma2_1 = 1 / np.nansum(var_fs[0, frequency_mask] ** -1)
    sigma2_N = 1 / np.nansum(var_fs[-1, frequency_mask] ** -1)
    sigma2IJ = 1 / sigma2_oo + 1 / sigma2_ee - (1 / 2) * (1 / sigma2_1 + 1 / sigma2_N)

    sigma2_oo, sigma2_ee, sigma2_1, sigma2_N = [
        s if s != np.inf else 0 for s in (sigma2_oo, sigma2_ee, sigma2_1, sigma2_N)
    ]

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

    var_f_new = 1 / inv_var_f_new

    return Y_f_new, var_f_new

def calc_Y_sigma_from_Yf_sigmaf(
    Y_f, sigma_f, frequency_mask=True, alpha=None, fref=None
):
    """
    Calculate the omega point estimate and sigma from their respective spectra,
    or spectrograms, taking into account the desired spectral weighting.
    To apply weighting, the frequency array associated to the spectra must be supplied.

    If applied to a 1D array, you get single numbers out. If applied to a 2D array, it combines
    over the second dimension. That is, if dimension is Ntimes x Nfrequencies, then the resulting
    spectra are Ntimes long.

    Parameters
    =======
    Y_f: ``pygwb.omega_spectrogram.OmegaSpectrogram``
        Point estimate spectrum.
    sigma_f: ``pygwb.omega_spectrogram.OmegaSpectrogram``
        Sigma spectrum.
    frequency_mask: ``array-like``, optional
        Boolean mask to apply to frequencies for the calculation. Default set to True including all frequencies.
    alpha: ``float``, optional
        Spectral index to use in case re-weighting is requested. Default set to None.
    fref: ``float``, optional
        Reference frequency to use in case re-weighting is requested. Default set to None.

    Returns
    =======
    Y: ``array-like`` or ``float``
        Point estimate or Point estimate spectrum.
    sigma: ``array-like`` or ``float``
        Point estimate standard deviation (theoretical) or spectrum of point estimate standard deviations.

    Notes
    -----
    If passing in spectrograms, the point estimate and sigma will be calculated per
    spectrum, without any time-averaging applied.
    Y_f and sigma_f can also be ``gwpy.spectrogram.Spectrogram`` objects, or numpy arrays. In these cases
    however the reweight functionality is not supported.
    """
    # Reweight in case one wants to pass it.
    if alpha is not None or fref is not None:
        Y_f.reweight(new_alpha=alpha, new_fref=fref)
        sigma_f.reweight(new_alpha=alpha, new_fref=fref)

    # now just strip off what we need...
    try:
        Y_f = np.real(Y_f.value)
        var_f = sigma_f.value ** 2
    except AttributeError:
        Y_f = np.real(Y_f)
        var_f = sigma_f ** 2

    if isinstance(frequency_mask, np.ndarray):
        pass
    elif frequency_mask == True:
        if len(Y_f.shape) == 1:
            frequency_mask = np.ones(Y_f.shape[0], dtype=bool)
        elif len(Y_f.shape) == 2:
            frequency_mask = np.ones(Y_f.shape[1], dtype=bool)
    if len(Y_f.shape) == 1 or Y_f.shape[0] == 1:
        if Y_f.shape[0] == 1:
            Y_f = Y_f[0]
            var_f = var_f[0]
        var = 1 / np.sum(var_f[frequency_mask] ** (-1), axis=-1).squeeze()
        Y = np.nansum(Y_f[frequency_mask] * (var / var_f[frequency_mask]), axis=-1)
    # need to make this nan-safe
    elif len(Y_f.shape) == 2:
        var = 1 / np.sum(var_f[:, frequency_mask] ** (-1), axis=-1).squeeze()
        Y = np.einsum(
            "tf, t -> t", Y_f[:, frequency_mask] / var_f[:, frequency_mask], var
        )
    else:
        raise ValueError("The input is neither a spectrum nor a spectrogram.")

    sigma = np.sqrt(var)

    return Y, sigma

def calculate_point_estimate_sigma_spectra(
    freqs,
    csd,
    avg_psd_1,
    avg_psd_2,
    orf,
    sample_rate,
    segment_duration,
    window_fftgram_dict={"window_fftgram": "hann"},
    overlap_factor=0.5,
    fref=25.0,
    alpha=0.0,
):
    """
    Calculate the Omega point estimate and associated sigma integrand,
    given a set of cross-spectral and power-spectral density spectrograms.
    This is particularly useful for statistical checks.

    If CSD is set to None, only returns variance.

    Parameters
    =======
    freqs: ``array_like``
        Frequencies associated to the spectrograms.
    csd: ``gwpy.spectrogram.Spectrogram``
        CSD spectrogram for detectors 1 and 2.
    avg_psd_1: ``gwpy.spectrogram.Spectrogram``
        Spectrogram of averaged PSDs for detector 1.
    avg_psd_2: ``gwpy.spectrogram.Spectrogram``
        Spectrogram of averaged PSDs for detector 2.
    orf: ``array_like``
        Overlap reduction function.
    sample_rate: ``float``
        Sampling rate of the data.
    segment_duration: ``float``
        Duration of each segment in seconds.
    window_fftgram_dict: ``dictionary``, optional
        Dictionary with window characteristics used in analysis segment estimation.
        Default is ``window_fftgram_dict={"window_fftgram": "hann"}``.
    overlap_factor: ``float``, optional
        Overlap factor used in analysis segment estimation. Default is 0.5.
    fref: ``float``, optional
        Reference frequency to use in the weighting calculation.
        Final result refers to this frequency.
    alpha: ``float``, optional
        Spectral index to use in the weighting.

    See also
    --------
    pygwb.util.window_factors
    """
    S_alpha = 3 * H0.si.value ** 2 / (10 * np.pi ** 2) / freqs ** 3
    S_alpha *= (freqs / fref) ** float(alpha)

    var_fs = (
        1
        / (2 * segment_duration * (freqs[1] - freqs[0]))
        * avg_psd_1
        * avg_psd_2
        / (orf ** 2 * S_alpha ** 2)
    )

    w1w2bar, w1w2squaredbar, _, _ = window_factors(
        int(sample_rate * segment_duration), window_fftgram_dict=window_fftgram_dict, overlap_factor=overlap_factor
    )
    var_fs = var_fs * w1w2squaredbar / w1w2bar ** 2
    if csd is not None:
        Y_fs = (csd) / (orf * S_alpha)
        return Y_fs, var_fs
    else:
        return var_fs

def combine_spectra_with_sigma_weights(main_spectra, weights_spectra):
    r"""
    Combine different statistically independent spectra :math:`S_i(f)` using spectral weights :math:`w_i(f)`, as

    .. math::

        S(f) = \frac{\sum_i \frac{S_i(f)}{w^2_i(f)}}{\sum_i \frac{1}{w^2_i(f)}},\,\,\,\, \sigma = \sqrt{\frac{1}{\sum_i \frac{1}{w^2_i(f)}}}.

    If main_spectra is 2D and has dimensions N_1 x N_2, final spectrum has dimension N_2
    (in contrast to ``calc_Y_sigma_from_Yf_sigmaf`` which combines across other dimension).

    Parameters
    =======
    main_spectra: ``np.ndarray``
        Array of arrays or FrequencySeries or OmegaSpectrum objects to be combined.
    weights_spectra: ``np.ndarray``
        Array of arrays or FrequencySeries or OmegaSpectrum objects to use as weights.

    Returns
    =======
    combined_weighted_spectrum: ``array_like``
        Final spectrum obtained combining the original spectra with given weights.
    combined_weights_spectrum: ``array_like``
        Variance associated to the final spectrum obtained combining the given weights.

    See also
    --------
    pygwb.omega_spectra.OmegaSpectrum
    
    pygwb.util._check_omegaspectra
    """
    if isinstance(main_spectra[0], OmegaSpectrum):
        _check_omegaspectra(main_spectra)

    if isinstance(weights_spectra[0], OmegaSpectrum):
        _check_omegaspectra(weights_spectra)

    w_spec = np.array(weights_spectra)
    m_spec = np.array(main_spectra)
    res_1 = 1 / np.nansum(1 / w_spec ** 2, axis=0)
    combined_weights_spectrum = np.sqrt(res_1)
    combined_weighted_spectrum = (
        np.nansum(m_spec / w_spec ** 2, axis=0) * res_1
    )
    if isinstance(main_spectra[0], OmegaSpectrum):
        combined_weighted_omegaspectrum = OmegaSpectrum(combined_weighted_spectrum, alpha=main_spectra[0].alpha, fref=main_spectra[0].fref, h0=main_spectra[0].h0, frequencies=main_spectra[0].frequencies, name='omega_spectrum')
        combined_weights_omegaspectrum = OmegaSpectrum(combined_weights_spectrum, alpha=main_spectra[0].alpha, fref=main_spectra[0].fref, h0=main_spectra[0].h0, frequencies=main_spectra[0].frequencies, name='sigma_spectrum')
        return combined_weighted_omegaspectrum, combined_weights_omegaspectrum
    else:
        return combined_weighted_spectrum, combined_weights_spectrum