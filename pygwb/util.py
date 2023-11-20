"""
The ``util`` module combines miscellaneous functions used in several parts of the ``pygwb`` package.
These functions mainly perform small computations, necessary at multiple stages of the analysis.
"""
import copy

import gwpy
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import get_window

from pygwb.constants import H0


def parse_window_dict(window_dict):
    """
    Parse the window dictionary properly for scipy compatibility.
    
    Parameters
    =======
    window_dict: ``dictionary``
        Dictionary containing the window characteristics.
    """
    bools = ['sym', 'norm']
    floats = ['center', 'tau', 'alpha', 'beta', 'nbar', 'sll', 'std', 'p', 'sig', 'at']
    for key in window_dict:
        if key in floats:
            window_dict[key] = float(window_dict[key])
        elif key in bools:
            window_dict[key] = bool(window_dict[key])
        else:
            pass
    return window_dict

def window_factors(N, window_fftgram_dict={"window_fftgram": "hann"}, overlap_factor=0.5):
    """
    Calculate window factors. By default for a hann window with 50% overlap.

    Parameters
    =======
    N: ``int``
        Length of the window.
    window_fftgram_dict: ``dictionary``, optional
        Dictionary with window characteristics. Default is ``window_fftgram_dict={"window_fftgram": "hann"}``
    overlap_factor: ``float``, optional
        Defines the overlap between consecutive data chunks used in the calculation. Default is 0.5.        

    Returns
    =======
    This functions returns the four window factors which correct the analysis for the effect of the window factor.
    w1w2bar: ``float``
    w1w2squaredbar: ``float``
    w1w2ovlbar: ``float``
    w1w2squaredovlbar: ``float``

    See also
    --------
    scipy.signal.get_window
        More information `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html>`_.
    """
    window_tuple = get_window_tuple(window_fftgram_dict)
    w = get_window(window_tuple, N, fftbins=False)
    w1w2bar = np.mean(w ** 2)
    w1w2squaredbar = np.mean(w ** 4)

    S = N - int(overlap_factor*N)
    
    if overlap_factor == 0.0 or overlap_factor == 0:
        w1w2squaredovlbar = 0.0
        w1w2ovlbar = 0.0
    else:
        w1w2squaredovlbar = 1 / (N*overlap_factor) * np.sum(w[0:N-S]**2*w[S:N]**2)
        w1w2ovlbar = 1 / (N*overlap_factor) * np.sum(w[0:N-S]*w[S:N])

    return w1w2bar, w1w2squaredbar, w1w2ovlbar, w1w2squaredovlbar

def get_window_tuple(window_fftgram_dict={"window_fftgram": "hann"}):
    """
    Unpack the ``window_fft_dict`` dictionary into a ``tuple`` that can be read by ``scipy.signal.get_window``.

    Parameters
    =======
    window_fftgram_dict: ``dictionary``, optional
        Dictionary with window characteristics. Default is ``window_fftgram_dict={"window_fftgram": "hann"}``.

    Returns
    =======
    window_tuple: ``tuple``
        A tuple containing the window_fft name as the first entry, followed by optional entries of the window_fft_dict.

    Notes
    -----
    ``window_fftgram_dict`` is expected to have at least one item, ``window_fftgram``.
    """
    window_dict = copy.deepcopy(window_fftgram_dict)
    out = tuple([window_dict["window_fftgram"]])
    window_dict.pop("window_fftgram")
    for name in window_dict:
        if name != "sym":
            out += tuple([window_dict[name]])
    if "sym" in window_fftgram_dict:
        out += tuple([window_dict["sym"]])
    return out

def calc_rho1(N, window_fftgram_dict={"window_fftgram": "hann"}, overlap_factor=0.5):
    """
    Calculate the combined window factor rho.

    Parameters
    =======
    N: ``int``
        Length of the window.
    window_fftgram_dict: ``dictionary``, optional
        Dictionary with window characteristics. Default is ``window_fftgram_dict={"window_fftgram": "hann"}``.
    overlap_factor: ``float``, optional
        Defines the overlap between consecutive data chunks used in the calculation. Default is 0.5.

    Returns
    =======
    rho1: ``float``
        The combined window factor.
    """
    w1w2bar, _, w1w2ovlbar, _ = window_factors(N, window_fftgram_dict, overlap_factor=overlap_factor)
    rho1 = (overlap_factor * w1w2ovlbar / w1w2bar) ** 2
    return rho1

def calc_rho(N, j, window_tuple="hann", overlap_factor=0.5):
    """
    Calculate the normalised correlation of a window with itself shifted ``j`` times. This is identical
    to the ``rho(j)`` from Welch (1967).

    Parameters
    =======
    N: ``int``
        Length of the window.
    j: ``int``
        Number of "shifts" to apply to the window when correlating with itself.
    window_tuple: ``str`` or ``tuple``, optional
        Window name or tuple as used in ``scipy.signal.get_window()``. Default is ``window_tuple="hann"``.
    overlap_factor: ``float``, optional
        Defines the overlap between consecutive segments used in the calculation. Default is 0.5.

    Returns
    =======
    rho: ``float``
        The normalised window correlation rho(j).

    See also
    --------
    scipy.signal.get_window
        More information `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html>`_.
    """
    # The base window for which we want to calculate the correlation
    w = get_window(window_tuple, N, fftbins=False)
    
    # S is the shift, that is, the number of samples by which the window is shifted from the base window
    S = N - int(overlap_factor*N)
    
    if (j*S < N):
        rho = (np.sum(w[0:N-j*S]*w[j*S:N])/sum(w*w))**2
    else:
        # j*S >= N so no overlap
        rho = 0

    return rho

def effective_welch_averages(nSamples, N, window_tuple="hann", overlap_factor=0.5):
    """
    Calculate the "effective" number of averages used in Welch's PSD estimate after taking into account windowing
    and overlap.

    Parameters
    =======
    nSamples: ``int``
        Number of samples to be used to estimate the PSD.
    N: ``int``
        Length of the window.
    window_tuple: ``str`` or ``tuple``, optional
        Window name or tuple as used in ``scipy.signal.get_window()``. Default is ``window_tuple="hann"``.
    overlap_factor: ``float``, optional
        Defines the overlap between consecutive segments used in the calculation. Default is 0.5.

    Returns
    =======
    Neff: ``float``
        The effective number of averages.
    """
    # S is the shift, that is, the number of samples by which the window is shifted from the base window
    S = N - int(overlap_factor*N)
    
    # K is the number of segments that will be averaged in the corresponding Welch estimate
    K = 1 + int((nSamples - N)/S)

    # Form the weighted sum of the window correlations for shifts j = 1 to K-1
    rho_sum = 0
    for j in range(1, K):
        rho_sum += (K - j)/K*calc_rho(N, j, window_tuple=window_tuple, overlap_factor=overlap_factor)
    
    Neff = K/(1 + 2 * rho_sum)

    return Neff

def calc_bias(
    segmentDuration,
    deltaF,
    deltaT,
    N_avg_segs=2,
    window_fftgram_dict={"window_fftgram": "hann"},
    overlap_factor=0.5
):
    """
    Calculate the bias factor introduced by Welch averaging.

    Parameters
    =======
    segmentDuration: ``float``
        Duration in seconds of Welch-averaged segment.
    deltaF: ``float``
        Frequency resolution (in Hz) of Welch-averaged segment.
    deltaT: ``float``
        Time sampling of Welch-averaged segment (in seconds).
    N_avg_segs: ``int``, optional
        Number of segments over which the average is performed.
    window_fftgram_dict: ``dictionary``, optional
        Dictionary with window characteristics. Default is ``window_fftgram_dict={"window_fftgram": "hann"}``.
    overlap_factor: ``float``, optional
        Defines the overlap between consecutive data chunks used in the calculation. Default is 0.5.

    Returns
    =======
    bias: ``float``
        The bias factor.
    """
    # Number of samples in a data chunk
    nSamples = int(segmentDuration / deltaT)

    # Number of samples in the window used for Welch's estimate
    N = int(1 / (deltaT * deltaF))

    # Effective number of segments that are averaged for this windowing scheme
    window_tuple = get_window_tuple(window_fftgram_dict)
    Neff = effective_welch_averages(nSamples, N, window_tuple, overlap_factor=overlap_factor)

    # if Neff is the same as the true number of segments, no bias.
    if Neff == (nSamples/N):
        bias = 1.0

    else:
        # Correction for number of PSDs that will be averaged 
        Neff = N_avg_segs * Neff
        bias = Neff / (Neff - 1)

    return bias

def omega_to_power(omega_GWB, frequencies):
    """
    Compute the GW power spectrum starting from the :math:`\Omega`\ :sub:`GWB`\
    spectrum. For more information, see https://arxiv.org/pdf/2303.15696.pdf.

    Parameters
    =======
    omega_GWB: ``array_like``
        The omega spectrum to turn into strain power.
    frequencies: ``array_like``
        Array of frequencies (in Hz) corresponding to the omega spectrum.

    Returns
    =======
    power: ``gwpy.frequencyseries.FrequencySeries``
        A gwpy FrequencySeries containing the GW power spectrum

    Notes
    -----
    The given frequencies need to match the given spectrum.
    """
    H_theor = (3 * H0.si.value ** 2) / (10 * np.pi ** 2)

    power = H_theor * omega_GWB * frequencies ** (-3)
    power = gwpy.frequencyseries.FrequencySeries(power, frequencies=frequencies)

    return power

def interpolate_frequency_series(fSeries, new_frequencies):
    """
    Interpolate a frequency series, given a new set of frequencies.

    Parameters
    =======
    fSeries: ``gwpy.frequencyseries.FrequencySeries``
        The FrequencySeries to interpolate.
    new_frequencies: ``array_like``
        The new set of frequencies to interpolate to.

    Returns
    =======
    fSeries_new: ``gwpy.frequencyseries.FrequencySeries``
        The interpolated FrequencySeries.

    See also
    --------
    scipy.interpolate.interp1d
        More information `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.
    """
    spectrum = fSeries.value
    frequencies = fSeries.frequencies.value

    spectrum_func = interp1d(
        frequencies, spectrum, kind="cubic", fill_value="extrapolate"
    )

    return gwpy.frequencyseries.FrequencySeries(
        spectrum_func(new_frequencies), frequencies=new_frequencies
    )

def StatKS(DKS):
    """
    Compute the Kolgomorov-Smirnov test.
    
    Parameters:
    =======
    DKS: ``float``
        DKS input value.
    
    Returns
    =======
    pvalue: ``float``
        The p-value for the KS test.
    """
    jmax = 500
    pvalue = 0.0
    for jj in np.arange(1, jmax + 1):
        pvalue += 2.0 * (-1) ** (jj + 1) * np.exp(-2.0 * jj ** 2 * DKS ** 2)
    return pvalue

def _check_omegaspectra(spectra):
    for spec in spectra:
        if spec.alpha != spectra[0].alpha:
            print(spec.alpha, spectra[0].alpha)
            raise ValueError(
                'spectra in this set have been weighted with different alphas. Please correct this before continuing.'
            )
        if spec.fref != spectra[0].fref:
            raise ValueError(
                'spectra in this set have been set at different reference frequencies. '
                'Please correct this before continuing.'
            )
        if spec.h0 != spectra[0].h0:
            raise ValueError(
                'spectra in this set have been set at different h0. Please correct this before continuing.'
            )
        if not np.allclose(spec.frequencies.value, spectra[0].frequencies.value):
            raise ValueError(
                'spectra in this set have different frequencies. Please correct this before continuing.'
            )
