import numpy as np


def coarse_grain(data, coarsening_factor):
    """
    Coarse grain a frequency series by an integer factor.

    If the coarsening factor is even there are coarsening_factor + 1 entries
    in the input data that contribute to each coarse frequency bin, however,
    the first an last contribute only a half to the frequency below and half
    to the frequency above.

    If the coarsening factor is odd, there are no edge effects that have to be
    considered.

    The length of the output is len(data) // coarsening_factor - 1

    If the coarsening factor is not an integer, :code:`coarse_grain_exact` is
    used.

    Parameters
    ==========
    data: array-like
        The data to coarse grain
    coarsening_factor: float
        The factor by which to coarsen the data

    Returns
    =======
    array-like
        The coarse-grained data
    """
    if coarsening_factor == 1:
        return data
    elif coarsening_factor % 1 != 0:
        return coarse_grain_exact(data, coarsening_factor)
    elif coarsening_factor % 2:
        data = data[:-1]
    coarsened = coarse_grain_naive(
        data=data[coarsening_factor // 2 + 1 : -(coarsening_factor // 2)],
        coarsening_factor=coarsening_factor,
    )
    if not coarsening_factor % 2:
        left_edges = data[coarsening_factor // 2 :: coarsening_factor][:-1]
        right_edges = data[int(coarsening_factor * 1.5) :: coarsening_factor]
        coarsened += (left_edges - right_edges) / 2 / coarsening_factor
    return coarsened


def coarse_grain_exact(data, coarsening_factor):
    """
    Coarse grain an array using any coarsening factor

    Each bin will contain the integral of the input array covering
    `coarsening_factor` bins.

    This is done by evaluating the difference between the cumulative integral
    of the data at the beginning and end of each bin.

    The i'th bin covers
    `[coarsening_factor * (ii - 0.5), coarsening_factor * (ii + 0.5)]`
    indexed for `1 <= ii < len(data) / coarsening_factor`.

    Parameters
    ==========
    data: array-like
        The data to coarse grain
    coarsening_factor: float
        The factor by which to coarsen the data

    Returns
    =======
    array-like
        The coarse-grained data
    """
    from scipy.integrate import cumtrapz

    n_input = len(data)
    first_full_bin_start = coarsening_factor / 2

    x_inputs = np.arange(n_input)
    x_values = np.arange(first_full_bin_start, n_input, coarsening_factor)

    cumulative_y = cumtrapz(data, x_inputs, initial=0)
    y_values = np.interp(x_values, x_inputs, cumulative_y)
    output = np.diff(y_values) / coarsening_factor

    return output


def coarse_grain_naive(data, coarsening_factor):
    """
    Naive implementation of a coarse graining factor that ignores edge effects.

    This is equivalent to the process performed for a Welch average

    Parameters
    ==========
    data: array-like
        The data to coarse grain
    coarsening_factor: int
        The factor by which to coarsen the data

    Returns
    =======
    array-like
        The coarse-grained data
    """
    coarsened = np.mean(data.reshape(-1, coarsening_factor), axis=-1)
    return coarsened


def coarse_grain_spectrogram(
    spectrogram,
    delta_t=None,
    delta_f=None,
    time_method="naive",
    frequency_method="full",
):
    """
    Coarsen a spectrogram in time and/or frequency

    The coarsening methods are either:
      - naive: this is equivalent to a Welch average
      - full: the full coarse-grain method

    Parameters
    ==========
    spectrogram: gwpy.spectrogram.Spectrogram
    delta_t: float
    delta_f: float
    time_method: str
    frequency_method: str

    Returns
    =======
    output: gwpy.spectrogram.Spectrogram
    """
    from gwpy.spectrogram import Spectrogram

    methods = dict(
        naive=coarse_grain_naive,
        full=coarse_grain,
    )

    value = spectrogram.value

    if delta_t is not None:
        factor = delta_t / spectrogram.dt.value
        func = methods[time_method]
        value = np.apply_along_axis(func, axis=0, arr=value, coarsening_factor=factor)
        coarse_times = func(spectrogram.times, coarsening_factor=factor)
    else:
        coarse_times = spectrogram.times

    if delta_f is not None:
        factor = delta_f / spectrogram.df.value
        func = methods[frequency_method]
        value = np.apply_along_axis(func, axis=1, arr=value, coarsening_factor=factor)
        coarse_frequencies = func(spectrogram.frequencies, coarsening_factor=factor)
    else:
        coarse_frequencies = spectrogram.frequencies

    output = Spectrogram(value, times=coarse_times, frequencies=coarse_frequencies)
    return output


def density(fft_gram_1, fft_gram_2):
    """
    Compute the spectral density from two inputs

    Parameters
    ----------
    fft_gram_1: array-like
    fft_gram_2: array-like

    Returns
    -------
    array-like
        The spectral density
    """
    return np.conj(fft_gram_1) * fft_gram_2