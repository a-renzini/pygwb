import numpy as np
from scipy.signal import get_window, spectrogram

import gwpy.spectrogram


def fftgram(
    time_series_data, fftlength, overlap_length=0, zeropad=False, window_fftgram="hann"
):
    """Function that creates an fftgram from a timeseries

    Parameters
    ==========
    time_series_data: gwpy_timeseries
        Timeseries from which to compute the fftgram

    fftlength: int_length
        Length (in no. of data points) of each segment in which
        to compute an FFT

    overlap: int_length
        Length (in no.of data points) of overlap in calculating FFT
        (default 0 (no overlap))

    zeropadd: bool
        Whether to zero pad the data equal to the length of FFT or not
        (default False)

    window_fftgram: string_like
        Type of window to compute the Fast Fourier
        transform

    Returns
    =======
    fftgram: FFTgram
        FFTgram containing several psds (or csds)
    """

    sample_rate = int(1 / time_series_data.dt.value)
    window_fftgram = get_window(window_fftgram, fftlength * sample_rate, fftbins=False)

    if zeropad:
        f, t, Sxx = spectrogram(
            time_series_data.data,
            fs=sample_rate,
            window=window_fftgram,
            nperseg=fftlength * sample_rate,
            noverlap=overlap_length * sample_rate,
            nfft=2 * fftlength * sample_rate,
            mode="complex",
            detrend=False,
        )
    else:
        f, t, Sxx = spectrogram(
            time_series_data.data,
            fs=sample_rate,
            window=window_fftgram,
            nperseg=fftlength * sample_rate,
            noverlap=overlap_length * sample_rate,
            nfft=fftlength * sample_rate,
            mode="complex",
            detrend=False,
        )

    data_fftgram = gwpy.spectrogram.Spectrogram(
        Sxx.T, times=t + time_series_data.t0.value - (fftlength / 2), frequencies=f
    )

    return data_fftgram


def pwelch_psd(data, segment_duration, do_overlap=True):
    """
    Estimate PSD using pwelch method.

    Parameters
    ==========
    data: gwpy fftgram
        The data to coarse grain
    segment_duration: int
        segment duration over which PSD needs to be averaged
    do_overlap: bool
        Whether to calculate overlapping PSD spectrograms

    Returns
    =======
    gwpy spectrogram
        averaged over segments
    """

    averaging_factor = int(segment_duration / data.dt.value)
    if do_overlap:
        seg_indices = np.arange(1, len(data), int(averaging_factor / 2))
        seg_indices = seg_indices[seg_indices <= len(data) + 2 - averaging_factor]
    else:  # NOT CHECKED
        seg_indices = np.arange(1, len(data), averaging_factor)[0:-1]

    averaged = data[0 : len(seg_indices)].copy()  # temporary spectrogram
    kk = 0
    for ii in seg_indices - 1:
        averaged[kk] = data[ii : ii + 11].mean(axis=0)
        kk = kk + 1
    averaged.times = (
        averaged.epoch.value * data.times.unit + (seg_indices - 1) * data.dt
    )

    return np.real(averaged)


def before_after_average(psd_gram, segment_duration, psd_duration):
    """
    Average the first independent entry before and after for a specified time
    offset.

    Parameters
    ==========
    psd_gram: psd.spectrogram.Spectrogram
        The input spectrogram
    segment_duration: float
        The duration of data going into each analyzed segment.
    psd_duration: float
        The duration of data going into each PSD estimate.
        This should probably be an integer multiple of the segment duration
        but it might still work if not.
    """
    stride = psd_gram.dx.value
    overlap = segment_duration - stride
    strides_per_psd = int(np.ceil(psd_duration / stride))
    strides_per_segment = int(np.ceil(segment_duration / stride))
    time_offset = strides_per_psd * overlap * psd_gram.times.unit
    after_segment_offset = strides_per_psd + strides_per_segment

    output = psd_gram.copy()
    output = (output[:-after_segment_offset:] + output[after_segment_offset:]) / 2
    output.times = psd_gram.times[:-after_segment_offset] + time_offset

    return output


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
    coarsening_factor = int(coarsening_factor)
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
    coarsening_factor = int(coarsening_factor)
    n_remove = len(data) % coarsening_factor
    if n_remove > 0:
        data = data[:-n_remove]
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
    Coarsen a spectrogram in time and/or frequency, e.g., Welch averaging /
    coarse-graining

    The coarsening methods are either:
      - naive: this is equivalent to a Welch average
      - full: the full coarse-grain method
      - running_mean:

    Parameters
    ==========
    spectrogram: gwpy.spectrogram.Spectrogram
        Spectrogram object to be coarsened
    delta_t: float
        Output time spacing
    delta_f: float
        Output frequency spacing
    time_method: str
        Should be one of the coarsening methods listed above
    frequency_method: str
        Should be one of the coarsening methods listed above

    Returns
    =======
    output: gwpy.spectrogram.Spectrogram
    """
    from gwpy.spectrogram import Spectrogram

    methods = dict(
        naive=coarse_grain_naive,
        full=coarse_grain,
        running_mean=running_mean,
    )

    value = spectrogram.value

    if delta_t is not None:
        factor = delta_t / spectrogram.dt.value
        func = methods[time_method]
        value = np.apply_along_axis(func, axis=0, arr=value, coarsening_factor=factor)
        coarse_times = func(spectrogram.times.value, coarsening_factor=factor)
        coarse_times += spectrogram.times.value[0] - coarse_times[0]
    else:
        coarse_times = spectrogram.times

    if delta_f is not None:
        factor = delta_f / spectrogram.df.value
        func = methods[frequency_method]
        value = np.apply_along_axis(func, axis=1, arr=value, coarsening_factor=factor)
        coarse_frequencies = func(
            spectrogram.frequencies.value, coarsening_factor=factor
        )
    else:
        coarse_frequencies = spectrogram.frequencies

    output = Spectrogram(value, times=coarse_times, frequencies=coarse_frequencies)
    return output


def cross_spectral_density(
    time_series1,
    time_series2,
    segment_duration,
    frequency_resolution,
    do_overlap=False,
    overlap_factor=0.5,
    zeropad=True,
    window_fftgram="hann",
):
    """
    Compute the cross spectral density from two time series inputs

    Parameters
    ----------
    time_series1: array-like
    time_series2: array-like

    Returns
    -------
    gwpy spectrogram of cross spectral density
    """

    fft_gram_1 = fftgram(
        time_series1,
        segment_duration,
        overlap_length=segment_duration * overlap_factor * int(do_overlap),
        zeropad=zeropad,
        window_fftgram=window_fftgram,
    )
    fft_gram_2 = fftgram(
        time_series2,
        segment_duration,
        overlap_length=segment_duration * overlap_factor * int(do_overlap),
        zeropad=zeropad,
        window_fftgram=window_fftgram,
    )

    csd_spectrogram = coarse_grain_spectrogram(
        2 * np.conj(fft_gram_1) * fft_gram_2, delta_f=frequency_resolution
    )

    return csd_spectrogram


def power_spectral_density(
    time_series_data,
    segment_duration,
    frequency_resolution,
    do_overlap=True,
    overlap_factor=0.5,
    do_overlap_welch_psd=True,
    zeropad=False,
    window_fftgram="hann",
):
    """
    Compute the power spectral density of a time series using pwelch method

    Parameters
    ----------
    time_series_data: gwpy timeseries

    Returns
    -------
    gwpy spectrogram of power spectral density
    """

    fftlength = int(1.0 / frequency_resolution)
    fft_gram_data = fftgram(
        time_series_data,
        fftlength,
        overlap_length=fftlength * overlap_factor * int(do_overlap),
        zeropad=zeropad,
        window_fftgram=window_fftgram,
    )
    psd_spectrogram = pwelch_psd(
        2 * np.conj(fft_gram_data) * fft_gram_data,
        segment_duration,
        do_overlap=do_overlap_welch_psd,
    )

    return psd_spectrogram


def running_mean(data, coarsening_factor=1, axis=-1):
    """
    Compute the running mean of an array, this uses the default axis of numpy
    cumsum.

    Parameters
    ----------
    data: array-like
        Array of size M to be average
    coarsening_factor: int
        Number of segments to average, default=1
    axis:
        Axis to apply the mean over, default=-1

    Returns
    -------
    array-like: the averaged array of size M - coarsening factor
    """
    coarsening_factor = int(coarsening_factor)
    if axis != -1:
        data = np.swapaxes(axis, -1)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (
        np.swapaxes(cumsum[coarsening_factor:] - cumsum[:-coarsening_factor], axis, -1)
        / coarsening_factor
    )
