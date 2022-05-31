import gwpy.spectrogram
import numpy as np
from scipy.signal import get_window, spectrogram

from pygwb.util import get_window_tuple


def fftgram(
    time_series_data,
    fftlength,
    overlap_factor=0,
    zeropad=False,
    window_fftgram="boxcar",
):
    """Function that creates an fftgram from a timeseries

    Parameters
    ----------
    time_series_data: gwpy_timeseries
        Timeseries from which to compute the fftgram
    fftlength: int
        Length of each segment (in seconds) for which
        to compute an FFT
    overlap_factor: float
        Factor of overlap between adjacent FFT segments (values range from 0 to 1)
        (default 0 (no overlap))
    zeropadd: bool
        Whether to zero pad the data equal to the length of FFT or not
        (default False)
    window_fftgram: str
        Type of window to use for FFT (default no window)

    Returns
    -------
    data_fftgram: gwpy fftgram (complex)
        fftgram containing several PSDs (or CSDs) in a matrix format
    """

    sample_rate = int(1 / time_series_data.dt.value)
    window_fftgram = get_window(window_fftgram, fftlength * sample_rate, fftbins=False)

    if zeropad:
        f, t, Sxx = spectrogram(
            time_series_data.data,
            fs=sample_rate,
            window=window_fftgram,
            nperseg=fftlength * sample_rate,
            noverlap=overlap_factor * fftlength * sample_rate,
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
            noverlap=overlap_factor * fftlength * sample_rate,
            nfft=fftlength * sample_rate,
            mode="complex",
            detrend=False,
        )

    data_fftgram = gwpy.spectrogram.Spectrogram(
        Sxx.T,
        times=t + time_series_data.t0.value - (fftlength / 2),
        frequencies=f,  # - (fftlength / 2)
    )

    return data_fftgram


def pwelch_psd(data_fftgram, segment_duration, overlap_factor=0):
    """
    Estimate PSD using pwelch method.

    Parameters
    ==========
    data_fftgram: gwpy fftgram (complex)
        fft gram data to be averaged
    segment_duration: int
        data duration over which PSDs need to be averaged;
        should be greater than or equal to the duration used for FFT
    overlap_factor: float
        Amount of overlap between adjacent average PSDs, can vary between 0 and 1 (default 0);
        This factor should be same as the one used for CSD estimation

    Returns
    =======
    averaged_psd: gwpy psd spectrogram
        averaged over segments
    """

    averaging_factor = round(segment_duration / data_fftgram.dt.value)
    if overlap_factor == 0:  # no overlap (TODO : Check whether this works)
        seg_indices = np.arange(1, len(data_fftgram), averaging_factor)[0:-1]
    else:  # overlapping segments (TODO : Check whether it works for overlap_factor!=0.5)
        seg_indices = np.arange(
            1, len(data_fftgram), round(averaging_factor * overlap_factor)
        )
        seg_indices = seg_indices[
            seg_indices <= len(data_fftgram) + 2 - averaging_factor
        ]

    averaged_psd = data_fftgram[0 : len(seg_indices)].copy()  # temporary spectrogram
    kk = 0
    for ii in seg_indices - 1:
        averaged_psd[kk] = data_fftgram[ii : ii + 11].mean(axis=0)
        kk = kk + 1
    averaged_psd.times = (
        averaged_psd.epoch.value * data_fftgram.times.unit
        + (seg_indices - 1) * data_fftgram.dt
    )

    return np.real(averaged_psd)


def before_after_average(psd_gram, segment_duration, N_avg_segs):
    """
    Average the requested number of PSDs from segments adjacent to the segment of interest
    (for which CDS is calculated)

    Parameters
    ----------
    psd_gram: gwpy psd spectrogram
        Input spectrogram
    segment_duration: float
        Duration of data going into each analyzed segment.
    N_avg_segs: int
        Number of segments used for PSD averaging (from both sides of the segment of interest)
        N_avg_segs should be even and >= 2

    Returns
    -------
    avg_psd: averaged psd gram
    """
    # TODO: Raise exception when N_avg_segs is not >=2 and even
    stride = round(psd_gram.dx.value)
    overlap = segment_duration - stride
    # TODO: Check whether the below conditions work when (segment_duration / stride) is not an integer
    strides_per_segment = int(np.ceil(segment_duration / stride))
    strides_per_psd = int(N_avg_segs / 2) * strides_per_segment
    no_of_strides_oneside = strides_per_psd + strides_per_segment

    avg_psd = psd_gram.copy()
    # TODO: Check whether this works for N_avg_seg >2
    # TODO: Resolve the issue with 3 segments case
    avg_psd = (
        avg_psd[:-no_of_strides_oneside] + avg_psd[no_of_strides_oneside:]
    ) / N_avg_segs
    # properly set the start times of the averaged PSDs
    time_offset = (N_avg_segs / 2) * segment_duration * psd_gram.times.unit
    avg_psd.times = psd_gram.times[:-no_of_strides_oneside] + time_offset

    return avg_psd


def coarse_grain(data, coarsening_factor):
    """
    Coarse grain a frequency series by an integer factor.

    If the coarsening factor is even, there are coarsening_factor + 1 entries
    in the input data that contribute to each coarse frequency bin, however,
    the first and last contribute only a half to the frequency below and half
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
    coarsened: array-like
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
    output: array-like
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
    coarsened:array-like
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
    time_series_data1,
    time_series_data2,
    segment_duration,
    frequency_resolution,
    overlap_factor=0,
    zeropad=False,
    window_fftgram="boxcar",
):
    """
    Compute the cross spectral density from two time series inputs

    Parameters
    ----------
    time_series_data1: gwpy timeseries
        Timeseries data of detector1
    time_series_data2: gwpy timeseries
        Timeseries data of detector2
    segment duration: int
        data duration over which CSDs need to be calculated
    frequency_resolution: float
        Frequency resolution of the final CSDs; This is achieved by averaing in
        frequency domain
    overlap_factor: float, optional
        Amount of overlap between adjacent segments (range between 0 and 1)
        This factor should be same as the one used for power_spectral_density
        (default 0, no overlap)
    zeropadd: bool, optional
        Whether to zero pad the data equal to the length of FFT used
        (default False)
    window_fftgram: str, optional
        Type of window to use for FFT (default no window)

    Returns
    -------
    csd_spectrogram: gwpy spectrogram
       cross spectral density of the two timeseries
    """

    fft_gram_1 = fftgram(
        time_series_data1,
        segment_duration,
        overlap_factor=overlap_factor,
        zeropad=zeropad,
        window_fftgram=window_fftgram,
    )
    fft_gram_2 = fftgram(
        time_series_data2,
        segment_duration,
        overlap_factor=overlap_factor,
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
    overlap_factor=0,
    window_fftgram="boxcar",
):
    """
    Compute the PSDs of every segment (defined by the segment duration)
    in the time series using pwelch method

    Parameters
    ----------
    time_series_data: gwpy timeseries
        Timeseries from which to compute PSDs
    segment duration: int
        data duration over which each PSDs need to be calculated
    frequency_resolution: float
        Frequency resolution of the final PSDs; This sets the time duration
        over which FFTs are calculated in the pwelch method
    overlap_factor: float, optional
        Amount of overlap between adjacent segments (range between 0 and 1)
        This factor should be same as the one used for cross_spectral_density
        (default 0, no overlap)
    window_fftgram: str, optional
        Type of window to use for FFT (default no window)

    Returns
    -------
    psd_spectrogram: gwpy PSD spectrogram
        PSD spectrogram with each PSD duration equal to segment duration
    """

    # Length of data blocks to be used in pwelch
    fftlength = int(1.0 / frequency_resolution)

    # No zero-pad is used in the PSD estimation
    fft_gram_data = fftgram(
        time_series_data,
        fftlength,
        overlap_factor=overlap_factor,
        zeropad=False,
        window_fftgram=window_fftgram,
    )

    # Use pwelch method (averaging) to get PSDs for each segment duration of data
    psd_spectrogram = pwelch_psd(
        2 * np.conj(fft_gram_data) * fft_gram_data,
        segment_duration,
        overlap_factor=overlap_factor,
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


