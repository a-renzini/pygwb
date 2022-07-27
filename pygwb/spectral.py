import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram
from gwpy.timeseries import TimeSeries
from scipy.signal import get_window, spectrogram

from pygwb.util import get_window_tuple


def fftgram(
    time_series_data: TimeSeries,
    fftlength: int,
    overlap_factor: float=0.5,
    zeropad: bool=False,
    window_fftgram_dict: dict={"window_fftgram": "hann"},
):
    """Create an fftgram from a timeseries

    Parameters
    ----------
    time_series_data: gwpy timeseries
        Timeseries from which to compute the fftgram.
    fftlength: int
        Length of each segment (in seconds) for computing FFT.
    overlap_factor: float, optional
        Factor of overlap between adjacent FFT segments (values range from 0 
        to 1). Users should provide proper combination of overlap_factor and
        window_fftgram_dict. For \"hann\" window use 0.5 overlap_factor and 
        for \"boxcar"\ window use 0 overlap_factor. Default 0.5 (50% overlap).
    zeropadd: bool, optional
        Before doing FFT whether to zero pad the data equal to the length of 
        FFT or not. Default is False.
    window_fftgram_dict: dictionary, optional
        Dictionary containing name and parameters describing which window to 
        use for producing FFTs. Default is \"hann\".

    Returns
    -------
    data_fftgram: gwpy spectrogram (complex)
        fftgram containing several FFTs in a matrix format
    """
    
    sample_rate = int(1 / time_series_data.dt.value)

    # get the window function
    window_tuple = get_window_tuple(window_fftgram_dict)
    window_fftgram = get_window(window_tuple, fftlength * sample_rate, 
                                fftbins=False)

    # calculate the spectrogram using scipy.signal.spectrogram
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

    # convert the above spectrogram into gwpy spectrogram object
    ffts = []
    for ii in range(len(t)):
        ffts.append(FrequencySeries(Sxx.T[ii], f0= f[0], df = f[1]-f[0]))

    data_fftgram = Spectrogram.from_spectra(*ffts, epoch=time_series_data.t0, 
                                            dt=t[1]-t[0])

    return data_fftgram


def pwelch_psd(
    psdgram: Spectrogram, 
    segment_duration: int, 
    overlap_factor: float = 0.5):

    """
    Estimate PSD using pwelch method.

    Parameters
    ==========
    psdgram: gwpy spectrogram (PSD)
       PSD gram data to be averaged
    segment_duration: int
        Data duration over which PSDs need to be averaged. Should be greater 
        than or equal to the duration used for FFT. 
    overlap_factor: float, optional
        Amount of overlap between adjacent average PSDs, can vary between 0 
        and 1. This factor should be same as the one used for CSD estimation. 
        Default is 0.5.

    Returns
    =======
    avg_psdgram: gwpy psd spectrogram
        averaged over segments within the segment_duration
    """
    
    averaging_factor = round(segment_duration * psdgram.dy.value)

    if averaging_factor < 1:    
        raise ValueError('''Segment_duration should be greater than the FFT
                        duration used for PSD calculation''')
    elif averaging_factor == 1: # nothing to average in this case
        avg_psdgram = np.real(psdgram)
    else:
        # total duration of the original time series
        job_duration = (psdgram.xindex.value[-1] +
                        1/psdgram.dy.value - psdgram.xindex.value[0])
        # possible number of (overlapping) segments
        stride = segment_duration*(1-overlap_factor)
        n_segments = int((job_duration - overlap_factor* segment_duration)/
                         stride)
        
        avg_psds = []        
        segments_start_times = np.zeros(n_segments)
        start_time = psdgram.xindex.value[0]
        for ii in range(n_segments):
            seg_indices = ((psdgram.xindex.value >= start_time) & 
                            (psdgram.xindex.value <= 
                             (start_time + segment_duration - 
                              1 / psdgram.dy.value)))
            avg_psds.append(psdgram[seg_indices].mean(axis=0))
            segments_start_times[ii] = start_time
            # move to next (overlapping) segment
            start_time = start_time + stride

        avg_psdgram = Spectrogram.from_spectra(*avg_psds, 
                             epoch=segments_start_times[0], 
                             dt=stride)

    return np.real(avg_psdgram)


def before_after_average(
    psdgram: Spectrogram, 
    segment_duration: int, 
    N_avg_segs: int):

    """
    Average the requested number of PSDs from segments adjacent to the segment 
    of interest (for which CDS is calculated)

    Parameters
    ----------
    psdgram: gwpy spectrogram (PSD)
        PSD spectrogram.
    segment_duration: int
        Duration of data used for each PSD calculation.
    N_avg_segs: int
        Number of segments to be used for PSD averaging (from both sides of 
        the segment of interest). N_avg_segs should be even and >= 2

    Returns
    -------
    avg_psdgram: gwpy spectrogram 
        averaged psd gram
    """

    if N_avg_segs < 2:
        raise ValueError('N_avg_segs should be >=2')

    if (N_avg_segs % 2) != 0:
        raise ValueError('N_avg_segs should be even')

    if ((psdgram.xindex.value[-1] - psdgram.xindex.value[0]) < 
        (N_avg_segs * segment_duration)):
        max_N_avg_segs = round((psdgram.xindex.value[-1] - 
                                psdgram.xindex.value[0])/segment_duration)
        raise ValueError(f''' Input (PSD) spectrogram does not have enough 
                           segments to be used with given N_avg_segs. Here 
                           N_avg_segs should be less than or equal to
                           {max_N_avg_segs}.''')

    # segments for which average PSDs need to be calculcated
    seg_times = [value for value in psdgram.xindex.value 
                 if  (((value - psdgram.xindex.value[0]) >= 
                       (N_avg_segs / 2 * segment_duration)) &
                      ((psdgram.xindex.value[-1] - value) >= 
                       (N_avg_segs / 2 * segment_duration)))]

    avg_psds = [] 
    for ii in seg_times:
        seg_diffs = abs(psdgram.xindex.value - ii) / segment_duration
        seg_indices = ((seg_diffs <= (N_avg_segs / 2)) & 
                       ((seg_diffs % 1) == 0) & 
                       ((psdgram.xindex.value - ii) !=0))
        # Below we only average non-overlapping segments (this is a choice)
        avg_psds.append(psdgram[seg_indices].mean(axis=0))
        
    avg_psdgram = Spectrogram.from_spectra(*avg_psds, 
                             epoch=seg_times[0], 
                             dt=psdgram.dx.value)    
    return avg_psdgram


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
        value = np.apply_along_axis(func, axis=0, arr=value, 
                                    coarsening_factor=factor)
        coarse_times = func(spectrogram.times.value, coarsening_factor=factor)
        coarse_times += spectrogram.times.value[0] - coarse_times[0]
    else:
        coarse_times = spectrogram.times

    if delta_f is not None:
        factor = delta_f / spectrogram.df.value
        func = methods[frequency_method]
        value = np.apply_along_axis(func, axis=1, arr=value, 
                                    coarsening_factor=factor)
        coarse_frequencies = func(
            spectrogram.frequencies.value, coarsening_factor=factor
        )
    else:
        coarse_frequencies = spectrogram.frequencies
        
    # create gwpy spectrogram object
    specgrams = []
    # to avoid precision issues
    deltaF = (coarse_frequencies[10] - coarse_frequencies[0])/10 
    for ii in value:
        specgrams.append(FrequencySeries(ii, f0=coarse_frequencies[0], 
                    df = deltaF))
        
    output = Spectrogram.from_spectra(*specgrams, epoch=coarse_times[0].value, 
                            dt=coarse_times[1].value-coarse_times[0].value)
    
    return output


def cross_spectral_density(
    time_series_data1:  TimeSeries,
    time_series_data2:  TimeSeries,
    segment_duration: int,
    frequency_resolution: float,
    overlap_factor: float = 0.5,
    zeropad: bool = False,
    window_fftgram_dict: dict={"window_fftgram": "hann"},
):
    """
    Compute the cross spectral density from two time series inputs

    Parameters
    ----------
    time_series_data1: gwpy timeseries
        Timeseries data of detector1.
    time_series_data2: gwpy timeseries
        Timeseries data of detector2.
    segment duration: int
        data duration over which CSDs need to be calculated.
    frequency_resolution: float
        Frequency resolution of the final CSDs. This is achieved by averaing in
        frequency domain.
    overlap_factor: float, optional
        Amount of overlap between adjacent segments (range between 0 and 1)
        This factor should be same as the one used for power_spectral_density.
        Users should provide proper combination of overlap_factor and
        window_fftgram_dict. For \"hann\" window use 0.5 overlap_factor and for \"boxcar"\ 
        window use 0 overlap_factor. Default id 0.5 (50% overlap).
    zeropadd: bool, optional
        Before doing FFT whether to zero pad the data equal to the length of 
        FFT or not. Default is False.
    window_fftgram_dict: dictionary, optional
        Dictionary containing name and parameters describing which window to 
        use for producing FFTs. Default is \"hann\".

    Returns
    -------
    csd_spectrogram: gwpy spectrogram
       Cross spectral density of the two timeseries
    """

    # Check if the lengths of two time-series are equal
    if len(time_series_data1.data) != len(time_series_data2.data):
        raise ValueError('Lengths of two input time series are not equal')
        
    # Check if the sample rates of two input time-series are equal
    if time_series_data1.dt.value != time_series_data2.dt.value:
        raise ValueError('Sample rates of two input time series are not equal')
    
    fft_gram_1 = fftgram(
        time_series_data1,
        segment_duration,
        overlap_factor=overlap_factor,
        zeropad=zeropad,
        window_fftgram_dict=window_fftgram_dict,
    )
    fft_gram_2 = fftgram(
        time_series_data2,
        segment_duration,
        overlap_factor=overlap_factor,
        zeropad=zeropad,
        window_fftgram_dict=window_fftgram_dict,
    )

    csd_spectrogram = coarse_grain_spectrogram(
        2 * np.conj(fft_gram_1) * fft_gram_2, delta_f=frequency_resolution
    )

    # Correct the scaling of DC and nyquist frequency components in agreement 
    # with scipy.signal.welch (and pwelch in matlab)
    for ii in range(len(csd_spectrogram)):
        csd_spectrogram[ii].value[0] = csd_spectrogram[ii].value[0]/2
        csd_spectrogram[ii].value[-1] = csd_spectrogram[ii].value[-1]/2
       
    return csd_spectrogram


def power_spectral_density(
    time_series_data:  TimeSeries,
    segment_duration:  TimeSeries,
    frequency_resolution: float,
    overlap_factor: float = 0.5,
    window_fftgram_dict: dict = {"window_fftgram": "hann"},
):
    """
    Compute the PSDs of every segment (defined by the segment duration)
    in the time series using pwelch method.

    Parameters
    ----------
    time_series_data: gwpy timeseries
        Timeseries from which to compute PSDs.
    segment duration: int
        Data duration over which each PSDs need to be calculated.
    frequency_resolution: float
        Frequency resolution of the final PSDs. This sets the time duration
        over which FFTs are calculated in the pwelch method.
    overlap_factor: float, optional
        Amount of overlap between adjacent segments (range between 0 and 1).
        This factor should be same as the one used for cross_spectral_density.
        Users should provide proper combination of overlap_factor and
        window_fftgram_dict. For \"hann\" window use 0.5 overlap_factor and 
        for \"boxcar"\ window use 0 overlap_factor. Default is 0.5 (50% overlap)
    window_fftgram_dict: dictionary, optional
        Dictionary containing name and parameters describing which window to 
        use for producing FFTs. Default is \"hann\".
     
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
        window_fftgram_dict=window_fftgram_dict,
    )

    # Use pwelch method (averaging) to get PSDs for each segment duration 
    # of data
    psd_spectrogram = pwelch_psd(
        2 * np.conj(fft_gram_data) * fft_gram_data,
        segment_duration,
        overlap_factor=overlap_factor,
    )

    # Correct the scaling of DC and nyquist frequency components in agreement  
    # with scipy.signal.welch (and pwelch in matlab)
    for ii in range(len(psd_spectrogram)):
        psd_spectrogram[ii].value[0] = psd_spectrogram[ii].value[0]/2
        psd_spectrogram[ii].value[-1] = psd_spectrogram[ii].value[-1]/2
       
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
        np.swapaxes(cumsum[coarsening_factor:] - cumsum[:-coarsening_factor], 
                    axis, -1)/ coarsening_factor
    )
