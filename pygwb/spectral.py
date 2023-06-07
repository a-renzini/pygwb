"""Spectral module contains all functions in the pygwb package that are related to the computation of power spectral (PSD) and cross spectral densities (CSD).

The functions in this module are capable of computing an fftgram from a timeseries in the form of a `gwpy.spectrogram.Spectrogram`, computing a Welch-averaged PSD and coarse-graining any data. They are also capable of computing general PSDs and CSDs using the conveniently called `power_spectral_density` and `cross_spectral_density` functions.

Examples
--------

To demonstrate the power of this module, we will compute the cross spectral and power densities of two timeseries 
representing data from two interferometers. We will look at the most important functions from the spectral module.

We read in data for two interferometers using functions from the preprocessing module. Hence we import preprocessing aswell.

>>> import pygwb.preprocessing as ppp
>>> import pygwb.spectral as psp

>>> IFO = "H1"
>>> data_type = "public"
>>> channel = "H1:GWOSC-4KHZ_R1_STRAIN"
>>> t0 = 1247644138
>>> tf = 1247648138
>>> local_data_path = ""
>>> input_sample_rate = 16384
>>> data_timeseries = ppp.read_data(
        IFO,
        data_type,
        channel,
        t0,
        tf,
        local_data_path,
        input_sample_rate,
    )
    
>>> IFO = "L1"
>>> data_type = "public"
>>> channel = "L1:GWOSC-4KHZ_R1_STRAIN"
>>> t0 = 1247644138
>>> tf = 1247648138
>>> local_data_path = ""
>>> input_sample_rate = 16384
>>> data_timeseries_L = ppp.read_data(
        IFO,
        data_type,
        channel,
        t0,
        tf,
        local_data_path,
        input_sample_rate,
    )

We compute now the PSD of H1 and L1.
To achieve this goal, we use common values for the parameters in this 
spectral module. These values are used in the more general pygwb analysis.

>>> PSD_H1 = psp.power_spectral_density(
        data_timeseries,
        segment_duration=192,
        frequency_resolution=1/32.,
        overlap_factor=0.5,
        window_fftgram_dict_welch_psd={"window_fftgram": "hann"},
        overlap_factor_welch_psd=0.5,
    )
    
>>> PSD_L1 = psp.power_spectral_density(
        data_timeseries_L,
        segment_duration=192,
        frequency_resolution=1/32.,
        overlap_factor=0.5,
        window_fftgram_dict_welch_psd={"window_fftgram": "hann"},
        overlap_factor_welch_psd=0.5,
    )

We can also compute the CSD of our baseline H1L1.

>>> CSD_baseline = psp.cross_spectral_density(
        data_timeseries,
        data_timeseries_L,
        segment_duration=192,
        frequency_resolution=1/32.,
        overlap_factor=0.5,
        zeropad=True,
        window_fftgram_dict={"window_fftgram": "hann"},
    )
    
The spectral module as shows here, has the capacity to compute PSDs of single detectors 
and CSDs of baselines (or network) of detectors.

"""

import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram
from gwpy.timeseries import TimeSeries
from scipy.signal import get_window, spectrogram

from pygwb.util import get_window_tuple, parse_window_dict


def fftgram(
    time_series_data: TimeSeries,
    fftlength: int,
    overlap_factor: float=0.5,
    zeropad: bool=False,
    window_fftgram_dict: dict={"window_fftgram": "hann"},
):
    """Create an fftgram from a timeseries.

    Parameters
    ----------
    time_series_data: ``gwpy.timeseries.TimeSeries``
        Timeseries from which to compute the fftgram.
    fftlength: `int`
        Length of each segment (in seconds) for computing FFT.
    overlap_factor: `float`, optional
        Factor of overlap between adjacent FFT segments (values range from 0 
        to 1). Users should provide proper combination of overlap_factor and
        window_fftgram_dict. For \"hann\" window use 0.5 overlap_factor and 
        for \"boxcar"\ window use 0 overlap_factor. Default 0.5 (50% overlap).
    zeropad: `bool`, optional
        Before doing FFT whether to zero pad the data equal to the length of 
        FFT or not. Default is False.
    window_fftgram_dict: `dictionary`, optional
        Dictionary containing name and parameters describing which window to 
        use for producing FFTs. Default is \"hann\".

    Returns
    -------
    data_fftgram: `gwpy.spectrogram.Spectrogram` (complex)
        fftgram containing several FFTs in a matrix format.
    """
    
    sample_rate = int(1 / time_series_data.dt.value)

    # get the window function
    window_tuple = get_window_tuple(parse_window_dict(window_fftgram_dict))
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
    ========
    psdgram: `gwpy.spectrogram.Spectrogram` (PSD)
       PSD gram data to be averaged.
    segment_duration: `int`
        Data duration over which PSDs need to be averaged. Should be greater 
        than or equal to the duration used for FFT. 
    overlap_factor: `float`, optional
        Amount of overlap between adjacent average PSDs, can vary between 0 
        and 1. This factor should be same as the one used for CSD estimation. 
        Default is 0.5.

    Returns
    =======
    avg_psdgram: PSD `gwpy.spectrogram.Spectrogram`
        Averaged over segments within the `segment_duration`.
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
    of interest (for which CSD is calculated).

    Parameters
    ----------
    psdgram: `gwpy.spectrogram.Spectrogram` (PSD)
        PSD spectrogram.
    segment_duration: `int`
        Duration of data used for each PSD calculation.
    N_avg_segs: `int`
        Number of segments to be used for PSD averaging (from both sides of 
        the segment of interest). N_avg_segs should be even and >= 2

    Returns
    -------
    avg_psdgram: `gwpy.spectrogram.Spectrogram`
        Averaged psd gram.
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

    The length of the output is len(data) // coarsening_factor - 1.

    If the coarsening factor is not an integer, :code:`coarse_grain_exact` is
    used.

    Parameters
    ==========
    data: array-like
        The data to coarse grain.
    coarsening_factor: `float`
        The factor by which to coarsen the data.

    Returns
    =======
    coarsened: array-like
        The coarse-grained data.
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
        The data to coarse grain.
    coarsening_factor: :code:`float`
        The factor by which to coarsen the data.

    Returns
    =======
    output: array-like
        The coarse-grained data.
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

    This is equivalent to the process performed for a Welch average.

    Parameters
    ==========
    data: array-like
        The data to coarse grain.
    coarsening_factor: :code:`int`
        The factor by which to coarsen the data.

    Returns
    =======
    coarsened: array-like
        The coarse-grained data.
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
    coarse-graining.

    The coarsening methods are either:
      - naive: this is equivalent to a Welch average
      - full: the full coarse-grain method
      - running_mean: computing the running mean of the array.

    Parameters
    ==========
    spectrogram: `gwpy.spectrogram.Spectrogram`
        Spectrogram object to be coarsened.
    delta_t: `float`, optional.
        Output time spacing.
        Default is None.
    delta_f: `float`, optional
        Output frequency spacing.
        Default is None. 
    time_method: `str`, optional
        Should be one of the coarsening methods listed above.
        Default is "naive".
    frequency_method: `str`, optional.
        Should be one of the coarsening methods listed above.
        Default is "full".

    Returns
    =======
    output: :code:`gwpy.spectrogram.Spectrogram`
        The coarse-grained spectrogram.
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
    Compute the cross spectral density from two time series inputs.

    Parameters
    ----------
    time_series_data1: :code:`gwpy.timeseries.TimeSeries`
        Timeseries data of detector1.
    time_series_data2: `gwpy.timeseries.TimeSeries`
        Timeseries data of detector2.
    segment duration: `int`
        Data duration over which CSDs need to be calculated.
    frequency_resolution: `float`
        Frequency resolution of the final CSDs. This is achieved by averaing in
        frequency domain.
    overlap_factor: `float`, optional
        Amount of overlap between adjacent segments (range between 0 and 1)
        This factor should be same as the one used for power_spectral_density.
        Users should provide proper combination of overlap_factor and
        window_fftgram_dict. For \"hann\" window use 0.5 overlap_factor and for \"boxcar"\ 
        window use 0 overlap_factor. Default id 0.5 (50% overlap).
    zeropad: `bool`, optional
        Before doing FFT whether to zero pad the data equal to the length of 
        FFT or not. Default is False.
    window_fftgram_dict: `dictionary`, optional
        Dictionary containing name and parameters describing which window to 
        use for producing FFTs. Default is \"hann\".

    Returns
    -------
    csd_spectrogram: `gwpy.spectrogram.Spectrogram`
       Cross spectral density of the two timeseries.
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
    segment_duration:  int,
    frequency_resolution: float,
    overlap_factor: float = 0.5,
    window_fftgram_dict_welch_psd: dict = {"window_fftgram": "hann"},
    overlap_factor_welch_psd: float = 0.5,
):
    """
    Compute the PSDs of every segment (defined by the segment duration)
    in the time series using pwelch method.

    Parameters
    ----------
    time_series_data: `gwpy.timeseries.TimeSeries`
        Timeseries from which to compute PSDs.
    segment duration: `int`
        Data duration over which each PSDs need to be calculated.
    frequency_resolution: `float`
        Frequency resolution of the final PSDs. This sets the time duration
        over which FFTs are calculated in the pwelch method.
    overlap_factor: `float`, optional
        Amount of overlap between adjacent segments (range between 0 and 1).
        This factor should be same as the one used for cross_spectral_density.
        Users should provide proper combination of overlap_factor and
        window_fftgram_dict. For \"hann\" window use 0.5 overlap_factor and 
        for \"boxcar"\ window use 0 overlap_factor. Default is 0.5 (50% overlap)
    window_fftgram_dict: `dictionary`, optional
        Dictionary containing name and parameters describing which window to 
        use for producing FFTs. Default is \"hann\".
    overlap_factor_welch_psd: `float`, optional
        Amount of overlap between adjacent segments when Welch averaging 
        in computing the fftgram from both timeseries. 
        Default is 0.5 (50% overlap).
     
    Returns
    -------
    psd_spectrogram: `gwpy.spectrogram.Spectrogram` PSD
        PSD spectrogram with each PSD duration equal to segment duration.
    """
    
    # Length of data blocks to be used in pwelch
    fftlength = int(1.0 / frequency_resolution)

    # No zero-pad is used in the PSD estimation
    fft_gram_data = fftgram(
        time_series_data,
        fftlength,
        overlap_factor=overlap_factor_welch_psd,
        zeropad=False,
        window_fftgram_dict=window_fftgram_dict_welch_psd,
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
        Array of size M to be averaged.
    coarsening_factor: `int`, optional
        Number of segments to average, default=1.
    axis: `int`, optional
        Axis to apply the mean over, default=-1.

    Returns
    -------
    running_mean_array: array-like
        The averaged array of size M - coarsening factor.
    """
    coarsening_factor = int(coarsening_factor)
    if axis != -1:
        data = np.swapaxes(axis, -1)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (
        np.swapaxes(cumsum[coarsening_factor:] - cumsum[:-coarsening_factor], 
                    axis, -1)/ coarsening_factor
    )
