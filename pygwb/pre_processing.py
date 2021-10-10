import lal
import numpy as np
from gwpy import signal, timeseries
from gwpy.spectrogram import Spectrogram
from scipy.signal import get_window, spectrogram


def set_start_time(
    job_start_GPS, job_end_GPS, buffer_secs, segment_duration, do_sidereal=0
):
    """Function to identify segment start times
    either with or without sidereal option

    Parameters
    ==========
    job_start_GPS: int_like
        Integer indicating the start time (in GPS)
        of the data

    job_end_GPS: int_like
        Integer indicating the end time (in GPS)
        of the data

    buffer_secs: int_like
        Number of cropped seconds

    segment_duration: int_like
        Duration of each segment

    do_sidereal: binary_like
        --

    Returns
    =======
    centered_start_time: int_like
        Integer with the initial time of the segment to be pre-processed
    """
    if not do_sidereal:
        job_duration = job_end_GPS - job_start_GPS
        M = np.floor((job_duration - 2 * buffer_secs) / segment_duration)
        centered_start_time = (
            job_start_GPS
            + buffer_secs
            + np.floor((job_duration - 2 * buffer_secs - M * segment_duration) / 2)
        )
    else:
        srfac = 23.9344696 / 24
        # sidereal time conversion factor
        srtime = (lal.GreenwichMeanSiderealTime(job_end_GPS) % (2 * np.pi)) * 3600
        md = np.mod(srtime, segment_duration / srfac)
        centered_start_time = np.round(job_end_GPS + segment_duration - md * srfac)
        if centered_start_time - job_end_GPS < buffer_secs:
            centered_start_time = centered_start_time + segment_duration
    return centered_start_time


def read_data(IFO, data_type, channel, t0, tf):
    """Function doing the reading of the data to be used in the
    stochastic pipeline

    Parameters
    ==========
    IFO: string_like
        Interferometer from which to retrieve the data

    data_type: string_like
        String indicating the type of data to be read,
        either 'public' or 'private'

    channel: string_like
        Name of the channel (e.g.: "L1:GWOSC-4KHZ_R1_STRAIN")

    t0: int_like
        GPS time of the start of the data taking

    tf: int_like
        GPS time of the end of the data taking

    Returns
    =======
    data: TimeSeries object
        Time series containing the requested data
    """
    if data_type == "public":
        data = timeseries.TimeSeries.fetch_open_data(IFO, t0, tf, sample_rate=16384)
    elif data_type == "private":
        data = timeseries.TimeSeries.get(channel, t0, tf)
    elif data_type == "injection":
        pass
    else:
        raise ValueError(
            "Wrong data type. Choose between: public, private and injection"
        )
    return data


def apply_high_pass_filter(
    timeseries, sample_rate, cutoff_frequency, number_cropped_seconds=2
):
    """Function to apply a high pass filter to a timeseries

    Parameters
    ==========
    timeseries: gwpy_timeseries
        Timeseries to which to apply the high pass filter

    sample_rate: int_like
        Sampling rate of the timeseries

    cutoff_frequency: int_like
        Frequency (in Hz) from which to start applying the
        high pass filter

    number_cropped_seconds: int_like
        Number of seconds to remove at the beginning and end
        of the high-passed data

    fstop: float
        Stop-band edge frequency, defaults to frequency * 1.5

    gpass: float_like
        The maximum loss in the passband (dB)

    gstop: float_like
        The minimum attenuation in the stopband (dB)

    type: str_like
        The filter type, either 'iir' or 'fir'

    Returns
    =======
    filtered: Timeseries
        High-pass filtered timeseries
    """
    hp = signal.filter_design.highpass(
        cutoff_frequency, sample_rate, fstop=6.0, gpass=0.01, gstop=60.0, type="iir"
    )
    filtered = timeseries.filter(hp, filtfilt=True)
    filtered = filtered.crop(*filtered.span.contract(number_cropped_seconds))
    return filtered


def fftgram(timeseries, fftlength, overlap, zeropad=False, window_fftgram="hann"):
    """Function that creates an fftgram from a timeseries

    Parameters
    ==========
    timeseries: gwpy_timeseries
        Timeseries from which to compute the fftgram

    fftlength: int_length
        Length (in s) of each segment in which
        to compute an FFT

    overlap: int_length
        Length (in s) of the overlap between segments

    zeropad: bool
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
    if overlap is None:
        overlap = fftlength / 2.0
    sample_rate = int(1 / timeseries.dt.value)
    window_fftgram = get_window(
        window=window_fftgram, Nx=fftlength * sample_rate, fftbins=False
    )
    if zeropad:
        f, t, Sxx = spectrogram(
            x=timeseries.data,
            fs=sample_rate,
            window=window_fftgram,
            nperseg=fftlength * sample_rate,
            noverlap=overlap * sample_rate,
            nfft=2 * fftlength * sample_rate,
            mode="complex",
            detrend=False,
        )
    else:
        f, t, Sxx = spectrogram(
            x=timeseries.data,
            fs=sample_rate,
            window=window_fftgram,
            nperseg=fftlength * sample_rate,
            noverlap=overlap * sample_rate,
            nfft=fftlength * sample_rate,
            mode="complex",
            detrend=False,
        )

    data_fftgram = Spectrogram(data=Sxx.T, times=t + timeseries.t0.value, frequencies=f)

    return data_fftgram


def resample_filter_fftgram(
    time_series_data,
    new_sample_rate,
    cutoff_frequency,
    fftlength,
    overlap,
    number_cropped_seconds=2,
    zeropad=False,
    window_downsampling="hamming",
    ftype="fir",
    window_fftgram="hann",
):
    """Function doing part of the pre-processing
    (resampling,filtering and fftgram computation)
    of the data to be used in the stochastic pipeline

    Parameters
    ==========

    time_series_data: gwpy_timeseries_like
        timeseries data to be analysed in the pipeline

    new_sample_rate: int_like
        Sampling rate of the downsampled-timeseries

    cutoff_frequency: int_like
        Frequency (in Hz) from which to start applying the
        high pass filter

    fftlength: int_length
        Length (in s) of each segment in which
        to compute an FFT

    overlap: int_length
        Length (in s) of the overlap between segments

    number_cropped_seconds: int_like
        Number of seconds to remove at the beginning and end
        of the high-passed data

    zeropad: bool
        Whether to zero pad the data equal to the length of FFT or not
        (default False)

    window_downsampling: string_like
        Type of window used to downsample

    ftype: string_like
        Type of filter to use in the downsampling

    window_fftgram: string_like
        Type of window to compute the Fast Fourier
        transform

    Returns
    =======
    filtered: gwpy_timeseries
        Timseries containing the filtered and high passed data
    """
    resampled = time_series_data.resample(new_sample_rate, window_downsampling, ftype)
    sample_rate = time_series_data.sample_rate.value
    filtered = apply_high_pass_filter(
        timeseries=resampled,
        sample_rate=sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_cropped_seconds=number_cropped_seconds,
    )
    # output = fftgram(
    #    timeseries=filtered,
    #    fftlength=fftlength,
    #    overlap=overlap,
    #    zeropad=zeropad,
    #    window_fftgram=window_fftgram,
    # )
    return filtered


def preprocessing_data_channel_name(
    IFO,
    t0,
    tf,
    data_type,
    channel,
    new_sample_rate,
    cutoff_frequency,
    fftlength,
    segment_duration,
    zeropad,
    overlap=None,
    number_cropped_seconds=2,
    window_downsampling="hamming",
    ftype="fir",
    window_fftgram="hann",
):
    """Function doing the pre-processing of the data to be used in the
    stochastic pipeline

    Parameters
    ==========
    IFO: string_like
        Interferometer from which to retrieve the data

    t0: int_like
        GPS time of the start of the data taking

    tf: int_like
        GPS time of the end of the data taking

    data_type: string_like
        String indicating the type of data to be read,
        either 'public' or 'private'

    channel: string_like
        Name of the channel (e.g.: "L1:GWOSC-4KHZ_R1_STRAIN")

    new_sample_rate:int_like
        Sampling rate of the downsampled-timeseries

    cutoff_frequency: int_like
        Frequency (in Hz) from which to start applying the
        high pass filter

    fftlength: int_length
        Length (in s) of each segment in which
        to compute an FFT

    segment_duration: int_like
        Duration of each segment (argument of set_start_time)

    overlap: int_length
        Length (in s) of the overlap between segments

    number_cropped_seconds: int_like
        Number of seconds to remove at the beginning and end
        of the high-passed data

    zeropad: bool
        Whether to zero pad the data equal to the length of FFT or not
        (default False)

    window_downsampling: string_like
        Type of window used to downsample

    ftype: string_like
        Type of filter to use in the downsampling

    window_fftgram: string_like
        Type of window to compute the Fast Fourier
        transform

    Returns
    =======
    filtered: gwpy_timeseries
        Timseries containing the filtered and high passed data
    """
    data_start_time = set_start_time(
        job_start_GPS=t0,
        job_end_GPS=tf,
        buffer_secs=number_cropped_seconds,
        segment_duration=segment_duration,
    )
    time_series_data = read_data(
        IFO=IFO,
        data_type=data_type,
        channel=channel,
        t0=data_start_time - number_cropped_seconds,
        tf=tf,
    )
    output = resample_filter_fftgram(
        time_series_data=time_series_data,
        new_sample_rate=new_sample_rate,
        cutoff_frequency=cutoff_frequency,
        fftlength=fftlength,
        overlap=overlap,
        number_cropped_seconds=number_cropped_seconds,
        zeropad=zeropad,
        window_downsampling=window_downsampling,
        ftype=ftype,
        window_fftgram=window_fftgram,
    )
    return output


def preprocessing_data_timeseries_array(
    t0,
    tf,
    IFO,
    array,
    new_sample_rate,
    cutoff_frequency,
    fftlength,
    segment_duration,
    zeropad,
    overlap=None,
    sample_rate=4096,
    number_cropped_seconds=2,
    window_downsampling="hamming",
    ftype="fir",
    window_fftgram="hann",
):
    """Function doing the pre-processing of a time-series array to be used in the
    stochastic pipeline

    Parameters
    ==========
    t0: int_like
        GPS time of the start of the data taking

    tf: int_like
        GPS time of the end of the data taking

    IFO: string_like
        Interferometer from which to retrieve the data

    array: array_like
        Array containing a timeseries

    new_sample_rate:int_like
        Sampling rate of the downsampled-timeseries

    cutoff_frequency: int_like
        Frequency (in Hz) from which to start applying the
        high pass filter

    fftlength: int_length
        Length (in s) of each segment in which
        to compute an FFT

    segment_duration: int_like
        Duration of each segment (argument of set_start_time)

    zeropad: bool
        Whether to zero pad the data equal to the length of FFT or not
        (default False)

    overlap: int_length
        Length (in s) of the overlap between segments

    sample_rate: int_like
        Sampling rate of the original timeseries

    number_cropped_seconds: int_like
        Number of seconds to remove at the beginning and end
        of the high-passed data

    window_downsampling: string_like
        Type of window used to downsample

    ftype: string_like
        Type of filter to use in the downsampling

    window_fftgram: string_like
        Type of window to compute the Fast Fourier
        transform

    Returns
    =======
    filtered: gwpy_timeseries
        Timseries containing the filtered and high passed data
    """
    data_start_time = set_start_time(
        job_start_GPS=t0,
        job_end_GPS=tf,
        buffer_secs=number_cropped_seconds,
        segment_duration=segment_duration,
    )
    time_series_data = timeseries.TimeSeries(
        array, t0=data_start_time, sample_rate=sample_rate
    )
    output = resample_filter_fftgram(
        time_series_data=time_series_data,
        new_sample_rate=new_sample_rate,
        cutoff_frequency=cutoff_frequency,
        fftlength=fftlength,
        overlap=overlap,
        number_cropped_seconds=number_cropped_seconds,
        zeropad=zeropad,
        window_downsampling=window_downsampling,
        ftype=ftype,
        window_fftgram=window_fftgram,
    )
    return output


def preprocessing_data_gwpy_timeseries(
    IFO,
    gwpy_timeseries,
    new_sample_rate,
    cutoff_frequency,
    fftlength,
    number_cropped_seconds,
    zeropad,
    overlap=None,
    window_downsampling="hamming",
    ftype="fir",
    window_fftgram="hann",
):
    """Function doing the pre-processing of a gwpy timeseries to be used in the
    stochastic pipeline

    Parameters
    ==========

    gwpy_timeseries: gwpy timeseries
        Timeseries from gwpy

    new_sample_rate:int_like
        Sampling rate of the downsampled-timeseries

    cutoff_frequency: int_like
        Frequency (in Hz) from which to start applying the
        high pass filter

    number_cropped_seconds: int_like
        Number of seconds to remove at the beginning and end
        of the high-passed data

    fftlength: int_length
        Length (in s) of each segment in which
        to compute an FFT

    number_cropped_seconds: int_like
        Number of seconds to remove at the beginning and end
        of the high-passed data

    zeropad: bool
        Whether to zero pad the data equal to the length of FFT or not
        (default False)

    overlap: int_length
        Length (in s) of the overlap between segments

    window_fftgram: string_like
        Type of window to compute the Fast Fourier
        transform

    window_downsampling: string_like
        Type of window used to downsample

    Returns
    =======
    filtered: gwpy_timeseries
        Timseries containing the filtered and high passed data
    """
    time_series_data = gwpy_timeseries
    output = resample_filter_fftgram(
        time_series_data=time_series_data,
        new_sample_rate=new_sample_rate,
        cutoff_frequency=cutoff_frequency,
        fftlength=fftlength,
        overlap=overlap,
        number_cropped_seconds=number_cropped_seconds,
        zeropad=zeropad,
        window_downsampling=window_downsampling,
        ftype=ftype,
        window_fftgram=window_fftgram,
    )
    return output
