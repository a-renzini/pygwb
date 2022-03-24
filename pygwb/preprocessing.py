import lal
import numpy as np
from gwpy import signal, timeseries


def set_start_time(
    job_start_GPS: int,
    job_end_GPS: int,
    buffer_secs: int,
    segment_duration: int,
    do_sidereal: bool = False,
):
    """
    Function to identify segment start times
    either with or without sidereal option

    Parameters
    ==========
    job_start_GPS: int
        Integer indicating the start time (in GPS)
        of the data

    job_end_GPS: int
        Integer indicating the end time (in GPS)
        of the data

    buffer_secs: int
        Number of cropped seconds

    segment_duration: int
        Duration of each segment

    do_sidereal: bool
        ---

    Returns
    =======
    centered_start_time: int
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


def read_data(IFO: str, data_type: str, channel: str, t0: int, tf: int, mock_data_path: str = ''):
    """
    Function doing the reading of the data to be used in the
    stochastic pipeline

    Parameters
    ==========
    IFO: string
        Interferometer from which to retrieve the data

    data_type: string
        String indicating the type of data to be read,
        either 'public' or 'private' or 'mock'
        if 'mock', need to also pass a mock_data_path

    channel: string
        Name of the channel (e.g.: "L1:GWOSC-4KHZ_R1_STRAIN")

    t0: int
        GPS time of the start of the data taking

    tf: int
        GPS time of the end of the data taking

    mock_data_path: str, optional

    Returns
    =======
    data: TimeSeries object
        Time series containing the requested data
    """
    if data_type == "public":
        data = timeseries.TimeSeries.fetch_open_data(IFO, t0, tf, sample_rate=16384)
        data.channel = channel
    elif data_type == "private":
        data = timeseries.TimeSeries.get(channel, start=t0, end=tf, verbose=True)
        data.channel = channel
    elif data_type == "mock":
        data = timeseries.TimeSeries.read(source=mock_data_path, channel=channel, start=t0, end=tf)
        data.channel = channel
        data.name = IFO
        
    else:
        raise ValueError(
            "Wrong data type. Choose between: public, private and mock"
        )
    return data


def apply_high_pass_filter(
    timeseries: timeseries.TimeSeries,
    sample_rate: int,
    cutoff_frequency: float,
    number_cropped_seconds: int = 2,
):
    """
    Function to apply a high pass filter to a timeseries

    Parameters
    ==========
    timeseries: gwpy_timeseries
        Timeseries to which to apply the high pass filter

    sample_rate: int
        Sampling rate of the timeseries

    cutoff_frequency: float
        Frequency (in Hz) from which to start applying the
        high pass filter

    number_cropped_seconds: int, optional
        Number of seconds to remove at the beginning and end
        of the high-passed data; default is 2

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


def resample_filter(
    time_series_data: timeseries.TimeSeries,
    new_sample_rate: int,
    cutoff_frequency: float,
    number_cropped_seconds: int = 2,
    window_downsampling: str = "hamming",
    ftype: str = "fir",
):
    """
    Function doing part of the pre-processing
    (resampling,filtering and fftgram computation)
    of the data to be used in the stochastic pipeline

    Parameters
    ==========

    time_series_data: gwpy_timeseries
        timeseries data to be analysed in the pipeline

    new_sample_rate: int
        Sampling rate of the downsampled-timeseries

    cutoff_frequency: float
        Frequency (in Hz) from which to start applying the
        high pass filter

    number_cropped_seconds: int
        Number of seconds to remove at the beginning and end
        of the high-passed data

    window_downsampling: string
        Type of window used to downsample

    ftype: string
        Type of filter to use in the downsampling


    Returns
    =======
    filtered: gwpy_timeseries
        Timeseries containing the filtered and high passed data
    """
    if new_sample_rate % 2 != 0:
        raise ValueError("New sample rate is not even.")
    resampled = time_series_data.resample(new_sample_rate, window_downsampling, ftype)
    sample_rate = resampled.sample_rate.value
    filtered = apply_high_pass_filter(
        timeseries=resampled,
        sample_rate=sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_cropped_seconds=number_cropped_seconds,
    )

    return filtered


def preprocessing_data_channel_name(
    IFO: str,
    t0: int,
    tf: int,
    data_type: str,
    channel: str,
    new_sample_rate: int,
    cutoff_frequency: float,
    segment_duration: int,
    number_cropped_seconds: int = 2,
    window_downsampling: str = "hamming",
    ftype: str = "fir",
    mock_data_path: str = '',
):
    """
    Function doing the pre-processing of the data to be used in the
    stochastic pipeline

    Parameters
    ==========
    IFO: string
        Interferometer from which to retrieve the data

    t0: int
        GPS time of the start of the data taking

    tf: int
        GPS time of the end of the data taking

    data_type: string
        String indicating the type of data to be read,
        either 'public' or 'private'

    channel: string
        Name of the channel (e.g.: "L1:GWOSC-4KHZ_R1_STRAIN")

    new_sample_rate:int
        Sampling rate of the downsampled-timeseries

    cutoff_frequency: float
        Frequency (in Hz) from which to start applying the
        high pass filter

    segment_duration: int
        Duration of each segment (argument of set_start_time)

    number_cropped_seconds: int
        Number of seconds to remove at the beginning and end
        of the high-passed data

    window_downsampling: string
        Type of window used to downsample

    ftype: string
        Type of filter to use in the downsampling


    Returns
    =======
    filtered: gwpy_timeseries
        Timeseries containing the filtered and high passed data
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
        mock_data_path=mock_data_path
    )

    filtered = resample_filter(
        time_series_data=time_series_data,
        new_sample_rate=new_sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_cropped_seconds=number_cropped_seconds,
        window_downsampling=window_downsampling,
        ftype=ftype,
    )
    return filtered


def preprocessing_data_timeseries_array(
    t0: int,
    tf: int,
    IFO: str,
    array: np.ndarray,
    new_sample_rate: int,
    cutoff_frequency: float,
    segment_duration: int,
    sample_rate: int = 4096,
    number_cropped_seconds: int = 2,
    window_downsampling: str = "hamming",
    ftype: str = "fir",
):
    """
    Function doing the pre-processing of a time-series array to be used in the
    stochastic pipeline

    Parameters
    ==========
    t0: int
        GPS time of the start of the data taking

    tf: int
        GPS time of the end of the data taking

    IFO: string
        Interferometer from which to retrieve the data

    array: array
        Array containing a timeseries

    new_sample_rate:int
        Sampling rate of the downsampled-timeseries

    cutoff_frequency: float
        Frequency (in Hz) from which to start applying the
        high pass filter

    segment_duration: int
        Duration of each segment (argument of set_start_time)

    sample_rate: int
        Sampling rate of the original timeseries

    number_cropped_seconds: int
        Number of seconds to remove at the beginning and end
        of the high-passed data

    window_downsampling: string
        Type of window used to downsample

    ftype: string
        Type of filter to use in the downsampling

    Returns
    =======
    filtered: gwpy_timeseries
        Timeseries containing the filtered and high passed data
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

    filtered = resample_filter(
        time_series_data=time_series_data,
        new_sample_rate=new_sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_cropped_seconds=number_cropped_seconds,
        window_downsampling=window_downsampling,
        ftype=ftype,
    )

    return filtered


def preprocessing_data_gwpy_timeseries(
    IFO: str,
    gwpy_timeseries: timeseries.TimeSeries,
    new_sample_rate: int,
    cutoff_frequency: float,
    number_cropped_seconds: int = 2,
    window_downsampling: str = "hamming",
    ftype: str = "fir",
):
    """
    Function doing the pre-processing of a gwpy timeseries to be used in the
    stochastic pipeline

    Parameters
    ==========

    IFO: string
        Interferometer from which to retrieve the data

    gwpy_timeseries: gwpy_timeseries
        Timeseries from gwpy

    new_sample_rate:int
        Sampling rate of the downsampled-timeseries

    cutoff_frequency: float
        Frequency (in Hz) from which to start applying the
        high pass filter

    number_cropped_seconds: int
        Number of seconds to remove at the beginning and end
        of the high-passed data

    window_downsampling: string
        Type of window used to downsample

    ftype: string
        Type of filter to use in the downsampling

    Returns
    =======
    filtered: gwpy_timeseries
        Timseries containing the filtered and high passed data
    """
    time_series_data = gwpy_timeseries
    filtered = resample_filter(
        time_series_data=time_series_data,
        new_sample_rate=new_sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_cropped_seconds=number_cropped_seconds,
        window_downsampling=window_downsampling,
        ftype=ftype,
    )
    return filtered

