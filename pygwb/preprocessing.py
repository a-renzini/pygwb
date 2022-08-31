import copy
import os
import warnings

import lal
import numpy as np
import scipy
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


def read_data(
    IFO: str,
    data_type: str,
    channel: str,
    t0: int,
    tf: int,
    local_data_path: str = "",
    tag: str = "C00",
    input_sample_rate: int = 16384,
):
    """
    Function doing the reading of the data to be used in the
    stochastic pipeline

    Parameters
    ==========
    IFO: string
        Interferometer from which to retrieve the data

    data_type: string
        String indicating the type of data to be read,
        either:
        - 'public' : data from GWOSC (https://www.gw-openscience.org/)
        - 'private' : data from the LIGO-Virgo servers restricted to members of the collaboration
        - 'local' (if 'local_data_path'): data the user has saved locally

    channel: string
        Name of the channel (e.g.: "L1:GWOSC-4KHZ_R1_STRAIN")

    t0: int
        GPS time of the start of the data taking

    tf: int
        GPS time of the end of the data taking

    local_data_path: str, optional
        path where local gwf is stored

    tag: str, optional
        tag identifying type of data, e.g.: 'C00', 'C01'

    input_sample_rate: int
        Sampling rate of the timeseries to be read

    Returns
    =======
    data: TimeSeries object
        Time series containing the requested data
    """
    if data_type == "public":
        data = timeseries.TimeSeries.fetch_open_data(
            IFO, t0, tf, sample_rate=input_sample_rate, tag=tag
        )
        data.channel = channel
    elif data_type == "private":
        data = timeseries.TimeSeries.get(channel, start=t0, end=tf, verbose=True)
        data.channel = channel
    elif data_type == "local":
        if os.path.isdir(local_data_path):
            local_data = []
            for f in os.listdir(local_data_path):
                local_data.append(os.path.join(local_data_path, f))
        else:
            local_data = local_data_path
        data = timeseries.TimeSeries.read(
            source=local_data, channel=channel, start=t0, end=tf, verbose=True
        )
        data.channel = channel
        data.name = IFO

    else:
        raise ValueError("Wrong data type. Choose between: public, private and local")
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
    zpk = scipy.signal.butter(
        16, cutoff_frequency, "high", analog=False, output="zpk", fs=sample_rate
    )
    filtered = timeseries.filter(zpk, filtfilt=True)
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
    if (new_sample_rate * number_cropped_seconds) < 18:
        warnings.warn(
            f"Number of cropped seconds requested {number_cropped_seconds}s is low compared to the sampling rate {new_sampling_rate}: cropped-seconds x sampling-rate = {number_cropped_seconds*new_sampling_rate}."
        )
    data_to_resample = copy.deepcopy(time_series_data.value)
    original_times = copy.deepcopy(time_series_data.times)
    nan_mask = np.isnan(time_series_data.value)  # .flatten()

    if np.sum(nan_mask) != 0:
        data_nansafe = data_to_resample[~nan_mask]
        times_nansafe = original_times[~nan_mask]
        interped_data = scipy.interpolate.CubicSpline(times_nansafe, data_nansafe)
        new = interped_data(original_times).view(time_series_data.__class__)
        new.__metadata_finalize__(time_series_data)
        new._unit = time_series_data.unit
        resampled = new.resample(new_sample_rate, window_downsampling, ftype)
        warnings.warn(
            f"There are {np.sum(nan_mask)} NaNs in the timestream ({np.sum(nan_mask)*100/len(time_series_data)}% of the data). These will be ignored in pre-processing."
        )
    else:
        resampled = time_series_data.resample(
            new_sample_rate, window_downsampling, ftype
        )

    sample_rate = resampled.sample_rate.value
    filtered = apply_high_pass_filter(
        timeseries=resampled,
        sample_rate=sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_cropped_seconds=number_cropped_seconds,
    )

    return filtered


def self_gate_data(
    time_series_data: timeseries.TimeSeries,
    tzero: float = 1.0,
    tpad: float = 0.5,
    gate_threshold: float = 50.0,
    cluster_window: float = 0.5,
    whiten: bool = True,
):
    """
    Function to self-gate
    data to be used in the stochastic pipeline

    Parameters
    ==========

    time_series_data: gwpy_timeseries
        timeseries data to be analysed in the pipeline

    tzero : `int`, optional
        half-width time duration (seconds) in which the timeseries is
        set to zero

    tpad : `int`, optional
        half-width time duration (seconds) in which the Planck window
        is tapered

    whiten : `bool`, optional
        if True, data will be whitened before gating points are discovered,
        use of this option is highly recommended

    threshold : `float`, optional
        amplitude threshold, if the data exceeds this value a gating window
        will be placed

    cluster_window : `float`, optional
        time duration (seconds) over which gating points will be clustered

    Returns
    =======
    gated: gwpy_timeseries
        Timeseries containing the gated data

    deadtime: `gwpy.segments.SegmentList`
        SegmentList containing the times that were gated, not including
        any padding applied

    Notes
    -----
    This method is based on `gwpy.timeseries.gate`. See
    https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries/?highlight=timeseries#gwpy.timeseries.TimeSeries.gate
    for additional details.
    """

    from gwpy.segments import Segment, SegmentList
    from scipy.signal import find_peaks

    # Find points to gate based on a threshold
    sample = time_series_data.sample_rate.to("Hz").value
    data = time_series_data.whiten() if whiten else time_series_data
    window_samples = cluster_window * sample
    gates = find_peaks(abs(data.value), height=gate_threshold, distance=window_samples)[
        0
    ]
    # represent gates as time segments
    deadtime = SegmentList(
        [
            Segment(
                time_series_data.t0.value + (k / sample) - tzero,
                time_series_data.t0.value + (k / sample) + tzero,
            )
            for k in gates
        ]
    ).coalesce()
    # return the self-gated timeseries
    gated = time_series_data.mask(deadtime=deadtime, const=0, tpad=tpad)
    return gated, deadtime


def shift_timeseries(time_series_data: timeseries.TimeSeries, time_shift: int = 0):

    """
    Function that shifts a timeseries by an amount time_shift
    in order to perform the time shifted analysis

    Parameters
    ==========

    time_series_data: gwpy_timeseries
        timeseries data to be analysed in the pipeline

    time_shift: int
        value of the time shift (in seconds)

    Returns
    =======
    shifted_data: gwpy_timeseries
        Timeseries containing the shifted_data
    """

    if time_shift > 0:
        shifted_data = np.roll(
            time_series_data, int(time_shift / time_series_data.dt.value)
        )
    else:
        shifted_data = time_series_data
    return shifted_data


def preprocessing_data_gwpy_timeseries(
    IFO: str,
    gwpy_timeseries: timeseries.TimeSeries,
    new_sample_rate: int,
    cutoff_frequency: float,
    number_cropped_seconds: int = 2,
    window_downsampling: str = "hamming",
    ftype: str = "fir",
    time_shift: int = 0,
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

    time_shift: int
        value of the time shift (in seconds)

    Returns
    =======
    pre_processed_data: gwpy_timeseries
        Timeseries containing the filtered and high passed data (shifted if time_shift>0)
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
    if time_shift > 0:
        return shift_timeseries(time_series_data=filtered, time_shift=time_shift)
    else:
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
    time_shift: int = 0,
    local_data_path: str = "",
    tag: str = "C00",
    input_sample_rate: int = 16384,
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

    time_shift: int
        value of the time shift (in seconds)

    tag: str, optional
        tag identifying type of data, e.g.: 'C00', 'C01'

    input_sample_rate: int
        Sampling rate of the timeseries to be read

    Returns
    =======
    pre_processed_data: gwpy_timeseries
        Timeseries containing the filtered and high passed data (shifted if time_shift>0)
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
        local_data_path=local_data_path,
        tag=tag,
        input_sample_rate=input_sample_rate,
    )

    return preprocessing_data_gwpy_timeseries(
        IFO=IFO,
        gwpy_timeseries=time_series_data,
        new_sample_rate=new_sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_cropped_seconds=2,
        window_downsampling=window_downsampling,
        ftype=ftype,
        time_shift=time_shift,
    )


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
    time_shift: int = 0,
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

    time_shift: int
        value of the time shift (in seconds)

    Returns
    =======
    pre_processed_data: gwpy_timeseries
        Timeseries containing the filtered and high passed data (shifted if time_shift>0)
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
    return preprocessing_data_gwpy_timeseries(
        IFO=IFO,
        gwpy_timeseries=time_series_data,
        new_sample_rate=new_sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_cropped_seconds=2,
        window_downsampling=window_downsampling,
        ftype=ftype,
        time_shift=time_shift,
    )
