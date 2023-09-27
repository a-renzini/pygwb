"""The ``preprocessing`` module combines all the functions that handle the preprocessing of the data used in the analysis.
This is anything related to the preparation of the data for the ``pygwb`` analysis run.
It can read data from frame files, locally or publicly (for additional information on frame files, see `here <https://gwpy.github.io/docs/v0.1/timeseries/gwf.html>`_).
Other functionalities include resampling the data, applying a high-pass filter to data or applying a timeshift.
These functionalities come together in the triplet of ``preprocessing_data`` functions
which read in data and resample and/or high-passe the data on the fly.
The triplet can work for a ``gwpy.timeseries.TimeSeries``, a normal array or using a gravitational-wave channel
that will read data from that channel using the provided local or public frame files. Another functionality of the module is to 
gate data based on the gating function in ``gwpy``, ``gwpy.timeseries.TimeSeries.gate``.
More information can be found `here <https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.gate>`_.

Examples
--------

As an example, we read in some data from a certain channel and then resample, high-pass and apply gating to the data.
First, we have to import the module.

>>> import pygwb.preprocessing as ppp

Then, we read in some data using the ``read_data`` method.
For concreteness, we read in public data from the LIGO Hanford "H1" detector. This can be done as shown below. 
The "public" tag indicates we are obtaining public data from the `GWOSC <https://gwosc.org/>`_ servers.

>>> IFO = "H1"
>>> data_timeseries = ppp.read_data(
        IFO,
        "public",                   # data_type
        "H1:GWOSC-16KHZ_R1_STRAIN", # channel
        1247644138,                 # t0
        1247648138,                 # tf
        "",                         # local_data_path
        16384                       # input_sample_rate
    )
>>> print(data_timeseries.sample_rate)
16384.0 Hz

The sample rate is shown for illustrative purposes. Now, we preprocess the data, 
meaning it is resampled and a high-pass
filter is applied to the data. As an example, the data is resampled to 4 kHz.

>>> new_sample_rate = 4096
>>> preprocessed_timeseries = ppp.preprocessing_data_gwpy_timeseries(
        IFO,
        data_timeseries,
        new_sample_rate,
        11,        # cutoff_frequency
        2,         # number_cropped_seconds
        "hamming", # window_downsampling
        "fir",     # ftype
        0          # timeshift
    )
>>> print(preprocessed_timeseries.sample_rate)
4096.0 Hz

One can see that the sample rate was indeed modified. 
Another important part of preprocessing is gating the data.
In that case, using again default values for parameters, one can run the following lines:

>>> gated_timeseries, deadtime = ppp.self_gate_data(
        preprocessed_timeseries,
        1.0,  # gate_tzero
        0.5,  # gate_tpad
        50.0, # gate_threshold
        0.5,  # cluster_window
        True  # gate_whiten
    )

More information on the gating procedure can be found `here <https://dcc.ligo.org/public/0172/P2000546/002/gating-mdc.pdf>`_.
"""
import copy
import os
import warnings

import lal
import numpy as np
import scipy
from gwpy import timeseries
from gwpy.segments import Segment, SegmentList
from gwsumm.data.timeseries import get_timeseries


def set_start_time(
    job_start_GPS: int,
    job_end_GPS: int,
    buffer_secs: int,
    segment_duration: int,
    do_sidereal: bool = False,
):
    """
    Function to identify segment start times either with or without sidereal option.

    Parameters
    =======
    job_start_GPS: ``int``
        Integer indicating the start time (in GPS)
        of the data.

    job_end_GPS: ``int``
        Integer indicating the end time (in GPS)
        of the data.

    buffer_secs: ``int``
        Number of cropped seconds.

    segment_duration: ``int``
        Duration of each segment.

    do_sidereal: ``bool``, optional
        When this option is turned on, the code 
        considers sidereal days instead of terrestrial days.

    Returns
    =======
    centered_start_time: ``int``
        Integer with the initial time of the segment to be pre-processed.
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
    frametype: str = "",
    input_sample_rate: int = 16384,
):
    """
    Function that read in the data to be used in the
    rest of the code.

    Parameters
    =======
    IFO: ``str``
        Interferometer name for which to retrieve the data.

    data_type: ``str``
        String indicating the type of data to be read,
        either:
        - 'public' : data from GWOSC (https://www.gw-openscience.org/)
        - 'private' : data from the LIGO-Virgo servers restricted to members of the LIGO-Virgo-KAGRA collaboration
        - 'local' (if 'local_data_path'): locally saved data

    channel: ``str``
        Name of the channel (e.g.: "L1:GWOSC-4KHZ_R1_STRAIN").

    t0: ``int``
        GPS time of the start of the data taking.

    tf: ``int``
        GPS time of the end of the data taking.

    frametype: ``str``
        Frame type that contains the channel, only used if data_type=private (e.g.: "L1_HOFT_C00").

    local_data_path: ``str``, optional
        Path where local data (gwf format) is stored.

    input_sample_rate: ``int``, optional
        Sampling rate of the timeseries to be read in Hz. Default is 16384 Hz.

    Returns
    =======
    data: ``gwpy.timeseries.TimeSeries``
        Time series containing the requested data.

    See also
    --------
    gwpy.timeseries.TimeSeries
        More information `here <https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries>`_.
    gwsumm.data.timeseries.get_timeseries
        More information `here <https://github.com/gwpy/gwsumm/blob/master/gwsumm/data/timeseries.py>`_.
    """
    if data_type == "public":
        data = timeseries.TimeSeries.fetch_open_data(
            IFO, t0, tf, sample_rate=input_sample_rate
        )
        data.channel = channel
    elif data_type == "private":
        if frametype == "":
            frametype = None
        data = get_timeseries(channel, segments=[[t0, tf]], frametype=frametype)
        if len(data) > 1:
            raise ValueError("Something went wrong while getting the data!"
                             "There was more than one data stretch returned.")
        else:
            data = data[0]
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
    Function to apply a high pass filter to a timeseries.

    Parameters
    =======
    timeseries: ``gwpy.timeseries.TimeSeries``
        Timeseries to which to apply the high pass filter.

    sample_rate: ``int``
        Sampling rate of the timeseries in Hz.

    cutoff_frequency: ``float``
        Frequency (in Hz) from which to start applying the
        high pass filter.

    number_cropped_seconds: ``int``, optional
        Number of seconds to remove at the beginning and end
        of the high-passed data; default is 2.

    Returns
    =======
    filtered: ``gwpy.timeseries.TimeSeries``
        High-pass filtered timeseries.
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
    Function doing part of the pre-processing (resampling and filtering)
    of the data to be used in the remainder of the code.

    Parameters
    =======
    time_series_data: ``gwpy.timeseries.TimeSeries``
        Timeseries data to be analysed.

    new_sample_rate: ``int``
        Sampling rate of the downsampled timeseries in Hz.

    cutoff_frequency: ``float``
        Frequency (in Hz) from which to start applying the
        high pass filter.

    number_cropped_seconds: ``int``, optional
        Number of seconds to remove at the beginning and end
        of the high-passed data. Default is 2.

    window_downsampling: ``str``, optional
        Type of window used to downsample. Default window is hamming.

    ftype: ``str``
        Type of filter to use in the downsampling. Default filter is fir.

    Returns
    =======
    filtered: ``gwpy.timeseries.TimeSeries``
        Timeseries containing the filtered and high-passed data.
    """
    if (new_sample_rate * number_cropped_seconds) < 18:
        warnings.warn(
            f"Number of cropped seconds requested {number_cropped_seconds}s is low compared to the sampling rate "
            f"{new_sample_rate}: cropped-seconds x sampling-rate = {number_cropped_seconds*new_sample_rate}."
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
            f"There are {np.sum(nan_mask)} NaNs in the timestream ({np.sum(nan_mask)*100/len(time_series_data)}%"
            f" of the data). These will be ignored in pre-processing."
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
    gates: SegmentList = None
):
    """
    Function to self-gate
    data to be used in the stochastic pipeline.

    Parameters
    =======
    time_series_data: ``gwpy.timeseries.TimeSeries``
        Timeseries data to be analysed in the pipeline.

    tzero : ``int``, optional
        Half-width time duration (seconds) in which the timeseries is
        set to zero. Default is 1.0.

    tpad : ``int``, optional
        Half-width time duration (seconds) in which the Planck window
        is tapered. Default is 0.5.

    whiten : ``bool``, optional
        If True, data will be whitened before gating points are discovered,
        use of this option is highly recommended. Default is True.

    gate_threshold : ``float``, optional
        Amplitude threshold, if the data exceeds this value a gating window
        will be placed. Default is 50.0.

    cluster_window : ``float``, optional
        Time duration (seconds) over which gating points will be clustered.
        Default is 0.5.

    gates: ``gwpy.segments.SegmentList``, optional
        Argument where gates can be explicitly given to this function.
        Those gates would then be applied to the timeseries data. If not applied, equal to None.

    Returns
    =======
    gated: ``gwpy.timeseries.TimeSeries``
        TimeSeries containing the gated data.

    deadtime: ``gwpy.segments.SegmentList``
        SegmentList containing the times that were gated, not including
        any padding applied.

    See also
    --------
    gwpy.timeseries.TimeSeries.gate
        More information `here <https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.gate>`_.
    gwpy.segments.SegmentList
        More information `here <https://gwpy.github.io/docs/stable/api/gwpy.segments.SegmentList/>`_.
    """
    from scipy.signal import find_peaks

    # Find points to gate based on a threshold
    sample = time_series_data.sample_rate.to("Hz").value
    if gates is None:
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
    else:
        deadtime = SegmentList([Segment(k[0], k[1]) for k in gates]).coalesce()

    # return the self-gated timeseries
    gated = time_series_data.mask(deadtime=deadtime, const=0, tpad=tpad)
    return gated, deadtime

def shift_timeseries(time_series_data: timeseries.TimeSeries, time_shift: int = 0):

    """
    Function that shifts a timeseries by an amount ``time_shift``
    in order to perform the timeshifted analysis.

    Parameters
    =======
    time_series_data: ``gwpy.timeseries.TimeSeries``
        Timeseries data to be analysed in the pipeline.

    time_shift: ``int``, optional
        Value of the time shift (in seconds).
        Default value is 0.

    Returns
    =======
    shifted_data: ``gwpy.timeseries.TimeSeries``
        TimeSeries containing the shifted_data.
    """

    if time_shift > 0:
        shifted_data = np.roll(
            time_series_data, int(time_shift / time_series_data.dt.value)
        )
    else:
        shifted_data = time_series_data
    return shifted_data

def preprocessing_data_gwpy_timeseries(
    gwpy_timeseries: timeseries.TimeSeries,
    new_sample_rate: int,
    cutoff_frequency: float,
    number_cropped_seconds: int = 2,
    window_downsampling: str = "hamming",
    ftype: str = "fir",
    time_shift: int = 0,
):
    """
    Function doing the pre-processing of a gwpy timeseries to be used in the remainder of the code.

    Parameters
    =======

    gwpy_timeseries: ``gwpy.timeseries.TimeSeries``
        Timeseries from gwpy.

    new_sample_rate: ``int``
        Sampling rate of the downsampled-timeseries in Hz.

    cutoff_frequency: ``float``
        Frequency (in Hz) from which to start applying the
        high pass filter.

    number_cropped_seconds: ``int``, optional
        Number of seconds to remove at the beginning and end
        of the high-passed data. Default is 2.

    window_downsampling: ``str``, optional
        Type of window used to downsample.Default value is "hamming".

    ftype: ``str``, optional
        Type of filter to use in the downsampling. Default is "fir".

    time_shift: ``int``, optional
        Value of the time shift (in seconds). Default is 0.

    Returns
    =======
    pre_processed_data: ``gwpy.timeseries.TimeSeries``
        Timeseries containing the filtered and high passed data (shifted if time_shift>0).
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
    frametype: str = "",
    input_sample_rate: int = 16384,
):
    """
    Function doing the pre-processing of the data to be used in the
    remainder of the code.

    Parameters
    =======
    IFO: ``str``
        Interferometer name for which to retrieve the data.

    t0: ``int``
        GPS time of the start of the data taking.

    tf: ``int``
        GPS time of the end of the data taking.

    data_type: ``str``
        String indicating the type of data to be read,
        either 'public', 'private' or 'local'.

    channel: ``str``
        Name of the channel (e.g.: "L1:GWOSC-4KHZ_R1_STRAIN").

    frametype: ``str``
        Frame type that contains the channel, only used if data_type=private (e.g.: "L1_HOFT_C00").

    new_sample_rate: ``int``
        Sampling rate of the downsampled-timeseries in Hz.

    cutoff_frequency: ``float``
        Frequency (in Hz) from which to start applying the
        high pass filter.

    segment_duration: ``int``
        Duration (in seconds) of each segment (argument of set_start_time).

    number_cropped_seconds: ``int``, optional
        Number of seconds to remove at the beginning and end of the high-passed data. Default is 2.

    window_downsampling: ``str``, optional
        Type of window used to downsample. Default is "hamming".

    ftype: ``str``, optional
        Type of filter to use in the downsampling. Default is "fir".

    time_shift: ``int``, optional
        Value of the time shift (in seconds). Default is 0.
        
    local_data_path: ``str``, optional
        Path where local gwf frame file is stored. Default is an empty string.

    input_sample_rate: ``int``, optional.
        Sampling rate of the timeseries to be read in Hz. Default is 16384 Hz.

    Returns
    =======
    pre_processed_data: ``gwpy.timeseries.TimeSeries``
        Timeseries containing the filtered and high passed data (shifted if time_shift>0).
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
        frametype=frametype,
        t0=data_start_time - number_cropped_seconds,
        tf=tf,
        local_data_path=local_data_path,
        input_sample_rate=input_sample_rate,
    )

    return preprocessing_data_gwpy_timeseries(
        gwpy_timeseries=time_series_data,
        new_sample_rate=new_sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_cropped_seconds=number_cropped_seconds,
        window_downsampling=window_downsampling,
        ftype=ftype,
        time_shift=time_shift,
    )

def preprocessing_data_timeseries_array(
    t0: int,
    tf: int,
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
    Function performing the pre-processing of a time-series array to be used in the remainder of the code.

    Parameters
    =======
    t0: ``int``
        GPS time of the start of the data taking.

    tf: ``int``
        GPS time of the end of the data taking.

    array: ``np.ndarray``
        Array containing a timeseries.

    new_sample_rate: ``int``
        Sampling rate of the downsampled-timeseries in Hz.

    cutoff_frequency: ``float``
        Frequency (in Hz) from which to start applying the
        high pass filter.

    segment_duration: ``int``
        Duration (in seconds) of each segment (argument of set_start_time).

    sample_rate: ``int``, optional
        Sampling rate of the original timeseries. Default is 4096 Hz.

    number_cropped_seconds: ``int``, optional
        Number of seconds to remove at the beginning and end
        of the high-passed data. Default is 2.

    window_downsampling: ``str``, optional
        Type of window used to downsample. Default is "hamming".

    ftype: ``str``, optional
        Type of filter to use in the downsampling. Default is "fir".

    time_shift: ``int``, optional
        Value of the time shift (in seconds). Default is no time shift.

    Returns
    =======
    pre_processed_data: ``gwpy.timeseries.TimeSeries``
        Timeseries containing the filtered and high passed data (shifted if time_shift>0).
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
        gwpy_timeseries=time_series_data,
        new_sample_rate=new_sample_rate,
        cutoff_frequency=cutoff_frequency,
        number_cropped_seconds=number_cropped_seconds,
        window_downsampling=window_downsampling,
        ftype=ftype,
        time_shift=time_shift,
    )