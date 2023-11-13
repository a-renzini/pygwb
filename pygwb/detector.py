"""
The ``detector`` module is a subclass of bilby's Interferometer class (more details `here <https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.detector.interferometer.Interferometer.html#bilby.gw.detector.interferometer.Interferometer>`_) which is charged with handling, storing and
saving all relevant interferometer data. 
It handles all data analysis parts relating to the individual detectors in a baseline of a network.
For example, it loads the data from a certain channel and computes the power spectral density (PSD) of the detector.
    
Examples
--------
    
In this example, we load in data from the publicly available `GWOSC <https://gwosc.org/>`_ servers using the ``detector`` module.
We gate the data, compute the PSD and the average PSD of the ``detector`` object.
This example gives a brief overview of the most important features of the ``pygwb.detector`` module.
We start by importing the Interferometer class from ``pygwb``.
    
>>> from pygwb.detector import Interferometer

To load in the data, an empty ``detector`` object is first created, for which,
based on the name of the object, the module will make an Interferometer object without data.
The name can be any one of the detectors supported in ``bilby.gw.detector``,
the parent class of our Interferometer class (more details `here <https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.detector.interferometer.Interferometer.html#bilby.gw.detector.interferometer.Interferometer>`_).
    
>>> ifo_1 = Interferometer.get_empty_interferometer("H1")
    
Then, we load the data using the ``set_timeseries_from`` methods, and pass a start and end time, ``t0`` and ``tf``, respectively.
We are obtaining data from the `GWOSC <https://gwosc.org/>`_ servers, i.e., public data, and indicate this by marking the ``data_type`` tag to public. We use 
the channel "H1:GWOSC-4KHZ_R1_STRAIN", for illustrative purposes. All the other parameters are set to their
default value.

>>> ifo_1.set_timeseries_from_channel_name(
    "H1:GWOSC-4KHZ_R1_STRAIN",
    t0=1247644138,
    tf=1247648138,
    data_type="public",
    local_data_path = "",
    new_sample_rate=4096,
    input_sample_rate=4096,
    cutoff_frequency=11,
    segment_duration=192,
    number_cropped_seconds=2,
    window_downsampling="hamming",
    ftype="fir",
    time_shift=0,
    )
    
To illustrate one of the features of the module, we gate the detector data. This procedure applies a window to stretches of data to get rid of
glitches and other artefacts. More information on the gating procedure can be found in `this technical note <https://dcc.ligo.org/public/0172/P2000546/002/gating-mdc.pdf>`_.  

>>> ifo_1.gate_data_apply(
    gate_tzero=1.0,
    gate_tpad=0.5,
    gate_threshold=50.0,
    cluster_window=0.5,
    gate_whiten=True,
    )

The module also allows to compute the PSD spectrogram of the detector. A spectrogram
shows the PSD both per time and per frequency. After specifying the desired frequency resolution, one can call the
``set_psd_spectrogram`` method.
    
>>> frequency_resolution = 1/32.
>>> ifo_1.set_psd_spectrogram(
        frequency_resolution,
        overlap_factor=0.5,
        window_fftgram_dict_welch_psd={"window_fftgram": "hann"},
        overlap_factor_welch_psd=0.5,
        )
        
Finally, the average PSD of the detector can also be computed, by invoking ``set_average_psd``.
    
>>> ifo_1.set_average_psd(N_average_segments_welch_psd=2)

This highlights some of the features of the ``detector`` module. For more details, we refer the reader 
to the remainder of the ``detector`` API documentation.
"""

import logging
import os

import bilby.gw.detector
from bilby.gw.detector import PowerSpectralDensity
from gwpy.segments import SegmentList

from .preprocessing import (
    preprocessing_data_channel_name,
    preprocessing_data_gwpy_timeseries,
    preprocessing_data_timeseries_array,
    self_gate_data,
)
from .spectral import before_after_average, power_spectral_density


class Interferometer(bilby.gw.detector.Interferometer):
    def __init__(self, *args, **kwargs):
        """Instantiate an Interferometer class

        Parameters
        =======
        
        *args : arguments passed to the (parent) bilby's Interferometer class.
        **kwargs : keyword arguments passed to the (parent) bilby's Interferometer class.

        name: ``str``
            Interferometer name, e.g. H1 for LIGO Hanford.
        power_spectral_density: ``bilby.gw.detector.PowerSpectralDensity``
            Power spectral density determining the sensitivity of the detector.
        minimum_frequency: ``float``
            Minimum frequency to analyse for detector.
        maximum_frequency: ``float``
            Maximum frequency to analyse for detector.
        length: ``float``
            Length of the interferometer in km.
        latitude: ``float``
            Latitude North in degrees (South is negative).
        longitude: ``float``
            Longitude East in degrees (West is negative).
        elevation: ``float``
            Height above surface in metres.
        xarm_azimuth: ``float``
            Orientation of the x arm in degrees North of East.
        yarm_azimuth: ``float``
            Orientation of the y arm in degrees North of East.
        xarm_tilt: ``float``, optional
            Tilt of the x arm in radians above the horizontal defined by
            ellipsoid earth model in LIGO-T980044-08.
        yarm_tilt: ``float``, optional
            Tilt of the y arm in radians above the horizontal.
        calibration_model: Recalibration
            Calibration model, this applies the calibration correction to the
            template, the default model applies no correction.
        timeseries : ``gwpy.timeseries.TimeSeries``
            TimeSeries object with resampling/high-pass filter applied. This is an additional attribute of the class.
        psd_spectrogram : ``gwpy.spectrogram.Spectrogram``
            gwpy Spectrogram of power spectral density. This is an additional attribute of the class.
        gates: ``gwpy.segments.SegmentList``
            List of segments that have been gated, not including any additional padding. This is an additional attribute of the class.
        gate_pad: ``float``
            Duration of padding used when applying gates. This is an additional attribute of the class.

        See also
        --------
        bilby.gw.detector.Interferometer
            Test
            
        gwpy.segments.SegmentList
            More information `here <https://gwpy.github.io/docs/stable/api/gwpy.segments.SegmentList/>`_.
        """
        self.gates = SegmentList()
        self.gate_pad = None
        super(Interferometer, self).__init__(*args, **kwargs)

    @classmethod
    def get_empty_interferometer(cls, name):
        """
        A class method to get an Interferometer class object from a given ifo name.
        Empty means no data has been read into this object.

        Parameters
        =======
        name : ``str``
            Interferometer name, e.g. H1 for LIGO Hanford.

        Returns
        =======
        interferometer: ``pygwb.Interferometer``
            Interferometer instance of pygwb.
        """
        filename = os.path.join(
            os.path.dirname(bilby.gw.detector.__file__),
            "detectors",
            f"{name}.interferometer",
        )
        try:
            parameters = {}
            with open(filename, "r") as parameter_file:
                lines = parameter_file.readlines()
                for line in lines:
                    if line[0] == "#" or line[0] == "\n":
                        continue
                    split_line = line.split("=")
                    key = split_line[0].strip()
                    value = eval(
                        "=".join(split_line[1:]),
                        dict(__builtins__=dict()),
                        dict(PowerSpectralDensity=PowerSpectralDensity),
                    )
                    parameters[key] = value
            if "shape" not in parameters.keys():
                logging.debug("Assuming L shape for name")
            elif parameters["shape"].lower() in ["l", "ligo"]:
                parameters.pop("shape")
            elif parameters["shape"].lower() in ["triangular", "triangle"]:
                raise ValueError("Triangular detectors are not implemented yet.")
            else:
                raise IOError(
                    f"{filename} could not be loaded. Invalid parameter 'shape'."
                )
            ifo_cls = cls(**parameters)
            ifo_cls._check_ifo_name(name)
            return ifo_cls
        except OSError:
            raise ValueError(f"Interferometer {name} not implemented")

    @classmethod
    def from_parameters(cls, name, parameters):
        """
        A class method to get an Interferometer class from a given argparser object.

        Parameters
        =======
        name : ``str``
            Interferometer name, e.g. H1 for LIGO Hanford.
        parameters : argparser object
            This contains attributes defined for command line options.

        Returns
        =======
        interferometer: ``pygwb.Interferometer``
            Instance of the pygwb interferometer object.
        """
        ifo = cls.get_empty_interferometer(name)
        channel = str(ifo.name + ":" + parameters.channel)

        ifo.set_timeseries_from_channel_name(
            channel,
            t0=parameters.t0,
            tf=parameters.tf,
            data_type=parameters.data_type,
            frametype=parameters.frametype,
            local_data_path=parameters.local_data_path,
            new_sample_rate=parameters.new_sample_rate,
            cutoff_frequency=parameters.cutoff_frequency,
            segment_duration=parameters.segment_duration,
            number_cropped_seconds=parameters.number_cropped_seconds,
            window_downsampling=parameters.window_downsampling,
            ftype=parameters.ftype,
            time_shift=parameters.time_shift,
            input_sample_rate=parameters.input_sample_rate,
        )
        return ifo

    def set_timeseries_from_channel_name(self, channel, **kwargs):
        """
        Set a timeseries attribute from a given channel name.

        Parameters
        =======
        channel: ``str``
            Name of the channel (e.g.: "L1:GWOSC-4KHZ_R1_STRAIN") from which to load the data.

        **kwargs : keyword arguments passed to preprocess module.

        See also
        --------
        pygwb.preprocessing.preprocessing_data_channel_name
        """
        t0 = kwargs.pop("t0")
        tf = kwargs.pop("tf")
        data_type = kwargs.pop("data_type")
        frametype = kwargs.pop("frametype")
        local_data_path = kwargs.pop("local_data_path")
        new_sample_rate = kwargs.pop("new_sample_rate")
        input_sample_rate = kwargs.pop("input_sample_rate")
        cutoff_frequency = kwargs.pop("cutoff_frequency")
        segment_duration = kwargs.pop("segment_duration")
        number_cropped_seconds = kwargs.pop("number_cropped_seconds")
        window_downsampling = kwargs.pop("window_downsampling")
        ftype = kwargs.pop("ftype")
        time_shift = kwargs.pop("time_shift")
        self.duration = segment_duration
        self.timeseries = preprocessing_data_channel_name(
            IFO=self.name,
            channel=channel,
            t0=t0,
            tf=tf,
            data_type=data_type,
            frametype=frametype,
            local_data_path=local_data_path,
            new_sample_rate=new_sample_rate,
            cutoff_frequency=cutoff_frequency,
            segment_duration=segment_duration,
            number_cropped_seconds=number_cropped_seconds,
            window_downsampling=window_downsampling,
            ftype=ftype,
            time_shift=time_shift,
            input_sample_rate=input_sample_rate,
        )
        self._check_timeseries_channel_name(channel)
        self.sampling_frequency = new_sample_rate

    def set_timeseries_from_timeseries_array(
        self, timeseries_array, sample_rate, **kwargs
    ):
        """
        Set a timeseries attribute from a given numpy array.

        Parameters
        =======
        timeseries_array: ``np.ndarray``
            Timeseries strain data as numpy array object
        sample_rate: ``int``
            Sample rate of the timeseries in the array (in Hz)
        **kwargs : keyword arguments passed to preprocess module.

        See also
        --------
        pygwb.preprocessing.preprocessing_data_timeseries_array
        """
        t0 = kwargs.pop("t0")
        tf = kwargs.pop("tf")
        data_type = kwargs.pop("data_type")
        new_sample_rate = kwargs.pop("new_sample_rate")
        cutoff_frequency = kwargs.pop("cutoff_frequency")
        segment_duration = kwargs.pop("segment_duration")
        number_cropped_seconds = kwargs.pop("number_cropped_seconds")
        window_downsampling = kwargs.pop("window_downsampling")
        ftype = kwargs.pop("ftype")
        time_shift = kwargs.pop("time_shift")
        self.duration = segment_duration
        self.timeseries = preprocessing_data_timeseries_array(
            array=timeseries_array,
            t0=t0,
            tf=tf,
            new_sample_rate=new_sample_rate,
            cutoff_frequency=cutoff_frequency,
            segment_duration=segment_duration,
            sample_rate=sample_rate,
            number_cropped_seconds=number_cropped_seconds,
            window_downsampling=window_downsampling,
            ftype=ftype,
            time_shift=time_shift,
        )
        self.timeseries.channel = kwargs.pop("channel")
        self._check_timeseries_sample_rate(new_sample_rate)
        self.sampling_frequency = sample_rate

    def set_timeseries_from_gwpy_timeseries(self, gwpy_timeseries, **kwargs):
        """
        Set a timeseries attribute from a given ``gwpy.timeseries`` object. More information 
        on the gwpy.timeseries object can be found `here <https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries>`_.

        Parameters
        =======
        gwpy_timeseries: ``gwpy.timeseries.TimeSeries``
            Timeseries strain data as ``gwpy.timeseries`` object.

        **kwargs : keyword arguments passed to preprocess module.

        See also
        --------
        pygwb.preprocessing.preprocessing_data_gwpy_timeseries
        """
        new_sample_rate = kwargs.pop("new_sample_rate")
        segment_duration = kwargs.pop("segment_duration")
        cutoff_frequency = kwargs.pop("cutoff_frequency")
        number_cropped_seconds = kwargs.pop("number_cropped_seconds")
        window_downsampling = kwargs.pop("window_downsampling")
        ftype = kwargs.pop("ftype")
        time_shift = kwargs.pop("time_shift")
        self.duration = segment_duration
        self.timeseries = preprocessing_data_gwpy_timeseries(
            gwpy_timeseries=gwpy_timeseries,
            new_sample_rate=new_sample_rate,
            cutoff_frequency=cutoff_frequency,
            number_cropped_seconds=number_cropped_seconds,
            window_downsampling=window_downsampling,
            ftype=ftype,
            time_shift=time_shift,
        )
        self.timeseries.channel = kwargs.pop("channel")
        self._check_timeseries_sample_rate(new_sample_rate)
        self.sampling_frequency = new_sample_rate

    def set_psd_spectrogram(
        self,
        frequency_resolution,
        coarse_grain=False,
        overlap_factor=0.5,
        window_fftgram_dict={"window_fftgram": "hann"},
        overlap_factor_welch=0.5,
    ):
        """
        Set psd_spectrogram attribute from given spectrum-related information.

        Parameters
        =======
        frequency_resolution: ``float``
            Frequency resolution of the final PSDs. This sets the time duration
            over which FFTs are calculated in the pwelch method.
        coarse_grain: ``bool``
            Coarse-graining flag. If True, PSD will be estimated via coarse-graining,
            as opposed to Welch-averaging. Default is False.
        overlap_factor: ``float``, optional
            Amount of overlap between adjacent segments (ranges between 0 and 1).
            This factor should be same as the one used for cross_spectral_density
            (default 0, no overlap).
        window_fftgram_dict: ``dictionary``, optional
            Dictionary containing name and parameters describing which window to use when producing fftgrams
            for psd estimation. Default is \"hann\".
        overlap_factor_welch: ``float``, optional
            Overlap factor to use when using Welch's method to estimate the PSD (NOT coarsegraining).
            For \"hann\" window use 0.5 overlap_factor and for \"boxcar\" window use 0 overlap_factor.
            Default is 0.5 (50% overlap), which is optimal when using Welch's method with a \"hann\" window.

        See also
        --------
        pygwb.spectral.power_spectral_density
        """
        # PSD estimation needs zeropadding when using coarse-graining
        zeropad_psd = coarse_grain

        self.psd_spectrogram = power_spectral_density(
            self.timeseries,
            self.duration,
            frequency_resolution,
            coarse_grain=coarse_grain,
            zeropad=zeropad_psd,
            overlap_factor=overlap_factor,
            window_fftgram_dict=window_fftgram_dict,
            overlap_factor_welch=overlap_factor_welch,
        )
        self.psd_spectrogram.channel = self.timeseries.channel
        self._check_spectrogram_channel_name(self.timeseries.channel.name)
        self._check_spectrogram_frequency_resolution(frequency_resolution)

    def set_average_psd(self, N_average_segments=2):
        """
        Set average_psd attribute from the existing raw PSD.

        Parameters
        =======
        N_average_segments: ``int``, optional
            Number of segments used for PSD averaging (from both sides of the segment of interest).
            N_avg_segs should be even and >= 2.
        
        See also
        --------
        pygwb.spectral.before_after_average
        """
        try:
            self.average_psd = before_after_average(
                self.psd_spectrogram, self.duration, N_average_segments
            )
        except AttributeError:
            print(
                "PSDs have not been calculated yet! Need to set_psd_spectrogram first."
            )

    def gate_data_apply(self, **kwargs):
        """
        Self-gate the gwpy.timeseries associated with this timeseries. The list
        of times gated and the padding applied are stored as properties of the Interferometer.

        Parameters
        =======
        gate_tzero : ``float``
            Half-width time duration (seconds) in which the timeseries is
            set to zero.
        gate_tpad : ``float``
            Half-width time duration (seconds) in which the Planck window
            is tapered.
        gate_threshold : ``float``
            Amplitude threshold, if the data exceeds this value a gating window
            will be placed.
        cluster_window : ``float``
            Time duration (seconds) over which gating points will be clustered.
        gate_whiten : ``bool``
            If True, data will be whitened before gating points are discovered,
            use of this option is highly recommended.
            
        See also
        --------
        pygwb.preprocessing.self_gate_data

        gwpy.timeseries.TimeSeries.gate : the function used for the gating of the data itself.
        """
        gate_tzero = kwargs.pop("gate_tzero")
        gate_tpad = kwargs.pop("gate_tpad")
        gate_threshold = kwargs.pop("gate_threshold")
        cluster_window = kwargs.pop("cluster_window")
        gate_whiten = kwargs.pop("gate_whiten")
        self.timeseries, new_gates = self_gate_data(
            self.timeseries,
            tzero=gate_tzero,
            tpad=gate_tpad,
            gate_threshold=gate_threshold,
            cluster_window=cluster_window,
            whiten=gate_whiten,
        )
        self.gates = self.gates | new_gates
        self.gate_pad = gate_tpad

    def apply_gates_from_file(self, loaded_object, index, **kwargs):
        """
        Load gates from a pygwb output file and apply them to the Interferometer object. 
        The gated times are stored as a property of the object.
        
        Parameters
        =======
        loaded_object : 
            Object that represents the data in the output file, e.g. a loaded npz-object.
        index : ``int``
            Integer representing the correct Interferometer object in the baseline.
        gate_tpad : ``float``
            Half-width time duration (in seconds) in which the Planck window
            is tapered.

        See also
        --------
        pygwb.preprocessing.self_gate_data
        """
        gates = loaded_object[f"ifo_{index}_gates"]
        gate_tpad = kwargs.pop("gate_tpad")

        self.timeseries, new_gates = self_gate_data(
            self.timeseries,
            tpad=gate_tpad,
            gates=gates,
        )

        self.gates = self.gates | new_gates
        self.gate_pad = gate_tpad

    def _check_ifo_name(self, name):
        if not self.name == name:
            raise AssertionError(
                "The ifo name in Interferometer class does not match given name!"
            )

    def _check_timeseries_channel_name(self, channel_name):
        if not self.timeseries.channel.name == channel_name:
            raise AssertionError(
                "Channel name in timeseries does not match given channel!"
            )

    def _check_timeseries_sample_rate(self, sample_rate):
        if not self.timeseries.sample_rate.value == sample_rate:
            raise AssertionError(
                "Sampling rate in timeseries does not match given sampling rate!"
            )

    def _check_spectrogram_channel_name(self, channel_name):
        if not self.psd_spectrogram.channel.name == channel_name:
            raise AssertionError(
                "Channel name in psd_spectrogram does not match given channel!"
            )

    def _check_spectrogram_frequency_resolution(self, frequency_resolution):
        if not self.psd_spectrogram.df.value == frequency_resolution:
            raise AssertionError(
                "Frequency resolution in psd_spectrogram does not match given frequency resolution!"
            )
