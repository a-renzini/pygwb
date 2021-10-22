import logging
import os

import bilby.gw.detector
import numpy as np
from bilby.gw.detector.psd import PowerSpectralDensity

from .pre_processing import (
    preprocessing_data_channel_name,
    preprocessing_data_gwpy_timeseries,
    preprocessing_data_timeseries_array,
)
from .spectral import power_spectral_density


class Interferometer(bilby.gw.detector.Interferometer):

    """Subclass of bilby's Interferometer class"""

    def __init__(self, *args, **kwargs):
        """Instantiate an Interferometer class

        Parameters
        ==========
        *args : arguments passed to the (parent) bilby's Interferometer class.
        **kwargs : keyword arguments passed to the (parent) bilby's Inteferometer class.

        Nominally, the bilby's Interferometer class takes the following arguments.
        name: str
            Interferometer name, e.g., H1.
        power_spectral_density: bilby.gw.detector.PowerSpectralDensity
            Power spectral density determining the sensitivity of the detector.
        minimum_frequency: float
            Minimum frequency to analyse for detector.
        maximum_frequency: float
            Maximum frequency to analyse for detector.
        length: float
            Length of the interferometer in km.
        latitude: float
            Latitude North in degrees (South is negative).
        longitude: float
            Longitude East in degrees (West is negative).
        elevation: float
            Height above surface in metres.
        xarm_azimuth: float
            Orientation of the x arm in degrees North of East.
        yarm_azimuth: float
            Orientation of the y arm in degrees North of East.
        xarm_tilt: float, optional
            Tilt of the x arm in radians above the horizontal defined by
            ellipsoid earth model in LIGO-T980044-08.
        yarm_tilt: float, optional
            Tilt of the y arm in radians above the horizontal.
        calibration_model: Recalibration
            Calibration model, this applies the calibration correction to the
            template, the default model applies no correction.

        See https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.detector.interferometer.Interferometer.html#bilby.gw.detector.interferometer.Interferometer
        for the detailed docs of the parent class.

        Additional attributes
        timeseries : gwpy timeseries
            timeseries object with resampling/high-pass filter applied.
        psd_spectrogram : gwpy spectrogram
            gwpy spectrogram of power spectral density

        """

        super(Interferometer, self).__init__(*args, **kwargs)

    @classmethod
    def get_empty_interferometer(cls, name):
        """
        A classmethod to get an Interferometer class from a given ifo name

        Parameters
        ==========
        name : str
            Interferometer identifier.

        Returns
        =======
        interferometer: Interferometer
            Interferometer instance

        """
        filename = os.path.join(
            os.path.dirname(bilby.gw.detector.__file__),
            "detectors",
            f"{name}.interferometer",
        )
        try:
            parameters = dict()
            with open(filename, "r") as parameter_file:
                lines = parameter_file.readlines()
                for line in lines:
                    if line[0] == "#" or line[0] == "\n":
                        continue
                    split_line = line.split("=")
                    key = split_line[0].strip()
                    value = eval("=".join(split_line[1:]))
                    parameters[key] = value
            if "shape" not in parameters.keys():
                logging.debug("Assuming L shape for name")
            elif parameters["shape"].lower() in ["l", "ligo"]:
                parameters.pop("shape")
            elif parameters["shape"].lower() in ["triangular", "triangle"]:
                raise ValueError("Triangular detectros are not implemented yet.")
            else:
                raise IOError(
                    f"{filename} could not be loaded. Invalid parameter 'shape'."
                )
            return cls(**parameters)
        except OSError:
            raise ValueError(f"Interferometer {name} not implemented")

    def set_timeseries_from_channel_name(self, channel, **kwargs):
        """
        A classmethod to get an Interferometer class from a given ifo name

        Parameters
        ==========
        channel: str
            Name of the channel (e.g.: "L1:GWOSC-4KHZ_R1_STRAIN") from which to load the data.

        **kwargs : keyword arguments passed to preprocess module.

        """

        self.timeseries = preprocessing_data_channel_name(
            IFO=self.name, channel=channel, **kwargs
        )

    def set_timeseries_from_timeseries_array(self, timeseries_array, **kwargs):
        """
        A classmethod to get an Interferometer class from a given ifo name

        Parameters
        ==========
        timeseries_array: numpy array
            timeseries strain data as numpy array object

        **kwargs : keyword arguments passed to preprocess module.

        """

        self.timeseries = preprocessing_data_timeseries_array(
            IFO=self.name, timeseries_array=timeseries_array, **kwargs
        )

    def set_timeseries_from_gwpy_timeseries(self, gwpy_timeseries, **kwargs):
        """
        A classmethod to get an Interferometer class from a given ifo name

        Parameters
        ==========
        gwpy_timeseries: gwpy.timeseries
            timeseries strain data as gwpy.timeseries object

        **kwargs : keyword arguments passed to preprocess module.

        """

        self.timeseries = preprocessing_data_gwpy_timeseries(
            IFO=self.name, gwpy_timeseries=gwpy_timeseries, **kwargs
        )

    def set_pre_processed_timeseries_from_channel_name(self, *args, **kwargs):
        """
        A classmethod to get an Interferometer class from a given ifo name

        Parameters
        ==========
        gwpy_timeseries: gwpy.timeseries
            timeseries strain data as gwpy.timeseries object

        **kwargs : keyword arguments passed to preprocess module.

        """

        self.timeseries = preprocessing_data_channel_name(*args, **kwargs)

    def set_psd_spectrogram(
        self,
        frequency_resolution,
        do_overlap=False,
        overlap_factor=0.5,
        zeropad=False,
        window_fftgram="hann",
    ):
        """
        A classmethod to set psd frequency series from a given timeseries as an attribute of given Interferometer object

        Parameters
        ==========
        frequencies : numpy array
            frequency array

        """

        # psd_array = spectral.psd(self.timeseries, frequencies)
        self.psd_spectrogram = power_spectral_density(
            self.timeseries,
            self.duration,
            frequency_resolution,
            do_overlap=do_overlap,
            overlap_factor=overlap_factor,
            zeropad=zeropad,
            window_fftgram=window_fftgram,
        )
