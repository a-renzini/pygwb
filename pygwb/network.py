import os
import warnings

import bilby
import gwpy
import numpy as np
from loguru import logger

from .baseline import Baseline
from .notch import StochNotchList
from .omega_spectra import OmegaSpectrum
from .postprocessing import (
    calc_Y_sigma_from_Yf_sigmaf,
    combine_spectra_with_sigma_weights,
)
from .simulator import Simulator


class Network(object):
    """
    Network object for stochastic analyses with multiple functionalities:
        * data simulation
        * stochastic pre-processing
        * isotropic stochastic analysis
    """

    def __init__(
        self,
        name,
        interferometers,
        duration=None,
        frequencies=None,
        calibration_epsilon=0,
        notch_list_path=None,
        overlap_factor=0.5,
        zeropad_csd=True,
        window_fftgram_dict={"window_fftgram": "hann"},
        N_average_segments_welch_psd=2,
    ):
        """
        Instantiate a Network object.

        Parameters
        ----------
        name: str
            Name for the network, e.g H1H2
        interferometers: list of intereferometer objects
        duration: float, optional
            the duration in seconds of each data segment in the interferometers. None by default, in which case duration is inherited from the interferometers.
        frequencies: array_like, optional
            the frequency array for the Baseline and
            interferometers
        calibration_epsilon: float, optional
            calibration uncertainty for this baseline -- currently only supports a single notch list for all baselines
        notch_list_path: str, optional
            file path of the baseline notch list -- currently only supports a single notch list for all baselines
        overlap_factor: float, optional
            factor by which to overlap the segments in the psd and csd estimation. Default is 1/2, if set to 0 no overlap is performed.
        zeropad_csd: bool, optional
            if True, applies zeropadding in the csd estimation. True by default.
        window_fftgram_dict: dictionary, optional
            Dictionary containing name and parameters describing which window to use when producing fftgrams for psds and csds. Default is \"hann\".
        N_average_segments_welch_psd: int, optional
            Number of segments used for PSD averaging (from both sides of the segment of interest)
            N_avg_segs should be even and >= 2
        """
        self.name = name
        self.interferometers = interferometers
        self.Nifo = len(interferometers)
        self.set_duration(duration)

        combo_tuples = []
        for j in range(1, len(interferometers)):
            for k in range(j):
                combo_tuples.append((k, j))

        baselines = []
        for i, j in combo_tuples:
            base_name = (
                f"{self.interferometers[i].name} - {self.interferometers[j].name}"
            )
            baselines.append(
                Baseline(
                    base_name,
                    self.interferometers[i],
                    self.interferometers[j],
                    duration=duration,
                    frequencies=frequencies,
                    calibration_epsilon=calibration_epsilon,
                    notch_list_path=notch_list_path,
                    overlap_factor=overlap_factor,
                    zeropad_csd=zeropad_csd,
                    window_fftgram_dict=window_fftgram_dict,
                    N_average_segments_welch_psd=N_average_segments_welch_psd,
                )
            )

        self.baselines = baselines

    @classmethod
    def from_baselines(cls, name, baselines):
        """
        Initialise a network from a set of baselines. Takes care to unpack the interferometers from each baselines and sets them in the Network.
        """
        if not all(baselines[0].duration == base.duration for base in baselines[1:]):
            raise AssertionError(
                "All baselines used to initialise must have same duration set."
            )
        network = cls(
            name,
            interferometers=[],
            duration=baselines[0].duration,
        )
        network.baselines = baselines
        ifo_visited = set()
        interferometers = []
        for base in baselines:
            if base.interferometer_1.name not in ifo_visited:
                ifo_visited.add(base.interferometer_1.name)
                interferometers.append(base.interferometer_1)
            if base.interferometer_2.name not in ifo_visited:
                ifo_visited.add(base.interferometer_2.name)
                interferometers.append(base.interferometer_2)
        network.interferometers = interferometers
        network.Nifo = len(interferometers)
        return network

    def set_duration(self, duration):
        """Sets the duration for the Network and Interferometers

        Note: the cross-checks that durations match in all the interferometers are done by each Baseline.

        Parameters
        ==========
        duration: float, optional
            The duration to set for the Network and interferometers
        """
        if duration is not None:
            self.duration = duration
            for ifo in self.interferometers:
                ifo.duration = duration
            return
        try:
            duration = next(
                ifo.duration for ifo in self.interferometers if ifo.duration is not None
            )
        except StopIteration:
            warnings.warn(
                "The Network duration is not set, "
                "and the interferometer durations are all None."
            )
            self.duration = None
            return

        check_dur = all(ifo.duration == duration for ifo in self.interferometers)
        if not check_dur:
            raise AssertionError(
                "The interferometer durations don't match! "
                "The Network can't handle this. "
                "Make sure that the interferometer durations are the same."
            )
            # for ifo in self.interferometers:
            #    ifo.duration = duration
        self.duration = duration

    def set_frequency_mask(self, notch_list_path="", flow=20, fhigh=1726):
        """
        Set frequency mask to frequencies attribute.

        Parameters
        ==========
        notch_list_path: str
            Path to notch list to apply to frequency array.
        """
        mask = (self.frequencies >= flow) & (self.frequencies <= fhigh)
        if notch_list_path:
            notch_list = StochNotchList.load_from_file(notch_list_path)
            notch_mask = notch_list.get_notch_mask(self.frequencies)
            mask = np.logical_and(mask, notch_mask)
        self.frequency_mask = mask

    def set_interferometer_data_from_simulator(
        self,
        GWB_intensity,
        N_segments,
        sampling_frequency=None,
        start_time=None,
        inject_into_data_flag=False,
    ):
        """
        Fill interferometers with data from simulation. Data can already be present in the intereferometers of the network,
        in which case the simulated data will be injected on top of the data already present.

        Parameters
        ==========

        GWB_intensity: gwpy.frequencyseries.FrequencySeries
            A gwpy.frequencyseries.FrequencySeries containing the desired strain power spectrum
        N_segments: int
            Number of segments to simulate
        sampling_frequency: float
            Sampling frequency at which the data needs to be simulated. If not specified (None), will check for interferometer's
            sampling frequency.
        start_time: float
            Start time of the simulated data. If not passed (None), will check for interferometer's timeseries start time.
            If not specified either, start time will default to zero.
        inject_into_data_flag: boolean
            Flag that specifies whether or not the simulated data needs to be injected into data, i.e. if there is already
            data present in the interferometers of the network. If so, only data will be simulated and no extra noise will
            be added on top of the simulated data.
        """
        no_noise = inject_into_data_flag
        if start_time == None:
            try:
                start_time = self.interferometers[0].timeseries.times.value[0]
            except:
                start_time = 0.0
                warnings.warn(
                    "User did not specify a start time. Setting start time to zero."
                )

        if sampling_frequency == None:
            try:
                sampling_frequency = self.interferometers[0].sampling_frequency
            except:
                raise ValueError(
                    "Sampling frequency was not set. Please set sampling frequency in interferometers or pass directly to function."
                )

        data_simulator = Simulator(
            self.interferometers,
            GWB_intensity,
            N_segments,
            duration=self.duration,
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            no_noise=no_noise,
        )
        data = data_simulator.get_data_for_interferometers()

        if inject_into_data_flag:
            for ifo in self.interferometers:
                ifo.set_strain_data_from_gwpy_timeseries(
                    ifo.strain_data.to_gwpy_timeseries().inject(data[ifo.name])
                )
        else:
            for ifo in self.interferometers:
                ifo.set_strain_data_from_gwpy_timeseries(data[ifo.name])

    def save_interferometer_data_to_file(self, save_dir="./", file_format="hdf5"):
        """
        Save interferometer strain data to a file. This method relies on the gwpy  TimeSeries.write method. Typically used when simulating a signal for a whole network of interferometers.
        Note: this will save a single frame file with a set of interferometer data; each strain channel is labelled by its interferometer.

        Parameters
        ==========
        save dir: str, optional
            The path of the output folder. Defaults to the local folder.
        file format: str, optional
            The format of the output file. Defaults to hdf5 file. Acceptable formats are standard gwpy TimeSeries.write formats.
        """
        file_name = f"{self.name}_STRAIN-{int(self.interferometers[0].strain_data.start_time)}-{int(self.interferometers[0].strain_data.duration)}.{file_format}"
        file_path = os.path.join(save_dir, file_name)
        data_dict = gwpy.timeseries.TimeSeriesDict()
        for ifo in self.interferometers:
            channel = f"STRAIN_{ifo.name}"
            data_dict[channel] = ifo.strain_data.to_gwpy_timeseries()
        data_dict.write(file_path, format=file_format)

    def combine_point_estimate_sigma_spectra(self):
        """
        Combines the point estimate and sigma spectra from different baselines in the Network and stores them as attributes.
        """
        try:
            point_estimate_spectra = [
                base.point_estimate_spectrum for base in self.baselines
            ]
            sigma_spectra = [base.sigma_spectrum for base in self.baselines]
        except AttributeError:
            raise AttributeError("The Baselines of the Network have not been set!")

        alphas = np.array([spec.alpha for spec in point_estimate_spectra])
        frefs = np.array([spec.fref for spec in point_estimate_spectra])
        h0s = np.array([spec.h0 for spec in point_estimate_spectra])
        dfs = np.array([spec.df.value for spec in point_estimate_spectra])
        f0s = np.array([spec.f0.value for spec in point_estimate_spectra])

        dict_attributes = {
            "alpha": [alphas, "spectral indices"],
            "fref": [frefs, "reference frequencies"],
            "h0": [h0s, "cosmology h0"],
            "df": [dfs, "sampling frequency"],
            "f0": [f0s, "begin frequency"],
        }

        for key in dict_attributes:
            if not np.all(dict_attributes[key][0] == dict_attributes[key][0][0]):
                raise ValueError(
                    f"The {dict_attributes[key][1]} of the spectra in each Baseline don't match! Spectra may not be combined."
                )

        pt_est_spec, sig_spec = combine_spectra_with_sigma_weights(
            np.array(point_estimate_spectra), np.array(sigma_spectra)
        )
        self.point_estimate_spectrum = OmegaSpectrum(
            pt_est_spec,
            alpha=alphas[0],
            fref=frefs[0],
            h0=h0s[0],
            name="Y_spectrum_network",
            frequencies=self.baselines[0].frequencies,
        )
        self.sigma_spectrum = OmegaSpectrum(
            sig_spec,
            alpha=alphas[0],
            fref=frefs[0],
            h0=h0s[0],
            name="sigma_spectrum_network",
            frequencies=self.baselines[0].frequencies,
        )

    def set_point_estimate_sigma(
        self,
        notch_list_path="",
        flow=20,
        fhigh=1726,
    ):
        """
        Set point estimate sigma based the combined spectra from each Baseline. This is estimate of omega_gw in each frequency bin.

        Parameters
        ==========
        notch_list_path: str, optional
            Path to the notch list to use in the spectrum; if the notch_list isn't set in the baseline,
            user can pass it directly here. If it is not set and if none is passed no notches will be applied.
        flow: float, optional
            Low frequency. Default is 20 Hz.
        fhigh: float, optional
            High frequency. Default is 1726 Hz.
        """
        if not hasattr(self, "point_estimate_spectrum"):
            logger.info(
                "Point estimate and sigma spectra have not been set before. Setting it now..."
            )
            self.combine_point_estimate_sigma_spectra()

        self.frequencies = self.point_estimate_spectrum.frequencies.value
        self.set_frequency_mask(notch_list_path, flow=flow, fhigh=fhigh)

        Y, sigma = calc_Y_sigma_from_Yf_sigmaf(
            self.point_estimate_spectrum,
            self.sigma_spectrum,
            frequency_mask=self.frequency_mask,
        )

        self.point_estimate = Y
        self.sigma = sigma


# TODO: add options in the network to apply dsc, frequency notching at the network level.
