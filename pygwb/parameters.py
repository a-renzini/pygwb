import argparse
import enum
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import json5

if sys.version_info >= (3, 0):
    import configparser
else:
    import ConfigParser as configparser


@dataclass
class Parameters:
    """
    Parameters class: a dataclass which contains all parameters required for initialising a pygwb Interferometer, a pygwb Baseline, and run pygwb_pipe.

    Attributes
    ----------
    t0 : float
        Initial time.
    tf: float
        Final time.
    data_type: str
        Type of data to access/download; options are private, public, local. Default is public.
    channel: str
        Channel name; needs to match an existing channel. Default is \"GWOSC-16KHZ_R1_STRAIN\"
    new_sample_rate: int
        Sample rate to use when downsampling the data (Hz). Default is 4096 Hz.
    input_sample_rate: int
        Sample rate of the read data (Hz). Default is 16384 Hz.
    cutoff_frequency: int
        Lower frequency cutoff; applied in filtering in preprocessing (Hz). Default is 11 Hz.
    segment_duration: int
        Duration of the individual segments to analyse (seconds). Default is 192 seconds.
    number_cropped_seconds: int
        Number of seconds to crop at the start and end of the analysed data (seconds). Default is 2 seconds.
    window_downsampling: str
        Type of window to use in preprocessing. Default is \"hamming\"
    ftype: str
        Type of filter to use in downsampling. Default is \"fir\"
    frequency_resolution: float
        Frequency resolution of the final output spectrum (Hz). Default is 1\/32 Hz.
    polarization: str
        Polarisation type for the overlap reduction function calculation; options are scalar, vector, tensor. Default is tensor.
    alpha: float
        Spectral index to filter the data for. Default is 0.
    fref: int
        Reference frequency to filter the data at (Hz). Default is 25 Hz.
    flow: int
        Lower frequency to include in the analysis (Hz). Default is 20 Hz.
    fhigh: int
        Higher frequency to include in the analysis (Hz). Default is 1726 Hz.
    coarse_grain: bool
        Whether to apply coarse graining to the spectra. Default is 0.
    interferometer_list: list
        List of interferometers to run the analysis with. Default is [\"H1\", \"L1\"]
    local_data_path_dict: dict
        Dictionary of local data, if the local data option is chosen. Default is empty.
    notch_list_path: str
        Path to the notch list file. Default is empty.
    N_average_segments_welch_psd: int
        Number of segments to average over when calculating the psd with Welch method. Default is 2.
    window_fft_dict: dict
        Dictionary containing name and parameters describing which window to use when producing fftgrams for psds and csds. Default is \"hann\".
    calibration_epsilon: float
        Calibation coefficient. Default is 0.
    overlap_factor: float
        Factor by which to overlap consecutive segments for analysis. Default is 0.5 (50%% overlap)
    zeropad_csd: bool
        Whether to zeropad the csd or not. Default is True.
    delta_sigma_cut: float
        Cutoff value for the delta sigma cut. Default is 0.2.
    alphas_delta_sigma_cut: list
        List of spectral indexes to use in delta sigma cut calculation. Default is [-5, 0, 3].
    save_data_type: str
        Suffix for the output data file. Options are hdf5, npz, json, pickle. Default is json.
    time_shift: int
        Seconds to timeshift the data by in preprocessing. Default is 0.
    gate_data: bool
        Whether to apply self-gating to the data in preprocessing. Default is False.
    gate_tzero: float
        Gate tzero. Default is 1.0.
    gate_tpad: float
        Gate tpad. Default is 0.5.
    gate_threshold: float
        Gate threshold. Default is 50.
    cluster_window: float
        Cluster window. Default is 0.5.
    gate_whiten: bool
        Whether to whiten when gating. Default is True.
    tag: str
        Hint for the read_data function to retrieve one specific type of data, e.g.: C00, C01
    return_naive_and_averaged_sigmas: bool
        option to return naive and sliding sigmas from delta sigma cut
    window_fftgram_dict: dictionary, optional
        Dictionary with window characteristics. Default is `(window_fftgram_dict={"window_fftgram": "hann"}`
    """

    t0: float = 0
    tf: float = 100
    data_type: str = "public"
    channel: str = "GWOSC-16KHZ_R1_STRAIN"
    new_sample_rate: int = 4096
    input_sample_rate: int = 16384
    cutoff_frequency: int = 11
    segment_duration: int = 192
    number_cropped_seconds: int = 2
    window_downsampling: str = "hamming"
    ftype: str = "fir"
    frequency_resolution: float = 0.03125
    polarization: str = "tensor"
    alpha: float = 0
    fref: int = 25
    flow: int = 20
    fhigh: int = 1726
    coarse_grain: bool = False
    interferometer_list: List = field(default_factory=lambda: ["H1", "L1"])
    local_data_path_dict: dict = field(default_factory=lambda: {})
    notch_list_path: str = ""
    N_average_segments_welch_psd: int = 2
    window_fft_dict: dict = field(default_factory=lambda: {"window_fftgram": "hann"})
    calibration_epsilon: float = 0
    overlap_factor: float = 0.5
    zeropad_csd: bool = True
    delta_sigma_cut: float = 0.2
    alphas_delta_sigma_cut: List = field(default_factory=lambda: [-5, 0, 3])
    save_data_type: str = "npz"
    time_shift: int = 0
    gate_data: bool = False
    gate_tzero: float = 1.0
    gate_tpad: float = 0.5
    gate_threshold: float = 50.0
    cluster_window: float = 0.5
    gate_whiten: bool = True
    tag: str = "C00"
    return_naive_and_averaged_sigmas: bool = False

    def __post_init__(self):
        if self.coarse_grain:
            self.fft_length = self.segment_duration
        else:
            self.fft_length = int(1 / self.frequency_resolution)

    def update_from_dictionary(self, kwargs):
        """Update parameters from a dictionary

        Parameters
        ----------
        kwargs: dictionary
            Dictionary of parameters to update.
        """
        ann = getattr(self, "__annotations__", {})
        for name, dtype in ann.items():
            if name in kwargs:
                try:
                    kwargs[name] = dtype(kwargs[name]) if kwargs[name] != 'False' else False
                except TypeError:
                    pass
                setattr(self, name, kwargs[name])
        for name in kwargs:
            if name not in ann.keys():
                warnings.warn(
                    f"{name} is not an expected parameter and will be ignored."
                )

    def update_from_file(self, path: str) -> None:
        """Update parameters from an ini file

        Parameters
        ----------
        path: str
            Path to parameters ini file to use to update class.
        """
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(path)
        mega_list = config.items("data_specs")
        mega_list.extend(config.items("preprocessing"))
        mega_list.extend(config.items("density_estimation"))
        mega_list.extend(config.items("postprocessing"))
        mega_list.extend(config.items("gating"))
        mega_list.extend(config.items("data_quality"))
        mega_list.extend(config.items("output"))
        dictionary = dict(mega_list)
        if dictionary["alphas_delta_sigma_cut"]:
            dictionary["alphas_delta_sigma_cut"] = json5.loads(
                dictionary["alphas_delta_sigma_cut"]
            )
        if dictionary["interferometer_list"]:
            dictionary["interferometer_list"] = json5.loads(
                dictionary["interferometer_list"]
            )
        dictionary["window_fft_dict"] = dict(config.items("window_fft_specs"))
        dictionary["local_data_path_dict"] = dict(config.items("local_data"))
        possible_ifos = ["H1", "L1", "V", "K"]
        for ifo in possible_ifos:
            if ifo in dictionary["local_data_path_dict"]:
                if dictionary["local_data_path_dict"][ifo].startswith("["):
                    dictionary["local_data_path_dict"][ifo] = json.loads(
                        dictionary["local_data_path_dict"][ifo]
                    )
                else:
                    dictionary["local_data_path_dict"][ifo] = dictionary[
                        "local_data_path_dict"
                    ][ifo]
        for item in dictionary.copy():
            if not dictionary[item]:
                dictionary.pop(item)
        self.update_from_dictionary(dictionary)

    def update_from_arguments(self, args: List[str]) -> None:
        """Update parameters from a set of arguments

        Parameters
        ----------
        args: list
            List of arguments to update in the Class. Format must coincide to argparse formatting, e.g.,
            ['--t0', '0', '--tf', '100']

        Notes
        -----
        Not all possible options are available through argument updating. The two exceptions are the dictionary
        attributes which can not be parsed easily by argparse. These are
        * local_data_path_dict: this is composed by paths passed individually using the following notation
            --H1 : path to data relative to H1
            --L1 : path to data relative to L1
            --V1 : path to data relative to V1
        These are the options currently supported for this dictionary. To add paths to different interferometers, pass
        these as part of an .ini file, in the relevant section [local_data].
        * window_fft_dict: this is composed by the single argument
            -- window_fftgram
        This is the only option currently supported. To use windows that require extra parameters, pass these as part of an
        .ini file, in the relevant section [window_fft_specs].
        """
        if not args:
            return
        ann = getattr(self, "__annotations__", {})
        parser = argparse.ArgumentParser()
        for name, dtype in ann.items():
            if dtype == List:
                parser.add_argument(f"--{name}", type=str, nargs='+', required=False)
            else:
                parser.add_argument(f"--{name}", type=dtype, required=False)

        parser.add_argument("--h1", type=str, required=False)
        parser.add_argument("--l1", type=str, required=False)
        parser.add_argument("--v", type=str, required=False)
        parser.add_argument("--window_fftgram", type=str, required=False)
        parsed, _ = parser.parse_known_args(args)
        dictionary = vars(parsed)
        for item in dictionary.copy():
            if dictionary[item] is None:
                dictionary.pop(item)
        local_data_path_dict = {}
        possible_ifos = ["H1", "L1", "V", "K"]
        for ifo in possible_ifos:
            if ifo.lower() in dictionary:
                if dictionary[ifo.lower()].startswith("["):
                    local_data_path_dict[ifo] = json.loads(dictionary[ifo.lower()])
                else:
                    local_data_path_dict[ifo] = dictionary[ifo.lower()]
                dictionary.pop(ifo.lower())
        if local_data_path_dict:
            dictionary["local_data_path_dict"] = local_data_path_dict
        if "window_fftgram" in dictionary:
            window_fft_dict = {}
            window_fft_dict["window_fftgram"] = dictionary["window_fftgram"]
            dictionary.pop("window_fftgram")
            dictionary["window_fft_dict"] = window_fft_dict
        self.update_from_dictionary(dictionary)

    def save_paramfile(self, output_path):
        """Save parameters to a parameters ini file.

        Parameters
        ----------
        output_path: str
            Full path for output parameters ini file.
        """
        param = configparser.ConfigParser()
        param.optionxform = str
        param_dict = asdict(self)
        # for key, value in param_dict.items():
        #    param_dict[key] = str(value)
        data_specs_dict = {}
        data_specs_dict["interferometer_list"] = param_dict["interferometer_list"]
        data_specs_dict["t0"] = param_dict["t0"]
        data_specs_dict["tf"] = param_dict["tf"]
        data_specs_dict["data_type"] = param_dict["data_type"]
        data_specs_dict["channel"] = param_dict["channel"]
        data_specs_dict["time_shift"] = param_dict["time_shift"]
        param["data_specs"] = data_specs_dict

        preprocessing_dict = {}
        preprocessing_dict["new_sample_rate"] = param_dict["new_sample_rate"]
        preprocessing_dict["cutoff_frequency"] = param_dict["cutoff_frequency"]
        preprocessing_dict["segment_duration"] = param_dict["segment_duration"]
        preprocessing_dict["number_cropped_seconds"] = param_dict[
            "number_cropped_seconds"
        ]
        preprocessing_dict["window_downsampling"] = param_dict["window_downsampling"]
        preprocessing_dict["ftype"] = param_dict["ftype"]
        param["preprocessing"] = preprocessing_dict
        
        gating_dict = {}
        gating_dict["gate_data"] = param_dict["gate_data"]
        gating_dict["gate_whiten"] = param_dict["gate_whiten"]
        gating_dict["gate_tzero"] = param_dict["gate_tzero"]
        gating_dict["gate_tpad"] = param_dict["gate_tpad"]
        gating_dict["gate_threshold"] = param_dict["gate_threshold"]
        gating_dict["cluster_window"] = param_dict["cluster_window"]
        param["gating"] = gating_dict

        param["window_fft_specs"] = self.window_fft_dict

        density_estimation_dict = {}
        density_estimation_dict["frequency_resolution"] = param_dict[
            "frequency_resolution"
        ]
        density_estimation_dict["N_average_segments_welch_psd"] = param_dict[
            "N_average_segments_welch_psd"
        ]
        density_estimation_dict["coarse_grain"] = param_dict["coarse_grain"]
        density_estimation_dict["overlap_factor"] = param_dict["overlap_factor"]
        density_estimation_dict["zeropad_csd"] = param_dict["zeropad_csd"]
        param["density_estimation"] = density_estimation_dict

        postprocessing_dict = {}
        postprocessing_dict["polarization"] = param_dict["polarization"]
        postprocessing_dict["alpha"] = param_dict["alpha"]
        postprocessing_dict["fref"] = param_dict["fref"]
        postprocessing_dict["flow"] = param_dict["flow"]
        postprocessing_dict["fhigh"] = param_dict["fhigh"]
        param["postprocessing"] = postprocessing_dict

        data_quality_dict = {}
        data_quality_dict["notch_list_path"] = param_dict["notch_list_path"]
        data_quality_dict["calibration_epsilon"] = param_dict["calibration_epsilon"]
        data_quality_dict["alphas_delta_sigma_cut"] = param_dict[
            "alphas_delta_sigma_cut"
        ]
        data_quality_dict["delta_sigma_cut"] = param_dict["delta_sigma_cut"]
        data_quality_dict["return_naive_and_averaged_sigmas"] = param_dict[
            "return_naive_and_averaged_sigmas"
        ]
        param["data_quality"] = data_quality_dict

        output_dict = {}
        output_dict["save_data_type"] = param_dict["save_data_type"]
        param["output"] = output_dict

        param["local_data"] = self.local_data_path_dict

        with open(output_path, "w") as configfile:
            param.write(configfile)


class ParametersHelp(enum.Enum):
    """Description of the arguments in the Parameters class. This is an enumeration class and is not meant for user interaction."""

    t0 = "Initial time."
    tf = "Final time."
    data_type = "Type of data to access/download; options are private, public, local. Default is public."
    channel = 'Channel name; needs to match an existing channel. Default is "GWOSC-16KHZ_R1_STRAIN" '
    new_sample_rate = (
        "Sample rate to use when downsampling the data (Hz). Default is 4096 Hz."
    )
    input_sample_rate = "Sample rate of the read data (Hz). Default is 16384 Hz."
    cutoff_frequency = "Lower frequency cutoff; applied in filtering in preprocessing (Hz). Default is 11 Hz."
    segment_duration = "Duration of the individual segments to analyse (seconds). Default is 192 seconds."
    number_cropped_seconds = "Number of seconds to crop at the start and end of the analysed data (seconds). Default is 2 seconds."
    window_downsampling = 'Type of window to use in preprocessing. Default is "hamming"'
    ftype = 'Type of filter to use in downsampling. Default is "fir"'
    frequency_resolution = (
        "Frequency resolution of the final output spectrum (Hz). Default is 1\/32 Hz."
    )
    polarization = "Polarisation type for the overlap reduction function calculation; options are scalar, vector, tensor. Default is tensor."
    alpha = "Spectral index to filter the data for. Default is 0."
    fref = "Reference frequency to filter the data at (Hz). Default is 25 Hz."
    flow = "Lower frequency to include in the analysis (Hz). Default is 20 Hz."
    fhigh = "Higher frequency to include in the analysis (Hz). Default is 1726 Hz."
    coarse_grain = "Whether to apply coarse graining to the spectra. Default is 0."
    interferometer_list = (
        'List of interferometers to run the analysis with. Default is ["H1", "L1"]'
    )
    local_data_path_dict = "Dictionary of local data, if the local data option is chosen. Default is empty."
    notch_list_path = "Path to the notch list file. Default is empty."
    N_average_segments_welch_psd = "Number of segments to average over when calculating the psd with Welch method. Default is 2."
    window_fft_dict = 'Dictionary containing name and parameters relative to which window to use when producing fftgrams for psds and csds. Default is "hann".'
    calibration_epsilon = "Calibation coefficient. Default is 0."
    overlap_factor = "Factor by which to overlap consecutive segments for analysis. Default is 0.5 (50%% overlap)"
    zeropad_csd = "Whether to zeropad the csd or not. Default is True."
    delta_sigma_cut = "Cutoff value for the delta sigma cut. Default is 0.2."
    alphas_delta_sigma_cut = "List of spectral indexes to use in delta sigma cut calculation. Default is [-5, 0, 3]."
    save_data_type = "Suffix for the output data file. Options are hdf5, npz, json, pickle. Default is json."
    time_shift = "Seconds to timeshift the data by in preprocessing. Default is 0."
    gate_data = (
        "Whether to apply self-gating to the data in preprocessing. Default is False."
    )
    gate_tzero = "Gate tzero. Default is 1.0."
    gate_tpad = "Gate tpad. Default is 0.5."
    gate_threshold = "Gate threshold. Default is 50."
    cluster_window = "Cluster window. Default is 0.5."
    gate_whiten = "Whether to whiten when gating. Default is True."
    tag = "Hint for the read_data function to retrieve one specific type of data, e.g.: C00, C01"
    return_naive_and_averaged_sigmas = "option to return naive and sliding sigmas from delta sigma cut. Default value: False"

    @property
    def help(self):
        return self.value

    @property
    def argument(self):
        return self.name.lower()
