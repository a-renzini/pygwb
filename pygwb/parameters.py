import argparse
import enum
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

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
    """
    t0: float = 0
    tf: float = 100
    data_type: str = "public"
    channel: str = "GWOSC-16KHZ_R1_STRAIN"
    new_sample_rate: int = 4096
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
    save_data_type: str = "json"
    time_shift: int = 0
    gate_data: bool = False
    gate_tzero: float = 1.0
    gate_tpad: float = 0.5 
    gate_threshold: float = 50.0
    cluster_window: float = 0.5
    gate_whiten: bool = True

    def __post_init__(self):
        if self.coarse_grain:
            self.fft_length = self.segment_duration
        else:
            self.fft_length = int(1 / self.frequency_resolution)

    def save_paramfile(self, output_path):
        """Save parameters to a parameters ini file.
        
        Parameters
        ----------
        output_path: str
            Full path for output parameters ini file. 
        """
        param = configparser.ConfigParser()
        param_dict = asdict(self)
        for key, value in param_dict.items():
            param_dict[key] = str(value)
        param["parameters"] = param_dict
        with open(output_path, "w") as configfile:
            param.write(configfile)

    def update_from_dictionary(self, **kwargs):
        """Update parameters from a dictionary
        
        Parameters
        ----------
        **kwargs: **dictionary
            Dictionary of parameters to update.
        """
        ann = getattr(self, "__annotations__", {})
        for name, dtype in ann.items():
            if name in kwargs:
                try:
                    kwargs[name] = dtype(kwargs[name])
                except TypeError:
                    pass
                setattr(self, name, kwargs[name])

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
        mega_list = config.items('data_specs')
        mega_list.extend(config.items('preprocessing'))
        mega_list.extend(config.items('density_estimation'))
        mega_list.extend(config.items('preprocessing'))
        mega_list.extend(config.items('data_quality'))
        mega_list.extend(config.items('output'))
        dictionary = dict(mega_list)
        if dictionary['alphas_delta_sigma_cut']: dictionary['alphas_delta_sigma_cut'] = json.loads(dictionary['alphas_delta_sigma_cut'])
        if dictionary['interferometer_list']: dictionary['interferometer_list'] = json.loads(dictionary['interferometer_list'])
        dictionary['window_fft_dict'] = dict(config.items("window_fft_specs"))
        dictionary['local_data_path_dict'] = dict(config.items("local_data"))
        for item in dictionary.copy():
            if not dictionary[item]:
                dictionary.pop(item)
        self.update_from_dictionary(**dictionary)

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
            parser.add_argument(f"--{name}", type=dtype, required=False)
        parser.add_argument("--H1", type=str, required=False)
        parser.add_argument("--L1", type=str, required=False)
        parser.add_argument("--V", type=str, required=False)
        parser.add_argument("--window_fftgram", type=str, required=False)
        parsed, _ = parser.parse_known_args(args)
        dictionary = vars(parsed)
        for item in dictionary.copy():
            if dictionary[item] is None:
                dictionary.pop(item)
        local_data_path_dict = {}
        if 'H1' in dictionary:
            local_data_path_dict['H1'] = dictionary['H1']
        if 'L1' in dictionary:
            local_data_path_dict['L1'] = dictionary['L1']
        if 'V' in dictionary:
            local_data_path_dict['V'] = dictionary['V']
        dictionary['local_data_path_dict'] = local_data_path_dict
        window_fft_dict = {}
        if 'window_fftgram' in dictionary:
            window_fft_dict['window_fftgram'] = dictionary['window_fftgram']
        dictionary['window_fft_dict'] = window_fft_dict
        self.update_from_dictionary(**dictionary)


class ParametersHelp(enum.Enum):
    """Description of the arguments in the Parameters class. This is an enumeration class and is not meant for user interaction."""
    t0 = "Initial time."
    tf = "Final time."
    data_type = "Type of data to access/download; options are private, public, local. Default is public."
    channel = "Channel name; needs to match an existing channel. Default is \"GWOSC-16KHZ_R1_STRAIN\" "
    new_sample_rate = "Sample rate to use when downsampling the data (Hz). Default is 4096 Hz."
    cutoff_frequency = "Lower frequency cutoff; applied in filtering in preprocessing (Hz). Default is 11 Hz." 
    segment_duration = "Duration of the individual segments to analyse (seconds). Default is 192 seconds."
    number_cropped_seconds = "Number of seconds to crop at the start and end of the analysed data (seconds). Default is 2 seconds."
    window_downsampling = "Type of window to use in preprocessing. Default is \"hamming\""
    ftype = "Type of filter to use in downsampling. Default is \"fir\""
    frequency_resolution = "Frequency resolution of the final output spectrum (Hz). Default is 1\/32 Hz."
    polarization = "Polarisation type for the overlap reduction function calculation; options are scalar, vector, tensor. Default is tensor."
    alpha = "Spectral index to filter the data for. Default is 0." 
    fref = "Reference frequency to filter the data at (Hz). Default is 25 Hz."
    flow = "Lower frequency to include in the analysis (Hz). Default is 20 Hz."
    fhigh = "Higher frequency to include in the analysis (Hz). Default is 1726 Hz."
    coarse_grain = "Whether to apply coarse graining to the spectra. Default is 0."
    interferometer_list = "List of interferometers to run the analysis with. Default is [\"H1\", \"L1\"]"
    local_data_path_dict = "Dictionary of local data, if the local data option is chosen. Default is empty."
    notch_list_path = "Path to the notch list file. Default is empty."
    N_average_segments_welch_psd = "Number of segments to average over when calculating the psd with Welch method. Default is 2."
    window_fft_dict = "Dictionary containing name and parameters relative to which window to use when producing fftgrams for psds and csds. Default is \"hann\"."
    calibration_epsilon = "Calibation coefficient. Default is 0."
    overlap_factor = "Factor by which to overlap consecutive segments for analysis. Default is 0.5 (50%% overlap)"
    zeropad_csd = "Whether to zeropad the csd or not. Default is True."
    delta_sigma_cut = "Cutoff value for the delta sigma cut. Default is 0.2."
    alphas_delta_sigma_cut = "List of spectral indexes to use in delta sigma cut calculation. Default is [-5, 0, 3]."
    save_data_type = "Suffix for the output data file. Options are hdf5, npz, json, pickle. Default is json."
    time_shift = "Seconds to timeshift the data by in preprocessing. Default is 0."
    gate_data = "Whether to apply self-gating to the data in preprocessing. Default is False."
    gate_tzero = "Gate tzero. Default is 1.0."
    gate_tpad = "Gate tpad. Default is 0.5."
    gate_threshold = "Gate threshold. Default is 50."
    cluster_window = "Cluster window. Default is 0.5."
    gate_whiten = "Whether to whiten when gating. Default is True."

    @property
    def help(self):
        return self.value

    @property
    def argument(self):
        return self.name.lower()
