"""
Many parameters enter in a full gravitational-wave background analysis. The ``parameters`` module is meant to facilitate
the handling of this large amount of parameters. It contains the ``Parameters`` class which is used to store and handle 
all parameters of ``pygwb``.

This module contains only one class, the ``Parameters`` class and its accompanying helper class ``ParametersHelp``.
An object of the ``Parameters`` class is used as the storage object of all parameters in the ``pygwb`` analysis.
It is a container for all the required ingredients to perform a stochastic gravitational-wave background analysis.

Examples
--------

When using a packaged script, parameters passed to the script directly will be stored in a ``Parameters`` object. 
To pass these, the standard ``argparse`` usage applies; for example, when using ``pygwb_pipe``.

>>> pygwb_pipe --parameter_to_pass [my-parameter-value]

A parameter class may also be instantiated from a parameter file using the appropriate method, ``Parameters.update_from_file``.
A parameter file must be structured specifically to be read by the ``Parameters`` class. The following structure should be followed::

    [data_specs]
    interferometer_list: ["IFO1", "IFO2"]
    t0:
    tf:
    data_type:
    channel:
    frametype:
    time_shift:
    [preprocessing]
    new_sample_rate: 
    input_sample_rate: 
    cutoff_frequency: 
    segment_duration: 
    number_cropped_seconds: 
    window_downsampling: 
    ftype: 
    [gating]
    gate_data: 
    gate_whiten: 
    gate_tzero: 
    gate_tpad: 
    gate_threshold: 
    cluster_window: 
    [window_fft_specs]
    window_fftgram: 
    alpha:
    [window_fft_welch_specs]
    window_fftgram: 
    alpha:
    [density_estimation]
    frequency_resolution: 
    N_average_segments_psd: 
    coarse_grain_psd: 
    coarse_grain_csd: 
    overlap_factor_welch: 
    overlap_factor: 
    [postprocessing]
    polarization:  
    alpha: 
    fref: 
    flow: 
    fhigh: 
    [data_quality]
    notch_list_path: 
    calibration_epsilon: 
    alphas_delta_sigma_cut:
    delta_sigma_cut: 
    return_naive_and_averaged_sigmas: 
    [output]
    save_data_type:
    [local_data]
    local_data_path:

All values of parameters have a default value and need not be passed when the user doesn't want to specify a different value. 
An example parameters file may be found in ``pygwb/pygwb_pipe/parameters.ini``.
Details on the usage may be found in the class descriptions and by calling the help function of the ``pygwb_pipe`` script. 

In a very small subset of cases, some parameters are passed differently through the parameter file compared to the command line, and not all combinations are possible through the command line. 
These are specifically the cases where sections of the parameter file are parsed as dictionaries, to allow for multiple parameters to be passed without having been specified in the ``Parameters`` class.
These are the ``window_fft_specs`` and ``window_fft_welch_specs`` sections which are parsed as dictionaries and passed directly to ``scipy`` which expects all the parameters relevant to the specific window employed (e.g., ``alpha`` for a Tukey window, etc.).
When passing arguments relative to these sections from the command line directly, these should be passed as, e.g.:

>>> pygwb_pipe --window_fftgram my-window-1 --window_fftgram_welch my-window-2

Notes
-----

**Special Parameters**

For the most part, the usage of the ``Parameters`` object is self evident and clarified by the descriptions provided by the class itself and ``ParametersHelp``. However, certain parameters, which are `interferometer-specific`, need to be passed with specific formatting to be properly parsed. 
These are: ``channel`` ``frametype`` ``input_sample_rate`` ``local_data_path`` ``time_shift``.
When passing in the same value of one of these parameters for both interferometers, this only needs to be passed once; for example, to pass in the same frame type for both interferometers, one simply adds to the parameter file (in the appropriate section):

>>> [data_specs]
>>> interferometer_list: ["IFO1", "IFO2"]
>>> frametype: MY_FRAME

However, when passing in different frame types for different interferometers, this needs to be specified as

>>> [data_specs]
>>> interferometer_list: ["IFO1", "IFO2"]
>>> frametype: IFO1:MY_FRAME_1,IFO2:MY_FRAME_2

where IFO1 and IFO2 match the names provided for the interferometers in the ``interferometer_list`` parameter.

Users should not interact with the ``ParametersHelp`` class.
"""
import argparse
import configparser
import enum
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import List

import json5


@dataclass
class Parameters:
    """
    A dataclass which contains all parameters required for initialising a pygwb ``Interferometer``,
    a pygwb ``Baseline``, and run ``pygwb_pipe``.

    Attributes
    =======
    t0 : ``float``
        Initial time.
    tf: ``float``
        Final time.
    data_type: ``str``
        Type of data to access/download; options are private, public, local. Default is public.
    channel: ``str``
        Channel name; needs to match an existing channel. Default is \"GWOSC-16KHZ_R1_STRAIN\".
    frametype: ``str``
        Frame type; Optional, desired channel needs to be found in listed frametype. Only used when data_type=private.
        Default is empty.
    new_sample_rate: ``int``
        Sample rate to use when downsampling the data (Hz). Default is 4096 Hz.
    input_sample_rate: ``int``
        Sample rate of the read data (Hz). Default is 16384 Hz.
    cutoff_frequency: ``int``
        Lower frequency cutoff; applied in filtering in preprocessing (Hz). Default is 11 Hz.
    segment_duration: ``int``
        Duration of the individual segments to analyse (seconds). Default is 192 seconds.
    number_cropped_seconds: ``int``
        Number of seconds to crop at the start and end of the analysed data (seconds). Default is 2 seconds.
    window_downsampling: ``str``
        Type of window to use in preprocessing. Default is \"hamming\".
    ftype: ``str``
        Type of filter to use in downsampling. Default is \"fir\".
    frequency_resolution: ``float``
        Frequency resolution of the final output spectrum (Hz). Default is 1\/32 Hz.
    polarization: ``str``
        Polarization type for the overlap reduction function calculation; options are scalar, vector, tensor.
        Default is tensor.
    alpha: ``float``
        Spectral index to filter the data for. Default is 0.
    fref: ``int``
        Reference frequency to filter the data at (Hz). Default is 25 Hz.
    flow: ``int``
        Lower frequency to include in the analysis (Hz). Default is 20 Hz.
    fhigh: ``int``
        Higher frequency to include in the analysis (Hz). Default is 1726 Hz.
    interferometer_list: ``list``
        List of interferometers to run the analysis with. Default is [\"H1\", \"L1\"].
    local_data_path: ``str``
        Path(s) to local data, if the local data option is chosen. Default is empty.
    notch_list_path: ``str``
        Path to the notch list file. Default is empty.
    coarse_grain_psd: ``bool``
        Whether to apply coarse graining to obtain PSD spectra. Default is False.
    coarse_grain_csd: ``bool``
        Whether to apply coarse graining to obtain CSD spectra. Default is True.
    overlap_factor_welch: ``float``
        Overlap factor to use when if using Welch's method to estimate spectra (NOT coarsegraining).
        For \"hann\" window use 0.5 overlap_factor and for \"boxcar\" window use 0 overlap_factor.
        Default is 0.5 (50% overlap), which is optimal when using Welch's method with a \"hann\" window.
    N_average_segments_psd: ``int``
        Number of segments to average over when calculating the psd with Welch method. Default is 2.
    window_fft_dict: ``dict``
        Dictionary containing name and parameters describing which window to use when producing fftgrams
        for psds and csds. Default is \"hann\".
    window_fft_dict_welch: ``dict``
        Dictionary containing name and parameters relative to which window to use when producing fftgrams
        for pwelch calculation. Default is \"hann\".
    calibration_epsilon: ``float``
        Calibration coefficient. Default is 0.
    overlap_factor: ``float``
        Factor by which to overlap consecutive segments for analysis. Default is 0.5 (50%% overlap).
    delta_sigma_cut: ``float``
        Cutoff value for the delta sigma cut. Default is 0.2.
    alphas_delta_sigma_cut: ``list``
        List of spectral indexes to use in delta sigma cut calculation. Default is [-5, 0, 3].
    save_data_type: ``str``
        Suffix for the output data file. Options are hdf5, npz, json, pickle. Default is json.
    time_shift: ``int``
        Seconds to timeshift the data by in preprocessing. Default is 0.
    gate_data: ``bool``
        Whether to apply self-gating to the data in preprocessing. Default is False.
    gate_tzero: ``float``
        Half-width time duration (seconds) in which the timeseries is set to zero. Default is 1.0.
    gate_tpad: ``float``
        Half-width time duration (seconds) in which the Planck window is tapered. Default is 0.5.
    gate_threshold: ``float``
        Amplitude threshold, if the data exceeds this value a gating window will be placed. Default is 50.
    cluster_window: ``float``
        Time duration (seconds) over which gating points will be clustered. Default is 0.5.
    gate_whiten: ``bool``
        Whether to whiten when gating. Default is True.
    tag: ``str``
        Hint for the read_data function to retrieve one specific type of data, e.g.: C00, C01.
    return_naive_and_averaged_sigmas: ``bool``
        Option to return naive and sliding sigmas from delta sigma cut.
    """
    t0: float = 0
    tf: float = 100
    interferometer_list: List = field(default_factory=lambda: ["H1", "L1"])
    data_type: str = "public"
    channel: str = "GWOSC-16KHZ_R1_STRAIN"
    frametype: str = ""
    new_sample_rate: int = 4096
    input_sample_rate: int = 16384
    cutoff_frequency: float = 11
    segment_duration: int = 192
    number_cropped_seconds: int = 2
    window_downsampling: str = "hamming"
    ftype: str = "fir"
    frequency_resolution: float = 0.03125
    polarization: str = "tensor"
    alpha: float = 0
    fref: float = 25
    flow: float = 20
    fhigh: float = 1726
    local_data_path: str = ""
    notch_list_path: str = ""
    coarse_grain_psd: bool = False
    coarse_grain_csd: bool = True
    overlap_factor_welch: float = 0.5
    N_average_segments_psd: int = 2
    window_fft_dict: dict = field(default_factory=lambda: {"window_fftgram": "hann"})
    window_fft_dict_welch: dict = field(default_factory=lambda: {"window_fftgram": "hann"})
    calibration_epsilon: float = 0
    overlap_factor: float = 0.5
    delta_sigma_cut: float = 0.2
    alphas_delta_sigma_cut: List = field(default_factory=lambda: [-5, 0, 3])
    save_data_type: str = "npz"
    time_shift: int = 0
    path_gate_data: str = ""
    gate_data: bool = False
    gate_tzero: float = 1.0
    gate_tpad: float = 0.5
    gate_threshold: float = 50.0
    cluster_window: float = 0.5
    gate_whiten: bool = True
    tag: str = ""
    return_naive_and_averaged_sigmas: bool = False

    def update_from_dictionary(self, kwargs):
        """Update parameters from a dictionary.

        Parameters
        =======
        kwargs: ``dict``
            Dictionary of parameters to update.
        """
        ann = getattr(self, "__annotations__", {})
        for name, dtype in ann.items():
            if name in kwargs:
                try:
                    if not bool(re.search("^.+:.+,.+:.+$", kwargs[name])): 
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
        """Update parameters from an ini file.

        Parameters
        =======
        path: ``str``
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
        mega_list.extend(config.items("local_data"))
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
        dictionary["window_fft_dict_welch"] = dict(config.items("window_fft_welch_specs"))
        for item in dictionary.copy():
            if not dictionary[item]:
                dictionary.pop(item)
        self.update_from_dictionary(dictionary)

    def update_from_arguments(self, args: List[str]) -> None:
        """Update parameters from a set of arguments.

        Parameters
        =======
        args: ``list``
            List of arguments to update in the class. Format must coincide to argparse formatting, e.g.,
            ['--t0', '0', '--tf', '100'].

        Notes
        -----
        Not all possible options are available through argument updating. The two exceptions are the dictionary
        attributes which can not be parsed easily by argparse. These are
        * window_fft_dict: this is composed by the single argument
            -- window_fftgram
        This is the only option currently supported. To use windows that require extra parameters, pass these as part
        of an .ini file, in the relevant section [window_fft_specs].
        * window_fft_dict_welch: this is composed by the single argument
            -- window_fftgram_welch
        """
        if not args:
            return
        ann = getattr(self, "__annotations__", {})
        parser = argparse.ArgumentParser()
        for name, dtype in ann.items():
            if dtype == List:
                parser.add_argument(f"--{name}", type=str, nargs='+', required=False)
            else:
                parser.add_argument(f"--{name}", type=str, required=False)

        parser.add_argument("--window_fftgram", type=str, required=False)
        parser.add_argument("--window_fftgram_welch", type=str, required=False)
        parsed, _ = parser.parse_known_args(args)
        dictionary = vars(parsed)
        for item in dictionary.copy():
            if dictionary[item] is None:
                dictionary.pop(item)
        if "window_fftgram" in dictionary:
            window_fft_dict = {}
            wfgram_name, *wfgram = dictionary.pop("window_fftgram").split(',')
            window_fft_dict["window_fftgram"] = wfgram_name
            if wfgram_name.lower() == 'tukey':
                window_fft_dict["alpha"] = wfgram[0]
            elif wfgram_name.lower() == 'hann':
                pass
            else: 
                raise ValueError(
                    f"Window {wfgram_name} not supported from command line! "
                    f"Please try submitting through a full parameter ini file."
                )
            dictionary["window_fft_dict"] = window_fft_dict
        if "window_fftgram_welch" in dictionary:
            window_fft_dict_welch = {}
            wfgram_name_welch, *wfgram_welch = dictionary.pop("window_fftgram_welch").split(',')
            window_fft_dict_welch["window_fftgram"] = wfgram_name_welch
            if wfgram_name_welch.lower() == 'tukey':
                window_fft_dict_welch["alpha"] = wfgram[0]
            elif wfgram_name_welch.lower() == 'hann':
                pass
            else: 
                raise ValueError(
                    f"Window {wfgram_name_welch} not supported from command line! "
                    f"Please try submitting through a full parameter ini file."
                )
            dictionary["window_fft_dict_welch"] = window_fft_dict_welch
        self.update_from_dictionary(dictionary)

    def save_paramfile(self, output_path):
        """Save parameters to a parameters ini file.

        Parameters
        =======
        output_path: ``str``
            Full path for output parameters ini file.
        """
        param = configparser.ConfigParser()
        param.optionxform = str
        param_dict = asdict(self)
        # for key, value in param_dict.items():
        #    param_dict[key] = str(value)
        data_specs_dict = {}
        data_specs_dict["t0"] = param_dict["t0"]
        data_specs_dict["tf"] = param_dict["tf"]
        data_specs_dict["interferometer_list"] = param_dict["interferometer_list"]
        data_specs_dict["data_type"] = param_dict["data_type"]
        data_specs_dict["channel"] = param_dict["channel"]
        data_specs_dict["frametype"] = param_dict["frametype"]
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
        gating_dict["path_gate_data"] = param_dict["path_gate_data"]
        gating_dict["gate_data"] = param_dict["gate_data"]
        gating_dict["gate_whiten"] = param_dict["gate_whiten"]
        gating_dict["gate_tzero"] = param_dict["gate_tzero"]
        gating_dict["gate_tpad"] = param_dict["gate_tpad"]
        gating_dict["gate_threshold"] = param_dict["gate_threshold"]
        gating_dict["cluster_window"] = param_dict["cluster_window"]
        param["gating"] = gating_dict

        param["window_fft_specs"] = self.window_fft_dict
        param["window_fft_welch_specs"] = self.window_fft_dict_welch

        density_estimation_dict = {}
        density_estimation_dict["frequency_resolution"] = param_dict[
            "frequency_resolution"
        ]
        density_estimation_dict["N_average_segments_psd"] = param_dict[
            "N_average_segments_psd"
        ]
        density_estimation_dict["coarse_grain_psd"] = param_dict["coarse_grain_psd"]
        density_estimation_dict["coarse_grain_csd"] = param_dict["coarse_grain_csd"]
        density_estimation_dict["overlap_factor_welch"] = param_dict["overlap_factor_welch"]
        density_estimation_dict["overlap_factor"] = param_dict["overlap_factor"]
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

        local_data_dict = {}
        local_data_dict["local_data_path"] = param_dict["local_data_path"]
        param["local_data"] = local_data_dict

        output_dict = {}
        output_dict["save_data_type"] = param_dict["save_data_type"]
        param["output"] = output_dict

        with open(output_path, "w") as configfile:
            param.write(configfile)

    def parse_ifo_parameters(self):
        """Parse the parameters of the analysis pipeline into a dictionary 
        with arguments per interferometer of the pipeline.
        
        Returns
        =======
        param_dict: ``dict``
            A dictionary containing the parameters of the analysis per interferometer used in the analysis.
        """
        ifo_parameters = ['channel', 'frametype', 'input_sample_rate', 'local_data_path', 'time_shift']
        ifo_list = self.interferometer_list
        param_dict = {}
        for ifo in ifo_list:
            param_dict[ifo] = Parameters()
        current_param_dict = self.__dict__
        for attr in current_param_dict:
            if attr in ifo_parameters:
                attr_str = str(current_param_dict[attr])
                attr_split = attr_str.split(',')
                if len(attr_split) > 1:
                    attr_dict = {key: value for key, value in (pair.split(':') for pair in attr_split)} 
                    for ifo in ifo_list:
                        param_dict[ifo].update_from_dictionary({attr: attr_dict[ifo]})
                else:
                    for ifo in ifo_list:
                        param_dict[ifo].update_from_dictionary({attr: current_param_dict[attr]})
            else:
                for ifo in ifo_list:
                    param_dict[ifo].update_from_dictionary({attr: current_param_dict[attr]})

        return param_dict

class ParametersHelp(enum.Enum):
    """
    Description of the arguments in the Parameters class. 
    
    Notes
    -----
    This is an enumeration class and is not meant for user interaction.
    """
    t0 = "Initial time."
    tf = "Final time."
    interferometer_list = (
        'List of interferometers to run the analysis with. Default is ["H1", "L1"]'
    )
    data_type = "Type of data to access/download; options are private, public, local. Default is public."
    channel = 'Channel name; needs to match an existing channel. Default is "GWOSC-16KHZ_R1_STRAIN" '
    frametype = (
        'Frame type; Optional, desired channel needs to be found in listed frametype. Only used when data_type=private.'
        ' Default is empty. '
    )
    new_sample_rate = (
        "Sample rate to use when downsampling the data (Hz). Default is 4096 Hz."
    )
    input_sample_rate = "Sample rate of the read data (Hz). Default is 16384 Hz."
    cutoff_frequency = "Lower frequency cutoff; applied in filtering in preprocessing (Hz). Default is 11 Hz."
    segment_duration = "Duration of the individual segments to analyse (seconds). Default is 192 seconds."
    number_cropped_seconds = (
        "Number of seconds to crop at the start and end of the analysed data (seconds). Default is 2 seconds."
    )
    window_downsampling = 'Type of window to use in preprocessing. Default is "hamming"'
    ftype = 'Type of filter to use in downsampling. Default is "fir"'
    frequency_resolution = (
        "Frequency resolution of the final output spectrum (Hz). Default is 1\/32 Hz."
    )
    polarization = (
        "Polarization type for the overlap reduction function calculation; options are scalar, vector, tensor."
        " Default is tensor."
    )
    alpha = "Spectral index to filter the data for. Default is 0."
    fref = "Reference frequency to filter the data at (Hz). Default is 25 Hz."
    flow = "Lower frequency to include in the analysis (Hz). Default is 20 Hz."
    fhigh = "Higher frequency to include in the analysis (Hz). Default is 1726 Hz."
    local_data_path = "Path(s) to local data, if the local data option is chosen. Default is empty."
    notch_list_path = "Path to the notch list file. Default is empty."
    coarse_grain_psd = "Whether to apply coarse graining to obtain PSD spectra. Default is False."
    coarse_grain_csd = "Whether to apply coarse graining to obtain CSD spectra. Default is True."
    overlap_factor_welch = (
        "Overlap factor to use when if using Welch's method to estimate spectra (NOT coarsegraining). "
        "For \"hann\" window use 0.5 overlap_factor and for \"boxcar\" window use 0 overlap_factor. "
        "Default is 0.5 (50%% overlap), which is optimal when using Welch's method with a \"hann\" window."
    )
    N_average_segments_psd = (
        "Number of segments to average over when calculating the psd with Welch's method. Default is 2."
    )
    window_fft_dict = (
        "Dictionary containing name and parameters relative to which window to use when producing fftgrams"
        " for psds and csds. Default is \"hann\"."
    )
    window_fft_dict_welch = (
        "Dictionary containing name and parameters relative to which window to use when producing fftgrams"
        " for pwelch calculation. Default is \"hann\"."
    )
    calibration_epsilon = "Calibration coefficient. Default is 0."
    overlap_factor = "Factor by which to overlap consecutive segments for analysis. Default is 0.5 (50%% overlap)"
    delta_sigma_cut = "Cutoff value for the delta sigma cut. Default is 0.2."
    alphas_delta_sigma_cut = "List of spectral indexes to use in delta sigma cut calculation. Default is [-5, 0, 3]."
    save_data_type = "Suffix for the output data file. Options are hdf5, npz, json, pickle. Default is json."
    time_shift = "Seconds to timeshift the data by in preprocessing. Default is 0."
    path_gate_data = (
        "Path to the pygwb output containing information about gates. "
        "If loading a single file, it has to be an .npzfile "
        "with the same structure as a pygwb output file. Default is an empty string."
    )
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
