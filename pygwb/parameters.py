import argparse
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
    coarse_grain: int = 0
    interferometer_list: List = field(default_factory=lambda: ["H1", "L1"])
    local_data_path_dict: dict = field(default_factory=lambda: {})
    notch_list_path: str = ""
    N_average_segments_welch_psd: int = 2
    window_fftgram: str = "hann"
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
        self.overlap = self.segment_duration / 2
        if self.coarse_grain:
            self.fft_length = self.segment_duration
        else:
            self.fft_length = int(1 / self.frequency_resolution)

    def save_paramfile(self, output_path):
        param = configparser.ConfigParser()
        param_dict = asdict(self)
        for key, value in param_dict.items():
            param_dict[key] = str(value)
        param["parameters"] = param_dict
        with open(output_path, "w") as configfile:
            param.write(configfile)

    def update_from_dictionary(self, **kwargs):
        """Update parameters from a dictionary"""
        ann = getattr(self, "__annotations__", {})
        for name, dtype in ann.items():
            if name in kwargs:
                try:
                    kwargs[name] = dtype(kwargs[name])
                except TypeError:
                    pass
                setattr(self, name, kwargs[name])

    def update_from_file(self, path: str) -> None:
        """Update parameters from an ini file"""
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(path)
        mega_list = config.items('parameters')
        dictionary = dict(mega_list)
        if dictionary['alphas_delta_sigma_cut']: dictionary['alphas_delta_sigma_cut'] = json.loads(dictionary['alphas_delta_sigma_cut'])
        if dictionary['interferometer_list']: dictionary['interferometer_list'] = json.loads(dictionary['interferometer_list'])
        dictionary['local_data_path_dict'] = dict(config.items("local_data"))
        for item in dictionary.copy():
            if not dictionary[item]:
                dictionary.pop(item)
        self.update_from_dictionary(**dictionary)

    def update_from_arguments(self, args: List[str]) -> None:
        """Update parameters from a set of arguments"""
        if not args:
            return
        ann = getattr(self, "__annotations__", {})
        parser = argparse.ArgumentParser()
        for name, dtype in ann.items():
            parser.add_argument(f"--{name}", type=dtype, required=False)
        parser.add_argument("--H1", type=str, required=False)
        parser.add_argument("--L1", type=str, required=False)
        parser.add_argument("--V", type=str, required=False)
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
        self.update_from_dictionary(**dictionary)

