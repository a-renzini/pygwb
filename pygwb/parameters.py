import argparse
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
    t0: float
    tf: float
    data_type: str
    channel: str
    new_sample_rate: int
    cutoff_frequency: int
    segment_duration: int
    number_cropped_seconds: int
    window_downsampling: str
    ftype: str
    frequency_resolution: float
    polarization: str
    alpha: float
    fref: int
    flow: int
    fhigh: int
    coarse_grain: int
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

    @classmethod
    def from_file(cls, param_file):
        if not Path(param_file).exists():
            raise OSError("Your paramfile doesn't exist!")

        param = configparser.ConfigParser()
        param.optionxform = str
        param.read(str(param_file))

        t0 = param.getfloat("parameters", "t0")
        tf = param.getfloat("parameters", "tf")
        data_type = param.get("parameters", "data_type")
        channel = param.get("parameters", "channel")
        new_sample_rate = param.getint("parameters", "new_sample_rate")
        cutoff_frequency = param.getint("parameters", "cutoff_frequency")
        segment_duration = param.getint("parameters", "segment_duration")
        number_cropped_seconds = param.getint("parameters", "number_cropped_seconds")
        window_downsampling = param.get("parameters", "window_downsampling")
        ftype = param.get("parameters", "ftype")
        window_fftgram = param.get("parameters", "window_fftgram")
        frequency_resolution = param.getfloat("parameters", "frequency_resolution")
        polarization = param.get("parameters", "polarization")
        alpha = param.getfloat("parameters", "alpha")
        fref = param.getint("parameters", "fref")
        flow = param.getint("parameters", "flow")
        fhigh = param.getint("parameters", "fhigh")
        notch_list_path = param.get("parameters", "notch_list_path")
        N_average_segments_welch_psd = param.getfloat(
            "parameters", "N_average_segments_welch_psd"
        )
        coarse_grain = param.getint("parameters", "coarse_grain")
        interferometer_list = param.get("parameters", "interferometer_list")
        calibration_epsilon = param.getfloat("parameters", "calibration_epsilon")
        overlap_factor = param.getfloat("parameters", "overlap_factor")
        zeropad_csd = param.getboolean("parameters", "zeropad_csd")
        delta_sigma_cut = param.getfloat("parameters", "delta_sigma_cut")
        alphas_delta_sigma_cut = param.get("parameters", "alphas_delta_sigma_cut")
        save_data_type = param.get("parameters", "save_data_type")
        time_shift = param.getint("parameters", "time_shift")
        if param.has_option("parameters", "gate_data"):
            gate_data = param.getboolean("parameters", "gate_data")
        else: 
            gate_data = False
        if param.has_option("parameters", "gate_tzero"):
            gate_tzero = param.getfloat("parameters", "gate_tzero")
        else:
            gate_tzero = 1.0
        if param.has_option("parameters", "gate_tpad"):
            gate_tpad = param.getfloat("parameters", "gate_tpad")  
        else:
            gate_tpad = 0.5
        if param.has_option("parameters", "gate_threshold"):
            gate_threshold = param.getfloat("parameters", "gate_threshold") 
        else:
            gate_threshold = 50.0
        if param.has_option("parameters", "cluster_window"):
            cluster_window = param.getfloat("parameters", "cluster_window") 
        else:
            cluster_window = 0.50
        if param.has_option("parameters", "gate_whiten"):
            gate_whiten = param.getboolean("parameters", "gate_whiten")
        else:
            gate_whiten = True

        local_data_path_dict = dict(param.items("local_data"))
        return cls(
            t0,
            tf,
            data_type,
            channel,
            new_sample_rate,
            cutoff_frequency,
            segment_duration,
            number_cropped_seconds,
            window_downsampling,
            ftype,
            frequency_resolution,
            polarization,
            alpha,
            fref,
            flow,
            fhigh,
            coarse_grain,
            interferometer_list,
            local_data_path_dict,
            notch_list_path,
            N_average_segments_welch_psd,
            window_fftgram,
            calibration_epsilon,
            overlap_factor,
            zeropad_csd,
            delta_sigma_cut,
            alphas_delta_sigma_cut,
            save_data_type,
            time_shift,
            gate_data,
            gate_tzero,
            gate_tpad,
            gate_threshold,
            cluster_window,
            gate_whiten,
        )

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
