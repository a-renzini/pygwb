import sys
import argparse
from dataclasses import dataclass

if sys.version_info >= (3, 0):
    import configparser
else:
    import ConfigParser as configparser


@dataclass
class Parameters:
    t0: int
    tf: int
    data_type: str
    channel: str
    new_sample_rate: int
    cutoff_frequency: int
    segment_duration: int
    number_cropped_seconds: int
    window_downsampling: str
    ftype: str
    window_fftgram: str
    frequency_resolution: float
    polarization: str
    alpha: float
    fref: int
    flow: int
    fhigh: int
    coarse_grain: int

    @classmethod
    def from_file(cls, param_file):
        param = configparser.ConfigParser()
        param.read(param_file)

        t0 = param.getfloat("parameters", "t0")
        # t0 = get_param('t0')
        #
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
        coarse_grain = param.getint("parameters", "coarse_grain")

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
            window_fftgram,
            frequency_resolution,
            polarization,
            alpha,
            fref,
            flow,
            fhigh,
            coarse_grain,
        )

    def __post_init__(self):
        self.overlap = self.segment_duration / 2
        if self.coarse_grain:
            self.fft_length = self.segment_duration
        else:
            self.fft_length = int(1 / self.frequency_resolution)


# if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-param_file', help="Parameter file", action="store", type=str)
#    parser.add_argument('-t0', help="Start time", action="store", type=str)
#    parser.add_argument('-tf', help="End time", action="store", type=str)
#
#    args = parser.parse_args()
#
#    Parameters(args.param_file)
