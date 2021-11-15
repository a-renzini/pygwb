import sys
import argparse

if sys.version_info >= (3, 0):
    import configparser
else:
    import ConfigParser as configparser

class parameters(object):
    def __init__(self, param_file):
        param = configparser.ConfigParser()
        param.read(param_file)
        
        self.t0 = param.getint('parameters','t0')
        self.tf= param.getint('parameters','tf')
        self.data_type= param.get('parameters','data_type')
        self.channel_suffix= param.get('parameters','channel_suffix')
        self.new_sample_rate= param.getint('parameters','new_sample_rate')
        self.cutoff_frequency= param.getint('parameters','cutoff_frequency')
        self.segment_duration= param.getint('parameters','segment_duration')
        self.number_cropped_seconds= param.getint('parameters','number_cropped_seconds')
        self.window_downsampling= param.get('parameters','window_downsampling')
        self.ftype= param.get('parameters','ftype')
        self.window_fftgram= param.get('parameters','window_fftgram')
        self.overlap= self.segment_duration/2
        self.frequency_resolution=param.getfloat('parameters','frequency_resolution')
        self.polarization=param.get('parameters','polarization') 
        self.alpha= param.getfloat('parameters','alpha')
        self.fref= param.getint('parameters','fref')
        self.flow= param.getint('parameters','flow')
        self.fhigh= param.getint('parameters','fhigh')
        self.coarse_grain= param.getint('parameters','coarse_grain')

        if self.coarse_grain:
            self.fft_length = self.segment_duration
        else:
            self.fft_length = int(1/self.frequency_resolution)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-param_file', help="Parameter file", action="store", type=str)
    
    args = parser.parse_args()

    parameters(args.param_file)
