import os

import bilby
import gwpy
import numpy as np

from .baseline import Baseline
from .simulator import Simulator


class Network(object):
    def __init__(
        self,
        name,
        interferometers,
        duration=None,
        frequencies=None,
        calibration_epsilon=0,
        notch_list=None,
        overlap_factor=0.5,
        zeropad_csd=True,
        window_fftgram="hann",
        overlap_factor_welch_psd=0,
        N_average_segments_welch_psd=2,
    ):
        """
        pygwb Network object with multiple functionalities
        * data simulation
        * stochastic pre-processing
        * isotropic stochastic analysis

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
        notch_list: str, optional
            filename of the baseline notch list -- currently only supports a single notch list for all baselines
        overlap_factor: float, optional
            factor by which to overlap the segments in the psd and csd estimation. Default is 1/2, if set to 0 no overlap is performed.
        zeropad_csd: bool, optional
            if True, applies zeropadding in the csd estimation. True by default.
        window_fftgram: str, optional
            what type of window to use to produce the fftgrams
        overlap_factor_welch_psd: float, optional
            Amount of overlap between data blocks used in pwelch method (range between 0 and 1)
            (default 0, no overlap)
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
            base_name = f"{self.interferometers[i]} - {self.interferometers[j]}"
            baselines.append(
                Baseline(
                    base_name,
                    self.interferometers[i],
                    self.interferometers[j],
                    duration=duration,
                    frequencies=frequencies,
                    calibration_epsilon=calibration_epsilon,
                    notch_list=notch_list,
                    overlap_factor=overlap_factor,
                    zeropad_csd=zeropad_csd,
                    window_fftgram=window_fftgram,
                    overlap_factor_welch_psd=overlap_factor_welch_psd,
                    N_average_segments_welch_psd=N_average_segments_welch_psd,
                )
            )

        self.baselines = baselines

    def set_duration(self, duration):
        """Sets the duration for the Network and Interferometers

        Note: the cross-checks that durations match in all the interferometers are done by each Baseline.

        Parameters
        ==========
        duration: float, optional
            The duration to set for the Network and interferometers
        """
        ifo_durations = []
        for ifo in self.interferometers:
            ifo_durations.append(ifo.duration)
        ifo_durations = np.array(ifo_durations)

        check_dur = np.all(ifo_durations == ifo_durations[0])

        if not check_dur:
            warnings.warn(
                "The interferometer durations don't match! The Network may not be able to handle this."
            )
        if duration is not None:
            self.duration = duration
            for ifo in self.interferometers:
                self.ifo.duration = duration
        elif check_dur:
            self.duration = self.interferometers[0].duration
        elif ifo_durations.any() is not None:
            for dur in ifo_durations:
                if dur is not None:
                    self.duration = duration
                    for ifo in self.interferometers:
                        self.ifo.duration = duration
        else:
            warnings.warn(
                "The Network duration is not set, and the interferometer durations don't match."
            )
            self.duration = duration

    def set_interferometer_data_from_simulator(
        self, GWB_intensity, N_segments, sampling_frequency, inject_into_data_flag=False
    ):
        """
        Fill interferometers with data from simulation
        """
        data_simulator = Simulator(
            self.interferometers,
            GWB_intensity,
            N_segments,
            duration=self.duration,
            sampling_frequency=sampling_frequency,
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

#    def set_interferometer_data_from_file(self, file):
#        """
#        Fill interferometers with data from file
#        """


#     These should be in the baseline class

#     def set_baseline_data_from_CSD(self):
#         """"""

#     def set_baseline_post_processing(self):
#         """"""
