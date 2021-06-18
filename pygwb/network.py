import bilby
import numpy as np

from baseline import Baseline #.
from simulator import Simulator #.


class Network(object):
    def __init__(
        self,
        interferometers,
        duration=None,
        sampling_frequency=None,
        calibration_epsilon=0,
    ):
        """
        pygwb Network object with multiple functionalities
        * data simulation
        * stochastic pre-processing
        * isotropic stochastic analysis

        [PARAMETERS]
        ------------------------
        baselines: list of baseline objects
            baseline list
        freqs: array_like
            frequency array
        """
        self.interferometers = interferometers
        self.Nifo = len(interferometers)

        combo_tuples = []
        for j in range(1, len(interferometers)):
            for k in range(j):
                combo_tuples.append((k, j))

        baselines = []
        for i, j in combo_tuples:
            base_name = f"{interferometers[i]} - {interferometers[j]}"
            baselines.append(
                Baseline(
                    base_name,
                    interferometers[i],
                    interferometers[j],
                    duration,
                    sampling_frequency,
                    calibration_epsilon
                )
            )

        self.baselines = baselines

        self.noise_PSD_array = self.get_noise_PSD_array()

    @classmethod
    def from_interferometer_list(cls, ifo_list, duration=None, 
        sampling_frequency=None, calibration_epsilon=0):
        """
        [PARAMETERS]
        ------------------------
        ifo_list: list of str
            list of interferometer names
        """
        interferometers = bilby.gw.detector.InterferometerList(ifo_list)
        
        return cls(interferometers, duration, sampling_frequency, calibration_epsilon)
    
    #I commented out the part below, otherwise it was computing the baselines twice.
#         combo_tuples = []
#         for j in range(1, len(ifo_list)):
#             for k in range(j):
#                 combo_tuples.append((k, j))

#         baselines = []
#         for i, j in combo_tuples:
#             base_name = f"{ifo_list[i]} - {ifo_list[j]}"
#             baselines.append(
#                 Baseline(
#                     base_name,
#                     interferometers[i],
#                     interferometers[j],
#                     duration,
#                     sampling_frequency,
#                     calibration_epsilon
#                 )
#             )

#         return cls(baselines, freqs)

    def get_noise_PSD_array(self):
        """ """
        noisePSDs = []
        for ifo in self.interferometers:
            psd = ifo.power_spectral_density_array
            psd[np.isinf(psd)] = 1
            noisePSDs.append(psd)

        return np.array(noisePSDs)

    def inject_GWB(self):
        """
        injection module for the Network object
        [PARAMETERS]
        ----------------------
        
        """
        pass
#         simulator.from_ifo_list()
