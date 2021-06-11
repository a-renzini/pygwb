import numpy as np
import bilby
from .baseline import Baseline
from .simulator import Simulator


class Network(object):
    def __init__(
        self,
        interferometers,
        freqs,
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

        self.Nifo = len(interferometers)

        combo_tuples = []
        for j in range(1, len(ifo_list)):
            for k in range(j):
                combo_tuples.append((k, j))

        baselines = []
        for i, j in combo_tuples:
            base_name = f"{ifo_list[i]} - {ifo_list[j]}"
            baselines.append(
                Baseline(
                    base_name,
                    interferometers[i],
                    interferometers[j],
                    freqs,
                )
            )

        self.baselines = baselines

        self.noise_PSD_array = get_noise_PSD_array()

    @classmethod
    def from_interferometer_list(cls, ifo_list, freqs):
        """
        [PARAMETERS]
        ------------------------
        ifo_list: list of str
            list of interferometer names
        """
        interferometers = bilby.gw.detector.InterferometerList(ifo_list)

        combo_tuples = []
        for j in range(1, len(ifo_list)):
            for k in range(j):
                combo_tuples.append((k, j))

        baselines = []
        for i, j in combo_tuples:
            base_name = f"{ifo_list[i]} - {ifo_list[j]}"
            baselines.append(
                Baseline(
                    base_name,
                    interferometers[i],
                    interferometers[j],
                    freqs,
                )
            )

        return cls(baselines, freqs)

    def get_noise_PSD_array(self):
        """ """
        noisePSDs = []
        for ifo in self.interferometers:
            psd = ifo.power_spectral_density_array
            psd[np.isinf(psd)] = 1
            noisePSDs.append(psd)

        return np.array(noisePSDs)

    def inject_GWB(self, ini_file):
        """
        injection module for the Network object
        [PARAMETERS]
        ----------------------
        ini_file: path
            path to the injection .ini file containing injection parameters
        """
        simulation_GWB.from_ini_file(self.baselines, ini_file)
