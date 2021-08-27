import bilby
import numpy as np

from .baseline import Baseline
from .simulator import Simulator


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
        interferometers: list of intereferometer objects
            baseline list
        duration: int
            segment duration
        sampling_frequency: float
            sampling frequency
        """
        self.interferometers = interferometers
        self.Nifo = len(interferometers)
        self.duration = duration
        self.sampling_frequency = sampling_frequency

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
                    calibration_epsilon,
                )
            )

        self.baselines = baselines

        self.noise_PSD_array = self.get_noise_PSD_array()

    @classmethod
    def from_interferometer_list(
        cls, ifo_list, duration=None, sampling_frequency=None, calibration_epsilon=0
    ):
        """
        [PARAMETERS]
        ------------------------
        ifo_list: list of str
            list of interferometer names
        duration: int
            segment duration
        sampling_frequency: float
            sampling frequency

        """
        interferometers = bilby.gw.detector.InterferometerList(ifo_list)

        for ifo in interferometers:
            ifo.duration = duration
            ifo.sampling_frequency = sampling_frequency
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                ifo.frequency_array,
                np.nan_to_num(ifo.power_spectral_density_array, posinf=1.0e-41),
            )

        return cls(interferometers, duration, sampling_frequency, calibration_epsilon)

    def get_noise_PSD_array(self):
        """
        Function that gets the noise PSD array of all the interferometers.
        
        Parameters
        ==========
        
        Returns
        =======
        noisePSD_array: array
            Array containing the noise PSD arrays for all interferometers in
            self.interferometers.
        
        """
        noisePSDs = []
        for ifo in self.interferometers:
            psd = ifo.power_spectral_density_array
            psd[np.isinf(psd)] = 1
            noisePSDs.append(psd)
        noisePSD_array = np.array(noisePSDs)
        return noisePSD_array

    def set_interferometer_data_from_simulator(self, GWB_intensity, N_segments):
        """
        Fill interferometers with data
        """
        data_simulator = Simulator(
            self.interferometers,
            GWB_intensity,
            N_segments,
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
        )
        data = data_simulator.get_data_for_interferometers()
        for ifo in self.interferometers:
            ifo.set_strain_data_from_gwpy_timeseries(data[ifo.name])
