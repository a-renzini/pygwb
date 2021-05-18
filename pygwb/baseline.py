import numpy as np
from .orfs import calc_orf


class Baseline(object):
    def __init__(
        self,
        name,
        interferometer_1,
        interferometer_2,
        frequencies=None,
        calibration_epsilon=0,
    ):
        """
        Parameters
        ----------
        name: str
            Name for the baseline, e.g H1H2
        interferometer_1/2: bilby Interferometer object
            the two detectors spanning the baseline
        calibration_epsilon: float
            calibration uncertainty for this baseline
        """
        self.name = name
        self.interferometer_1 = interferometer_1
        self.interferometer_2 = interferometer_2
        self.calibration_epsilon = calibration_epsilon
        self._tensor_orf_calculated = False
        self._vector_orf_calculated = False
        self._scalar_orf_calculated = False
        self.set_frequencies(frequencies)

    @property
    def overlap_reduction_function(self):
        if not self._tensor_orf_calculated:
            self._tensor_orf = self.calc_baseline_orf("tensor")
            self._tensor_orf_calculated = True
        return self._tensor_orf

    @property
    def vector_overlap_reduction_function(self):
        if not self._vector_orf_calculated:
            self._vector_orf = self.calc_baseline_orf("vector")
            self._vector_orf_calculated = True
        return self._vector_orf

    @property
    def scalar_overlap_reduction_function(self):
        if not self._scalar_orf_calculated:
            self._scalar_orf = self.calc_baseline_orf("scalar")
            self._scalar_orf_calculated = True
        return self._scalar_orf

    def set_frequencies(self, frequencies):
        frequencies_ifo_1 = (
            self.interferometer_1.duration and self.interferometer_1.sampling_frequency
        )
        frequencies_ifo_2 = (
            self.interferometer_2.duration and self.interferometer_2.sampling_frequency
        )
        if frequencies is not None:
            self.check_frequencies_match_baseline_ifos(frequencies)
            self.frequencies = frequencies
            if not frequencies_ifo_1:
                self.interferometer_1.frequency_array = frequencies
            if not frequencies_ifo_2:
                self.interferometer_2.frequency_array = frequencies
        elif frequencies_ifo_1 and frequencies_ifo_2:
            self.check_ifo_frequencies_match()
            self.frequencies = self.interferometer_1.frequency_array
        elif frequencies_ifo_1:
            self.frequencies = self.interferometer_1.frequency_array
            self.interferometer_2.duration = self.interferometer_1.duration
            self.interferometer_2.sampling_frequency = (
                self.interferometer_1.sampling_frequency
            )
        elif frequencies_ifo_2:
            self.frequencies = self.interferometer_2.frequency_array
            self.interferometer_1.duration = self.interferometer_2.duration
            self.interferometer_1.sampling_frequency = (
                self.interferometer_2.sampling_frequency
            )
        else:
            raise AttributeError(
                "Need either interferometer frequencies or frequencies passed to __init__!"
            )

    def check_frequencies_match_baseline_ifos(self, frequencies):
        frequencies_ifo_1 = (
            self.interferometer_1.duration and self.interferometer_1.sampling_frequency
        )
        frequencies_ifo_2 = (
            self.interferometer_2.duration and self.interferometer_2.sampling_frequency
        )
        if frequencies_ifo_1 and frequencies_ifo_2:
            self.check_ifo_frequencies_match()
            if not np.array_equal(frequencies, self.interferometer_2.frequency_array):
                raise AssertionError(
                    "Interferometer frequencies do not match given Baseline frequencies!"
                )
        elif frequencies_ifo_1:
            if not np.array_equal(frequencies, self.interferometer_1.frequency_array):
                raise AssertionError(
                    "Interferometer_1 frequencies do not match given Baseline frequencies!"
                )
        elif frequencies_ifo_2:
            if not np.array_equal(frequencies, self.interferometer_2.frequency_array):
                raise AssertionError(
                    "Interferometer_2 frequencies do not match given Baseline frequencies!"
                )

    def check_ifo_frequencies_match(self):
        if not np.array_equal(
            self.interferometer_1.frequency_array, self.interferometer_2.frequency_array
        ):
            raise AssertionError("Interferometer frequencies do not match each other!")

    def calc_baseline_orf(self, polarization):
        return calc_orf(
            self.frequencies,
            self.interferometer_1.vertex,
            self.interferometer_2.vertex,
            self.interferometer_1.x,
            self.interferometer_2.x,
            self.interferometer_1.y,
            self.interferometer_2.y,
            polarization,
        )
