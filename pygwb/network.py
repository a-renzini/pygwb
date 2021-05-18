import numpy as np
import bilby
from .pe import Baseline


class Network(object):
    def __init__(
        self,
        baselines,
        freqs,
    ):
        """
        [PARAMETERS]
        ------------------------
        baselines: list of baseline objects
            baseline list
        freqs: array_like
            frequency array
        """
        for base in baselines:
            print(base.name)


    @classmethod
    def from_interferometers(cls, ifo_list, freqs):
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
                    np.array([1,2]),
                    np.array([1,2]),
                    freqs,
                )
            )

        return cls(baselines, freqs)
