import unittest

import bilby.gw.detector
import numpy as np

import pygwb.orfs as orfs


class OverlapReductionFunctionTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_TVSplus(self):
        for polarization in ["tensor", "vector", "scalar"]:
            self.assertAlmostEqual(
                getattr(orfs, f"{polarization[0].upper()}plus")(alpha=0, beta=0),
                0,
                places=5,
            )
            self.assertAlmostEqual(
                getattr(orfs, f"{polarization[0].upper()}plus")(
                    alpha=0, beta=np.pi / 2
                ),
                -1.0 / 4,
                places=5,
            )
            self.assertAlmostEqual(
                getattr(orfs, f"{polarization[0].upper()}plus")(alpha=0, beta=np.pi),
                -1.0,
                places=5,
            )

    def test_TVSminus(self):
        for polarization in ["tensor", "vector", "scalar"]:
            self.assertAlmostEqual(
                getattr(orfs, f"{polarization[0].upper()}minus")(alpha=0, beta=0),
                1.0,
                places=5,
            )
            self.assertAlmostEqual(
                getattr(orfs, f"{polarization[0].upper()}minus")(
                    alpha=0, beta=np.pi / 2
                ),
                1.0 / 4,
                places=5,
            )
            self.assertAlmostEqual(
                getattr(orfs, f"{polarization[0].upper()}minus")(alpha=0, beta=np.pi),
                0.0,
                places=5,
            )

    def test_calc_orfs(self):
        freqs = np.arange(2000)
        for polarization in ["tensor", "vector", "scalar"]:
            for baseline, orf_f0 in [
                ("HL", -0.89077),
                ("HV", -0.00990),
                ("LV", -0.24715),
            ]:
                print(baseline + " " + polarization)
                interferometer_1 = bilby.gw.detector.get_empty_interferometer(
                    baseline[0] + "1"
                )
                interferometer_2 = bilby.gw.detector.get_empty_interferometer(
                    baseline[1] + "1"
                )

                orf = orfs.calc_orf(
                    freqs,
                    interferometer_1.vertex,
                    interferometer_2.vertex,
                    interferometer_1.x,
                    interferometer_2.x,
                    interferometer_1.y,
                    interferometer_2.y,
                    polarization=polarization,
                )
                self.assertAlmostEqual(
                    orf[0],
                    orf_f0 / 3.0 if polarization == "scalar" else orf_f0,
                    places=2,
                )
                self.assertAlmostEqual(orf[-1], 0, places=2)
                self.assertTrue(np.all(np.abs(orf) < 1))
                if baseline == "HV":
                    self.assertTrue(np.argmax(abs(orf)) < 25)
                else:
                    self.assertEqual(np.argmax(abs(orf)), 0)
