import unittest

import bilby.gw.detector
import bilby.gw.detector as bilbydet
import numpy as np

import pygwb.orfs as orfs
from pygwb.baseline import Baseline


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

    def test_calc_orf_from_beta_omegas(self):

        freqs = 11.0
        H1 = bilbydet.get_empty_interferometer("H1")
        L1 = bilbydet.get_empty_interferometer("L1")
        HL = Baseline("HL", H1, L1)
        HL.orf_polarization = 'tensor'
        HL.frequencies = freqs
        
        beta = 0.4757189334754699
        omega_det1 = 3.4296676473913137
        omega_det2 = -1.2848910022634024
        omega_plus = (omega_det1 + omega_det2) / 2.
        omega_minus = (omega_det1 - omega_det2) / 2.

        np.testing.assert_allclose(
            orfs.calc_orf_from_beta_omegas(freqs, beta, omega_det1, omega_det2, omega_minus, omega_plus, 'tensor'), HL.tensor_overlap_reduction_function, rtol=1e-4, atol=0
        )
        
if __name__ == "__main__":
    unittest.main()
