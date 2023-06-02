import unittest

import bilby.gw.detector as bilbydet
import numpy as np

from pygwb.baseline import Baseline
from pygwb.pe import *


class Test(unittest.TestCase):
    def setUp(self) -> None:
        H1 = bilbydet.get_empty_interferometer("H1")
        L1 = bilbydet.get_empty_interferometer("L1")
        self.HL = Baseline("H1L1", H1, L1)
        self.HL.orf_polarization = 'tensor'
        freqs = 8.0
        self.HL.frequencies = freqs
        return None

    def tearDown(self) -> None:
        del self.HL
        pass

    def test_power_law_model(self):
        kwargs = {"baselines": [self.HL], "model_name": "PL", "fref": 11}
        test_model = PowerLawModel(**kwargs)
        test_model.parameters = {"omega_ref": 8.9, "alpha": 3}

        # print(test_model.model_function(HL))
        # print(type(test_model.model_function(HL)))

        self.assertEqual(test_model.model_function(self.HL), 3.42359128474831)

    def test_broken_power_law_model(self):
        kwargs = {"baselines": [self.HL], "model_name": "BPL"}
        test_model = BrokenPowerLawModel(**kwargs)
        test_model.parameters = {
            "omega_ref": 8.9,
            "alpha_1": 3,
            "alpha_2": 5,
            "fbreak": 12.0,
        }

        # print(test_model.model_function(HL))
        # print(type(test_model.model_function(HL)))

        self.assertEqual(test_model.model_function(self.HL), 2.6370370370370364)

    def test_tripple_BPL(self):
        freqs = 15.0
        self.HL.frequencies = freqs

        kwargs = {"baselines": [self.HL], "model_name": "TBPL"}
        test_model = TripleBrokenPowerLawModel(**kwargs)
        test_model.parameters = {
            "omega_ref": 8.9,
            "alpha_1": 3,
            "alpha_2": 5,
            "alpha_3": 2,
            "fbreak1": 6.5,
            "fbreak2": 12.4,
        }

        # print(test_model.model_function(HL))
        # print(type(test_model.model_function(HL)))

        self.assertEqual(test_model.model_function(self.HL), 329.0567447272102)

    def test_smooth_BPL(self):

        freqs = 15.0
        self.HL.frequencies = freqs

        kwargs = {"baselines": [self.HL], "model_name": "T_smooth_PL"}
        test_model = SmoothBrokenPowerLawModel(**kwargs)
        test_model.parameters = {
            "omega_ref": 8.9,
            "fbreak": 12.0,
            "alpha_1": 3.0,
            "alpha_2": 5.0,
            "delta": 2.5,
        }

        # print("results ",test_model.model_function(HL))
        # print(type(test_model.model_function(HL)))

        self.assertEqual(test_model.model_function(self.HL), 39.0119698805036)

    ### Test model involve piecewise

    def test_broken_power_law_model_2(self):

        freqs = np.array([8.0, 15.0])
        self.HL.frequencies = freqs

        kwargs = {"baselines": [self.HL], "model_name": "BPL_piece_wise"}
        test_model = BrokenPowerLawModel(**kwargs)
        test_model.parameters = {
            "omega_ref": 8.9,
            "alpha_1": 3,
            "alpha_2": 5,
            "fbreak": 12.0,
        }

        # print("results: ",test_model.model_function(HL))
        # print(type(test_model.model_function(HL)))

        np.testing.assert_allclose(
            test_model.model_function(self.HL),
            np.array([2.63703704, 27.16064453]),
            rtol=1e-5,
            atol=0,
        )

    def test_tripple_BPL_2(self):

        freqs = np.array([3.5, 8.7, 15.0])
        self.HL.frequencies = freqs

        kwargs = {"baselines": [self.HL], "model_name": "TBPL_piece_wise"}
        test_model = TripleBrokenPowerLawModel(**kwargs)
        test_model.parameters = {
            "omega_ref": 8.9,
            "alpha_1": 3,
            "alpha_2": 5,
            "alpha_3": 2,
            "fbreak1": 6.5,
            "fbreak2": 12.4,
        }

        # print(test_model.model_function(HL))
        # print(type(test_model.model_function(HL)))

        np.testing.assert_allclose(
            test_model.model_function(self.HL),
            np.array([1.389485662, 38.23133703, 329.0567447]),
            rtol=1e-5,
            atol=0,
        )

    def test_PV_PL1(self):

        freqs = 15.0
        self.HL.frequencies = freqs

        kwargs = {"baselines": [self.HL], "model_name": "PV_PL1", "fref": 25}
        test_model = PVPowerLawModel(**kwargs)
        test_model.parameters = {"omega_ref": 8.9, "alpha": 4.6, "Pi": 2.8}

        # print(test_model.model_function(HL))
        # print("gamma_V ", HL.gamma_v)
        # print("Gamma_I ", HL.tensor_overlap_reduction_function)
        # print(type(test_model.model_function(HL)))

        np.testing.assert_allclose(
            test_model.model_function(self.HL), 0.8634145976022678, rtol=1e-5, atol=0
        )

    def test_PV_PL2(self):

        freqs = 16.0
        self.HL.frequencies = freqs

        kwargs = {"baselines": [self.HL], "model_name": "PV_PL2", "fref": 25.0}
        test_model = PVPowerLawModel2(**kwargs)
        test_model.parameters = {"omega_ref": 8.9, "alpha": 4.6, "beta": 3.7}

        # print(test_model.model_function(HL))
        # print("gamma_V ", HL.gamma_v)
        # print("Gamma_I ", HL.tensor_overlap_reduction_function)
        # print(type(test_model.model_function(HL)))

        np.testing.assert_allclose(
            test_model.model_function(self.HL), 200.6063814276104, rtol=1e-5, atol=0
        )

    def test_TVS(self):

        freqs = np.array([8.0])
        self.HL.frequencies = freqs

        kwargs = {
            "baselines": [self.HL],
            "model_name": "TVS_PL",
            "fref": 24.7,
            "polarizations": ["tensor", "scalar", "vector"],
        }
        test_model = TVSPowerLawModel(**kwargs)
        test_model.parameters = {
            "omega_ref_tensor": 8.9,
            "omega_ref_vector": 13.7,
            "omega_ref_scalar": 25.4,
            "alpha_tensor": 3.0,
            "alpha_vector": 5.0,
            "alpha_scalar": 4.2,
        }

        # print('tensor_overlap_reduction_function: ',HL.tensor_overlap_reduction_function)
        # print('vector_overlap_reduction_function: ',HL.vector_overlap_reduction_function)
        # print('scalar_overlap_reduction_function: ',HL.scalar_overlap_reduction_function)
        # print("tensor_vector_scalar ",test_model.model_function(HL))
        # print(type(test_model.model_function(HL)))

        np.testing.assert_allclose(
            test_model.model_function(self.HL),
            np.array([0.4226662681053238]),
            rtol=1e-5,
            atol=0,
        )

    def test_schumann_model(self):

        freqs = 11.0
        M_psd = 1 / 3 + 5j
        H1 = bilbydet.get_empty_interferometer("H1")
        L1 = bilbydet.get_empty_interferometer("L1")
        HL = Baseline("HL", H1, L1)
        HL.orf_polarization = 'tensor'
        HL.frequencies = freqs
        HL.M_f = M_psd

        kwargs = {"baselines": [HL], "model_name": "Schu"}
        test_model = SchumannModel(**kwargs)
        test_model.parameters = {
            "kappa_H": 0.7,
            "beta_H": 1.6,
            "kappa_L": 0.9,
            "beta_L": 2.3,
        }

        # print(test_model.model_function(HL))
        # print(type(test_model.model_function(HL)))

        np.testing.assert_allclose(
            test_model.model_function(HL), 1.448064220e-47, rtol=1e-5, atol=0
        )
        
    def test_geodesy(self):

        freqs = 11.0
        H1 = bilbydet.get_empty_interferometer("H1")
        L1 = bilbydet.get_empty_interferometer("L1")
        HL = Baseline("HL", H1, L1)
        HL.orf_polarization = 'tensor'
        HL.frequencies = freqs

        kwargs = {"baselines": [HL], "model_name": "Geodesy", "fref":11}
        test_model = Geodesy(**kwargs)
        test_model.parameters = {
            "omega_ref": 8.9,
            "alpha": 3,
            "beta": 0.4757189334754699,
            "omega_det1": 3.4296676473913137,
            "omega_det2": -1.2848910022634024,
        }
        
        kwargs_comparison = {"baselines": [self.HL], "model_name": "PL", "fref": 11}
        test_comparison = PowerLawModel(**kwargs)
        test_comparison.parameters = {"omega_ref": 8.9, "alpha": 3}
        
        np.testing.assert_allclose(
            test_model.model_function(HL), test_comparison.model_function(HL), rtol=1e-4, atol=0
        )
        
    def test_calc_orf_new(self):

        freqs = 11.0
        H1 = bilbydet.get_empty_interferometer("H1")
        L1 = bilbydet.get_empty_interferometer("L1")
        HL = Baseline("HL", H1, L1)
        HL.orf_polarization = 'tensor'
        HL.frequencies = freqs
        
        beta = 0.4757189334754699
        omega_det1 = 3.4296676473913137
        omega_det2 = -1.2848910022634024

        np.testing.assert_allclose(
            calc_orf_new(freqs, beta, omega_det1, omega_det2), HL.tensor_overlap_reduction_function, rtol=1e-4, atol=0
        )


if __name__ == "__main__":
    unittest.main()
