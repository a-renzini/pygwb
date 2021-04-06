import unittest

import numpy as np

from stochastic_lite.spectral import coarse_grain, coarse_grain_exact, coarse_grain_frequencies


class TestCoarseGrain(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_coarse_grain_matches_coarse_grain_exact_even(self):
        self.assertTrue(np.array_equal(
            coarse_grain(np.arange(17, dtype=float), 4),
            coarse_grain_exact(np.arange(17, dtype=float), 4)
        ))

    def test_coarse_grain_matches_coarse_grain_exact_odd(self):
        self.assertTrue(np.array_equal(
            coarse_grain(np.arange(25, dtype=float), 3),
            coarse_grain_exact(np.arange(25, dtype=float), 3)
        ))

    def test_coarse_grain_matches_coarse_grain_exact_float(self):
        self.assertTrue(np.array_equal(
            coarse_grain(np.arange(25, dtype=float), 3.3),
            coarse_grain_exact(np.arange(25, dtype=float), 3.3)
        ))

    def test_coarse_grain_frequencies_matches_expected(self):
        self.assertTrue(np.array_equal(
            np.linspace(0, 1, 33)[2:-2:2],
            coarse_grain_frequencies(np.linspace(0, 1, 33), 32, 2)
        ))
