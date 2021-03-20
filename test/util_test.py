import unittest

from stochastic_lite.util import calc_rho1


class WindowTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_rho1(self):
        self.assertEqual(calc_rho1(0), 0.027775555605193483)
