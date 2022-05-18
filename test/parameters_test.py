import argparse
import unittest

import pytest

from pygwb.parameters import Parameters, ParametersHelp


class TestParameters(unittest.TestCase):
    def setUp(self):
        self.t0 = 100
        self.tf = 200
        self.params = Parameters()

    def tearDown(self):
        del self.t0
        del self.tf
        del self.params

    def test_update_from_arguments(self):
        arguments = ['--t0', f'{self.t0}', '--tf', f'{self.tf}']
        self.params.update_from_arguments(arguments)
        self.assertEqual(self.params.t0, self.t0)
        self.assertEqual(self.params.tf, self.tf)

    def test_help(self):
        ann = getattr(Parameters, "__annotations__", {})
        parser = argparse.ArgumentParser()
        for name, dtype in ann.items():
            name_help = ParametersHelp[name].help
            parser.add_argument(f"--{name}", help=name_help, type=dtype, required=False)
        parser.print_help()  # for help

    def test_save_paramfile(self):
        self.params.save_paramfile("new_file.ini")
        self.params.update_from_file("new_file.ini")

if __name__ == "__main__":
    unittest.main()
