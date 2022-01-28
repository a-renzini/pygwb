import unittest

import numpy as np

from pygwb import notch


class TestNotch(unittest.TestCase):
    def setUp(self):
        self.stoch_notch_list_1 = notch.StochNotchList([])
        self.stoch_notch_list_1.append(notch.StochNotch(10.,15.,'This is a test notch'))

    def tearDown(self):
        del self.StochNotchList1

    def test_check_frequency(self):
        test_freqs = np.arange(10.,15.,0.1)
        test_results = []
        for i in test_freqs:
            test_results.append(self.stoch_notch_list_1.check_frequency(freq))
        self.assertTrue(np.all(test_results))

    def test_get_idxs(self):
        freqs = np.np.arange(5.,1000.,0.1)
        test_idxs = np.ones(len(freqs))
        test_inv_idxs = np.ones(len(freqs))
        for f in freqs:
            if f >= 10. and f<= 15.:
                test_idxs = True
                test_inv_idxs = False
            else:
                test_idxs = False
                test_inv_idxs = True
        idxs,inv_idxs = self.stoch_notch_list_1.get_idxs(freqs)
        self.assertTrue(np.array_equal(idxs,test_idxs))
        self.assertTrue(np.array_equal(inv_idxs,test_inv_idxs))       

    def test_save_to_txt(self):

    def test_sort_list(self):

    def test_load_from_file(self):

    def test_load_from_file_pre_pyGWB(self):

    def test_power_lines(self):

    def test_comb(self):

    def test_pulsar_injections(self):





