import os
import unittest

import numpy as np

from pygwb import notch


class TestNotch(unittest.TestCase):
    def setUp(self):
        self.stoch_notch_list_1 = notch.StochNotchList([])
        self.stoch_notch_list_1.append(notch.StochNotch(10.,15.,'This is a test notch'))

    def tearDown(self):
        del self.stoch_notch_list_1

    def test_check_frequency(self):
        epsilon = 1e-4
        test_freqs = np.arange(10.+epsilon,15.+epsilon,0.1)
        print(test_freqs)
        test_results = []
        for freq in test_freqs:
            test_results.append(self.stoch_notch_list_1.check_frequency(freq))
        self.assertTrue(np.all(test_results))

    def test_get_idxs(self):
        freqs = np.arange(5.,1000.,0.1)
        test_idxs = np.ones(len(freqs))
        test_inv_idxs = np.ones(len(freqs))
        for i,f in enumerate(freqs):
            if f >= 10. and f<= 15.:
                test_idxs[i] = True
                test_inv_idxs[i] = False
            else:
                test_idxs[i] = False
                test_inv_idxs[i] = True
        idxs,inv_idxs = self.stoch_notch_list_1.get_idxs(freqs)

        self.assertTrue(np.array_equal(idxs,test_idxs))
        self.assertTrue(np.array_equal(inv_idxs,test_inv_idxs))       
    
    def test_save_to_and_load_from_txt(self):
        
        self.stoch_notch_list_1.save_to_txt("TestNotchList.dat")
        
        self.assertTrue(os.path.isfile("TestNotchList.dat"))

        my_compare_notch_list = notch.StochNotchList([])
        my_compare_notch_list.load_from_file("TestNotchList.dat")

        if len(my_compare_notch_list) == len(self.stoch_notch_list_1):
            check_2 = True
        else:
            check_2 = False

        self.assertTrue(check_2)

 
        for i in range(len(my_compare_notch_list)):
            if my_compare_notch_list[i].minimum_frequency == self.stoch_notch_list_1.minimum_frequency:
                check_3 = True
            else:
                check_3 = False
            if my_compare_notch_list[i].maximum_frequency == self.stoch_notch_list_1.maximum_frequency:
                check_4 = True
            else:
                check_4 = False
            if my_compare_notch_list[i].description == self.stoch_notch_list_1.description:
                check_5 = True
            else:
                check_5 = False

        self.assertTrue(check_3)
        self.assertTrue(check_4)
        self.assertTrue(check_5)

      
        


    #def test_sort_list(self):
    """
    def test_load_from_file(self):

    def test_load_from_file_pre_pyGWB(self):

    def test_power_lines(self):

    def test_comb(self):

    def test_pulsar_injections(self):
    """




