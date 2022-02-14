import os
import unittest

import numpy as np

from pygwb import notch


class TestNotch(unittest.TestCase):
    def setUp(self):
        self.stoch_notch_list_1 = notch.StochNotchList([])
        self.stoch_notch_list_1.append(notch.StochNotch(10.,15.,'This is a test notch'))
        self.stoch_notch_list_2 = notch.StochNotchList([])
        self.stoch_notch_list_2.append(notch.StochNotch(10.,15.,'This is a test notch 1'))
        self.stoch_notch_list_2.append(notch.StochNotch(35.,36.,'This is a test notch 2'))
        self.stoch_notch_list_2.append(notch.StochNotch(21.,23.,'This is a test notch 3'))
        self.stoch_notch_list_3 = notch.StochNotchList([])
        self.stoch_notch_list_3.append(notch.StochNotch(12.42,12.45,'Pulsar injection'))
        self.stoch_notch_list_3.append(notch.StochNotch(16.58,17.61,'H1 calibration line'))




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
        
        self.stoch_notch_list_1.save_to_txt("test/TestNotchList_1.dat")
        
        self.assertTrue(os.path.isfile("test/TestNotchList_1.dat"))
        print("Does the file exist?   ",os.path.isfile("test/TestNotchList_1.dat"))

        my_compare_notch_list = notch.StochNotchList([])
        my_compare_notch_list = my_compare_notch_list.load_from_file("test/test_data/TestNotchList.dat")

        print(len(my_compare_notch_list),len(self.stoch_notch_list_1))
        print(my_compare_notch_list,self.stoch_notch_list_1)


        if len(my_compare_notch_list) == len(self.stoch_notch_list_1):
            check_2 = True
        else:
            check_2 = False

        self.assertTrue(check_2)

 
        for i in range(len(my_compare_notch_list)):
            if my_compare_notch_list[i].minimum_frequency == self.stoch_notch_list_1[i].minimum_frequency:
                check_3 = True
            else:
                check_3 = False
            if my_compare_notch_list[i].maximum_frequency == self.stoch_notch_list_1[i].maximum_frequency:
                check_4 = True
            else:
                check_4 = False
            if my_compare_notch_list[i].description == '  ' + self.stoch_notch_list_1[i].description:
                check_5 = True
            else:
                check_5 = False
            print(my_compare_notch_list[i].minimum_frequency , self.stoch_notch_list_1[i].minimum_frequency)
            print(my_compare_notch_list[i].maximum_frequency , self.stoch_notch_list_1[i].maximum_frequency)
            print(my_compare_notch_list[i].description , self.stoch_notch_list_1[i].description)

        self.assertTrue(check_3)
        self.assertTrue(check_4)
        self.assertTrue(check_5)

      
        


    def test_sort_list(self):
    
        self.stoch_notch_list_2.sort_list()
        for i in range(len(self.stoch_notch_list_2)-1):
            if self.stoch_notch_list_2[i].minimum_frequency <= self.stoch_notch_list_2[i+1].minimum_frequency:
                check = True
            else:
                check = False
                break
        self.assertTrue(check)

   
    def test_load_from_file_pre_pyGWB(self):


        my_compare_notch_list = notch.StochNotchList([])
        my_compare_notch_list = my_compare_notch_list.load_from_file_pre_pyGWB("test/test_data/TestNotchList_pre-pyGWB.dat")

    

        print(len(my_compare_notch_list),len(self.stoch_notch_list_3))
        print(my_compare_notch_list,self.stoch_notch_list_3)


        if len(my_compare_notch_list) == len(self.stoch_notch_list_3):
            check_1 = True
        else:
            check_1 = False

        self.assertTrue(check_1)



        for i in range(len(my_compare_notch_list)):
            if my_compare_notch_list[i].minimum_frequency == self.stoch_notch_list_3[i].minimum_frequency:
                check_2 = True
            else:
                check_2 = False
                break
            if my_compare_notch_list[i].maximum_frequency == self.stoch_notch_list_3[i].maximum_frequency:
                check_3 = True
            else:
                check_3 = False
                break
            if my_compare_notch_list[i].description == + self.stoch_notch_list_3[i].description:
                check_4 = True
            else:
                check_4 = False
                break
            print(my_compare_notch_list[i].minimum_frequency , self.stoch_notch_list_3[i].minimum_frequency)
            print(my_compare_notch_list[i].maximum_frequency , self.stoch_notch_list_3[i].maximum_frequency)
            print(my_compare_notch_list[i].description , self.stoch_notch_list_3[i].description)


        print(check_2,check_3,check_4)
        self.assertTrue(check_2)
        self.assertTrue(check_3)
        self.assertTrue(check_4)



    
    def test_power_lines(self):

        fmin_comp = [59.9,119.9,179.9]
        fmax_comp = [60.1,120.1,180.1]        
        my_compare_notch_list = notch.power_lines(fundamental=60, nharmonics=3, df=0.2)

        for i,notch in enumerate(my_compare_notch_list):
            if notch.fmin == fmin_comp[i] and notch.fmax == fmax_comp[i]:
                check = True
            else:
                check = False
                break

        self.assertTrue(check)


    """
    def test_comb(self):

    def test_pulsar_injections(self):
    """




