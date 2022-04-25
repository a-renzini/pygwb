import os
import unittest

import numpy as np

from pygwb import notch


class TestNotch(unittest.TestCase):
    def setUp(self):
        self.stoch_notch_list_1 = notch.StochNotchList([])
        self.stoch_notch_list_1.append(
            notch.StochNotch(10.0, 15.0, "This is a test notch")
        )
        self.stoch_notch_list_2 = notch.StochNotchList([])
        self.stoch_notch_list_2.append(
            notch.StochNotch(10.0, 15.0, "This is a test notch 1")
        )
        self.stoch_notch_list_2.append(
            notch.StochNotch(35.0, 36.0, "This is a test notch 2")
        )
        self.stoch_notch_list_2.append(
            notch.StochNotch(21.0, 23.0, "This is a test notch 3")
        )
        self.stoch_notch_list_3 = notch.StochNotchList([])
        self.stoch_notch_list_3.append(
            notch.StochNotch(12.42, 12.45, "Pulsar injection")
        )
        self.stoch_notch_list_3.append(
            notch.StochNotch(16.58, 17.61, "H1 calibration line")
        )
        self.stoch_notch_list_4 = notch.StochNotchList([])
        self.stoch_notch_list_4.append(
            notch.StochNotch(4.9375, 4.99, "This is a test notch")
        )
        self.stoch_notch_list_4.append(
            notch.StochNotch(5.4375, 5.475, "This is a test notch")
        )
        self.stoch_notch_list_4.append(
            notch.StochNotch(5.7875, 6.025, "This is a test notch")
        )
        self.stoch_notch_list_4.append(
            notch.StochNotch(6.3875, 6.5125, "This is a test notch")
        )
        self.stoch_notch_list_4.append(
            notch.StochNotch(6.875, 7, "This is a test notch")
        )
        self.stoch_notch_list_4.append(
            notch.StochNotch(7.51, 7.6, "This is a test notch")
        )
        self.stoch_notch_list_5 = notch.StochNotchList([])
        self.stoch_notch_list_5.append(
            notch.StochNotch(4.8, 4.875, "This is a test notch")
        )
        self.stoch_notch_list_5.append(
            notch.StochNotch(5.4375, 5.475, "This is a test notch")
        )
        self.stoch_notch_list_5.append(
            notch.StochNotch(5.7875, 6.025, "This is a test notch")
        )
        self.stoch_notch_list_5.append(
            notch.StochNotch(6.3875, 6.5125, "This is a test notch")
        )
        self.stoch_notch_list_5.append(
            notch.StochNotch(6.875, 7, "This is a test notch")
        )
        self.stoch_notch_list_5.append(
            notch.StochNotch(7.625, 7.7, "This is a test notch")
        )

    def tearDown(self):
        del self.stoch_notch_list_1

    def test_check_frequency(self):
        epsilon = 1e-4
        test_freqs = np.arange(10.0 + epsilon, 15.0 + epsilon, 0.1)
        print(test_freqs)
        test_results = []
        for freq in test_freqs:
            test_results.append(self.stoch_notch_list_1.check_frequency(freq))
        self.assertTrue(np.all(test_results))

    def test_get_notch_mask(self):
        epsilon = 1e-4
        freqs = np.arange(5.0, 7.5 + epsilon, 0.125)
        anwser_1 = [
            True,
            False,
            False,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
        ]
        anwser_1_b = [not elem for elem in anwser_1]
        anwser_2 = [
            False,
            False,
            False,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
        ]
        anwser_2_b = [not elem for elem in anwser_2]
        print(len(freqs))

        notched1 = self.stoch_notch_list_4.get_notch_mask(freqs)
        notched2 = self.stoch_notch_list_5.get_notch_mask(freqs)

        print(len(notched1))
        print(freqs)
        print(notched1)
        print(anwser_1_b)
        print(notched2)
        print(anwser_2_b)

        self.assertTrue(np.array_equal(notched1, anwser_1_b))
        self.assertTrue(np.array_equal(notched2, anwser_2_b))


    def test_save_to_and_load_from_txt(self):

        self.stoch_notch_list_1.save_to_txt("test/TestNotchList_1.dat")

        self.assertTrue(os.path.isfile("test/TestNotchList_1.dat"))
        print("Does the file exist?   ", os.path.isfile("test/TestNotchList_1.dat"))

        my_compare_notch_list = notch.StochNotchList([])
        my_compare_notch_list = my_compare_notch_list.load_from_file(
            "test/test_data/TestNotchList.dat"
        )

        print(len(my_compare_notch_list), len(self.stoch_notch_list_1))
        print(my_compare_notch_list, self.stoch_notch_list_1)

        if len(my_compare_notch_list) == len(self.stoch_notch_list_1):
            check_2 = True
        else:
            check_2 = False

        self.assertTrue(check_2)

        for i in range(len(my_compare_notch_list)):
            if (
                my_compare_notch_list[i].minimum_frequency
                == self.stoch_notch_list_1[i].minimum_frequency
            ):
                check_3 = True
            else:
                check_3 = False
            if (
                my_compare_notch_list[i].maximum_frequency
                == self.stoch_notch_list_1[i].maximum_frequency
            ):
                check_4 = True
            else:
                check_4 = False
            if (
                my_compare_notch_list[i].description
                == "  " + self.stoch_notch_list_1[i].description
            ):
                check_5 = True
            else:
                check_5 = False
            print(
                my_compare_notch_list[i].minimum_frequency,
                self.stoch_notch_list_1[i].minimum_frequency,
            )
            print(
                my_compare_notch_list[i].maximum_frequency,
                self.stoch_notch_list_1[i].maximum_frequency,
            )
            print(
                my_compare_notch_list[i].description,
                self.stoch_notch_list_1[i].description,
            )

        self.assertTrue(check_3)
        self.assertTrue(check_4)
        self.assertTrue(check_5)

    def test_sort_list(self):

        self.stoch_notch_list_2.sort_list()
        for i in range(len(self.stoch_notch_list_2) - 1):
            if (
                self.stoch_notch_list_2[i].minimum_frequency
                <= self.stoch_notch_list_2[i + 1].minimum_frequency
            ):
                check = True
            else:
                check = False
                break
        self.assertTrue(check)

    def test_power_lines(self):

        fmin_comp = [59.9, 119.9, 179.9]
        fmax_comp = [60.1, 120.1, 180.1]
        my_compare_notch_list = notch.power_lines(fundamental=60, nharmonics=3, df=0.2)

        for i, my_notch in enumerate(my_compare_notch_list):
            if (
                my_notch.minimum_frequency == fmin_comp[i]
                and my_notch.maximum_frequency == fmax_comp[i]
            ):
                check = True
            else:
                check = False
                break

        self.assertTrue(check)

    def test_comb(self):

        fmin_comp = [59.9, 119.9, 179.9]
        fmax_comp = [60.1, 120.1, 180.1]

        my_compare_notch_list = notch.comb(
            f0=60, f_spacing=60, n_harmonics=3, df=0.2, description=None
        )

        for i, my_notch in enumerate(my_compare_notch_list):
            if (
                my_notch.minimum_frequency == fmin_comp[i]
                and my_notch.maximum_frequency == fmax_comp[i]
            ):
                check = True
            else:
                check = False
                break

        self.assertTrue(check)

    def test_pulsar_injections(self):

        my_compare_notch_list = notch.pulsar_injections(
            filename="test/test_data/pulsar.dat",
            t_start=1238112018,
            t_end=1269363618,
            doppler=1e-4,
        )

        fmin_comp = [12.425369523451533, 26.32748345804803]
        fmax_comp = [12.428167393140068, 26.335406132980573]

        for i, my_notch in enumerate(my_compare_notch_list):
            print(my_notch.minimum_frequency, my_notch.maximum_frequency)
            if (
                my_notch.minimum_frequency == fmin_comp[i]
                and my_notch.maximum_frequency == fmax_comp[i]
            ):
                check = True
            else:
                check = False

        self.assertTrue(check)
