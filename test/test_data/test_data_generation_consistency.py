import pickle
import unittest

from .create_naive_and_sliding_psds_pickle import create_psd_data


class TestRunningSamplers(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_data_matches_on_disk(self):
        file_directory = Path(__file__).parent.resolve()
        existing_data_file = file_directory / "naive_and_sliding_psds.pickle"
        with open(existing_data_file, "rb") as ff:
            existing_data = pickle.load(ff)
        new_data = create_psd_data()

        self.assertDictEqual(existing_data, new_data)
