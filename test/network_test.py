import copy
import os
import unittest

import bilby
import numpy as np

from pygwb import network


class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.interferometer_1 = bilby.gw.detector.get_empty_interferometer("H1")
        self.interferometer_2 = bilby.gw.detector.get_empty_interferometer("H1")
        self.interferometer_2.name = "H2"
        self.interferometer_3 = bilby.gw.detector.get_empty_interferometer("L1")
        self.duration = 4.0
        self.sampling_frequency = 2048.0
        self.frequencies = np.arange(100)

    def tearDown(self):
        del self.interferometer_1
        del self.interferometer_2
        del self.interferometer_3
        del self.duration
        del self.frequencies

    def test_network_initialisation(self):
        ifos = [self.interferometer_1, self.interferometer_2, self.interferometer_3]
        net = network.Network("test_net", ifos)

    def test_set_duration_from_ifo_1(self):
        self.interferometer_1.duration = self.duration
        ifos = [self.interferometer_1, self.interferometer_2, self.interferometer_3]
        net = network.Network("test_net", ifos)
        self.assertTrue(net.duration, self.duration)

    def test_set_duration_from_ifos(self):
        ifos = [self.interferometer_1, self.interferometer_2, self.interferometer_3]
        for ifo in ifos:
            ifo.duration = self.duration
        net = network.Network("test_net", ifos)
        self.assertTrue(net.duration, self.duration)

    def test_set_duration_from_network(self):
        ifos = [self.interferometer_1, self.interferometer_2, self.interferometer_3]
        net = network.Network("test_net", ifos, duration=self.duration)
        self.assertTrue(net.duration, self.duration)
        for ifo in net.interferometers:
            self.assertTrue(ifo.duration, self.duration)
