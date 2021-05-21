import numpy as np
import pygwb.network

# freqs = read_from_file('freqs.txt')
A = pygwb.network.Network.from_interferometer_list(['H1', 'L1', 'V1'], np.linspace(0,1000)) 

A.inject_GWB('./pygwb/example.ini')   
