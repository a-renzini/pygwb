"""
The ``constants`` module contains several constants that are used in stochastic gravitational-wave 
background searches, such as the speed of light and the Hubble parameter. This module relies on 
``astropy`` (more information `here <https://www.astropy.org/>`_) to obtain the value of these constants. 

Examples
--------

To import one of the constants from this module inside another code, one would call the following
line:

>>> from pygwb.constants import H0
"""

from astropy.constants import c
from astropy.cosmology import Planck18 as cosmo

H0 = cosmo.H(0)
h0 = H0.value / 100.0
deprecated_H0 = 3.240779289444365023237687716352957261e-18  # LAL_H0FAC_SI = 100 km/s/Mpc in SI units
# H0=3.2407792903e-18 # stochastic.m value
speed_of_light = c.value  # m/s