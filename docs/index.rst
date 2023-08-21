pygwb documentation
==============================================
.. image:: pygwb_logo_plasma.png
   :width: 200

.. automodule::
    :members:

`pygwb`: A python-based, user-friendly library for gravitational-wave background (GWB) searches with ground-based interferometers.

`pygwb` provides a modular and flexible codebase to analyse laser interferometer data and design a GWB search pipeline. It is tailored to current ground-based interferometers: LIGO Hanford, LIGO Livingston, and Virgo, but can be generalized to other configurations. It is based on the existing packages `gwpy` and `bilby`, for optimal integration with widely-used GW data anylsis tools.

`pygwb` also includes a set of pre-packaged analysis scripts which may be used to analyse data and perform large-scale searches on a high-performance computing cluster efficiently.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   about
   installation
   contributing
   citing
   pipeline
   pe
   tutorials


.. currentmodule:: pygwb

API:
----

.. autosummary::
   :toctree: api
   :template: custom-module-template.rst
   :caption: API:
   :recursive:
   
   detector
   baseline
   network

   preprocessing
   spectral
   coherence
   postprocessing
   omega_spectra
   pe
   statistical_checks

   simulator

   delta_sigma_cut
   notch
   
   constants
   orfs   
   parameters
   util

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   generate_stochastic_background_in_network
   inject_simulated_data_in_network
   simulate_CBC_GWB
   test_orfs
   make_notchlist
   run_statistical_checks
   run_pe
