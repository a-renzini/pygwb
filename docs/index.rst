pygwb documentation
==============================================

.. automodule:: pygwb
    :members:

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   about
   installation
   contributing
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
   Understand_dsc
   make_notchlist
   run_statistical_checks
   run_pe
