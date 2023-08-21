pygwb documentation
==============================================
.. image:: pygwb_logo_plasma.png
   :width: 250
   :align: center

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
   api
   tutorials
   demos

.. toctree::
    :maxdepth: 1
    :hidden:
    
    pygwb paper <https://arxiv.org/pdf/2303.15696.pdf>
    GitHub <https://github.com/a-renzini/pygwb>
    Submit an issue <https://github.com/a-renzini/pygwb/issues/new>