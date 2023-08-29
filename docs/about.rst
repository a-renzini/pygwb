==============================================
About ``pygwb``
==============================================

A gravitational-wave background (GWB) is expected from the superposition of all gravitational waves (GWs) too faint to be detected individually, or by the incoherent overlap of a large number of signals in the same band `(Renzini+ 2022) <https://www.mdpi.com/2075-4434/10/1/34>`_. A GWB is primarily characterized by its spectral emission, usually parameterized by the GW fractional energy density spectrum, which is the target for stochastic GW searches `(Allen & Romano 1999) <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.59.102001>`_,

.. math:: 

   \Omega_{\rm GW}(f) = \frac{1}{\rho_c}\frac{d\rho_{\rm GW}(f)}{d\ln f},


where :math:`d\rho_{\rm GW}` is the energy density of GWs in the frequency band :math:`f` to :math:`df` , and :math:`\rho_c` is the critical energy density of the Universe. Different categories of GW sources may be identified by the unique spectral shape of their background emission; hence, the detection of a GWB will provide invaluable information about the evolution of the Universe and the population of GW sources within it.

Due to the considerable amount of data to analyze, and the vast panorama of GWB models to test, the detection and characterization of a GWB requires a community effort. Furthermore, data handling and model building entail a number of different choices, depending on specific analysis purposes. This exemplifies the need for an accessible, flexible, and user-friendly open-source codebase: `pygwb`.

The `pygwb` package is class-based and modular to facilitate the evolution of the code and to increase flexibility of the analysis pipeline. The advantage of the Python language lies in rapid code execution, while maintaining a certain level of user-friendliness, which results in a shallow learning curve and will encourage future contributions to the code from the whole GW community.

For additional information about the package, we refer the reader to the `pygwb paper <https://arxiv.org/pdf/2303.15696.pdf>`_.