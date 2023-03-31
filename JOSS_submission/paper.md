---
title: '`pygwb`: a Python-based library for gravitational-wave background searches'
tags:
  - Python
  - astronomy®
  - physics
  - gravitational waves
authors:
  - name: Arianna I. Renzini
    orcid: 0000-0002-4589-3987
    corresponding: true
    email: arenzini@caltech.edu
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Alba Romero-Rodriguez
    orcid: 0000-0000-0000-0000
    affiliation: "3"
  - name: Colm Talbot
    orcid: 0000-0000-0000-0000
    affiliation: "4"
  - name: Max Lalleman
    orcid: 0000-0000-0000-0000
    affiliation: "5"
  - name: Shivaraj Kandhasamy
    orcid: 0000-0000-0000-0000
    affiliation: "6"
  - name: Kevin Turbang
    orcid: 0000-0000-0000-0000
    affiliation: "3, 5"
  - name: Sylvia Biscoveanu
    orcid: 0000-0000-0000-0000
    affiliation: "4, 7"
  - name: Katarina Martinovic
    orcid: 0000-0000-0000-0000
    affiliation: "8"
  - name: Patrick Meyers
    orcid: 0000-0000-0000-0000
    affiliation: "9"
  - name: Leo Tsukada
    orcid: 0000-0000-0000-0000
    affiliation: "10, 11"
  - name: Kamiel Janssens 
    orcid: 0000-0000-0000-0000
    affiliation: "5, 12"
  - name: Derek Davis
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2"
  - name: Andrew Matas 
    orcid: 0000-0000-0000-0000
    affiliation: "13"
  - name: Philip Charlton
    orcid: 0000-0000-0000-0000
    affiliation: "14"
  - name: Guo-chin Liu
    orcid: 0000-0000-0000-0000
    affiliation: "15"
  - name: Irina Dvorkin
    orcid: 0000-0000-0000-0000
    affiliation: "16"
    
affiliations:
 - name: LIGO Laboratory,  California  Institute  of  Technology,  Pasadena,  California  91125,  USA
   index: 1
 - name: Department of Physics, California Institute of Technology, Pasadena, California 91125, USA
   index: 2
 - name: Theoretische Natuurkunde, Vrije Universiteit Brussel, Pleinlaan 2, B-1050 Brussels, Belgium
   index: 3
 - name: Kavli Institute for Astrophysics and Space Research, Massachusetts Institute of Technology, 77 Massachusetts Ave, Cambridge, MA 02139, USA
   index: 4
 - name: Universiteit Antwerpen, Prinsstraat 13, 2000 Antwerpen, België
   index: 5
 - name: Inter-University Centre for Astronomy and Astrophysics, Pune 411007, India
   index: 6
 - name: LIGO Laboratory, Massachusetts Institute of Technology, 185 Albany St, Cambridge, MA 02139, USA
   index: 7
 - name: Theoretical Particle Physics and Cosmology Group, Physics Department, King’s College London, University of London, Strand, London WC2R 2LS, United Kingdom
   index: 8
 - name: Theoretical Astrophysics Group, California Institute of Technology, Pasadena, CA 91125, USA
   index: 9
 - name: Department of Physics, The Pennsylvania State University, University Park, Pennsylvania 16802, USA
   index: 10
 - name: Institute for Gravitation and the Cosmos, The Pennsylvania State University, University Park, Pennsylvania 16802, USA
   index: 11
 - name: Université Côte d’Azur, Observatoire Côte d’Azur, ARTEMIS, Nice, France
   index: 12
 - name: Max Planck Institute for Gravitational Physics (Albert Einstein Institute), D-14476 Potsdam, Germany
   index: 13
 - name: OzGrav, Charles Sturt University, Wagga Wagga, New South Wales 2678, Australia
   index: 14
 - name: Department of Physics, Tamkang University, Danshui Dist., New Taipei City 25137, Taiwan
   index: 15
 - name: Institut d’Astrophysique de Paris, Sorbonne Université & CNRS, UMR 7095, 98 bis bd Arago, F-75014 Paris, France
   index: 16
   
date: 
bibliography: paper.bib

aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Introduction

A gravitational-wave background (GWB) is expected from the superposition of all gravitational waves (GWs) too faint to be detected individually, or by the incoherent overlap of a large number of signals in the same band [@AIReview]. A GWB is primarily characterized by its spectral emission, usually parameterized by the GW fractional energy density spectrum $\Omega_{\rm GW}(f)$, which is the target for stochastic GW searches [@AllenRomano],

$$
\Omega_{\rm GW}(f) = \frac{1}{\rho_c}\frac{d\rho_{\rm GW}(f)}{d\ln f},
$$

where $d\rho_{\rm GW}$ is the energy density of GWs in the frequency band $f$ to $f + df$, and $\rho_c$ is the critical energy density of the Universe. Different categories of GW sources may be identified by the unique spectral shape of their background emission; hence, the detection of a GWB will provide invaluable information about the evolution of the Universe and the population of GW sources within it.

Due to the considerable amount of data to analyze, and the vast panorama of GWB models to test, the detection and characterization of a GWB requires a community effort, justifying the need for an accessible and user-friendly open-source code. 

# Method

The GWB spectrum estimation implemented in `pygwb` is based on the unbiased minimum variance cross-correlation estimator [@RomanoRev],

$$
\hat{\Omega}_{{\rm GW}, f} = \frac{{\rm Re}[C_{IJ, f}]}{\gamma_{IJ}(f) S_0(f)}.
$$

Here, $C_{IJ, f}$ is the cross-correlation spectral density between two detectors $I$ and $J$, $\gamma_{IJ}$ is the overlap reduction function [@AllenRomano], and $S_0(f)$ = $\frac{3H_0^2}{10\pi^2}\frac{1}{f^3}$, where $H_0$ is the Hubble constant today [@Planck2018]. The variance of the estimator is given by

$$
\sigma^2_{{\rm GW,} f} = \frac{1}{2 T \Delta f} \frac{P_{I, f} P_{J, f}}{\gamma^2_{IJ}(f) S^2_0(f)},
$$

where $P_{I,f}$ is the power spectral density from detector $I$ and $T$ is the duration of data used to produce the above spectral densities. This estimator is optimal and unbiased under the assumption that the signal is Gaussian, isotropic, and continuous. Details on how the estimation is carried out, as well as the implementation of the estimator on large datasets and with many potentially overlapping datasegments can be found in our companion methods paper [@pygwb_paper].

Model testing in `pygwb` is performed through Bayesian inference on a select set of parameters, given a parametric GWB model and a likelihood $p$ of observing the data given the model. Concretely, the above cross-correlation estimator is input data to a Gaussian residual likelihood,

$$
p\left(\hat{\Omega}^{IJ}_{{\rm GW}, f} | \lambda\right) \propto\exp\left[  -\frac{1}{2} \sum_{IJ}^B \sum_f \left(\frac{\hat{\Omega}^{IJ}_{{\rm GW}, f}  - \Omega_{\rm M}(f|\lambda)}{\hat{\sigma}^{IJ}_{{\rm GW}, f}}\right)^2  \right],
$$

where $\Omega_{\rm M}(f|\lambda)$ is the GWB model and $\lambda$ are its parameters. `pygwb` currently admits a variety of GWB models, compatible with the Gaussian likelihood above. More information about the parameter estimation and the implemented models can be found in our companion methods paper [@pygwb_paper].

# `pygwb`

`pygwb` is a Python-based, open-source stochastic GW analysis package specifically tailored to searches for isotropic GWBs with current ground-based interferometers, namely the Laser Interferometer Gravitational-wave Observatory (LIGO), the Virgo observatory, and the KAGRA detector.  

The `pygwb` package is class-based and modular to facilitate the evolution of the code and to increase flexibility of the analysis pipeline. The advantage of the Python language lies in rapid code execution, while maintaining a certain level of user-friendliness, which results in a shallow learning curve and will encourage future contributions to the code from the whole GW community. A summary of all `pygwb` modules and its main external dependencies can be found in the `pygwb` schema \autoref{fig:schema}.

The package is compatible with GW frame files in a variety of formats, relying on the I/O functionality of `gwpy` [@gwpy]. `NumPy` [@harris2020array] is heavily used within the `pygwb` code, as well as `matplotlib` [@Hunter:2007] for plotting purposes. Some of the frequency-related computations rely on functionalities of the `scipy` [@2020SciPy-NMeth] package. The `astropy` [@astropy] package is employed for cosmology-related computations. The parameter estimation module included in `pygwb` is based on `Bilby` [@Ashton_2019] and the  `dynesty` [@Speagle_2020] sampler package.

A customizable pipeline script, `pygwb_pipe`, is provided with the package and can be run in default mode, which reproduces the methodology of the LIGO-Virgo-KAGRA Collaboration (LVK) isotropic analysis implemented on the most recent observation run [@Abbott_2021]. On the other hand, the modularity of the package allows users to develop custom `pygwb` pipelines to fit their needs. 
A set of simple statistical checks can be performed on the data after a `pygwb` run by using the `statistical_checks` module.
In addition, a parameter estimation script, `pygwb_pe`, is also included and allows to test a subset of default models with user-defined parameters. `pygwb_pe` is based on the `pygwb` parameter estimation module, `pe`, which allows the user to test both predefined and user-defined models and obtain posterior distributions on the parameters of interest. Users are encouraged to develop and test their own models within the `pe` module.
The `pygwb` package also contains built-in support for running on `HTCondor`-supported servers using `dag` files to parallelize the analysis of long stretches of data. Using the dedicated `pygwb_combine` script, the output can be combined into an overall estimation of the GWB for the whole data set.

The source code can be found at https://github.com/a-renzini/pygwb, and can be installed from `PyPi` via `pip install pygwb`. The online documentation, tutorials and examples are hosted at https://pygwb.docs.ligo.org/pygwb/index.html. The package includes a unit test suite which currently covers 80% of the modules. `pygwb` is released under a OSI Approved :: MIT License.

![`pygwb` schema.\label{fig:schema}](../docs/pygwb_modules.png){width=80%}


# Acknowledgements

We thank S. Banagiri, S. Bose, T. Callister, F. De Lillo, L. D'Onofrio, F. Garufi, G. Harry, J. Lawrence, V. Mandic, A. Macquet, I. Michaloliakos, S. Mitra, K. Pham, R. Poggiani, T. Regimbau, J. Romano, N. van Remortel, and H. Zhong for contributing to the review and tests of the code.

We thank the LVK stochastic group for its support. AIR is supported by the NSF award 1912594. ARR is supported in part by the Strategic Research Program “High-Energy Physics” of the Research Council of the Vrije Universiteit Brussel and by the iBOF “Unlocking the Dark Universe with Gravitational Wave Observations: from Quantum Optics to Quantum Gravity” of the Vlaamse Interuniversitaire Raad and by the FWO IRI grant I002123N “Essential Technologies for the Einstein Telescope”. KT is supported by FWO-Vlaanderen through grant number 1179522N. PMM is supported by the NANOGrav Physics Frontiers Center, National Science Foundation (NSF), award number 2020265. LT is supported by the National Science Foundation through OAC-2103662 and PHY-2011865. KJ is supported by FWO-Vlaanderen via grant number 11C5720N. DD is supported by the NSF as a part of the LIGO Laboratory.

This material is based upon work supported by NSF’s LIGO Laboratory which is a major facility fully funded by the National Science Foundation. LIGO was constructed by the California Institute of Technology and Massachusetts Institute of Technology with funding from the National Science Foundation, and operates under cooperative agreement PHY-1764464. Advanced LIGO was built under award PHY-0823459. The authors are grateful for computational resources provided by the LIGO Laboratory and supported by NSF Grants PHY-0757058 and PHY-0823459. 


# References
