---
title: '{\tt pygwb}: a Python-based library for gravitational-wave background searches'
tags:
  - Python
  - astronomy®
  - physics
  - gravitational waves
authors:
  - name: Arianna I. Renzini
    orcid: 0000-0000-0000-0000
    corresponding: true
    email: arenzini@caltech.edu
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Alba Romero-Rodriguez
    orcid: 0000-0000-0000-0000
    affiliation:
  - name: Colm Talbot
    orcid: 0000-0000-0000-0000
    affiliation:
  - name: Max Lalleman
    orcid: 0000-0000-0000-0000
    affiliation: "5"
  - name: Shivaraj Kandhasamy
    orcid: 0000-0000-0000-0000
    affiliation:    
  - name: Kevin Turbang
    orcid: 0000-0000-0000-0000
    affiliation: "3, 5"
  - name: Sylvia Biscoveanu
    orcid: 0000-0000-0000-0000
    affiliation:
  - name: Katarina Martinovic
    orcid: 0000-0000-0000-0000
    affiliation:
  - name: Patrick Meyers
    orcid: 0000-0000-0000-0000
    affiliation:
  - name: Leo Tsukada
    orcid: 0000-0000-0000-0000
    affiliation:
  - name: Kamiel Janssens 
    orcid: 0000-0000-0000-0000
    affiliation: "5"
  - name: Derek Davis
    orcid: 0000-0000-0000-0000
    affiliation:
  - name: Andrew Matas 
    orcid: 0000-0000-0000-0000
    affiliation:   
  - name: Philip Charlton
    orcid: 0000-0000-0000-0000
    affiliation: 
  - name: Guo-chin Liu
    orcid: 0000-0000-0000-0000
    affiliation:
  - name: Irina Dvorkin
    orcid: 0000-0000-0000-0000
    affiliation:
    
affiliations:
 - name: LIGO Laboratory,  California  Institute  of  Technology,  Pasadena,  California  91125,  USA
   index: 1
 - name: Department of Physics, California Institute of Technology, Pasadena, California 91125, USA
   index: 2
 - name: Theoretische Natuurkunde, Vrije Universiteit Brussel, Pleinlaan 2, B-1050 Brussels, Belgium
   index: 3
 - name: Universiteit Antwerpen, Prinsstraat 13, 2000 Antwerpen, België
   index: 5

date: 
bibliography: paper.bib

aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Introduction

A gravitational-wave background (GWB) is expected from the superposition of all gravitational waves (GWs) too faint to be detected individually, or by the incoherent overlap of a large number of signals in the same band. A GWB is primarily characterized by its spectral emission, usually parameterized by the GW fractional energy density spectrum $\Omega_{\rm GW}(f)$, which is the target for stochastic GW searches,

$$
\Omega_{\rm GW}(f) = \frac{1}{\rho_c}\frac{d\rho_{\rm GW}(f)}{d\ln f},
$$

where $d\rho_{\rm GW}$ is the energy density of GWs in the frequency band $f$ to $f + df$, and $\rho_c$ is the critical energy density in the Universe. The detection of a GWB will provide invaluable information about the evolution of the Universe and the population of GW sources within it.

The detection and characterization of a GWB requires a community effort, justifying the need for an accessible and user-friendly open-source code.


# Summary

{\tt pygwb} is a Python-based, open-source stochastic GW analysis package specifically tailored to searches for isotropic GWBs with current ground-based interferometers, namely the Laser Interferometer Gravitational-wave Observatory (LIGO), the Virgo observatory, and the KAGRA detector.  

The {\tt pygwb} package is class-based and modular to facilitate the evolution of the code and to increase flexibility of the analysis pipeline. The advantage of the Python language lies in rapid code execution, while maintaining a certain level of user-friendliness, which results in a shallow learning curve and will encourage future contributions to the code from the whole GW community. A summary of all {\tt pygwb} modules and it main external dependencies can be found in the {\tt pygwb} schema \autoref{fig:schema}.

The package is compatible with GW frame files in a variety of formats, relying on the I/O functionality of {\tt gwpy} [@gwpy]. {\tt NumPy} [@harris2020array] is heavily used within the {\tt pygwb} code, as well as {\tt matplotlib} [@Hunter:2007] for plotting purposes. Some of the frequency-related computations rely on functionalities of the {\tt scipy} [@2020SciPy-NMeth] package. The {\tt astropy} [@astropy] package is employed for cosmology-related computations. The parameter estimation module included in {\tt pygwb} is based on {\tt Bilby} [@Ashton_2019] and the {\tt dynesty} [@Speagle_2020] sampler package.

 A customizable pipeline script, {\tt pygwb\_pipe}, is provided with the package and can be run in default mode, which reproduces the methodology of the LVK isotropic analysis implemented on the most recent observation run [@Abbott_2021]. On the other hand, the modularity of the package allows users are to develop custom {\tt pygwb} pipelines to fit their needs. 
A parameter estimation script, {\tt pygwb_pe}, is also included and allows to test for a subset of default models with user-defined parameters. {\tt pygwb_pe} is based on the {\tt pygwb} parameter estimation module, {\tt pe}, which allows the user to test both predefined and user-defined models and obtain posterior distributions on the parameters of interest. Users are encouraged to develop and test their own models within the {\tt pe} module.
 The {\tt pygwb} package also contains built-in support for running on {\tt HTCondor}-supported servers using {\tt dag} files to parallelise the analysis of long stretches of data. Using the dedicated {\tt pygwb_combine} script, the output can be combined into an overall estimation of the GWB for the whole data set.

The source code can be found at https://github.com/a-renzini/pygwb, or it can be installed from {\tt PyPi} via {\tt pip install pygwb}. The online documentation, tutorials and examples are hosted at https://pygwb.docs.ligo.org/pygwb/index.html. {\tt pygwb} is released under a OSI Approved :: MIT License.

![pygwb schema.\label{fig:schema}](../docs/pygwb_modules.png){width=80%}


# Method

The GWB spectrum estimation implemented in {\tt pygwb} is based on the unbiased minimum variance cross-correlation estimator [@RomanoRev],

$$
\hat{\Omega}_{{GW}, f} = \frac{\mathbb{R}\[C_{IJ, f}\]}{\gamma_{IJ}(f) S_0(f)}.
$$

Here, $C_{IJ, f}$ is the cross-correlation spectral density between two detectors $I$ and $J$, $\gamma_{IJ}$ is the overlap reduction function, and $S_0(f)$ = $\frac{3H_0^2}{10\pi^2}\frac{1}{f^3}$, where $H_0$ is the Hubble constant today [@Planck2018]. The variance of the estimator is given by

$$
\sigma^2_{{\rm GW,} f} = \frac{1}{2 T \Delta f} \frac{P_{I, f} P_{J, f}}{\gamma^2_{IJ}(f) S^2_0(f)},
$$

where $P_{I,f}$ is the power spectral density from the detector $I$ and $T$ is the duration of data used to produce the above spectral densities. Details on how the estimation is carried out, as well as the implementation of the estimator on large datasets and with many potentially overlapping datasegments can be found in our companion methods paper [@].

In addition, the {\tt pygwb} package offers the possibility to perform parameter estimation on predefined and user-defined models. Concretely, the above cross-correlation estimator is used in a Gaussian likelihood to perform Bayesian inference:
$$
p\qty(\hat{\Omega}^{IJ}_{{\rm GW}, f} | \mathbf{\Theta}) \propto\exp\left[  -\frac{1}{2} \sum_{IJ}^B \sum_f \left(\frac{\hat{\Omega}^{IJ}_{{\rm GW}, f}  - \Omega_{\rm M}(f|\mathbf{\Theta})}{\hat{\sigma}^{IJ}_{{\rm GW}, f}}\right)^2  \right],
$$
where $\Omega_{\rm M}(f|\mathbf{\Theta})$ is the GWB model and $\mathbf{\Theta}$ are its parameters. More information about the parameter estimation and the implemented models can be found in our companion methods paper [@].


# Acknowledgements

We would like to thank the LVK stochastic group for its continued support. AIR is supported by the NSF award 1912594. ARR is supported in part by the Strategic Research Program “High-Energy Physics” of the Research Council of the Vrije Universiteit Brussel and by the iBOF “Unlocking the Dark Universe with Gravitational Wave Observations: from Quantum Optics to Quantum Gravity” of the Vlaamse Interuniversitaire Raad and by the FWO IRI grant I002123N “Essential Technologies for the Einstein Telescope”. KT is supported by FWO-Vlaanderen through grant number 1179522N. PMM is supported by the NANOGrav Physics Frontiers Center, National Science Foundation (NSF), award number 2020265. LT is supported by the National Science Foundation through OAC-2103662 and PHY-2011865. KJ is supported by FWO-Vlaanderen via grant number 11C5720N. DD is supported by the NSF as a part of the LIGO Laboratory. AM is supported by the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No. 754510.

This material is based upon work supported by NSF’s LIGO Laboratory which is a major facility fully funded by the National Science Foundation. LIGO was constructed by the California Institute of Technology and Massachusetts Institute of Technology with funding from the National Science Foundation, and operates under cooperative agreement PHY-1764464. Advanced LIGO was built under award PHY-0823459. The authors are grateful for computational resources provided by the LIGO Laboratory and supported by NSF Grants PHY-0757058 and PHY-0823459. This work carries LIGO document number ... .


# References


#numpy
#scipy==1.8.0
#matplotlib
#corner
#gwpy==3.0.1
#bilby
#astropy>=4.3.0
#lalsuite==7.3
#loguru
#json5
#jinja2==3.0.3
#seaborn