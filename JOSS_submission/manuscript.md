---
title: '``pygwb``: a Python-based library for gravitational-wave background searches'
tags:
  - Python
  - astronomy
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
    affiliation: " 5 "
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
 - name: Universiteit Antwerpen, Prinsstraat 13, 2000 Antwerpen, België
   index: 5

date: 
bibliography: paper.bib

aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Aiming to make the isotropic search for a gravitational wave background (GWB) more accessible and user friendly, `pygwb` is a pure Python, open source analysis package which will be used in the LIGO-Virgo-KAGRA Collaboration (LVK) and is open for everyone interested in gravitational waves analysis. It is designed via a class-based and modular approach to facilitate adding more capabilities and user options to the analysis pipeline. The pygwb package uses the I/O functionality provided by `gwpy` [@gwpy]. 
It is built using assets from `Bilby` [@Ashton_2019]. Bilbys Bayesian inference library is also used, specifically with the `dynesty` [@Speagle_2020] package. The values of the constants $H_0$ and $c$ are provided by `Astropy` [@Collaboration2022TheAP]. `NumPy` [@harris2020array] provides significant support for the `pygwb` code as does `matplotlib` [@Hunter:2007]. For some frequency related calculations, `scipy` [@2020SciPy-NMeth] functionalities are used. 

The discovery of the GWB would be a major breakthrough in the gravitational wave (GW) community, giving us information about the population and distribution of extremely massive objects in the Universe. It would also pave the way for encovering interesting and possibly new physics in the form of primordial black holes, dark matter and relics of inflation. The analysis of the GWB needs efficient and user-friendly code to make the community able to work quickly and dedicatedly on this problem. Writing `pygwb` in Python allows for a rapid code execution but also allows for not cutting down on easy user access for the (GW) community and the possibility to make fast adjustments and add more GWB models to the code itself in an efficient way.   

The package can read public available gwf frame files, but also locally made gwf files of any type of data. After reading in the data, the default set-up of `pygwb` will run the LVK isotropic analysis [@Abbott_2021], however due to the modular approach of the package, one could make different `pygwb` pipelines depending on ones needs. The package also contains built-in support for running on HTCondor supported servers using dag files. The output files will give information about the analysis, such as PSDs and CSDs computed during the computations. Using the scripts provided by `pygwb`, the output can be combined into one overall estimation of the GWB for the analysis and one can run parameter estimation on the data choosing your own priors and model. 

The source code can be found at https://github.com/a-renzini/pygwb, or it can be installed from `PyPi` via `pip install pygwb`. The online documentation, tutorials and examples are hosted at https://pygwb.docs.ligo.org/pygwb/index.html.

# Statement of need

# Method

$$
\hat{\Omega}_{{GW}, f} = \frac{\mathbb{R}\[C_{IJ, f}\]}{\gamma_{IJ}(f) S_0(f)} 
$$

$$
\sigma^2_{{\rm GW,} f} = \frac{1}{2 T \Delta f} \frac{P_{I, f} P_{J, f}}{\gamma^2_{IJ}(f) S^2_0(f)},
$$

Pygwb schema figure \autoref{fig:schema}.
![pygwb schema.\label{fig:schema}](../docs/pygwb_modules.png){width=80%}


# Acknowledgements

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