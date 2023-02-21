[![pipeline status](https://git.ligo.org/pygwb/pygwb/badges/master/pipeline.svg)](https://git.ligo.org/pygwb/pygwb/-/pipelines)
[![coverage report](https://git.ligo.org/pygwb/pygwb/badges/master/coverage.svg)](https://git.ligo.org/pygwb/pygwb/-/commits/master)

# pygwb

[**Documentation**](https://pygwb.docs.ligo.org/pygwb/)

## Installation instructions

* Essentials to support `pygwb` are present in live igwn conda environments
https://computing.docs.ligo.org/conda/

* More precisely, current dependencies are
  * `numpy`
  * `scipy==1.8.0`
  * `matplotlib`
  * `corner`
  * `gwpy==3.0.1`
  * `bilby`
  * `astropy>=4.3.0`
  * `lalsuite==7.3`
  * `loguru`
  * `json5`
  * `jinja2==3.0.3`
  * `seaborn`

  ## Modules

  The code is structured into many small modules.

  * `pre-processing.py` applies initial data-conditioning steps (high-pass filter and downsampling) on data from individual detector. Also supports importing simualted data.
  * `spectral.py` calculated CSDs and PSDs for each segment in a job (a coincident time segment of a pair of detectors)
  * `postprocessing.py` combines the cross-correlation spectrograms into a final spectrum.
  * `pe.py` defines classes to perform pe with bilby for various models.
  * `constants.py` contains numerical values of constants used throughout the code.
     Constants should never be redefined elsewhere.
  * `util.py` contains miscellaneous useful classes and functionality.

  ## Scripts

  A set of scripts are supported to run every-day stochastic tasks.
  * `pygwb_pipe` runs the cross-correlation stochastic analysis over data from selected detector pair, within the timeframes requested.
  * `pygwb_combine` combines over multiple `pygwb_pipe` output files. Useful when running long analyses in parallel.
  * `pygwb_pe` runs parameter estimation on desired parameters.
  * `pygwb_stats` produces regular statistical checks output.
