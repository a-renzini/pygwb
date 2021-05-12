# pygwb

## Installation instructions

* Everything needed is in igwn conda environment
https://computing.docs.ligo.org/conda/

* More precisely what you need is
  * `numpy`
  * `scipy`
  * `matplotlib`
  * `corner`
  * `gwpy`
  * `bilby`

  ## Modules

  The code is currently structured into 5 modules.

  * `cross_correlation.py` produces cross-correlation spectrograms given data from 2 channels.
  * `postprocessing.py` combines the cross-correlation spectrograms into a final spectrum.
  * `pe.py` defines classes to perform pe with bilby for various models.
  * `constants.py` contains numerical values of constants used throughout the code.
     Constants should never be redefined elsewhere.
  * `util.py` contains miscellaneous useful classes and functionality.
