[![pipeline status](https://git.ligo.org/pygwb/pygwb/badges/master/pipeline.svg)](https://git.ligo.org/pygwb/pygwb/-/pipelines)
[![coverage report](https://git.ligo.org/pygwb/pygwb/badges/master/coverage.svg)](https://git.ligo.org/pygwb/pygwb/-/commits/master)

# pygwb

[**Documentation**](https://pygwb.docs.ligo.org/pygwb/)

## Installation instructions

* Essentials to support `pygwb` are present in live igwn conda environments
https://computing.docs.ligo.org/conda/

* More precisely, current dependencies are
  * `numpy`
  * `scipy>=1.8.0`
  * `matplotlib`
  * `corner`
  * `gwpy>=3.0.1`
  * `bilby>=1.4`
  * `astropy>=5.2`
  * `lalsuite>=7.3`
  * `gwdetchar`
  * `gwsumm`
  * `pycondor`
  * `loguru`
  * `json5`
  * `seaborn`

  ## Modules

  The code is structured into a set of modules and objects.

  * `detector.py`: contains the `Interferometer` object. The `Interferometer` manages data reading, preprocessing, and PSD estimation.
  * `baseline.py`: contains the `Baseline` object. The `Baseline` is the core manager object in the stochastic analysis.
  * `network.py`:  contains the `Network` object. The `Network` is used to combine results from indibidual `Baselines` as well as simulating data across an `Interferometer` network.
  * `preprocessing.py`: methods for initial data-conditioning steps (high-pass filter and downsampling) on data from an individual detector. Supports importing public, private, or local data.
  * `spectral.py`: methods to calculate CSDs and PSDs for sub-segments in a dataset, made of coincident time segments for a pair of detectors.
  * `postprocessing.py`: methods to combine individual segment cross-correlation spectrograms into a final spectrum, in units of fractional energy density.
  * `omega_spectra.py`: contains the `OmegaSpectrum` and `OmegaSpectrogram` objects.
  * `pe.py`: contains model objects to perform pe with `Bilby`.
  * `statistical_checks.py`: Contains the `StatisticalChecks` object, and methods to run statistical checks on results from an analysis run.
  * `simulator.py`: Contains the `Simulator` object, which can simulate data for a set of detectors.
  * `delta_sigma_cut.py`: Methods to perform the delta-sigma data quality cut.
  * `notch.py`: Contains the `StochNotch`and `StochNotchList` objects, which store information about frequency notches to be applied to the analyzed data spectra.
  * `constants.py`: contains numerical values of constants used throughout the codebase.
  * `orfs.py`: Methods to calcuate overlap reduction functions.
  * `parameters.py`: Contains the `Parameters` dataclass.
  * `util.py`: contains miscellaneous useful functions used throughout the codebase.

  ## Scripts

  A set of scripts are included and maintained to run every-day stochastic tasks.
  * `pygwb_pipe`: runs the cross-correlation stochastic analysis over data from selected detector pair, within the timeframes requested.
  * `pygwb_combine`: combines over multiple `pygwb_pipe` output files. Useful when running long analyses in parallel.
  * `pygwb_pe`: runs parameter estimation on desired model.
  * `pygwb_stats`: produces regular statistical checks output.
  * `pygwb_dag`: supports the creation of a dag file for condor job submission.
