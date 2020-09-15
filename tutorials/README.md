**PLEASE CLEAR ALL OUTPUTS BEFORE COMMITTING CHANGES TO NOTEBOOKS**

There are four tutorial notebooks here.

* `analyze_O2_open_data.ipynb` will download a short stretch of O2 data from GWOSC and run stochastic_lite, and produce data products that can be compared with stochastic.m run on the same data.

* `run_PE_on_O1_O2.ipynb` will download the O1 and O2 cross-correlation spectra from the data releases, run PE, and reproduce the corner plot and upper limits from the O2 paper (for tensor modes, with a log-uniform prior).

* `simulate_and_analyze_white_noise_and_signal.ipynb` will simulate a stretch of white noise and a signal, run the cross correlation analysis to produce cross-correlation spectra, and finally run PE to recover the injection.

* `analyze_colored_noise_and_signal.ipynb` will analyze a stretch of simulated data with a colored noise spectrum and an alpha=2/3 background, produced in advance by Sylvia Biscoveanu. 
  * **NOTE** To run this notebook requires several large data files which have not been added to the repo. Please contact Andrew if you want to run this notebook.

