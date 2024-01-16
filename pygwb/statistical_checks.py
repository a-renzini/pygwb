"""
The ``statistical_checks`` module performs various tests by plotting different quantities and saving these plots. 
This allows the user to check for consistency with expected results. Concretely, the following tests and plots
can be generated: running point estimate, running sigma, (cumulative) point estimate integrand, real and imaginary 
part of point estimate integrand, FFT of the point estimate integrand, (cumulative) sensitivity, evolution of omega 
and sigma as a function of time, omega and sigma distribution, KS test, and a linear trend analysis of omega in time. 
Furthermore, part of these plots compares the values of these quantities before and after the delta sigma cut.

For additional information on how to run the statistical checks, and interpret them, we refer the user to the dedicatedplot_
tutorials and demos, as well as the `pygwb paper <https://arxiv.org/pdf/2303.15696.pdf>`_.
"""
import json
import warnings
from os import listdir
from os.path import isfile, join
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
from astropy.time import Time
from loguru import logger

matplotlib.rcParams['figure.figsize'] = (8,6)
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linestyle'] = ':'
matplotlib.rcParams['grid.color'] = 'grey'
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['legend.handlelength'] = 3

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# FIXME - changed to get workflow running
rc('text', usetex=False)

import numpy as np
import seaborn as sns
from scipy import integrate, stats
from scipy.optimize import curve_fit

from pygwb.baseline import Baseline
from pygwb.notch import StochNotchList
from pygwb.parameters import Parameters
from pygwb.util import StatKS, calc_bias, effective_welch_averages, get_window_tuple


class StatisticalChecks:
    def __init__(
        self,
        sliding_times_all,
        sliding_omega_all,
        sliding_sigmas_all,
        naive_sigmas_all,
        coherence_spectrum,
        coherence_n_segs,
        point_estimate_spectrum,
        sigma_spectrum,
        frequencies,
        badGPSTimes,
        delta_sigmas,
        plot_dir,
        baseline_name,
        param_file,
        frequency_mask = None,
        coherence_far = 1.0,
        gates_ifo1 = None,
        gates_ifo2 = None,
        file_tag = None,
        legend_fontsize = 16,
        convention = 'pygwb',
        seaborn_palette = 'tab10',
    ):
        """
        Instantiate a StatisticalChecks object.

        Parameters
        ==========

        sliding_times_all: ``array_like``
            Array of GPS times before the bad GPS times from the delta sigma cut are applied.
        sliding_omega_all: ``array_like``
            Array of sliding omegas before the bad GPS times from the delta sigma cut are applied.
        sliding_sigmas_all: ``array_like``
            Array of sliding sigmas before the bad GPS times from the delta sigma cut are applied.
        naive_sigmas_all: ``array_like``
            Array of naive sigmas before the bad GPS times from the delta sigma cut are applied.
        coherence_spectrum: ``array_like``
            Array containing a coherence spectrum. Each entry in this array corresponds to the 2-detector coherence spectrum evaluated at the corresponding frequency in the frequencies array.
        coherence_n_segs: ``int``
            Number of segments used for coherence calculation.
        point_estimate_spectrum: ``array_like``
            Array containing the point estimate spectrum. Each entry in this array corresponds to the point estimate spectrum evaluated at the corresponding frequency in the frequencies array.
        sigma_spectrum: ``array_like``
            Array containing the sigma spectrum. Each entry in this array corresponds to the sigma spectrum evaluated at the corresponding frequency in the frequencies array.
        frequencies: ``array_like``
            Array containing the frequencies.
        badGPStimes: ``array_like``
            Array of bad GPS times, i.e. times that do not pass the delta sigma cut.
        delta_sigmas: ``array_like``
            Array containing the value of delta sigma for all times in sliding_times_all.
        plot_dir: ``str``
            String with the path to which the output of the statistical checks (various plots) will be saved.
        baseline_name: ``str``
            Name of the baseline under consideration.
        param_file: ``str``
            String with path to the file containing the parameters that were used for the analysis run.
        frequency_mask: ``array_like``
            Boolean mask applied to the specrtra in broad-band analyses. 
        coherence_far: ``float``
            Target false alarm rate for number of frequency bins in the coherence spectrum exceeding the coherence threshold.
        gates_ifo1/gates_ifo2: ``list``
            List of gates applied to interferometer 1/2.
        file_tag: ``str``
            Tag to be used in file naming convention.
        legend_fontsize: ``int``
            Font size for plot legends. Default is 16. All other fonts are scaled to this font.
        """
        self.params = Parameters()
        self.params.update_from_file(param_file)

        self.sliding_times_all = sliding_times_all
        self.days_all = (sliding_times_all - sliding_times_all[0]) / 86400.0
        self.sliding_omega_all = sliding_omega_all
        self.sliding_sigmas_all = sliding_sigmas_all
        self.naive_sigmas_all = naive_sigmas_all
        self.badGPStimes = badGPSTimes
        self.delta_sigmas_all = delta_sigmas
        self.sliding_deviate_all = (
            self.sliding_omega_all - np.nanmean(self.sliding_omega_all)
        ) / self.sliding_sigmas_all
        self.gates_ifo1 = gates_ifo1
        self.gates_ifo2 = gates_ifo2
        self.frequencies = frequencies
        if frequency_mask is not None:
            self.frequency_mask = frequency_mask
        else:
            self.frequency_mask = True

        self.coherence_spectrum = coherence_spectrum
        self.coherence_far = coherence_far
        if coherence_n_segs is not None:
            # FFT length in seconds
            fftlength = int(1.0 / (self.frequencies[1] - self.frequencies[0]))

            # FFT number of samples
            nFFT = int(fftlength*self.params.new_sample_rate)

            # Number of samples in a segment used for calculating PSD
            nSamples = int(self.params.segment_duration*self.params.new_sample_rate)
            
            window_tuple = get_window_tuple(self.params.window_fft_dict_welch)

            # Total number of samples included in the all coherence segments combined. This is only approximate
            # since the coherences are combined across several discrete science segments. A more accurate count
            # could be obtained by saving this quantity for each science segment.
            N_tot = int(nSamples + (coherence_n_segs - 1)*(1 - self.params.overlap_factor)*nSamples)

            # Total number of effective segments - need to carefully check how many independent segments
            # there should be.
            self.n_segs = effective_welch_averages(N_tot, nFFT, window_tuple, self.params.overlap_factor_welch)
            
            # The old method, gives a slightly larger number of segments
            # self.n_segs = coherence_n_segs*(1.-self.params.overlap_factor)
            #     * int(np.floor(self.params.segment_duration/(fftlength*(1.-self.params.overlap_factor_welch)))-1)

            if self.params.coarse_grain_csd:
                # Note: this breaks down when self.params.segment_duration/fftlength < 3
                self.n_segs = coherence_n_segs*(1.-self.params.overlap_factor) * int(np.floor(self.params.segment_duration/(fftlength)))
            self.n_segs_statement = r"The number of segments is" + f" {self.n_segs}."

        self.sigma_spectrum = sigma_spectrum
        self.point_estimate_spectrum = point_estimate_spectrum

        self.plot_dir = Path(plot_dir)

        self.baseline_name = baseline_name
        self.segment_duration = self.params.segment_duration
        self.deltaF = self.params.frequency_resolution
        self.new_sample_rate = self.params.new_sample_rate
        self.deltaT = 1.0 / self.new_sample_rate
        self.fref = self.params.fref
        self.flow = self.params.flow
        self.fhigh = self.params.fhigh

        self.alpha = self.params.alpha
        (
            self.sliding_times_cut,
            self.days_cut,
            self.sliding_omega_cut,
            self.sliding_sigma_cut,
            self.naive_sigma_cut,
            self.delta_sigmas_cut,
            self.sliding_deviate_cut,
            self.sliding_deviate_KS,
        ) = self.get_data_after_dsc()
        self.dsc_percent = (len(self.sliding_times_all) - len(self.sliding_times_cut))/len(self.sliding_times_all) * 100
        self.dsc_statement = r"The $\Delta\sigma$ cut removed" + f"{float(f'{self.dsc_percent:.2g}'):g}% of the data."

        tot_tot_segs = ((sliding_times_all - sliding_times_all[0])[-1])/(self.params.segment_duration*(1-self.params.overlap_factor))
        self.percent_obs_segs = len(self.naive_sigmas_all)/tot_tot_segs * 100

        (
            self.running_pt_estimate,
            self.running_sigmas,
        ) = self.compute_running_quantities()

        t0_gps = Time(self.sliding_times_all[0], format='gps')
        t0 = Time(t0_gps, format='iso', scale='utc', precision=0, out_subfmt='date_hm')
        tf_gps = Time(self.sliding_times_all[-1], format='gps')
        tf = Time(tf_gps, format='iso', scale='utc', precision=0, out_subfmt='date_hm')
        self.time_tag = f"{t0}"+" $-$ "+f"{tf}"

        if file_tag:
            self.file_tag = file_tag
        else:
            self.file_tag = f"{int(t0_gps.value)}-{int(tf_gps.value)}"

        self.legend_fontsize = legend_fontsize
        self.axes_labelsize = legend_fontsize + 2
        self.title_fontsize = legend_fontsize + 4
        self.annotate_fontsize = legend_fontsize - 4

        ## convention: stochmon
        if convention == 'stochmon':
            self.days_all = self.days_all*24 + t0.ymdhms.hour + t0.ymdhms.minute/60
            self.days_cut = self.days_cut*24 + t0.ymdhms.hour + t0.ymdhms.minute/60
            self.xaxis = f"Hours since {t0}"
        else:
            self.xaxis = f"Days since {t0}"

        self.sea = sns.color_palette(seaborn_palette)


    def get_data_after_dsc(self):
        """
        Function that returns the GPS times, the sliding omegas, the sliding sigmas, the naive sigmas, 
        the delta sigmas and the sliding deviates after the bad GPS times from the delta sigma cut were applied. 
        This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `self.sliding_times_all`).

        Returns
        =======

        sliding_times_cut: ``array_like``
            Array of GPS times after the bad GPS times were applied.
        days_cut: ``array_like``
            Array of days after the bad GPS times were applied.
        sliding_omega_cut: ``array_like``
            Array of the sliding omega values after the bad GPS times were applied.
        sliding_sigma_cut: ``array_like``
            Array of sliding sigmas after the bad GPS times were applied.
        naive_sigma_cut: ``array_like``
            Array of naive sigmas after the bad GPS times were applied.
        delta_sigma_cut: ``array_like``
            Array of the delta sigma values after the bad GPS times were applied.
        sliding_deviate_cut: ``array_like``
            Array of the deviates after the bad GPS times were applied.
        """
        bad_gps_times = self.badGPStimes
        bad_gps_mask = np.array([(t not in bad_gps_times) for t in self.sliding_times_all])

        bad_gps_mask[~np.isfinite(self.sliding_omega_all)] = False
        sliding_times_cut = self.sliding_times_all.copy()
        days_cut = self.days_all.copy()
        sliding_omega_cut = self.sliding_omega_all.copy()
        sliding_sigma_cut = self.sliding_sigmas_all.copy()
        naive_sigma_cut = self.naive_sigmas_all.copy()
        delta_sigma_cut = self.delta_sigmas_all.copy()
        sliding_deviate_cut = self.sliding_deviate_all.copy()

        sliding_times_cut = sliding_times_cut[bad_gps_mask]

        days_cut = days_cut[bad_gps_mask]
        sliding_omega_cut = sliding_omega_cut[bad_gps_mask]
        sliding_sigma_cut = sliding_sigma_cut[bad_gps_mask]
        naive_sigma_cut = naive_sigma_cut[bad_gps_mask]
        delta_sigma_cut = delta_sigma_cut[bad_gps_mask]
        sliding_deviate_cut = (
            sliding_omega_cut - np.nanmean(self.sliding_omega_all)
        ) / sliding_sigma_cut

        sliding_deviate_KS = (
            sliding_omega_cut - np.nanmean(sliding_omega_cut)
        ) / sliding_sigma_cut

        return (
            sliding_times_cut,
            days_cut,
            sliding_omega_cut,
            sliding_sigma_cut,
            naive_sigma_cut,
            delta_sigma_cut,
            sliding_deviate_cut,
            sliding_deviate_KS,
        )

    def compute_running_quantities(self):
        """
        Function that computes the running point estimate and running sigmas from the sliding point estimate and sliding sigmas. 
        This is done only for the values after the delta sigma cut. This method does not require any input parameters, as it accesses 
        the data through the attributes of the class (e.g. `self.sliding_sigma_cut`).

        Returns
        =======

        running_pt_estimate: ``array_like``
            Array containing the values of the running point estimate.
        running_sigmas: ``array_like``
            Array containing the values of the running sigmas.
        """
        running_pt_estimate = self.sliding_omega_cut.copy()
        running_sigmas = self.sliding_sigma_cut.copy()

        ii = 0
        while ii < self.sliding_times_cut.shape[0] - 1:
            ii += 1
            numerator = np.nansum([running_pt_estimate[ii - 1] / (
                running_sigmas[ii - 1] ** 2
            ) , self.sliding_omega_cut[ii] / (self.sliding_sigma_cut[ii] ** 2)])
            denominator = np.nansum([1.0 / (running_sigmas[ii - 1] ** 2) , 1 / (
                self.sliding_sigma_cut[ii] ** 2
            )])
            running_pt_estimate[ii] = numerator / denominator
            running_sigmas[ii] = np.sqrt(1.0 / denominator)

        return running_pt_estimate, running_sigmas

    def compute_ifft_integrand(self):
        """
        Function that computes the inverse Fourier transform of the point estimate integrand. 
        This function does not require any input parameters, as it accesses the data through the 
        attributes of the class (e.g. `self.point_estimate_integrand`).

        Returns
        =======

        t_array: ``array_like``
            Array containing the time lag values (in seconds).
        omega_t: ``array_like``
            Array containing the

        See also
        --------
        numpy.fft.fft
            More information `here <https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html>`_.
        """
        numFreqs = self.point_estimate_spectrum.shape[0]
        fhigh = self.flow + self.deltaF * numFreqs

        fNyq = 1 / (2 * self.deltaT)

        numFreqs_pre = np.floor(self.flow / self.deltaF) - 1
        f_pre = self.deltaF * np.arange(1, numFreqs_pre + 1)
        numFreqs_post = np.floor((fNyq - fhigh) / self.deltaF)
        f_post = fhigh + self.deltaF * np.arange(0, numFreqs_post)
        fp = np.concatenate((f_pre, self.frequencies, f_post))
        fn = -np.flipud(fp)
        f_tot = np.concatenate((fn, np.array([0]), fp))

        integrand_pre = np.zeros(int(numFreqs_pre))
        integrand_post = np.zeros(int(numFreqs_post))
        integrand_p = np.concatenate(
            (integrand_pre, 0.5 * self.point_estimate_spectrum, integrand_post)
        )

        integrand_n = np.flipud(np.conj(integrand_p))

        integrand_tot = np.concatenate((np.array([0]), integrand_p, integrand_n))

        fft_integrand = np.fft.fftshift(np.fft.fft(self.deltaF * integrand_tot))

        t_array = np.arange(
            -1.0 / (2 * self.deltaF) + self.deltaT, 1.0 / (2 * self.deltaF), self.deltaT
        )
        
        # Take the real part to eliminate residual imaginary parts
        omega_t = np.real(np.flipud(fft_integrand))

        return t_array, omega_t

    def plot_running_point_estimate(self, ymin=None, ymax=None):
        """
        Generates and saves a plot of the running point estimate. The plotted values are the 
        ones after the delta sigma cut. This function does not require any input parameters, 
        as it accesses the data through the attributes of the class (e.g. `self.days_cut`).

        Parameters
        =======

        ymin: ``array_like``
            Minimum value on the y-axis.
        ymax: ``array_like``
            Maximum value on the y-axis.
        """
        if self.days_cut.size==0:
            return
        fig = plt.figure(figsize=(10, 8))
        plt.plot(
            self.days_cut,
            self.running_pt_estimate,
            ".",
            color="black",
            markersize=2,
            label=self.baseline_name,
        )
        plt.plot(
            self.days_cut,
            self.running_pt_estimate + 1.65 * self.running_sigmas,
            ".",
            color=self.sea[2],
            markersize=2,
        )
        plt.plot(
            self.days_cut,
            self.running_pt_estimate - 1.65 * self.running_sigmas,
            ".",
            color=self.sea[0],
            markersize=2,
        )
        plt.grid(True)
        plt.xlim(self.days_cut[0], self.days_cut[-1])
        if ymin and ymax:
            plt.ylim(ymin, ymax)
        plt.xlabel(self.xaxis, size=self.axes_labelsize)
        plt.ylabel(r"Point estimate $\pm 1.65 \sigma$", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.annotate(
            f"baseline time: {float(f'{self.percent_obs_segs:.2g}'):g}%",
            xy=(0.5, 0.9),
            xycoords="axes fraction",
            size = self.annotate_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        plt.title(f'Running point estimate in {self.time_tag}', fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-running_point_estimate.png", bbox_inches='tight'
        )
        plt.close()

    def plot_running_sigma(self):
        """
        Generates and saves a plot of the running sigma. The plotted values are the ones after the delta sigma cut. 
        This function does not require any input parameters, as it accesses the data through the attributes of the 
        class (e.g. `self.days_cut`).
        """
        if self.days_cut.size==0:
            return
        fig = plt.figure(figsize=(10, 8))
        plt.plot(
            self.days_cut, self.running_sigmas, '.', markersize=2, color=self.sea[0], label=self.baseline_name
        )
        plt.grid(True)
        plt.yscale("log")
        plt.xlim(self.days_cut[0], self.days_cut[-1])
        plt.xlabel(self.xaxis, size=self.axes_labelsize)
        plt.ylabel(r"$\sigma$", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(r'Running $\sigma$ ' + f'in {self.time_tag}', fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-running_sigma.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_IFFT_point_estimate_integrand(self):
        """
        Generates and saves a plot of the IFFT of the point estimate integrand. The IFFT of the point 
        estimate integrand is computed using the method "compute_ifft_integrand". This function does not 
        require any input parameters, as it accesses the data through the attributes of the class (e.g. `self.point_estimate_integrand`).
        """
        t_array, omega_array = self.compute_ifft_integrand()
        if len(t_array) != len(omega_array):
            warnings.warn("Times and Omega arrays don't match in the IFFT. No plot could be generated. Investigation is highly recommended.")
            return
        
        fig = plt.figure(figsize=(10, 8))
        plt.plot(t_array, omega_array, color=self.sea[0], label=self.baseline_name)
        plt.grid(True)
        plt.xlim(t_array[0], t_array[-1])
        plt.xlabel("Lag (s)", size=self.axes_labelsize)
        plt.ylabel(r"$\Omega$ integrand IFFT", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(r"$\Omega$ integrand IFFT" + f" in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-IFFT_point_estimate_integrand.png", bbox_inches='tight'
        )
        plt.close()

    def plot_SNR_spectrum(self):
        """
        Generates and saves a plot of the point estimate integrand. This function does not require any input parameters, 
        as it accesses the data through the attributes of the class (e.g. `self.point_estimate_integrand`).
        """
        if np.isnan(self.point_estimate_spectrum).all() or not np.real(self.point_estimate_spectrum).any():
            return
        fig, axs = plt.subplots(figsize=(10, 8))
        axs.semilogy(
            self.frequencies,
            abs(self.point_estimate_spectrum / self.sigma_spectrum),
            color=self.sea[0],
        )
        trans = mt.blended_transform_factory(axs.transData, axs.transAxes) 
        axs.vlines(self.frequencies[~self.frequency_mask], ymin=0, ymax=1, linewidth=1, linestyle=':', color='black', alpha=0.5, transform=trans)
        axs.set_xlabel("Frequency (Hz)", size=self.axes_labelsize)
        axs.set_ylabel(r"$|{\rm SNR}(f)|$", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        axs.set_xscale("log")
        plt.title(f"|SNR| in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-abs_point_estimate_integrand.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_cumulative_SNR_spectrum(self):
        """
        Generates and saves a plot of the cumulative point estimate integrand. This function does not 
        require any input parameters, as it accesses the data through the attributes of the class (e.g. `self.point_estimate_integrand`).
        """
        pt_est_cumul = self.point_estimate_spectrum.copy()
        pt_est_cumul[~self.frequency_mask] = 0
        cum_pt_estimate = integrate.cumtrapz(
            np.abs(pt_est_cumul/ self.sigma_spectrum), self.frequencies
        )
        cum_pt_estimate = cum_pt_estimate / cum_pt_estimate[-1]
        fig, axs = plt.subplots(figsize=(10, 8))
        axs.plot(self.frequencies[:-1], cum_pt_estimate, color=self.sea[0])
        axs.set_xlabel("Frequency (Hz)", size=self.axes_labelsize)
        axs.set_ylabel(r"Cumulative $|{\rm SNR}(f)|$", size=self.axes_labelsize)
        axs.set_xscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(f"Cumulative SNR in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-cumulative_SNR_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_real_SNR_spectrum(self):
        """
        Generates and saves a plot of the real part of the SNR spectrum. This function does not require 
        any input parameters, as it accesses the data through the attributes of the class 
        (e.g. `self.point_estimate_spectrum` and `self.sigma_spectrum`).
        """
        fig, axs =  plt.subplots(figsize=(10, 8))
        axs.plot(
            self.frequencies,
            np.real(self.point_estimate_spectrum / self.sigma_spectrum),
            color=self.sea[0],
        )
        trans = mt.blended_transform_factory(axs.transData, axs.transAxes) 
        axs.vlines(self.frequencies[~self.frequency_mask], ymin=0, ymax=1, linewidth=1, linestyle=':', color='black', alpha=0.5, transform=trans)
        axs.set_xlabel("Frequency (Hz)", size=self.axes_labelsize)
        axs.set_ylabel(r"Re$({\rm SNR}(f))$", size=self.axes_labelsize)
        axs.set_xscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(f"Real SNR in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-real_SNR_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_imag_SNR_spectrum(self):
        """
        Generates and saves a plot of the imaginary part of the SNR spectrum. This function does not 
        require any input parameters, as it accesses the data through the attributes of the class 
        (e.g. `self.point_estimate_spectrum` and `self.sigma_spectrum`).
        """
        fig, axs = plt.subplots(figsize=(10, 8))
        axs.plot(
            self.frequencies,
            np.imag(self.point_estimate_spectrum / self.sigma_spectrum),
            color=self.sea[0],
        )
        trans = mt.blended_transform_factory(axs.transData, axs.transAxes) 
        axs.vlines(self.frequencies[~self.frequency_mask], ymin=0, ymax=1, linewidth=1, linestyle=':', color='black', alpha=0.5, transform=trans)
        axs.set_xlabel("Frequency (Hz)", size=self.axes_labelsize)
        axs.set_ylabel(r"Im$({\rm SNR}(f))$", size=self.axes_labelsize)
        axs.set_xscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(f"Imaginary SNR in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-imag_SNR_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_sigma_spectrum(self):
        """
        Generates and saves a plot of the sigma spectrum. This function does not 
        require any input parameters, as it accesses the data through the attributes 
        of the class (e.g. `self.sigma_spectrum`).
        """
        if np.isinf(self.sigma_spectrum).all() or not np.real(self.point_estimate_spectrum).any():
            return

        fig, axs = plt.subplots(figsize=(10, 8))
        axs.plot(self.frequencies, self.sigma_spectrum, color=self.sea[0])
        trans = mt.blended_transform_factory(axs.transData, axs.transAxes) 
        axs.vlines(self.frequencies[~self.frequency_mask], ymin=0, ymax=1, linewidth=1, linestyle=':', color='black', alpha=0.5, transform=trans)
        axs.set_xlabel("Frequency (Hz)", size=self.axes_labelsize)
        axs.set_ylabel(r"$\sigma(f)$", size=self.axes_labelsize)
        axs.set_xscale("log")
        axs.set_yscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(r"Total $\sigma$ spectrum" + f" in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-sigma_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

    def coherence_pdf(self, gamma):
        """
        Theoretical pdf of coherences assuming Gaussian noise

        Parameters

        ==========

        gamma: ``array_like``
            Array of coherence values

        Returns

        ==========

        coherence_pdf: ``array_like``
            Value of PDF at each gamma
        """
        return (self.n_segs - 1) * (1 - gamma)**(self.n_segs - 2)

    def coherence_pvalue(self, gamma):
        """
        Upper tail p-value of the given coherences assuming Gaussian noise

        Parameters

        ==========

        gamma: ``array_like``
            Array of coherence values

        Returns

        ==========

        coherence_pvalue: ``array_like``
            p-value of each gamma
        """
        return (1 - gamma)**(self.n_segs - 1)

    def coherence_threshold(self):
        """
        Returns the coherence threshold corresponding to the given FAR.
        """
        threshold = 1 - (self.coherence_far/len(self.frequencies))**(1/(self.n_segs - 1))

        return threshold

    def plot_coherence_spectrum(self, flow=None, fhigh=None):
        """
        Generates and saves a plot of the coherence spectrum, if present. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `coherence_spectrum`).
        """
        if self.coherence_spectrum is None or self.coherence_spectrum.size==1:
            return

        flow = flow or self.flow
        fhigh = fhigh or self.fhigh

        threshold = self.coherence_threshold()
        
        plt.figure(figsize=(10, 8))
        plt.plot(self.frequencies, self.coherence_spectrum, color=self.sea[0])

        # Plot a reference line representing the mean of the theoretical coherence
        plt.axhline(y=1./self.n_segs,dashes=(4,3),color='black',label='Theoretical coherence level')

        # Plot a line representing the coherence threshold
        plt.axhline(y=threshold,dashes=(4,3),color='red',label='Outlier threshold')

        plt.xlim(flow, fhigh)
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel(r"Coherence", size=self.axes_labelsize)
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.annotate(
            f"{self.params.channel}, threshold $\gamma = ${threshold:.3f}",
            xy=(0.01, 0.03),
            xycoords="axes fraction",
            size = self.annotate_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        plt.legend()
        plt.title(r"Coherence ($\Delta f$ = " + f"{float(f'{self.deltaF:.5g}'):g} Hz) in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-coherence_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.plot(self.frequencies, self.coherence_spectrum, color=self.sea[0])

        # Plot a reference line representing the mean of the theoretical coherence
        plt.axhline(y=1./self.n_segs,dashes=(4,3),color='black',label='Theoretical coherence level')

        # Plot a line representing the coherence threshold
        plt.axhline(y=threshold,dashes=(4,3),color='red',label='Outlier threshold')

        plt.xlim(flow, 200)
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel(r"Coherence", size=self.axes_labelsize)
        plt.yscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.annotate(
            f"{self.params.channel}, threshold $\gamma =$ {threshold:.3f}",
            xy=(0.01, 0.03),
            xycoords="axes fraction",
            size = self.annotate_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        plt.legend()
        plt.title(r"Coherence ($\Delta f$ = " + f"{float(f'{self.deltaF:.5g}'):g} Hz) in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-coherence_spectrum_zoom.png",
            bbox_inches="tight",
        )
        plt.close()


    def plot_hist_coherence(self,total_bins = None):
        r"""
        Generates and saves a histogram of the coherence distribution. The plot shows the data after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class.
        Furthermore, it also saves a text file which contains the frequencies at which outliers of the coherence distribution were identified, i.e. spectral artefacts.
        """
        if self.coherence_spectrum is None or self.coherence_spectrum.size==1:
            return

        coherence = self.coherence_spectrum
        coherence_clipped = np.ones(len(coherence))
        # For zoomed plots, coherence is clipped at 50 times the theoretical mean
        clip_val = 50*(1/self.n_segs)
        for i in range(len(coherence_clipped)):
            if coherence[i] >= clip_val:
                coherence_clipped[i] = clip_val
            else:
                coherence_clipped[i] = coherence[i]
        frequencies = self.frequencies
        if total_bins is None:
            total_bins = 250

        # Bins are chosen so that the highest coherence value is the centre of the last bin
        upper_edge = max(coherence)*total_bins/(total_bins - 0.5)
        bins =  np.linspace(0, upper_edge, total_bins, endpoint=True)
        delta_coherence = bins[1] - bins[0]

        upper_edge_clipped = max(coherence_clipped)*total_bins/(total_bins - 0.5)
        bins_clipped =  np.linspace(0, upper_edge_clipped, total_bins, endpoint=True)
        delta_coherence_clipped = bins_clipped[1] - bins_clipped[0]

        # Note that the number of frequencies should be equal to the total number of counts
        # for the un-notched coherences, but is different to the total number of counts
        # for notched coherences. Care may be needed when comparing predicted counts
        # with the notched coherence histogram.
        n_frequencies = len(frequencies)
        resolution = frequencies[1] - frequencies[0]
        
        coherence_notched = coherence[self.frequency_mask]
        coherence_notched_clipped = coherence_clipped[self.frequency_mask]

        # Coherence values aligned with bin centres up to twice the max value of the coherence
        coherence_highres = np.linspace(delta_coherence/2, 2*upper_edge + delta_coherence/2, num=2*total_bins, endpoint=True)

        # Theoretical pdf of coherences assuming Gaussian noise
        predicted_highres = self.coherence_pdf(coherence_highres)

        # Threshold to give a false-alarm rate of FAR frequency bins per coherence spectrum
        threshold = self.coherence_threshold()

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
   
        axs.hist(
            coherence,
            bins,
            color=self.sea[3],
            ec="k",
            lw = 0.1,
            zorder=1,
            density = True,
            label = 'Before notching',
        )
        axs.hist(
            coherence_notched,
            bins,
            color=self.sea[0],
            ec="k",
            lw = 0.1,
            zorder=2,
            density = True,
            label = 'After notching',
        )
        axs.plot(
            coherence_highres, 
            predicted_highres,
            color=self.sea[1],
            zorder=3,
            alpha = 0.8,
            label="Predicted",
        )
        axs.axvline(
            np.abs(threshold),
            zorder=4,
            color=self.sea[8],
            linestyle='dashed',
            label=f"Threshold ($\\gamma = ${threshold:.3f})",
        )

        axs.set_xlabel(r"Coherence", size=self.axes_labelsize)
        axs.set_ylabel(r"Probability distribution", size=self.axes_labelsize)
        axs.legend(fontsize=self.legend_fontsize)
        axs.set_yscale("log")
        max_coh = max(np.append(coherence, threshold))
        # Go up to nearest 10th
        axs.set_xlim(0, np.ceil(10*max_coh)/10)
        axs.set_ylim(10**np.floor(np.log10(100.0/n_frequencies)), 10**np.ceil(np.log10(predicted_highres[0])))
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)
        plt.title(r"Coherence hist ($\Delta f$ = " + f"{resolution:.5f} Hz) in" f" {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-histogram_coherence.png", bbox_inches = 'tight'
        )
        plt.close()

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))      

        axs.hist(
            coherence_clipped,
            bins_clipped,
            color=self.sea[3],
            ec="k",
            lw = 0.1,
            zorder=1,
            density = True,
            label = 'Before notching',
        )
        axs.hist(
            coherence_notched_clipped,
            bins_clipped,
            color=self.sea[0],
            ec="k",
            lw = 0.1,
            zorder=2,
            density = True,
            label = 'After notching',
        )
        axs.plot(
            coherence_highres, 
            predicted_highres,
            color=self.sea[1],
            zorder=3,
            alpha = 0.8,
            label="Predicted",
        )
        axs.axvline(
            np.abs(threshold),
            zorder=4,
            color=self.sea[8],
            linestyle='dashed',
            label=f"Threshold ($\\gamma = ${threshold:.3f})",
        )

        axs.set_xlabel(r"Coherence", size=self.axes_labelsize)
        axs.set_ylabel(r"Probability distribution", size=self.axes_labelsize)
        axs.legend(fontsize=self.legend_fontsize)
        axs.set_yscale("log")
        max_coh = max(np.append(coherence_clipped, threshold))
        # For zoomed plot, go up to nearest 100th
        axs.set_xlim(0, np.ceil(100*max_coh)/100)
        axs.set_ylim(10**np.floor(np.log10(100.0/n_frequencies)), 10**np.ceil(np.log10(predicted_highres[0])))
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)

        plt.title(r"Coherence hist (zoomed) ($\Delta f$ = " + f"{resolution:.5f} Hz) in" f" {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-histogram_coherence_zoom.png", bbox_inches = 'tight'
        )
        plt.close()

        outlier_coherence = []
        for i in range(len(coherence)):
            if (coherence[i] > np.abs(threshold) and self.frequency_mask[i] == True):
                try:
                    outlier_coherence.append((i, frequencies[i], coherence[i], n_frequencies*self.coherence_pvalue(coherence[i])))
                except IndexError as err:
                    warnings.warn(
                            '\n In outlier_coherence, Frequency now is %f, and coherence is %f, which is out of the boundary 1, please check it'
                            %(frequencies[i],coherence[i])
                            )
                    outlier_coherence_notched.append((frequencies[i], coherence[i],'nan'))

        outlier_coherence_notched = []
        for i in range(len(coherence)):
            if (coherence[i] > np.abs(threshold) and self.frequency_mask[i] == False):
                try:
                    outlier_coherence_notched.append((i, frequencies[i], coherence[i], n_frequencies*self.coherence_pvalue(coherence[i])))
                except IndexError as err:
                    warnings.warn(
                            '\n In outlier_coherence_notched, Frequency now is %f, and coherence is %f, which is out of the boundary 1, please check it'
                            %(frequencies[i],coherence[i])
                            )
                    outlier_coherence_notched.append((frequencies[i], coherence[i], 'nan'))
        
        n_outlier = len(outlier_coherence)
        file_name = f"{self.plot_dir / self.baseline_name}-{self.file_tag}-list_coherence_outlier.txt"
        with open(file_name, 'w') as f:
            f.write('Bin \tFrequency \tCoherence \tExpected counts above this coherence\n')
            for tup in outlier_coherence:
                f.write(f'{tup[0]}\t{tup[1]}\t{tup[2]}\t{tup[3]}\n')
            f.write('\n The outliers below are already included in the applied version of the notch-list\n')
            for tup in outlier_coherence_notched:
                f.write(f'{tup[0]}\t{tup[1]}\t{tup[2]}\t{tup[3]}\n')
                
    def plot_cumulative_sensitivity(self):
        """
        Generates and saves a plot of the cumulative sensitivity. This function does not 
        require any input parameters, as it accesses the data through the attributes of 
        the class (e.g. `self.sigma_spectrum`).
        """
        if np.isinf(self.sigma_spectrum).all() or not np.real(self.point_estimate_spectrum).any():
            return

        sigma_cumul = self.sigma_spectrum.copy()
        sigma_cumul[~self.frequency_mask] = np.inf
        cumul_sens = integrate.cumtrapz((1 / sigma_cumul ** 2), self.frequencies)
        cumul_sens = cumul_sens / cumul_sens[-1]
        plt.figure(figsize=(10, 8))
        plt.plot(self.frequencies[:-1], cumul_sens, color=self.sea[0])
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel("Cumulative sensitivity", size=self.axes_labelsize)
        plt.xscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(r"$1/\sigma^2$ " + f"in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-cumulative_sigma_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_omega_sigma_in_time(self):
        r"""
        Generates and saves a panel plot with a scatter plot of :math:`\sigma` vs 
        :math:`\Delta{\rm SNR}_i`, as well as the evolution of :math:`\Omega`, :math:`\sigma`, 
        and :math:`(\Omega-\langle\Omega\rangle)/\sigma` as a function of the days since the 
        start of the run. All plots show the data before and after the delta-sigma cut (bad GPS times) 
        was applied. This function does not require any input parameters, as it accesses the data 
        through the attributes of the class (e.g. `self.sliding_sigmas_all`).
        """
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), constrained_layout=True)
        fig.suptitle(r"$\Omega$, $\sigma$, and" + f" SNR variations in {self.time_tag} with/out " + r"$\Delta\sigma$ cut", fontsize=self.title_fontsize)

        axs[0].plot(self.days_all, self.sliding_omega_all, color=self.sea[3], linewidth=1, alpha=0.5, label="All data")
        axs[0].plot(
            self.days_cut,
            self.sliding_omega_cut,
            color=self.sea[0], linewidth=1, alpha=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs[0].plot(self.days_all, self.sliding_omega_all, '.', color=self.sea[3])
        axs[0].plot(
            self.days_cut,
            self.sliding_omega_cut, '.',
            color=self.sea[0],
        )
        axs[0].set_xlabel(self.xaxis, size=self.axes_labelsize)
        axs[0].set_ylabel(r"$\Omega$", size=self.axes_labelsize)
        axs[0].legend(loc="upper left", fontsize=self.legend_fontsize)
        axs[0].set_xlim(0, self.days_all[-1])
        axs[0].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[0].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[0].yaxis.offsetText.set_fontsize(self.legend_fontsize)

        axs[1].plot(self.days_all, self.sliding_sigmas_all, color=self.sea[3], linewidth=1, alpha=0.5, label="All data")
        axs[1].plot(
            self.days_cut,
            self.sliding_sigma_cut,
            color=self.sea[0],
            linewidth=1,
            alpha=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs[1].plot(self.days_all, self.sliding_sigmas_all,'.', color=self.sea[3])
        axs[1].plot(
            self.days_cut,
            self.sliding_sigma_cut,'.',
            color=self.sea[0]
        )
        axs[1].set_xlabel(self.xaxis, size=self.axes_labelsize)
        axs[1].set_ylabel(r"$\sigma$", size=self.axes_labelsize)
        axs[1].legend(loc="upper left", fontsize=self.legend_fontsize)
        axs[1].set_xlim(0, self.days_all[-1])
        axs[1].set_yscale('log')
        axs[1].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[1].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[1].yaxis.offsetText.set_fontsize(self.legend_fontsize)

        axs[2].plot(self.days_all, self.sliding_deviate_all, color=self.sea[3], linewidth=1, alpha=0.5, label="All data")
        axs[2].plot(
            self.days_cut,
            self.sliding_deviate_cut,
            color=self.sea[0], linewidth=1, alpha=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs[2].plot(self.days_all, self.sliding_deviate_all, '.', color=self.sea[3])
        axs[2].plot(
            self.days_cut,
            self.sliding_deviate_cut, '.',
            color=self.sea[0],
        )
        axs[2].set_xlabel(self.xaxis, size=self.axes_labelsize)
        axs[2].set_ylabel(r"$\Delta{\rm SNR}_i$", size=self.axes_labelsize)
        axs[2].legend(loc="upper left", fontsize=self.legend_fontsize)
        axs[2].set_xlim(0, self.days_all[-1])
        axs[2].set_yscale("symlog")
        axs[2].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[2].tick_params(axis="y", labelsize=self.legend_fontsize)

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-omega_sigma_time.png", bbox_inches='tight'
        )
        plt.close()

    def plot_hist_sigma_dsc(self):
        r"""
        Generates and saves a panel plot with a histogram of :math:`|\Delta\sigma|/\sigma`, 
        as well as a histogram of :math:`\sigma`. Both plots show the data before and after 
        the delta-sigma cut (bad GPS times) was applied. This function does not require any 
        input parameters, as it accesses the data through the attributes of the class (e.g. `self.delta_sigmas_all`).

        """
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 14), constrained_layout=True)
        fig.suptitle(r"$\Delta\sigma$ and $\sigma$ distributions in" f" {self.time_tag} with/out " + r"$\Delta\sigma$ cut", fontsize=self.title_fontsize)

        axs[0].hist(
            self.delta_sigmas_all,
            bins=80,
            color=self.sea[3],
            ec="k",
            lw=0.5,
            label="All data",
            range=(0.0001, 1),
        )
        axs[0].hist(
            self.delta_sigmas_cut,
            bins=80,
            color=self.sea[0],
            alpha = 0.6,
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
            range=(0.0001, 1),
        )
        axs[0].set_xlabel(r"$|\Delta\sigma|/\sigma$", size=self.axes_labelsize)
        axs[0].set_ylabel(r"count", size=self.axes_labelsize)
        axs[0].legend(fontsize=self.legend_fontsize)
        axs[0].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[0].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[0].yaxis.offsetText.set_fontsize(self.legend_fontsize)

        if self.sliding_sigma_cut.size==0:
            minx1 = min(self.sliding_sigmas_all)
            maxx1 = max(self.sliding_sigmas_all)
        else:
            minx1 = min(self.sliding_sigma_cut)
            maxx1 = max(self.sliding_sigma_cut)
        nx = 50

        axs[1].hist(
            self.sliding_sigmas_all,
            bins=nx,
            color=self.sea[3],
            ec="k",
            lw=0.5,
            label="All data",
            range=(minx1, maxx1),
        )
        axs[1].hist(
            self.sliding_sigma_cut,
            bins=nx,
            color=self.sea[0],
            alpha = 0.6,
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
            range=(minx1, maxx1),
        )
        axs[1].set_xlabel(r"$\sigma$", size=self.axes_labelsize)
        axs[1].set_ylabel(r"count", size=self.axes_labelsize)
        axs[1].legend(fontsize=self.legend_fontsize)
        axs[1].set_yscale("log")
        axs[1].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[1].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[1].xaxis.offsetText.set_fontsize(self.legend_fontsize)

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-histogram_sigma_dsc.png", bbox_inches='tight'
        )
        plt.close()

    def plot_scatter_sigma_dsc(self):
        """
        Generates and saves a scatter plot of :math:`|\Delta\sigma]/\sigma` vs :math:`\sigma`. 
        The plot shows the data before and after the delta-sigma cut (bad GPS times) was applied. 
        This function does not require any input parameters, as it accesses the data through the 
        attributes of the class (e.g. `self.delta_sigmas_all`).
        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

        axs.scatter(
            self.delta_sigmas_all,
            self.naive_sigmas_all,
            marker=".",
            color=self.sea[3],
            label="All data",
            s=5,
        )
        axs.scatter(
            self.delta_sigmas_cut,
            self.naive_sigma_cut,
            marker=".",
            color=self.sea[0],
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",

            s=5,
        )
        axs.set_xlabel(r"$|\Delta\sigma|/\sigma$", size=self.axes_labelsize)
        axs.set_ylabel(r"$\sigma$", size=self.axes_labelsize)
        axs.set_yscale("log")
        axs.set_xscale("log")
        axs.legend(fontsize=self.legend_fontsize)
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)

        axs.annotate(
            r"Data cut by $\Delta\sigma$ cut"+f": {float(f'{self.dsc_percent:.2g}'):g}%",
            xy=(0.05, 0.8),
            xycoords="axes fraction",
            size = self.annotate_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        plt.title(r"$\Delta\sigma$ distribution in" f" {self.time_tag} with/out " + r"$\Delta\sigma$ cut", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-scatter_sigma_dsc.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_scatter_omega_sigma_dsc(self):
        r"""
        Generates and saves a panel plot with scatter plots of :math:`|\Delta\sigma|/\sigma` vs :math:`\Delta{\rm SNR}_i`, 
        as well as :math:`\sigma` vs :math:`(\Omega-\langle\Omega\rangle)/\sigma`. All plots show the data before and after 
        the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses 
        the data through the attributes of the class (e.g. `self.delta_sigmas_all`).
        """
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 13), constrained_layout=True)
        fig.suptitle(r"$\Delta$SNR spread" + f" in {self.time_tag} with/out " + r"$\Delta\sigma$ cut", fontsize=self.title_fontsize)

        if self.delta_sigmas_cut.size==0:
            maxx0 = max(self.delta_sigmas_all)
            maxx0 += maxx0 / 10.0
            minx0 = min(self.delta_sigmas_all)
            minx0 -= minx0 / 10.0
            maxy0 = np.nanmax(self.sliding_deviate_all)
            maxy0 += maxy0 / 10.0
            miny0 = np.nanmin(self.sliding_deviate_all)
            miny0 -= miny0 / 10.0

            maxx1 = max(self.sliding_sigmas_all)
            maxx1 += maxx1 / 10.0
            minx1 = min(self.sliding_sigmas_all)
            minx1 -= minx1 / 10.0
            maxy1 = max(self.sliding_deviate_all)
            maxy1 += maxy1 / 10.0
            miny1 = min(self.sliding_deviate_all)
            miny1 -= miny1 / 10.0

        else:
            maxx0 = max(self.delta_sigmas_cut)
            maxx0 += maxx0 / 10.0
            minx0 = min(self.delta_sigmas_cut)
            minx0 -= minx0 / 10.0
            maxy0 = np.nanmax(self.sliding_deviate_cut)
            maxy0 += maxy0 / 10.0
            miny0 = np.nanmin(self.sliding_deviate_cut)
            miny0 -= miny0 / 10.0

            maxx1 = max(self.sliding_sigma_cut)
            maxx1 += maxx1 / 10.0
            minx1 = min(self.sliding_sigma_cut)
            minx1 -= minx1 / 10.0
            maxy1 = max(self.sliding_deviate_cut)
            maxy1 += maxy1 / 10.0
            miny1 = min(self.sliding_deviate_cut)
            miny1 -= miny1 / 10.0

        axs[0].scatter(
            self.delta_sigmas_all,
            self.sliding_deviate_all,
            marker=".",
            color=self.sea[3],
            label="All data",
            s=3,
        )
        axs[0].scatter(
            self.delta_sigmas_cut,
            self.sliding_deviate_cut,
            marker=".",
            color=self.sea[0],
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
            s=3,
        )
        axs[0].set_xlabel(r"$|\Delta\sigma|/\sigma$", size=self.axes_labelsize)
        axs[0].set_ylabel(r"$\Delta{\rm SNR}_i$", size=self.axes_labelsize)
        axs[0].set_xlim(minx0, maxx0)
        axs[0].set_ylim(miny0, maxy0)
        axs[0].legend(fontsize=self.legend_fontsize)
        axs[0].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[0].tick_params(axis="y", labelsize=self.legend_fontsize)

        axs[1].scatter(
            self.sliding_sigmas_all,
            self.sliding_deviate_all,
            marker=".",
            color=self.sea[3],
            label="All data",
            s=3,
        )
        axs[1].scatter(
            self.sliding_sigma_cut,
            self.sliding_deviate_cut,
            marker=".",
            color=self.sea[0],
            label=r"Data after $|\Delta\sigma/\sigma|$ outlier cut",
            s=3,
        )
        axs[1].set_xlabel(r"$\sigma$", size=self.axes_labelsize)
        axs[1].set_ylabel(r"$\Delta{\rm SNR}_i$", size=self.axes_labelsize)
        axs[1].legend(fontsize=self.legend_fontsize)
        axs[1].set_xlim(minx1, maxx1)
        axs[1].set_ylim(miny1, maxy1)
        axs[1].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[1].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[1].xaxis.offsetText.set_fontsize(self.legend_fontsize)

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-scatter_omega_sigma_dsc.png", bbox_inches = 'tight')
        plt.close()

    def plot_hist_omega_pre_post_dsc(self):
        r"""
        Generates and saves a histogram of the :math:`\Delta{\rm SNR}_i` distribution. The plot 
        shows the data before and after the delta-sigma cut (bad GPS times) was applied. This function
          does not require any input parameters, as it accesses the data through the attributes of the 
          class (e.g. `self.sliding_deviate_all`).
        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

        # nan-safing the histograms for good measure...
        bins=np.histogram(np.hstack((self.sliding_deviate_all[~np.isnan(self.sliding_deviate_all)], self.sliding_deviate_cut[~np.isnan(self.sliding_deviate_cut)])), bins=202)[1]
        
        axs.hist(
            self.sliding_deviate_all,
            bins,
            color=self.sea[3],
            ec="k",
            lw=0.5,
            label="All data",
        )
        axs.hist(
            self.sliding_deviate_cut,
            bins,
            color=self.sea[0],
            alpha = 0.6,
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        
        axs.set_xlabel(r"$\Delta{\rm SNR}_i$", size=self.axes_labelsize)
        axs.set_ylabel(r"count", size=self.axes_labelsize)
        axs.legend(fontsize=self.legend_fontsize)
        axs.set_yscale("log")
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)

        plt.title(r"$\Delta$SNR distribution in" f" {self.time_tag} with/out " + r"$\Delta\sigma$ cut", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-histogram_omega_dsc.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_KS_test(self, bias_factor=None):
        """
        Generates and saves a panel plot with results of the Kolmogorov-Smirnov test for Gaussianity. 
        The cumulative distribution of the data (after the delta-sigma (bad GPS times) cut) is compared 
        to the one of Gaussian data, where the bias factor for the sigmas is taken into account. This 
        function does not require any input parameters, as it accesses the data through the attributes 
        of the class (e.g. `self.sliding_deviate_cut`).

        Parameters
        =======

        bias_factor: ``float``, optional
            Bias factor to consider in the KS calculation. Defaults to None, in which case it 
            computes the bias factor on the fly.

        See also
        --------
        pygwb.util.calc_bias

        pygwb.util.StatKS

        """
        if self.delta_sigmas_cut.size==0:
            return

        if bias_factor is None:
            bias_factor = calc_bias(self.segment_duration, self.deltaF, self.deltaT)
        dof_scale_factor = 1.0 / (1.0 + 3.0 / 35.0)
        lx = len(self.sliding_deviate_cut)

        sorted_deviates = np.sort(self.sliding_deviate_KS / bias_factor)

        nanmask = ~np.isnan(sorted_deviates)
        sorted_deviates_nansafe = sorted_deviates[nanmask]
        count, bins_count = np.histogram(sorted_deviates_nansafe, bins=500)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)

        normal = stats.norm(0, 1)
        normal_cdf = normal.cdf(bins_count[1:])

        dks_x = max(abs(cdf - normal_cdf))
        lx_eff = lx * dof_scale_factor

        lam = (np.sqrt(lx_eff) + 0.12 + 0.11 / np.sqrt(lx_eff)) * dks_x
        pval_KS = StatKS(lam)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), constrained_layout=True)
        fig.suptitle(f"Kolmogorov-Smirnov test in {self.time_tag}", fontsize=self.title_fontsize)

        axs[0].plot(bins_count[1:], cdf, "k", label="Data")
        axs[0].plot(
            bins_count[1:],
            normal_cdf,
            color=self.sea[3],
            label=r"Erf with $\sigma$=" + str(round(bias_factor, 2)),
        )
        axs[0].text(
            bins_count[1],
            0.8,
            "KS-statistic: "
            + str(round(dks_x, 3))
            + "\n"
            + "p-value: "
            + str(round(pval_KS, 3)),
        )
        axs[0].legend(loc="lower right", fontsize=self.legend_fontsize)
        axs[0].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[0].tick_params(axis="y", labelsize=self.legend_fontsize)

        axs[1].plot(
            bins_count[1:],
            cdf - normal_cdf,
        )
        axs[1].annotate(
            "Maximum absolute difference: " + str(round(dks_x, 3)),
            xy=(0.025, 0.9),
            xycoords="axes fraction",
            size = self.annotate_fontsize
        )
        axs[1].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[1].tick_params(axis="y", labelsize=self.legend_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-KS_test.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_hist_sigma_squared(self, max_val = 5, total_bins=100, label_number = 6):
        """
        Generates and saves a histogram of :math:`\sigma^2/\langle\sigma^2\rangle`. The plot shows 
        data after the delta-sigma (bad GPS times) cut. This function does not require any input parameters, 
        as it accesses the data through the attributes of the class (e.g. self.sliding_sigma_cut).

        """
        if self.delta_sigmas_cut.size==0:
            return
        
        values = 1 / np.nanmean(self.sliding_sigma_cut ** 2) * self.sliding_sigma_cut ** 2
        
        values_clipped = np.ones(len(values))
        
        for i in range(len(values_clipped)):
            if values[i] >= max_val:
                values_clipped[i] = max_val
            else:
                values_clipped[i] = values[i]
                
        bins = np.linspace(0, max(values_clipped), total_bins)
        
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        _, bins, patches = plt.hist(
            values_clipped,
            bins=bins,
            color=self.sea[0],
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        
        axs.set_xlabel(r"$\sigma^2/\langle\sigma^2\rangle$", size=self.axes_labelsize)
        axs.set_ylabel(r"count", size=self.axes_labelsize)
        axs.set_yscale("log")
        axs.set_xlim(0, max_val)
        axs.legend(fontsize=self.legend_fontsize)
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)
        
        xticks_tmp = np.linspace(0, max_val, label_number)
        labels = [str(i) for i in xticks_tmp]
        labels[-1]+="+"
        plt.xticks(xticks_tmp, labels)
        
        plt.title(f"Relative variance in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-histogram_sigma_squared.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_omega_time_fit(self):
        """
        Generates and saves a plot of :math:`\Omega` as a function of time and fits the data to perform 
        a linear trend analysis. The plot shows data after the delta-sigma (bad GPS times) cut. This function 
        does not require any input parameters, as it accesses the data through the attributes of the 
        class (e.g. self.sliding_omega_cut).

        """
        if self.days_cut.size==0:
            return
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

        t_obs = self.days_cut[-1]
        scale = np.sqrt(np.var(self.sliding_omega_cut))

        def func(x, a, b):
            return a * (x - t_obs / 2) * t_obs + b

        nanmask = ~np.isnan(self.sliding_omega_cut)
        sliding_omega_cut_nansafe = self.sliding_omega_cut[nanmask]
        sliding_sigma_cut_nansafe = self.sliding_sigma_cut[nanmask]

        popt, pcov = curve_fit(
            func,
            self.days_cut[nanmask],
            sliding_omega_cut_nansafe,
            sigma=sliding_sigma_cut_nansafe,
        )
        c1, c2 = popt[0], popt[1]
        axs.plot(self.days_cut, func(self.days_cut, c1, c2), color=self.sea[3])
        axs.plot(self.days_cut, self.sliding_omega_cut, '.', color=self.sea[0], markersize=1)
        axs.plot(self.days_cut, 3 * self.sliding_sigma_cut, color=self.sea[0], linewidth=1.5)
        axs.plot(self.days_cut, -3 * self.sliding_sigma_cut, color=self.sea[0], linewidth=1.5)
        axs.set_xlabel(self.xaxis, size=self.axes_labelsize)
        axs.set_ylabel(r"$\Omega_i$", size=self.axes_labelsize)
        axs.set_xlim(self.days_cut[0], self.days_cut[-1])
        axs.annotate(
            r"Linear trend analysis: $\Omega(t) = C_1 (t-T_{\rm obs}/2) T_{\rm obs} + C_2 C_1 = $"
            + str(f"{c1:.3e}")
            + "\nC$_2$ = "
            + str(f"{c2:.3e}"),
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            size = self.annotate_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)
        axs.xaxis.offsetText.set_fontsize(self.legend_fontsize)
        plt.title(r"Time evolution of $\Omega$ " + f"in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-omega_time_fit.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_sigma_time_fit(self):
        """
        Generates and saves a plot of :math:`\sigma` as a function of time and fits the data to perform 
        a linear trend analysis. The plot shows data after the delta-sigma (bad GPS times) cut. This function 
        does not require any input parameters, as it accesses the data through the attributes of the 
        class (e.g. `self.sliding_sigma_cut`).
        """
        if self.days_cut.size==0:
            return

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

        t_obs = self.days_cut[-1]
        scale = np.sqrt(np.var(self.sliding_sigma_cut))

        def func(x, a, b):
            return a * (x - t_obs / 2) * t_obs + b

        nanmask = ~np.isnan(self.sliding_sigma_cut)
        sliding_sigma_cut_nansafe = self.sliding_sigma_cut[nanmask]
        mean_sigma = np.nanmean(self.sliding_sigma_cut)

        popt, pcov = curve_fit(
            func,
            self.days_cut[nanmask],
            sliding_sigma_cut_nansafe,
        )
        c1, c2 = popt[0], popt[1]
        axs.plot(self.days_cut, func(self.days_cut, c1, c2), color=self.sea[3])
        axs.plot(self.days_cut, self.sliding_sigma_cut, ".", color=self.sea[0], markersize=1)
        axs.set_xlabel(self.xaxis, size=self.axes_labelsize)
        axs.set_ylabel(r"$\sigma_i$", size=self.axes_labelsize)
        axs.set_xlim(self.days_cut[0], self.days_cut[-1])
        axs.set_ylim(mean_sigma - 1.2 * mean_sigma, mean_sigma + 2.2 * mean_sigma)
        axs.annotate(
            r"Linear trend analysis: $\sigma(t) = C_1 (t-T_{\rm obs}/2) T_{\rm obs} + C_2 C_1 = $"
            + str(f"{c1:.3e}")
            + "\nC$_2$ = "
            + str(f"{c2:.3e}"),
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            size = self.annotate_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)
        plt.title(r"Time evolution of $\sigma$ " + f"in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-sigma_time_fit.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_gates_in_time(self):
        if self.gates_ifo1 is None and self.gates_ifo2 is None:
            self.gates_ifo1_statement=None
            self.gates_ifo2_statement=None
            return
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        if self.gates_ifo1 is None:
            self.gates_ifo1_statement=None
        else:
            self.total_gated_time_ifo1 = np.sum(self.gates_ifo1[:,1]-self.gates_ifo1[:,0])
            self.total_gated_percent_ifo1 = self.total_gated_time_ifo1/(int(self.sliding_times_all[-1])- int(self.sliding_times_all[0]))*100
            gate_times_in_days_ifo1 = (np.array(self.gates_ifo1[:,0]) - self.sliding_times_all[0]) / 86400.0
            self.gates_ifo1_statement= f"Data gated out: {self.total_gated_time_ifo1} s\n" f"Percentage: {float(f'{self.total_gated_percent_ifo1:.2g}'):g}%"
            gatefig1 = ax.plot(gate_times_in_days_ifo1, self.gates_ifo1[:,1]-self.gates_ifo1[:,0], 's', color=self.sea[3], label="IFO1:\n" f"{self.gates_ifo1_statement}")
            first_legend = ax.legend(handles=gatefig1, loc=(0.05,0.75), fontsize = self.axes_labelsize)
            ax.add_artist(first_legend)
        if self.gates_ifo2 is None:
            self.gates_ifo2_statement=None
        else:
            self.total_gated_time_ifo2 = np.sum(self.gates_ifo2[:,1]-self.gates_ifo2[:,0])
            self.total_gated_percent_ifo2 = self.total_gated_time_ifo2/(int(self.sliding_times_all[-1])- int(self.sliding_times_all[0]))*100
            gate_times_in_days_ifo2 = (np.array(self.gates_ifo2[:,0]) - self.sliding_times_all[0]) / 86400.0
            self.gates_ifo2_statement= f"Data gated out: {self.total_gated_time_ifo2} s\n" f"Percentage: {float(f'{self.total_gated_percent_ifo2:.2g}'):g}%"
            gatefig2 = ax.plot(gate_times_in_days_ifo2, self.gates_ifo2[:,1]-self.gates_ifo2[:,0], 's', color=self.sea[0], label="IFO2:\n" f"{self.gates_ifo2_statement}")
            ax.legend(handles=gatefig2, loc=(0.05, 0.1), fontsize = self.axes_labelsize)

        ax.set_xlabel(self.xaxis, size=self.axes_labelsize)
        ax.set_ylabel("Gate length (s)", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(f"Gates applied to {self.baseline_name} in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-gates_time.png",
            bbox_inches="tight",
        )
        plt.close()

    def save_all_statements(self):
        """
        Saves all useful statements gathered throughout the checks to a json file.
        """
        statements = {}
        statements['dsc'] = self.dsc_statement
        statements['gates_ifo1'] = self.gates_ifo1_statement
        statements['gates_ifo2'] = self.gates_ifo2_statement
        statements['n_segs'] = self.n_segs_statement
        with open("stats_statements.json", "w") as outfile:
                json.dump(statements, outfile)

    def generate_all_plots(self):
        """
        Generates and saves all the statistical analysis plots.
        """
        self.plot_running_point_estimate()
        self.plot_running_sigma()
        self.plot_IFFT_point_estimate_integrand()
        self.plot_SNR_spectrum()
        self.plot_cumulative_SNR_spectrum()
        self.plot_real_SNR_spectrum()
        self.plot_imag_SNR_spectrum()
        self.plot_sigma_spectrum()
        self.plot_cumulative_sensitivity()
        self.plot_omega_sigma_in_time()
        self.plot_hist_sigma_dsc()
        self.plot_scatter_sigma_dsc()
        self.plot_scatter_omega_sigma_dsc()
        self.plot_hist_omega_pre_post_dsc()
        self.plot_KS_test()
        self.plot_hist_sigma_squared()
        self.plot_omega_time_fit()
        self.plot_sigma_time_fit()
        self.plot_gates_in_time()
        if self.coherence_spectrum is not None:
            self.plot_coherence_spectrum()
            self.plot_hist_coherence()
        self.save_all_statements()

def sortingFunction(item):
    return float(item[5:].partition("-")[0])

def run_statistical_checks_from_file(
    combine_file_path, dsc_file_path, plot_dir, param_file, coherence_far=1.0, legend_fontsize=16, coherence_file_path = None, file_tag = None, convention='pygwb', seaborn_palette='tab10',
):
    """
    Method to generate an instance of the statistical checks class from a set of files.

    Parameters
    =======

    combine_file_path: ``str``
        Full path to the file containing the output of the pygwb.combine script, i.e., with the 
        combined results of the run.

    dsc_file_path: ``str``
        Full path to the file containing the results of the delta sigma cut.

    plot_dir: ``str``
        Full path where the plots generated by the statistical checks module should be saved.
    param_file: ``str``
        Full path to the parameter file that was used for the analysis.
    coherence_far: ``float``
        Coherence false alarm rate
    legend_fontsize: ``int``, optional
        Fontsize used in the plots generated by the module. Defaults to 16.

    See also
    --------
    pygwb.notch.StochNotchList

    pygwb.parameters.Parameters
    """
    params = Parameters()
    params.update_from_file(param_file)
    spectra_file = np.load(combine_file_path)
    dsc_file = np.load(dsc_file_path)

    badGPStimes = dsc_file["badGPStimes"]
    delta_sigmas = dsc_file["delta_sigma_values"]
    sliding_times = dsc_file["delta_sigma_times"]
    naive_sigma_all = dsc_file["naive_sigma_values"]
    gates_ifo1 = dsc_file["ifo_1_gates"]
    gates_ifo2 = dsc_file["ifo_2_gates"]
    if gates_ifo1.size==0:
        gates_ifo1=None
    if gates_ifo2.size==0:
        gates_ifo2=None

    sliding_omega_all, sliding_sigmas_all = (
        spectra_file["point_estimates_seg_UW"],
        spectra_file["sigmas_seg_UW"],
    )

    frequencies = np.arange(
        0,
        params.new_sample_rate / 2.0 + params.frequency_resolution,
        params.frequency_resolution,
    )
    frequency_cut = (params.fhigh >= frequencies) & (frequencies >= params.flow)
    try:
        frequency_mask = spectra_file["frequency_mask"]
    except KeyError:
        try:
            notch_list_path = params.notch_list_path
            logger.debug("loading notches from " + notch_list_path)
            notch_list = StochNotchList.load_from_file(notch_list_path)
            frequency_mask = notch_list.get_notch_mask(frequencies[frequency_cut])

        except:
            frequency_mask = None

    spectrum_file = np.load(combine_file_path, mmap_mode="r")

    point_estimate_spectrum = spectrum_file["point_estimate_spectrum"]
    sigma_spectrum = spectrum_file["sigma_spectrum"]

    baseline_name = params.interferometer_list[0] + params.interferometer_list[1]

    # select alpha for statistical checks
    delta_sigmas_sel = delta_sigmas[1]
    naive_sigmas_sel = naive_sigma_all[1]

    if coherence_file_path is not None:
        coh_data = np.load(coherence_file_path, allow_pickle=True)
        coherence_spectrum = coh_data['coherence']
        coherence_n_segs = coh_data['n_segs_coh']
    else:
        coherence_spectrum = None
        coherence_n_segs = None

    return StatisticalChecks(
        sliding_times,
        sliding_omega_all,
        sliding_sigmas_all,
        naive_sigmas_sel,
        coherence_spectrum,
        coherence_n_segs,
        point_estimate_spectrum,
        sigma_spectrum,
        frequencies[frequency_cut],
        badGPStimes,
        delta_sigmas_sel,
        plot_dir,
        baseline_name,
        param_file,
        frequency_mask,
        coherence_far,
        gates_ifo1,
        gates_ifo2,
        file_tag=file_tag,
        legend_fontsize=legend_fontsize,
        convention=convention,
        seaborn_palette=seaborn_palette,
    )
