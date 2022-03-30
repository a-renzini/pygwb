import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate, special, stats
from scipy.optimize import curve_fit
import pickle
import h5py
import gwpy

from pygwb.util import StatKS, calc_bias, calc_Y_sigma_from_Yf_varf, interpolate_frequency_series
from pygwb.delta_sigma_cut import calc_sens_integrand, calc_sigma_alpha, calc_Hf, WindowFactors
from pygwb.parameters import Parameters
from pygwb.baseline import Baseline

class StatisticalChecks(object):
    def __init__(self, sliding_times_all, sliding_omega_all, sliding_sigmas_all, naive_sigma_all, sensitivity_integrand, point_estimate_integrand, freqs, badGPSTimes, delta_sigmas, plot_dir, baseline_name, param_file):
        """
        The statistical checks class performs various tests by plotting different quantities and saving this plots. This allows the user to check for consistency with expected results. Concretely, the following tests and plots can be generated: running point estimate, running sigma, (cumulative) point estimate integrand, real and imaginary part of point estimate integrand, FFT of the point estimate integrand, (cumulative) sensitivity integrand, evolution of omega and sigma as a function of time, omega and sigma distribution, KS test, and a linear trend analysis of omega in time. Furthermore, part of these plots compares the values of these quantities before and after the delta sigma cut. Each of these plots can be made by calling the relevant class method (e.g. self.plot_running_point_estimate()).
        Parameters
        ==========
        sliding_times_all: array
            Array of GPS times before the bad GPS times from the delta sigma cut are applied.
        sliding_omega_all: array
            Array of sliding omegas before the bad GPS times from the delta sigma cut are applied.
        sliding_sigmas_all: array
            Array of sliding sigmas before the bad GPS times from the delta sigma cut are applied.
        naive_sigma_all: array
            Array of naive sigmas before the bad GPS times from the delta sigma cut are applied.
        sensitivity_integrand: array
            Array containing the sensitivity integrand. Each entry in this array corresponds to the sensitivity integrand evaluated at the corresponding frequency in the freqs array.
        point_estimate_integrand: array
            Array containing the point estimate integrand. Each entry in this array corresponds to the sensitivity integrand evaluated at the corresponding frequency in the freqs array.
        freqs: array
            Array containing the frequencies.
        badGPStimes: array
            Array of bad GPS times, i.e. times that do not pass the delta sigma cut.
        delta_sigmas: array
            Array containing the value of delta sigma for all times in sliding_times_all.
        plot_dir: str
            String with the path to which the output of the statistical checks (various plots) will be saved.
        baseline_name: str
            Name of the baseline under consideration.
        param_file: str
            String with path to the file containing the parameters that were used for the analysis run.
        
        Returns
        =======
        Initializes an instance of the statistical checks class.
        """
        self.sliding_times_all=sliding_times_all
        self.days_all=(sliding_times_all-sliding_times_all[0])/86400.
        self.sliding_omega_all=sliding_omega_all
        self.sliding_sigmas_all=sliding_sigmas_all
        self.naive_sigma_all = naive_sigma_all
        self.badGPStimes = badGPSTimes
        self.delta_sigmas_all = delta_sigmas
        self.sliding_deviate_all = (
            self.sliding_omega_all - np.mean(self.sliding_omega_all)
        ) / self.sliding_sigmas_all
        
        self.sensitivity_integrand = sensitivity_integrand
        self.point_estimate_integrand = point_estimate_integrand
        
        self.plot_dir = plot_dir
        
        self.params = Parameters.from_file(param_file)
        self.baseline_name = baseline_name
        self.segment_duration = self.params.segment_duration
        self.deltaF = self.params.frequency_resolution
        self.new_sample_rate=self.params.new_sample_rate
        self.deltaT=1./self.new_sample_rate
        self.fref = self.params.fref
        self.flow = self.params.flow
        self.fhigh = self.params.fhigh
        
        self.freqs = freqs

        self.alpha = self.params.alpha
        
        self.sliding_times_cut, self.days_cut, self.sliding_omega_cut, self.sliding_sigmas_cut, self.naive_sigma_cut, self.delta_sigmas_cut, self.sliding_deviate_cut = self.get_data_after_dsc()
        
        self.running_pt_estimate, self.running_sigmas = self.compute_running_quantities()
        
    @classmethod
    def from_baseline_pickle(
        cls,
        file_path,
        param_file,
        plot_dir
    ):
        """
        Method to initialize the statistical checks class from a baseline object saved to a pickle file.
        
        Parameters
        ==========
        file_path: str
            String with path to the file containing the pickled baseline.
        param_file: str
            String with path to the file containing the parameters that were used for the analysis run.
        plot_dir: str
            String with the path to which the output of the statistical checks (various plots) will be saved.
        Returns
        =======
        Initializes an instance of the statistical checks class.
        """
        params = Parameters.from_file(param_file)
        
        baseline = Baseline.load_from_pickle(file_path)
        
        orf = baseline.overlap_reduction_function
        
        orf = gwpy.frequencyseries.FrequencySeries(
        baseline.overlap_reduction_function, frequencies=baseline.frequencies
    )
        orf_new = interpolate_frequency_series(orf, baseline.interferometer_1.psd_spectrogram.frequencies.value)
        
        sliding_times_all=baseline.point_estimate_spectrogram.times.value
        sliding_omega_all=np.zeros_like(sliding_times_all)
        sliding_sigmas_all=np.zeros_like(sliding_times_all)
        naive_sigma_all= np.zeros_like(sliding_times_all)
        
        for time in range(len(sliding_times_all)):
            sliding_omega_all[time], sliding_sigmas_all[time] = calc_Y_sigma_from_Yf_varf(baseline.point_estimate_spectrogram.value[time], baseline.sigma_spectrogram.value[time], freqs = baseline.point_estimate_spectrogram.frequencies.value, alpha = params.alpha, fref=params.fref, weight_spectrum=False)
            naive_sigma_all[time] = calc_naive_sigma(baseline.interferometer_1.psd_spectrogram.frequencies.value, baseline.interferometer_1.psd_spectrogram.value[time], baseline.interferometer_2.psd_spectrogram.value[time], orf_new.value, params.frequency_resolution, params.segment_duration, params.new_sample_rate, params.alpha)
        
        badGPSTimes=baseline.badGPStimes
        delta_sigmas=abs(np.random.randn(sliding_times_all.shape[0]))
        #delta_sigmas=baseline.delta_sigmas[1] #Selects the value for alpha=0
        
        sensitivity_integrand = 1./baseline.sigma_spectrum.value**2
        point_estimate_integrand = baseline.point_estimate_spectrum.value
        return cls(sliding_times_all, sliding_omega_all, sliding_sigmas_all, naive_sigma_all, sensitivity_integrand, point_estimate_integrand, baseline.point_estimate_spectrogram.frequencies.value, badGPSTimes, delta_sigmas, plot_dir, baseline.name, param_file)
    
    
    @classmethod
    def from_pickle(cls, point_est_file_path, PSD_file_path, param_file, plot_dir, baseline_name, orf):
        """
        Method to initialize the statistical checks class from a pickle file. 
        
        Parameters
        ==========
        point_estimate_file_path: str
            String with the path to the file containing the point estimate and sigma spectrograms, point estimate and sigma spectra, the bad GPS times, and the value of the delta sigmas.
        PSD_file_path: str
            String with path to the file containing the power spectral densities and cross spectral densities of the two interferometers.
        param_file: str
            String with path to the file containing the parameters that were used for the analysis run.
        plot_dir: str
            String with the path to which the output of the statistical checks (various plots) will be saved.
        basline_name: str
            Name of the baseline
        orf: gwpy.frequencyseries
            Overlap reduction function (as a gwpy frequency series object) for the baseline under consideration. This overlap reduction function will be used in the computation of the naive sigmas.
            
        Returns
        =======
        Initializes an instance of the statistical checks class.
        """        
        
        params = Parameters.from_file(param_file)
        
        with (open(point_est_file_path, "rb")) as f:
            while True:
                try:
                    point_est_file_data = pickle.load(f)
                except EOFError:
                    break
                    
        sliding_times_all = point_est_file_data['point_estimate_spectrogram'].times.value
        sliding_omega_all=np.zeros_like(sliding_times_all)
        sliding_sigmas_all=np.zeros_like(sliding_times_all)
        
        for time in range(len(sliding_times_all)):
            sliding_omega_all[time], sliding_sigmas_all[time] = calc_Y_sigma_from_Yf_varf(point_est_file_data['point_estimate_spectrogram'].value[time], point_est_file_data['sigma_spectrogram'].value[time], freqs = point_est_file_data['point_estimate_spectrogram'].frequencies.value, alpha = params.alpha, fref=params.fref, weight_spectrum=False)
        
        badGPSTimes=point_est_file_data['badGPStimes']
        delta_sigmas=point_est_file_data['delta_sigmas'][1] #Selects value corresponding to alpha=0
        
        sensitivity_integrand = 1./point_est_file_data['sigma_spectrum'].value**2
        point_estimate_integrand = point_est_file_data['point_estimate_spectrum'].value
        
        with (open(PSD_file_path, "rb")) as f:
            while True:
                try:
                    PSD_file_data = pickle.load(f)
                except EOFError:
                    break
                    
        freqs =  PSD_file_data['psd_1'].frequencies.value[1:]
        
        orf_new = interpolate_frequency_series(orf, freqs)
        
        naive_sigma_all= np.zeros_like(sliding_times_all)
        
        for time in range(len(sliding_times_all)):
            naive_sigma_all[time] = calc_naive_sigma(PSD_file_data['psd_1'].frequencies.value[1:], PSD_file_data['psd_1'].value[time][1:], PSD_file_data['psd_2'].value[time][1:], orf_new.value, params.frequency_resolution, params.segment_duration, params.new_sample_rate, params.alpha)
        
        return cls(sliding_times_all, sliding_omega_all, sliding_sigmas_all, naive_sigma_all, sensitivity_integrand, point_estimate_integrand, point_est_file_data['sigma_spectrum'].frequencies, badGPSTimes, delta_sigmas, plot_dir, baseline_name, param_file)
    
    
    @classmethod
    def from_file(cls, point_est_file_path, PSD_file_path, param_file, plot_dir, baseline_name, orf):
        file_name, file_extension = os.path.splitext(point_est_file_path)
    
        if file_extension==(".p" or ".pickle"):
            pass
        elif filefile_extension==".h5":
            pass
        else:
            pass #return error
    
    @classmethod
    def from_hdf5(cls, point_est_file_path, PSD_file_path, param_file, plot_dir, baseline_name, orf):
        """
        """
        with h5py.File(PSD_file_path, "r") as f:
            sliding_times_all = f['psds_group']['psd_1']['psd_1_times'][:]
            naive_sigma_all= np.zeros_like(sliding_times_all)
            orf = f['psds_group']['psd_1']['psd_1'][0][:] #Need to change
            sampling_frequency=1/deltaT
            for time in range(len(sliding_times_all)):
                naive_sigma_all[time] = calc_naive_sigma(f['psds_group']['psd_1']['psd_1'][time][:], f['psds_group']['psd_1']['psd_1'][time][:], f['psds_group']['psd_2']['psd_2'][time][:], orf, deltaF, segment_duration, sampling_frequency, alpha)
        sliding_omega_all=np.zeros_like(sliding_times_all)
        sliding_sigmas_all=np.zeros_like(sliding_times_all)
        
        with h5py.File(point_est_file_path, "r") as f:
            point_estimate_spectrogram = f['point_estimate_spectrogram'][:]
            sigma_spectrogram = f['sigma_spectrogram'][:]
            freqs = f['freqs'][:]
            print(point_estimate_spectrogram.shape)
            for time in range(len(sliding_times_all)):
                sliding_omega_all[time], sliding_sigmas_all[time] = calc_Y_sigma_from_Yf_varf(point_estimate_spectrogram[time][:], sigma_spectrogram[time][:], freqs = freqs, alpha = alpha, fref=fref, weight_spectrum=True)

            badGPSTimes = f['badGPStimes'][:]
            delta_sigmas = f['delta_sigmas'][0][:]
            #point_estimate_spectrum = f['point_estimate_spectrum']
            
        return cls(sliding_times_all, sliding_omega_all, sliding_sigmas_all, naive_sigma_all, badGPSTimes, delta_sigmas, plot_dir, baseline_name, segment_duration, deltaF, deltaT)

        
    def get_data_after_dsc(self):
        """
        Function that returns the GPS times, the sliding omegas, the sliding sigmas, the naive sigmas, the delta sigmas and the sliding deviates after the bad GPS times from the delta sigma cut were applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_times_all).
        Parameters
        ==========
        
        
        Returns
        =======
        sliding_times_cut: array
            Array of GPS times after the bad GPS times were applied.
        days_cut: array
            Array of days after the bad GPS times were applied.
        sliding_omega_cut: array
            Array of the sliding omega values after the bad GPS times were applied.
        sliding_sigmas_cut: array
            Array of sliding sigmas after the bad GPS times were applied.
        naive_sigma_cut: array
            Array of naive sigmas after the bad GPS times were applied.
        delta_sigma_cut: array
            Array of the delta sigma values after the bad GPS times were applied.
        sliding_deviate_cut: array
            Array of the deviates after the bad GPS times were applied.
        """
        bad_gps_times = self.badGPStimes
        bad_gps_mask = [(t not in bad_gps_times) for t in self.sliding_times_all]
        
        sliding_times_cut = self.sliding_times_all.copy()
        days_cut = self.days_all.copy()
        sliding_omega_cut = self.sliding_omega_all.copy()
        sliding_sigmas_cut = self.sliding_sigmas_all.copy()
        naive_sigma_cut = self.naive_sigma_all.copy()
        delta_sigma_cut=self.delta_sigmas_all.copy()
        sliding_deviate_cut=self.sliding_deviate_all.copy()
        
        sliding_times_cut = sliding_times_cut[bad_gps_mask]
        days_cut = days_cut[bad_gps_mask]
        sliding_omega_cut = sliding_omega_cut[bad_gps_mask]
        sliding_sigmas_cut = sliding_sigmas_cut[bad_gps_mask]
        naive_sigma_cut = naive_sigma_cut[bad_gps_mask]
        delta_sigma_cut = delta_sigma_cut[bad_gps_mask]
        sliding_deviate_cut = (sliding_omega_cut- np.mean(sliding_omega_cut)) / sliding_sigmas_cut
        
        return sliding_times_cut, days_cut, sliding_omega_cut, sliding_sigmas_cut, naive_sigma_cut, delta_sigma_cut, sliding_deviate_cut
        
    def compute_running_quantities(self):
        """
        Function that computes the running point estimate and running sigmas from the sliding point estimate and sliding sigmas. This is done only for the values after the delta sigma cut. This method does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_sigmas_cut).
        
        Parameters
        ==========
        
        Returns
        =======
        running_pt_estimate: array
            Array containing the values of the running point estimate.
        running_sigmas: array
            Array containing the values of the running sigmas.
        """
        running_pt_estimate = self.sliding_omega_cut.copy()
        running_sigmas = self.sliding_sigmas_cut.copy()
        
        ii=0
        while ii < self.sliding_times_cut.shape[0]-1:
            ii+=1
            numerator = running_pt_estimate[ii-1]/(running_sigmas[ii-1]**2) + self.sliding_omega_cut[ii]/(self.sliding_sigmas_cut[ii]**2);
            denominator = 1./(running_sigmas[ii-1]**2) + 1/(self.sliding_sigmas_cut[ii]**2)
            running_pt_estimate[ii] = numerator/denominator
            running_sigmas[ii] = np.sqrt(1./denominator);
        
        return running_pt_estimate, running_sigmas

    
    def compute_ifft_integrand(self):
        """
        Function that computes the inverse Fourier transform of the point estimate integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.point_estimate_integrand).
        
        Parameters
        ==========
        
        
        Returns
        =======
        t_array: array
            Array containing the time lag values (in seconds).
        omega_t: array
            Array containing the 
        
        """

        numFreqs = self.point_estimate_integrand.shape[0]
        freqs = self.flow + self.deltaF * np.arange(0,numFreqs)
        fhigh = self.flow + self.deltaF*numFreqs

        fNyq = 1/(2*self.deltaT)

        numFreqs_pre = np.floor(self.flow/self.deltaF)-1
        f_pre = self.deltaF*np.arange(1,numFreqs_pre+1)
        numFreqs_post = np.floor((fNyq - fhigh)/self.deltaF)
        f_post = fhigh + self.deltaF*np.arange(0,numFreqs_post)
        fp =  np.concatenate((f_pre, freqs, f_post))
        fn = -np.flipud(fp)
        f_tot = np.concatenate((fn, np.array([0]), fp))

        integrand_pre  = np.zeros(int(numFreqs_pre))
        integrand_post = np.zeros(int(numFreqs_post))
        integrand_p = np.concatenate((integrand_pre, self.point_estimate_integrand, integrand_post))

        integrand_n = np.flipud(np.conj(integrand_p))

        integrand_tot = np.concatenate((np.array([0]),integrand_p, integrand_n))

        fft_integrand = np.fft.fftshift(np.fft.fft(self.deltaF*integrand_tot))
        
        t_array = np.arange(-1./(2*self.deltaF)+self.deltaT,1./(2*self.deltaF),self.deltaT)
        omega_t = np.flipud(fft_integrand)
        
        return t_array, omega_t
    
    def plot_running_point_estimate(self):
        """
        Generates and saves a plot of the running point estimate. The plotted values are the ones after the delta sigma cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.days_cut).

        Parameters
        ==========

        Returns
        =======

        """
        fig = plt.figure()
        plt.plot(
            self.days_cut, self.running_pt_estimate, ".", color="black", markersize=2, label=self.baseline_name)
        plt.plot(self.days_cut, self.running_pt_estimate + 1.65 * self.running_sigmas, ".", color="green", markersize=2)
        plt.plot(self.days_cut, self.running_pt_estimate - 1.65 * self.running_sigmas, ".", color="blue", markersize=2)
        plt.grid(True)
        #plt.ylim(-5e-7, max())
        plt.xlim(self.days_cut[0], self.days_cut[-1])
        plt.xlabel("Days since start of run")
        plt.ylabel("Point estimate +/- 1.65\u03C3")
        plt.legend()
        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-running_point_estimate.png")

    def plot_running_sigma(self):
        """
        Generates and saves a plot of the running sigma. The plotted values are the ones after the delta sigma cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.days_cut).

        Parameters
        ==========

        Returns
        =======

        """
        fig = plt.figure()
        plt.semilogy(self.days_cut, self.running_sigmas, color="blue", label=self.baseline_name)
        plt.grid(True)
        #plt.ylim(1e-8, 1e-3)
        plt.xlim(self.days_cut[0], self.days_cut[-1])
        plt.xlabel("Days since start of run")
        plt.ylabel("\u03C3")
        plt.legend()
        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-running_sigma.png")

    def plot_IFFT_point_estimate_integrand(self):
        """
        Generates and saves a plot of the IFFT of the point estimate integrand. The IFFT of the point estimate integrand is computed using the method "compute_ifft_integrand". This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.point_estimate_integrand).

        Parameters
        ==========

        Returns
        =======

        """
        t_array, omega_array = self.compute_ifft_integrand()
        
        fig = plt.figure()
        plt.plot(t_array, omega_array, color="b", label=self.baseline_name)
        plt.grid(True)
        #plt.ylim(-1.5e-7, 1e-7)
        plt.xlim(t_array[0], t_array[-1])
        plt.xlabel("Lag (s)")
        plt.ylabel("IFFT of Integrand of Pt Estimate")
        plt.legend()
        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-IFFT_point_estimate_integrand.png")

    def plot_point_estimate_integrand(self):
        """
        Generates and saves a plot of the point estimate integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.point_estimate_integrand).

        Parameters
        ==========

        Returns
        =======

        """
        plt.figure(figsize=(10, 8))
        plt.semilogy(
            self.freqs,
            abs(self.point_estimate_integrand))
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Abs(point estimate integrand)", size=18)
        plt.savefig(
            f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-abs_point_estimate_integrand.png", bbox_inches="tight"
        )

    def plot_cumulative_point_estimate(self):
        """
        Generates and saves a plot of the cumulative point estimate integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.point_estimate_integrand).

        Parameters
        ==========

        Returns
        =======

        """
        cum_pt_estimate = integrate.cumtrapz(
            np.real(self.point_estimate_integrand), self.freqs
        )
        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs[:-1], cum_pt_estimate)
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Cumulative point estimate", size=18)
        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-cumulative_point_estimate.png", bbox_inches="tight")

    def plot_real_point_estimate_integrand(self):
        """
        Generates and saves a plot of the real part of the point estimate integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.point_estimate_integrand).

        Parameters
        ==========

        Returns
        =======

        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs, np.real(self.point_estimate_integrand))
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Re(point estimate integrand)", size=18)
        plt.savefig(
            f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-real_point_estimate_integrand.png", bbox_inches="tight"
        )

    def plot_imaginary_point_estimate_integrand(self):
        """
        Generates and saves a plot of the imaginary part of the point estimate integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.point_estimate_integrand).

        Parameters
        ==========

        Returns
        =======

        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs, np.imag(self.point_estimate_integrand))
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Im(point estimate integrand)", size=18)
        plt.savefig(
            f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-imag_point_estimate_integrand.png", bbox_inches="tight"
        )

    def plot_sensitivity_integrand(self):
        """
        Generates and saves a plot of the sensitivity integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sensitivity_integrand).

        Parameters
        ==========

        Returns
        =======

        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs, self.sensitivity_integrand)
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Sensitivity integrand", size=18)
        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-sensitivity_integrand.png", bbox_inches="tight")

    def plot_cumulative_sensitivity_integrand(self):
        """
        Generates and saves a plot of the cumulative sensitivity integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sensitivity_integrand).

        Parameters
        ==========

        Returns
        =======

        """
        
        cumul_sensitivity = integrate.cumtrapz(
            self.sensitivity_integrand, self.freqs
        )
        cumul_sensitivity=cumul_sensitivity/cumul_sensitivity[-1]
        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs[:-1], cumul_sensitivity)
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Cumulative sensitivity", size=18)
        plt.savefig(
            f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-cumulative_sensitivity_integrand.png", bbox_inches="tight"
        )

    def plot_omega_sigma_in_time(self):
        """
        Generates and saves a panel plot with a scatter plot of \u03C3 vs (\u03A9-<\u03A9>)/\u03C3, as well as the evolution of \u03A9, \u03C3, and (\u03A9-<\u03A9>)/\u03C3 as a function of the days since the start of the run. All plots show the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_sigmas_all).

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
        fig.suptitle(
            "Results from " + self.baseline_name + " for Abs[\u03B4\u03C3]/\u03C3 cut",
            fontsize=20,
        )

        axs[0, 0].scatter(
            self.sliding_sigmas_all,
            self.sliding_deviate_all,
            marker=".",
            c="r",
            s=3,
            label="All data",
        )
        axs[0, 0].scatter(
            self.sliding_sigmas_cut,
            self.sliding_deviate_cut,
            marker=".",
            c="b",
            s=3,
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
        )
        axs[0, 0].set_xlabel("\u03C3")
        axs[0, 0].set_ylabel("(\u03A9-<\u03A9>)/\u03C3")
        axs[0, 0].legend(loc="upper left")

        axs[0, 1].plot(self.days_all, self.sliding_omega_all, c="r", label="All data")
        axs[0, 1].plot(
            self.days_cut,
            self.sliding_omega_cut,
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
        )
        axs[0, 1].set_xlabel("Days since start of run")
        axs[0, 1].set_ylabel("\u03A9")
        axs[0, 1].legend(loc="upper left")

        axs[1, 0].plot(self.days_all, self.sliding_sigmas_all, c="r", label="All data")
        axs[1, 0].plot(
            self.days_cut,
            self.sliding_sigmas_cut,
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
        )
        axs[1, 0].set_xlabel("Days since start of run")
        axs[1, 0].set_ylabel("\u03C3")
        axs[1, 0].legend(loc="upper left")

        axs[1, 1].plot(self.days_all, self.sliding_deviate_all, c="r", label="All data")
        axs[1, 1].plot(
            self.days_cut,
            self.sliding_deviate_cut,
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
        )
        axs[1, 1].set_xlabel("Days since start of run")
        axs[1, 1].set_ylabel("(\u03A9-<\u03A9>)/\u03C3")
        axs[1, 1].legend(loc="upper left")

        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-omega_sigma_time.png")

    def plot_hist_sigma_dsc(self):
        """
        Generates and saves a panel plot with a histogram of Abs[\u03B4\u03C3]/\u03C3, as well as a histogram of \u03C3. Both plots show the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.delta_sigmas_all).

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(13, 13))
        fig.suptitle(
            "Results from " + self.baseline_name + " for Abs[\u03B4\u03C3]/\u03C3",
            fontsize=20,
        )

        axs[0].hist(
            self.delta_sigmas_all,
            bins=80,
            color="r",
            ec="k",
            lw=0.5,
            label="All data",
            range=(0.0001, 1),
        )
        axs[0].hist(
            self.delta_sigmas_cut,
            bins=80,
            color="b",
            ec="k",
            lw=0.5,
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            range=(0.0001, 1),
        )
        axs[0].set_xlabel("Abs[\u03B4\u03C3]/\u03C3")
        axs[0].set_ylabel("# per bin")
        #axs[0].set_xlim([0, 1])
        axs[0].legend()

        minx1 = min(self.sliding_sigmas_cut)
        maxx1 = max(self.sliding_sigmas_cut)
        nx = 50

        axs[1].hist(
            self.sliding_sigmas_all,
            bins=nx,
            color="r",
            ec="k",
            lw=0.5,
            label="All data",
            range=(minx1, maxx1),
        )
        axs[1].hist(
            self.sliding_sigmas_cut,
            bins=nx,
            color="b",
            ec="k",
            lw=0.5,
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            range=(minx1, maxx1),
        )
        axs[1].set_xlabel("\u03C3")
        axs[1].set_ylabel("# per bin")
        axs[1].legend()
        axs[1].set_yscale("log")
        #axs[1].set_xlim([0, maxx1])
        #axs[1].set_ylim([1, 10 ** 4])

        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-histogram_sigma_dsc.png")

    def plot_scatter_sigma_dsc(self):
        """
        Generates and saves a scatter plot of Abs[\u03B4\u03C3]/\u03C3 vs \u03C3. The plot shows the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.delta_sigmas_all).

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        fig.suptitle(
            "Results from " + self.baseline_name + " for Abs[\u03B4\u03C3]/\u03C3",
            fontsize=20,
        )

        axs.scatter(
            self.delta_sigmas_all,
            self.naive_sigma_all,
            marker=".",
            c="r",
            label="All data",
            s=5,
        )
        axs.scatter(
            self.delta_sigmas_cut,
            self.naive_sigma_cut,
            marker=".",
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            s=5,
        )
        axs.set_xlabel("Abs[\u03B4\u03C3]/\u03C3")
        axs.set_ylabel("\u03C3")
        axs.legend()

        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-scatter_sigma_dsc.png")

    def plot_scatter_omega_sigma_dsc(self):
        """
        Generates and saves a panel plot with scatter plots of Abs[\u03B4\u03C3]/\u03C3 vs (\u03A9-<\u03A9>)/\u03C3, as well as \u03C3 vs (\u03A9-<\u03A9>)/\u03C3. All plots show the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.delta_sigmas_all).

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(13, 13))
        fig.suptitle(
            "Results from " + self.baseline_name + " for Abs[\u03B4\u03C3]/\u03C3",
            fontsize=20,
        )

        axs[0].scatter(
            self.delta_sigmas_all,
            self.sliding_deviate_all,
            marker=".",
            c="r",
            label="All data",
            s=3,
        )
        axs[0].scatter(
            self.delta_sigmas_cut,
            self.sliding_deviate_cut,
            marker=".",
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            s=3,
        )
        axs[0].set_xlabel("Abs[\u03B4\u03C3]/\u03C3")
        axs[0].set_ylabel("(\u03A9-<\u03A9>)/\u03C3")
        axs[0].legend()
        #axs[0].set_xlim(0, 2)
        #axs[0].set_ylim(-20, 20)

        maxx1 = max(self.sliding_sigmas_cut)

        axs[1].scatter(
            self.sliding_sigmas_all,
            self.sliding_deviate_all,
            marker=".",
            c="r",
            label="All data",
            s=3,
        )
        axs[1].scatter(
            self.sliding_sigmas_cut,
            self.sliding_deviate_cut,
            marker=".",
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            s=3,
        )
        axs[1].set_xlabel("\u03C3")
        axs[1].set_ylabel("(\u03A9-<\u03A9>)/\u03C3")
        #axs[1].set_xlim([0, maxx1])
        #axs[1].set_ylim([-20, 20])
        axs[1].legend()

        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-scatter_omega_sigma_dsc.png")

    def plot_hist_omega_pre_post_dsc(self):
        """
        Generates and saves a histogram of the (\u03A9-<\u03A9>)/\u03C3 distribution. The plot shows the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_deviate_all).

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        fig.suptitle(
            "Results from "
            + self.baseline_name
            + ": Histogram of (\u03A9-<\u03A9>)/\u03C3 with outlier cuts",
            fontsize=20,
        )

        axs.hist(
            self.sliding_deviate_all,
            bins=101,
            color="r",
            ec="k",
            lw=0.5,
            label="All data")
        axs.hist(
            self.sliding_deviate_cut,
            bins=101,
            color="b",
            ec="k",
            lw=0.5,
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut"
        )
        axs.set_xlabel("(\u03A9-<\u03A9>)/\u03C3")
        axs.set_ylabel("# per bin")
        #axs.set_xlim(-60, 60)
        #axs.set_ylim(1, 1e4)
        axs.legend()
        axs.set_yscale("log")

        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-histogram_omega_dsc.png")

    def plot_KS_test(self):
        """
        Generates and saves a panel plot with results of the Kolmogorov-Smirnov test for Gaussianity. The cumulative distribution of the data (after the delta-sigma (bad GPS times) cut) is compared to the one of Gaussian data, where the bias factor for the sigmas is taken into account. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_deviate_cut).

        Parameters
        ==========

        Returns
        =======

        """
        bias_factor = calc_bias(self.segment_duration, self.deltaF, self.deltaT)
        dof_scale_factor = 1. / (1. + 3. / 35.)
        lx = len(self.sliding_deviate_cut)
        
        sorted_deviates = np.sort(self.sliding_deviate_cut/bias_factor)
        
        count, bins_count = np.histogram(sorted_deviates, bins=500)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        
        normal = stats.norm(0, 1)
        normal_cdf = normal.cdf(bins_count[1:])
        
        dks_x = max(abs(cdf - normal_cdf))
        lx_eff = lx * dof_scale_factor

        lam = (np.sqrt(lx_eff) + 0.12 + 0.11 / np.sqrt(lx_eff)) * dks_x
        pval_KS = StatKS(lam)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 9))
        fig.suptitle(
            "Results from " + self.baseline_name + " for Abs[\u03B4\u03C3]/\u03C3",
            fontsize=16,
        )
        axs[0].plot(bins_count[1:], cdf, "k", label="Data")
        axs[0].plot(
            bins_count[1:],
            normal_cdf,
            "r",
            label="Erf with \u03C3=" + str(round(bias_factor, 2)),
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
        axs[0].legend(loc="lower right")

        axs[1].plot(
            bins_count[1:],
            cdf - normal_cdf,
        )
        axs[1].annotate(
            "Maximum absolute difference: " + str(round(dks_x, 3)),
            xy=(0.025, 0.9),
            xycoords="axes fraction",
        )
        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-KS_test.png")

    def plot_hist_scatter_omega_sigma(self):
        """
        Generates and saves a panel plot with a scatter plot of \u03C3 vs (\u03A9-<\u03A9>)/\u03C3, as well as histograms of the \u03C3 and (\u03A9-<\u03A9>)/\u03C3 distributions. All plots show the data after the delta-sigma (bad GPS times) cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_deviate_cut).

        Parameters
        ==========

        Returns
        =======

        """
        muXY = np.cov(self.sliding_sigmas_cut, self.sliding_deviate_cut)
        binsize = 0.2
        bins = np.arange(-5, 5, binsize)
        count, bins_count = np.histogram(self.sliding_deviate_cut, bins=bins)

        bias_factor = calc_bias(self.segment_duration, self.deltaF, self.deltaT)
        dof_scale_factor = 1.0 / (1.0 + 3.0 / 35.0)
        lx = len(self.sliding_deviate_cut)
        cdf_x = 1.0 / lx * np.linspace(1, lx, lx)

        sorted_deviates_U = np.sort(self.sliding_deviate_cut)
        phi_x_U = 0.5 * (
            1.0 + special.erf(sorted_deviates_U / (np.sqrt(2) * bias_factor))
        )
        dks_x = max(abs(cdf_x - phi_x_U))

        muR = np.mean(self.sliding_deviate_cut)
        sigmaR = bias_factor
        theory = np.exp(-((bins_count - muR + binsize / 2) ** 2) / (2 * sigmaR ** 2))
        tmax = 10 ** np.log10(np.max(count))

        nbins = 200
        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
        (ax1, ax2), (ax3, ax4) = gs.subplots(sharex="col", sharey="row")
        ax1.hist(
            self.sliding_sigmas_cut,
            bins=nbins,
            color="b",
            ec="k",
            lw=0.5,
            orientation="vertical",
        )
        ax1.set_ylabel("Relative Frequency, N")

        ax2.axis("off")

        maxx1 = np.max(self.sliding_sigmas_cut) / 4
        ax3.text(
            1.35 * maxx1 / 3,
            2.75,
            "N = "
            + str(len(self.sliding_sigmas_cut))
            + "\n\u03C3$_{XY}^2$ = "
            + str(f"{muXY[0, 0]:0.2e}")
            + "\n\u03C1 = \u03C3$_{XY}^2$/\u03C3$_{X}$\u03C3$_{Y}$ = "
            + str(f"{muXY[0, 1] / np.sqrt(muXY[0, 0] * muXY[1, 1]):0.2e}")
            + "\n\u03A3 (\u03A9$_i$-<\u03A9>)$^2$/\u03C3$_i$$^2$ = "
            + str(np.floor(np.sum(self.sliding_deviate_cut ** 2))),
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
        )
        ax3.scatter(
            self.sliding_sigmas_cut, self.sliding_deviate_cut, marker=".", s=1, c="b"
        )
        ax3.set_xlabel("\u03C3")
        ax3.set_ylabel("(\u03A9-<\u03A9>)/\u03C3")
        #ax3.set_xlim(0, maxx1)
        #ax3.set_ylim(-6, 6)

        ax4.hist(
            self.sliding_deviate_cut,
            bins=bins,
            color="b",
            ec="k",
            lw=0.5,
            orientation="horizontal",
        )
        ax4.plot(tmax * theory, bins_count, color="r")
        ax4.text(
            10,
            4.4,
            "Fit: $e^{-x^2/(2\u03C3^2)}$; \u03C3 = "
            + str(round(bias_factor, 2))
            + "\nKS-test: "
            + str(round(dks_x, 3)),
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        ax4.set_xlabel("Relative Frequency, N")

        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-histogram_scatter_omega_sigma.png")

    def plot_hist_sigma_squared(self):
        """
        Generates and saves a histogram of \u03C3^2/<\u03C3^2>. The plot shows data after the delta-sigma (bad GPS times) cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_sigmas_cut).

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        fig.suptitle(
            "Results from "
            + self.baseline_name
            + ": Distribution of \u03C3$^2$/<\u03C3$^2$>",
            fontsize=20,
        )

        axs.hist(
            1 / np.mean(self.sliding_sigmas_cut ** 2) * self.sliding_sigmas_cut ** 2,
            bins=300,
            color="b",
            ec="k",
            lw=0.5,
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
        )
        axs.set_xlabel("\u03C3$^2$/<\u03C3$^2$>")
        axs.set_ylabel("# per bin")
        axs.legend()

        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-histogram_sigma_squared.png")

    def plot_omega_time_fit(self):
        """
        Generates and saves a plot of \u03A9 as a function of time and fits the data to perform a linear trend analysis. The plot shows data after the delta-sigma (bad GPS times) cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_omega_cut).

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        fig.suptitle("\u03A9 vs time for each segment", fontsize=20)

        t_obs = self.days_cut[-1]
        scale = np.sqrt(np.var(self.sliding_omega_cut))

        def func(x, a, b):
            return a * (x - t_obs / 2) * t_obs + b

        popt, pcov = curve_fit(
            func, self.days_cut, self.sliding_omega_cut, sigma=self.sliding_sigmas_cut
        )
        c1, c2 = popt[0], popt[1]
        axs.plot(self.days_cut, func(self.days_cut, c1, c2), "r")
        axs.plot(self.days_cut, self.sliding_omega_cut, "b.", markersize=1)
        axs.plot(self.days_cut, 3 * self.sliding_sigmas_cut, "b", linewidth=1.5)
        axs.plot(self.days_cut, -3 * self.sliding_sigmas_cut, "b", linewidth=1.5)
        axs.set_xlabel("Days since start of run")
        axs.set_ylabel("\u03A9$_i$")
        axs.set_xlim(self.days_cut[0], self.days_cut[-1])
        #axs.set_ylim(-6 * scale, 6 * scale)
        axs.annotate("Linear trend analysis: \u03A9(t) = C$_1$*(t-T$_{obs}$/2)*T$_{obs}$ + C$_2$\nC$_1$ = "
            + str(f"{c1:.3e}")
            + "\nC$_2$ = "
            + str(f"{c2:.3e}"), xy=(0.05, 0.05), xycoords='axes fraction',fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=1))
        plt.savefig(f"{self.plot_dir}{self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-omega_time_fit.png")

    def generate_all_plots(self):
        """
        Generates and saves all the statistical analysis plots.

        Parameters
        ==========

        Returns
        =======

        """
        self.plot_running_point_estimate()
        self.plot_running_sigma()
        self.plot_IFFT_point_estimate_integrand()
        self.plot_point_estimate_integrand()
        self.plot_cumulative_point_estimate()
        self.plot_real_point_estimate_integrand()
        self.plot_imaginary_point_estimate_integrand()
        self.plot_sensitivity_integrand()
        self.plot_cumulative_sensitivity_integrand()
        self.plot_omega_sigma_in_time()
        self.plot_hist_sigma_dsc()
        self.plot_scatter_sigma_dsc()
        self.plot_scatter_omega_sigma_dsc()
        self.plot_hist_omega_pre_post_dsc()
        self.plot_KS_test()
        self.plot_hist_scatter_omega_sigma()
        self.plot_hist_sigma_squared()
        self.plot_omega_time_fit()
    
def calc_naive_sigma(freqs, psd1_naive, psd2_naive, orf, deltaF, segment_duration, sampling_frequency, alpha):
    """
    Function to compute the naive sigma starting from the naive power spectral density (PSD) of two interferometers.
    
    Parameters
    ==========
    freqs: array
        Array containing the frequencies.
    psd1_naive: array
        Array containing the naive PSD for interferometer 1. Each entry corresponds to the value oof the naive PSD evaluated at the corresponding frequency in the freqs array.
    psd2_naive: array
        Array containing the naive PSD for interferometer 2. Each entry corresponds to the value oof the naive PSD evaluated at the corresponding frequency in the freqs array.
    orf: array
        Array containing the overlap reduction function. Each entry in this array corresponds to the overlap reduction function evaluated at the corresponding frequency in the freqs array.
    deltaF: float
        Frequency resolution
    segment_duration: int
        Duration of a segment
    sampling_frequency: float
        Sampling frequency
    alpha: float
        Spectral index
    
    Returns
    =======
    naive_sigma_alpha: float
        Naive sigma
    
    """
    Hf = calc_Hf(freqs, alpha)
    deltaT=1./sampling_frequency
        
    window1 = np.hanning(segment_duration * sampling_frequency)
    window2 = window1

    naive_sensitivity_integrand_with_Hf = (
            calc_sens_integrand(
                freqs, psd1_naive, psd2_naive, window1, window2, deltaF, orf, T=deltaT
            )
            / Hf**2
        )

    naive_sigma_alpha = calc_sigma_alpha(
            naive_sensitivity_integrand_with_Hf
        )
    return naive_sigma_alpha