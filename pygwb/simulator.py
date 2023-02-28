import bilby
import gwpy
import numpy as np
from bilby.core.utils import create_frequency_series
from gwpy.timeseries import TimeSeries
from loguru import logger
from tqdm import tqdm

from pygwb.baseline import Baseline, get_baselines
from pygwb.util import interpolate_frequency_series


class Simulator(object):
    def __init__(
        self,
        interferometers,
        N_segments,
        duration,
        sampling_frequency,
        intensity_GW=None,
        injection_dict=None,
        start_time=0.0,
        no_noise=False,
        orf_polarization="tensor",
        seed=None,
        continuous=False,
    ):
        """
        Class that simulates an isotropic stochastic background.

        Parameters
        ==========
        interferometers: list of bilby interferometer objects
        intensity_GW: gwpy.frequencyseries.FrequencySeries
            A gwpy.frequencyseries.FrequencySeries containing the desired strain power spectrum
            which needs to be simulated. Note: A range of spectral indices (from -3 to 3) was
            tested. However, one should be careful for spectral indices outside of this range,
            as the splicing procedure implemented in this module is known to introduce a bias for
            some values of the spectral index (usually large negative numbers).
        injection_dict: dict
            Dictionary containing the parameters of CBC injections.
        N_segments: int
            Number of segments that needs to be generated for the simulation
        duration: float
            Duration of a simulated data segment
        sampling_frequency: float
            Sampling frequency
        start_time: float
            Start time of the simulation
        no_noise: boolean
            Flag that sets the noise_PSDs to 0
        seed: int
            Seed to use for random data generation
        continuous: boolean
            Flag to generate continuous segments. If True, a seed must also be provided 

        Returns
        =======
        """
        if len(interferometers) < 2:
            raise ValueError("Number of interferometers should be at least 2")
        else:
            self.interferometers = interferometers
            self.Nd = len(self.interferometers)

            self.sampling_frequency = sampling_frequency
            self.duration = duration

            for ifo in interferometers:
                ifo.sampling_frequency = self.sampling_frequency
                ifo.duration = self.duration

            self.N_segments = N_segments
            self.frequencies = self.get_frequencies()

            self.Nf = len(self.frequencies)
            self.t0 = start_time
            self.N_samples_per_segment = int(self.sampling_frequency * self.duration)
            self.deltaT = 1 / self.sampling_frequency

            self.noise_PSD_array = self.get_noise_PSD_array()

            self.no_noise = no_noise

            self.seed = seed
            self.continuous = continuous
            if (self.continuous == True) and not self.seed:
                raise ValueError("Must provide a seed to generate continuous segments")

            if no_noise == True:
                self.noise_PSD_array = np.zeros_like(self.noise_PSD_array)

            self.baselines = get_baselines(
                self.interferometers, frequencies=self.frequencies
            )
            for baseline in self.baselines:
                if not baseline._orf_polarization_set:
                    baseline.orf_polarization = "tensor"
            self.orf = self.get_orf(polarization=orf_polarization)

            self.intensity_GW = intensity_GW
            if self.intensity_GW:
                self.intensity_GW = interpolate_frequency_series(
                    intensity_GW, self.frequencies
                )
            self.injection_dict = injection_dict

            if not self.intensity_GW and not self.injection_dict:
                raise ValueError('Must provide at least one of intensity_GW or self.injection_dict')

    def get_frequencies(self):
        """
        Computes an array of frequencies with given sampling frequency and duration.

        Parameters
        ==========

        Returns
        =======
        frequencies: array_like
            Array containing the computed frequencies
        """
        frequencies = create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        return frequencies

    def get_noise_PSD_array(self):
        """
        Function that gets the noise PSD array of all the interferometers.

        Parameters
        ==========

        Returns
        =======
        noise_PSDs_array: array_like
            Array containing the noise PSD arrays for all interferometers in
            self.interferometers.
        """
        noise_PSDs = []
        try:
            for ifo in self.interferometers:
                psd_temp = ifo.power_spectral_density.psd_array
                freqs_temp = ifo.power_spectral_density.frequency_array

                if np.isinf(psd_temp).any() == True:
                    raise ValueError(
                        f"The noisePSD of interferometer {ifo.name} contains infs!"
                    )
                psd = gwpy.frequencyseries.FrequencySeries(
                    psd_temp, frequencies=freqs_temp
                )
                psd_interpolated = interpolate_frequency_series(psd, self.frequencies)

                noise_PSDs.append(psd_interpolated.value[:])

                noise_PSDs_array = np.array(noise_PSDs)
            return noise_PSDs_array

        except:
            raise AttributeError(
                "The noisePSD of all the detectors needs to be specified!"
            )

    def get_orf(self, polarization="tensor"):
        """
        Function that returns a list containing the overlap reduction functions
        for all the baselines in self.baselines.

        Parameters
        ==========

        Returns
        =======
        orf_list: list
            List containing the overlap reduction functions for all the baselines
            in self.baselines.
        """
        orf_list = []
        for base in self.baselines:
            base.orf_polarization = polarization
        orfs = np.array(
            [baseline.overlap_reduction_function for baseline in self.baselines]
        )
        for orf in orfs:
            if orf.shape[0] != self.frequencies.shape[0]:
                orf = orf[1:]
            orf_list.append(orf)
        return orf_list

    def get_data_for_interferometers(self):
        """
        Get a data dictionary for interferometers

        Parameters
        ==========

        Returns
        =======
        interferometer_data: dict
            A dictionary with the simulated data for the interferometers.
        """
        data = self.generate_data(data_start=self.t0)
        interferometer_data = {}
        for idx, ifo in enumerate(self.interferometers):
            interferometer_data[ifo.name] = data[idx]
        if self.continuous:
            pad_len = int(self.N_segments * self.N_samples_per_segment/2)
            total_dur = self.N_segments*self.duration
            early_data = self.generate_data(data_start=self.t0-total_dur/2)
            late_data = self.generate_data(data_start=self.t0+total_dur/2)
            for idx, ifo in enumerate(self.interferometers):
                ed = early_data[idx] * TimeSeries(np.sin((early_data[idx].times.value-early_data[idx].t0.value
                                                   )/total_dur*np.pi), times=early_data[idx].times)
                ld = late_data[idx] * TimeSeries(np.sin((late_data[idx].times.value-late_data[idx].t0.value
                                                  )/total_dur*np.pi), times=late_data[idx].times)
                cd = interferometer_data[ifo.name] * TimeSeries(np.sin((interferometer_data[ifo.name].times.value-interferometer_data[ifo.name].t0.value
                                                            )/total_dur*np.pi), times=interferometer_data[ifo.name].times)
                interferometer_data[ifo.name] = \
                               ed.pad((0, 2*pad_len)) + \
                               cd.pad(pad_len) + \
                               ld.pad((2*pad_len, 0))
                interferometer_data[ifo.name] = \
                    interferometer_data[ifo.name].crop(self.t0,
                                              self.t0+total_dur)
                interferometer_data[ifo.name] = TimeSeries(interferometer_data[ifo.name].value, 
                                                           t0=self.t0, sample_rate=self.sampling_frequency)
        return interferometer_data

    def generate_data(self, data_start=None):
        """
        Function that simulates an isotropic stochastic background given the
        input parameters. The data is simulated and spliced together to prevent
        periodicity artifacts related to IFFTs.

        Parameters
        ==========
        data_start: float
            Start time of the data that is being generated.

        Returns
        =======
        data: array_like
            An array of size Nd (number of detectors) with gwpy TimeSeries with the
            data containing the simulated isotropic stochastic background.
        """
        if not data_start:
            data_start = self.t0
        data_signal_temp = np.zeros(
            (self.Nd, self.N_samples_per_segment * self.N_segments), dtype=np.ndarray
        )
        if self.intensity_GW:
            y_signal = self.simulate("signal")
            if self.seed:
                # seed is based on start time of segment
                np.random.seed(int(self.seed+(data_start%1000)))
            data_spliced = self.splice_segments(y_signal)
            for ii in range(self.Nd):
                data_signal_temp[ii] = data_signal_temp[ii] + data_spliced[ii]
        if self.injection_dict:
            data_cbc = self.inject_CBC()
            for ii in range(self.Nd):
                data_signal_temp[ii] = data_signal_temp[ii] + data_cbc[ii]
        if not self.no_noise:
            y_noise = self.simulate("noise")
            data_noise_temp = self.splice_segments(y_noise)
        else:
            data_noise_temp = np.zeros_like(data_signal_temp)
        data = np.zeros(self.Nd, dtype=gwpy.timeseries.TimeSeries)
        for ii in range(self.Nd):
            logger.info(
                f"Adding data to channel {self.interferometers[ii].name}:SIM-STOCH_INJ"
            )
            data[ii] = gwpy.timeseries.TimeSeries(
                (data_signal_temp[ii] + data_noise_temp[ii]).astype("float64"),
                t0=data_start,
                dt=self.deltaT,
                channel=f"{self.interferometers[ii].name}:SIM-STOCH_INJ",
                name=f"{self.interferometers[ii].name}:SIM-STOCH_INJ",
            )
        return data

    def orf_to_array(self):
        """
        Function that converts the list of overlap reduction functions into an array
        to facilitate the correct implementation when computing the covariance matrix.

        Parameters
        ==========

        Returns
        =======
        orf_array: array_like
            Array of shape Nd x Nd containing the orfs, where Nd is the number of
            detectors. The convention used for consistency with the remainder of the
            simulation is as follows. Orfs are only present in the off-diagonal slots
            in the array. Only the part below the diagonal is filled, after which this
            is copied to the upper part by transposing and summing. The array is filled
            by starting from the first free slot below the diagonal, from left to right,
            until the diagonal is reached, after which the line below in the array is
            filled analogously and so on.
        """
        index = 0
        orf_array = np.zeros(
            (self.Nd, self.Nd),
            dtype=gwpy.frequencyseries.FrequencySeries,
        )
        for ii in range(self.Nd):
            for jj in range(ii):
                orf_array[ii, jj] = self.orf[index]
                index += 1
        orf_array = orf_array + orf_array.transpose()

        for ii in range(self.Nd):
            baseline_name = (
                f"{self.interferometers[ii].name} - {self.interferometers[ii].name}"
            )
            baseline_temp = Baseline(
                baseline_name,
                self.interferometers[ii],
                self.interferometers[ii],
                frequencies=self.frequencies,
            )
            if not baseline_temp._orf_polarization_set:
                baseline_temp.orf_polarization = "tensor"
            orf_temp = baseline_temp.overlap_reduction_function

            if orf_temp.shape[0] != self.frequencies.shape[0]:
                orf_temp = orf_temp[1:]
            orf_array[ii, ii] = orf_temp

        return orf_array

    def covariance_matrix(self, flag):
        """
        Function to compute the covariance matrix corresponding to a stochastic
        background in the various detectors.

        Parameters
        ==========
        flaf: str
            Either flagged as "noise" or "signal", to allow for different generation of
            covariance matrix in both cases.

        Returns
        =======
        C: array_like
            Covariance matrix corresponding to a stochastic background in the
            various detectors. Dimensions are Nd x Nd x Nf, where Nd is the
            number of detectors and Nf the number of frequencies.
        """
        orf_array = self.orf_to_array()

        C = np.zeros((self.Nd, self.Nd, self.Nf))
        if flag == "noise":
            for ii in range(self.Nd):
                for jj in range(self.Nd):
                    if ii == jj:
                        C[ii, jj, :] = self.noise_PSD_array[ii]
        elif flag == "signal":
            for ii in range(self.Nd):
                for jj in range(self.Nd):
                    C[ii, jj, :] = orf_array[ii, jj] * self.intensity_GW.value[:]

        C[C == 0.0] = 1.0e-60

        C = self.N_samples_per_segment / (self.deltaT * 4) * C
        return C

    def compute_eigval_eigvec(self, C):
        """
        Function to compute the eigenvalues and eigenvectors of the covariance
        matrix corresponding to a stochastic background in the various detectors.

        Parameters
        ==========
        C: array_like
            Covariance matrix corresponding to a stochastic background in the
            various detectors. Dimensions are Nd x Nd x Nf, where Nd is the
            number of detectors and Nf the number of frequencies.

        Returns
        =======
        eigval: array_like
            Array of diagonal matrices containing the eigenvalues of the
            covariance matrix C.
        eigvec: array_like
            Array of matrices containing the eigenvectors of the covariance
            matrix C.
        """
        eigval, eigvec = np.linalg.eig(C.transpose((2, 0, 1)))
        eigval = np.array([np.diag(x) for x in eigval])
        return eigval, eigvec

    def generate_freq_domain_data(self):
        """
        Function that generates the uncorrelated frequency domain data with
        random phases for the stochastic background.

        Parameters
        ==========

        Returns
        =======
        z: array_like
            Array of size Nf x Nd containing uncorrelated frequency domain data.
        """
        z = np.zeros((self.Nf, self.Nd), dtype="complex_")
        re = np.random.randn(self.Nf, self.Nd)
        im = np.random.randn(self.Nf, self.Nd)
        z = re + im * 1j
        return z

    def transform_to_correlated_data(self, z, C):
        """
        Function that transforms the uncorrelated stochastic background
        simulated data, to correlated data.

        Parameters
        ==========
        z: array_like
            Array containing the uncorrelated data with random phase.
        C: array_like
            Array of size Nd x Nd x Nf representing the covariance matrices
            between detectors for a desired stochastic background, where Nd
            is the number of detectors and Nf is the number of frequencies.

        Returns
        =======
        x: array_like
            Array of size Nf x Nd, containing the correlated stochastic
            background data.
        """
        eigval, eigvec = self.compute_eigval_eigvec(C)

        A = np.einsum("...ij,jk...", np.sqrt(eigval + 0j), eigvec.transpose())
        x = np.einsum("...j,...jk", z, A)
        return x

    def simulate(self, flag):
        """
        Function that simulates the data corresponding to an isotropic stochastic
        background.

        Parameters
        ==========

        Returns
        =======
        y: array_like
            Array of size Nd x 2*(N_segments+1) x N_samples_per_segment containing the
            various segments with the simulated data.
        """
        C = self.covariance_matrix(flag)

        y = np.zeros(
            (self.Nd, 2 * self.N_segments + 1, self.N_samples_per_segment),
            dtype=np.ndarray,
        )

        for kk in range(2 * self.N_segments + 1):
            z = self.generate_freq_domain_data()

            xtemp = self.transform_to_correlated_data(z, C)

            for ii in range(self.Nd):
                if self.N_samples_per_segment % 2 == 0:
                    xtilde = np.concatenate(
                        (
                            np.array([0]),
                            xtemp[:, ii],
                            np.array([0]),
                            np.flipud(np.conjugate(xtemp[:, ii])),
                        )
                    )
                else:
                    xtilde = np.concatenate(
                        (
                            np.array([0]),
                            xtemp[:, ii],
                            np.flipud(np.conjugate(xtemp[:, ii])),
                        )
                    )
                y[ii, kk, :] = np.real(np.fft.ifft(xtilde))[
                    : self.N_samples_per_segment
                ]
        return y

    def splice_segments(self, segments):
        """
        This function splices together the various segments to prevent
        artifacts related to the periodicity that can arise from inverse
        Fourier transforms. Note: A range of spectral indices (from -3 to 3) was
        tested for the GW power spectrum to inject. However, one should be careful
        for spectral indices outside of this range, as the splicing procedure
        implemented here is known to introduce a bias for some values of the spectral
        index (usually large negative numbers).

        Parameters
        ==========
        segments: array_like
            Array of size Nd x (2*N_segments+1) x N_samples_per_segment containing the
            various segments with the simulated data that need to be spliced
            together, where Nd is the number of detectors.

        Returns
        =======
        data: array_like
            Array of size Nd x (N_segments*N_samples_per_segment) containing the simulated
            data corresponding to an isotropic stochastic background for each of the
            detectors, where Nd is the number of detectors.
        """
        w = np.zeros(self.N_samples_per_segment)

        for ii in range(self.N_samples_per_segment):
            w[ii] = np.sin(np.pi * ii / self.N_samples_per_segment)

        data = np.zeros(
            (self.Nd, self.N_samples_per_segment * self.N_segments), dtype=np.ndarray
        )

        for ii in range(self.Nd):
            for jj in range(self.N_segments):
                y0 = w * segments[ii][2 * jj][:]
                y1 = w * segments[ii][2 * jj + 1][:]
                y2 = w * segments[ii][2 * jj + 2][:]

                z0 = np.concatenate(
                    (
                        y0[
                            int(
                                self.N_samples_per_segment / 2
                            ) : self.N_samples_per_segment
                        ],
                        np.zeros(int(self.N_samples_per_segment / 2)),
                    )
                )
                z1 = y1[:]
                z2 = np.concatenate(
                    (
                        np.zeros(int(self.N_samples_per_segment / 2)),
                        y2[0 : int(self.N_samples_per_segment / 2)],
                    )
                )

                data[
                    ii,
                    jj
                    * self.N_samples_per_segment : (jj + 1)
                    * self.N_samples_per_segment,
                ] = (
                    z0 + z1 + z2
                )

        return data

    def inject_CBC(self):
        """
        This function uses the provided dictionary of injection parameters to 
        simulate a background of CBC sources. 

        Parameters
        ==========

        Returns
        =======
        data: array_like
            Array of size Nd x (N_segments*N_samples_per_segment) containing the simulated
            data corresponding to a background of CBC sources for each of the
            detectors, where Nd is the number of detectors.
        """

        data = np.zeros(
            (self.Nd, self.N_samples_per_segment * self.N_segments), dtype=np.ndarray
        )
        empty_ts = gwpy.timeseries.TimeSeries(np.zeros(self.N_samples_per_segment * self.N_segments),
                                              t0=self.t0, sample_rate = self.sampling_frequency)
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=self.N_segments*self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={
                "waveform_approximant": "IMRPhenomPv2",
                "reference_frequency": 50,
            },
        )
        for ii in range(self.Nd):
            det = bilby.gw.detector.get_empty_interferometer(self.interferometers[ii].name) 
            det.strain_data.set_from_gwpy_timeseries(empty_ts) 
            det.minimum_frequency=10.

            for n in tqdm(range(len(self.injection_dict['geocent_time']))):
                inj_params = {}
                for k in self.injection_dict.keys():
                    inj_params[k] = self.injection_dict[k][n]
            
                det.inject_signal(
                    waveform_generator=waveform_generator, parameters=inj_params
                )
            data[ii] = det.strain_data.to_gwpy_timeseries().value

        return data
