import numpy as np
#from constants import H0
from scipy.interpolate import interp1d
import gwpy

H0 = 1.

import h5py
import sys

if sys.version_info >= (3, 0):
    import configparser
else:
    import ConfigParser as configparser


class initialize(object):
    def __init__(self, param_file, baselines):
        """
        Class that initializes the necessary parameters for the simulation
        of an isotropic stochastic background.

        Parameters
        ==========
        param_file: str
            File containing the various parameter values used to simulate
            an isotropic stochastic background.
        baselines: Baseline object
            List of baseline objects

        Returns
        =======
        """
        self.param_file = param_file
        param = configparser.ConfigParser()
        param.read(param_file)
    
        # [Frequency handling]
        self.NSegments = param.getint("parameters", "NSegments")
        self.Fs = param.getfloat("parameters", "Fs")
        self.TAvg = param.getfloat("parameters", "TAvg")
        
        # [Time handling]
        self.segmentDuration = param.getfloat("parameters", "segmentDuration")
        self.t0 = param.getfloat("parameters", "t0")
        
        # [Derived quantities]
        self.NSamplesPerSegment = int(self.segmentDuration * self.Fs)
        self.N = self.NSegments * self.NSamplesPerSegment
        self.deltaT = 1 / self.Fs
        self.NAvgs = 2 * int(self.segmentDuration / self.TAvg) - 1
        self.jobDuration = self.NSegments * self.segmentDuration
        self.fNyquist = 1 / (2 * self.deltaT)
        self.deltaF = 1 / self.segmentDuration
        self.deltaFStoch = 1 / self.TAvg
        
        # [Signal injection]
        self.fref = param.getfloat("parameters", "fref")
        self.alpha = param.getfloat("parameters", "alpha")
        self.omegaRef = param.getfloat("parameters", "omegaRef")
        
        self.baselines=baselines
        self.Nd = param.getint("parameters", "Nd")

        self.read_noise_file = param.getboolean("parameters", "read_noise_file")
        self.read_orf_file = param.getboolean("parameters", "read_orf_file")

    def make_freqs(self):
        """
        Function that makes an array of frequencies given the sampling rate
        and the segment duration specified in the initial parameter file.

        Parmeters
        =========

        Returns
        =======
        freqs: array_like
            Array of frequencies for which an isotropic stochastic background
            will be simulated.
        """
        if self.NSamplesPerSegment % 2 == 0:
            numFreqs = self.NSamplesPerSegment / 2 - 1
        else:
            numFreqs = (self.NSamplesPerSegment - 1) / 2

        freqs = np.array([self.deltaF * (i + 1) for i in range(int(numFreqs))])
        return freqs

    def make_times(self):
        """
        Function that makes an array of times given thenumber of segments
        and the sampling frequency specified in the initial parameter file.

        Parameters
        ==========

        Returns
        =======
        times: array_like
            Array of times for which an isotropic stochastic background
            will be simulated.
        """
        T = self.N * self.deltaT
        times = np.array([self.t0 + self.deltaT * i for i in range(int(self.N))])
        return times

    @property
    def omegaGW(self):
        """
        Parameters
        ==========

        Returns
        =======
        omegaGW: gwpy.frequencyseries.FrequencySeries
            A gwpy FrequencySeries containing the omegaGW spectrum of the
            isotropic stochastic background that will be simulated.
        """
        freqs = self.make_freqs()
        omegaGW = self.omegaRef * (freqs / self.fref) ** self.alpha
        omegaGW = gwpy.frequencyseries.FrequencySeries(omegaGW, f0=self.deltaF, df=self.deltaF)
        return omegaGW

    @property
    def noise_PSD(self):
        """
        Function that creates the noise PSDs for the detectors that enter in
        the simulation. Two options: either the noise PSDs are read in from
        a file or they can be obtained from the intereferometers in bibly (still
        needs to be implemented).

        Parameters
        ==========

        Returns
        =======
        noisePSD_array: array_like
            Array of size Nd, the number of detectors, containing gwpy FrequencySeries
            of the noisePSDs for each of the detectors.
        """
        if self.read_noise_file:
            param = configparser.ConfigParser()
            param.read(self.param_file)
            paths = param.get("parameters", "noisePSD_file")
            filenames = paths.split(", ")
            noisePSD_array = np.zeros(
                self.Nd, dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries
            )
            for ii in range(len(filenames)):
                noisePSD = self.read_from_file(filenames[ii])
                noisePSD_array[ii] = noisePSD
            return noisePSD_array

        else:
#             noisePSD_array = np.array([baseline.power_spectral_density_array for baseline in self.baselines])
# Think about how to include the PSD only once for each detector. Looping over all baselines would include it more than once
            pass  # Should then get noise PSD from other module
                  # return interferometer.noise_PSD
            
    @property
    def orfs(self):
        """
        Function that creates the noise PSDs for the detectors that enter in
        the simulation. Two options: either the noise PSDs are read in from
        a file or they can be obtained from the Baseline class (still needs to
        be implemented).

        Parameters
        ==========

        Returns
        =======
        orf_array: array_like
            Array containing gwpy FrequencySeries of the orfs for the various
            baseline combinations that enter in the simulation.
        """
        if self.read_orf_file:
            param = configparser.ConfigParser()
            param.read(self.param_file)
            paths = param.get("parameters", "orf_file")
            filenames = paths.split(", ")
            orf_array = np.zeros(
                len(filenames),
                dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries,
            )
            for ii in range(len(filenames)):
                orf = self.read_from_file(filenames[ii])
                orf_array[ii] = orf
            return orf_array

        else:
            orf = np.array([baseline.overlap_reduction_function for baseline in self.baselines])
            orf_array=np.zeros(orf.shape[0],dtype=gwpy.frequencyseries.FrequencySeries)
            for ii in range(orf.shape[0]):
                orf_array[ii] = gwpy.frequencyseries.FrequencySeries(orf[ii], f0=self.deltaF, df=self.deltaF)
            return orf_array

    def read_from_file(self, filename):
        """
        Function that reads from file to produce frequency series of a
        quantity of interest (e.g. noisePSD, orf, ...).

        Parameters
        ==========
        filename: str
            File containing the frequency and corresponding quantity of
            interest (e.g. noisePSD, orf, ...).

        Return
        ======
        freq_series: gwpy.frequencyseries.FrequencySeries
            A gwpy FrequencySeries containing the quantity of interest
            (e.g. noisePSD, orf, ...).
        """
        content = [i.strip().split() for i in open(filename).readlines()]

        freqs = [float(content[i][0]) for i in range(len(content))]
        data = [float(content[i][1]) for i in range(len(content))]

        func = interp1d(freqs, data, kind="cubic", fill_value="extrapolate")
        f = self.make_freqs()
        freq_series = gwpy.frequencyseries.FrequencySeries(func(f), f0=self.deltaF, df=self.deltaF)
        return freq_series


class simulation_GWB(object):
    def __init__(self, noisePSD, omegaGW, orf, Fs, segmentDuration, NSegments, t0):
        """
        Class that simulates an isotropic stochastic background.

        Parameters
        ==========

        Returns
        =======
        """
        # [detector attributes]
        self.noisePSD = noisePSD
        self.OmegaGW = omegaGW
        self.orf = orf

        # [frequency handling]
        self.Fs = Fs
        self.freqs = omegaGW.frequencies.value
        self.Nf = omegaGW.size
        self.Nd = noisePSD.shape[0]
        self.deltaF = omegaGW.df.value
        
        # [time handling]
        self.segmentDuration = segmentDuration
        self.NSegments = NSegments
        self.t0 = t0
        self.NSamplesPerSegment = int(self.Fs * self.segmentDuration)
        self.deltaT = 1 / self.Fs

        self.gen_data = self.generate_data()
        print(self.gen_data)
#         self.save_data_to_npz()

    @classmethod
    def from_ini_file(cls, baselines, ini_file):
        """
        Initializes the class from an ini_file for a given list of baselines.
        
        Parameters
        ==========
        baselines: Baseline object
            List of baseline objects.
        ini_file: str
            File containing the various parameter values used to simulate
            an isotropic stochastic background. 
        Returns
        =======
        """
        params_ini = initialize(ini_file, baselines)
        return cls(
            params_ini.noise_PSD,
            params_ini.omegaGW,
            params_ini.orfs,
            params_ini.Fs,
            params_ini.segmentDuration,
            params_ini.NSegments,
            params_ini.t0
        )

    def save_data_to_npz(self):
        """
        """
        np.savez('data.npz', data = self.gen_data)

#     def save_data_to_hdf5(self):
#         """
#         """
#         timeseries_data = gwpy.timeseries.Timeseries.self.gen_data
#         timeseries_data.t0 = my_t0
#         Timeseries.write_to_h5(self.gen_data)

    def generate_data(self):
        """
        Function that simulates an isotropic stochastic background given the
        input parameters. The data is simulated and spliced together to prevent
        periodicity artifacts related to IFFTs.

        Parameters
        ==========

        Returns
        =======
        data: array_like
            An array of size Nd (number of detectors) with gwpy TimeSeries with the
            data containing the simulated isotropic stochastic background.
        """
        y = self.simulate_data()
        datatemp = self.splice_segments(y)
#         data = np.zeros(
#             self.Nd, dtype=gwpy.timeseries.TimeSeries
#         )
       
#         for ii in range(self.Nd):
#             data[ii] = gwpy.timeseries.TimeSeries(datatemp[ii], t0=self.t0, dt=self.deltaT)
            
        return datatemp

    def orfToArray(self):
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
            dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries,
        )
        for ii in range(self.Nd):
            for jj in range(ii):
                orf_array[ii, jj] = self.orf[index]
                index += 1
        orf_array = orf_array + orf_array.transpose()
        return orf_array

    def omegaToPower(self):
        """
        Function that computes the GW power spectrum starting from the OmegaGW
        spectrum.

        Parameters
        ==========

        Returns
        =======
        power: gwpy.frequencyseries.FrequencySeries
            A gwpy FrequencySeries conatining the GW power spectrum
        """
        H_theor = (3 * H0 ** 2) / (10 * np.pi ** 2)

        power = H_theor * self.OmegaGW.value * self.freqs ** (-3)
        power = gwpy.frequencyseries.FrequencySeries(power, f0= self.deltaF, df=self.deltaF)
        return power

    def covariance_matrix(self):
        """
        Function to compute the covariance matrix corresponding to a stochastic
        background in the various detectors.

        Parameters
        ==========

        Returns
        =======
        C: array_like
            Covariance matrix corresponding to a stochastic background in the
            various detectors. Dimensions are Nd x Nd x Nf, where Nd is the
            number of detectors and Nf the number of frequencies.
        """
        GWBPower = self.omegaToPower()
        orf_array = self.orfToArray()

        C = np.zeros((self.Nd, self.Nd, self.Nf))

        for ii in range(self.Nd):
            for jj in range(self.Nd):
                if ii == jj:
                    C[ii, jj, :] = self.noisePSD[ii].value[:] + GWBPower.value[:]
                else:
                    C[ii, jj, :] = orf_array[ii, jj].value[:] * GWBPower.value[:]

        C = self.NSamplesPerSegment / (self.deltaT * 4) * C
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

        A = np.einsum("...ij,jk...", np.sqrt(eigval), eigvec.transpose())
        x = np.einsum("...j,...jk", z, A)
        return x

    def simulate_data(self):
        """
        Function that simulates the data corresponding to an isotropic stochastic
        background.

        Parameters
        ==========

        Returns
        =======
        y: array_like
            Array of size Nd x 2*(NSegments+1) x NSamplesPerSegment containing the
            various segments with the simulated data.
        """
        C = self.covariance_matrix()

        y = np.zeros(
            (self.Nd, 2 * self.NSegments + 1, self.NSamplesPerSegment), dtype=np.ndarray
        )

        for kk in range(2 * self.NSegments + 1):
            z = self.generate_freq_domain_data()

            xtemp = self.transform_to_correlated_data(z, C)

            for ii in range(self.Nd):
                if self.NSamplesPerSegment % 2 == 0:
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
                y[ii, kk, :] = np.real(np.fft.ifft(xtilde))
        return y

    def splice_segments(self, segments):
        """
        This function splices together the various segments to prevent
        artifacts related to the periodicity that can arise from inverse
        Fourier transforms.

        Parameters
        ==========
        segments: array_like
            Array of size Nd x (2*NSegments+1) x NSamplesPerSegment containing the
            various segments with the simulated data that need to be spliced
            together, where Nd is the number of detectors.

        Returns
        =======
        data: array_like
            Array of size Nd x (NSegments*NSamplesPerSegment) containing the simulated
            data corresponding to an isotropic stochastic background for each of the
            detectors, where Nd is the number of detectors.
        """
        w = np.zeros(self.NSamplesPerSegment)

        for ii in range(self.NSamplesPerSegment):
            w[ii] = np.sin(np.pi * ii / self.NSamplesPerSegment)

        data = np.zeros(
            (self.Nd, self.NSamplesPerSegment * self.NSegments), dtype=np.ndarray
        )

        for ii in range(self.Nd):
            for jj in range(self.NSegments):
                y0 = w * segments[ii][2 * jj][:]
                y1 = w * segments[ii][2 * jj + 1][:]
                y2 = w * segments[ii][2 * jj + 2][:]

                z0 = np.concatenate(
                    (
                        y0[int(self.NSamplesPerSegment / 2) : self.NSamplesPerSegment],
                        np.zeros(int(self.NSamplesPerSegment / 2)),
                    )
                )
                z1 = y1[:]
                z2 = np.concatenate(
                    (
                        np.zeros(int(self.NSamplesPerSegment / 2)),
                        y2[0 : int(self.NSamplesPerSegment / 2)],
                    )
                )

                data[
                    ii,
                    jj * self.NSamplesPerSegment : (jj + 1) * self.NSamplesPerSegment,
                ] = (
                    z0 + z1 + z2
                )
            
        return data
