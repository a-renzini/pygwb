import numpy as np
from scipy.interpolate import interp1d
import gwpy
import h5py
import sys

from .constants import h0
from .util import omegaToPower, interpolate_frequencySeries

if sys.version_info >= (3, 0):
    import configparser
else:
    import ConfigParser as configparser

"""
TODOS
--------

urgent
-------
* convert the output of simulate_data into a gwpy timeseries
* everything is initiated via baseline_1; how to change this?
* check the PSD import in get_NoisePSD: correct format? correct import?

less urgent
-----------
* I would change the way OmegaGW is in injected, as right now there is a risk of mismatch between frequencies in the detectors and in the OmegaGW file which is being read for the injection. We should probably discuss this.
* write all the necessary import checks

"""


class Simulator(object):
    def __init__(
        baselines, omegaGW, Nsegments, save_to_file=False
    ):  # (self, noisePSD, omegaGW, orf, sampling_frequency, duration, NSegments):
        """
        Class that simulates an isotropic stochastic background.

        Parameters
        ==========
        baselines: list of baselines #make into dictionary
        omegaGW:

        Returns
        =======
        """

        self.baselines = baselines
        # [detector attributes]
        self.noisePSD = self.set_noisePSD
        self.orf = np.array(
            [baseline.overlap_reduction_function for baseline in baselines]
        )

        # [time/frequency handling]
        baseline_1 = self.baselines[0]
        self.sampling_frequency = (
            baseline_1.sampling_frequency
        )  # inherited from baselines/interferometer objects
        self.duration = (
            baseline_1.duration
        )  # inherited from baselines/interferometer objects
        self.NSegments = NSegments
        self.frequencies = baseline_1.frequencies
        self.Nf = len(self.frequencies)
        # self.t0 = t0
        
        self.Nd = int(1 + np.sqrt(1 + 8 * len(self.baselines)))

        self.NSamplesPerSegment = int(self.sampling_frequency * self.duration)
        self.deltaT = 1 / self.sampling_frequency

        self.OmegaGW = interpolate_frequencySeries(omegaGW, frequencies)

        self.gen_data = self.generate_data()

        # if save_to_file = True: ...

    @classmethod
    def from_ini_file(cls, baselines, ini_file):
        params_ini = initialize(ini_file)
        orfs = np.array([baseline.overlap_reduction_function for baseline in baselines])
        return cls(
            params_ini.noise_PSD,
            params_ini.omegaGW,
            orfs,
            params_ini.sampling_frequency,
            params_ini.duration,
            params_ini.NSegments,
        )

    def write(self):
        """
        TODO: develop write method; make decisions here
        """

        def save_data_to_npz(self):
            """ """
            np.savez("data.npz", data=self.gen_data)

        def save_data_h5(self):
            """ """
            timeseries_data = gwpy.timeseries.Timeseries.self.gen_data
            timeseries_data.t0 = my_t0
            Timeseries.write_to_h5(self.gen_data)

    def set_noisePSD(self):
        """
        TODO: implement check to see whether ifos have PSDs set; else give error
        """
        noisePSD = []
        for baseline in baselines:
            PSD_1 = baseline.interferometer_1.power_spectral_density_array
            PSD_2 = baseline.interferometer_2.power_spectral_density_array
            noisePSD.append([PSD_1, PSD_2])
        return noise_PSD

    def timeseries_data(self):
        return self.gen_data
    
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
        data = self.splice_segments(y)
        #         data = gwpy.timeseries.TimeSeries(data, times=t)
        return data

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
        GWBPower = omegaToPower(self.OmegaGW, self.frequencies)
        orf_array = self.orfToArray()

        C = np.zeros((self.Nd, self.Nd, self.Nf))

        for ii in range(self.Nd):
            for jj in range(self.Nd):
                if ii == jj:
                    C[ii, jj, :] = self.noisePSD[ii].value[:] + GWBPower.value[:]
                else:
                    C[ii, jj, :] = orf_array[ii, jj] * GWBPower.value[:]

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
        r
