import numpy as np
from constants import H0
from scipy.interpolate import interp1d
import gwpy

# Add dimension of arrays in the explanation?

class simulation_GWB(object):
    def __init__(self, noisePSD, OmegaGW, orf, Fs=None, segmentDuration=None, NSegments=1):
        """
        Parameters
        ==========
        
        """
        self.noisePSD = noisePSD
        self.OmegaGW = OmegaGW
        self.orf = orf
        self.Fs = Fs
        self.segmentDuration = segmentDuration
        self.NSegments = NSegments
        
        self.freqs = OmegaGW.frequencies.value
        self.Nf = OmegaGW.size
        self.Nd = noisePSD.shape[0]
        self.deltaF = OmegaGW.df.value
    
        if Fs==None:
            self.Fs=(self.freqs[-1]+self.deltaF)*2 #Assumes freq is power of two
        
        if segmentDuration==None:
            self.segmentDuration=1/(self.deltaF)

#     OmegaGW_new, noisePSD_new, orf_new = pre_processing(OmegaGW,noisePSD,orf,Fs,segmentDuration)
#     freqs_new = OmegaGW_new.frequencies.value
    
        self.NSamplesPerSegment = int(self.Fs*self.segmentDuration)
        self.deltaT = 1/self.Fs
    
    def generate_data(self):
        """
        
        Parameters
        ==========
        
        Returns
        =======
        """
        y = self.simulate_data()
        data = self.splice_segments(y)
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
            
        """
        index = 0
        orf_array = np.zeros((self.Nd,self.Nd), dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries)
        for ii in range(self.Nd):
            for jj in range(ii):
                orf_array[ii,jj] = self.orf[index]
                index += 1
        orf_array=orf_array+orf_array.transpose()
        return orf_array
    
    def omegaToPower(self):
        """
        Function that computes the GW power spectrum starting from the OmegaGW 
        spectrum.
        
        Parameters
        ==========
        omegaGW: gwpy.frequencyseries.FrequencySeries
            A gwpy FrequencySeries containing the Omega spectrum
    
        Returns
        =======
        power: gwpy.frequencyseries.FrequencySeries
            A gwpy FrequencySeries conatining the GW power spectrum
        """
        
        H_theor = (3*H0**2)/(10*np.pi**2)
        
        power = H_theor*self.OmegaGW.value*self.freqs**(-3)
        power = gwpy.frequencyseries.FrequencySeries(power, frequencies=self.freqs)
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
                if ii==jj:
                    C[ii,jj,:] = self.noisePSD[ii].value[:]+GWBPower.value[:]
                else:
                    C[ii,jj,:] = orf_array[ii,jj].value[:]*GWBPower.value[:]
                    
        C = self.NSamplesPerSegment/(self.deltaT*4)*C  
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
            
        eigvec: array_like
        """
        
        eigval, eigvec = np.linalg.eig(C.transpose((2,0,1)))
        eigval = np.array([np.diag(x) for x in eigval])
        
        return eigval, eigvec
    
    def generate_freq_domain_data(self):
        """
        Function that generates the uncorrelated frequency domain data for the
        stochastic background of gravitational waves.
        
        Parameters
        ==========
                
        Returns
        =======
        z: array_like
        """
        z = np.zeros((self.Nf, self.Nd), dtype = 'complex_')
        re = np.random.randn(self.Nf, self.Nd)
        im = np.random.randn(self.Nf, self.Nd)
        z = re + im*1j
        return z
    
    def transform_to_correlated_data(self, z, eigval, eigvec):
        """
        Function that transforms the uncorrelated stochastic background 
        simulated data, to correlated data.
        
        Parameters
        ==========
        z: array_like
            Array containing the uncorrelated data.
        eigval: array_like
        eigvec: array_like
        Returns
        =======
        x: array_like
        """
        
        A = np.einsum('...ij,jk...',np.sqrt(eigval),eigvec.transpose())
        x = np.einsum('...j,...jk',z,A)
            
        return x
    
    def simulate_data(self):
        """
        Function that simulates a stochastic background 
        
        Parameters
        ==========
        noisePSD: array_like
            Array of gwpy FrequencySeries containing the noise power spectral
            density of the various detectors. Array of length Nd, the number
            of detectors.
        OmegaGW: gwpy.frequencyseries.FrequencySeries
            The Omega spectrum of the stochastic background of gravitational
            waves to be simulated.
        orf: 
        
        Returns
        =======
        y
        
        """

        C = self.covariance_matrix()
        eigval, eigvec = self.compute_eigval_eigvec(C)
        
        #Generate three time segments that will be spliced together to prevent periodicity of ifft
        y = np.zeros((self.Nd, 2*self.NSegments+1, self.NSamplesPerSegment), dtype=np.ndarray)
    
        for kk in range(2*self.NSegments+1):
            z = self.generate_freq_domain_data()
            
            xtemp = self.transform_to_correlated_data(z, eigval, eigvec)
        
            for ii in range(self.Nd):
                # Set DC and Nyquist = 0, then add negative freq parts in proper order
                if self.NSamplesPerSegment%2==0:
                    # Note that most negative frequency is -f_Nyquist when N=even
                    xtilde = np.concatenate((np.array([0]), xtemp[:,ii],np.array([0]), np.flipud(np.conjugate(xtemp[:,ii]))))
                else:
                    print('inside else statement')
                    # No Nyquist frequency when N=odd
                    xtilde = np.concatenate((np.array([0]), xtemp[:,ii], np.flipud(np.conjugate(xtemp[:,ii]))))
                # Fourier transform back to time domain and take real part (imag part = 0)
                y[ii,kk,:] = np.real(np.fft.ifft(xtilde))
        return y
    
    def splice_segments(self, segments):
        """
        This function splices together the various segments to prevent 
        artifacts related to the periodicity that can arise from inverse
        Fourier transforms.
        
        Parameters
        ==========
        segments: array_like
            Nd x 2*NSegments+1 x NSamplesPerSegment
        Returns
        =======
        data:
        """
        
        w = np.zeros(self.NSamplesPerSegment)
        
        for ii in range(self.NSamplesPerSegment):
            w[ii] = np.sin(np.pi*ii/self.NSamplesPerSegment)
        
        data = np.zeros((self.Nd,self.NSamplesPerSegment*self.NSegments),dtype=np.ndarray)
    
        for ii in range(self.Nd):
            for jj in range(self.NSegments):
                y0 = w*segments[ii][2*jj][:]
                y1 = w*segments[ii][2*jj+1][:]
                y2 = w*segments[ii][2*jj+2][:]
                
                z0=np.concatenate((y0[int(self.NSamplesPerSegment/2):self.NSamplesPerSegment], np.zeros(int(self.NSamplesPerSegment/2))))
                z1=y1[:]
                z2=np.concatenate((np.zeros(int(self.NSamplesPerSegment/2)),y2[0:int(self.NSamplesPerSegment/2)]))
                
                data[ii,jj*self.NSamplesPerSegment:(jj+1)*self.NSamplesPerSegment] = z0 + z1 + z2
                
        return data

# def pre_processing(omegaGW,noisePSD,orf,Fs,segmentDuration):
#     """
    
#     Parameters
#     ==========
    
#     Returns
#     =======
    
#     """
#     freqs=omegaGW.frequencies.value
    
#     deltaF_new = 1/segmentDuration
#     fmax_new=Fs/2
#     freqs_new=np.arange(deltaF_new,fmax_new,deltaF_new) #Check with Andrew!
    
#     omegaSpline=interp1d(freqs,omegaGW.value,kind='cubic',fill_value='extrapolate')
#     omegaGW_new = omegaSpline(freqs_new)
#     omegaGW_new = gwpy.frequencyseries.FrequencySeries(omegaGW_new, frequencies=freqs_new)
    
#     Nd = noisePSD.shape[0]
#     noisePSD_new = np.zeros((Nd),dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries)
#     for ii in range(Nd):
#         noiseSpline=interp1d(freqs,noisePSD[ii].value,kind='cubic',fill_value='extrapolate')
#         noise_new=noiseSpline(freqs_new)
#         noise_new = gwpy.frequencyseries.FrequencySeries(noise_new, frequencies=freqs_new)
#         noisePSD_new[ii]=noise_new
    
#     orf_new_array = np.zeros((orf.shape[0]),dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries)
#     for ii in range(orf.shape[0]):
#         print(type(orf[ii]))
#         orfSpline=interp1d(freqs,orf[ii].value,kind='cubic',fill_value='extrapolate')
#         orf_new = orfSpline(freqs_new)
#         orf_new_array[ii] = gwpy.frequencyseries.FrequencySeries(orf_new, frequencies=freqs_new)
    
#     return omegaGW_new, noisePSD_new, orf_new_array