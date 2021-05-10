import numpy as np
from constants import H0
from scipy.interpolate import interp1d
import gwpy

# Add dimension of arrays in the explanation?

def simulate_data(noisePSD, OmegaGW, orf, Fs=None, segmentDuration=None, NSegments=1):
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
    freqs = OmegaGW.frequencies.value
    Nf = OmegaGW.size
    #Ntold = 2*(Nf+1)
    #print(f'Nt_old={Ntold}')
    Nd = noisePSD.shape[0]
    deltaF = OmegaGW.df.value
    #deltaTold = 1/(Ntold*deltaF)
    if Fs==None:
        Fs=(freqs[-1]+deltaF)*2 #Assumes freq is power of two
        
    if segmentDuration==None:
        segmentDuration=1/(deltaF)

    OmegaGW_new, noisePSD_new, orf_new = pre_processing(OmegaGW,noisePSD,orf,Fs,segmentDuration)
    freqs_new = OmegaGW_new.frequencies.value
    
    NSamplesPerSegment=int(Fs*segmentDuration)
    
    Nt=NSamplesPerSegment
    deltaT=1/Fs
    
    H_theor = (3*H0**2)/(10*np.pi**2)
    
    GWBPower = omegaToPower(OmegaGW)
    
    #orf_array=np.array([[0,orf],[orf,0]],dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries)
    C = covariance_matrix(noisePSD, GWBPower, orf)
    eigval, eigvec = compute_eigval_eigvec(C)
    
    #Generate three time segments that will be spliced together to prevent periodicity of ifft
    y=np.zeros((Nd,2*NSegments+1,NSamplesPerSegment),dtype=np.ndarray)

    for kk in range(2*NSegments+1):
        z = generate_freq_domain_data(Nf,Nd)
        
        xtemp = transform_to_correlated_data(z, eigval, eigvec)
        
        for ii in range(Nd):
            # Set DC and Nyquist = 0, then add negative freq parts in proper order
            if Nt%2==0:
                # Note that most negative frequency is -f_Nyquist when N=even
                xtilde = np.concatenate((np.array([0]),xtemp[:,ii],np.array([0]),np.flipud(np.conjugate(xtemp[:,ii]))))
            else:
                print('inside else statement')
                # No Nyquist frequency when N=odd
                xtilde = np.concatenate((np.array([0]),xtemp[:,ii],np.flipud(np.conjugate(xtemp[:,ii]))))
            # Fourier transform back to time domain and take real part (imag part = 0)
            y[ii,kk,:]=np.real(np.fft.ifft(xtilde))
    return y

def pre_processing(omegaGW,noisePSD,orf,Fs,segmentDuration):
    """
    
    Parameters
    ==========
    
    Returns
    =======
    
    """
    freqs=omegaGW.frequencies.value
    
    deltaF_new = 1/segmentDuration
    fmax_new=Fs/2
    freqs_new=np.arange(deltaF_new,fmax_new,deltaF_new) #Check with Andrew!
    
    omegaSpline=interp1d(freqs,omegaGW.value,kind='cubic',fill_value='extrapolate')
    omegaGW_new = omegaSpline(freqs_new)
    omegaGW_new = gwpy.frequencyseries.FrequencySeries(omegaGW_new, frequencies=freqs_new)
    
    Nd = noisePSD.shape[0]
    noisePSD_new = np.zeros((Nd),dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries)
    for ii in range(Nd):
        noiseSpline=interp1d(freqs,noisePSD[ii].value,kind='cubic',fill_value='extrapolate')
        noise_new=noiseSpline(freqs_new)
        noise_new = gwpy.frequencyseries.FrequencySeries(noise_new, frequencies=freqs_new)
        noisePSD_new[ii]=noise_new
    
    orf_new_array = np.zeros((orf.shape[0]),dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries)
    for ii in range(orf.shape[0]):
        print(type(orf[ii]))
        orfSpline=interp1d(freqs,orf[ii].value,kind='cubic',fill_value='extrapolate')
        orf_new = orfSpline(freqs_new)
        orf_new_array[ii] = gwpy.frequencyseries.FrequencySeries(orf_new, frequencies=freqs_new)
    
    return omegaGW_new, noisePSD_new, orf_new_array

def orfToArray(orf):
    
    return orf_array

def PowerToOmega(power):
    H_theor = (3*H0**2)/(10*np.pi**2)
    freqs = power.frequencies.value
    omegaGW = power.value/H_theor*freqs**3
    omegaGW = gwpy.frequencyseries.FrequencySeries(omegaGW, frequencies=freqs)
    return omegaGW

def omegaToPower(omegaGW):
    """
    Function that computes the GW power spectrum starting from the Omega spectrum.
    
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
    freqs = omegaGW.frequencies.value
    power = H_theor*omegaGW.value*freqs**(-3)
    power = gwpy.frequencyseries.FrequencySeries(power, frequencies=freqs)
    return power

def covariance_matrix(noisePower, GWBPower, orf):
    """
    Function to compute the covariance matrix corresponding to a stochastic 
    background in the various detectors.
    
    Parameters
    ==========
    noisePower:
    
    GWBPower:
    
    orf:
    
    Returns
    =======
    C: array_like
        Covariance matric corresponding to a stochastic background in the 
        various detectors.
    """
    Nf = GWBPower.size
    Nt = 2*(Nf+1)
    deltaF = GWBPower.df.value
    deltaT = 1/(Nt*deltaF)
    Nd = noisePower.shape[0]
    
    #C = np.zeros((Nd,Nd,Nf))
    #for ii in range(Nd):
    #    for jj in range(Nd):
    #        if ii==jj:
    #            C[ii,jj,:] = noisePower[ii].value[:]+GWBPower.value[:]
    #        else:
    #            C[ii,jj,:] = orf[ii,jj].value[:]*GWBPower.value[:]
    #C = Nt/deltaT/4*C  
    Cp = Nt/deltaT/4*np.array(
        [[noisePower[0].value[:]+GWBPower.value[:],orf[0].value[:]*GWBPower.value[:]],
         [orf[0].value[:]*GWBPower.value[:],noisePower[1].value[:]+GWBPower.value[:]]])
    return Cp

def compute_eigval_eigvec(C):
    """
    Function to compute the eigenvalues and eigenvectors of the covariance
    matrix corresponding to a stochastic background in the various detectors.
    
    Parameters
    ==========
    
    Returns
    =======
    """

    Nf=C.shape[2]
    
    eigval,eigvec = np.linalg.eig(C.transpose((2,0,1)))
    eigval = np.array([np.diag(x) for x in eigval])
    
    return eigval, eigvec

def generate_freq_domain_data(Nf, Nd):
    """
    Function that generates the uncorrelated frequency domain data for the
    stochastic background of gravitational waves.
    
    Parameters
    ==========
    Nf: int
        Number of frequencies
    Nd: int
        Number of detectors
    
    Returns
    =======
    z: array_like
    """
    z = np.zeros((Nf,Nd),dtype = 'complex_')
    re = np.random.randn(Nf,Nd)
    im = np.random.randn(Nf,Nd)
    z = re+1j*im
    
    return z

def transform_to_correlated_data(z, eigval, eigvec):
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
    Nf = z.shape[0]
    Nd = z.shape[1]
    
    A = np.einsum('...ij,jk...',np.sqrt(eigval),eigvec.transpose())
    x = np.einsum('...j,...jk',z,A)
        
    return x

def splice_segments(y):
    """
    
    Parameters
    ==========
    y: Nd x 2N+1 x NSamplesPerSegment
    
    Returns
    =======
    data:
    """
    Nd=y.shape[0]
    NSegments = int((y.shape[1]-1)/2)
    NSamplesPerSegment=y.shape[2]
    
    w=np.zeros(NSamplesPerSegment)
    
    for ii in range(NSamplesPerSegment):
        w[ii]=np.sin(np.pi*ii/NSamplesPerSegment)
    
    data=np.zeros((Nd,NSamplesPerSegment*NSegments),dtype=np.ndarray)

    for ii in range(Nd):
        for jj in range(NSegments):
            y0 = w*y[ii][2*jj][:]
            y1 = w*y[ii][2*jj+1][:]
            y2 = w*y[ii][2*jj+2][:]
    
            z0=np.concatenate((y0[int(NSamplesPerSegment/2):NSamplesPerSegment],np.zeros(int(NSamplesPerSegment/2))))
            z1=y1[:]
            z2=np.concatenate((np.zeros(int(NSamplesPerSegment/2)),y2[0:int(NSamplesPerSegment/2)]))
            
            data[ii,jj*NSamplesPerSegment:(jj+1)*NSamplesPerSegment] = z0 + z1 + z2
            
    return data