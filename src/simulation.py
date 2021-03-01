import numpy as np
from constants import H0
from scipy.interpolate import interp1d

# Interpolate ORF for two LIGO detectors -- Can be generalized for other detectors
orf_file='../src/orfs/ORF_HL.dat'

datContent = [i.strip().split() for i in open(orf_file).readlines()]

x=[float(datContent[i][0]) for i in range(len(datContent))]
y=[float(datContent[i][1]) for i in range(len(datContent))]

orf_f = interp1d(x, y, kind='cubic')

def orf_func(x):
    if x<10:
        return orf_f(10)
    else:
        return orf_f(x)

def simulate_data(freqs,noisePSD,OmegaGW):
    Nf = len(freqs)
    Nt = 2*(Nf+1)
    Nd = noisePSD.shape[0]
    deltaF=freqs[1]-freqs[0]
    deltaT = 1/(Nt*deltaF)
    
    H_theor = (3*H0**2)/(10*np.pi**2)
    
    norm = np.sqrt(Nt/(2*deltaT))*np.sqrt(H_theor)*np.sqrt(OmegaGW)*freqs**(-3/2)
    normPSD = np.sqrt(Nt/(2*deltaT))*np.sqrt(noisePSD)
    
    # Implement ORF frequency dependence here -- Can be generalized to other/more detectors
    gamma11=np.ones(int(Nf)) #Response function for detector 1
    gamma22=np.ones(int(Nf)) #Response function for detector 2
    gamma12=np.array([orf_func(i) for i in freqs]) #ORF

    C=[]
    for i in range(len(gamma11)):
        C.append(np.array([[gamma11[i],gamma12[i]],[gamma12[i],gamma22[i]]]))
    C=np.array(C) #Correlation matrix for each of the frequencies. Has shape (numFreqs,M,M) with M=2 for two detectors

    eigval,eigvec=np.linalg.eig(C) #Compute eigenvalues and eigenvectors of C for each frequency

    eigval=np.array([np.diag(x) for x in eigval]) #Array with diagonal M x M matrices containing the eigenvalues
    
    #Generate three time segments that will be spliced together to prevent periodicity of ifft
    y=np.zeros((Nd,3),dtype=np.ndarray)
    
    for kk in range(3):
        z = np.zeros((Nf,Nd),dtype = 'complex_')
        for ii in range(Nd):
            # Construct real and imaginary parts, with random phases (uncorrelated)
            re = (norm/np.sqrt(2))*np.random.randn(int(Nf))
            im = (norm/np.sqrt(2))*np.random.randn(int(Nf))
            z[:,ii] = re+1j*im
    
        znoise = np.zeros((Nf,Nd),dtype = 'complex_')
        for ii in range(Nd):
            # Construct real and imaginary parts, with random phases (uncorrelated)
            re = (normPSD[ii]/np.sqrt(2))*np.random.randn(int(Nf))
            im = (normPSD[ii]/np.sqrt(2))*np.random.randn(int(Nf))
            znoise[:,ii] = re+1j*im
        
        # Transform from uncorrelated z to correlated x:
        xtemp = np.zeros((Nf,Nd),dtype = 'complex_')
        for ii in range(Nf):
            xtemp[ii,:] = (z[ii]).dot(np.sqrt(eigval[ii]).dot(eigvec[ii].transpose())) #This ensures data is correlated properly
            xtemp[ii,:] += znoise[ii,:] #Add noise PSD to the data
        
        for ii in range(Nd):
            # Set DC and Nyquist = 0, then add negative freq parts in proper order
            if Nt%2==0:
                # Note that most negative frequency is -f_Nyquist when N=even
                xtilde = np.concatenate((np.array([0]),xtemp[:,ii],np.array([0]),np.flipud(np.conjugate(xtemp[:,ii]))))
            else:
                # No Nyquist frequency when N=odd
                xtilde = np.concatenate((np.array([0]),xtemp[:,ii],np.flipud(np.conjugate(xtemp[:,ii]))))
            # Fourier transform back to time domain and take real part (imag part = 0)
            y[ii][kk]=np.real(np.fft.ifft(xtilde))
    return y

def splice_segments(y):
    Nt=len(y[0][0])
    Nd=y.shape[0]
    
    w=np.zeros(Nt)
    
    for ii in range(Nt):
        w[ii]=np.sin(np.pi * ((ii+1)-1)/Nt)
    
    xdata=np.zeros(Nd,dtype=np.ndarray)

    for ii in range(Nd):
        y0 = w*y[ii][0]
        y1 = w*y[ii][1]
        y2 = w*y[ii][2]
    
        z0=np.concatenate((y0[int(Nt/2):Nt],np.zeros(int(Nt/2))))
        z1=y1[:]
        z2=np.concatenate((np.zeros(int(Nt/2)),y2[0:int(Nt/2)]))
    
        xdata[ii] = z0 + z1 + z2 #Correlated strain data for ii-th detector
    
    return xdata