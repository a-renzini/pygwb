import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import bilby
import gwpy.timeseries

from .util import *
from .cross_correlation import *
from .postprocessing import *
from .pe import *
from .constants import *

from .simulation import *

matplotlib.rcParams.update({'font.size':15})

##################
# Inputs     
##################

NSegments         = 16
Fs                = 1024. # Hz
segmentDuration   = 64.0 # s
t0                = 0 # s
TAvg              = 4.0 # s

####################
# Computed quantities
####################

NSamplesPerSegment=int(segmentDuration*Fs) 
deltaT=1/Fs
fNyquist=1/(2*deltaT)
deltaF=1/segmentDuration
deltaFStoch=1/TAvg
NAvgs = 2 * int(segmentDuration / TAvg) - 1
jobDuration = NSegments * segmentDuration
N = NSegments*NSamplesPerSegment   #Total number of samples

Nd = 2   #Number of detectors

# Discrete times
T = N*deltaT
t = np.array([t0 + deltaT*i for i in range(int(N))])

if NSamplesPerSegment%2==0:
    numFreqs = NSamplesPerSegment/2-1
else:
    numFreqs = (NSamplesPerSegment-1)/2

#Discrete positive frequencies
deltaF = 1/(NSamplesPerSegment*deltaT)
f = np.array([deltaF*(i+1) for i in range(int(numFreqs))])

# Generate a noise PSD for each of the detectors (here an example of colored noise, but could use H1/L1 PSD)

noisePSD=np.zeros((Nd),dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries)

noisePSD0=1e-45*f**(-.5)+1e-45*f**(-.5)*np.random.randn(int(numFreqs))/10
noisePSD[0]=gwpy.frequencyseries.FrequencySeries(noisePSD0,frequencies=f)
noisePSD1=1e-45*f**(-.5)+1e-45*f**(-.5)*np.random.randn(int(numFreqs))/10
noisePSD[1]=gwpy.frequencyseries.FrequencySeries(noisePSD1,frequencies=f)

###################
# Signal properties
###################

fref = 25   #reference frequency (Hz)
alpha = 2/3  #spectral index
OmegaRef=1e-6 #Amplitude GW spectrum as in h^2Omega=OmegaRef*(f/fref)**alpha

OmegaGW = OmegaRef*(f/fref)**alpha

OmegaGW = gwpy.frequencyseries.FrequencySeries(OmegaGW, frequencies=f)

# Interpolate ORF for two LIGO detectors -- Can be generalized for other detectors
orf_file='../src/orfs/ORF_HL.dat'

datContent = [i.strip().split() for i in open(orf_file).readlines()]

x=[float(datContent[i][0]) for i in range(len(datContent))]
y=[float(datContent[i][1]) for i in range(len(datContent))]

orf_f = interp1d(x, y, kind='cubic', fill_value='extrapolate')

orf1 = gwpy.frequencyseries.FrequencySeries(orf_f(f), frequencies=f)
orf=np.zeros(3,dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries)
orf[0]=orf1

simul1 = simulation_GWB(noisePSD,OmegaGW,orf,Fs=None,segmentDuration=None,NSegments=NSegments)

xdata = simul1.generate_data()

d1=TimeSeries(t,xdata[0])
d2=TimeSeries(t,xdata[1])

# run stochastic pipeline
alpha=0 #This alpha simply follows from cross-corelation statistic, don't change even for other spectral alphas
fref=25
Y_t,sig_t,Y_ft,var_ft,segmentStartTimes,freqs=cross_correlation(d1,d2,segmentDuration,deltaFStoch,orf_file='../src/orfs/ORF_HL.dat',fref=fref,alpha=alpha,fmin=deltaFStoch,fmax=fNyquist/2.)

Y_f,var_f=postprocessing_spectra(Y_ft,var_ft,jobDuration,segmentDuration,
                                 deltaFStoch,deltaT)

# test optimal filtering

alphas=np.linspace(-8,8,100)

snrs=np.zeros(alphas.shape)
for ii,a in enumerate(alphas):
    y,s=calc_Y_sigma_from_Yf_varf(np.real(Y_f),
                          var_f,
                          freqs=freqs,
                          alpha=a,
                          fref=25)
    snrs[ii]=y/s
fig1,axs1=plt.subplots()
axs1.plot(alphas,snrs)
axs1.axvline(2/3)
axs1.set_xlabel('alpha')
axs1.set_ylabel('SNR')
axs1.set_title('Optimal SNR vs alpha')
plt.savefig("OptimalSNR23")

# Run parameter estimation
label = 'GWB_powerlaw'
outdir = 'outdir'

cleanup_dir(outdir)

Amin,Amax=1e-13,1e-2
    
fref=25

omegaGW_inj=OmegaRef
alpha_inj=0
#injection_parameters=dict('A'=omegaGW_inj,'alpha'=alpha_inj)

likelihood = BasicPowerLawGWBLikelihood(Y_f[1:],var_f[1:],freqs[1:],fref)
priors = dict(A=bilby.core.prior.LogUniform(Amin,Amax, 'A'),
              alpha=bilby.core.prior.Gaussian(0,3.5, 'alpha'))

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
    walks=10, outdir=outdir, label=label,maxmcmc=10000)
#result.plot_corner()

A1=result.samples[:,0]
fig=plt.figure()
x=np.linspace(Amin,Amax,10)

plt.hist(np.log10(A1),bins=30,histtype='step',color='blue',density=True,label='')
plt.axvline(np.log10(OmegaRef),color='red')

plt.xlabel('log$_{10}$ \u03A9$_{ref}$')
plt.ylabel('Posterior probability density')
plt.savefig("OmegaRefInj23")
