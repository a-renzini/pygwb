import numpy as np
from matplotlib import pyplot as plt
import bilby
from scipy.interpolate import CubicSpline
from scipy.special import erf
import shutil

from util import *
from constants import *

def cross_correlation(d1,d2,segmentDuration,deltaF,verbose=True,doOverlap=True,alpha=0,fref=1,orf_file=None,fmin=0,fmax=0):
    
    # error checking
    assert d1.deltaT == d2.deltaT
    
    # compute length of segment and interval
    length_of_segment = int(segmentDuration / d1.deltaT)
    length_of_interval = int(3 * length_of_segment)
    
    if doOverlap:
        # overlapping intervals
        # see https://git.ligo.org/stochastic-public/stochastic/-/blob/master/CrossCorr/src_cc/loadAuxiliaryInput.m/#L288
        bufferSecs = 0 
        jobDuration = d1.times[-1] - d1.times[0] + d1.deltaT
        # number of non-overlapping segments
        M = int( np.floor( (jobDuration - 2*bufferSecs)/segmentDuration ) )
        numSegmentsTotal = 2*M-1
        numIntervalsTotal = 2*(M-2)-1
        intervalTimeStride = int(segmentDuration/2)
        
        stride = int(intervalTimeStride / d1.deltaT)
    else:
        numIntervalsTotal = int(len(d1.data) / length_of_interval)
        stride = length_of_interval

    # initialize point estimate and error bar arrays
    Ys=np.zeros(numIntervalsTotal)
    sigs=np.zeros(numIntervalsTotal)
    
    # initialize Y(f,t) and variance(f,t) 2d-arrays
    # also define frequency and start time arrays
    d1_test=slice_time_series(d1,0,length_of_segment)
    segmentStartTimes=d1.times[2*stride:-4*stride+1:stride]
    
    epsilon=deltaF/100.
    freqs=np.arange(fmin,fmax+epsilon,deltaF)

    numFreqs = len(freqs)
    Y_ft = np.zeros((numFreqs,numIntervalsTotal))
    var_ft = np.zeros((numFreqs,numIntervalsTotal))
   
    # initialize orf, set to 1 if no file provided
    if orf_file is not None:
        f_orf,orf=np.loadtxt(orf_file,unpack=True)
        orfSpline=CubicSpline(f_orf,orf)
        orf = orfSpline(freqs)
    else:
        orf=np.ones(freqs.shape)
   
    # window factors
    w1w2bar,w1w2squaredbar,_,_=window_factors(int(segmentDuration/d1.deltaT))
    
    # loop over intervals
    for II in range(numIntervalsTotal):
        offset = II * stride
        
        # split interval into segments 1, 2, 3
        d1_left=slice_time_series(d1,offset,offset+length_of_segment)
        d1_mid =slice_time_series(d1,offset+length_of_segment,
                                   offset+2*length_of_segment)
        d1_right=slice_time_series(d1,offset+2*length_of_segment,
                                   offset+3*length_of_segment)
        
        d2_left=slice_time_series(d2,offset,offset+length_of_segment)
        d2_mid =slice_time_series(d2,offset+length_of_segment,
                                   offset+2*length_of_segment)
        d2_right=slice_time_series(d2,offset+2*length_of_segment,
                                    offset+3*length_of_segment)
    
        # window, zero pad, fft
        d1_mid_tilde = d1_mid.window_and_fft()
        d2_mid_tilde = d2_mid.window_and_fft()
        
        
        # compute cross correlation d1*d2 [segment 2]
        c = FrequencySeries(d1_mid_tilde.freqs, 
                            d1_mid_tilde.data * np.conj(d2_mid_tilde.data))
        
    
        # coarse grain
        c_cg=c.coarse_grain(deltaF,fmin,fmax)
    
        # compute power spectra [segment 1 + 3]
        psd_freqs,P1_left = welch_psd(d1_left.data,
                            window='hann',
                            nperseg=int(d1_left.Fs/deltaF),
                            fs=d1_left.Fs)
        _,P1_right = welch_psd(d1_right.data,
                            window='hann',
                            nperseg=int(d1_right.Fs/deltaF),
                            fs=d1_right.Fs)  
        P1 = FrequencySeries(psd_freqs,
                            0.5*(P1_left+P1_right))

        _,P2_left = welch_psd(d2_left.data,
                            window='hann',
                            nperseg=int(d2_left.Fs/deltaF),
                            fs=d2_left.Fs)
        _,P2_right = welch_psd(d2_right.data,
                            window='hann',
                            nperseg=int(d2_right.Fs/deltaF),
                            fs=d2_right.Fs)  
        P2 = FrequencySeries(psd_freqs,
                            0.5*(P2_left+P2_right))       

       
        imin=np.argmin(np.abs(P1.freqs-fmin))
        imax=np.argmin(np.abs(P1.freqs-fmax)) + 1
        P1.freqs=P1.freqs[imin:imax]
        P1.data=P1.data[imin:imax]
        P2.freqs=P2.freqs[imin:imax]
        P2.data=P2.data[imin:imax]

        assert(np.sum(np.abs(P1.freqs-freqs))==0)

        # construct cross-correlation spectra
                
        S0 = FrequencySeries(P1.freqs, ( (3*H0**2)/
                                         (10*np.pi**2*fref**3)*
                                         (P1.freqs/fref)**(alpha-3)) )
        Y_f = FrequencySeries(P1.freqs, 2*c_cg.data/
                                         (segmentDuration * orf * S0.data ))
        Y_f = Y_f / w1w2bar
        var_f = FrequencySeries(P1.freqs, (P1.data*P2.data) / (2*segmentDuration*deltaF)
                                             * orf**(-2) * S0.data**(-2))
        var_f = var_f * w1w2squaredbar / w1w2bar**2
        
        Ys[II],sigs[II] = calc_Y_sigma_from_Yf_varf(Y_f.data,var_f.data) 
        Y_ft[:,II],var_ft[:,II] = Y_f.data, var_f.data
                
        
        if verbose:
            print('stochastic: Done with Interval %u / %u'%(II+1,numIntervalsTotal))
            print('\tY     = %e'%Ys[II])
            print('\tsigma = %e'%sigs[II])
            print('\tSNR   = %f'%(Ys[II]/sigs[II]))
        
    return Ys,sigs,Y_ft,var_ft,segmentStartTimes,freqs
