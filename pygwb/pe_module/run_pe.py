#!/usr/bin/env python
import numpy as np
import bilby
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
from pe_filters import *
from orf import ORF

#import data
#take e.g. O3
Y_HL, sigma_HL, Y_HV, sigma_HV, Y_LV, sigma_LV, frequencies = np.loadtxt('o3.dat', unpack=True, usecols=(0,1,2,3,4,5,6))

#import ORFs -- check for consistency with ORF module
Torf_HL, Vorf_HL, Sorf_HL = ORF(frequencies, baseline='HL')
Torf_HV, Vorf_HV, Sorf_HV = ORF(frequencies, baseline='HV')
Torf_LV, Vorf_LV, Sorf_LV = ORF(frequencies, baseline='LV')

#choose pair likelihoods for the models you want to constrain with the data
#power law
ll_HL_pl = PowerLawLikelihood(Y_HL, sigma_HL, Torf_HL, frequencies)
ll_HV_pl = PowerLawLikelihood(Y_HV, sigma_HV, Torf_HL, frequencies)
ll_LV_pl = PowerLawLikelihood(Y_LV, sigma_LV, Torf_HL, frequencies)

#extra polarisations power law
tvs_HL_like = TVSPowerLawLikelihood(Y_HL, sigma_HL, frequencies, Torf_HL, Vorf_HL, Sorf_HL)
tvs_HV_like = TVSPowerLawLikelihood(Y_HV, sigma_HV, frequencies, Torf_HV, Vorf_HV, Sorf_HV)
tvs_LV_like = TVSPowerLawLikelihood(Y_LV, sigma_LV, frequencies, Torf_LV, Vorf_LV, Sorf_LV)

#multi-detector likelihoods
multiIFO_like_pl = MultiIFOPowerLawLikelihood([ll_HL_pl, ll_HV_pl, ll_LV_pl])
tvs_multiIFO_like_pl = TVSMultiIFOPowerLawLikelihood([tvs_HL_like, tvs_HV_like, tvs_LV_like])

# priors
priors_pl = {'omega_alpha': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\alpha}$'),
                       'alpha': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha$')}

priors_tvs_pl = {'omegaT_alpha': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\alpha}^T$'),
                       'alphaT': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha^T$'),
                 'omegaV_alpha': bilby.core.prior.DeltaFunction(0),
                       'alphaV': bilby.core.prior.DeltaFunction(0),
               #'omegaV_alpha': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\alpha}^V$'),
               #        'alphaV': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha^V$'),
               'omegaS_alpha': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\alpha}^S$'),
                       'alphaS': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha^S$')}

#run PE sampler
#print("HL work")
#hl_pl = bilby.run_sampler(likelihood=ll_HL_pl,
#                            priors=priors_pl,
#                            sampler='dynesty', npoints=1000, walks=10,
#                            outdir='./',
#                            label= 'hl_pl',
#                            resume=False)
#hl_pl.plot_corner()

#print("HLV work")
#multiIFO_pl = bilby.run_sampler(likelihood=multiIFO_like_pl,
#                                priors=priors_pl,
#                                sampler='dynesty', npoints=1000, walks=10,
#                                outdir='./',
#                                label= 'multiIFO_pl',
#                                resume=False)
#multiIFO_pl.plot_corner()

print("HLV work")
multiIFO_tvs_pl = bilby.run_sampler(likelihood=tvs_multiIFO_like_pl,
                                priors=priors_tvs_pl,
                                sampler='dynesty', npoints=1000, walks=10,
                                outdir='./',
                                label= 'multiIFO_ts_pl',
                                resume=False)
multiIFO_tvs_pl.plot_corner()