#!/usr/bin/env python
import numpy as np
import bilby
from scipy.io import loadmat
from .pe_filters import *
from .baseline import Baseline
import bilby.gw.detector as bilbydet

#import data
#take e.g. O3
frequencies, Y_HL, sigma_HL = np.loadtxt('C_O3_HL.dat', unpack=True, usecols=(0,1,2))
Y_HV, sigma_HV = np.loadtxt('C_O3_HV.dat', unpack=True, usecols=(1,2))
Y_LV, sigma_LV = np.loadtxt('C_O3_LV.dat', unpack=True, usecols=(1,2))

#go up to 256 Hz
idx=np.argmin(np.abs(frequencies-256))
frequencies = frequencies[:idx]
#cut all of the data for frequencies > 256 Hz
sigma_HL = sigma_HL[:idx]
Y_HL = Y_HL[:idx]
sigma_HV = sigma_HV[:idx]
Y_HV = Y_HV[:idx]
sigma_LV = sigma_LV[:idx]
Y_LV = Y_LV[:idx]

#remove infinities from sigma measurements(O3a)
inf_array=np.isinf(sigma_HL)
not_inf_array = ~ inf_array
sigma_HL = sigma_HL[not_inf_array]
Y_HL=Y_HL[not_inf_array]
sigma_HV = sigma_HV[not_inf_array]
Y_HV=Y_HV[not_inf_array]
sigma_LV = sigma_LV[not_inf_array]
Y_LV=Y_LV[not_inf_array]
frequencies=frequencies[not_inf_array]

inf_array=np.isinf(sigma_HV)
not_inf_array = ~ inf_array
sigma_HL = sigma_HL[not_inf_array]
Y_HL=Y_HL[not_inf_array]
sigma_HV = sigma_HV[not_inf_array]
Y_HV=Y_HV[not_inf_array]
sigma_LV = sigma_LV[not_inf_array]
Y_LV=Y_LV[not_inf_array]
frequencies=frequencies[not_inf_array]

inf_array=np.isinf(sigma_LV)
not_inf_array = ~ inf_array
sigma_HL = sigma_HL[not_inf_array]
Y_HL=Y_HL[not_inf_array]
sigma_HV = sigma_HV[not_inf_array]
Y_HV=Y_HV[not_inf_array]
sigma_LV = sigma_LV[not_inf_array]
Y_LV=Y_LV[not_inf_array]
frequencies=frequencies[not_inf_array]


point_estimates = [Y_HL, Y_HV, Y_LV]
sigmas = [sigma_HL, sigma_HV, sigma_LV]
H1 = bilbydet.get_empty_interferometer('H1')
L1 = bilbydet.get_empty_interferometer('L1')
V1 = bilbydet.get_empty_interferometer('V1')

HL = Baseline('H1L1', H1, L1, duration=10, sampling_frequency=10)
HV = Baseline('H1V1', H1, V1, duration=10, sampling_frequency=10)
LV = Baseline('L1V1', L1, V1, duration=10, sampling_frequency=10)

HL.frequencies = frequencies
HV.frequencies = frequencies
LV.frequencies = frequencies


HL.point_estimate = Y_HL
HL.sigma = sigma_HL
HV.point_estimate = Y_HV
HV.sigma = sigma_HV
LV.point_estimate = Y_LV
LV.sigma = sigma_LV


# ###############################################
# ###############Testing pl##################
# ###############################################

#choose pair likelihoods for the models you want to constrain with the data
#power law
# kwargs_pl = {"baselines":[HL,HV,LV], "model_name":'PL', "fref":25}
# model_pl = PowerLawModel(**kwargs_pl)
# priors_pl = {'omega_ref': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\rm ref}$'),
#                         'alpha': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha$')}
# hlv_pl=bilby.run_sampler(likelihood=model_pl,priors=priors_pl,sampler='dynesty', npoints=1000, walks=10,outdir='./',label= 'hlv_pl', resume=False)
# hlv_pl.plot_corner()


# ###############################################
# ###############Testing bpl##################
# ###############################################

# kwargs_bpl = {"baselines": [HL, HV, LV], "model_name":'BPL'}
# model_bpl = BrokenPowerLawModel(**kwargs_bpl)
# priors_bpl = {'omega_ref': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\rm ref}$'),
#             'fbreak': bilby.core.prior.Uniform(1, 100,'$f_{\\rm break}$'),
#             'alpha_1': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha_1$'),
#             'alpha_2': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha_2$')}
# hlv_bpl=bilby.run_sampler(likelihood=model_bpl,priors=priors_bpl,sampler='dynesty', npoints=1000, walks=10,outdir='./',label= 'hlv_bpl', resume=False)
# hlv_bpl.plot_corner()


###############################################
######### Testing triple_BPL##################
###############################################

# kwargs_triple_bpl = {"baselines":[HL,HV,LV],"model_name": 'TBPL'}
# model_triple_bpl = TripleBrokenPowerLawModel(**kwargs_triple_bpl)
# priors_triple_bpl = {'omega_ref': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\rm ref}$'),
#                        'alpha_1': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha_1$'),
#                       'alpha_2': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha_2$'),
#                        'alpha_3': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha_3$'),
#                      'fbreak1': bilby.core.prior.Uniform(1, 100,'$f_{\\rm break}^1$'), 
#                      'fbreak2': bilby.core.prior.Uniform(1, 100,'$f_{\\rm break}^2$')}
# hlv_triple_bpl = bilby.run_sampler(likelihood=model_triple_bpl,priors=priors_triple_bpl,sampler='dynesty', npoints=1000, walks=10,outdir='./',label= 'hlv_tbpl', resume=False)
# hlv_triple_bpl.plot_corner()


# ###############################################
# ######### Testing Smooth Broken PL#############
# ###############################################

# kwargs_sbpl = {"baselines":[HL,HV,LV],"model_name": 'SBPL'}
# model_sbpl = SmoothBrokenPowerLawModel(**kwargs_sbpl)
# priors_sbpl = {'omega_ref': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\rm ref}$'),
#                   'fbreak': bilby.core.prior.Uniform(1, 256, '$f_{\\rm break}$'),
#                         'alpha_1': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha_1$'),
#                        'alpha_2': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha_2$'),
#                         'delta': bilby.core.prior.Uniform(0, 8, '$\\Delta$')}
# hlv_sbpl = bilby.run_sampler(likelihood=model_sbpl,priors=priors_sbpl,sampler='dynesty', npoints=1000, walks=10,outdir='./',label= 'hlv_sbpl', resume=False)
# hlv_sbpl.plot_corner()


# ###############################################
# ###############Testing tvs pl##################
# ###############################################

# kwargs_pl_sv={"baselines":[HL, HV, LV], "model_name":'PL_SV', "fref":25, "polarizations":['scalar', 'vector']}
# model_pl_sv = TVSPowerLawModel(**kwargs_pl_sv)
# priors_pl_sv = {'omega_ref_scalar': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\rm ref,s}$'),
#                       'alpha_scalar': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha_s$'),
#           'omega_ref_vector': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\rm ref,v}$'),
#                       'alpha_vector': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha_v$')}
# hlv_pl_sv=bilby.run_sampler(likelihood=model_pl_sv,priors=priors_pl_sv,sampler='dynesty', npoints=1000, walks=10,outdir='./',label= 'hlv_pl_sv', resume=False)
# hlv_pl_sv.plot_corner()


# ###############################################
# ######### Testing Parity Violation PL 1 #######
# ###############################################

# kwargs_pl_pv = {"baselines":[HL, HV, LV],"model_name": 'PL_PV', 'fref': 25}
# model_pl_pv = PVPowerLawModel(**kwargs_pl_pv)
# priors_pl_pv = {'omega_ref': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\rm ref}$'),
#                        'alpha': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha$'),
#                        'Pi': bilby.core.prior.Uniform(-1,1,'$\\Pi$')}
# hlv_pl_pv = bilby.run_sampler(likelihood=model_pl_pv,priors=priors_pl_pv,sampler='dynesty', npoints=1000, walks=10,outdir='./',label= 'hlv_pl_pv', resume=False)
# hlv_pl_pv.plot_corner()

##############################################
######## Testing Parity Violation PL 2 #######
##############################################

# kwargs_pv_pl_2 = {"baselines":[HL, HV, LV],"model_name": 'PL_PV_2', 'fref': 25}
# model_pv_pl_2 = PVPowerLawModel2(**kwargs_pv_pl_2)
# priors_pv_pl_2 = {'omega_ref': bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\rm ref}$'),
#                      'alpha': bilby.core.prior.Gaussian(0, 3.5, '$\\alpha$'),
#                        'beta': bilby.core.prior.Uniform(-2,0,'$\\beta$')}
# hlv_pv_pl_2 = bilby.run_sampler(likelihood=model_pv_pl_2,priors=priors_pv_pl_2,sampler='dynesty', npoints=1000, walks=10,outdir='./',label= 'hlv_pv_pl_2', resume=False)
# hlv_pv_pl_2.plot_corner()







