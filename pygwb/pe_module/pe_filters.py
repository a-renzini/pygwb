import bilby
import numpy as np
from astropy import cosmology, units

#define models here

def power_law_model(omega_alpha, alpha, frequencies, orf, fref=25):
    return orf * omega_alpha * (frequencies / fref)**(alpha)

def multi_power_law_model(omega_alpha_1, alpha_1, omega_alpha_2, alpha_2, frequencies, orf, fref=25):
    return orf * (omega_alpha_1 * (frequencies / fref)**(alpha_1) + omega_alpha_2 * (frequencies / fref)**(alpha_2))

def broken_power_law_model(omega_alpha, alpha_1, alpha_2, frequencies, orf, fbreak=100):
    return omega_alpha * (np.piecewise(frequencies, [frequencies<=fbreak, frequencies>fbreak], [lambda frequencies: (frequencies / fbreak)**(alpha_1), lambda frequencies: (frequencies / fbreak)**(alpha_2)]))

def smooth_broken_power_law_model(omega_alpha, fbreak, alpha_1, alpha_2, delta, frequencies, orf):
    return orf * omega_alpha * (frequencies / fbreak)**(alpha_1) * (1+(frequencies / fbreak)**(delta))**((alpha_2-alpha_1)/delta)

def triple_broken_power_law_model(omega_alpha, alpha_1, alpha_2, alpha_3, fbreak_1, fbreak_2, frequencies, orf):
    return orf * omega_alpha * (np.piecewise(frequencies, [frequencies<=fbreak_1, (frequencies<=fbreak_2) & (frequencies>fbreak_1)], [lambda frequencies: (frequencies / fbreak_1)**(alpha_1), lambda frequencies: (frequencies / fbreak_1)**(alpha_2), lambda frequencies: (fbreak_2 / fbreak_1)**(alpha_2) * (frequencies / fbreak_2)**(alpha_3)]))

#magnetic models

def schumann_model(kappa1, kappa2, beta1, beta2, M_f, freqs):
    #assumes simple power law model for transfer function
    T1 = kappa1 *  (freqs / 10)**(-beta1) * 1e-23
    T2 = kappa2 *  (freqs / 10)**(-beta2) * 1e-23
    H_amp = T1 * T2 * np.real(M_f)  #units of transfer function strain/pT
    return H_amp

def schumann_model2(kappa1, kappa2, beta11, beta12, beta21, beta22, fbreak1, fbreak2, M_f, freqs):
    #assumes broken power law model for transfer function
    const1 = (fbreak1 / 10)**(beta11)
    const2 = (fbreak2 / 10)**(beta21)
    T1 = kappa1 * np.piecewise(freqs, [freqs<=fbreak1, freqs>fbreak1], [lambda freqs: (freqs / fbreak1)**(beta11), lambda freqs: const1 * (freqs / fbreak1)**(beta12)]) * 1e-23
    T2 = kappa2 * np.piecewise(freqs, [freqs<=fbreak2, freqs>fbreak2], [lambda freqs: (freqs / fbreak2)**(beta21), lambda freqs: const2 * (freqs / fbreak2)**(beta22)]) * 1e-23
    H_amp = T1 * T2 * np.real(M_f)  #units of coupling transfer strain/pT
    return H_amp

#parity violation models

def pv_power_law_model(omega_alpha, alpha, Pi, frequencies, gammaI, gammaV, fref=25):
    return (1 + Pi * gammaV / gammaI) * omega_alpha * (frequencies / fref)**(alpha)

def pv_power_law_model2(omega_alpha, alpha, beta, frequencies, gammaI, gammaV, fref=25):
    return (1 + ((frequencies)**(beta)) * gammaV / gammaI) * omega_alpha * (frequencies / fref)**(alpha)

#extra polarisations

def tvs_power_law_model(omegaT_alpha, omegaV_alpha, omegaS_alpha, alphaT, alphaV, alphaS, frequencies, Torf, Vorf, Sorf, fref=25):
    modelT = Torf * omegaT_alpha * (frequencies / fref)**(alphaT)
    modelV = Vorf * omegaV_alpha * (frequencies / fref)**(alphaV)
    modelS = Sorf * omegaS_alpha * (frequencies / fref)**(alphaS)   
    return modelT + modelV + modelS

#define likelihoods here (single-baseline likelihoods first, then multiIFO likelihoods)

class PowerLawLikelihood(bilby.Likelihood):
    """
    likelihood for a simple power law model
    """
    def __init__(self, point_estimate, sigma, frequencies, orf=None):
        if orf is None:
            orf = np.ones(frequencies.size)
        self.orf = orf
        self.parameters = {'omega_alpha': None, 'alpha': None}
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        # i have no idea why i have to do this. but i do.
        # whatever.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        # gaussian log likelihood
        model = power_law_model(self.parameters['omega_alpha'], self.parameters['alpha'], self.frequencies, self.orf)
        res = model - self.point_estimate
        return -0.5 * np.sum(res**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))

    def noise_log_likelihood(self):
        # noise log likelihood is just calculating
        # gaussian log likelihood with no model.
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))

class MultiPowerLawLikelihood(bilby.Likelihood):
    """
    likelihood for a multiple power law model
    """
    def __init__(self, point_estimate, sigma, frequencies, orf=None):
        if orf is None:
            orf = np.ones(frequencies.size)
        self.orf = orf
        self.parameters = {'omega_alpha_1': None, 'alpha_1': None, 'omega_alpha_2': None, 'alpha_2': None}
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        # i have no idea why i have to do this. but i do.
        # whatever.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        # gaussian log likelihood
        model = multi_power_law_model(self.parameters['omega_alpha_1'], self.parameters['alpha_1'],
                                      self.parameters['omega_alpha_2'], self.parameters['alpha_2'],
                                      self.frequencies, self.orf)
        res = model - self.point_estimate
        return -0.5 * np.sum(res**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))

    def noise_log_likelihood(self):
        # noise log likelihood is just calculating
        # gaussian log likelihood with no model.
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))   
    
class BrokenPowerLawLikelihood(bilby.Likelihood):
    """
    likelihood for a broken power law model
    """
    def __init__(self, point_estimate, sigma, frequencies, orf=None):
        if orf is None:
            orf = np.ones(frequencies.size)
        self.orf = orf
        self.parameters = {'omega_alpha': None, 'alpha_1': None, 'alpha_2': None}
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        # i have no idea why i have to do this. but i do.
        # whatever.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        # gaussian log likelihood
        model = broken_power_law_model(self.parameters['omega_alpha'], self.parameters['alpha_1'],
                                      self.parameters['alpha_2'],
                                      self.frequencies, self.orf)
        res = model - self.point_estimate
        return -0.5 * np.sum(res**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))

    def noise_log_likelihood(self):
        # noise log likelihood is just calculating
        # gaussian log likelihood with no model.
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))
    
class SmoothBrokenPowerLawLikelihood(bilby.Likelihood):
    """
    likelihood for a smooth broken power law model
    """
    def __init__(self, point_estimate, sigma, frequencies, orf=None):
        if orf is None:
            orf = np.ones(frequencies.size)
        self.orf = orf
        self.parameters = {'omega_alpha': None, 'fbreak': None, 'alpha_1': None, 'alpha_2': None, 'delta': None}
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        # i have no idea why i have to do this. but i do.
        # whatever.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        # gaussian log likelihood
        model = smooth_broken_power_law_model(self.parameters['omega_alpha'], self.parameters['fbreak'] ,self.parameters['alpha_1'],
                                      self.parameters['alpha_2'], self.parameters['delta'],
                                      self.frequencies, self.orf)
        res = model - self.point_estimate
        return -0.5 * np.sum(res**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))
    
    def noise_log_likelihood(self):
        # noise log likelihood is just calculating
        # gaussian log likelihood with no model.
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))      

class TripleBrokenPowerLawLikelihood(bilby.Likelihood):
    """
    likelihood for a triple broken power law model
    """
    def __init__(self, point_estimate, sigma, frequencies, orf=None):
        if orf is None:
            orf = np.ones(frequencies.size)
        self.orf = orf
        self.parameters = {'omega_alpha': None, 'alpha_1': None, 'alpha_2': None, 'alpha_3': None, 'fbreak_1': None, 'fbreak_2': None}
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        # i have no idea why i have to do this. but i do.
        # whatever.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        # gaussian log likelihood
        model = triple_broken_power_law_model(self.parameters['omega_alpha'], self.parameters['alpha_1'],
                                      self.parameters['alpha_2'], self.parameters['alpha_3'], self.parameters['fbreak_1'],  self.parameters['fbreak_2'],
                                      self.frequencies, self.orf)
        res = model - self.point_estimate
        return -0.5 * np.sum(res**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))
    def noise_log_likelihood(self):
        # noise log likelihood is just calculating
        # gaussian log likelihood with no model.
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))         
    
class SchumannLikelihood(bilby.Likelihood):
    """
    likelihood for simple schumann model w/ power law transfer functions
    """
    def __init__(self, point_estimate, sigma, M_f, frequencies, baseline='HL'):
        self.parameters = {'kappa'+baseline[0]: None, 'kappa'+baseline[1]: None,
                           'beta'+baseline[0]: None, 'beta'+baseline[1]: None}
        self.baseline = baseline
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        self.M_f = M_f
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        d1 = self.baseline[0]
        d2 = self.baseline[1]
        model = schumann_model(self.parameters['kappa' + d1], self.parameters['kappa' + d2],
                               self.parameters['beta' + d1], self.parameters['beta' + d2],
                               self.M_f,
                               self.frequencies)
        res = model - self.point_estimate
        newsigma2 = self.sigma**2 #+ np.abs(model)**2
        return -0.5 * np.sum(res**2 / newsigma2) - 0.5 * np.sum(np.log(2 * np.pi * newsigma2))

    def noise_log_likelihood(self):
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))
    
class SchumannLikelihood2(bilby.Likelihood):
    """
    likelihood for schumann model w/ broken power law transfer functions
    """
    def __init__(self, point_estimate, sigma, M_f, frequencies, baseline='HL'):
        self.parameters = {'kappa'+baseline[0]: None, 'kappa'+baseline[1]: None,
                           'beta1'+baseline[0]: None, 'beta2'+baseline[0]: None,
                           'beta1'+baseline[1]: None, 'beta2'+baseline[1]: None,
                           'fbreak'+baseline[0]: None, 'fbreak'+baseline[1]: None}
        self.baseline = baseline
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        self.M_f = M_f
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        d1 = self.baseline[0]
        d2 = self.baseline[1]
        model = schumann_model2(self.parameters['kappa' + d1], self.parameters['kappa' + d2],
                               self.parameters['beta1' + d1], self.parameters['beta2' + d1], 
                               self.parameters['beta1' + d2] ,self.parameters['beta2' + d2],
                               self.parameters['fbreak' + d1], self.parameters['fbreak' + d2],
                               self.M_f,
                               self.frequencies)
        res = model - self.point_estimate
        newsigma2 = self.sigma**2 #+ np.abs(model)**2
        return -0.5 * np.sum(res**2 / newsigma2) - 0.5 * np.sum(np.log(2 * np.pi * newsigma2))

    def noise_log_likelihood(self):
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))

class SchumannPowerLawLikelihood(bilby.Likelihood):
    """
    Schumann w/ power law transfer functions + Power Law Likelihood
    """
    def __init__(self, point_estimate, sigma, M_f, frequencies, baseline='HL', orf=None):
        self.parameters = {'kappa'+baseline[0]: None, 'kappa'+baseline[1]: None,
                           'beta'+baseline[0]: None, 'beta'+baseline[1]: None,
                           'omega_alpha': None, 'alpha': None}
        self.baseline = baseline
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        self.M_f = M_f
        if orf is None:
            self.orf = np.ones(self.frequencies.size)
        else:
            self.orf = orf
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        d1 = self.baseline[0]
        d2 = self.baseline[1]
        pl_model = power_law_model(self.parameters['omega_alpha'], self.parameters['alpha'],
                                   self.frequencies, self.orf)
        model = schumann_model(self.parameters['kappa' + d1], self.parameters['kappa' + d2],
                               self.parameters['beta' + d1], self.parameters['beta' + d2],
                               self.M_f,
                               self.frequencies)
        res = model + pl_model - self.point_estimate
        newsigma2 = self.sigma**2 # + np.abs(model)**2
        return -0.5 * np.sum(res**2 / newsigma2) - 0.5 * np.sum(np.log(2 * np.pi * newsigma2))

    def noise_log_likelihood(self):
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))    

class SchumannPowerLawLikelihood2(bilby.Likelihood):
    """
    Schumann w/ broken power law transfer functions + Power Law Likelihood
    """
    def __init__(self, point_estimate, sigma, M_f, frequencies, baseline='HL', orf=None):
        self.parameters = {'kappa'+baseline[0]: None, 'kappa'+baseline[1]: None,
                           'beta1'+baseline[0]: None, 'beta2'+baseline[0]: None,
                           'beta1'+baseline[1]: None, 'beta2'+baseline[1]: None,
                           'fbreak'+baseline[0]: None, 'fbreak'+baseline[1]: None,
                           'omega_alpha': None, 'alpha': None}
        self.baseline = baseline
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        self.M_f = M_f
        if orf is None:
            self.orf = np.ones(self.frequencies.size)
        else:
            self.orf = orf
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        d1 = self.baseline[0]
        d2 = self.baseline[1]
        pl_model = power_law_model(self.parameters['omega_alpha'], self.parameters['alpha'],
                                   self.frequencies, self.orf)
        model = schumann_model2(self.parameters['kappa' + d1], self.parameters['kappa' + d2],
                               self.parameters['beta1' + d1], self.parameters['beta2' + d1], 
                               self.parameters['beta1' + d2] ,self.parameters['beta2' + d2],
                               self.parameters['fbreak' + d1], self.parameters['fbreak' + d2],
                               self.M_f,
                               self.frequencies)
        res = model + pl_model - self.point_estimate
        newsigma2 = self.sigma**2 # + np.abs(model)**2
        return -0.5 * np.sum(res**2 / newsigma2) - 0.5 * np.sum(np.log(2 * np.pi * newsigma2))

    def noise_log_likelihood(self):
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))      

class SchumannSmoothBrokenPowerLawLikelihood(bilby.Likelihood):
    """
    Schumann w/ power law transfer functions + Smooth Broken Power Law Likelihood
    """
    def __init__(self, point_estimate, sigma, M_f, frequencies, baseline='HL', orf=None):
        self.parameters = {'kappa'+baseline[0]: None, 'kappa'+baseline[1]: None,
                           'beta'+baseline[0]: None, 'beta'+baseline[1]: None,
                           'omega_alpha': None, 'fbreak': None, 'alpha_1': None, 'alpha_2': None, 'delta': None}
        self.baseline = baseline
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        self.M_f = M_f
        if orf is None:
            self.orf = np.ones(self.frequencies.size)
        else:
            self.orf = orf
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        d1 = self.baseline[0]
        d2 = self.baseline[1]
        gw_model = smooth_broken_power_law_model(self.parameters['omega_alpha'], self.parameters['fbreak'] ,self.parameters['alpha_1'],
                                      self.parameters['alpha_2'], self.parameters['delta'],
                                      self.frequencies, self.orf)
        model = schumann_model(self.parameters['kappa' + d1], self.parameters['kappa' + d2],
                               self.parameters['beta' + d1], self.parameters['beta' + d2],
                               self.M_f,
                               self.frequencies)
        res = model + gw_model - self.point_estimate
        newsigma2 = self.sigma**2 # + np.abs(model)**2
        return -0.5 * np.sum(res**2 / newsigma2) - 0.5 * np.sum(np.log(2 * np.pi * newsigma2))

    def noise_log_likelihood(self):
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))        
    
class PVPowerLawLikelihood(bilby.Likelihood):
    """
    likelihood for a simple power law model in parity violating case Pi=const
    """
    def __init__(self, point_estimate, sigma, frequencies, gammaI, gammaV):
        self.parameters = {'omega_alpha': None, 'alpha': None, 'Pi': None}
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        self.gammaI = gammaI
        self.gammaV = gammaV
        # i have no idea why i have to do this. but i do.
        # whatever.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        # gaussian log likelihood
        model = pv_power_law_model(self.parameters['omega_alpha'], self.parameters['alpha'], self.parameters['Pi'], self.frequencies, self.gammaI, self.gammaV)
        res = model - self.point_estimate
        return -0.5 * np.sum(res**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))

    def noise_log_likelihood(self):
        # noise log likelihood is just calculating
        # gaussian log likelihood with no model.
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))
    
class PVPowerLawLikelihood2(bilby.Likelihood):
    """
    likelihood for a simple power law model in parity violating case Pi=f^(beta)
    """
    def __init__(self, point_estimate, sigma, frequencies, gammaI, gammaV):
        self.parameters = {'omega_alpha': None, 'alpha': None, 'beta': None}
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        self.gammaI = gammaI
        self.gammaV = gammaV
        # i have no idea why i have to do this. but i do.
        # whatever.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        # gaussian log likelihood
        model = pv_power_law_model2(self.parameters['omega_alpha'], self.parameters['alpha'], self.parameters['beta'], self.frequencies, self.gammaI, self.gammaV)
        res = model - self.point_estimate
        return -0.5 * np.sum(res**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))

    def noise_log_likelihood(self):
        # noise log likelihood is just calculating
        # gaussian log likelihood with no model.
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))    

class TVSPowerLawLikelihood(bilby.Likelihood):
    """
    likelihood for extra polarisations -- simple power law model
    """
    def __init__(self, point_estimate, sigma, frequencies, Torf=None, Vorf=None, Sorf=None):            
        self.Torf = Torf
        self.Vorf = Vorf
        self.Sorf = Sorf
        self.parameters = {'omegaT_alpha': None, 'omegaV_alpha': None, 'omegaS_alpha': None, 'alphaT': None, 'alphaV': None, 'alphaS': None}
        self.point_estimate = point_estimate
        self.sigma = sigma
        self.frequencies = frequencies
        # i have no idea why i have to do this. but i do.
        # whatever.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        # gaussian log likelihood
        model = tvs_power_law_model(self.parameters['omegaT_alpha'], self.parameters['omegaV_alpha'], self.parameters['omegaS_alpha'], self.parameters['alphaT'], self.parameters['alphaV'], self.parameters['alphaS'], self.frequencies, self.Torf, self.Vorf, self.Sorf)
        res = model - self.point_estimate
        return -0.5 * np.sum(res**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))

    def noise_log_likelihood(self):
        # noise log likelihood is just calculating
        # gaussian log likelihood with no model.
        return -0.5 * np.sum(self.point_estimate**2 / self.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * self.sigma**2))
    
    
class MultiIFOPowerLawLikelihood(bilby.Likelihood):
    """
    multi baseline power law likelihood"
    """
    def __init__(self, ll_list):
        self.parameters = {'omega_alpha': None, 'alpha': None}
        self.ll_list = ll_list
        
        # again...whatever the heck this is for.
        # probably this is supposed to be a super(MultiIFOPowerLawLikelihood).__init__()
        # but for some reason the monash guys do it this way.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        log_like = 0
        # loop over list of individual baseline log likelihoods
        for ll in self.ll_list:
            # get baseline log likelihood
            ll.parameters = self.parameters
            # add it to three-baseline log likelihood
            log_like += ll.log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_list:
            nll += ll.noise_log_likelihood()
        return nll    
    
class MultiIFOMultiPowerLawLikelihood(bilby.Likelihood):
    """
    multi baseline multiple power law likelihood"
    """
    def __init__(self, ll_list):
        self.parameters = {'omega_alpha_1': None, 'alpha_1': None, 'omega_alpha_2': None, 'alpha_2': None}
        self.ll_list = ll_list
        
        # again...whatever the heck this is for.
        # probably this is supposed to be a super(MultiIFOPowerLawLikelihood).__init__()
        # but for some reason the monash guys do it this way.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        log_like = 0
        # loop over list of individual baseline log likelihoods
        for ll in self.ll_list:
            # get baseline log likelihood
            ll.parameters = self.parameters
            # add it to three-baseline log likelihood
            log_like += ll.log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_list:
            nll += ll.noise_log_likelihood()
        return nll        
    
class MultiIFOBrokenPowerLawLikelihood(bilby.Likelihood):
    """
    multi baseline broken power law likelihood"
    """
    def __init__(self, ll_list):
        self.parameters = {'omega_alpha': None, 'alpha_1': None, 'alpha_2': None}
        self.ll_list = ll_list
        
        # again...whatever the heck this is for.
        # probably this is supposed to be a super(MultiIFOPowerLawLikelihood).__init__()
        # but for some reason the monash guys do it this way.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        log_like = 0
        # loop over list of individual baseline log likelihoods
        for ll in self.ll_list:
            # get baseline log likelihood
            ll.parameters = self.parameters
            # add it to three-baseline log likelihood
            log_like += ll.log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_list:
            nll += ll.noise_log_likelihood()
        return nll            
    
class MultiIFOSmoothBrokenPowerLawLikelihood(bilby.Likelihood):
    """
    multi baseline smooth broken power law likelihood"
    """
    def __init__(self, ll_list):
        self.parameters = {'omega_alpha': None, 'fbreak': None, 'alpha_1': None, 'alpha_2': None, 'delta': None}
        self.ll_list = ll_list
        
        # again...whatever the heck this is for.
        # probably this is supposed to be a super(MultiIFOPowerLawLikelihood).__init__()
        # but for some reason the monash guys do it this way.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        log_like = 0
        # loop over list of individual baseline log likelihoods
        for ll in self.ll_list:
            # get baseline log likelihood
            ll.parameters = self.parameters
            # add it to three-baseline log likelihood
            log_like += ll.log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_list:
            nll += ll.noise_log_likelihood()
        return nll  

class MultiIFOTripleBrokenPowerLawLikelihood(bilby.Likelihood):
    def __init__(self, ll_list):
        self.parameters = {'omega_alpha': None, 'alpha_1': None, 'alpha_2': None, 'alpha_3': None, 'fbreak_1': None, 'fbreak_2': None}
        self.ll_list = ll_list
        
        # again...whatever the heck this is for.
        # probably this is supposed to be a super(MultiIFOPowerLawLikelihood).__init__()
        # but for some reason the monash guys do it this way.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        log_like = 0
        # loop over list of individual baseline log likelihoods
        for ll in self.ll_list:
            # get baseline log likelihood
            ll.parameters = self.parameters
            # add it to three-baseline log likelihood
            log_like += ll.log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_list:
            nll += ll.noise_log_likelihood()
        return nll      
    
class MultiIFOSchumannLikelihood(bilby.Likelihood):
    """
    multi baseline Schumann Likelihood w/ power law transfer functions
    """
    def __init__(self, ll_dict):
        self.parameters = {'kappaH': None, 'kappaV': None, 'kappaL': None,
                           'betaH': None, 'betaL': None, 'betaV': None
                           }
        self.ll_dict = ll_dict
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        log_like = 0
        # loop over baselines
        for ll in self.ll_dict:
            # pick out parameters from THIS likelihoods draws and feed it to
            # the individual baseline likelihoods.
            self.ll_dict[ll].parameters['kappa' + ll[0]] = self.parameters['kappa' + ll[0]]
            self.ll_dict[ll].parameters['kappa' + ll[1]] = self.parameters['kappa' + ll[1]]
            self.ll_dict[ll].parameters['beta' + ll[0]] = self.parameters['beta' + ll[0]]
            self.ll_dict[ll].parameters['beta' + ll[1]] = self.parameters['beta' + ll[1]]
            # calculate individual baseline log-likelihood,
            # add it to multi-baseline log-likelihood.
            log_like += self.ll_dict[ll].log_likelihood()
        return log_like

    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_dict:
            nll += self.ll_dict[ll].noise_log_likelihood()
        return nll

class MultiIFOSchumannLikelihood2(bilby.Likelihood):
    """
    multi baseline Schumann Likelihood w/ broken power law transfer functions
    """
    def __init__(self, ll_dict):
        self.parameters = {'kappaH': None, 'kappaV': None, 'kappaL': None,
                           'beta1H': None, 'beta2H': None, 'beta1L': None, 
                           'beta2L': None, 'beta1V': None, 'beta2V': None,
                           'fbreakH' : None, 'fbreakL' : None, 'fbreakV' : None
                           }
        self.ll_dict = ll_dict
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        log_like = 0
        # loop over baselines
        for ll in self.ll_dict:
            # pick out parameters from THIS likelihoods draws and feed it to
            # the individual baseline likelihoods.
            self.ll_dict[ll].parameters['kappa' + ll[0]] = self.parameters['kappa' + ll[0]]
            self.ll_dict[ll].parameters['kappa' + ll[1]] = self.parameters['kappa' + ll[1]]
            self.ll_dict[ll].parameters['beta1' + ll[0]] = self.parameters['beta1' + ll[0]]
            self.ll_dict[ll].parameters['beta2' + ll[0]] = self.parameters['beta2' + ll[0]]
            self.ll_dict[ll].parameters['beta1' + ll[1]] = self.parameters['beta1' + ll[1]]
            self.ll_dict[ll].parameters['beta2' + ll[1]] = self.parameters['beta2' + ll[1]]
            self.ll_dict[ll].parameters['fbreak' + ll[0]] = self.parameters['fbreak' + ll[0]]
            self.ll_dict[ll].parameters['fbreak' + ll[1]] = self.parameters['fbreak' + ll[1]]
            # calculate individual baseline log-likelihood,
            # add it to multi-baseline log-likelihood.
            log_like += self.ll_dict[ll].log_likelihood()
        return log_like

    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_dict:
            nll += self.ll_dict[ll].noise_log_likelihood()
        return nll
    
    
class MultiIFOSchumannPowerLawLikelihood(bilby.Likelihood):
    """
    multi baseline Schumann Likelihood w/ power law transfer functions + GW power law
    """
    def __init__(self, ll_dict):
        self.parameters = {'kappaH': None, 'kappaV': None, 'kappaL': None,
                           'betaH': None, 'betaL': None, 'betaV': None,
                           'omega_alpha': None, 'alpha': None
                           }
        self.ll_dict = ll_dict
        bilby.Likelihood.__init__(self, parameters=self.parameters)
        
    def log_likelihood(self):
        log_like = 0
        for ll in self.ll_dict:
            self.ll_dict[ll].parameters['kappa' + ll[0]] = self.parameters['kappa' + ll[0]]
            self.ll_dict[ll].parameters['kappa' + ll[1]] = self.parameters['kappa' + ll[1]]
            self.ll_dict[ll].parameters['beta' + ll[0]] = self.parameters['beta' + ll[0]]
            self.ll_dict[ll].parameters['beta' + ll[1]] = self.parameters['beta' + ll[1]]
            self.ll_dict[ll].parameters['omega_alpha'] = self.parameters['omega_alpha']
            self.ll_dict[ll].parameters['alpha'] = self.parameters['alpha']
#             print(ll)
#             print(self.ll_dict[ll].parameters)
            log_like += self.ll_dict[ll].log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_dict:
            nll += self.ll_dict[ll].noise_log_likelihood()
        return nll    
    
class MultiIFOSchumannPowerLawLikelihood2(bilby.Likelihood):
    """
    multi baseline Schumann Likelihood w/ broken power law transfer functions + GW power law
    """
    def __init__(self, ll_dict):
        self.parameters = {'kappaH': None, 'kappaV': None, 'kappaL': None,
                           'beta1H': None, 'beta2H': None, 'beta1L': None, 
                           'beta2L': None, 'beta1V': None, 'beta2V': None,
                           'fbreakH' : None, 'fbreakL' : None, 'fbreakV' : None,
                           'omega_alpha': None, 'alpha': None
                           }
        self.ll_dict = ll_dict
        bilby.Likelihood.__init__(self, parameters=self.parameters)
        
    def log_likelihood(self):
        log_like = 0
        for ll in self.ll_dict:
            self.ll_dict[ll].parameters['kappa' + ll[0]] = self.parameters['kappa' + ll[0]]
            self.ll_dict[ll].parameters['kappa' + ll[1]] = self.parameters['kappa' + ll[1]]
            self.ll_dict[ll].parameters['beta1' + ll[0]] = self.parameters['beta1' + ll[0]]
            self.ll_dict[ll].parameters['beta2' + ll[0]] = self.parameters['beta2' + ll[0]]
            self.ll_dict[ll].parameters['beta1' + ll[1]] = self.parameters['beta1' + ll[1]]
            self.ll_dict[ll].parameters['beta2' + ll[1]] = self.parameters['beta2' + ll[1]]
            self.ll_dict[ll].parameters['fbreak' + ll[0]] = self.parameters['fbreak' + ll[0]]
            self.ll_dict[ll].parameters['fbreak' + ll[1]] = self.parameters['fbreak' + ll[1]]
            self.ll_dict[ll].parameters['omega_alpha'] = self.parameters['omega_alpha']
            self.ll_dict[ll].parameters['alpha'] = self.parameters['alpha']
#             print(ll)
#             print(self.ll_dict[ll].parameters)
            log_like += self.ll_dict[ll].log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_dict:
            nll += self.ll_dict[ll].noise_log_likelihood()
        return nll

class MultiIFOSchumannSmoothBrokenPowerLawLikelihood(bilby.Likelihood):
    """
    multi baseline Schumann Likelihood w/ power law transfer functions + smooth broken power law
    """
    def __init__(self, ll_dict):
        self.parameters = {'kappaH': None, 'kappaV': None, 'kappaL': None,
                           'betaH': None, 'betaL': None, 'betaV': None,
                           'omega_alpha': None, 'fbreak': None, 'alpha_1': None, 'alpha_2': None, 'delta': None
                           }
        self.ll_dict = ll_dict
        bilby.Likelihood.__init__(self, parameters=self.parameters)
        
    def log_likelihood(self):
        log_like = 0
        for ll in self.ll_dict:
            self.ll_dict[ll].parameters['kappa' + ll[0]] = self.parameters['kappa' + ll[0]]
            self.ll_dict[ll].parameters['kappa' + ll[1]] = self.parameters['kappa' + ll[1]]
            self.ll_dict[ll].parameters['beta' + ll[0]] = self.parameters['beta' + ll[0]]
            self.ll_dict[ll].parameters['beta' + ll[1]] = self.parameters['beta' + ll[1]]
            self.ll_dict[ll].parameters['omega_alpha'] = self.parameters['omega_alpha']
            self.ll_dict[ll].parameters['fbreak'] = self.parameters['fbreak']
            self.ll_dict[ll].parameters['alpha_1'] = self.parameters['alpha_1']
            self.ll_dict[ll].parameters['alpha_2'] = self.parameters['alpha_2']
            self.ll_dict[ll].parameters['delta'] = self.parameters['delta']
#             print(ll)
#             print(self.ll_dict[ll].parameters)
            log_like += self.ll_dict[ll].log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_dict:
            nll += self.ll_dict[ll].noise_log_likelihood()
        return nll        
    
class MultiIFOPVPowerLawLikelihood(bilby.Likelihood):
    """
    multi baseline likelihood for a simple power law model in parity violating case, Pi=const
    """
    def __init__(self, ll_list):
        self.parameters = {'omega_alpha': None, 'alpha': None, 'Pi': None}
        self.ll_list = ll_list
        
        # again...whatever the heck this is for.
        # probably this is supposed to be a super(MultiIFOPowerLawLikelihood).__init__()
        # but for some reason the monash guys do it this way.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        log_like = 0
        # loop over list of individual baseline log likelihoods
        for ll in self.ll_list:
            # get baseline log likelihood
            ll.parameters = self.parameters
            # add it to three-baseline log likelihood
            log_like += ll.log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_list:
            nll += ll.noise_log_likelihood()
        return nll  
    
class MultiIFOPVPowerLawLikelihood2(bilby.Likelihood):
    """
    multi baseline likelihood for a simple power law model in parity violating case, Pi=f^(beta)
    """
    def __init__(self, ll_list):
        self.parameters = {'omega_alpha': None, 'alpha': None, 'beta': None}
        self.ll_list = ll_list
        
        # again...whatever the heck this is for.
        # probably this is supposed to be a super(MultiIFOPowerLawLikelihood).__init__()
        # but for some reason the monash guys do it this way.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        log_like = 0
        # loop over list of individual baseline log likelihoods
        for ll in self.ll_list:
            # get baseline log likelihood
            ll.parameters = self.parameters
            # add it to three-baseline log likelihood
            log_like += ll.log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_list:
            nll += ll.noise_log_likelihood()
        return nll      
    
class TVSMultiIFOPowerLawLikelihood(bilby.Likelihood):
    """
    extra polarisations multi baseline power law likelihood"
    """
    def __init__(self, ll_list):
        self.parameters = {'omegaT_alpha': None, 'omegaV_alpha': None, 'omegaS_alpha': None, 'alphaT': None, 'alphaV': None, 'alphaS': None}
        self.ll_list = ll_list
        
        # again...whatever the heck this is for.
        # probably this is supposed to be a super(MultiIFOPowerLawLikelihood).__init__()
        # but for some reason the monash guys do it this way.
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    def log_likelihood(self):
        log_like = 0
        # loop over list of individual baseline log likelihoods
        for ll in self.ll_list:
            # get baseline log likelihood
            ll.parameters = self.parameters
            # add it to three-baseline log likelihood
            log_like += ll.log_likelihood()
        return log_like
    
    def noise_log_likelihood(self):
        nll = 0
        for ll in self.ll_list:
            nll += ll.noise_log_likelihood()
        return nll    