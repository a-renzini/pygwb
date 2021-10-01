import numpy as np
from abc import abstractmethod
import bilby
from .baseline import Baseline

class GWBModel(bilby.Likelihood):
    """
    GWB Model
    ---------
    generic model,
    contains definitions of log likelihood and noise
    """
    def __init__(self, baselines=None, model_name=None, polarizations=None):
        super(GWBModel, self).__init__()
        # list of baselines
        if baselines is None:
            raise ValueError('list of baselines must be supplied')
        if model_name is None:
            raise ValueError('model_name must be supplied')
        # if single baseline supplied, that's fine
        # just make it a list
        if isinstance(baselines, Baseline):
            baselines = [baselines]
        self.baselines = baselines
        self.orfs = []
        self.polarizations = polarizations
        # if polarizations is not supplied
        if polarizations is None:
            self.polarizations = ['tensor' for ii in range(len(baselines))]
        for bline in self.baselines:
            if self.polarizations[0].lower() == 'tensor':
                self.orfs.append(bline.overlap_reduction_function)
            elif self.polarizations[0].lower() == 'vector':
                self.orfs.append(bline.vector_overlap_reduction_function)
            elif self.polarizations[0].lower() == 'scalar':
                self.orfs.append(bline.scalar_overlap_reduction_function)
            else:
                raise ValueError('unexpected type for polarizations {}'.format(type(polarizations)))         
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    @abstractmethod
    def parameters(self):
        """Parameters. Should return a dict"""
        pass

    @abstractmethod
    def model_function(self):
        """function for evaluating model"""
        pass

    def log_likelihood(self):
        """
        log likelihood
        """
        ll = 0
        for orf, bline in zip(self.orfs, self.baselines):
            model = orf * self.model_function(bline)
            res = model - bline.point_estimate
            ll += -0.5 * np.sum(res**2 / bline.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * bline.sigma**2))
        return ll

    def noise_log_likelihood(self):
        # noise log likelihood is just calculating
        # gaussian log likelihood with no model.
        ll = 0
        for bline in self.baselines:
            ll += -0.5 * np.sum(bline.point_estimate**2 / bline.sigma**2) - 0.5 * np.sum(np.log(2 * np.pi * bline.sigma**2))
        return ll

class PowerLawModel(GWBModel):
    """
    Parameters:
    -----------
    fref : float
        reference frequency for defining the model
    omega_ref : float
        amplitude of signal at fref
    alpha : float
        spectral index of the power law
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """
    def __init__(self, **kwargs):
        try:
            fref = kwargs.pop('fref')
        except KeyError:
            raise KeyError("fref must be supplied")
        super(PowerLawModel, self).__init__(**kwargs)
        self.fref = fref

    @property
    def parameters(self):
        if self._parameters is None:
            #include this so that the parent class GWBModel doesn't complain
            return {'omega_ref': None, 'alpha': None}
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError('unexpected type for parameters {}'.format(type(parameters)))

    def model_function(self, bline):
        return self.parameters['omega_ref'] * (bline.frequencies / self.fref)**self.parameters['alpha']


class BrokenPowerLawModel(GWBModel):
    """
    Parameters:
    -----------
    omega_ref : float
        amplitude of signal at fref
    alpha_1 : float
        spectral index of the broken power law
    alpha_2 : float
        spectral index of the broken power law
    fbreak : float
        break frequency for the broken power law
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """
    def __init__(self, **kwargs):
        super(BrokenPowerLawModel, self).__init__(**kwargs)

    @property
    def parameters(self):
        if self._parameters is None:
            #include this so that the parent class GWBModel doesn't complain
            return {'omega_ref': None, 'fbreak': None,'alpha_1': None, 'alpha_2': None}
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError('unexpected type for parameters {}'.format(type(parameters)))

    def model_function(self, bline):
        frequencies = bline.frequencies
        return self.parameters['omega_ref'] * (np.piecewise(frequencies, [frequencies<=self.parameters['fbreak'], frequencies>self.parameters['fbreak']], [lambda frequencies: (frequencies / self.parameters['fbreak'])**(self.parameters['alpha_1']), lambda frequencies: (frequencies / self.parameters['fbreak'])**(self.parameters['alpha_2'])]))
    


class TripleBrokenPowerLawModel(GWBModel):
    """
    Parameters:
    -----------
    omega_ref : float
        amplitude of signal at fref
    alpha_1 : float
        spectral index of the broken power law
    alpha_2 : float
        spectral index of the broken power law
    alpha_3 : float
        spectral index of the broken power law
    fbreak1 : float
        break frequency for the broken power law
    fbreak1 : float
        break frequency for the broken power law
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """
    def __init__(self, **kwargs):
        super(TripleBrokenPowerLawModel, self).__init__(**kwargs)


    @property
    def parameters(self):
        if self._parameters is None:
            #include this so that the parent class GWBModel doesn't complain
            return {'omega_ref': None,'alpha_1': None, 'alpha_2': None, 'alpha_3': None,  'fbreak1': None, 'fbreak2': None}
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError('unexpected type for parameters {}'.format(type(parameters)))

    def model_function(self, bline):
        frequencies = bline.frequencies
        piecewise = np.piecewise(frequencies, [frequencies <= self.parameters['fbreak1'], (frequencies <= self.parameters['fbreak2']) & (frequencies > self.parameters['fbreak1'])], 
                                [lambda frequencies: (frequencies / self.parameters['fbreak1'])**(self.parameters['alpha_1']), 
                                lambda frequencies: (frequencies / self.parameters['fbreak1'])**(self.parameters['alpha_2']), 
                                lambda frequencies: (self.parameters['fbreak2'] / self.parameters['fbreak1'])**(self.parameters['alpha_2']) * (frequencies / self.parameters['fbreak2'])**(self.parameters['alpha_3'])])
        return self.parameters['omega_ref'] * piecewise
    

class SmoothBrokenPowerLawModel(GWBModel):
    """
    Parameters:
    -----------
    omega_ref : float
        amplitude of signal at fref
    delta : float
        smoothing variable for the smooth broken power law
    alpha_1 : float
        low-frequency spectral index of the smooth broken power law
    alpha_2 : float
       (alpha_2 - alpha_1)/Delta is high-frequency spectral index of the smooth broken power law
    fbreak : float
        break frequency for the smooth broken power law
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """
    def __init__(self, **kwargs):
        super(SmoothBrokenPowerLawModel, self).__init__(**kwargs)
        
    @property
    def parameters(self):
        if self._parameters is None:
            #include this so that the parent class GWBModel doesn't complain
            return {'omega_ref': None, 'fbreak': None,'alpha_1': None, 'alpha_2': None, 'delta': None}
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError('unexpected type for parameters {}'.format(type(parameters)))

    def model_function(self, bline):
        return self.parameters['omega_ref'] * (bline.frequencies / self.parameters['fbreak'])**self.parameters['alpha_1'] * (1+(bline.frequencies / self.parameters['fbreak'])**(self.parameters['delta']))**((self.parameters['alpha_2']-self.parameters['alpha_1'])/self.parameters['delta'])


class SchumannModel(GWBModel):
    """
    Parameters:
    -----------
    fref : float
        reference frequency for defining the model
    kappa_i : float
        amplitude of coupling function of ifo i at 10 Hz
    beta_i : float
        spectral index of coupling function of ifo i
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """
    def __init__(self, **kwargs):
        # Set valid ifos
        # get ifo's associated with this set of
        # baselines.
        valid_ifos = ['H','L','V','K']
        self.ifos = []
        for bline in kwargs["baselines"]:
            if bline.name[0] not in valid_ifos:
                raise ValueError('baseline names must be two ifo letters from list of H, L, V, K')
            if bline.name[1] not in valid_ifos:
                raise ValueError('baseline names must be two ifo letters from list of H, L, V, K')
        # get list of ifos
        for bline in kwargs["baselines"]:
            self.ifos.append(bline.name[0])
            self.ifos.append(bline.name[1])
        super(SchumannModel, self).__init__(**kwargs)
        # set error handling to make sure baselines
        # are named properly
        # make it unique
        # be careful, this doesn't preserve
        # the order!
        self.ifos = list(set(self.ifos))

    @property
    def parameters(self):
        if self._parameters is None:
            schu_params={}
            for ifo in self.ifos:
                schu_params['kappa_{}'.format(ifo)]= None
                schu_params['beta_{}'.format(ifo)]= None
            return schu_params
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError('unexpected type for parameters {}'.format(type(parameters)))

    def model_function(self, bline):
        #assumes simple power law model for transfer function
        ifo1 = bline.name[0]
        ifo2 = bline.name[1]
        TF1 = self.parameters['kappa_{}'.format(ifo1)] *  (bline.frequencies / 10)**(-self.parameters['beta_{}'.format(ifo1)]) * 1e-23
        TF2 = self.parameters['kappa_{}'.format(ifo2)] *  (bline.frequencies / 10)**(-self.parameters['beta_{}'.format(ifo2)]) * 1e-23
        #units of transfer function strain/pT
        return TF1 * TF2 * np.real(bline.M_f)

    
class TVSPowerLawModel(GWBModel):
    """
    Parameters:
    -----------
    fref : float
        reference frequency for defining the model
    omega_ref_pol : float
        amplitude of signal at fref for polarization pol
    alpha_pol : float
        spectral index of the power law for polarization pol
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """    
    def __init__(self, **kwargs):
        try:
            fref = kwargs.pop('fref')
        except KeyError:
            raise KeyError('fref must be supplied')
        super(TVSPowerLawModel, self).__init__(**kwargs)
        self.fref = fref

    @property
    def parameters(self):
        if self._parameters is None:
            params={}
            for pol in self.polarizations:
                params['omega_ref_{}'.format(pol)]= None
                params['alpha_{}'.format(pol)]= None
            return params
        elif isinstance(self._parameters, dict):
            return self._parameters
            
    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError('unexpected type for parameters {}'.format(type(parameters)))

    def model_function(self, bline):
        model = np.zeros(bline.frequencies.shape)
        for pol in self.polarizations:
            orf_pol = eval('bline.{}_overlap_reduction_function'.format(pol))
            orf_parent = eval('bline.{}_overlap_reduction_function'.format(self.polarizations[0]))
            model += orf_pol/orf_parent * self.parameters['omega_ref_{}'.format(pol)] * (bline.frequencies / self.fref)**(self.parameters['alpha_{}'.format(pol)])
        return model

#Parity violation models

class PVPowerLawModel(GWBModel):
    """
    Parameters:
    -----------
    fref : float
        reference frequency for defining the model
    omega_ref : float
        amplitude of signal at fref
    alpha : float
        spectral index of the power law
    Pi : float
        degree of parity violation
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """
    def __init__(self, **kwargs):
        try:
            fref = kwargs.pop('fref')
        except KeyError:
            raise KeyError("fref must be supplied")      
        super(PVPowerLawModel, self).__init__(**kwargs)
        self.fref = fref

    @property
    def parameters(self):
        if self._parameters is None:
            #include this so that the parent class GWBModel doesn't complain
            return {'omega_ref': None,'alpha': None, 'Pi': None}
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError('unexpected type for parameters {}'.format(type(parameters)))
  
    def model_function(self, bline):
        return (1 + self.parameters['Pi'] * bline.gamma_v / bline.overlap_reduction_function) * self.parameters['omega_ref']  * (bline.frequencies / self.fref)**(self.parameters['alpha'])
    
class PVPowerLawModel2(GWBModel):
    """
    Parameters:
    -----------
    fref : float
        reference frequency for defining the model
    omega_ref : float
        amplitude of signal at fref
    alpha : float
        spectral index of the power law
    beta : float
        spectral index of the degree of parity violation
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """
    def __init__(self, **kwargs):
        try:
            fref = kwargs.pop('fref')
        except KeyError:
            raise KeyError("fref must be supplied")            
        super(PVPowerLawModel2, self).__init__(**kwargs)
        self.fref = fref

    @property
    def parameters(self):
        if self._parameters is None:
            #include this so that the parent class GWBModel doesn't complain
            return {'omega_ref': None,'alpha': None, 'beta': None}
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError('unexpected type for parameters {}'.format(type(parameters)))

    def model_function(self, bline):
        return  (1 + ((bline.frequencies)**(self.parameters['beta'])) * bline.gamma_v/ (bline.overlap_reduction_function)) * self.parameters['omega_ref'] * (bline.frequencies / self.fref)**(self.parameters['alpha'])
