from abc import abstractmethod

import bilby
import numpy as np
from scipy.special import erf

from .baseline import Baseline


class GWBModel(bilby.Likelihood):
    r"""
    Generic model, contains the definition of likelihood:

    .. math::

        p(\hat{C}^{IJ}(f_k) | \mathbf{\Theta}) \propto\exp\left[  -\frac{1}{2} \sum_{IJ}^N \sum_k \left(\frac{\hat{C}^{IJ}(f_k) - \Omega_{\rm M}(f_k|\mathbf{\Theta})}{\sigma^2_{IJ}(f_k)}\right)^2  \right],

    where :math:`\Omega_{\rm M}(f_k|\mathbf{\Theta})` is the model being fit to data, and :math:`\mathbf{\Theta}` are the model's parameters.

    The noise likelihood is given by setting :math:`\Omega_{\rm M}(f_k|\mathbf{\Theta})=0`.

    """

    def __init__(self, baselines=None, model_name=None, polarizations=None):
        super(GWBModel, self).__init__()
        # list of baselines
        if baselines is None:
            raise ValueError("list of baselines must be supplied")
        if model_name is None:
            raise ValueError("model_name must be supplied")
        # if single baseline supplied, that's fine
        # just make it a list
        if isinstance(baselines, Baseline):
            baselines = [baselines]
        self.baselines = baselines
        self.orfs = []
        self.polarizations = polarizations
        # if polarizations is not supplied
        if polarizations is None:
            self.polarizations = ["tensor" for ii in range(len(baselines))]
        for bline in self.baselines:
            if self.polarizations[0].lower() == "tensor":
                self.orfs.append(bline.tensor_overlap_reduction_function)
            elif self.polarizations[0].lower() == "vector":
                self.orfs.append(bline.vector_overlap_reduction_function)
            elif self.polarizations[0].lower() == "scalar":
                self.orfs.append(bline.scalar_overlap_reduction_function)
            else:
                raise ValueError(
                    f"unexpected type for polarizations {type(polarizations)}"
                )
        bilby.Likelihood.__init__(self, parameters=self.parameters)

    @abstractmethod
    def parameters(self):
        """Parameters. Should return a dictionary"""
        pass

    @abstractmethod
    def model_function(self):
        """Function for evaluating model"""
        pass

    def log_likelihood_IJ(self, baseline, noise=False):
        """Function for evaluating log likelihood of IJ baseline pair"""
        if noise:
            Y_model_f = 0
        else:
            Y_model_f = self.model_function(baseline)

        # simple likelihood without calibration uncertainty
        if baseline.calibration_epsilon == 0:
            logL_IJ = -0.5 * (
                np.sum(
                    (baseline.point_estimate_spectrum - Y_model_f) ** 2
                    / baseline.sigma_spectrum ** 2
                )
                + np.sum(np.log(2 * np.pi * baseline.sigma_spectrum ** 2))
            )

        # likelihood with calibration uncertainty marginalizatione done analytically
        # see https://stochastic-alog.ligo.org/aLOG//index.php?callRep=339711
        # note \cal{N} = \Prod_j sqrt(2*pi*sigma_j^2)
        else:
            A = baseline.calibration_epsilon ** (-2) + np.sum(
                Y_model_f ** 2 / baseline.sigma_spectrum ** 2
            )
            B = baseline.calibration_epsilon ** (-2) + np.sum(
                Y_model_f
                * baseline.point_estimate_spectrum
                / baseline.sigma_spectrum ** 2
            )
            C = baseline.calibration_epsilon ** (-2) + np.sum(
                baseline.point_estimate_spectrum ** 2 / baseline.sigma_spectrum ** 2
            )
            log_norm = -0.5 * np.sum(np.log(2 * np.pi * baseline.sigma_spectrum ** 2))

            logL_IJ = (
                log_norm
                - 0.5 * np.log(A * baseline.calibration_epsilon ** 2)
                + np.log(1 + erf(B / np.sqrt(2 * A)))
                - np.log(1 + erf(1 / np.sqrt(2 * baseline.calibration_epsilon ** 2)))
                - 0.5 * (C - B ** 2 / A)
            )

        return logL_IJ

    def log_likelihood(self):
        """Function for evaluating log likelihood of detector network"""
        ll = 0
        for baseline in self.baselines:
            ll = ll + self.log_likelihood_IJ(baseline, noise=False)
        return ll

    def noise_log_likelihood(self):
        """Function for evaluating noise log likelihood of detector network"""
        ll = 0
        for baseline in self.baselines:
            ll = ll + self.log_likelihood_IJ(baseline, noise=True)
        return ll


class PowerLawModel(GWBModel):
    r"""
    Power law model is defined as:

    .. math::
        \Omega(f) = \Omega_{\text{ref}} \left(\frac{f}{f_{\text{ref}}}\right)^{\alpha}
    Parameters:
    -----------
    fref : float
        reference frequency for defining the model (:math:`f_{\text{ref}}`)
    omega_ref : float
        amplitude of signal at fref (:math:`\Omega_{\text{ref}}`)
    alpha : float
        spectral index of the power law (:math:`\alpha`)
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """

    def __init__(self, **kwargs):
        try:
            fref = kwargs.pop("fref")
        except KeyError:
            raise KeyError("fref must be supplied")
        super(PowerLawModel, self).__init__(**kwargs)
        self.fref = fref

    @property
    def parameters(self):
        if self._parameters is None:
            # include this so that the parent class GWBModel doesn't complain
            return {"omega_ref": None, "alpha": None}
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError(f"unexpected type for parameters {type(parameters)}")

    def model_function(self, bline):
        return (
            self.parameters["omega_ref"]
            * (bline.frequencies / self.fref) ** self.parameters["alpha"]
        )


class BrokenPowerLawModel(GWBModel):
    r"""
    Broken Power law model is defined as: 
    
    .. math:: 
        \Omega(f) = \begin{cases}
            \Omega_{\text{ref}} \left( \frac{f}{f_{\text{ref}}} \right) ^ {\alpha_1}, f \leqslant f_{\text{ref}} \\
            \Omega_{\text{ref}} \left( \frac{f}{f_{\text{ref}}} \right) ^ {\alpha_2}, f > f_{\text{ref}}
        \end{cases}
    Parameters:
    -----------
    omega_ref : float
        amplitude of signal at fref (:math:`\Omega_{\text{ref}}`)
    alpha_1 : float
        spectral index of the broken power law (:math:`\alpha_1`)
    alpha_2 : float
        spectral index of the broken power law (:math:`\alpha_2`)
    fbreak : float
        break frequency for the broken power law (:math:`f_{\text{ref}}`)
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """

    def __init__(self, **kwargs):
        super(BrokenPowerLawModel, self).__init__(**kwargs)

    @property
    def parameters(self):
        if self._parameters is None:
            # include this so that the parent class GWBModel doesn't complain
            return {"omega_ref": None, "fbreak": None, "alpha_1": None, "alpha_2": None}
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError(f"unexpected type for parameters {type(parameters)}")

    def model_function(self, bline):
        frequencies = bline.frequencies
        return self.parameters["omega_ref"] * (
            np.piecewise(
                frequencies,
                [
                    frequencies <= self.parameters["fbreak"],
                    frequencies > self.parameters["fbreak"],
                ],
                [
                    lambda frequencies: (frequencies / self.parameters["fbreak"])
                    ** (self.parameters["alpha_1"]),
                    lambda frequencies: (frequencies / self.parameters["fbreak"])
                    ** (self.parameters["alpha_2"]),
                ],
            )
        )


class TripleBrokenPowerLawModel(GWBModel):
    r"""
    The triple broken power law is defined as: 
    
    .. math:: 
        \Omega(f) = \begin{cases}
            \Omega_{\text{ref}} \left( \frac{f}{f_1} \right) ^ {\alpha_1}, f \leqslant f_1 \\
            \Omega_{\text{ref}} \left( \frac{f}{f_1} \right) ^ {\alpha_2}, f_1 < f \leqslant f_2  \\
            \Omega_{\text{ref}} \left( \frac{f_2}{f_1} \right) ^{\alpha_2} \left( \frac{f}{f_2} \right)^{\alpha_3}, f > f_2
        \end{cases}
    Parameters:
    -----------
    omega_ref : float
        amplitude of signal at fref (:math:`\Omega_{\text{ref}}`) 
    alpha_1 : float
        spectral index of the broken power law (:math:`\alpha_1`)
    alpha_2 : float
        spectral index of the broken power law (:math:`\alpha_2`)
    alpha_3 : float
        spectral index of the broken power law (:math:`\alpha_3`)
    fbreak1 : float
        1st break frequency for the triple broken power law (:math:`f_1`)
    fbreak2 : float
        2nd break frequency for the triple broken power law (:math:`f_2`)
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """

    def __init__(self, **kwargs):
        super(TripleBrokenPowerLawModel, self).__init__(**kwargs)

    @property
    def parameters(self):
        if self._parameters is None:
            # include this so that the parent class GWBModel doesn't complain
            return {
                "omega_ref": None,
                "alpha_1": None,
                "alpha_2": None,
                "alpha_3": None,
                "fbreak1": None,
                "fbreak2": None,
            }
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError(f"unexpected type for parameters {type(parameters)}")

    def model_function(self, bline):
        frequencies = bline.frequencies
        piecewise = np.piecewise(
            frequencies,
            [
                frequencies <= self.parameters["fbreak1"],
                (frequencies <= self.parameters["fbreak2"])
                & (frequencies > self.parameters["fbreak1"]),
            ],
            [
                lambda frequencies: (frequencies / self.parameters["fbreak1"])
                ** (self.parameters["alpha_1"]),
                lambda frequencies: (frequencies / self.parameters["fbreak1"])
                ** (self.parameters["alpha_2"]),
                lambda frequencies: (
                    self.parameters["fbreak2"] / self.parameters["fbreak1"]
                )
                ** (self.parameters["alpha_2"])
                * (frequencies / self.parameters["fbreak2"])
                ** (self.parameters["alpha_3"]),
            ],
        )
        return self.parameters["omega_ref"] * piecewise


class SmoothBrokenPowerLawModel(GWBModel):
    r"""

    The smooth broken power law is defined as:

    .. math::
        \Omega(f) = \Omega_{\text{ref}}\left(\frac{f}{f_{\text{ref}}}\right) ^{\alpha_1} \left[1+\left(\frac{f}{f_{\text{ref}}}\right)^{\Delta}\right]^{\frac{\alpha_2-\alpha_1}{\Delta}}
    Parameters:
    -----------
    omega_ref : float
        amplitude of signal (:math:`\Omega_{\text{ref}}`)
    Delta : float
        smoothing variable for the smooth broken power law (:math:`\Delta`)
    alpha_1 : float
        low-frequency spectral index of the smooth broken power law (:math:`\alpha_1`)
    alpha_2 : float
        high-frequency spectral index of the smooth broken power law (:math:`\alpha_2`)
    fbreak : float
        break frequency for the smooth broken power law (:math:`f_{\text{ref}}`)
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """

    def __init__(self, **kwargs):
        super(SmoothBrokenPowerLawModel, self).__init__(**kwargs)

    @property
    def parameters(self):
        if self._parameters is None:
            # include this so that the parent class GWBModel doesn't complain
            return {
                "omega_ref": None,
                "fbreak": None,
                "alpha_1": None,
                "alpha_2": None,
                "delta": None,
            }
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError(f"unexpected type for parameters {type(parameters)}")

    def model_function(self, bline):
        return (
            self.parameters["omega_ref"]
            * (bline.frequencies / self.parameters["fbreak"])
            ** self.parameters["alpha_1"]
            * (
                1
                + (bline.frequencies / self.parameters["fbreak"])
                ** (self.parameters["delta"])
            )
            ** (
                (self.parameters["alpha_2"] - self.parameters["alpha_1"])
                / self.parameters["delta"]
            )
        )


class SchumannModel(GWBModel):
    r"""

    The Schumann model is defined as:


    .. math::
         \Omega(f) = \sum_{ij} \kappa_i \kappa_j \left(\frac{f}{f_{\text{ref}}}\right)^{-\beta_i-\beta_j} M_{ij}(f) \times 10^{-46}
    Parameters:
    -----------
    fref : float
        reference frequency for defining the model (:math:`f_{\text{ref}}`)
    kappa_i : float
        amplitude of coupling function of interferometer i at 10 Hz (:math:`\kappa_i`)
    beta_i : float
        spectral index of coupling function of interferometer i (:math:`\beta_i`)
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """

    def __init__(self, **kwargs):
        # Set valid ifos
        # get ifo's associated with this set of
        # baselines.
        valid_ifos = ["H", "L", "V", "K"]
        self.ifos = []
        for bline in kwargs["baselines"]:
            if bline.name[0] not in valid_ifos:
                raise ValueError(
                    "baseline names must be two ifo letters from list of H, L, V, K"
                )
            if bline.name[1] not in valid_ifos:
                raise ValueError(
                    "baseline names must be two ifo letters from list of H, L, V, K"
                )
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
            schu_params = {}
            for ifo in self.ifos:
                schu_params[f"kappa_{ifo}"] = None
                schu_params[f"beta_{ifo}"] = None
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
            raise ValueError(f"unexpected type for parameters {type(parameters)}")

    def model_function(self, bline):
        # assumes simple power law model for transfer function
        ifo1 = bline.name[0]
        ifo2 = bline.name[1]
        TF1 = (
            self.parameters[f"kappa_{ifo1}"]
            * (bline.frequencies / 10) ** (-self.parameters[f"beta_{ifo1}"])
            * 1e-23
        )
        TF2 = (
            self.parameters[f"kappa_{ifo2}"]
            * (bline.frequencies / 10) ** (-self.parameters[f"beta_{ifo2}"])
            * 1e-23
        )
        # units of transfer function strain/pT
        return TF1 * TF2 * np.real(bline.M_f)


class TVSPowerLawModel(GWBModel):
    r"""
    The Tensor-Vector-Scalar polarization (T,V,S) power-law model is defined as:

    .. math::

        \Omega(f) = \Omega _T + \Omega _V + \Omega _S

        \Omega _T = \Omega _{{\text{ref}},T} \left( \frac{f}{f_{\text{ref}}}\right)^{\alpha _T}

        \Omega _V = (\gamma _V/\gamma_T)~\Omega _{{\text{ref}},V} \left( \frac{f}{f_{\text{ref}}}\right)^{\alpha _V}

        \Omega _S = (\gamma_S/\gamma_T)~\Omega _{{\text{ref}},S} \left( \frac{f}{f_{\text{ref}}}\right)^{\alpha _S}
    Parameters:
    -----------
    fref : float
        reference frequency for defining the model (:math:`f_{\text{ref}}`)
    omega_ref_pol : float
        amplitude of signal at fref for polarization pol (:math:`\Omega_{\text{ref},\text{pol}}`)
    alpha_pol : float
        spectral index of the power law for polarization pol (:math:`\alpha_{\text{pol}}`)
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """

    def __init__(self, **kwargs):
        try:
            fref = kwargs.pop("fref")
        except KeyError:
            raise KeyError("fref must be supplied")
        super(TVSPowerLawModel, self).__init__(**kwargs)
        self.fref = fref

    @property
    def parameters(self):
        if self._parameters is None:
            params = {}
            for pol in self.polarizations:
                params[f"omega_ref_{pol}"] = None
                params[f"alpha_{pol}"] = None
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
            raise ValueError(f"unexpected type for parameters {type(parameters)}")

    def model_function(self, bline):
        model = np.zeros(bline.frequencies.shape)
        for pol in self.polarizations:
            orf_pol = eval(f"bline.{pol}_overlap_reduction_function")
            orf_parent = eval(
                f"bline.{self.polarizations[0]}_overlap_reduction_function"
            )
            model += (
                orf_pol
                / orf_parent
                * self.parameters[f"omega_ref_{pol}"]
                * (bline.frequencies / self.fref) ** (self.parameters[f"alpha_{pol}"])
            )
        return model


# Parity violation models


class PVPowerLawModel(GWBModel):
    r"""
    The parity violation model can be defined as:

    .. math::
        \Omega(f) = \left(1 + \Pi \frac{\gamma _V}{\gamma _I}\right) \Omega_{\text{ref}} \left( \frac{f}{f_{\text{ref}}} \right)^{\alpha}
    Parameters:
    -----------
    fref : float
        reference frequency for defining the model (:math:`f_{\text{ref}}`)
    omega_ref : float
        amplitude of signal at fref (:math:`\Omega_{\text{ref}}`)
    alpha : float
        spectral index of the power law (:math:`\alpha`)
    Pi : float
        degree of parity violation (:math:`\Pi`)
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """

    def __init__(self, **kwargs):
        try:
            fref = kwargs.pop("fref")
        except KeyError:
            raise KeyError("fref must be supplied")
        super(PVPowerLawModel, self).__init__(**kwargs)
        self.fref = fref

    @property
    def parameters(self):
        if self._parameters is None:
            # include this so that the parent class GWBModel doesn't complain
            return {"omega_ref": None, "alpha": None, "Pi": None}
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError(f"unexpected type for parameters {type(parameters)}")

    def model_function(self, bline):
        return (
            (
                1
                + self.parameters["Pi"]
                * bline.gamma_v
                / bline.overlap_reduction_function
            )
            * self.parameters["omega_ref"]
            * (bline.frequencies / self.fref) ** (self.parameters["alpha"])
        )


class PVPowerLawModel2(GWBModel):
    r"""
    The parity violation model 2 can be defined as:

    .. math::
         \Omega(f) = \left(1 + f^{\beta} \frac{\gamma_V}{\gamma _I}\right) \Omega_{\text{ref}}\left(\frac{f}{f_{\text{ref}}} \right)^{\alpha}
    Parameters:
    -----------
    fref : float
        reference frequency for defining the model (:math:`f_{\text{ref}}`)
    omega_ref : float
        amplitude of signal at fref (:math:`\Omega_{\text{ref}}`)
    alpha : float
        spectral index of the power law (:math:`\alpha`)
    beta : float
        spectral index of the degree of parity violation (:math:`\beta`)
    frequencies : numpy.ndarray
        array of frequencies at which to evaluate the model
    """

    def __init__(self, **kwargs):
        try:
            fref = kwargs.pop("fref")
        except KeyError:
            raise KeyError("fref must be supplied")
        super(PVPowerLawModel2, self).__init__(**kwargs)
        self.fref = fref

    @property
    def parameters(self):
        if self._parameters is None:
            # include this so that the parent class GWBModel doesn't complain
            return {"omega_ref": None, "alpha": None, "beta": None}
        elif isinstance(self._parameters, dict):
            return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is None:
            self._parameters = parameters
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise ValueError(f"unexpected type for parameters {type(parameters)}")

    def model_function(self, bline):
        return (
            (
                1
                + ((bline.frequencies) ** (self.parameters["beta"]))
                * bline.gamma_v
                / (bline.overlap_reduction_function)
            )
            * self.parameters["omega_ref"]
            * (bline.frequencies / self.fref) ** (self.parameters["alpha"])
        )
