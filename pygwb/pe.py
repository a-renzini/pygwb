import bilby
import numpy as np
from scipy.special import erf


class GWBLikelihood(bilby.Likelihood):
    def __init__(self, baselines, parameters=None):
        """
        A parent class for GWB data analysis with Bilby

        Parameters
        ----------
        baselines: a list of Baseline objects containing cross correlation data.
        """
        super(GWBLikelihood, self).__init__(parameters=parameters)
        self.baselines = baselines

    def log_likelihood_IJ(self, baseline, noise=False):
        if noise:
            Y_model_f = 0
        else:
            Y_model_f = self.OmegaGW(baseline.freqs)

        # simple likelihood without calibration uncertainty
        if baseline.calibration_epsilon == 0:
            logL_IJ = -0.5 * (
                np.sum((baseline.Y_f - Y_model_f) ** 2 / baseline.var_f)
                + np.sum(np.log(2 * np.pi * baseline.var_f))
            )

        # likelihood with calibration uncertainty marginalizatione done analytically
        # see https://stochastic-alog.ligo.org/aLOG//index.php?callRep=339711
        # note \cal{N} = \Prod_j sqrt(2*pi*sigma_j^2)
        else:
            A = baseline.calibration_epsilon ** (-2) + np.sum(
                Y_model_f ** 2 / baseline.var_f
            )
            B = baseline.calibration_epsilon ** (-2) + np.sum(
                Y_model_f * baseline.Y_f / baseline.var_f
            )
            C = baseline.calibration_epsilon ** (-2) + np.sum(
                baseline.Y_f ** 2 / baseline.var_f
            )
            log_norm = -0.5 * np.sum(np.log(2 * np.pi * baseline.var_f))

            logL_IJ = (
                log_norm
                - 0.5 * np.log(A * baseline.calibration_epsilon ** 2)
                + np.log(1 + erf(B / np.sqrt(2 * A)))
                - np.log(1 + erf(1 / np.sqrt(2 * baseline.calibration_epsilon ** 2)))
                - 0.5 * (C - B ** 2 / A)
            )

        return logL_IJ

    def log_likelihood(self):
        logL = 0
        for baseline in self.baselines:
            logL = logL + self.log_likelihood_IJ(baseline, noise=False)
        return logL

    def noise_log_likelihood(self):
        logL = 0
        for baseline in self.baselines:
            logL = logL + self.log_likelihood_IJ(baseline, noise=True)
        return logL

    def OmegaGW(self, freqs):
        """
        Subclasses should implement this function
        """
        pass


class PowerLawGWBLikelihood(GWBLikelihood):
    """
    Power law GWB model

    Omega_GW(f|{A,alpha,fref=fixed}) = A * (f/fref)**alpha
    """

    def __init__(self, baselines, fref=1):
        super(PowerLawGWBLikelihood, self).__init__(
            baselines, parameters={"A": None, "alpha": None}
        )
        self.fref = fref

    def OmegaGW(self, freqs):
        A = self.parameters["A"]
        alpha = self.parameters["alpha"]
        return A * (freqs / self.fref) ** alpha


class BrokenPowerLawGWBLikelihood(GWBLikelihood):
    """
    Broken power law GWB model

    Omega_GW(f|{A,alpha1,alpha2,fbreak,fref=fixed}) = ...
          if fref < fbreak:
              A * (f/fref)**alpha1, f<fbreak
              A * (fbreak/fref)**(alpha1-alpha2) *(f/fref)**alpha2, f>fbreak
          if fref > fbreak:
              A * (fbreak/fref)**(alpha2-alpha1) * (f/fref)**alpha1, f<fbreak
              A * (f/fref)**alpha2, f>fbreak


    This is parameterized so that:
        Omega_GW(fref) = A (regardless of the value of fbreak)
        Omega_GW is continuous at fbreak
    It probably makes sense to chose a convention where fref is always very small
    or very large compared to fbreak

    """

    def __init__(self, baselines, fref):
        super(BrokenPowerLawGWBLikelihood, self).__init__(
            baselines,
            parameters={"A": None, "alpha1": None, "alpha2": None, "fbreak": None},
        )
        self.fref = fref

    def OmegaGW(self, freqs):
        A = self.parameters["A"]
        alpha1 = self.parameters["alpha1"]
        alpha2 = self.parameters["alpha2"]
        fbreak = self.parameters["fbreak"]
        Omega = np.zeros(freqs.shape)
        fref = self.fref

        if fref <= fbreak:
            Omega[freqs <= fbreak] = A * (freqs[freqs <= fbreak] / fref) ** alpha1
            Omega[freqs > fbreak] = (
                A
                * (fbreak / fref) ** (alpha1 - alpha2)
                * (freqs[freqs > fbreak] / fref) ** alpha2
            )
        else:
            Omega[freqs <= fbreak] = (
                A
                * (fbreak / fref) ** (alpha2 - alpha1)
                * (freqs[freqs <= fbreak] / fref) ** alpha1
            )
            Omega[freqs > fbreak] = A * (freqs[freqs <= fbreak] / fref) ** alpha2
        return Omega


class SVTPowerLawGWBLikelihood(GWBLikelihood):
    """
    Power law GWB model

    Omega_GW(f|{A_p,alpha_p,fref=fixed,ORFs=fixed}) = sum_IJ sum_p A_p * beta^p_IJ(f) * (f/fref)**alpha_p
    where
    beta^p_IJ(f) = gamma^p_IJ(f) / gamma_IJ(f) is the ratio of ORFs
    """

    def __init__(self, baselines, fref=1):
        super(SVTPowerLawGWBLikelihood, self).__init__(
            baselines,
            parameters={
                "AT": None,
                "alphaT": None,
                "AV": None,
                "alphaV": None,
                "AS": None,
                "alphaS": None,
            },
        )
        self.fref = fref

    def OmegaGW(self, freqs):
        A = self.parameters["A"]
        alpha = self.parameters["alpha"]
        return A * (freqs / self.fref) ** alpha


class BasicGWBLikelihood(GWBLikelihood):
    """
    Basic version of GWB Likelihood, in which the user
    just provides Y_f, sigma_f, and freqs, and doesn't need
    to worry about the Baseline class
    """

    def __init__(self, Y_f, var_f, freqs, parameters=None):
        baselines = [Baseline(None, Y_f, var_f, freqs)]
        super(BasicGWBLikelihood, self).__init__(baselines, parameters=parameters)


class BasicPowerLawGWBLikelihood(BasicGWBLikelihood):
    """
    Power law GWB model
    """

    def __init__(self, Y_f, var_f, freqs, fref=1):
        super(BasicPowerLawGWBLikelihood, self).__init__(
            Y_f, var_f, freqs, parameters={"A": None, "alpha": None}
        )
        self.fref = fref

    def OmegaGW(self, freqs):
        A = self.parameters["A"]
        alpha = self.parameters["alpha"]
        return A * (freqs / self.fref) ** alpha
