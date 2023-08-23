=======================================
Parameter estimation using ``pygwb_pe``
=======================================

The ``pygwb_pe`` script performs parameter estimation (PE) as an integral part of the
analysis, which naturally follows the computation of the optimal estimate of the gravitational-wave background (GWB) performed in ``pygwb``.
Parameter estimation consists of inferring parameters from the data, using Bayesian inference. Concretely, one uses the likelihood

.. math::

    p(\hat{C}^{IJ}(f_k) | \mathbf{\Theta}) \propto\exp\left[  -\frac{1}{2} \sum_{IJ}^N \sum_k \left(\frac{\hat{C}^{IJ}(f_k) - \Omega_{\rm M}(f_k|\mathbf{\Theta})}{\sigma^2_{IJ}(f_k)}\right)^2  \right],

where :math:`\Omega_{\rm M}(f_k|\mathbf{\Theta})` is the model being fit to data, and :math:`\mathbf{\Theta}` are the model's parameters.
This allows to infer, or put constraints on, some GWB model parameters :math:`\mathbf{\Theta}`, given the optimal GWB estimator :math:`\hat{C}^{IJ}(f_k)` computed
by ``pygwb``.

The ``pygwb`` package comes with a script, ``pygwb_pe``, which wraps the ``pygwb.pe`` module. In this tutorial, we show how to
use this script and escribe the basic functionalities. However, we show some additional features of the model in one of the demos, 
to illustrate the customizability of the module.

.. tip::

    Make sure to have a look at the ``pygwb.pe`` `module page <api/pygwb.pe.html>`_ before going through the tutorial for additional information.


**1. Script parameters**
========================

The input paramaters of the ``pygwb_pe`` script  can be accessed by running

.. code-block:: shell

   pygwb_pe --help
   
This will display the parameters of the script as follows: 

.. code-block:: shell

    pygwb_pe [-h] --input_file INPUT_FILE [--ifos IFOS] [--model MODEL][--model_prior_file MODEL_PRIOR_FILE]
    [--non_prior_args NON_PRIOR_ARGS] [--outdir OUTDIR][--apply_notching APPLY_NOTCHING]
    [--notch_list_path NOTCH_LIST_PATH][--injection_parameters INJECTION_PARAMETERS]
    [--quantiles QUANTILES][--calibration_epsilon CALIBRATION_EPSILON]

    optional arguments:
        -h, --help      show this help message and exit
        --input_file INPUT_FILE
                        Path to data file (or pickled baseline) to use for analysis.
        --ifos IFOS     List of names of two interferometers for which you
                        want to run PE, default is H1 and L1. Not needed when
                        running from a pickled baseline.
        --model MODEL   Model for which to run the PE. Default is "Power-law".
        --model_prior_file MODEL_PRIOR_FILE
                        Points to a file with the required parameters for the
                        model and their priors. Default value is power-law
                        parameters omega_ref with LogUniform bilby prior from
                        1e-11 to 1e-8 and alpha with Uniform bilby prior from
                        -4 to 4.
        --non_prior_args NON_PRIOR_ARGS
                        A dictionary with the parameters of the model that are
                        not associated with a prior, such as the reference
                        frequency for Power-Law. Default value is reference
                        frequency at 25 Hz for the power-law model.
        --outdir OUTDIR 
                        Output directory of the PE (sampler). Default is "./pe_output".
        --apply_notching APPLY_NOTCHING
                        Apply notching to the data. Default is True.
        --notch_list_path NOTCH_LIST_PATH
                        Absolute path from where you can load in notch list
                        file.
        --injection_parameters INJECTION_PARAMETERS
                        The injected parameters.
        --quantiles QUANTILES
                        The quantiles used for plotting in plot corner.
                        Default is [0.05, 0.95].
        --calibration_epsilon CALIBRATION_EPSILON
                        Calibration uncertainty. Default is 0.

The only required parameter is ``path_to_file``, which refers to an output file from a ``pygwb`` analysis run. The script can be run simply with

.. code-block:: shell

    pygwb_pe --input_file {path_to_pygwb_output_file}
    
This produces the usual output files of a ``bilby`` parameter estimation run and is saved in the ``output_dir``, which by default is ``./PE_Output``. 
Note that upon successful completion of the parameter estimation the ``output_dir`` should contain a ``result.json`` file and a ``bilby`` corner plot 
(in ``png`` format) summarizing the results of the PE run.

**2. Customizing a run**
========================

Although the ``pygwb_pe`` can be run with the default setup as outlined above, 
most users will want to customize some of the parameters for the run. For example, one may choose to run using a different model and different priors. 
The model may be specified using the ``--model argument``, and must be a ``pygwb``--supported model.
These inculde

.. code-block:: python

        "Power-Law": pe.PowerLawModel
        "Broken-Power-Law": pe.BrokenPowerLawModel
        "Triple-Broken-Power-Law": pe.TripleBrokenPowerLawModel
        "Smooth-Broken-Power-Law": pe.SmoothBrokenPowerLawModel
        "Schumann": pe.SchumannModel
        "Parity-Violation": pe.PVPowerLawModel
        "Parity-Violation-2": pe.PVPowerLawModel2

We refer the user to the ``pe`` documentation for a full overview of available models. 
The priors may be specified by passing a ``prior.json`` file through the ``--model_prior_file`` argument. 

This is illustrated with an example below for power-law model priors. A Log-uniform prior for Omega from 
1e-13 to 1e-5 and a Gaussian prior for alpha with mean of 2/3 and sigma 1.5 are taken. Such a json file 
can be made using the following code:

.. code-block:: python

    import bilby
    
    priors = bilby.core.prior.PriorDict()
    
    priors['omega_ref'] = bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\rm ref}$')
    priors['alpha'] = bilby.core.prior.Gaussian(mu = 2/3, sigma = 1.5, latex_label = '$\\alpha$')
    
    priors.to_json({path_to_where_you_want_to_save_json}, label='pe')
    
Now you can run your script with the json file containing the information about the priors on the parameters 

.. code-block:: python

    pygwb_pe --path_to_file {path_to_pygwb_output_file} --model_prior_file {path_to_json_file} --model {model_you_want_to_examine}

.. tip::
    Feeling overwhelmed by this tutorial? Make sure to have a look at the ``pygwb.pe`` `module page <api/pygwb.pe.html>`_ for additional information
    about the methods of the module.