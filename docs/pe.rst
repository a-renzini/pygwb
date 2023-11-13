=======================================
Parameter estimation using ``pygwb_pe``
=======================================

The ``pygwb_pe`` script performs parameter estimation (PE) as an integral part of the
analysis, which naturally follows the computation of the optimal estimate of the gravitational-wave background (GWB) performed in ``pygwb``.
Parameter estimation consists of inferring parameters from the data, using Bayesian inference. Concretely, one uses the likelihood

.. math::

    p(\hat{C}^{IJ}(f_k) | \mathbf{\Theta}) \propto\exp\left[  -\frac{1}{2} \sum_{IJ}^N \sum_k \left(\frac{\hat{C}^{IJ}(f_k) - \Omega_{\rm M}(f_k|\mathbf{\Theta})}{\sigma_{IJ}(f_k)}\right)^2  \right],

where :math:`\Omega_{\rm M}(f_k|\mathbf{\Theta})` is the model being fit to data, and :math:`\mathbf{\Theta}` are the model's parameters.
This allows to infer, or put constraints on, some GWB model parameters :math:`\mathbf{\Theta}`, given the optimal GWB estimator :math:`\hat{C}^{IJ}(f_k)` computed
by ``pygwb``.

The ``pygwb`` package comes with a script, ``pygwb_pe``, which wraps the ``pygwb.pe`` module. In this tutorial, we show how to
use this script and describe the basic functionalities. In addition, we show some advanced features of the module in one of the `demos <run_pe.html>`_, 
to illustrate the customizability of the module.

.. tip::

    Make sure to also have a look at the ``pygwb.pe`` `module page <api/pygwb.pe.html>`_ before going through the tutorial for additional information.

.. note::
    The ``pygwb.pe`` module is based on the ``bilby`` package. For more information, we refer the reader to the documentation of the ``bilby`` package `here <https://lscsoft.docs.ligo.org/bilby/index.html>`_.

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

**2. Running the script**
=========================

The only required argument is the ``input_file``, which should point to the output file of a ``pygwb`` run containing the point estimate spectrum and its variance. 
The script can then simply be run with

.. code-block:: shell

    pygwb_pe --input_file {path_to_pygwb_output_file}
    
The above command would run ``pygwb_pe`` with all script parameter values set to their default values. However, the various script parameters of ``pygwb_pe``, 
as shown above, allow for a certain level of customization. In particular, the ``pygwb_pe`` script accommodates all the models present in the ``pygwb.pe`` module through the 
``--model`` argument (more information on available models `here <api/pygwb.pe.html>`_). These include:

.. code-block:: python

        "Power-Law": pe.PowerLawModel
        "Broken-Power-Law": pe.BrokenPowerLawModel
        "Triple-Broken-Power-Law": pe.TripleBrokenPowerLawModel
        "Smooth-Broken-Power-Law": pe.SmoothBrokenPowerLawModel
        "Schumann": pe.SchumannModel
        "Parity-Violation": pe.PVPowerLawModel
        "Parity-Violation-2": pe.PVPowerLawModel2

Depending on the model choice above, the prior on the model parameters will have to be modified as well. This is handled by a prior file (``json`` file format),
which contains the priors in a dictionary format used for PE (as expected by `bilby <https://lscsoft.docs.ligo.org/bilby/prior.html>`_ to run PE). To create such a file,
one can run the following lines of code (here for a power-law model):

.. code-block:: python

    import bilby
    
    priors = bilby.core.prior.PriorDict()
    
    priors['omega_ref'] = bilby.core.prior.LogUniform(1e-13, 1e-5, '$\\Omega_{\\rm ref}$')
    priors['alpha'] = bilby.core.prior.Gaussian(mu = 2/3, sigma = 1.5, latex_label = '$\\alpha$')
    
    priors.to_json({path_to_where_you_want_to_save_json}, label='pe') 

This file can then be passed to the script through the ``--model_prior_file`` argument:

.. code-block:: python

    pygwb_pe --path_to_file {path_to_pygwb_output_file} --model_prior_file {path_to_json_file} --model {chosenn_model}

.. warning::
    Make sure to specify all the model parameters of the chosen model in the code above to avoid errors when running the script.

.. tip::
    For more information about the model parameters, see the relevant API PE documentation `pages <api/pygwb.pe.html>`_. Additional information about ``bilby`` priors can be found
    `here <https://lscsoft.docs.ligo.org/bilby/prior.html>`_.

Other script arguments allow for further customization of the PE run. For example, a notch list can be passed through the ``--notch_list_path`` to exclude specific
frequency bins from the analysis. For more information on notching, we refer the reader to the `notch module <api/pygwb.notch.html>`_ API page.

**3. Output of the script**
===========================

The ``pygwb_pe`` script produces the usual output files of a ``bilby`` PE run and is saved in the ``--output_dir`` passed when running the script from
the command line (defaults to ``./pe_output``). This directory should contain a ``result.json`` file and a so-called corner plot (or posterior plot), 
in ``png`` format, summarizing the results of the PE run. For examples of these corner plots and additional information on the output, we refer the 
reader to the `bilby documentation <https://lscsoft.docs.ligo.org/bilby/bilby-output.html>`_.

Note that the output of the ``pygwb_pe`` script can be read in with dedicated ``bilby`` methods. For example, one can load a PE result as follows:

.. code-block:: python

    result = bilby.core.result.Result.from_json("my_file.json")

For additional information about the ``bilby.core.result`` object and its functionalities, we refer the reader to the 
`documentation of the class <https://lscsoft.docs.ligo.org/bilby/api/bilby.core.result.Result.html#bilby.core.result.Result>`_.

.. tip::
    Feeling overwhelmed by this tutorial? Make sure to have a look at the ``pygwb.pe`` `module page <api/pygwb.pe.html>`_ for additional information
    about the methods of the module.

.. seealso::

    For more information about how to customize your PE runs, make sure to check out the `PE demo <run_pe.html>`_.