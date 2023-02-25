============
Parameter estimation using ``pygwb_pe``
============

**1. ``pygwb_pe`` and its parameters**
=========

You can see the script's run options by running

.. code-block:: shell

   pygwb_pe --help
   
This will show all the parameters of the script as follows: 

.. code-block:: shell

    usage: pygwb_pe [-h] --path_to_file PATH_TO_FILE [--ifos IFOS] [--model MODEL] [--model_prior_file MODEL_PRIOR_FILE] [--non_prior_args NON_PRIOR_ARGS] [--output_dir OUTPUT_DIR]
                [--injection_parameters INJECTION_PARAMETERS] [--quantiles QUANTILES] [--f0 F0] [--fhigh FHIGH] [--df DF]

    optional arguments:
      -h, --help            show this help message and exit
      --path_to_file PATH_TO_FILE
                            Path to data file (or pickled baseline) to use for analysis.
      --ifos IFOS           List of names of two interferometers for which you want to run PE, default is H1 and L1. Not needed when running from a pickled baseline.
      --model MODEL         Model for which to run the PE. Default is "Power-law".
      --model_prior_file MODEL_PRIOR_FILE
                            The required parameters for the model and their priors. Default value is power-law parameters omega_ref with LogUniform bilby prior from 1e-11 to 1e-8 and alpha with Uniform bilby
                            prior from -4 to 4.
      --non_prior_args NON_PRIOR_ARGS
                            A dictionary with the parameters of the model that are not associated with a prior, such as the reference frequency for Power-Law. Default value is reference frequency at 25 Hz
                            for the power-law model.
      --output_dir OUTPUT_DIR
                            Output directory of the PE (sampler). Default: ./PE_Output.
      --injection_parameters INJECTION_PARAMETERS
                            The injected parameters.
      --quantiles QUANTILES
                            The quantiles used for plotting in plot corner, default is [0.05, 0.95].
      --f0 F0               If no frequencies are saved in the loaded npz file, you can give f0, fhigh and df to the script. This is f0
      --fhigh FHIGH         If no frequencies are saved in the loaded npz file, you can give f0, fhigh and df to the script. This is fhigh
      --df DF               If no frequencies are saved in the loaded npz file, you can give f0, fhigh and df to the script. This is df
      
The only required parameter is ``path_to_file``, which refers to an output file from a ``pygwb`` analysis run. The script can be run simply with

.. code-block:: shell

    pygwb_pe --path_to_file {path_to_pygwb_output_file}
    
This produces the usual output files of a ``bilby`` parameter estimation run and is saved in the ``output_dir``, which by default is ``./PE_Output``. 
Note that upon successful completion of the parameter estimation the ``output_dir`` should contain a ``result.json`` file and a ``bilby`` corner plot (in ``png`` format) summarising the results of the PE run.

**2. Customizing a ``pygwb_pe`` run**
=========

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
    









