========================================
Simulating data with ``pygwb.simulator``
========================================

In this tutorial, we show how the ``pygwb.simulator`` module can be used to simulate a gravitational-wave background. We consider both the case
where a power-law spectrum is injected, and where individual compact binary coalescences (CBCs) are injected. For more information about the module, we refer
the reader to the ``pygwb.simulator`` `documentation page <api/pygwb.simulator.html>`_.

**1. Simulating a stochastic gravitational-wave background**
============================================================

**1.1 Injecting a power spectrum in random LIGO noise**
-------------------------------------------------------

For concreteness, we consider a broken power-law signal power spectral density (PSD) to be injected in random LIGO Gaussian noise.  
The ``simulator`` module takes a ``gwpy.FrequencySeries`` as input for the signal PSD to be injected. 
We start by building a custom input signal by defining an ``IntensityGW`` function, which outputs the 
desired signal PSD in the form of a ``gwpy.FrequencySeries``.

.. code-block:: python

    frequencies_x = np.linspace(0, 1000, 10000)

    alpha1 = 6
    alpha2 = 0
    fref = 10
    omegaRef = 5.e-5

    def IntensityGW(freqs, omegaRef, alpha1, fref, alpha2 = 2/3):
        ''' GW Intensity function from broken power law in OmegaGW
    
        Parameters
        ----------

        freqs: np.array
            frequency array
        fref: 
            reference frequency
        omegaRef: 
            Value of OmegaGW at reference frequency
        alpha1:
            first spectral index
        alpha2:
            second spectral index
        
        Return
        ------
        FrequencySeries
        '''
        from pygwb.constants import H0
        H_theor = (3 * H0.si.value ** 2) / (10 * np.pi ** 2)
        
        fknee = fref
        
        power = np.zeros_like(freqs)
        
        power[freqs<fknee] = H_theor * omegaRef * (freqs[freqs<fknee]) ** (alpha1 -3) * fref**(-alpha1)
        power[freqs>fknee] = H_theor * omegaRef * (freqs[freqs>fknee]) ** (alpha2 - 3) * fref**(-alpha2)
        power[freqs==fknee] = H_theor * omegaRef * (fknee) ** (alpha2 -3) * fref**(-alpha2)
        
        power[0] = power[1]
        
        return gwpy.frequencyseries.FrequencySeries(power, frequencies=freqs)

    Intensity_GW_inject = IntensityGW(frequencies_x, omegaRef = omegaRef, alpha1 = alpha1, fref = fref)

Note that the above signal PSD was chosen for illustrative purposes. However, in practice, any 
signal PSD can be chosen to be injected, and one does not need to restrict themselves to a broken power-law.

.. seealso::

    More information about ``gwpy.frequencyseries.FrequencySeries`` can be found `here <https://gwpy.github.io/docs/stable/api/gwpy.frequencyseries.FrequencySeries/>`_.

One also needs to specify the parameters that will serve as input to the ``simulator``. Concretely, we specify 
the duration of each simulated segment, the number of segments, and the sampling frequency.
   
.. code-block:: python

    duration = 60 # duration of each segment of data (s)
    N_segs = 10  # number of data segments to generate
    sampling_frequency = 1024 # Hz

.. tip::

    Not sure about what the above parameters do? Make sure to check out the `documentation <api/pygwb.simulator.html>`_ of the ``simulator`` module.

The detectors for which data with the above signal PSD need to be simulated, have to be passed 
to the ``simulator`` module. By relying on the ``detector`` module, we instantiate various detectors below.  
In addition, we note that these detectors are ``Interferometer`` objects, but are based on ``bilby`` detectors, 
which have default noise PSDs saved in them, in the ``power_spectral_density`` attribute of the ``bilby`` detector. 
Below, we load in this noise PSD and make sure the duration and sampling frequency of the detector are set to the desired value of 
these parameters.

.. code-block:: python

    H1 = Interferometer.get_empty_interferometer("H1") #LIGO Hanford detector
    L1 = Interferometer.get_empty_interferometer("L1") #LIGO Livingston detector

    ifo_list = [H1, L1]

    for ifo in ifo_list:
        ifo.duration = duration
        ifo.sampling_frequency = sampling_frequency
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(ifo.frequency_array, np.nan_to_num(ifo.power_spectral_density_array, posinf=1.e-41))
    
     net_HL = Network('HL', ifo_list)

.. seealso::

    Additional information about the ``Interferometer`` object can be found `here <api/pygwb.detector.Interferometer.html>`_. For more information, we also refer the user to the ``bilby``
    `documentation <https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.detector.html>`_.


We are now ready to simulate the data, consisting of a signal and Gaussian noise, colored by the noise PSD saved in each of the detectors. 
We rely on the ``network`` module to simulate the data by calling the ``set_interferometer_data_from_simulator()`` method (which uses the ``simulator`` module).
More information on the method can be found `here <api/pygwb.network.Network.html#pygwb.network.Network.set_interferometer_data_from_simulator>`_.

.. code-block:: python

     net_HL.set_interferometer_data_from_simulator(N_segments=N_segs, GWB_intensity=Intensity_GW_inject, sampling_frequency=sampling_frequency)


.. note::

    One may save the data by calling ``pygwb.network.save_interferometer_data_to_file()`` (see `here <api/pygwb.network.Network.html#pygwb.network.Network.save_interferometer_data_to_file>`_) 
    and specifying the file format as an argument. This wraps the ``gwpy.TimeSeries.write()`` method (more details can be found 
    `here <https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.write>`_).

**1.2 Injecting a power spectrum in real data**
-----------------------------------------------

Alternatively, one could decide to inject a SGWB in real detector data. To illustrate this functionality, we inject the same signal as above
in real LIGO data. The detectors are instantiated through the ``parameters`` module, which allows to load the parameters, including the GPS
times used to retrieve real data.

.. code-block:: python

    params = Parameters()
    params.update_from_file(path="../test/test_data/parameters_baseline_test.ini")
    params.t0=1247644204
    params.tf=1247645100
    params.segment_duration=128

.. tip::

    Not sure how the ``parameters`` module works anymore? Make sure to check out the `documentation <api/pygwb.parameters.html>`_.

We now create the two ``Interferometer`` objects that will be used for the data simulation (LIGO Hanford (H1) and LIGO Livingstn (L1) for this concrete example).

.. code-block:: python

    H1 = Interferometer.from_parameters(params.interferometer_list[0], params)
    L1 = Interferometer.from_parameters(params.interferometer_list[1], params)

    ifo_list = [H1, L1]

.. seealso::

    Additional informational information about the ``Interferometer`` object can be found `here <api/pygwb.detector.Interferometer.html>`_.

Note that the interferometers above contain the desired data in which we want to inject the signal. We now make sure the 
duration and sampling frequency of the detector are set to the desired value of these parameters, as specified in the parameters 
object defined at the start of this example.  The strain data in the interferometer is also set to the real data considered in this example.

.. code-block:: python

    for ifo in ifo_list:
        ifo.sampling_frequency = params.new_sample_rate
        ifo.set_strain_data_from_gwpy_timeseries(gwpy.timeseries.TimeSeries(data=ifo.timeseries.value, times=ifo.timeseries.times))
        ifo.duration=params.segment_duration

To inject a signal in real data, we rely on the ``network`` module, for which an object is instantiated below. To simulate the data, one calls
``set_interferometer_data_from_simulator()`` method (which uses the ``simulator`` module). More information on the method can be found 
`here <api/pygwb.network.Network.html#pygwb.network.Network.set_interferometer_data_from_simulator>`_. Note that the ``inject_into_data_flag`` is 
set to ``True``, indicating the data will be injected in real data, and that additional Gaussian colored noise therefore does not need to be simulated, nor injected on top of the signal.

.. code-block:: python

    HL_baseline = Baseline.from_parameters(H1, L1, params)
    net_HL = Network.from_baselines("HL_network", [HL_baseline])

    net_HL.set_interferometer_data_from_simulator(N_segments=7, GWB_intensity=Intensity_GW_inject, sampling_frequency=H1.sampling_frequency, inject_into_data_flag=True)

.. note::

    One may save the data by calling ``pygwb.network.save_interferometer_data_to_file()`` (see `here <api/pygwb.network.Network.html#pygwb.network.Network.save_interferometer_data_to_file>`_) 
    and specifying the file format as an argument. This wraps the ``gwpy.TimeSeries.write()`` method (more details can be found 
    `here <https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.write>`_).

**2. Injecting individual CBC events**
======================================

**2.1 Initialising empty interferometers and parameters for simulation**
------------------------------------------------------------------------

We start by specifying the parameters that will serve as input to the ``simulator``. 
Concretely, we specify the duration of each simulated segment, the number of segments, and the sampling frequency.

.. code-block:: python

    duration = 64 # duration of each segment of data (s)
    N_segs = 5  # number of data segments to generate
    sampling_frequency = 1024 # Hz

.. tip::

    Not sure about what the above parameters do? Make sure to check out the `documentation <api/pygwb.simulator.html>`_ of the ``simulator`` module.

The detectors for which data need to be simulated, have to be passed to the simulator module. 
By relying on the detector module, we instantiate various detectors below. We decide to use H1 and L1 
as an example. However, note that the data can be simulated for an arbitrary amount of detectors. One would simply add more 
detectors to the ``ifo_list`` below.

.. code-block:: python

    ifo_H1 = Interferometer.get_empty_interferometer('H1')
    ifo_L1 = Interferometer.get_empty_interferometer('L1')

    ifo_list = [ifo_H1, ifo_L1]

.. seealso::

    Additional informational information about the ``Interferometer`` object can be found `here <api/pygwb.detector.Interferometer.html>`_. For more information, we also refer the reader to the ``bilby``
    `documentation <https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.detector.html>`_.

The above detectors are ``Interferometer`` objects, but are based on ``bilby`` detectors, which have default noise PSDs saved in 
them, in the ``power_spectral_density`` attribute of the ``bilby`` detector. Below, we load in this noise PSD and make sure the 
duration and sampling frequency of the detector are set to the desired value of these parameters.

.. code-block:: python

    for ifo in ifo_list:
        ifo.duration = duration
        ifo.sampling_frequency = sampling_frequency
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(ifo.frequency_array, np.nan_to_num(ifo.power_spectral_density_array, posinf=1.e-41))
    net_HL = Network('HL', ifo_list)

**2.2 Specifying the CBC population**
-------------------------------------

Before being able to simulate CBCs, we need to specify which population the CBC events are drawn from. This is done by using ``bilby`` priors.
This allows the user to specify the distributions of the various parameters that come into play in CBC waveforms. A few examples are given below.

.. code-block:: python

    priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    priors['chirp_mass'] = bilby.core.prior.Uniform(2, 30, name="chirp_mass")
    priors['mass_ratio'] = 1.0
    priors['chi_1'] = 0
    priors['chi_2'] = 0
    priors['luminosity_distance'] = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', 
                                                          minimum=10, maximum=100, 
                                                          unit='Mpc')
    priors["geocent_time"] = bilby.core.prior.Uniform(0, duration*N_segs, name="geocent_time")

    # create 20 injections
    injections = priors.sample(20)

.. seealso::

    For additional information on ``bilby`` prior dictionaries, we refer the user to the `documentation <https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.prior.BBHPriorDict.html>`_.


The output of the cell above is a dictionary containing the injections, which will serve as input for the ``simulator``. 
It can be very useful to save these injections to file for later use. This is done by executing the following lines of code:

.. code-block:: python

    import json

    with open("injections.json", "w") as file:
        json.dump(
            injections, file, indent=2, cls=bilby.core.result.BilbyJsonEncoder
        )

**2.3 Simulating CBCs and Gaussian noise**
------------------------------------------

We are now ready to simulate the data, consisting of CBCs and Gaussian noise, colored by the noise PSD saved in each of the detectors. 
We rely on the ``pygwb.network`` module to simulate the data by calling the ``set_interferometer_data_from_simulator()`` method (which uses the ``pygwb.simulator`` module).
More information on the method can be found `here <api/pygwb.network.Network.html#pygwb.network.Network.set_interferometer_data_from_simulator>`_.

.. code-block:: python

    net_HL.set_interferometer_data_from_simulator(N_segs, CBC_dict=injections, sampling_frequency = sampling_frequency)
    
.. note::

    One may save the data by calling ``pygwb.network.save_interferometer_data_to_file()`` (see `here <api/pygwb.network.Network.html#pygwb.network.Network.save_interferometer_data_to_file>`_) 
    and specifying the file format as an argument. This wraps the ``gwpy.TimeSeries.write()`` method (more details can be found 
    `here <https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.write>`_).







