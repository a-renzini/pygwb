"""The notch module handles all things considering notches. It bookkeeps and can calculates them with the provided classes and functions.

This module has two main classes and a couple of functions all dealing with the createn of a notchlist used in the analysis to get rid of bad behaving frequencies.
First of all there is the class StochNotch, based on the Notch class from ``bilby.gw.detector.strain_data``. This class stores a single Notch object containing a small description of the notch and the corresponding minimum and maximum frequency. 
Next, StochNotchList is the combination of multiple StochNotch objects. It will contain information about multiple notches, every notch represented as a StochNotch and the information inside.

Three independent functions are also defined in this module. Those created StochNotchList objects for three of the most prevalent types of notches. 
The first function is ``comb`` which creates a StochNotchList for a certain set of lines in a comb structure. Secondly, we have ``power_lines`` which makes a StochNotchList object for the notches coming with the power line harmonics, e.g. 60 Hz harmonics in USA and 50 Hz in Italy. The third and final function, ``pursat_injection``, generates a StochNotchList object with notches which are contaminated by pulsar injections.

Examples
--------

The notch list can be made using the functions from the notch module as we will show here.

We will generate a StochNotchList from a .txt file containing multiple StochNotch objects and utilise some notch functions from the module.
First we import the notch module and load notches into a StochNotchList object from a .txt file.

>>> import pygwb.notch as pn
>>> notch_list = pn.StochNotchList.load_from_file("./test/test_data/Official_O3_HL_notchlist.txt")

Now, :code:`notch_list` contains information about all notches in the considered .txt file.
The StochNotchList object itself is a container object that collects all notches of the file
in different StochNotch objects.

One of the more famous lines is created by the power line harmonics.
Let's check if their fundamental frequency is available in the StochNotchList object.
We know that that there should be a power line notched around 60 Hz.

>>> is_60_in_notch_list = notch_list.check_frequency(60)
True

So, it is possible to check if a certain frequency is present in the container object.
If one wants to have information about all the notches in the StochNotchList, you can run

>>> for notch in notch_list:
>>>    notch.print_notch()

This will show the minimum and maximum frequency for all notches in StochNotchList.
It will also print a small description of the notch itself.

Another important function of the List object is computing the frequency mask that gives an array that will 
be False for frequencies in the StochNotchList object. 
That array can then be utilised to "mask" a real pygwb analysis spectrum and notch out the contaminated frequencies.

We take a random frequency array and mask it using our StochNotchList object.

>>> frequency_array = np.arange(0, 1700,1/32.)
>>> notch_list.get_notch_mask(frequency_array, save_file_flag=False, filename="")

The List function utilises the :code:`get_notch_mask` function from the StochNotch object.
And it will combine that information for all notches. 

One can also save the StochNotchList object to a .txt file using :code:`save_to_txt(filename)`.
That will create a .txt file in the same structure as required to make a StochNotchList 
object from a file. You can also save the mask itself using 
:code:`save_notch_mask(frequency_array, filename)`

"""

import numpy as np
from bilby.gw.detector.strain_data import Notch


class StochNotch(Notch):
    def __init__(self, minimum_frequency, maximum_frequency, description):
        """A notch object storing the maximum and minimum frequency of the notch, as well as a description.

        Parameters
        ========
        minimum_frequency, maximum_frequency: `float`
            The minimum and maximum frequency of the notch.
        description: `str`
            A description of the origin/reason of the notch.
            
        See also
        --------
        bilby.gw.detector.strain_data.Notch : The parent class used for this implementation.
        
        """
        super().__init__(minimum_frequency, maximum_frequency)
        self.description = description

    def print_notch(self):
        """
        Small function that prints out the defining contents of the notch.
        It will show you the minimum and maximum frequency and the description of the notch.
        """
        print(self.minimum_frequency, self.maximum_frequency, self.description)

    def get_notch_mask(self, frequency_array):
        """Get a boolean mask for the frequencies in frequency_array in the notch.

        Parameters
        ========
        frequency_array: `np.ndarray`
            An array of frequencies.

        Returns
        =======
        notch_mask: `np.ndarray`
            An array of booleans that are False for frequencies in the notch.

        Notes
        =====
        This notches any frequency that may have overlapping frequency content with the notch.
        """
        df = np.abs(frequency_array[1] - frequency_array[0])
        frequencies_below = np.concatenate(
            [frequency_array[:1] - df, frequency_array[:-1]]
        )
        frequencies_above = np.concatenate(
            [frequency_array[1:], frequency_array[-1:] + df]
        )
        lower = frequencies_below + df / 2 <= self.maximum_frequency
        upper = frequencies_above - df / 2 >= self.minimum_frequency
        notch_mask = [not elem for elem in (lower & upper)]
        return notch_mask


class StochNotchList(list):
    def __init__(self, notch_list):
        """A list of notches. All these notches are represented by an object of the StochNotch class.

        Parameters
        ========
        notch_list: `list`
            A list of length-3 tuples of the (min, max) frequency; description for the notches.

        Notes
        =====
        :raises:
            ValueError: If the list is malformed.
        """

        if notch_list is not None:
            for notch in notch_list:
                if isinstance(notch, tuple) and len(notch) == 3:
                    self.append(StochNotch(*notch))
                else:
                    msg = f"notch_list {notch_list} is malformed"
                    raise ValueError(msg)

    def check_frequency(self, freq):
        """Check if freq is inside the notch list.

        Parameters
        ========
        freq: `float`
            The frequency to check.

        Returns
        =======
        True/False:
            If freq inside any of the notches, return True, else False.
        """

        for notch in self:
            if notch.check_frequency(freq):
                return True
        return False

    def get_notch_mask(self, frequency_array, save_file_flag=False, filename=""):
        """Get a boolean mask for the frequencies in frequency_array in the notch list.

        Parameters
        ========
        frequency_array: `np.ndarray`
            An array of frequencies.
        save_file_flag: `bool`
            A boolean flag indicating whether to save the notch mask in a file or not.
        filename: `str`
            The name of the file where to store the notch mask if save_file_flag is true.

        Returns
        =======
        notch_mask: `np.ndarray`
            An array of booleans that are False for frequencies in the notch.

        Notes
        =====
        This notches any frequency that may have overlapping frequency content with the notch.
        """
        notch_mask = np.ones(len(frequency_array), dtype=bool)
        for notch in self:
            notch_mask = notch_mask & notch.get_notch_mask(frequency_array)

        if save_file_flag == True:
            if len(filename) == 0:
                filename = "Notch_mask.txt"
            save_notch_mask(self, frequency_array, filename)
        return notch_mask

    def save_notch_mask(self, frequency_array, filename):
        """Saves a boolean mask for the frequencies in frequency_array in the notch list.

        Parameters
        ========
        frequency_array: `np.ndarray`
            An array of frequencies.

        filename: `str`
            Name of the target file.

        Notes
        =====
        This saves notch_mask (see get_notch_mask for more information) in a text file. notch_mask is an array of booleans that are False for frequencies in the notch.
        """

        notch_mask = self.get_notch_mask(frequency_array)
        np.savetxt(
            filename,
            np.transpose([frequency_array, notch_mask]),
        )

    def save_to_txt(self, filename):
        """Save the notch list to a txt-file (after sorting).

        Parameters
        ========
        filename: `str`
            Name of the target file

        """

        fmin = []
        fmax = []
        desc = []
        self.sort_list()
        for n in self:
            fmin.append(n.minimum_frequency)
            fmax.append(n.maximum_frequency)
            desc.append(n.description)

        np.savetxt(
            filename,
            np.transpose([fmin, fmax, desc]),
            fmt=("%-20s  ,  %-20s  ,  %-" + str(len(max(desc)) + 5) + "s"),
        )

    def sort_list(self):
        """Sorts the notch list based on the minimum frequency of the notches.

        Parameters
        ==========

        """

        self.sort(key=lambda elem: elem.minimum_frequency)

    @classmethod
    def load_from_file(cls, filename):
        """Load an already existing notch list from a txt-file (with formatting as produced by this code).

        Parameters
        ========
        filename: `str`
            Filename of the file containing the notchlist to be read in

        """

        fmin, fmax = np.loadtxt(filename, delimiter=",", unpack=True, usecols=(0, 1))
        desc = np.loadtxt(
            filename, delimiter=",", unpack=True, usecols=(2), dtype="str"
        )

        cls = StochNotchList([])
        if np.ndim(fmin) == 1:
            for i in range(len(fmin)):
                cls.append(StochNotch(fmin[i], fmax[i], desc[i]))
        elif np.ndim(fmin) == 0:
            cls.append(StochNotch(fmin, fmax, desc))
        else:
            raise TypeError("Notch list from file has too many dimensions.")
        return cls


def power_lines(fundamental=60, nharmonics=40, df=0.2):
    """
    Create list of power line harmonics (nharmonics*fundamental Hz) to remove.

    Parameters
    ========
    fundamental: `float`, optional
        Fundamental frequency of the power line.
        Default value is 60 Hz.
    nharmonics: `float`, optional
        Number of harmonics (should include all harmonics within studied frequency range of the study).
        Default is 40.
    df: `float`, optional
        Frequency width of considered power line.

    Returns
    =======
    notches: `StochNotchList`
        StochNotchList object containing lines you want to be notched.

    """
    freqs = fundamental * np.arange(1, nharmonics + 1)

    notches = StochNotchList([])
    for f0 in freqs:
        notch = StochNotch(f0 - df / 2, f0 + df / 2, "Power Lines")
        notches.append(notch)

    return notches


def comb(f0, f_spacing, n_harmonics, df, description=None):
    """
    Create a list of comb lines to remove with the form 'f0+n*f_spacing, n=0,1,...,n_harmonics-1'.

    Parameters
    ========
    f0: `float`
        Fundamental frequency of the comb.
    f_spacing: `float`
        Spacing between two subsequent harmonics.
    nharmonics: `float`
        Number of harmonics (should include all harmonics within studied frequency range of the study).
    df: `float`
        Width of the comb-lines.
    description: `str`, optional
        Optional additional description, e.g. known source of the comb.

    Returns
    =======
    notches: `StochNotchList`
        StochNotchList object of lines you want to be notched.

    """

    notches = StochNotchList([])
    freqs = [f0 + n * f_spacing for n in range(n_harmonics)]
    for f in freqs:
        TotalDescription = f"Comb with fundamental freq {f0} and spacing {f_spacing}"
        if description:
            TotalDescription += " " + description
        notch = StochNotch(f - df / 2, f + df / 2, TotalDescription)
        notches.append(notch)

    return notches


def pulsar_injections(filename, t_start, t_end, doppler=1e-4):
    """
    Create list of frequencies contaminated by pulsar injections.

    Parameters
    ========
    filename: `str`
        Filename of list containing information about pulsar injections, e.g. for O3 at https://git.ligo.org/stochastic/stochasticdetchar/-/blob/master/O3/notchlists/make_notchlist/input/pulsars.dat.
    t_start: `int`
        GPS start time of run/analysis.
    t_end: `int`
        GPS end time of run/analysis.
    doppler: `float`, optional
        Doppler shift; typical value of v/c for Earth motion in solar system = 1e-4 (default).

    Returns
    =======
    notches: `StochNotchList`
        StochNotchList object of lines you want to be notched.
    """

    """
    f_start: pulsar freq at start of time period
    f_end:   pulsar freq at end of time period
    f1:      allow for doppler shifting
    f2:      allow for doppler shifting
    f0:      central freq over entire period
    df:      width
    binary:  pulsar binary system, yes or no. If yes the affected with is ~two times larger (by design). We use a conservative factor of 3. 
    """

    t_refs, f_refs, f_dots, binary = np.loadtxt(
        filename,
        unpack=True,
        dtype=[
            ("t_refs", float),
            ("f_refs", float),
            ("f_dots", float),
            ("binary", str, 3),
        ],
    )
    notches = StochNotchList([])

    for t_ref, f_ref, f_dot, my_binary in zip(t_refs, f_refs, f_dots, binary):
        f_start = f_ref + f_dot * (t_start - t_ref)
        f_end = f_ref + f_dot * (t_end - t_ref)
        f1 = f_start * (1 + doppler)
        f2 = f_end * (1 - doppler)
        f0 = (f1 + f2) / 2.0
        df = f1 - f2
        if my_binary == "yes":
            df = 3.0 * df
        notch = StochNotch(f0 - df / 2, f0 + df / 2, "Pulsar injection")
        notches.append(notch)
    return notches
