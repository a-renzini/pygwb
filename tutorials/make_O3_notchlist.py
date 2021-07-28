# from pygwb.notch import *
import sys

import numpy as np

from notch import *

sys.path.append("../pygwb")

####################
#### DISCLAIMER ####
####################

# Due to the new code method the nocth list produced in this tutorial will not exactely match the official O3 notch list
# Consider this code only be an example and for any further (official) processing, base your work on the official code & notchlists
# Official O3 code can be found here: https://git.ligo.org/stochastic/stochasticdetchar/-/tree/master/O3/notchlists/make_notchlist


def violin_modes(det):
    """
    Create list of bands around the violin modes we will remove

    Parameters
    ----------
    det: str
        Abbreviation for the detector being H1,L1 or V1 for Hanford, Livingston and Virgo respectively

    Returns
    -------
    notches: StochNotchList object
        List of lines you want to be notched in StochNotch format
    """

    notches = StochNotchList([])
    if det == "H1" or det == "L1":
        notches.append(
            StochNotch(480, 520, "Violin mode 1st harmonic")
        )  # range [480,520]
        notches.append(
            StochNotch(960, 1040, "Violin mode 2nd harmonic")
        )  # range [960,1040]
        notches.append(
            StochNotch(1455, 1540, "Violin mode 3rd harmonic")
        )  # range [1455,1540]

    if det == "V1":
        notches.append(
            StochNotch(275, 281, "BS Violin mode 1st harmonic")
        )  # range[275,281]
        notches.append(
            StochNotch(550, 561, "BS Violin mode 2nd harmonic")
        )  # range[550,561]
        notches.append(
            StochNotch(442, 457, "Violin mode 1st harmonic")
        )  # range [442,457]
        notches.append(
            StochNotch(881.5, 908.5, "Violin mode 2nd harmonic")
        )  # range [881.5,908.5]
        notches.append(
            StochNotch(1325.5, 1360.5, "Violin mode 3rd harmonic")
        )  # range [1325.5,1360.5]
        notches.append(
            StochNotch(1774, 1814, "Violin mode 4th harmonic")
        )  # range [1774,1814]
        notches.append(
            StochNotch(2192, 2256, "Violin mode 5th harmonic")
        )  # range [2192,2256]

    return notches


def calibration_lines(det):
    """
    Create list of calibration lines

    Parameters
    ----------
    det: str
        Abbreviation for the detector being H1,L1 or V1 for Hanford, Livingston and Virgo respectively

    Returns
    -------
    notches: StochNotchList object
        List of lines you want to be notched in StochNotch format

    """
    # ignore lines below 20 Hz

    notches = StochNotchList([])
    if det == "H1":
        notches.append(StochNotch(16.6, 17.6, "H1 calibration line"))
        notches.append(StochNotch(17.1, 18.1, "H1 calibration line"))
        #        notches.append(StochNotch(35.9,1.0,'H1 calibration line - first two weeks O3') # First two weeks are not analyze)d
        #        notches.append(StochNotch(36.7,1.0,'H1 calibration line - first two weeks O3') # First two weeks are not analyze)d
        notches.append(StochNotch(409.8, 410.8, "H1 calibration line"))
        notches.append(StochNotch(1083.2, 1084.2, "H1 calibration line"))
        notches.append(StochNotch(331.4, 332.4, "H1 calibration line"))

    if det == "L1":
        notches.append(StochNotch(434.4, 435.4, "L1 calibration line"))
        notches.append(StochNotch(1082.6, 1083.6, "L1 calibration line"))

    if det == "V1":
        notches.append(StochNotch(15.7, 15.9, "V1 WE EM calibration line"))
        notches.append(StochNotch(16.2, 16.4, "V1 BS EM calibration line"))
        notches.append(StochNotch(16.9, 17.1, "V1 NE EM calibration line"))
        notches.append(StochNotch(28.9, 29.1, "V1 calibration line"))
        notches.append(StochNotch(30.9, 31.1, "V1 calibration line"))
        notches.append(StochNotch(31.4, 31.6, "V1 calibration line"))
        notches.append(StochNotch(32.4, 32.6, "V1 calibration line"))
        notches.append(StochNotch(34.4, 34.6, "V1 WE PCAL calibration line"))
        notches.append(StochNotch(36.4, 36.6, "V1 NE PCAL calibration line"))
        notches.append(StochNotch(37.4, 37.6, "V1 NE EM calibration line"))
        notches.append(StochNotch(56.4, 56.6, "V1 WE EM calibration line"))
        notches.append(StochNotch(60.4, 60.6, "V1 WE PCAL calibration line"))
        notches.append(StochNotch(60.9, 61.1, "V1 BS EM calibration line"))
        notches.append(StochNotch(61.4, 61.6, "V1 WE EM calibration line"))
        notches.append(StochNotch(62.4, 62.6, "V1 NE EM calibration line"))
        notches.append(StochNotch(62.9, 63.1, "V1 PR EM calibration line"))
        notches.append(StochNotch(63.4, 63.6, "V1 NE PCAL calibration line"))
        notches.append(StochNotch(77.4, 77.6, "V1 NE EM calibration line"))
        notches.append(StochNotch(87.0, 87.2, "SDB1 OMC"))
        notches.append(StochNotch(106.4, 106.6, "V1 WE EM calibration line"))
        notches.append(StochNotch(107.4, 107.6, "V1 NE EM calibration line"))
        notches.append(StochNotch(137.4, 137.6, "V1 NE EM calibration line"))
        notches.append(StochNotch(206.4, 206.6, "V1 WE EM calibration line"))
        notches.append(StochNotch(355.4, 355.6, "V1 WE PCAL calibration line"))
        notches.append(StochNotch(355.9, 356.1, "V1 BS EM calibration line"))
        notches.append(StochNotch(356.4, 356.6, "V1 WE EM calibration line"))
        notches.append(StochNotch(357.4, 357.6, "V1 NE EM calibration line"))
        notches.append(StochNotch(357.9, 358.1, "V1 PR EM calibration line"))
        notches.append(StochNotch(359.4, 359.6, "V1 NE PCALcalibration line"))
        notches.append(StochNotch(406.4, 406.6, "V1 WE EM calibration line"))
        notches.append(StochNotch(1900.4, 1900.6, "V1 WE PCAL calibration line"))
        notches.append(
            StochNotch(2012.4, 2012.6, "V1 PR,BS,NI,NE,WI,WI EM calibration line")
        )
        notches.append(
            StochNotch(
                31.4,
                31.6,
                "V1 second order harmonic of calibration line at 15.8 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                32.5,
                32.7,
                "V1 second order harmonic of calibration line at 16.3 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                33.5,
                33.7,
                "V1 second order harmonic of calibration line at 16.8 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                120.9,
                121.1,
                "V1 second order harmonic of calibration line at 60.5 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                121.9,
                122.1,
                "V1 second order harmonic of calibration line at 61 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                122.9,
                123.1,
                "V1 second order harmonic of calibration line at 61.5 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                124.9,
                125.1,
                "V1 second order harmonic of calibration line at 62.5 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                125.9,
                126.1,
                "V1 second order harmonic of calibration line at 63 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                126.9,
                127.1,
                "V1 second order harmonic of calibration line at 63.5 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                181.4,
                181.6,
                "V1 third order harmonic of calibration line at 60.5 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                122.9,
                123.1,
                "V1 third order harmonic of calibration line at 61 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                184.4,
                184.6,
                "V1 third order harmonic of calibration line at 61.5 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                187.4,
                187.6,
                "V1 third order harmonic of calibration line at 62.5 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                188.9,
                189.1,
                "V1 third order harmonic of calibration line at 63 - identified by CW people",
            )
        )
        notches.append(
            StochNotch(
                190.4,
                190.6,
                "V1 third order harmonic of calibration line at 63.5 - identified by CW people",
            )
        )

    return notches


def instrumental_lines(baseline):
    """
    Creates a list of notched lines which were identified as being instrumental/environmental

    Parameters
    ----------
    baseline: str
        Abbreviation for the detector baseline pair being HL,HV or LV

    Returns
    -------
    extra_lines: StochNotchList object
        List of lines you want to be notched in StochNotch format
    """

    extra_lines = StochNotchList([])
    if baseline == "HL":
        extra_lines.append(
            StochNotch(33.199, 33.201, "Calibration line nonlinearity")
        )  # calibration line nonlinearity
        extra_lines.append(
            StochNotch(
                31.9,
                32.1,
                "environmental identification: link with H corner station accelerometers - regularly intermittent",
            )
        )  # environmental identification
        extra_lines.append(
            StochNotch(
                47.0,
                49.0,
                "H non-stationary, non-linear noise - scat. light -fixed at end of Sep 2019",
            )
        )  # non-stationary, non-linear noise - scat. light -fixed at end of Sep 2019
        extra_lines.append(
            StochNotch(
                49.95,
                50.05,
                "environmental identification: In L coherence with many channels including magnetic, seismic and suspension isolation. For LLO harmonic of 10 Hz comb and LHO harmonic of 1Hz comb according to CW studies",
            )
        )  # environmental identification
        extra_lines.append(StochNotch(20.075, 20.175, "ASC dither line"))
        extra_lines.append(StochNotch(20.73125, 20.83125, "ASC dither line"))
        extra_lines.append(StochNotch(21.85625, 21.95625, "ASC dither line"))
        extra_lines.append(StochNotch(22.29375, 22.39375, "ASC dither line"))
        extra_lines.append(
            StochNotch(27.41875, 27.51875, "Triple suspension bounce mode possibly SRM")
        )
        extra_lines.append(StochNotch(27.66875, 27.76875, "SR3 bounce mode"))
        extra_lines.append(
            StochNotch(
                35.66875, 35.76875, "Environmental disturbance also seen in O1/O2 "
            )
        )
        extra_lines.append(StochNotch(40.85625, 40.95625, "SR2/MC2 roll mode"))
        extra_lines.append(StochNotch(40.8875, 40.9875, "PR2 roll mode "))
        extra_lines.append(
            StochNotch(
                60.45,
                60.55,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                60.95,
                91.05,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                61.45,
                61.55,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                62.45,
                62.55,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                62.95,
                63.05,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                63.45,
                63.55,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                24.95,
                25.05,
                "Unknown comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(394.6375, 394.7375, "H1 Calibration line non-linearity")
        )
        extra_lines.append(StochNotch(1152.85, 1153.35, "H1 & L1 Calibration line"))
        extra_lines.append(
            StochNotch(
                31.45,
                31.55,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                32.45,
                32.55,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                33.45,
                33.55,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                33.95,
                34.05,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                29.95,
                30.05,
                "Unknown comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                39.95,
                40.05,
                "Unknown comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                99.95,
                100.05,
                "Unknown comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                53.35625,
                53.45625,
                "Calibration line non-linearity (Hanford, central freq rounded to nearest 1/32)",
            )
        )
        extra_lines.append(
            StochNotch(33.45, 33.55, "1Hz comb with 0.5Hz offset in H and L")
        )
        extra_lines.append(StochNotch(39.95, 40.05, "1Hz comb in H and 10Hz comb in L"))
        extra_lines.append(
            StochNotch(31.45, 31.55, "1Hz comb with 0.5Hz offset in H and L")
        )
        extra_lines.append(
            StochNotch(32.45, 32.55, "1Hz comb with 0.5Hz offset in H and L")
        )
        extra_lines.append(StochNotch(29.95, 30.05, "1Hz comb in H and 10Hz comb in L"))
        extra_lines.append(
            StochNotch(99.95, 100.05, "1Hz comb in H and 10Hz comb in L")
        )
        extra_lines.append(
            StochNotch(436.48125, 436.58125, "Input Mode Cleaner pitch mode")
        )

        # O3B added
        extra_lines.append(
            StochNotch(
                20.212,
                20.252,
                "H: EBAY seirack magnetometers - L: Input Mode Cleaner pitch mode",
            )
        )
        extra_lines.append(
            StochNotch(
                20.223,
                20.263,
                "H: EBAY seirack magnetometers - L:Input Mode Cleaner pitch mode",
            )
        )
        extra_lines.append(
            StochNotch(20.339, 20.379, "H: BLND_GS13Z - L:Input Mode Cleaner yaw mode")
        )
        extra_lines.append(
            StochNotch(
                174.5425,
                174.5825,
                "H: ASC-INP1_Y_OUT,PEM-CS_ADC_4_30_16K_OUT - L:Input Mode Cleaner yaw mode",
            )
        )
        extra_lines.append(
            StochNotch(258.4488, 258.4888, "L:Input Mode Cleaner yaw mode")
        )
        extra_lines.append(
            StochNotch(276.6675, 276.7075, "L:Input Mode Cleaner yaw mode")
        )
        extra_lines.append(
            StochNotch(409.9175, 409.9475, "H & L:Input Mode Cleaner yaw mode")
        )

    elif baseline == "HV":

        extra_lines.extend(comb(20, 1, 1726, 0.01))
        extra_lines.append(StochNotch(33.199, 33.201, "Calibration line nonlinearity"))
        extra_lines.append(
            StochNotch(
                295.95,
                296.05,
                "environmental identification: broadband coherence with V CEB_MAG_N and narrowband coherence with other magnetometers",
            )
        )  # environmental identification
        extra_lines.append(
            StochNotch(
                23.95,
                24.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                24.95,
                25.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                25.95,
                26.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                26.95,
                27.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                27.95,
                28.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                31.95,
                32.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                32.95,
                33.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                33.95,
                34.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                34.95,
                35.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                96.95,
                97.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                98.95,
                99.05,
                "1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.",
            )
        )
        extra_lines.append(
            StochNotch(
                33.45,
                33.55,
                "Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible",
            )
        )

        extra_lines.append(
            StochNotch(
                34.1375,
                34.2375,
                "Calibration line non-linearity (Hanford, central freq rounded to nearest 1/32)",
            )
        )
        extra_lines.append(
            StochNotch(
                49.45,
                50.55,
                "Virgo calibration: bad sensitivity in this segment -> advise to remove from analysis",
            )
        )

        # O3B added
        extra_lines.append(
            StochNotch(
                26.171, 26.204, "H: Input Mode Cleaner pitch mode - V: WEB_MAG_V"
            )
        )
        extra_lines.append(
            StochNotch(
                46.0,
                51.0,
                "V to large calib error due to active damping of 48hz mechanical resonance - advise to notch 46-51Hz region",
            )
        )  # 46-51

    elif baseline == "LV":

        extra_lines.append(
            StochNotch(
                31.09, 32.01, "Seen in time shifted run; comb or cal non-linearity?"
            )
        )
        extra_lines.append(
            StochNotch(
                33.09, 34.01, "Seen in time shifted run; comb or cal non-linearity?"
            )
        )
        extra_lines.append(StochNotch(39.05, 40.05, "1Hz comb in V and 10Hz comb in L"))
        extra_lines.append(
            StochNotch(
                31.95,
                32.05,
                "1Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                33.95,
                34.05,
                "1Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                34.95,
                35.05,
                "1Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                35.95,
                36.05,
                "1Hz comb; probably digital; not all comb teeth may be visible",
            )
        )
        extra_lines.append(
            StochNotch(
                29.76,
                29.86,
                "line in zero lag run - witness in Seis an ACC @ Virgo, witness in HPI-HAM3_BLND_L4C_Y_IN1 @ L",
            )
        )
        extra_lines.append(
            StochNotch(
                49.45,
                50.55,
                "Virgo calibration: bad sensitivity in this segment -> advise to remove from analysis",
            )
        )

        # O3B added
        extra_lines.append(
            StochNotch(
                71.23,
                71.27,
                "L: Input Mode Cleaner yaw mode - V: F7 crossbar mechanical mode",
            )
        )
        extra_lines.append(
            StochNotch(
                46.0,
                51.0,
                "V to large calib error due to active damping of 48hz mechanical resonance - advise to notch 46-51Hz region",
            )
        )  # 46-51

    return extra_lines


def Produce_O3_Isotropic_notchlist(src, version=0, runPart="A"):
    """
    Creates the final O3 isotropic notchlist combing all previously identified and decleared lines
    Gives as output 3 files containing the desired notchlist.

    Parameters
    ----------
    src: str
        path for storage of the output notchlist files
    binwidth: float
        Binwidth used in the analysis you want to apply this notch list to.
    version: int
        Number of the version of the notchlist, 0,1,2, ...
    runPart: str
        Part of the run the notchlist applies to e.g. A or B

    Output
    -------
    3 files as output containing the desired notchlist.
    """

    pulsar_src = "/home/kamiel.janssens/StochasticCoherenceRuns/JobFiles/stochasticdetchar/O3/notchlists/make_notchlist/input/"

    ###############################
    # HL
    ###############################
    output_filename = f"{src}/notchlist_HL_O3{runPart}_v{version}.txt"

    noise_power_lines = power_lines(nharmonics=28)  # 28*60=1680 < 1726
    noise_violin_modes = violin_modes("H1")
    noise_calibration_lines1 = calibration_lines(det="H1")
    noise_calibration_lines2 = calibration_lines(det="L1")
    noise_pulsar_injections = pulsar_injections(pulsar_src + "/pulsars.dat")
    noise_instrumental_lines = instrumental_lines("HL")

    noise_lines = StochNotchList([])
    noise_lines.extend(noise_power_lines)
    noise_lines.extend(noise_violin_modes)
    noise_lines.extend(noise_calibration_lines1)
    noise_lines.extend(noise_calibration_lines2)
    noise_lines.extend(noise_pulsar_injections)
    noise_lines.extend(noise_instrumental_lines)

    noise_lines.save_to_txt(output_filename)

    append_copy = open(output_filename, "r")
    original_text = append_copy.read()
    append_copy.close()

    append_copy = open(output_filename, "w")
    append_copy.write("####################\n")
    append_copy.write("#### DISCLAIMER ####\n")
    append_copy.write("####################\n")
    append_copy.write(
        "# Due to the new code method the nocth list produced in this tutorial will not exactely match the official O3 notch list\n"
    )
    append_copy.write(
        "# Consider this code only be an example and for any further (official) processing, base your work on the official code & notchlists\n"
    )
    append_copy.write(
        "# Official O3 code can be found here: https://git.ligo.org/stochastic/stochasticdetchar/-/tree/master/O3/notchlists/make_notchlist\n\n"
    )
    append_copy.write(original_text)
    append_copy.close()

    ###############################
    # HV
    ###############################
    output_filename = f"{src}/notchlist_HV_O3{runPart}_v{version}.txt"

    noise_power_lines = power_lines(nharmonics=28)  # 28*60=1680 < 1726
    noise_power_linesV = power_lines(fundamental=50, nharmonics=34)  # 34*50=1700 < 1726
    noise_violin_modes = violin_modes("H1")
    noise_violin_modesV = violin_modes("V1")
    noise_calibration_lines1 = calibration_lines(det="H1")
    noise_calibration_lines2 = calibration_lines(det="V1")
    noise_pulsar_injections = pulsar_injections(pulsar_src + "/pulsars.dat")
    noise_instrumental_lines = instrumental_lines("HV")

    noise_lines = StochNotchList([])
    noise_lines.extend(noise_power_lines)
    noise_lines.extend(noise_power_linesV)
    noise_lines.extend(noise_violin_modes)
    noise_lines.extend(noise_violin_modesV)
    noise_lines.extend(noise_calibration_lines1)
    noise_lines.extend(noise_calibration_lines2)
    noise_lines.extend(noise_pulsar_injections)
    noise_lines.extend(noise_instrumental_lines)

    noise_lines.save_to_txt(output_filename)

    append_copy = open(output_filename, "r")
    original_text = append_copy.read()
    append_copy.close()

    append_copy = open(output_filename, "w")
    append_copy.write("####################\n")
    append_copy.write("#### DISCLAIMER ####\n")
    append_copy.write("####################\n")
    append_copy.write(
        "# Due to the new code method the nocth list produced in this tutorial will not exactely match the official O3 notch list\n"
    )
    append_copy.write(
        "# Consider this code only be an example and for any further (official) processing, base your work on the official code & notchlists\n"
    )
    append_copy.write(
        "# Official O3 code can be found here: https://git.ligo.org/stochastic/stochasticdetchar/-/tree/master/O3/notchlists/make_notchlist\n\n"
    )
    append_copy.write(original_text)
    append_copy.close()

    ###############################
    # LV
    ###############################
    output_filename = f"{src}/notchlist_LV_O3{runPart}_v{version}.txt"

    noise_power_lines = power_lines(nharmonics=28)  # 28*60=1680 < 1726
    noise_power_linesV = power_lines(fundamental=50, nharmonics=34)  # 34*50=1700 < 1726
    noise_violin_modes = violin_modes("L1")
    noise_violin_modesV = violin_modes("V1")
    noise_calibration_lines1 = calibration_lines(det="L1")
    noise_calibration_lines2 = calibration_lines(det="V1")
    noise_pulsar_injections = pulsar_injections(pulsar_src + "/pulsars.dat")
    noise_instrumental_lines = instrumental_lines("LV")

    noise_lines = StochNotchList([])
    noise_lines.extend(noise_power_lines)
    noise_lines.extend(noise_power_linesV)
    noise_lines.extend(noise_violin_modes)
    noise_lines.extend(noise_violin_modesV)
    noise_lines.extend(noise_calibration_lines1)
    noise_lines.extend(noise_calibration_lines2)
    noise_lines.extend(noise_pulsar_injections)
    noise_lines.extend(noise_instrumental_lines)

    noise_lines.save_to_txt(output_filename)

    append_copy = open(output_filename, "r")
    original_text = append_copy.read()
    append_copy.close()

    append_copy = open(output_filename, "w")
    append_copy.write("####################\n")
    append_copy.write("#### DISCLAIMER ####\n")
    append_copy.write("####################\n")
    append_copy.write(
        "# Due to the new code method the nocth list produced in this tutorial will not exactely match the official O3 notch list\n"
    )
    append_copy.write(
        "# Consider this code only be an example and for any further (official) processing, base your work on the official code & notchlists\n"
    )
    append_copy.write(
        "# Official O3 code can be found here: https://git.ligo.org/stochastic/stochasticdetchar/-/tree/master/O3/notchlists/make_notchlist\n\n"
    )
    append_copy.write(original_text)
    append_copy.close()


if __name__ == "__main__":

    version = 10
    runPart = "B"

    Produce_O3_Isotropic_notchlist(src="./", version=version, runPart=runPart)
