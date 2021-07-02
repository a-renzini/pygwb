import numpy as np
from NotchList_BasisFunctions import *

def power_lines(fundamental=60,nharmonics=40,df=0.2):
    """
    Create list of power line harmonics (nharmonics*fundamental Hz) to remove

    Parameters
    ----------
    fundamental: float
        Fundamental frequeny of the first harmonic
    nharmonics: float
        Number of harmonics (should include all harmonics within studied frequency range of the study)

    Returns
    -------
    notches: list of NoiseLine objects
        List of lines you want to be notched in NoisLine format

    """
    freqs=fundamental*np.arange(1,nharmonics+1)

    notches=[]
    for f0 in freqs:
        notch=NoiseLine(f0,df,'Power Lines')
        notches.append(notch)

    return notches


def comb(f0,f_spacing,n_harmonics,df,description = None):
    """
    Create a list of comb lines to remove with the form 'f0+n*f_spacing, n=0,1,...,n_harmonics-1'

    Parameters
    ----------
    f0: float
        Fundamental frequeny of the first harmonic
    f_spacing: float
        spacing between two subsequent harmonics
    nharmonics: float
        Number of harmonics (should include all harmonics within studied frequency range of the study)
    df: float
        Width of the comb-lines
    description: str (Optional)
        Optional additional description, e.g. known source of the comb

    Returns
    -------
    notches: list of NoiseLine objects
        List of lines you want to be notched in NoisLine format

    """

    notches=[]
    freqs=[f0+n*f_spacing for n in range(n_harmonics)]
    for f in freqs:
        TotalDescription = 'Comb with fundamental freq {} and spacing {}'.format(f0,f_spacing)
        if description:
            TotalDescription += ' ' + description
        notch=NoiseLine(f,df,TotalDescription)
        notches.append(notch)

    return notches




def violin_modes(det):
    """
    Create list of bands around the violin modes we will remove

    Parameters
    ----------
    det: str
        Abreviation for the detector being H1,L1 or V1 for Hanford, Livingston and Virgo respectively

    Returns
    -------
    []: list of NoiseLine objects
        List of lines you want to be notched in NoisLine format
    """

    notches = []
    if det=='H1' or det=='L1':
        mode1=NoiseLine(500,40,'Violin mode 1st harmonic') # range [480,520]
        mode2=NoiseLine(1000,80,'Violin mode 2nd harmonic') # range [960,1040]
        mode3=NoiseLine(1497.5,85,'Violin mode 3rd harmonic') # range [1455,1540]
        
        return [mode1,mode2,mode3]

    if det=='V1':
        mode1BS=NoiseLine(278,6,'BS Violin mode 1st harmonic') # range[275,281]
        mode2BS=NoiseLine(555.5,11,'BS Violin mode 2nd harmonic') # range[550,561]
        mode1=NoiseLine(449.5,15,'Violin mode 1st harmonic') # range [442,457]
        mode2=NoiseLine(895,27,'Violin mode 2nd harmonic') # range [881.5,908.5]
        mode3=NoiseLine(1343,35,'Violin mode 3rd harmonic') # range [1325.5,1360.5]
        mode4=NoiseLine(1794,40,'Violin mode 4th harmonic') # range [1774,1814]
        mode5=NoiseLine(2224,64,'Violin mode 5th harmonic') # range [2192,2256]

        return [mode1BS,mode2BS,mode1,mode2,mode3,mode4,mode5]


def calibration_lines(det):
    """
    Create list of calibration lines

    Parameters
    ----------
    det: str
        Abreviation for the detector being H1,L1 or V1 for Hanford, Livingston and Virgo respectively

    Returns
    -------
    []: list of NoiseLine objects
        List of lines you want to be notched in NoisLine format
    """
    # ignore lines below 20 Hz

    if det=='H1':
        H1mode0=NoiseLine(17.1,1.0,'H1 calibration line')
        H1mode1=NoiseLine(17.6,1.0,'H1 calibration line')
#        H1mode2=NoiseLine(35.9,1.0,'H1 calibration line - first two weeks O3') # First two weeks are not analyzed
#        H1mode3=NoiseLine(36.7,1.0,'H1 calibration line - first two weeks O3') # First two weeks are not analyzed
        H1mode4=NoiseLine(410.3,1.0,'H1 calibration line')
        H1mode5=NoiseLine(1083.7,1.0,'H1 calibration line')
        H1mode6=NoiseLine(331.9,1.0,'H1 calibration line')
        return [H1mode0,H1mode1,H1mode4,H1mode5,H1mode6]
    if det=='L1':
        L1mode4=NoiseLine(434.9,1.0,'L1 calibration line')
        L1mode5=NoiseLine(1083.1,1.0,'L1 calibration line')
        return [L1mode4,L1mode5]
    if det=='V1':
        V1modes=[NoiseLine(15.8,0.1,'V1 WE EM calibration line'),
                 NoiseLine(16.3,0.1,'V1 BS EM calibration line'),
                 NoiseLine(16.8,0.1,'V1 NE EM calibration line'),
                 NoiseLine(29.0,0.1,'V1 calibration line'),
                 NoiseLine(31.0,0.1,'V1 calibration line'),
                 NoiseLine(31.5,0.1,'V1 calibration line'),
                 NoiseLine(32.5,0.1,'V1 calibration line'),
                 NoiseLine(34.5,0.1,'V1 WE PCAL calibration line'),
                 NoiseLine(36.5,0.1,'V1 NE PCAL calibration line'),
                 NoiseLine(37.5,0.1,'V1 NE EM calibration line'),
                 NoiseLine(56.5,0.1,'V1 WE EM calibration line'),
                 NoiseLine(60.5,0.1,'V1 WE PCAL calibration line'),
                 NoiseLine(61,0.1,'V1 BS EM calibration line'),
                 NoiseLine(61.5,0.1,'V1 WE EM calibration line'),
                 NoiseLine(62.5,0.1,'V1 NE EM calibration line'),
                 NoiseLine(63,0.1,'V1 PR EM calibration line'),
                 NoiseLine(63.5,0.1,'V1 NE PCAL calibration line'),
                 NoiseLine(77.5,0.1,'V1 NE EM calibration line'),
                 NoiseLine(87.1,0.1,'SDB1 OMC'),
                 NoiseLine(106.5,0.1,'V1 WE EM calibration line'),
                 NoiseLine(107.5,0.1,'V1 NE EM calibration line'),
                 NoiseLine(137.5,0.1,'V1 NE EM calibration line'),
                 NoiseLine(206.5,0.1,'V1 WE EM calibration line'),
                 NoiseLine(355.5,0.1,'V1 WE PCAL calibration line'),
                 NoiseLine(356.0,0.1,'V1 BS EM calibration line'),
                 NoiseLine(356.5,0.1,'V1 WE EM calibration line'),
                 NoiseLine(357.5,0.1,'V1 NE EM calibration line'),
                 NoiseLine(358.0,0.1,'V1 PR EM calibration line'),
                 NoiseLine(359.5,0.1,'V1 NE PCALcalibration line'),
                 NoiseLine(406.5,0.1,'V1 WE EM calibration line'),
                 NoiseLine(1900.5,0.1,'V1 WE PCAL calibration line'),
                 NoiseLine(2012.5,0.1,'V1 PR,BS,NI,NE,WI,WI EM calibration line'),
                 NoiseLine(31.6,0.1,'V1 second order harmonic of calibration line at 15.8 - identified by CW people'),
                 NoiseLine(32.6,0.1,'V1 second order harmonic of calibration line at 16.3 - identified by CW people'),
                 NoiseLine(33.6,0.1,'V1 second order harmonic of calibration line at 16.8 - identified by CW people'),
                 NoiseLine(121,0.1,'V1 second order harmonic of calibration line at 60.5 - identified by CW people'),
                 NoiseLine(122,0.1,'V1 second order harmonic of calibration line at 61 - identified by CW people'),
                 NoiseLine(123,0.1,'V1 second order harmonic of calibration line at 61.5 - identified by CW people'),
                 NoiseLine(125,0.1,'V1 second order harmonic of calibration line at 62.5 - identified by CW people'),
                 NoiseLine(126,0.1,'V1 second order harmonic of calibration line at 63 - identified by CW people'),
                 NoiseLine(127,0.1,'V1 second order harmonic of calibration line at 63.5 - identified by CW people'),
                 NoiseLine(181.5,0.1,'V1 third order harmonic of calibration line at 60.5 - identified by CW people'),
                 NoiseLine(183,0.1,'V1 third order harmonic of calibration line at 61 - identified by CW people'),
                 NoiseLine(184.5,0.1,'V1 third order harmonic of calibration line at 61.5 - identified by CW people'),
                 NoiseLine(187.5,0.1,'V1 third order harmonic of calibration line at 62.5 - identified by CW people'),
                 NoiseLine(189,0.1,'V1 third order harmonic of calibration line at 63 - identified by CW people'),
                 NoiseLine(190.5,0.1,'V1 third order harmonic of calibration line at 63.5 - identified by CW people'),

                 ]
        return V1modes

def pulsar_injections(filename='input/pulsars.dat'):
    """
    Create list of frequencies contaminated by pulsar injections

    Parameters
    ----------
    filename: str
        Filename of list containg information about pulsar injections. e.g. for O3 at https://git.ligo.org/stochastic/stochasticdetchar/-/blob/master/O3/notchlists/make_notchlist/input/pulsars.dat

    Returns
    -------
    notches: list of NoiseLine objects
        List of lines you want to be notched in NoisLine format
    """
    trefs,frefs,fdots=np.loadtxt(filename,unpack=True)
    doppler=1e-4 # typical value of v/c for Earth motion in solar system
    Tstart=1238112018 # Apr 1 2019 00:00:00 UTC
    Tend= 1269363618  # Mar 27 2020 17:00:00 UTC

    notches=[]
    for tref,fref,fdot in zip(trefs,frefs,fdots):
        fstart = fref + fdot*(Tstart - tref) # pulsar freq at start of O2
        fend = fref + fdot*(Tend - tref) # pulsar freq at end of O2
        f1=fstart*(1+doppler) # allow for doppler shifting
        f2=fend*(1-doppler) # allow for doppler shifting
        f0 = (f1+f2)/2.0 # central freq over all of O2
        df = (f1-f2) # width
        notch=NoiseLine(f0,df,'Pulsar injection')
        notches.append(notch)
    return notches


def instrumental_lines(baseline):
    """
    Creates a list of notched lines which were identified as being instrumental/environmental

    Parameters
    ----------
    baseline: str
        Abreviation for the detector baseline pair being HL,HV or LV

    Returns
    -------
    extra_lines: list of NoiseLine objects
        List of lines you want to be notched in NoisLine format
    """

    extra_lines = []
    if baseline == 'HL':
        extra_lines.append(NoiseLine(33.2,0.001,'Calibration line nonlinearity')) # calibration line nonlinearity
        extra_lines.append(NoiseLine(32.0,0.2,'environmental identification: link with H corner station accelerometers - regularly intermittent')) # environmental identification
        extra_lines.append(NoiseLine(48.0,2,'H non-stationary, non-linear noise - scat. light -fixed at end of Sep 2019')) # non-stationary, non-linear noise - scat. light -fixed at end of Sep 2019
        extra_lines.append(NoiseLine(50.0,0.1,'environmental identification: In L coherence with many channels including magnetic, seismic and suspension isolation. For LLO harmonic of 10 Hz comb and LHO harmonic of 1Hz comb according to CW studies')) # environmental identification
        extra_lines.append(NoiseLine(20.125,0.1,'ASC dither line'))
        extra_lines.append(NoiseLine(20.78125,0.1,'ASC dither line'))
        extra_lines.append(NoiseLine(21.90625,0.1,'ASC dither line'))
        extra_lines.append(NoiseLine(22.34375,0.1,'ASC dither line'))
        extra_lines.append(NoiseLine(27.46875,0.1,'Triple suspension bounce mode possibly SRM'))
        extra_lines.append(NoiseLine(27.71875,0.1,'SR3 bounce mode'))
        extra_lines.append(NoiseLine(35.71875,0.1,'Environmental disturbance also seen in O1/O2 '))
        extra_lines.append(NoiseLine(40.90625,0.1,'SR2/MC2 roll mode'))
        extra_lines.append(NoiseLine(40.9375,0.1,'PR2 roll mode '))
        extra_lines.append(NoiseLine(60.5,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(61.0,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(61.5,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(62.5,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(63.0,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(63.5,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(25.0,0.1,'Unknown comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(394.6875,0.1,'H1 Calibration line non-linearity'))
        extra_lines.append(NoiseLine(1153.1,0.5,'H1 & L1 Calibration line'))
        extra_lines.append(NoiseLine(31.5,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(32.5,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(33.5,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(34.0,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(30.0,0.1,'Unknown comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(40.0,0.1,'Unknown comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(100.0,0.1,'Unknown comb; probably digital; not all comb teeth may be visible'))
        extra_lines.append(NoiseLine(53.40625,0.1,'Calibration line non-linearity (Hanford, central freq rounded to nearest 1/32)'))
        extra_lines.append(NoiseLine(33.5,0.1,'1Hz comb with 0.5Hz offset in H and L'))
        extra_lines.append(NoiseLine(40.0,0.1,'1Hz comb in H and 10Hz comb in L'))
        extra_lines.append(NoiseLine(31.5,0.1,'1Hz comb with 0.5Hz offset in H and L'))
        extra_lines.append(NoiseLine(32.5,0.1,'1Hz comb with 0.5Hz offset in H and L'))
        extra_lines.append(NoiseLine(30.0,0.1,'1Hz comb in H and 10Hz comb in L'))
        extra_lines.append(NoiseLine(100.0,0.1,'1Hz comb in H and 10Hz comb in L'))
        extra_lines.append(NoiseLine(436.53125,0.1,'Input Mode Cleaner pitch mode'))

        #O3B added
        extra_lines.append(NoiseLine(20.232,0.03125,'H: EBAY seirack magnetometers - L: Input Mode Cleaner pitch mode'))
        extra_lines.append(NoiseLine(20.243,0.03125,'H: EBAY seirack magnetometers - L:Input Mode Cleaner pitch mode'))
        extra_lines.append(NoiseLine(20.359,0.03125,'H: BLND_GS13Z - L:Input Mode Cleaner yaw mode'))
        extra_lines.append(NoiseLine(174.5625,0.03125,'H: ASC-INP1_Y_OUT,PEM-CS_ADC_4_30_16K_OUT - L:Input Mode Cleaner yaw mode'))
        extra_lines.append(NoiseLine(258.4688,0.03125,'L:Input Mode Cleaner yaw mode'))
        extra_lines.append(NoiseLine(276.6875,0.03125,'L:Input Mode Cleaner yaw mode'))
        extra_lines.append(NoiseLine(409.9375,0.03125,'H & L:Input Mode Cleaner yaw mode'))
        

    elif baseline == 'HV':

        extra_lines=np.append(extra_lines,comb(20,1,1726,0.01))
        extra_lines=np.append(extra_lines,NoiseLine(33.2,0.001,'Calibration line nonlinearity'))
        extra_lines=np.append(extra_lines,NoiseLine(296,0.1,'environmental identification: broadband coherence with V CEB_MAG_N and narrowband coherence with other magnetometers')) # environmental identification
        extra_lines=np.append(extra_lines,NoiseLine(24.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(25.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(26.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(27.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(28.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(32.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(33.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(34.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(35.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(97.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(99.0,0.1,'1Hz comb with larger bin width since in coh spectra adjecent bins where also loud lines.'))
        extra_lines=np.append(extra_lines,NoiseLine(33.5,0.1,'Unknown 0.5Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines=np.append(extra_lines,NoiseLine(34.1875,0.1,'Calibration line non-linearity (Hanford, central freq rounded to nearest 1/32)'))
        extra_lines=np.append(extra_lines,NoiseLine(50.0,1.1,'Virgo calibration: bad sensitivity in this segment -> advise to remove from analysis'))

        #O3B added
        extra_lines=np.append(extra_lines,NoiseLine(26.1875,0.03125,'H: Input Mode Cleaner pitch mode - V: WEB_MAG_V'))
        extra_lines=np.append(extra_lines,NoiseLine(48.5,5.0,'V to large calib error due to active damping of 48hz mechanical resonance - advise to notch 46-51Hz region')) #46-51


    elif baseline == 'LV':

        extra_lines=np.append(extra_lines,NoiseLine(32,0.001,'Seen in time shifted run; comb or cal non-linearity?'))
        extra_lines=np.append(extra_lines,NoiseLine(34,0.001,'Seen in time shifted run; comb or cal non-linearity?'))
        extra_lines=np.append(extra_lines,NoiseLine(40.0,0.1,'1Hz comb in V and 10Hz comb in L'))
        extra_lines=np.append(extra_lines,NoiseLine(32.0,0.1,'1Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines=np.append(extra_lines,NoiseLine(34.0,0.1,'1Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines=np.append(extra_lines,NoiseLine(35.0,0.1,'1Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines=np.append(extra_lines,NoiseLine(36.0,0.1,'1Hz comb; probably digital; not all comb teeth may be visible'))
        extra_lines=np.append(extra_lines,NoiseLine(29.81,0.1,'line in zero lag run - witness in Seis an ACC @ Virgo, witness in HPI-HAM3_BLND_L4C_Y_IN1 @ L'))
        extra_lines=np.append(extra_lines,NoiseLine(50.0,1.1,'Virgo calibration: bad sensitivity in this segment -> advise to remove from analysis'))

        #O3B added
        extra_lines=np.append(extra_lines,NoiseLine(71.25,0.03125,'L: Input Mode Cleaner yaw mode - V: F7 crossbar mechanical mode'))
        extra_lines=np.append(extra_lines,NoiseLine(48.5,5.0,'V to large calib error due to active damping of 48hz mechanical resonance - advise to notch 46-51Hz region')) #46-51
    
    return extra_lines

def Produce_O3_Isotropic_notchlist(src,binwidth,version=0,runPart='A'):
    """
    Creates the final O3 isotropic notchlist combing all previously identified and decleared lines
    Gives as output 6 files containing the desired notchlist. For each baseline 1 file in a format easily readible by the code, 1 in more human readable format.

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
    6 files as output containing the desired notchlist. For each baseline 1 file in a format easily readible by the code, 1 in more human readable format.
    """ 

    pulsar_src = "/home/kamiel.janssens/StochasticCoherenceRuns/JobFiles/stochasticdetchar/O3/notchlists/make_notchlist/input/"


    ###############################
    # HL
    ###############################
    output_filename1='{output}/notchlist_HL_{bw}_O3{AorB}_v{v}.txt'.format(output=src,bw=binwidth,AorB=runPart,v=version)
    output_filename2='{output}/notchlist_HL_{bw}_O3{AorB}_v{v}_HumanReadable.txt'.format(output=src,bw=binwidth,AorB=runPart,v=version)

    noise_power_lines = power_lines(nharmonics=28) # 28*60=1680 < 1726
    noise_violin_modes = violin_modes('H1')
    noise_calibration_lines1 = calibration_lines(det='H1')
    noise_calibration_lines2 = calibration_lines(det='L1')
    noise_pulsar_injections = pulsar_injections(pulsar_src+'/pulsars.dat')
    noise_instrumental_lines = instrumental_lines('HL')
    print(noise_instrumental_lines)

    noise_lines=[noise_power_lines,noise_violin_modes,noise_calibration_lines1,noise_calibration_lines2,noise_pulsar_injections,noise_instrumental_lines]

    notches=make_notchlist(noise_lines,binwidth)

    make_txt_file(noise_lines,binwidth,outfile=output_filename2)

    ###############################
    # HV
    ###############################
    output_filename1='{output}/notchlist_HV_{bw}_O3{AorB}_v{v}.txt'.format(output=src,bw=binwidth,AorB=runPart,v=version)
    output_filename2='{output}/notchlist_HV_{bw}_O3{AorB}_v{v}_HumanReadable.txt'.format(output=src,bw=binwidth,AorB=runPart,v=version)


    noise_power_lines=power_lines(nharmonics=28) # 28*60=1680 < 1726
    noise_power_linesV=power_lines(fundamental=50,nharmonics=34) # 34*50=1700 < 1726
    noise_violin_modes=violin_modes('H1')
    noise_violin_modesV=violin_modes('V1')
    noise_calibration_lines1=calibration_lines(det='H1')
    noise_calibration_lines2=calibration_lines(det='V1')
    noise_pulsar_injections=pulsar_injections(pulsar_src+'/pulsars.dat')
    noise_instrumental_lines = instrumental_lines('HV')

    noise_lines=[noise_power_lines,noise_power_linesV,noise_violin_modes,noise_violin_modesV,noise_calibration_lines1,noise_calibration_lines2,noise_pulsar_injections,noise_instrumental_lines]

    notches=make_notchlist(noise_lines,binwidth)

    make_txt_file(noise_lines,binwidth,outfile=output_filename2)


    ###############################
    # LV
    ###############################
    output_filename1='{output}/notchlist_LV_{bw}_O3{AorB}_v{v}.txt'.format(output=src,bw=binwidth,AorB=runPart,v=version)
    output_filename2='{output}/notchlist_LV_{bw}_O3{AorB}_v{v}_HumanReadable.txt'.format(output=src,bw=binwidth,AorB=runPart,v=version)


    noise_power_lines=power_lines(nharmonics=28) # 28*60=1680 < 1726
    noise_power_linesV=power_lines(fundamental=50,nharmonics=34) # 34*50=1700 < 1726
    noise_violin_modes=violin_modes('L1')
    noise_violin_modesV=violin_modes('V1')
    noise_calibration_lines1=calibration_lines(det='L1')
    noise_calibration_lines2=calibration_lines(det='V1')
    noise_pulsar_injections=pulsar_injections(pulsar_src+'/pulsars.dat')
    noise_instrumental_lines = instrumental_lines('LV')

    noise_lines=[noise_power_lines,noise_power_linesV,noise_violin_modes,noise_violin_modesV,noise_calibration_lines1,noise_calibration_lines2,noise_pulsar_injections,noise_instrumental_lines]

    notches=make_notchlist(noise_lines,binwidth)

    make_txt_file(noise_lines,binwidth,outfile=output_filename2)


if __name__=='__main__':
    binwidth=0.03125 # bin width of search
    version=10
    runPart='B'

    Produce_O3_Isotropic_notchlist(src='./',binwidth=binwidth,version=version,runPart=runPart)
    
