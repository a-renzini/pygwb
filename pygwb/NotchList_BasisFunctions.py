import numpy as np


class NoiseLine():
    def __init__(self,f0,df,description=None):
        """
        Represents a noise line with central frequency f0 and width df, so line is (f0-df/2,f0+df/2)

        Parameters
        ----------
        f0: float
            Central frequency
        df: float
            width
        description: str
            Reason to notch the line
        """
        self.f0=f0 
        self.df=df 
        self.description=description

class NotchedLine():
    def __init__(self,f0,nbins,description,binwidth):
        """
        Represents line in a form useful for notch list. Includes extra parameter, bin width of the search.

        Parameters
        ----------
        f0: float
            Central frequency
        nbins: int
            width in nr of bins
        description: str
            Reason to notch the line 
        binwidth = float
            Binwidth used in the analysis you want to apply this notch list to.
        """

        self.f0=f0
        self.nbins=nbins
        self.description=description
        self.binwidth=binwidth



def notch_line(noise_line,binwidth):
    """
    Convert a noise line to a notched line

    Parameters
    ----------
    noise_line: object of NoiseLine class
        The line you want to convert
    binwidth: float
        Binwidth used in the analysis you want to apply this notch list to.

    Returns
    -------
    myNotchedLine: NotchedLine objects
        The line you want to notch in NotchedLine format
    """
    fcentral=np.round(noise_line.f0/binwidth)*binwidth
    nbins=np.ceil(noise_line.df/binwidth) # using ceil is conservative
    # nbins should be odd
    if np.mod(nbins,2)==0:
        nbins=nbins+1
    myNotchedLine = NotchedLine(fcentral,nbins,noise_line.description,binwidth)
    return myNotchedLine


def make_notchlist(list_of_noise_lines,binwidth):
    """
    Produces a notchlist of NotchedLines given a list of noise lines
   

    Parameters
    ----------
    list_of_noise_lines: list of NoiseLine objects
        List of lines you want to be notched
    binwidth: float
        Binwidth used in the analysis you want to apply this notch list to.    

    Returns
    -------
    notchlist: list of NotchedLine objects
        List of lines you want to be notched in NotchedLine format with binwidth of the search included
    """
    notchlist=[]
    for notchtype in list_of_noise_lines:
        if notchtype is None:
            continue
        for line in notchtype:
            notchlist.append(notch_line(line,binwidth))
    return notchlist



def print_notchlist(notchlist,output_filename):
    """
    Print the notchlist in format that can be fed into paramfile

    Parameters
    ----------
    notchlist: a list of NotchedLine objects
        The list you want to notch
    output_filename: str
        Name of your output file
    """
    # freqsToRemove
    outstr='freqsToRemove '
    for notch in notchlist:
        outstr=outstr+str(notch.f0)+','
    outstr=outstr[:-1] # remove the last ','

    # nBinsToRemove
    outstr=outstr+'\nnBinsToRemove '
    for notch in notchlist:
        outstr=outstr+str(notch.nbins)+','
    outstr=outstr[:-1] # remove the last ','

    # write to file
    ff=open(output_filename,'w')
    ff.write(outstr)

#
###  I think this function is deprecated ####
###  at least not used for O3 as far as I know ###
#
#def make_notch_file(list_of_noise_lines,binwidth, outfile='all_notches.txt'):
#    try:
#        import os
#        os.system('rm {}'.format(outfile))
#    except:
#        pass
#   
#    notchlist = make_notchlist(list_of_noise_lines,binwidth)
#
#    notchlist = sorted(notchlist, key=lambda tup: tup[0])
#    final = []
#    for nn in notchlist:
#        f, b, d = nn
#        b = int(b)
#        for ii in range(-2*b, 2*b +1):
#            tf = f + ii/32.0
#            final.append(tf)
#
#    final = list(sorted(set(final)))
#    with open(outfile, 'a+') as g:
#        for ff in final:
#            g.write('{0:.5f}\n'.format(ff))


def make_txt_file(list_of_noise_lines,binwidth, outfile='notchlist.txt'):
    """
    Writes the notchlist to a txt-file in a easily human-readable format

    Parameters
    ----------
    notchlist: a list of NotchedLine objects
        The list you want to notch
    binwidth: float
        Binwidth used in the analysis you want to apply this notch list to.    
     output_filename: str
        Name of your output file
    """
    try:
        import os
        os.system('rm {}'.format(outfile))
    except:
        pass

#    notchlist = make_notchlist(list_of_noise_lines,binwidth)
    notchlist=[]
    for notchtype in list_of_noise_lines:
        if notchtype is None:
            continue
        for line in notchtype:
            nl = notch_line(line,binwidth)
            notchlist.append((nl.f0, nl.nbins, nl.description))

#    print(notchlist)

    notchlist = sorted(notchlist, key=lambda tup: tup[0])
    with open(outfile, 'a+') as g:
        g.write('Band\tDescription\n')
        for nn in notchlist:
            f, b, d = nn
            g.write('[{0:.2f}, {1:.2f}] Hz \t{2}\n'.format(f-b*binwidth/2.0,f+b*binwidth/2.0,d))



