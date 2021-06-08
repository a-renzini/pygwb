import numpy as np
from scipy.special import spherical_jn
from scipy.linalg import cholesky

# http://www.ligo.org/scientists/GW100916/detectors.txt


def ORF(freqs, baseline="HL"):

    # Detector geometry (Hanford)
    H1loc = np.array([-2.16141492636e6, -3.83469517889e6, 4.60035022664e6])
    H1x = np.array([-0.22389266154, 0.79983062746, 0.55690487831])
    H1y = np.array([-0.91397818574, 0.02609403989, -0.40492342125])

    # Livingston
    L1loc = np.array([-7.42760447238e4, -5.49628371971e6, 3.22425701744e6])
    L1x = np.array([-0.95457412153, -0.14158077340, -0.26218911324])
    L1y = np.array([0.29774156894, -0.48791033647, -0.82054461286])

    # Virgo
    V1loc = np.array([4.54637409900e6, 8.42989697626e5, 4.37857696241e6])
    V1x = np.array([-0.70045821479, 0.20848948619, 0.68256166277])
    V1y = np.array([-0.05379255368, -0.96908180549, 0.24080451708])

    # India (Geographical center, arms in the due west and north direction)
    lat = 25.484194
    lon = 78.884207
    R = 6.378e6
    theta = (90. - lat) * np.pi / 180.
    phi = lon * np.pi / 180.
    I1loc = R * np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi), np.cos(theta)])
    I1x = np.cross(I1loc / R, [0, 0, 1]) / \
        np.linalg.norm(np.cross(I1loc / R, [0, 0, 1]))
    I1y = np.cross(I1x, I1loc / R)

    # Construct detector tensors and separation vectors
    c = 2.998e8
    if baseline == "HL" or baseline == "LH":

        dX = np.linalg.norm(H1loc - L1loc)
        d = (H1loc - L1loc) / dX
        d2 = np.outer(d, d)
        D1 = (np.outer(H1x, H1x) - np.outer(H1y, H1y)) / 2.
        D2 = (np.outer(L1x, L1x) - np.outer(L1y, L1y)) / 2.

    elif baseline == "HV" or baseline == "VH":

        dX = np.linalg.norm(H1loc - V1loc)
        d = (H1loc - V1loc) / dX
        d2 = np.outer(d, d)
        D1 = (np.outer(H1x, H1x) - np.outer(H1y, H1y)) / 2.
        D2 = (np.outer(V1x, V1x) - np.outer(V1y, V1y)) / 2.

    elif baseline == "LV" or baseline == "VL":

        dX = np.linalg.norm(L1loc - V1loc)
        d = (L1loc - V1loc) / dX
        d2 = np.outer(d, d)
        D1 = (np.outer(L1x, L1x) - np.outer(L1y, L1y)) / 2.
        D2 = (np.outer(V1x, V1x) - np.outer(V1y, V1y)) / 2.

    elif baseline == "HI" or baseline == "IH":

        dX = np.linalg.norm(H1loc - I1loc)
        d = (H1loc - I1loc) / dX
        d2 = np.outer(d, d)
        D1 = (np.outer(H1x, H1x) - np.outer(H1y, H1y)) / 2.
        D2 = (np.outer(I1x, I1x) - np.outer(I1y, I1y)) / 2.

    elif baseline == "LI" or baseline == "IL":

        dX = np.linalg.norm(L1loc - I1loc)
        d = (L1loc - I1loc) / dX
        d2 = np.outer(d, d)
        D1 = (np.outer(L1x, L1x) - np.outer(L1y, L1y)) / 2.
        D2 = (np.outer(I1x, I1x) - np.outer(I1y, I1y)) / 2.

    else:
        print("Baseline not recognized.")

    # Helper array
    geometryVector = np.array([
        np.tensordot(D1, D2, axes=2),
        np.tensordot(np.tensordot(D1, D2, axes=1), d2, axes=2),
        np.tensordot(D1, d2, axes=2) * np.tensordot(D2, d2, axes=2)
    ])

    # Compute Bessel functions used in ORFs
    alphas = 2. * np.pi * freqs * dX / c
    j0 = spherical_jn(0, alphas)
    j2 = spherical_jn(2, alphas)
    j4 = spherical_jn(4, alphas)

    # Matrices from Nishizawa et al
    Tmatrix = np.array([
        [28., -40., 2.],
        [0., 120., -20.],
        [0., 0., 35.]
    ])
    Vmatrix = np.array([
        [7., 5., -2.],
        [0., -15., 20.],
        [0., 0., -35.]
    ])
    Smatrix = np.array([
        [14., 20., 6.],
        [0., -60., -60.],
        [0., 0., 105.]
    ])

    # Obtain ORF coefficients
    Tcoeffs = (1. / 14.) * np.dot(Tmatrix, np.array([j0, j2, j4]))
    Vcoeffs = (2. / 7.) * np.dot(Vmatrix, np.array([j0, j2, j4]))
    Scoeffs = (1. / 7.) * np.dot(Smatrix, np.array([j0, j2, j4]))

    # Compute ORFs
    Torf = np.dot(Tcoeffs.transpose(), geometryVector)
    Vorf = np.dot(Vcoeffs.transpose(), geometryVector)
    Sorf = np.dot(Scoeffs.transpose(), geometryVector)
    
  
    return Torf, Vorf, Sorf

def strain(freqs, psd, ORF12, ORF13, ORF23):
    """
    This function
    assumes frequency array goes from:
    
    df -> f_nyquist
    """

    #finding strain using the beta matrix for 3 detectors from Cella's paper
    #ORF12 = ORF(freqs, baseline = "HL")[0]
    #ORF13 = ORF(freqs, baseline = "HV")[0]
    #ORF23 = ORF(freqs, baseline = "LV")[0]
    
    fmax = np.amax(freqs) #use freqs.size instead cause will not always have steps of 1
    #frequency range, assumes we have taken steps of 1
    B = np.zeros((3, 3, freqs.size))
    

    for f in range (0, freqs.size):
        U = np.array([[1, ORF12[f], ORF13[f]], [ORF12[f], 1, ORF23[f]], [ORF13[f], ORF23[f], 1]])
        B[:, :, f] = cholesky(U, lower=True)
        #B[:, :, f] = np.array([[1, 0, 0],
                               #[Torf12[f], np.sqrt(1 - (Torf12[f])**2), 0],
                               #[Torf13[f], (Torf23[f] - Torf12[f] * Torf13[f])/(1 - np.sqrt(1 - (Torf12[f])**2)), np.sqrt((1 + 2 * Torf12[f] * Torf13[f] * Torf23[f] - (Torf12[f])**2 - (Torf13[f])**2 - (Torf23[f])**2)/(1 - (Torf12[f])**2))]])
    # B = np.split(B, freqs.size + 1, axis=1)
    # B = B[1:(freqs.size + 1)]   #this is our beta matrix

    # noise must be complex
    noise = np.random.randn(freqs.size, 3) + 1j*np.random.randn(freqs.size, 3)
    #psd = freqs # np.ones(freqs.size) #psd linear in freq

    df = freqs[2] - freqs[1]
    fsamp = (freqs[-1] + df) * 2
    deltaT  = 1. / fsamp

    Tdur = 1. / df
    Nsamps = Tdur / deltaT
    norm = np.sqrt(Nsamps / (2 * deltaT))
    
    # strain in frequrency domain
    H = np.zeros((3, freqs.size), dtype='complex')

    for f in range (0, freqs.size):
        #take psd(f) = const, should make this an argument of the f-on
        H[:, f] = norm * (psd[f]**0.5) * (1/np.sqrt(2)) * B[:, :, f].dot(noise[f, :])

    # H_final is strain in time domain
    if np.mod(H[0].size, 2) == 0:
        H_final = np.zeros((H.shape[0], 2 * (H[0].size + 1)), dtype='complex')
    else:
        H_final = np.zeros((H.shape[0], 2 * H[0].size + 1), dtype='complex')
    # take inverse Fourier transforms        
    for ii in range(H.shape[0]):
        if np.mod(H[ii].size, 2) == 0:
            H_final[ii] = np.real(np.fft.ifft(np.append(np.append([0], H[ii]), np.append([0], np.flipud(np.conj(H[ii]))))))
        else:
            H_final[ii] = np.real(np.fft.ifft(np.append(np.append([0], H[ii]), np.flipud(np.conj(H[ii])))))
    return H_final