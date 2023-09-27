"""
The relative position and orientation of two detectors is taken into account by the 
overlap reduction function (ORF) in gravitational-wave background searches. The ``orfs``
module combines the different component methods to compute the ORF for a given detector
baseline, and polarization. By default, general relativity (GR) polarization is assumed, i.e., tensor. However, 
the ``orfs`` module also supports non-GR polarizations (scalar and vector). For more information about
the ORF, see `here <https://arxiv.org/pdf/1608.06889.pdf>`_.

Examples
--------
    
To illustrate how to compute the ORF, we start by 
importing the relevant packages:
    
>>> import numpy as np
>>> from pygwb.orfs import *
>>> import matplotlib.pyplot as plt
    
For concreteness, we consider the LIGO Hanford-Livingston baseline, and compute
the ORF for this baseline. We define empty detectors:
    
>>> H1 = bilbydet.get_empty_interferometer('H1')
>>> L1 = bilbydet.get_empty_interferometer('L1')
    
We  now compute the ORF for a set of frequencies by using the relevant information 
contained in the interferometer objects defined above:
    
>>> freqs = np.arange(10.25, 256.25, 0.25)
>>> orf = calc_orf(freqs, H1.vertex, L1.vertex, H1.x, L1.x, H1.y, L1.y, polarization = "tensor")
    
Note that the ``calc_orf`` method combines the various other methods of the module. 
    
Note that, in practice, these methods are not called by the user, but are
called by the ``baseline`` module directly. For more information on how the ``orfs`` module
interacts with the ``baseline`` module, see :doc:`pygwb.baseline`.
"""

import numpy as np
from scipy.special import spherical_jn

from .constants import speed_of_light


def Tplus(alpha, beta):
    """
    Function used in the computation of the tensor ORF, as given by 
    Eq. (34) of https://arxiv.org/pdf/0903.0528.pdf.
    
    Parameters
    =======
    
    alpha: ``array_like``
        Given below Eq. (32) of https://arxiv.org/pdf/0903.0528.pdf. 
        Has same shape as frequencies used to compute the ORF.
    
    beta: ``float``
        Angle between detectors from center of the Earth.
        
    Returns
    =======
    
    Tplus: ``array_like``
        Tplus parameter, as defined in Eq. (34) of https://arxiv.org/pdf/0903.0528.pdf.

    See also
    --------
    scipy.special.spherical_jn
        More information `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_jn.html>`_.
    """
    return (
        -(
            3.0 / 8 * spherical_jn(0, alpha)
            - 45.0 / 56 * spherical_jn(2, alpha)
            + 169.0 / 896 * spherical_jn(4, alpha)
        )
        + (
            0.5 * spherical_jn(0, alpha)
            - 5.0 / 7 * spherical_jn(2, alpha)
            - 27.0 / 224 * spherical_jn(4, alpha)
        )
        * np.cos(beta)
        - (
            1.0 / 8 * spherical_jn(0, alpha)
            + 5.0 / 56 * spherical_jn(2, alpha)
            + 3.0 / 896 * spherical_jn(4, alpha)
        )
        * np.cos(2 * beta)
    )

def Tminus(alpha, beta):
    """
    Function used in the computation of the tensor ORF, as given by 
    Eq. (35) of https://arxiv.org/pdf/0903.0528.pdf.
    
    Parameters
    =======
    
    alpha: ``array_like``
        Given below Eq. (32) of https://arxiv.org/pdf/0903.0528.pdf. 
        Has same shape as frequencies used to compute the ORF.
    
    beta: ``float``
        Angle between detectors from center of the Earth.
    
    Returns
    =======
    
    Tminus: ``array_like``
        Tminus parameter, as defined in Eq. (35) of https://arxiv.org/pdf/0903.0528.pdf.

    See also
    --------
    scipy.special.spherical_jn
        More information `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_jn.html>`_.
    """
    return (
        spherical_jn(0, alpha)
        + 5.0 / 7 * spherical_jn(2, alpha)
        + 3.0 / 112 * spherical_jn(4, alpha)
    ) * np.cos(beta / 2) ** 4

def Vplus(alpha, beta):
    """
    Function used in the computation of the vector ORF, as given by 
    Eq. (37) of https://arxiv.org/pdf/0903.0528.pdf.
    
    Parameters
    =======
    
    alpha: ``array_like``
        Given below Eq. (32) of https://arxiv.org/pdf/0903.0528.pdf. 
        Has same shape as frequencies used to compute the ORF.
    
    beta: ``float``
        Angle between detectors from center of the Earth.
    
    Returns
    =======
    
    Vplus: ``array_like``
        Vplus parameter, as defined in Eq. (37) of https://arxiv.org/pdf/0903.0528.pdf.

    See also
    --------
    scipy.special.spherical_jn
        More information `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_jn.html>`_.
    """
    return (
        -(
            3.0 / 8 * spherical_jn(0, alpha)
            + 45.0 / 112 * spherical_jn(2, alpha)
            + 169.0 / 224 * spherical_jn(4, alpha)
        )
        + (
            0.5 * spherical_jn(0, alpha)
            + 5.0 / 14 * spherical_jn(2, alpha)
            + 27.0 / 56 * spherical_jn(4, alpha)
        )
        * np.cos(beta)
        - (
            1.0 / 8 * spherical_jn(0, alpha)
            - 5.0 / 112 * spherical_jn(2, alpha)
            - 3.0 / 224 * spherical_jn(4, alpha)
        )
        * np.cos(2 * beta)
    )

def Vminus(alpha, beta):
    """
    Function used in the computation of the vector ORF, as given by 
    Eq. (38) of https://arxiv.org/pdf/0903.0528.pdf
    
    Parameters
    =======
    
    alpha: ``array_like``
        Given below Eq. (32) of https://arxiv.org/pdf/0903.0528.pdf. 
        Has same shape as frequencies used to compute the ORF.
    
    beta: ``float``
        Angle between detectors from center of the Earth.
    
    Returns
    =======
    
    Vminus: ``array_like``
        Vminus parameter, as defined in Eq. (38) of https://arxiv.org/pdf/0903.0528.pdf.

    See also
    --------
    scipy.special.spherical_jn
        More information `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_jn.html>`_.
    """
    return (
        spherical_jn(0, alpha)
        - 5.0 / 14 * spherical_jn(2, alpha)
        - 3.0 / 28 * spherical_jn(4, alpha)
    ) * np.cos(beta / 2) ** 4

def Splus(alpha, beta):
    """
    Function used in the computation of the scalar ORF, as given by 
    Eq. (40) of https://arxiv.org/pdf/0903.0528.pdf.
    
    Parameters
    =======
    
    alpha: ``array_like``
        Given below Eq. (32) of https://arxiv.org/pdf/0903.0528.pdf. 
        Has same shape as frequencies used to compute the ORF.
    
    beta: ``float``
        Angle between detectors from center of the Earth.
    
    Returns
    =======
    
    Splus: ``array_like``
        Splus parameter, as defined in Eq. (40) of https://arxiv.org/pdf/0903.0528.pdf.

    See also
    --------
    scipy.special.spherical_jn
        More information `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_jn.html>`_.
    """
    return (
        -(
            3.0 / 8 * spherical_jn(0, alpha)
            + 45.0 / 56 * spherical_jn(2, alpha)
            + 507.0 / 448 * spherical_jn(4, alpha)
        )
        + (
            0.5 * spherical_jn(0, alpha)
            + 5.0 / 7 * spherical_jn(2, alpha)
            - 81.0 / 112 * spherical_jn(4, alpha)
        )
        * np.cos(beta)
        - (
            1.0 / 8 * spherical_jn(0, alpha)
            - 5.0 / 56 * spherical_jn(2, alpha)
            + 9.0 / 448 * spherical_jn(4, alpha)
        )
        * np.cos(2 * beta)
    )

def Sminus(alpha, beta):
    """
    Function used in the computation of the scalar ORF, as given by 
    Eq. (41) of https://arxiv.org/pdf/0903.0528.pdf
    
    Parameters
    =======
    
    alpha: ``array_like``
        Given below Eq. (32) of https://arxiv.org/pdf/0903.0528.pdf. 
        Has same shape as frequencies used to compute the ORF.
    
    beta: ``float``
        Angle between detectors from center of the Earth.
    
    Returns
    =======
    
    Sminus: ``array_like``
        Sminus parameter, as defined in Eq. (41) of https://arxiv.org/pdf/0903.0528.pdf.

    See also
    --------
    scipy.special.spherical_jn
        More information `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_jn.html>`_.
    """
    return (
        spherical_jn(0, alpha)
        - 5.0 / 7 * spherical_jn(2, alpha)
        + 9.0 / 56 * spherical_jn(4, alpha)
    ) * np.cos(beta / 2) ** 4

def T_right_left(alpha, beta):
    """
    Parameters
    =======
    
    alpha: ``array_like``
        Given below Eq. (32) of https://arxiv.org/pdf/0903.0528.pdf. 
        Has same shape as frequencies used to compute the ORF.
    
    beta: ``float``
        Angle between detectors from center of the Earth.
    
    Returns
    =======
    T_right_left: ``array_like``
        T_right_left parameter as defined below Eq. (8) of 
        https://arxiv.org/pdf/0707.0535.pdf.

    See also
    --------
    scipy.special.spherical_jn
        More information `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_jn.html>`_.
    """
    return -np.sin(beta / 2) * (
        (-spherical_jn(1, alpha) + 7.0 / 8 * spherical_jn(3, alpha))
        + (spherical_jn(1, alpha) + 3.0 / 8 * spherical_jn(3, alpha)) * np.cos(beta)
    )

def tangent_vector(vector1, vector2):
    """
    Method to compute the tangent vector given two vectors.
    
    Parameters
    =======
    
    vector1: ``array_like``
    
    vector2: ``array_like``
    
    Returns
    =======
    
    tanget_vector: ``array_like``
        Tangent vector to vector1 and vector2.
    
    """
    return np.subtract(
        vector2,
        np.multiply(np.dot(vector1, vector2) / np.dot(vector1, vector1), vector1),
    )

def omega_tangent_bisector(bisector, tangent_vector, perp):
    """
    Method to compute the angle between bisector and tangent vector.
    
    Parameters
    =======
    
    bisector: ``array_like``
        Bisector vector.
        
    tangent_vector: ``array_like``
        Tangent vector at detector X along great circle between detectors.
        
    perp: ``array_like``
        Outward radial vector perpendicular to the detector plane.
        
    Returns
    =======
    
    omega_detX: ``float``
        Angle between bisector and tangent vector at detector X.
    
    """
    norm = np.linalg.norm(bisector) * np.linalg.norm(tangent_vector)
    sin_omega = np.dot(np.cross(bisector, tangent_vector), perp) / norm
    cos_omega = np.dot(bisector, tangent_vector) / norm
    return np.arctan2(sin_omega, cos_omega)

def calc_orf(
    frequencies,
    det1_vertex,
    det2_vertex,
    det1_xarm,
    det2_xarm,
    det1_yarm,
    det2_yarm,
    polarization="tensor",
):
    """
    Calculates the tensor, scalar, and vector overlap reduction functions, 
    following Section IVb of https://arxiv.org/abs/0903.0528. See Appendix A
    of https://arxiv.org/abs/1704.08373 for a
    discussion of the normalization of the scalar ORF and
    https://arxiv.org/pdf/0707.0535.pdf for the vector ORF function.
    
    Parameters
    =======
    
    frequencies: ``array_like``
        Frequencies at which to evaluate the ORFs.
        
    det1_vertex: ``array_like``
        Coordinates (Earth-fixed cartesian, in meters) of the vertex of detector 1.
    
    det2_vertex: ``array_like``
        Coordinates (Earth-fixed cartesian, in meters) of the vertex of detector 2.
    
    det1_xarm: ``array_like``
        Unit vector (Earth-fixed cartesian) along the x arm of detector 1.
    
    det2_xarm: ``array_like``
        Unit vector (Earth-fixed cartesian) along the x arm of detector 2.
    
    det1_yarm: ``array_like``
        Unit vector (Earth-fixed cartesian) along the y arm of detector 1.
    
    det2_yarm: ``array_like``
        Unit vector (Earth-fixed cartesian) along the y arm of detector 2.
    
    polarization: ``str``, optional
        Polarization used in the computation of the overlap reduction function. Default is tesnor.
    

    Intermediate parameters
    =======================

    beta: ``float``
        Angle between detectors from center of earth.
        
    tan_detX: ``array_like``
        Tangent vector at detector X along great circle between detectors.
     
    bisector_detX: ``array_like``
        Bisector vector for detector X.
        
    perp_detX: ``array_like``
        Inward radial unit vector perpendicular to the detector plane for detector X.
        
    omega_detX: ``float``
        Angle between bisector and tangent vector at detector X.
        
    perp: ``array_like``
        Vector at theta=90 along great circle with det1_vertex theta=0.

    Returns
    =======
    
    overlap_reduction_function: ``array_like``
        Overlap reduction function at given frequencies for specified polarization.
    """
    delta_x = np.subtract(det1_vertex, det2_vertex)
    alpha = 2 * np.pi * frequencies * np.linalg.norm(delta_x) / speed_of_light

    beta = np.arccos(
        np.dot(det1_vertex, det2_vertex)
        / (np.linalg.norm(det1_vertex) * np.linalg.norm(det2_vertex))
    )

    tan_det1 = tangent_vector(det1_vertex, det2_vertex)
    bisector_det1 = np.add(det1_xarm, det1_yarm)
    perp1_unnormalized = np.cross(det1_xarm, det1_yarm)
    perp_det1 = -perp1_unnormalized / np.linalg.norm(perp1_unnormalized)

    perp = np.cross(np.cross(det1_vertex, det2_vertex), det1_vertex)
    tan_det2 = tangent_vector(det2_vertex, perp)
    bisector_det2 = np.add(det2_xarm, det2_yarm)
    perp2_unnormalized = np.cross(det2_xarm, det2_yarm)
    perp_det2 = -perp2_unnormalized / np.linalg.norm(perp2_unnormalized)

    if np.linalg.norm(delta_x) != 0:
        omega_det1 = omega_tangent_bisector(bisector_det1, tan_det1, perp_det1)
        omega_det2 = omega_tangent_bisector(bisector_det2, tan_det2, perp_det2)
        omega_plus = (omega_det1 + omega_det2) / 2
        omega_minus = (omega_det1 - omega_det2) / 2
    else:
        omega_plus = 1
        omega_minus = (
            omega_tangent_bisector(bisector_det1, bisector_det2, perp_det1) / 2
        )

    if polarization.lower() == "tensor":
        overlap_reduction_function = Tplus(alpha, beta) * np.cos(
            4 * omega_plus
        ) + Tminus(alpha, beta) * np.cos(4 * omega_minus)
    elif polarization.lower() == "vector":
        overlap_reduction_function = Vplus(alpha, beta) * np.cos(
            4 * omega_plus
        ) + Vminus(alpha, beta) * np.cos(4 * omega_minus)
    elif polarization.lower() == "scalar":
        overlap_reduction_function = (
            1.0
            / 3
            * (
                Splus(alpha, beta) * np.cos(4 * omega_plus)
                + Sminus(alpha, beta) * np.cos(4 * omega_minus)
            )
        )
    elif polarization.lower() == "right_left":
        overlap_reduction_function = T_right_left(alpha, beta) * np.sin(4 * omega_plus)
    else:
        raise ValueError(
            "Unrecognized polarization! Must be either tensor, vector, scalar, or right_left"
        )
    return overlap_reduction_function