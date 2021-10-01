import numpy as np
from scipy.special import spherical_jn

from .constants import speed_of_light


def Tplus(alpha, beta):
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
    return (
        spherical_jn(0, alpha)
        + 5.0 / 7 * spherical_jn(2, alpha)
        + 3.0 / 112 * spherical_jn(4, alpha)
    ) * np.cos(beta / 2) ** 4


def Vplus(alpha, beta):
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
    return (
        spherical_jn(0, alpha)
        - 5.0 / 14 * spherical_jn(2, alpha)
        - 3.0 / 28 * spherical_jn(4, alpha)
    ) * np.cos(beta / 2) ** 4


def Splus(alpha, beta):
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
    return (
        spherical_jn(0, alpha)
        - 5.0 / 7 * spherical_jn(2, alpha)
        + 9.0 / 56 * spherical_jn(4, alpha)
    ) * np.cos(beta / 2) ** 4


def T_right_left(alpha, beta):
    return -np.sin(beta / 2) * (
        (-spherical_jn(1, alpha) + 7.0 / 8 * spherical_jn(3, alpha))
        + (spherical_jn(1, alpha) + 3.0 / 8 * spherical_jn(3, alpha)) * np.cos(beta)
    )


def tangent_vector(vector1, vector2):
    return np.subtract(
        vector2,
        np.multiply(np.dot(vector1, vector2) / np.dot(vector1, vector1), vector1),
    )


def omega_tangent_bisector(bisector, tangent_vector, perp):
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
    Calculates the tensor, scalar, and vector overlap reduction functions
    Following Section IVb of https://arxiv.org/abs/0903.0528
    See Appendix A of https://arxiv.org/abs/1704.08373 for a
    discussion of the normalization of the scalar ORF and
    https://arxiv.org/pdf/0707.0535.pdf for the gamma_V function

    Inputs:
    frequencies: frequencies at which to evaluate the ORFs
    det1_vertex: Coordinates of the vertex of detector 1
    det2_vertex: Coordinates of the vertex of detector 2
    det1_xarm: Coordinates of the x arm of detector 1
    det2_xarm: Coordinates of the x arm of detector 2
    det1_yarm: Coordinates of the y arm of detector 1
    det2_yarm: Coordinates of the y arm of detector 2
    Coordinates are always Earth-fixed cartesian

    Description of the intermediate parameters:
    beta: angle between detectors from center of earth
    tan_detX: tangent vector at detX along great circle between detectors
    bisector_detX: detX arm bisector vector
    perp_detX: outward radial vector perpendicular to the detector plane
    omega_detX: angle between bisector and tangent vector at detX
    perp: vector at theta=90 along great circle with det1_vertex theta=0

    Outputs:
    overlap_reduction_function: overlap reduction function at given frequencies
        for specified polarization
    """

    delta_x = np.subtract(det1_vertex, det2_vertex)
    alpha = 2 * np.pi * frequencies * np.linalg.norm(delta_x) / speed_of_light

    beta = np.arccos(
        np.dot(det1_vertex, det2_vertex)
        / (np.linalg.norm(det1_vertex) * np.linalg.norm(det2_vertex))
    )

    tan_det1 = tangent_vector(det1_vertex, det2_vertex)
    bisector_det1 = np.add(det1_xarm, det1_yarm)
    perp_det1 = -np.cross(det1_xarm, det1_yarm) / (
        np.linalg.norm(det1_xarm) * np.linalg.norm(det1_yarm)
    )

    perp = np.cross(np.cross(det1_vertex, det2_vertex), det1_vertex)
    tan_det2 = tangent_vector(det2_vertex, perp)
    bisector_det2 = np.add(det2_xarm, det2_yarm)
    perp_det2 = -np.cross(det2_xarm, det2_yarm) / (
        np.linalg.norm(det2_xarm) * np.linalg.norm(det2_yarm)
    )

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
