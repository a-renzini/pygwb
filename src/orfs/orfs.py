import numpy as np
from scipy.special import sperical_jn
from ..constants import speed_of_light

def Tplus(alpha,beta):
    return (-(3./8*spherical_jn(0,alpha)-45./56*spherical_jn(2,alpha)+169./896*spherical_jn(4,alpha))+
            (.5*spherical_jn(0,alpha)-5./7*spherical_jn(2,alpha)-27./224*spherical_jn(4,alpha))*np.cos(beta)-
            (1./8*spherical_jn(0,alpha)+5./56*spherical_jn(2,alpha)+3./896*spherical_jn(4,alpha))*np.cos(2*beta))

def Tminus(alpha,beta):
    return (spherical_jn(0,alpha)+5./7*spherical_jn(2,alpha)+3./112*spherical_jn(4,alpha))*np.cos(beta/2)**4

def Vplus(alpha,beta):
    return (-(3./8*spherical_jn(0,alpha)+45./112*spherical_jn(2,alpha)+169./224*spherical_jn(4,alpha))+
            (.5*spherical_jn(0,alpha)+5./14*spherical_jn(2,alpha)+27./56*spherical_jn(4,alpha))*np.cos(beta)-
            (1./8*spherical_jn(0,alpha)-5./122*spherical_jn(2,alpha)-3./224*spherical_jn(4,alpha))*np.cos(2*beta))

def Vminus(alpha,beta):
    return (spherical_jn(0,alpha)-5./14*spherical_jn(2,alpha)-3./28*spherical_jn(4,alpha))*np.cos(beta/2)**4

def Splus(alpha,beta):
    return (-(3./8*spherical_jn(0,alpha)+45./56*spherical_jn(2,alpha)+507./448*spherical_jn(4,alpha))+
            (0.5*spherical_jn(0,alpha)+5./7*spherical_jn(2,alpha)-81./112*spherical_jn(4,alpha))*np.cos(beta)-
            (1./8*spherical_jn(0,alpha)-5./56*spherical_jn(2,alpha)+9./448*spherical_jn(4,alpha))*np.cos(2*beta))

def Sminus(alpha,beta):
    return (spherical_jn(0,alpha)-5./7*spherical_jn(2,alpha)+9./56*spherical_jn(4,alpha))*np.cos(beta/2)**4

def calc_orfs(freqs, det1_vertex, det2_vertex, det1_xarm, det2_xarm, det1_yarm, det2_yarm):
    '''
    Calculates the tensor, scalar, and vector overlap reduction funtions
    
    Inputs:
    freqs: frequencies at which to evaluate the ORFs
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
    omega_detX: angle between bisector and tangent vector at detX
    perp: vector at theta=90 along great circle with det1_vertex theta=0

    Outputs:
    orf_T: tensor overlap reduction function at given frequencies
    orf_V: vector ORF at given frequencies
    orf_S: scalar ORF at given frequencies
    '''

    delta_x = np.subtract(det1_vertex,det2_vertex)
    alpha = 2*np.pi*freqs*np.linalg.norm(delta_x)/speed_of_light

    beta = np.arccos(np.dot(det1_vertex,det2_vertex)/(np.linalg.norm(det1_vertex)*np.linalg.norm(det2_vertex)))
    tan_det1 = np.subtract(det2_vertex,np.multiply(np.dot(det1_vertex,det2_vertex)/np.dot(det1_vertex,det1_vertex),det1_vertex))
    bisector_det1 = np.add(det1_xarm,det1_yarm)
    omega_det1 = np.arccos(np.dot(bisector_det1,tan_det1)/(np.linalg.norm(bisector_det1)*np.linalg.norm(tan_det1)))

    perp = np.cross(np.cross(det1_vertex,det2_vertex),det1_vertex)
    tan_det2 = np.subtract(perp,np.multiply(np.dot(det2_vertex,perp)/np.dot(det2_vertex,det2_vertex),det2_vertex))
    bisector_det2 = np.add(det2_xarm,det2_yarm)
    omega_det2 = np.arccos(np.dot(bisector_det2,tan_det2)/(np.linalg.norm(bisector_det2)*np.linalg.norm(tan_det2)))

    omega_plus = (omega_det1+omega_det2)/2
    omega_minus = (omega_det1-omega_det2)/2

    orf_T = Tplus(alpha,beta)*np.cos(4*omega_plus)+Tminus(alpha,beta)*np.cos(4*omega_minus)
    orf_V = Vplus(alpha,beta)*np.cos(4*omega_plus)+Vminus(alpha,beta)*np.cos(4*omega_minus)
    orf_S = Splus(alpha,beta)*np.cos(4*omega_plus)+Sminus(alpha,beta)*np.cos(4*omega_minus)
    return orf_T, orf_V, orf_S

