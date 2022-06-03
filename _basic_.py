# Author: Nabil Atlam 

"""
Basic Methods in Python
"""

import numpy as np
import numpy.linalg as la 
import typing as ty
import numpy.typing as npt 
import scipy.io as scio
import scipy.linalg as sla 
import warnings 

# -------------------------------------------------------------------------#
# ------ COMPUTE RECIPROCAL LATTICE VECTORS FROM UNIT CELL VECTORS ------- #
# -------------------------------------------------------------------------#

def _get_rec_vec_(
        
        uc_vecs: npt.NDArray[np.float_]
        
        ) -> npt.NDArray[np.float_]:

    """
    compute the reciprocal lattice from the position space lattice
    """
    transposed = np.matrix.transpose(uc_vecs)
    rec_vecs = la.inv(transposed)
    
    return rec_vecs





# -------------------------------------------------------------------------------------#
# ------- GET TIGHT-BINDING HOPPING AND OVERLAP PARAMETERS FROM MAT DATA FILE -------- #
# -------------------------------------------------------------------------------------#
def _hash_from_mat(
        mat_filename: str,
        hop_table_name: str,
        overlap_table_name: str,
        size = None,
        dim = 3) -> ty.Sequence[ty.Dict]:
    """
    Get two Python dictionaries: hop and overlap from matlab data file
    """
    hop = dict()
    overlap = dict()
    
    # Read File
    data = scio.loadmat(mat_filename)
    _hop_data = np.array(data[hop_table_name], dtype = complex)
    _overlap_data = np.array(data[overlap_table_name], dtype = complex)

    # Get Size
    if size is None:
        size = _hop_data.max(axis = 0)[0]
        size = int(np.real(size))
        
    # Create Hopping and Overlap Matrices as Python Dictionaries #
    # Getting Hopping Parameters
    hop = _hop_overlap_as_dict(_hop_data, size, dim)
    overlap = _hop_overlap_as_dict(_overlap_data, size, dim)

    return hop, overlap


# HELPER FUNCTIONS FOR _hash_from_mat
def _hop_overlap_as_dict(matrix: npt.NDArray[np.complex_], size, dim = 3) -> dict:
    """
    takes a matrix and return a dictionary containing the hopping/overlap parameters
    """
    
    cut = int(-1 * dim) # useful variable. Usually = -3

    hop_overlp = dict()
    
    for entry in matrix:
        R = np.real(entry[cut:])
        R = tuple(R)

        orb_1, orb_2, t = entry[0], entry[1], entry[2]
        orb_1, orb_2 = np.real(orb_1), np.real(orb_2)
        orb_1, orb_2 = int(orb_1), int(orb_2)

        i = orb_1 - 1
        j = orb_2 - 1

        keys = list(hop_overlp.keys())
        mat = np.zeros((size, size) ,dtype = complex)
        mat[i, j] = t

        if R not in keys:
            new_key = {R:mat}
            hop_overlp.update(new_key)
        else:
            hop_overlp[R] = hop_overlp[R] + mat

    return hop_overlp





# -------------------------------------------------------------------------#
# ------------- COMPUTING BLOCH HAMILTONIAN AT A K POINT ----------------- #
# -------------------------------------------------------------------------#


def bloch_hamilton(
        hop: dict, 
        k: ty.Sequence[float],
        size: int = None,
        ) -> npt.NDArray[np.complex_]:
    """
    This computes the Bloch Hamiltonian at a given K point
    size: number of quantum states in the unit cell 
    k: the coordinates of the k point in terms of reciprocal lattice vectors 
    """

    if size is None:
        size = next(iter(hop.values())).shape[0] 

    k_array = np.array(k, dtype = float)
    H = np.zeros((size, size), dtype = complex)

    for R, t in hop.items():

        tmp = np.zeros((size, size) ,dtype = complex)
        kR = np.dot(k_array, R)
        np.multiply(np.exp(2j * np.pi * kR) ,t ,tmp)
        H = H + tmp 
    
    return H


def overlap_at_kpnt(
        overlap: dict,
        k: ty.Sequence[float],
        size: int = None,
        ) -> npt.NDArray[np.complex_]:
    """
    This computes the Overlap Matrix at a given k point
    size: number of quantum states in the unit cell 
    k: the coordinates of the k point in terms of reciprocal lattice vectors
    """
    if size is None:
        size = next(iter(overlap.values())).shape[0]

    k_array = np.array(k, dtype = float)
    overlap_kpnt = np.zeros((size, size), dtype = complex)

    for R, o in overlap.items():

        tmp = np.zeros((size, size) ,dtype = complex)
        kR = np.dot(k_array, R)
        np.multiply(np.exp(2j * np.pi * kR) ,o ,tmp)
        overlap_kpnt = overlap_kpnt + tmp

    return overlap_kpnt





# ----------------------------------------------------------------------------------------- #
# --------------------------- EIGENVALUES AND EIGENVECTORS AT A K POINT ------------------- #
# ----------------------------------------------------------------------------------------- #



def spectrum(
        bloch_ham: npt.NDArray[np.complex_],
        overlap: npt.NDArray[np.complex_],
        size: int = None,
        lowerbnd = None,
        upperbnd = None,
        ) -> float:
    """ 
    Computes the eigenvalues and the Eigenvectors 
    Solves Generalized Eigenvalue Problem
    """
    if size is None:
        size = bloch_ham.shape[0]
    
    evals, evecs = sla.eigh(a = bloch_ham, b = overlap)
    return evals, evecs


def eigvals(
        bloch_ham: npt.NDArray[np.complex_],
        overlap: npt.NDArray[np.complex_],
        size: int = None,
        lowerbnd = None,
        upperbnd = None,
        ) -> float:
    """
    Computes the Eigenvalues Only
    """
    if size is None:
        size = bloch_ham.shape[0]

    evals = sla.eigh(a = bloch_ham, b = overlap, eigvals_only = True)
    return evals 

# -------------------------------------------------------------------------------------------# 
# --------------------------- EIGENVALUES AT A SEQUENCE OF K_POINTS -------------------------#
# -------------------------------------------------------------------------------------------#

def bands(
        kpath: ty.Sequence[ty.Sequence[float]],
        hop: dict,
        overlap: dict,
        ) -> npt.NDArray[np.complex_]:
    """
    Compute the Band Structure along a K-Path
    kpath: a sequence of k-points 
    hop: the hopping dictionary
    overlap: the overlap dictionary 
    output: an array, first three elements encode the coordinates of the k point. the rest are the
    eigenvalues
    """
    size = next(iter(hop.values())).shape[0]
    dim = len(next(iter(hop.keys())))
    kpath_array = np.array(kpath, dtype = float)
    nkpnts = kpath_array.shape[0]

    #bs = np.zeros((nkpnts, dim + size), dtype = float)

    tmp_list_bs = []
    for k in kpath:
        bh = bloch_hamilton(hop, k) 
        om = overlap_at_kpnt(overlap, k)  
        evals = eigvals(bh, om) # List of Eigenvalues
        k_nd_eigns = np.append(k, evals)
        
        tmp_list_bs.append(k_nd_eigns)

    bs = np.stack(tmp_list_bs)

    return bs 


# ------------------------------------------------------------------------------------ #
# ---------------------------- PLOTTING BS ------------------------------------------- #
# ------------------------------------------------------------------------------------ #
def plot_bs(
        kpath: ty.Sequence[ty.Sequence[float]],
        hop,
        overlap,
        ):
    pass 









#hop, overlap = _hash_from_mat(mat_filename = "ftnfile(1layer).mat",
