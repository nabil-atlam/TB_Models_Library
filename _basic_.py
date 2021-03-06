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
import matplotlib.pyplot as plt 
import math 





# ------------------------------------------------------------------------ #
# ------------------------ MODEL OBJECT ---------------------------------- #
# ------------------------------------------------------------------------ #
class model:
    """
    Class encoding a TB Model. 
    Initialization:
    hop = hopping parameters as dict 
    uc = unit cell 
    pos = positions of the orbitals 
    size = number of the orbitals in the unit cell 
    dim = nimber of spatial dimensions 
    orbs = orbitals as dict "index and name" 
    """

    # Initialization 
    def __init__(
            self,
            system_name: str = None,
            hop: dict = None,
            uc: npt.NDArray[float] = None,
            pos: ty.Sequence[ty.Sequence[float]] = None,
            orbs: dict = None,
            size: int = None,
            dim: int = None,
            orthonormal: bool = True,
            overlap: dict = None,
            ):
        # set parameters
        self.hop = hop
        self.system_name = system_name
        self.uc = uc
        self.pos = pos
        self.orbs = orbs
        self.size = size
        self.dim = dim
        self.orthonormal = orthonormal

        if orthonormal is False:
            # Non orthonormal basis
            self.overlap = overlap 
        
        """
        
        """




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
def _model_from_mat(
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

def spectrum_from_model(
        k: ty.Sequence[float],
        hop: dict,
        overlap: dict,
        size: int = None,
        lowerbnd = None,
        upperbnd = None,
        ) -> float:
    """
    computes the spectrum given a model (hoppings and overlaps)
    """

    bl_ham_k = bloch_hamilton(hop, k)
    overlap_k = overlap_at_kpnt(overlap, k)
    eigvals, eigmodes = spectrum(bl_ham_k, overlap_k)
    return eigvals, eigmodes



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
# -------------------------- GENERATE K-PATH FROM A SEQUENCE OF HSPs------------------ #
# ------------------------------------------------------------------------------------ #
def kpath_from_hsps(
        hsps: ty.Sequence[ty.Sequence[float]],
        nkpts = 1000,
        ) -> ty.Sequence[ty.Sequence[float]]:

    """
    Generate K-Path
    """

    dim = hsps.shape[1] 
    hsps = np.array(hsps, dtype = float)
    pts = []
    num_hsps = hsps.shape[0]
    
    for i in range(num_hsps - 1):
        k1 = hsps[i]
        k2 = hsps[i + 1]
        ki_line = np.linspace(k1, k2, nkpts) # generate K-Points on a line 
        pts.append(ki_line)

    pts = np.stack(pts)

    num_kpts_total = int(pts.shape[0] * pts.shape[1])
    pts = np.reshape(pts, (num_kpts_total, dim))
    return pts




# ------------------------------------------------------------------------------------ #
# ---------------------------- PLOTTING BS ------------------------------------------- #
# ------------------------------------------------------------------------------------ #
def plot_bs(
        hsps: list[tuple],
        hop: dict,
        overlap: dict,
        uc: npt.NDArray[float],
        nkpts: int = 1000,
        energy_ran: tuple = None,
        ):
    """
    Plot the BS
    hsps: a list of tuples. First entry of tuple = labels e.g. K, M, Gamma etc. Sec. entry = coordinates of HSPs
    """

    print("Plotting ...")
    
    hsps_labels = list(item[0] for item in hsps)
    hsps_vals = list(item[1] for item in hsps)
    hsps_vals = np.stack(hsps_vals)
    uc = np.array(uc, dtype = float) 

    dim = uc.shape[0] # Number of Dimensions


    # Generate K-Path
    kpath = kpath_from_hsps(hsps_vals, nkpts)

    # Calculate Bands
    bs = bands(kpath, hop, overlap)
    bs = bs[:, dim : ]
    
    # Number of Bands (Number of orbitals in uc)
    size = bs.shape[1] 

    # Generate x array
    distances = [0.]
    x = []
    num_hsps = len(hsps_labels)

    for i in range(num_hsps -1):
        k1 = hsps_vals[i]
        k2 = hsps_vals[i + 1]
        d = dist_kpts(k1, k2, uc)
        d = distances[-1] + d
        _x_ = np.linspace(distances[-1], d, nkpts)
        distances.append(d)
        x.append(_x_)

    
    x = np.stack(x)
    x = x.flatten()

    for i in range(size):
        bnd_vals = bs[: , i] 
        plt.plot(x, bnd_vals, 'k', linewidth = 0.9)

    
    plt.xticks(distances, hsps_labels)
    #plt.tick_params(labelleft = False, left = False)
    if energy_ran is not None:
        plt.ylim(energy_ran)
    plt.savefig('plot.pdf')





# HELPER FUNCTIONS FOR plot_bs
def dist_kpts(
        k1: ty.Sequence[float],
        k2: ty.Sequence[float],
        uc: npt.NDArray[float],
        ) -> float:
    
    k1 = np.array(k1)
    k2 = np.array(k2)
    uc = np.array(uc)
    deltak = k1 - k2

    b = _get_rec_vec_(uc)
    b_transpose = np.matrix.transpose(b)
    deltak_cart = np.dot(b_transpose, deltak) # deltak in cartesian
    deltak_norm = la.norm(deltak_cart)

    return deltak_norm




# -------------------------------------------------------------------------- #
# -----------------------------WANNIER90 OUTPUT----------------------------- #
# -------------------------------------------------------------------------- #
def _wann_hr_to_arr(
        wann_hr_fn: str,
        ) -> npt.NDArray[float]:
    """
    Reads hopping data from wann hr dat file and produces array  
    """
    with open(wann_hr_fn, 'r') as wf:
        next(wf)

        num_wann = int(wf.readline())
        nrpts = int(wf.readline())

        # get number of rows containing WS grid degeneracies
        num_ints_per_row = 15
        
        if nrpts % num_ints_per_row == 0:
            num_deg_rows = int(nrpts / num_ints_per_row)
        else:
            num_deg_rows = math.floor(nrpts / num_ints_per_row) + 1
            

    wf = np.loadtxt(wann_hr_fn, dtype = float, skiprows = num_deg_rows + 3)

    return wf, num_wann, nrpts




# Reads hr wann file and returns a mapping 
def _get_hop_from_wann(
        wann_hr_fn: str
        ) -> dict:
    """ 
    Reads hopping parameters from hr file and returns a mapping
    """
    filedata = _wann_hr_to_arr(wann_hr_fn)
    hop_as_arr = filedata[0]
    size = filedata[1] # num_wann = system size 

    hop = dict() # dictionary to store the mapping

    ndim = 3 # number of dimensions
    
    for entry in hop_as_arr:
        R = entry[ : ndim]
        R = tuple(R)
        
        orb_1, orb_2 = entry[ndim], entry[ndim + 1]
        orb_1 = int(orb_1)
        orb_2 = int(orb_2)

        t = complex(entry[ndim + 2], entry[ndim + 3])

        i = orb_1 - 1
        j = orb_2 - 2

        keys = list(hop.keys())

        mat = np.zeros((size, size), dtype = complex)
        mat[i, j] = t

        if R not in keys:
            new_key = {R:mat}
            hop.update(new_key)
        else:
            hop[R] = hop[R] + mat

    return hop

        
# Compute eigenvalues of the model Hamiltonian assuming orthonormal basis         
def eigvals_orth(
        bloch_ham: npt.NDArray[np.complex_],
        size: int = None,
        lowerbnd = None,
        upperbnd = None,
        ) -> npt.NDArray[float]:
    """
    Computes the Eigenvalues Only -- Assumes orthonormal basis 
    """
    if size is None:
        size = bloch_ham.shape[0]

    evals = sla.eigh(a = bloch_ham, eigvals_only = True)
    return evals
    



# Compute bands assuming orthonormal basis
def bands_orth(
        kpath: ty.Sequence[ty.Sequence[float]],
        hop: dict,
        ) -> npt.NDArray[np.complex_]:
    """
    Compute the band structure along a K-path
    kpath: a sequence of k-points 
    hop: the hopping dictionary
    output: an array. First three elemends encode the kpoint coordinates. the rest are the 
    eigenvalues 
    """
    size = next(iter(hop.values())).shape[0]
    dim = len(next(iter(hop.keys())))
    kpath_array = np.array(kpath, dtype = float)
    nkpnts = kpath_array.shape[0]

    tmp_list_bs = []
    for k in kpath:
        bh = bloch_hamilton(hop, k)
        evals = eigvals_orth(bh)
        k_nd_eigns = np.append(k, evals)

        tmp_list_bs.append(k_nd_eigns)

    bs = np.stack(tmp_list_bs)

    return bs



# Plot BS From Wannier output file
def plot_bs_wann(
        hsps: list[tuple],
        hop: dict,
        uc: npt.NDArray[float],
        nkpts: int = 1000,
        energy_ran: tuple = None,
        ):
    """
    Plot the BS assuming orthonormal basis 
    hsps: a list of tuples. First entry of tuple = labels e.g. K, M, etc. Sec. entry 
    = coordinates of HSPs
    """

    print("Plotting ...")

    hsps_labels = list(item[0] for item in hsps)
    hsps_vals = list(item[1] for item in hsps)
    hsps_vals = np.stack(hsps_vals)
    uc = np.array(uc, dtype = float)

    dim = uc.shape[0] # number of dimensions

    # Generate K-Path
    kpath = kpath_from_hsps(hsps_vals, nkpts)

    # Calculate bands
    bs = bands_orth(kpath, hop)
    bs = bs[:, dim : ]

    # Num of Bands
    size = bs.shape[1]

    # Generate x array
    distances = [0.]
    x = []
    num_hsps = len(hsps_labels)

    for i in range(num_hsps -1):
        k1 = hsps_vals[i]
        k2 = hsps_vals[i + 1]
        d = dist_kpts(k1, k2, uc)
        d = distances[-1] + d
        _x_ = np.linspace(distances[-1], d, nkpts)
        distances.append(d)
        x.append(_x_)

    x = np.stack(x)
    x = x.flatten()

    for i in range(size):
        bnd_vals = bs[: , i]
        plt.plot(x, bnd_vals, 'b', linewidth = 0.9)

    plt.xticks(distances, hsps_labels)
    if energy_ran is not None:
        plt.ylim(energy_ran)
    plt.savefig('MnBi2Se4band.pdf')

    




# ------------------------------------------------------------------ #
# ------------------------- MISC ----------------------------------- #
# ------------------------------------------------------------------ #
def frm_wann_in_to_poscar(wann_in_fn: str) -> str:
    """
    Reads structure data from Wannier90 input file
    Returns data in POSCAR format 
    """
    pass













# --------------------------------------------------------------------- #
# ----------------------RUN AB INITIO CODE (VASP)---------------------- #
# --------------------------------------------------------------------- #

















    
# TODO: WRITE A FUNCTION TO GET MODEL FROM WANNIER 
#def _read_wann(
#        wann_file: str) -> model:
#    return 



hsps = [
    ('M', [0.5, 0.0, 0.0]),
    (r'$\Gamma$', [0.0,0.0,0.0]),
    ('M', [0.5, 0.0, 0.0])
    ]
    #('K', [0.333 ,0.333, 0.0]),
    #(r'$\Gamma$', [0.0, 0.0, 0.0])
    #]




# MAT file name and the names of the tables 
#mat_file_name = "ftnfile(1layer).mat"
#hop_tbl_name = "ftn60_h"
#overlap_tbl_name = "ftn60_s"

uc = [[4.3337998, 0.0 ,0.0],[-2.1668999 ,3.753180, 0.0],[0.0, 0.0, 26.9090614]]

# Run 
#hop, overlap = _model_from_mat(mat_file_name, hop_tbl_name, overlap_tbl_name)

#plot_bs(hsps = hsps, hop = hop, overlap = overlap, uc = uc, energy_ran = (-3, 2))

wann_fn = "wannier90_hr.dat"

hop = _get_hop_from_wann(wann_fn)

plot_bs_wann(hsps = hsps, hop = hop, uc = uc, energy_ran = (-.5,.5))


