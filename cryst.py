import numpy as np
import numpy.linalg as la
import typing as ty
import numpy.typing as npt
import itertools as iter 

# GENERATE CUBIC GRID
def grid(
        ndim: int,
        lim: int,
        ):
    """
    Generate Regular Grid
    lim is the limit, since we want a bounded grid
    Num of points in each dimension is 2 * lim + 1 
    """

    size = 2 * lim + 1
    # Sequence of int from -lim to +lim 
    m = np.linspace(start = -lim, stop = lim, num = size, endpoint = True, dtype = int)
    grid = iter.product(m, repeat = ndim)
    return np.array(grid)




# COMPUTE THE BASIS FOR THE DUAL LATTICE
def get_rec(
        uc: ty.Sequence[ty.Sequence[float]]
        ) -> npt.NDArray[np.float_]:
    """
    compute the reciprocal lattice
    takes uc vectors and returns rec lattice vectors basis vecs as ndarray 
    """
    uc = np.array(uc) # to array 
    
    mat_transpose = np.matrix.transpose(uc)
    rec = la.inv(mat_transpose)
    
    # forget about 2 * pi factor 
    # rec = 2 * np.pi * rec_vecs

    return rec



# COMPUTE FIRST BRILLOUIN ZONE (VORONOI CONSTRUCTION)
def bz(
        uc: ty.Sequence[ty.Sequence[float]],
        supercell_size = 3
        ):
    """
    Compute the BZ
    Uses Scipy computational geometry algorithms 
    """
    rec = get_rec(uc)   # compute dual lattice basis vectors
    ndim = rec.shape[0] # number of dimensions

    # Getting Voronoi Cells 
    supercell_size = 3
    _grid = grid(lim = supercell_size, ndim = ndim) # generate grid 
    gvectors = [] # G vectors

    # Generate G vectors
    for i in range(-supercell_size, supercell_size + 1):
        for j in range(-supercell_size, supercell_size + 1):
            for k in range(-supercell_size, supercell_size + 1):
                G = []
                gvectors.append(G)
    return np.array(G_lst)
