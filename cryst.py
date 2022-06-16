import numpy as np
import numpy.linalg as la
import typing as ty
import numpy.typing as npt
import itertools as iter




#########################################################
# COMPUTE FIRST BRILLOUIN ZONE (VORONOI CONSTRUCTION) #
########################################################
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

    # Getting a set of G vectors  
    supercell_size = 3
    _grid = grid(lim = supercell_size, ndim = ndim) # generate grid 
    gvecs = [] # G vectors

    
    for p in _grid:
        gvec = np.dot(p, rec)
        gvecs.append(gvec) 

    # generate Voronoi tessellation and extract vertices surrounding Gamma point 
    vertices = _vertices_central_voronoi(gvecs)
    print(vertices)


# HELPERS FOR the Brilloiun Zone Function bz()
# -------------------------------------------------------- #
def _vertices_central_voronoi(
        seeds: ty.Sequence[ty.Sequence[float]]
        ):

    """
    Given seeds, construct a voronoi tessellation 
    compute the vertices forming the central region around the origin
    """
    
    seeds = np.array(seeds, dtype = float) # seeds to construct tessellation 
    
    from scipy.spatial import Voronoi
    vor = Voronoi(seeds) # create voronoi tessellation 
    nseeds = seeds.shape[0] # number of seeds 
    ndim = seeds.shape[1]   # number of dimensions 
    seed_idx_dict = dict()

    # create dict: keys store the cartesian coordinates of the seeds
    #              vals store the index of the points (index of the point in the array point_region) 
    for i in range(nseeds):
        seed_idx_dict[tuple(seeds[i])] = vor.point_region[i]

    central = tuple(np.zeros(shape = ndim, dtype = float)) # coords of origin 
    idx_central = seed_idx_dict[central] # index of the origin inside the point_region array 
    idx_central_vertices = vor.regions[idx_central] # gets the indices of the vertices surrounding the seed with index idx_central 

    central_vertices = np.array([vor.vertices[i] for i in idx_central_vertices], dtype = float) # gets the vertices 
    return central_vertices
    
    

        
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
    grid = list(iter.product(m, repeat = ndim))
    grid = np.array(grid)
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
