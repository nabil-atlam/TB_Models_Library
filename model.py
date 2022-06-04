import numpy as np
import typing as ty
import numpy.typing as npt




class model():
    """
    A class representing a model 
    Initialization: 
    hop = hopping parameters as dict 
    uc = unit cell 
    pos = positions of the orbitals 
    size = number of the orbitals in the unit cell 
    dim = number of spatial dimensions 
    orbs = orbitals as dict "index and name" 
    """

    # INITIALIZATION 
    def _init_(
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
        # set model parameters 
        self.hop = hop
        self.system_name = system_name
        self.uc = uc
        self.pos = pos
        self.orbs = orbs
        self.size = size
        self.dim = dim
        self.orthonormal = True

        if orthonormal is False:
            # non-orthonormal basis 
            self.overlap = overlap 
    
            
