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
            hop: dict = None,
            uc: npt.NDArray[float] = None, 
            pos: ty.Sequence[ty.Sequence[float]] = None,
            orbs: dict = None,
            size: int = None,
            dim: int = None,
            orthonormal: bool = True,
            overlap: dict = None, 
            ):
        pass
    
            
