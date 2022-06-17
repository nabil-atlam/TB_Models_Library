from sympy import * 
from sympy.vector import *



# ----------------------------------------------------------------------------#
# ---------------------- DEFINING MODEL ------------------------------------- #
# --------------------------------------------------------------------------- #
# orthonormal system
N = CoordSys3D('N')

# define momentum, lattice, and model parameters
ndim = 2 
ta, tb, tc, p, a1, a2 = symbols('t\u2090 t\u2091 t\u2092 p a\u2081 a\u2082', real = True)

p1, p2 = symbols('p\u2081 p\u2082', real = True)

#p1 = Dot(p, a1)
#p2 = Dot(p, a2)

elem = -tc - ta * exp(I * p1) - tb * exp(I * p2) 
H = Matrix([[0, elem], [conjugate(elem), 0]])
# write the Hamiltonian 

print("Bloch Hamiltonian:")
pprint(H)

eigvals = H.eigenvals()

print("Eigenvalues")

pprint(eigvals)

