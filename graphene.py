from sympy import * 
from sympy.vector import *



# ----------------------------------------------------------------------------#
# ---------------------- DEFINING MODEL ------------------------------------- #
# --------------------------------------------------------------------------- #

# TIGHT-BINDING MODEL FOR GRAPHENE 

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

eigvals_muls = H.eigenvals()
eigvals = list(eigvals_muls.keys())

E1, E2 = eigvals[0], eigvals[1]  

print("Eigenvalues:")

pprint(simplify(E1))
pprint(simplify(E2))

# Modify the model:
# We study the finely tuned point in parameter space ta = tb = tc = t

t = symbols('t')

E1 = E1.subs(ta, t)
E1 = E1.subs(tb, t)
E1 = E1.subs(tc, t)

E2 = E2.subs(ta, t)
E2 = E2.subs(tb, t)
E2 = E2.subs(tc, t)

E1, E2 = simplify(E1), simplify(E2)

E1, E2 = E1.as_real_imag(), E2.as_real_imag() 
pprint(E1)
pprint(E2) 
