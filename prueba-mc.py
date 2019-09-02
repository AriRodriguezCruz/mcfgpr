from pyModelChecking.LTL import *
from pyModelChecking import *

K=Kripke(R=[(0,1),(1,2),(2,3),(3,3)],L={0: ['p'], 1: ['p','q'],3 : ['p']})
psi=A(F(G('p')))

print(psi)
print(modelcheck(K,psi))