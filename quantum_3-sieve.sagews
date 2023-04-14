import numpy as np
k=3
CL23, KMPM19 = 0, 1

# Volume of a spherical cap of angle alpha
def V(alpha): return n( log(sin(alpha))/log(2) )

def Exp(x): return n( log(x)/log(2))

# Volume of the spherical cap satisfying the partial configuration Cij
def RatioFilter(Cij): return V(acos(-Cij))

# Determinant of configuration C of size 3x3
def det3(C):
    C12, C13, C23 = C
    return Exp(1 + 2*C12*C13*C23 -C12**2 -C13**2 -C23**2)

def det3bis(C):
    C12, C13, C23 = C
    MatC = np.array([
                [1, C12, C13],
                [C12, 1, C23],
                [C13, C23, 1]], dtype=np.float64)
    detC = np.linalg.det(MatC)
    return Exp(detC)



#========== Configurations ==========
# C is the configuration of the lattice vectors x
# C_ is the configuration of the residial vectors y

# Return C'_ij given C_ij (See Lemma 1)
def xy_config(alpha, Cij): k = 3 ; return (Cij + cos(alpha)**2/(k-1))/sin(alpha)**2

# Return C' given C
def residual(alpha, C):
    C12, C13, C23 = C
    C12_ = xy_config(alpha, C12)
    C13_ = xy_config(alpha, C13)
    C23_ = xy_config(alpha, C23)
    return (C12_, C13_, C23_)

# Return C_ij given C'_ij
def yx_config(alpha, Cij_): k = 3 ; return -cos(alpha)**2/(k-1) + sin(alpha)**2 * Cij_

# Recover C from C'
def lattice_config(alpha, C_):
    C12_, C13_, C23_ = C_
    C12 = yx_config(alpha, C12_)
    C13 = yx_config(alpha, C13_)
    C23 = yx_config(alpha, C23_)
    return (C12, C13, C23)



#========== Complexity of the 3-sieve ==========

# Size of the input list L for the sieve
def L(C): k = 3 ; return -det3(C)/(2*(k-1))

# Size of the input lists for the subproblem
def SizeLi(alpha, C): return L(C) + V(alpha)

# Number of tuple-filters (See Equation 4)
def NbFilters(alpha): return -V(alpha)

# Probability that a triplet is a solution of the subproblem
def PrSol(alpha, C): C_ = residual(alpha, C) ; return det3(C_)/2

# Number of solutions-triplets in one tuple-filter
def NbSolPerFilter(alpha, C): k = 3 ; return SizeLi(alpha, C)*k + PrSol(alpha, C)

# Number of repeats of steps 1 and 2 (See Lemma 3)
def NbRep(alpha, C): return max(0, L(C) - (NbSolPerFilter(alpha, C) + NbFilters(alpha)) )

# Number of tuple-filters we need to look in for a solution (in the case there are less than one solution per tuple-filter)
def NbFiltersToLook(alpha, C): return max(0, -NbSolPerFilter(alpha, C) )

# Size of the intermediate lists L_ij (See Proposition 6)
def SizeLiyj(alpha, C, Cij_): return max(0, SizeLi(alpha, C) + RatioFilter(Cij_) )


# Time to find all the triplets of configuration C in a tuple-filter of angle alpha
def FASq3(alpha, C):
    C_ = residual(alpha, C)
    C12_, C13_, C23_ = C_
    Li = SizeLi(alpha, C)
    Sol = NbSolPerFilter(alpha, C)

    grover1 = max(0, -RatioFilter(C12_)/2, -RatioFilter(C13_)/2)

    L2y1 = SizeLiyj(alpha, C, C12_)
    L3y1 = SizeLiyj(alpha, C, C13_)
    AA = (Li-Sol)/2 + (L2y1/2 + L3y1/2)

    C = lattice_config(alpha, C_)
    ToLook = max(0, NbFiltersToLook(alpha, C))
    return Sol + grover1 + AA + ToLook/2

# Time to solve SVP by the quantum 3-sieve with parameters angle alpha and configuration C
def quantum_sieve3(alpha, C):
    NbReps = NbRep(alpha, C)
    NbTupleFilters = -V(alpha)
    FAS = FASq3(alpha, C)
    return NbReps + max(L(C), NbTupleFilters + FAS)

# Time to solve SVP by searching triplets of balanced configuration, using a prefiltering of angle alpha
def quantum_sieve3_balanced(alpha): return quantum_sieve3(alpha, (-1/3, -1/3, -1/3))



#========== Displaying ==========
def aff(alpha, C):
    C_ = residual(alpha, C)
    C12_, C13_, C23_ = C_
    print("qtime =", quantum_sieve3(alpha, C))
    print("|L| =", L(C))
    print("alpha =", alpha)
    print("C =", C)
    print("C_ =", residual(alpha, C) )
    #print("PrSol =", PrSol(alpha, C) )
    print("|Sol|=", NbSolPerFilter(alpha, C))
    print("NbRep =", NbRep(alpha, C) )
    #print("QNbFiltersToLook =", NbFiltersToLook(alpha, C)/2)
    print("1/V(alpha) =", NbFilters(alpha), "(nombre de tuple-filters)")
    print("Li =", SizeLi(alpha, C) )
    print("L2(y1) =", SizeLi(alpha, C) + RatioFilter(C12_))
    print("L3(y1) =", SizeLi(alpha, C) + RatioFilter(C13_))
    print("FAS =", FASq3(alpha, C))
    print("grover1 =", max(0, -RatioFilter(C12_)/2, -RatioFilter(C13_)/2))

    Li = SizeLi(alpha, C)
    Sol = NbSolPerFilter(alpha, C)
    L2y1 = SizeLiyj(alpha, C, C12_)
    L3y1 = SizeLiyj(alpha, C, C13_)
    grover1 = max(0, -RatioFilter(C12_)/2, -RatioFilter(C13_)/2)
    AA = (Li-Sol)/2 + (L2y1/2 + L3y1/2)
    print("AA =", AA)
    print("find1sol =", grover1 + AA)
    print( )


#========== Optimization ==========
# Parameters:
# - alpha: angle of prefiltering
# - Configuration C =(C12, C13, C23).
#       As our algorithm uses C12 and C13 symmetrically, we can set C12=C13.
#       C23 depends on C12 and C13 (See Sec 2.3): C23 = -1-C12-C13 = -1-2*C12

# Return (optimized time over C12, optimal C12) given alpha
def opt_C12(alpha):
    def partial(C12):
        C = (C12, C12, -1-2*C12)
        return quantum_sieve3(alpha, C)
    return find_local_minimum(partial, -1, 0, tol=0.001) #t, C12

# Return (optimized time over alpha, optimal alpha)
def opt_alpha():
    def partial(alpha): return opt_C12(alpha)[0]
    return find_local_minimum(partial, pi/4, pi/2, tol=0.001) #t, alpha



#========== Results ==========
if 1: # [KMPM19] BLS Alg 4.1
    print("\n====== Balanced configuration & No prefiltering ======")
    alpha = n(pi/2) ; C=(-1/3, -1/3, -1/3)
    print("time =", quantum_sieve3(alpha, C))
    print("mem  =", L(C))

if 1: # Theorem 6
    print("\n==== Balanced configuration & Prefiltering (Memory-optimizing parameters) ====")
    time, alpha = find_local_minimum(quantum_sieve3_balanced, pi/4, pi/2)
    C = (-1/3, -1/3, -1/3)
    print("time =", quantum_sieve3(alpha, C))
    print("mem  =", L(C))
    #aff(alpha, C)

if 1: # Theorem 7
    print("\n==== Any configuration & Prefiltering (Time-optimizing parameters) ====")
    alpha = opt_alpha()[1]
    C12 = C13 = opt_C12(alpha)[1]
    C23 = -1-C12-C13
    C = (C12, C13, C23)
    print("time =", quantum_sieve3(alpha, C))
    print("mem  =", L(C))
    #aff(alpha, C)

if 0: # Trade-off
    ALGO = CL23 # <--- Choose KMPM19 or CL23
    print("\n==== Trade-off Quantum 3-sieve", ["[CL23]", "[KMPM19] Alg 4.1 (BLS)"][ALGO],  "====")
    T = []
    M = []
    A = []
    for C12 in np.linspace(-0.3657694572837665, -1/3, num=70):
        C = (C12, C12, -1-2*C12)
        def f(alpha): return quantum_sieve3(alpha, C)
        if ALGO == KMPM19: alpha = n(pi/2)
        elif ALGO == CL23: alpha = find_local_minimum(f, pi/4, pi/2, tol=0.001)[1]
        t = quantum_sieve3(alpha, C)
        T.append(t)
        M.append(L(C))
        A.append(alpha)
    print("T =", T)
    print("M =", M)
    print("A =", A)
