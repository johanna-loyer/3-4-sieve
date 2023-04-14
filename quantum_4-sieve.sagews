import numpy as np
k=4

# Volume of a spherical cap of angle alpha
def V(alpha): return n( log(sin(alpha))/log(2) )

def Exp(x): return n( log(x)/log(2))

# Volume of the spherical cap satisfying the partial configuration Cij
def RatioFilter(Cxx): return V(acos(-Cxx))

# Determinant of configuration C of size 3x3
def det3(C):
    C12, C13, C23 = C
    return Exp(1 + 2*C12*C13*C23 -C12**2 -C13**2 -C23**2)

# Determinant of configuration C of size 4x4
def det4(C):
    C12, C13, C14, C23, C24, C34 = C
    MatC = np.array([
                [1, C12, C13, C14],
                [C12, 1, C23, C24],
                [C13, C23, 1, C34],
                [C14, C24, C34, 1]], dtype=np.float64)
    detC = np.linalg.det(MatC)
    return Exp(detC)


#========== Configurations ==========
# C is the configuration of the lattice vectors x
# C_ is the configuration of the residial vectors y

# Return C'_ij given C_ij (See Lemma 1)
def xy_config(alpha, Cij): k = 4 ; return (Cij + cos(alpha)**2/(k-1))/sin(alpha)**2

# Return C' given C
def residual(alpha, C):
    C12, C13, C14, C23, C24, C34 = C
    C12_ = xy_config(alpha, C12)
    C13_ = xy_config(alpha, C13)
    C14_ = xy_config(alpha, C14)
    C23_ = xy_config(alpha, C23)
    C24_ = xy_config(alpha, C24)
    C34_ = xy_config(alpha, C34)
    return (C12_, C13_, C14_, C23_, C24_, C34_)

# Return C_ij given C'_ij
def yx_config(alpha, Cij_): k = 4 ; return -cos(alpha)**2/(k-1) + sin(alpha)**2 * Cij_

# Recover C from C'
def lattice(alpha, C_):
    C12_, C13_, C14_, C23_, C24_, C34_ = C_
    C12 = yx_config(alpha, C12_)
    C13 = yx_config(alpha, C13_)
    C14 = yx_config(alpha, C14_)
    C23 = yx_config(alpha, C23_)
    C24 = yx_config(alpha, C24_)
    C34 = yx_config(alpha, C34_)
    return (C12, C13, C14, C23, C24, C34)



#========== Complexity of the 4-sieve ==========
# Size of the input list L for the sieve
def L(C): k = 4 ; return -det4(C)/(2*(k-1))

# Size of the input lists for the subproblem
def SizeLi(alpha, C): return L(C) + V(alpha)

# Number of tuple-filters (See Equation 4)
def NbFilters(alpha): return -V(alpha)

# Probability that a triplet is a solution of the subproblem
def PrSol(alpha, C): C_ = residual(alpha, C) ; return det4(C_)/2

# Number of solutions-triplets in one tuple-filter
def NbSolPerFilter(alpha, C): k = 4 ; return SizeLi(alpha, C)*k + PrSol(alpha, C)

# Number of repeats of steps 1 and 2 (See Lemma 3)
def NbRep(alpha, C): return max(0, L(C) - (NbSolPerFilter(alpha, C) + NbFilters(alpha)) )

# Number of tuple-filters we need to look in for a solution (in the case there are less than one solution per tuple-filter)
def NbFiltersToLook(alpha, C): return max(0, -NbSolPerFilter(alpha, C) )

# Size of the intermediate lists L_ij (See Proposition 6)
def sizeLiy1y2(Li, C123, C12):
    """C123 configuration des (x1, ..., xi, xj) ; C12 configuration des (x1, ..., xi)
    Retourne |Lj(x1,...xi)|"""
    return max(0, Li + (det3(C123)- Exp(1-C12**2))/2 )

# Time to find all the 4-tuples of configuration C in a tuple-filter of angle alpha
def FASq4(alpha, C, pr=False):
    """Hybrid. See article klist p14"""
    C_ = residual(alpha, C)
    C12_, C13_, C14_, C23_, C24_, C34_ = C_
    Li = SizeLi(alpha, C)
    Sol = NbSolPerFilter(alpha, C)

    L2y1 = max(0, Li + RatioFilter(C12_) )
    C123_ = (C12_, C13_, C23_)
    C124_ = (C12_, C14_, C24_)
    L3y1y2 = sizeLiy1y2(Li, C123_, C12_) # max(0, Li + RatioFilter(Z23))
    L4y1y2 = sizeLiy1y2(Li, C124_, C12_) # max(0, Li + RatioFilter(Z24))
    groverL2 = max(0, (Li-L2y1)/2)
    groverL3 = max(0, (Li-L3y1y2)/2)
    groverL4 = max(0, (Li-L4y1y2)/2)

    AA = (Li-Sol)/2 + (L2y1/2 + L3y1y2/2 + L4y1y2/2)

    C = lattice(alpha, C_)
    ToLook = max(0, NbFiltersToLook(alpha, C))

    if pr:
        print("C_ =", C_)
        print("L2y1 =", L2y1)
        print("L3y1y2 =", L3y1y2)
        print("L4y1y2 =", L4y1y2)
        print("groverL2 =", groverL2)
        print("groverL3 =", groverL3)
        print("groverL4 =", groverL4)
        print("AA =", AA)
        print("Find 1 sol =", max(groverL2, groverL3, groverL4) + AA + ToLook/2)
        print("Sol =", Sol)
        print("FAS =", Sol + max(groverL2, groverL3, groverL4) + AA + ToLook/2)
    return Sol + max(groverL2, groverL3, groverL4) + AA + ToLook/2

# Time to solve SVP by the quantum 4-sieve with parameters angle alpha and configuration C
def quantum_sieve4(alpha, C):
    NbReps = NbRep(alpha, C)
    NbTupleFilters = -V(alpha)
    FAS = FASq4(alpha, C)
    return NbReps + max(L(C), NbTupleFilters + FAS)

# Time to solve SVP by the quantum 4-sieve with parameters angle alpha and balanced configuration
def quantum_sieve4_balanced(alpha):
    k = 4
    C = (-1/k, -1/k, -1/k, -1/k, -1/k, -1/k)
    return quantum_sieve4(alpha, C)




#========== Displaying ==========
def aff(alpha, C): # à mettre à jour pour le cas k=4
    C_ = residual(alpha, C)
    C12_, C13_, C14_, C23_, C24_, C34_ = C_
    print("\nC_ =", residual(alpha, C) )
    print("|L| =", L(C))
    print("PrSol =", PrSol(alpha, C) )
    print("NbSolPerFilter=", NbSolPerFilter(alpha, C))
    print("NbRep =", NbRep(alpha, C) )
    #print("QNbFiltersToLook =", NbFiltersToLook(alpha, C)/2)
    print("1/V(alpha) = NbFilters =", NbFilters(alpha))
    print("Li =", SizeLi(alpha, C) )

    print("FAS =", FASq4(alpha, C))
    print( )



#========== Optimization ==========
# Parameters:
# - alpha: angle of prefiltering
# - Configuration C =(C12, C13, C14, C23, C24, C34)
#       As our algorithm uses C13 and C14 symmetrically (resp. C23 and C24), we can set C13=C14 (and C23=C24).
#       C34 depends on C12 and C13 (See Sec 2.3): C34 = -3/2-C12-C13-C14-C23-C24 = -3/2-C12-2*C13-2*C23

# Return (optimized time over C23, optimal C23)
def opt_C23(alpha, C12, C13):
    def partial(C23):
        C = ( C12, C13, C13, C23, C23, -3/2-C12-2*C13-2*C23 )
        return quantum_sieve4(alpha, C)
    mini = max(-1, (-3/2 -C12 -2*C13)/2)
    return find_local_minimum(partial, mini, 0, tol=0.01) #t, C23=24

# Return (optimized time over C13, optimal C13)
def opt_C13(alpha, C12):
    def partial(C13): return opt_C23(alpha, C12, C13)[0]
    mini = max(-1, (-3/2 -C12)/2)
    return find_local_minimum(partial, mini, 0, tol=0.01) #t, C13=C14

# Return (optimized time over C12, optimal C12)
def opt_C12(alpha):
    def partial(C12): return opt_C13(alpha, C12)[0]
    return find_local_minimum(partial, max(-1, -3/2), 0, tol=0.01) #t, C12

# Return (optimized time over alpha, optimal alpha)
def opt_alpha():
    def partial(alpha): return opt_C12(alpha)[0]
    return find_local_minimum(partial, pi/4, pi/2, tol=0.01) #t, alpha





#=============== Results ===============#
PRINT = False # <--- Choose True to print all the intermediate values

if 0: # [KMPM19] BLS Alg 4.1
    alpha = n(pi/2)
    print("\n====== Balanced configuration & No prefiltering ======")
    C = (-1/k, -1/k, -1/k, -1/k, -1/k, -1/k)
    print("time =", quantum_sieve4(alpha, C))
    print("mem  =", L(C))

    print("\n====== Any configuration & No prefiltering ======")
    C12 = opt_C12(alpha)[1]
    C13 = opt_C13(alpha, C12)[1]
    t, C23 = opt_C23(alpha, C12, C13)
    C = (C12, C13, C13, C23, C23, -3/2-C12-2*C13-2*C23)
    print("time =", quantum_sieve4(alpha, C) , "([KMPM19] T=0.3289, M=0.1796)")
    print("mem  =", L(C))
    print("C =", C)

    #aff(alpha, C)

if 0: # Theorem 8
    print("\n==== Balanced configuration & Prefiltering (Memory-optimizing parameters) ====")
    time, alpha = find_local_minimum(quantum_sieve4_balanced, pi/4, pi/2, tol=0.001)
    print("time =", time)
    C = (-1/k, -1/k, -1/k, -1/k, -1/k, -1/k) # Cij = -0.24420997060531358
    print("mem  =", L(C))        # 0.172369285889652
    print("alpha =", alpha)      # 1.3130673568166376
    if PRINT: aff(alpha, C)
    t = FASq4(alpha, C, pr=PRINT) # 0.327592281608114

if 1: # Theorem 9
    print("\n==== Any configuration & Prefiltering (Time-optimizing parameters) ====")
    alpha = opt_alpha()[1]
    C12 = opt_C12(alpha)[1]
    C13 = opt_C13(alpha, C12)[1]
    C23 = opt_C23(alpha, C12, C13)[1]
    C = (C12, C13, C13, C23, C23, -3/2-C12-2*C13-2*C23)
    # C = (-0.3819660112501052, -0.22935056273510615, -0.22935056273510615, -0.2297432171041298, -0.2297432171041298, -0.19984642907142308)
    print("qtime =", quantum_sieve4(alpha, C)) # 0.311994703610206
    print("M =", L(C))                         # 0.181285118552484
    print("alpha =", alpha)                    # 1.2953622586776552
    print("C =", C)
    if PRINT: aff(alpha, C)
    t = FASq4(alpha, C, pr=PRINT)


#=============== Trade-offs ===============#

# Return True if C is a valid configuration, i.e. a tuple satisfying C is reducing.
def valid(C):
    C12, C13, C14, C23, C24, C34 = C
    if not(-1 < C12 < 0) or not(-1 < C13 < 0) or not(-1 < C14 < 0) or not(-1 < C23 < 0) or not(-1 < C24 < 0) or not(-1 < C34 < 0): return False
    return (4 + 2*C12 + 2*C13 + 2*C14 + 2*C23 + 2*C24 + 2*C34) <= 1
def index(M, val):
    for i in range(len(M)-1):
        if M[i] <= val <= M[i+1]: return i

if 0:
    print("\n===== Trade-off Quantum 4-sieve KMPM19 BLS Algo 4.1 =====")
    T = []
    M = []
    NUM = 27
    NUM2= 10
    T = [2]*NUM2
    M = np.linspace(0.1724, 0.180665415447899, NUM2)
    MM= np.linspace(0.1724, 0.180665415447899, NUM2)
    for C12 in np.linspace(-1,0, num=NUM):
        for C13 in np.linspace(max(-1, (-3/2 -C12)/2), 0, num=NUM):
            for C23 in np.linspace(max(-1, (-3/2 -C12 -2*C13)/2), 0, num=NUM):
                C = (C12, C13, C13, C23, C23, -3/2-C12-2*C13-2*C23)
                if valid(C):
                    t, m = quantum_sieve4(n(pi/2), C), L(C)
                    if m <= max(M):
                        i = index(M, m)
                        if t < T[i]:
                            MM[i] = m
                            T[i] = t
    print("T =", [0.341887262167812] + T[:-1] + [0.328888551908956])
    print("M =", [0.172369285889652] + list(MM)[:-1] + [0.180665415447900])

if 0:
    print("\n===== Trade-off Quantum 4-sieve [CL23] =====")
    T = []
    M = []
    NUM = 27 # Nombre de valeurs testés pour chaque Cij
    NUM2= 10 # Nombre de points
    T = [2]*NUM2
    M = np.linspace(0.172369285889652, 0.180665415447899, NUM2)
    MM= np.linspace(0.172369285889652, 0.180665415447899, NUM2)

    for alpha in [1.2953622586776552, 1.3, 1.3130673568166376]: #[1.3]
        for C12 in np.linspace(-1,0, num=NUM):
            for C13 in np.linspace(max(-1, (-3/2 -C12)/2), 0, num=NUM):
                for C23 in np.linspace(max(-1, (-3/2 -C12 -2*C13)/2), 0, num=NUM):
                    C = (C12, C13, C13, C23, C23, -3/2-C12-2*C13-2*C23)
                    m = L(C)
                    if m <= max(M):
                        t = quantum_sieve4(alpha, C)
                        i = index(M, m)
                        if t < T[i]:
                            MM[i] = m
                            T[i] = t
    # Affiche les listes des points calculés + les 2 extrémités connues
    print("T =", [0.327592281608114] + T[:-1]        + [0.311994703610206])
    print("M =", [0.172369285889652] + list(MM)[:-1] + [0.181285118552484])