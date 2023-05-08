import numpy as np
k=3

def Exp(x): return n( log(x)/log(2))
def L(x): return n(log(x)/log(2));
def V(alpha): return L(sin(alpha));
def InvV(N):
    if (N > 0): return n(pi/2)
    def partial(alpha): return V(alpha) - N;
    res = find_root(partial,0,pi/2);
    return n(res);
def EE(C12): return (1/2)*L(1 - C12^2);

def FAS3p(L1,C,pr=False):
    C12,C13,C23 = C;
    L2x1 = 2*L1 + EE(C12) - L1;
    L12 = L2x1 + L1;
    alpha = InvV(-L1)
    alphap = InvV(-L2x1);
    C12p = n((1/sin(alpha)^2)*(C12 + cos(alpha)^2));
    T12 = 2*L1 + EE(C12) - EE(C12p);
    Y23 = n((1/(1 - C12^2))*(C23 - C12^2));
    Y23p = (1/(sin(alphap)^2))*(Y23 + cos(alphap)^2);
    T123 = L1 + 2*L2x1 + EE(Y23) - EE(Y23p);
    if (pr):
        print("---3p---")
        print("L1 =", L1);
        print("C12,C23 =", C12,C23)
        print("L12 =", L12);
        print("L2x1 =", L2x1);
        print("alpha =", alpha);
        print("alphap =", alphap);
        print("C12p =", C12p);
        print("T12 =", T12);
        print("Y23 =", Y23);
        print("Y23p =", Y23p);
        print("T123 =", T123);
        print("---3p(END)---")
    return max(T12,T123);

def FAS3A(L1,C,alpha,pr=False):
    C12,C13,C23 = C;
    C12p = (1/sin(alpha)^2)*(C12 + (1/2)*cos(alpha)^2)
    C13p = (1/sin(alpha)^2)*(C13 + (1/2)*cos(alpha)^2)
    C23p = (1/sin(alpha)^2)*(C23 + (1/2)*cos(alpha)^2)
    NbRep = max(0,(1/2)*(det3(C) - det3((C12p,C13p,C23p))) - (k-1)*V(alpha));
    L1part = L1 + V(alpha);
    if (pr):
        print("L1 =", L1);
        print("alpha =", alpha);
        print("C12p =", C12p);
        print("C13p =", C13p);
        print("C23p =", C23p);
        print("NbRep =", NbRep);
        print("L1part =", L1part);
    res1 = FAS3p(L1part,(C12p,C13p,C23p));
    Lint = 2*L1part + EE(C12p);
    NbFilters = -V(alpha);
    if (pr):
        print("res1 =", res1);
        print("Lint =", Lint);
        print("NbFilters =", NbFilters);
    return NbFilters + NbRep + res1;

def FAS3AM(L1,C,alpha,MEM,pr=False):
    C12,C13,C23 = C;
    C12p = (1/sin(alpha)^2)*(C12 + (1/2)*cos(alpha)^2)
    C13p = (1/sin(alpha)^2)*(C13 + (1/2)*cos(alpha)^2)
    C23p = (1/sin(alpha)^2)*(C23 + (1/2)*cos(alpha)^2)
    NbRep = max(0,(1/2)*(det3(C) - det3((C12p,C13p,C23p))) - (k-1)*V(alpha));
    L1part = L1 + V(alpha);
    if (pr):
        print("L1 =", L1);
        print("alpha =", alpha);
        print("C12p =", C12p);
        print("C13p =", C13p);
        print("C23p =", C23p);
        print("NbRep =", NbRep);
        print("L1part =", L1part);
    res1 = FAS3p(L1part,(C12p,C13p,C23p));
    Lint = 2*L1part + EE(C12p);
    NbFilters = -V(alpha);
    if (pr):
        print("res1 =", res1);
        print("Lint =", Lint);
        print("NbFilters =", NbFilters);
    if (Lint > MEM): return 1 + MEM + alpha;
    if (L1 > MEM): return 1 + MEM + alpha;
    return NbFilters + NbRep + res1;

def FAS3(L1,C,pr=False):
    C12,C13,C23 = C;
    def partial(alpha): return FAS3A(L1,C,alpha);
    (res,alpha0) = find_local_minimum(partial,pi/3,pi/2);
    return res;

def FAS3C(C12,alpha,pr=False):
    C13 = C12;
    C23 = -1 - 2*C12;
    p = (1/2)*det3((C12,C13,C23))
    L1 = -p/2;
    return FAS3A(L1,(C12,C13,C23),alpha,pr);

def FAS3CM(C12,alpha,MEM,pr=False):
    C13 = C12;
    C23 = -1 - 2*C12;
    p = (1/2)*det3((C12,C13,C23))
    L1 = -p/2;
    return FAS3AM(L1,(C12,C13,C23),alpha,MEM,pr);


def mini4(pr = False):
    L1 = 0.1724;
    C12 = -1/4;
    C13 = -1/4;
    C23 = -1/4;
    L2x1 = L1 + EE(C12);
    Y23 = n((1/(1 - C12^2))*(C23 - C12^2));
    TimeList = L1;
    res1 = FAS3p(L2x1,(Y23,Y23,Y23),pr=True);
    if (pr):
        print("L1 =", L1);
        print("L2x1 =", L2x1);
        print("Y23 =", Y23);
        print("res1 =", res1);
        print("TimeList =", L1)
    return max(res1,TimeList) + L1;

C0 = (-1/3,-1/3,-1/3)
#print(FAS3A(0.1887,C0,1.2954,pr=True));
#print(FAS3C(-1/3,pi/2,pr=True))
def Min(alpha,MEM,pr=False):
    def partial(c): return FAS3CM(c,alpha,MEM,pr=False);
    (res,C12) = find_local_minimum(partial,-0.5,-0.25,tol=0.001);
    (resbis,C12bis) = find_local_minimum(partial,-0.4,-0.3,tol=0.001);
    (rester,C12) = find_local_minimum(partial,-0.36,-0.31,tol=0.001);
    #print("res,C12 =", res,C12);
    res2 = FAS3CM(C12,alpha,MEM,pr);
    return min(res,resbis,rester);

#print(Min(1.2954,0.19))

def MinVal(MEM):
    #print("MEM =", MEM);
    def partial(alpha): return Min(alpha,MEM);
    (res,value) = find_local_minimum(partial,1.2,pi/2,tol=0.001);
    #print("res,alpha =", res,value);
    return res;

def FAS3ww(L1,C,pr=False):
    C12,C13,C23 = C;
    def partial(alpha): return FAS3A(L1,C,alpha);
    (res,alpha0) = find_local_minimum(partial,1.3,pi/2);
    if (pr):
        print("res,alpha0 =", res, alpha0);
    return res;

def FAS4p(L1,C,pr=False):
    C12,C13,C14,C23,C24,C34 = C;
    L2x1 = 2*L1 + EE(C12) - L1;
    L12 = L2x1 + L1;
    alpha = InvV(-L1)
    alphap = InvV(-L2x1);
    C12p = n((1/sin(alpha)^2)*(C12 + cos(alpha)^2));
    T12 = 2*L1 + EE(C12) - EE(C12p);
    Y23 = n((1/(1 - C12^2))*(C23 - C12^2));
    Y23p = (1/(sin(alphap)^2))*(Y23 + cos(alphap)^2);
    Y24 = n((1/(1 - C12^2))*(C24 - C12^2));
    Y24p = (1/(sin(alphap)^2))*(Y24 + cos(alphap)^2);
    Y34 = n((1/(1 - C12^2))*(C34 - C12^2));
    Y34p = (1/(sin(alphap)^2))*(Y34 + cos(alphap)^2);
    T123x1 = FAS3ww(L2x1,(Y23,Y24,Y34));
    T123 = L1 + T123x1;
    if (pr):
        print("L1 =", L1);
        print("L12 =", L12);
        print("L2x1 =", L2x1);
        print("alpha =", alpha);
        print("alphap =", alphap);
        print("C12p =", C12p);
        print("T12 =", T12);
        print("Y23 =", Y23);
        print("Y23p =", Y23p);
        print("T123 =", T123);
    return max(T12,T123);

def FAS4bis(L1,C,pr=False):
    C12,C13,C14,C23,C24,C34 = C;
    L2x1 = 2*L1 + EE(C12) - L1;
    L12 = L2x1 + L1;
    alpha = InvV(-L1)
    alphap = InvV(-L2x1);
    C12p = n((1/sin(alpha)^2)*(C12 + cos(alpha)^2));
    T12 = L1;
    Y23 = n((1/(1 - C12^2))*(C23 - C12^2));
    Y23p = (1/(sin(alphap)^2))*(Y23 + cos(alphap)^2);
    Y24 = n((1/(1 - C12^2))*(C24 - C12^2));
    Y24p = (1/(sin(alphap)^2))*(Y24 + cos(alphap)^2);
    Y34 = n((1/(1 - C12^2))*(C34 - C12^2));
    Y34p = (1/(sin(alphap)^2))*(Y34 + cos(alphap)^2);
    T123x1 = FAS3p(L2x1,(Y23,Y24,Y34),pr);
    if (pr):
        print("L1 =", L1);
        print("L12 =", L12);
        print("L2x1 =", L2x1);
        print("alpha =", alpha);
        print("alphap =", alphap);
        print("C12p =", C12p);
        print("T12 =", T12);
        print("Y23 =", Y23);
        print("Y23p =", Y23p);
        print("T123x1 =", T123x1);
    return L1 + max(T12,T123x1);
#plot(MinVal,0.1889,0.28,plot_points=5)

def det3(C):
    C12, C13, C23 = C
    MatC = np.array([
                [1, C12, C13],
                [C12, 1, C23],
                [C13, C23, 1]], dtype=np.float64)
    detC = np.linalg.det(MatC)
    return Exp(detC)

def det4(C):
    C12, C13, C14, C23, C24, C34 = C
    MatC = np.array([
                [1, C12, C13, C14],
                [C12, 1, C23, C24],
                [C13, C23, 1, C34],
                [C14, C24, C34, 1]], dtype=np.float64)
    detC = np.linalg.det(MatC)
    return Exp(detC)

def FAS4A(L1,C,alpha,pr=False):
    k = 4;
    C12,C13,C14,C23,C24,C34 = C;
    C12p = (1/sin(alpha)^2)*(C12 + (1/3)*cos(alpha)^2)
    C13p = (1/sin(alpha)^2)*(C13 + (1/3)*cos(alpha)^2)
    C23p = (1/sin(alpha)^2)*(C23 + (1/3)*cos(alpha)^2)
    C14p = (1/sin(alpha)^2)*(C14 + (1/3)*cos(alpha)^2)
    C24p = (1/sin(alpha)^2)*(C24 + (1/3)*cos(alpha)^2)
    C34p = (1/sin(alpha)^2)*(C34 + (1/3)*cos(alpha)^2)
    Cp = (C12p,C13p,C14p,C23p,C24p,C34p);
    NbRep = max(0,(1/2)*(det4(C) - det4(Cp)) - (k-1)*V(alpha));
    L1part = L1 + V(alpha);
    if (pr):
        print("L1 =", L1);
        print("alpha =", alpha);
        print("C12p =", C12p);
        print("C13p =", C13p);
        print("C23p =", C23p);
        print("NbRep =", NbRep);
        print("L1part =", L1part);
    res1 = FAS4bis(L1part,Cp,pr);
    Lint = 2*L1part + EE(C12p);
    NbFilters = -V(alpha);
    FINAL = NbFilters + NbRep + res1;
    if (pr):
        print("res1 =", res1);
        print("Lint =", Lint);
        print("NbFilters =", NbFilters);
        print("FINAL =", FINAL);
    return FINAL;

C0 = (-1/4,-1/4,-1/4,-1/4,-1/4,-1/4);
def Conf4(c1,c2,c3):
    return (c1,c1,c1,c2,c2,c3);
def Size4(C):
    k = 4;
    p = (1/2)*det4(C);
    return -p/(k-1)

def FAS4Full(C):
    L1 = Size4(C);
    print("L1 =", L1);
    def partial(a): return FAS4A(L1,C,a);
    (res,value) = find_local_minimum(partial,1.2,n(pi/2),tol=0.01);
    print("(res,alpha) =", (res,alpha));
    return res;


#print(FAS4bis(0.1724,C0,pr=True));
#print(FAS4A(0.1724,C0,1.4,pr=True));



def partial(a):
    print("a =", a);
    return FAS4A(0.1724,C0,a);

a0 = 1.2546638931515364;
#FAS4A(0.1724,C0,a0,pr=True);

#find_local_minimum(partial,1.36,pi/2)
M0 = 0.1890
M1 = 0.275
rr = 8
for i in range(0,rr+1):
    M = M0 + i*(M1 - M0)/rr;
#    MinVal(M);
C03 = (-1/3,-1/3,-1/3);
#print(FAS3A(0.1887,C03,1.2954));
#print(FAS4Full(C0))


# Optimizes the configuration
def Opt1(c1):
    def partial(c2):
        c3 = -(3/2) - 3*c1 - 2*c2;
        C = Conf4(c1,c2,c3);
        return FAS4Full(C);
    print("c1 =", c1);
    (res,value) = find_local_minimum(partial, -0.4,-0.1,tol=0.01);
    print("res,c2 =", res,value);
    return res;

#print(find_local_minimum(Opt1,-0.4,-0.2,tol=0.01))




# Trade-off classical 3-sieve
 M = list(np.linspace(0.18877, 0.285, 10))
 T = []
 for m in M: 
        T.append(MinVal(m))
print("M =", M)
print("T =", T)