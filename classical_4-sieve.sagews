def V(alpha): return n( log(sin(alpha))/log(2) )
def NbSols(N,theta): return 2*N + V(theta);
def RatioFilter(Cxx): return V(acos(-Cxx));

def configy_x(alpha, Cij): return -cos(alpha)**2/2 + sin(alpha)**2 * Cij # from configuration Cij on the residual vectors yi, yj, returns the corresponding configuration on the list vectors xi, xj
def configx_y(alpha, xij): return (xij + cos(alpha)^2/2)/(sin(alpha)^2); # reverse

def MM(x12,x13,x23): # number of needed points given the configuration
    detC = 1 + 2*x12*x13*x23 -x12**2 -x13**2 -x23**2
    return n( -1/4*log(detC)/log(2) )


def Mval(C12,C13):
    C23 = -1 - C12 - C13;
    return MM(C12,C13,C23);

def ProbaReduce(theta,alpha): # Probability that 2 vectors theta-reduce knowing that they are at angle alpha from the same point.
    cosGamma = (cos(theta) - cos(alpha)^2)/sin(alpha)^2;
    return RatioFilter(cosGamma);


def InvV(N):
    if (N > 0): return "oops";
    if (N == 0): return pi/2;
    def partial(alpha): return V(alpha) - N;
    res = find_root(partial,0,pi/2);
    return res;

def FAS2n(N,C12):
    theta = arccos(-C12);
    alpha = InvV(-N);
    TotalSols = 2*N + V((theta));
    Cprime = (C12 + cos(alpha)^2)/(sin(alpha)^2);
    Sol1Found = V(arccos(-Cprime)) - V(alpha);
    NbReps = TotalSols - Sol1Found;
    if 0:
        print("alpha =", alpha);
        print("TotalSols =", TotalSols);
        print("Cprime =", Cprime);
        print("Sol1Found =", Sol1Found);
        print("NbReps =", NbReps);
    return NbReps + N;

def det2(C12): return n(log(1 - C12^2)/log(2));

#print("FAS2test =", FAS2test(0.2075,-1/2), FAS2test(0.18,-1/3));
#print("FAS2n =", FAS2n(0.2075,-1/2), FAS2n(0.18,-1/3));




#C = matrix([[1,-1/3,-1/3],[-1/3,1,-1/3],[-1/3,-1/3,1]]);
#D = det(C);
#print(n(log(D)/log(2))/2)

def matrix4(C12,C13,C14,C23,C24,C34):
    return matrix([[1,C12,C13,C14],[C12,1,C23,C24],[C13,C23,1,C34],[C14,C24,C34,1]]);

def Probareduce4(C12,C13,C14,C23,C24,C34):
    C = matrix4(n(C12),n(C13),n(C14),n(C23),n(C24),n(C34));
    D = det(C);
    return n(log(D)/log(2))/2;

C4 = matrix4(-1/4,-1/4,-1/4,-1/4,-1/4,-1/4);
P4 = Probareduce4(-1/4,-1/4,-1/4,-1/4,-1/4,-1/4);
M4 = -P4/(3);

#print(P4);
#print(M4);

def cprime4(C12,alpha):
    return (C12 + cos(alpha)^2/3)/(sin(alpha)^2);


def FAS4nBis(C12,alpha,b):
    C13 = (-(3/2) - 2*C12)/4;
    p = Probareduce4(C12,C13,C13,C13,C13,C12);
    N = -p/3;
    TotalSols = N;
    SmallListSize = N + V(alpha);
    Cprime12 = cprime4(C12,alpha);
    Cprime13 = cprime4(C13,alpha);
    pprime = Probareduce4(Cprime12,Cprime13,Cprime13,Cprime13,Cprime13,Cprime12);
    IntermediateListSize = 2*SmallListSize + V(arccos(-Cprime12));
    NbFilters = -V(alpha);
    SolutionsFound = 4*SmallListSize + pprime + NbFilters;
    NbRepetitions = TotalSols - SolutionsFound;
    Time12 = FAS2n(SmallListSize,Cprime12);
    R = 2*Cprime12 + 2;
    FinalC = ((1/sin(alpha)^2) - 2*R)/(2*R);
    Time1234 = FAS2n(IntermediateListSize,FinalC);
    Time = NbRepetitions + NbFilters + max(Time12,Time1234);
    if (b):
        print("InitialListSize =", N);
        print("NbRepetitions =", NbRepetitions);
        print("NbFilters =", NbFilters);
        print("Time12 =", Time12);
        print("SmallListSize =", SmallListSize);
        print("IntermediateListSize =", IntermediateListSize);
        print("SolutionsFound =", SolutionsFound);
        print("SolutionsFoundPerFilter =", SolutionsFound - NbFilters);
        print("Time1234 =", Time1234);
        print("Cprime12 =", Cprime12);
        print("Cprime13 =", Cprime13);
    return Time;

#FAS4nBis(-1/4,pi/2,1);

def partialM(a):
    return FAS4nBis(-0.427,a,1);

def FAS4nBisMemory(M):
    def partial(C12):
        C13 = (-(3/2) - 2*C12)/4;
        p = Probareduce4(C12,C13,C13,C13,C13,C12);
        return -p/3 - M;
    C12 = find_root(partial,-0.5,-0.25);
    C13 = (-(3/2) - 2*C12)/4;
    def partial2(alpha):
        SmallListSize = M + V(alpha);
        Cprime12 = cprime4(C12,alpha);
        Cprime13 = cprime4(C13,alpha);
        pprime = Probareduce4(Cprime12,Cprime13,Cprime13,Cprime13,Cprime13,Cprime12);
        IntermediateListSize = 2*SmallListSize + V(arccos(-Cprime12));
        return IntermediateListSize - M;
    alpha0 = find_root(partial2,pi/4,pi/2);
    # print("M,alpha0 =", M,alpha0);
    return FAS4nBis(C12,alpha0,0);



#print(find_local_minimum(T1,-0.4,-0.2))
#plot(FAS4nBisMemory,M4,0.2075)

#for i in range(0,11):
#    M = M4 + i*(0.2075 - M4)/10;
#    print(FAS4nBisMemory(M));


# Trade-off
import numpy
M = list(numpy.linspace(0.20752, M4, 200))
T = []
for m in M:
    T.append(FAS4nBisMemory(m))
print("M =", M)
print("T =", T)
