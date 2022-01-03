import numpy as np
import rundec

# -------------------------------------------------------------------------------------------------------
# Module contains initial vectors, physical parameters and calculates Gamma0 and a_s at a given scale. 
# All parameters are in GeV and are taken from the PDG. 
# -------------------------------------------------------------------------------------------------------

# Initialize rundec
crd = rundec.CRunDec()

# Initial vectors for the full moments and as direct input for the q2-moments.
# Order of arguments is (mb, mc, mus, (muw), (mu0), api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi).
q2cuts = np.array([3.6, 4.1, 4.6, 5.099999999999999, 5.599999999999998, 6.099999999999998,
                   6.599999999999998, 7.099999999999997, 7.599999999999996, 8.099999999999996, 8.599999999999996])
kin_init = np.array([4.526, 1.13, 0.36156, -0.132, 0.019,
                     -0.8, -0.072, -0.006, 0.145, 0.432375])
ms_init = np.array([4.526, 1.099, 0.36156, -0.132, 0.019,
                    -0.8, -0.072, -0.006, 0.145, 0.432375])
kin_init_raw = np.array([4.526, 1.099, 4.526, 0.22 / np.pi, 0.36156,
                         -0.132, 0.019, -0.8, -0.072, -0.006, 0.145, 0.432375])
ms_init_raw = np.array([4.526, 1.099, 4.526, 2, 0.22 / np.pi, 0.36156,
                        -0.132, 0.019, -0.8, -0.072, -0.006, 0.145, 0.432375])
kin_init_cent = np.array([4.526, 1.099, 4.526, 1, 0.22 / np.pi, 0.36156,
                          -0.132, 0.019, -0.8, -0.072, -0.006, 0.145, 0.432375])
ms_init_cent = np.array([4.526, 1.099, 4.526, 1, 2, 0.22 / np.pi, 0.36156,
                         -0.132, 0.019, -0.8, -0.072, -0.006, 0.145, 0.432375])

# General parameters.
G_F = 1.1663787e-5
Vcb_pdg = 0.0422
A_ew = 1.014
asmZ = 0.1179

# Physical masses.
mZ = 91.1876
mbOS = 4.7
mcOS = 1.3
mbkin = 4.565
mckin = 1.130
mbMS = 4.198
mcMS_3 = 0.993
mcMS_2 = 1.099


def Gamma_0(mb=mbkin, Vcb=Vcb_pdg):
    res = A_ew * G_F ** 2 * Vcb ** 2 * mb ** 5 / (192 * np.pi ** 3)
    return res


# Returns as_4 with 4 loop accuracy for a given scale mu of mbkin.
def as_4(mu):
    as_res = (crd.AlphasExact(crd.DecAsDownSI(crd.AlphasExact(asmZ, mZ, 2 * mbMS, 5, 4),
                                              mbMS, 2 * mbMS, 4, 4), 2 * mbMS, mu, 4, 4))
    return as_res