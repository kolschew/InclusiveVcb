import numpy as np

import nlo_moments_fixed_order_cut as nlo
import q2_moments_raw as q2raw #mathcal{Q}_n (4.9) in 1812.07472.
import q2_moments_kinetic as q2m # q_n (5.1) in 1812.07472.

# Module contains the full NLO contributions to the moments. It is only known for the non-normalized
# moments and hence the expressions had to be reexpanded.

# NLO corrections for the non-centralized moments from the reexpansion in a_s of (5.2) in 1812.07472

def nlo_1(q_cut, mbkin, mc):
    fnlo1 = ( -2/3*mbkin**2/(q2raw.q2moment0(q_cut, mbkin, mc)**2)
               *(nlo.q2_nlo_1(q_cut/mbkin**2, mc/mbkin)*q2raw.q2moment0(q_cut, mbkin, mc) 
               - nlo.q2_nlo_0(q_cut/mbkin**2, mc/mbkin)*q2raw.q2moment1(q_cut, mbkin, mc)) )
    return fnlo1

def nlo_2(q_cut, mbkin, mc):
    fnlo2 = ( -2/3*mbkin**4/(q2raw.q2moment0(q_cut, mbkin, mc)**2)
               *(nlo.q2_nlo_2(q_cut/mbkin**2, mc/mbkin)*q2raw.q2moment0(q_cut, mbkin, mc) 
               - nlo.q2_nlo_0(q_cut/mbkin**2, mc/mbkin)*q2raw.q2moment2(q_cut, mbkin, mc)) )
    return fnlo2

def nlo_3(q_cut, mbkin, mc):
    fnlo3 = ( -2/3*mbkin**6/(q2raw.q2moment0(q_cut, mbkin, mc)**2)
               *(nlo.q2_nlo_3(q_cut/mbkin**2, mc/mbkin)*q2raw.q2moment0(q_cut, mbkin, mc) 
               - nlo.q2_nlo_0(q_cut/mbkin**2, mc/mbkin)*q2raw.q2moment3(q_cut, mbkin, mc)) )
    return fnlo3

def nlo_4(q_cut, mbkin, mc):
    fnlo4 = ( -2/3*mbkin**8/(q2raw.q2moment0(q_cut, mbkin, mc)**2)
               *(nlo.q2_nlo_4(q_cut/mbkin**2, mc/mbkin)*q2raw.q2moment0(q_cut, mbkin, mc) 
               - nlo.q2_nlo_0(q_cut/mbkin**2, mc/mbkin)*q2raw.q2moment4(q_cut, mbkin, mc)) )
    return fnlo4

# NLO contribution for the centralized moments from reexpanding (5.6) in 1812.07472.
# Vector zz strips all HQE and api contributions from the kinetic moments in order to obtain raw moments.
zz = np.zeros(10)

def nlo_cent_2(q_cut, mbkin, mc):
    cfnlo2 = ( nlo_2(q_cut, mbkin, mc)
              - 2*nlo_1(q_cut, mbkin, mc)*q2m.q2moment1Kin(q_cut, mbkin, mc, *zz) )
    return cfnlo2

def nlo_cent_3(q_cut, mbkin, mc):
    cfnlo3 = ( 6*q2m.q2moment1Kin(q_cut, mbkin, mc, *zz)**2*nlo_1(q_cut, mbkin, mc) 
              - 3*nlo_1(q_cut, mbkin, mc)*q2m.q2moment2Kin(q_cut, mbkin, mc, *zz)
              - 3*q2m.q2moment1Kin(q_cut, mbkin, mc, *zz)*nlo_2(q_cut, mbkin, mc)
              + nlo_3(q_cut, mbkin, mc) )
    return cfnlo3

def nlo_cent_4(q_cut, mbkin, mc):
    cfnlo4 = ( -12*q2m.q2moment1Kin(q_cut, mbkin, mc, *zz)**3*nlo_1(q_cut, mbkin, mc)
              + 6*q2m.q2moment1Kin(q_cut, mbkin, mc, *zz)**2*nlo_2(q_cut, mbkin, mc)
              - 4*q2m.q2moment3Kin(q_cut, mbkin, mc, *zz)*nlo_1(q_cut, mbkin, mc)
              - 4*q2m.q2moment1Kin(q_cut, mbkin, mc, *zz)*nlo_3(q_cut, mbkin, mc)
              + 12*q2m.q2moment1Kin(q_cut, mbkin, mc, *zz)*q2m.q2moment2Kin(q_cut, mbkin, mc, *zz)
              *nlo_1(q_cut, mbkin, mc) 
              + nlo_4(q_cut, mbkin, mc) )
    return cfnlo4