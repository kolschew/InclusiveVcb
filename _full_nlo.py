"""
Description:
    Module contains the full NLO contributions to the moments.
    Since the NLO corrections from the scheme change are incorporated in the q2moment-files,
    the expressions were reexpanded in a_s analytically.

Content:
    nlo_1,..., nlo_4: NLO contributions for the non-centralized moments
    nlo_cent_1,..., nlo_cent_4: NLO contributions for the centralized moments

Args:
    q_cut (array): cuts for the lepton invariant mass q2
    mbkin (float): mass of the b-quark in the kinetic scheme
    mc (flaot): mass of the c-quark mass in either kinetic or MS scheme
"""

import numpy as np

import _nlo_moments as nlo
from moments import q2_moments_kinetic as q2m, q2_moments_raw as q2raw

ZZ = np.zeros(10)


# NLO corrections for the non-centralized moments from the reexpansion in a_s of (5.2) in 1812.07472
def nlo_1(q_cut, mbkin, mc):
    fnlo1 = (-2 / 3 * mbkin ** 2 / (q2raw.q2moment0(q_cut, mbkin, mc) ** 2) * 
             (nlo.q2_nlo_1(q_cut / mbkin ** 2, mc / mbkin) * q2raw.q2moment0(q_cut, mbkin, mc) - 
             nlo.q2_nlo_0(q_cut / mbkin ** 2, mc / mbkin) * q2raw.q2moment1(q_cut, mbkin, mc)))
    return fnlo1


def nlo_2(q_cut, mbkin, mc):
    fnlo2 = (-2 / 3 * mbkin ** 4 / (q2raw.q2moment0(q_cut, mbkin, mc) ** 2)
             * (nlo.q2_nlo_2(q_cut / mbkin ** 2, mc / mbkin) * q2raw.q2moment0(q_cut, mbkin, mc)
                - nlo.q2_nlo_0(q_cut / mbkin ** 2, mc / mbkin) * q2raw.q2moment2(q_cut, mbkin, mc)))
    return fnlo2


def nlo_3(q_cut, mbkin, mc):
    fnlo3 = (-2 / 3 * mbkin ** 6 / (q2raw.q2moment0(q_cut, mbkin, mc) ** 2)
             * (nlo.q2_nlo_3(q_cut / mbkin ** 2, mc / mbkin) * q2raw.q2moment0(q_cut, mbkin, mc)
                - nlo.q2_nlo_0(q_cut / mbkin ** 2, mc / mbkin) * q2raw.q2moment3(q_cut, mbkin, mc)))
    return fnlo3


def nlo_4(q_cut, mbkin, mc):
    fnlo4 = (-2 / 3 * mbkin ** 8 / (q2raw.q2moment0(q_cut, mbkin, mc) ** 2)
             * (nlo.q2_nlo_4(q_cut / mbkin ** 2, mc / mbkin) * q2raw.q2moment0(q_cut, mbkin, mc)
                - nlo.q2_nlo_0(q_cut / mbkin ** 2, mc / mbkin) * q2raw.q2moment4(q_cut, mbkin, mc)))
    return fnlo4


# NLO contribution for the centralized moments from reexpanding (5.6) in 1812.07472.
def nlo_cent_2(q_cut, mbkin, mc):
    cfnlo2 = (nlo_2(q_cut, mbkin, mc)
              - 2 * nlo_1(q_cut, mbkin, mc) * q2m.q2moment1Kin(q_cut, mbkin, mc, *ZZ))
    return cfnlo2


def nlo_cent_3(q_cut, mbkin, mc):
    cfnlo3 = (6 * q2m.q2moment1Kin(q_cut, mbkin, mc, *ZZ) ** 2 * nlo_1(q_cut, mbkin, mc)
              - 3 * nlo_1(q_cut, mbkin, mc) * q2m.q2moment2Kin(q_cut, mbkin, mc, *ZZ)
              - 3 * q2m.q2moment1Kin(q_cut, mbkin, mc, *ZZ) * nlo_2(q_cut, mbkin, mc)
              + nlo_3(q_cut, mbkin, mc))
    return cfnlo3


def nlo_cent_4(q_cut, mbkin, mc):
    cfnlo4 = (-12 * q2m.q2moment1Kin(q_cut, mbkin, mc, *ZZ) ** 3 * nlo_1(q_cut, mbkin, mc)
              + 6 * q2m.q2moment1Kin(q_cut, mbkin, mc, *ZZ) ** 2 * nlo_2(q_cut, mbkin, mc)
              - 4 * q2m.q2moment3Kin(q_cut, mbkin, mc, *ZZ) * nlo_1(q_cut, mbkin, mc)
              - 4 * q2m.q2moment1Kin(q_cut, mbkin, mc, *ZZ) * nlo_3(q_cut, mbkin, mc)
              + 12 * q2m.q2moment1Kin(q_cut, mbkin, mc, *ZZ) * q2m.q2moment2Kin(q_cut, mbkin, mc, *ZZ)
              * nlo_1(q_cut, mbkin, mc)
              + nlo_4(q_cut, mbkin, mc))
    return cfnlo4
