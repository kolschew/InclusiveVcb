import numpy as np

import theory_parameters as tpar
import total_rates as tr
import q2_moments_kinetic as q2mKin
import q2_moments_MS as q2mMS
import central_moments_kinetic as cmk
import central_moments_MS as cmMS
import full_nlo as fnlo


# Class contains the moments in the kinetic scheme with NLO-correction. 
# The value for mu_s and mu0 is internally fixed in init.

class InclusiveVcb:
    def __init__(self, mu0=2, mus=tpar.mbkin, muW=1):

        self.mus = mus
        self.mu0 = mu0
        self.muW = muW

        self.hbar = 6.58212e-25
        self.tauB = (1.519 + 1.638) * 0.5e-12
        self.api4 = tpar.as_4(self.mus)/np.pi

    # ---------------------------------------------------------------------------------
    # Functions for the total rate in kinetic and MS scheme.
    # ---------------------------------------------------------------------------------

    def total_rate_kin(self, Vcb, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        tot_kin = ( tpar.Gamma_0(mbkin, Vcb)
                   *tr.total_rate_kin(mbkin, mckin, self.mus, self.api4,
                                      muG, sB, rE, sqB, sE, rG, rhoD, mupi) )
        return tot_kin*self.tauB/self.hbar

    def total_rate_MS(self, Vcb, mbkin, mcMS, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        tot_MS = ( tpar.Gamma_0(mbkin, Vcb)
                  *tr.total_rate_MS(mbkin, mcMS, self.mus, self.mu0, self.api4,
                                    muG, sB, rE, sqB, sE, rG, rhoD, mupi) )
        return tot_MS*self.tauB/self.hbar

# ----------------------------------------------------------------------------------------------------
# Functions for the q2 moments in the kinetic scheme.
# ----------------------------------------------------------------------------------------------------

    def q2_moment_kin_1(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm1 = ( q2mKin.q2moment1Kin(q_cut, mbkin, mckin, self.mus, self.api4,
                                    muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.api4*fnlo.nlo_1(q_cut, mbkin, mckin) )
        return qm1

    def q2_moment_kin_2(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm2 = ( q2mKin.q2moment2Kin(q_cut, mbkin, mckin, self.mus, self.api4,
                                    muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.api4*fnlo.nlo_2(q_cut, mbkin, mckin) )
        return qm2

    def q2_moment_kin_3(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm3 = ( q2mKin.q2moment3Kin(q_cut, mbkin, mckin, self.mus, self.api4,
                                    muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.api4*fnlo.nlo_3(q_cut, mbkin, mckin) )
        return qm3

    def q2_moment_kin_4(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm4 = ( q2mKin.q2moment4Kin(q_cut, mbkin, mckin, self.mus, self.api4,
                                    muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.api4*fnlo.nlo_4(q_cut, mbkin, mckin) )
        return qm4

# ----------------------------------------------------------------------------------------------------
# Functions for the q2 moments in the MS scheme.
# ----------------------------------------------------------------------------------------------------

    def q2_moment_MS_1(self, q_cut, mbkin, mcMS, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm1 = ( q2mMS.q2moment1MS(q_cut, mbkin, mcMS, self.mus, self.mu0,
                                  self.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.api4*fnlo.nlo_1(q_cut, mbkin, mcMS) )
        return qm1

    def q2_moment_MS_2(self, q_cut, mbkin, mcMS, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm2 = ( q2mMS.q2moment2MS(q_cut, mbkin, mcMS, self.mus, self.mu0,
                                  self.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.api4*fnlo.nlo_2(q_cut, mbkin, mcMS) )
        return qm2

    def q2_moment_MS_3(self, q_cut, mbkin, mcMS, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm3 = ( q2mMS.q2moment3MS(q_cut, mbkin, mcMS, self.mus, self.mu0,
                                  self.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.api4*fnlo.nlo_3(q_cut, mbkin, mcMS) )
        return qm3

    def q2_moment_MS_4(self, q_cut, mbkin, mcMS, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm4 = ( q2mMS.q2moment4MS(q_cut, mbkin, mcMS, self.mus, self.mu0,
                                  self.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.api4*fnlo.nlo_4(q_cut, mbkin, mcMS) )
        return qm4

# ----------------------------------------------------------------------------------------------------
# Functions for the centralized moments in the kinetic scheme.
# ----------------------------------------------------------------------------------------------------

    def central_moment_kin_1(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm1 = self.q2_moment_kin_1(q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
        return cm1

    def central_moment_kin_2(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm2 = ( cmk.centmomKin2(q_cut, mbkin, mckin, self.mus, self.muW,
                               self.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
                + self.api4*(fnlo.nlo_cent_2(q_cut, mbkin, mckin)) )
        return cm2

    def central_moment_kin_3(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm3 = ( cmk.centmomKin3(q_cut, mbkin, mckin, self.mus, self.muW,
                               self.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
                + self.api4*fnlo.nlo_cent_3(q_cut, mbkin, mckin) )
        return cm3

    def central_moment_kin_4(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm4 = ( cmk.centmomKin4(q_cut, mbkin, mckin, self.mus, self.muW,
                               self.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
                + self.api4*fnlo.nlo_cent_4(q_cut, mbkin, mckin) )
        return cm4

# ----------------------------------------------------------------------------------------------------
# Functions for the centralized moments in the MS scheme.
# ----------------------------------------------------------------------------------------------------

    def central_moment_MS_1(self, q_cut, mbkin, mcMS, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm1 = self.q2_moment_MS_1(q_cut, mbkin, mcMS, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
        return cm1

    def central_moment_MS_2(self, q_cut, mbkin, mcMS, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm2 = ( cmMS.centmomMS2(q_cut, mbkin, mcMS, self.mus, self.muW, self.mu0,
                               self.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
                + self.api4*(fnlo.nlo_cent_2(q_cut, mbkin, mcMS)) )
        return cm2

    def central_moment_MS_3(self, q_cut, mbkin, mcMS, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm3 = ( cmMS.centmomMS3(q_cut, mbkin, mcMS, self.mus, self.muW, self.mu0,
                               self.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
                + self.api4*fnlo.nlo_cent_3(q_cut, mbkin, mcMS) )
        return cm3

    def central_moment_MS_4(self, q_cut, mbkin, mcMS, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm4 = ( cmMS.centmomMS4(q_cut, mbkin, mcMS, self.mus, self.muW, self.mu0,
                               self.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
                + self.api4*fnlo.nlo_cent_4(q_cut, mbkin, mcMS) )
        return cm4
