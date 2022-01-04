from base import AbstractInclusiveVcb

from moments.total_rates import total_rate_kin
from moments.q2_moments_kinetic import *
from moments.central_moments_kinetic import *
from full_nlo import *


class InclusiveVcb(AbstractInclusiveVcb):

    def total_rate(self, Vcb, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        tot_kin = (self.data.tauB / self.data.hbar * self.data.gamma_0(mbkin, Vcb) *
                   total_rate_kin(mbkin, mckin, self.data.mus, self.data.api4,
                                  muG, sB, rE, sqB, sE, rG, rhoD, mupi))
        return tot_kin

    def q2_moment_1(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm1 = (q2moment1Kin(q_cut, mbkin, mckin, self.data.mus,
               self.data.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.data.api4 * nlo_1(q_cut, mbkin, mckin))
        return qm1

    def q2_moment_2(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm2 = (q2moment2Kin(q_cut, mbkin, mckin, self.data.mus,
               self.data.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.data.api4 * nlo_2(q_cut, mbkin, mckin))
        return qm2

    def q2_moment_3(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm3 = (q2moment3Kin(q_cut, mbkin, mckin, self.data.mus,
               self.data.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.data.api4 * nlo_3(q_cut, mbkin, mckin))
        return qm3

    def q2_moment_4(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        qm4 = (q2moment4Kin(q_cut, mbkin, mckin, self.data.mus,
               self.data.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.data.api4 * nlo_4(q_cut, mbkin, mckin))
        return qm4


class InclusiveVcbCentralized(AbstractInclusiveVcb):

    def total_rate(self, Vcb, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        tot_kin = (self.data.tauB / self.data.hbar * self.data.gamma_0(mbkin, Vcb) *
                   total_rate_kin(mbkin, mckin, self.data.mus, self.data.api4,
                                  muG, sB, rE, sqB, sE, rG, rhoD, mupi))
        return tot_kin

    def q2_moment_1(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm1 = (q2moment1Kin(q_cut, mbkin, mckin, self.data.mus,
               self.data.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.data.api4 * nlo_1(q_cut, mbkin, mckin))
        return cm1

    def q2_moment_2(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm2 = (centmomKin2(q_cut, mbkin, mckin, self.data.mus, self.data.muW,
                           self.data.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.data.api4 * (nlo_cent_2(q_cut, mbkin, mckin)))
        return cm2

    def q2_moment_3(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm3 = (centmomKin3(q_cut, mbkin, mckin, self.data.mus, self.data.muW,
                           self.data.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.data.api4 * nlo_cent_3(q_cut, mbkin, mckin))
        return cm3

    def q2_moment_4(self, q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        cm4 = (centmomKin4(q_cut, mbkin, mckin, self.data.mus, self.data.muW,
                           self.data.api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
               + self.data.api4 * nlo_cent_4(q_cut, mbkin, mckin))
        return cm4
