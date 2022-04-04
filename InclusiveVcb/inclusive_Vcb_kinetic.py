from InclusiveVcb._base import AbstractInclusiveVcb

from InclusiveVcb.moments.total_rates import total_rate_kin
from InclusiveVcb.moments.q2_moments_kinetic import *
from InclusiveVcb.moments.central_moments_kinetic import *
from InclusiveVcb._full_nlo import *

import InclusiveVcb.covariance_matrix as cm


class NormalizedMomentsKin(AbstractInclusiveVcb):
    """Total Rate and first four q2-moments in the kinetic scheme.

    Attributes
    ----------
    mus : float
        Scale for the strong coupling a_s (default is 4.546)
    mckin : float
        Mass of the charm quark in the kinetic scheme (default is 1.130)
    mbkin : float
        Mass of the bottom quark in the kinetic scheme (default is 4.546)

    Methods
    -------
    total_rate : (Vcb, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
        Total rate up to a_s^3 and 1/mb^4
    q2_moment_i : (q_cut, mbkin, mckin, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
        The i-th q2-moment up to a_s and 1/mb^4. First argument must be provided as np.array.
    covariance_matrix : (cuts, shifts, multi=1, decorr=None)
        Covariance matrix of the rate and moments for shifts in the given parameters.
    """

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

    def covariance_matrix(self, cuts, shifts, multi=1, decorr=None):
        mat = cm.CovarianceMatrix(cuts, self.data.default, shifts,
                                  scheme='kin', cent=False,
                                  multi=multi, decorr=decorr
                                  ).summed_covariance()
        return mat


class CentralizedMomentsKin(AbstractInclusiveVcb):
    """Total Rate and first four centralized q2-moments in the kinetic scheme"""

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

    def covariance_matrix(self, cuts, shifts, multi=1, decorr=None):
        mat = cm.CovarianceMatrix(cuts, self.data.default, shifts,
                                  scheme='kin', cent=True,
                                  multi=multi, decorr=decorr
                                  ).summed_covariance()
        return mat
