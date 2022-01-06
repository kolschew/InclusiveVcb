import numpy as np
from typing import Any, Union

import inclusive_Vcb_MS as incms
import inclusive_Vcb_kinetic as inckin


class CovarianceMatrix:

    def __init__(self, q_cuts: np.array, hqe_pars: list, shifts: dict,
                 scheme: str, cent: bool, multi=1, decorr=None):
        """Calculates the covariance and correlation matrix for given shifts in the parameters
        of inclusive Vcb.

        Attributes:
            q_cuts      Experimental cuts on q2
            hqe_pars    Default values for the HQE parameters
            shifts      Procentual shift in given parameters (exept 'mus' must be absolute shift)
            scheme      Mass scheme, either 'kin' or 'MS'
            cent        Flag for centralized moments
            multi       Extends the covariance matrix to use more than one dataset for the moments
            decorr      Flag for the decorrelation scheme
        """

        self.V_cb = 0.042
        self.scheme = scheme
        self.cent = cent
        self.shifts = shifts
        self.cuts = q_cuts
        self.hqe_pars = hqe_pars
        self.multi = multi
        self.decorr = decorr

        # Initialize the q2 moments - if mus is provided also initiate a class with different mus.
        self._tp = self._instantiate_moments(self.scheme, self.cent)

        self._func_list, self._funcs = self._build_moments(self._tp())
        if 'mus' in self.shifts:
            _, self._funcs_shift = self._build_moments(self._tp(mus=self.shifts['mus']))

        # Dimension for the covariance matrix
        self._dim = len(self.cuts) * len(self._func_list) + 1
        self._full_dim = len(self.cuts) * len(self._func_list) * self.multi + 1

        # Initialize the scheme for the decorrelation.
        self._decorr_matrix = self._set_decorr_matrix(self.decorr)

    def covariance(self, par: str):
        """Builds the covariance matrix of given parameter

        The generated covariance matrix rows and columns are
        [tot_rate, q1(c1),...,q1(cn),...,q4(c1),...,q4(cn)]
        for given cuts ci and moments qi.
        Parameter multi multiplies all moments to allow for
        additional datasets to be incorporated.
        """

        cut = np.concatenate((np.array([1]), np.tile(self.cuts, len(self._func_list))))
        cov = np.zeros((self._full_dim, self._full_dim))
        block = np.zeros((self._dim, self._dim))
        hqe_shift = self.hqe_pars * self._build_shift(par)

        for ii in range(len(cut)):
            for jj in range(ii, len(self._funcs)):
                if (ii == 0 or jj == 0) and ii != jj:
                    block[ii][jj] = 0
                else:
                    if par == 'mus':
                        block[ii][jj] = ((self._funcs[ii](np.array([cut[ii]]), *self.hqe_pars) -
                                          self._funcs_shift[ii](np.array([cut[ii]]), *self.hqe_pars)) *
                                         (self._funcs[jj](np.array([cut[jj]]), *self.hqe_pars) -
                                          self._funcs_shift[jj](np.array([cut[jj]]), *self.hqe_pars)))
                        block[jj][ii] = block[ii][jj]

                    elif par in self.shifts:
                        block[ii][jj] = ((self._funcs[ii](np.array([cut[ii]]), *self.hqe_pars) -
                                          self._funcs[ii](np.array([cut[ii]]), *hqe_shift)) *
                                         (self._funcs[jj](np.array([cut[jj]]), *self.hqe_pars) -
                                          self._funcs[jj](np.array([cut[jj]]), *hqe_shift)))
                        block[jj][ii] = block[ii][jj]

        cov[0, 0] = block[0, 0]
        cov[1:, 1:] = np.block([[block[1:, 1:]] * self.multi] * self.multi)
        return cov * self._decorr_matrix

    def summed_covariance(self):
        """Sums up the covariance matrices for all shifts"""

        summed_mat = np.zeros((self._full_dim, self._full_dim))
        for par in self.shifts:
            summed_mat += self.covariance(par)
        return summed_mat

    def _rate_wrapper(self, rate: callable):
        """Wrapper for the call of the total rate"""

        def wrapper(q_cut, mbkin, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
            res = rate(self.V_cb, mbkin, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
            return res
        return wrapper

    @staticmethod
    def _build_correlation_matrix(cov_mat: np.array):
        """Generates a correlation matrix from a given covariance matrix"""

        inv = np.diag(1 / np.sqrt(np.diag(cov_mat)))
        return inv @ cov_mat @ inv

    @staticmethod
    def _instantiate_moments(scheme: str, cent: bool):
        """Initializer for mass scheme and centralized moments"""

        if scheme == 'kin':
            if cent:
                return inckin.InclusiveVcbCentralized
            else:
                return inckin.InclusiveVcb
        elif scheme == 'MS':
            if cent:
                return incms.InclusiveVcbCentralized
            else:
                return incms.InclusiveVcb
        else:
            raise KeyError('Please choose a valid mass scheme.')

    def _build_shift(self, parshift: str):
        """Translates the input dict to a vector with a shift for the respective parameter."""

        if parshift not in self.shifts:
            raise KeyError(f'No input value provided for the variation of {parshift}.')

        if parshift == 'mb':
            shift = np.array([1 + self.shifts[parshift], 1, 1, 1, 1, 1, 1, 1, 1, 1])
        elif parshift == 'mc':
            shift = np.array([1, 1 + self.shifts[parshift], 1, 1, 1, 1, 1, 1, 1, 1])
        if parshift == 'muG':
            shift = np.array([1, 1, 1 + self.shifts[parshift], 1, 1, 1, 1, 1, 1, 1])
        elif parshift == 'sB':
            shift = np.array([1, 1, 1, 1 + self.shifts[parshift], 1, 1, 1, 1, 1, 1])
        elif parshift == 'rE':
            shift = np.array([1, 1, 1, 1, 1 + self.shifts[parshift], 1, 1, 1, 1, 1])
        elif parshift == 'sqB':
            shift = np.array([1, 1, 1, 1, 1, 1 + self.shifts[parshift], 1, 1, 1, 1])
        elif parshift == 'sE':
            shift = np.array([1, 1, 1, 1, 1, 1, 1 + self.shifts[parshift], 1, 1, 1])
        elif parshift == 'rG':
            shift = np.array([1, 1, 1, 1, 1, 1, 1, 1 + self.shifts[parshift], 1, 1])
        elif parshift == 'rhoD':
            shift = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1 + self.shifts[parshift], 1])
        elif parshift == 'mupi':
            shift = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1 + self.shifts[parshift]])
        else:
            raise KeyError('This parameter is not included in the covariance matrix.')
        return np.array(shift)

    def _build_moments(self, obj: Any):
        """Builds list of functions for the q2moments"""

        func_list = [obj.q2_moment_1, obj.q2_moment_2,
                     obj.q2_moment_3, obj.q2_moment_4]
        funcs = ([self._rate_wrapper(obj.total_rate)] +
                 list(np.repeat(func_list, len(self.cuts))))
        return func_list, funcs

    def _set_decorr_matrix(self, decorr: Union[None, str, list]):
        """Sets the decorrelation matrix"""

        if decorr is None:
            return np.ones((self._full_dim, self._full_dim))
        elif decorr == 'Block':
            return self._build_decorr_matrix_block()
        elif type(decorr) == list:
            return self._build_decorr_matrix_dec()
        else:
            raise NameError('The provided input does not match any decorrelation scheme.')

    def _build_decorr_matrix_dec(self):
        """Builds decorrelation matrix with decreasing correlation
         for growing distance between observables.
         """

        decorr_block = np.zeros((self._dim - 1, self._dim - 1))
        decorr_mat = np.eye(self._full_dim)

        for ii in range(len(self._funcs) - 1):
            for jj in range(len(self._funcs) - 1):
                decorr_block[ii][jj] = (self.decorr[0] **
                                        (np.abs(ii - jj) % len(self._func_list)) *
                                        self.decorr[1] **
                                        (np.abs(ii - jj) // len(self._func_list)))

        decorr_mat[1:, 1:] = np.block([[decorr_block] * self.multi] * self.multi)
        return decorr_mat

    def _build_decorr_matrix_block(self):
        """Builds a decorrelation matrix with block-diagonal form
         to fully decorrelate different q2moments
         """

        decorr_block = np.zeros((self._dim - 1, self._dim - 1))
        decorr_mat = np.eye(self._full_dim)

        for ii in range(len(self._funcs) - 1):
            for jj in range(len(self._funcs) - 1):
                if self._funcs[ii + 1] == self._funcs[jj + 1]:
                    decorr_block[ii][jj] = 1
                else:
                    decorr_block[ii][jj] = 0

        decorr_mat[1:, 1:] = np.block([[decorr_block] * self.multi] * self.multi)
        return decorr_mat


# cuts = np.array([3, 4])
# kin_start = np.array([4.565, 1.13, 0.36156, -0.132, 0.019, -0.8, -0.072, -0.006, 0.145, 0.432375])
# MS_start = np.array([4.565, 1.093, 0.36156, -0.132, 0.019, -0.8, -0.072, -0.006, 0.145, 0.432375])
# shift = {'rhoD': 0.3, 'muG': 0.2, 'mus': 4.565/2}
