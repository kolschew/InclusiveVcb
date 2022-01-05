import numpy as np
from typing import Any, Union

import inclusive_Vcb_MS as incms
import inclusive_Vcb_kinetic as inckin

cuts = np.array([3, 4])
kin_start = np.array([4.565, 1.13, 0.36156, -0.132, 0.019, -0.8, -0.072, -0.006, 0.145, 0.432375])
MS_start = np.array([4.565, 1.093, 0.36156, -0.132, 0.019, -0.8, -0.072, -0.006, 0.145, 0.432375])
shift = {'rhoD': 0.3, 'muG': 0.2, 'mus': 4.565/2}

# ----------------------------------------------------------------------------------------------------------------------
#  Class calculates the covariance matrix for a given set of q_cuts, hqe-pars and shifts (provided as a dictionary).
# One can choose between the kinetic and MS scheme and have the covariance matrices summed up for all shifts.
# The generated covariance matrix will have tot_rate, q1(c1),...,q1(cn),...,q4(c1),...,q4(cn) for given cuts ci
# and moments qi in its rows and columns. If multi is provided, the part with the moments in the matrix is multiplied.
# If decorr is provided as a vector with two entries, decorrelation is performed with decreasing correlation
# decor1^Mod(i - j)*decor2^Quot(i - j). If decor is provided as 'Block' decorrelation is done with a block matrix.
# ----------------------------------------------------------------------------------------------------------------------


class CovarianceMatrix:

    def __init__(self, q_cuts: np.array, hqe_pars: list, shifts: dict,
                 scheme: str, cent: bool, multi=1, decorr=None):

        # Store outside parameters for later use
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

        # Dimension for the covariance matrix depending on the input of the functions and cuts.
        self._dim = len(self.cuts) * len(self._func_list) + 1
        self._full_dim = len(self.cuts) * len(self._func_list) * self.multi + 1

        # Initialize the scheme for the decorrelation.
        self._decorr_matrix = self._set_decorr_matrix(self.decorr)

    # Wrapper for the call of the total rate, so all functions can be called with the same input.
    def _rate_wrapper(self, rate: callable):
        def wrapper(q_cut, mbkin, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
            res = rate(self.V_cb, mbkin, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
            return res
        return wrapper

    @staticmethod
    def _instantiate_moments(scheme: str, cent: bool):
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

    # Translates the input dic to a vector with a shift for the respective parameter.
    def _build_shift(self, parshift: str):
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

    # Builds the list for the functions with the provided choice for the scheme.
    def _build_moments(self, obj: Any):
        func_list = [obj.q2_moment_1, obj.q2_moment_2,
                     obj.q2_moment_3, obj.q2_moment_4]
        funcs = ([self._rate_wrapper(obj.total_rate)] +
                 list(np.repeat(func_list, len(self.cuts))))
        return func_list, funcs

    def _set_decorr_matrix(self, decorr: Union[None, str, list]):
        if decorr is None:
            return np.ones((self._full_dim, self._full_dim))
        elif decorr == 'Block':
            return self._build_decorr_matrix_block()
        elif type(decorr) == list:
            return self._build_decorr_matrix_dec()
        else:
            raise NameError('The provided input does not match any decorrelation scheme.')

    # Build a decorrelation matrix with decreasing correlation for growing distance between observables.
    def _build_decorr_matrix_dec(self):
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

    # Build a decorrelation matrix with block-diagonal form to decorrelate different q2moments fully.
    def _build_decorr_matrix_block(self):
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

    # Generates a correlation matrix from a given covariance matrix.
    @staticmethod
    def _build_correlation_matrix(cov_mat: np.array):
        inv = np.diag(1 / np.sqrt(np.diag(cov_mat)))
        return inv @ cov_mat @ inv

    # Builds the covariance matrix by recursively iterating through all functions and cuts.
    def covariance(self, par: str):
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

    # Sums up the covariance matrices for all the shifts provided.
    def summed_covariance(self):
        summed_mat = np.zeros((self._full_dim, self._full_dim))
        for par in self.shifts:
            summed_mat += self.covariance(par)
        return summed_mat
