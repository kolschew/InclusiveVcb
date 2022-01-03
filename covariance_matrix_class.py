import numpy as np

from full_moments import TheoryPrediction

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

    def __init__(self, q_cuts, hqe_pars, shifts, scheme, multi=1, decorr=None):

        # Store outside parameters for later use
        self.V_cb = 0.042
        self.scheme = scheme
        self.shifts = shifts
        self.cuts = q_cuts
        self.hqe_pars = hqe_pars
        self.multi = multi
        self.decorr = decorr

        # Initialize the q2 moments - if mus is provided also initiate a class with different mus.
        self.tp = TheoryPrediction()
        if 'mus' in self.shifts:
            self.tps = TheoryPrediction(mus=self.shifts['mus'])
            self.BuildMusShift()

        # Initialize the func list for the matrices and the scheme for the decorrelation.
        self.BuildFuncLists()

        # Dimension for the covariance matrix depending on the input of the functions and cuts.
        self.dim = len(self.cuts)*len(self.func_list) + 1
        self.full_dim = len(self.cuts)*len(self.func_list)*self.multi + 1

        # Initialize the scheme for the decorrelation.
        self.SetDecorrMatrix()

    # Wrapper for the call of the total rate, so all functions can be called with the same input.
    def RateWrapper(self, rate):
        def wrapper(q_cut, mbkin, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
            res = rate(self.V_cb, mbkin, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi)
            return res
        return wrapper

    # Translates the input dic to a vector with a shift for the respective parameter.
    def BuildShift(self, parshift):
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

    # Builds the shift for the variation in mus with the provided choice for the scheme.
    def BuildMusShift(self):
        if self.scheme == 'kin':
            self.funcs_shift = ( [self.RateWrapper(self.tps.total_rate_kin)]
                + np.repeat([self.tps.q2_moment_kin_1, self.tps.q2_moment_kin_2,
                   self.tps.q2_moment_kin_3, self.tps.q2_moment_kin_4], len(self.cuts)).tolist()*self.multi )
        elif self.scheme == 'MS':
            self.funcs_shift = ( [self.RateWrapper(self.tps.total_rate_MS)]
                + np.repeat([self.tps.q2_moment_MS_1, self.tps.q2_moment_MS_2,
                   self.tps.q2_moment_MS_3, self.tps.q2_moment_MS_4], len(self.cuts)).tolist()*self.multi )
        return

    # Builds the list for the functions with the provided choice for the scheme.
    def BuildFuncLists(self):
        if self.scheme == 'kin':
            self.func_list = [self.tp.q2_moment_kin_1, self.tp.q2_moment_kin_2,
                              self.tp.q2_moment_kin_3, self.tp.q2_moment_kin_4]
            self.funcs = ( [self.RateWrapper(self.tp.total_rate_kin)]
                          + np.repeat(self.func_list, len(self.cuts)).tolist() )
        elif self.scheme == 'MS':
            self.func_list = [self.tp.q2_moment_MS_1, self.tp.q2_moment_MS_2,
                              self.tp.q2_moment_MS_3, self.tp.q2_moment_MS_4]
            self.funcs = ( [self.RateWrapper(self.tp.total_rate_MS)]
                          + np.repeat(self.func_list, len(self.cuts)).tolist() )
        else:
            raise NameError('Please choose a valid mass scheme.')
        return

    def SetDecorrMatrix(self):
        if self.decorr == None:
            self.decorr_mat = np.ones((self.full_dim, self.full_dim))
        elif self.decorr == 'Block':
            self.BuildDecorrMatrixBlock()
        elif type(self.decorr) == list:
            self.BuildDecorrMatrixDec()
        else:
            raise NameError('The provided input does not match any decorrelation scheme.')

    # Build a decorrelation matrix with decreasing correlation for growing distance between observables.
    def BuildDecorrMatrixDec(self):
        decorr_block = np.zeros((self.dim - 1, self.dim - 1))
        self.decorr_mat = np.eye(self.full_dim)

        for ii in range(len(self.funcs) - 1):
            for jj in range(len(self.funcs) - 1):
                decorr_block[ii][jj] = ( self.decorr[0]**(np.abs(ii - jj) % len(self.func_list))
                                        *self.decorr[1]**(np.abs(ii - jj) // len(self.func_list)) )

        self.decorr_mat[1:, 1:] = np.block([[decorr_block]*self.multi]*self.multi)
        return self.decorr_mat

    # Build a decorrelation matrix with block-diagonal form to decorrelate different q2moments fully.
    def BuildDecorrMatrixBlock(self):
        decorr_block = np.zeros((self.dim - 1, self.dim - 1))
        self.decorr_mat = np.eye(self.full_dim)

        for ii in range(len(self.funcs) - 1):
            for jj in range(len(self.funcs) - 1):
                if self.funcs[ii + 1] == self.funcs[jj + 1]:
                    decorr_block[ii][jj] = 1
                else:
                    decorr_block[ii][jj] = 0

        self.decorr_mat[1:, 1:] = np.block([[decorr_block]*self.multi]*self.multi)
        return

    # Generates a correlation matrix from a given covariance matrix.
    @staticmethod
    def BuildCorrelationMatrix(cov_mat):
        inv = np.diag(1 / np.sqrt(np.diag(cov_mat)))
        return inv @ cov_mat @ inv

    # Builds the covariance matrix by recursively iterating through all functions and cuts.
    def BuildCovarianceMatrix(self, par):
        cut = np.concatenate((np.array([1]), np.tile(self.cuts, len(self.func_list))))
        cov = np.zeros((self.full_dim, self.full_dim))
        block = np.zeros((self.dim, self.dim))

        if par == 'mus':
            for ii in range(len(cut)):
                for jj in range(ii, len(self.funcs)):
                    if ((ii == 0 or jj == 0) and ii != jj):
                        block[ii][jj] = 0
                    else:
                        block[ii][jj] = ( (self.funcs[ii](np.array([cut[ii]]), *self.hqe_pars)
                                          - self.funcs_shift[ii](np.array([cut[ii]]), *self.hqe_pars))
                                         *(self.funcs[jj](np.array([cut[jj]]), *self.hqe_pars)
                                          - self.funcs_shift[jj](np.array([cut[jj]]), *self.hqe_pars)) )
                        block[jj][ii] = block[ii][jj]
        elif par in self.shifts:
            hqe_shift = self.hqe_pars*self.BuildShift(par)
            for ii in range(len(cut)):
                for jj in range(ii, len(self.funcs)):
                    if ii == 0 or jj == 0 and ii != jj:
                        block[ii][jj] = 0
                    else:
                        block[ii][jj] = ( (self.funcs[ii](np.array([cut[ii]]), *self.hqe_pars)
                                          - self.funcs[ii](np.array([cut[ii]]), *hqe_shift))
                                          *(self.funcs[jj](np.array([cut[jj]]), *self.hqe_pars)
                                          - self.funcs[jj](np.array([cut[jj]]), *hqe_shift)) )
                        block[jj][ii] = block[ii][jj]
        else:
            raise KeyError('No input value provided for the variation of %s'%par)

        cov[0, 0] = block[0, 0]
        cov[1:, 1:] = np.block([[block[1:, 1:]]*self.multi]*self.multi)
        return cov*self.decorr_mat

    # Sums up the covariance matrices for all the shifts provided.
    def BuildSummedCovariance(self):
        summed_mat = np.zeros((self.full_dim, self.full_dim))
        for par in self.shifts:
            summed_mat += self.BuildCovarianceMatrix(par)
        self.corr_sum = self.BuildCorrelationMatrix(summed_mat)
        return summed_mat
