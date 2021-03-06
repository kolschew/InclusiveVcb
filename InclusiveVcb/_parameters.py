import numpy as np
from dataclasses import dataclass, field
import rundec


@dataclass
class VcbData:
    """Class holds all parameters and physical constants for inclusive Vcb."""

    # Physical constants
    g_f: float = field(default=1.166378e-5, init=False)
    vcb_pdg: float = field(default=0.0422, init=False)
    a_ew: float = field(default=1.014, init=False)
    hbar: float = field(default=6.58212e-25, init=False)
    tauB: float = field(default=(1.519 + 1.638) * 0.5e-12, init=False)
    muW: int = field(default=1, init=False)

    # Default masses in OS, MS and kinetic scheme
    mbkin: float = 4.546
    mckin: float = 1.130
    mcMS_3: float = 0.987
    mcMS_2: float = 1.093

    # Default values for the HQE parameters
    muG: float = 0.36156
    sB: float = -0.132
    rE: float = 0.019
    sqB: float = -0.8
    sE: float = -0.072
    rG: float = -0.006
    rhoD: float = 0.145
    mupi: float = 0.432375

    # Scale for a_s and m_c
    mus: float = 4.546
    mu0: float = 2

    def __post_init__(self):
        self.api4 = self._run_api(self.mus) / np.pi
        self.mcMS = self.mcMS_3 if self.mu0 == 3 else self.mcMS_2
        self.default = (self.mbkin, self.mcMS, self.muG, self.sB, self.rE,
                        self.sqB, self.sE, self.rG, self.rhoD, self.mupi)
        self.noHQE = (self.mbkin, self.mcMS, 0, 0, 0, 0, 0, 0, 0, 0)

    def gamma_0(self, mb, vcb):
        """Prefactor of the total rate.

        Parameters
        ----------
        mb : float
            Mass of the b-quark in respective scheme
        vcb : float
            Value for CKM-element V_cb

        Return
        ------
        float
            Value for the prefactor Gamma_0 with electroweak correction.
        """

        res = self.a_ew * self.g_f ** 2 * vcb ** 2 * mb ** 5 / (192 * np.pi ** 3)
        return res

    @staticmethod
    def _run_api(scale):
        """Runs a_s down to the desired scale at 4 loops with n_f = 4 by using rundec.

        Parameters
        ----------
        scale : float
            Scale to which a_s is run and evaluated

        Return
        ------
        float
            Value of the strong coupling constant at given scale.
        """

        # Define parameters for running a_s
        _asmZ = 0.1179
        _mZ = 91.1876
        _mbMS = 4.198

        # Initialize rundec and run a_s. n_f = 4
        crd = rundec.CRunDec()
        as_res = (crd.AlphasExact(crd.DecAsDownSI(crd.AlphasExact(_asmZ, _mZ, 2 * _mbMS, 5, 4),
                                                  _mbMS, 2 * _mbMS, 4, 4), 2 * _mbMS, scale, 4, 4))
        return as_res
