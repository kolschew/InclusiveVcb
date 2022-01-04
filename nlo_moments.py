"""
Description:
    Module contains the genuine NLO corrections to the lepton invariant mass moments.
    The differential width is integrated with different powers in omega for the respective moment.

Contains:
    diff_sec: the differential width taken from arXiv:----
    q2_nlo_0,...q2_nlo_4: genuine NLO correction for the rate and 1st - 4th q2-moment
"""

import numpy as np
from mpmath import polylog, fp
from scipy.integrate import fixed_quad

KNOTS = 71  # knots used in the fixed quadrature
CUT = 1e-10  # cutoff at the endpoint for numerical stability


@np.vectorize
def np_polylog(nn, arr):
    return fp.polylog(nn, arr)


def diff_sec(omega, rho):
    kaellen = 1 + omega ** 2 + rho ** 4 - 2 * (omega + rho ** 2 + omega * rho ** 2)
    f_0 = 4 * ((1 - rho ** 2) ** 2 + omega * (1 + rho ** 2) - 2 * omega ** 2)
    u_q = (1 + rho ** 2 - omega - np.sqrt(kaellen)) / (1 + rho ** 2 - omega + np.sqrt(kaellen))
    u_w = (1 - rho ** 2 + omega - np.sqrt(kaellen)) / (1 - rho ** 2 + omega + np.sqrt(kaellen))
    
    ret_val = (1 / 2 * f_0 * (1 + rho ** 2 - omega) * (
            np.pi ** 2 + 2 * np_polylog(2, u_w) - 2 * np_polylog(2, (1 - u_w)) - 4 * np_polylog(2, u_q)
            - 4 * np_polylog(2, u_q * u_w) + np.log((1 - u_q) / omega) * np.log(1 - u_q) - np.log(1 - u_q * u_w) ** 2
            + 1 / 4 * np.log(omega / u_w) ** 2 - np.log(u_w) * np.log((1 - u_q * u_w) ** 2 / (1 - u_q))
            - 2 * np.log(u_q) * np.log((1 - u_q) * (1 - u_q * u_w)))
               - np.sqrt(kaellen) * f_0 * (2 * np.log(np.sqrt(omega)) + 3 * np.log(rho) - 2 * np.log(kaellen))
               + 4 * (1 - rho ** 2) * ((1 - rho ** 2) ** 2 + omega * (1 + rho ** 2) - 4 * omega ** 2) * np.log(u_w)
               + ((3 - rho ** 2 + 11 * rho ** 4 - rho ** 6) + omega * (6 - 12 * rho ** 2 + 2 * rho ** 4)
                  - omega ** 2 * (21 + 5 * rho ** 2) + 12 * omega ** 3) * np.log(u_q)
               + 6 * np.sqrt(kaellen) * (1 - rho ** 2) * (1 + rho ** 2 - omega) * np.log(rho)
               + np.sqrt(kaellen) * (22 * rho ** 2 - 5 - 5 * rho ** 4 - 9 * omega * (1 + rho ** 2) + 6 * omega ** 2))
    return ret_val


def q2_nlo_0(q2cut, rho):
    def integrand(omega):
        res = diff_sec(omega, rho)
        return res

    mom0_res = []
    for cc in q2cut:
        mom0_res.append(fixed_quad(integrand, cc, (1 - rho) ** 2 - CUT, n=KNOTS)[0])
    return np.array(mom0_res)


def q2_nlo_1(q2cut, rho):
    def integrand(omega):
        res = omega * diff_sec(omega, rho)
        return res

    mom1_res = []
    for cc in q2cut:
        mom1_res.append(fixed_quad(integrand, cc, (1 - rho) ** 2 - CUT, n=KNOTS)[0])
    return np.array(mom1_res)


def q2_nlo_2(q2cut, rho):
    def integrand(omega):
        res = omega ** 2 * diff_sec(omega, rho)
        return res

    mom2_res = []
    for cc in q2cut:
        mom2_res.append(fixed_quad(integrand, cc, (1 - rho) ** 2 - CUT, n=KNOTS)[0])
    return np.array(mom2_res)


def q2_nlo_3(q2cut, rho):
    def integrand(omega):
        res = omega ** 3 * diff_sec(omega, rho)
        return res

    mom3_res = []
    for cc in q2cut:
        mom3_res.append(fixed_quad(integrand, cc, (1 - rho) ** 2 - CUT, n=KNOTS)[0])
    return np.array(mom3_res)


def q2_nlo_4(q2cut, rho):
    def integrand(omega):
        res = omega ** 4 * diff_sec(omega, rho)
        return res

    mom4_res = []
    for cc in q2cut:
        mom4_res.append(fixed_quad(integrand, cc, (1 - rho) ** 2 - CUT, n=KNOTS)[0])
    return np.array(mom4_res)
