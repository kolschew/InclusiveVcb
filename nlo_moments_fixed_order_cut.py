import numpy as np
from mpmath import polylog, fp
from scipy.integrate import fixed_quad

knots = 71 #number of knots used in the fixed quadrature.
cut = 1e-10


def np_polylog(nn, arr):
    ret = []
    if type(arr) == np.ndarray:
        for ii in arr:
            ret.append(fp.polylog(nn, ii))
    else:
        ret = [fp.polylog(nn, arr)]
    return np.array(ret)


def diff_sec(omega, rho):
    kaellen = 1 + omega**2 + rho**4 - 2*(omega + rho**2 + omega*rho**2)
    f_0 = 4*((1 - rho**2)**2 + omega*(1 + rho**2) - 2*omega**2)
    u_q = (1 + rho**2 - omega - np.sqrt(kaellen))/(1 + rho**2 - omega + np.sqrt(kaellen))
    u_w = (1 - rho**2 + omega - np.sqrt(kaellen))/(1 - rho**2 + omega + np.sqrt(kaellen))
    ret_val = ( 1/2*f_0*(1 + rho**2 - omega)*( np.pi**2 + 2*np_polylog(2, u_w) - 2*np_polylog(2, (1 - u_w)) - 4*np_polylog(2, u_q)
                - 4*np_polylog(2, u_q*u_w) + np.log((1 - u_q)/omega)*np.log(1 - u_q) - np.log(1 - u_q*u_w)**2
                + 1/4*np.log(omega/u_w)**2 - np.log(u_w)*np.log((1 - u_q*u_w)**2/(1 - u_q))
                - 2*np.log(u_q)*np.log((1 - u_q)*(1 - u_q*u_w)) )
               - np.sqrt(kaellen)*f_0*(2*np.log(np.sqrt(omega)) + 3*np.log(rho) - 2*np.log(kaellen))
               + 4*(1 - rho**2)*((1 - rho**2)**2 + omega*(1 + rho**2) - 4*omega**2)*np.log(u_w)
               + ((3 - rho**2 + 11*rho**4 - rho**6) + omega*(6 - 12*rho**2 + 2*rho**4)
                - omega**2*(21 + 5*rho**2) + 12*omega**3)*np.log(u_q)
               + 6*np.sqrt(kaellen)*(1 - rho**2)*(1 + rho**2 - omega)*np.log(rho)
               + np.sqrt(kaellen)*(22*rho**2 - 5 - 5*rho**4 - 9*omega*(1 + rho**2) + 6*omega**2) )
    return ret_val


def q2_nlo_0(q2cut, rho):
    mom0_res = []
    for cc in q2cut:
        mom0_res.append(fixed_quad(diff_sec, cc, (1 - rho)**2 - cut, args=(rho,), n=knots)[0])
    return np.array(mom0_res)


def q2_nlo_1(q2cut, rho):
    mom1_res = []
    integrand = lambda omega, rho: omega*diff_sec(omega, rho)
    for cc in q2cut:
        mom1_res.append(fixed_quad(integrand, cc, (1 - rho)**2 - cut, args=(rho,), n=knots)[0])
    return np.array(mom1_res)


def q2_nlo_2(q2cut, rho):
    mom2_res = []
    integrand = lambda omega, rho: omega**2*diff_sec(omega, rho)
    for cc in q2cut:
        mom2_res.append(fixed_quad(integrand, cc, (1 - rho)**2 - cut, args=(rho,), n=knots)[0])
    return np.array(mom2_res)


def q2_nlo_3(q2cut, rho):
    mom3_res = []
    integrand = lambda omega, rho: omega**3*diff_sec(omega, rho)
    for cc in q2cut:
        mom3_res.append(fixed_quad(integrand, cc, (1 - rho)**2 - cut, args=(rho,), n=knots)[0])
    return np.array(mom3_res)


def q2_nlo_4(q2cut, rho):
    mom4_res = []
    integrand = lambda omega, rho_diff: omega ** 4 * diff_sec(omega, rho_diff)
    for cc in q2cut:
        mom4_res.append(fixed_quad(integrand, cc, (1 - rho)**2 - cut, args=(rho,), n=knots)[0])
    return np.array(mom4_res)
