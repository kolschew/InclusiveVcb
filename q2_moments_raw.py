import numpy as np
import warnings 

def q2moment0(q_cut, mbkin, mc):
 res = ( (np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - (2*(mc**2 + q_cut))/mbkin**2)*
       (mbkin**6 + (mc**2 - q_cut)**2*(mc**2 + q_cut) - mbkin**4*(7*mc**2 + q_cut) - 
        mbkin**2*(7*mc**4 + q_cut**2)) - 12*mbkin**2*mc**4*
       np.log((mc**2 - q_cut - mbkin**2*(-1 + np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - 
              (2*(mc**2 + q_cut))/mbkin**2)))/(mc**2 - q_cut + 
          mbkin**2*(1 + np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - (2*(mc**2 + q_cut))/mbkin**
                2)))))/mbkin**6 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

def q2moment1(q_cut, mbkin, mc):
 res = ( (np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - (2*(mc**2 + q_cut))/mbkin**2)*
       (3*mbkin**8 + mbkin**6*(-42*mc**2 + 3*q_cut) - 
        mbkin**4*(282*mc**4 + 33*mc**2*q_cut + 7*q_cut**2) + (mc**2 - q_cut)**2*
         (3*mc**4 + 9*mc**2*q_cut + 8*q_cut**2) - mbkin**2*(42*mc**6 + 33*mc**4*q_cut - 
          2*mc**2*q_cut**2 + 7*q_cut**3)))/(10*mbkin**8) - 
     (18*mc**4*(mbkin**2 + mc**2)*
       np.log((mc**2 - q_cut - mbkin**2*(-1 + np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - 
              (2*(mc**2 + q_cut))/mbkin**2)))/(mc**2 - q_cut + 
          mbkin**2*(1 + np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - (2*(mc**2 + q_cut))/mbkin**
                2)))))/mbkin**6 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

def q2moment2(q_cut, mbkin, mc):
 res = ( (2*np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - (2*(mc**2 + q_cut))/mbkin**2)*
       (mbkin**10 + mbkin**8*(-23*mc**2 + q_cut) + 
        mbkin**6*(-398*mc**4 - 20*mc**2*q_cut + q_cut**2) - 
        mbkin**4*(398*mc**6 + 102*mc**4*q_cut + 15*mc**2*q_cut**2 + 4*q_cut**3) + 
        (mc**2 - q_cut)**2*(mc**6 + 3*mc**4*q_cut + 6*mc**2*q_cut**2 + 5*q_cut**3) - 
        mbkin**2*(23*mc**8 + 20*mc**6*q_cut + 15*mc**4*q_cut**2 - 2*mc**2*q_cut**3 + 
          4*q_cut**4)))/(15*mbkin**10) - 
     (8*mc**4*(3*mbkin**4 + 8*mbkin**2*mc**2 + 3*mc**4)*
       np.log((mc**2 - q_cut - mbkin**2*(-1 + np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - 
              (2*(mc**2 + q_cut))/mbkin**2)))/(mc**2 - q_cut + 
          mbkin**2*(1 + np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - (2*(mc**2 + q_cut))/mbkin**
                2)))))/mbkin**8 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

def q2moment3(q_cut, mbkin, mc):
 res = ( (np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - (2*(mc**2 + q_cut))/mbkin**2)*
       (mbkin**12 + mbkin**10*(-34*mc**2 + q_cut) + 
        mbkin**8*(-1133*mc**4 - 31*mc**2*q_cut + q_cut**2) + 
        mbkin**6*(-2708*mc**6 - 390*mc**4*q_cut - 26*mc**2*q_cut**2 + q_cut**3) - 
        mbkin**4*(1133*mc**8 + 390*mc**6*q_cut + 118*mc**4*q_cut**2 + 19*mc**2*q_cut**3 + 
          6*q_cut**4) + (mc**2 - q_cut)**2*(mc**8 + 3*mc**6*q_cut + 6*mc**4*q_cut**2 + 
          10*mc**2*q_cut**3 + 8*q_cut**4) - mbkin**2*(34*mc**10 + 31*mc**8*q_cut + 
          26*mc**6*q_cut**2 + 19*mc**4*q_cut**3 - 4*mc**2*q_cut**4 + 6*q_cut**5)) - 
      420*mbkin**2*mc**4*(mbkin**2 + mc**2)*(mbkin**4 + 4*mbkin**2*mc**2 + mc**4)*
       np.log((mc**2 - q_cut - mbkin**2*(-1 + np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - 
              (2*(mc**2 + q_cut))/mbkin**2)))/(mc**2 - q_cut + 
          mbkin**2*(1 + np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - (2*(mc**2 + q_cut))/mbkin**
                2)))))/(14*mbkin**12) )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

def q2moment4(q_cut, mbkin, mc):
 res = ( (np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - (2*(mc**2 + q_cut))/mbkin**2)*
       (3*mbkin**14 + 3*mbkin**12*(-47*mc**2 + q_cut) - 
        3*mbkin**10*(2595*mc**4 + 44*mc**2*q_cut - q_cut**2) - 
        3*mbkin**8*(11219*mc**6 + 1051*mc**4*q_cut + 39*mc**2*q_cut**2 - q_cut**3) - 
        3*mbkin**6*(11219*mc**8 + 2432*mc**6*q_cut + 424*mc**4*q_cut**2 + 32*mc**2*q_cut**3 - 
          q_cut**4) - mbkin**4*(7785*mc**10 + 3153*mc**8*q_cut + 1272*mc**6*q_cut**2 + 
          408*mc**4*q_cut**3 + 69*mc**2*q_cut**4 + 25*q_cut**5) + (mc**2 - q_cut)**2*
         (3*mc**10 + 9*mc**8*q_cut + 18*mc**6*q_cut**2 + 30*mc**4*q_cut**3 + 45*mc**2*q_cut**4 + 
          35*q_cut**5) - mbkin**2*(141*mc**12 + 132*mc**10*q_cut + 117*mc**8*q_cut**2 + 
          96*mc**6*q_cut**3 + 69*mc**4*q_cut**4 - 20*mc**2*q_cut**5 + 25*q_cut**6)))/
      (70*mbkin**14) - (36*mc**4*(mbkin**8 + 8*mbkin**6*mc**2 + 15*mbkin**4*mc**4 + 
        8*mbkin**2*mc**6 + mc**8)*
       np.log((mc**2 - q_cut - mbkin**2*(-1 + np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - 
              (2*(mc**2 + q_cut))/mbkin**2)))/(mc**2 - q_cut + 
          mbkin**2*(1 + np.sqrt(0j + 1 + (mc**2 - q_cut)**2/mbkin**4 - (2*(mc**2 + q_cut))/mbkin**
                2)))))/mbkin**12 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

