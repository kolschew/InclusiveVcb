import numpy as np
import warnings 

def q2moment1MS(q_cut, mbkin, mcMS, mus, mu0, api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
 res = ( ((18*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
           2*mcMS**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
           7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
           mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)**2*(3*mbkin**8 - 
          42*mbkin**6*mcMS**2 - 282*mbkin**4*mcMS**4 - 42*mbkin**2*mcMS**6 + 
          3*mcMS**8 + 3*mbkin**6*q_cut - 33*mbkin**4*mcMS**2*q_cut - 
          33*mbkin**2*mcMS**4*q_cut + 3*mcMS**6*q_cut - 7*mbkin**4*q_cut**2 + 
          2*mbkin**2*mcMS**2*q_cut**2 - 7*mcMS**4*q_cut**2 - 7*mbkin**2*q_cut**3 - 
          7*mcMS**2*q_cut**3 + 8*q_cut**4))/mbkin**28 - 
       (216*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
           2*mcMS**2*q_cut + q_cut**2)**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
            2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
         (21*mbkin**14 - 321*mbkin**12*mcMS**2 + 297*mbkin**10*mcMS**4 + 
          6483*mbkin**8*mcMS**6 + 6483*mbkin**6*mcMS**8 + 297*mbkin**4*mcMS**10 - 
          321*mbkin**2*mcMS**12 + 21*mcMS**14 - 30*mbkin**12*q_cut + 
          156*mbkin**10*mcMS**2*q_cut + 1302*mbkin**8*mcMS**4*q_cut + 
          1464*mbkin**6*mcMS**6*q_cut + 1302*mbkin**4*mcMS**8*q_cut + 
          156*mbkin**2*mcMS**10*q_cut - 30*mcMS**12*q_cut - 41*mbkin**10*q_cut**2 + 
          411*mbkin**8*mcMS**2*q_cut**2 + 1394*mbkin**6*mcMS**4*q_cut**2 + 
          1394*mbkin**4*mcMS**6*q_cut**2 + 411*mbkin**2*mcMS**8*q_cut**2 - 
          41*mcMS**10*q_cut**2 + 60*mbkin**8*q_cut**3 - 64*mbkin**6*mcMS**2*q_cut**3 - 
          568*mbkin**4*mcMS**4*q_cut**3 - 64*mbkin**2*mcMS**6*q_cut**3 + 60*mcMS**8*q_cut**3 + 
          35*mbkin**6*q_cut**4 - 139*mbkin**4*mcMS**2*q_cut**4 - 139*mbkin**2*mcMS**4*
           q_cut**4 + 35*mcMS**6*q_cut**4 - 46*mbkin**4*q_cut**5 - 28*mbkin**2*mcMS**2*q_cut**5 - 
          46*mcMS**4*q_cut**5 - 15*mbkin**2*q_cut**6 - 15*mcMS**2*q_cut**6 + 16*q_cut**7)*
         np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                 mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**
                4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
             np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**22 + 
       (2592*mcMS**8*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
           2*mcMS**2*q_cut + q_cut**2)**2*(33*mbkin**8 - 222*mbkin**6*mcMS**2 - 
          702*mbkin**4*mcMS**4 - 222*mbkin**2*mcMS**6 + 33*mcMS**8 - 
          27*mbkin**6*q_cut - 63*mbkin**4*mcMS**2*q_cut - 63*mbkin**2*mcMS**4*q_cut - 
          27*mcMS**6*q_cut - 37*mbkin**4*q_cut**2 - 58*mbkin**2*mcMS**2*q_cut**2 - 
          37*mcMS**4*q_cut**2 + 23*mbkin**2*q_cut**3 + 23*mcMS**2*q_cut**3 + 8*q_cut**4)*
         np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**20 - 
       (466560*mcMS**12*(mbkin**2 + mcMS**2)*
         ((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
            q_cut**2)/mbkin**4)**(3/2)*np.log((mbkin**2 + mcMS**2 - q_cut - 
             mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                  q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
             mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                  q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**10)/
      (180*mbkin**2*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
          2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
            7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
            mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**3) + 
     (((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
         q_cut**2/mbkin**4)*(-24*mbkin**2*muG*((-1 + mcMS**2/mbkin**2)**2 - 
            (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)**2*
          (6 - (41*mcMS**2)/mbkin**2 + (20*mcMS**4)/mbkin**4 - 
           (800*mcMS**6)/mbkin**6 + (3184*mcMS**8)/mbkin**8 - 
           (8374*mcMS**10)/mbkin**10 - (7172*mcMS**12)/mbkin**12 + 
           (4912*mcMS**14)/mbkin**14 - (358*mcMS**16)/mbkin**16 - 
           (17*mcMS**18)/mbkin**18 + ((-15 + (95*mcMS**2)/mbkin**2 + 
              (173*mcMS**4)/mbkin**4 - (3465*mcMS**6)/mbkin**6 + 
              (395*mcMS**8)/mbkin**8 + (2825*mcMS**10)/mbkin**10 - 
              (5061*mcMS**12)/mbkin**12 + (689*mcMS**14)/mbkin**14 + 
              (44*mcMS**16)/mbkin**16)*q_cut)/mbkin**2 + 
           ((-6 + (31*mcMS**2)/mbkin**2 + (180*mcMS**4)/mbkin**4 - 
              (3195*mcMS**6)/mbkin**6 - (10526*mcMS**8)/mbkin**8 - 
              (3891*mcMS**10)/mbkin**10 + (552*mcMS**12)/mbkin**12 + 
              (7*mcMS**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
           ((45 - (263*mcMS**2)/mbkin**2 - (1114*mcMS**4)/mbkin**4 + 
              (3670*mcMS**6)/mbkin**6 + (5017*mcMS**8)/mbkin**8 - 
              (1723*mcMS**10)/mbkin**10 - (104*mcMS**12)/mbkin**12)*q_cut**3)/
            mbkin**6 + ((-30 + (221*mcMS**2)/mbkin**2 + (80*mcMS**4)/mbkin**4 - 
              (2094*mcMS**6)/mbkin**6 + (302*mcMS**8)/mbkin**8 + 
              (65*mcMS**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
           ((-21 + (37*mcMS**2)/mbkin**2 + (1245*mcMS**4)/mbkin**4 + 
              (1095*mcMS**6)/mbkin**6 + (52*mcMS**8)/mbkin**8)*q_cut**5)/mbkin**10 - 
           ((-30 + (119*mcMS**2)/mbkin**2 + (592*mcMS**4)/mbkin**4 + 
              (59*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
           ((-9 + (35*mcMS**2)/mbkin**2 + (8*mcMS**4)/mbkin**4)*q_cut**7)/mbkin**14 + 
           (4*mcMS**2*q_cut**8)/mbkin**18) - 12*muG*mupi*
          ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)**2*(6 - (41*mcMS**2)/mbkin**2 + (20*mcMS**4)/mbkin**4 - 
           (800*mcMS**6)/mbkin**6 + (3184*mcMS**8)/mbkin**8 - 
           (8374*mcMS**10)/mbkin**10 - (7172*mcMS**12)/mbkin**12 + 
           (4912*mcMS**14)/mbkin**14 - (358*mcMS**16)/mbkin**16 - 
           (17*mcMS**18)/mbkin**18 + ((-15 + (95*mcMS**2)/mbkin**2 + 
              (173*mcMS**4)/mbkin**4 - (3465*mcMS**6)/mbkin**6 + 
              (395*mcMS**8)/mbkin**8 + (2825*mcMS**10)/mbkin**10 - 
              (5061*mcMS**12)/mbkin**12 + (689*mcMS**14)/mbkin**14 + 
              (44*mcMS**16)/mbkin**16)*q_cut)/mbkin**2 + 
           ((-6 + (31*mcMS**2)/mbkin**2 + (180*mcMS**4)/mbkin**4 - 
              (3195*mcMS**6)/mbkin**6 - (10526*mcMS**8)/mbkin**8 - 
              (3891*mcMS**10)/mbkin**10 + (552*mcMS**12)/mbkin**12 + 
              (7*mcMS**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
           ((45 - (263*mcMS**2)/mbkin**2 - (1114*mcMS**4)/mbkin**4 + 
              (3670*mcMS**6)/mbkin**6 + (5017*mcMS**8)/mbkin**8 - 
              (1723*mcMS**10)/mbkin**10 - (104*mcMS**12)/mbkin**12)*q_cut**3)/
            mbkin**6 + ((-30 + (221*mcMS**2)/mbkin**2 + (80*mcMS**4)/mbkin**4 - 
              (2094*mcMS**6)/mbkin**6 + (302*mcMS**8)/mbkin**8 + 
              (65*mcMS**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
           ((-21 + (37*mcMS**2)/mbkin**2 + (1245*mcMS**4)/mbkin**4 + 
              (1095*mcMS**6)/mbkin**6 + (52*mcMS**8)/mbkin**8)*q_cut**5)/mbkin**10 - 
           ((-30 + (119*mcMS**2)/mbkin**2 + (592*mcMS**4)/mbkin**4 + 
              (59*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
           ((-9 + (35*mcMS**2)/mbkin**2 + (8*mcMS**4)/mbkin**4)*q_cut**7)/mbkin**14 + 
           (4*mcMS**2*q_cut**8)/mbkin**18) + 12*muG**2*((-1 + mcMS**2/mbkin**2)**2 - 
            (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)**2*
          (-18 + (27*mcMS**2)/mbkin**2 - (316*mcMS**4)/mbkin**4 + 
           (1304*mcMS**6)/mbkin**6 - (7248*mcMS**8)/mbkin**8 + 
           (15050*mcMS**10)/mbkin**10 - (31540*mcMS**12)/mbkin**12 + 
           (16248*mcMS**14)/mbkin**14 - (2062*mcMS**16)/mbkin**16 - 
           (85*mcMS**18)/mbkin**18 + ((-3 - (101*mcMS**2)/mbkin**2 - 
              (1279*mcMS**4)/mbkin**4 + (5419*mcMS**6)/mbkin**6 - 
              (22225*mcMS**8)/mbkin**8 + (31533*mcMS**10)/mbkin**10 - 
              (21625*mcMS**12)/mbkin**12 + (3741*mcMS**14)/mbkin**14 + 
              (220*mcMS**16)/mbkin**16)*q_cut)/mbkin**2 + 
           ((42 - (341*mcMS**2)/mbkin**2 - (3860*mcMS**4)/mbkin**4 + 
              (8857*mcMS**6)/mbkin**6 - (24270*mcMS**8)/mbkin**8 - 
              (6319*mcMS**10)/mbkin**10 + (3248*mcMS**12)/mbkin**12 + 
              (35*mcMS**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
           ((57 + (133*mcMS**2)/mbkin**2 - (4242*mcMS**4)/mbkin**4 + 
              (894*mcMS**6)/mbkin**6 + (19333*mcMS**8)/mbkin**8 - 
              (9167*mcMS**10)/mbkin**10 - (520*mcMS**12)/mbkin**12)*q_cut**3)/
            mbkin**6 + ((-78 + (601*mcMS**2)/mbkin**2 + (1040*mcMS**4)/mbkin**4 - 
              (13694*mcMS**6)/mbkin**6 + (1358*mcMS**8)/mbkin**8 + 
              (325*mcMS**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
           ((-105 + (81*mcMS**2)/mbkin**2 + (6697*mcMS**4)/mbkin**4 + 
              (5635*mcMS**6)/mbkin**6 + (260*mcMS**8)/mbkin**8)*q_cut**5)/mbkin**10 - 
           ((-150 + (595*mcMS**2)/mbkin**2 + (2928*mcMS**4)/mbkin**4 + 
              (295*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
           (5*(-9 + (35*mcMS**2)/mbkin**2 + (8*mcMS**4)/mbkin**4)*q_cut**7)/
            mbkin**14 + (20*mcMS**2*q_cut**8)/mbkin**18) - 
         8*mbkin*(-((-1 + mcMS**2/mbkin**2)**4*(1 + mcMS**2/mbkin**2)**2*
             (351 - (5815*mcMS**2)/mbkin**2 + (36915*mcMS**4)/mbkin**4 - 
              (109923*mcMS**6)/mbkin**6 + (69257*mcMS**8)/mbkin**8 - 
              (22209*mcMS**10)/mbkin**10 + (1077*mcMS**12)/mbkin**12 + 
              (107*mcMS**14)/mbkin**14)) + (2*(-1 + mcMS**2/mbkin**2)**2*
             (1089 - (13483*mcMS**2)/mbkin**2 + (53852*mcMS**4)/mbkin**4 - 
              (45772*mcMS**6)/mbkin**6 - (288518*mcMS**8)/mbkin**8 - 
              (239134*mcMS**10)/mbkin**10 + (43532*mcMS**12)/mbkin**12 + 
              (52492*mcMS**14)/mbkin**14 - (52219*mcMS**16)/mbkin**16 + 
              (3977*mcMS**18)/mbkin**18 + (344*mcMS**20)/mbkin**20)*q_cut)/mbkin**2 + 
           ((-4929 + (52403*mcMS**2)/mbkin**2 - (176271*mcMS**4)/mbkin**4 + 
              (115749*mcMS**6)/mbkin**6 + (605090*mcMS**8)/mbkin**8 + 
              (898826*mcMS**10)/mbkin**10 + (657918*mcMS**12)/mbkin**12 - 
              (174202*mcMS**14)/mbkin**14 - (222817*mcMS**16)/mbkin**16 + 
              (202947*mcMS**18)/mbkin**18 - (17775*mcMS**20)/mbkin**20 - 
              (1579*mcMS**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
           (2*(1659 - (10934*mcMS**2)/mbkin**2 + (14621*mcMS**4)/mbkin**4 + 
              (22408*mcMS**6)/mbkin**6 - (61810*mcMS**8)/mbkin**8 + 
              (4308*mcMS**10)/mbkin**10 + (92322*mcMS**12)/mbkin**12 - 
              (26296*mcMS**14)/mbkin**14 - (44537*mcMS**16)/mbkin**16 + 
              (7826*mcMS**18)/mbkin**18 + (433*mcMS**20)/mbkin**20)*q_cut**3)/
            mbkin**6 + (4*(1353 - (9240*mcMS**2)/mbkin**2 + (13738*mcMS**4)/mbkin**
                4 + (9254*mcMS**6)/mbkin**6 - (21795*mcMS**8)/mbkin**8 - 
              (9861*mcMS**10)/mbkin**10 - (9696*mcMS**12)/mbkin**12 - 
              (33052*mcMS**14)/mbkin**14 + (4192*mcMS**16)/mbkin**16 + 
              (675*mcMS**18)/mbkin**18)*q_cut**4)/mbkin**8 - 
           (2*(6003 - (25356*mcMS**2)/mbkin**2 + (10882*mcMS**4)/mbkin**4 + 
              (16908*mcMS**6)/mbkin**6 - (30624*mcMS**8)/mbkin**8 - 
              (109320*mcMS**10)/mbkin**10 - (93822*mcMS**12)/mbkin**12 + 
              (22600*mcMS**14)/mbkin**14 + (2793*mcMS**16)/mbkin**16)*q_cut**5)/
            mbkin**10 + (2*(3483 - (12003*mcMS**2)/mbkin**2 - (3121*mcMS**4)/
               mbkin**4 - (19139*mcMS**6)/mbkin**6 - (74989*mcMS**8)/mbkin**8 - 
              (52799*mcMS**10)/mbkin**10 + (11627*mcMS**12)/mbkin**12 + 
              (1677*mcMS**14)/mbkin**14)*q_cut**6)/mbkin**12 + 
           (4*(765 + (1989*mcMS**2)/mbkin**2 + (3559*mcMS**4)/mbkin**4 + 
              (13202*mcMS**6)/mbkin**6 + (14387*mcMS**8)/mbkin**8 + 
              (5569*mcMS**10)/mbkin**10 + (417*mcMS**12)/mbkin**12)*q_cut**7)/
            mbkin**14 - ((6399 + (6327*mcMS**2)/mbkin**2 + (19022*mcMS**4)/mbkin**
                4 + (37806*mcMS**6)/mbkin**6 + (34907*mcMS**8)/mbkin**8 + 
              (3747*mcMS**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
           (2*(1719 + (1825*mcMS**2)/mbkin**2 + (6279*mcMS**4)/mbkin**4 + 
              (8751*mcMS**6)/mbkin**6 + (1150*mcMS**8)/mbkin**8)*q_cut**9)/mbkin**18 - 
           ((693 + (829*mcMS**2)/mbkin**2 + (3367*mcMS**4)/mbkin**4 + 
              (607*mcMS**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
           ((6 + (88*mcMS**2)/mbkin**2 + (58*mcMS**4)/mbkin**4)*q_cut**11)/mbkin**22 - 
           (2*(3 + (7*mcMS**2)/mbkin**2)*q_cut**12)/mbkin**24 + (6*q_cut**13)/mbkin**26)*
          rhoD + ((mbkin**6 - 7*mbkin**4*mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - 
            mbkin**4*q_cut - mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)*
           (-16*(-((-1 + mcMS**2/mbkin**2)**4*(-317 + (2200*mcMS**2)/mbkin**2 + 
                 (1589*mcMS**4)/mbkin**4 - (2096*mcMS**6)/mbkin**6 - 
                 (2611*mcMS**8)/mbkin**8 + (2536*mcMS**10)/mbkin**10 + 
                 (139*mcMS**12)/mbkin**12)) + ((-1 + mcMS**2/mbkin**2)**2*
                (-1535 + (7527*mcMS**2)/mbkin**2 + (11043*mcMS**4)/mbkin**4 + 
                 (3449*mcMS**6)/mbkin**6 - (13681*mcMS**8)/mbkin**8 - 
                 (8751*mcMS**10)/mbkin**10 + (12717*mcMS**12)/mbkin**12 + 
                 (751*mcMS**14)/mbkin**14)*q_cut)/mbkin**2 - 
              ((-2610 + (10073*mcMS**2)/mbkin**2 + (8271*mcMS**4)/mbkin**4 + 
                 (9663*mcMS**6)/mbkin**6 + (16889*mcMS**8)/mbkin**8 - 
                 (24081*mcMS**10)/mbkin**10 - (19599*mcMS**12)/mbkin**12 + 
                 (23033*mcMS**14)/mbkin**14 + (1401*mcMS**16)/mbkin**16)*q_cut**2)/
               mbkin**4 + ((-1137 + (2623*mcMS**2)/mbkin**2 + (347*mcMS**4)/
                  mbkin**4 - (3231*mcMS**6)/mbkin**6 - (18327*mcMS**8)/mbkin**8 + 
                 (13433*mcMS**10)/mbkin**10 + (16165*mcMS**12)/mbkin**12 + 
                 (447*mcMS**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
              ((-1962 - (917*mcMS**2)/mbkin**2 + (8227*mcMS**4)/mbkin**4 + 
                 (11800*mcMS**6)/mbkin**6 - (1490*mcMS**8)/mbkin**8 + 
                 (10753*mcMS**10)/mbkin**10 + (2205*mcMS**12)/mbkin**12)*q_cut**4)/
               mbkin**8 - ((-3015 - (1561*mcMS**2)/mbkin**2 + (11820*mcMS**4)/
                  mbkin**4 + (20886*mcMS**6)/mbkin**6 + (25655*mcMS**8)/mbkin**8 + 
                 (3759*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
              ((-1710 + (271*mcMS**2)/mbkin**2 + (11609*mcMS**4)/mbkin**4 + 
                 (16225*mcMS**6)/mbkin**6 + (2709*mcMS**8)/mbkin**8)*q_cut**6)/mbkin**
                12 - ((-465 + (689*mcMS**2)/mbkin**2 + (3797*mcMS**4)/mbkin**4 + 
                 (939*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
              ((-63 + (115*mcMS**2)/mbkin**2 + (138*mcMS**4)/mbkin**4)*q_cut**8)/
               mbkin**16 - (4*(2 + (5*mcMS**2)/mbkin**2)*q_cut**9)/mbkin**18 + 
              (8*q_cut**10)/mbkin**20)*rE - ((-1 + mcMS**2/mbkin**2)**2 - 
              (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
             (-4*(-((-1 + mcMS**2/mbkin**2)**2*(-731 + (4718*mcMS**2)/mbkin**2 - 
                   (4699*mcMS**4)/mbkin**4 - (3384*mcMS**6)/mbkin**6 - 
                   (2189*mcMS**8)/mbkin**8 + (1882*mcMS**10)/mbkin**10 + 
                   (83*mcMS**12)/mbkin**12)) + ((-2179 + (9001*mcMS**2)/
                    mbkin**2 - (4197*mcMS**4)/mbkin**4 - (9313*mcMS**6)/mbkin**6 - 
                   (10973*mcMS**8)/mbkin**8 - (5289*mcMS**10)/mbkin**10 + 
                   (5381*mcMS**12)/mbkin**12 + (289*mcMS**14)/mbkin**14)*q_cut)/
                 mbkin**2 - (2*(-694 + (1014*mcMS**2)/mbkin**2 - (1881*mcMS**4)/
                    mbkin**4 - (304*mcMS**6)/mbkin**6 + (612*mcMS**8)/mbkin**8 + 
                   (1866*mcMS**10)/mbkin**10 + (107*mcMS**12)/mbkin**12)*q_cut**2)/
                 mbkin**4 - (2*(-856 - (348*mcMS**2)/mbkin**2 + (691*mcMS**4)/
                    mbkin**4 + (2447*mcMS**6)/mbkin**6 + (2013*mcMS**8)/mbkin**8 + 
                   (181*mcMS**10)/mbkin**10)*q_cut**3)/mbkin**6 + 
                ((-2641 - (1196*mcMS**2)/mbkin**2 + (6054*mcMS**4)/mbkin**4 + 
                   (6404*mcMS**6)/mbkin**6 + (715*mcMS**8)/mbkin**8)*q_cut**4)/
                 mbkin**8 - ((-1103 + (405*mcMS**2)/mbkin**2 + (2415*mcMS**4)/
                    mbkin**4 + (427*mcMS**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
                (4*(-28 + (27*mcMS**2)/mbkin**2 + (19*mcMS**4)/mbkin**4)*q_cut**6)/
                 mbkin**12 + (4*(-1 + mcMS**2/mbkin**2)*q_cut**7)/mbkin**14 + 
                (2*q_cut**8)/mbkin**16)*rG - 4*(-((-1 + mcMS**2/mbkin**2)**2*
                  (351 - (1796*mcMS**2)/mbkin**2 - (5235*mcMS**4)/mbkin**4 - 
                   (5200*mcMS**6)/mbkin**6 - (7135*mcMS**8)/mbkin**8 + 
                   (4404*mcMS**10)/mbkin**10 + (211*mcMS**12)/mbkin**12)) + 
                ((909 - (1687*mcMS**2)/mbkin**2 - (18237*mcMS**4)/mbkin**4 - 
                   (23505*mcMS**6)/mbkin**6 - (8845*mcMS**8)/mbkin**8 - 
                   (19449*mcMS**10)/mbkin**10 + (12477*mcMS**12)/mbkin**12 + 
                   (737*mcMS**14)/mbkin**14)*q_cut)/mbkin**2 - 
                (6*(57 + (175*mcMS**2)/mbkin**2 - (475*mcMS**4)/mbkin**4 + 
                   (270*mcMS**6)/mbkin**6 - (725*mcMS**8)/mbkin**8 + 
                   (1403*mcMS**10)/mbkin**10 + (95*mcMS**12)/mbkin**12)*q_cut**2)/
                 mbkin**4 - (2*(390 + (1352*mcMS**2)/mbkin**2 + (3467*mcMS**4)/
                    mbkin**4 + (6483*mcMS**6)/mbkin**6 + (5015*mcMS**8)/mbkin**8 + 
                   (413*mcMS**10)/mbkin**10)*q_cut**3)/mbkin**6 + 
                ((561 + (4034*mcMS**2)/mbkin**2 + (13860*mcMS**4)/mbkin**4 + 
                   (15570*mcMS**6)/mbkin**6 + (1655*mcMS**8)/mbkin**8)*q_cut**4)/
                 mbkin**8 - (3*(-49 + (463*mcMS**2)/mbkin**2 + (1961*mcMS**4)/
                    mbkin**4 + (313*mcMS**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
                (2*(-69 + (139*mcMS**2)/mbkin**2 + (64*mcMS**4)/mbkin**4)*q_cut**6)/
                 mbkin**12 + (4*(-3 + (5*mcMS**2)/mbkin**2)*q_cut**7)/mbkin**14 + 
                (6*q_cut**8)/mbkin**16)*sB - 1968*sE + (14432*mcMS**2*sE)/mbkin**2 - 
              (26560*mcMS**4*sE)/mbkin**4 + (35616*mcMS**6*sE)/mbkin**6 - 
              (28000*mcMS**8*sE)/mbkin**8 - (15712*mcMS**10*sE)/mbkin**10 + 
              (32448*mcMS**12*sE)/mbkin**12 - (9760*mcMS**14*sE)/mbkin**14 - 
              (496*mcMS**16*sE)/mbkin**16 + (5760*q_cut*sE)/mbkin**2 - 
              (18112*mcMS**2*q_cut*sE)/mbkin**4 - (21504*mcMS**4*q_cut*sE)/mbkin**6 - 
              (21120*mcMS**6*q_cut*sE)/mbkin**8 - (44800*mcMS**8*q_cut*sE)/mbkin**10 - 
              (25920*mcMS**10*q_cut*sE)/mbkin**12 + (31872*mcMS**12*q_cut*sE)/mbkin**
                14 + (1664*mcMS**14*q_cut*sE)/mbkin**16 - (3424*q_cut**2*sE)/mbkin**4 + 
              (2160*mcMS**2*q_cut**2*sE)/mbkin**6 + (1488*mcMS**4*q_cut**2*sE)/mbkin**8 + 
              (12992*mcMS**6*q_cut**2*sE)/mbkin**10 - (28704*mcMS**8*q_cut**2*sE)/mbkin**
                12 - (25776*mcMS**10*q_cut**2*sE)/mbkin**14 - (976*mcMS**12*q_cut**2*
                sE)/mbkin**16 - (4656*q_cut**3*sE)/mbkin**6 - (7456*mcMS**2*q_cut**3*
                sE)/mbkin**8 - (9952*mcMS**4*q_cut**3*sE)/mbkin**10 + 
              (6432*mcMS**6*q_cut**3*sE)/mbkin**12 - (16816*mcMS**8*q_cut**3*sE)/mbkin**
                14 - (2752*mcMS**10*q_cut**3*sE)/mbkin**16 + (6336*q_cut**4*sE)/mbkin**
                8 + (11504*mcMS**2*q_cut**4*sE)/mbkin**10 + (18864*mcMS**4*q_cut**4*sE)/
               mbkin**12 + (32784*mcMS**6*q_cut**4*sE)/mbkin**14 + (4880*mcMS**8*q_cut**4*
                sE)/mbkin**16 - (1936*q_cut**5*sE)/mbkin**10 - (2976*mcMS**2*q_cut**5*
                sE)/mbkin**12 - (12816*mcMS**4*q_cut**5*sE)/mbkin**14 - 
              (2944*mcMS**6*q_cut**5*sE)/mbkin**16 - (144*q_cut**6*sE)/mbkin**12 + 
              (512*mcMS**2*q_cut**6*sE)/mbkin**14 + (656*mcMS**4*q_cut**6*sE)/mbkin**16 - 
              (64*mcMS**2*q_cut**7*sE)/mbkin**16 + (32*q_cut**8*sE)/mbkin**16 - 57*sqB + 
              (338*mcMS**2*sqB)/mbkin**2 + (5756*mcMS**4*sqB)/mbkin**4 - 
              (10002*mcMS**6*sqB)/mbkin**6 + (3590*mcMS**8*sqB)/mbkin**8 - 
              (2098*mcMS**10*sqB)/mbkin**10 + (3012*mcMS**12*sqB)/mbkin**12 - 
              (526*mcMS**14*sqB)/mbkin**14 - (13*mcMS**16*sqB)/mbkin**16 + 
              (75*q_cut*sqB)/mbkin**2 + (713*mcMS**2*q_cut*sqB)/mbkin**4 - 
              (13683*mcMS**4*q_cut*sqB)/mbkin**6 - (18057*mcMS**6*q_cut*sqB)/mbkin**8 - 
              (7867*mcMS**8*q_cut*sqB)/mbkin**10 - (3177*mcMS**10*q_cut*sqB)/mbkin**
                12 + (1635*mcMS**12*q_cut*sqB)/mbkin**14 + (41*mcMS**14*q_cut*sqB)/
               mbkin**16 + (116*q_cut**2*sqB)/mbkin**4 - (1062*mcMS**2*q_cut**2*sqB)/
               mbkin**6 + (2688*mcMS**4*q_cut**2*sqB)/mbkin**8 - (2668*mcMS**6*q_cut**2*
                sqB)/mbkin**10 - (1236*mcMS**8*q_cut**2*sqB)/mbkin**12 - 
              (1182*mcMS**10*q_cut**2*sqB)/mbkin**14 - (16*mcMS**12*q_cut**2*sqB)/mbkin**
                16 - (138*q_cut**3*sqB)/mbkin**6 - (868*mcMS**2*q_cut**3*sqB)/mbkin**8 + 
              (1346*mcMS**4*q_cut**3*sqB)/mbkin**10 - (18*mcMS**6*q_cut**3*sqB)/mbkin**
                12 - (1024*mcMS**8*q_cut**3*sqB)/mbkin**14 - (82*mcMS**10*q_cut**3*sqB)/
               mbkin**16 - (171*q_cut**4*sqB)/mbkin**8 + (1028*mcMS**2*q_cut**4*sqB)/
               mbkin**10 + (1686*mcMS**4*q_cut**4*sqB)/mbkin**12 + (1716*mcMS**6*q_cut**4*
                sqB)/mbkin**14 + (125*mcMS**8*q_cut**4*sqB)/mbkin**16 + 
              (251*q_cut**5*sqB)/mbkin**10 - (201*mcMS**2*q_cut**5*sqB)/mbkin**12 - 
              (663*mcMS**4*q_cut**5*sqB)/mbkin**14 - (67*mcMS**6*q_cut**5*sqB)/mbkin**
                16 - (66*q_cut**6*sqB)/mbkin**12 + (56*mcMS**2*q_cut**6*sqB)/mbkin**14 + 
              (14*mcMS**4*q_cut**6*sqB)/mbkin**16 - (12*q_cut**7*sqB)/mbkin**14 - 
              (4*mcMS**2*q_cut**7*sqB)/mbkin**16 + (2*q_cut**8*sqB)/mbkin**16)))/
          mbkin**6) - 6*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
           2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
        (32*(-((-1 + mcMS**2/mbkin**2)**4*(-11 + (131*mcMS**2)/mbkin**2 + 
              (343*mcMS**4)/mbkin**4 - (4901*mcMS**6)/mbkin**6 - 
              (3931*mcMS**8)/mbkin**8 + (4513*mcMS**10)/mbkin**10 + 
              (4201*mcMS**12)/mbkin**12 - (3931*mcMS**14)/mbkin**14 - 
              (842*mcMS**16)/mbkin**16 + (108*mcMS**18)/mbkin**18)) + 
           ((-1 + mcMS**2/mbkin**2)**2*(-69 + (686*mcMS**2)/mbkin**2 + 
              (1998*mcMS**4)/mbkin**4 - (17384*mcMS**6)/mbkin**6 - 
              (28414*mcMS**8)/mbkin**8 - (7896*mcMS**10)/mbkin**10 + 
              (25876*mcMS**12)/mbkin**12 + (14624*mcMS**14)/mbkin**14 - 
              (19869*mcMS**16)/mbkin**16 - (4814*mcMS**18)/mbkin**18 + 
              (702*mcMS**20)/mbkin**20)*q_cut)/mbkin**2 - 
           (2*(-80 + (683*mcMS**2)/mbkin**2 + (1558*mcMS**4)/mbkin**4 - 
              (10930*mcMS**6)/mbkin**6 - (12701*mcMS**8)/mbkin**8 - 
              (15107*mcMS**10)/mbkin**10 - (14557*mcMS**12)/mbkin**12 + 
              (20503*mcMS**14)/mbkin**14 + (17401*mcMS**16)/mbkin**16 - 
              (16868*mcMS**18)/mbkin**18 - (5317*mcMS**20)/mbkin**20 + 
              (855*mcMS**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
           (2*(-60 + (337*mcMS**2)/mbkin**2 + (688*mcMS**4)/mbkin**4 - 
              (2600*mcMS**6)/mbkin**6 - (1620*mcMS**8)/mbkin**8 + 
              (3788*mcMS**10)/mbkin**10 + (14332*mcMS**12)/mbkin**12 - 
              (9532*mcMS**14)/mbkin**14 - (15876*mcMS**16)/mbkin**16 - 
              (2417*mcMS**18)/mbkin**18 + (720*mcMS**20)/mbkin**20)*q_cut**3)/
            mbkin**6 + (2*(-75 + (292*mcMS**2)/mbkin**2 + (3006*mcMS**4)/mbkin**
                4 + (1155*mcMS**6)/mbkin**6 - (7341*mcMS**8)/mbkin**8 - 
              (10120*mcMS**10)/mbkin**10 - (1458*mcMS**12)/mbkin**12 - 
              (8299*mcMS**14)/mbkin**14 - (3536*mcMS**16)/mbkin**16 + 
              (720*mcMS**18)/mbkin**18)*q_cut**4)/mbkin**8 + 
           ((378 - (886*mcMS**2)/mbkin**2 - (8966*mcMS**4)/mbkin**4 - 
              (4238*mcMS**6)/mbkin**6 + (22414*mcMS**8)/mbkin**8 + 
              (43870*mcMS**10)/mbkin**10 + (48834*mcMS**12)/mbkin**12 + 
              (10862*mcMS**14)/mbkin**14 - (4284*mcMS**16)/mbkin**16)*q_cut**5)/
            mbkin**10 + (2*(-126 + (313*mcMS**2)/mbkin**2 + (2593*mcMS**4)/mbkin**
                4 - (837*mcMS**6)/mbkin**6 - (12889*mcMS**8)/mbkin**8 - 
              (16866*mcMS**10)/mbkin**10 - (3746*mcMS**12)/mbkin**12 + 
              (1638*mcMS**14)/mbkin**14)*q_cut**6)/mbkin**12 + 
           (2*(-30 - (317*mcMS**2)/mbkin**2 - (672*mcMS**4)/mbkin**4 + 
              (2136*mcMS**6)/mbkin**6 + (6062*mcMS**8)/mbkin**8 + 
              (2953*mcMS**10)/mbkin**10 + (180*mcMS**12)/mbkin**12)*q_cut**7)/
            mbkin**14 - ((-195 - (551*mcMS**2)/mbkin**2 + (491*mcMS**4)/mbkin**4 + 
              (3933*mcMS**6)/mbkin**6 + (5026*mcMS**8)/mbkin**8 + 
              (2340*mcMS**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
           ((-125 - (234*mcMS**2)/mbkin**2 + (913*mcMS**4)/mbkin**4 + 
              (2412*mcMS**6)/mbkin**6 + (1710*mcMS**8)/mbkin**8)*q_cut**9)/mbkin**18 + 
           ((36 + (36*mcMS**2)/mbkin**2 - (446*mcMS**4)/mbkin**4 - 
              (558*mcMS**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
           ((-4 + (72*mcMS**4)/mbkin**4)*q_cut**11)/mbkin**22)*rE + 
         ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
           q_cut**2/mbkin**4)*((180*mcMS**2*muG**2)/mbkin**2 - (1452*mcMS**4*muG**2)/
            mbkin**4 + (4512*mcMS**6*muG**2)/mbkin**6 + (1584*mcMS**8*muG**2)/
            mbkin**8 - (38424*mcMS**10*muG**2)/mbkin**10 + (160680*mcMS**12*muG**2)/
            mbkin**12 - (250128*mcMS**14*muG**2)/mbkin**14 + 
           (135888*mcMS**16*muG**2)/mbkin**16 - (924*mcMS**18*muG**2)/mbkin**18 - 
           (14076*mcMS**20*muG**2)/mbkin**20 + (2160*mcMS**22*muG**2)/mbkin**22 + 
           (60*mcMS**2*muG*mupi)/mbkin**2 - (996*mcMS**4*muG*mupi)/mbkin**4 + 
           (3360*mcMS**6*muG*mupi)/mbkin**6 + (1488*mcMS**8*muG*mupi)/mbkin**8 - 
           (25224*mcMS**10*muG*mupi)/mbkin**10 - (24648*mcMS**12*muG*mupi)/
            mbkin**12 + (124176*mcMS**14*muG*mupi)/mbkin**14 - 
           (90960*mcMS**16*muG*mupi)/mbkin**16 + (8652*mcMS**18*muG*mupi)/
            mbkin**18 + (4524*mcMS**20*muG*mupi)/mbkin**20 - 
           (432*mcMS**22*muG*mupi)/mbkin**22 - (240*mcMS**2*muG**2*q_cut)/mbkin**4 + 
           (1032*mcMS**4*muG**2*q_cut)/mbkin**6 - (1344*mcMS**6*muG**2*q_cut)/mbkin**8 + 
           (16128*mcMS**8*muG**2*q_cut)/mbkin**10 - (66624*mcMS**10*muG**2*q_cut)/
            mbkin**12 - (32784*mcMS**12*muG**2*q_cut)/mbkin**14 - 
           (183936*mcMS**14*muG**2*q_cut)/mbkin**16 + (23040*mcMS**16*muG**2*q_cut)/
            mbkin**18 + (47088*mcMS**18*muG**2*q_cut)/mbkin**20 - 
           (9720*mcMS**20*muG**2*q_cut)/mbkin**22 - (240*mcMS**2*muG*mupi*q_cut)/
            mbkin**4 + (3288*mcMS**4*muG*mupi*q_cut)/mbkin**6 - 
           (7680*mcMS**6*muG*mupi*q_cut)/mbkin**8 - (24000*mcMS**8*muG*mupi*q_cut)/
            mbkin**10 + (73920*mcMS**10*muG*mupi*q_cut)/mbkin**12 + 
           (84624*mcMS**12*muG*mupi*q_cut)/mbkin**14 + (129216*mcMS**14*muG*mupi*
             q_cut)/mbkin**16 - (38976*mcMS**16*muG*mupi*q_cut)/mbkin**18 - 
           (14736*mcMS**18*muG*mupi*q_cut)/mbkin**20 + (1944*mcMS**20*muG*mupi*q_cut)/
            mbkin**22 - (240*mcMS**2*muG**2*q_cut**2)/mbkin**6 + 
           (3000*mcMS**4*muG**2*q_cut**2)/mbkin**8 - (5208*mcMS**6*muG**2*q_cut**2)/
            mbkin**10 - (20472*mcMS**8*muG**2*q_cut**2)/mbkin**12 - 
           (9768*mcMS**10*muG**2*q_cut**2)/mbkin**14 + (34152*mcMS**12*muG**2*q_cut**2)/
            mbkin**16 + (22008*mcMS**14*muG**2*q_cut**2)/mbkin**18 - 
           (53352*mcMS**16*muG**2*q_cut**2)/mbkin**20 + (12600*mcMS**18*muG**2*q_cut**2)/
            mbkin**22 + (240*mcMS**2*muG*mupi*q_cut**2)/mbkin**6 - 
           (2520*mcMS**4*muG*mupi*q_cut**2)/mbkin**8 + (3384*mcMS**6*muG*mupi*q_cut**2)/
            mbkin**10 + (16152*mcMS**8*muG*mupi*q_cut**2)/mbkin**12 + 
           (15048*mcMS**10*muG*mupi*q_cut**2)/mbkin**14 - 
           (46152*mcMS**12*muG*mupi*q_cut**2)/mbkin**16 + 
           (19752*mcMS**14*muG*mupi*q_cut**2)/mbkin**18 + 
           (13896*mcMS**16*muG*mupi*q_cut**2)/mbkin**20 - 
           (2520*mcMS**18*muG*mupi*q_cut**2)/mbkin**22 - (240*mcMS**2*muG**2*q_cut**3)/
            mbkin**8 - (1080*mcMS**4*muG**2*q_cut**3)/mbkin**10 + 
           (4224*mcMS**6*muG**2*q_cut**3)/mbkin**12 - (96840*mcMS**8*muG**2*q_cut**3)/
            mbkin**14 - (119760*mcMS**10*muG**2*q_cut**3)/mbkin**16 - 
           (151272*mcMS**12*muG**2*q_cut**3)/mbkin**18 + (20160*mcMS**14*muG**2*q_cut**3)/
            mbkin**20 + (6120*mcMS**16*muG**2*q_cut**3)/mbkin**22 + 
           (240*mcMS**2*muG*mupi*q_cut**3)/mbkin**8 - (3624*mcMS**4*muG*mupi*q_cut**3)/
            mbkin**10 + (7104*mcMS**6*muG*mupi*q_cut**3)/mbkin**12 + 
           (66216*mcMS**8*muG*mupi*q_cut**3)/mbkin**14 + 
           (105168*mcMS**10*muG*mupi*q_cut**3)/mbkin**16 + 
           (71112*mcMS**12*muG*mupi*q_cut**3)/mbkin**18 + 
           (1536*mcMS**14*muG*mupi*q_cut**3)/mbkin**20 - 
           (1224*mcMS**16*muG*mupi*q_cut**3)/mbkin**22 + (1560*mcMS**2*muG**2*q_cut**4)/
            mbkin**10 - (8400*mcMS**4*muG**2*q_cut**4)/mbkin**12 - 
           (7416*mcMS**6*muG**2*q_cut**4)/mbkin**14 + (114336*mcMS**8*muG**2*q_cut**4)/
            mbkin**16 + (167928*mcMS**10*muG**2*q_cut**4)/mbkin**18 + 
           (12960*mcMS**12*muG**2*q_cut**4)/mbkin**20 - (29160*mcMS**14*muG**2*q_cut**4)/
            mbkin**22 - (600*mcMS**2*muG*mupi*q_cut**4)/mbkin**10 + 
           (8112*mcMS**4*muG*mupi*q_cut**4)/mbkin**12 - (12072*mcMS**6*muG*mupi*
             q_cut**4)/mbkin**14 - (95712*mcMS**8*muG*mupi*q_cut**4)/mbkin**16 - 
           (93048*mcMS**10*muG*mupi*q_cut**4)/mbkin**18 - 
           (10560*mcMS**12*muG*mupi*q_cut**4)/mbkin**20 + 
           (5832*mcMS**14*muG*mupi*q_cut**4)/mbkin**22 - (720*mcMS**2*muG**2*q_cut**5)/
            mbkin**12 + (7032*mcMS**4*muG**2*q_cut**5)/mbkin**14 - 
           (15648*mcMS**6*muG**2*q_cut**5)/mbkin**16 - (98928*mcMS**8*muG**2*q_cut**5)/
            mbkin**18 - (46224*mcMS**10*muG**2*q_cut**5)/mbkin**20 + 
           (21240*mcMS**12*muG**2*q_cut**5)/mbkin**22 + (240*mcMS**2*muG*mupi*q_cut**5)/
            mbkin**12 - (4824*mcMS**4*muG*mupi*q_cut**5)/mbkin**14 + 
           (11808*mcMS**6*muG*mupi*q_cut**5)/mbkin**16 + 
           (40560*mcMS**8*muG*mupi*q_cut**5)/mbkin**18 + 
           (11376*mcMS**10*muG*mupi*q_cut**5)/mbkin**20 - 
           (4248*mcMS**12*muG*mupi*q_cut**5)/mbkin**22 - (1200*mcMS**2*muG**2*q_cut**6)/
            mbkin**14 + (10728*mcMS**4*muG**2*q_cut**6)/mbkin**16 + 
           (57048*mcMS**6*muG**2*q_cut**6)/mbkin**18 + (64872*mcMS**8*muG**2*q_cut**6)/
            mbkin**20 + (6120*mcMS**10*muG**2*q_cut**6)/mbkin**22 + 
           (240*mcMS**2*muG*mupi*q_cut**6)/mbkin**14 - (1608*mcMS**4*muG*mupi*q_cut**6)/
            mbkin**16 - (11832*mcMS**6*muG*mupi*q_cut**6)/mbkin**18 - 
           (11784*mcMS**8*muG*mupi*q_cut**6)/mbkin**20 - 
           (1224*mcMS**10*muG*mupi*q_cut**6)/mbkin**22 + (1200*mcMS**2*muG**2*q_cut**7)/
            mbkin**16 - (18600*mcMS**4*muG**2*q_cut**7)/mbkin**18 - 
           (40608*mcMS**6*muG**2*q_cut**7)/mbkin**20 - (16200*mcMS**8*muG**2*q_cut**7)/
            mbkin**22 - (240*mcMS**2*muG*mupi*q_cut**7)/mbkin**16 + 
           (3720*mcMS**4*muG*mupi*q_cut**7)/mbkin**18 + (7584*mcMS**6*muG*mupi*q_cut**7)/
            mbkin**20 + (3240*mcMS**8*muG*mupi*q_cut**7)/mbkin**22 - 
           (300*mcMS**2*muG**2*q_cut**8)/mbkin**18 + (9180*mcMS**4*muG**2*q_cut**8)/
            mbkin**20 + (8280*mcMS**6*muG**2*q_cut**8)/mbkin**22 + 
           (60*mcMS**2*muG*mupi*q_cut**8)/mbkin**18 - (1836*mcMS**4*muG*mupi*q_cut**8)/
            mbkin**20 - (1656*mcMS**6*muG*mupi*q_cut**8)/mbkin**22 - 
           (1440*mcMS**4*muG**2*q_cut**9)/mbkin**22 + (288*mcMS**4*muG*mupi*q_cut**9)/
            mbkin**22 - 24*mcMS**2*muG*((-1 + mcMS**2/mbkin**2)**2*
              (-5 + (73*mcMS**2)/mbkin**2 - (129*mcMS**4)/mbkin**4 - (455*mcMS**6)/
                mbkin**6 + (1321*mcMS**8)/mbkin**8 + (5151*mcMS**10)/mbkin**10 - 
               (1367*mcMS**12)/mbkin**12 - (305*mcMS**14)/mbkin**14 + 
               (36*mcMS**16)/mbkin**16) - (2*(-10 + (137*mcMS**2)/mbkin**2 - 
                (320*mcMS**4)/mbkin**4 - (1000*mcMS**6)/mbkin**6 + (3080*mcMS**8)/
                 mbkin**8 + (3526*mcMS**10)/mbkin**10 + (5384*mcMS**12)/
                 mbkin**12 - (1624*mcMS**14)/mbkin**14 - (614*mcMS**16)/
                 mbkin**16 + (81*mcMS**18)/mbkin**18)*q_cut)/mbkin**2 + 
             (2*(-10 + (105*mcMS**2)/mbkin**2 - (141*mcMS**4)/mbkin**4 - 
                (673*mcMS**6)/mbkin**6 - (627*mcMS**8)/mbkin**8 + (1923*mcMS**10)/
                 mbkin**10 - (823*mcMS**12)/mbkin**12 - (579*mcMS**14)/mbkin**14 + 
                (105*mcMS**16)/mbkin**16)*q_cut**2)/mbkin**4 + 
             (2*(-10 + (151*mcMS**2)/mbkin**2 - (296*mcMS**4)/mbkin**4 - 
                (2759*mcMS**6)/mbkin**6 - (4382*mcMS**8)/mbkin**8 - 
                (2963*mcMS**10)/mbkin**10 - (64*mcMS**12)/mbkin**12 + 
                (51*mcMS**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
             ((50 - (676*mcMS**2)/mbkin**2 + (1006*mcMS**4)/mbkin**4 + 
                (7976*mcMS**6)/mbkin**6 + (7754*mcMS**8)/mbkin**8 + (880*mcMS**10)/
                 mbkin**10 - (486*mcMS**12)/mbkin**12)*q_cut**4)/mbkin**8 + 
             ((-20 + (402*mcMS**2)/mbkin**2 - (984*mcMS**4)/mbkin**4 - 
                (3380*mcMS**6)/mbkin**6 - (948*mcMS**8)/mbkin**8 + (354*mcMS**10)/
                 mbkin**10)*q_cut**5)/mbkin**10 + (2*(-10 + (67*mcMS**2)/mbkin**2 + 
                (493*mcMS**4)/mbkin**4 + (491*mcMS**6)/mbkin**6 + (51*mcMS**8)/
                 mbkin**8)*q_cut**6)/mbkin**12 - (2*(-10 + (155*mcMS**2)/mbkin**2 + 
                (316*mcMS**4)/mbkin**4 + (135*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
             ((-5 + (153*mcMS**2)/mbkin**2 + (138*mcMS**4)/mbkin**4)*q_cut**8)/
              mbkin**16 - (24*mcMS**2*q_cut**9)/mbkin**20) - 
           4*(-((-1 + mcMS**2/mbkin**2)**2*(-43 + (438*mcMS**2)/mbkin**2 + 
                (1390*mcMS**4)/mbkin**4 - (15994*mcMS**6)/mbkin**6 + 
                (14652*mcMS**8)/mbkin**8 + (24394*mcMS**10)/mbkin**10 + 
                (8738*mcMS**12)/mbkin**12 - (6774*mcMS**14)/mbkin**14 - 
                (1025*mcMS**16)/mbkin**16 + (144*mcMS**18)/mbkin**18)) + 
             (4*(-49 + (417*mcMS**2)/mbkin**2 + (935*mcMS**4)/mbkin**4 - 
                (7247*mcMS**6)/mbkin**6 + (2361*mcMS**8)/mbkin**8 + 
                (13385*mcMS**10)/mbkin**10 + (16711*mcMS**12)/mbkin**12 + 
                (4863*mcMS**14)/mbkin**14 - (4600*mcMS**16)/mbkin**16 - 
                (1018*mcMS**18)/mbkin**18 + (162*mcMS**20)/mbkin**20)*q_cut)/
              mbkin**2 - (4*(-65 + (360*mcMS**2)/mbkin**2 + (625*mcMS**4)/
                 mbkin**4 - (1220*mcMS**6)/mbkin**6 + (2577*mcMS**8)/mbkin**8 + 
                (1258*mcMS**10)/mbkin**10 - (1589*mcMS**12)/mbkin**12 - 
                (3504*mcMS**14)/mbkin**14 - (812*mcMS**16)/mbkin**16 + 
                (210*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
             (4*(-29 + (101*mcMS**2)/mbkin**2 + (1989*mcMS**4)/mbkin**4 + 
                (173*mcMS**6)/mbkin**6 - (2343*mcMS**8)/mbkin**8 - (5835*mcMS**10)/
                 mbkin**10 - (4455*mcMS**12)/mbkin**12 - (359*mcMS**14)/
                 mbkin**14 + (102*mcMS**16)/mbkin**16)*q_cut**3)/mbkin**6 + 
             (2*(-299 + (600*mcMS**2)/mbkin**2 + (5761*mcMS**4)/mbkin**4 + 
                (944*mcMS**6)/mbkin**6 - (14463*mcMS**8)/mbkin**8 - 
                (14012*mcMS**10)/mbkin**10 - (1655*mcMS**12)/mbkin**12 + 
                (972*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 - 
             (4*(-113 + (249*mcMS**2)/mbkin**2 + (1309*mcMS**4)/mbkin**4 - 
                (833*mcMS**6)/mbkin**6 - (3006*mcMS**8)/mbkin**8 - (728*mcMS**10)/
                 mbkin**10 + (354*mcMS**12)/mbkin**12)*q_cut**5)/mbkin**10 - 
             (4*(-29 - (268*mcMS**2)/mbkin**2 - (177*mcMS**4)/mbkin**4 + 
                (714*mcMS**6)/mbkin**6 + (802*mcMS**8)/mbkin**8 + (102*mcMS**10)/
                 mbkin**10)*q_cut**6)/mbkin**12 + (4*(-85 - (195*mcMS**2)/mbkin**2 + 
                (167*mcMS**4)/mbkin**4 + (563*mcMS**6)/mbkin**6 + (270*mcMS**8)/
                 mbkin**8)*q_cut**7)/mbkin**14 + ((179 + (204*mcMS**2)/mbkin**2 - 
                (571*mcMS**4)/mbkin**4 - (552*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 - 
             (32*(mbkin**4 - 3*mcMS**4)*q_cut**9)/mbkin**22)*rG + 
           8*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*(21 - (236*mcMS**2)/mbkin**2 + 
                (280*mcMS**4)/mbkin**4 + (5400*mcMS**6)/mbkin**6 - (43634*mcMS**8)/
                 mbkin**8 - (31160*mcMS**10)/mbkin**10 + (14712*mcMS**12)/
                 mbkin**12 - (4424*mcMS**14)/mbkin**14 - (1619*mcMS**16)/
                 mbkin**16 + (180*mcMS**18)/mbkin**18)) + 
             (2*(51 - (508*mcMS**2)/mbkin**2 + (671*mcMS**4)/mbkin**4 + 
                (7066*mcMS**6)/mbkin**6 - (40432*mcMS**8)/mbkin**8 - 
                (66010*mcMS**10)/mbkin**10 - (36356*mcMS**12)/mbkin**12 + 
                (22310*mcMS**14)/mbkin**14 - (4915*mcMS**16)/mbkin**16 - 
                (3242*mcMS**18)/mbkin**18 + (405*mcMS**20)/mbkin**20)*q_cut)/
              mbkin**2 - (2*(75 - (535*mcMS**2)/mbkin**2 + (234*mcMS**4)/
                 mbkin**4 + (3612*mcMS**6)/mbkin**6 - (2692*mcMS**8)/mbkin**8 + 
                (6882*mcMS**10)/mbkin**10 + (9870*mcMS**12)/mbkin**12 - 
                (5092*mcMS**14)/mbkin**14 - (2799*mcMS**16)/mbkin**16 + 
                (525*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
             (2*(-21 + (124*mcMS**2)/mbkin**2 + (650*mcMS**4)/mbkin**4 - 
                (4312*mcMS**6)/mbkin**6 + (1928*mcMS**8)/mbkin**8 + (728*mcMS**10)/
                 mbkin**10 + (7154*mcMS**12)/mbkin**12 + (1396*mcMS**14)/
                 mbkin**14 - (255*mcMS**16)/mbkin**16)*q_cut**3)/mbkin**6 + 
             (2*(168 - (595*mcMS**2)/mbkin**2 - (1396*mcMS**4)/mbkin**4 + 
                (3013*mcMS**6)/mbkin**6 - (1930*mcMS**8)/mbkin**8 - 
                (13117*mcMS**10)/mbkin**10 - (3590*mcMS**12)/mbkin**12 + 
                (1215*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
             (2*(-147 + (436*mcMS**2)/mbkin**2 + (723*mcMS**4)/mbkin**4 + 
                (726*mcMS**6)/mbkin**6 + (6157*mcMS**8)/mbkin**8 + (2478*mcMS**10)/
                 mbkin**10 - (885*mcMS**12)/mbkin**12)*q_cut**5)/mbkin**10 - 
             (2*(21 + (227*mcMS**2)/mbkin**2 + (112*mcMS**4)/mbkin**4 + 
                (1090*mcMS**6)/mbkin**6 + (1463*mcMS**8)/mbkin**8 + (255*mcMS**10)/
                 mbkin**10)*q_cut**6)/mbkin**12 + (2*(105 + (140*mcMS**2)/mbkin**2 + 
                (248*mcMS**4)/mbkin**4 + (856*mcMS**6)/mbkin**6 + (675*mcMS**8)/
                 mbkin**8)*q_cut**7)/mbkin**14 - ((123 + (88*mcMS**2)/mbkin**2 + 
                (447*mcMS**4)/mbkin**4 + (690*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
             (24*(mbkin**4 + 5*mcMS**4)*q_cut**9)/mbkin**22)*rhoD + 60*sB - 
           (520*mcMS**2*sB)/mbkin**2 - (8228*mcMS**4*sB)/mbkin**4 + 
           (63424*mcMS**6*sB)/mbkin**6 + (35656*mcMS**8*sB)/mbkin**8 - 
           (82832*mcMS**10*sB)/mbkin**10 - (57080*mcMS**12*sB)/mbkin**12 - 
           (98912*mcMS**14*sB)/mbkin**14 + (191356*mcMS**16*sB)/mbkin**16 - 
           (30056*mcMS**18*sB)/mbkin**18 - (14308*mcMS**20*sB)/mbkin**20 + 
           (1440*mcMS**22*sB)/mbkin**22 - (240*q_cut*sB)/mbkin**2 + 
           (880*mcMS**2*q_cut*sB)/mbkin**4 + (28672*mcMS**4*q_cut*sB)/mbkin**6 - 
           (57136*mcMS**6*q_cut*sB)/mbkin**8 - (436016*mcMS**8*q_cut*sB)/mbkin**10 - 
           (582320*mcMS**10*q_cut*sB)/mbkin**12 - (280240*mcMS**12*q_cut*sB)/
            mbkin**14 - (229712*mcMS**14*q_cut*sB)/mbkin**16 + 
           (134176*mcMS**16*q_cut*sB)/mbkin**18 + (46016*mcMS**18*q_cut*sB)/mbkin**20 - 
           (6480*mcMS**20*q_cut*sB)/mbkin**22 + (240*q_cut**2*sB)/mbkin**4 + 
           (320*mcMS**2*q_cut**2*sB)/mbkin**6 - (16896*mcMS**4*q_cut**2*sB)/mbkin**8 - 
           (22800*mcMS**6*q_cut**2*sB)/mbkin**10 + (41120*mcMS**8*q_cut**2*sB)/
            mbkin**12 - (18960*mcMS**10*q_cut**2*sB)/mbkin**14 + 
           (28800*mcMS**12*q_cut**2*sB)/mbkin**16 - (95344*mcMS**14*q_cut**2*sB)/
            mbkin**18 - (40080*mcMS**16*q_cut**2*sB)/mbkin**20 + 
           (8400*mcMS**18*q_cut**2*sB)/mbkin**22 + (240*q_cut**3*sB)/mbkin**6 + 
           (80*mcMS**2*q_cut**3*sB)/mbkin**8 - (23840*mcMS**4*q_cut**3*sB)/mbkin**10 - 
           (80912*mcMS**6*q_cut**3*sB)/mbkin**12 - (158912*mcMS**8*q_cut**3*sB)/
            mbkin**14 - (228848*mcMS**10*q_cut**3*sB)/mbkin**16 - 
           (174080*mcMS**12*q_cut**3*sB)/mbkin**18 - (13648*mcMS**14*q_cut**3*sB)/
            mbkin**20 + (4080*mcMS**16*q_cut**3*sB)/mbkin**22 - (600*q_cut**4*sB)/
            mbkin**8 - (2480*mcMS**2*q_cut**4*sB)/mbkin**10 + (20488*mcMS**4*q_cut**4*sB)/
            mbkin**12 + (113552*mcMS**6*q_cut**4*sB)/mbkin**14 + 
           (253960*mcMS**8*q_cut**4*sB)/mbkin**16 + (257200*mcMS**10*q_cut**4*sB)/
            mbkin**18 + (41000*mcMS**12*q_cut**4*sB)/mbkin**20 - 
           (19440*mcMS**14*q_cut**4*sB)/mbkin**22 + (240*q_cut**5*sB)/mbkin**10 + 
           (1040*mcMS**2*q_cut**5*sB)/mbkin**12 + (1056*mcMS**4*q_cut**5*sB)/mbkin**14 - 
           (39792*mcMS**6*q_cut**5*sB)/mbkin**16 - (110944*mcMS**8*q_cut**5*sB)/
            mbkin**18 - (35232*mcMS**10*q_cut**5*sB)/mbkin**20 + 
           (14160*mcMS**12*q_cut**5*sB)/mbkin**22 + (240*q_cut**6*sB)/mbkin**12 + 
           (2240*mcMS**2*q_cut**6*sB)/mbkin**14 + (5056*mcMS**4*q_cut**6*sB)/mbkin**16 + 
           (29104*mcMS**6*q_cut**6*sB)/mbkin**18 + (31664*mcMS**8*q_cut**6*sB)/
            mbkin**20 + (4080*mcMS**10*q_cut**6*sB)/mbkin**22 - (240*q_cut**7*sB)/
            mbkin**14 - (2000*mcMS**2*q_cut**7*sB)/mbkin**16 - 
           (10496*mcMS**4*q_cut**7*sB)/mbkin**18 - (20560*mcMS**6*q_cut**7*sB)/
            mbkin**20 - (10800*mcMS**8*q_cut**7*sB)/mbkin**22 + 
           (60*q_cut**8*sB)/mbkin**16 + (440*mcMS**2*q_cut**8*sB)/mbkin**18 + 
           (5148*mcMS**4*q_cut**8*sB)/mbkin**20 + (5520*mcMS**6*q_cut**8*sB)/mbkin**22 - 
           (960*mcMS**4*q_cut**9*sB)/mbkin**22 - 96*sE + (928*mcMS**2*sE)/mbkin**2 + 
           (5792*mcMS**4*sE)/mbkin**4 - (54880*mcMS**6*sE)/mbkin**6 + 
           (55616*mcMS**8*sE)/mbkin**8 - (48448*mcMS**10*sE)/mbkin**10 + 
           (78656*mcMS**12*sE)/mbkin**12 + (33728*mcMS**14*sE)/mbkin**14 - 
           (99040*mcMS**16*sE)/mbkin**16 + (20384*mcMS**18*sE)/mbkin**18 + 
           (8224*mcMS**20*sE)/mbkin**20 - (864*mcMS**22*sE)/mbkin**22 + 
           (432*q_cut*sE)/mbkin**2 - (2656*mcMS**2*q_cut*sE)/mbkin**4 - 
           (22480*mcMS**4*q_cut*sE)/mbkin**6 + (67264*mcMS**6*q_cut*sE)/mbkin**8 + 
           (180512*mcMS**8*q_cut*sE)/mbkin**10 + (197120*mcMS**10*q_cut*sE)/mbkin**12 + 
           (166816*mcMS**12*q_cut*sE)/mbkin**14 + (84800*mcMS**14*q_cut*sE)/mbkin**16 - 
           (97744*mcMS**16*q_cut*sE)/mbkin**18 - (24992*mcMS**18*q_cut*sE)/mbkin**20 + 
           (3888*mcMS**20*q_cut*sE)/mbkin**22 - (560*q_cut**2*sE)/mbkin**4 + 
           (2000*mcMS**2*q_cut**2*sE)/mbkin**6 + (14336*mcMS**4*q_cut**2*sE)/mbkin**8 + 
           (1792*mcMS**6*q_cut**2*sE)/mbkin**10 - (20512*mcMS**8*q_cut**2*sE)/mbkin**12 - 
           (23968*mcMS**10*q_cut**2*sE)/mbkin**14 + (98176*mcMS**12*q_cut**2*sE)/
            mbkin**16 + (101504*mcMS**14*q_cut**2*sE)/mbkin**18 + 
           (16592*mcMS**16*q_cut**2*sE)/mbkin**20 - (5040*mcMS**18*q_cut**2*sE)/
            mbkin**22 - (272*q_cut**3*sE)/mbkin**6 - (32*mcMS**2*q_cut**3*sE)/mbkin**8 + 
           (25472*mcMS**4*q_cut**3*sE)/mbkin**10 + (46368*mcMS**6*q_cut**3*sE)/
            mbkin**12 + (54944*mcMS**8*q_cut**3*sE)/mbkin**14 - 
           (1120*mcMS**10*q_cut**3*sE)/mbkin**16 + (51072*mcMS**12*q_cut**3*sE)/
            mbkin**18 + (18016*mcMS**14*q_cut**3*sE)/mbkin**20 - 
           (2448*mcMS**16*q_cut**3*sE)/mbkin**22 + (1296*q_cut**4*sE)/mbkin**8 + 
           (80*mcMS**2*q_cut**4*sE)/mbkin**10 - (32080*mcMS**4*q_cut**4*sE)/mbkin**12 - 
           (69200*mcMS**6*q_cut**4*sE)/mbkin**14 - (89296*mcMS**8*q_cut**4*sE)/
            mbkin**16 - (119440*mcMS**10*q_cut**4*sE)/mbkin**18 - 
           (28400*mcMS**12*q_cut**4*sE)/mbkin**20 + (11664*mcMS**14*q_cut**4*sE)/
            mbkin**22 - (944*q_cut**5*sE)/mbkin**10 + (992*mcMS**2*q_cut**5*sE)/
            mbkin**12 + (9584*mcMS**4*q_cut**5*sE)/mbkin**14 + 
           (19072*mcMS**6*q_cut**5*sE)/mbkin**16 + (49520*mcMS**8*q_cut**5*sE)/
            mbkin**18 + (15776*mcMS**10*q_cut**5*sE)/mbkin**20 - 
           (8496*mcMS**12*q_cut**5*sE)/mbkin**22 - (272*q_cut**6*sE)/mbkin**12 - 
           (3344*mcMS**2*q_cut**6*sE)/mbkin**14 - (3424*mcMS**4*q_cut**6*sE)/mbkin**16 - 
           (9312*mcMS**6*q_cut**6*sE)/mbkin**18 - (11408*mcMS**8*q_cut**6*sE)/mbkin**20 - 
           (2448*mcMS**10*q_cut**6*sE)/mbkin**22 + (720*q_cut**7*sE)/mbkin**14 + 
           (2720*mcMS**2*q_cut**7*sE)/mbkin**16 + (4640*mcMS**4*q_cut**7*sE)/mbkin**18 + 
           (8608*mcMS**6*q_cut**7*sE)/mbkin**20 + (6480*mcMS**8*q_cut**7*sE)/mbkin**22 - 
           (368*q_cut**8*sE)/mbkin**16 - (688*mcMS**2*q_cut**8*sE)/mbkin**18 - 
           (2416*mcMS**4*q_cut**8*sE)/mbkin**20 - (3312*mcMS**6*q_cut**8*sE)/mbkin**22 + 
           (64*q_cut**9*sE)/mbkin**18 + (576*mcMS**4*q_cut**9*sE)/mbkin**22 - 3*sqB + 
           (34*mcMS**2*sqB)/mbkin**2 + (893*mcMS**4*sqB)/mbkin**4 - 
           (6652*mcMS**6*sqB)/mbkin**6 - (19834*mcMS**8*sqB)/mbkin**8 + 
           (34484*mcMS**10*sqB)/mbkin**10 + (5222*mcMS**12*sqB)/mbkin**12 - 
           (4768*mcMS**14*sqB)/mbkin**14 - (11203*mcMS**16*sqB)/mbkin**16 + 
           (1514*mcMS**18*sqB)/mbkin**18 + (349*mcMS**20*sqB)/mbkin**20 - 
           (36*mcMS**22*sqB)/mbkin**22 + (6*q_cut*sqB)/mbkin**2 + 
           (32*mcMS**2*q_cut*sqB)/mbkin**4 - (3202*mcMS**4*q_cut*sqB)/mbkin**6 + 
           (4780*mcMS**6*q_cut*sqB)/mbkin**8 + (72536*mcMS**8*q_cut*sqB)/mbkin**10 + 
           (107300*mcMS**10*q_cut*sqB)/mbkin**12 + (57856*mcMS**12*q_cut*sqB)/
            mbkin**14 + (9620*mcMS**14*q_cut*sqB)/mbkin**16 - (6142*mcMS**16*q_cut*sqB)/
            mbkin**18 - (1028*mcMS**18*q_cut*sqB)/mbkin**20 + (162*mcMS**20*q_cut*sqB)/
            mbkin**22 + (10*q_cut**2*sqB)/mbkin**4 - (250*mcMS**2*q_cut**2*sqB)/
            mbkin**6 + (1892*mcMS**4*q_cut**2*sqB)/mbkin**8 + (3880*mcMS**6*q_cut**2*sqB)/
            mbkin**10 - (7312*mcMS**8*q_cut**2*sqB)/mbkin**12 + 
           (7772*mcMS**10*q_cut**2*sqB)/mbkin**14 + (9004*mcMS**12*q_cut**2*sqB)/
            mbkin**16 + (4712*mcMS**14*q_cut**2*sqB)/mbkin**18 + 
           (662*mcMS**16*q_cut**2*sqB)/mbkin**20 - (210*mcMS**18*q_cut**2*sqB)/
            mbkin**22 - (26*q_cut**3*sqB)/mbkin**6 + (64*mcMS**2*q_cut**3*sqB)/mbkin**8 + 
           (2588*mcMS**4*q_cut**3*sqB)/mbkin**10 + (8856*mcMS**6*q_cut**3*sqB)/
            mbkin**12 + (3320*mcMS**8*q_cut**3*sqB)/mbkin**14 + 
           (4664*mcMS**10*q_cut**3*sqB)/mbkin**16 + (5580*mcMS**12*q_cut**3*sqB)/
            mbkin**18 + (592*mcMS**14*q_cut**3*sqB)/mbkin**20 - 
           (102*mcMS**16*q_cut**3*sqB)/mbkin**22 - (12*q_cut**4*sqB)/mbkin**8 + 
           (410*mcMS**2*q_cut**4*sqB)/mbkin**10 - (2188*mcMS**4*q_cut**4*sqB)/mbkin**12 - 
           (10862*mcMS**6*q_cut**4*sqB)/mbkin**14 - (12784*mcMS**8*q_cut**4*sqB)/
            mbkin**16 - (8026*mcMS**10*q_cut**4*sqB)/mbkin**18 - 
           (800*mcMS**12*q_cut**4*sqB)/mbkin**20 + (486*mcMS**14*q_cut**4*sqB)/
            mbkin**22 + (58*q_cut**5*sqB)/mbkin**10 - (304*mcMS**2*q_cut**5*sqB)/
            mbkin**12 - (202*mcMS**4*q_cut**5*sqB)/mbkin**14 + 
           (1540*mcMS**6*q_cut**5*sqB)/mbkin**16 + (2402*mcMS**8*q_cut**5*sqB)/
            mbkin**18 + (476*mcMS**10*q_cut**5*sqB)/mbkin**20 - 
           (354*mcMS**12*q_cut**5*sqB)/mbkin**22 - (26*q_cut**6*sqB)/mbkin**12 - 
           (62*mcMS**2*q_cut**6*sqB)/mbkin**14 + (8*mcMS**4*q_cut**6*sqB)/mbkin**16 - 
           (372*mcMS**6*q_cut**6*sqB)/mbkin**18 - (662*mcMS**8*q_cut**6*sqB)/mbkin**20 - 
           (102*mcMS**10*q_cut**6*sqB)/mbkin**22 - (30*q_cut**7*sqB)/mbkin**14 + 
           (80*mcMS**2*q_cut**7*sqB)/mbkin**16 + (344*mcMS**4*q_cut**7*sqB)/mbkin**18 + 
           (568*mcMS**6*q_cut**7*sqB)/mbkin**20 + (270*mcMS**8*q_cut**7*sqB)/mbkin**22 + 
           (31*q_cut**8*sqB)/mbkin**16 - (4*mcMS**2*q_cut**8*sqB)/mbkin**18 - 
           (157*mcMS**4*q_cut**8*sqB)/mbkin**20 - (138*mcMS**6*q_cut**8*sqB)/mbkin**22 - 
           (8*q_cut**9*sqB)/mbkin**18 + (24*mcMS**4*q_cut**9*sqB)/mbkin**22))*
        np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
              mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
            np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*
                mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
       (144*mcMS**4*(-16*(-((-1 + mcMS**2/mbkin**2)**4*(-16 + (84*mcMS**2)/
                mbkin**2 + (397*mcMS**4)/mbkin**4 + (7*mcMS**6)/mbkin**6 - 
               (273*mcMS**8)/mbkin**8 + (53*mcMS**10)/mbkin**10 + (108*mcMS**12)/
                mbkin**12)) + ((-1 + mcMS**2/mbkin**2)**2*(-83 + (309*mcMS**2)/
                mbkin**2 + (1688*mcMS**4)/mbkin**4 + (1592*mcMS**6)/mbkin**6 - 
               (423*mcMS**8)/mbkin**8 - (1055*mcMS**10)/mbkin**10 + (258*mcMS**12)/
                mbkin**12 + (594*mcMS**14)/mbkin**14)*q_cut)/mbkin**2 - 
            ((-158 + (492*mcMS**2)/mbkin**2 + (2013*mcMS**4)/mbkin**4 + 
               (1738*mcMS**6)/mbkin**6 + (2412*mcMS**8)/mbkin**8 + (120*mcMS**10)/
                mbkin**10 - (2075*mcMS**12)/mbkin**12 - (6*mcMS**14)/mbkin**14 + 
               (1224*mcMS**16)/mbkin**16)*q_cut**2)/mbkin**4 + 
            ((-101 + (215*mcMS**2)/mbkin**2 + (929*mcMS**4)/mbkin**4 + 
               (641*mcMS**6)/mbkin**6 - (688*mcMS**8)/mbkin**8 - (784*mcMS**10)/
                mbkin**10 + (1330*mcMS**12)/mbkin**12 + (918*mcMS**14)/mbkin**14)*
              q_cut**3)/mbkin**6 + ((-80 - (260*mcMS**2)/mbkin**2 + (156*mcMS**4)/
                mbkin**4 + (895*mcMS**6)/mbkin**6 + (742*mcMS**8)/mbkin**8 - 
               (45*mcMS**10)/mbkin**10 + (540*mcMS**12)/mbkin**12)*q_cut**4)/
             mbkin**8 - ((-179 - (393*mcMS**2)/mbkin**2 + (414*mcMS**4)/mbkin**4 + 
               (1540*mcMS**6)/mbkin**6 + (1914*mcMS**8)/mbkin**8 + (1602*mcMS**10)/
                mbkin**10)*q_cut**5)/mbkin**10 + (2*(-61 - (110*mcMS**2)/mbkin**2 + 
               (307*mcMS**4)/mbkin**4 + (770*mcMS**6)/mbkin**6 + (648*mcMS**8)/
                mbkin**8)*q_cut**6)/mbkin**12 + ((37 + (37*mcMS**2)/mbkin**2 - 
               (366*mcMS**4)/mbkin**4 - (486*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
            ((-4 + (72*mcMS**4)/mbkin**4)*q_cut**8)/mbkin**16)*rE + 
          ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)*((-60*mcMS**2*muG**2)/mbkin**2 + (384*mcMS**4*muG**2)/
             mbkin**4 - (1704*mcMS**6*muG**2)/mbkin**6 + (6120*mcMS**8*muG**2)/
             mbkin**8 - (8700*mcMS**10*muG**2)/mbkin**10 + (4080*mcMS**12*muG**2)/
             mbkin**12 + (96*mcMS**14*muG**2)/mbkin**14 - (216*mcMS**16*muG**2)/
             mbkin**16 - (60*mcMS**2*muG*mupi)/mbkin**2 + (576*mcMS**4*muG*mupi)/
             mbkin**4 - (1296*mcMS**6*muG*mupi)/mbkin**6 - 
            (1320*mcMS**8*muG*mupi)/mbkin**8 + (4500*mcMS**10*muG*mupi)/
             mbkin**10 - (2160*mcMS**12*muG*mupi)/mbkin**12 - 
            (456*mcMS**14*muG*mupi)/mbkin**14 + (216*mcMS**16*muG*mupi)/
             mbkin**16 - (60*mcMS**2*muG**2*q_cut)/mbkin**4 + (876*mcMS**4*muG**2*q_cut)/
             mbkin**6 - (1128*mcMS**6*muG**2*q_cut)/mbkin**8 - 
            (2388*mcMS**8*muG**2*q_cut)/mbkin**10 - (7248*mcMS**10*muG**2*q_cut)/
             mbkin**12 + (552*mcMS**12*muG**2*q_cut)/mbkin**14 + 
            (756*mcMS**14*muG**2*q_cut)/mbkin**16 + (180*mcMS**2*muG*mupi*q_cut)/
             mbkin**4 - (1236*mcMS**4*muG*mupi*q_cut)/mbkin**6 + 
            (408*mcMS**6*muG*mupi*q_cut)/mbkin**8 + (5508*mcMS**8*muG*mupi*q_cut)/
             mbkin**10 + (4008*mcMS**10*muG*mupi*q_cut)/mbkin**12 + 
            (528*mcMS**12*muG*mupi*q_cut)/mbkin**14 - (756*mcMS**14*muG*mupi*q_cut)/
             mbkin**16 + (120*mcMS**2*muG**2*q_cut**2)/mbkin**6 - 
            (360*mcMS**4*muG**2*q_cut**2)/mbkin**8 - (1188*mcMS**6*muG**2*q_cut**2)/
             mbkin**10 + (2976*mcMS**8*muG**2*q_cut**2)/mbkin**12 - 
            (108*mcMS**10*muG**2*q_cut**2)/mbkin**14 - (720*mcMS**12*muG**2*q_cut**2)/
             mbkin**16 - (120*mcMS**2*muG*mupi*q_cut**2)/mbkin**6 + 
            (600*mcMS**4*muG*mupi*q_cut**2)/mbkin**8 - (12*mcMS**6*muG*mupi*q_cut**2)/
             mbkin**10 - (1296*mcMS**8*muG*mupi*q_cut**2)/mbkin**12 - 
            (612*mcMS**10*muG*mupi*q_cut**2)/mbkin**14 + (720*mcMS**12*muG*mupi*
              q_cut**2)/mbkin**16 + (360*mcMS**2*muG**2*q_cut**3)/mbkin**8 - 
            (1020*mcMS**4*muG**2*q_cut**3)/mbkin**10 - (2868*mcMS**6*muG**2*q_cut**3)/
             mbkin**12 - (3048*mcMS**8*muG**2*q_cut**3)/mbkin**14 - 
            (360*mcMS**10*muG**2*q_cut**3)/mbkin**16 - (120*mcMS**2*muG*mupi*q_cut**3)/
             mbkin**8 + (780*mcMS**4*muG*mupi*q_cut**3)/mbkin**10 + 
            (1668*mcMS**6*muG*mupi*q_cut**3)/mbkin**12 + (2328*mcMS**8*muG*mupi*
              q_cut**3)/mbkin**14 + (360*mcMS**10*muG*mupi*q_cut**3)/mbkin**16 - 
            (540*mcMS**2*muG**2*q_cut**4)/mbkin**10 + (2160*mcMS**4*muG**2*q_cut**4)/
             mbkin**12 + (3732*mcMS**6*muG**2*q_cut**4)/mbkin**14 + 
            (1080*mcMS**8*muG**2*q_cut**4)/mbkin**16 + (180*mcMS**2*muG*mupi*q_cut**4)/
             mbkin**10 - (1440*mcMS**4*muG*mupi*q_cut**4)/mbkin**12 - 
            (2652*mcMS**6*muG*mupi*q_cut**4)/mbkin**14 - (1080*mcMS**8*muG*mupi*
              q_cut**4)/mbkin**16 + (180*mcMS**2*muG**2*q_cut**5)/mbkin**12 - 
            (1224*mcMS**4*muG**2*q_cut**5)/mbkin**14 - (684*mcMS**6*muG**2*q_cut**5)/
             mbkin**16 - (60*mcMS**2*muG*mupi*q_cut**5)/mbkin**12 + 
            (864*mcMS**4*muG*mupi*q_cut**5)/mbkin**14 + (684*mcMS**6*muG*mupi*q_cut**5)/
             mbkin**16 + (144*mcMS**4*muG**2*q_cut**6)/mbkin**16 - 
            (144*mcMS**4*muG*mupi*q_cut**6)/mbkin**16 + 24*mcMS**2*muG*
             ((-1 + mcMS**2/mbkin**2)**2*(-5 + (38*mcMS**2)/mbkin**2 - 
                (27*mcMS**4)/mbkin**4 - (202*mcMS**6)/mbkin**6 - (2*mcMS**8)/
                 mbkin**8 + (18*mcMS**10)/mbkin**10) + ((15 - (103*mcMS**2)/
                  mbkin**2 + (34*mcMS**4)/mbkin**4 + (459*mcMS**6)/mbkin**6 + 
                 (334*mcMS**8)/mbkin**8 + (44*mcMS**10)/mbkin**10 - (63*mcMS**12)/
                  mbkin**12)*q_cut)/mbkin**2 + ((-10 + (50*mcMS**2)/mbkin**2 - 
                 mcMS**4/mbkin**4 - (108*mcMS**6)/mbkin**6 - (51*mcMS**8)/
                  mbkin**8 + (60*mcMS**10)/mbkin**10)*q_cut**2)/mbkin**4 + 
              ((-10 + (65*mcMS**2)/mbkin**2 + (139*mcMS**4)/mbkin**4 + 
                 (194*mcMS**6)/mbkin**6 + (30*mcMS**8)/mbkin**8)*q_cut**3)/mbkin**6 - 
              ((-15 + (120*mcMS**2)/mbkin**2 + (221*mcMS**4)/mbkin**4 + 
                 (90*mcMS**6)/mbkin**6)*q_cut**4)/mbkin**8 + 
              ((-5 + (72*mcMS**2)/mbkin**2 + (57*mcMS**4)/mbkin**4)*q_cut**5)/mbkin**
                10 - (12*mcMS**2*q_cut**6)/mbkin**14) + 
            4*(-((-1 + mcMS**2/mbkin**2)**2*(-19 + (41*mcMS**2)/mbkin**2 + 
                 (383*mcMS**4)/mbkin**4 - (917*mcMS**6)/mbkin**6 - (752*mcMS**8)/
                  mbkin**8 + (112*mcMS**10)/mbkin**10 + (72*mcMS**12)/
                  mbkin**12)) + ((-69 + (122*mcMS**2)/mbkin**2 + (539*mcMS**4)/
                  mbkin**4 - (814*mcMS**6)/mbkin**6 - (3029*mcMS**8)/mbkin**8 - 
                 (1520*mcMS**10)/mbkin**10 + (199*mcMS**12)/mbkin**12 + 
                 (252*mcMS**14)/mbkin**14)*q_cut)/mbkin**2 + 
              ((70 - (68*mcMS**2)/mbkin**2 - (99*mcMS**4)/mbkin**4 + (581*mcMS**6)/
                  mbkin**6 + (377*mcMS**8)/mbkin**8 - (261*mcMS**10)/mbkin**10 - 
                 (240*mcMS**12)/mbkin**12)*q_cut**2)/mbkin**4 - 
              (2*(-15 - (91*mcMS**2)/mbkin**2 + (79*mcMS**4)/mbkin**4 + 
                 (143*mcMS**6)/mbkin**6 + (238*mcMS**8)/mbkin**8 + (60*mcMS**10)/
                  mbkin**10)*q_cut**3)/mbkin**6 + ((-105 - (253*mcMS**2)/mbkin**2 + 
                 (295*mcMS**4)/mbkin**4 + (779*mcMS**6)/mbkin**6 + (360*mcMS**8)/
                  mbkin**8)*q_cut**4)/mbkin**8 + ((71 + (96*mcMS**2)/mbkin**2 - 
                 (273*mcMS**4)/mbkin**4 - (228*mcMS**6)/mbkin**6)*q_cut**5)/mbkin**
                10 - (16*(mbkin**4 - 3*mcMS**4)*q_cut**6)/mbkin**16)*rG - 
            8*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*(3 - (17*mcMS**2)/mbkin**2 + 
                 (443*mcMS**4)/mbkin**4 + (1923*mcMS**6)/mbkin**6 + (188*mcMS**8)/
                  mbkin**8 - (110*mcMS**10)/mbkin**10 + (90*mcMS**12)/
                  mbkin**12)) + ((18 - (129*mcMS**2)/mbkin**2 + (1036*mcMS**4)/
                  mbkin**4 + (4331*mcMS**6)/mbkin**6 + (4596*mcMS**8)/mbkin**8 + 
                 (443*mcMS**10)/mbkin**10 - (530*mcMS**12)/mbkin**12 + 
                 (315*mcMS**14)/mbkin**14)*q_cut)/mbkin**2 + 
              ((-30 + (136*mcMS**2)/mbkin**2 - (287*mcMS**4)/mbkin**4 - 
                 (779*mcMS**6)/mbkin**6 + (165*mcMS**8)/mbkin**8 + (255*mcMS**10)/
                  mbkin**10 - (300*mcMS**12)/mbkin**12)*q_cut**2)/mbkin**4 - 
              (2*mcMS**2*(-8 - (38*mcMS**2)/mbkin**2 + (95*mcMS**4)/mbkin**4 + 
                 (100*mcMS**6)/mbkin**6 + (75*mcMS**8)/mbkin**8)*q_cut**3)/mbkin**8 + 
              ((45 - (39*mcMS**2)/mbkin**2 - (45*mcMS**4)/mbkin**4 + (305*mcMS**6)/
                  mbkin**6 + (450*mcMS**8)/mbkin**8)*q_cut**4)/mbkin**8 - 
              ((42 + (7*mcMS**2)/mbkin**2 + (120*mcMS**4)/mbkin**4 + (285*mcMS**6)/
                  mbkin**6)*q_cut**5)/mbkin**10 + (12*(mbkin**4 + 5*mcMS**4)*q_cut**6)/
               mbkin**16)*rhoD - 60*sB + (100*mcMS**2*sB)/mbkin**2 + 
            (4440*mcMS**4*sB)/mbkin**4 - (2040*mcMS**6*sB)/mbkin**6 - 
            (5180*mcMS**8*sB)/mbkin**8 - (1740*mcMS**10*sB)/mbkin**10 + 
            (4080*mcMS**12*sB)/mbkin**12 + (1120*mcMS**14*sB)/mbkin**14 - 
            (720*mcMS**16*sB)/mbkin**16 + (180*q_cut*sB)/mbkin**2 + 
            (480*mcMS**2*q_cut*sB)/mbkin**4 - (8980*mcMS**4*q_cut*sB)/mbkin**6 - 
            (25160*mcMS**6*q_cut*sB)/mbkin**8 - (19140*mcMS**8*q_cut*sB)/mbkin**10 - 
            (6800*mcMS**10*q_cut*sB)/mbkin**12 - (700*mcMS**12*q_cut*sB)/mbkin**14 + 
            (2520*mcMS**14*q_cut*sB)/mbkin**16 - (120*q_cut**2*sB)/mbkin**4 - 
            (640*mcMS**2*q_cut**2*sB)/mbkin**6 + (2540*mcMS**4*q_cut**2*sB)/mbkin**8 + 
            (4100*mcMS**6*q_cut**2*sB)/mbkin**10 + (1260*mcMS**8*q_cut**2*sB)/mbkin**12 + 
            (60*mcMS**10*q_cut**2*sB)/mbkin**14 - (2400*mcMS**12*q_cut**2*sB)/mbkin**16 - 
            (120*q_cut**3*sB)/mbkin**6 - (1000*mcMS**2*q_cut**3*sB)/mbkin**8 - 
            (2320*mcMS**4*q_cut**3*sB)/mbkin**10 - (3920*mcMS**6*q_cut**3*sB)/mbkin**12 - 
            (4720*mcMS**8*q_cut**3*sB)/mbkin**14 - (1200*mcMS**10*q_cut**3*sB)/
             mbkin**16 + (180*q_cut**4*sB)/mbkin**8 + (1500*mcMS**2*q_cut**4*sB)/
             mbkin**10 + (4500*mcMS**4*q_cut**4*sB)/mbkin**12 + 
            (6460*mcMS**6*q_cut**4*sB)/mbkin**14 + (3600*mcMS**8*q_cut**4*sB)/mbkin**16 - 
            (60*q_cut**5*sB)/mbkin**10 - (440*mcMS**2*q_cut**5*sB)/mbkin**12 - 
            (2220*mcMS**4*q_cut**5*sB)/mbkin**14 - (2280*mcMS**6*q_cut**5*sB)/mbkin**16 + 
            (480*mcMS**4*q_cut**6*sB)/mbkin**16 + 48*sE + (32*mcMS**2*sE)/mbkin**2 - 
            (2928*mcMS**4*sE)/mbkin**4 + (2928*mcMS**6*sE)/mbkin**6 + 
            (1280*mcMS**8*sE)/mbkin**8 + (192*mcMS**10*sE)/mbkin**10 - 
            (1392*mcMS**12*sE)/mbkin**12 - (592*mcMS**14*sE)/mbkin**14 + 
            (432*mcMS**16*sE)/mbkin**16 - (168*q_cut*sE)/mbkin**2 - 
            (456*mcMS**2*q_cut*sE)/mbkin**4 + (5272*mcMS**4*q_cut*sE)/mbkin**6 + 
            (11720*mcMS**6*q_cut*sE)/mbkin**8 + (6120*mcMS**8*q_cut*sE)/mbkin**10 + 
            (2408*mcMS**10*q_cut*sE)/mbkin**12 - (344*mcMS**12*q_cut*sE)/mbkin**14 - 
            (1512*mcMS**14*q_cut*sE)/mbkin**16 + (160*q_cut**2*sE)/mbkin**4 + 
            (384*mcMS**2*q_cut**2*sE)/mbkin**6 - (1568*mcMS**4*q_cut**2*sE)/mbkin**8 - 
            (2160*mcMS**6*q_cut**2*sE)/mbkin**10 + (448*mcMS**8*q_cut**2*sE)/mbkin**12 + 
            (2256*mcMS**10*q_cut**2*sE)/mbkin**14 + (1440*mcMS**12*q_cut**2*sE)/
             mbkin**16 + (80*q_cut**3*sE)/mbkin**6 + (944*mcMS**2*q_cut**3*sE)/mbkin**8 + 
            (944*mcMS**4*q_cut**3*sE)/mbkin**10 + (736*mcMS**6*q_cut**3*sE)/mbkin**12 - 
            (224*mcMS**8*q_cut**3*sE)/mbkin**14 + (720*mcMS**10*q_cut**3*sE)/mbkin**16 - 
            (240*q_cut**4*sE)/mbkin**8 - (1376*mcMS**2*q_cut**4*sE)/mbkin**10 - 
            (1920*mcMS**4*q_cut**4*sE)/mbkin**12 - (1984*mcMS**6*q_cut**4*sE)/mbkin**14 - 
            (2160*mcMS**8*q_cut**4*sE)/mbkin**16 + (152*q_cut**5*sE)/mbkin**10 + 
            (472*mcMS**2*q_cut**5*sE)/mbkin**12 + (888*mcMS**4*q_cut**5*sE)/mbkin**14 + 
            (1368*mcMS**6*q_cut**5*sE)/mbkin**16 - (32*q_cut**6*sE)/mbkin**12 - 
            (288*mcMS**4*q_cut**6*sE)/mbkin**16 + 9*sqB - (49*mcMS**2*sqB)/mbkin**2 - 
            (552*mcMS**4*sqB)/mbkin**4 - (108*mcMS**6*sqB)/mbkin**6 + 
            (1415*mcMS**8*sqB)/mbkin**8 - (69*mcMS**10*sqB)/mbkin**10 - 
            (666*mcMS**12*sqB)/mbkin**12 + (2*mcMS**14*sqB)/mbkin**14 + 
            (18*mcMS**16*sqB)/mbkin**16 - (24*q_cut*sqB)/mbkin**2 - 
            (3*mcMS**2*q_cut*sqB)/mbkin**4 + (1294*mcMS**4*q_cut*sqB)/mbkin**6 + 
            (3941*mcMS**6*q_cut*sqB)/mbkin**8 + (3966*mcMS**8*q_cut*sqB)/mbkin**10 + 
            (1085*mcMS**10*q_cut*sqB)/mbkin**12 - (116*mcMS**12*q_cut*sqB)/mbkin**14 - 
            (63*mcMS**14*q_cut*sqB)/mbkin**16 + (10*q_cut**2*sqB)/mbkin**4 + 
            (72*mcMS**2*q_cut**2*sqB)/mbkin**6 - (359*mcMS**4*q_cut**2*sqB)/mbkin**8 - 
            (699*mcMS**6*q_cut**2*sqB)/mbkin**10 - (83*mcMS**8*q_cut**2*sqB)/mbkin**12 + 
            (159*mcMS**10*q_cut**2*sqB)/mbkin**14 + (60*mcMS**12*q_cut**2*sqB)/
             mbkin**16 + (20*q_cut**3*sqB)/mbkin**6 + (92*mcMS**2*q_cut**3*sqB)/
             mbkin**8 + (152*mcMS**4*q_cut**3*sqB)/mbkin**10 - (26*mcMS**6*q_cut**3*sqB)/
             mbkin**12 + (124*mcMS**8*q_cut**3*sqB)/mbkin**14 + 
            (30*mcMS**10*q_cut**3*sqB)/mbkin**16 - (15*q_cut**4*sqB)/mbkin**8 - 
            (143*mcMS**2*q_cut**4*sqB)/mbkin**10 - (285*mcMS**4*q_cut**4*sqB)/mbkin**12 - 
            (271*mcMS**6*q_cut**4*sqB)/mbkin**14 - (90*mcMS**8*q_cut**4*sqB)/mbkin**16 - 
            (4*q_cut**5*sqB)/mbkin**10 + (31*mcMS**2*q_cut**5*sqB)/mbkin**12 + 
            (102*mcMS**4*q_cut**5*sqB)/mbkin**14 + (57*mcMS**6*q_cut**5*sqB)/mbkin**16 + 
            (4*q_cut**6*sqB)/mbkin**12 - (12*mcMS**4*q_cut**6*sqB)/mbkin**16))*
         np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 - 
       (4320*mcMS**8*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
            2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(24*mcMS**2*muG - 
          (72*mcMS**4*muG)/mbkin**2 - (12*mcMS**2*muG**2)/mbkin**2 + 
          (36*mcMS**4*muG**2)/mbkin**4 + (12*mcMS**2*muG*mupi)/mbkin**2 - 
          (36*mcMS**4*muG*mupi)/mbkin**4 + 32*(1 + mcMS**2/mbkin**2)*rE + 
          4*(1 - (4*mcMS**2)/mbkin**2 + (15*mcMS**4)/mbkin**4)*rG + 
          24*mbkin*rhoD + (80*mcMS**2*rhoD)/mbkin + (120*mcMS**4*rhoD)/
           mbkin**3 + 12*sB + (88*mcMS**2*sB)/mbkin**2 + (60*mcMS**4*sB)/
           mbkin**4 - (64*mcMS**2*sE)/mbkin**2 - 3*sqB - (10*mcMS**2*sqB)/
           mbkin**2 - (15*mcMS**4*sqB)/mbkin**4)*
         np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8)/
      (180*mbkin**2*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
          2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
            7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
            mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**3) + 
     (api4*(((18*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
              7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
              mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)**2*(3*mbkin**8 - 
             42*mbkin**6*mcMS**2 - 282*mbkin**4*mcMS**4 - 42*mbkin**2*mcMS**6 + 
             3*mcMS**8 + 3*mbkin**6*q_cut - 33*mbkin**4*mcMS**2*q_cut - 
             33*mbkin**2*mcMS**4*q_cut + 3*mcMS**6*q_cut - 7*mbkin**4*q_cut**2 + 
             2*mbkin**2*mcMS**2*q_cut**2 - 7*mcMS**4*q_cut**2 - 7*mbkin**2*q_cut**3 - 
             7*mcMS**2*q_cut**3 + 8*q_cut**4))/mbkin**28 - 
          (216*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**
                4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
            (21*mbkin**14 - 321*mbkin**12*mcMS**2 + 297*mbkin**10*mcMS**4 + 
             6483*mbkin**8*mcMS**6 + 6483*mbkin**6*mcMS**8 + 297*mbkin**4*
              mcMS**10 - 321*mbkin**2*mcMS**12 + 21*mcMS**14 - 30*mbkin**12*q_cut + 
             156*mbkin**10*mcMS**2*q_cut + 1302*mbkin**8*mcMS**4*q_cut + 
             1464*mbkin**6*mcMS**6*q_cut + 1302*mbkin**4*mcMS**8*q_cut + 
             156*mbkin**2*mcMS**10*q_cut - 30*mcMS**12*q_cut - 41*mbkin**10*q_cut**2 + 
             411*mbkin**8*mcMS**2*q_cut**2 + 1394*mbkin**6*mcMS**4*q_cut**2 + 
             1394*mbkin**4*mcMS**6*q_cut**2 + 411*mbkin**2*mcMS**8*q_cut**2 - 
             41*mcMS**10*q_cut**2 + 60*mbkin**8*q_cut**3 - 64*mbkin**6*mcMS**2*q_cut**3 - 
             568*mbkin**4*mcMS**4*q_cut**3 - 64*mbkin**2*mcMS**6*q_cut**3 + 
             60*mcMS**8*q_cut**3 + 35*mbkin**6*q_cut**4 - 139*mbkin**4*mcMS**2*q_cut**4 - 
             139*mbkin**2*mcMS**4*q_cut**4 + 35*mcMS**6*q_cut**4 - 46*mbkin**4*q_cut**5 - 
             28*mbkin**2*mcMS**2*q_cut**5 - 46*mcMS**4*q_cut**5 - 15*mbkin**2*q_cut**6 - 
             15*mcMS**2*q_cut**6 + 16*q_cut**7)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
               mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                   2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**22 + 
          (2592*mcMS**8*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**2*(33*mbkin**8 - 222*mbkin**6*mcMS**2 - 
             702*mbkin**4*mcMS**4 - 222*mbkin**2*mcMS**6 + 33*mcMS**8 - 
             27*mbkin**6*q_cut - 63*mbkin**4*mcMS**2*q_cut - 63*mbkin**2*mcMS**4*q_cut - 
             27*mcMS**6*q_cut - 37*mbkin**4*q_cut**2 - 58*mbkin**2*mcMS**2*q_cut**2 - 
             37*mcMS**4*q_cut**2 + 23*mbkin**2*q_cut**3 + 23*mcMS**2*q_cut**3 + 8*q_cut**4)*
            np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**20 - 
          (466560*mcMS**12*(mbkin**2 + mcMS**2)*((mbkin**4 - 2*mbkin**2*mcMS**2 + 
               mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**10)*
         (((-4*(3 + 8*mbkin))/(9*mbkin**4*((mbkin**4 - 2*mbkin**2*mcMS**2 + 
                 mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/
                2)) - (3*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                 (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + (4*q_cut*
                 (6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                  12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**6)))/(2*mbkin**2*((mbkin**4 - 2*mbkin**2*mcMS**2 + 
                 mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
              ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
               q_cut**2/mbkin**4)))/((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                  2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 
                7*mbkin**4*mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - 
                mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
             (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                      2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
            3 - (3*((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*
                 mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - 
                mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)*(
                (-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                  (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                   9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                   12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                 (9*mbkin**6)))/(2*mbkin**6*((-1 + mcMS**2/mbkin**2)**2 - 
                (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)) + 
             np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*((-4*(3 + 8*mbkin)*q_cut**3)/
                (3*mbkin**8) - (14*mcMS**2*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) - (28*mcMS**4*
                 (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**6) + (2*mcMS**6*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(3*mbkin**8) + (2*q_cut**2*
                 (12*mbkin**2 + 32*mbkin**3 + 18*mcMS**2 + 48*mbkin*mcMS**2 - 
                  12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**8) + (4*q_cut*(3*mbkin**4 + 8*mbkin**5 + 9*mcMS**4 + 
                  24*mbkin*mcMS**4 - 12*mbkin**2*mcMS**4 - 9*mbkin**2*mcMS**4*
                   np.log(mu0**2/mcMS**2)))/(9*mbkin**8)) - 
             12*((mcMS**4*(16/3 + 4*np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - 
                    q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                   (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4))))/mbkin**4 + mcMS**4*
                ((-8*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 6*mbkin**2*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                     np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                         q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                    9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2)))/(9*mbkin**4*(mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)*
                   (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + mbkin**2*
                     np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                         q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                 (8*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                      mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                          q_cut**2)/mbkin**4))))/(9*mbkin**6)))))/
           (mbkin**2*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*
                mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
                 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
                 mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
              (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                       2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                       2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**
                4)**4)) + (18*mbkin**4*(3*(1 - (14*mcMS**2)/mbkin**2 - 
              (94*mcMS**4)/mbkin**4 - (14*mcMS**6)/mbkin**6 + mcMS**8/mbkin**8) + 
            (3*(mbkin**6 - 11*mbkin**4*mcMS**2 - 11*mbkin**2*mcMS**4 + mcMS**6)*q_cut)/
             mbkin**8 + ((-7 + (2*mcMS**2)/mbkin**2 - (7*mcMS**4)/mbkin**4)*q_cut**2)/
             mbkin**4 - (7*(mbkin**2 + mcMS**2)*q_cut**3)/mbkin**8 + 
            (8*q_cut**4)/mbkin**8)*((-1 + mcMS**2/mbkin**2)**2*(1 - (7*mcMS**2)/
                mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/mbkin**6) + 
             ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + (14*mcMS**6)/
                 mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/mbkin**2 + 
             (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*mcMS**4 + mcMS**6)*q_cut**
                2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/
              mbkin**10 - (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + 
             q_cut**5/mbkin**10)**2*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
            (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 9*
                mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
            (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 12*
                mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
             (9*mbkin**6)) + ((-1 + mcMS**2/mbkin**2)**2 - 
            (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
           ((64*mbkin*(-((-1 + mcMS**2/mbkin**2)**4*(1 + mcMS**2/mbkin**2)**2*
                 (351 - (5815*mcMS**2)/mbkin**2 + (36915*mcMS**4)/mbkin**4 - 
                  (109923*mcMS**6)/mbkin**6 + (69257*mcMS**8)/mbkin**8 - 
                  (22209*mcMS**10)/mbkin**10 + (1077*mcMS**12)/mbkin**12 + 
                  (107*mcMS**14)/mbkin**14)) + (2*(-1 + mcMS**2/mbkin**2)**2*
                 (1089 - (13483*mcMS**2)/mbkin**2 + (53852*mcMS**4)/mbkin**4 - 
                  (45772*mcMS**6)/mbkin**6 - (288518*mcMS**8)/mbkin**8 - 
                  (239134*mcMS**10)/mbkin**10 + (43532*mcMS**12)/mbkin**12 + 
                  (52492*mcMS**14)/mbkin**14 - (52219*mcMS**16)/mbkin**16 + 
                  (3977*mcMS**18)/mbkin**18 + (344*mcMS**20)/mbkin**20)*q_cut)/
                mbkin**2 + ((-4929 + (52403*mcMS**2)/mbkin**2 - (176271*mcMS**4)/
                   mbkin**4 + (115749*mcMS**6)/mbkin**6 + (605090*mcMS**8)/
                   mbkin**8 + (898826*mcMS**10)/mbkin**10 + (657918*mcMS**12)/
                   mbkin**12 - (174202*mcMS**14)/mbkin**14 - (222817*mcMS**16)/
                   mbkin**16 + (202947*mcMS**18)/mbkin**18 - (17775*mcMS**20)/
                   mbkin**20 - (1579*mcMS**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
               (2*(1659 - (10934*mcMS**2)/mbkin**2 + (14621*mcMS**4)/mbkin**4 + 
                  (22408*mcMS**6)/mbkin**6 - (61810*mcMS**8)/mbkin**8 + 
                  (4308*mcMS**10)/mbkin**10 + (92322*mcMS**12)/mbkin**12 - 
                  (26296*mcMS**14)/mbkin**14 - (44537*mcMS**16)/mbkin**16 + 
                  (7826*mcMS**18)/mbkin**18 + (433*mcMS**20)/mbkin**20)*q_cut**3)/
                mbkin**6 + (4*(1353 - (9240*mcMS**2)/mbkin**2 + (13738*mcMS**4)/
                   mbkin**4 + (9254*mcMS**6)/mbkin**6 - (21795*mcMS**8)/mbkin**8 - 
                  (9861*mcMS**10)/mbkin**10 - (9696*mcMS**12)/mbkin**12 - 
                  (33052*mcMS**14)/mbkin**14 + (4192*mcMS**16)/mbkin**16 + 
                  (675*mcMS**18)/mbkin**18)*q_cut**4)/mbkin**8 - (2*(6003 - 
                  (25356*mcMS**2)/mbkin**2 + (10882*mcMS**4)/mbkin**4 + 
                  (16908*mcMS**6)/mbkin**6 - (30624*mcMS**8)/mbkin**8 - 
                  (109320*mcMS**10)/mbkin**10 - (93822*mcMS**12)/mbkin**12 + 
                  (22600*mcMS**14)/mbkin**14 + (2793*mcMS**16)/mbkin**16)*q_cut**5)/
                mbkin**10 + (2*(3483 - (12003*mcMS**2)/mbkin**2 - (3121*mcMS**4)/
                   mbkin**4 - (19139*mcMS**6)/mbkin**6 - (74989*mcMS**8)/
                   mbkin**8 - (52799*mcMS**10)/mbkin**10 + (11627*mcMS**12)/
                   mbkin**12 + (1677*mcMS**14)/mbkin**14)*q_cut**6)/mbkin**12 + 
               (4*(765 + (1989*mcMS**2)/mbkin**2 + (3559*mcMS**4)/mbkin**4 + 
                  (13202*mcMS**6)/mbkin**6 + (14387*mcMS**8)/mbkin**8 + 
                  (5569*mcMS**10)/mbkin**10 + (417*mcMS**12)/mbkin**12)*q_cut**7)/
                mbkin**14 - ((6399 + (6327*mcMS**2)/mbkin**2 + (19022*mcMS**4)/
                   mbkin**4 + (37806*mcMS**6)/mbkin**6 + (34907*mcMS**8)/
                   mbkin**8 + (3747*mcMS**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
               (2*(1719 + (1825*mcMS**2)/mbkin**2 + (6279*mcMS**4)/mbkin**4 + 
                  (8751*mcMS**6)/mbkin**6 + (1150*mcMS**8)/mbkin**8)*q_cut**9)/
                mbkin**18 - ((693 + (829*mcMS**2)/mbkin**2 + (3367*mcMS**4)/
                   mbkin**4 + (607*mcMS**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
               ((6 + (88*mcMS**2)/mbkin**2 + (58*mcMS**4)/mbkin**4)*q_cut**11)/
                mbkin**22 - (2*(3 + (7*mcMS**2)/mbkin**2)*q_cut**12)/mbkin**24 + 
               (6*q_cut**13)/mbkin**26))/9 + 18*(mbkin**4*((-1 + mcMS**2/mbkin**2)**2*
                  (1 - (7*mcMS**2)/mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/
                    mbkin**6) + ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/
                     mbkin**4 + (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/
                  mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*
                     mcMS**4 + mcMS**6)*q_cut**2)/mbkin**10 + (2*(mbkin**4 + 
                    mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/mbkin**10 - 
                 (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2*(
                (-128*(3 + 8*mbkin)*q_cut**4)/(9*mbkin**10) - 
                (4*(7*mbkin**6*mcMS**2 + 94*mbkin**4*mcMS**4 + 21*mbkin**2*
                    mcMS**6 - 2*mcMS**8)*(-6 - 16*mbkin + 12*mbkin**2 + 
                   9*mbkin**2*np.log(mu0**2/mcMS**2)))/(3*mbkin**10) + 
                (14*q_cut**3*(18*mbkin**2 + 48*mbkin**3 + 24*mcMS**2 + 64*mbkin*
                    mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                    np.log(mu0**2/mcMS**2)))/(9*mbkin**10) + q_cut**2*
                 ((-8*(3 + 8*mbkin)*(-7 + (2*mcMS**2)/mbkin**2 - (7*mcMS**4)/
                      mbkin**4))/(9*mbkin**6) + (4*(mbkin**2*mcMS**2 - 7*mcMS**4)*
                    (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                        mcMS**2)))/(9*mbkin**10)) + 3*q_cut*((-4*(3 + 8*mbkin)*
                    (1 - (11*mcMS**2)/mbkin**2 - (11*mcMS**4)/mbkin**4 + 
                     mcMS**6/mbkin**6))/(9*mbkin**4) - (2*(11*mbkin**4*mcMS**2 + 
                     22*mbkin**2*mcMS**4 - 3*mcMS**6)*(-6 - 16*mbkin + 
                     12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                   (9*mbkin**10))) + (3*(1 - (14*mcMS**2)/mbkin**2 - (94*mcMS**4)/
                   mbkin**4 - (14*mcMS**6)/mbkin**6 + mcMS**8/mbkin**8) + 
                (3*(mbkin**6 - 11*mbkin**4*mcMS**2 - 11*mbkin**2*mcMS**4 + mcMS**6)*
                  q_cut)/mbkin**8 + ((-7 + (2*mcMS**2)/mbkin**2 - (7*mcMS**4)/
                    mbkin**4)*q_cut**2)/mbkin**4 - (7*(mbkin**2 + mcMS**2)*q_cut**3)/
                 mbkin**8 + (8*q_cut**4)/mbkin**8)*((8*mbkin**2*(3 + 8*mbkin)*
                  ((-1 + mcMS**2/mbkin**2)**2*(1 - (7*mcMS**2)/mbkin**2 - 
                      (7*mcMS**4)/mbkin**4 + mcMS**6/mbkin**6) + 
                    ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + 
                       (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/
                     mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*
                        mcMS**4 + mcMS**6)*q_cut**2)/mbkin**10 + (2*(mbkin**4 + 
                       mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/mbkin**10 - 
                    (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2)/
                 9 + 2*mbkin**4*((-1 + mcMS**2/mbkin**2)**2*(1 - (7*mcMS**2)/
                     mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/mbkin**6) + 
                  ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + 
                     (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/mbkin**2 + 
                  (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*mcMS**4 + mcMS**6)*
                    q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mcMS**2 + mcMS**4)*
                    q_cut**3)/mbkin**10 - (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + 
                  q_cut**5/mbkin**10)*((-20*(3 + 8*mbkin)*q_cut**5)/(9*mbkin**12) - 
                  (2*(mbkin**2 - mcMS**2)*(9*mbkin**6*mcMS**2 - 7*mbkin**4*
                      mcMS**4 - 31*mbkin**2*mcMS**6 + 5*mcMS**8)*(-6 - 16*mbkin + 
                     12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                   (9*mbkin**12) + (2*q_cut**4*(24*mbkin**2 + 64*mbkin**3 + 
                     30*mcMS**2 + 80*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 
                     9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/(3*mbkin**12) + 
                  2*q_cut**3*((-4*(3 + 8*mbkin)*(1 + mcMS**2/mbkin**2 + mcMS**4/
                        mbkin**4))/(3*mbkin**8) + (2*(mbkin**2*mcMS**2 + 
                       2*mcMS**4)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                        np.log(mu0**2/mcMS**2)))/(9*mbkin**12)) + 2*q_cut**2*
                   ((-8*(3 + 8*mbkin)*(1 - (2*mcMS**2)/mbkin**2 - (2*mcMS**4)/
                        mbkin**4 + mcMS**6/mbkin**6))/(9*mbkin**6) - 
                    (2*(2*mbkin**4*mcMS**2 + 4*mbkin**2*mcMS**4 - 3*mcMS**6)*
                      (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**12)) + q_cut*((-4*(3 + 8*mbkin)*
                      (-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + 
                       (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8))/
                     (9*mbkin**4) + (4*(7*mbkin**6*mcMS**2 + 26*mbkin**4*mcMS**4 + 
                       21*mbkin**2*mcMS**6 - 6*mcMS**8)*(-6 - 16*mbkin + 
                       12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                     (9*mbkin**12)))))) - 
          6*(np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(36*mcMS**4*(
                3*(-1 + mcMS**2/mbkin**2)**2*(7 - (107*mcMS**2)/mbkin**2 + 
                  (99*mcMS**4)/mbkin**4 + (2161*mcMS**6)/mbkin**6 + (2161*mcMS**8)/
                   mbkin**8 + (99*mcMS**10)/mbkin**10 - (107*mcMS**12)/mbkin**12 + 
                  (7*mcMS**14)/mbkin**14) - (24*(3 - (34*mcMS**2)/mbkin**2 - 
                   (42*mcMS**4)/mbkin**4 + (606*mcMS**6)/mbkin**6 + (1094*mcMS**8)/
                    mbkin**8 + (606*mcMS**10)/mbkin**10 - (42*mcMS**12)/
                    mbkin**12 - (34*mcMS**14)/mbkin**14 + (3*mcMS**16)/mbkin**16)*
                  q_cut)/mbkin**2 + (8*(5 - (10*mcMS**2)/mbkin**2 - (261*mcMS**4)/
                    mbkin**4 - (4*mcMS**6)/mbkin**6 - (4*mcMS**8)/mbkin**8 - 
                   (261*mcMS**10)/mbkin**10 - (10*mcMS**12)/mbkin**12 + 
                   (5*mcMS**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
                (16*(7 - (48*mcMS**2)/mbkin**2 - (168*mcMS**4)/mbkin**4 - 
                   (194*mcMS**6)/mbkin**6 - (168*mcMS**8)/mbkin**8 - (48*mcMS**10)/
                    mbkin**10 + (7*mcMS**12)/mbkin**12)*q_cut**3)/mbkin**6 - 
                (6*(21 - (35*mcMS**2)/mbkin**2 - (472*mcMS**4)/mbkin**4 - 
                   (472*mcMS**6)/mbkin**6 - (35*mcMS**8)/mbkin**8 + (21*mcMS**10)/
                    mbkin**10)*q_cut**4)/mbkin**8 - (8*(7 - (26*mcMS**2)/mbkin**2 + 
                   (6*mcMS**4)/mbkin**4 - (26*mcMS**6)/mbkin**6 + (7*mcMS**8)/
                    mbkin**8)*q_cut**5)/mbkin**10 + (8*(14 + (3*mcMS**2)/mbkin**2 + 
                   (3*mcMS**4)/mbkin**4 + (14*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 - 
                (47*(mbkin**2 + mcMS**2)*q_cut**8)/mbkin**18 + (16*q_cut**9)/mbkin**18)*(
                (-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                  (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                   9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                   12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                 (9*mbkin**6)) + ((-1 + mcMS**2/mbkin**2)**2 - 
                (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*(
                (-64*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*(21 - (236*mcMS**2)/
                       mbkin**2 + (280*mcMS**4)/mbkin**4 + (5400*mcMS**6)/
                       mbkin**6 - (43634*mcMS**8)/mbkin**8 - (31160*mcMS**10)/
                       mbkin**10 + (14712*mcMS**12)/mbkin**12 - (4424*mcMS**14)/
                       mbkin**14 - (1619*mcMS**16)/mbkin**16 + (180*mcMS**18)/
                       mbkin**18)) + (2*(51 - (508*mcMS**2)/mbkin**2 + 
                      (671*mcMS**4)/mbkin**4 + (7066*mcMS**6)/mbkin**6 - 
                      (40432*mcMS**8)/mbkin**8 - (66010*mcMS**10)/mbkin**10 - 
                      (36356*mcMS**12)/mbkin**12 + (22310*mcMS**14)/mbkin**14 - 
                      (4915*mcMS**16)/mbkin**16 - (3242*mcMS**18)/mbkin**18 + 
                      (405*mcMS**20)/mbkin**20)*q_cut)/mbkin**2 - 
                   (2*(75 - (535*mcMS**2)/mbkin**2 + (234*mcMS**4)/mbkin**4 + 
                      (3612*mcMS**6)/mbkin**6 - (2692*mcMS**8)/mbkin**8 + 
                      (6882*mcMS**10)/mbkin**10 + (9870*mcMS**12)/mbkin**12 - 
                      (5092*mcMS**14)/mbkin**14 - (2799*mcMS**16)/mbkin**16 + 
                      (525*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
                   (2*(-21 + (124*mcMS**2)/mbkin**2 + (650*mcMS**4)/mbkin**4 - 
                      (4312*mcMS**6)/mbkin**6 + (1928*mcMS**8)/mbkin**8 + 
                      (728*mcMS**10)/mbkin**10 + (7154*mcMS**12)/mbkin**12 + 
                      (1396*mcMS**14)/mbkin**14 - (255*mcMS**16)/mbkin**16)*q_cut**3)/
                    mbkin**6 + (2*(168 - (595*mcMS**2)/mbkin**2 - (1396*mcMS**4)/
                       mbkin**4 + (3013*mcMS**6)/mbkin**6 - (1930*mcMS**8)/
                       mbkin**8 - (13117*mcMS**10)/mbkin**10 - (3590*mcMS**12)/
                       mbkin**12 + (1215*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
                   (2*(-147 + (436*mcMS**2)/mbkin**2 + (723*mcMS**4)/mbkin**4 + 
                      (726*mcMS**6)/mbkin**6 + (6157*mcMS**8)/mbkin**8 + 
                      (2478*mcMS**10)/mbkin**10 - (885*mcMS**12)/mbkin**12)*q_cut**5)/
                    mbkin**10 - (2*(21 + (227*mcMS**2)/mbkin**2 + (112*mcMS**4)/
                       mbkin**4 + (1090*mcMS**6)/mbkin**6 + (1463*mcMS**8)/
                       mbkin**8 + (255*mcMS**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
                   (2*(105 + (140*mcMS**2)/mbkin**2 + (248*mcMS**4)/mbkin**4 + 
                      (856*mcMS**6)/mbkin**6 + (675*mcMS**8)/mbkin**8)*q_cut**7)/
                    mbkin**14 - ((123 + (88*mcMS**2)/mbkin**2 + (447*mcMS**4)/
                       mbkin**4 + (690*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
                   (24*(mbkin**4 + 5*mcMS**4)*q_cut**9)/mbkin**22))/9 + 
                36*(mcMS**4*(3*(-1 + mcMS**2/mbkin**2)**2*(7 - (107*mcMS**2)/
                       mbkin**2 + (99*mcMS**4)/mbkin**4 + (2161*mcMS**6)/
                       mbkin**6 + (2161*mcMS**8)/mbkin**8 + (99*mcMS**10)/
                       mbkin**10 - (107*mcMS**12)/mbkin**12 + (7*mcMS**14)/
                       mbkin**14) - (24*(3 - (34*mcMS**2)/mbkin**2 - (42*mcMS**4)/
                        mbkin**4 + (606*mcMS**6)/mbkin**6 + (1094*mcMS**8)/
                        mbkin**8 + (606*mcMS**10)/mbkin**10 - (42*mcMS**12)/
                        mbkin**12 - (34*mcMS**14)/mbkin**14 + (3*mcMS**16)/
                        mbkin**16)*q_cut)/mbkin**2 + (8*(5 - (10*mcMS**2)/mbkin**2 - 
                       (261*mcMS**4)/mbkin**4 - (4*mcMS**6)/mbkin**6 - (4*mcMS**8)/
                        mbkin**8 - (261*mcMS**10)/mbkin**10 - (10*mcMS**12)/
                        mbkin**12 + (5*mcMS**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
                    (16*(7 - (48*mcMS**2)/mbkin**2 - (168*mcMS**4)/mbkin**4 - 
                       (194*mcMS**6)/mbkin**6 - (168*mcMS**8)/mbkin**8 - 
                       (48*mcMS**10)/mbkin**10 + (7*mcMS**12)/mbkin**12)*q_cut**3)/
                     mbkin**6 - (6*(21 - (35*mcMS**2)/mbkin**2 - (472*mcMS**4)/
                        mbkin**4 - (472*mcMS**6)/mbkin**6 - (35*mcMS**8)/
                        mbkin**8 + (21*mcMS**10)/mbkin**10)*q_cut**4)/mbkin**8 - 
                    (8*(7 - (26*mcMS**2)/mbkin**2 + (6*mcMS**4)/mbkin**4 - 
                       (26*mcMS**6)/mbkin**6 + (7*mcMS**8)/mbkin**8)*q_cut**5)/
                     mbkin**10 + (8*(14 + (3*mcMS**2)/mbkin**2 + (3*mcMS**4)/
                        mbkin**4 + (14*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 - 
                    (47*(mbkin**2 + mcMS**2)*q_cut**8)/mbkin**18 + (16*q_cut**9)/
                     mbkin**18)*(16/3 + 4*np.log(mu0**2/mcMS**2)) + mcMS**4*
                   ((-64*(3 + 8*mbkin)*q_cut**9)/mbkin**20 - (2*(mbkin**2 - mcMS**2)*
                      (121*mbkin**14*mcMS**2 - 519*mbkin**12*mcMS**4 - 
                       6087*mbkin**10*mcMS**6 + 2161*mbkin**8*mcMS**8 + 
                       12471*mbkin**6*mcMS**10 + 1335*mbkin**4*mcMS**12 - 
                       905*mbkin**2*mcMS**14 + 63*mcMS**16)*(-6 - 16*mbkin + 
                       12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                     (3*mbkin**20) + (94*q_cut**8*(48*mbkin**2 + 128*mbkin**3 + 
                       54*mcMS**2 + 144*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 
                       9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**20) + 
                    8*q_cut**6*((-8*(3 + 8*mbkin)*(14 + (3*mcMS**2)/mbkin**2 + 
                         (3*mcMS**4)/mbkin**4 + (14*mcMS**6)/mbkin**6))/
                       (3*mbkin**14) + (2*(mbkin**4*mcMS**2 + 2*mbkin**2*mcMS**4 + 
                         14*mcMS**6)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(3*mbkin**20)) - 8*q_cut**5*
                     ((-20*(3 + 8*mbkin)*(7 - (26*mcMS**2)/mbkin**2 + 
                         (6*mcMS**4)/mbkin**4 - (26*mcMS**6)/mbkin**6 + 
                         (7*mcMS**8)/mbkin**8))/(9*mbkin**12) - 
                      (4*(13*mbkin**6*mcMS**2 - 6*mbkin**4*mcMS**4 + 39*mbkin**2*
                          mcMS**6 - 14*mcMS**8)*(-6 - 16*mbkin + 12*mbkin**2 + 
                         9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**20)) - 
                    6*q_cut**4*((-16*(3 + 8*mbkin)*(21 - (35*mcMS**2)/mbkin**2 - 
                         (472*mcMS**4)/mbkin**4 - (472*mcMS**6)/mbkin**6 - 
                         (35*mcMS**8)/mbkin**8 + (21*mcMS**10)/mbkin**10))/
                       (9*mbkin**10) - (2*(35*mbkin**8*mcMS**2 + 944*mbkin**6*
                          mcMS**4 + 1416*mbkin**4*mcMS**6 + 140*mbkin**2*mcMS**8 - 
                         105*mcMS**10)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**20)) + 16*q_cut**3*
                     ((-4*(3 + 8*mbkin)*(7 - (48*mcMS**2)/mbkin**2 - 
                         (168*mcMS**4)/mbkin**4 - (194*mcMS**6)/mbkin**6 - 
                         (168*mcMS**8)/mbkin**8 - (48*mcMS**10)/mbkin**10 + 
                         (7*mcMS**12)/mbkin**12))/(3*mbkin**8) - 
                      (4*(8*mbkin**10*mcMS**2 + 56*mbkin**8*mcMS**4 + 97*mbkin**6*
                          mcMS**6 + 112*mbkin**4*mcMS**8 + 40*mbkin**2*mcMS**10 - 
                         7*mcMS**12)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(3*mbkin**20)) + 8*q_cut**2*
                     ((-8*(3 + 8*mbkin)*(5 - (10*mcMS**2)/mbkin**2 - 
                         (261*mcMS**4)/mbkin**4 - (4*mcMS**6)/mbkin**6 - 
                         (4*mcMS**8)/mbkin**8 - (261*mcMS**10)/mbkin**10 - 
                         (10*mcMS**12)/mbkin**12 + (5*mcMS**14)/mbkin**14))/
                       (9*mbkin**6) - (2*(10*mbkin**12*mcMS**2 + 522*mbkin**10*
                          mcMS**4 + 12*mbkin**8*mcMS**6 + 16*mbkin**6*mcMS**8 + 
                         1305*mbkin**4*mcMS**10 + 60*mbkin**2*mcMS**12 - 
                         35*mcMS**14)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**20)) - 24*q_cut*
                     ((-4*(3 + 8*mbkin)*(3 - (34*mcMS**2)/mbkin**2 - 
                         (42*mcMS**4)/mbkin**4 + (606*mcMS**6)/mbkin**6 + 
                         (1094*mcMS**8)/mbkin**8 + (606*mcMS**10)/mbkin**10 - 
                         (42*mcMS**12)/mbkin**12 - (34*mcMS**14)/mbkin**14 + 
                         (3*mcMS**16)/mbkin**16))/(9*mbkin**4) - 
                      (4*(17*mbkin**14*mcMS**2 + 42*mbkin**12*mcMS**4 - 
                         909*mbkin**10*mcMS**6 - 2188*mbkin**8*mcMS**8 - 
                         1515*mbkin**6*mcMS**10 + 126*mbkin**4*mcMS**12 + 
                         119*mbkin**2*mcMS**14 - 12*mcMS**16)*(-6 - 16*mbkin + 
                         12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                       (9*mbkin**20))))))*np.log((mbkin**2 + mcMS**2 - q_cut - 
                mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                    2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                    mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) + 
            (36*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                2*mcMS**2*q_cut + q_cut**2)**2*(21*mbkin**14 - 321*mbkin**12*mcMS**2 + 
               297*mbkin**10*mcMS**4 + 6483*mbkin**8*mcMS**6 + 6483*mbkin**6*
                mcMS**8 + 297*mbkin**4*mcMS**10 - 321*mbkin**2*mcMS**12 + 21*
                mcMS**14 - 30*mbkin**12*q_cut + 156*mbkin**10*mcMS**2*q_cut + 1302*
                mbkin**8*mcMS**4*q_cut + 1464*mbkin**6*mcMS**6*q_cut + 1302*mbkin**4*
                mcMS**8*q_cut + 156*mbkin**2*mcMS**10*q_cut - 30*mcMS**12*q_cut - 41*
                mbkin**10*q_cut**2 + 411*mbkin**8*mcMS**2*q_cut**2 + 1394*mbkin**6*mcMS**4*
                q_cut**2 + 1394*mbkin**4*mcMS**6*q_cut**2 + 411*mbkin**2*mcMS**8*q_cut**2 - 
               41*mcMS**10*q_cut**2 + 60*mbkin**8*q_cut**3 - 64*mbkin**6*mcMS**2*q_cut**3 - 
               568*mbkin**4*mcMS**4*q_cut**3 - 64*mbkin**2*mcMS**6*q_cut**3 + 60*mcMS**8*
                q_cut**3 + 35*mbkin**6*q_cut**4 - 139*mbkin**4*mcMS**2*q_cut**4 - 139*
                mbkin**2*mcMS**4*q_cut**4 + 35*mcMS**6*q_cut**4 - 46*mbkin**4*q_cut**5 - 28*
                mbkin**2*mcMS**2*q_cut**5 - 46*mcMS**4*q_cut**5 - 15*mbkin**2*q_cut**6 - 15*
                mcMS**2*q_cut**6 + 16*q_cut**7)*((-8*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                   mbkin**4)*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4) + 6*mbkin**2*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4) + 16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                      2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                       q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*mcMS**2*q_cut*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 9*mbkin**6*mcMS**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 
                  9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                      mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                   np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**2*q_cut*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2)))/
                (9*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)*(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                      2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) + 
               (np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*((-8*(3 + 8*mbkin)*q_cut**2)/
                   (9*mbkin**6) + (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 
                     16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                   (9*mbkin**4) + (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 
                     32*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                      np.log(mu0**2/mcMS**2)))/(9*mbkin**6))*np.log((mbkin**2 + 
                    mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                   (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4))))/(2*((-1 + mcMS**2/mbkin**2)**2 - 
                  (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4))))/
             mbkin**22) - 144*((mcMS**4*(-18*mcMS**4*(3*(-1 + mcMS**2/mbkin**2)**2*
                  (11 - (74*mcMS**2)/mbkin**2 - (234*mcMS**4)/mbkin**4 - 
                   (74*mcMS**6)/mbkin**6 + (11*mcMS**8)/mbkin**8) + 
                 ((-93 + (369*mcMS**2)/mbkin**2 + (1884*mcMS**4)/mbkin**4 + 
                    (1884*mcMS**6)/mbkin**6 + (369*mcMS**8)/mbkin**8 - 
                    (93*mcMS**10)/mbkin**10)*q_cut)/mbkin**2 + 
                 ((50 - (26*mcMS**2)/mbkin**2 - (408*mcMS**4)/mbkin**4 - 
                    (26*mcMS**6)/mbkin**6 + (50*mcMS**8)/mbkin**8)*q_cut**2)/
                  mbkin**4 + (2*(35 + (52*mcMS**2)/mbkin**2 + (52*mcMS**4)/
                     mbkin**4 + (35*mcMS**6)/mbkin**6)*q_cut**3)/mbkin**6 - 
                 ((75 + (166*mcMS**2)/mbkin**2 + (75*mcMS**4)/mbkin**4)*q_cut**4)/
                  mbkin**8 + (7*(mbkin**2 + mcMS**2)*q_cut**5)/mbkin**12 + 
                 (8*q_cut**6)/mbkin**12)*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
                 (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 
                    12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                 (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                     mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                     np.log(mu0**2/mcMS**2)))/(9*mbkin**6)) + 
               ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
                 q_cut**2/mbkin**4)*((64*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*
                      (3 - (17*mcMS**2)/mbkin**2 + (443*mcMS**4)/mbkin**4 + 
                       (1923*mcMS**6)/mbkin**6 + (188*mcMS**8)/mbkin**8 - 
                       (110*mcMS**10)/mbkin**10 + (90*mcMS**12)/mbkin**12)) + 
                    ((18 - (129*mcMS**2)/mbkin**2 + (1036*mcMS**4)/mbkin**4 + 
                       (4331*mcMS**6)/mbkin**6 + (4596*mcMS**8)/mbkin**8 + 
                       (443*mcMS**10)/mbkin**10 - (530*mcMS**12)/mbkin**12 + 
                       (315*mcMS**14)/mbkin**14)*q_cut)/mbkin**2 + 
                    ((-30 + (136*mcMS**2)/mbkin**2 - (287*mcMS**4)/mbkin**4 - 
                       (779*mcMS**6)/mbkin**6 + (165*mcMS**8)/mbkin**8 + 
                       (255*mcMS**10)/mbkin**10 - (300*mcMS**12)/mbkin**12)*q_cut**2)/
                     mbkin**4 - (2*mcMS**2*(-8 - (38*mcMS**2)/mbkin**2 + 
                       (95*mcMS**4)/mbkin**4 + (100*mcMS**6)/mbkin**6 + 
                       (75*mcMS**8)/mbkin**8)*q_cut**3)/mbkin**8 + 
                    ((45 - (39*mcMS**2)/mbkin**2 - (45*mcMS**4)/mbkin**4 + 
                       (305*mcMS**6)/mbkin**6 + (450*mcMS**8)/mbkin**8)*q_cut**4)/
                     mbkin**8 - ((42 + (7*mcMS**2)/mbkin**2 + (120*mcMS**4)/
                        mbkin**4 + (285*mcMS**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
                    (12*(mbkin**4 + 5*mcMS**4)*q_cut**6)/mbkin**16))/9 - 
                 18*(mcMS**4*(3*(-1 + mcMS**2/mbkin**2)**2*(11 - (74*mcMS**2)/
                        mbkin**2 - (234*mcMS**4)/mbkin**4 - (74*mcMS**6)/
                        mbkin**6 + (11*mcMS**8)/mbkin**8) + ((-93 + (369*mcMS**2)/
                         mbkin**2 + (1884*mcMS**4)/mbkin**4 + (1884*mcMS**6)/
                         mbkin**6 + (369*mcMS**8)/mbkin**8 - (93*mcMS**10)/
                         mbkin**10)*q_cut)/mbkin**2 + ((50 - (26*mcMS**2)/mbkin**2 - 
                        (408*mcMS**4)/mbkin**4 - (26*mcMS**6)/mbkin**6 + 
                        (50*mcMS**8)/mbkin**8)*q_cut**2)/mbkin**4 + 
                     (2*(35 + (52*mcMS**2)/mbkin**2 + (52*mcMS**4)/mbkin**4 + 
                        (35*mcMS**6)/mbkin**6)*q_cut**3)/mbkin**6 - 
                     ((75 + (166*mcMS**2)/mbkin**2 + (75*mcMS**4)/mbkin**4)*q_cut**4)/
                      mbkin**8 + (7*(mbkin**2 + mcMS**2)*q_cut**5)/mbkin**12 + 
                     (8*q_cut**6)/mbkin**12)*(16/3 + 4*np.log(mu0**2/mcMS**2)) + 
                   mcMS**4*((-64*(3 + 8*mbkin)*q_cut**6)/(3*mbkin**14) - 
                     (4*(mbkin**2 - mcMS**2)*(16*mbkin**8*mcMS**2 + 41*mbkin**6*
                         mcMS**4 - 119*mbkin**4*mcMS**6 - 69*mbkin**2*mcMS**8 + 
                        11*mcMS**10)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                         np.log(mu0**2/mcMS**2)))/mbkin**14 - (14*q_cut**5*
                       (30*mbkin**2 + 80*mbkin**3 + 36*mcMS**2 + 96*mbkin*
                         mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                         np.log(mu0**2/mcMS**2)))/(9*mbkin**14) + q_cut**4*
                      ((16*(3 + 8*mbkin)*(75 + (166*mcMS**2)/mbkin**2 + 
                          (75*mcMS**4)/mbkin**4))/(9*mbkin**10) - 
                       (4*(83*mbkin**2*mcMS**2 + 75*mcMS**4)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                        (9*mbkin**14)) + 2*q_cut**3*((-4*(3 + 8*mbkin)*(35 + 
                          (52*mcMS**2)/mbkin**2 + (52*mcMS**4)/mbkin**4 + 
                          (35*mcMS**6)/mbkin**6))/(3*mbkin**8) + 
                       (2*(52*mbkin**4*mcMS**2 + 104*mbkin**2*mcMS**4 + 
                          105*mcMS**6)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**14)) + q_cut**2*
                      ((-8*(3 + 8*mbkin)*(50 - (26*mcMS**2)/mbkin**2 - 
                          (408*mcMS**4)/mbkin**4 - (26*mcMS**6)/mbkin**6 + 
                          (50*mcMS**8)/mbkin**8))/(9*mbkin**6) - 
                       (4*(13*mbkin**6*mcMS**2 + 408*mbkin**4*mcMS**4 + 
                          39*mbkin**2*mcMS**6 - 100*mcMS**8)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                        (9*mbkin**14)) + q_cut*((-4*(3 + 8*mbkin)*(-93 + 
                          (369*mcMS**2)/mbkin**2 + (1884*mcMS**4)/mbkin**4 + 
                          (1884*mcMS**6)/mbkin**6 + (369*mcMS**8)/mbkin**8 - 
                          (93*mcMS**10)/mbkin**10))/(9*mbkin**4) + 
                       (2*(123*mbkin**8*mcMS**2 + 1256*mbkin**6*mcMS**4 + 
                          1884*mbkin**4*mcMS**6 + 492*mbkin**2*mcMS**8 - 
                          155*mcMS**10)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(3*mbkin**14))))))*
              np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 - 
            (18*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                2*mcMS**2*q_cut + q_cut**2)**2*(33*mbkin**8 - 222*mbkin**6*mcMS**2 - 702*
                mbkin**4*mcMS**4 - 222*mbkin**2*mcMS**6 + 33*mcMS**8 - 27*mbkin**6*
                q_cut - 63*mbkin**4*mcMS**2*q_cut - 63*mbkin**2*mcMS**4*q_cut - 27*mcMS**6*
                q_cut - 37*mbkin**4*q_cut**2 - 58*mbkin**2*mcMS**2*q_cut**2 - 37*mcMS**4*
                q_cut**2 + 23*mbkin**2*q_cut**3 + 23*mcMS**2*q_cut**3 + 8*q_cut**4)*
              ((mcMS**4*(16/3 + 4*np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - 
                     q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                    (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 + mcMS**4*
                ((-16*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 6*mbkin**2*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                     np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                         q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                    9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - q_cut - 
                      mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                     (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))))/(9*mbkin**4*(mbkin**4 - 
                    2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                    q_cut**2)*(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + 
                    mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                 (8*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                        np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                       mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                          q_cut**2)/mbkin**4)))**2)/(9*mbkin**6))))/mbkin**16) - 
          4320*((mcMS**8*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                  q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*((-64*mbkin)/3 - 
               (640*mcMS**2)/(9*mbkin) - (320*mcMS**4)/(3*mbkin**3) + 144*mcMS**4*
                (4 + 3*np.log(mu0**2/mcMS**2)) + (24*mcMS**6*(-6 - 16*mbkin + 
                  36*mbkin**2 + 27*mbkin**2*np.log(mu0**2/mcMS**2)))/mbkin**4)*
              np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8 + 
            (108*mcMS**4 + (108*mcMS**6)/mbkin**2)*
             ((3*mcMS**8*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                    q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
                ((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                   (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                    9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                 (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                     mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                     np.log(mu0**2/mcMS**2)))/(9*mbkin**6))*
                np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                    mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/(2*
                mbkin**8*((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/
                  mbkin**4 + q_cut**2/mbkin**4)) + ((mbkin**4 - 2*mbkin**2*mcMS**2 + 
                  mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(
                (mcMS**8*(32/3 + 8*np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - 
                      q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                          mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                         mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8 + 
                mcMS**8*((-8*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                         q_cut**2)/mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + 
                       (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                         2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 6*mbkin**2*mcMS**4*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 16*mbkin**3*
                      mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 
                     12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                        mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*
                      mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                     9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                      np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                      mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                      np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - q_cut - 
                        mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                       (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)))**2)/(3*mbkin**8*(mbkin**4 - 
                     2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                     q_cut**2)*(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + 
                     mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                  (16*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                         np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                       (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)))**3)/(9*mbkin**10))))))/
         (mbkin**2*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
             2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 7*
                mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - mbkin**2*
                q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
            (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                   (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                     2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                 mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                     2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
           3)))/180 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

def q2moment2MS(q_cut, mbkin, mcMS, mus, mu0, api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
 res = ( ((72*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
           2*mcMS**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
           7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
           mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)**2*(mbkin**10 - 
          23*mbkin**8*mcMS**2 - 398*mbkin**6*mcMS**4 - 398*mbkin**4*mcMS**6 - 
          23*mbkin**2*mcMS**8 + mcMS**10 + mbkin**8*q_cut - 20*mbkin**6*mcMS**2*q_cut - 
          102*mbkin**4*mcMS**4*q_cut - 20*mbkin**2*mcMS**6*q_cut + mcMS**8*q_cut + 
          mbkin**6*q_cut**2 - 15*mbkin**4*mcMS**2*q_cut**2 - 15*mbkin**2*mcMS**4*q_cut**2 + 
          mcMS**6*q_cut**2 - 4*mbkin**4*q_cut**3 + 2*mbkin**2*mcMS**2*q_cut**3 - 
          4*mcMS**4*q_cut**3 - 4*mbkin**2*q_cut**4 - 4*mcMS**2*q_cut**4 + 5*q_cut**5))/
        mbkin**30 - (864*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
           2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)**2*
         np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
            2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(17*mbkin**16 - 230*mbkin**14*mcMS**2 - 
          508*mbkin**12*mcMS**4 + 7790*mbkin**10*mcMS**6 + 16102*mbkin**8*mcMS**8 + 
          7790*mbkin**6*mcMS**10 - 508*mbkin**4*mcMS**12 - 230*mbkin**2*mcMS**14 + 
          17*mcMS**16 - 30*mbkin**14*q_cut + 122*mbkin**12*mcMS**2*q_cut + 
          1566*mbkin**10*mcMS**4*q_cut + 3382*mbkin**8*mcMS**6*q_cut + 
          3382*mbkin**6*mcMS**8*q_cut + 1566*mbkin**4*mcMS**10*q_cut + 
          122*mbkin**2*mcMS**12*q_cut - 30*mcMS**14*q_cut - 17*mbkin**12*q_cut**2 + 
          180*mbkin**10*mcMS**2*q_cut**2 + 2125*mbkin**8*mcMS**4*q_cut**2 + 
          3656*mbkin**6*mcMS**6*q_cut**2 + 2125*mbkin**4*mcMS**8*q_cut**2 + 
          180*mbkin**2*mcMS**10*q_cut**2 - 17*mcMS**12*q_cut**2 + 50*mbkin**10*q_cut**3 + 
          62*mbkin**8*mcMS**2*q_cut**3 - 1104*mbkin**6*mcMS**4*q_cut**3 - 
          1104*mbkin**4*mcMS**6*q_cut**3 + 62*mbkin**2*mcMS**8*q_cut**3 + 
          50*mcMS**10*q_cut**3 - 15*mbkin**8*q_cut**4 + 22*mbkin**6*mcMS**2*q_cut**4 + 
          34*mbkin**4*mcMS**4*q_cut**4 + 22*mbkin**2*mcMS**6*q_cut**4 - 15*mcMS**8*q_cut**4 - 
          2*mbkin**6*q_cut**5 - 198*mbkin**4*mcMS**2*q_cut**5 - 198*mbkin**2*mcMS**4*
           q_cut**5 - 2*mcMS**6*q_cut**5 + 5*mbkin**4*q_cut**6 + 60*mbkin**2*mcMS**2*q_cut**6 + 
          5*mcMS**4*q_cut**6 - 18*mbkin**2*q_cut**7 - 18*mcMS**2*q_cut**7 + 10*q_cut**8)*
         np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                 mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**
                4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
             np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**24 + 
       (10368*mcMS**8*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
           2*mcMS**2*q_cut + q_cut**2)**2*(31*mbkin**10 - 153*mbkin**8*mcMS**2 - 
          1138*mbkin**6*mcMS**4 - 1138*mbkin**4*mcMS**6 - 153*mbkin**2*mcMS**8 + 
          31*mcMS**10 - 29*mbkin**8*q_cut - 100*mbkin**6*mcMS**2*q_cut - 
          162*mbkin**4*mcMS**4*q_cut - 100*mbkin**2*mcMS**6*q_cut - 29*mcMS**8*q_cut - 
          29*mbkin**6*q_cut**2 - 125*mbkin**4*mcMS**2*q_cut**2 - 125*mbkin**2*mcMS**4*
           q_cut**2 - 29*mcMS**6*q_cut**2 + 26*mbkin**4*q_cut**3 + 82*mbkin**2*mcMS**2*q_cut**3 + 
          26*mcMS**4*q_cut**3 - 4*mbkin**2*q_cut**4 - 4*mcMS**2*q_cut**4 + 5*q_cut**5)*
         np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**22 - 
       (622080*mcMS**12*(3*mbkin**4 + 8*mbkin**2*mcMS**2 + 3*mcMS**4)*
         ((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
            q_cut**2)/mbkin**4)**(3/2)*np.log((mbkin**2 + mcMS**2 - q_cut - 
             mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                  q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
             mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                  q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**12)/
      (540*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
          2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
            7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
            mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**3) + 
     (((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
         q_cut**2/mbkin**4)*(-72*mbkin**2*muG*((-1 + mcMS**2/mbkin**2)**2 - 
            (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)**2*
          (-((1 + mcMS**2/mbkin**2)**2*(-6 - (33*mcMS**2)/mbkin**2 + 
              (1163*mcMS**4)/mbkin**4 - (5343*mcMS**6)/mbkin**6 + 
              (6489*mcMS**8)/mbkin**8 + (22085*mcMS**10)/mbkin**10 - 
              (10023*mcMS**12)/mbkin**12 + (771*mcMS**14)/mbkin**14 + 
              (17*mcMS**16)/mbkin**16)) + ((-10 - (115*mcMS**2)/mbkin**2 + 
              (1918*mcMS**4)/mbkin**4 - (8212*mcMS**6)/mbkin**6 - 
              (6930*mcMS**8)/mbkin**8 - (1330*mcMS**10)/mbkin**10 - 
              (11526*mcMS**12)/mbkin**12 - (5492*mcMS**14)/mbkin**14 + 
              (1428*mcMS**16)/mbkin**16 + (29*mcMS**18)/mbkin**18)*q_cut)/mbkin**2 + 
           (2*(-8 + (5*mcMS**2)/mbkin**2 + (578*mcMS**4)/mbkin**4 - 
              (3580*mcMS**6)/mbkin**6 - (14183*mcMS**8)/mbkin**8 - 
              (13151*mcMS**10)/mbkin**10 - (4092*mcMS**12)/mbkin**12 + 
              (642*mcMS**14)/mbkin**14 + (21*mcMS**16)/mbkin**16)*q_cut**2)/mbkin**4 + 
           ((30 + (192*mcMS**2)/mbkin**2 - (3912*mcMS**4)/mbkin**4 + 
              (2716*mcMS**6)/mbkin**6 + (10742*mcMS**8)/mbkin**8 + 
              (2184*mcMS**10)/mbkin**10 - (3172*mcMS**12)/mbkin**12 - 
              (84*mcMS**14)/mbkin**14)*q_cut**3)/mbkin**6 - 
           (2*(-10 + (102*mcMS**2)/mbkin**2 - (253*mcMS**4)/mbkin**4 + 
              (327*mcMS**6)/mbkin**6 - (299*mcMS**8)/mbkin**8 + (69*mcMS**10)/
               mbkin**10 + (20*mcMS**12)/mbkin**12)*q_cut**4)/mbkin**8 + 
           (2*(-23 + (43*mcMS**2)/mbkin**2 + (1111*mcMS**4)/mbkin**4 + 
              (1490*mcMS**6)/mbkin**6 + (1026*mcMS**8)/mbkin**8 + 
              (51*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
           (2*mcMS**2*(23 - (218*mcMS**2)/mbkin**2 - (158*mcMS**4)/mbkin**4 + 
              (3*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**14 - 
           (2*(-13 + (52*mcMS**2)/mbkin**2 + (186*mcMS**4)/mbkin**4 + 
              (26*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
           ((-10 + (39*mcMS**2)/mbkin**2 + (9*mcMS**4)/mbkin**4)*q_cut**8)/mbkin**16 + 
           (5*mcMS**2*q_cut**9)/mbkin**20) - 36*muG*mupi*
          ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)**2*(-((1 + mcMS**2/mbkin**2)**2*
             (-6 - (33*mcMS**2)/mbkin**2 + (1163*mcMS**4)/mbkin**4 - 
              (5343*mcMS**6)/mbkin**6 + (6489*mcMS**8)/mbkin**8 + 
              (22085*mcMS**10)/mbkin**10 - (10023*mcMS**12)/mbkin**12 + 
              (771*mcMS**14)/mbkin**14 + (17*mcMS**16)/mbkin**16)) + 
           ((-10 - (115*mcMS**2)/mbkin**2 + (1918*mcMS**4)/mbkin**4 - 
              (8212*mcMS**6)/mbkin**6 - (6930*mcMS**8)/mbkin**8 - 
              (1330*mcMS**10)/mbkin**10 - (11526*mcMS**12)/mbkin**12 - 
              (5492*mcMS**14)/mbkin**14 + (1428*mcMS**16)/mbkin**16 + 
              (29*mcMS**18)/mbkin**18)*q_cut)/mbkin**2 + 
           (2*(-8 + (5*mcMS**2)/mbkin**2 + (578*mcMS**4)/mbkin**4 - 
              (3580*mcMS**6)/mbkin**6 - (14183*mcMS**8)/mbkin**8 - 
              (13151*mcMS**10)/mbkin**10 - (4092*mcMS**12)/mbkin**12 + 
              (642*mcMS**14)/mbkin**14 + (21*mcMS**16)/mbkin**16)*q_cut**2)/mbkin**4 + 
           ((30 + (192*mcMS**2)/mbkin**2 - (3912*mcMS**4)/mbkin**4 + 
              (2716*mcMS**6)/mbkin**6 + (10742*mcMS**8)/mbkin**8 + 
              (2184*mcMS**10)/mbkin**10 - (3172*mcMS**12)/mbkin**12 - 
              (84*mcMS**14)/mbkin**14)*q_cut**3)/mbkin**6 - 
           (2*(-10 + (102*mcMS**2)/mbkin**2 - (253*mcMS**4)/mbkin**4 + 
              (327*mcMS**6)/mbkin**6 - (299*mcMS**8)/mbkin**8 + (69*mcMS**10)/
               mbkin**10 + (20*mcMS**12)/mbkin**12)*q_cut**4)/mbkin**8 + 
           (2*(-23 + (43*mcMS**2)/mbkin**2 + (1111*mcMS**4)/mbkin**4 + 
              (1490*mcMS**6)/mbkin**6 + (1026*mcMS**8)/mbkin**8 + 
              (51*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
           (2*mcMS**2*(23 - (218*mcMS**2)/mbkin**2 - (158*mcMS**4)/mbkin**4 + 
              (3*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**14 - 
           (2*(-13 + (52*mcMS**2)/mbkin**2 + (186*mcMS**4)/mbkin**4 + 
              (26*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
           ((-10 + (39*mcMS**2)/mbkin**2 + (9*mcMS**4)/mbkin**4)*q_cut**8)/mbkin**16 + 
           (5*mcMS**2*q_cut**9)/mbkin**20) + 36*muG**2*((-1 + mcMS**2/mbkin**2)**2 - 
            (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)**2*
          (-18 - (231*mcMS**2)/mbkin**2 + (1641*mcMS**4)/mbkin**4 - 
           (5542*mcMS**6)/mbkin**6 + (26*mcMS**8)/mbkin**8 + (672*mcMS**10)/
            mbkin**10 - (58660*mcMS**12)/mbkin**12 - (20842*mcMS**14)/mbkin**14 + 
           (26856*mcMS**16)/mbkin**16 - (4297*mcMS**18)/mbkin**18 - 
           (85*mcMS**20)/mbkin**20 + ((-18 - (239*mcMS**2)/mbkin**2 - 
              (530*mcMS**4)/mbkin**4 + (1020*mcMS**6)/mbkin**6 - 
              (37266*mcMS**8)/mbkin**8 + (34518*mcMS**10)/mbkin**10 - 
              (10102*mcMS**12)/mbkin**12 - (24964*mcMS**14)/mbkin**14 + 
              (7196*mcMS**16)/mbkin**16 + (145*mcMS**18)/mbkin**18)*q_cut)/mbkin**2 + 
           (2*(16 - (67*mcMS**2)/mbkin**2 - (4098*mcMS**4)/mbkin**4 + 
              (6356*mcMS**6)/mbkin**6 - (27619*mcMS**8)/mbkin**8 - 
              (24327*mcMS**10)/mbkin**10 - (7848*mcMS**12)/mbkin**12 + 
              (3554*mcMS**14)/mbkin**14 + (105*mcMS**16)/mbkin**16)*q_cut**2)/
            mbkin**4 - (2*(-51 - (400*mcMS**2)/mbkin**2 + (6112*mcMS**4)/mbkin**
                4 + (474*mcMS**6)/mbkin**6 - (16791*mcMS**8)/mbkin**8 - 
              (5188*mcMS**10)/mbkin**10 + (7926*mcMS**12)/mbkin**12 + 
              (210*mcMS**14)/mbkin**14)*q_cut**3)/mbkin**6 - 
           (2*(14 - (94*mcMS**2)/mbkin**2 - (1493*mcMS**4)/mbkin**4 + 
              (11407*mcMS**6)/mbkin**6 + (4013*mcMS**8)/mbkin**8 + 
              (681*mcMS**10)/mbkin**10 + (100*mcMS**12)/mbkin**12)*q_cut**4)/
            mbkin**8 + (2*(-75 - (97*mcMS**2)/mbkin**2 + (6039*mcMS**4)/mbkin**4 + 
              (6826*mcMS**6)/mbkin**6 + (5110*mcMS**8)/mbkin**8 + 
              (255*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
           (2*mcMS**2*(55 - (814*mcMS**2)/mbkin**2 - (694*mcMS**4)/mbkin**4 + 
              (15*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**14 + 
           (130*(mbkin**6 - 4*mbkin**4*mcMS**2 - 14*mbkin**2*mcMS**4 - 2*mcMS**6)*
             q_cut**7)/mbkin**20 + (5*(-10 + (39*mcMS**2)/mbkin**2 + 
              (9*mcMS**4)/mbkin**4)*q_cut**8)/mbkin**16 + (25*mcMS**2*q_cut**9)/
            mbkin**20) - 24*mbkin*(-((-1 + mcMS**2/mbkin**2)**4*
             (1 + mcMS**2/mbkin**2)**2*(503 - (9464*mcMS**2)/mbkin**2 + 
              (69322*mcMS**4)/mbkin**4 - (179128*mcMS**6)/mbkin**6 - 
              (217124*mcMS**8)/mbkin**8 + (134968*mcMS**10)/mbkin**10 - 
              (44170*mcMS**12)/mbkin**12 + (3064*mcMS**14)/mbkin**14 + 
              (109*mcMS**16)/mbkin**16)) + ((-1 + mcMS**2/mbkin**2)**2*
             (2903 - (41547*mcMS**2)/mbkin**2 + (196111*mcMS**4)/mbkin**4 + 
              (84389*mcMS**6)/mbkin**6 - (2226350*mcMS**8)/mbkin**8 - 
              (4060522*mcMS**10)/mbkin**10 - (2007262*mcMS**12)/mbkin**12 + 
              (332278*mcMS**14)/mbkin**14 + (144215*mcMS**16)/mbkin**16 - 
              (185931*mcMS**18)/mbkin**18 + (19663*mcMS**20)/mbkin**20 + 
              (613*mcMS**22)/mbkin**22)*q_cut)/mbkin**2 - 
           (2*(2870 - (34643*mcMS**2)/mbkin**2 + (127322*mcMS**4)/mbkin**4 + 
              (190963*mcMS**6)/mbkin**6 - (1214040*mcMS**8)/mbkin**8 - 
              (2984930*mcMS**10)/mbkin**10 - (3081596*mcMS**12)/mbkin**12 - 
              (1121550*mcMS**14)/mbkin**14 + (431494*mcMS**16)/mbkin**16 + 
              (81077*mcMS**18)/mbkin**18 - (159838*mcMS**20)/mbkin**20 + 
              (20891*mcMS**22)/mbkin**22 + (540*mcMS**24)/mbkin**24)*q_cut**2)/
            mbkin**4 - (2*(-969 + (3860*mcMS**2)/mbkin**2 + (20318*mcMS**4)/mbkin**
                4 - (143553*mcMS**6)/mbkin**6 + (43662*mcMS**8)/mbkin**8 + 
              (243236*mcMS**10)/mbkin**10 - (166416*mcMS**12)/mbkin**12 - 
              (159266*mcMS**14)/mbkin**14 + (150619*mcMS**16)/mbkin**16 + 
              (20136*mcMS**18)/mbkin**18 - (11694*mcMS**20)/mbkin**20 + 
              (67*mcMS**22)/mbkin**22)*q_cut**3)/mbkin**6 + 
           ((8985 - (76768*mcMS**2)/mbkin**2 + (136327*mcMS**4)/mbkin**4 + 
              (661168*mcMS**6)/mbkin**6 + (524386*mcMS**8)/mbkin**8 + 
              (437592*mcMS**10)/mbkin**10 + (491062*mcMS**12)/mbkin**12 - 
              (151632*mcMS**14)/mbkin**14 - (284475*mcMS**16)/mbkin**16 + 
              (57128*mcMS**18)/mbkin**18 + (2563*mcMS**20)/mbkin**20)*q_cut**4)/
            mbkin**8 - ((12407 - (70063*mcMS**2)/mbkin**2 + (19312*mcMS**4)/mbkin**
                4 + (726800*mcMS**6)/mbkin**6 + (1038918*mcMS**8)/mbkin**8 + 
              (304322*mcMS**10)/mbkin**10 - (490544*mcMS**12)/mbkin**12 - 
              (209264*mcMS**14)/mbkin**14 + (102019*mcMS**16)/mbkin**16 + 
              (1997*mcMS**18)/mbkin**18)*q_cut**5)/mbkin**10 - 
           (4*(-354 + (2543*mcMS**2)/mbkin**2 - (2380*mcMS**4)/mbkin**4 - 
              (59099*mcMS**6)/mbkin**6 - (24326*mcMS**8)/mbkin**8 + 
              (44915*mcMS**10)/mbkin**10 + (6624*mcMS**12)/mbkin**12 - 
              (8999*mcMS**14)/mbkin**14 + (564*mcMS**16)/mbkin**16)*q_cut**6)/
            mbkin**12 + (4*(2103 - (3452*mcMS**2)/mbkin**2 - (14684*mcMS**4)/
               mbkin**4 - (17263*mcMS**6)/mbkin**6 - (5615*mcMS**8)/mbkin**8 + 
              (7734*mcMS**10)/mbkin**10 + (11390*mcMS**12)/mbkin**12 + 
              (1035*mcMS**14)/mbkin**14)*q_cut**7)/mbkin**14 + 
           ((-5829 + (9694*mcMS**2)/mbkin**2 + (45611*mcMS**4)/mbkin**4 + 
              (25144*mcMS**6)/mbkin**6 - (44863*mcMS**8)/mbkin**8 - 
              (44134*mcMS**10)/mbkin**10 - (1047*mcMS**12)/mbkin**12)*q_cut**8)/
            mbkin**16 - ((343 + (6467*mcMS**2)/mbkin**2 + (16652*mcMS**4)/mbkin**
                4 - (4820*mcMS**6)/mbkin**6 - (5767*mcMS**8)/mbkin**8 + 
              (2021*mcMS**10)/mbkin**10)*q_cut**9)/mbkin**18 + 
           (2*(830 + (1771*mcMS**2)/mbkin**2 + (3242*mcMS**4)/mbkin**4 + 
              (3405*mcMS**6)/mbkin**6 + (928*mcMS**8)/mbkin**8)*q_cut**10)/mbkin**20 - 
           (2*(247 + (420*mcMS**2)/mbkin**2 + (1194*mcMS**4)/mbkin**4 + 
              (291*mcMS**6)/mbkin**6)*q_cut**11)/mbkin**22 + 
           ((3 + (92*mcMS**2)/mbkin**2 + (65*mcMS**4)/mbkin**4)*q_cut**12)/mbkin**24 - 
           ((9 + (19*mcMS**2)/mbkin**2)*q_cut**13)/mbkin**26 + (8*q_cut**14)/mbkin**28)*
          rhoD + ((mbkin**6 - 7*mbkin**4*mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - 
            mbkin**4*q_cut - mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)*
           (-16*(-((-1 + mcMS**2/mbkin**2)**4*(-2158 + (14281*mcMS**2)/mbkin**2 + 
                 (16728*mcMS**4)/mbkin**4 - (3957*mcMS**6)/mbkin**6 - 
                 (23262*mcMS**8)/mbkin**8 - (4629*mcMS**10)/mbkin**10 + 
                 (14812*mcMS**12)/mbkin**12 + (425*mcMS**14)/mbkin**14)) + 
              (4*(-1 + mcMS**2/mbkin**2)**2*(-2507 + (11037*mcMS**2)/mbkin**2 + 
                 (27788*mcMS**4)/mbkin**4 + (14892*mcMS**6)/mbkin**6 - 
                 (16983*mcMS**8)/mbkin**8 - (31701*mcMS**10)/mbkin**10 + 
                 (3740*mcMS**12)/mbkin**12 + (17724*mcMS**14)/mbkin**14 + 
                 (490*mcMS**16)/mbkin**16)*q_cut)/mbkin**2 - 
              ((-15790 + (48321*mcMS**2)/mbkin**2 + (127224*mcMS**4)/mbkin**4 + 
                 (95584*mcMS**6)/mbkin**6 + (114300*mcMS**8)/mbkin**8 - 
                 (90522*mcMS**10)/mbkin**10 - (246488*mcMS**12)/mbkin**12 + 
                 (43344*mcMS**14)/mbkin**14 + (117234*mcMS**16)/mbkin**16 + 
                 (2633*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
              ((-4746 + (2616*mcMS**2)/mbkin**2 + (37008*mcMS**4)/mbkin**4 - 
                 (24212*mcMS**6)/mbkin**6 - (142016*mcMS**8)/mbkin**8 + 
                 (4768*mcMS**10)/mbkin**10 + (173592*mcMS**12)/mbkin**12 + 
                 (47004*mcMS**14)/mbkin**14 - (894*mcMS**16)/mbkin**16)*q_cut**3)/
               mbkin**6 + (4*(-3324 - (3576*mcMS**2)/mbkin**2 + (6375*mcMS**4)/
                  mbkin**4 + (22309*mcMS**6)/mbkin**6 + (6100*mcMS**8)/mbkin**8 + 
                 (17121*mcMS**10)/mbkin**10 + (24243*mcMS**12)/mbkin**12 + 
                 (1140*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 - 
              (2*(-7473 - (14868*mcMS**2)/mbkin**2 + (13473*mcMS**4)/mbkin**4 + 
                 (68584*mcMS**6)/mbkin**6 + (112941*mcMS**8)/mbkin**8 + 
                 (61452*mcMS**10)/mbkin**10 + (483*mcMS**12)/mbkin**12)*q_cut**5)/
               mbkin**10 - (12*(347 + (965*mcMS**2)/mbkin**2 - (2210*mcMS**4)/
                  mbkin**4 - (7196*mcMS**6)/mbkin**6 - (2977*mcMS**8)/mbkin**8 + 
                 (476*mcMS**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
              (6*(-265 + (350*mcMS**2)/mbkin**2 + (1064*mcMS**4)/mbkin**4 + 
                 (2354*mcMS**6)/mbkin**6 + (1073*mcMS**8)/mbkin**8)*q_cut**7)/mbkin**
                14 - (3*(-382 + (609*mcMS**2)/mbkin**2 + (2628*mcMS**4)/
                  mbkin**4 + (917*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
              ((-214 + (368*mcMS**2)/mbkin**2 + (470*mcMS**4)/mbkin**4)*q_cut**9)/
               mbkin**18 - ((34 + (79*mcMS**2)/mbkin**2)*q_cut**10)/mbkin**20 + 
              (32*q_cut**11)/mbkin**22)*rE - ((-1 + mcMS**2/mbkin**2)**2 - 
              (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
             (-4*(-((-1 + mcMS**2/mbkin**2)**2*(-3991 + (22369*mcMS**2)/mbkin**2 + 
                   (9981*mcMS**4)/mbkin**4 - (62043*mcMS**6)/mbkin**6 - 
                   (57033*mcMS**8)/mbkin**8 - (897*mcMS**10)/mbkin**10 + 
                   (10723*mcMS**12)/mbkin**12 + (251*mcMS**14)/mbkin**14)) + 
                (6*(-1763 + (4604*mcMS**2)/mbkin**2 + (13346*mcMS**4)/mbkin**4 - 
                   (19732*mcMS**6)/mbkin**6 - (33036*mcMS**8)/mbkin**8 - 
                   (24348*mcMS**10)/mbkin**10 + (2270*mcMS**12)/mbkin**12 + 
                   (4788*mcMS**14)/mbkin**14 + (111*mcMS**16)/mbkin**16)*q_cut)/
                 mbkin**2 - (12*(-322 - (930*mcMS**2)/mbkin**2 + (741*mcMS**4)/
                    mbkin**4 - (1797*mcMS**6)/mbkin**6 + (282*mcMS**8)/mbkin**8 + 
                   (3186*mcMS**10)/mbkin**10 + (1079*mcMS**12)/mbkin**12 + 
                   mcMS**14/mbkin**14)*q_cut**2)/mbkin**4 - 
                (4*(-2673 - (2970*mcMS**2)/mbkin**2 + (618*mcMS**4)/mbkin**4 + 
                   (9080*mcMS**6)/mbkin**6 + (13851*mcMS**8)/mbkin**8 + 
                   (6882*mcMS**10)/mbkin**10 + (300*mcMS**12)/mbkin**12)*q_cut**3)/
                 mbkin**6 + (12*(-748 - (1798*mcMS**2)/mbkin**2 + (2525*mcMS**4)/
                    mbkin**4 + (5355*mcMS**6)/mbkin**6 + (2166*mcMS**8)/mbkin**8 + 
                   (36*mcMS**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
                (6*(-169 + (308*mcMS**2)/mbkin**2 - (834*mcMS**4)/mbkin**4 + 
                   (68*mcMS**6)/mbkin**6 + (211*mcMS**8)/mbkin**8)*q_cut**5)/
                 mbkin**10 - (24*(-100 + (40*mcMS**2)/mbkin**2 + (199*mcMS**4)/
                    mbkin**4 + (49*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
                (12*(-31 + (30*mcMS**2)/mbkin**2 + (21*mcMS**4)/mbkin**4)*q_cut**7)/
                 mbkin**14 + (15*(-1 + mcMS**2/mbkin**2)*q_cut**8)/mbkin**16 + 
                (8*q_cut**9)/mbkin**18)*rG - 12*(-((-1 + mcMS**2/mbkin**2)**2*
                  (773 - (1157*mcMS**2)/mbkin**2 - (30827*mcMS**4)/mbkin**4 - 
                   (37389*mcMS**6)/mbkin**6 - (23869*mcMS**8)/mbkin**8 - 
                   (7027*mcMS**10)/mbkin**10 + (8563*mcMS**12)/mbkin**12 + 
                   (213*mcMS**14)/mbkin**14)) + (2*(959 + (2370*mcMS**2)/
                    mbkin**2 - (35340*mcMS**4)/mbkin**4 - (73922*mcMS**6)/
                    mbkin**6 - (52234*mcMS**8)/mbkin**8 - (30418*mcMS**10)/
                    mbkin**10 - (4644*mcMS**12)/mbkin**12 + (11506*mcMS**14)/
                    mbkin**14 + (283*mcMS**16)/mbkin**16)*q_cut)/mbkin**2 - 
                (4*(138 + (1678*mcMS**2)/mbkin**2 - (129*mcMS**4)/mbkin**4 - 
                   (321*mcMS**6)/mbkin**6 - (426*mcMS**8)/mbkin**8 + 
                   (3978*mcMS**10)/mbkin**10 + (2637*mcMS**12)/mbkin**12 + 
                   (5*mcMS**14)/mbkin**14)*q_cut**2)/mbkin**4 - 
                (4*(456 + (2987*mcMS**2)/mbkin**2 + (4858*mcMS**4)/mbkin**4 + 
                   (9938*mcMS**6)/mbkin**6 + (9260*mcMS**8)/mbkin**8 + 
                   (5515*mcMS**10)/mbkin**10 + (250*mcMS**12)/mbkin**12)*q_cut**3)/
                 mbkin**6 + (4*(337 + (3220*mcMS**2)/mbkin**2 + (8155*mcMS**4)/
                    mbkin**4 + (10439*mcMS**6)/mbkin**6 + (5285*mcMS**8)/
                    mbkin**8 + (98*mcMS**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
                (2*(-63 - (414*mcMS**2)/mbkin**2 - (1068*mcMS**4)/mbkin**4 + 
                   (66*mcMS**6)/mbkin**6 + (479*mcMS**8)/mbkin**8)*q_cut**5)/
                 mbkin**10 - (8*(-21 + (146*mcMS**2)/mbkin**2 + (479*mcMS**4)/
                    mbkin**4 + (107*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
                (4*(-38 + (77*mcMS**2)/mbkin**2 + (35*mcMS**4)/mbkin**4)*q_cut**7)/
                 mbkin**14 + (5*(-3 + (5*mcMS**2)/mbkin**2)*q_cut**8)/mbkin**16 + 
                (8*q_cut**9)/mbkin**18)*sB - 10496*sE + (55248*mcMS**2*sE)/mbkin**
                2 + (72576*mcMS**4*sE)/mbkin**4 - (154048*mcMS**6*sE)/mbkin**6 + 
              (21120*mcMS**8*sE)/mbkin**8 - (52704*mcMS**10*sE)/mbkin**10 - 
              (24448*mcMS**12*sE)/mbkin**12 + (153024*mcMS**14*sE)/mbkin**14 - 
              (58752*mcMS**16*sE)/mbkin**16 - (1520*mcMS**18*sE)/mbkin**18 + 
              (27264*q_cut*sE)/mbkin**2 - (4128*mcMS**2*q_cut*sE)/mbkin**4 - 
              (502944*mcMS**4*q_cut*sE)/mbkin**6 - (430944*mcMS**6*q_cut*sE)/mbkin**8 - 
              (400608*mcMS**8*q_cut*sE)/mbkin**10 - (306912*mcMS**10*q_cut*sE)/mbkin**
                12 + (16800*mcMS**12*q_cut*sE)/mbkin**14 + (169056*mcMS**14*q_cut*sE)/
               mbkin**16 + (3936*mcMS**16*q_cut*sE)/mbkin**18 - (8832*q_cut**2*sE)/
               mbkin**4 - (62592*mcMS**2*q_cut**2*sE)/mbkin**6 + (51840*mcMS**4*q_cut**2*
                sE)/mbkin**8 + (108864*mcMS**6*q_cut**2*sE)/mbkin**10 - 
              (150912*mcMS**8*q_cut**2*sE)/mbkin**12 - (296064*mcMS**10*q_cut**2*sE)/
               mbkin**14 - (84096*mcMS**12*q_cut**2*sE)/mbkin**16 + (192*mcMS**14*
                q_cut**2*sE)/mbkin**18 - (28032*q_cut**3*sE)/mbkin**6 - 
              (89760*mcMS**2*q_cut**3*sE)/mbkin**8 - (92448*mcMS**4*q_cut**3*sE)/mbkin**
                10 - (12992*mcMS**6*q_cut**3*sE)/mbkin**12 - (88896*mcMS**8*q_cut**3*
                sE)/mbkin**14 - (155808*mcMS**10*q_cut**3*sE)/mbkin**16 - 
              (7200*mcMS**12*q_cut**3*sE)/mbkin**18 + (21600*q_cut**4*sE)/mbkin**8 + 
              (117600*mcMS**2*q_cut**4*sE)/mbkin**10 + (148800*mcMS**4*q_cut**4*sE)/
               mbkin**12 + (251520*mcMS**6*q_cut**4*sE)/mbkin**14 + (160992*mcMS**8*
                q_cut**4*sE)/mbkin**16 + (1632*mcMS**10*q_cut**4*sE)/mbkin**18 + 
              (2784*q_cut**5*sE)/mbkin**10 - (10752*mcMS**2*q_cut**5*sE)/mbkin**12 - 
              (28800*mcMS**4*q_cut**5*sE)/mbkin**14 - (9216*mcMS**6*q_cut**5*sE)/mbkin**
                16 + (9120*mcMS**8*q_cut**5*sE)/mbkin**18 - (3936*q_cut**6*sE)/mbkin**
                12 - (7104*mcMS**2*q_cut**6*sE)/mbkin**14 - (23904*mcMS**4*q_cut**6*sE)/
               mbkin**16 - (8256*mcMS**6*q_cut**6*sE)/mbkin**18 - (480*q_cut**7*sE)/
               mbkin**14 + (1728*mcMS**2*q_cut**7*sE)/mbkin**16 + (2208*mcMS**4*q_cut**7*
                sE)/mbkin**18 - (240*mcMS**2*q_cut**8*sE)/mbkin**18 + 
              (128*q_cut**9*sE)/mbkin**18 - 803*sqB + (1899*mcMS**2*sqB)/mbkin**2 + 
              (58452*mcMS**4*sqB)/mbkin**4 - (23284*mcMS**6*sqB)/mbkin**6 - 
              (87534*mcMS**8*sqB)/mbkin**8 + (10878*mcMS**10*sqB)/mbkin**10 + 
              (32876*mcMS**12*sqB)/mbkin**12 + (10548*mcMS**14*sqB)/mbkin**14 - 
              (2991*mcMS**16*sqB)/mbkin**16 - (41*mcMS**18*sqB)/mbkin**18 + 
              (1914*q_cut*sqB)/mbkin**2 + (8832*mcMS**2*q_cut*sqB)/mbkin**4 - 
              (110436*mcMS**4*q_cut*sqB)/mbkin**6 - (338112*mcMS**6*q_cut*sqB)/mbkin**
                8 - (279840*mcMS**8*q_cut*sqB)/mbkin**10 - (102144*mcMS**10*q_cut*sqB)/
               mbkin**12 + (5028*mcMS**12*q_cut*sqB)/mbkin**14 + (8256*mcMS**14*q_cut*
                sqB)/mbkin**16 + (102*mcMS**16*q_cut*sqB)/mbkin**18 - 
              (552*q_cut**2*sqB)/mbkin**4 - (8088*mcMS**2*q_cut**2*sqB)/mbkin**6 - 
              (5796*mcMS**4*q_cut**2*sqB)/mbkin**8 - (2556*mcMS**6*q_cut**2*sqB)/mbkin**
                10 - (24408*mcMS**8*q_cut**2*sqB)/mbkin**12 - (22248*mcMS**10*q_cut**2*
                sqB)/mbkin**14 - (3564*mcMS**12*q_cut**2*sqB)/mbkin**16 + 
              (12*mcMS**14*q_cut**2*sqB)/mbkin**18 - (1548*q_cut**3*sqB)/mbkin**6 - 
              (16692*mcMS**2*q_cut**3*sqB)/mbkin**8 - (11460*mcMS**4*q_cut**3*sqB)/mbkin**
                10 - (14288*mcMS**6*q_cut**3*sqB)/mbkin**12 - (17340*mcMS**8*q_cut**3*
                sqB)/mbkin**14 - (7932*mcMS**10*q_cut**3*sqB)/mbkin**16 - 
              (180*mcMS**12*q_cut**3*sqB)/mbkin**18 + (1164*q_cut**4*sqB)/mbkin**8 + 
              (16356*mcMS**2*q_cut**4*sqB)/mbkin**10 + (40188*mcMS**4*q_cut**4*sqB)/
               mbkin**12 + (29892*mcMS**6*q_cut**4*sqB)/mbkin**14 + (7428*mcMS**8*
                q_cut**4*sqB)/mbkin**16 + (12*mcMS**10*q_cut**4*sqB)/mbkin**18 - 
              (510*q_cut**5*sqB)/mbkin**10 - (2076*mcMS**2*q_cut**5*sqB)/mbkin**12 - 
              (5256*mcMS**4*q_cut**5*sqB)/mbkin**14 - (36*mcMS**6*q_cut**5*sqB)/mbkin**
                16 + (246*mcMS**8*q_cut**5*sqB)/mbkin**18 + (588*q_cut**6*sqB)/mbkin**
                12 - (408*mcMS**2*q_cut**6*sqB)/mbkin**14 - (1308*mcMS**4*q_cut**6*sqB)/
               mbkin**16 - (192*mcMS**6*q_cut**6*sqB)/mbkin**18 - (216*q_cut**7*sqB)/
               mbkin**14 + (192*mcMS**2*q_cut**7*sqB)/mbkin**16 + (48*mcMS**4*q_cut**7*
                sqB)/mbkin**18 - (45*q_cut**8*sqB)/mbkin**16 - (15*mcMS**2*q_cut**8*sqB)/
               mbkin**18 + (8*q_cut**9*sqB)/mbkin**18)))/mbkin**6) - 
       12*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
           2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
        (16*(-((-1 + mcMS**2/mbkin**2)**4*(-58 + (585*mcMS**2)/mbkin**2 + 
              (4376*mcMS**4)/mbkin**4 - (36483*mcMS**6)/mbkin**6 - 
              (43458*mcMS**8)/mbkin**8 + (11933*mcMS**10)/mbkin**10 + 
              (47124*mcMS**12)/mbkin**12 + (8583*mcMS**14)/mbkin**14 - 
              (26848*mcMS**16)/mbkin**16 - (2978*mcMS**18)/mbkin**18 + 
              (504*mcMS**20)/mbkin**20)) + (2*(-1 + mcMS**2/mbkin**2)**2*
             (-176 + (1403*mcMS**2)/mbkin**2 + (11699*mcMS**4)/mbkin**4 - 
              (60791*mcMS**6)/mbkin**6 - (146769*mcMS**8)/mbkin**8 - 
              (73597*mcMS**10)/mbkin**10 + (82981*mcMS**12)/mbkin**12 + 
              (120123*mcMS**14)/mbkin**14 - (8527*mcMS**16)/mbkin**16 - 
              (67374*mcMS**18)/mbkin**18 - (7400*mcMS**20)/mbkin**20 + 
              (1548*mcMS**22)/mbkin**22)*q_cut)/mbkin**2 + 
           ((770 - (4873*mcMS**2)/mbkin**2 - (39796*mcMS**4)/mbkin**4 + 
              (144026*mcMS**6)/mbkin**6 + (339380*mcMS**8)/mbkin**8 + 
              (299362*mcMS**10)/mbkin**10 + (220952*mcMS**12)/mbkin**12 - 
              (216028*mcMS**14)/mbkin**14 - (441230*mcMS**16)/mbkin**16 + 
              (32151*mcMS**18)/mbkin**18 + (233484*mcMS**20)/mbkin**20 + 
              (26162*mcMS**22)/mbkin**22 - (6840*mcMS**24)/mbkin**24)*q_cut**2)/
            mbkin**4 + (2*(-235 + (591*mcMS**2)/mbkin**2 + (7800*mcMS**4)/mbkin**
                4 - (8734*mcMS**6)/mbkin**6 - (52158*mcMS**8)/mbkin**8 + 
              (25714*mcMS**10)/mbkin**10 + (129232*mcMS**12)/mbkin**12 + 
              (2040*mcMS**14)/mbkin**14 - (152887*mcMS**16)/mbkin**16 - 
              (63857*mcMS**18)/mbkin**18 + (984*mcMS**20)/mbkin**20 + 
              (2070*mcMS**22)/mbkin**22)*q_cut**3)/mbkin**6 + 
           (2*(-435 + (981*mcMS**2)/mbkin**2 + (22494*mcMS**4)/mbkin**4 + 
              (23775*mcMS**6)/mbkin**6 - (20508*mcMS**8)/mbkin**8 - 
              (76853*mcMS**10)/mbkin**10 - (40824*mcMS**12)/mbkin**12 - 
              (68001*mcMS**14)/mbkin**14 - (91813*mcMS**16)/mbkin**16 - 
              (9258*mcMS**18)/mbkin**18 + (4050*mcMS**20)/mbkin**20)*q_cut**4)/
            mbkin**8 - (4*(-426 + (6*mcMS**2)/mbkin**2 + (14202*mcMS**4)/mbkin**
                4 + (22767*mcMS**6)/mbkin**6 - (11589*mcMS**8)/mbkin**8 - 
              (74018*mcMS**10)/mbkin**10 - (112690*mcMS**12)/mbkin**12 - 
              (63853*mcMS**14)/mbkin**14 + (2397*mcMS**16)/mbkin**16 + 
              (3888*mcMS**18)/mbkin**18)*q_cut**5)/mbkin**10 + 
           (2*(-378 - (159*mcMS**2)/mbkin**2 + (11412*mcMS**4)/mbkin**4 + 
              (16246*mcMS**6)/mbkin**6 - (37738*mcMS**8)/mbkin**8 - 
              (96055*mcMS**10)/mbkin**10 - (44068*mcMS**12)/mbkin**12 + 
              (11250*mcMS**14)/mbkin**14 + (3024*mcMS**16)/mbkin**16)*q_cut**6)/
            mbkin**12 + (4*(-165 - (738*mcMS**2)/mbkin**2 - (591*mcMS**4)/mbkin**
                4 + (940*mcMS**6)/mbkin**6 + (4641*mcMS**8)/mbkin**8 + 
              (2194*mcMS**10)/mbkin**10 + (1377*mcMS**12)/mbkin**12 + 
              (1890*mcMS**14)/mbkin**14)*q_cut**7)/mbkin**14 - 
           ((-870 - (3543*mcMS**2)/mbkin**2 + (1320*mcMS**4)/mbkin**4 + 
              (10451*mcMS**6)/mbkin**6 + (14996*mcMS**8)/mbkin**8 + 
              (19998*mcMS**10)/mbkin**10 + (8640*mcMS**12)/mbkin**12)*q_cut**8)/
            mbkin**16 + (2*(-140 - (631*mcMS**2)/mbkin**2 + (631*mcMS**4)/mbkin**
                4 + (3208*mcMS**6)/mbkin**6 + (3818*mcMS**8)/mbkin**8 + 
              (900*mcMS**10)/mbkin**10)*q_cut**9)/mbkin**18 + 
           ((-62 - (9*mcMS**2)/mbkin**2 + (476*mcMS**4)/mbkin**4 + 
              (1114*mcMS**6)/mbkin**6 + (1656*mcMS**8)/mbkin**8)*q_cut**10)/
            mbkin**20 + ((58 + (58*mcMS**2)/mbkin**2 - (788*mcMS**4)/mbkin**4 - 
              (1044*mcMS**6)/mbkin**6)*q_cut**11)/mbkin**22 - 
           (10*(mbkin**4 - 18*mcMS**4)*q_cut**12)/mbkin**28)*rE + 
         ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
           q_cut**2/mbkin**4)*((540*mcMS**2*muG**2)/mbkin**2 - (4248*mcMS**4*muG**2)/
            mbkin**4 + (7200*mcMS**6*muG**2)/mbkin**6 + (41904*mcMS**8*muG**2)/
            mbkin**8 - (183960*mcMS**10*muG**2)/mbkin**10 + 
           (687024*mcMS**12*muG**2)/mbkin**12 - (719712*mcMS**14*muG**2)/
            mbkin**14 - (252000*mcMS**16*muG**2)/mbkin**16 + 
           (549756*mcMS**18*muG**2)/mbkin**18 - (109080*mcMS**20*muG**2)/
            mbkin**20 - (22464*mcMS**22*muG**2)/mbkin**22 + (5040*mcMS**24*muG**2)/
            mbkin**24 + (180*mcMS**2*muG*mupi)/mbkin**2 - (2664*mcMS**4*muG*mupi)/
            mbkin**4 + (2736*mcMS**6*muG*mupi)/mbkin**6 + 
           (68112*mcMS**8*muG*mupi)/mbkin**8 - (257544*mcMS**10*muG*mupi)/
            mbkin**10 - (90288*mcMS**12*muG*mupi)/mbkin**12 + 
           (528192*mcMS**14*muG*mupi)/mbkin**14 - (27936*mcMS**16*muG*mupi)/
            mbkin**16 - (281772*mcMS**18*muG*mupi)/mbkin**18 + 
           (53784*mcMS**20*muG*mupi)/mbkin**20 + (8208*mcMS**22*muG*mupi)/
            mbkin**22 - (1008*mcMS**24*muG*mupi)/mbkin**24 - 
           (720*mcMS**2*muG**2*q_cut)/mbkin**4 + (3168*mcMS**4*muG**2*q_cut)/mbkin**6 - 
           (6768*mcMS**6*muG**2*q_cut)/mbkin**8 + (114336*mcMS**8*muG**2*q_cut)/
            mbkin**10 - (356544*mcMS**10*muG**2*q_cut)/mbkin**12 - 
           (417312*mcMS**12*muG**2*q_cut)/mbkin**14 - (1368000*mcMS**14*muG**2*q_cut)/
            mbkin**16 - (565344*mcMS**16*muG**2*q_cut)/mbkin**18 + 
           (387792*mcMS**18*muG**2*q_cut)/mbkin**20 + (52992*mcMS**20*muG**2*q_cut)/
            mbkin**22 - (20880*mcMS**22*muG**2*q_cut)/mbkin**24 - 
           (720*mcMS**2*muG*mupi*q_cut)/mbkin**4 + (8352*mcMS**4*muG*mupi*q_cut)/
            mbkin**6 + (1872*mcMS**6*muG*mupi*q_cut)/mbkin**8 - 
           (204480*mcMS**8*muG*mupi*q_cut)/mbkin**10 + (266400*mcMS**10*muG*mupi*
             q_cut)/mbkin**12 + (936576*mcMS**12*muG*mupi*q_cut)/mbkin**14 + 
           (1141920*mcMS**14*muG*mupi*q_cut)/mbkin**16 + 
           (230400*mcMS**16*muG*mupi*q_cut)/mbkin**18 - 
           (186768*mcMS**18*muG*mupi*q_cut)/mbkin**20 - 
           (20448*mcMS**20*muG*mupi*q_cut)/mbkin**22 + (4176*mcMS**22*muG*mupi*q_cut)/
            mbkin**24 - (720*mcMS**2*muG**2*q_cut**2)/mbkin**6 + 
           (7200*mcMS**4*muG**2*q_cut**2)/mbkin**8 + (5328*mcMS**6*muG**2*q_cut**2)/
            mbkin**10 - (113760*mcMS**8*muG**2*q_cut**2)/mbkin**12 - 
           (187920*mcMS**10*muG**2*q_cut**2)/mbkin**14 + (251136*mcMS**12*muG**2*q_cut**2)/
            mbkin**16 + (101520*mcMS**14*muG**2*q_cut**2)/mbkin**18 - 
           (235296*mcMS**16*muG**2*q_cut**2)/mbkin**20 - (30528*mcMS**18*muG**2*q_cut**2)/
            mbkin**22 + (21600*mcMS**20*muG**2*q_cut**2)/mbkin**24 + 
           (720*mcMS**2*muG*mupi*q_cut**2)/mbkin**6 - (5760*mcMS**4*muG*mupi*q_cut**2)/
            mbkin**8 - (8784*mcMS**6*muG*mupi*q_cut**2)/mbkin**10 + 
           (100224*mcMS**8*muG*mupi*q_cut**2)/mbkin**12 + 
           (92880*mcMS**10*muG*mupi*q_cut**2)/mbkin**14 - 
           (143136*mcMS**12*muG*mupi*q_cut**2)/mbkin**16 + 
           (42480*mcMS**14*muG*mupi*q_cut**2)/mbkin**18 + 
           (99072*mcMS**16*muG*mupi*q_cut**2)/mbkin**20 + 
           (8064*mcMS**18*muG*mupi*q_cut**2)/mbkin**22 - 
           (4320*mcMS**20*muG*mupi*q_cut**2)/mbkin**24 - (720*mcMS**2*muG**2*q_cut**3)/
            mbkin**8 - (1080*mcMS**4*muG**2*q_cut**3)/mbkin**10 + 
           (34776*mcMS**6*muG**2*q_cut**3)/mbkin**12 - (323064*mcMS**8*muG**2*q_cut**3)/
            mbkin**14 - (826920*mcMS**10*muG**2*q_cut**3)/mbkin**16 - 
           (943848*mcMS**12*muG**2*q_cut**3)/mbkin**18 - (345528*mcMS**14*muG**2*q_cut**3)/
            mbkin**20 + (37080*mcMS**16*muG**2*q_cut**3)/mbkin**22 + 
           (22680*mcMS**18*muG**2*q_cut**3)/mbkin**24 + (720*mcMS**2*muG*mupi*q_cut**3)/
            mbkin**8 - (8136*mcMS**4*muG*mupi*q_cut**3)/mbkin**10 - 
           (8280*mcMS**6*muG*mupi*q_cut**3)/mbkin**12 + (221400*mcMS**8*muG*mupi*
             q_cut**3)/mbkin**14 + (500616*mcMS**10*muG*mupi*q_cut**3)/mbkin**16 + 
           (469800*mcMS**12*muG*mupi*q_cut**3)/mbkin**18 + 
           (200952*mcMS**14*muG*mupi*q_cut**3)/mbkin**20 + 
           (6408*mcMS**16*muG*mupi*q_cut**3)/mbkin**22 - 
           (4536*mcMS**18*muG*mupi*q_cut**3)/mbkin**24 + (4680*mcMS**2*muG**2*q_cut**4)/
            mbkin**10 - (21240*mcMS**4*muG**2*q_cut**4)/mbkin**12 - 
           (100224*mcMS**6*muG**2*q_cut**4)/mbkin**14 + (330840*mcMS**8*muG**2*q_cut**4)/
            mbkin**16 + (918216*mcMS**10*muG**2*q_cut**4)/mbkin**18 + 
           (432792*mcMS**12*muG**2*q_cut**4)/mbkin**20 - (87840*mcMS**14*muG**2*q_cut**4)/
            mbkin**22 - (57240*mcMS**16*muG**2*q_cut**4)/mbkin**24 - 
           (1800*mcMS**2*muG*mupi*q_cut**4)/mbkin**10 + (14328*mcMS**4*muG*mupi*
             q_cut**4)/mbkin**12 + (28800*mcMS**6*muG*mupi*q_cut**4)/mbkin**14 - 
           (246456*mcMS**8*muG*mupi*q_cut**4)/mbkin**16 - 
           (465480*mcMS**10*muG*mupi*q_cut**4)/mbkin**18 - 
           (205272*mcMS**12*muG*mupi*q_cut**4)/mbkin**20 + 
           (8928*mcMS**14*muG*mupi*q_cut**4)/mbkin**22 + 
           (11448*mcMS**16*muG*mupi*q_cut**4)/mbkin**24 - (2160*mcMS**2*muG**2*q_cut**5)/
            mbkin**12 + (15048*mcMS**4*muG**2*q_cut**5)/mbkin**14 + 
           (7416*mcMS**6*muG**2*q_cut**5)/mbkin**16 - (256464*mcMS**8*muG**2*q_cut**5)/
            mbkin**18 - (269568*mcMS**10*muG**2*q_cut**5)/mbkin**20 + 
           (1224*mcMS**12*muG**2*q_cut**5)/mbkin**22 + (18360*mcMS**14*muG**2*q_cut**5)/
            mbkin**24 + (720*mcMS**2*muG*mupi*q_cut**5)/mbkin**12 - 
           (3816*mcMS**4*muG*mupi*q_cut**5)/mbkin**14 - (12024*mcMS**6*muG*mupi*
             q_cut**5)/mbkin**16 + (52272*mcMS**8*muG*mupi*q_cut**5)/mbkin**18 + 
           (39744*mcMS**10*muG*mupi*q_cut**5)/mbkin**20 - 
           (6408*mcMS**12*muG*mupi*q_cut**5)/mbkin**22 - 
           (3672*mcMS**14*muG*mupi*q_cut**5)/mbkin**24 - (3600*mcMS**2*muG**2*q_cut**6)/
            mbkin**14 + (21672*mcMS**4*muG**2*q_cut**6)/mbkin**16 + 
           (154224*mcMS**6*muG**2*q_cut**6)/mbkin**18 + (236880*mcMS**8*muG**2*q_cut**6)/
            mbkin**20 + (119808*mcMS**10*muG**2*q_cut**6)/mbkin**22 + 
           (33480*mcMS**12*muG**2*q_cut**6)/mbkin**24 + (720*mcMS**2*muG*mupi*q_cut**6)/
            mbkin**14 - (7272*mcMS**4*muG*mupi*q_cut**6)/mbkin**16 - 
           (19440*mcMS**6*muG*mupi*q_cut**6)/mbkin**18 - 
           (18864*mcMS**8*muG*mupi*q_cut**6)/mbkin**20 - 
           (18432*mcMS**10*muG*mupi*q_cut**6)/mbkin**22 - 
           (6696*mcMS**12*muG*mupi*q_cut**6)/mbkin**24 + (3600*mcMS**2*muG**2*q_cut**7)/
            mbkin**16 - (26280*mcMS**4*muG**2*q_cut**7)/mbkin**18 - 
           (98712*mcMS**6*muG**2*q_cut**7)/mbkin**20 - (78552*mcMS**8*muG**2*q_cut**7)/
            mbkin**22 - (27000*mcMS**10*muG**2*q_cut**7)/mbkin**24 - 
           (720*mcMS**2*muG*mupi*q_cut**7)/mbkin**16 + (6120*mcMS**4*muG*mupi*q_cut**7)/
            mbkin**18 + (17496*mcMS**6*muG*mupi*q_cut**7)/mbkin**20 + 
           (16056*mcMS**8*muG*mupi*q_cut**7)/mbkin**22 + 
           (5400*mcMS**10*muG*mupi*q_cut**7)/mbkin**24 - (900*mcMS**2*muG**2*q_cut**8)/
            mbkin**18 + (720*mcMS**4*muG**2*q_cut**8)/mbkin**20 + 
           (1440*mcMS**6*muG**2*q_cut**8)/mbkin**22 - (1080*mcMS**8*muG**2*q_cut**8)/
            mbkin**24 + (180*mcMS**2*muG*mupi*q_cut**8)/mbkin**18 - 
           (144*mcMS**4*muG*mupi*q_cut**8)/mbkin**20 - (1008*mcMS**6*muG*mupi*q_cut**8)/
            mbkin**22 + (216*mcMS**8*muG*mupi*q_cut**8)/mbkin**24 + 
           (6840*mcMS**4*muG**2*q_cut**9)/mbkin**22 + (6840*mcMS**6*muG**2*q_cut**9)/
            mbkin**24 - (1368*mcMS**4*muG*mupi*q_cut**9)/mbkin**22 - 
           (1368*mcMS**6*muG*mupi*q_cut**9)/mbkin**24 - (1800*mcMS**4*muG**2*q_cut**10)/
            mbkin**24 + (360*mcMS**4*muG*mupi*q_cut**10)/mbkin**24 - 
           72*mcMS**2*muG*((-1 + mcMS**2/mbkin**2)**2*(-5 + (64*mcMS**2)/mbkin**2 + 
               (57*mcMS**4)/mbkin**4 - (1842*mcMS**6)/mbkin**6 + (3413*mcMS**8)/
                mbkin**8 + (11176*mcMS**10)/mbkin**10 + (4267*mcMS**12)/
                mbkin**12 - (1866*mcMS**14)/mbkin**14 - (172*mcMS**16)/mbkin**16 + 
               (28*mcMS**18)/mbkin**18) - (4*(-5 + (58*mcMS**2)/mbkin**2 + 
                (13*mcMS**4)/mbkin**4 - (1420*mcMS**6)/mbkin**6 + (1850*mcMS**8)/
                 mbkin**8 + (6504*mcMS**10)/mbkin**10 + (7930*mcMS**12)/
                 mbkin**12 + (1600*mcMS**14)/mbkin**14 - (1297*mcMS**16)/
                 mbkin**16 - (142*mcMS**18)/mbkin**18 + (29*mcMS**20)/mbkin**20)*
               q_cut)/mbkin**2 + (4*(-5 + (40*mcMS**2)/mbkin**2 + (61*mcMS**4)/
                 mbkin**4 - (696*mcMS**6)/mbkin**6 - (645*mcMS**8)/mbkin**8 + 
                (994*mcMS**10)/mbkin**10 - (295*mcMS**12)/mbkin**12 - 
                (688*mcMS**14)/mbkin**14 - (56*mcMS**16)/mbkin**16 + 
                (30*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
             (2*(-10 + (113*mcMS**2)/mbkin**2 + (115*mcMS**4)/mbkin**4 - 
                (3075*mcMS**6)/mbkin**6 - (6953*mcMS**8)/mbkin**8 - 
                (6525*mcMS**10)/mbkin**10 - (2791*mcMS**12)/mbkin**12 - 
                (89*mcMS**14)/mbkin**14 + (63*mcMS**16)/mbkin**16)*q_cut**3)/
              mbkin**6 + ((50 - (398*mcMS**2)/mbkin**2 - (800*mcMS**4)/mbkin**4 + 
                (6846*mcMS**6)/mbkin**6 + (12930*mcMS**8)/mbkin**8 + 
                (5702*mcMS**10)/mbkin**10 - (248*mcMS**12)/mbkin**12 - 
                (318*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
             (2*(-10 + (53*mcMS**2)/mbkin**2 + (167*mcMS**4)/mbkin**4 - 
                (726*mcMS**6)/mbkin**6 - (552*mcMS**8)/mbkin**8 + (89*mcMS**10)/
                 mbkin**10 + (51*mcMS**12)/mbkin**12)*q_cut**5)/mbkin**10 + 
             (2*(-10 + (101*mcMS**2)/mbkin**2 + (270*mcMS**4)/mbkin**4 + 
                (262*mcMS**6)/mbkin**6 + (256*mcMS**8)/mbkin**8 + (93*mcMS**10)/
                 mbkin**10)*q_cut**6)/mbkin**12 - (2*(-10 + (85*mcMS**2)/mbkin**2 + 
                (243*mcMS**4)/mbkin**4 + (223*mcMS**6)/mbkin**6 + (75*mcMS**8)/
                 mbkin**8)*q_cut**7)/mbkin**14 + ((-5 + (4*mcMS**2)/mbkin**2 + 
                (28*mcMS**4)/mbkin**4 - (6*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
             (38*mcMS**2*(mbkin**2 + mcMS**2)*q_cut**9)/mbkin**22 - (10*mcMS**2*q_cut**10)/
              mbkin**22) - 4*(-((-1 + mcMS**2/mbkin**2)**2*(-97 + (735*mcMS**2)/
                 mbkin**2 + (7763*mcMS**4)/mbkin**4 - (49677*mcMS**6)/mbkin**6 - 
                (29811*mcMS**8)/mbkin**8 + (166397*mcMS**10)/mbkin**10 + 
                (165981*mcMS**12)/mbkin**12 + (5133*mcMS**14)/mbkin**14 - 
                (23212*mcMS**16)/mbkin**16 - (1628*mcMS**18)/mbkin**18 + 
                (336*mcMS**20)/mbkin**20)) + (4*(-101 + (552*mcMS**2)/mbkin**2 + 
                (5871*mcMS**4)/mbkin**4 - (19052*mcMS**6)/mbkin**6 - 
                (43986*mcMS**8)/mbkin**8 + (71160*mcMS**10)/mbkin**10 + 
                (151138*mcMS**12)/mbkin**12 + (98832*mcMS**14)/mbkin**14 - 
                (5145*mcMS**16)/mbkin**16 - (16416*mcMS**18)/mbkin**18 - 
                (1281*mcMS**20)/mbkin**20 + (348*mcMS**22)/mbkin**22)*q_cut)/
              mbkin**2 - (12*(-35 + (45*mcMS**2)/mbkin**2 + (1125*mcMS**4)/
                 mbkin**4 + (885*mcMS**6)/mbkin**6 - (2003*mcMS**8)/mbkin**8 + 
                (3409*mcMS**10)/mbkin**10 - (399*mcMS**12)/mbkin**12 - 
                (6723*mcMS**14)/mbkin**14 - (3108*mcMS**16)/mbkin**16 - 
                (36*mcMS**18)/mbkin**18 + (120*mcMS**20)/mbkin**20)*q_cut**2)/
              mbkin**4 - (4*(-111 + (258*mcMS**2)/mbkin**2 + (6843*mcMS**4)/
                 mbkin**4 + (6336*mcMS**6)/mbkin**6 - (7977*mcMS**8)/mbkin**8 - 
                (27428*mcMS**10)/mbkin**10 - (35087*mcMS**12)/mbkin**12 - 
                (15128*mcMS**14)/mbkin**14 - (660*mcMS**16)/mbkin**16 + 
                (378*mcMS**18)/mbkin**18)*q_cut**3)/mbkin**6 + 
             (2*(-561 - (651*mcMS**2)/mbkin**2 + (15192*mcMS**4)/mbkin**4 + 
                (20940*mcMS**6)/mbkin**6 - (48597*mcMS**8)/mbkin**8 - 
                (82731*mcMS**10)/mbkin**10 - (30908*mcMS**12)/mbkin**12 + 
                (2496*mcMS**14)/mbkin**14 + (1908*mcMS**16)/mbkin**16)*q_cut**4)/
              mbkin**8 - (12*(-29 - (102*mcMS**2)/mbkin**2 + (461*mcMS**4)/
                 mbkin**4 + (182*mcMS**6)/mbkin**6 - (2061*mcMS**8)/mbkin**8 - 
                (718*mcMS**10)/mbkin**10 + (405*mcMS**12)/mbkin**12 + 
                (102*mcMS**14)/mbkin**14)*q_cut**5)/mbkin**10 - 
             (4*(-171 - (513*mcMS**2)/mbkin**2 + (393*mcMS**4)/mbkin**4 + 
                (1009*mcMS**6)/mbkin**6 + (1000*mcMS**8)/mbkin**8 + 
                (1164*mcMS**10)/mbkin**10 + (558*mcMS**12)/mbkin**12)*q_cut**6)/
              mbkin**12 + (4*(-135 - (510*mcMS**2)/mbkin**2 + (273*mcMS**4)/
                 mbkin**4 + (1460*mcMS**6)/mbkin**6 + (1302*mcMS**8)/mbkin**8 + 
                (450*mcMS**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
             (3*(-13 + (69*mcMS**2)/mbkin**2 - (172*mcMS**4)/mbkin**4 - 
                (180*mcMS**6)/mbkin**6 + (24*mcMS**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
             (8*(-19 - (19*mcMS**2)/mbkin**2 + (49*mcMS**4)/mbkin**4 + 
                (57*mcMS**6)/mbkin**6)*q_cut**9)/mbkin**18 - 
             (40*(mbkin**4 - 3*mcMS**4)*q_cut**10)/mbkin**24)*rG + 
           24*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*(13 - (155*mcMS**2)/mbkin**2 + 
                (233*mcMS**4)/mbkin**4 + (9069*mcMS**6)/mbkin**6 - (59845*mcMS**8)/
                 mbkin**8 - (147269*mcMS**10)/mbkin**10 - (55861*mcMS**12)/
                 mbkin**12 + (18451*mcMS**14)/mbkin**14 - (5640*mcMS**16)/
                 mbkin**16 - (1056*mcMS**18)/mbkin**18 + (140*mcMS**20)/
                 mbkin**20)) + (4*(14 - (143*mcMS**2)/mbkin**2 + (150*mcMS**4)/
                 mbkin**4 + (6469*mcMS**6)/mbkin**6 - (22516*mcMS**8)/mbkin**8 - 
                (106182*mcMS**10)/mbkin**10 - (107880*mcMS**12)/mbkin**12 - 
                (20198*mcMS**14)/mbkin**14 + (12998*mcMS**16)/mbkin**16 - 
                (3867*mcMS**18)/mbkin**18 - (910*mcMS**20)/mbkin**20 + 
                (145*mcMS**22)/mbkin**22)*q_cut)/mbkin**2 - 
             (4*(15 - (95*mcMS**2)/mbkin**2 - (89*mcMS**4)/mbkin**4 + 
                (2989*mcMS**6)/mbkin**6 + (4773*mcMS**8)/mbkin**8 + 
                (2337*mcMS**10)/mbkin**10 + (10585*mcMS**12)/mbkin**12 + 
                (2765*mcMS**14)/mbkin**14 - (2854*mcMS**16)/mbkin**16 - 
                (416*mcMS**18)/mbkin**18 + (150*mcMS**20)/mbkin**20)*q_cut**2)/
              mbkin**4 - (2*(33 - (229*mcMS**2)/mbkin**2 - (526*mcMS**4)/
                 mbkin**4 + (11552*mcMS**6)/mbkin**6 + (19468*mcMS**8)/mbkin**8 + 
                (17570*mcMS**10)/mbkin**10 + (6430*mcMS**12)/mbkin**12 - 
                (8984*mcMS**14)/mbkin**14 - (1277*mcMS**16)/mbkin**16 + 
                (315*mcMS**18)/mbkin**18)*q_cut**3)/mbkin**6 + 
             (2*(84 - (301*mcMS**2)/mbkin**2 - (1280*mcMS**4)/mbkin**4 + 
                (11016*mcMS**6)/mbkin**6 + (26043*mcMS**8)/mbkin**8 + 
                (1603*mcMS**10)/mbkin**10 - (11796*mcMS**12)/mbkin**12 - 
                (692*mcMS**14)/mbkin**14 + (795*mcMS**16)/mbkin**16)*q_cut**4)/
              mbkin**8 - (2*(21 - (37*mcMS**2)/mbkin**2 - (145*mcMS**4)/mbkin**4 + 
                (2777*mcMS**6)/mbkin**6 + (927*mcMS**8)/mbkin**8 - (2211*mcMS**10)/
                 mbkin**10 + (413*mcMS**12)/mbkin**12 + (255*mcMS**14)/mbkin**14)*
               q_cut**5)/mbkin**10 + ((-126 + (92*mcMS**2)/mbkin**2 + (1574*mcMS**4)/
                 mbkin**4 + (2084*mcMS**6)/mbkin**6 + (2298*mcMS**8)/mbkin**8 - 
                (560*mcMS**10)/mbkin**10 - (930*mcMS**12)/mbkin**12)*q_cut**6)/
              mbkin**12 + (2*(45 - (25*mcMS**2)/mbkin**2 - (476*mcMS**4)/
                 mbkin**4 - (338*mcMS**6)/mbkin**6 + (523*mcMS**8)/mbkin**8 + 
                (375*mcMS**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
             ((21 + (77*mcMS**2)/mbkin**2 + (272*mcMS**4)/mbkin**4 - 
                (64*mcMS**6)/mbkin**6 + (30*mcMS**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
             (2*(19 + (19*mcMS**2)/mbkin**2 + (63*mcMS**4)/mbkin**4 + 
                (95*mcMS**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
             (10*(mbkin**4 + 5*mcMS**4)*q_cut**10)/mbkin**24)*rhoD + 180*sB - 
           (300*mcMS**2*sB)/mbkin**2 - (44160*mcMS**4*sB)/mbkin**4 + 
           (177168*mcMS**6*sB)/mbkin**6 + (875160*mcMS**8*sB)/mbkin**8 - 
           (698568*mcMS**10*sB)/mbkin**10 - (945120*mcMS**12*sB)/mbkin**12 + 
           (23040*mcMS**14*sB)/mbkin**14 + (261012*mcMS**16*sB)/mbkin**16 + 
           (525300*mcMS**18*sB)/mbkin**18 - (150432*mcMS**20*sB)/mbkin**20 - 
           (26640*mcMS**22*sB)/mbkin**22 + (3360*mcMS**24*sB)/mbkin**24 - 
           (720*q_cut*sB)/mbkin**2 - (2400*mcMS**2*q_cut*sB)/mbkin**4 + 
           (134064*mcMS**4*q_cut*sB)/mbkin**6 + (54816*mcMS**6*q_cut*sB)/mbkin**8 - 
           (2573472*mcMS**8*q_cut*sB)/mbkin**10 - (5234112*mcMS**10*q_cut*sB)/
            mbkin**12 - (4081632*mcMS**12*q_cut*sB)/mbkin**14 - 
           (1732416*mcMS**14*q_cut*sB)/mbkin**16 - (211728*mcMS**16*q_cut*sB)/
            mbkin**18 + (531360*mcMS**18*q_cut*sB)/mbkin**20 + 
           (66480*mcMS**20*q_cut*sB)/mbkin**22 - (13920*mcMS**22*q_cut*sB)/mbkin**24 + 
           (720*q_cut**2*sB)/mbkin**4 + (6000*mcMS**2*q_cut**2*sB)/mbkin**6 - 
           (66864*mcMS**4*q_cut**2*sB)/mbkin**8 - (258384*mcMS**6*q_cut**2*sB)/
            mbkin**10 - (20016*mcMS**8*q_cut**2*sB)/mbkin**12 + 
           (37296*mcMS**10*q_cut**2*sB)/mbkin**14 - (112944*mcMS**12*q_cut**2*sB)/
            mbkin**16 - (362832*mcMS**14*q_cut**2*sB)/mbkin**18 - 
           (303936*mcMS**16*q_cut**2*sB)/mbkin**20 - (22080*mcMS**18*q_cut**2*sB)/
            mbkin**22 + (14400*mcMS**20*q_cut**2*sB)/mbkin**24 + 
           (720*q_cut**3*sB)/mbkin**6 + (5280*mcMS**2*q_cut**3*sB)/mbkin**8 - 
           (106176*mcMS**4*q_cut**3*sB)/mbkin**10 - (524784*mcMS**6*q_cut**3*sB)/
            mbkin**12 - (866112*mcMS**8*q_cut**3*sB)/mbkin**14 - 
           (1283568*mcMS**10*q_cut**3*sB)/mbkin**16 - (1067328*mcMS**12*q_cut**3*sB)/
            mbkin**18 - (541008*mcMS**14*q_cut**3*sB)/mbkin**20 - 
           (35088*mcMS**16*q_cut**3*sB)/mbkin**22 + (15120*mcMS**18*q_cut**3*sB)/
            mbkin**24 - (1800*q_cut**4*sB)/mbkin**8 - (20040*mcMS**2*q_cut**4*sB)/
            mbkin**10 + (86640*mcMS**4*q_cut**4*sB)/mbkin**12 + 
           (603360*mcMS**6*q_cut**4*sB)/mbkin**14 + (1157832*mcMS**8*q_cut**4*sB)/
            mbkin**16 + (1188024*mcMS**10*q_cut**4*sB)/mbkin**18 + 
           (561840*mcMS**12*q_cut**4*sB)/mbkin**20 - (17472*mcMS**14*q_cut**4*sB)/
            mbkin**22 - (38160*mcMS**16*q_cut**4*sB)/mbkin**24 + 
           (720*q_cut**5*sB)/mbkin**10 + (8160*mcMS**2*q_cut**5*sB)/mbkin**12 - 
           (16608*mcMS**4*q_cut**5*sB)/mbkin**14 - (95088*mcMS**6*q_cut**5*sB)/
            mbkin**16 - (156240*mcMS**8*q_cut**5*sB)/mbkin**18 - 
           (82368*mcMS**10*q_cut**5*sB)/mbkin**20 + (31200*mcMS**12*q_cut**5*sB)/
            mbkin**22 + (12240*mcMS**14*q_cut**5*sB)/mbkin**24 + 
           (720*q_cut**6*sB)/mbkin**12 + (11760*mcMS**2*q_cut**6*sB)/mbkin**14 + 
           (45888*mcMS**4*q_cut**6*sB)/mbkin**16 + (59088*mcMS**6*q_cut**6*sB)/
            mbkin**18 + (32640*mcMS**8*q_cut**6*sB)/mbkin**20 + 
           (39360*mcMS**10*q_cut**6*sB)/mbkin**22 + (22320*mcMS**12*q_cut**6*sB)/
            mbkin**24 - (720*q_cut**7*sB)/mbkin**14 - (11040*mcMS**2*q_cut**7*sB)/
            mbkin**16 - (40224*mcMS**4*q_cut**7*sB)/mbkin**18 - 
           (52944*mcMS**6*q_cut**7*sB)/mbkin**20 - (42960*mcMS**8*q_cut**7*sB)/
            mbkin**22 - (18000*mcMS**10*q_cut**7*sB)/mbkin**24 + 
           (180*q_cut**8*sB)/mbkin**16 + (2580*mcMS**2*q_cut**8*sB)/mbkin**18 + 
           (4848*mcMS**4*q_cut**8*sB)/mbkin**20 + (3408*mcMS**6*q_cut**8*sB)/mbkin**22 - 
           (720*mcMS**8*q_cut**8*sB)/mbkin**24 + (3792*mcMS**4*q_cut**9*sB)/mbkin**22 + 
           (4560*mcMS**6*q_cut**9*sB)/mbkin**24 - (1200*mcMS**4*q_cut**10*sB)/mbkin**24 - 
           224*sE + (1168*mcMS**2*sE)/mbkin**2 + (30464*mcMS**4*sE)/mbkin**4 - 
           (154784*mcMS**6*sE)/mbkin**6 - (282656*mcMS**8*sE)/mbkin**8 + 
           (506464*mcMS**10*sE)/mbkin**10 + (114688*mcMS**12*sE)/mbkin**12 - 
           (93248*mcMS**14*sE)/mbkin**14 + (51424*mcMS**16*sE)/mbkin**16 - 
           (275184*mcMS**18*sE)/mbkin**18 + (88320*mcMS**20*sE)/mbkin**20 + 
           (15584*mcMS**22*sE)/mbkin**22 - (2016*mcMS**24*sE)/mbkin**24 + 
           (928*q_cut*sE)/mbkin**2 - (1056*mcMS**2*q_cut*sE)/mbkin**4 - 
           (98592*mcMS**4*q_cut*sE)/mbkin**6 + (53920*mcMS**6*q_cut*sE)/mbkin**8 + 
           (1396416*mcMS**8*q_cut*sE)/mbkin**10 + (1653312*mcMS**10*q_cut*sE)/
            mbkin**12 + (1115200*mcMS**12*q_cut*sE)/mbkin**14 + 
           (512448*mcMS**14*q_cut*sE)/mbkin**16 + (6816*mcMS**16*q_cut*sE)/mbkin**18 - 
           (325920*mcMS**18*q_cut*sE)/mbkin**20 - (36384*mcMS**20*q_cut*sE)/mbkin**22 + 
           (8352*mcMS**22*q_cut*sE)/mbkin**24 - (960*q_cut**2*sE)/mbkin**4 - 
           (2880*mcMS**2*q_cut**2*sE)/mbkin**6 + (53184*mcMS**4*q_cut**2*sE)/mbkin**8 + 
           (146112*mcMS**6*q_cut**2*sE)/mbkin**10 - (116352*mcMS**8*q_cut**2*sE)/
            mbkin**12 - (159168*mcMS**10*q_cut**2*sE)/mbkin**14 + 
           (324864*mcMS**12*q_cut**2*sE)/mbkin**16 + (547008*mcMS**14*q_cut**2*sE)/
            mbkin**18 + (216384*mcMS**16*q_cut**2*sE)/mbkin**20 + 
           (2688*mcMS**18*q_cut**2*sE)/mbkin**22 - (8640*mcMS**20*q_cut**2*sE)/
            mbkin**24 - (1008*q_cut**3*sE)/mbkin**6 - (1776*mcMS**2*q_cut**3*sE)/
            mbkin**8 + (89088*mcMS**4*q_cut**3*sE)/mbkin**10 + 
           (285888*mcMS**6*q_cut**3*sE)/mbkin**12 + (308256*mcMS**8*q_cut**3*sE)/
            mbkin**14 + (170720*mcMS**10*q_cut**3*sE)/mbkin**16 + 
           (181376*mcMS**12*q_cut**3*sE)/mbkin**18 + (290240*mcMS**14*q_cut**3*sE)/
            mbkin**20 + (33360*mcMS**16*q_cut**3*sE)/mbkin**22 - 
           (9072*mcMS**18*q_cut**3*sE)/mbkin**24 + (2544*q_cut**4*sE)/mbkin**8 + 
           (13344*mcMS**2*q_cut**4*sE)/mbkin**10 - (85824*mcMS**4*q_cut**4*sE)/
            mbkin**12 - (367680*mcMS**6*q_cut**4*sE)/mbkin**14 - 
           (424416*mcMS**8*q_cut**4*sE)/mbkin**16 - (497376*mcMS**10*q_cut**4*sE)/
            mbkin**18 - (324160*mcMS**12*q_cut**4*sE)/mbkin**20 + 
           (11712*mcMS**14*q_cut**4*sE)/mbkin**22 + (22896*mcMS**16*q_cut**4*sE)/
            mbkin**24 - (816*q_cut**5*sE)/mbkin**10 - (6768*mcMS**2*q_cut**5*sE)/
            mbkin**12 + (13680*mcMS**4*q_cut**5*sE)/mbkin**14 + 
           (54576*mcMS**6*q_cut**5*sE)/mbkin**16 + (68016*mcMS**8*q_cut**5*sE)/
            mbkin**18 + (26736*mcMS**10*q_cut**5*sE)/mbkin**20 - 
           (35184*mcMS**12*q_cut**5*sE)/mbkin**22 - (7344*mcMS**14*q_cut**5*sE)/
            mbkin**24 - (1488*q_cut**6*sE)/mbkin**12 - (9024*mcMS**2*q_cut**6*sE)/
            mbkin**14 - (17904*mcMS**4*q_cut**6*sE)/mbkin**16 - 
           (13888*mcMS**6*q_cut**6*sE)/mbkin**18 + (10256*mcMS**8*q_cut**6*sE)/
            mbkin**20 - (9216*mcMS**10*q_cut**6*sE)/mbkin**22 - 
           (13392*mcMS**12*q_cut**6*sE)/mbkin**24 + (1200*q_cut**7*sE)/mbkin**14 + 
           (8880*mcMS**2*q_cut**7*sE)/mbkin**16 + (19680*mcMS**4*q_cut**7*sE)/mbkin**18 + 
           (21152*mcMS**6*q_cut**7*sE)/mbkin**20 + (22512*mcMS**8*q_cut**7*sE)/
            mbkin**22 + (10800*mcMS**10*q_cut**7*sE)/mbkin**24 + 
           (48*q_cut**8*sE)/mbkin**16 - (1584*mcMS**2*q_cut**8*sE)/mbkin**18 - 
           (2784*mcMS**4*q_cut**8*sE)/mbkin**20 - (3360*mcMS**6*q_cut**8*sE)/mbkin**22 + 
           (432*mcMS**8*q_cut**8*sE)/mbkin**24 - (304*q_cut**9*sE)/mbkin**18 - 
           (304*mcMS**2*q_cut**9*sE)/mbkin**20 - (1712*mcMS**4*q_cut**9*sE)/mbkin**22 - 
           (2736*mcMS**6*q_cut**9*sE)/mbkin**24 + (80*q_cut**10*sE)/mbkin**20 + 
           (720*mcMS**4*q_cut**10*sE)/mbkin**24 - 17*sqB + (49*mcMS**2*sqB)/
            mbkin**2 + (5132*mcMS**4*sqB)/mbkin**4 - (20288*mcMS**6*sqB)/
            mbkin**6 - (146474*mcMS**8*sqB)/mbkin**8 + (48502*mcMS**10*sqB)/
            mbkin**10 + (253600*mcMS**12*sqB)/mbkin**12 - (4568*mcMS**14*sqB)/
            mbkin**14 - (119537*mcMS**16*sqB)/mbkin**16 - (24231*mcMS**18*sqB)/
            mbkin**18 + (7380*mcMS**20*sqB)/mbkin**20 + (536*mcMS**22*sqB)/
            mbkin**22 - (84*mcMS**24*sqB)/mbkin**24 + (64*q_cut*sqB)/mbkin**2 + 
           (252*mcMS**2*q_cut*sqB)/mbkin**4 - (15648*mcMS**4*q_cut*sqB)/mbkin**6 - 
           (12404*mcMS**6*q_cut*sqB)/mbkin**8 + (354816*mcMS**8*q_cut*sqB)/mbkin**10 + 
           (956568*mcMS**10*q_cut*sqB)/mbkin**12 + (871600*mcMS**12*q_cut*sqB)/
            mbkin**14 + (308568*mcMS**14*q_cut*sqB)/mbkin**16 - 
           (19776*mcMS**16*q_cut*sqB)/mbkin**18 - (24276*mcMS**18*q_cut*sqB)/
            mbkin**20 - (912*mcMS**20*q_cut*sqB)/mbkin**22 + (348*mcMS**22*q_cut*sqB)/
            mbkin**24 - (60*q_cut**2*sqB)/mbkin**4 - (660*mcMS**2*q_cut**2*sqB)/
            mbkin**6 + (7428*mcMS**4*q_cut**2*sqB)/mbkin**8 + 
           (34908*mcMS**6*q_cut**2*sqB)/mbkin**10 + (21180*mcMS**8*q_cut**2*sqB)/
            mbkin**12 + (4812*mcMS**10*q_cut**2*sqB)/mbkin**14 + 
           (61356*mcMS**12*q_cut**2*sqB)/mbkin**16 + (60252*mcMS**14*q_cut**2*sqB)/
            mbkin**18 + (13416*mcMS**16*q_cut**2*sqB)/mbkin**20 - 
           (672*mcMS**18*q_cut**2*sqB)/mbkin**22 - (360*mcMS**20*q_cut**2*sqB)/
            mbkin**24 - (54*q_cut**3*sqB)/mbkin**6 - (618*mcMS**2*q_cut**3*sqB)/
            mbkin**8 + (12876*mcMS**4*q_cut**3*sqB)/mbkin**10 + 
           (70464*mcMS**6*q_cut**3*sqB)/mbkin**12 + (94992*mcMS**8*q_cut**3*sqB)/
            mbkin**14 + (94100*mcMS**10*q_cut**3*sqB)/mbkin**16 + 
           (69044*mcMS**12*q_cut**3*sqB)/mbkin**18 + (21776*mcMS**14*q_cut**3*sqB)/
            mbkin**20 + (678*mcMS**16*q_cut**3*sqB)/mbkin**22 - 
           (378*mcMS**18*q_cut**3*sqB)/mbkin**24 + (132*q_cut**4*sqB)/mbkin**8 + 
           (2142*mcMS**2*q_cut**4*sqB)/mbkin**10 - (10176*mcMS**4*q_cut**4*sqB)/
            mbkin**12 - (80640*mcMS**6*q_cut**4*sqB)/mbkin**14 - 
           (148506*mcMS**8*q_cut**4*sqB)/mbkin**16 - (99042*mcMS**10*q_cut**4*sqB)/
            mbkin**18 - (20104*mcMS**12*q_cut**4*sqB)/mbkin**20 + 
           (2760*mcMS**14*q_cut**4*sqB)/mbkin**22 + (954*mcMS**16*q_cut**4*sqB)/
            mbkin**24 - (78*q_cut**5*sqB)/mbkin**10 - (714*mcMS**2*q_cut**5*sqB)/
            mbkin**12 + (1782*mcMS**4*q_cut**5*sqB)/mbkin**14 + 
           (14034*mcMS**6*q_cut**5*sqB)/mbkin**16 + (18006*mcMS**8*q_cut**5*sqB)/
            mbkin**18 + (1194*mcMS**10*q_cut**5*sqB)/mbkin**20 - 
           (2526*mcMS**12*q_cut**5*sqB)/mbkin**22 - (306*mcMS**14*q_cut**5*sqB)/
            mbkin**24 + (6*q_cut**6*sqB)/mbkin**12 - (1332*mcMS**2*q_cut**6*sqB)/
            mbkin**14 - (4638*mcMS**4*q_cut**6*sqB)/mbkin**16 - 
           (5260*mcMS**6*q_cut**6*sqB)/mbkin**18 - (3130*mcMS**8*q_cut**6*sqB)/
            mbkin**20 - (1488*mcMS**10*q_cut**6*sqB)/mbkin**22 - 
           (558*mcMS**12*q_cut**6*sqB)/mbkin**24 + (30*q_cut**7*sqB)/mbkin**14 + 
           (1170*mcMS**2*q_cut**7*sqB)/mbkin**16 + (4176*mcMS**4*q_cut**7*sqB)/
            mbkin**18 + (4628*mcMS**6*q_cut**7*sqB)/mbkin**20 + 
           (2202*mcMS**8*q_cut**7*sqB)/mbkin**22 + (450*mcMS**10*q_cut**7*sqB)/
            mbkin**24 - (51*q_cut**8*sqB)/mbkin**16 - (327*mcMS**2*q_cut**8*sqB)/
            mbkin**18 - (912*mcMS**4*q_cut**8*sqB)/mbkin**20 - (528*mcMS**6*q_cut**8*sqB)/
            mbkin**22 + (18*mcMS**8*q_cut**8*sqB)/mbkin**24 + (38*q_cut**9*sqB)/
            mbkin**18 + (38*mcMS**2*q_cut**9*sqB)/mbkin**20 - (50*mcMS**4*q_cut**9*sqB)/
            mbkin**22 - (114*mcMS**6*q_cut**9*sqB)/mbkin**24 - (10*q_cut**10*sqB)/
            mbkin**20 + (30*mcMS**4*q_cut**10*sqB)/mbkin**24))*
        np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
              mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
            np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*
                mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
       (144*mcMS**4*(-16*(-((-1 + mcMS**2/mbkin**2)**4*(-88 + (274*mcMS**2)/
                mbkin**2 + (4140*mcMS**4)/mbkin**4 + (919*mcMS**6)/mbkin**6 - 
               (2041*mcMS**8)/mbkin**8 - (1623*mcMS**10)/mbkin**10 + 
               (975*mcMS**12)/mbkin**12 + (504*mcMS**14)/mbkin**14)) + 
            ((-1 + mcMS**2/mbkin**2)**2*(-444 + (658*mcMS**2)/mbkin**2 + 
               (16802*mcMS**4)/mbkin**4 + (19797*mcMS**6)/mbkin**6 - 
               (2506*mcMS**8)/mbkin**8 - (12332*mcMS**10)/mbkin**10 - 
               (5424*mcMS**12)/mbkin**12 + (5337*mcMS**14)/mbkin**14 + 
               (2592*mcMS**16)/mbkin**16)*q_cut)/mbkin**2 - 
            (3*(-268 + (194*mcMS**2)/mbkin**2 + (7472*mcMS**4)/mbkin**4 + 
               (7533*mcMS**6)/mbkin**6 + (6459*mcMS**8)/mbkin**8 + (256*mcMS**10)/
                mbkin**10 - (6086*mcMS**12)/mbkin**12 - (3839*mcMS**14)/
                mbkin**14 + (3015*mcMS**16)/mbkin**16 + (1584*mcMS**18)/mbkin**18)*
              q_cut**2)/mbkin**4 + ((-438 - (218*mcMS**2)/mbkin**2 + (10234*mcMS**4)/
                mbkin**4 + (12285*mcMS**6)/mbkin**6 - (8678*mcMS**8)/mbkin**8 - 
               (11348*mcMS**10)/mbkin**10 + (5918*mcMS**12)/mbkin**12 + 
               (11121*mcMS**14)/mbkin**14 + (2484*mcMS**16)/mbkin**16)*q_cut**3)/
             mbkin**6 + ((-480 - (2270*mcMS**2)/mbkin**2 - (1602*mcMS**4)/
                mbkin**4 + (4773*mcMS**6)/mbkin**6 + (7609*mcMS**8)/mbkin**8 + 
               (2511*mcMS**10)/mbkin**10 + (5895*mcMS**12)/mbkin**12 + 
               (3240*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 - 
            (3*(-274 - (1282*mcMS**2)/mbkin**2 - (488*mcMS**4)/mbkin**4 + 
               (2601*mcMS**6)/mbkin**6 + (5406*mcMS**8)/mbkin**8 + (5847*mcMS**10)/
                mbkin**10 + (1692*mcMS**12)/mbkin**12)*q_cut**5)/mbkin**10 + 
            ((-396 - (1946*mcMS**2)/mbkin**2 + (166*mcMS**4)/mbkin**4 + 
               (7217*mcMS**6)/mbkin**6 + (9357*mcMS**8)/mbkin**8 + (1728*mcMS**10)/
                mbkin**10)*q_cut**6)/mbkin**12 + ((6 + (202*mcMS**2)/mbkin**2 + 
               (24*mcMS**4)/mbkin**4 - (405*mcMS**6)/mbkin**6 + (972*mcMS**8)/
                mbkin**8)*q_cut**7)/mbkin**14 + (48*(mbkin**6 + mbkin**4*mcMS**2 - 12*
                mbkin**2*mcMS**4 - 18*mcMS**6)*q_cut**8)/mbkin**22 - 
            (10*(mbkin**4 - 18*mcMS**4)*q_cut**9)/mbkin**22)*rE + 
          ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)*((-360*mcMS**2*muG**2)/mbkin**2 + (1872*mcMS**4*muG**2)/
             mbkin**4 - (8100*mcMS**6*muG**2)/mbkin**6 + (34668*mcMS**8*muG**2)/
             mbkin**8 - (15480*mcMS**10*muG**2)/mbkin**10 - (54720*mcMS**12*muG**2)/
             mbkin**12 + (47988*mcMS**14*muG**2)/mbkin**14 - (4860*mcMS**16*muG**2)/
             mbkin**16 - (1008*mcMS**18*muG**2)/mbkin**18 - (360*mcMS**2*muG*mupi)/
             mbkin**2 + (3168*mcMS**4*muG*mupi)/mbkin**4 - 
            (1980*mcMS**6*muG*mupi)/mbkin**6 - (34668*mcMS**8*muG*mupi)/
             mbkin**8 + (40680*mcMS**10*muG*mupi)/mbkin**10 + 
            (19440*mcMS**12*muG*mupi)/mbkin**12 - (27828*mcMS**14*muG*mupi)/
             mbkin**14 + (540*mcMS**16*muG*mupi)/mbkin**16 + 
            (1008*mcMS**18*muG*mupi)/mbkin**18 - (360*mcMS**2*muG**2*q_cut)/
             mbkin**4 + (4968*mcMS**4*muG**2*q_cut)/mbkin**6 + 
            (2124*mcMS**6*muG**2*q_cut)/mbkin**8 - (43848*mcMS**8*muG**2*q_cut)/
             mbkin**10 - (91728*mcMS**10*muG**2*q_cut)/mbkin**12 - 
            (75888*mcMS**12*muG**2*q_cut)/mbkin**14 + (20124*mcMS**14*muG**2*q_cut)/
             mbkin**16 + (3168*mcMS**16*muG**2*q_cut)/mbkin**18 + 
            (1080*mcMS**2*muG*mupi*q_cut)/mbkin**4 - (6408*mcMS**4*muG*mupi*q_cut)/
             mbkin**6 - (10764*mcMS**6*muG*mupi*q_cut)/mbkin**8 + 
            (58248*mcMS**8*muG*mupi*q_cut)/mbkin**10 + (103968*mcMS**10*muG*mupi*
              q_cut)/mbkin**12 + (45648*mcMS**12*muG*mupi*q_cut)/mbkin**14 - 
            (7164*mcMS**14*muG*mupi*q_cut)/mbkin**16 - (3168*mcMS**16*muG*mupi*q_cut)/
             mbkin**18 + (720*mcMS**2*muG**2*q_cut**2)/mbkin**6 - 
            (1440*mcMS**4*muG**2*q_cut**2)/mbkin**8 - (12384*mcMS**6*muG**2*q_cut**2)/
             mbkin**10 + (14904*mcMS**8*muG**2*q_cut**2)/mbkin**12 + 
            (30024*mcMS**10*muG**2*q_cut**2)/mbkin**14 - (14544*mcMS**12*muG**2*q_cut**2)/
             mbkin**16 - (2160*mcMS**14*muG**2*q_cut**2)/mbkin**18 - 
            (720*mcMS**2*muG*mupi*q_cut**2)/mbkin**6 + (2880*mcMS**4*muG*mupi*q_cut**2)/
             mbkin**8 + (6624*mcMS**6*muG*mupi*q_cut**2)/mbkin**10 - 
            (16344*mcMS**8*muG*mupi*q_cut**2)/mbkin**12 - 
            (15624*mcMS**10*muG*mupi*q_cut**2)/mbkin**14 + 
            (5904*mcMS**12*muG*mupi*q_cut**2)/mbkin**16 + 
            (2160*mcMS**14*muG*mupi*q_cut**2)/mbkin**18 + (2160*mcMS**2*muG**2*q_cut**3)/
             mbkin**8 - (2880*mcMS**4*muG**2*q_cut**3)/mbkin**10 - 
            (27864*mcMS**6*muG**2*q_cut**3)/mbkin**12 - (42048*mcMS**8*muG**2*q_cut**3)/
             mbkin**14 - (22104*mcMS**10*muG**2*q_cut**3)/mbkin**16 - 
            (2520*mcMS**12*muG**2*q_cut**3)/mbkin**18 - (720*mcMS**2*muG*mupi*q_cut**3)/
             mbkin**8 + (2880*mcMS**4*muG*mupi*q_cut**3)/mbkin**10 + 
            (14904*mcMS**6*muG*mupi*q_cut**3)/mbkin**12 + 
            (21888*mcMS**8*muG*mupi*q_cut**3)/mbkin**14 + 
            (13464*mcMS**10*muG*mupi*q_cut**3)/mbkin**16 + 
            (2520*mcMS**12*muG*mupi*q_cut**3)/mbkin**18 - (3240*mcMS**2*muG**2*q_cut**4)/
             mbkin**10 + (6480*mcMS**4*muG**2*q_cut**4)/mbkin**12 + 
            (38196*mcMS**6*muG**2*q_cut**4)/mbkin**14 + (30636*mcMS**8*muG**2*q_cut**4)/
             mbkin**16 + (3600*mcMS**10*muG**2*q_cut**4)/mbkin**18 + 
            (1080*mcMS**2*muG*mupi*q_cut**4)/mbkin**10 - (4320*mcMS**4*muG*mupi*
              q_cut**4)/mbkin**12 - (20916*mcMS**6*muG*mupi*q_cut**4)/mbkin**14 - 
            (17676*mcMS**8*muG*mupi*q_cut**4)/mbkin**16 - 
            (3600*mcMS**10*muG*mupi*q_cut**4)/mbkin**18 + (1080*mcMS**2*muG**2*q_cut**5)/
             mbkin**12 - (2592*mcMS**4*muG**2*q_cut**5)/mbkin**14 - 
            (8244*mcMS**6*muG**2*q_cut**5)/mbkin**16 - (432*mcMS**8*muG**2*q_cut**5)/
             mbkin**18 - (360*mcMS**2*muG*mupi*q_cut**5)/mbkin**12 + 
            (1152*mcMS**4*muG*mupi*q_cut**5)/mbkin**14 + (3924*mcMS**6*muG*mupi*
              q_cut**5)/mbkin**16 + (432*mcMS**8*muG*mupi*q_cut**5)/mbkin**18 - 
            (1008*mcMS**4*muG**2*q_cut**6)/mbkin**16 - (1008*mcMS**6*muG**2*q_cut**6)/
             mbkin**18 + (1008*mcMS**4*muG*mupi*q_cut**6)/mbkin**16 + 
            (1008*mcMS**6*muG*mupi*q_cut**6)/mbkin**18 + (360*mcMS**4*muG**2*q_cut**7)/
             mbkin**18 - (360*mcMS**4*muG*mupi*q_cut**7)/mbkin**18 + 
            72*mcMS**2*muG*((-1 + mcMS**2/mbkin**2)**2*(-10 + (68*mcMS**2)/
                 mbkin**2 + (91*mcMS**4)/mbkin**4 - (849*mcMS**6)/mbkin**6 - 
                (659*mcMS**8)/mbkin**8 + (71*mcMS**10)/mbkin**10 + (28*mcMS**12)/
                 mbkin**12) + ((30 - (178*mcMS**2)/mbkin**2 - (299*mcMS**4)/
                  mbkin**4 + (1618*mcMS**6)/mbkin**6 + (2888*mcMS**8)/mbkin**8 + 
                 (1268*mcMS**10)/mbkin**10 - (199*mcMS**12)/mbkin**12 - 
                 (88*mcMS**14)/mbkin**14)*q_cut)/mbkin**2 + 
              (2*(-10 + (40*mcMS**2)/mbkin**2 + (92*mcMS**4)/mbkin**4 - 
                 (227*mcMS**6)/mbkin**6 - (217*mcMS**8)/mbkin**8 + (82*mcMS**10)/
                  mbkin**10 + (30*mcMS**12)/mbkin**12)*q_cut**2)/mbkin**4 + 
              ((-20 + (80*mcMS**2)/mbkin**2 + (414*mcMS**4)/mbkin**4 + 
                 (608*mcMS**6)/mbkin**6 + (374*mcMS**8)/mbkin**8 + (70*mcMS**10)/
                  mbkin**10)*q_cut**3)/mbkin**6 - ((-30 + (120*mcMS**2)/mbkin**2 + 
                 (581*mcMS**4)/mbkin**4 + (491*mcMS**6)/mbkin**6 + (100*mcMS**8)/
                  mbkin**8)*q_cut**4)/mbkin**8 + ((-10 + (32*mcMS**2)/mbkin**2 + 
                 (109*mcMS**4)/mbkin**4 + (12*mcMS**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
              (28*mcMS**2*(mbkin**2 + mcMS**2)*q_cut**6)/mbkin**16 - 
              (10*mcMS**2*q_cut**7)/mbkin**16) + 4*(-((-1 + mcMS**2/mbkin**2)**2*
                (-82 - (224*mcMS**2)/mbkin**2 + (4995*mcMS**4)/mbkin**4 - 
                 (3584*mcMS**6)/mbkin**6 - (17674*mcMS**8)/mbkin**8 - 
                 (5232*mcMS**10)/mbkin**10 + (1305*mcMS**12)/mbkin**12 + 
                 (336*mcMS**14)/mbkin**14)) + ((-262 - (926*mcMS**2)/mbkin**2 + 
                 (9471*mcMS**4)/mbkin**4 + (3031*mcMS**6)/mbkin**6 - 
                 (41278*mcMS**8)/mbkin**8 - (45366*mcMS**10)/mbkin**10 - 
                 (9987*mcMS**12)/mbkin**12 + (3621*mcMS**14)/mbkin**14 + 
                 (1056*mcMS**16)/mbkin**16)*q_cut)/mbkin**2 + 
              ((180 + (996*mcMS**2)/mbkin**2 - (3076*mcMS**4)/mbkin**4 + 
                 (2042*mcMS**6)/mbkin**6 + (9992*mcMS**8)/mbkin**8 + 
                 (594*mcMS**10)/mbkin**10 - (3288*mcMS**12)/mbkin**12 - 
                 (720*mcMS**14)/mbkin**14)*q_cut**2)/mbkin**4 - 
              (2*(-110 - (688*mcMS**2)/mbkin**2 + (59*mcMS**4)/mbkin**4 + 
                 (1248*mcMS**6)/mbkin**6 + (2690*mcMS**8)/mbkin**8 + 
                 (1869*mcMS**10)/mbkin**10 + (420*mcMS**12)/mbkin**12)*q_cut**3)/
               mbkin**6 + ((-310 - (2224*mcMS**2)/mbkin**2 + (641*mcMS**4)/
                  mbkin**4 + (7288*mcMS**6)/mbkin**6 + (5757*mcMS**8)/mbkin**8 + 
                 (1200*mcMS**10)/mbkin**10)*q_cut**4)/mbkin**8 - 
              (3*(-6 - (202*mcMS**2)/mbkin**2 + (331*mcMS**4)/mbkin**4 + 
                 (485*mcMS**6)/mbkin**6 + (48*mcMS**8)/mbkin**8)*q_cut**5)/mbkin**10 - 
              (8*(-14 - (14*mcMS**2)/mbkin**2 + (33*mcMS**4)/mbkin**4 + 
                 (42*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 - (40*(mbkin**4 - 
                 3*mcMS**4)*q_cut**7)/mbkin**18)*rG - 24*mbkin*
             (-((-1 + mcMS**2/mbkin**2)**2*(-2 - (44*mcMS**2)/mbkin**2 + 
                 (729*mcMS**4)/mbkin**4 + (9244*mcMS**6)/mbkin**6 + 
                 (10134*mcMS**8)/mbkin**8 + (36*mcMS**10)/mbkin**10 - 
                 (77*mcMS**12)/mbkin**12 + (140*mcMS**14)/mbkin**14)) + 
              ((-2 - (226*mcMS**2)/mbkin**2 + (1025*mcMS**4)/mbkin**4 + 
                 (19673*mcMS**6)/mbkin**6 + (39598*mcMS**8)/mbkin**8 + 
                 (21350*mcMS**10)/mbkin**10 - (861*mcMS**12)/mbkin**12 - 
                 (357*mcMS**14)/mbkin**14 + (440*mcMS**16)/mbkin**16)*q_cut)/mbkin**
                2 - (2*mcMS**2*(-88 - (232*mcMS**2)/mbkin**2 + (2115*mcMS**4)/
                  mbkin**4 + (1978*mcMS**6)/mbkin**6 - (633*mcMS**8)/mbkin**8 + 
                 (70*mcMS**10)/mbkin**10 + (150*mcMS**12)/mbkin**12)*q_cut**2)/mbkin**
                6 + ((-10 + (216*mcMS**2)/mbkin**2 + (1252*mcMS**4)/mbkin**4 + 
                 (972*mcMS**6)/mbkin**6 + (866*mcMS**8)/mbkin**8 - (370*mcMS**10)/
                  mbkin**10 - (350*mcMS**12)/mbkin**12)*q_cut**3)/mbkin**6 + 
              ((10 - (284*mcMS**2)/mbkin**2 - (2029*mcMS**4)/mbkin**4 - 
                 (1764*mcMS**6)/mbkin**6 + (735*mcMS**8)/mbkin**8 + (500*mcMS**10)/
                  mbkin**10)*q_cut**4)/mbkin**8 + ((18 + (106*mcMS**2)/mbkin**2 + 
                 (601*mcMS**4)/mbkin**4 - (157*mcMS**6)/mbkin**6 - (60*mcMS**8)/
                  mbkin**8)*q_cut**5)/mbkin**10 - (4*(7 + (7*mcMS**2)/mbkin**2 + 
                 (17*mcMS**4)/mbkin**4 + (35*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
              (10*(mbkin**4 + 5*mcMS**4)*q_cut**7)/mbkin**18)*rhoD - 360*sB - 
            (1920*mcMS**2*sB)/mbkin**2 + (45204*mcMS**4*sB)/mbkin**4 + 
            (44976*mcMS**6*sB)/mbkin**6 - (128580*mcMS**8*sB)/mbkin**8 - 
            (16800*mcMS**10*sB)/mbkin**10 + (19980*mcMS**12*sB)/mbkin**12 + 
            (41616*mcMS**14*sB)/mbkin**14 - (756*mcMS**16*sB)/mbkin**16 - 
            (3360*mcMS**18*sB)/mbkin**18 + (1080*q_cut*sB)/mbkin**2 + 
            (10440*mcMS**2*q_cut*sB)/mbkin**4 - (74364*mcMS**4*q_cut*sB)/mbkin**6 - 
            (348228*mcMS**6*q_cut*sB)/mbkin**8 - (436248*mcMS**8*q_cut*sB)/mbkin**10 - 
            (214728*mcMS**10*q_cut*sB)/mbkin**12 - (55908*mcMS**12*q_cut*sB)/
             mbkin**14 + (18756*mcMS**14*q_cut*sB)/mbkin**16 + 
            (10560*mcMS**16*q_cut*sB)/mbkin**18 - (720*q_cut**2*sB)/mbkin**4 - 
            (8880*mcMS**2*q_cut**2*sB)/mbkin**6 + (13200*mcMS**4*q_cut**2*sB)/mbkin**8 + 
            (67512*mcMS**6*q_cut**2*sB)/mbkin**10 + (40176*mcMS**8*q_cut**2*sB)/
             mbkin**12 + (5832*mcMS**10*q_cut**2*sB)/mbkin**14 - 
            (19200*mcMS**12*q_cut**2*sB)/mbkin**16 - (7200*mcMS**14*q_cut**2*sB)/
             mbkin**18 - (720*q_cut**3*sB)/mbkin**6 - (11040*mcMS**2*q_cut**3*sB)/
             mbkin**8 - (26280*mcMS**4*q_cut**3*sB)/mbkin**10 - 
            (41328*mcMS**6*q_cut**3*sB)/mbkin**12 - (50448*mcMS**8*q_cut**3*sB)/
             mbkin**14 - (30120*mcMS**10*q_cut**3*sB)/mbkin**16 - 
            (8400*mcMS**12*q_cut**3*sB)/mbkin**18 + (1080*q_cut**4*sB)/mbkin**8 + 
            (16560*mcMS**2*q_cut**4*sB)/mbkin**10 + (43980*mcMS**4*q_cut**4*sB)/
             mbkin**12 + (60792*mcMS**6*q_cut**4*sB)/mbkin**14 + 
            (44100*mcMS**8*q_cut**4*sB)/mbkin**16 + (12000*mcMS**10*q_cut**4*sB)/
             mbkin**18 - (360*q_cut**5*sB)/mbkin**10 - (5160*mcMS**2*q_cut**5*sB)/
             mbkin**12 - (9804*mcMS**4*q_cut**5*sB)/mbkin**14 - 
            (10284*mcMS**6*q_cut**5*sB)/mbkin**16 - (1440*mcMS**8*q_cut**5*sB)/
             mbkin**18 - (2496*mcMS**4*q_cut**6*sB)/mbkin**16 - 
            (3360*mcMS**6*q_cut**6*sB)/mbkin**18 + (1200*mcMS**4*q_cut**7*sB)/mbkin**18 + 
            224*sE + (1920*mcMS**2*sE)/mbkin**2 - (28320*mcMS**4*sE)/mbkin**4 - 
            (3568*mcMS**6*sE)/mbkin**6 + (70704*mcMS**8*sE)/mbkin**8 - 
            (36384*mcMS**10*sE)/mbkin**10 + (7520*mcMS**12*sE)/mbkin**12 - 
            (14160*mcMS**14*sE)/mbkin**14 + (48*mcMS**16*sE)/mbkin**16 + 
            (2016*mcMS**18*sE)/mbkin**18 - (704*q_cut*sE)/mbkin**2 - 
            (8032*mcMS**2*q_cut*sE)/mbkin**4 + (45120*mcMS**4*q_cut*sE)/mbkin**6 + 
            (159920*mcMS**6*q_cut*sE)/mbkin**8 + (127264*mcMS**8*q_cut*sE)/mbkin**10 + 
            (38208*mcMS**10*q_cut*sE)/mbkin**12 + (13536*mcMS**12*q_cut*sE)/mbkin**14 - 
            (11856*mcMS**14*q_cut*sE)/mbkin**16 - (6336*mcMS**16*q_cut*sE)/mbkin**18 + 
            (480*q_cut**2*sE)/mbkin**4 + (6432*mcMS**2*q_cut**2*sE)/mbkin**6 - 
            (10592*mcMS**4*q_cut**2*sE)/mbkin**8 - (34400*mcMS**6*q_cut**2*sE)/
             mbkin**10 - (4544*mcMS**8*q_cut**2*sE)/mbkin**12 + 
            (17856*mcMS**10*q_cut**2*sE)/mbkin**14 + (17568*mcMS**12*q_cut**2*sE)/
             mbkin**16 + (4320*mcMS**14*q_cut**2*sE)/mbkin**18 + (560*q_cut**3*sE)/
             mbkin**6 + (7552*mcMS**2*q_cut**3*sE)/mbkin**8 + (12304*mcMS**4*q_cut**3*sE)/
             mbkin**10 + (10944*mcMS**6*q_cut**3*sE)/mbkin**12 + 
            (1840*mcMS**8*q_cut**3*sE)/mbkin**14 + (8448*mcMS**10*q_cut**3*sE)/
             mbkin**16 + (5040*mcMS**12*q_cut**3*sE)/mbkin**18 - (800*q_cut**4*sE)/
             mbkin**8 - (11648*mcMS**2*q_cut**4*sE)/mbkin**10 - 
            (20288*mcMS**4*q_cut**4*sE)/mbkin**12 - (18928*mcMS**6*q_cut**4*sE)/
             mbkin**14 - (21552*mcMS**8*q_cut**4*sE)/mbkin**16 - 
            (7200*mcMS**10*q_cut**4*sE)/mbkin**18 + (96*q_cut**5*sE)/mbkin**10 + 
            (3552*mcMS**2*q_cut**5*sE)/mbkin**12 + (4032*mcMS**4*q_cut**5*sE)/mbkin**14 + 
            (6480*mcMS**6*q_cut**5*sE)/mbkin**16 + (864*mcMS**8*q_cut**5*sE)/mbkin**18 + 
            (224*q_cut**6*sE)/mbkin**12 + (224*mcMS**2*q_cut**6*sE)/mbkin**14 + 
            (864*mcMS**4*q_cut**6*sE)/mbkin**16 + (2016*mcMS**6*q_cut**6*sE)/mbkin**18 - 
            (80*q_cut**7*sE)/mbkin**14 - (720*mcMS**4*q_cut**7*sE)/mbkin**18 + 62*sqB - 
            (5685*mcMS**4*sqB)/mbkin**4 - (10018*mcMS**6*sqB)/mbkin**6 + 
            (16491*mcMS**8*sqB)/mbkin**8 + (15108*mcMS**10*sqB)/mbkin**10 - 
            (11035*mcMS**12*sqB)/mbkin**12 - (5430*mcMS**14*sqB)/mbkin**14 + 
            (423*mcMS**16*sqB)/mbkin**16 + (84*mcMS**18*sqB)/mbkin**18 - 
            (182*q_cut*sqB)/mbkin**2 - (886*mcMS**2*q_cut*sqB)/mbkin**4 + 
            (9915*mcMS**4*q_cut*sqB)/mbkin**6 + (53819*mcMS**6*q_cut*sqB)/mbkin**8 + 
            (84730*mcMS**8*q_cut*sqB)/mbkin**10 + (49362*mcMS**10*q_cut*sqB)/
             mbkin**12 + (7041*mcMS**12*q_cut*sqB)/mbkin**14 - 
            (1935*mcMS**14*q_cut*sqB)/mbkin**16 - (264*mcMS**16*q_cut*sqB)/mbkin**18 + 
            (120*q_cut**2*sqB)/mbkin**4 + (936*mcMS**2*q_cut**2*sqB)/mbkin**6 - 
            (1256*mcMS**4*q_cut**2*sqB)/mbkin**8 - (10490*mcMS**6*q_cut**2*sqB)/
             mbkin**10 - (8804*mcMS**8*q_cut**2*sqB)/mbkin**12 + 
            (702*mcMS**10*q_cut**2*sqB)/mbkin**14 + (1812*mcMS**12*q_cut**2*sqB)/
             mbkin**16 + (180*mcMS**14*q_cut**2*sqB)/mbkin**18 + (110*q_cut**3*sqB)/
             mbkin**6 + (1336*mcMS**2*q_cut**3*sqB)/mbkin**8 + 
            (3232*mcMS**4*q_cut**3*sqB)/mbkin**10 + (3012*mcMS**6*q_cut**3*sqB)/
             mbkin**12 + (2938*mcMS**8*q_cut**3*sqB)/mbkin**14 + 
            (1482*mcMS**10*q_cut**3*sqB)/mbkin**16 + (210*mcMS**12*q_cut**3*sqB)/
             mbkin**18 - (170*q_cut**4*sqB)/mbkin**8 - (1964*mcMS**2*q_cut**4*sqB)/
             mbkin**10 - (5759*mcMS**4*q_cut**4*sqB)/mbkin**12 - 
            (6484*mcMS**6*q_cut**4*sqB)/mbkin**14 - (2763*mcMS**8*q_cut**4*sqB)/
             mbkin**16 - (300*mcMS**10*q_cut**4*sqB)/mbkin**18 + (78*q_cut**5*sqB)/
             mbkin**10 + (606*mcMS**2*q_cut**5*sqB)/mbkin**12 + 
            (1611*mcMS**4*q_cut**5*sqB)/mbkin**14 + (969*mcMS**6*q_cut**5*sqB)/
             mbkin**16 + (36*mcMS**8*q_cut**5*sqB)/mbkin**18 - (28*q_cut**6*sqB)/
             mbkin**12 - (28*mcMS**2*q_cut**6*sqB)/mbkin**14 + (12*mcMS**4*q_cut**6*sqB)/
             mbkin**16 + (84*mcMS**6*q_cut**6*sqB)/mbkin**18 + (10*q_cut**7*sqB)/
             mbkin**14 - (30*mcMS**4*q_cut**7*sqB)/mbkin**18))*
         np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 - 
       (8640*mcMS**8*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
            2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(72*mcMS**2*muG - 
          (144*mcMS**4*muG)/mbkin**2 - (432*mcMS**6*muG)/mbkin**4 - 
          (36*mcMS**2*muG**2)/mbkin**2 + (72*mcMS**4*muG**2)/mbkin**4 + 
          (216*mcMS**6*muG**2)/mbkin**6 + (36*mcMS**2*muG*mupi)/mbkin**2 - 
          (72*mcMS**4*muG*mupi)/mbkin**4 - (216*mcMS**6*muG*mupi)/mbkin**6 + 
          (96 + (368*mcMS**2)/mbkin**2 - (192*mcMS**4)/mbkin**4)*rE + 
          4*(3 - (35*mcMS**2)/mbkin**2 + (78*mcMS**4)/mbkin**4 + 
            (66*mcMS**6)/mbkin**6)*rG + 72*mbkin*rhoD + (408*mcMS**2*rhoD)/
           mbkin + (1680*mcMS**4*rhoD)/mbkin**3 + (528*mcMS**6*rhoD)/mbkin**5 + 
          36*sB + (516*mcMS**2*sB)/mbkin**2 + (696*mcMS**4*sB)/mbkin**4 + 
          (264*mcMS**6*sB)/mbkin**6 - (304*mcMS**2*sE)/mbkin**2 - 
          (192*mcMS**4*sE)/mbkin**4 - 9*sqB - (67*mcMS**2*sqB)/mbkin**2 - 
          (138*mcMS**4*sqB)/mbkin**4 - (66*mcMS**6*sqB)/mbkin**6)*
         np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8)/
      (540*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
          2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
            7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
            mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**3) + 
     (api4*(((72*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
              7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
              mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)**2*(mbkin**10 - 
             23*mbkin**8*mcMS**2 - 398*mbkin**6*mcMS**4 - 398*mbkin**4*mcMS**6 - 
             23*mbkin**2*mcMS**8 + mcMS**10 + mbkin**8*q_cut - 20*mbkin**6*mcMS**2*
              q_cut - 102*mbkin**4*mcMS**4*q_cut - 20*mbkin**2*mcMS**6*q_cut + mcMS**8*q_cut + 
             mbkin**6*q_cut**2 - 15*mbkin**4*mcMS**2*q_cut**2 - 15*mbkin**2*mcMS**4*q_cut**2 + 
             mcMS**6*q_cut**2 - 4*mbkin**4*q_cut**3 + 2*mbkin**2*mcMS**2*q_cut**3 - 
             4*mcMS**4*q_cut**3 - 4*mbkin**2*q_cut**4 - 4*mcMS**2*q_cut**4 + 5*q_cut**5))/
           mbkin**30 - (864*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
              2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)**2*
            np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*
                mcMS**2*q_cut + q_cut**2)/mbkin**4)*(17*mbkin**16 - 230*mbkin**14*
              mcMS**2 - 508*mbkin**12*mcMS**4 + 7790*mbkin**10*mcMS**6 + 
             16102*mbkin**8*mcMS**8 + 7790*mbkin**6*mcMS**10 - 
             508*mbkin**4*mcMS**12 - 230*mbkin**2*mcMS**14 + 17*mcMS**16 - 
             30*mbkin**14*q_cut + 122*mbkin**12*mcMS**2*q_cut + 1566*mbkin**10*mcMS**4*
              q_cut + 3382*mbkin**8*mcMS**6*q_cut + 3382*mbkin**6*mcMS**8*q_cut + 
             1566*mbkin**4*mcMS**10*q_cut + 122*mbkin**2*mcMS**12*q_cut - 
             30*mcMS**14*q_cut - 17*mbkin**12*q_cut**2 + 180*mbkin**10*mcMS**2*q_cut**2 + 
             2125*mbkin**8*mcMS**4*q_cut**2 + 3656*mbkin**6*mcMS**6*q_cut**2 + 
             2125*mbkin**4*mcMS**8*q_cut**2 + 180*mbkin**2*mcMS**10*q_cut**2 - 
             17*mcMS**12*q_cut**2 + 50*mbkin**10*q_cut**3 + 62*mbkin**8*mcMS**2*q_cut**3 - 
             1104*mbkin**6*mcMS**4*q_cut**3 - 1104*mbkin**4*mcMS**6*q_cut**3 + 
             62*mbkin**2*mcMS**8*q_cut**3 + 50*mcMS**10*q_cut**3 - 15*mbkin**8*q_cut**4 + 
             22*mbkin**6*mcMS**2*q_cut**4 + 34*mbkin**4*mcMS**4*q_cut**4 + 
             22*mbkin**2*mcMS**6*q_cut**4 - 15*mcMS**8*q_cut**4 - 2*mbkin**6*q_cut**5 - 
             198*mbkin**4*mcMS**2*q_cut**5 - 198*mbkin**2*mcMS**4*q_cut**5 - 
             2*mcMS**6*q_cut**5 + 5*mbkin**4*q_cut**6 + 60*mbkin**2*mcMS**2*q_cut**6 + 
             5*mcMS**4*q_cut**6 - 18*mbkin**2*q_cut**7 - 18*mcMS**2*q_cut**7 + 10*q_cut**8)*
            np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                  mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**24 + 
          (10368*mcMS**8*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**2*(31*mbkin**10 - 153*mbkin**8*mcMS**2 - 
             1138*mbkin**6*mcMS**4 - 1138*mbkin**4*mcMS**6 - 153*mbkin**2*mcMS**8 + 
             31*mcMS**10 - 29*mbkin**8*q_cut - 100*mbkin**6*mcMS**2*q_cut - 
             162*mbkin**4*mcMS**4*q_cut - 100*mbkin**2*mcMS**6*q_cut - 29*mcMS**8*q_cut - 
             29*mbkin**6*q_cut**2 - 125*mbkin**4*mcMS**2*q_cut**2 - 125*mbkin**2*mcMS**4*
              q_cut**2 - 29*mcMS**6*q_cut**2 + 26*mbkin**4*q_cut**3 + 82*mbkin**2*mcMS**2*
              q_cut**3 + 26*mcMS**4*q_cut**3 - 4*mbkin**2*q_cut**4 - 4*mcMS**2*q_cut**4 + 
             5*q_cut**5)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                  (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                    2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/
           mbkin**22 - (622080*mcMS**12*(3*mbkin**4 + 8*mbkin**2*mcMS**2 + 
             3*mcMS**4)*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
               2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**12)*
         ((-3*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
             (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
             (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
              (9*mbkin**6)))/(2*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*
                mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
             q_cut**2/mbkin**4)*((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                   2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 
                 7*mbkin**4*mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - 
                 mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
              (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                       2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                       2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**
                4)**3) - (3*((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                  2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 
                7*mbkin**4*mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - 
                mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)*(
                (-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                  (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                   9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                   12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                 (9*mbkin**6)))/(2*mbkin**6*((-1 + mcMS**2/mbkin**2)**2 - 
                (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)) + 
             np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*((-4*(3 + 8*mbkin)*q_cut**3)/
                (3*mbkin**8) - (14*mcMS**2*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) - (28*mcMS**4*
                 (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**6) + (2*mcMS**6*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(3*mbkin**8) + (2*q_cut**2*
                 (12*mbkin**2 + 32*mbkin**3 + 18*mcMS**2 + 48*mbkin*mcMS**2 - 
                  12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**8) + (4*q_cut*(3*mbkin**4 + 8*mbkin**5 + 9*mcMS**4 + 
                  24*mbkin*mcMS**4 - 12*mbkin**2*mcMS**4 - 9*mbkin**2*mcMS**4*
                   np.log(mu0**2/mcMS**2)))/(9*mbkin**8)) - 
             12*((mcMS**4*(16/3 + 4*np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - 
                    q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                   (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4))))/mbkin**4 + mcMS**4*
                ((-8*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 6*mbkin**2*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                     np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                         q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                    9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2)))/(9*mbkin**4*(mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)*
                   (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + mbkin**2*
                     np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                         q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                 (8*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                      mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                          q_cut**2)/mbkin**4))))/(9*mbkin**6)))))/
           (((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                q_cut + q_cut**2)/mbkin**4)**(3/2)*
            ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
                 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
                 mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
              (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                       2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                       2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**
                4)**4)) + (72*mbkin**4*((-1 + mcMS**2/mbkin**2)**2*
              (1 - (7*mcMS**2)/mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/
                mbkin**6) + ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + 
                (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/mbkin**2 + 
             (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*mcMS**4 + mcMS**6)*q_cut**
                2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/
              mbkin**10 - (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + 
             q_cut**5/mbkin**10)**2*(1 - (23*mcMS**2)/mbkin**2 - (398*mcMS**4)/
             mbkin**4 - (398*mcMS**6)/mbkin**6 - (23*mcMS**8)/mbkin**8 + 
            mcMS**10/mbkin**10 + ((mbkin**8 - 20*mbkin**6*mcMS**2 - 102*mbkin**4*
                mcMS**4 - 20*mbkin**2*mcMS**6 + mcMS**8)*q_cut)/mbkin**10 + 
            ((mbkin**6 - 15*mbkin**4*mcMS**2 - 15*mbkin**2*mcMS**4 + mcMS**6)*q_cut**2)/
             mbkin**10 + ((-4 + (2*mcMS**2)/mbkin**2 - (4*mcMS**4)/mbkin**4)*q_cut**3)/
             mbkin**6 - (4*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + 
            (5*q_cut**5)/mbkin**10)*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
            (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 9*
                mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
            (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 12*
                mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
             (9*mbkin**6)) + ((-1 + mcMS**2/mbkin**2)**2 - 
            (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
           ((64*mbkin*(-((-1 + mcMS**2/mbkin**2)**4*(1 + mcMS**2/mbkin**2)**2*
                 (503 - (9464*mcMS**2)/mbkin**2 + (69322*mcMS**4)/mbkin**4 - 
                  (179128*mcMS**6)/mbkin**6 - (217124*mcMS**8)/mbkin**8 + 
                  (134968*mcMS**10)/mbkin**10 - (44170*mcMS**12)/mbkin**12 + 
                  (3064*mcMS**14)/mbkin**14 + (109*mcMS**16)/mbkin**16)) + 
               ((-1 + mcMS**2/mbkin**2)**2*(2903 - (41547*mcMS**2)/mbkin**2 + 
                  (196111*mcMS**4)/mbkin**4 + (84389*mcMS**6)/mbkin**6 - 
                  (2226350*mcMS**8)/mbkin**8 - (4060522*mcMS**10)/mbkin**10 - 
                  (2007262*mcMS**12)/mbkin**12 + (332278*mcMS**14)/mbkin**14 + 
                  (144215*mcMS**16)/mbkin**16 - (185931*mcMS**18)/mbkin**18 + 
                  (19663*mcMS**20)/mbkin**20 + (613*mcMS**22)/mbkin**22)*q_cut)/
                mbkin**2 - (2*(2870 - (34643*mcMS**2)/mbkin**2 + (127322*mcMS**4)/
                   mbkin**4 + (190963*mcMS**6)/mbkin**6 - (1214040*mcMS**8)/
                   mbkin**8 - (2984930*mcMS**10)/mbkin**10 - (3081596*mcMS**12)/
                   mbkin**12 - (1121550*mcMS**14)/mbkin**14 + (431494*mcMS**16)/
                   mbkin**16 + (81077*mcMS**18)/mbkin**18 - (159838*mcMS**20)/
                   mbkin**20 + (20891*mcMS**22)/mbkin**22 + (540*mcMS**24)/
                   mbkin**24)*q_cut**2)/mbkin**4 - (2*(-969 + (3860*mcMS**2)/
                   mbkin**2 + (20318*mcMS**4)/mbkin**4 - (143553*mcMS**6)/
                   mbkin**6 + (43662*mcMS**8)/mbkin**8 + (243236*mcMS**10)/
                   mbkin**10 - (166416*mcMS**12)/mbkin**12 - (159266*mcMS**14)/
                   mbkin**14 + (150619*mcMS**16)/mbkin**16 + (20136*mcMS**18)/
                   mbkin**18 - (11694*mcMS**20)/mbkin**20 + (67*mcMS**22)/
                   mbkin**22)*q_cut**3)/mbkin**6 + ((8985 - (76768*mcMS**2)/
                   mbkin**2 + (136327*mcMS**4)/mbkin**4 + (661168*mcMS**6)/
                   mbkin**6 + (524386*mcMS**8)/mbkin**8 + (437592*mcMS**10)/
                   mbkin**10 + (491062*mcMS**12)/mbkin**12 - (151632*mcMS**14)/
                   mbkin**14 - (284475*mcMS**16)/mbkin**16 + (57128*mcMS**18)/
                   mbkin**18 + (2563*mcMS**20)/mbkin**20)*q_cut**4)/mbkin**8 - 
               ((12407 - (70063*mcMS**2)/mbkin**2 + (19312*mcMS**4)/mbkin**4 + 
                  (726800*mcMS**6)/mbkin**6 + (1038918*mcMS**8)/mbkin**8 + 
                  (304322*mcMS**10)/mbkin**10 - (490544*mcMS**12)/mbkin**12 - 
                  (209264*mcMS**14)/mbkin**14 + (102019*mcMS**16)/mbkin**16 + 
                  (1997*mcMS**18)/mbkin**18)*q_cut**5)/mbkin**10 - 
               (4*(-354 + (2543*mcMS**2)/mbkin**2 - (2380*mcMS**4)/mbkin**4 - 
                  (59099*mcMS**6)/mbkin**6 - (24326*mcMS**8)/mbkin**8 + 
                  (44915*mcMS**10)/mbkin**10 + (6624*mcMS**12)/mbkin**12 - 
                  (8999*mcMS**14)/mbkin**14 + (564*mcMS**16)/mbkin**16)*q_cut**6)/
                mbkin**12 + (4*(2103 - (3452*mcMS**2)/mbkin**2 - (14684*mcMS**4)/
                   mbkin**4 - (17263*mcMS**6)/mbkin**6 - (5615*mcMS**8)/mbkin**8 + 
                  (7734*mcMS**10)/mbkin**10 + (11390*mcMS**12)/mbkin**12 + 
                  (1035*mcMS**14)/mbkin**14)*q_cut**7)/mbkin**14 + 
               ((-5829 + (9694*mcMS**2)/mbkin**2 + (45611*mcMS**4)/mbkin**4 + 
                  (25144*mcMS**6)/mbkin**6 - (44863*mcMS**8)/mbkin**8 - 
                  (44134*mcMS**10)/mbkin**10 - (1047*mcMS**12)/mbkin**12)*q_cut**8)/
                mbkin**16 - ((343 + (6467*mcMS**2)/mbkin**2 + (16652*mcMS**4)/
                   mbkin**4 - (4820*mcMS**6)/mbkin**6 - (5767*mcMS**8)/mbkin**8 + 
                  (2021*mcMS**10)/mbkin**10)*q_cut**9)/mbkin**18 + 
               (2*(830 + (1771*mcMS**2)/mbkin**2 + (3242*mcMS**4)/mbkin**4 + 
                  (3405*mcMS**6)/mbkin**6 + (928*mcMS**8)/mbkin**8)*q_cut**10)/
                mbkin**20 - (2*(247 + (420*mcMS**2)/mbkin**2 + (1194*mcMS**4)/
                   mbkin**4 + (291*mcMS**6)/mbkin**6)*q_cut**11)/mbkin**22 + 
               ((3 + (92*mcMS**2)/mbkin**2 + (65*mcMS**4)/mbkin**4)*q_cut**12)/
                mbkin**24 - ((9 + (19*mcMS**2)/mbkin**2)*q_cut**13)/mbkin**26 + 
               (8*q_cut**14)/mbkin**28))/3 + 72*(mbkin**4*((-1 + mcMS**2/mbkin**2)**2*
                  (1 - (7*mcMS**2)/mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/
                    mbkin**6) + ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/
                     mbkin**4 + (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/
                  mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*
                     mcMS**4 + mcMS**6)*q_cut**2)/mbkin**10 + (2*(mbkin**4 + 
                    mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/mbkin**10 - 
                 (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2*(
                (-100*(3 + 8*mbkin)*q_cut**5)/(9*mbkin**12) - (46*mcMS**2*
                  (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                 (9*mbkin**4) - (1592*mcMS**4*(-6 - 16*mbkin + 12*mbkin**2 + 
                   9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**6) - 
                (796*mcMS**6*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                    np.log(mu0**2/mcMS**2)))/(3*mbkin**8) - (184*mcMS**8*
                  (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                 (9*mbkin**10) + (10*mcMS**10*(-6 - 16*mbkin + 12*mbkin**2 + 
                   9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**12) + 
                (8*q_cut**4*(24*mbkin**2 + 64*mbkin**3 + 30*mcMS**2 + 80*mbkin*
                    mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                    np.log(mu0**2/mcMS**2)))/(9*mbkin**12) + q_cut**3*
                 ((-4*(3 + 8*mbkin)*(-4 + (2*mcMS**2)/mbkin**2 - (4*mcMS**4)/
                      mbkin**4))/(3*mbkin**8) + (4*(mbkin**2*mcMS**2 - 4*mcMS**4)*
                    (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                        mcMS**2)))/(9*mbkin**12)) + q_cut**2*((-8*(3 + 8*mbkin)*
                    (1 - (15*mcMS**2)/mbkin**2 - (15*mcMS**4)/mbkin**4 + 
                     mcMS**6/mbkin**6))/(9*mbkin**6) - (2*(5*mbkin**4*mcMS**2 + 
                     10*mbkin**2*mcMS**4 - mcMS**6)*(-6 - 16*mbkin + 
                     12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                   (3*mbkin**12)) + q_cut*((-4*(3 + 8*mbkin)*(1 - (20*mcMS**2)/
                      mbkin**2 - (102*mcMS**4)/mbkin**4 - (20*mcMS**6)/mbkin**6 + 
                     mcMS**8/mbkin**8))/(9*mbkin**4) - (8*(5*mbkin**6*mcMS**2 + 
                     51*mbkin**4*mcMS**4 + 15*mbkin**2*mcMS**6 - mcMS**8)*
                    (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                        mcMS**2)))/(9*mbkin**12))) + (1 - (23*mcMS**2)/mbkin**2 - 
                (398*mcMS**4)/mbkin**4 - (398*mcMS**6)/mbkin**6 - (23*mcMS**8)/
                 mbkin**8 + mcMS**10/mbkin**10 + ((mbkin**8 - 20*mbkin**6*mcMS**2 - 
                   102*mbkin**4*mcMS**4 - 20*mbkin**2*mcMS**6 + mcMS**8)*q_cut)/
                 mbkin**10 + ((mbkin**6 - 15*mbkin**4*mcMS**2 - 15*mbkin**2*
                    mcMS**4 + mcMS**6)*q_cut**2)/mbkin**10 + 
                ((-4 + (2*mcMS**2)/mbkin**2 - (4*mcMS**4)/mbkin**4)*q_cut**3)/
                 mbkin**6 - (4*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + 
                (5*q_cut**5)/mbkin**10)*((8*mbkin**2*(3 + 8*mbkin)*
                  ((-1 + mcMS**2/mbkin**2)**2*(1 - (7*mcMS**2)/mbkin**2 - 
                      (7*mcMS**4)/mbkin**4 + mcMS**6/mbkin**6) + 
                    ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + 
                       (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/
                     mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*
                        mcMS**4 + mcMS**6)*q_cut**2)/mbkin**10 + (2*(mbkin**4 + 
                       mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/mbkin**10 - 
                    (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2)/
                 9 + 2*mbkin**4*((-1 + mcMS**2/mbkin**2)**2*(1 - (7*mcMS**2)/
                     mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/mbkin**6) + 
                  ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + 
                     (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/mbkin**2 + 
                  (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*mcMS**4 + mcMS**6)*
                    q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mcMS**2 + mcMS**4)*
                    q_cut**3)/mbkin**10 - (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + 
                  q_cut**5/mbkin**10)*((-20*(3 + 8*mbkin)*q_cut**5)/(9*mbkin**12) - 
                  (2*(mbkin**2 - mcMS**2)*(9*mbkin**6*mcMS**2 - 7*mbkin**4*
                      mcMS**4 - 31*mbkin**2*mcMS**6 + 5*mcMS**8)*(-6 - 16*mbkin + 
                     12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                   (9*mbkin**12) + (2*q_cut**4*(24*mbkin**2 + 64*mbkin**3 + 
                     30*mcMS**2 + 80*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 
                     9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/(3*mbkin**12) + 
                  2*q_cut**3*((-4*(3 + 8*mbkin)*(1 + mcMS**2/mbkin**2 + mcMS**4/
                        mbkin**4))/(3*mbkin**8) + (2*(mbkin**2*mcMS**2 + 
                       2*mcMS**4)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                        np.log(mu0**2/mcMS**2)))/(9*mbkin**12)) + 2*q_cut**2*
                   ((-8*(3 + 8*mbkin)*(1 - (2*mcMS**2)/mbkin**2 - (2*mcMS**4)/
                        mbkin**4 + mcMS**6/mbkin**6))/(9*mbkin**6) - 
                    (2*(2*mbkin**4*mcMS**2 + 4*mbkin**2*mcMS**4 - 3*mcMS**6)*
                      (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**12)) + q_cut*((-4*(3 + 8*mbkin)*
                      (-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + 
                       (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8))/
                     (9*mbkin**4) + (4*(7*mbkin**6*mcMS**2 + 26*mbkin**4*mcMS**4 + 
                       21*mbkin**2*mcMS**6 - 6*mcMS**8)*(-6 - 16*mbkin + 
                       12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                     (9*mbkin**12)))))) - 
          12*(np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(72*mcMS**4*(
                (-1 + mcMS**4/mbkin**4)**2*(17 - (264*mcMS**2)/mbkin**2 + 
                  (3*mcMS**4)/mbkin**4 + (8048*mcMS**6)/mbkin**6 + (3*mcMS**8)/
                   mbkin**8 - (264*mcMS**10)/mbkin**10 + (17*mcMS**12)/
                   mbkin**12) - (16*(4 - (38*mcMS**2)/mbkin**2 - (173*mcMS**4)/
                    mbkin**4 + (887*mcMS**6)/mbkin**6 + (3100*mcMS**8)/mbkin**8 + 
                   (3100*mcMS**10)/mbkin**10 + (887*mcMS**12)/mbkin**12 - 
                   (173*mcMS**14)/mbkin**14 - (38*mcMS**16)/mbkin**16 + 
                   (4*mcMS**18)/mbkin**18)*q_cut)/mbkin**2 + 
                (4*(15 - (50*mcMS**2)/mbkin**2 - (534*mcMS**4)/mbkin**4 - 
                   (630*mcMS**6)/mbkin**6 - (122*mcMS**8)/mbkin**8 - 
                   (630*mcMS**10)/mbkin**10 - (534*mcMS**12)/mbkin**12 - 
                   (50*mcMS**14)/mbkin**14 + (15*mcMS**16)/mbkin**16)*q_cut**2)/
                 mbkin**4 + ((54 - (242*mcMS**2)/mbkin**2 - (4222*mcMS**4)/
                    mbkin**4 - (7014*mcMS**6)/mbkin**6 - (7014*mcMS**8)/mbkin**8 - 
                   (4222*mcMS**10)/mbkin**10 - (242*mcMS**12)/mbkin**12 + 
                   (54*mcMS**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
                ((-132 + (8*mcMS**2)/mbkin**2 + (4184*mcMS**4)/mbkin**4 + 
                   (8048*mcMS**6)/mbkin**6 + (4184*mcMS**8)/mbkin**8 + 
                   (8*mcMS**10)/mbkin**10 - (132*mcMS**12)/mbkin**12)*q_cut**4)/
                 mbkin**8 + (2*(39 - (73*mcMS**2)/mbkin**2 - (510*mcMS**4)/
                    mbkin**4 - (510*mcMS**6)/mbkin**6 - (73*mcMS**8)/mbkin**8 + 
                   (39*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
                ((-6 + (472*mcMS**2)/mbkin**2 + (716*mcMS**4)/mbkin**4 + 
                   (472*mcMS**6)/mbkin**6 - (6*mcMS**8)/mbkin**8)*q_cut**6)/
                 mbkin**12 - (10*(3 + (31*mcMS**2)/mbkin**2 + (31*mcMS**4)/
                    mbkin**4 + (3*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
                ((51 + (112*mcMS**2)/mbkin**2 + (51*mcMS**4)/mbkin**4)*q_cut**8)/
                 mbkin**16 - (38*(mbkin**2 + mcMS**2)*q_cut**9)/mbkin**20 + 
                (10*q_cut**10)/mbkin**20)*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
                (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                   9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                   12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                 (9*mbkin**6)) + ((-1 + mcMS**2/mbkin**2)**2 - 
                (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*(
                (-64*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*(13 - (155*mcMS**2)/
                       mbkin**2 + (233*mcMS**4)/mbkin**4 + (9069*mcMS**6)/
                       mbkin**6 - (59845*mcMS**8)/mbkin**8 - (147269*mcMS**10)/
                       mbkin**10 - (55861*mcMS**12)/mbkin**12 + (18451*mcMS**14)/
                       mbkin**14 - (5640*mcMS**16)/mbkin**16 - (1056*mcMS**18)/
                       mbkin**18 + (140*mcMS**20)/mbkin**20)) + 
                   (4*(14 - (143*mcMS**2)/mbkin**2 + (150*mcMS**4)/mbkin**4 + 
                      (6469*mcMS**6)/mbkin**6 - (22516*mcMS**8)/mbkin**8 - 
                      (106182*mcMS**10)/mbkin**10 - (107880*mcMS**12)/mbkin**12 - 
                      (20198*mcMS**14)/mbkin**14 + (12998*mcMS**16)/mbkin**16 - 
                      (3867*mcMS**18)/mbkin**18 - (910*mcMS**20)/mbkin**20 + 
                      (145*mcMS**22)/mbkin**22)*q_cut)/mbkin**2 - 
                   (4*(15 - (95*mcMS**2)/mbkin**2 - (89*mcMS**4)/mbkin**4 + 
                      (2989*mcMS**6)/mbkin**6 + (4773*mcMS**8)/mbkin**8 + 
                      (2337*mcMS**10)/mbkin**10 + (10585*mcMS**12)/mbkin**12 + 
                      (2765*mcMS**14)/mbkin**14 - (2854*mcMS**16)/mbkin**16 - 
                      (416*mcMS**18)/mbkin**18 + (150*mcMS**20)/mbkin**20)*q_cut**2)/
                    mbkin**4 - (2*(33 - (229*mcMS**2)/mbkin**2 - (526*mcMS**4)/
                       mbkin**4 + (11552*mcMS**6)/mbkin**6 + (19468*mcMS**8)/
                       mbkin**8 + (17570*mcMS**10)/mbkin**10 + (6430*mcMS**12)/
                       mbkin**12 - (8984*mcMS**14)/mbkin**14 - (1277*mcMS**16)/
                       mbkin**16 + (315*mcMS**18)/mbkin**18)*q_cut**3)/mbkin**6 + 
                   (2*(84 - (301*mcMS**2)/mbkin**2 - (1280*mcMS**4)/mbkin**4 + 
                      (11016*mcMS**6)/mbkin**6 + (26043*mcMS**8)/mbkin**8 + 
                      (1603*mcMS**10)/mbkin**10 - (11796*mcMS**12)/mbkin**12 - 
                      (692*mcMS**14)/mbkin**14 + (795*mcMS**16)/mbkin**16)*q_cut**4)/
                    mbkin**8 - (2*(21 - (37*mcMS**2)/mbkin**2 - (145*mcMS**4)/
                       mbkin**4 + (2777*mcMS**6)/mbkin**6 + (927*mcMS**8)/
                       mbkin**8 - (2211*mcMS**10)/mbkin**10 + (413*mcMS**12)/
                       mbkin**12 + (255*mcMS**14)/mbkin**14)*q_cut**5)/mbkin**10 + 
                   ((-126 + (92*mcMS**2)/mbkin**2 + (1574*mcMS**4)/mbkin**4 + 
                      (2084*mcMS**6)/mbkin**6 + (2298*mcMS**8)/mbkin**8 - 
                      (560*mcMS**10)/mbkin**10 - (930*mcMS**12)/mbkin**12)*q_cut**6)/
                    mbkin**12 + (2*(45 - (25*mcMS**2)/mbkin**2 - (476*mcMS**4)/
                       mbkin**4 - (338*mcMS**6)/mbkin**6 + (523*mcMS**8)/
                       mbkin**8 + (375*mcMS**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
                   ((21 + (77*mcMS**2)/mbkin**2 + (272*mcMS**4)/mbkin**4 - 
                      (64*mcMS**6)/mbkin**6 + (30*mcMS**8)/mbkin**8)*q_cut**8)/
                    mbkin**16 - (2*(19 + (19*mcMS**2)/mbkin**2 + (63*mcMS**4)/
                       mbkin**4 + (95*mcMS**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
                   (10*(mbkin**4 + 5*mcMS**4)*q_cut**10)/mbkin**24))/3 + 
                72*(mcMS**4*((-1 + mcMS**4/mbkin**4)**2*(17 - (264*mcMS**2)/
                       mbkin**2 + (3*mcMS**4)/mbkin**4 + (8048*mcMS**6)/mbkin**6 + 
                      (3*mcMS**8)/mbkin**8 - (264*mcMS**10)/mbkin**10 + 
                      (17*mcMS**12)/mbkin**12) - (16*(4 - (38*mcMS**2)/mbkin**2 - 
                       (173*mcMS**4)/mbkin**4 + (887*mcMS**6)/mbkin**6 + 
                       (3100*mcMS**8)/mbkin**8 + (3100*mcMS**10)/mbkin**10 + 
                       (887*mcMS**12)/mbkin**12 - (173*mcMS**14)/mbkin**14 - 
                       (38*mcMS**16)/mbkin**16 + (4*mcMS**18)/mbkin**18)*q_cut)/
                     mbkin**2 + (4*(15 - (50*mcMS**2)/mbkin**2 - (534*mcMS**4)/
                        mbkin**4 - (630*mcMS**6)/mbkin**6 - (122*mcMS**8)/
                        mbkin**8 - (630*mcMS**10)/mbkin**10 - (534*mcMS**12)/
                        mbkin**12 - (50*mcMS**14)/mbkin**14 + (15*mcMS**16)/
                        mbkin**16)*q_cut**2)/mbkin**4 + ((54 - (242*mcMS**2)/
                        mbkin**2 - (4222*mcMS**4)/mbkin**4 - (7014*mcMS**6)/
                        mbkin**6 - (7014*mcMS**8)/mbkin**8 - (4222*mcMS**10)/
                        mbkin**10 - (242*mcMS**12)/mbkin**12 + (54*mcMS**14)/
                        mbkin**14)*q_cut**3)/mbkin**6 + ((-132 + (8*mcMS**2)/
                        mbkin**2 + (4184*mcMS**4)/mbkin**4 + (8048*mcMS**6)/
                        mbkin**6 + (4184*mcMS**8)/mbkin**8 + (8*mcMS**10)/
                        mbkin**10 - (132*mcMS**12)/mbkin**12)*q_cut**4)/mbkin**8 + 
                    (2*(39 - (73*mcMS**2)/mbkin**2 - (510*mcMS**4)/mbkin**4 - 
                       (510*mcMS**6)/mbkin**6 - (73*mcMS**8)/mbkin**8 + 
                       (39*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
                    ((-6 + (472*mcMS**2)/mbkin**2 + (716*mcMS**4)/mbkin**4 + 
                       (472*mcMS**6)/mbkin**6 - (6*mcMS**8)/mbkin**8)*q_cut**6)/
                     mbkin**12 - (10*(3 + (31*mcMS**2)/mbkin**2 + (31*mcMS**4)/
                        mbkin**4 + (3*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
                    ((51 + (112*mcMS**2)/mbkin**2 + (51*mcMS**4)/mbkin**4)*q_cut**8)/
                     mbkin**16 - (38*(mbkin**2 + mcMS**2)*q_cut**9)/mbkin**20 + 
                    (10*q_cut**10)/mbkin**20)*(16/3 + 4*np.log(mu0**2/mcMS**2)) + 
                  mcMS**4*((-400*(3 + 8*mbkin)*q_cut**10)/(9*mbkin**22) - 
                    (4*(mbkin**4 - mcMS**4)*(132*mbkin**14*mcMS**2 + 31*mbkin**12*
                        mcMS**4 - 12732*mbkin**10*mcMS**6 + 3*mbkin**8*mcMS**8 + 
                       28828*mbkin**6*mcMS**10 - 39*mbkin**4*mcMS**12 - 
                       1188*mbkin**2*mcMS**14 + 85*mcMS**16)*(-6 - 16*mbkin + 
                       12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                     (9*mbkin**22) + (76*q_cut**9*(54*mbkin**2 + 144*mbkin**3 + 
                       60*mcMS**2 + 160*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 
                       9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**22) + 
                    q_cut**8*((-32*(3 + 8*mbkin)*(51 + (112*mcMS**2)/mbkin**2 + 
                         (51*mcMS**4)/mbkin**4))/(9*mbkin**18) + 
                      (4*(56*mbkin**2*mcMS**2 + 51*mcMS**4)*(-6 - 16*mbkin + 
                         12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                       (9*mbkin**22)) - 10*q_cut**7*((-28*(3 + 8*mbkin)*(3 + 
                         (31*mcMS**2)/mbkin**2 + (31*mcMS**4)/mbkin**4 + 
                         (3*mcMS**6)/mbkin**6))/(9*mbkin**16) + 
                      (2*(31*mbkin**4*mcMS**2 + 62*mbkin**2*mcMS**4 + 9*mcMS**6)*
                        (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**22)) + q_cut**6*((-8*(3 + 8*mbkin)*
                        (-6 + (472*mcMS**2)/mbkin**2 + (716*mcMS**4)/mbkin**4 + 
                         (472*mcMS**6)/mbkin**6 - (6*mcMS**8)/mbkin**8))/
                       (3*mbkin**14) + (16*(59*mbkin**6*mcMS**2 + 179*mbkin**4*
                          mcMS**4 + 177*mbkin**2*mcMS**6 - 3*mcMS**8)*(-6 - 
                         16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**
                          2)))/(9*mbkin**22)) + 2*q_cut**5*((-20*(3 + 8*mbkin)*
                        (39 - (73*mcMS**2)/mbkin**2 - (510*mcMS**4)/mbkin**4 - 
                         (510*mcMS**6)/mbkin**6 - (73*mcMS**8)/mbkin**8 + 
                         (39*mcMS**10)/mbkin**10))/(9*mbkin**12) - 
                      (2*(73*mbkin**8*mcMS**2 + 1020*mbkin**6*mcMS**4 + 1530*
                          mbkin**4*mcMS**6 + 292*mbkin**2*mcMS**8 - 195*mcMS**10)*
                        (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**22)) + q_cut**4*((-16*(3 + 8*mbkin)*
                        (-132 + (8*mcMS**2)/mbkin**2 + (4184*mcMS**4)/mbkin**4 + 
                         (8048*mcMS**6)/mbkin**6 + (4184*mcMS**8)/mbkin**8 + 
                         (8*mcMS**10)/mbkin**10 - (132*mcMS**12)/mbkin**12))/
                       (9*mbkin**10) + (16*(mbkin**10*mcMS**2 + 1046*mbkin**8*
                          mcMS**4 + 3018*mbkin**6*mcMS**6 + 2092*mbkin**4*
                          mcMS**8 + 5*mbkin**2*mcMS**10 - 99*mcMS**12)*(-6 - 
                         16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**
                          2)))/(9*mbkin**22)) + q_cut**3*((-4*(3 + 8*mbkin)*
                        (54 - (242*mcMS**2)/mbkin**2 - (4222*mcMS**4)/mbkin**4 - 
                         (7014*mcMS**6)/mbkin**6 - (7014*mcMS**8)/mbkin**8 - 
                         (4222*mcMS**10)/mbkin**10 - (242*mcMS**12)/mbkin**12 + 
                         (54*mcMS**14)/mbkin**14))/(3*mbkin**8) - 
                      (4*(121*mbkin**12*mcMS**2 + 4222*mbkin**10*mcMS**4 + 
                         10521*mbkin**8*mcMS**6 + 14028*mbkin**6*mcMS**8 + 
                         10555*mbkin**4*mcMS**10 + 726*mbkin**2*mcMS**12 - 
                         189*mcMS**14)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**22)) + 4*q_cut**2*
                     ((-8*(3 + 8*mbkin)*(15 - (50*mcMS**2)/mbkin**2 - 
                         (534*mcMS**4)/mbkin**4 - (630*mcMS**6)/mbkin**6 - 
                         (122*mcMS**8)/mbkin**8 - (630*mcMS**10)/mbkin**10 - 
                         (534*mcMS**12)/mbkin**12 - (50*mcMS**14)/mbkin**14 + 
                         (15*mcMS**16)/mbkin**16))/(9*mbkin**6) - 
                      (4*(25*mbkin**14*mcMS**2 + 534*mbkin**12*mcMS**4 + 
                         945*mbkin**10*mcMS**6 + 244*mbkin**8*mcMS**8 + 1575*
                          mbkin**6*mcMS**10 + 1602*mbkin**4*mcMS**12 + 
                         175*mbkin**2*mcMS**14 - 60*mcMS**16)*(-6 - 16*mbkin + 
                         12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                       (9*mbkin**22)) - 16*q_cut*((-4*(3 + 8*mbkin)*(4 - 
                         (38*mcMS**2)/mbkin**2 - (173*mcMS**4)/mbkin**4 + 
                         (887*mcMS**6)/mbkin**6 + (3100*mcMS**8)/mbkin**8 + 
                         (3100*mcMS**10)/mbkin**10 + (887*mcMS**12)/mbkin**12 - 
                         (173*mcMS**14)/mbkin**14 - (38*mcMS**16)/mbkin**16 + 
                         (4*mcMS**18)/mbkin**18))/(9*mbkin**4) - 
                      (2*(38*mbkin**16*mcMS**2 + 346*mbkin**14*mcMS**4 - 
                         2661*mbkin**12*mcMS**6 - 12400*mbkin**10*mcMS**8 - 
                         15500*mbkin**8*mcMS**10 - 5322*mbkin**6*mcMS**12 + 
                         1211*mbkin**4*mcMS**14 + 304*mbkin**2*mcMS**16 - 
                         36*mcMS**18)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**22))))))*
             np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) + 
            (72*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                2*mcMS**2*q_cut + q_cut**2)**2*(17*mbkin**16 - 230*mbkin**14*mcMS**2 - 
               508*mbkin**12*mcMS**4 + 7790*mbkin**10*mcMS**6 + 16102*mbkin**8*
                mcMS**8 + 7790*mbkin**6*mcMS**10 - 508*mbkin**4*mcMS**12 - 230*
                mbkin**2*mcMS**14 + 17*mcMS**16 - 30*mbkin**14*q_cut + 122*mbkin**12*
                mcMS**2*q_cut + 1566*mbkin**10*mcMS**4*q_cut + 3382*mbkin**8*mcMS**6*
                q_cut + 3382*mbkin**6*mcMS**8*q_cut + 1566*mbkin**4*mcMS**10*q_cut + 122*
                mbkin**2*mcMS**12*q_cut - 30*mcMS**14*q_cut - 17*mbkin**12*q_cut**2 + 180*
                mbkin**10*mcMS**2*q_cut**2 + 2125*mbkin**8*mcMS**4*q_cut**2 + 3656*
                mbkin**6*mcMS**6*q_cut**2 + 2125*mbkin**4*mcMS**8*q_cut**2 + 180*mbkin**2*
                mcMS**10*q_cut**2 - 17*mcMS**12*q_cut**2 + 50*mbkin**10*q_cut**3 + 62*
                mbkin**8*mcMS**2*q_cut**3 - 1104*mbkin**6*mcMS**4*q_cut**3 - 1104*mbkin**4*
                mcMS**6*q_cut**3 + 62*mbkin**2*mcMS**8*q_cut**3 + 50*mcMS**10*q_cut**3 - 15*
                mbkin**8*q_cut**4 + 22*mbkin**6*mcMS**2*q_cut**4 + 34*mbkin**4*mcMS**4*
                q_cut**4 + 22*mbkin**2*mcMS**6*q_cut**4 - 15*mcMS**8*q_cut**4 - 2*mbkin**6*
                q_cut**5 - 198*mbkin**4*mcMS**2*q_cut**5 - 198*mbkin**2*mcMS**4*q_cut**5 - 2*
                mcMS**6*q_cut**5 + 5*mbkin**4*q_cut**6 + 60*mbkin**2*mcMS**2*q_cut**6 + 5*
                mcMS**4*q_cut**6 - 18*mbkin**2*q_cut**7 - 18*mcMS**2*q_cut**7 + 10*q_cut**8)*
              ((-8*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(-6*mbkin**4*mcMS**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 16*mbkin**5*mcMS**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 12*mbkin**6*mcMS**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 6*mbkin**2*mcMS**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 16*mbkin**3*mcMS**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*mcMS**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*mcMS**2*q_cut*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 9*mbkin**6*mcMS**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 
                  9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                      mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                   np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**2*q_cut*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2)))/
                (9*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)*(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                      2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) + 
               (np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*((-8*(3 + 8*mbkin)*q_cut**2)/
                   (9*mbkin**6) + (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 
                     16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                   (9*mbkin**4) + (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 
                     32*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                      np.log(mu0**2/mcMS**2)))/(9*mbkin**6))*np.log((mbkin**2 + 
                    mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                   (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4))))/(2*((-1 + mcMS**2/mbkin**2)**2 - 
                  (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4))))/
             mbkin**24) - 144*((mcMS**4*(-72*mcMS**4*((-1 + mcMS**2/mbkin**2)**2*
                  (31 - (153*mcMS**2)/mbkin**2 - (1138*mcMS**4)/mbkin**4 - 
                   (1138*mcMS**6)/mbkin**6 - (153*mcMS**8)/mbkin**8 + 
                   (31*mcMS**10)/mbkin**10) + ((-91 + (202*mcMS**2)/mbkin**2 + 
                    (2591*mcMS**4)/mbkin**4 + (4676*mcMS**6)/mbkin**6 + 
                    (2591*mcMS**8)/mbkin**8 + (202*mcMS**10)/mbkin**10 - 
                    (91*mcMS**12)/mbkin**12)*q_cut)/mbkin**2 + 
                 ((60 + (38*mcMS**2)/mbkin**2 - (518*mcMS**4)/mbkin**4 - 
                    (518*mcMS**6)/mbkin**6 + (38*mcMS**8)/mbkin**8 + (60*mcMS**10)/
                     mbkin**10)*q_cut**2)/mbkin**4 + ((55 + (238*mcMS**2)/mbkin**2 + 
                    (226*mcMS**4)/mbkin**4 + (238*mcMS**6)/mbkin**6 + (55*mcMS**8)/
                     mbkin**8)*q_cut**3)/mbkin**6 - ((85 + (337*mcMS**2)/mbkin**2 + 
                    (337*mcMS**4)/mbkin**4 + (85*mcMS**6)/mbkin**6)*q_cut**4)/
                  mbkin**8 + ((39 + (88*mcMS**2)/mbkin**2 + (39*mcMS**4)/mbkin**4)*
                   q_cut**5)/mbkin**10 - (14*(mbkin**2 + mcMS**2)*q_cut**6)/mbkin**14 + 
                 (5*q_cut**7)/mbkin**14)*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
                 (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 
                    12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                 (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                     mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                     np.log(mu0**2/mcMS**2)))/(9*mbkin**6)) + 
               ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
                 q_cut**2/mbkin**4)*((64*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*
                      (-2 - (44*mcMS**2)/mbkin**2 + (729*mcMS**4)/mbkin**4 + 
                       (9244*mcMS**6)/mbkin**6 + (10134*mcMS**8)/mbkin**8 + 
                       (36*mcMS**10)/mbkin**10 - (77*mcMS**12)/mbkin**12 + 
                       (140*mcMS**14)/mbkin**14)) + ((-2 - (226*mcMS**2)/
                        mbkin**2 + (1025*mcMS**4)/mbkin**4 + (19673*mcMS**6)/
                        mbkin**6 + (39598*mcMS**8)/mbkin**8 + (21350*mcMS**10)/
                        mbkin**10 - (861*mcMS**12)/mbkin**12 - (357*mcMS**14)/
                        mbkin**14 + (440*mcMS**16)/mbkin**16)*q_cut)/mbkin**2 - 
                    (2*mcMS**2*(-88 - (232*mcMS**2)/mbkin**2 + (2115*mcMS**4)/
                        mbkin**4 + (1978*mcMS**6)/mbkin**6 - (633*mcMS**8)/
                        mbkin**8 + (70*mcMS**10)/mbkin**10 + (150*mcMS**12)/
                        mbkin**12)*q_cut**2)/mbkin**6 + ((-10 + (216*mcMS**2)/
                        mbkin**2 + (1252*mcMS**4)/mbkin**4 + (972*mcMS**6)/
                        mbkin**6 + (866*mcMS**8)/mbkin**8 - (370*mcMS**10)/
                        mbkin**10 - (350*mcMS**12)/mbkin**12)*q_cut**3)/mbkin**6 + 
                    ((10 - (284*mcMS**2)/mbkin**2 - (2029*mcMS**4)/mbkin**4 - 
                       (1764*mcMS**6)/mbkin**6 + (735*mcMS**8)/mbkin**8 + 
                       (500*mcMS**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
                    ((18 + (106*mcMS**2)/mbkin**2 + (601*mcMS**4)/mbkin**4 - 
                       (157*mcMS**6)/mbkin**6 - (60*mcMS**8)/mbkin**8)*q_cut**5)/
                     mbkin**10 - (4*(7 + (7*mcMS**2)/mbkin**2 + (17*mcMS**4)/
                        mbkin**4 + (35*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
                    (10*(mbkin**4 + 5*mcMS**4)*q_cut**7)/mbkin**18))/3 - 
                 72*(mcMS**4*((-1 + mcMS**2/mbkin**2)**2*(31 - (153*mcMS**2)/
                        mbkin**2 - (1138*mcMS**4)/mbkin**4 - (1138*mcMS**6)/
                        mbkin**6 - (153*mcMS**8)/mbkin**8 + (31*mcMS**10)/
                        mbkin**10) + ((-91 + (202*mcMS**2)/mbkin**2 + 
                        (2591*mcMS**4)/mbkin**4 + (4676*mcMS**6)/mbkin**6 + 
                        (2591*mcMS**8)/mbkin**8 + (202*mcMS**10)/mbkin**10 - 
                        (91*mcMS**12)/mbkin**12)*q_cut)/mbkin**2 + 
                     ((60 + (38*mcMS**2)/mbkin**2 - (518*mcMS**4)/mbkin**4 - 
                        (518*mcMS**6)/mbkin**6 + (38*mcMS**8)/mbkin**8 + 
                        (60*mcMS**10)/mbkin**10)*q_cut**2)/mbkin**4 + 
                     ((55 + (238*mcMS**2)/mbkin**2 + (226*mcMS**4)/mbkin**4 + 
                        (238*mcMS**6)/mbkin**6 + (55*mcMS**8)/mbkin**8)*q_cut**3)/
                      mbkin**6 - ((85 + (337*mcMS**2)/mbkin**2 + (337*mcMS**4)/
                         mbkin**4 + (85*mcMS**6)/mbkin**6)*q_cut**4)/mbkin**8 + 
                     ((39 + (88*mcMS**2)/mbkin**2 + (39*mcMS**4)/mbkin**4)*q_cut**5)/
                      mbkin**10 - (14*(mbkin**2 + mcMS**2)*q_cut**6)/mbkin**14 + 
                     (5*q_cut**7)/mbkin**14)*(16/3 + 4*np.log(mu0**2/mcMS**2)) + 
                   mcMS**4*((-140*(3 + 8*mbkin)*q_cut**7)/(9*mbkin**16) - 
                     (2*(mbkin**2 - mcMS**2)*(215*mbkin**10*mcMS**2 + 1817*
                         mbkin**8*mcMS**4 - 1138*mbkin**6*mcMS**6 - 5078*mbkin**4*
                         mcMS**8 - 1073*mbkin**2*mcMS**10 + 217*mcMS**12)*
                       (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**16) + (28*q_cut**6*(36*mbkin**2 + 
                        96*mbkin**3 + 42*mcMS**2 + 112*mbkin*mcMS**2 - 
                        12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**16) + q_cut**5*((-20*(3 + 8*mbkin)*
                         (39 + (88*mcMS**2)/mbkin**2 + (39*mcMS**4)/mbkin**4))/
                        (9*mbkin**12) + (4*(44*mbkin**2*mcMS**2 + 39*mcMS**4)*
                         (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**16)) + q_cut**4*((16*(3 + 8*mbkin)*
                         (85 + (337*mcMS**2)/mbkin**2 + (337*mcMS**4)/mbkin**4 + 
                          (85*mcMS**6)/mbkin**6))/(9*mbkin**10) - 
                       (2*(337*mbkin**4*mcMS**2 + 674*mbkin**2*mcMS**4 + 
                          255*mcMS**6)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**16)) + q_cut**3*
                      ((-4*(3 + 8*mbkin)*(55 + (238*mcMS**2)/mbkin**2 + 
                          (226*mcMS**4)/mbkin**4 + (238*mcMS**6)/mbkin**6 + 
                          (55*mcMS**8)/mbkin**8))/(3*mbkin**8) + 
                       (4*(119*mbkin**6*mcMS**2 + 226*mbkin**4*mcMS**4 + 
                          357*mbkin**2*mcMS**6 + 110*mcMS**8)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                        (9*mbkin**16)) + q_cut**2*((-8*(3 + 8*mbkin)*(60 + 
                          (38*mcMS**2)/mbkin**2 - (518*mcMS**4)/mbkin**4 - 
                          (518*mcMS**6)/mbkin**6 + (38*mcMS**8)/mbkin**8 + 
                          (60*mcMS**10)/mbkin**10))/(9*mbkin**6) + 
                       (4*(19*mbkin**8*mcMS**2 - 518*mbkin**6*mcMS**4 - 
                          777*mbkin**4*mcMS**6 + 76*mbkin**2*mcMS**8 + 150*
                          mcMS**10)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**16)) + 
                     q_cut*((-4*(3 + 8*mbkin)*(-91 + (202*mcMS**2)/mbkin**2 + 
                          (2591*mcMS**4)/mbkin**4 + (4676*mcMS**6)/mbkin**6 + 
                          (2591*mcMS**8)/mbkin**8 + (202*mcMS**10)/mbkin**10 - 
                          (91*mcMS**12)/mbkin**12))/(9*mbkin**4) + 
                       (4*(101*mbkin**10*mcMS**2 + 2591*mbkin**8*mcMS**4 + 
                          7014*mbkin**6*mcMS**6 + 5182*mbkin**4*mcMS**8 + 
                          505*mbkin**2*mcMS**10 - 273*mcMS**12)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                        (9*mbkin**16))))))*np.log((mbkin**2 + mcMS**2 - q_cut - 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                      2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                 (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4)))**2)/mbkin**4 - (72*mcMS**4*(mbkin**4 - 
                2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                q_cut**2)**2*(31*mbkin**10 - 153*mbkin**8*mcMS**2 - 1138*mbkin**6*
                mcMS**4 - 1138*mbkin**4*mcMS**6 - 153*mbkin**2*mcMS**8 + 31*
                mcMS**10 - 29*mbkin**8*q_cut - 100*mbkin**6*mcMS**2*q_cut - 162*mbkin**4*
                mcMS**4*q_cut - 100*mbkin**2*mcMS**6*q_cut - 29*mcMS**8*q_cut - 29*mbkin**6*
                q_cut**2 - 125*mbkin**4*mcMS**2*q_cut**2 - 125*mbkin**2*mcMS**4*q_cut**2 - 29*
                mcMS**6*q_cut**2 + 26*mbkin**4*q_cut**3 + 82*mbkin**2*mcMS**2*q_cut**3 + 26*
                mcMS**4*q_cut**3 - 4*mbkin**2*q_cut**4 - 4*mcMS**2*q_cut**4 + 5*q_cut**5)*
              ((mcMS**4*(16/3 + 4*np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - 
                     q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                    (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 + mcMS**4*
                ((-16*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 6*mbkin**2*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                     np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                         q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                    9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - q_cut - 
                      mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                     (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))))/(9*mbkin**4*(mbkin**4 - 
                    2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                    q_cut**2)*(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + 
                    mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                 (8*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                        np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                       mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                          q_cut**2)/mbkin**4)))**2)/(9*mbkin**6))))/mbkin**18) - 
          8640*((mcMS**8*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                  q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(-64*mbkin - 
               (1088*mcMS**2)/(3*mbkin) - (4480*mcMS**4)/(3*mbkin**3) - 
               (1408*mcMS**6)/(3*mbkin**5) + 288*mcMS**4*(4 + 
                 3*np.log(mu0**2/mcMS**2)) + (192*mcMS**8*(-3 - 8*mbkin + 
                  12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/mbkin**6 + 
               (128*mcMS**6*(-6 - 16*mbkin + 36*mbkin**2 + 27*mbkin**2*
                   np.log(mu0**2/mcMS**2)))/mbkin**4)*np.log((mbkin**2 + mcMS**2 - q_cut - 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                      2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                 (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                     mbkin**4)))**3)/mbkin**8 + (216*mcMS**4 + (576*mcMS**6)/mbkin**
                2 + (216*mcMS**8)/mbkin**4)*((3*mcMS**8*((mbkin**4 - 2*mbkin**2*
                    mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                  mbkin**4)**(3/2)*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
                 (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 
                    12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                 (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                     mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                     np.log(mu0**2/mcMS**2)))/(9*mbkin**6))*
                np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                    mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/(2*
                mbkin**8*((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/
                  mbkin**4 + q_cut**2/mbkin**4)) + ((mbkin**4 - 2*mbkin**2*mcMS**2 + 
                  mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(
                (mcMS**8*(32/3 + 8*np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - 
                      q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                          mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                         mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8 + 
                mcMS**8*((-8*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                         q_cut**2)/mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + 
                       (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                         2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 6*mbkin**2*mcMS**4*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 16*mbkin**3*
                      mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 
                     12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                        mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*
                      mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                     9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                      np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                      mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                      np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - q_cut - 
                        mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                       (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)))**2)/(3*mbkin**8*(mbkin**4 - 
                     2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                     q_cut**2)*(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + 
                     mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                  (16*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                         np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                       (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)))**3)/(9*mbkin**10))))))/
         (((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
             2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 7*
                mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - mbkin**2*
                q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
            (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                   (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                     2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                 mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                     2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
           3)))/540 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

def q2moment3MS(q_cut, mbkin, mcMS, mus, mu0, api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
 res = ( 
    (mbkin**2*((180*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
            2*mcMS**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
            7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
            mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)**2*(mbkin**12 - 
           34*mbkin**10*mcMS**2 - 1133*mbkin**8*mcMS**4 - 2708*mbkin**6*mcMS**6 - 
           1133*mbkin**4*mcMS**8 - 34*mbkin**2*mcMS**10 + mcMS**12 + mbkin**10*q_cut - 
           31*mbkin**8*mcMS**2*q_cut - 390*mbkin**6*mcMS**4*q_cut - 390*mbkin**4*mcMS**6*
            q_cut - 31*mbkin**2*mcMS**8*q_cut + mcMS**10*q_cut + mbkin**8*q_cut**2 - 
           26*mbkin**6*mcMS**2*q_cut**2 - 118*mbkin**4*mcMS**4*q_cut**2 - 
           26*mbkin**2*mcMS**6*q_cut**2 + mcMS**8*q_cut**2 + mbkin**6*q_cut**3 - 
           19*mbkin**4*mcMS**2*q_cut**3 - 19*mbkin**2*mcMS**4*q_cut**3 + mcMS**6*q_cut**3 - 
           6*mbkin**4*q_cut**4 + 4*mbkin**2*mcMS**2*q_cut**4 - 6*mcMS**4*q_cut**4 - 
           6*mbkin**2*q_cut**5 - 6*mcMS**2*q_cut**5 + 8*q_cut**6))/mbkin**32 - 
        (2160*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
            2*mcMS**2*q_cut + q_cut**2)**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
             2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
          (37*mbkin**18 - 397*mbkin**16*mcMS**2 - 2854*mbkin**14*mcMS**4 + 
           18134*mbkin**12*mcMS**6 + 75800*mbkin**10*mcMS**8 + 
           75800*mbkin**8*mcMS**10 + 18134*mbkin**6*mcMS**12 - 
           2854*mbkin**4*mcMS**14 - 397*mbkin**2*mcMS**16 + 37*mcMS**18 - 
           70*mbkin**16*q_cut + 132*mbkin**14*mcMS**2*q_cut + 4424*mbkin**12*mcMS**4*
            q_cut + 15500*mbkin**10*mcMS**6*q_cut + 20508*mbkin**8*mcMS**8*q_cut + 
           15500*mbkin**6*mcMS**10*q_cut + 4424*mbkin**4*mcMS**12*q_cut + 
           132*mbkin**2*mcMS**14*q_cut - 70*mcMS**16*q_cut - 37*mbkin**14*q_cut**2 + 
           307*mbkin**12*mcMS**2*q_cut**2 + 6201*mbkin**10*mcMS**4*q_cut**2 + 
           18225*mbkin**8*mcMS**6*q_cut**2 + 18225*mbkin**6*mcMS**8*q_cut**2 + 
           6201*mbkin**4*mcMS**10*q_cut**2 + 307*mbkin**2*mcMS**12*q_cut**2 - 
           37*mcMS**14*q_cut**2 + 140*mbkin**12*q_cut**3 + 272*mbkin**10*mcMS**2*q_cut**3 - 
           2796*mbkin**8*mcMS**4*q_cut**3 - 7136*mbkin**6*mcMS**6*q_cut**3 - 
           2796*mbkin**4*mcMS**8*q_cut**3 + 272*mbkin**2*mcMS**10*q_cut**3 + 
           140*mcMS**12*q_cut**3 - 49*mbkin**10*q_cut**4 + 13*mbkin**8*mcMS**2*q_cut**4 - 
           300*mbkin**6*mcMS**4*q_cut**4 - 300*mbkin**4*mcMS**6*q_cut**4 + 
           13*mbkin**2*mcMS**8*q_cut**4 - 49*mcMS**10*q_cut**4 - 70*mbkin**8*q_cut**5 - 
           372*mbkin**6*mcMS**2*q_cut**5 - 668*mbkin**4*mcMS**4*q_cut**5 - 
           372*mbkin**2*mcMS**6*q_cut**5 - 70*mcMS**8*q_cut**5 + 77*mbkin**6*q_cut**6 + 
           41*mbkin**4*mcMS**2*q_cut**6 + 41*mbkin**2*mcMS**4*q_cut**6 + 77*mcMS**6*q_cut**6 - 
           16*mbkin**4*q_cut**7 + 32*mbkin**2*mcMS**2*q_cut**7 - 16*mcMS**4*q_cut**7 - 
           28*mbkin**2*q_cut**8 - 28*mcMS**2*q_cut**8 + 16*q_cut**9)*
          np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**26 + 
        (25920*mcMS**8*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
            2*mcMS**2*q_cut + q_cut**2)**2*(71*mbkin**12 - 174*mbkin**10*mcMS**2 - 
           3723*mbkin**8*mcMS**4 - 7468*mbkin**6*mcMS**6 - 3723*mbkin**4*mcMS**8 - 
           174*mbkin**2*mcMS**10 + 71*mcMS**12 - 69*mbkin**10*q_cut - 
           381*mbkin**8*mcMS**2*q_cut - 810*mbkin**6*mcMS**4*q_cut - 
           810*mbkin**4*mcMS**6*q_cut - 381*mbkin**2*mcMS**8*q_cut - 69*mcMS**10*q_cut - 
           69*mbkin**8*q_cut**2 - 446*mbkin**6*mcMS**2*q_cut**2 - 818*mbkin**4*mcMS**4*
            q_cut**2 - 446*mbkin**2*mcMS**6*q_cut**2 - 69*mcMS**8*q_cut**2 + 
           71*mbkin**6*q_cut**3 + 331*mbkin**4*mcMS**2*q_cut**3 + 331*mbkin**2*mcMS**4*
            q_cut**3 + 71*mcMS**6*q_cut**3 - 6*mbkin**4*q_cut**4 + 4*mbkin**2*mcMS**2*q_cut**4 - 
           6*mcMS**4*q_cut**4 - 6*mbkin**2*q_cut**5 - 6*mcMS**2*q_cut**5 + 8*q_cut**6)*
          np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**24 - 
        (10886400*mcMS**12*(mbkin**6 + 5*mbkin**4*mcMS**2 + 5*mbkin**2*mcMS**4 + 
           mcMS**6)*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
             2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**14))/
      (2520*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
          2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
            7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
            mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**3) + 
     (mbkin**2*(((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
          q_cut**2/mbkin**4)*(-72*mbkin**2*muG*((-1 + mcMS**2/mbkin**2)**2 - 
             (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)**2*
           (25 + (789*mcMS**2)/mbkin**2 - (11707*mcMS**4)/mbkin**4 + 
            (28953*mcMS**6)/mbkin**6 + (73278*mcMS**8)/mbkin**8 - 
            (325002*mcMS**10)/mbkin**10 - (894522*mcMS**12)/mbkin**12 - 
            (477522*mcMS**14)/mbkin**14 + (113193*mcMS**16)/mbkin**16 + 
            (47093*mcMS**18)/mbkin**18 - (6027*mcMS**20)/mbkin**20 - 
            (71*mcMS**22)/mbkin**22 + ((-35 - (1504*mcMS**2)/mbkin**2 + 
               (15413*mcMS**4)/mbkin**4 - (30432*mcMS**6)/mbkin**6 - 
               (180506*mcMS**8)/mbkin**8 - (158416*mcMS**10)/mbkin**10 - 
               (166746*mcMS**12)/mbkin**12 - (202976*mcMS**14)/mbkin**14 - 
               (10627*mcMS**16)/mbkin**16 + (9968*mcMS**18)/mbkin**18 + 
               (101*mcMS**20)/mbkin**20)*q_cut)/mbkin**2 + 
            (2*(-30 - (544*mcMS**2)/mbkin**2 + (7709*mcMS**4)/mbkin**4 - 
               (22753*mcMS**6)/mbkin**6 - (158279*mcMS**8)/mbkin**8 - 
               (280433*mcMS**10)/mbkin**10 - (161069*mcMS**12)/mbkin**12 - 
               (18783*mcMS**14)/mbkin**14 + (5109*mcMS**16)/mbkin**16 + 
               (81*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
            (2*(-35 - (1674*mcMS**2)/mbkin**2 + (14879*mcMS**4)/mbkin**4 + 
               (17828*mcMS**6)/mbkin**6 - (66689*mcMS**8)/mbkin**8 - 
               (71234*mcMS**10)/mbkin**10 + (21933*mcMS**12)/mbkin**12 + 
               (10784*mcMS**14)/mbkin**14 + (96*mcMS**16)/mbkin**16)*q_cut**3)/
             mbkin**6 - (2*(-35 + (183*mcMS**2)/mbkin**2 + (448*mcMS**4)/
                mbkin**4 + (902*mcMS**6)/mbkin**6 + (14355*mcMS**8)/mbkin**8 - 
               (503*mcMS**10)/mbkin**10 + (1208*mcMS**12)/mbkin**12 + 
               (98*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
            (2*mcMS**2*(-1184 + (8011*mcMS**2)/mbkin**2 + (29884*mcMS**4)/
                mbkin**4 + (29424*mcMS**6)/mbkin**6 + (6156*mcMS**8)/mbkin**8 + 
               (21*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**12 + 
            (2*(-70 + (692*mcMS**2)/mbkin**2 - (1269*mcMS**4)/mbkin**4 - 
               (6871*mcMS**6)/mbkin**6 - (389*mcMS**8)/mbkin**8 + (147*mcMS**10)/
                mbkin**10)*q_cut**6)/mbkin**12 + (2*(5 + (14*mcMS**2)/mbkin**2 - 
               (339*mcMS**4)/mbkin**4 - (284*mcMS**6)/mbkin**6 + (4*mcMS**8)/
                mbkin**8)*q_cut**7)/mbkin**14 - (3*(-35 + (141*mcMS**2)/mbkin**2 + 
               (439*mcMS**4)/mbkin**4 + (71*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
            ((-45 + (176*mcMS**2)/mbkin**2 + (41*mcMS**4)/mbkin**4)*q_cut**9)/
             mbkin**18 + (24*mcMS**2*q_cut**10)/mbkin**22) - 36*muG*mupi*
           ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
             q_cut**2/mbkin**4)**2*(25 + (789*mcMS**2)/mbkin**2 - (11707*mcMS**4)/
             mbkin**4 + (28953*mcMS**6)/mbkin**6 + (73278*mcMS**8)/mbkin**8 - 
            (325002*mcMS**10)/mbkin**10 - (894522*mcMS**12)/mbkin**12 - 
            (477522*mcMS**14)/mbkin**14 + (113193*mcMS**16)/mbkin**16 + 
            (47093*mcMS**18)/mbkin**18 - (6027*mcMS**20)/mbkin**20 - 
            (71*mcMS**22)/mbkin**22 + ((-35 - (1504*mcMS**2)/mbkin**2 + 
               (15413*mcMS**4)/mbkin**4 - (30432*mcMS**6)/mbkin**6 - 
               (180506*mcMS**8)/mbkin**8 - (158416*mcMS**10)/mbkin**10 - 
               (166746*mcMS**12)/mbkin**12 - (202976*mcMS**14)/mbkin**14 - 
               (10627*mcMS**16)/mbkin**16 + (9968*mcMS**18)/mbkin**18 + 
               (101*mcMS**20)/mbkin**20)*q_cut)/mbkin**2 + 
            (2*(-30 - (544*mcMS**2)/mbkin**2 + (7709*mcMS**4)/mbkin**4 - 
               (22753*mcMS**6)/mbkin**6 - (158279*mcMS**8)/mbkin**8 - 
               (280433*mcMS**10)/mbkin**10 - (161069*mcMS**12)/mbkin**12 - 
               (18783*mcMS**14)/mbkin**14 + (5109*mcMS**16)/mbkin**16 + 
               (81*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
            (2*(-35 - (1674*mcMS**2)/mbkin**2 + (14879*mcMS**4)/mbkin**4 + 
               (17828*mcMS**6)/mbkin**6 - (66689*mcMS**8)/mbkin**8 - 
               (71234*mcMS**10)/mbkin**10 + (21933*mcMS**12)/mbkin**12 + 
               (10784*mcMS**14)/mbkin**14 + (96*mcMS**16)/mbkin**16)*q_cut**3)/
             mbkin**6 - (2*(-35 + (183*mcMS**2)/mbkin**2 + (448*mcMS**4)/
                mbkin**4 + (902*mcMS**6)/mbkin**6 + (14355*mcMS**8)/mbkin**8 - 
               (503*mcMS**10)/mbkin**10 + (1208*mcMS**12)/mbkin**12 + 
               (98*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
            (2*mcMS**2*(-1184 + (8011*mcMS**2)/mbkin**2 + (29884*mcMS**4)/
                mbkin**4 + (29424*mcMS**6)/mbkin**6 + (6156*mcMS**8)/mbkin**8 + 
               (21*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**12 + 
            (2*(-70 + (692*mcMS**2)/mbkin**2 - (1269*mcMS**4)/mbkin**4 - 
               (6871*mcMS**6)/mbkin**6 - (389*mcMS**8)/mbkin**8 + (147*mcMS**10)/
                mbkin**10)*q_cut**6)/mbkin**12 + (2*(5 + (14*mcMS**2)/mbkin**2 - 
               (339*mcMS**4)/mbkin**4 - (284*mcMS**6)/mbkin**6 + (4*mcMS**8)/
                mbkin**8)*q_cut**7)/mbkin**14 - (3*(-35 + (141*mcMS**2)/mbkin**2 + 
               (439*mcMS**4)/mbkin**4 + (71*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
            ((-45 + (176*mcMS**2)/mbkin**2 + (41*mcMS**4)/mbkin**4)*q_cut**9)/
             mbkin**18 + (24*mcMS**2*q_cut**10)/mbkin**22) + 
          36*muG**2*((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/
              mbkin**4 + q_cut**2/mbkin**4)**2*(-75 - (2767*mcMS**2)/mbkin**2 + 
            (18697*mcMS**4)/mbkin**4 - (48675*mcMS**6)/mbkin**6 - 
            (55770*mcMS**8)/mbkin**8 - (17058*mcMS**10)/mbkin**10 - 
            (650082*mcMS**12)/mbkin**12 - (1112826*mcMS**14)/mbkin**14 + 
            (320421*mcMS**16)/mbkin**16 + (128241*mcMS**18)/mbkin**18 - 
            (31271*mcMS**20)/mbkin**20 - (355*mcMS**22)/mbkin**22 + 
            ((-95 - (2840*mcMS**2)/mbkin**2 + (9705*mcMS**4)/mbkin**4 - 
               (48896*mcMS**6)/mbkin**6 - (295810*mcMS**8)/mbkin**8 - 
               (193920*mcMS**10)/mbkin**10 + (406750*mcMS**12)/mbkin**12 - 
               (605248*mcMS**14)/mbkin**14 - (45663*mcMS**16)/mbkin**16 + 
               (49752*mcMS**18)/mbkin**18 + (505*mcMS**20)/mbkin**20)*q_cut)/
             mbkin**2 + (2*(30 + (436*mcMS**2)/mbkin**2 - (26411*mcMS**4)/
                mbkin**4 - (7345*mcMS**6)/mbkin**6 - (184695*mcMS**8)/mbkin**8 - 
               (575529*mcMS**10)/mbkin**10 - (334469*mcMS**12)/mbkin**12 - 
               (11887*mcMS**14)/mbkin**14 + (26633*mcMS**16)/mbkin**16 + 
               (405*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
            (2*(-215 - (5310*mcMS**2)/mbkin**2 + (48067*mcMS**4)/mbkin**4 + 
               (66136*mcMS**6)/mbkin**6 - (122925*mcMS**8)/mbkin**8 - 
               (294958*mcMS**10)/mbkin**10 + (100777*mcMS**12)/mbkin**12 + 
               (53196*mcMS**14)/mbkin**14 + (480*mcMS**16)/mbkin**16)*q_cut**3)/
             mbkin**6 - (2*(-75 + (535*mcMS**2)/mbkin**2 - (8184*mcMS**4)/
                mbkin**4 + (51670*mcMS**6)/mbkin**6 + (174771*mcMS**8)/mbkin**8 + 
               (22097*mcMS**10)/mbkin**10 + (6520*mcMS**12)/mbkin**12 + 
               (490*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
            ((-480 - (7288*mcMS**2)/mbkin**2 + (79886*mcMS**4)/mbkin**4 + 
               (239560*mcMS**6)/mbkin**6 + (261440*mcMS**8)/mbkin**8 + 
               (58976*mcMS**10)/mbkin**10 + (210*mcMS**12)/mbkin**12)*q_cut**5)/
             mbkin**10 + (2*(-170 + (2000*mcMS**2)/mbkin**2 - (3941*mcMS**4)/
                mbkin**4 - (35663*mcMS**6)/mbkin**6 - (1921*mcMS**8)/mbkin**8 + 
               (735*mcMS**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
            (2*(25 - (206*mcMS**2)/mbkin**2 - (415*mcMS**4)/mbkin**4 - 
               (968*mcMS**6)/mbkin**6 + (20*mcMS**8)/mbkin**8)*q_cut**7)/mbkin**14 - 
            (3*(-175 + (705*mcMS**2)/mbkin**2 + (2131*mcMS**4)/mbkin**4 + 
               (355*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
            (5*(-45 + (176*mcMS**2)/mbkin**2 + (41*mcMS**4)/mbkin**4)*q_cut**9)/
             mbkin**18 + (120*mcMS**2*q_cut**10)/mbkin**22) - 
          24*mbkin*(-((-1 + mcMS**2/mbkin**2)**4*(1 + mcMS**2/mbkin**2)**2*
              (2803 - (59389*mcMS**2)/mbkin**2 + (425638*mcMS**4)/mbkin**4 - 
               (142434*mcMS**6)/mbkin**6 - (7007992*mcMS**8)/mbkin**8 - 
               (411584*mcMS**10)/mbkin**10 + (753258*mcMS**12)/mbkin**12 - 
               (298894*mcMS**14)/mbkin**14 + (24853*mcMS**16)/mbkin**16 + 
               (461*mcMS**18)/mbkin**18)) + ((-1 + mcMS**2/mbkin**2)**2*
              (15765 - (259196*mcMS**2)/mbkin**2 + (1031280*mcMS**4)/mbkin**4 + 
               (5842740*mcMS**6)/mbkin**6 - (24509183*mcMS**8)/mbkin**8 - 
               (87162472*mcMS**10)/mbkin**10 - (89842656*mcMS**12)/mbkin**12 - 
               (25831304*mcMS**14)/mbkin**14 + (7046591*mcMS**16)/mbkin**16 - 
               (151836*mcMS**18)/mbkin**18 - (1156752*mcMS**20)/mbkin**20 + 
               (149588*mcMS**22)/mbkin**22 + (2475*mcMS**24)/mbkin**24)*q_cut)/
             mbkin**2 - (2*(15112 - (210025*mcMS**2)/mbkin**2 + (472499*mcMS**4)/
                mbkin**4 + (5851497*mcMS**6)/mbkin**6 - (11357051*mcMS**8)/
                mbkin**8 - (56972502*mcMS**10)/mbkin**10 - (85340102*mcMS**12)/
                mbkin**12 - (61750930*mcMS**14)/mbkin**14 - (11562978*mcMS**16)/
                mbkin**16 + (7576839*mcMS**18)/mbkin**18 - (785517*mcMS**20)/
                mbkin**20 - (914135*mcMS**22)/mbkin**22 + (150229*mcMS**24)/
                mbkin**24 + (2104*mcMS**26)/mbkin**26)*q_cut**2)/mbkin**4 + 
            (2*(4935 - (22515*mcMS**2)/mbkin**2 - (355240*mcMS**4)/mbkin**4 + 
               (1715991*mcMS**6)/mbkin**6 + (2655127*mcMS**8)/mbkin**8 - 
               (4426866*mcMS**10)/mbkin**10 - (3311288*mcMS**12)/mbkin**12 + 
               (4104698*mcMS**14)/mbkin**14 + (1043701*mcMS**16)/mbkin**16 - 
               (1487323*mcMS**18)/mbkin**18 + (15840*mcMS**20)/mbkin**20 + 
               (62895*mcMS**22)/mbkin**22 + (45*mcMS**24)/mbkin**24)*q_cut**3)/
             mbkin**6 + ((43537 - (458971*mcMS**2)/mbkin**2 - (93925*mcMS**4)/
                mbkin**4 + (12415735*mcMS**6)/mbkin**6 + (21452530*mcMS**8)/
                mbkin**8 + (19866834*mcMS**10)/mbkin**10 + (19989798*mcMS**12)/
                mbkin**12 + (10268870*mcMS**14)/mbkin**14 - (3574435*mcMS**16)/
                mbkin**16 - (1439255*mcMS**18)/mbkin**18 + (436447*mcMS**20)/
                mbkin**20 + (7139*mcMS**22)/mbkin**22)*q_cut**4)/mbkin**8 - 
            ((58149 - (424380*mcMS**2)/mbkin**2 - (1661061*mcMS**4)/mbkin**4 + 
               (11127360*mcMS**6)/mbkin**6 + (29799650*mcMS**8)/mbkin**8 + 
               (27800000*mcMS**10)/mbkin**10 + (5553270*mcMS**12)/mbkin**12 - 
               (6210720*mcMS**14)/mbkin**14 - (461543*mcMS**16)/mbkin**16 + 
               (648988*mcMS**18)/mbkin**18 + (6063*mcMS**20)/mbkin**20)*q_cut**5)/
             mbkin**10 + (4*(2930 - (20127*mcMS**2)/mbkin**2 - (145371*mcMS**4)/
                mbkin**4 + (861647*mcMS**6)/mbkin**6 + (1702217*mcMS**8)/
                mbkin**8 + (293611*mcMS**10)/mbkin**10 - (560795*mcMS**12)/
                mbkin**12 + (4239*mcMS**14)/mbkin**14 + (35411*mcMS**16)/
                mbkin**16 + (110*mcMS**18)/mbkin**18)*q_cut**6)/mbkin**12 - 
            (4*(-5667 + (16545*mcMS**2)/mbkin**2 + (282306*mcMS**4)/mbkin**4 + 
               (576507*mcMS**6)/mbkin**6 + (446228*mcMS**8)/mbkin**8 + 
               (21045*mcMS**10)/mbkin**10 - (206250*mcMS**12)/mbkin**12 - 
               (83609*mcMS**14)/mbkin**14 + (951*mcMS**16)/mbkin**16)*q_cut**7)/
             mbkin**14 + ((-13425 + (43605*mcMS**2)/mbkin**2 + (1070385*mcMS**4)/
                mbkin**4 + (1555763*mcMS**6)/mbkin**6 + (247781*mcMS**8)/
                mbkin**8 - (859585*mcMS**10)/mbkin**10 - (271765*mcMS**12)/
                mbkin**12 + (8265*mcMS**14)/mbkin**14)*q_cut**8)/mbkin**16 + 
            ((3211 - (5958*mcMS**2)/mbkin**2 - (298825*mcMS**4)/mbkin**4 - 
               (133880*mcMS**6)/mbkin**6 + (266305*mcMS**8)/mbkin**8 + 
               (71726*mcMS**10)/mbkin**10 - (643*mcMS**12)/mbkin**12)*q_cut**9)/
             mbkin**18 - (2*(3444 + (8075*mcMS**2)/mbkin**2 + (7199*mcMS**4)/
                mbkin**4 + (24877*mcMS**6)/mbkin**6 + (14553*mcMS**8)/mbkin**8 + 
               (4308*mcMS**10)/mbkin**10)*q_cut**10)/mbkin**20 + 
            (2*(3295 + (6917*mcMS**2)/mbkin**2 + (13436*mcMS**4)/mbkin**4 + 
               (14159*mcMS**6)/mbkin**6 + (3805*mcMS**8)/mbkin**8)*q_cut**11)/
             mbkin**22 - ((1869 + (3645*mcMS**2)/mbkin**2 + (9103*mcMS**4)/
                mbkin**4 + (2463*mcMS**6)/mbkin**6)*q_cut**12)/mbkin**24 + 
            (5*(mbkin**4 + 80*mbkin**2*mcMS**2 + 59*mcMS**4)*q_cut**13)/mbkin**30 - 
            (48*(mbkin**2 + 2*mcMS**2)*q_cut**14)/mbkin**30 + (40*q_cut**15)/mbkin**30)*
           rhoD + ((mbkin**6 - 7*mbkin**4*mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - 
             mbkin**4*q_cut - mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)*
            (-16*(-((-1 + mcMS**2/mbkin**2)**4*(-16529 + (91326*mcMS**2)/
                   mbkin**2 + (305370*mcMS**4)/mbkin**4 - (252698*mcMS**6)/
                   mbkin**6 - (524676*mcMS**8)/mbkin**8 - (176622*mcMS**10)/
                   mbkin**10 + (103798*mcMS**12)/mbkin**12 + (105354*mcMS**14)/
                   mbkin**14 + (1797*mcMS**16)/mbkin**16)) + 
               (2*(-1 + mcMS**2/mbkin**2)**2*(-37623 + (116641*mcMS**2)/mbkin**2 + 
                  (808206*mcMS**4)/mbkin**4 - (28274*mcMS**6)/mbkin**6 - 
                  (1319312*mcMS**8)/mbkin**8 - (1291200*mcMS**10)/mbkin**10 - 
                  (352094*mcMS**12)/mbkin**12 + (401986*mcMS**14)/mbkin**14 + 
                  (246231*mcMS**16)/mbkin**16 + (3919*mcMS**18)/mbkin**18)*q_cut)/
                mbkin**2 + ((114885 - (167094*mcMS**2)/mbkin**2 - 
                  (2139307*mcMS**4)/mbkin**4 + (43388*mcMS**6)/mbkin**6 + 
                  (2018354*mcMS**8)/mbkin**8 + (2751296*mcMS**10)/mbkin**10 + 
                  (4252250*mcMS**12)/mbkin**12 + (1266308*mcMS**14)/mbkin**14 - 
                  (1541527*mcMS**16)/mbkin**16 - (782218*mcMS**18)/mbkin**18 - 
                  (10255*mcMS**20)/mbkin**20)*q_cut**2)/mbkin**4 - 
               (2*(15345 + (35887*mcMS**2)/mbkin**2 - (319224*mcMS**4)/mbkin**4 + 
                  (127134*mcMS**6)/mbkin**6 + (1286022*mcMS**8)/mbkin**8 + 
                  (745104*mcMS**10)/mbkin**10 - (914648*mcMS**12)/mbkin**12 - 
                  (813694*mcMS**14)/mbkin**14 - (131991*mcMS**16)/mbkin**16 + 
                  (305*mcMS**18)/mbkin**18)*q_cut**3)/mbkin**6 + (2*(-47995 - 
                  (123687*mcMS**2)/mbkin**2 + (174111*mcMS**4)/mbkin**4 + 
                  (564235*mcMS**6)/mbkin**6 + (577645*mcMS**8)/mbkin**8 + 
                  (446275*mcMS**10)/mbkin**10 + (745753*mcMS**12)/mbkin**12 + 
                  (310881*mcMS**14)/mbkin**14 + (4830*mcMS**16)/mbkin**16)*q_cut**4)/
                mbkin**8 - (2*(-51671 - (201155*mcMS**2)/mbkin**2 + 
                  (163523*mcMS**4)/mbkin**4 + (897993*mcMS**6)/mbkin**6 + 
                  (1593699*mcMS**8)/mbkin**8 + (1338455*mcMS**10)/mbkin**10 + 
                  (348873*mcMS**12)/mbkin**12 + (2523*mcMS**14)/mbkin**14)*q_cut**5)/
                mbkin**10 + (2*(-17689 - (72181*mcMS**2)/mbkin**2 + 
                  (150624*mcMS**4)/mbkin**4 + (551050*mcMS**6)/mbkin**6 + 
                  (492577*mcMS**8)/mbkin**8 + (109095*mcMS**10)/mbkin**10 + 
                  (4452*mcMS**12)/mbkin**12)*q_cut**6)/mbkin**12 - 
               (2*(-3065 - (177*mcMS**2)/mbkin**2 + (31314*mcMS**4)/mbkin**4 + 
                  (22874*mcMS**6)/mbkin**6 + (17991*mcMS**8)/mbkin**8 + 
                  (12615*mcMS**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
               ((-7235 + (7924*mcMS**2)/mbkin**2 + (34922*mcMS**4)/mbkin**4 + 
                  (61668*mcMS**6)/mbkin**6 + (26385*mcMS**8)/mbkin**8)*q_cut**8)/
                mbkin**16 - (16*(-290 + (471*mcMS**2)/mbkin**2 + (1857*mcMS**4)/
                   mbkin**4 + (735*mcMS**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
               ((-971 + (1612*mcMS**2)/mbkin**2 + (2143*mcMS**4)/mbkin**4)*q_cut**10)/
                mbkin**20 - (8*(22 + (49*mcMS**2)/mbkin**2)*q_cut**11)/mbkin**22 + 
               (160*q_cut**12)/mbkin**24)*rE - ((-1 + mcMS**2/mbkin**2)**2 - 
               (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
              (-4*(-((-1 + mcMS**2/mbkin**2)**2*(-27053 + (86226*mcMS**2)/
                     mbkin**2 + (575046*mcMS**4)/mbkin**4 - (609158*mcMS**6)/
                     mbkin**6 - (1436880*mcMS**8)/mbkin**8 - (602802*mcMS**10)/
                     mbkin**10 + (123994*mcMS**12)/mbkin**12 + (75174*mcMS**14)/
                     mbkin**14 + (1053*mcMS**16)/mbkin**16)) + 
                 (4*(-17144 - (6564*mcMS**2)/mbkin**2 + (379329*mcMS**4)/
                     mbkin**4 - (45999*mcMS**6)/mbkin**6 - (941649*mcMS**8)/
                     mbkin**8 - (1045641*mcMS**10)/mbkin**10 - (306441*mcMS**12)/
                     mbkin**12 + (120231*mcMS**14)/mbkin**14 + (48849*mcMS**16)/
                     mbkin**16 + (629*mcMS**18)/mbkin**18)*q_cut)/mbkin**2 - 
                 (6*(-3605 - (29460*mcMS**2)/mbkin**2 + (11114*mcMS**4)/
                     mbkin**4 - (6348*mcMS**6)/mbkin**6 - (33800*mcMS**8)/
                     mbkin**8 + (77508*mcMS**10)/mbkin**10 + (72846*mcMS**12)/
                     mbkin**12 + (12540*mcMS**14)/mbkin**14 + (5*mcMS**16)/
                     mbkin**16)*q_cut**2)/mbkin**4 - (2*(-32263 - (142675*mcMS**2)/
                     mbkin**2 + (7835*mcMS**4)/mbkin**4 + (190079*mcMS**6)/
                     mbkin**6 + (442691*mcMS**8)/mbkin**8 + (369935*mcMS**10)/
                     mbkin**10 + (90345*mcMS**12)/mbkin**12 + (1413*mcMS**14)/
                     mbkin**14)*q_cut**3)/mbkin**6 + ((-47518 - (330552*mcMS**2)/
                     mbkin**2 + (106806*mcMS**4)/mbkin**4 + (962320*mcMS**6)/
                     mbkin**6 + (758274*mcMS**8)/mbkin**8 + (148632*mcMS**10)/
                     mbkin**10 - (42*mcMS**12)/mbkin**12)*q_cut**4)/mbkin**8 + 
                 (24*(53 + (1383*mcMS**2)/mbkin**2 - (2122*mcMS**4)/mbkin**4 - 
                    (3098*mcMS**6)/mbkin**6 - (303*mcMS**8)/mbkin**8 + 
                    (7*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
                 ((-5854 + (3896*mcMS**2)/mbkin**2 - (7780*mcMS**4)/mbkin**4 + 
                    (8184*mcMS**6)/mbkin**6 + (4914*mcMS**8)/mbkin**8)*q_cut**6)/
                  mbkin**12 - (10*(-917 + (375*mcMS**2)/mbkin**2 + (1749*mcMS**4)/
                     mbkin**4 + (489*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
                 (3*(-557 + (540*mcMS**2)/mbkin**2 + (377*mcMS**4)/mbkin**4)*
                   q_cut**8)/mbkin**16 + (72*(-1 + mcMS**2/mbkin**2)*q_cut**9)/
                  mbkin**18 + (40*q_cut**10)/mbkin**20)*rG - 12*
                (-((-1 + mcMS**2/mbkin**2)**2*(5857 + (22438*mcMS**2)/mbkin**2 - 
                    (394686*mcMS**4)/mbkin**4 - (890034*mcMS**6)/mbkin**6 - 
                    (634464*mcMS**8)/mbkin**8 - (262246*mcMS**10)/mbkin**10 + 
                    (35198*mcMS**12)/mbkin**12 + (60722*mcMS**14)/mbkin**14 + 
                    (895*mcMS**16)/mbkin**16)) + (4*(3566 + (30072*mcMS**2)/
                     mbkin**2 - (173049*mcMS**4)/mbkin**4 - (699873*mcMS**6)/
                     mbkin**6 - (769467*mcMS**8)/mbkin**8 - (377943*mcMS**10)/
                     mbkin**10 - (159039*mcMS**12)/mbkin**12 + (49257*mcMS**14)/
                     mbkin**14 + (39621*mcMS**16)/mbkin**16 + (535*mcMS**18)/
                     mbkin**18)*q_cut)/mbkin**2 - (2*(1905 + (41030*mcMS**2)/
                     mbkin**2 + (64112*mcMS**4)/mbkin**4 - (1502*mcMS**6)/
                     mbkin**6 + (24078*mcMS**8)/mbkin**8 + (60282*mcMS**10)/
                     mbkin**10 + (121880*mcMS**12)/mbkin**12 + (30910*mcMS**14)/
                     mbkin**14 + (25*mcMS**16)/mbkin**16)*q_cut**2)/mbkin**4 - 
                 (2*(6707 + (83567*mcMS**2)/mbkin**2 + (176017*mcMS**4)/
                     mbkin**4 + (278501*mcMS**6)/mbkin**6 + (332897*mcMS**8)/
                     mbkin**8 + (246037*mcMS**10)/mbkin**10 + (73947*mcMS**12)/
                     mbkin**12 + (1175*mcMS**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
                 ((9642 + (159460*mcMS**2)/mbkin**2 + (478626*mcMS**4)/mbkin**4 + 
                    (641576*mcMS**6)/mbkin**6 + (516490*mcMS**8)/mbkin**8 + 
                    (123972*mcMS**10)/mbkin**10 - (70*mcMS**12)/mbkin**12)*q_cut**4)/
                  mbkin**8 + (8*(-81 - (1917*mcMS**2)/mbkin**2 - (4518*mcMS**4)/
                     mbkin**4 - (7162*mcMS**6)/mbkin**6 - (901*mcMS**8)/mbkin**8 + 
                    (35*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
                 (2*(-107 - (654*mcMS**2)/mbkin**2 - (184*mcMS**4)/mbkin**4 + 
                    (3018*mcMS**6)/mbkin**6 + (1855*mcMS**8)/mbkin**8)*q_cut**6)/
                  mbkin**12 - (10*(-75 + (469*mcMS**2)/mbkin**2 + (1395*mcMS**4)/
                     mbkin**4 + (355*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
                 ((-681 + (1384*mcMS**2)/mbkin**2 + (625*mcMS**4)/mbkin**4)*q_cut**8)/
                  mbkin**16 + (24*(-3 + (5*mcMS**2)/mbkin**2)*q_cut**9)/mbkin**18 + 
                 (40*q_cut**10)/mbkin**20)*sB - 70480*sE + (105488*mcMS**2*sE)/
                mbkin**2 + (2685536*mcMS**4*sE)/mbkin**4 - (2646400*mcMS**6*sE)/
                mbkin**6 - (1847776*mcMS**8*sE)/mbkin**8 + (1481312*mcMS**10*sE)/
                mbkin**10 - (685888*mcMS**12*sE)/mbkin**12 + (827264*mcMS**14*sE)/
                mbkin**14 + (580400*mcMS**16*sE)/mbkin**16 - (423024*mcMS**18*sE)/
                mbkin**18 - (6432*mcMS**20*sE)/mbkin**20 + (175840*q_cut*sE)/
                mbkin**2 + (763584*mcMS**2*q_cut*sE)/mbkin**4 - (5905152*mcMS**4*q_cut*
                 sE)/mbkin**6 - (11687616*mcMS**6*q_cut*sE)/mbkin**8 - (6505152*
                 mcMS**8*q_cut*sE)/mbkin**10 - (2639808*mcMS**10*q_cut*sE)/mbkin**12 - 
               (2264832*mcMS**12*q_cut*sE)/mbkin**14 + (1738944*mcMS**14*q_cut*sE)/
                mbkin**16 + (1149408*mcMS**16*q_cut*sE)/mbkin**18 + (15104*mcMS**18*
                 q_cut*sE)/mbkin**20 - (50880*q_cut**2*sE)/mbkin**4 - (734880*mcMS**2*
                 q_cut**2*sE)/mbkin**6 - (354720*mcMS**4*q_cut**2*sE)/mbkin**8 + 
               (1468896*mcMS**6*q_cut**2*sE)/mbkin**10 - (348576*mcMS**8*q_cut**2*sE)/
                mbkin**12 - (3542496*mcMS**10*q_cut**2*sE)/mbkin**14 - (2898144*
                 mcMS**12*q_cut**2*sE)/mbkin**16 - (474720*mcMS**14*q_cut**2*sE)/
                mbkin**18 + (480*mcMS**16*q_cut**2*sE)/mbkin**20 - (165760*q_cut**3*sE)/
                mbkin**6 - (1377664*mcMS**2*q_cut**3*sE)/mbkin**8 - (1563904*mcMS**4*
                 q_cut**3*sE)/mbkin**10 - (872704*mcMS**6*q_cut**3*sE)/mbkin**12 - 
               (531712*mcMS**8*q_cut**3*sE)/mbkin**14 - (2099584*mcMS**10*q_cut**3*sE)/
                mbkin**16 - (1037184*mcMS**12*q_cut**3*sE)/mbkin**18 - 
               (17664*mcMS**14*q_cut**3*sE)/mbkin**20 + (116000*q_cut**4*sE)/mbkin**8 + 
               (1421952*mcMS**2*q_cut**4*sE)/mbkin**10 + (2406240*mcMS**4*q_cut**4*sE)/
                mbkin**12 + (2765056*mcMS**6*q_cut**4*sE)/mbkin**14 + (3099936*
                 mcMS**8*q_cut**4*sE)/mbkin**16 + (872640*mcMS**10*q_cut**4*sE)/
                mbkin**18 + (672*mcMS**12*q_cut**4*sE)/mbkin**20 - (3840*q_cut**5*sE)/
                mbkin**10 - (137088*mcMS**2*q_cut**5*sE)/mbkin**12 - (262656*mcMS**4*
                 q_cut**5*sE)/mbkin**14 - (336384*mcMS**6*q_cut**5*sE)/mbkin**16 - 
               (13056*mcMS**8*q_cut**5*sE)/mbkin**18 - (2688*mcMS**10*q_cut**5*sE)/
                mbkin**20 + (15200*q_cut**6*sE)/mbkin**12 - (20224*mcMS**2*q_cut**6*sE)/
                mbkin**14 - (55168*mcMS**4*q_cut**6*sE)/mbkin**16 + (3072*mcMS**6*
                 q_cut**6*sE)/mbkin**18 + (35616*mcMS**8*q_cut**6*sE)/mbkin**20 - 
               (14560*q_cut**7*sE)/mbkin**14 - (27840*mcMS**2*q_cut**7*sE)/mbkin**16 - 
               (84960*mcMS**4*q_cut**7*sE)/mbkin**18 - (34560*mcMS**6*q_cut**7*sE)/
                mbkin**20 - (2160*q_cut**8*sE)/mbkin**16 + (7824*mcMS**2*q_cut**8*sE)/
                mbkin**18 + (9984*mcMS**4*q_cut**8*sE)/mbkin**20 - (1152*mcMS**2*q_cut**9*
                 sE)/mbkin**20 + (640*q_cut**10*sE)/mbkin**20 - 7651*sqB - 
               (17644*mcMS**2*sqB)/mbkin**2 + (688991*mcMS**4*sqB)/mbkin**4 + 
               (773432*mcMS**6*sqB)/mbkin**6 - (1727530*mcMS**8*sqB)/mbkin**8 - 
               (969472*mcMS**10*sqB)/mbkin**10 + (709394*mcMS**12*sqB)/
                mbkin**12 + (562184*mcMS**14*sqB)/mbkin**14 + (9293*mcMS**16*sqB)/
                mbkin**16 - (20820*mcMS**18*sqB)/mbkin**18 - (177*mcMS**20*sqB)/
                mbkin**20 + (18532*q_cut*sqB)/mbkin**2 + (170328*mcMS**2*q_cut*sqB)/
                mbkin**4 - (941220*mcMS**4*q_cut*sqB)/mbkin**6 - (5610852*mcMS**6*q_cut*
                 sqB)/mbkin**8 - (8028036*mcMS**8*q_cut*sqB)/mbkin**10 - 
               (4624524*mcMS**10*q_cut*sqB)/mbkin**12 - (851580*mcMS**12*q_cut*sqB)/
                mbkin**14 + (216708*mcMS**14*q_cut*sqB)/mbkin**16 + (54720*mcMS**16*
                 q_cut*sqB)/mbkin**18 + (404*mcMS**18*q_cut*sqB)/mbkin**20 - 
               (5070*q_cut**2*sqB)/mbkin**4 - (109200*mcMS**2*q_cut**2*sqB)/mbkin**6 - 
               (277932*mcMS**4*q_cut**2*sqB)/mbkin**8 - (147432*mcMS**6*q_cut**2*sqB)/
                mbkin**10 - (336624*mcMS**8*q_cut**2*sqB)/mbkin**12 - (512448*
                 mcMS**10*q_cut**2*sqB)/mbkin**14 - (224244*mcMS**12*q_cut**2*sqB)/
                mbkin**16 - (20040*mcMS**14*q_cut**2*sqB)/mbkin**18 + (30*mcMS**16*
                 q_cut**2*sqB)/mbkin**20 - (17422*q_cut**3*sqB)/mbkin**6 - 
               (229786*mcMS**2*q_cut**3*sqB)/mbkin**8 - (532666*mcMS**4*q_cut**3*sqB)/
                mbkin**10 - (503470*mcMS**6*q_cut**3*sqB)/mbkin**12 - (496234*mcMS**8*
                 q_cut**3*sqB)/mbkin**14 - (294526*mcMS**10*q_cut**3*sqB)/mbkin**16 - 
               (50286*mcMS**12*q_cut**3*sqB)/mbkin**18 - (474*mcMS**14*q_cut**3*sqB)/
                mbkin**20 + (13586*q_cut**4*sqB)/mbkin**8 + (212652*mcMS**2*q_cut**4*
                 sqB)/mbkin**10 + (757458*mcMS**4*q_cut**4*sqB)/mbkin**12 + 
               (806704*mcMS**6*q_cut**4*sqB)/mbkin**14 + (343986*mcMS**8*q_cut**4*sqB)/
                mbkin**16 + (39636*mcMS**10*q_cut**4*sqB)/mbkin**18 + (42*mcMS**12*
                 q_cut**4*sqB)/mbkin**20 - (1464*q_cut**5*sqB)/mbkin**10 - 
               (19656*mcMS**2*q_cut**5*sqB)/mbkin**12 - (54288*mcMS**4*q_cut**5*sqB)/
                mbkin**14 - (34992*mcMS**6*q_cut**5*sqB)/mbkin**16 - (600*mcMS**8*
                 q_cut**5*sqB)/mbkin**18 - (168*mcMS**10*q_cut**5*sqB)/mbkin**20 - 
               (1702*q_cut**6*sqB)/mbkin**12 - (6028*mcMS**2*q_cut**6*sqB)/mbkin**14 - 
               (13792*mcMS**4*q_cut**6*sqB)/mbkin**16 + (1500*mcMS**6*q_cut**6*sqB)/
                mbkin**18 + (966*mcMS**8*q_cut**6*sqB)/mbkin**20 + (2330*q_cut**7*sqB)/
                mbkin**14 - (1470*mcMS**2*q_cut**7*sqB)/mbkin**16 - (4770*mcMS**4*
                 q_cut**7*sqB)/mbkin**18 - (810*mcMS**6*q_cut**7*sqB)/mbkin**20 - 
               (963*q_cut**8*sqB)/mbkin**16 + (876*mcMS**2*q_cut**8*sqB)/mbkin**18 + 
               (219*mcMS**4*q_cut**8*sqB)/mbkin**20 - (216*q_cut**9*sqB)/mbkin**18 - 
               (72*mcMS**2*q_cut**9*sqB)/mbkin**20 + (40*q_cut**10*sqB)/mbkin**20)))/
           mbkin**6) - 12*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
            2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
         (16*(-((-1 + mcMS**2/mbkin**2)**4*(-380 + (2665*mcMS**2)/mbkin**2 + 
               (51248*mcMS**4)/mbkin**4 - (302302*mcMS**6)/mbkin**6 - 
               (716770*mcMS**8)/mbkin**8 + (773636*mcMS**10)/mbkin**10 + 
               (1352942*mcMS**12)/mbkin**12 + (338714*mcMS**14)/mbkin**14 - 
               (183346*mcMS**16)/mbkin**16 - (220253*mcMS**18)/mbkin**18 - 
               (10574*mcMS**20)/mbkin**20 + (3060*mcMS**22)/mbkin**22)) + 
            (2*(-1 + mcMS**2/mbkin**2)**2*(-1145 + (5525*mcMS**2)/mbkin**2 + 
               (132776*mcMS**4)/mbkin**4 - (475932*mcMS**6)/mbkin**6 - 
               (2047932*mcMS**8)/mbkin**8 + (363678*mcMS**10)/mbkin**10 + 
               (3806364*mcMS**12)/mbkin**12 + (3201960*mcMS**14)/mbkin**14 + 
               (640143*mcMS**16)/mbkin**16 - (713347*mcMS**18)/mbkin**18 - 
               (545492*mcMS**20)/mbkin**20 - (21308*mcMS**22)/mbkin**22 + 
               (9270*mcMS**24)/mbkin**24)*q_cut)/mbkin**2 - 
            ((-4980 + (15565*mcMS**2)/mbkin**2 + (460790*mcMS**4)/mbkin**4 - 
               (1082073*mcMS**6)/mbkin**6 - (5400314*mcMS**8)/mbkin**8 + 
               (329986*mcMS**10)/mbkin**10 + (6336868*mcMS**12)/mbkin**12 + 
               (9252922*mcMS**14)/mbkin**14 + (9659440*mcMS**16)/mbkin**16 + 
               (2356861*mcMS**18)/mbkin**18 - (2620314*mcMS**20)/mbkin**20 - 
               (1870561*mcMS**22)/mbkin**22 - (56450*mcMS**24)/mbkin**24 + 
               (40500*mcMS**26)/mbkin**26)*q_cut**2)/mbkin**4 + 
            (4*(-770 - (425*mcMS**2)/mbkin**2 + (42860*mcMS**4)/mbkin**4 - 
               (6553*mcMS**6)/mbkin**6 - (460604*mcMS**8)/mbkin**8 + 
               (213734*mcMS**10)/mbkin**10 + (1527392*mcMS**12)/mbkin**12 + 
               (766954*mcMS**14)/mbkin**14 - (799798*mcMS**16)/mbkin**16 - 
               (839309*mcMS**18)/mbkin**18 - (191076*mcMS**20)/mbkin**20 + 
               (13455*mcMS**22)/mbkin**22 + (6300*mcMS**24)/mbkin**24)*q_cut**3)/
             mbkin**6 + (2*(-2665 - (3660*mcMS**2)/mbkin**2 + (216220*mcMS**4)/
                mbkin**4 + (333804*mcMS**6)/mbkin**6 - (482042*mcMS**8)/mbkin**8 - 
               (1290220*mcMS**10)/mbkin**10 - (1494860*mcMS**12)/mbkin**12 - 
               (1281730*mcMS**14)/mbkin**14 - (1412601*mcMS**16)/mbkin**16 - 
               (676392*mcMS**18)/mbkin**18 + (22620*mcMS**20)/mbkin**20 + 
               (21510*mcMS**22)/mbkin**22)*q_cut**4)/mbkin**8 - 
            (4*(-2655 - (9270*mcMS**2)/mbkin**2 + (135271*mcMS**4)/mbkin**4 + 
               (293165*mcMS**6)/mbkin**6 - (273693*mcMS**8)/mbkin**8 - 
               (1178103*mcMS**10)/mbkin**10 - (1794639*mcMS**12)/mbkin**12 - 
               (1361095*mcMS**14)/mbkin**14 - (355238*mcMS**16)/mbkin**16 + 
               (69807*mcMS**18)/mbkin**18 + (21330*mcMS**20)/mbkin**20)*q_cut**5)/
             mbkin**10 + (2*(-2640 - (8715*mcMS**2)/mbkin**2 + (123148*mcMS**4)/
                mbkin**4 + (214717*mcMS**6)/mbkin**6 - (539178*mcMS**8)/mbkin**8 - 
               (1417455*mcMS**10)/mbkin**10 - (1141044*mcMS**12)/mbkin**12 - 
               (236615*mcMS**14)/mbkin**14 + (69426*mcMS**16)/mbkin**16 + 
               (21060*mcMS**18)/mbkin**18)*q_cut**6)/mbkin**12 + 
            (4*(-690 - (8535*mcMS**2)/mbkin**2 - (20320*mcMS**4)/mbkin**4 + 
               (23073*mcMS**6)/mbkin**6 + (142274*mcMS**8)/mbkin**8 + 
               (174489*mcMS**10)/mbkin**10 + (129296*mcMS**12)/mbkin**12 + 
               (60345*mcMS**14)/mbkin**14 + (4860*mcMS**16)/mbkin**16)*q_cut**7)/
             mbkin**14 - ((-4320 - (43275*mcMS**2)/mbkin**2 - (35240*mcMS**4)/
                mbkin**4 + (192068*mcMS**6)/mbkin**6 + (508934*mcMS**8)/mbkin**8 + 
               (602131*mcMS**10)/mbkin**10 + (312570*mcMS**12)/mbkin**12 + 
               (28620*mcMS**14)/mbkin**14)*q_cut**8)/mbkin**16 + 
            (2*(-985 - (9315*mcMS**2)/mbkin**2 - (1095*mcMS**4)/mbkin**4 + 
               (62681*mcMS**6)/mbkin**6 + (118342*mcMS**8)/mbkin**8 + 
               (67430*mcMS**10)/mbkin**10 + (6390*mcMS**12)/mbkin**12)*q_cut**9)/
             mbkin**18 - ((-700 - (3555*mcMS**2)/mbkin**2 + (4658*mcMS**4)/
                mbkin**4 + (29699*mcMS**6)/mbkin**6 + (28886*mcMS**8)/mbkin**8 + 
               (8820*mcMS**10)/mbkin**10)*q_cut**10)/mbkin**20 + 
            (32*(-15 - (25*mcMS**2)/mbkin**2 + (131*mcMS**4)/mbkin**4 + 
               (317*mcMS**6)/mbkin**6 + (270*mcMS**8)/mbkin**8)*q_cut**11)/mbkin**22 - 
            (10*(-23 - (23*mcMS**2)/mbkin**2 + (318*mcMS**4)/mbkin**4 + 
               (414*mcMS**6)/mbkin**6)*q_cut**12)/mbkin**24 - 
            (40*(mbkin**4 - 18*mcMS**4)*q_cut**13)/mbkin**30)*rE + 
          ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)*((3780*mcMS**2*muG**2)/mbkin**2 - (23760*mcMS**4*muG**2)/
             mbkin**4 - (39348*mcMS**6*muG**2)/mbkin**6 + (673560*mcMS**8*muG**2)/
             mbkin**8 - (1433376*mcMS**10*muG**2)/mbkin**10 + 
            (4650048*mcMS**12*muG**2)/mbkin**12 + (2922696*mcMS**14*muG**2)/
             mbkin**14 - (16610832*mcMS**16*muG**2)/mbkin**16 + 
            (7950420*mcMS**18*muG**2)/mbkin**18 + (3388176*mcMS**20*muG**2)/
             mbkin**20 - (1472148*mcMS**22*muG**2)/mbkin**22 - 
            (39816*mcMS**24*muG**2)/mbkin**24 + (30600*mcMS**26*muG**2)/mbkin**26 + 
            (1260*mcMS**2*muG*mupi)/mbkin**2 - (15840*mcMS**4*muG*mupi)/
             mbkin**4 - (41148*mcMS**6*muG*mupi)/mbkin**6 + 
            (855864*mcMS**8*muG*mupi)/mbkin**8 - (2008800*mcMS**10*muG*mupi)/
             mbkin**10 - (4847904*mcMS**12*muG*mupi)/mbkin**12 + 
            (5623128*mcMS**14*muG*mupi)/mbkin**14 + (7036848*mcMS**16*muG*mupi)/
             mbkin**16 - (5620644*mcMS**18*muG*mupi)/mbkin**18 - 
            (1584000*mcMS**20*muG*mupi)/mbkin**20 + (577764*mcMS**22*muG*mupi)/
             mbkin**22 + (29592*mcMS**24*muG*mupi)/mbkin**24 - 
            (6120*mcMS**26*muG*mupi)/mbkin**26 - (5040*mcMS**2*muG**2*q_cut)/
             mbkin**4 + (13320*mcMS**4*muG**2*q_cut)/mbkin**6 - 
            (7776*mcMS**6*muG**2*q_cut)/mbkin**8 + (1056600*mcMS**8*muG**2*q_cut)/
             mbkin**10 - (2260800*mcMS**10*muG**2*q_cut)/mbkin**12 - 
            (9882000*mcMS**12*muG**2*q_cut)/mbkin**14 - (20960928*mcMS**14*muG**2*q_cut)/
             mbkin**16 - (25217712*mcMS**16*muG**2*q_cut)/mbkin**18 + 
            (700272*mcMS**18*muG**2*q_cut)/mbkin**20 + (4524552*mcMS**20*muG**2*q_cut)/
             mbkin**22 - (91008*mcMS**22*muG**2*q_cut)/mbkin**24 - 
            (124200*mcMS**24*muG**2*q_cut)/mbkin**26 - (5040*mcMS**2*muG*mupi*q_cut)/
             mbkin**4 + (47160*mcMS**4*muG*mupi*q_cut)/mbkin**6 + 
            (197856*mcMS**6*muG*mupi*q_cut)/mbkin**8 - (2047608*mcMS**8*muG*mupi*
              q_cut)/mbkin**10 - (561024*mcMS**10*muG*mupi*q_cut)/mbkin**12 + 
            (15349392*mcMS**12*muG*mupi*q_cut)/mbkin**14 + 
            (25799328*mcMS**14*muG*mupi*q_cut)/mbkin**16 + 
            (16085232*mcMS**16*muG*mupi*q_cut)/mbkin**18 - 
            (815184*mcMS**18*muG*mupi*q_cut)/mbkin**20 - (1783368*mcMS**20*muG*mupi*
              q_cut)/mbkin**22 - (36864*mcMS**22*muG*mupi*q_cut)/mbkin**24 + 
            (24840*mcMS**24*muG*mupi*q_cut)/mbkin**26 - (5040*mcMS**2*muG**2*q_cut**2)/
             mbkin**6 + (40320*mcMS**4*muG**2*q_cut**2)/mbkin**8 + 
            (177840*mcMS**6*muG**2*q_cut**2)/mbkin**10 - (877248*mcMS**8*muG**2*q_cut**2)/
             mbkin**12 - (3926304*mcMS**10*muG**2*q_cut**2)/mbkin**14 + 
            (735264*mcMS**12*muG**2*q_cut**2)/mbkin**16 + (4323744*mcMS**14*muG**2*
              q_cut**2)/mbkin**18 - (2464704*mcMS**16*muG**2*q_cut**2)/mbkin**20 - 
            (2724912*mcMS**18*muG**2*q_cut**2)/mbkin**22 + 
            (240480*mcMS**20*muG**2*q_cut**2)/mbkin**24 + (126000*mcMS**22*muG**2*
              q_cut**2)/mbkin**26 + (5040*mcMS**2*muG*mupi*q_cut**2)/mbkin**6 - 
            (30240*mcMS**4*muG*mupi*q_cut**2)/mbkin**8 - (169200*mcMS**6*muG*mupi*
              q_cut**2)/mbkin**10 + (720000*mcMS**8*muG*mupi*q_cut**2)/mbkin**12 + 
            (2313504*mcMS**10*muG*mupi*q_cut**2)/mbkin**14 - 
            (372384*mcMS**12*muG*mupi*q_cut**2)/mbkin**16 - 
            (815904*mcMS**14*muG*mupi*q_cut**2)/mbkin**18 + 
            (1880064*mcMS**16*muG*mupi*q_cut**2)/mbkin**20 + 
            (886320*mcMS**18*muG*mupi*q_cut**2)/mbkin**22 - 
            (37440*mcMS**20*muG*mupi*q_cut**2)/mbkin**24 - 
            (25200*mcMS**22*muG*mupi*q_cut**2)/mbkin**26 - (5040*mcMS**2*muG**2*q_cut**3)/
             mbkin**8 - (11880*mcMS**4*muG**2*q_cut**3)/mbkin**10 + 
            (407808*mcMS**6*muG**2*q_cut**3)/mbkin**12 - (2325168*mcMS**8*muG**2*q_cut**3)/
             mbkin**14 - (12438432*mcMS**10*muG**2*q_cut**3)/mbkin**16 - 
            (17994816*mcMS**12*muG**2*q_cut**3)/mbkin**18 - 
            (13595904*mcMS**14*muG**2*q_cut**3)/mbkin**20 - 
            (2472912*mcMS**16*muG**2*q_cut**3)/mbkin**22 + 
            (702288*mcMS**18*muG**2*q_cut**3)/mbkin**24 + (124200*mcMS**20*muG**2*
              q_cut**3)/mbkin**26 + (5040*mcMS**2*muG*mupi*q_cut**3)/mbkin**8 - 
            (39960*mcMS**4*muG*mupi*q_cut**3)/mbkin**10 - 
            (266976*mcMS**6*muG*mupi*q_cut**3)/mbkin**12 + 
            (1733904*mcMS**8*muG*mupi*q_cut**3)/mbkin**14 + 
            (6554304*mcMS**10*muG*mupi*q_cut**3)/mbkin**16 + 
            (8608032*mcMS**12*muG*mupi*q_cut**3)/mbkin**18 + 
            (6371424*mcMS**14*muG*mupi*q_cut**3)/mbkin**20 + 
            (1519344*mcMS**16*muG*mupi*q_cut**3)/mbkin**22 - 
            (74736*mcMS**18*muG*mupi*q_cut**3)/mbkin**24 - 
            (24840*mcMS**20*muG*mupi*q_cut**3)/mbkin**26 + 
            (32760*mcMS**2*muG**2*q_cut**4)/mbkin**10 - (81000*mcMS**4*muG**2*q_cut**4)/
             mbkin**12 - (1250064*mcMS**6*muG**2*q_cut**4)/mbkin**14 + 
            (1785384*mcMS**8*muG**2*q_cut**4)/mbkin**16 + (13305744*mcMS**10*muG**2*
              q_cut**4)/mbkin**18 + (14415336*mcMS**12*muG**2*q_cut**4)/mbkin**20 + 
            (2808000*mcMS**14*muG**2*q_cut**4)/mbkin**22 - 
            (1709064*mcMS**16*muG**2*q_cut**4)/mbkin**24 - 
            (307800*mcMS**18*muG**2*q_cut**4)/mbkin**26 - (12600*mcMS**2*muG*mupi*
              q_cut**4)/mbkin**10 + (66600*mcMS**4*muG*mupi*q_cut**4)/mbkin**12 + 
            (499248*mcMS**6*muG*mupi*q_cut**4)/mbkin**14 - 
            (1575720*mcMS**8*muG*mupi*q_cut**4)/mbkin**16 - 
            (6865776*mcMS**10*muG*mupi*q_cut**4)/mbkin**18 - 
            (6544296*mcMS**12*muG*mupi*q_cut**4)/mbkin**20 - 
            (1386720*mcMS**14*muG*mupi*q_cut**4)/mbkin**22 + 
            (305928*mcMS**16*muG*mupi*q_cut**4)/mbkin**24 + 
            (61560*mcMS**18*muG*mupi*q_cut**4)/mbkin**26 - 
            (15120*mcMS**2*muG**2*q_cut**5)/mbkin**12 + (68040*mcMS**4*muG**2*q_cut**5)/
             mbkin**14 + (291168*mcMS**6*muG**2*q_cut**5)/mbkin**16 - 
            (2070936*mcMS**8*muG**2*q_cut**5)/mbkin**18 - (5521680*mcMS**10*muG**2*
              q_cut**5)/mbkin**20 - (2596392*mcMS**12*muG**2*q_cut**5)/mbkin**22 + 
            (455616*mcMS**14*muG**2*q_cut**5)/mbkin**24 + (113400*mcMS**16*muG**2*
              q_cut**5)/mbkin**26 + (5040*mcMS**2*muG*mupi*q_cut**5)/mbkin**12 - 
            (27720*mcMS**4*muG*mupi*q_cut**5)/mbkin**14 - 
            (120672*mcMS**6*muG*mupi*q_cut**5)/mbkin**16 + 
            (514296*mcMS**8*muG*mupi*q_cut**5)/mbkin**18 + 
            (1383984*mcMS**10*muG*mupi*q_cut**5)/mbkin**20 + 
            (487656*mcMS**12*muG*mupi*q_cut**5)/mbkin**22 - 
            (104832*mcMS**14*muG*mupi*q_cut**5)/mbkin**24 - 
            (22680*mcMS**16*muG*mupi*q_cut**5)/mbkin**26 - 
            (25200*mcMS**2*muG**2*q_cut**6)/mbkin**14 + (68040*mcMS**4*muG**2*q_cut**6)/
             mbkin**16 + (1398456*mcMS**6*muG**2*q_cut**6)/mbkin**18 + 
            (3709584*mcMS**8*muG**2*q_cut**6)/mbkin**20 + (3680928*mcMS**10*muG**2*
              q_cut**6)/mbkin**22 + (1273608*mcMS**12*muG**2*q_cut**6)/mbkin**24 + 
            (113400*mcMS**14*muG**2*q_cut**6)/mbkin**26 + (5040*mcMS**2*muG*mupi*
              q_cut**6)/mbkin**14 - (7560*mcMS**4*muG*mupi*q_cut**6)/mbkin**16 - 
            (302616*mcMS**6*muG*mupi*q_cut**6)/mbkin**18 - 
            (713232*mcMS**8*muG*mupi*q_cut**6)/mbkin**20 - 
            (725472*mcMS**10*muG*mupi*q_cut**6)/mbkin**22 - 
            (261576*mcMS**12*muG*mupi*q_cut**6)/mbkin**24 - 
            (22680*mcMS**14*muG*mupi*q_cut**6)/mbkin**26 + 
            (25200*mcMS**2*muG**2*q_cut**7)/mbkin**16 - (94680*mcMS**4*muG**2*q_cut**7)/
             mbkin**18 - (1141920*mcMS**6*muG**2*q_cut**7)/mbkin**20 - 
            (2010528*mcMS**8*muG**2*q_cut**7)/mbkin**22 - (1093680*mcMS**10*muG**2*
              q_cut**7)/mbkin**24 - (81000*mcMS**12*muG**2*q_cut**7)/mbkin**26 - 
            (5040*mcMS**2*muG*mupi*q_cut**7)/mbkin**16 + (8280*mcMS**4*muG*mupi*
              q_cut**7)/mbkin**18 + (264960*mcMS**6*muG*mupi*q_cut**7)/mbkin**20 + 
            (486720*mcMS**8*muG*mupi*q_cut**7)/mbkin**22 + 
            (234000*mcMS**10*muG*mupi*q_cut**7)/mbkin**24 + 
            (16200*mcMS**12*muG*mupi*q_cut**7)/mbkin**26 - (6300*mcMS**2*muG**2*q_cut**8)/
             mbkin**18 + (27000*mcMS**4*muG**2*q_cut**8)/mbkin**20 + 
            (288612*mcMS**6*muG**2*q_cut**8)/mbkin**22 + (305712*mcMS**8*muG**2*q_cut**8)/
             mbkin**24 + (10800*mcMS**10*muG**2*q_cut**8)/mbkin**26 + 
            (1260*mcMS**2*muG*mupi*q_cut**8)/mbkin**18 - (1800*mcMS**4*muG*mupi*
              q_cut**8)/mbkin**20 - (67284*mcMS**6*muG*mupi*q_cut**8)/mbkin**22 - 
            (59904*mcMS**8*muG*mupi*q_cut**8)/mbkin**24 - 
            (2160*mcMS**10*muG*mupi*q_cut**8)/mbkin**26 - (25200*mcMS**4*muG**2*q_cut**9)/
             mbkin**22 - (71136*mcMS**6*muG**2*q_cut**9)/mbkin**24 - 
            (25200*mcMS**8*muG**2*q_cut**9)/mbkin**26 + (5040*mcMS**4*muG*mupi*q_cut**9)/
             mbkin**22 + (11232*mcMS**6*muG*mupi*q_cut**9)/mbkin**24 + 
            (5040*mcMS**8*muG*mupi*q_cut**9)/mbkin**26 + (27000*mcMS**4*muG**2*q_cut**10)/
             mbkin**24 + (27000*mcMS**6*muG**2*q_cut**10)/mbkin**26 - 
            (5400*mcMS**4*muG*mupi*q_cut**10)/mbkin**24 - 
            (5400*mcMS**6*muG*mupi*q_cut**10)/mbkin**26 - (7200*mcMS**4*muG**2*q_cut**11)/
             mbkin**26 + (1440*mcMS**4*muG*mupi*q_cut**11)/mbkin**26 - 
            72*mcMS**2*muG*((-1 + mcMS**2/mbkin**2)**2*(-35 + (370*mcMS**2)/
                 mbkin**2 + (1918*mcMS**4)/mbkin**4 - (20308*mcMS**6)/mbkin**6 + 
                (13266*mcMS**8)/mbkin**8 + (181504*mcMS**10)/mbkin**10 + 
                (193544*mcMS**12)/mbkin**12 + (10116*mcMS**14)/mbkin**14 - 
                (17183*mcMS**16)/mbkin**16 - (482*mcMS**18)/mbkin**18 + 
                (170*mcMS**20)/mbkin**20) - (2*(-70 + (655*mcMS**2)/mbkin**2 + 
                 (2748*mcMS**4)/mbkin**4 - (28439*mcMS**6)/mbkin**6 - 
                 (7792*mcMS**8)/mbkin**8 + (213186*mcMS**10)/mbkin**10 + 
                 (358324*mcMS**12)/mbkin**12 + (223406*mcMS**14)/mbkin**14 - 
                 (11322*mcMS**16)/mbkin**16 - (24769*mcMS**18)/mbkin**18 - 
                 (512*mcMS**20)/mbkin**20 + (345*mcMS**22)/mbkin**22)*q_cut)/mbkin**
                2 + (4*(-35 + (210*mcMS**2)/mbkin**2 + (1175*mcMS**4)/mbkin**4 - 
                 (5000*mcMS**6)/mbkin**6 - (16066*mcMS**8)/mbkin**8 + 
                 (2586*mcMS**10)/mbkin**10 + (5666*mcMS**12)/mbkin**12 - 
                 (13056*mcMS**14)/mbkin**14 - (6155*mcMS**16)/mbkin**16 + 
                 (260*mcMS**18)/mbkin**18 + (175*mcMS**20)/mbkin**20)*q_cut**2)/mbkin**
                4 + (2*(-70 + (555*mcMS**2)/mbkin**2 + (3708*mcMS**4)/mbkin**4 - 
                 (24082*mcMS**6)/mbkin**6 - (91032*mcMS**8)/mbkin**8 - 
                 (119556*mcMS**10)/mbkin**10 - (88492*mcMS**12)/mbkin**12 - 
                 (21102*mcMS**14)/mbkin**14 + (1038*mcMS**16)/mbkin**16 + 
                 (345*mcMS**18)/mbkin**18)*q_cut**3)/mbkin**6 - 
              (2*(-175 + (925*mcMS**2)/mbkin**2 + (6934*mcMS**4)/mbkin**4 - 
                 (21885*mcMS**6)/mbkin**6 - (95358*mcMS**8)/mbkin**8 - 
                 (90893*mcMS**10)/mbkin**10 - (19260*mcMS**12)/mbkin**12 + 
                 (4249*mcMS**14)/mbkin**14 + (855*mcMS**16)/mbkin**16)*q_cut**4)/
               mbkin**8 + (2*(-70 + (385*mcMS**2)/mbkin**2 + (1676*mcMS**4)/
                  mbkin**4 - (7143*mcMS**6)/mbkin**6 - (19222*mcMS**8)/mbkin**8 - 
                 (6773*mcMS**10)/mbkin**10 + (1456*mcMS**12)/mbkin**12 + 
                 (315*mcMS**14)/mbkin**14)*q_cut**5)/mbkin**10 + 
              (2*(-70 + (105*mcMS**2)/mbkin**2 + (4203*mcMS**4)/mbkin**4 + 
                 (9906*mcMS**6)/mbkin**6 + (10076*mcMS**8)/mbkin**8 + 
                 (3633*mcMS**10)/mbkin**10 + (315*mcMS**12)/mbkin**12)*q_cut**6)/
               mbkin**12 - (10*(-14 + (23*mcMS**2)/mbkin**2 + (736*mcMS**4)/
                  mbkin**4 + (1352*mcMS**6)/mbkin**6 + (650*mcMS**8)/mbkin**8 + 
                 (45*mcMS**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
              ((-35 + (50*mcMS**2)/mbkin**2 + (1869*mcMS**4)/mbkin**4 + 
                 (1664*mcMS**6)/mbkin**6 + (60*mcMS**8)/mbkin**8)*q_cut**8)/mbkin**
                16 - (4*mcMS**2*(35 + (78*mcMS**2)/mbkin**2 + (35*mcMS**4)/
                  mbkin**4)*q_cut**9)/mbkin**20 + (150*mcMS**2*(mbkin**2 + mcMS**2)*
                q_cut**10)/mbkin**24 - (40*mcMS**2*q_cut**11)/mbkin**24) - 
            4*(-((-1 + mcMS**2/mbkin**2)**2*(-575 + (1630*mcMS**2)/mbkin**2 + 
                 (85106*mcMS**4)/mbkin**4 - (282442*mcMS**6)/mbkin**6 - 
                 (1378372*mcMS**8)/mbkin**8 + (1535906*mcMS**10)/mbkin**10 + 
                 (4125290*mcMS**12)/mbkin**12 + (1796594*mcMS**14)/mbkin**14 - 
                 (249373*mcMS**16)/mbkin**16 - (188768*mcMS**18)/mbkin**18 - 
                 (3836*mcMS**20)/mbkin**20 + (2040*mcMS**22)/mbkin**22)) + 
              (4*(-585 + (285*mcMS**2)/mbkin**2 + (64813*mcMS**4)/mbkin**4 - 
                 (59927*mcMS**6)/mbkin**6 - (921308*mcMS**8)/mbkin**8 + 
                 (129898*mcMS**10)/mbkin**10 + (2676538*mcMS**12)/mbkin**12 + 
                 (3044602*mcMS**14)/mbkin**14 + (904567*mcMS**16)/mbkin**16 - 
                 (261887*mcMS**18)/mbkin**18 - (134543*mcMS**20)/mbkin**20 - 
                 (1323*mcMS**22)/mbkin**22 + (2070*mcMS**24)/mbkin**24)*q_cut)/mbkin**
                2 - (4*(-595 - (2310*mcMS**2)/mbkin**2 + (33975*mcMS**4)/
                  mbkin**4 + (77820*mcMS**6)/mbkin**6 - (95082*mcMS**8)/mbkin**8 - 
                 (7416*mcMS**10)/mbkin**10 + (117360*mcMS**12)/mbkin**12 - 
                 (247224*mcMS**14)/mbkin**14 - (273243*mcMS**16)/mbkin**16 - 
                 (64490*mcMS**18)/mbkin**18 + (5505*mcMS**20)/mbkin**20 + 
                 (2100*mcMS**22)/mbkin**22)*q_cut**2)/mbkin**4 - 
              (4*(-585 - (2055*mcMS**2)/mbkin**2 + (59033*mcMS**4)/mbkin**4 + 
                 (170055*mcMS**6)/mbkin**6 - (69365*mcMS**8)/mbkin**8 - 
                 (373829*mcMS**10)/mbkin**10 - (644871*mcMS**12)/mbkin**12 - 
                 (469615*mcMS**14)/mbkin**14 - (106890*mcMS**16)/mbkin**16 + 
                 (4692*mcMS**18)/mbkin**18 + (2070*mcMS**20)/mbkin**20)*q_cut**3)/
               mbkin**6 + (2*(-2895 - (21000*mcMS**2)/mbkin**2 + (119453*mcMS**4)/
                  mbkin**4 + (428252*mcMS**6)/mbkin**6 - (312001*mcMS**8)/
                  mbkin**8 - (1414660*mcMS**10)/mbkin**10 - (1024579*mcMS**12)/
                  mbkin**12 - (162412*mcMS**14)/mbkin**14 + (50862*mcMS**16)/
                  mbkin**16 + (10260*mcMS**18)/mbkin**18)*q_cut**4)/mbkin**8 - 
              (4*(-525 - (4515*mcMS**2)/mbkin**2 + (17451*mcMS**4)/mbkin**4 + 
                 (35671*mcMS**6)/mbkin**6 - (71799*mcMS**8)/mbkin**8 - 
                 (98271*mcMS**10)/mbkin**10 - (16201*mcMS**12)/mbkin**12 + 
                 (10059*mcMS**14)/mbkin**14 + (1890*mcMS**16)/mbkin**16)*q_cut**5)/
               mbkin**10 - (4*(-525 - (7560*mcMS**2)/mbkin**2 - (15337*mcMS**4)/
                  mbkin**4 + (10568*mcMS**6)/mbkin**6 + (44285*mcMS**8)/mbkin**8 + 
                 (50502*mcMS**10)/mbkin**10 + (20097*mcMS**12)/mbkin**12 + 
                 (1890*mcMS**14)/mbkin**14)*q_cut**6)/mbkin**12 + 
              (20*(-69 - (1443*mcMS**2)/mbkin**2 - (1933*mcMS**4)/mbkin**4 + 
                 (4085*mcMS**6)/mbkin**6 + (7964*mcMS**8)/mbkin**8 + 
                 (3702*mcMS**10)/mbkin**10 + (270*mcMS**12)/mbkin**12)*q_cut**7)/
               mbkin**14 + ((135 + (7260*mcMS**2)/mbkin**2 + (2877*mcMS**4)/
                  mbkin**4 - (24560*mcMS**6)/mbkin**6 - (18312*mcMS**8)/mbkin**8 - 
                 (720*mcMS**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
              (16*(-35 - (70*mcMS**2)/mbkin**2 + (29*mcMS**4)/mbkin**4 + 
                 (181*mcMS**6)/mbkin**6 + (105*mcMS**8)/mbkin**8)*q_cut**9)/mbkin**
                18 - (120*(-5 - (5*mcMS**2)/mbkin**2 + (13*mcMS**4)/mbkin**4 + 
                 (15*mcMS**6)/mbkin**6)*q_cut**10)/mbkin**20 - (160*(mbkin**4 - 
                 3*mcMS**4)*q_cut**11)/mbkin**26)*rG + 24*mbkin*
             (-((-1 + mcMS**2/mbkin**2)**2*(65 - (810*mcMS**2)/mbkin**2 - 
                 (516*mcMS**4)/mbkin**4 + (110224*mcMS**6)/mbkin**6 - 
                 (410120*mcMS**8)/mbkin**8 - (2939462*mcMS**10)/mbkin**10 - 
                 (3104774*mcMS**12)/mbkin**12 - (475058*mcMS**14)/mbkin**14 + 
                 (163447*mcMS**16)/mbkin**16 - (52384*mcMS**18)/mbkin**18 - 
                 (4742*mcMS**20)/mbkin**20 + (850*mcMS**22)/mbkin**22)) + 
              (2*(135 - (1420*mcMS**2)/mbkin**2 - (3196*mcMS**4)/mbkin**4 + 
                 (162656*mcMS**6)/mbkin**6 - (138305*mcMS**8)/mbkin**8 - 
                 (3423984*mcMS**10)/mbkin**10 - (6202792*mcMS**12)/mbkin**12 - 
                 (3867608*mcMS**14)/mbkin**14 - (95475*mcMS**16)/mbkin**16 + 
                 (226796*mcMS**18)/mbkin**18 - (77740*mcMS**20)/mbkin**20 - 
                 (7352*mcMS**22)/mbkin**22 + (1725*mcMS**24)/mbkin**24)*q_cut)/mbkin**
                2 - (4*(70 - (455*mcMS**2)/mbkin**2 - (3470*mcMS**4)/mbkin**4 + 
                 (37665*mcMS**6)/mbkin**6 + (138922*mcMS**8)/mbkin**8 + 
                 (83452*mcMS**10)/mbkin**10 + (121324*mcMS**12)/mbkin**12 + 
                 (190868*mcMS**14)/mbkin**14 + (16424*mcMS**16)/mbkin**16 - 
                 (25445*mcMS**18)/mbkin**18 - (790*mcMS**20)/mbkin**20 + 
                 (875*mcMS**22)/mbkin**22)*q_cut**2)/mbkin**4 - 
              (2*(135 - (880*mcMS**2)/mbkin**2 - (7621*mcMS**4)/mbkin**4 + 
                 (133182*mcMS**6)/mbkin**6 + (543892*mcMS**8)/mbkin**8 + 
                 (658370*mcMS**10)/mbkin**10 + (525528*mcMS**12)/mbkin**12 + 
                 (55122*mcMS**14)/mbkin**14 - (76843*mcMS**16)/mbkin**16 - 
                 (2082*mcMS**18)/mbkin**18 + (1725*mcMS**20)/mbkin**20)*q_cut**3)/
               mbkin**6 + (2*(330 - (1015*mcMS**2)/mbkin**2 - (21973*mcMS**4)/
                  mbkin**4 + (116514*mcMS**6)/mbkin**6 + (621711*mcMS**8)/
                  mbkin**8 + (591448*mcMS**10)/mbkin**10 - (14853*mcMS**12)/
                  mbkin**12 - (96558*mcMS**14)/mbkin**14 + (6769*mcMS**16)/
                  mbkin**16 + (4275*mcMS**18)/mbkin**18)*q_cut**4)/mbkin**8 - 
              (2*(105 - (280*mcMS**2)/mbkin**2 - (8822*mcMS**4)/mbkin**4 + 
                 (25052*mcMS**6)/mbkin**6 + (79356*mcMS**8)/mbkin**8 - 
                 (7796*mcMS**10)/mbkin**10 - (25990*mcMS**12)/mbkin**12 + 
                 (3136*mcMS**14)/mbkin**14 + (1575*mcMS**16)/mbkin**16)*q_cut**5)/
               mbkin**10 - (2*(105 + (455*mcMS**2)/mbkin**2 - (8521*mcMS**4)/
                  mbkin**4 - (25699*mcMS**6)/mbkin**6 - (20961*mcMS**8)/mbkin**8 + 
                 (4285*mcMS**10)/mbkin**10 + (11193*mcMS**12)/mbkin**12 + 
                 (1575*mcMS**14)/mbkin**14)*q_cut**6)/mbkin**12 + 
              (10*(3 + (88*mcMS**2)/mbkin**2 - (1667*mcMS**4)/mbkin**4 - 
                 (3298*mcMS**6)/mbkin**6 + (671*mcMS**8)/mbkin**8 + 
                 (2090*mcMS**10)/mbkin**10 + (225*mcMS**12)/mbkin**12)*q_cut**7)/
               mbkin**14 + ((45 - (310*mcMS**2)/mbkin**2 + (4303*mcMS**4)/
                  mbkin**4 + (694*mcMS**6)/mbkin**6 - (5204*mcMS**8)/mbkin**8 - 
                 (300*mcMS**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
              (4*(35 + (70*mcMS**2)/mbkin**2 + (94*mcMS**4)/mbkin**4 + 
                 (218*mcMS**6)/mbkin**6 + (175*mcMS**8)/mbkin**8)*q_cut**9)/mbkin**
                18 - (30*(5 + (5*mcMS**2)/mbkin**2 + (17*mcMS**4)/mbkin**4 + 
                 (25*mcMS**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
              (40*(mbkin**4 + 5*mcMS**4)*q_cut**11)/mbkin**26)*rhoD + 1260*sB + 
            (9240*mcMS**2*sB)/mbkin**2 - (444348*mcMS**4*sB)/mbkin**4 + 
            (379944*mcMS**6*sB)/mbkin**6 + (15149640*mcMS**8*sB)/mbkin**8 + 
            (2327616*mcMS**10*sB)/mbkin**10 - (26470152*mcMS**12*sB)/mbkin**12 - 
            (6409872*mcMS**14*sB)/mbkin**14 + (7311276*mcMS**16*sB)/mbkin**16 + 
            (7117560*mcMS**18*sB)/mbkin**18 + (2594244*mcMS**20*sB)/mbkin**20 - 
            (1478808*mcMS**22*sB)/mbkin**22 - (108000*mcMS**24*sB)/mbkin**24 + 
            (20400*mcMS**26*sB)/mbkin**26 - (5040*q_cut*sB)/mbkin**2 - 
            (62160*mcMS**2*q_cut*sB)/mbkin**4 + (1228416*mcMS**4*q_cut*sB)/mbkin**6 + 
            (4170192*mcMS**6*q_cut*sB)/mbkin**8 - (27720144*mcMS**8*q_cut*sB)/
             mbkin**10 - (98929248*mcMS**10*q_cut*sB)/mbkin**12 - 
            (111450432*mcMS**12*q_cut*sB)/mbkin**14 - (55878048*mcMS**14*q_cut*sB)/
             mbkin**16 - (16085904*mcMS**16*q_cut*sB)/mbkin**18 + 
            (3898032*mcMS**18*q_cut*sB)/mbkin**20 + (4635456*mcMS**20*q_cut*sB)/
             mbkin**22 + (171600*mcMS**22*q_cut*sB)/mbkin**24 - 
            (82800*mcMS**24*q_cut*sB)/mbkin**26 + (5040*q_cut**2*sB)/mbkin**4 + 
            (87360*mcMS**2*q_cut**2*sB)/mbkin**6 - (505200*mcMS**4*q_cut**2*sB)/
             mbkin**8 - (4070880*mcMS**6*q_cut**2*sB)/mbkin**10 - 
            (4525536*mcMS**8*q_cut**2*sB)/mbkin**12 + (142656*mcMS**10*q_cut**2*sB)/
             mbkin**14 - (2771904*mcMS**12*q_cut**2*sB)/mbkin**16 - 
            (4946496*mcMS**14*q_cut**2*sB)/mbkin**18 - (5925840*mcMS**16*q_cut**2*sB)/
             mbkin**20 - (2336640*mcMS**18*q_cut**2*sB)/mbkin**22 + 
            (87600*mcMS**20*q_cut**2*sB)/mbkin**24 + (84000*mcMS**22*q_cut**2*sB)/
             mbkin**26 + (5040*q_cut**3*sB)/mbkin**6 + (82320*mcMS**2*q_cut**3*sB)/
             mbkin**8 - (854496*mcMS**4*q_cut**3*sB)/mbkin**10 - 
            (7703856*mcMS**6*q_cut**3*sB)/mbkin**12 - (16016976*mcMS**8*q_cut**3*sB)/
             mbkin**14 - (21826608*mcMS**10*q_cut**3*sB)/mbkin**16 - 
            (21509136*mcMS**12*q_cut**3*sB)/mbkin**18 - (13371216*mcMS**14*q_cut**3*sB)/
             mbkin**20 - (3961056*mcMS**16*q_cut**3*sB)/mbkin**22 + 
            (110880*mcMS**18*q_cut**3*sB)/mbkin**24 + (82800*mcMS**20*q_cut**3*sB)/
             mbkin**26 - (12600*q_cut**4*sB)/mbkin**8 - (253680*mcMS**2*q_cut**4*sB)/
             mbkin**10 + (404088*mcMS**4*q_cut**4*sB)/mbkin**12 + 
            (8097600*mcMS**6*q_cut**4*sB)/mbkin**14 + (19759944*mcMS**8*q_cut**4*sB)/
             mbkin**16 + (21465504*mcMS**10*q_cut**4*sB)/mbkin**18 + 
            (14271960*mcMS**12*q_cut**4*sB)/mbkin**20 + (3432288*mcMS**14*q_cut**4*sB)/
             mbkin**22 - (769440*mcMS**16*q_cut**4*sB)/mbkin**24 - 
            (205200*mcMS**18*q_cut**4*sB)/mbkin**26 + (5040*q_cut**5*sB)/mbkin**10 + 
            (102480*mcMS**2*q_cut**5*sB)/mbkin**12 - (16512*mcMS**4*q_cut**5*sB)/
             mbkin**14 - (1479984*mcMS**6*q_cut**5*sB)/mbkin**16 - 
            (2775456*mcMS**8*q_cut**5*sB)/mbkin**18 - (2796144*mcMS**10*q_cut**5*sB)/
             mbkin**20 - (875232*mcMS**12*q_cut**5*sB)/mbkin**22 + 
            (290640*mcMS**14*q_cut**5*sB)/mbkin**24 + (75600*mcMS**16*q_cut**5*sB)/
             mbkin**26 + (5040*q_cut**6*sB)/mbkin**12 + (127680*mcMS**2*q_cut**6*sB)/
             mbkin**14 + (597024*mcMS**4*q_cut**6*sB)/mbkin**16 + 
            (1262448*mcMS**6*q_cut**6*sB)/mbkin**18 + (1846128*mcMS**8*q_cut**6*sB)/
             mbkin**20 + (1700544*mcMS**10*q_cut**6*sB)/mbkin**22 + 
            (725760*mcMS**12*q_cut**6*sB)/mbkin**24 + (75600*mcMS**14*q_cut**6*sB)/
             mbkin**26 - (5040*q_cut**7*sB)/mbkin**14 - (122640*mcMS**2*q_cut**7*sB)/
             mbkin**16 - (494400*mcMS**4*q_cut**7*sB)/mbkin**18 - 
            (1044720*mcMS**6*q_cut**7*sB)/mbkin**20 - (1306080*mcMS**8*q_cut**7*sB)/
             mbkin**22 - (661440*mcMS**10*q_cut**7*sB)/mbkin**24 - 
            (54000*mcMS**12*q_cut**7*sB)/mbkin**26 + (1260*q_cut**8*sB)/mbkin**16 + 
            (29400*mcMS**2*q_cut**8*sB)/mbkin**18 + (84036*mcMS**4*q_cut**8*sB)/
             mbkin**20 + (198456*mcMS**6*q_cut**8*sB)/mbkin**22 + 
            (166080*mcMS**8*q_cut**8*sB)/mbkin**24 + (7200*mcMS**10*q_cut**8*sB)/
             mbkin**26 - (8928*mcMS**4*q_cut**9*sB)/mbkin**22 - 
            (28800*mcMS**6*q_cut**9*sB)/mbkin**24 - (16800*mcMS**8*q_cut**9*sB)/
             mbkin**26 + (15120*mcMS**4*q_cut**10*sB)/mbkin**24 + 
            (18000*mcMS**6*q_cut**10*sB)/mbkin**26 - (4800*mcMS**4*q_cut**11*sB)/
             mbkin**26 - 1360*sE - (2000*mcMS**2*sE)/mbkin**2 + 
            (309680*mcMS**4*sE)/mbkin**4 - (670736*mcMS**6*sE)/mbkin**6 - 
            (7412592*mcMS**8*sE)/mbkin**8 + (6194640*mcMS**10*sE)/mbkin**10 + 
            (8632992*mcMS**12*sE)/mbkin**12 - (5378784*mcMS**14*sE)/mbkin**14 - 
            (700464*mcMS**16*sE)/mbkin**16 - (312368*mcMS**18*sE)/mbkin**18 - 
            (1548880*mcMS**20*sE)/mbkin**20 + (836848*mcMS**22*sE)/mbkin**22 + 
            (65264*mcMS**24*sE)/mbkin**24 - (12240*mcMS**26*sE)/mbkin**26 + 
            (5520*q_cut*sE)/mbkin**2 + (32160*mcMS**2*q_cut*sE)/mbkin**4 - 
            (906080*mcMS**4*q_cut*sE)/mbkin**6 - (1643168*mcMS**6*q_cut*sE)/mbkin**8 + 
            (17160304*mcMS**8*q_cut*sE)/mbkin**10 + (36106432*mcMS**10*q_cut*sE)/
             mbkin**12 + (22402624*mcMS**12*q_cut*sE)/mbkin**14 + 
            (4453696*mcMS**14*q_cut*sE)/mbkin**16 + (2931184*mcMS**16*q_cut*sE)/
             mbkin**18 - (2327648*mcMS**18*q_cut*sE)/mbkin**20 - 
            (2691296*mcMS**20*q_cut*sE)/mbkin**22 - (94368*mcMS**22*q_cut*sE)/
             mbkin**24 + (49680*mcMS**24*q_cut*sE)/mbkin**26 - (5600*q_cut**2*sE)/
             mbkin**4 - (57120*mcMS**2*q_cut**2*sE)/mbkin**6 + 
            (427680*mcMS**4*q_cut**2*sE)/mbkin**8 + (2262240*mcMS**6*q_cut**2*sE)/
             mbkin**10 + (581760*mcMS**8*q_cut**2*sE)/mbkin**12 - 
            (2877312*mcMS**10*q_cut**2*sE)/mbkin**14 + (1913472*mcMS**12*q_cut**2*sE)/
             mbkin**16 + (6809472*mcMS**14*q_cut**2*sE)/mbkin**18 + 
            (5638368*mcMS**16*q_cut**2*sE)/mbkin**20 + (1425440*mcMS**18*q_cut**2*sE)/
             mbkin**22 - (101280*mcMS**20*q_cut**2*sE)/mbkin**24 - 
            (50400*mcMS**22*q_cut**2*sE)/mbkin**26 - (5520*q_cut**3*sE)/mbkin**6 - 
            (54240*mcMS**2*q_cut**3*sE)/mbkin**8 + (691120*mcMS**4*q_cut**3*sE)/
             mbkin**10 + (4345728*mcMS**6*q_cut**3*sE)/mbkin**12 + 
            (5522528*mcMS**8*q_cut**3*sE)/mbkin**14 + (4002368*mcMS**10*q_cut**3*sE)/
             mbkin**16 + (2024544*mcMS**12*q_cut**3*sE)/mbkin**18 + 
            (3545728*mcMS**14*q_cut**3*sE)/mbkin**20 + (2277168*mcMS**16*q_cut**3*sE)/
             mbkin**22 + (21408*mcMS**18*q_cut**3*sE)/mbkin**24 - 
            (49680*mcMS**20*q_cut**3*sE)/mbkin**26 + (13680*q_cut**4*sE)/mbkin**8 + 
            (186480*mcMS**2*q_cut**4*sE)/mbkin**10 - (511520*mcMS**4*q_cut**4*sE)/
             mbkin**12 - (4919264*mcMS**6*q_cut**4*sE)/mbkin**14 - 
            (7467680*mcMS**8*q_cut**4*sE)/mbkin**16 - (6060512*mcMS**10*q_cut**4*sE)/
             mbkin**18 - (5694752*mcMS**12*q_cut**4*sE)/mbkin**20 - 
            (1869920*mcMS**14*q_cut**4*sE)/mbkin**22 + (413616*mcMS**16*q_cut**4*sE)/
             mbkin**24 + (123120*mcMS**18*q_cut**4*sE)/mbkin**26 - 
            (5040*q_cut**5*sE)/mbkin**10 - (77280*mcMS**2*q_cut**5*sE)/mbkin**12 + 
            (108480*mcMS**4*q_cut**5*sE)/mbkin**14 + (874016*mcMS**6*q_cut**5*sE)/
             mbkin**16 + (894432*mcMS**8*q_cut**5*sE)/mbkin**18 + 
            (642528*mcMS**10*q_cut**5*sE)/mbkin**20 + (23872*mcMS**12*q_cut**5*sE)/
             mbkin**22 - (213024*mcMS**14*q_cut**5*sE)/mbkin**24 - 
            (45360*mcMS**16*q_cut**5*sE)/mbkin**26 - (5040*q_cut**6*sE)/mbkin**12 - 
            (105840*mcMS**2*q_cut**6*sE)/mbkin**14 - (365840*mcMS**4*q_cut**6*sE)/
             mbkin**16 - (419792*mcMS**6*q_cut**6*sE)/mbkin**18 - 
            (359504*mcMS**8*q_cut**6*sE)/mbkin**20 - (464784*mcMS**10*q_cut**6*sE)/
             mbkin**22 - (371952*mcMS**12*q_cut**6*sE)/mbkin**24 - 
            (45360*mcMS**14*q_cut**6*sE)/mbkin**26 + (3600*q_cut**7*sE)/mbkin**14 + 
            (102240*mcMS**2*q_cut**7*sE)/mbkin**16 + (292720*mcMS**4*q_cut**7*sE)/
             mbkin**18 + (389440*mcMS**6*q_cut**7*sE)/mbkin**20 + 
            (544240*mcMS**8*q_cut**7*sE)/mbkin**22 + (363360*mcMS**10*q_cut**7*sE)/
             mbkin**24 + (32400*mcMS**12*q_cut**7*sE)/mbkin**26 - 
            (480*q_cut**8*sE)/mbkin**16 - (25440*mcMS**2*q_cut**8*sE)/mbkin**18 - 
            (44400*mcMS**4*q_cut**8*sE)/mbkin**20 - (82928*mcMS**6*q_cut**8*sE)/
             mbkin**22 - (85728*mcMS**8*q_cut**8*sE)/mbkin**24 - 
            (4320*mcMS**10*q_cut**8*sE)/mbkin**26 + (1120*q_cut**9*sE)/mbkin**18 + 
            (2240*mcMS**2*q_cut**9*sE)/mbkin**20 + (2240*mcMS**4*q_cut**9*sE)/mbkin**22 + 
            (9664*mcMS**6*q_cut**9*sE)/mbkin**24 + (10080*mcMS**8*q_cut**9*sE)/
             mbkin**26 - (1200*q_cut**10*sE)/mbkin**20 - (1200*mcMS**2*q_cut**10*sE)/
             mbkin**22 - (6960*mcMS**4*q_cut**10*sE)/mbkin**24 - 
            (10800*mcMS**6*q_cut**10*sE)/mbkin**26 + (320*q_cut**11*sE)/mbkin**22 + 
            (2880*mcMS**4*q_cut**11*sE)/mbkin**26 - 145*sqB - (800*mcMS**2*sqB)/
             mbkin**2 + (54167*mcMS**4*sqB)/mbkin**4 - (39482*mcMS**6*sqB)/
             mbkin**6 - (2079312*mcMS**8*sqB)/mbkin**8 - (1917294*mcMS**10*sqB)/
             mbkin**10 + (4531890*mcMS**12*sqB)/mbkin**12 + 
            (3302724*mcMS**14*sqB)/mbkin**14 - (2155053*mcMS**16*sqB)/mbkin**16 - 
            (1741748*mcMS**18*sqB)/mbkin**18 - (24841*mcMS**20*sqB)/mbkin**20 + 
            (69430*mcMS**22*sqB)/mbkin**22 + (974*mcMS**24*sqB)/mbkin**24 - 
            (510*mcMS**26*sqB)/mbkin**26 + (570*q_cut*sqB)/mbkin**2 + 
            (6480*mcMS**2*q_cut*sqB)/mbkin**4 - (148544*mcMS**4*q_cut*sqB)/mbkin**6 - 
            (579896*mcMS**6*q_cut*sqB)/mbkin**8 + (3426370*mcMS**8*q_cut*sqB)/
             mbkin**10 + (16467184*mcMS**10*q_cut*sqB)/mbkin**12 + 
            (23592832*mcMS**12*q_cut*sqB)/mbkin**14 + (14304928*mcMS**14*q_cut*sqB)/
             mbkin**16 + (2600590*mcMS**16*q_cut*sqB)/mbkin**18 - 
            (688736*mcMS**18*q_cut*sqB)/mbkin**20 - (201440*mcMS**20*q_cut*sqB)/
             mbkin**22 + (4152*mcMS**22*q_cut*sqB)/mbkin**24 + 
            (2070*mcMS**24*q_cut*sqB)/mbkin**26 - (560*q_cut**2*sqB)/mbkin**4 - 
            (9660*mcMS**2*q_cut**2*sqB)/mbkin**6 + (55200*mcMS**4*q_cut**2*sqB)/
             mbkin**8 + (549540*mcMS**6*q_cut**2*sqB)/mbkin**10 + 
            (928824*mcMS**8*q_cut**2*sqB)/mbkin**12 + (375504*mcMS**10*q_cut**2*sqB)/
             mbkin**14 + (785808*mcMS**12*q_cut**2*sqB)/mbkin**16 + 
            (1426416*mcMS**14*q_cut**2*sqB)/mbkin**18 + (718968*mcMS**16*q_cut**2*sqB)/
             mbkin**20 + (84620*mcMS**18*q_cut**2*sqB)/mbkin**22 - 
            (13680*mcMS**20*q_cut**2*sqB)/mbkin**24 - (2100*mcMS**22*q_cut**2*sqB)/
             mbkin**26 - (570*q_cut**3*sqB)/mbkin**6 - (8760*mcMS**2*q_cut**3*sqB)/
             mbkin**8 + (108574*mcMS**4*q_cut**3*sqB)/mbkin**10 + 
            (1034772*mcMS**6*q_cut**3*sqB)/mbkin**12 + (2407232*mcMS**8*q_cut**3*sqB)/
             mbkin**14 + (2729180*mcMS**10*q_cut**3*sqB)/mbkin**16 + 
            (2234328*mcMS**12*q_cut**3*sqB)/mbkin**18 + (1038412*mcMS**14*q_cut**3*sqB)/
             mbkin**20 + (159642*mcMS**16*q_cut**3*sqB)/mbkin**22 - 
            (7812*mcMS**18*q_cut**3*sqB)/mbkin**24 - (2070*mcMS**20*q_cut**3*sqB)/
             mbkin**26 + (1440*q_cut**4*sqB)/mbkin**8 + (27510*mcMS**2*q_cut**4*sqB)/
             mbkin**10 - (43082*mcMS**4*q_cut**4*sqB)/mbkin**12 - 
            (1078244*mcMS**6*q_cut**4*sqB)/mbkin**14 - (3002066*mcMS**8*q_cut**4*sqB)/
             mbkin**16 - (2994488*mcMS**10*q_cut**4*sqB)/mbkin**18 - 
            (1173962*mcMS**12*q_cut**4*sqB)/mbkin**20 - (79652*mcMS**14*q_cut**4*sqB)/
             mbkin**22 + (42966*mcMS**16*q_cut**4*sqB)/mbkin**24 + 
            (5130*mcMS**18*q_cut**4*sqB)/mbkin**26 - (630*q_cut**5*sqB)/mbkin**10 - 
            (10920*mcMS**2*q_cut**5*sqB)/mbkin**12 - (5652*mcMS**4*q_cut**5*sqB)/
             mbkin**14 + (184832*mcMS**6*q_cut**5*sqB)/mbkin**16 + 
            (365376*mcMS**8*q_cut**5*sqB)/mbkin**18 + (177024*mcMS**10*q_cut**5*sqB)/
             mbkin**20 - (14900*mcMS**12*q_cut**5*sqB)/mbkin**22 - 
            (20664*mcMS**14*q_cut**5*sqB)/mbkin**24 - (1890*mcMS**16*q_cut**5*sqB)/
             mbkin**26 - (630*q_cut**6*sqB)/mbkin**12 - (13650*mcMS**2*q_cut**6*sqB)/
             mbkin**14 - (70346*mcMS**4*q_cut**6*sqB)/mbkin**16 - 
            (131414*mcMS**6*q_cut**6*sqB)/mbkin**18 - (124586*mcMS**8*q_cut**6*sqB)/
             mbkin**20 - (86790*mcMS**10*q_cut**6*sqB)/mbkin**22 - 
            (26502*mcMS**12*q_cut**6*sqB)/mbkin**24 - (1890*mcMS**14*q_cut**6*sqB)/
             mbkin**26 + (810*q_cut**7*sqB)/mbkin**14 + (12840*mcMS**2*q_cut**7*sqB)/
             mbkin**16 + (61750*mcMS**4*q_cut**7*sqB)/mbkin**18 + 
            (109420*mcMS**6*q_cut**7*sqB)/mbkin**20 + (85210*mcMS**8*q_cut**7*sqB)/
             mbkin**22 + (27180*mcMS**10*q_cut**7*sqB)/mbkin**24 + 
            (1350*mcMS**12*q_cut**7*sqB)/mbkin**26 - (255*q_cut**8*sqB)/mbkin**16 - 
            (2910*mcMS**2*q_cut**8*sqB)/mbkin**18 - (11409*mcMS**4*q_cut**8*sqB)/
             mbkin**20 - (15662*mcMS**6*q_cut**8*sqB)/mbkin**22 - 
            (6588*mcMS**8*q_cut**8*sqB)/mbkin**24 - (180*mcMS**10*q_cut**8*sqB)/
             mbkin**26 - (140*q_cut**9*sqB)/mbkin**18 - (280*mcMS**2*q_cut**9*sqB)/
             mbkin**20 - (568*mcMS**4*q_cut**9*sqB)/mbkin**22 + 
            (184*mcMS**6*q_cut**9*sqB)/mbkin**24 + (420*mcMS**8*q_cut**9*sqB)/mbkin**26 + 
            (150*q_cut**10*sqB)/mbkin**20 + (150*mcMS**2*q_cut**10*sqB)/mbkin**22 - 
            (210*mcMS**4*q_cut**10*sqB)/mbkin**24 - (450*mcMS**6*q_cut**10*sqB)/
             mbkin**26 - (40*q_cut**11*sqB)/mbkin**22 + (120*mcMS**4*q_cut**11*sqB)/
             mbkin**26))*np.log((mbkin**2 + mcMS**2 - q_cut - 
            mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                 q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
            mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                 q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
        (144*mcMS**4*(-16*(-((-1 + mcMS**2/mbkin**2)**4*(-590 - (170*mcMS**2)/
                 mbkin**2 + (47529*mcMS**4)/mbkin**4 + (10177*mcMS**6)/mbkin**6 - 
                (120338*mcMS**8)/mbkin**8 - (39978*mcMS**10)/mbkin**10 - 
                (3053*mcMS**12)/mbkin**12 + (12643*mcMS**14)/mbkin**14 + 
                (3060*mcMS**16)/mbkin**16)) + ((-1 + mcMS**2/mbkin**2)**2*(-2960 - 
                (5850*mcMS**2)/mbkin**2 + (190377*mcMS**4)/mbkin**4 + 
                (219388*mcMS**6)/mbkin**6 - (442203*mcMS**8)/mbkin**8 - 
                (597078*mcMS**10)/mbkin**10 - (182993*mcMS**12)/mbkin**12 + 
                (13020*mcMS**14)/mbkin**14 + (67059*mcMS**16)/mbkin**16 + 
                (15480*mcMS**18)/mbkin**18)*q_cut)/mbkin**2 - 
             (3*(-1780 - (4930*mcMS**2)/mbkin**2 + (89849*mcMS**4)/mbkin**4 + 
                (83367*mcMS**6)/mbkin**6 - (134199*mcMS**8)/mbkin**8 - 
                (202701*mcMS**10)/mbkin**10 - (265421*mcMS**12)/mbkin**12 - 
                (100835*mcMS**14)/mbkin**14 + (4207*mcMS**16)/mbkin**16 + 
                (39243*mcMS**18)/mbkin**18 + (9360*mcMS**20)/mbkin**20)*q_cut**2)/
              mbkin**4 + ((-2970 - (11480*mcMS**2)/mbkin**2 + (121169*mcMS**4)/
                 mbkin**4 + (162838*mcMS**6)/mbkin**6 - (396629*mcMS**8)/
                 mbkin**8 - (498572*mcMS**10)/mbkin**10 - (54133*mcMS**12)/
                 mbkin**12 + (158794*mcMS**14)/mbkin**14 + (97083*mcMS**16)/
                 mbkin**16 + (15660*mcMS**18)/mbkin**18)*q_cut**3)/mbkin**6 + 
             ((-2940 - (26390*mcMS**2)/mbkin**2 - (12875*mcMS**4)/mbkin**4 + 
                (74925*mcMS**6)/mbkin**6 + (184456*mcMS**8)/mbkin**8 + 
                (125662*mcMS**10)/mbkin**10 + (82895*mcMS**12)/mbkin**12 + 
                (75915*mcMS**14)/mbkin**14 + (15120*mcMS**16)/mbkin**16)*q_cut**4)/
              mbkin**8 - (3*(-1750 - (14980*mcMS**2)/mbkin**2 - (1663*mcMS**4)/
                 mbkin**4 + (43998*mcMS**6)/mbkin**6 + (91212*mcMS**8)/mbkin**8 + 
                (97050*mcMS**10)/mbkin**10 + (54369*mcMS**12)/mbkin**12 + 
                (8820*mcMS**14)/mbkin**14)*q_cut**5)/mbkin**10 + 
             ((-2940 - (23870*mcMS**2)/mbkin**2 + (9833*mcMS**4)/mbkin**4 + 
                (105581*mcMS**6)/mbkin**6 + (151243*mcMS**8)/mbkin**8 + 
                (84441*mcMS**10)/mbkin**10 + (15120*mcMS**12)/mbkin**12)*q_cut**6)/
              mbkin**12 - ((-810 - (4360*mcMS**2)/mbkin**2 + (9023*mcMS**4)/
                 mbkin**4 + (21544*mcMS**6)/mbkin**6 + (14031*mcMS**8)/mbkin**8 + 
                (7020*mcMS**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
             (6*(-55 - (70*mcMS**2)/mbkin**2 + (431*mcMS**4)/mbkin**4 + 
                (672*mcMS**6)/mbkin**6 + (990*mcMS**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
             (10*(-19 - (19*mcMS**2)/mbkin**2 + (230*mcMS**4)/mbkin**4 + 
                (342*mcMS**6)/mbkin**6)*q_cut**9)/mbkin**18 - 
             (40*(mbkin**4 - 18*mcMS**4)*q_cut**10)/mbkin**24)*rE + 
           ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
             q_cut**2/mbkin**4)*((-2520*mcMS**2*muG**2)/mbkin**2 + 
             (7740*mcMS**4*muG**2)/mbkin**4 - (25884*mcMS**6*muG**2)/mbkin**6 + 
             (165564*mcMS**8*muG**2)/mbkin**8 + (515340*mcMS**10*muG**2)/
              mbkin**10 - (1157940*mcMS**12*muG**2)/mbkin**12 + 
             (117180*mcMS**14*muG**2)/mbkin**14 + (465444*mcMS**16*muG**2)/
              mbkin**16 - (78804*mcMS**18*muG**2)/mbkin**18 - 
             (6120*mcMS**20*muG**2)/mbkin**20 - (2520*mcMS**2*muG*mupi)/mbkin**2 + 
             (17460*mcMS**4*muG*mupi)/mbkin**4 + (51084*mcMS**6*muG*mupi)/
              mbkin**6 - (442764*mcMS**8*muG*mupi)/mbkin**8 - 
             (36540*mcMS**10*muG*mupi)/mbkin**10 + (910980*mcMS**12*muG*mupi)/
              mbkin**12 - (243180*mcMS**14*muG*mupi)/mbkin**14 - 
             (289044*mcMS**16*muG*mupi)/mbkin**16 + (28404*mcMS**18*muG*mupi)/
              mbkin**18 + (6120*mcMS**20*muG*mupi)/mbkin**20 - 
             (2520*mcMS**2*muG**2*q_cut)/mbkin**4 + (30060*mcMS**4*muG**2*q_cut)/
              mbkin**6 + (108504*mcMS**6*muG**2*q_cut)/mbkin**8 - 
             (486972*mcMS**8*muG**2*q_cut)/mbkin**10 - (1597032*mcMS**10*muG**2*q_cut)/
              mbkin**12 - (2135052*mcMS**12*muG**2*q_cut)/mbkin**14 - 
             (563832*mcMS**14*muG**2*q_cut)/mbkin**16 + (273564*mcMS**16*muG**2*q_cut)/
              mbkin**18 + (18720*mcMS**18*muG**2*q_cut)/mbkin**20 + 
             (7560*mcMS**2*muG*mupi*q_cut)/mbkin**4 - (30060*mcMS**4*muG*mupi*q_cut)/
              mbkin**6 - (214344*mcMS**6*muG*mupi*q_cut)/mbkin**8 + 
             (476892*mcMS**8*muG*mupi*q_cut)/mbkin**10 + (2035512*mcMS**10*muG*mupi*
               q_cut)/mbkin**12 + (1832652*mcMS**12*muG*mupi*q_cut)/mbkin**14 + 
             (387432*mcMS**14*muG*mupi*q_cut)/mbkin**16 - (122364*mcMS**16*muG*mupi*
               q_cut)/mbkin**18 - (18720*mcMS**18*muG*mupi*q_cut)/mbkin**20 + 
             (5040*mcMS**2*muG**2*q_cut**2)/mbkin**6 - (131400*mcMS**6*muG**2*q_cut**2)/
              mbkin**10 - (42912*mcMS**8*muG**2*q_cut**2)/mbkin**12 + 
             (603144*mcMS**10*muG**2*q_cut**2)/mbkin**14 + (130968*mcMS**12*muG**2*q_cut**
                2)/mbkin**16 - (189360*mcMS**14*muG**2*q_cut**2)/mbkin**18 - 
             (12600*mcMS**16*muG**2*q_cut**2)/mbkin**20 - (5040*mcMS**2*muG*mupi*q_cut**
                2)/mbkin**6 + (10080*mcMS**4*muG*mupi*q_cut**2)/mbkin**8 + 
             (111240*mcMS**6*muG*mupi*q_cut**2)/mbkin**10 - 
             (98208*mcMS**8*muG*mupi*q_cut**2)/mbkin**12 - (401544*mcMS**10*muG*mupi*
               q_cut**2)/mbkin**14 - (80568*mcMS**12*muG*mupi*q_cut**2)/mbkin**16 + 
             (88560*mcMS**14*muG*mupi*q_cut**2)/mbkin**18 + 
             (12600*mcMS**16*muG*mupi*q_cut**2)/mbkin**20 + 
             (15120*mcMS**2*muG**2*q_cut**3)/mbkin**8 + (12600*mcMS**4*muG**2*q_cut**3)/
              mbkin**10 - (302760*mcMS**6*muG**2*q_cut**3)/mbkin**12 - 
             (719352*mcMS**8*muG**2*q_cut**3)/mbkin**14 - (676512*mcMS**10*muG**2*q_cut**
                3)/mbkin**16 - (239760*mcMS**12*muG**2*q_cut**3)/mbkin**18 - 
             (12600*mcMS**14*muG**2*q_cut**3)/mbkin**20 - (5040*mcMS**2*muG*mupi*q_cut**
                3)/mbkin**8 + (7560*mcMS**4*muG*mupi*q_cut**3)/mbkin**10 + 
             (161640*mcMS**6*muG*mupi*q_cut**3)/mbkin**12 + 
             (316152*mcMS**8*muG*mupi*q_cut**3)/mbkin**14 + 
             (323712*mcMS**10*muG*mupi*q_cut**3)/mbkin**16 + 
             (138960*mcMS**12*muG*mupi*q_cut**3)/mbkin**18 + 
             (12600*mcMS**14*muG*mupi*q_cut**3)/mbkin**20 - 
             (22680*mcMS**2*muG**2*q_cut**4)/mbkin**10 - (3780*mcMS**4*muG**2*q_cut**4)/
              mbkin**12 + (454500*mcMS**6*muG**2*q_cut**4)/mbkin**14 + 
             (754668*mcMS**8*muG**2*q_cut**4)/mbkin**16 + (343620*mcMS**10*muG**2*q_cut**
                4)/mbkin**18 + (17640*mcMS**12*muG**2*q_cut**4)/mbkin**20 + 
             (7560*mcMS**2*muG*mupi*q_cut**4)/mbkin**10 - (11340*mcMS**4*muG*mupi*q_cut**
                4)/mbkin**12 - (227700*mcMS**6*muG*mupi*q_cut**4)/mbkin**14 - 
             (376668*mcMS**8*muG*mupi*q_cut**4)/mbkin**16 - 
             (192420*mcMS**10*muG*mupi*q_cut**4)/mbkin**18 - 
             (17640*mcMS**12*muG*mupi*q_cut**4)/mbkin**20 + 
             (7560*mcMS**2*muG**2*q_cut**5)/mbkin**12 - (6300*mcMS**4*muG**2*q_cut**5)/
              mbkin**14 - (143496*mcMS**6*muG**2*q_cut**5)/mbkin**16 - 
             (109476*mcMS**8*muG**2*q_cut**5)/mbkin**18 - (5040*mcMS**10*muG**2*q_cut**5)/
              mbkin**20 - (2520*mcMS**2*muG*mupi*q_cut**5)/mbkin**12 + 
             (6300*mcMS**4*muG*mupi*q_cut**5)/mbkin**14 + (67896*mcMS**6*muG*mupi*q_cut**
                5)/mbkin**16 + (59076*mcMS**8*muG*mupi*q_cut**5)/mbkin**18 + 
             (5040*mcMS**10*muG*mupi*q_cut**5)/mbkin**20 + (2520*mcMS**4*muG**2*q_cut**6)/
              mbkin**16 + (4176*mcMS**6*muG**2*q_cut**6)/mbkin**18 + 
             (2520*mcMS**8*muG**2*q_cut**6)/mbkin**20 - (2520*mcMS**4*muG*mupi*q_cut**6)/
              mbkin**16 - (4176*mcMS**6*muG*mupi*q_cut**6)/mbkin**18 - 
             (2520*mcMS**8*muG*mupi*q_cut**6)/mbkin**20 - (3960*mcMS**4*muG**2*q_cut**7)/
              mbkin**18 - (3960*mcMS**6*muG**2*q_cut**7)/mbkin**20 + 
             (3960*mcMS**4*muG*mupi*q_cut**7)/mbkin**18 + (3960*mcMS**6*muG*mupi*q_cut**
                7)/mbkin**20 + (1440*mcMS**4*muG**2*q_cut**8)/mbkin**20 - 
             (1440*mcMS**4*muG*mupi*q_cut**8)/mbkin**20 + 72*mcMS**2*muG*
              ((-1 + mcMS**2/mbkin**2)**2*(-70 + (345*mcMS**2)/mbkin**2 + 
                 (2179*mcMS**4)/mbkin**4 - (8286*mcMS**6)/mbkin**6 - 
                 (19766*mcMS**8)/mbkin**8 - (5941*mcMS**10)/mbkin**10 + 
                 (1129*mcMS**12)/mbkin**12 + (170*mcMS**14)/mbkin**14) + 
               ((210 - (835*mcMS**2)/mbkin**2 - (5954*mcMS**4)/mbkin**4 + 
                  (13247*mcMS**6)/mbkin**6 + (56542*mcMS**8)/mbkin**8 + 
                  (50907*mcMS**10)/mbkin**10 + (10762*mcMS**12)/mbkin**12 - 
                  (3399*mcMS**14)/mbkin**14 - (520*mcMS**16)/mbkin**16)*q_cut)/
                mbkin**2 + (2*(-70 + (140*mcMS**2)/mbkin**2 + (1545*mcMS**4)/
                   mbkin**4 - (1364*mcMS**6)/mbkin**6 - (5577*mcMS**8)/mbkin**8 - 
                  (1119*mcMS**10)/mbkin**10 + (1230*mcMS**12)/mbkin**12 + 
                  (175*mcMS**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
               (2*(-70 + (105*mcMS**2)/mbkin**2 + (2245*mcMS**4)/mbkin**4 + 
                  (4391*mcMS**6)/mbkin**6 + (4496*mcMS**8)/mbkin**8 + 
                  (1930*mcMS**10)/mbkin**10 + (175*mcMS**12)/mbkin**12)*q_cut**3)/
                mbkin**6 - ((-210 + (315*mcMS**2)/mbkin**2 + (6325*mcMS**4)/
                   mbkin**4 + (10463*mcMS**6)/mbkin**6 + (5345*mcMS**8)/mbkin**8 + 
                  (490*mcMS**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
               ((-70 + (175*mcMS**2)/mbkin**2 + (1886*mcMS**4)/mbkin**4 + 
                  (1641*mcMS**6)/mbkin**6 + (140*mcMS**8)/mbkin**8)*q_cut**5)/
                mbkin**10 - (2*mcMS**2*(35 + (58*mcMS**2)/mbkin**2 + (35*mcMS**4)/
                   mbkin**4)*q_cut**6)/mbkin**14 + (110*mcMS**2*(mbkin**2 + mcMS**2)*
                 q_cut**7)/mbkin**18 - (40*mcMS**2*q_cut**8)/mbkin**18) + 
             4*(-((-1 + mcMS**2/mbkin**2)**2*(-470 - (5510*mcMS**2)/mbkin**2 + 
                  (55113*mcMS**4)/mbkin**4 + (61525*mcMS**6)/mbkin**6 - 
                  (282350*mcMS**8)/mbkin**8 - (267510*mcMS**10)/mbkin**10 - 
                  (31085*mcMS**12)/mbkin**12 + (14647*mcMS**14)/mbkin**14 + 
                  (2040*mcMS**16)/mbkin**16)) + ((-1450 - (18580*mcMS**2)/
                   mbkin**2 + (103993*mcMS**4)/mbkin**4 + (231998*mcMS**6)/
                   mbkin**6 - (425575*mcMS**8)/mbkin**8 - (1094180*mcMS**10)/
                   mbkin**10 - (618145*mcMS**12)/mbkin**12 - (41638*mcMS**14)/
                   mbkin**14 + (42937*mcMS**16)/mbkin**16 + (6240*mcMS**18)/
                   mbkin**18)*q_cut)/mbkin**2 - (2*(-490 - (7300*mcMS**2)/mbkin**2 + 
                  (13660*mcMS**4)/mbkin**4 + (22907*mcMS**6)/mbkin**6 - 
                  (77531*mcMS**8)/mbkin**8 - (62779*mcMS**10)/mbkin**10 + 
                  (16973*mcMS**12)/mbkin**12 + (16860*mcMS**14)/mbkin**14 + 
                  (2100*mcMS**16)/mbkin**16)*q_cut**2)/mbkin**4 - 
               (2*(-490 - (8770*mcMS**2)/mbkin**2 - (7155*mcMS**4)/mbkin**4 + 
                  (17517*mcMS**6)/mbkin**6 + (38840*mcMS**8)/mbkin**8 + 
                  (45853*mcMS**10)/mbkin**10 + (18785*mcMS**12)/mbkin**12 + 
                  (2100*mcMS**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
               ((-1330 - (26770*mcMS**2)/mbkin**2 - (18885*mcMS**4)/mbkin**4 + 
                  (89471*mcMS**6)/mbkin**6 + (131639*mcMS**8)/mbkin**8 + 
                  (55915*mcMS**10)/mbkin**10 + (5880*mcMS**12)/mbkin**12)*q_cut**4)/
                mbkin**8 - ((-350 - (8440*mcMS**2)/mbkin**2 + (3047*mcMS**4)/
                   mbkin**4 + (29380*mcMS**6)/mbkin**6 + (16563*mcMS**8)/
                   mbkin**8 + (1680*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
               (8*(-35 - (30*mcMS**2)/mbkin**2 + (14*mcMS**4)/mbkin**4 + 
                  (76*mcMS**6)/mbkin**6 + (105*mcMS**8)/mbkin**8)*q_cut**6)/
                mbkin**12 - (40*(-11 - (11*mcMS**2)/mbkin**2 + (26*mcMS**4)/
                   mbkin**4 + (33*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 - 
               (160*(mbkin**4 - 3*mcMS**4)*q_cut**8)/mbkin**20)*rG - 
             24*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*(-40 - (530*mcMS**2)/
                   mbkin**2 + (1237*mcMS**4)/mbkin**4 + (117101*mcMS**6)/
                   mbkin**6 + (301726*mcMS**8)/mbkin**8 + (144226*mcMS**10)/
                   mbkin**10 - (5749*mcMS**12)/mbkin**12 + (619*mcMS**14)/
                   mbkin**14 + (850*mcMS**16)/mbkin**16)) + 
               ((-110 - (2180*mcMS**2)/mbkin**2 - (8043*mcMS**4)/mbkin**4 + 
                  (231598*mcMS**6)/mbkin**6 + (844217*mcMS**8)/mbkin**8 + 
                  (919012*mcMS**10)/mbkin**10 + (277427*mcMS**12)/mbkin**12 - 
                  (28230*mcMS**14)/mbkin**14 + (1469*mcMS**16)/mbkin**16 + 
                  (2600*mcMS**18)/mbkin**18)*q_cut)/mbkin**2 - (2*(-35 - 
                  (870*mcMS**2)/mbkin**2 - (7720*mcMS**4)/mbkin**4 + 
                  (19403*mcMS**6)/mbkin**6 + (68795*mcMS**8)/mbkin**8 + 
                  (21911*mcMS**10)/mbkin**10 - (10719*mcMS**12)/mbkin**12 + 
                  (1600*mcMS**14)/mbkin**14 + (875*mcMS**16)/mbkin**16)*q_cut**2)/
                mbkin**4 + (2*(35 + (975*mcMS**2)/mbkin**2 + (11590*mcMS**4)/
                   mbkin**4 + (21792*mcMS**6)/mbkin**6 + (19921*mcMS**8)/
                   mbkin**8 + (9944*mcMS**10)/mbkin**10 - (2650*mcMS**12)/
                   mbkin**12 - (875*mcMS**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
               ((-140 - (2810*mcMS**2)/mbkin**2 - (35835*mcMS**4)/mbkin**4 - 
                  (69801*mcMS**6)/mbkin**6 - (30887*mcMS**8)/mbkin**8 + 
                  (8455*mcMS**10)/mbkin**10 + (2450*mcMS**12)/mbkin**12)*q_cut**4)/
                mbkin**8 + ((70 + (900*mcMS**2)/mbkin**2 + (10637*mcMS**4)/
                   mbkin**4 + (9736*mcMS**6)/mbkin**6 - (2311*mcMS**8)/mbkin**8 - 
                  (700*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
               ((70 + (60*mcMS**2)/mbkin**2 + (308*mcMS**4)/mbkin**4 + 
                  (76*mcMS**6)/mbkin**6 + (350*mcMS**8)/mbkin**8)*q_cut**6)/
                mbkin**12 - (10*(11 + (11*mcMS**2)/mbkin**2 + (27*mcMS**4)/
                   mbkin**4 + (55*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
               (40*(mbkin**4 + 5*mcMS**4)*q_cut**8)/mbkin**20)*rhoD - 2520*sB - 
             (36120*mcMS**2*sB)/mbkin**2 + (411444*mcMS**4*sB)/mbkin**4 + 
             (1483596*mcMS**6*sB)/mbkin**6 - (1465380*mcMS**8*sB)/mbkin**8 - 
             (1936620*mcMS**10*sB)/mbkin**10 + (711900*mcMS**12*sB)/mbkin**12 + 
             (534996*mcMS**14*sB)/mbkin**14 + (377244*mcMS**16*sB)/mbkin**16 - 
             (58140*mcMS**18*sB)/mbkin**18 - (20400*mcMS**20*sB)/mbkin**20 + 
             (7560*q_cut*sB)/mbkin**2 + (141120*mcMS**2*q_cut*sB)/mbkin**4 - 
             (497964*mcMS**4*q_cut*sB)/mbkin**6 - (5101848*mcMS**6*q_cut*sB)/mbkin**8 - 
             (9831468*mcMS**8*q_cut*sB)/mbkin**10 - (7031328*mcMS**10*q_cut*sB)/
              mbkin**12 - (2379828*mcMS**12*q_cut*sB)/mbkin**14 - 
             (327624*mcMS**14*q_cut*sB)/mbkin**16 + (283140*mcMS**16*q_cut*sB)/
              mbkin**18 + (62400*mcMS**18*q_cut*sB)/mbkin**20 - (5040*q_cut**2*sB)/
              mbkin**4 - (107520*mcMS**2*q_cut**2*sB)/mbkin**6 - 
             (50400*mcMS**4*q_cut**2*sB)/mbkin**8 + (1033512*mcMS**6*q_cut**2*sB)/
              mbkin**10 + (1250856*mcMS**8*q_cut**2*sB)/mbkin**12 + 
             (328632*mcMS**10*q_cut**2*sB)/mbkin**14 - (123720*mcMS**12*q_cut**2*sB)/
              mbkin**16 - (228000*mcMS**14*q_cut**2*sB)/mbkin**18 - 
             (42000*mcMS**16*q_cut**2*sB)/mbkin**20 - (5040*q_cut**3*sB)/mbkin**6 - 
             (122640*mcMS**2*q_cut**3*sB)/mbkin**8 - (457800*mcMS**4*q_cut**3*sB)/
              mbkin**10 - (691848*mcMS**6*q_cut**3*sB)/mbkin**12 - 
             (888528*mcMS**8*q_cut**3*sB)/mbkin**14 - (681720*mcMS**10*q_cut**3*sB)/
              mbkin**16 - (307800*mcMS**12*q_cut**3*sB)/mbkin**18 - 
             (42000*mcMS**14*q_cut**3*sB)/mbkin**20 + (7560*q_cut**4*sB)/mbkin**8 + 
             (183960*mcMS**2*q_cut**4*sB)/mbkin**10 + (716940*mcMS**4*q_cut**4*sB)/
              mbkin**12 + (1069212*mcMS**6*q_cut**4*sB)/mbkin**14 + 
             (894780*mcMS**8*q_cut**4*sB)/mbkin**16 + (445260*mcMS**10*q_cut**4*sB)/
              mbkin**18 + (58800*mcMS**12*q_cut**4*sB)/mbkin**20 - 
             (2520*q_cut**5*sB)/mbkin**10 - (58800*mcMS**2*q_cut**5*sB)/mbkin**12 - 
             (185724*mcMS**4*q_cut**5*sB)/mbkin**14 - (197424*mcMS**6*q_cut**5*sB)/
              mbkin**16 - (131820*mcMS**8*q_cut**5*sB)/mbkin**18 - 
             (16800*mcMS**10*q_cut**5*sB)/mbkin**20 + (3024*mcMS**4*q_cut**6*sB)/
              mbkin**16 + (7200*mcMS**6*q_cut**6*sB)/mbkin**18 + 
             (8400*mcMS**8*q_cut**6*sB)/mbkin**20 - (9840*mcMS**4*q_cut**7*sB)/
              mbkin**18 - (13200*mcMS**6*q_cut**7*sB)/mbkin**20 + 
             (4800*mcMS**4*q_cut**8*sB)/mbkin**20 + 1360*sE + (27200*mcMS**2*sE)/
              mbkin**2 - (259360*mcMS**4*sE)/mbkin**4 - (553312*mcMS**6*sE)/
              mbkin**6 + (1278032*mcMS**8*sE)/mbkin**8 + (59360*mcMS**10*sE)/
              mbkin**10 - (668080*mcMS**12*sE)/mbkin**12 + (218720*mcMS**14*sE)/
              mbkin**14 - (143008*mcMS**16*sE)/mbkin**16 + (26848*mcMS**18*sE)/
              mbkin**18 + (12240*mcMS**20*sE)/mbkin**20 - (4160*q_cut*sE)/mbkin**2 - 
             (96800*mcMS**2*q_cut*sE)/mbkin**4 + (332240*mcMS**4*q_cut*sE)/mbkin**6 + 
             (2431312*mcMS**6*q_cut*sE)/mbkin**8 + (3031744*mcMS**8*q_cut*sE)/
              mbkin**10 + (707744*mcMS**10*q_cut*sE)/mbkin**12 + 
             (12784*mcMS**12*q_cut*sE)/mbkin**14 + (68464*mcMS**14*q_cut*sE)/
              mbkin**16 - (155968*mcMS**16*q_cut*sE)/mbkin**18 - 
             (37440*mcMS**18*q_cut*sE)/mbkin**20 + (2800*q_cut**2*sE)/mbkin**4 + 
             (71200*mcMS**2*q_cut**2*sE)/mbkin**6 - (16000*mcMS**4*q_cut**2*sE)/
              mbkin**8 - (560480*mcMS**6*q_cut**2*sE)/mbkin**10 - 
             (296416*mcMS**8*q_cut**2*sE)/mbkin**12 + (240832*mcMS**10*q_cut**2*sE)/
              mbkin**14 + (251584*mcMS**12*q_cut**2*sE)/mbkin**16 + 
             (160320*mcMS**14*q_cut**2*sE)/mbkin**18 + (25200*mcMS**16*q_cut**2*sE)/
              mbkin**20 + (2800*q_cut**3*sE)/mbkin**6 + (79600*mcMS**2*q_cut**3*sE)/
              mbkin**8 + (217200*mcMS**4*q_cut**3*sE)/mbkin**10 + 
             (182640*mcMS**6*q_cut**3*sE)/mbkin**12 + (108784*mcMS**8*q_cut**3*sE)/
              mbkin**14 + (46544*mcMS**10*q_cut**3*sE)/mbkin**16 + 
             (112720*mcMS**12*q_cut**3*sE)/mbkin**18 + (25200*mcMS**14*q_cut**3*sE)/
              mbkin**20 - (3920*q_cut**4*sE)/mbkin**8 - (120320*mcMS**2*q_cut**4*sE)/
              mbkin**10 - (337440*mcMS**4*q_cut**4*sE)/mbkin**12 - 
             (306080*mcMS**6*q_cut**4*sE)/mbkin**14 - (255776*mcMS**8*q_cut**4*sE)/
              mbkin**16 - (207040*mcMS**10*q_cut**4*sE)/mbkin**18 - 
             (35280*mcMS**12*q_cut**4*sE)/mbkin**20 + (1120*q_cut**5*sE)/mbkin**10 + 
             (38720*mcMS**2*q_cut**5*sE)/mbkin**12 + (80720*mcMS**4*q_cut**5*sE)/
              mbkin**14 + (62992*mcMS**6*q_cut**5*sE)/mbkin**16 + 
             (56832*mcMS**8*q_cut**5*sE)/mbkin**18 + (10080*mcMS**10*q_cut**5*sE)/
              mbkin**20 - (560*q_cut**6*sE)/mbkin**12 - (480*mcMS**2*q_cut**6*sE)/
              mbkin**14 - (1120*mcMS**4*q_cut**6*sE)/mbkin**16 + 
             (2848*mcMS**6*q_cut**6*sE)/mbkin**18 - (5040*mcMS**8*q_cut**6*sE)/
              mbkin**20 + (880*q_cut**7*sE)/mbkin**14 + (880*mcMS**2*q_cut**7*sE)/
              mbkin**16 + (3440*mcMS**4*q_cut**7*sE)/mbkin**18 + 
             (7920*mcMS**6*q_cut**7*sE)/mbkin**20 - (320*q_cut**8*sE)/mbkin**16 - 
             (2880*mcMS**4*q_cut**8*sE)/mbkin**20 + 460*sqB + (3110*mcMS**2*sqB)/
              mbkin**2 - (53971*mcMS**4*sqB)/mbkin**4 - (232651*mcMS**6*sqB)/
              mbkin**6 + (64337*mcMS**8*sqB)/mbkin**8 + (481775*mcMS**10*sqB)/
              mbkin**10 - (5635*mcMS**12*sqB)/mbkin**12 - (222829*mcMS**14*sqB)/
              mbkin**14 - (41029*mcMS**16*sqB)/mbkin**16 + (5923*mcMS**18*sqB)/
              mbkin**18 + (510*mcMS**20*sqB)/mbkin**20 - (1370*q_cut*sqB)/mbkin**2 - 
             (15620*mcMS**2*q_cut*sqB)/mbkin**4 + (66491*mcMS**4*q_cut*sqB)/mbkin**6 + 
             (746374*mcMS**6*q_cut*sqB)/mbkin**8 + (1788331*mcMS**8*q_cut*sqB)/
              mbkin**10 + (1693796*mcMS**10*q_cut*sqB)/mbkin**12 + 
             (617581*mcMS**12*q_cut*sqB)/mbkin**14 + (27010*mcMS**14*q_cut*sqB)/
              mbkin**16 - (22153*mcMS**16*q_cut*sqB)/mbkin**18 - 
             (1560*mcMS**18*q_cut*sqB)/mbkin**20 + (910*q_cut**2*sqB)/mbkin**4 + 
             (12940*mcMS**2*q_cut**2*sqB)/mbkin**6 + (12080*mcMS**4*q_cut**2*sqB)/
              mbkin**8 - (138098*mcMS**6*q_cut**2*sqB)/mbkin**10 - 
             (257050*mcMS**8*q_cut**2*sqB)/mbkin**12 - (92426*mcMS**10*q_cut**2*sqB)/
              mbkin**14 + (34474*mcMS**12*q_cut**2*sqB)/mbkin**16 + 
             (17880*mcMS**14*q_cut**2*sqB)/mbkin**18 + (1050*mcMS**16*q_cut**2*sqB)/
              mbkin**20 + (910*q_cut**3*sqB)/mbkin**6 + (15670*mcMS**2*q_cut**3*sqB)/
              mbkin**8 + (63360*mcMS**4*q_cut**3*sqB)/mbkin**10 + 
             (93252*mcMS**6*q_cut**3*sqB)/mbkin**12 + (86746*mcMS**8*q_cut**3*sqB)/
              mbkin**14 + (58364*mcMS**10*q_cut**3*sqB)/mbkin**16 + 
             (16480*mcMS**12*q_cut**3*sqB)/mbkin**18 + (1050*mcMS**14*q_cut**3*sqB)/
              mbkin**20 - (1400*q_cut**4*sqB)/mbkin**8 - (23390*mcMS**2*q_cut**4*sqB)/
              mbkin**10 - (101145*mcMS**4*q_cut**4*sqB)/mbkin**12 - 
             (159083*mcMS**6*q_cut**4*sqB)/mbkin**14 - (104861*mcMS**8*q_cut**4*sqB)/
              mbkin**16 - (27235*mcMS**10*q_cut**4*sqB)/mbkin**18 - 
             (1470*mcMS**12*q_cut**4*sqB)/mbkin**20 + (490*q_cut**5*sqB)/mbkin**10 + 
             (7340*mcMS**2*q_cut**5*sqB)/mbkin**12 + (26891*mcMS**4*q_cut**5*sqB)/
              mbkin**14 + (27988*mcMS**6*q_cut**5*sqB)/mbkin**16 + 
             (8787*mcMS**8*q_cut**5*sqB)/mbkin**18 + (420*mcMS**10*q_cut**5*sqB)/
              mbkin**20 + (70*q_cut**6*sqB)/mbkin**12 + (60*mcMS**2*q_cut**6*sqB)/
              mbkin**14 + (644*mcMS**4*q_cut**6*sqB)/mbkin**16 + 
             (268*mcMS**6*q_cut**6*sqB)/mbkin**18 - (210*mcMS**8*q_cut**6*sqB)/
              mbkin**20 - (110*q_cut**7*sqB)/mbkin**14 - (110*mcMS**2*q_cut**7*sqB)/
              mbkin**16 + (50*mcMS**4*q_cut**7*sqB)/mbkin**18 + 
             (330*mcMS**6*q_cut**7*sqB)/mbkin**20 + (40*q_cut**8*sqB)/mbkin**16 - 
             (120*mcMS**4*q_cut**8*sqB)/mbkin**20))*
          np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 - 
        (60480*mcMS**8*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
             2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(72*mcMS**2*muG - 
           (1080*mcMS**6*muG)/mbkin**4 - (720*mcMS**8*muG)/mbkin**6 - 
           (36*mcMS**2*muG**2)/mbkin**2 + (540*mcMS**6*muG**2)/mbkin**6 + 
           (360*mcMS**8*muG**2)/mbkin**8 + (36*mcMS**2*muG*mupi)/mbkin**2 - 
           (540*mcMS**6*muG*mupi)/mbkin**6 - (360*mcMS**8*muG*mupi)/mbkin**8 - 
           16*(-6 - (47*mcMS**2)/mbkin**2 + (70*mcMS**4)/mbkin**4 + 
             (55*mcMS**6)/mbkin**6)*rE + 4*(3 - (68*mcMS**2)/mbkin**2 + 
             (55*mcMS**4)/mbkin**4 + (280*mcMS**6)/mbkin**6 + (90*mcMS**8)/
              mbkin**8)*rG + 72*mbkin*rhoD + (624*mcMS**2*rhoD)/mbkin + 
           (4200*mcMS**4*rhoD)/mbkin**3 + (5040*mcMS**6*rhoD)/mbkin**5 + 
           (720*mcMS**8*rhoD)/mbkin**7 + 36*sB + (840*mcMS**2*sB)/mbkin**2 + 
           (2340*mcMS**4*sB)/mbkin**4 + (1320*mcMS**6*sB)/mbkin**6 + 
           (360*mcMS**8*sB)/mbkin**8 - (448*mcMS**2*sE)/mbkin**2 - 
           (880*mcMS**4*sE)/mbkin**4 + (80*mcMS**6*sE)/mbkin**6 - 9*sqB - 
           (118*mcMS**2*sqB)/mbkin**2 - (385*mcMS**4*sqB)/mbkin**4 - 
           (370*mcMS**6*sqB)/mbkin**6 - (90*mcMS**8*sqB)/mbkin**8)*
          np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8))/
      (2520*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
          2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
            7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
            mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**3) + 
     (api4*(((180*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
              7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
              mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)**2*(mbkin**12 - 
             34*mbkin**10*mcMS**2 - 1133*mbkin**8*mcMS**4 - 2708*mbkin**6*mcMS**6 - 
             1133*mbkin**4*mcMS**8 - 34*mbkin**2*mcMS**10 + mcMS**12 + 
             mbkin**10*q_cut - 31*mbkin**8*mcMS**2*q_cut - 390*mbkin**6*mcMS**4*q_cut - 
             390*mbkin**4*mcMS**6*q_cut - 31*mbkin**2*mcMS**8*q_cut + mcMS**10*q_cut + 
             mbkin**8*q_cut**2 - 26*mbkin**6*mcMS**2*q_cut**2 - 118*mbkin**4*mcMS**4*
              q_cut**2 - 26*mbkin**2*mcMS**6*q_cut**2 + mcMS**8*q_cut**2 + mbkin**6*q_cut**3 - 
             19*mbkin**4*mcMS**2*q_cut**3 - 19*mbkin**2*mcMS**4*q_cut**3 + mcMS**6*q_cut**3 - 
             6*mbkin**4*q_cut**4 + 4*mbkin**2*mcMS**2*q_cut**4 - 6*mcMS**4*q_cut**4 - 
             6*mbkin**2*q_cut**5 - 6*mcMS**2*q_cut**5 + 8*q_cut**6))/mbkin**32 - 
          (2160*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**
                4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
            (37*mbkin**18 - 397*mbkin**16*mcMS**2 - 2854*mbkin**14*mcMS**4 + 
             18134*mbkin**12*mcMS**6 + 75800*mbkin**10*mcMS**8 + 
             75800*mbkin**8*mcMS**10 + 18134*mbkin**6*mcMS**12 - 
             2854*mbkin**4*mcMS**14 - 397*mbkin**2*mcMS**16 + 37*mcMS**18 - 
             70*mbkin**16*q_cut + 132*mbkin**14*mcMS**2*q_cut + 4424*mbkin**12*mcMS**4*
              q_cut + 15500*mbkin**10*mcMS**6*q_cut + 20508*mbkin**8*mcMS**8*q_cut + 
             15500*mbkin**6*mcMS**10*q_cut + 4424*mbkin**4*mcMS**12*q_cut + 
             132*mbkin**2*mcMS**14*q_cut - 70*mcMS**16*q_cut - 37*mbkin**14*q_cut**2 + 
             307*mbkin**12*mcMS**2*q_cut**2 + 6201*mbkin**10*mcMS**4*q_cut**2 + 
             18225*mbkin**8*mcMS**6*q_cut**2 + 18225*mbkin**6*mcMS**8*q_cut**2 + 
             6201*mbkin**4*mcMS**10*q_cut**2 + 307*mbkin**2*mcMS**12*q_cut**2 - 
             37*mcMS**14*q_cut**2 + 140*mbkin**12*q_cut**3 + 272*mbkin**10*mcMS**2*q_cut**3 - 
             2796*mbkin**8*mcMS**4*q_cut**3 - 7136*mbkin**6*mcMS**6*q_cut**3 - 
             2796*mbkin**4*mcMS**8*q_cut**3 + 272*mbkin**2*mcMS**10*q_cut**3 + 
             140*mcMS**12*q_cut**3 - 49*mbkin**10*q_cut**4 + 13*mbkin**8*mcMS**2*q_cut**4 - 
             300*mbkin**6*mcMS**4*q_cut**4 - 300*mbkin**4*mcMS**6*q_cut**4 + 
             13*mbkin**2*mcMS**8*q_cut**4 - 49*mcMS**10*q_cut**4 - 70*mbkin**8*q_cut**5 - 
             372*mbkin**6*mcMS**2*q_cut**5 - 668*mbkin**4*mcMS**4*q_cut**5 - 
             372*mbkin**2*mcMS**6*q_cut**5 - 70*mcMS**8*q_cut**5 + 77*mbkin**6*q_cut**6 + 
             41*mbkin**4*mcMS**2*q_cut**6 + 41*mbkin**2*mcMS**4*q_cut**6 + 
             77*mcMS**6*q_cut**6 - 16*mbkin**4*q_cut**7 + 32*mbkin**2*mcMS**2*q_cut**7 - 
             16*mcMS**4*q_cut**7 - 28*mbkin**2*q_cut**8 - 28*mcMS**2*q_cut**8 + 16*q_cut**9)*
            np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                  mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**26 + 
          (25920*mcMS**8*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**2*(71*mbkin**12 - 174*mbkin**10*mcMS**2 - 
             3723*mbkin**8*mcMS**4 - 7468*mbkin**6*mcMS**6 - 3723*mbkin**4*
              mcMS**8 - 174*mbkin**2*mcMS**10 + 71*mcMS**12 - 69*mbkin**10*q_cut - 
             381*mbkin**8*mcMS**2*q_cut - 810*mbkin**6*mcMS**4*q_cut - 
             810*mbkin**4*mcMS**6*q_cut - 381*mbkin**2*mcMS**8*q_cut - 69*mcMS**10*q_cut - 
             69*mbkin**8*q_cut**2 - 446*mbkin**6*mcMS**2*q_cut**2 - 818*mbkin**4*mcMS**4*
              q_cut**2 - 446*mbkin**2*mcMS**6*q_cut**2 - 69*mcMS**8*q_cut**2 + 
             71*mbkin**6*q_cut**3 + 331*mbkin**4*mcMS**2*q_cut**3 + 331*mbkin**2*mcMS**4*
              q_cut**3 + 71*mcMS**6*q_cut**3 - 6*mbkin**4*q_cut**4 + 4*mbkin**2*mcMS**2*
              q_cut**4 - 6*mcMS**4*q_cut**4 - 6*mbkin**2*q_cut**5 - 6*mcMS**2*q_cut**5 + 8*q_cut**6)*
            np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**24 - 
          (10886400*mcMS**12*(mbkin**6 + 5*mbkin**4*mcMS**2 + 5*mbkin**2*mcMS**4 + 
             mcMS**6)*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*
                mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**14)*
         (((4*(3 + 8*mbkin))/(9*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)) - 
            (3*mbkin**2*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                 (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + (4*q_cut*
                 (6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                  12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**6)))/(2*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
              ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
               q_cut**2/mbkin**4)))/((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                  2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 
                7*mbkin**4*mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - 
                mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
             (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                      2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
            3 - (3*mbkin**2*((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                  2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 
                7*mbkin**4*mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - 
                mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)*(
                (-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                  (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                   9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                   12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                 (9*mbkin**6)))/(2*mbkin**6*((-1 + mcMS**2/mbkin**2)**2 - 
                (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)) + 
             np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*((-4*(3 + 8*mbkin)*q_cut**3)/
                (3*mbkin**8) - (14*mcMS**2*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) - (28*mcMS**4*
                 (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**6) + (2*mcMS**6*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(3*mbkin**8) + (2*q_cut**2*
                 (12*mbkin**2 + 32*mbkin**3 + 18*mcMS**2 + 48*mbkin*mcMS**2 - 
                  12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**8) + (4*q_cut*(3*mbkin**4 + 8*mbkin**5 + 9*mcMS**4 + 
                  24*mbkin*mcMS**4 - 12*mbkin**2*mcMS**4 - 9*mbkin**2*mcMS**4*
                   np.log(mu0**2/mcMS**2)))/(9*mbkin**8)) - 
             12*((mcMS**4*(16/3 + 4*np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - 
                    q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                   (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4))))/mbkin**4 + mcMS**4*
                ((-8*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 6*mbkin**2*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                     np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                         q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                    9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2)))/(9*mbkin**4*(mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)*
                   (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + mbkin**2*
                     np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                         q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                 (8*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                      mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                          q_cut**2)/mbkin**4))))/(9*mbkin**6)))))/
           (((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                q_cut + q_cut**2)/mbkin**4)**(3/2)*
            ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
                 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
                 mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
              (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                       2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                       2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**
                4)**4)) + (mbkin**2*(180*mbkin**4*((-1 + mcMS**2/mbkin**2)**2*(1 - 
                (7*mcMS**2)/mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/mbkin**6) + 
              ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + (14*mcMS**6)/
                  mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/mbkin**2 + 
              (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*mcMS**4 + mcMS**6)*
                q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/
               mbkin**10 - (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + 
              q_cut**5/mbkin**10)**2*(1 - (34*mcMS**2)/mbkin**2 - (1133*mcMS**4)/
              mbkin**4 - (2708*mcMS**6)/mbkin**6 - (1133*mcMS**8)/mbkin**8 - 
             (34*mcMS**10)/mbkin**10 + mcMS**12/mbkin**12 + 
             ((mbkin**10 - 31*mbkin**8*mcMS**2 - 390*mbkin**6*mcMS**4 - 
                390*mbkin**4*mcMS**6 - 31*mbkin**2*mcMS**8 + mcMS**10)*q_cut)/
              mbkin**12 + ((mbkin**8 - 26*mbkin**6*mcMS**2 - 118*mbkin**4*mcMS**4 - 
                26*mbkin**2*mcMS**6 + mcMS**8)*q_cut**2)/mbkin**12 + 
             ((mbkin**6 - 19*mbkin**4*mcMS**2 - 19*mbkin**2*mcMS**4 + mcMS**6)*q_cut**
                3)/mbkin**12 + ((-6 + (4*mcMS**2)/mbkin**2 - (6*mcMS**4)/mbkin**4)*
               q_cut**4)/mbkin**8 - (6*(mbkin**2 + mcMS**2)*q_cut**5)/mbkin**12 + 
             (8*q_cut**6)/mbkin**12)*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
             (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
             (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
              (9*mbkin**6)) + ((-1 + mcMS**2/mbkin**2)**2 - 
             (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
            ((64*mbkin*(-((-1 + mcMS**2/mbkin**2)**4*(1 + mcMS**2/mbkin**2)**2*
                  (2803 - (59389*mcMS**2)/mbkin**2 + (425638*mcMS**4)/mbkin**4 - 
                   (142434*mcMS**6)/mbkin**6 - (7007992*mcMS**8)/mbkin**8 - 
                   (411584*mcMS**10)/mbkin**10 + (753258*mcMS**12)/mbkin**12 - 
                   (298894*mcMS**14)/mbkin**14 + (24853*mcMS**16)/mbkin**16 + 
                   (461*mcMS**18)/mbkin**18)) + ((-1 + mcMS**2/mbkin**2)**2*
                  (15765 - (259196*mcMS**2)/mbkin**2 + (1031280*mcMS**4)/
                    mbkin**4 + (5842740*mcMS**6)/mbkin**6 - (24509183*mcMS**8)/
                    mbkin**8 - (87162472*mcMS**10)/mbkin**10 - (89842656*
                     mcMS**12)/mbkin**12 - (25831304*mcMS**14)/mbkin**14 + 
                   (7046591*mcMS**16)/mbkin**16 - (151836*mcMS**18)/mbkin**18 - 
                   (1156752*mcMS**20)/mbkin**20 + (149588*mcMS**22)/mbkin**22 + 
                   (2475*mcMS**24)/mbkin**24)*q_cut)/mbkin**2 - 
                (2*(15112 - (210025*mcMS**2)/mbkin**2 + (472499*mcMS**4)/
                    mbkin**4 + (5851497*mcMS**6)/mbkin**6 - (11357051*mcMS**8)/
                    mbkin**8 - (56972502*mcMS**10)/mbkin**10 - (85340102*
                     mcMS**12)/mbkin**12 - (61750930*mcMS**14)/mbkin**14 - 
                   (11562978*mcMS**16)/mbkin**16 + (7576839*mcMS**18)/mbkin**18 - 
                   (785517*mcMS**20)/mbkin**20 - (914135*mcMS**22)/mbkin**22 + 
                   (150229*mcMS**24)/mbkin**24 + (2104*mcMS**26)/mbkin**26)*q_cut**2)/
                 mbkin**4 + (2*(4935 - (22515*mcMS**2)/mbkin**2 - 
                   (355240*mcMS**4)/mbkin**4 + (1715991*mcMS**6)/mbkin**6 + 
                   (2655127*mcMS**8)/mbkin**8 - (4426866*mcMS**10)/mbkin**10 - 
                   (3311288*mcMS**12)/mbkin**12 + (4104698*mcMS**14)/mbkin**14 + 
                   (1043701*mcMS**16)/mbkin**16 - (1487323*mcMS**18)/mbkin**18 + 
                   (15840*mcMS**20)/mbkin**20 + (62895*mcMS**22)/mbkin**22 + 
                   (45*mcMS**24)/mbkin**24)*q_cut**3)/mbkin**6 + 
                ((43537 - (458971*mcMS**2)/mbkin**2 - (93925*mcMS**4)/mbkin**4 + 
                   (12415735*mcMS**6)/mbkin**6 + (21452530*mcMS**8)/mbkin**8 + 
                   (19866834*mcMS**10)/mbkin**10 + (19989798*mcMS**12)/
                    mbkin**12 + (10268870*mcMS**14)/mbkin**14 - 
                   (3574435*mcMS**16)/mbkin**16 - (1439255*mcMS**18)/mbkin**18 + 
                   (436447*mcMS**20)/mbkin**20 + (7139*mcMS**22)/mbkin**22)*q_cut**4)/
                 mbkin**8 - ((58149 - (424380*mcMS**2)/mbkin**2 - 
                   (1661061*mcMS**4)/mbkin**4 + (11127360*mcMS**6)/mbkin**6 + 
                   (29799650*mcMS**8)/mbkin**8 + (27800000*mcMS**10)/mbkin**10 + 
                   (5553270*mcMS**12)/mbkin**12 - (6210720*mcMS**14)/mbkin**14 - 
                   (461543*mcMS**16)/mbkin**16 + (648988*mcMS**18)/mbkin**18 + 
                   (6063*mcMS**20)/mbkin**20)*q_cut**5)/mbkin**10 + 
                (4*(2930 - (20127*mcMS**2)/mbkin**2 - (145371*mcMS**4)/mbkin**4 + 
                   (861647*mcMS**6)/mbkin**6 + (1702217*mcMS**8)/mbkin**8 + 
                   (293611*mcMS**10)/mbkin**10 - (560795*mcMS**12)/mbkin**12 + 
                   (4239*mcMS**14)/mbkin**14 + (35411*mcMS**16)/mbkin**16 + 
                   (110*mcMS**18)/mbkin**18)*q_cut**6)/mbkin**12 - 
                (4*(-5667 + (16545*mcMS**2)/mbkin**2 + (282306*mcMS**4)/
                    mbkin**4 + (576507*mcMS**6)/mbkin**6 + (446228*mcMS**8)/
                    mbkin**8 + (21045*mcMS**10)/mbkin**10 - (206250*mcMS**12)/
                    mbkin**12 - (83609*mcMS**14)/mbkin**14 + (951*mcMS**16)/
                    mbkin**16)*q_cut**7)/mbkin**14 + ((-13425 + (43605*mcMS**2)/
                    mbkin**2 + (1070385*mcMS**4)/mbkin**4 + (1555763*mcMS**6)/
                    mbkin**6 + (247781*mcMS**8)/mbkin**8 - (859585*mcMS**10)/
                    mbkin**10 - (271765*mcMS**12)/mbkin**12 + (8265*mcMS**14)/
                    mbkin**14)*q_cut**8)/mbkin**16 + ((3211 - (5958*mcMS**2)/
                    mbkin**2 - (298825*mcMS**4)/mbkin**4 - (133880*mcMS**6)/
                    mbkin**6 + (266305*mcMS**8)/mbkin**8 + (71726*mcMS**10)/
                    mbkin**10 - (643*mcMS**12)/mbkin**12)*q_cut**9)/mbkin**18 - 
                (2*(3444 + (8075*mcMS**2)/mbkin**2 + (7199*mcMS**4)/mbkin**4 + 
                   (24877*mcMS**6)/mbkin**6 + (14553*mcMS**8)/mbkin**8 + 
                   (4308*mcMS**10)/mbkin**10)*q_cut**10)/mbkin**20 + 
                (2*(3295 + (6917*mcMS**2)/mbkin**2 + (13436*mcMS**4)/mbkin**4 + 
                   (14159*mcMS**6)/mbkin**6 + (3805*mcMS**8)/mbkin**8)*q_cut**11)/
                 mbkin**22 - ((1869 + (3645*mcMS**2)/mbkin**2 + (9103*mcMS**4)/
                    mbkin**4 + (2463*mcMS**6)/mbkin**6)*q_cut**12)/mbkin**24 + 
                (5*(mbkin**4 + 80*mbkin**2*mcMS**2 + 59*mcMS**4)*q_cut**13)/
                 mbkin**30 - (48*(mbkin**2 + 2*mcMS**2)*q_cut**14)/mbkin**30 + 
                (40*q_cut**15)/mbkin**30))/3 + 180*(mbkin**4*
                ((-1 + mcMS**2/mbkin**2)**2*(1 - (7*mcMS**2)/mbkin**2 - (7*mcMS**4)/
                     mbkin**4 + mcMS**6/mbkin**6) + ((-3 + (14*mcMS**2)/mbkin**2 + 
                     (26*mcMS**4)/mbkin**4 + (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/
                      mbkin**8)*q_cut)/mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 
                     2*mbkin**2*mcMS**4 + mcMS**6)*q_cut**2)/mbkin**10 + 
                  (2*(mbkin**4 + mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/mbkin**10 - 
                  (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2*
                ((-64*(3 + 8*mbkin)*q_cut**6)/(3*mbkin**14) - (68*mcMS**2*
                   (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                       mcMS**2)))/(9*mbkin**4) - (4532*mcMS**4*(-6 - 16*mbkin + 
                    12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**6) - 
                 (5416*mcMS**6*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                     np.log(mu0**2/mcMS**2)))/(3*mbkin**8) - (9064*mcMS**8*
                   (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                       mcMS**2)))/(9*mbkin**10) - (340*mcMS**10*(-6 - 16*mbkin + 
                    12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**12) + 
                 (4*mcMS**12*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                     np.log(mu0**2/mcMS**2)))/(3*mbkin**14) + (4*q_cut**5*(30*mbkin**2 + 
                    80*mbkin**3 + 36*mcMS**2 + 96*mbkin*mcMS**2 - 12*mbkin**2*
                     mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                  (3*mbkin**14) + q_cut**4*((-16*(3 + 8*mbkin)*(-6 + (4*mcMS**2)/
                       mbkin**2 - (6*mcMS**4)/mbkin**4))/(9*mbkin**10) + 
                   (8*(mbkin**2*mcMS**2 - 3*mcMS**4)*(-6 - 16*mbkin + 
                      12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                    (9*mbkin**14)) + q_cut**3*((-4*(3 + 8*mbkin)*(1 - (19*mcMS**2)/
                       mbkin**2 - (19*mcMS**4)/mbkin**4 + mcMS**6/mbkin**6))/
                    (3*mbkin**8) - (2*(19*mbkin**4*mcMS**2 + 38*mbkin**2*mcMS**4 - 
                      3*mcMS**6)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                       np.log(mu0**2/mcMS**2)))/(9*mbkin**14)) + q_cut**2*
                  ((-8*(3 + 8*mbkin)*(1 - (26*mcMS**2)/mbkin**2 - (118*mcMS**4)/
                       mbkin**4 - (26*mcMS**6)/mbkin**6 + mcMS**8/mbkin**8))/
                    (9*mbkin**6) - (4*(13*mbkin**6*mcMS**2 + 118*mbkin**4*
                       mcMS**4 + 39*mbkin**2*mcMS**6 - 2*mcMS**8)*(-6 - 
                      16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                    (9*mbkin**14)) + q_cut*((-4*(3 + 8*mbkin)*(1 - (31*mcMS**2)/
                       mbkin**2 - (390*mcMS**4)/mbkin**4 - (390*mcMS**6)/
                       mbkin**6 - (31*mcMS**8)/mbkin**8 + mcMS**10/mbkin**10))/
                    (9*mbkin**4) - (2*(31*mbkin**8*mcMS**2 + 780*mbkin**6*
                       mcMS**4 + 1170*mbkin**4*mcMS**6 + 124*mbkin**2*mcMS**8 - 
                      5*mcMS**10)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                       np.log(mu0**2/mcMS**2)))/(9*mbkin**14))) + (1 - (34*mcMS**2)/
                  mbkin**2 - (1133*mcMS**4)/mbkin**4 - (2708*mcMS**6)/mbkin**6 - 
                 (1133*mcMS**8)/mbkin**8 - (34*mcMS**10)/mbkin**10 + 
                 mcMS**12/mbkin**12 + ((mbkin**10 - 31*mbkin**8*mcMS**2 - 
                    390*mbkin**6*mcMS**4 - 390*mbkin**4*mcMS**6 - 31*mbkin**2*
                     mcMS**8 + mcMS**10)*q_cut)/mbkin**12 + ((mbkin**8 - 26*mbkin**6*
                     mcMS**2 - 118*mbkin**4*mcMS**4 - 26*mbkin**2*mcMS**6 + 
                    mcMS**8)*q_cut**2)/mbkin**12 + ((mbkin**6 - 19*mbkin**4*mcMS**2 - 
                    19*mbkin**2*mcMS**4 + mcMS**6)*q_cut**3)/mbkin**12 + 
                 ((-6 + (4*mcMS**2)/mbkin**2 - (6*mcMS**4)/mbkin**4)*q_cut**4)/
                  mbkin**8 - (6*(mbkin**2 + mcMS**2)*q_cut**5)/mbkin**12 + 
                 (8*q_cut**6)/mbkin**12)*((8*mbkin**2*(3 + 8*mbkin)*
                   ((-1 + mcMS**2/mbkin**2)**2*(1 - (7*mcMS**2)/mbkin**2 - 
                       (7*mcMS**4)/mbkin**4 + mcMS**6/mbkin**6) + 
                     ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + 
                        (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/
                      mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*
                         mcMS**4 + mcMS**6)*q_cut**2)/mbkin**10 + (2*(mbkin**4 + 
                        mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/mbkin**10 - 
                     (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2)/
                  9 + 2*mbkin**4*((-1 + mcMS**2/mbkin**2)**2*(1 - (7*mcMS**2)/
                      mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/mbkin**6) + 
                   ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + 
                      (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/mbkin**2 + 
                   (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*mcMS**4 + 
                      mcMS**6)*q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mcMS**2 + 
                      mcMS**4)*q_cut**3)/mbkin**10 - (3*(mbkin**2 + mcMS**2)*q_cut**4)/
                    mbkin**10 + q_cut**5/mbkin**10)*((-20*(3 + 8*mbkin)*q_cut**5)/
                    (9*mbkin**12) - (2*(mbkin**2 - mcMS**2)*(9*mbkin**6*mcMS**2 - 
                      7*mbkin**4*mcMS**4 - 31*mbkin**2*mcMS**6 + 5*mcMS**8)*
                     (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                         mcMS**2)))/(9*mbkin**12) + (2*q_cut**4*(24*mbkin**2 + 
                      64*mbkin**3 + 30*mcMS**2 + 80*mbkin*mcMS**2 - 12*mbkin**2*
                       mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                    (3*mbkin**12) + 2*q_cut**3*((-4*(3 + 8*mbkin)*(1 + mcMS**2/
                         mbkin**2 + mcMS**4/mbkin**4))/(3*mbkin**8) + 
                     (2*(mbkin**2*mcMS**2 + 2*mcMS**4)*(-6 - 16*mbkin + 
                        12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                      (9*mbkin**12)) + 2*q_cut**2*((-8*(3 + 8*mbkin)*(1 - 
                        (2*mcMS**2)/mbkin**2 - (2*mcMS**4)/mbkin**4 + mcMS**6/
                         mbkin**6))/(9*mbkin**6) - (2*(2*mbkin**4*mcMS**2 + 
                        4*mbkin**2*mcMS**4 - 3*mcMS**6)*(-6 - 16*mbkin + 
                        12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                      (9*mbkin**12)) + q_cut*((-4*(3 + 8*mbkin)*(-3 + (14*mcMS**2)/
                         mbkin**2 + (26*mcMS**4)/mbkin**4 + (14*mcMS**6)/
                         mbkin**6 - (3*mcMS**8)/mbkin**8))/(9*mbkin**4) + 
                     (4*(7*mbkin**6*mcMS**2 + 26*mbkin**4*mcMS**4 + 21*mbkin**2*
                         mcMS**6 - 6*mcMS**8)*(-6 - 16*mbkin + 12*mbkin**2 + 
                        9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**12)))))) - 
           12*(np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(180*mcMS**4*
                ((-1 + mcMS**2/mbkin**2)**2*(37 - (397*mcMS**2)/mbkin**2 - 
                   (2854*mcMS**4)/mbkin**4 + (18134*mcMS**6)/mbkin**6 + 
                   (75800*mcMS**8)/mbkin**8 + (75800*mcMS**10)/mbkin**10 + 
                   (18134*mcMS**12)/mbkin**12 - (2854*mcMS**14)/mbkin**14 - 
                   (397*mcMS**16)/mbkin**16 + (37*mcMS**18)/mbkin**18) - 
                 (16*(9 - (62*mcMS**2)/mbkin**2 - (662*mcMS**4)/mbkin**4 + 
                    (1486*mcMS**6)/mbkin**6 + (12121*mcMS**8)/mbkin**8 + 
                    (19576*mcMS**10)/mbkin**10 + (12121*mcMS**12)/mbkin**12 + 
                    (1486*mcMS**14)/mbkin**14 - (662*mcMS**16)/mbkin**16 - 
                    (62*mcMS**18)/mbkin**18 + (9*mcMS**20)/mbkin**20)*q_cut)/
                  mbkin**2 + (4*(35 - (35*mcMS**2)/mbkin**2 - (1604*mcMS**4)/
                     mbkin**4 - (3896*mcMS**6)/mbkin**6 - (2060*mcMS**8)/
                     mbkin**8 - (2060*mcMS**10)/mbkin**10 - (3896*mcMS**12)/
                     mbkin**12 - (1604*mcMS**14)/mbkin**14 - (35*mcMS**16)/
                     mbkin**16 + (35*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
                 (16*(9 - (26*mcMS**2)/mbkin**2 - (737*mcMS**4)/mbkin**4 - 
                    (2164*mcMS**6)/mbkin**6 - (2732*mcMS**8)/mbkin**8 - 
                    (2164*mcMS**10)/mbkin**10 - (737*mcMS**12)/mbkin**12 - 
                    (26*mcMS**14)/mbkin**14 + (9*mcMS**16)/mbkin**16)*q_cut**3)/
                  mbkin**6 - (2*(183 + (203*mcMS**2)/mbkin**2 - (5437*mcMS**4)/
                     mbkin**4 - (19201*mcMS**6)/mbkin**6 - (19201*mcMS**8)/
                     mbkin**8 - (5437*mcMS**10)/mbkin**10 + (203*mcMS**12)/
                     mbkin**12 + (183*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
                 (8*(21 + (14*mcMS**2)/mbkin**2 - (277*mcMS**4)/mbkin**4 - 
                    (668*mcMS**6)/mbkin**6 - (277*mcMS**8)/mbkin**8 + 
                    (14*mcMS**10)/mbkin**10 + (21*mcMS**12)/mbkin**12)*q_cut**5)/
                  mbkin**10 + (8*(21 + (98*mcMS**2)/mbkin**2 + (227*mcMS**4)/
                     mbkin**4 + (227*mcMS**6)/mbkin**6 + (98*mcMS**8)/mbkin**8 + 
                    (21*mcMS**10)/mbkin**10)*q_cut**6)/mbkin**12 - 
                 (16*(15 + (34*mcMS**2)/mbkin**2 + (58*mcMS**4)/mbkin**4 + 
                    (34*mcMS**6)/mbkin**6 + (15*mcMS**8)/mbkin**8)*q_cut**7)/
                  mbkin**14 + ((81 + (37*mcMS**2)/mbkin**2 + (37*mcMS**4)/
                     mbkin**4 + (81*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
                 (56*(mbkin**2 + mcMS**2)**2*q_cut**9)/mbkin**22 - 
                 (60*(mbkin**2 + mcMS**2)*q_cut**10)/mbkin**22 + (16*q_cut**11)/
                  mbkin**22)*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
                 (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 
                    12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                 (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                     mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                     np.log(mu0**2/mcMS**2)))/(9*mbkin**6)) + 
               ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
                 q_cut**2/mbkin**4)*((-64*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*
                      (65 - (810*mcMS**2)/mbkin**2 - (516*mcMS**4)/mbkin**4 + 
                       (110224*mcMS**6)/mbkin**6 - (410120*mcMS**8)/mbkin**8 - 
                       (2939462*mcMS**10)/mbkin**10 - (3104774*mcMS**12)/
                        mbkin**12 - (475058*mcMS**14)/mbkin**14 + (163447*
                         mcMS**16)/mbkin**16 - (52384*mcMS**18)/mbkin**18 - 
                       (4742*mcMS**20)/mbkin**20 + (850*mcMS**22)/mbkin**22)) + 
                    (2*(135 - (1420*mcMS**2)/mbkin**2 - (3196*mcMS**4)/mbkin**4 + 
                       (162656*mcMS**6)/mbkin**6 - (138305*mcMS**8)/mbkin**8 - 
                       (3423984*mcMS**10)/mbkin**10 - (6202792*mcMS**12)/
                        mbkin**12 - (3867608*mcMS**14)/mbkin**14 - 
                       (95475*mcMS**16)/mbkin**16 + (226796*mcMS**18)/mbkin**18 - 
                       (77740*mcMS**20)/mbkin**20 - (7352*mcMS**22)/mbkin**22 + 
                       (1725*mcMS**24)/mbkin**24)*q_cut)/mbkin**2 - 
                    (4*(70 - (455*mcMS**2)/mbkin**2 - (3470*mcMS**4)/mbkin**4 + 
                       (37665*mcMS**6)/mbkin**6 + (138922*mcMS**8)/mbkin**8 + 
                       (83452*mcMS**10)/mbkin**10 + (121324*mcMS**12)/mbkin**12 + 
                       (190868*mcMS**14)/mbkin**14 + (16424*mcMS**16)/mbkin**16 - 
                       (25445*mcMS**18)/mbkin**18 - (790*mcMS**20)/mbkin**20 + 
                       (875*mcMS**22)/mbkin**22)*q_cut**2)/mbkin**4 - 
                    (2*(135 - (880*mcMS**2)/mbkin**2 - (7621*mcMS**4)/mbkin**4 + 
                       (133182*mcMS**6)/mbkin**6 + (543892*mcMS**8)/mbkin**8 + 
                       (658370*mcMS**10)/mbkin**10 + (525528*mcMS**12)/mbkin**
                         12 + (55122*mcMS**14)/mbkin**14 - (76843*mcMS**16)/
                        mbkin**16 - (2082*mcMS**18)/mbkin**18 + (1725*mcMS**20)/
                        mbkin**20)*q_cut**3)/mbkin**6 + (2*(330 - (1015*mcMS**2)/
                        mbkin**2 - (21973*mcMS**4)/mbkin**4 + (116514*mcMS**6)/
                        mbkin**6 + (621711*mcMS**8)/mbkin**8 + (591448*mcMS**10)/
                        mbkin**10 - (14853*mcMS**12)/mbkin**12 - (96558*mcMS**14)/
                        mbkin**14 + (6769*mcMS**16)/mbkin**16 + (4275*mcMS**18)/
                        mbkin**18)*q_cut**4)/mbkin**8 - (2*(105 - (280*mcMS**2)/
                        mbkin**2 - (8822*mcMS**4)/mbkin**4 + (25052*mcMS**6)/
                        mbkin**6 + (79356*mcMS**8)/mbkin**8 - (7796*mcMS**10)/
                        mbkin**10 - (25990*mcMS**12)/mbkin**12 + (3136*mcMS**14)/
                        mbkin**14 + (1575*mcMS**16)/mbkin**16)*q_cut**5)/mbkin**10 - 
                    (2*(105 + (455*mcMS**2)/mbkin**2 - (8521*mcMS**4)/mbkin**4 - 
                       (25699*mcMS**6)/mbkin**6 - (20961*mcMS**8)/mbkin**8 + 
                       (4285*mcMS**10)/mbkin**10 + (11193*mcMS**12)/mbkin**12 + 
                       (1575*mcMS**14)/mbkin**14)*q_cut**6)/mbkin**12 + 
                    (10*(3 + (88*mcMS**2)/mbkin**2 - (1667*mcMS**4)/mbkin**4 - 
                       (3298*mcMS**6)/mbkin**6 + (671*mcMS**8)/mbkin**8 + 
                       (2090*mcMS**10)/mbkin**10 + (225*mcMS**12)/mbkin**12)*
                      q_cut**7)/mbkin**14 + ((45 - (310*mcMS**2)/mbkin**2 + 
                       (4303*mcMS**4)/mbkin**4 + (694*mcMS**6)/mbkin**6 - 
                       (5204*mcMS**8)/mbkin**8 - (300*mcMS**10)/mbkin**10)*q_cut**8)/
                     mbkin**16 + (4*(35 + (70*mcMS**2)/mbkin**2 + (94*mcMS**4)/
                        mbkin**4 + (218*mcMS**6)/mbkin**6 + (175*mcMS**8)/
                        mbkin**8)*q_cut**9)/mbkin**18 - (30*(5 + (5*mcMS**2)/
                        mbkin**2 + (17*mcMS**4)/mbkin**4 + (25*mcMS**6)/mbkin**6)*
                      q_cut**10)/mbkin**20 + (40*(mbkin**4 + 5*mcMS**4)*q_cut**11)/
                     mbkin**26))/3 + 180*(mcMS**4*((-1 + mcMS**2/mbkin**2)**2*
                      (37 - (397*mcMS**2)/mbkin**2 - (2854*mcMS**4)/mbkin**4 + 
                       (18134*mcMS**6)/mbkin**6 + (75800*mcMS**8)/mbkin**8 + 
                       (75800*mcMS**10)/mbkin**10 + (18134*mcMS**12)/mbkin**12 - 
                       (2854*mcMS**14)/mbkin**14 - (397*mcMS**16)/mbkin**16 + 
                       (37*mcMS**18)/mbkin**18) - (16*(9 - (62*mcMS**2)/
                         mbkin**2 - (662*mcMS**4)/mbkin**4 + (1486*mcMS**6)/
                         mbkin**6 + (12121*mcMS**8)/mbkin**8 + (19576*mcMS**10)/
                         mbkin**10 + (12121*mcMS**12)/mbkin**12 + (1486*mcMS**14)/
                         mbkin**14 - (662*mcMS**16)/mbkin**16 - (62*mcMS**18)/
                         mbkin**18 + (9*mcMS**20)/mbkin**20)*q_cut)/mbkin**2 + 
                     (4*(35 - (35*mcMS**2)/mbkin**2 - (1604*mcMS**4)/mbkin**4 - 
                        (3896*mcMS**6)/mbkin**6 - (2060*mcMS**8)/mbkin**8 - 
                        (2060*mcMS**10)/mbkin**10 - (3896*mcMS**12)/mbkin**12 - 
                        (1604*mcMS**14)/mbkin**14 - (35*mcMS**16)/mbkin**16 + 
                        (35*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
                     (16*(9 - (26*mcMS**2)/mbkin**2 - (737*mcMS**4)/mbkin**4 - 
                        (2164*mcMS**6)/mbkin**6 - (2732*mcMS**8)/mbkin**8 - 
                        (2164*mcMS**10)/mbkin**10 - (737*mcMS**12)/mbkin**12 - 
                        (26*mcMS**14)/mbkin**14 + (9*mcMS**16)/mbkin**16)*q_cut**3)/
                      mbkin**6 - (2*(183 + (203*mcMS**2)/mbkin**2 - 
                        (5437*mcMS**4)/mbkin**4 - (19201*mcMS**6)/mbkin**6 - 
                        (19201*mcMS**8)/mbkin**8 - (5437*mcMS**10)/mbkin**10 + 
                        (203*mcMS**12)/mbkin**12 + (183*mcMS**14)/mbkin**14)*
                       q_cut**4)/mbkin**8 + (8*(21 + (14*mcMS**2)/mbkin**2 - 
                        (277*mcMS**4)/mbkin**4 - (668*mcMS**6)/mbkin**6 - 
                        (277*mcMS**8)/mbkin**8 + (14*mcMS**10)/mbkin**10 + 
                        (21*mcMS**12)/mbkin**12)*q_cut**5)/mbkin**10 + 
                     (8*(21 + (98*mcMS**2)/mbkin**2 + (227*mcMS**4)/mbkin**4 + 
                        (227*mcMS**6)/mbkin**6 + (98*mcMS**8)/mbkin**8 + 
                        (21*mcMS**10)/mbkin**10)*q_cut**6)/mbkin**12 - 
                     (16*(15 + (34*mcMS**2)/mbkin**2 + (58*mcMS**4)/mbkin**4 + 
                        (34*mcMS**6)/mbkin**6 + (15*mcMS**8)/mbkin**8)*q_cut**7)/
                      mbkin**14 + ((81 + (37*mcMS**2)/mbkin**2 + (37*mcMS**4)/
                         mbkin**4 + (81*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
                     (56*(mbkin**2 + mcMS**2)**2*q_cut**9)/mbkin**22 - 
                     (60*(mbkin**2 + mcMS**2)*q_cut**10)/mbkin**22 + (16*q_cut**11)/
                      mbkin**22)*(16/3 + 4*np.log(mu0**2/mcMS**2)) + mcMS**4*
                    ((-704*(3 + 8*mbkin)*q_cut**11)/(9*mbkin**24) - 
                     (2*(mbkin**2 - mcMS**2)*(471*mbkin**18*mcMS**2 + 4517*
                         mbkin**16*mcMS**4 - 65818*mbkin**14*mcMS**6 - 212530*
                         mbkin**12*mcMS**8 + 75800*mbkin**10*mcMS**10 + 421796*
                         mbkin**8*mcMS**12 + 165050*mbkin**6*mcMS**14 - 22510*
                         mbkin**4*mcMS**16 - 4303*mbkin**2*mcMS**18 + 407*
                         mcMS**20)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                         np.log(mu0**2/mcMS**2)))/(9*mbkin**24) - (224*(mbkin**2 + 
                        mcMS**2)*q_cut**9*(27*mbkin**2 + 72*mbkin**3 + 33*mcMS**2 + 
                        88*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*
                         mcMS**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**24) + 
                     (40*q_cut**10*(60*mbkin**2 + 160*mbkin**3 + 66*mcMS**2 + 
                        176*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*
                         mcMS**2*np.log(mu0**2/mcMS**2)))/(3*mbkin**24) + 
                     q_cut**8*((-32*(3 + 8*mbkin)*(81 + (37*mcMS**2)/mbkin**2 + 
                          (37*mcMS**4)/mbkin**4 + (81*mcMS**6)/mbkin**6))/
                        (9*mbkin**18) + (2*(37*mbkin**4*mcMS**2 + 74*mbkin**2*
                          mcMS**4 + 243*mcMS**6)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**24)) - 
                     16*q_cut**7*((-28*(3 + 8*mbkin)*(15 + (34*mcMS**2)/mbkin**2 + 
                          (58*mcMS**4)/mbkin**4 + (34*mcMS**6)/mbkin**6 + 
                          (15*mcMS**8)/mbkin**8))/(9*mbkin**16) + 
                       (4*(17*mbkin**6*mcMS**2 + 58*mbkin**4*mcMS**4 + 51*mbkin**2*
                          mcMS**6 + 30*mcMS**8)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**24)) + 
                     8*q_cut**6*((-8*(3 + 8*mbkin)*(21 + (98*mcMS**2)/mbkin**2 + 
                          (227*mcMS**4)/mbkin**4 + (227*mcMS**6)/mbkin**6 + 
                          (98*mcMS**8)/mbkin**8 + (21*mcMS**10)/mbkin**10))/
                        (3*mbkin**14) + (2*(98*mbkin**8*mcMS**2 + 454*mbkin**6*
                          mcMS**4 + 681*mbkin**4*mcMS**6 + 392*mbkin**2*mcMS**8 + 
                          105*mcMS**10)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**24)) + 
                     8*q_cut**5*((-20*(3 + 8*mbkin)*(21 + (14*mcMS**2)/mbkin**2 - 
                          (277*mcMS**4)/mbkin**4 - (668*mcMS**6)/mbkin**6 - 
                          (277*mcMS**8)/mbkin**8 + (14*mcMS**10)/mbkin**10 + 
                          (21*mcMS**12)/mbkin**12))/(9*mbkin**12) + 
                       (4*(7*mbkin**10*mcMS**2 - 277*mbkin**8*mcMS**4 - 1002*
                          mbkin**6*mcMS**6 - 554*mbkin**4*mcMS**8 + 35*mbkin**2*
                          mcMS**10 + 63*mcMS**12)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**24)) - 
                     2*q_cut**4*((-16*(3 + 8*mbkin)*(183 + (203*mcMS**2)/mbkin**2 - 
                          (5437*mcMS**4)/mbkin**4 - (19201*mcMS**6)/mbkin**6 - 
                          (19201*mcMS**8)/mbkin**8 - (5437*mcMS**10)/mbkin**10 + 
                          (203*mcMS**12)/mbkin**12 + (183*mcMS**14)/mbkin**14))/
                        (9*mbkin**10) + (2*(203*mbkin**12*mcMS**2 - 10874*
                          mbkin**10*mcMS**4 - 57603*mbkin**8*mcMS**6 - 76804*
                          mbkin**6*mcMS**8 - 27185*mbkin**4*mcMS**10 + 1218*
                          mbkin**2*mcMS**12 + 1281*mcMS**14)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                        (9*mbkin**24)) + 16*q_cut**3*((-4*(3 + 8*mbkin)*(9 - 
                          (26*mcMS**2)/mbkin**2 - (737*mcMS**4)/mbkin**4 - 
                          (2164*mcMS**6)/mbkin**6 - (2732*mcMS**8)/mbkin**8 - 
                          (2164*mcMS**10)/mbkin**10 - (737*mcMS**12)/mbkin**12 - 
                          (26*mcMS**14)/mbkin**14 + (9*mcMS**16)/mbkin**16))/
                        (3*mbkin**8) - (4*(13*mbkin**14*mcMS**2 + 737*mbkin**12*
                          mcMS**4 + 3246*mbkin**10*mcMS**6 + 5464*mbkin**8*
                          mcMS**8 + 5410*mbkin**6*mcMS**10 + 2211*mbkin**4*
                          mcMS**12 + 91*mbkin**2*mcMS**14 - 36*mcMS**16)*
                         (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**24)) + 4*q_cut**2*((-8*(3 + 8*mbkin)*
                         (35 - (35*mcMS**2)/mbkin**2 - (1604*mcMS**4)/mbkin**4 - 
                          (3896*mcMS**6)/mbkin**6 - (2060*mcMS**8)/mbkin**8 - 
                          (2060*mcMS**10)/mbkin**10 - (3896*mcMS**12)/mbkin**12 - 
                          (1604*mcMS**14)/mbkin**14 - (35*mcMS**16)/mbkin**16 + 
                          (35*mcMS**18)/mbkin**18))/(9*mbkin**6) - 
                       (2*(35*mbkin**16*mcMS**2 + 3208*mbkin**14*mcMS**4 + 
                          11688*mbkin**12*mcMS**6 + 8240*mbkin**10*mcMS**8 + 
                          10300*mbkin**8*mcMS**10 + 23376*mbkin**6*mcMS**12 + 
                          11228*mbkin**4*mcMS**14 + 280*mbkin**2*mcMS**16 - 
                          315*mcMS**18)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**24)) - 
                     16*q_cut*((-4*(3 + 8*mbkin)*(9 - (62*mcMS**2)/mbkin**2 - 
                          (662*mcMS**4)/mbkin**4 + (1486*mcMS**6)/mbkin**6 + 
                          (12121*mcMS**8)/mbkin**8 + (19576*mcMS**10)/mbkin**10 + 
                          (12121*mcMS**12)/mbkin**12 + (1486*mcMS**14)/mbkin**
                          14 - (662*mcMS**16)/mbkin**16 - (62*mcMS**18)/mbkin**
                          18 + (9*mcMS**20)/mbkin**20))/(9*mbkin**4) - 
                       (4*(31*mbkin**18*mcMS**2 + 662*mbkin**16*mcMS**4 - 
                          2229*mbkin**14*mcMS**6 - 24242*mbkin**12*mcMS**8 - 
                          48940*mbkin**10*mcMS**10 - 36363*mbkin**8*mcMS**12 - 
                          5201*mbkin**6*mcMS**14 + 2648*mbkin**4*mcMS**16 + 
                          279*mbkin**2*mcMS**18 - 45*mcMS**20)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                        (9*mbkin**24))))))*np.log((mbkin**2 + mcMS**2 - q_cut - 
                 mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                     2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                      mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                    mbkin**4))) + (180*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + 
                 mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)**2*(37*mbkin**18 - 
                397*mbkin**16*mcMS**2 - 2854*mbkin**14*mcMS**4 + 18134*mbkin**12*
                 mcMS**6 + 75800*mbkin**10*mcMS**8 + 75800*mbkin**8*mcMS**10 + 
                18134*mbkin**6*mcMS**12 - 2854*mbkin**4*mcMS**14 - 397*mbkin**2*
                 mcMS**16 + 37*mcMS**18 - 70*mbkin**16*q_cut + 132*mbkin**14*mcMS**2*
                 q_cut + 4424*mbkin**12*mcMS**4*q_cut + 15500*mbkin**10*mcMS**6*q_cut + 
                20508*mbkin**8*mcMS**8*q_cut + 15500*mbkin**6*mcMS**10*q_cut + 
                4424*mbkin**4*mcMS**12*q_cut + 132*mbkin**2*mcMS**14*q_cut - 
                70*mcMS**16*q_cut - 37*mbkin**14*q_cut**2 + 307*mbkin**12*mcMS**2*q_cut**2 + 
                6201*mbkin**10*mcMS**4*q_cut**2 + 18225*mbkin**8*mcMS**6*q_cut**2 + 
                18225*mbkin**6*mcMS**8*q_cut**2 + 6201*mbkin**4*mcMS**10*q_cut**2 + 
                307*mbkin**2*mcMS**12*q_cut**2 - 37*mcMS**14*q_cut**2 + 140*mbkin**12*
                 q_cut**3 + 272*mbkin**10*mcMS**2*q_cut**3 - 2796*mbkin**8*mcMS**4*q_cut**3 - 
                7136*mbkin**6*mcMS**6*q_cut**3 - 2796*mbkin**4*mcMS**8*q_cut**3 + 
                272*mbkin**2*mcMS**10*q_cut**3 + 140*mcMS**12*q_cut**3 - 49*mbkin**10*
                 q_cut**4 + 13*mbkin**8*mcMS**2*q_cut**4 - 300*mbkin**6*mcMS**4*q_cut**4 - 
                300*mbkin**4*mcMS**6*q_cut**4 + 13*mbkin**2*mcMS**8*q_cut**4 - 
                49*mcMS**10*q_cut**4 - 70*mbkin**8*q_cut**5 - 372*mbkin**6*mcMS**2*q_cut**5 - 
                668*mbkin**4*mcMS**4*q_cut**5 - 372*mbkin**2*mcMS**6*q_cut**5 - 
                70*mcMS**8*q_cut**5 + 77*mbkin**6*q_cut**6 + 41*mbkin**4*mcMS**2*q_cut**6 + 
                41*mbkin**2*mcMS**4*q_cut**6 + 77*mcMS**6*q_cut**6 - 16*mbkin**4*q_cut**7 + 
                32*mbkin**2*mcMS**2*q_cut**7 - 16*mcMS**4*q_cut**7 - 28*mbkin**2*q_cut**8 - 
                28*mcMS**2*q_cut**8 + 16*q_cut**9)*((-8*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                      mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                    mbkin**4)*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                        mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                      mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                        mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                      mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                        mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                      mbkin**4) + 6*mbkin**2*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                        mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                      mbkin**4) + 16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                        mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                      mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                        mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                      mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                       2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                        q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*mcMS**2*
                    q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 9*mbkin**6*mcMS**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 
                   9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                       mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                    np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**2*q_cut*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                       2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2)))/
                 (9*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)*(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - 
                   mcMS**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                       mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**
                       4))) + (np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                     2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                  ((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                     (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                      9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                   (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                       mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                       np.log(mu0**2/mcMS**2)))/(9*mbkin**6))*np.log((mbkin**2 + 
                     mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                        mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/
                 (2*((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/
                    mbkin**4 + q_cut**2/mbkin**4))))/mbkin**26) - 
           144*((mcMS**4*(-180*mcMS**4*((-1 + mcMS**2/mbkin**2)**2*(71 - 
                    (174*mcMS**2)/mbkin**2 - (3723*mcMS**4)/mbkin**4 - 
                    (7468*mcMS**6)/mbkin**6 - (3723*mcMS**8)/mbkin**8 - 
                    (174*mcMS**10)/mbkin**10 + (71*mcMS**12)/mbkin**12) + 
                  ((-211 - (37*mcMS**2)/mbkin**2 + (7677*mcMS**4)/mbkin**4 + 
                     (22811*mcMS**6)/mbkin**6 + (22811*mcMS**8)/mbkin**8 + 
                     (7677*mcMS**10)/mbkin**10 - (37*mcMS**12)/mbkin**12 - 
                     (211*mcMS**14)/mbkin**14)*q_cut)/mbkin**2 + 
                  (2*(70 + (209*mcMS**2)/mbkin**2 - (668*mcMS**4)/mbkin**4 - 
                     (1742*mcMS**6)/mbkin**6 - (668*mcMS**8)/mbkin**8 + 
                     (209*mcMS**10)/mbkin**10 + (70*mcMS**12)/mbkin**12)*q_cut**2)/
                   mbkin**4 + (2*(70 + (419*mcMS**2)/mbkin**2 + (729*mcMS**4)/
                      mbkin**4 + (729*mcMS**6)/mbkin**6 + (419*mcMS**8)/mbkin**8 + 
                     (70*mcMS**10)/mbkin**10)*q_cut**3)/mbkin**6 - 
                  ((217 + (1234*mcMS**2)/mbkin**2 + (2162*mcMS**4)/mbkin**4 + 
                     (1234*mcMS**6)/mbkin**6 + (217*mcMS**8)/mbkin**8)*q_cut**4)/
                   mbkin**8 + (11*(7 + (31*mcMS**2)/mbkin**2 + (31*mcMS**4)/
                      mbkin**4 + (7*mcMS**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
                  (2*(7 + (6*mcMS**2)/mbkin**2 + (7*mcMS**4)/mbkin**4)*q_cut**6)/
                   mbkin**12 - (22*(mbkin**2 + mcMS**2)*q_cut**7)/mbkin**16 + 
                  (8*q_cut**8)/mbkin**16)*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
                  (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 
                     12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                  (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                      mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                      np.log(mu0**2/mcMS**2)))/(9*mbkin**6)) + 
                ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/
                   mbkin**4 + q_cut**2/mbkin**4)*((64*mbkin*
                    (-((-1 + mcMS**2/mbkin**2)**2*(-40 - (530*mcMS**2)/mbkin**2 + 
                        (1237*mcMS**4)/mbkin**4 + (117101*mcMS**6)/mbkin**6 + 
                        (301726*mcMS**8)/mbkin**8 + (144226*mcMS**10)/mbkin**10 - 
                        (5749*mcMS**12)/mbkin**12 + (619*mcMS**14)/mbkin**14 + 
                        (850*mcMS**16)/mbkin**16)) + ((-110 - (2180*mcMS**2)/
                         mbkin**2 - (8043*mcMS**4)/mbkin**4 + (231598*mcMS**6)/
                         mbkin**6 + (844217*mcMS**8)/mbkin**8 + (919012*mcMS**10)/
                         mbkin**10 + (277427*mcMS**12)/mbkin**12 - 
                        (28230*mcMS**14)/mbkin**14 + (1469*mcMS**16)/mbkin**16 + 
                        (2600*mcMS**18)/mbkin**18)*q_cut)/mbkin**2 - 
                     (2*(-35 - (870*mcMS**2)/mbkin**2 - (7720*mcMS**4)/mbkin**4 + 
                        (19403*mcMS**6)/mbkin**6 + (68795*mcMS**8)/mbkin**8 + 
                        (21911*mcMS**10)/mbkin**10 - (10719*mcMS**12)/mbkin**12 + 
                        (1600*mcMS**14)/mbkin**14 + (875*mcMS**16)/mbkin**16)*
                       q_cut**2)/mbkin**4 + (2*(35 + (975*mcMS**2)/mbkin**2 + 
                        (11590*mcMS**4)/mbkin**4 + (21792*mcMS**6)/mbkin**6 + 
                        (19921*mcMS**8)/mbkin**8 + (9944*mcMS**10)/mbkin**10 - 
                        (2650*mcMS**12)/mbkin**12 - (875*mcMS**14)/mbkin**14)*
                       q_cut**3)/mbkin**6 + ((-140 - (2810*mcMS**2)/mbkin**2 - 
                        (35835*mcMS**4)/mbkin**4 - (69801*mcMS**6)/mbkin**6 - 
                        (30887*mcMS**8)/mbkin**8 + (8455*mcMS**10)/mbkin**10 + 
                        (2450*mcMS**12)/mbkin**12)*q_cut**4)/mbkin**8 + 
                     ((70 + (900*mcMS**2)/mbkin**2 + (10637*mcMS**4)/mbkin**4 + 
                        (9736*mcMS**6)/mbkin**6 - (2311*mcMS**8)/mbkin**8 - 
                        (700*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
                     ((70 + (60*mcMS**2)/mbkin**2 + (308*mcMS**4)/mbkin**4 + 
                        (76*mcMS**6)/mbkin**6 + (350*mcMS**8)/mbkin**8)*q_cut**6)/
                      mbkin**12 - (10*(11 + (11*mcMS**2)/mbkin**2 + (27*mcMS**4)/
                         mbkin**4 + (55*mcMS**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
                     (40*(mbkin**4 + 5*mcMS**4)*q_cut**8)/mbkin**20))/3 - 
                  180*(mcMS**4*((-1 + mcMS**2/mbkin**2)**2*(71 - (174*mcMS**2)/
                         mbkin**2 - (3723*mcMS**4)/mbkin**4 - (7468*mcMS**6)/
                         mbkin**6 - (3723*mcMS**8)/mbkin**8 - (174*mcMS**10)/
                         mbkin**10 + (71*mcMS**12)/mbkin**12) + 
                      ((-211 - (37*mcMS**2)/mbkin**2 + (7677*mcMS**4)/mbkin**4 + 
                         (22811*mcMS**6)/mbkin**6 + (22811*mcMS**8)/mbkin**8 + 
                         (7677*mcMS**10)/mbkin**10 - (37*mcMS**12)/mbkin**12 - 
                         (211*mcMS**14)/mbkin**14)*q_cut)/mbkin**2 + 
                      (2*(70 + (209*mcMS**2)/mbkin**2 - (668*mcMS**4)/mbkin**4 - 
                         (1742*mcMS**6)/mbkin**6 - (668*mcMS**8)/mbkin**8 + 
                         (209*mcMS**10)/mbkin**10 + (70*mcMS**12)/mbkin**12)*
                        q_cut**2)/mbkin**4 + (2*(70 + (419*mcMS**2)/mbkin**2 + 
                         (729*mcMS**4)/mbkin**4 + (729*mcMS**6)/mbkin**6 + 
                         (419*mcMS**8)/mbkin**8 + (70*mcMS**10)/mbkin**10)*q_cut**3)/
                       mbkin**6 - ((217 + (1234*mcMS**2)/mbkin**2 + 
                         (2162*mcMS**4)/mbkin**4 + (1234*mcMS**6)/mbkin**6 + 
                         (217*mcMS**8)/mbkin**8)*q_cut**4)/mbkin**8 + 
                      (11*(7 + (31*mcMS**2)/mbkin**2 + (31*mcMS**4)/mbkin**4 + 
                         (7*mcMS**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
                      (2*(7 + (6*mcMS**2)/mbkin**2 + (7*mcMS**4)/mbkin**4)*q_cut**6)/
                       mbkin**12 - (22*(mbkin**2 + mcMS**2)*q_cut**7)/mbkin**16 + 
                      (8*q_cut**8)/mbkin**16)*(16/3 + 4*np.log(mu0**2/mcMS**2)) + 
                    mcMS**4*((-256*(3 + 8*mbkin)*q_cut**8)/(9*mbkin**18) - 
                      (8*(mbkin**2 - mcMS**2)*(79*mbkin**12*mcMS**2 + 1731*
                          mbkin**10*mcMS**4 + 1878*mbkin**8*mcMS**6 - 5612*
                          mbkin**6*mcMS**8 - 5367*mbkin**4*mcMS**10 - 411*mbkin**2*
                          mcMS**12 + 142*mcMS**14)*(-6 - 16*mbkin + 
                         12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                       (9*mbkin**18) + (44*q_cut**7*(42*mbkin**2 + 112*mbkin**3 + 
                         48*mcMS**2 + 128*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 
                         9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**18) + 
                      2*q_cut**6*((-8*(3 + 8*mbkin)*(7 + (6*mcMS**2)/mbkin**2 + 
                          (7*mcMS**4)/mbkin**4))/(3*mbkin**14) + 
                        (4*(3*mbkin**2*mcMS**2 + 7*mcMS**4)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                         (9*mbkin**18)) + 11*q_cut**5*((-20*(3 + 8*mbkin)*
                          (7 + (31*mcMS**2)/mbkin**2 + (31*mcMS**4)/mbkin**4 + 
                          (7*mcMS**6)/mbkin**6))/(9*mbkin**12) + 
                        (2*(31*mbkin**4*mcMS**2 + 62*mbkin**2*mcMS**4 + 
                          21*mcMS**6)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**18)) + q_cut**4*
                       ((16*(3 + 8*mbkin)*(217 + (1234*mcMS**2)/mbkin**2 + 
                          (2162*mcMS**4)/mbkin**4 + (1234*mcMS**6)/mbkin**6 + 
                          (217*mcMS**8)/mbkin**8))/(9*mbkin**10) - 
                        (4*(617*mbkin**6*mcMS**2 + 2162*mbkin**4*mcMS**4 + 
                          1851*mbkin**2*mcMS**6 + 434*mcMS**8)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                         (9*mbkin**18)) + 2*q_cut**3*((-4*(3 + 8*mbkin)*(70 + 
                          (419*mcMS**2)/mbkin**2 + (729*mcMS**4)/mbkin**4 + 
                          (729*mcMS**6)/mbkin**6 + (419*mcMS**8)/mbkin**8 + 
                          (70*mcMS**10)/mbkin**10))/(3*mbkin**8) + 
                        (2*(419*mbkin**8*mcMS**2 + 1458*mbkin**6*mcMS**4 + 
                          2187*mbkin**4*mcMS**6 + 1676*mbkin**2*mcMS**8 + 
                          350*mcMS**10)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**18)) + 
                      2*q_cut**2*((-8*(3 + 8*mbkin)*(70 + (209*mcMS**2)/mbkin**2 - 
                          (668*mcMS**4)/mbkin**4 - (1742*mcMS**6)/mbkin**6 - 
                          (668*mcMS**8)/mbkin**8 + (209*mcMS**10)/mbkin**10 + 
                          (70*mcMS**12)/mbkin**12))/(9*mbkin**6) + 
                        (2*(209*mbkin**10*mcMS**2 - 1336*mbkin**8*mcMS**4 - 
                          5226*mbkin**6*mcMS**6 - 2672*mbkin**4*mcMS**8 + 
                          1045*mbkin**2*mcMS**10 + 420*mcMS**12)*(-6 - 
                          16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**
                          2)))/(9*mbkin**18)) + q_cut*((-4*(3 + 8*mbkin)*(-211 - 
                          (37*mcMS**2)/mbkin**2 + (7677*mcMS**4)/mbkin**4 + 
                          (22811*mcMS**6)/mbkin**6 + (22811*mcMS**8)/mbkin**8 + 
                          (7677*mcMS**10)/mbkin**10 - (37*mcMS**12)/mbkin**12 - 
                          (211*mcMS**14)/mbkin**14))/(9*mbkin**4) - 
                        (2*(37*mbkin**12*mcMS**2 - 15354*mbkin**10*mcMS**4 - 
                          68433*mbkin**8*mcMS**6 - 91244*mbkin**6*mcMS**8 - 
                          38385*mbkin**4*mcMS**10 + 222*mbkin**2*mcMS**12 + 
                          1477*mcMS**14)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**18))))))*
               np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                        mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                      mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 - 
             (180*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                  q_cut - 2*mcMS**2*q_cut + q_cut**2)**2*(71*mbkin**12 - 174*mbkin**10*
                 mcMS**2 - 3723*mbkin**8*mcMS**4 - 7468*mbkin**6*mcMS**6 - 
                3723*mbkin**4*mcMS**8 - 174*mbkin**2*mcMS**10 + 71*mcMS**12 - 
                69*mbkin**10*q_cut - 381*mbkin**8*mcMS**2*q_cut - 810*mbkin**6*mcMS**4*
                 q_cut - 810*mbkin**4*mcMS**6*q_cut - 381*mbkin**2*mcMS**8*q_cut - 
                69*mcMS**10*q_cut - 69*mbkin**8*q_cut**2 - 446*mbkin**6*mcMS**2*q_cut**2 - 
                818*mbkin**4*mcMS**4*q_cut**2 - 446*mbkin**2*mcMS**6*q_cut**2 - 
                69*mcMS**8*q_cut**2 + 71*mbkin**6*q_cut**3 + 331*mbkin**4*mcMS**2*q_cut**3 + 
                331*mbkin**2*mcMS**4*q_cut**3 + 71*mcMS**6*q_cut**3 - 6*mbkin**4*q_cut**4 + 
                4*mbkin**2*mcMS**2*q_cut**4 - 6*mcMS**4*q_cut**4 - 6*mbkin**2*q_cut**5 - 
                6*mcMS**2*q_cut**5 + 8*q_cut**6)*((mcMS**4*(16/3 + 4*np.log(mu0**2/mcMS**2))*
                  np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                      mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/
                 mbkin**4 + mcMS**4*((-16*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + 
                       (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                         2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 12*mbkin**6*mcMS**2*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 6*mbkin**2*
                      mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                     16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                        mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*
                      mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 
                     12*mbkin**4*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                        mbkin**4) + 9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                         q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**4*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/
                        mcMS**2) - 9*mbkin**4*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2))*
                    np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                       mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/
                   (9*mbkin**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                     2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)*(mbkin**2 + mcMS**2 - 
                     q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))*
                    (-mbkin**2 - mcMS**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))) - (8*(3 + 8*mbkin)*
                    np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                        mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/
                   (9*mbkin**6))))/mbkin**20) - 
           60480*((mcMS**8*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(-64*mbkin - 
                (1664*mcMS**2)/(3*mbkin) - (11200*mcMS**4)/(3*mbkin**3) - 
                (4480*mcMS**6)/mbkin**5 - (640*mcMS**8)/mbkin**7 + 240*mcMS**4*
                 (4 + 3*np.log(mu0**2/mcMS**2)) + (800*mcMS**8*(-3 - 8*mbkin + 
                   12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/mbkin**6 + 
                (120*mcMS**10*(-6 - 16*mbkin + 20*mbkin**2 + 15*mbkin**2*
                    np.log(mu0**2/mcMS**2)))/mbkin**8 + (200*mcMS**6*(-6 - 
                   16*mbkin + 36*mbkin**2 + 27*mbkin**2*np.log(mu0**2/mcMS**2)))/
                 mbkin**4)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                       2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                       2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/
              mbkin**8 + (180*mcMS**4 + (900*mcMS**6)/mbkin**2 + (900*mcMS**8)/
                mbkin**4 + (180*mcMS**10)/mbkin**6)*((3*mcMS**8*
                 ((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                    2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*((-8*(3 + 8*mbkin)*
                    q_cut**2)/(9*mbkin**6) + (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*
                    (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                        mcMS**2)))/(9*mbkin**4) + (4*q_cut*(6*mbkin**2 + 
                     16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 12*mbkin**2*
                      mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                   (9*mbkin**6))*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                     mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                        mbkin**4)))**3)/(2*mbkin**8*((-1 + mcMS**2/mbkin**2)**2 - 
                  (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)) + 
               ((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
                ((mcMS**8*(32/3 + 8*np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - 
                       q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                          mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                          mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                        np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8 + 
                 mcMS**8*((-8*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                          q_cut**2)/mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + 
                        (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                          2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 12*mbkin**6*mcMS**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 6*mbkin**2*
                       mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                      16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                          mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                         mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*
                       mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 
                      12*mbkin**4*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                          mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                         mbkin**4) + 9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                       mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                       np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**2*q_cut*np.sqrt(0j + 
                        (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                          2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2))*
                     np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                         mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/
                    (3*mbkin**8*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                      2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)*(mbkin**2 + mcMS**2 - 
                      q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                          mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                         mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + mbkin**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                   (16*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                          np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                        (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)))**3)/(9*mbkin**10)))))))/
         (((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
             2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 7*
                mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - mbkin**2*
                q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
            (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                   (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                     2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                 mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                     2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
           3)))/2520 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

def q2moment4MS(q_cut, mbkin, mcMS, mus, mu0, api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
 res = ( 
    (mbkin**4*((18*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
            2*mcMS**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
            7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
            mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)**2*(3*mbkin**14 - 
           141*mbkin**12*mcMS**2 - 7785*mbkin**10*mcMS**4 - 33657*mbkin**8*
            mcMS**6 - 33657*mbkin**6*mcMS**8 - 7785*mbkin**4*mcMS**10 - 
           141*mbkin**2*mcMS**12 + 3*mcMS**14 + 3*mbkin**12*q_cut - 
           132*mbkin**10*mcMS**2*q_cut - 3153*mbkin**8*mcMS**4*q_cut - 
           7296*mbkin**6*mcMS**6*q_cut - 3153*mbkin**4*mcMS**8*q_cut - 
           132*mbkin**2*mcMS**10*q_cut + 3*mcMS**12*q_cut + 3*mbkin**10*q_cut**2 - 
           117*mbkin**8*mcMS**2*q_cut**2 - 1272*mbkin**6*mcMS**4*q_cut**2 - 
           1272*mbkin**4*mcMS**6*q_cut**2 - 117*mbkin**2*mcMS**8*q_cut**2 + 
           3*mcMS**10*q_cut**2 + 3*mbkin**8*q_cut**3 - 96*mbkin**6*mcMS**2*q_cut**3 - 
           408*mbkin**4*mcMS**4*q_cut**3 - 96*mbkin**2*mcMS**6*q_cut**3 + 3*mcMS**8*q_cut**3 + 
           3*mbkin**6*q_cut**4 - 69*mbkin**4*mcMS**2*q_cut**4 - 69*mbkin**2*mcMS**4*q_cut**4 + 
           3*mcMS**6*q_cut**4 - 25*mbkin**4*q_cut**5 + 20*mbkin**2*mcMS**2*q_cut**5 - 
           25*mcMS**4*q_cut**5 - 25*mbkin**2*q_cut**6 - 25*mcMS**2*q_cut**6 + 35*q_cut**7))/
         mbkin**34 - (432*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
            2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)**2*
          np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
             2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(108*mbkin**20 - 
           792*mbkin**18*mcMS**2 - 13329*mbkin**16*mcMS**4 + 40518*mbkin**14*
            mcMS**6 + 387441*mbkin**12*mcMS**8 + 668988*mbkin**10*mcMS**10 + 
           387441*mbkin**8*mcMS**12 + 40518*mbkin**6*mcMS**14 - 
           13329*mbkin**4*mcMS**16 - 792*mbkin**2*mcMS**18 + 108*mcMS**20 - 
           210*mbkin**18*q_cut - 222*mbkin**16*mcMS**2*q_cut + 15402*mbkin**14*mcMS**4*
            q_cut + 81210*mbkin**12*mcMS**6*q_cut + 153300*mbkin**10*mcMS**8*q_cut + 
           153300*mbkin**8*mcMS**10*q_cut + 81210*mbkin**6*mcMS**12*q_cut + 
           15402*mbkin**4*mcMS**14*q_cut - 222*mbkin**2*mcMS**16*q_cut - 
           210*mcMS**18*q_cut - 108*mbkin**16*q_cut**2 + 552*mbkin**14*mcMS**2*q_cut**2 + 
           22257*mbkin**12*mcMS**4*q_cut**2 + 101844*mbkin**10*mcMS**6*q_cut**2 + 
           158394*mbkin**8*mcMS**8*q_cut**2 + 101844*mbkin**6*mcMS**10*q_cut**2 + 
           22257*mbkin**4*mcMS**12*q_cut**2 + 552*mbkin**2*mcMS**14*q_cut**2 - 
           108*mcMS**16*q_cut**2 + 420*mbkin**14*q_cut**3 + 2088*mbkin**12*mcMS**2*q_cut**3 - 
           8028*mbkin**10*mcMS**4*q_cut**3 - 43584*mbkin**8*mcMS**6*q_cut**3 - 
           43584*mbkin**6*mcMS**8*q_cut**3 - 8028*mbkin**4*mcMS**10*q_cut**3 + 
           2088*mbkin**2*mcMS**12*q_cut**3 + 420*mcMS**14*q_cut**3 - 105*mbkin**12*q_cut**4 - 
           642*mbkin**10*mcMS**2*q_cut**4 - 966*mbkin**8*mcMS**4*q_cut**4 - 
           2118*mbkin**6*mcMS**6*q_cut**4 - 966*mbkin**4*mcMS**8*q_cut**4 - 
           642*mbkin**2*mcMS**10*q_cut**4 - 105*mcMS**12*q_cut**4 - 238*mbkin**10*q_cut**5 - 
           1650*mbkin**8*mcMS**2*q_cut**5 - 5522*mbkin**6*mcMS**4*q_cut**5 - 
           5522*mbkin**4*mcMS**6*q_cut**5 - 1650*mbkin**2*mcMS**8*q_cut**5 - 
           238*mcMS**10*q_cut**5 + 105*mbkin**8*q_cut**6 + 940*mbkin**6*mcMS**2*q_cut**6 + 
           1705*mbkin**4*mcMS**4*q_cut**6 + 940*mbkin**2*mcMS**6*q_cut**6 + 
           105*mcMS**8*q_cut**6 + 88*mbkin**6*q_cut**7 - 284*mbkin**4*mcMS**2*q_cut**7 - 
           284*mbkin**2*mcMS**4*q_cut**7 + 88*mcMS**6*q_cut**7 - 35*mbkin**4*q_cut**8 + 
           70*mbkin**2*mcMS**2*q_cut**8 - 35*mcMS**4*q_cut**8 - 60*mbkin**2*q_cut**9 - 
           60*mcMS**2*q_cut**9 + 35*q_cut**10)*np.log((mbkin**2 + mcMS**2 - q_cut - 
             mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                  q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
             mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                  q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**28 + 
        (2592*mcMS**8*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
            2*mcMS**2*q_cut + q_cut**2)**2*(423*mbkin**14 + 279*mbkin**12*mcMS**2 - 
           27945*mbkin**10*mcMS**4 - 97497*mbkin**8*mcMS**6 - 
           97497*mbkin**6*mcMS**8 - 27945*mbkin**4*mcMS**10 + 
           279*mbkin**2*mcMS**12 + 423*mcMS**14 - 417*mbkin**12*q_cut - 
           3492*mbkin**10*mcMS**2*q_cut - 9873*mbkin**8*mcMS**4*q_cut - 
           14016*mbkin**6*mcMS**6*q_cut - 9873*mbkin**4*mcMS**8*q_cut - 
           3492*mbkin**2*mcMS**10*q_cut - 417*mcMS**12*q_cut - 417*mbkin**10*q_cut**2 - 
           3897*mbkin**8*mcMS**2*q_cut**2 - 10932*mbkin**6*mcMS**4*q_cut**2 - 
           10932*mbkin**4*mcMS**6*q_cut**2 - 3897*mbkin**2*mcMS**8*q_cut**2 - 
           417*mcMS**10*q_cut**2 + 423*mbkin**8*q_cut**3 + 3264*mbkin**6*mcMS**2*q_cut**3 + 
           5892*mbkin**4*mcMS**4*q_cut**3 + 3264*mbkin**2*mcMS**6*q_cut**3 + 
           423*mcMS**8*q_cut**3 + 3*mbkin**6*q_cut**4 - 69*mbkin**4*mcMS**2*q_cut**4 - 
           69*mbkin**2*mcMS**4*q_cut**4 + 3*mcMS**6*q_cut**4 - 25*mbkin**4*q_cut**5 + 
           20*mbkin**2*mcMS**2*q_cut**5 - 25*mcMS**4*q_cut**5 - 25*mbkin**2*q_cut**6 - 
           25*mcMS**2*q_cut**6 + 35*q_cut**7)*np.log((mbkin**2 + mcMS**2 - q_cut - 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - 
              q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                  2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**26 - 
        (6531840*mcMS**12*(mbkin**8 + 8*mbkin**6*mcMS**2 + 15*mbkin**4*mcMS**4 + 
           8*mbkin**2*mcMS**6 + mcMS**8)*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
             2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**16))/
      (1260*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
          2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
            7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
            mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**3) + 
     (mbkin**4*(((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
          q_cut**2/mbkin**4)*(-24*mbkin**2*muG*((-1 + mcMS**2/mbkin**2)**2 - 
             (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)**2*
           (-((1 + mcMS**2/mbkin**2)**2*(-33 - (2131*mcMS**2)/mbkin**2 + 
               (30294*mcMS**4)/mbkin**4 - (78660*mcMS**6)/mbkin**6 - 
               (303336*mcMS**8)/mbkin**8 + (1430424*mcMS**10)/mbkin**10 + 
               (1908216*mcMS**12)/mbkin**12 - (417984*mcMS**14)/mbkin**14 - 
               (84555*mcMS**16)/mbkin**16 + (12471*mcMS**18)/mbkin**18 + 
               (94*mcMS**20)/mbkin**20)) + ((-42 - (3763*mcMS**2)/mbkin**2 + 
               (26558*mcMS**4)/mbkin**4 + (44403*mcMS**6)/mbkin**6 - 
               (669144*mcMS**8)/mbkin**8 - (1235460*mcMS**10)/mbkin**10 - 
               (1259364*mcMS**12)/mbkin**12 - (1346004*mcMS**14)/mbkin**14 - 
               (612774*mcMS**16)/mbkin**16 + (45903*mcMS**18)/mbkin**18 + 
               (19966*mcMS**20)/mbkin**20 + (121*mcMS**22)/mbkin**22)*q_cut)/
             mbkin**2 + ((-75 - (3521*mcMS**2)/mbkin**2 + (35220*mcMS**4)/
                mbkin**4 - (13530*mcMS**6)/mbkin**6 - (1006434*mcMS**8)/mbkin**8 - 
               (2777886*mcMS**10)/mbkin**10 - (2724486*mcMS**12)/mbkin**12 - 
               (998682*mcMS**14)/mbkin**14 - (17175*mcMS**16)/mbkin**16 + 
               (21963*mcMS**18)/mbkin**18 + (206*mcMS**20)/mbkin**20)*q_cut**2)/
             mbkin**4 + ((84 + (8007*mcMS**2)/mbkin**2 - (44916*mcMS**4)/
                mbkin**4 - (259494*mcMS**6)/mbkin**6 + (280412*mcMS**8)/mbkin**8 + 
               (873934*mcMS**10)/mbkin**10 + (232148*mcMS**12)/mbkin**12 - 
               (222734*mcMS**14)/mbkin**14 - (42208*mcMS**16)/mbkin**16 - 
               (233*mcMS**18)/mbkin**18)*q_cut**3)/mbkin**6 - 
            (2*(-21 - (375*mcMS**2)/mbkin**2 + (4350*mcMS**4)/mbkin**4 + 
               (858*mcMS**6)/mbkin**6 + (40570*mcMS**8)/mbkin**8 + (22642*
                 mcMS**10)/mbkin**10 + (7617*mcMS**12)/mbkin**12 + (3239*mcMS**14)/
                mbkin**14 + (56*mcMS**16)/mbkin**16)*q_cut**4)/mbkin**8 + 
            (2*mcMS**2*(-2439 + (10338*mcMS**2)/mbkin**2 + (103080*mcMS**4)/
                mbkin**4 + (158590*mcMS**6)/mbkin**6 + (93216*mcMS**8)/mbkin**8 + 
               (11622*mcMS**10)/mbkin**10 - (7*mcMS**12)/mbkin**12)*q_cut**5)/
             mbkin**12 - (2*(-21 + (3*mcMS**2)/mbkin**2 - (369*mcMS**4)/mbkin**4 + 
               (16671*mcMS**6)/mbkin**6 + (12001*mcMS**8)/mbkin**8 + 
               (1097*mcMS**10)/mbkin**10 + (28*mcMS**12)/mbkin**12)*q_cut**6)/
             mbkin**12 + (2*(-90 + (771*mcMS**2)/mbkin**2 - (786*mcMS**4)/
                mbkin**4 - (5817*mcMS**6)/mbkin**6 + (112*mcMS**8)/mbkin**8 + 
               (187*mcMS**10)/mbkin**10)*q_cut**7)/mbkin**14 - 
            ((-21 + (51*mcMS**2)/mbkin**2 + (453*mcMS**4)/mbkin**4 + (495*mcMS**6)/
                mbkin**6 + (2*mcMS**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
            ((-138 + (559*mcMS**2)/mbkin**2 + (1610*mcMS**4)/mbkin**4 + 
               (283*mcMS**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
            ((-63 + (247*mcMS**2)/mbkin**2 + (58*mcMS**4)/mbkin**4)*q_cut**10)/
             mbkin**20 + (35*mcMS**2*q_cut**11)/mbkin**24) - 12*muG*mupi*
           ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
             q_cut**2/mbkin**4)**2*(-((1 + mcMS**2/mbkin**2)**2*(-33 - (2131*mcMS**2)/
                mbkin**2 + (30294*mcMS**4)/mbkin**4 - (78660*mcMS**6)/mbkin**6 - 
               (303336*mcMS**8)/mbkin**8 + (1430424*mcMS**10)/mbkin**10 + 
               (1908216*mcMS**12)/mbkin**12 - (417984*mcMS**14)/mbkin**14 - 
               (84555*mcMS**16)/mbkin**16 + (12471*mcMS**18)/mbkin**18 + 
               (94*mcMS**20)/mbkin**20)) + ((-42 - (3763*mcMS**2)/mbkin**2 + 
               (26558*mcMS**4)/mbkin**4 + (44403*mcMS**6)/mbkin**6 - 
               (669144*mcMS**8)/mbkin**8 - (1235460*mcMS**10)/mbkin**10 - 
               (1259364*mcMS**12)/mbkin**12 - (1346004*mcMS**14)/mbkin**14 - 
               (612774*mcMS**16)/mbkin**16 + (45903*mcMS**18)/mbkin**18 + 
               (19966*mcMS**20)/mbkin**20 + (121*mcMS**22)/mbkin**22)*q_cut)/
             mbkin**2 + ((-75 - (3521*mcMS**2)/mbkin**2 + (35220*mcMS**4)/
                mbkin**4 - (13530*mcMS**6)/mbkin**6 - (1006434*mcMS**8)/mbkin**8 - 
               (2777886*mcMS**10)/mbkin**10 - (2724486*mcMS**12)/mbkin**12 - 
               (998682*mcMS**14)/mbkin**14 - (17175*mcMS**16)/mbkin**16 + 
               (21963*mcMS**18)/mbkin**18 + (206*mcMS**20)/mbkin**20)*q_cut**2)/
             mbkin**4 + ((84 + (8007*mcMS**2)/mbkin**2 - (44916*mcMS**4)/
                mbkin**4 - (259494*mcMS**6)/mbkin**6 + (280412*mcMS**8)/mbkin**8 + 
               (873934*mcMS**10)/mbkin**10 + (232148*mcMS**12)/mbkin**12 - 
               (222734*mcMS**14)/mbkin**14 - (42208*mcMS**16)/mbkin**16 - 
               (233*mcMS**18)/mbkin**18)*q_cut**3)/mbkin**6 - 
            (2*(-21 - (375*mcMS**2)/mbkin**2 + (4350*mcMS**4)/mbkin**4 + 
               (858*mcMS**6)/mbkin**6 + (40570*mcMS**8)/mbkin**8 + (22642*
                 mcMS**10)/mbkin**10 + (7617*mcMS**12)/mbkin**12 + (3239*mcMS**14)/
                mbkin**14 + (56*mcMS**16)/mbkin**16)*q_cut**4)/mbkin**8 + 
            (2*mcMS**2*(-2439 + (10338*mcMS**2)/mbkin**2 + (103080*mcMS**4)/
                mbkin**4 + (158590*mcMS**6)/mbkin**6 + (93216*mcMS**8)/mbkin**8 + 
               (11622*mcMS**10)/mbkin**10 - (7*mcMS**12)/mbkin**12)*q_cut**5)/
             mbkin**12 - (2*(-21 + (3*mcMS**2)/mbkin**2 - (369*mcMS**4)/mbkin**4 + 
               (16671*mcMS**6)/mbkin**6 + (12001*mcMS**8)/mbkin**8 + 
               (1097*mcMS**10)/mbkin**10 + (28*mcMS**12)/mbkin**12)*q_cut**6)/
             mbkin**12 + (2*(-90 + (771*mcMS**2)/mbkin**2 - (786*mcMS**4)/
                mbkin**4 - (5817*mcMS**6)/mbkin**6 + (112*mcMS**8)/mbkin**8 + 
               (187*mcMS**10)/mbkin**10)*q_cut**7)/mbkin**14 - 
            ((-21 + (51*mcMS**2)/mbkin**2 + (453*mcMS**4)/mbkin**4 + (495*mcMS**6)/
                mbkin**6 + (2*mcMS**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
            ((-138 + (559*mcMS**2)/mbkin**2 + (1610*mcMS**4)/mbkin**4 + 
               (283*mcMS**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
            ((-63 + (247*mcMS**2)/mbkin**2 + (58*mcMS**4)/mbkin**4)*q_cut**10)/
             mbkin**20 + (35*mcMS**2*q_cut**11)/mbkin**24) + 
          12*muG**2*((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/
              mbkin**4 + q_cut**2/mbkin**4)**2*(-99 - (7119*mcMS**2)/mbkin**2 + 
            (37829*mcMS**4)/mbkin**4 - (17113*mcMS**6)/mbkin**6 - 
            (532974*mcMS**8)/mbkin**8 - (15204*mcMS**10)/mbkin**10 - 
            (2224272*mcMS**12)/mbkin**12 - (6676824*mcMS**14)/mbkin**14 - 
            (2350377*mcMS**16)/mbkin**16 + (1791459*mcMS**18)/mbkin**18 + 
            (80763*mcMS**20)/mbkin**20 - (64799*mcMS**22)/mbkin**22 - 
            (470*mcMS**24)/mbkin**24 + ((-138 - (7727*mcMS**2)/mbkin**2 + 
               (32390*mcMS**4)/mbkin**4 - (100577*mcMS**6)/mbkin**6 - 
               (1118888*mcMS**8)/mbkin**8 - (1883828*mcMS**10)/mbkin**10 + 
               (1076956*mcMS**12)/mbkin**12 - (1208036*mcMS**14)/mbkin**14 - 
               (2113334*mcMS**16)/mbkin**16 + (233467*mcMS**18)/mbkin**18 + 
               (99510*mcMS**20)/mbkin**20 + (605*mcMS**22)/mbkin**22)*q_cut)/
             mbkin**2 + ((33 + (2803*mcMS**2)/mbkin**2 - (82700*mcMS**4)/
                mbkin**4 - (325442*mcMS**6)/mbkin**6 - (916186*mcMS**8)/mbkin**8 - 
               (5230838*mcMS**10)/mbkin**10 - (5833550*mcMS**12)/mbkin**12 - 
               (2100802*mcMS**14)/mbkin**14 + (236197*mcMS**16)/mbkin**16 + 
               (112255*mcMS**18)/mbkin**18 + (1030*mcMS**20)/mbkin**20)*q_cut**2)/
             mbkin**4 + ((492 + (28915*mcMS**2)/mbkin**2 - (164660*mcMS**4)/
                mbkin**4 - (809006*mcMS**6)/mbkin**6 + (248124*mcMS**8)/mbkin**8 + 
               (2883926*mcMS**10)/mbkin**10 + (1025940*mcMS**12)/mbkin**12 - 
               (1069270*mcMS**14)/mbkin**14 - (209496*mcMS**16)/mbkin**16 - 
               (1165*mcMS**18)/mbkin**18)*q_cut**3)/mbkin**6 - 
            (2*(-141 + (393*mcMS**2)/mbkin**2 - (5790*mcMS**4)/mbkin**4 + 
               (26550*mcMS**6)/mbkin**6 + (712286*mcMS**8)/mbkin**8 + 
               (454454*mcMS**10)/mbkin**10 + (76233*mcMS**12)/mbkin**12 + 
               (15871*mcMS**14)/mbkin**14 + (280*mcMS**16)/mbkin**16)*q_cut**4)/
             mbkin**8 + (2*(-132 - (11435*mcMS**2)/mbkin**2 + (60198*mcMS**4)/
                mbkin**4 + (474024*mcMS**6)/mbkin**6 + (702514*mcMS**8)/mbkin**8 + 
               (444840*mcMS**10)/mbkin**10 + (57426*mcMS**12)/mbkin**12 - 
               (35*mcMS**14)/mbkin**14)*q_cut**5)/mbkin**10 - 
            (2*(195 - (2837*mcMS**2)/mbkin**2 - (1149*mcMS**4)/mbkin**4 + 
               (116747*mcMS**6)/mbkin**6 + (78417*mcMS**8)/mbkin**8 + 
               (7121*mcMS**10)/mbkin**10 + (140*mcMS**12)/mbkin**12)*q_cut**6)/
             mbkin**12 + (2*(-198 + (1767*mcMS**2)/mbkin**2 - (386*mcMS**4)/
                mbkin**4 - (29909*mcMS**6)/mbkin**6 + (684*mcMS**8)/mbkin**8 + 
               (935*mcMS**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
            ((105 - (1039*mcMS**2)/mbkin**2 + (1391*mcMS**4)/mbkin**4 - 
               (1171*mcMS**6)/mbkin**6 - (10*mcMS**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
            (5*(-138 + (559*mcMS**2)/mbkin**2 + (1554*mcMS**4)/mbkin**4 + 
               (283*mcMS**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
            (5*(-63 + (247*mcMS**2)/mbkin**2 + (58*mcMS**4)/mbkin**4)*q_cut**10)/
             mbkin**20 + (175*mcMS**2*q_cut**11)/mbkin**24) - 
          8*mbkin*(-4*(-1 + mcMS**2/mbkin**2)**4*(1 + mcMS**2/mbkin**2)**2*
             (1179 - (27994*mcMS**2)/mbkin**2 + (151119*mcMS**4)/mbkin**4 + 
              (982305*mcMS**6)/mbkin**6 - (6893010*mcMS**8)/mbkin**8 - 
              (8620596*mcMS**10)/mbkin**10 + (1136781*mcMS**12)/mbkin**12 + 
              (171093*mcMS**14)/mbkin**14 - (137043*mcMS**16)/mbkin**16 + 
              (13572*mcMS**18)/mbkin**18 + (154*mcMS**20)/mbkin**20) + 
            ((-1 + mcMS**2/mbkin**2)**2*(26163 - (491315*mcMS**2)/mbkin**2 + 
               (609844*mcMS**4)/mbkin**4 + (29190880*mcMS**6)/mbkin**6 - 
               (45147167*mcMS**8)/mbkin**8 - (424453401*mcMS**10)/mbkin**10 - 
               (729940536*mcMS**12)/mbkin**12 - (481708704*mcMS**14)/mbkin**14 - 
               (67588491*mcMS**16)/mbkin**16 + (33128619*mcMS**18)/mbkin**18 - 
               (4511740*mcMS**20)/mbkin**20 - (1905472*mcMS**22)/mbkin**22 + 
               (315767*mcMS**24)/mbkin**24 + (3233*mcMS**26)/mbkin**26)*q_cut)/
             mbkin**2 + (4*(-12321 + (196421*mcMS**2)/mbkin**2 + (301638*mcMS**4)/
                mbkin**4 - (12526887*mcMS**6)/mbkin**6 + (1399932*mcMS**8)/
                mbkin**8 + (127033574*mcMS**10)/mbkin**10 + (284550615*mcMS**12)/
                mbkin**12 + (300700218*mcMS**14)/mbkin**14 + (150895899*mcMS**16)/
                mbkin**16 + (5517729*mcMS**18)/mbkin**18 - (15244856*mcMS**20)/
                mbkin**20 + (2907981*mcMS**22)/mbkin**22 + (671466*mcMS**24)/
                mbkin**24 - (153900*mcMS**26)/mbkin**26 - (1349*mcMS**28)/
                mbkin**28)*q_cut**2)/mbkin**4 + ((14763 - (64787*mcMS**2)/mbkin**2 - 
               (2819568*mcMS**4)/mbkin**4 + (7404252*mcMS**6)/mbkin**6 + 
               (43250901*mcMS**8)/mbkin**8 - (6393169*mcMS**10)/mbkin**10 - 
               (79307576*mcMS**12)/mbkin**12 - (4041216*mcMS**14)/mbkin**14 + 
               (48737885*mcMS**16)/mbkin**16 + (556627*mcMS**18)/mbkin**18 - 
               (8219224*mcMS**20)/mbkin**20 + (656612*mcMS**22)/mbkin**22 + 
               (224419*mcMS**24)/mbkin**24 + (81*mcMS**26)/mbkin**26)*q_cut**3)/
             mbkin**6 + (4*(18273 - (235702*mcMS**2)/mbkin**2 - (1086112*mcMS**4)/
                mbkin**4 + (11266643*mcMS**6)/mbkin**6 + (36598184*mcMS**8)/
                mbkin**8 + (45843340*mcMS**10)/mbkin**10 + (44974590*mcMS**12)/
                mbkin**12 + (36547042*mcMS**14)/mbkin**14 + (10361895*mcMS**16)/
                mbkin**16 - (3866986*mcMS**18)/mbkin**18 - (424058*mcMS**20)/
                mbkin**20 + (224799*mcMS**22)/mbkin**22 + (2444*mcMS**24)/
                mbkin**24)*q_cut**4)/mbkin**8 - ((98721 - (917479*mcMS**2)/mbkin**2 - 
               (9760513*mcMS**4)/mbkin**4 + (30195627*mcMS**6)/mbkin**6 + 
               (170463350*mcMS**8)/mbkin**8 + (259417590*mcMS**10)/mbkin**10 + 
               (167078226*mcMS**12)/mbkin**12 + (15469354*mcMS**14)/mbkin**14 - 
               (21555727*mcMS**16)/mbkin**16 + (973161*mcMS**18)/mbkin**18 + 
               (1273159*mcMS**20)/mbkin**20 + (10643*mcMS**22)/mbkin**22)*q_cut**5)/
             mbkin**10 + (4*(5934 - (43901*mcMS**2)/mbkin**2 - (695976*mcMS**4)/
                mbkin**4 + (2309184*mcMS**6)/mbkin**6 + (11870802*mcMS**8)/
                mbkin**8 + (11318680*mcMS**10)/mbkin**10 + (724070*mcMS**12)/
                mbkin**12 - (1728188*mcMS**14)/mbkin**14 + (119948*mcMS**16)/
                mbkin**16 + (74017*mcMS**18)/mbkin**18 + (526*mcMS**20)/mbkin**20)*
              q_cut**6)/mbkin**12 + ((44175 - (243147*mcMS**2)/mbkin**2 - 
               (5579352*mcMS**4)/mbkin**4 - (17016744*mcMS**6)/mbkin**6 - 
               (22924990*mcMS**8)/mbkin**8 - (14689794*mcMS**10)/mbkin**10 - 
               (2030576*mcMS**12)/mbkin**12 + (2336816*mcMS**14)/mbkin**14 + 
               (572671*mcMS**16)/mbkin**16 + (6717*mcMS**18)/mbkin**18)*q_cut**7)/
             mbkin**14 - (4*(11646 - (53280*mcMS**2)/mbkin**2 - (1304004*mcMS**4)/
                mbkin**4 - (3516303*mcMS**6)/mbkin**6 - (3269188*mcMS**8)/
                mbkin**8 - (647651*mcMS**10)/mbkin**10 + (523431*mcMS**12)/
                mbkin**12 + (121214*mcMS**14)/mbkin**14 + (3663*mcMS**16)/
                mbkin**16)*q_cut**8)/mbkin**16 + ((24633 - (76675*mcMS**2)/mbkin**2 - 
               (1532353*mcMS**4)/mbkin**4 - (3216569*mcMS**6)/mbkin**6 - 
               (1176415*mcMS**8)/mbkin**8 + (448897*mcMS**10)/mbkin**10 + 
               (145151*mcMS**12)/mbkin**12 + (15283*mcMS**14)/mbkin**14)*q_cut**9)/
             mbkin**18 - (4*(1197 - (4127*mcMS**2)/mbkin**2 - (21706*mcMS**4)/
                mbkin**4 - (39039*mcMS**6)/mbkin**6 - (14018*mcMS**8)/mbkin**8 + 
               (2910*mcMS**10)/mbkin**10 + (449*mcMS**12)/mbkin**12)*q_cut**10)/
             mbkin**20 - ((8559 + (20945*mcMS**2)/mbkin**2 + (29208*mcMS**4)/
                mbkin**4 + (54804*mcMS**6)/mbkin**6 + (34199*mcMS**8)/mbkin**8 + 
               (11197*mcMS**10)/mbkin**10)*q_cut**11)/mbkin**22 + 
            (4*(2121 + (4502*mcMS**2)/mbkin**2 + (8820*mcMS**4)/mbkin**4 + 
               (9231*mcMS**6)/mbkin**6 + (2526*mcMS**8)/mbkin**8)*q_cut**12)/
             mbkin**24 - ((2379 + (5003*mcMS**2)/mbkin**2 + (11693*mcMS**4)/
                mbkin**4 + (3329*mcMS**6)/mbkin**6)*q_cut**13)/mbkin**26 + 
            (4*mcMS**2*(137 + (104*mcMS**2)/mbkin**2)*q_cut**14)/mbkin**30 - 
            (5*(15 + (29*mcMS**2)/mbkin**2)*q_cut**15)/mbkin**30 + 
            (60*q_cut**16)/mbkin**32)*rhoD + ((mbkin**6 - 7*mbkin**4*mcMS**2 - 
             7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
             mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)*
            (-16*(-((-1 + mcMS**2/mbkin**2)**4*(-11807 + (43798*mcMS**2)/
                   mbkin**2 + (479384*mcMS**4)/mbkin**4 - (709662*mcMS**6)/
                   mbkin**6 - (2037632*mcMS**8)/mbkin**8 - (998650*mcMS**10)/
                   mbkin**10 + (38820*mcMS**12)/mbkin**12 + (204634*mcMS**14)/
                   mbkin**14 + (72155*mcMS**16)/mbkin**16 + (800*mcMS**18)/
                   mbkin**18)) + ((-1 + mcMS**2/mbkin**2)**2*(-53075 + 
                  (56430*mcMS**2)/mbkin**2 + (2219469*mcMS**4)/mbkin**4 - 
                  (1103324*mcMS**6)/mbkin**6 - (11042008*mcMS**8)/mbkin**8 - 
                  (11699532*mcMS**10)/mbkin**10 - (3837272*mcMS**12)/mbkin**12 + 
                  (583124*mcMS**14)/mbkin**14 + (1195347*mcMS**16)/mbkin**16 + 
                  (332166*mcMS**18)/mbkin**18 + (3395*mcMS**20)/mbkin**20)*q_cut)/
                mbkin**2 + ((79585 + (71237*mcMS**2)/mbkin**2 - (3051995*mcMS**4)/
                   mbkin**4 + (927717*mcMS**6)/mbkin**6 + (13296168*mcMS**8)/
                   mbkin**8 + (18284376*mcMS**10)/mbkin**10 + (15027276*mcMS**12)/
                   mbkin**12 + (5791152*mcMS**14)/mbkin**14 - (1097145*mcMS**16)/
                   mbkin**16 - (2117941*mcMS**18)/mbkin**18 - (515489*mcMS**20)/
                   mbkin**20 - (4381*mcMS**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
               ((-19567 - (124096*mcMS**2)/mbkin**2 + (851901*mcMS**4)/mbkin**4 + 
                  (142210*mcMS**6)/mbkin**6 - (6133506*mcMS**8)/mbkin**8 - 
                  (6567734*mcMS**10)/mbkin**10 + (601502*mcMS**12)/mbkin**12 + 
                  (3352138*mcMS**14)/mbkin**14 + (1456325*mcMS**16)/mbkin**16 + 
                  (156130*mcMS**18)/mbkin**18 - (183*mcMS**20)/mbkin**20)*q_cut**3)/
                mbkin**6 + ((-67452 - (313401*mcMS**2)/mbkin**2 + 
                  (755046*mcMS**4)/mbkin**4 + (2414684*mcMS**6)/mbkin**6 + 
                  (3075992*mcMS**8)/mbkin**8 + (2747000*mcMS**10)/mbkin**10 + 
                  (2478366*mcMS**12)/mbkin**12 + (1825168*mcMS**14)/mbkin**14 + 
                  (418744*mcMS**16)/mbkin**16 + (4885*mcMS**18)/mbkin**18)*q_cut**4)/
                mbkin**8 - (2*(-35815 - (226435*mcMS**2)/mbkin**2 + 
                  (358442*mcMS**4)/mbkin**4 + (1835272*mcMS**6)/mbkin**6 + 
                  (2982792*mcMS**8)/mbkin**8 + (2754841*mcMS**10)/mbkin**10 + 
                  (1366980*mcMS**12)/mbkin**12 + (235624*mcMS**14)/mbkin**14 + 
                  (2391*mcMS**16)/mbkin**16)*q_cut**5)/mbkin**10 + 
               (2*(-12395 - (83570*mcMS**2)/mbkin**2 + (199368*mcMS**4)/
                   mbkin**4 + (884692*mcMS**6)/mbkin**6 + (991494*mcMS**8)/
                   mbkin**8 + (506815*mcMS**10)/mbkin**10 + (88823*mcMS**12)/
                   mbkin**12 + (1353*mcMS**14)/mbkin**14)*q_cut**6)/mbkin**12 + 
               (2*(715 + (5343*mcMS**2)/mbkin**2 - (21471*mcMS**4)/mbkin**4 - 
                  (34137*mcMS**6)/mbkin**6 - (33187*mcMS**8)/mbkin**8 - 
                  (13378*mcMS**10)/mbkin**10 + (1443*mcMS**12)/mbkin**12)*q_cut**7)/
                mbkin**14 - ((-2087 + (2386*mcMS**2)/mbkin**2 + (18520*mcMS**4)/
                   mbkin**4 + (2730*mcMS**6)/mbkin**6 + (10355*mcMS**8)/mbkin**8 + 
                  (10962*mcMS**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
               ((-3243 + (3494*mcMS**2)/mbkin**2 + (16102*mcMS**4)/mbkin**4 + 
                  (26896*mcMS**6)/mbkin**6 + (11675*mcMS**8)/mbkin**8)*q_cut**9)/
                mbkin**18 - ((-2053 + (3361*mcMS**2)/mbkin**2 + (12637*mcMS**4)/
                   mbkin**4 + (5333*mcMS**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
               ((-455 + (738*mcMS**2)/mbkin**2 + (1009*mcMS**4)/mbkin**4)*q_cut**11)/
                mbkin**22 - (15*(6 + (13*mcMS**2)/mbkin**2)*q_cut**12)/mbkin**24 + 
               (80*q_cut**13)/mbkin**26)*rE - ((-1 + mcMS**2/mbkin**2)**2 - 
               (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
              (-4*(-2*(-1 + mcMS**2/mbkin**2)**2*(-8941 - (6827*mcMS**2)/
                    mbkin**2 + (409306*mcMS**4)/mbkin**4 + (95050*mcMS**6)/
                    mbkin**6 - (1229126*mcMS**8)/mbkin**8 - (1188302*mcMS**10)/
                    mbkin**10 - (194666*mcMS**12)/mbkin**12 + (101926*mcMS**14)/
                    mbkin**14 + (25507*mcMS**16)/mbkin**16 + (233*mcMS**18)/
                    mbkin**18) + ((-44221 - (216428*mcMS**2)/mbkin**2 + 
                    (1611477*mcMS**4)/mbkin**4 + (2327952*mcMS**6)/mbkin**6 - 
                    (4524756*mcMS**8)/mbkin**8 - (9498816*mcMS**10)/mbkin**10 - 
                    (5978964*mcMS**12)/mbkin**12 - (429264*mcMS**14)/mbkin**14 + 
                    (655281*mcMS**16)/mbkin**16 + (129964*mcMS**18)/mbkin**18 + 
                    (1055*mcMS**20)/mbkin**20)*q_cut)/mbkin**2 - 
                 (3*(-4221 - (64891*mcMS**2)/mbkin**2 - (48868*mcMS**4)/
                     mbkin**4 + (76692*mcMS**6)/mbkin**6 - (88296*mcMS**8)/
                     mbkin**8 + (102712*mcMS**10)/mbkin**10 + (313380*mcMS**12)/
                     mbkin**12 + (141964*mcMS**14)/mbkin**14 + (15045*mcMS**16)/
                     mbkin**16 + (3*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
                 (2*(-20817 - (183792*mcMS**2)/mbkin**2 - (135051*mcMS**4)/
                     mbkin**4 + (238636*mcMS**6)/mbkin**6 + (639405*mcMS**8)/
                     mbkin**8 + (822464*mcMS**10)/mbkin**10 + (396761*mcMS**12)/
                     mbkin**12 + (60252*mcMS**14)/mbkin**14 + (574*mcMS**16)/
                     mbkin**16)*q_cut**3)/mbkin**6 + (2*(-15621 - (180327*mcMS**2)/
                     mbkin**2 - (164400*mcMS**4)/mbkin**4 + (580364*mcMS**6)/
                     mbkin**6 + (870223*mcMS**8)/mbkin**8 + (378225*mcMS**10)/
                     mbkin**10 + (48724*mcMS**12)/mbkin**12 + (284*mcMS**14)/
                     mbkin**14)*q_cut**4)/mbkin**8 - (2*(-1756 - (14964*mcMS**2)/
                     mbkin**2 - (2592*mcMS**4)/mbkin**4 + (68024*mcMS**6)/
                     mbkin**6 + (36849*mcMS**8)/mbkin**8 + (5232*mcMS**10)/
                     mbkin**10 + (287*mcMS**12)/mbkin**12)*q_cut**5)/mbkin**10 + 
                 (10*(-66 + (654*mcMS**2)/mbkin**2 - (1140*mcMS**4)/mbkin**4 - 
                    (1204*mcMS**6)/mbkin**6 + (117*mcMS**8)/mbkin**8 + 
                    (7*mcMS**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
                 (2*(-1347 + (648*mcMS**2)/mbkin**2 - (945*mcMS**4)/mbkin**4 + 
                    (2092*mcMS**6)/mbkin**6 + (1052*mcMS**8)/mbkin**8)*q_cut**7)/
                  mbkin**14 - (2*(-1960 + (810*mcMS**2)/mbkin**2 + (3651*mcMS**4)/
                     mbkin**4 + (1091*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
                 ((-779 + (756*mcMS**2)/mbkin**2 + (527*mcMS**4)/mbkin**4)*q_cut**9)/
                  mbkin**18 + (35*(-1 + mcMS**2/mbkin**2)*q_cut**10)/mbkin**20 + 
                 (20*q_cut**11)/mbkin**22)*rG - 4*(-2*(-1 + mcMS**2/mbkin**2)**2*
                  (6246 + (67217*mcMS**2)/mbkin**2 - (529254*mcMS**4)/mbkin**4 - 
                   (2490332*mcMS**6)/mbkin**6 - (2668594*mcMS**8)/mbkin**8 - 
                   (1145520*mcMS**10)/mbkin**10 - (280166*mcMS**12)/mbkin**12 + 
                   (158480*mcMS**14)/mbkin**14 + (62208*mcMS**16)/mbkin**16 + 
                   (595*mcMS**18)/mbkin**18) + ((30051 + (480428*mcMS**2)/
                     mbkin**2 - (1230867*mcMS**4)/mbkin**4 - (12753936*mcMS**6)/
                     mbkin**6 - (21644916*mcMS**8)/mbkin**8 - (14619072*mcMS**10)/
                     mbkin**10 - (5141604*mcMS**12)/mbkin**12 - (1089024*
                      mcMS**14)/mbkin**14 + (1095105*mcMS**16)/mbkin**16 + 
                    (318180*mcMS**18)/mbkin**18 + (2695*mcMS**20)/mbkin**20)*q_cut)/
                  mbkin**2 - (3*(2529 + (82713*mcMS**2)/mbkin**2 + 
                    (305552*mcMS**4)/mbkin**4 + (155956*mcMS**6)/mbkin**6 + 
                    (105796*mcMS**8)/mbkin**8 + (230008*mcMS**10)/mbkin**10 + 
                    (330632*mcMS**12)/mbkin**12 + (264908*mcMS**14)/mbkin**14 + 
                    (37251*mcMS**16)/mbkin**16 + (15*mcMS**18)/mbkin**18)*q_cut**2)/
                  mbkin**4 - (2*(14181 + (280842*mcMS**2)/mbkin**2 + 
                    (943095*mcMS**4)/mbkin**4 + (1413102*mcMS**6)/mbkin**6 + 
                    (1811119*mcMS**8)/mbkin**8 + (1516738*mcMS**10)/mbkin**10 + 
                    (804843*mcMS**12)/mbkin**12 + (148334*mcMS**14)/mbkin**14 + 
                    (1442*mcMS**16)/mbkin**16)*q_cut**3)/mbkin**6 + 
                 (2*(10191 + (248865*mcMS**2)/mbkin**2 + (1087626*mcMS**4)/
                     mbkin**4 + (1689732*mcMS**6)/mbkin**6 + (1520309*mcMS**8)/
                     mbkin**8 + (788349*mcMS**10)/mbkin**10 + (120732*mcMS**12)/
                     mbkin**12 + (712*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 - 
                 (6*(282 + (7474*mcMS**2)/mbkin**2 + (28672*mcMS**4)/mbkin**4 + 
                    (40064*mcMS**6)/mbkin**6 + (27057*mcMS**8)/mbkin**8 + 
                    (4094*mcMS**10)/mbkin**10 + (245*mcMS**12)/mbkin**12)*q_cut**5)/
                  mbkin**10 + (10*(-18 - (906*mcMS**2)/mbkin**2 - (1926*mcMS**4)/
                     mbkin**4 - (3108*mcMS**6)/mbkin**6 + (149*mcMS**8)/mbkin**8 + 
                    (35*mcMS**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
                 (2*(-81 - (462*mcMS**2)/mbkin**2 + (1233*mcMS**4)/mbkin**4 + 
                    (4734*mcMS**6)/mbkin**6 + (2380*mcMS**8)/mbkin**8)*q_cut**7)/
                  mbkin**14 - (6*(-173 + (1028*mcMS**2)/mbkin**2 + (2902*mcMS**4)/
                     mbkin**4 + (791*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
                 ((-951 + (1936*mcMS**2)/mbkin**2 + (871*mcMS**4)/mbkin**4)*q_cut**9)/
                  mbkin**18 + (35*(-3 + (5*mcMS**2)/mbkin**2)*q_cut**10)/mbkin**20 + 
                 (60*q_cut**11)/mbkin**22)*sB - 46416*sE - (180016*mcMS**2*sE)/
                mbkin**2 + (3327728*mcMS**4*sE)/mbkin**4 + (1108272*mcMS**6*sE)/
                mbkin**6 - (9079776*mcMS**8*sE)/mbkin**8 + (1599360*mcMS**10*sE)/
                mbkin**10 + (4680480*mcMS**12*sE)/mbkin**12 - (2416128*mcMS**14*
                 sE)/mbkin**14 + (1409328*mcMS**16*sE)/mbkin**16 - (108624*
                 mcMS**18*sE)/mbkin**18 - (291344*mcMS**20*sE)/mbkin**20 - 
               (2864*mcMS**22*sE)/mbkin**22 + (113376*q_cut*sE)/mbkin**2 + 
               (1176992*mcMS**2*q_cut*sE)/mbkin**4 - (4539072*mcMS**4*q_cut*sE)/
                mbkin**6 - (22322496*mcMS**6*q_cut*sE)/mbkin**8 - (17816832*mcMS**8*
                 q_cut*sE)/mbkin**10 - (449664*mcMS**10*q_cut*sE)/mbkin**12 + 
               (1554240*mcMS**12*q_cut*sE)/mbkin**14 - (806976*mcMS**14*q_cut*sE)/
                mbkin**16 + (2644128*mcMS**16*q_cut*sE)/mbkin**18 + (765024*mcMS**18*
                 q_cut*sE)/mbkin**20 + (6400*mcMS**20*q_cut*sE)/mbkin**22 - 
               (30384*q_cut**2*sE)/mbkin**4 - (719568*mcMS**2*q_cut**2*sE)/mbkin**6 - 
               (1578336*mcMS**4*q_cut**2*sE)/mbkin**8 + (1362048*mcMS**6*q_cut**2*sE)/
                mbkin**10 + (1248288*mcMS**8*q_cut**2*sE)/mbkin**12 - (3449280*
                 mcMS**10*q_cut**2*sE)/mbkin**14 - (5232480*mcMS**12*q_cut**2*sE)/
                mbkin**16 - (2609664*mcMS**14*q_cut**2*sE)/mbkin**18 - (280368*
                 mcMS**16*q_cut**2*sE)/mbkin**20 + (144*mcMS**18*q_cut**2*sE)/mbkin**22 - 
               (106992*q_cut**3*sE)/mbkin**6 - (1520496*mcMS**2*q_cut**3*sE)/mbkin**8 - 
               (3246960*mcMS**4*q_cut**3*sE)/mbkin**10 - (1996656*mcMS**6*q_cut**3*sE)/
                mbkin**12 - (667760*mcMS**8*q_cut**3*sE)/mbkin**14 - (1415408*
                 mcMS**10*q_cut**3*sE)/mbkin**16 - (2723568*mcMS**12*q_cut**3*sE)/
                mbkin**18 - (698608*mcMS**14*q_cut**3*sE)/mbkin**20 - (7168*mcMS**16*
                 q_cut**3*sE)/mbkin**22 + (78384*q_cut**4*sE)/mbkin**8 + (1397952*
                 mcMS**2*q_cut**4*sE)/mbkin**10 + (4097136*mcMS**4*q_cut**4*sE)/
                mbkin**12 + (3924288*mcMS**6*q_cut**4*sE)/mbkin**14 + (4126192*
                 mcMS**8*q_cut**4*sE)/mbkin**16 + (3239904*mcMS**10*q_cut**4*sE)/
                mbkin**18 + (575280*mcMS**12*q_cut**4*sE)/mbkin**20 + (3488*mcMS**14*
                 q_cut**4*sE)/mbkin**22 - (9328*q_cut**5*sE)/mbkin**10 - (115056*mcMS**2*
                 q_cut**5*sE)/mbkin**12 - (386496*mcMS**4*q_cut**5*sE)/mbkin**14 - 
               (386944*mcMS**6*q_cut**5*sE)/mbkin**16 - (361392*mcMS**8*q_cut**5*sE)/
                mbkin**18 - (63792*mcMS**10*q_cut**5*sE)/mbkin**20 - (3136*mcMS**12*
                 q_cut**5*sE)/mbkin**22 + (1200*q_cut**6*sE)/mbkin**12 - (24960*mcMS**2*
                 q_cut**6*sE)/mbkin**14 - (55200*mcMS**4*q_cut**6*sE)/mbkin**16 - 
               (51840*mcMS**6*q_cut**6*sE)/mbkin**18 + (19760*mcMS**8*q_cut**6*sE)/
                mbkin**20 - (1120*mcMS**10*q_cut**6*sE)/mbkin**22 + (6960*q_cut**7*sE)/
                mbkin**14 - (5904*mcMS**2*q_cut**7*sE)/mbkin**16 - (15888*mcMS**4*
                 q_cut**7*sE)/mbkin**18 + (5232*mcMS**6*q_cut**7*sE)/mbkin**20 + 
               (15296*mcMS**8*q_cut**7*sE)/mbkin**22 - (6112*q_cut**8*sE)/mbkin**16 - 
               (12048*mcMS**2*q_cut**8*sE)/mbkin**18 - (34848*mcMS**4*q_cut**8*sE)/
                mbkin**20 - (15472*mcMS**6*q_cut**8*sE)/mbkin**22 - (1008*q_cut**9*sE)/
                mbkin**18 + (3664*mcMS**2*q_cut**9*sE)/mbkin**20 + (4672*mcMS**4*q_cut**9*
                 sE)/mbkin**22 - (560*mcMS**2*q_cut**10*sE)/mbkin**22 + (320*q_cut**11*
                 sE)/mbkin**22 - 6084*sqB - (51628*mcMS**2*sqB)/mbkin**2 + 
               (647540*mcMS**4*sqB)/mbkin**4 + (2185704*mcMS**6*sqB)/mbkin**6 - 
               (1160436*mcMS**8*sqB)/mbkin**8 - (3871392*mcMS**10*sqB)/
                mbkin**10 - (5292*mcMS**12*sqB)/mbkin**12 + (1785960*mcMS**14*
                 sqB)/mbkin**14 + (538296*mcMS**16*sqB)/mbkin**16 - 
               (48564*mcMS**18*sqB)/mbkin**18 - (14024*mcMS**20*sqB)/mbkin**20 - 
               (80*mcMS**22*sqB)/mbkin**22 + (14649*q_cut*sqB)/mbkin**2 + 
               (230462*mcMS**2*q_cut*sqB)/mbkin**4 - (478107*mcMS**4*q_cut*sqB)/
                mbkin**6 - (7544880*mcMS**6*q_cut*sqB)/mbkin**8 - (17208996*mcMS**8*
                 q_cut*sqB)/mbkin**10 - (15625344*mcMS**10*q_cut*sqB)/mbkin**12 - 
               (5968452*mcMS**12*q_cut*sqB)/mbkin**14 - (316608*mcMS**14*q_cut*sqB)/
                mbkin**16 + (291675*mcMS**16*q_cut*sqB)/mbkin**18 + (35826*mcMS**18*
                 q_cut*sqB)/mbkin**20 + (175*mcMS**20*q_cut*sqB)/mbkin**22 - 
               (3789*q_cut**2*sqB)/mbkin**4 - (117945*mcMS**2*q_cut**2*sqB)/mbkin**6 - 
               (543048*mcMS**4*q_cut**2*sqB)/mbkin**8 - (622812*mcMS**6*q_cut**2*sqB)/
                mbkin**10 - (569172*mcMS**8*q_cut**2*sqB)/mbkin**12 - (1021296*
                 mcMS**10*q_cut**2*sqB)/mbkin**14 - (789072*mcMS**12*q_cut**2*sqB)/
                mbkin**16 - (201876*mcMS**14*q_cut**2*sqB)/mbkin**18 - 
               (11799*mcMS**16*q_cut**2*sqB)/mbkin**20 + (9*mcMS**18*q_cut**2*sqB)/
                mbkin**22 - (13812*q_cut**3*sqB)/mbkin**6 - (270534*mcMS**2*q_cut**3*
                 sqB)/mbkin**8 - (1041540*mcMS**4*q_cut**3*sqB)/mbkin**10 - 
               (1476162*mcMS**6*q_cut**3*sqB)/mbkin**12 - (1406192*mcMS**8*q_cut**3*sqB)/
                mbkin**14 - (991022*mcMS**10*q_cut**3*sqB)/mbkin**16 - (329844*
                 mcMS**12*q_cut**3*sqB)/mbkin**18 - (33178*mcMS**14*q_cut**3*sqB)/
                mbkin**20 - (196*mcMS**16*q_cut**3*sqB)/mbkin**22 + (9996*q_cut**4*sqB)/
                mbkin**8 + (239034*mcMS**2*q_cut**4*sqB)/mbkin**10 + (1181778*mcMS**4*
                 q_cut**4*sqB)/mbkin**12 + (1946208*mcMS**6*q_cut**4*sqB)/mbkin**14 + 
               (1272532*mcMS**8*q_cut**4*sqB)/mbkin**16 + (332286*mcMS**10*q_cut**4*sqB)/
                mbkin**18 + (26154*mcMS**12*q_cut**4*sqB)/mbkin**20 + (92*mcMS**14*
                 q_cut**4*sqB)/mbkin**22 - (442*q_cut**5*sqB)/mbkin**10 - (22842*mcMS**2*
                 q_cut**5*sqB)/mbkin**12 - (87144*mcMS**4*q_cut**5*sqB)/mbkin**14 - 
               (111472*mcMS**6*q_cut**5*sqB)/mbkin**16 - (31992*mcMS**8*q_cut**5*sqB)/
                mbkin**18 - (2910*mcMS**10*q_cut**5*sqB)/mbkin**20 - (70*mcMS**12*
                 q_cut**5*sqB)/mbkin**22 - (330*q_cut**6*sqB)/mbkin**12 - (4020*mcMS**2*
                 q_cut**6*sqB)/mbkin**14 - (10620*mcMS**4*q_cut**6*sqB)/mbkin**16 - 
               (5640*mcMS**6*q_cut**6*sqB)/mbkin**18 + (800*mcMS**8*q_cut**6*sqB)/
                mbkin**20 - (70*mcMS**10*q_cut**6*sqB)/mbkin**22 - (672*q_cut**7*sqB)/
                mbkin**14 - (2298*mcMS**2*q_cut**7*sqB)/mbkin**16 - (4992*mcMS**4*
                 q_cut**7*sqB)/mbkin**18 + (810*mcMS**6*q_cut**7*sqB)/mbkin**20 + 
               (416*mcMS**8*q_cut**7*sqB)/mbkin**22 + (1016*q_cut**8*sqB)/mbkin**16 - 
               (606*mcMS**2*q_cut**8*sqB)/mbkin**18 - (1986*mcMS**4*q_cut**8*sqB)/
                mbkin**20 - (364*mcMS**6*q_cut**8*sqB)/mbkin**22 - (447*q_cut**9*sqB)/
                mbkin**18 + (412*mcMS**2*q_cut**9*sqB)/mbkin**20 + (103*mcMS**4*q_cut**9*
                 sqB)/mbkin**22 - (105*q_cut**10*sqB)/mbkin**20 - (35*mcMS**2*q_cut**10*
                 sqB)/mbkin**22 + (20*q_cut**11*sqB)/mbkin**22)))/mbkin**6) - 
        6*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
            2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
         (16*(-((-1 + mcMS**2/mbkin**2)**4*(-487 + (1378*mcMS**2)/mbkin**2 + 
               (103211*mcMS**4)/mbkin**4 - (451836*mcMS**6)/mbkin**6 - 
               (2360958*mcMS**8)/mbkin**8 + (4610472*mcMS**10)/mbkin**10 + 
               (11668476*mcMS**12)/mbkin**12 + (5391216*mcMS**14)/mbkin**14 - 
               (326961*mcMS**16)/mbkin**16 - (795314*mcMS**18)/mbkin**18 - 
               (332527*mcMS**20)/mbkin**20 - (1436*mcMS**22)/mbkin**22 + 
               (3726*mcMS**24)/mbkin**24)) + (2*(-1 + mcMS**2/mbkin**2)**2*
              (-1464 + (878*mcMS**2)/mbkin**2 + (264747*mcMS**4)/mbkin**4 - 
               (654866*mcMS**6)/mbkin**6 - (6024249*mcMS**8)/mbkin**8 + 
               (5029272*mcMS**10)/mbkin**10 + (32640138*mcMS**12)/mbkin**12 + 
               (33450960*mcMS**14)/mbkin**14 + (10105578*mcMS**16)/mbkin**16 - 
               (1628862*mcMS**18)/mbkin**18 - (2359709*mcMS**20)/mbkin**20 - 
               (806502*mcMS**22)/mbkin**22 + (8687*mcMS**24)/mbkin**24 + 
               (11232*mcMS**26)/mbkin**26)*q_cut)/mbkin**2 + 
            ((6355 + (7208*mcMS**2)/mbkin**2 - (936656*mcMS**4)/mbkin**4 + 
               (1373882*mcMS**6)/mbkin**6 + (17299235*mcMS**8)/mbkin**8 - 
               (10086138*mcMS**10)/mbkin**10 - (79195350*mcMS**12)/mbkin**12 - 
               (110084676*mcMS**14)/mbkin**14 - (87026223*mcMS**16)/mbkin**16 - 
               (28155028*mcMS**18)/mbkin**18 + (5642380*mcMS**20)/mbkin**20 + 
               (8425514*mcMS**22)/mbkin**22 + (2715769*mcMS**24)/mbkin**24 - 
               (80762*mcMS**26)/mbkin**26 - (48870*mcMS**28)/mbkin**28)*q_cut**2)/
             mbkin**4 + (2*(-1960 - (9526*mcMS**2)/mbkin**2 + (177979*mcMS**4)/
                mbkin**4 + (103540*mcMS**6)/mbkin**6 - (2959273*mcMS**8)/
                mbkin**8 + (220386*mcMS**10)/mbkin**10 + (17116804*mcMS**12)/
                mbkin**12 + (16837442*mcMS**14)/mbkin**14 - (700248*mcMS**16)/
                mbkin**16 - (6799396*mcMS**18)/mbkin**18 - (3305847*mcMS**20)/
                mbkin**20 - (420862*mcMS**22)/mbkin**22 + (77361*mcMS**24)/
                mbkin**24 + (15120*mcMS**26)/mbkin**26)*q_cut**3)/mbkin**6 + 
            ((-6845 - (37942*mcMS**2)/mbkin**2 + (791885*mcMS**4)/mbkin**4 + 
               (1728634*mcMS**6)/mbkin**6 - (5957916*mcMS**8)/mbkin**8 - 
               (15444608*mcMS**10)/mbkin**10 - (18941330*mcMS**12)/mbkin**12 - 
               (18044424*mcMS**14)/mbkin**14 - (12783065*mcMS**16)/mbkin**16 - 
               (7164330*mcMS**18)/mbkin**18 - (1748003*mcMS**20)/mbkin**20 + 
               (205406*mcMS**22)/mbkin**22 + (52650*mcMS**24)/mbkin**24)*q_cut**4)/
             mbkin**8 - (4*(-3434 - (26020*mcMS**2)/mbkin**2 + (254961*mcMS**4)/
                mbkin**4 + (749581*mcMS**6)/mbkin**6 - (1690441*mcMS**8)/
                mbkin**8 - (6219248*mcMS**10)/mbkin**10 - (8487200*mcMS**12)/
                mbkin**12 - (6481863*mcMS**14)/mbkin**14 - (2731315*mcMS**16)/
                mbkin**16 - (318150*mcMS**18)/mbkin**18 + (155525*mcMS**20)/
                mbkin**20 + (26532*mcMS**22)/mbkin**22)*q_cut**5)/mbkin**10 + 
            ((-6969 - (50096*mcMS**2)/mbkin**2 + (479318*mcMS**4)/mbkin**4 + 
               (1199560*mcMS**6)/mbkin**6 - (3991626*mcMS**8)/mbkin**8 - 
               (11924636*mcMS**10)/mbkin**10 - (10756192*mcMS**12)/mbkin**12 - 
               (4517112*mcMS**14)/mbkin**14 - (393005*mcMS**16)/mbkin**16 + 
               (317212*mcMS**18)/mbkin**18 + (54882*mcMS**20)/mbkin**20)*q_cut**6)/
             mbkin**12 + (4*(-960 - (14642*mcMS**2)/mbkin**2 - (31541*mcMS**4)/
                mbkin**4 + (94173*mcMS**6)/mbkin**6 + (501370*mcMS**8)/mbkin**8 + 
               (725465*mcMS**10)/mbkin**10 + (578564*mcMS**12)/mbkin**12 + 
               (295434*mcMS**14)/mbkin**14 + (75253*mcMS**16)/mbkin**16 + 
               (7200*mcMS**18)/mbkin**18)*q_cut**7)/mbkin**14 - 
            ((-6645 - (78502*mcMS**2)/mbkin**2 - (8337*mcMS**4)/mbkin**4 + 
               (664936*mcMS**6)/mbkin**6 + (1551613*mcMS**8)/mbkin**8 + 
               (1896026*mcMS**10)/mbkin**10 + (1248611*mcMS**12)/mbkin**12 + 
               (395792*mcMS**14)/mbkin**14 + (54090*mcMS**16)/mbkin**16)*q_cut**8)/
             mbkin**16 + (2*(-1760 - (16554*mcMS**2)/mbkin**2 + (17955*mcMS**4)/
                mbkin**4 + (167540*mcMS**6)/mbkin**6 + (275800*mcMS**8)/mbkin**8 + 
               (197764*mcMS**10)/mbkin**10 + (71587*mcMS**12)/mbkin**12 + 
               (16560*mcMS**14)/mbkin**14)*q_cut**9)/mbkin**18 - 
            ((-777 - (4296*mcMS**2)/mbkin**2 + (18684*mcMS**4)/mbkin**4 + 
               (39826*mcMS**6)/mbkin**6 + (6697*mcMS**8)/mbkin**8 + 
               (6210*mcMS**10)/mbkin**10 + (8946*mcMS**12)/mbkin**12)*q_cut**10)/
             mbkin**20 - (2*(-136 - (442*mcMS**2)/mbkin**2 - (79*mcMS**4)/
                mbkin**4 + (5286*mcMS**6)/mbkin**6 + (5739*mcMS**8)/mbkin**8 + 
               (2448*mcMS**10)/mbkin**10)*q_cut**11)/mbkin**22 + 
            (5*(-83 - (138*mcMS**2)/mbkin**2 + (735*mcMS**4)/mbkin**4 + 
               (1766*mcMS**6)/mbkin**6 + (1494*mcMS**8)/mbkin**8)*q_cut**12)/
             mbkin**24 + (200*(mbkin**6 + mbkin**4*mcMS**2 - 14*mbkin**2*mcMS**4 - 
               18*mcMS**6)*q_cut**13)/mbkin**32 - (35*(mbkin**4 - 18*mcMS**4)*q_cut**14)/
             mbkin**32)*rE + ((-1 + mcMS**2/mbkin**2)**2 - 
            (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
           ((5040*mcMS**2*muG**2)/mbkin**2 - (17868*mcMS**4*muG**2)/mbkin**4 - 
            (215088*mcMS**6*muG**2)/mbkin**6 + (1321848*mcMS**8*muG**2)/mbkin**8 - 
            (532800*mcMS**10*muG**2)/mbkin**10 + (2895012*mcMS**12*muG**2)/
             mbkin**12 + (26575200*mcMS**14*muG**2)/mbkin**14 - 
            (40119408*mcMS**16*muG**2)/mbkin**16 - (19286928*mcMS**18*muG**2)/
             mbkin**18 + (31501980*mcMS**20*muG**2)/mbkin**20 + 
            (485328*mcMS**22*muG**2)/mbkin**22 - (2758152*mcMS**24*muG**2)/
             mbkin**24 + (108576*mcMS**26*muG**2)/mbkin**26 + 
            (37260*mcMS**28*muG**2)/mbkin**28 + (1680*mcMS**2*muG*mupi)/mbkin**2 - 
            (15972*mcMS**4*muG*mupi)/mbkin**4 - (151440*mcMS**6*muG*mupi)/
             mbkin**6 + (1496040*mcMS**8*muG*mupi)/mbkin**8 - 
            (734400*mcMS**10*muG*mupi)/mbkin**10 - (18498132*mcMS**12*muG*mupi)/
             mbkin**12 + (1055520*mcMS**14*muG*mupi)/mbkin**14 + 
            (33922224*mcMS**16*muG*mupi)/mbkin**16 + (353232*mcMS**18*muG*mupi)/
             mbkin**18 - (17921580*mcMS**20*muG*mupi)/mbkin**20 - 
            (528528*mcMS**22*muG*mupi)/mbkin**22 + (1024872*mcMS**24*muG*mupi)/
             mbkin**24 + (3936*mcMS**26*muG*mupi)/mbkin**26 - 
            (7452*mcMS**28*muG*mupi)/mbkin**28 - (6720*mcMS**2*muG**2*q_cut)/
             mbkin**4 - (1920*mcMS**4*muG**2*q_cut)/mbkin**6 + 
            (70728*mcMS**6*muG**2*q_cut)/mbkin**8 + (1722336*mcMS**8*muG**2*q_cut)/
             mbkin**10 - (391224*mcMS**10*muG**2*q_cut)/mbkin**12 - 
            (29557776*mcMS**12*muG**2*q_cut)/mbkin**14 - (66936864*mcMS**14*muG**2*
              q_cut)/mbkin**16 - (114608208*mcMS**16*muG**2*q_cut)/mbkin**18 - 
            (53646816*mcMS**18*muG**2*q_cut)/mbkin**20 + (17158224*mcMS**20*muG**2*
              q_cut)/mbkin**22 + (7600536*mcMS**22*muG**2*q_cut)/mbkin**24 - 
            (752976*mcMS**24*muG**2*q_cut)/mbkin**26 - (150120*mcMS**26*muG**2*q_cut)/
             mbkin**28 - (6720*mcMS**2*muG*mupi*q_cut)/mbkin**4 + 
            (42240*mcMS**4*muG*mupi*q_cut)/mbkin**6 + (554712*mcMS**6*muG*mupi*q_cut)/
             mbkin**8 - (3011040*mcMS**8*muG*mupi*q_cut)/mbkin**10 - 
            (8769960*mcMS**10*muG*mupi*q_cut)/mbkin**12 + 
            (32194512*mcMS**12*muG*mupi*q_cut)/mbkin**14 + 
            (98364960*mcMS**14*muG*mupi*q_cut)/mbkin**16 + 
            (99688464*mcMS**16*muG*mupi*q_cut)/mbkin**18 + 
            (30618720*mcMS**18*muG*mupi*q_cut)/mbkin**20 - 
            (7389840*mcMS**20*muG*mupi*q_cut)/mbkin**22 - 
            (2901432*mcMS**22*muG*mupi*q_cut)/mbkin**24 + 
            (86160*mcMS**24*muG*mupi*q_cut)/mbkin**26 + (30024*mcMS**26*muG*mupi*
              q_cut)/mbkin**28 - (6720*mcMS**2*muG**2*q_cut**2)/mbkin**6 + 
            (33600*mcMS**4*muG**2*q_cut**2)/mbkin**8 + (469416*mcMS**6*muG**2*q_cut**2)/
             mbkin**10 - (866592*mcMS**8*muG**2*q_cut**2)/mbkin**12 - 
            (10653072*mcMS**10*muG**2*q_cut**2)/mbkin**14 - 
            (9608352*mcMS**12*muG**2*q_cut**2)/mbkin**16 + 
            (14150016*mcMS**14*muG**2*q_cut**2)/mbkin**18 + 
            (2844480*mcMS**16*muG**2*q_cut**2)/mbkin**20 - 
            (13364592*mcMS**18*muG**2*q_cut**2)/mbkin**22 - 
            (4050912*mcMS**20*muG**2*q_cut**2)/mbkin**24 + 
            (943128*mcMS**22*muG**2*q_cut**2)/mbkin**26 + (151200*mcMS**24*muG**2*
              q_cut**2)/mbkin**28 + (6720*mcMS**2*muG*mupi*q_cut**2)/mbkin**6 - 
            (20160*mcMS**4*muG*mupi*q_cut**2)/mbkin**8 - (408456*mcMS**6*muG*mupi*
              q_cut**2)/mbkin**10 + (674208*mcMS**8*muG*mupi*q_cut**2)/mbkin**12 + 
            (6670800*mcMS**10*muG*mupi*q_cut**2)/mbkin**14 + 
            (5097888*mcMS**12*muG*mupi*q_cut**2)/mbkin**16 - 
            (2860416*mcMS**14*muG*mupi*q_cut**2)/mbkin**18 + 
            (3660480*mcMS**16*muG*mupi*q_cut**2)/mbkin**20 + 
            (6068016*mcMS**18*muG*mupi*q_cut**2)/mbkin**22 + 
            (1275744*mcMS**20*muG*mupi*q_cut**2)/mbkin**24 - 
            (176184*mcMS**22*muG*mupi*q_cut**2)/mbkin**26 - 
            (30240*mcMS**24*muG*mupi*q_cut**2)/mbkin**28 - (6720*mcMS**2*muG**2*q_cut**3)/
             mbkin**8 - (29304*mcMS**4*muG**2*q_cut**3)/mbkin**10 + 
            (741624*mcMS**6*muG**2*q_cut**3)/mbkin**12 - (2216208*mcMS**8*muG**2*q_cut**3)/
             mbkin**14 - (30858432*mcMS**10*muG**2*q_cut**3)/mbkin**16 - 
            (65947824*mcMS**12*muG**2*q_cut**3)/mbkin**18 - 
            (68148864*mcMS**14*muG**2*q_cut**3)/mbkin**20 - 
            (32704464*mcMS**16*muG**2*q_cut**3)/mbkin**22 - 
            (2054112*mcMS**18*muG**2*q_cut**3)/mbkin**24 + 
            (1490184*mcMS**20*muG**2*q_cut**3)/mbkin**26 + 
            (150120*mcMS**22*muG**2*q_cut**3)/mbkin**28 + (6720*mcMS**2*muG*mupi*
              q_cut**3)/mbkin**8 - (30696*mcMS**4*muG*mupi*q_cut**3)/mbkin**10 - 
            (616728*mcMS**6*muG*mupi*q_cut**3)/mbkin**12 + 
            (1846224*mcMS**8*muG*mupi*q_cut**3)/mbkin**14 + 
            (16026048*mcMS**10*muG*mupi*q_cut**3)/mbkin**16 + 
            (29689968*mcMS**12*muG*mupi*q_cut**3)/mbkin**18 + 
            (29566464*mcMS**14*muG*mupi*q_cut**3)/mbkin**20 + 
            (14890128*mcMS**16*muG*mupi*q_cut**3)/mbkin**22 + 
            (2012256*mcMS**18*muG*mupi*q_cut**3)/mbkin**24 - 
            (221160*mcMS**20*muG*mupi*q_cut**3)/mbkin**26 - 
            (30024*mcMS**22*muG*mupi*q_cut**3)/mbkin**28 + 
            (43680*mcMS**2*muG**2*q_cut**4)/mbkin**10 + (21372*mcMS**4*muG**2*q_cut**4)/
             mbkin**12 - (2449776*mcMS**6*muG**2*q_cut**4)/mbkin**14 - 
            (764592*mcMS**8*muG**2*q_cut**4)/mbkin**16 + (31572960*mcMS**10*muG**2*
              q_cut**4)/mbkin**18 + (62188536*mcMS**12*muG**2*q_cut**4)/mbkin**20 + 
            (35488272*mcMS**14*muG**2*q_cut**4)/mbkin**22 - 
            (47280*mcMS**16*muG**2*q_cut**4)/mbkin**24 - (3628512*mcMS**18*muG**2*
              q_cut**4)/mbkin**26 - (377460*mcMS**20*muG**2*q_cut**4)/mbkin**28 - 
            (16800*mcMS**2*muG*mupi*q_cut**4)/mbkin**10 + 
            (35508*mcMS**4*muG*mupi*q_cut**4)/mbkin**12 + (1075248*mcMS**6*muG*mupi*
              q_cut**4)/mbkin**14 - (884880*mcMS**8*muG*mupi*q_cut**4)/mbkin**16 - 
            (16597728*mcMS**10*muG*mupi*q_cut**4)/mbkin**18 - 
            (28130712*mcMS**12*muG*mupi*q_cut**4)/mbkin**20 - 
            (14960208*mcMS**14*muG*mupi*q_cut**4)/mbkin**22 - 
            (1225488*mcMS**16*muG*mupi*q_cut**4)/mbkin**24 + 
            (675168*mcMS**18*muG*mupi*q_cut**4)/mbkin**26 + 
            (75492*mcMS**20*muG*mupi*q_cut**4)/mbkin**28 - 
            (20160*mcMS**2*muG**2*q_cut**5)/mbkin**12 + (29568*mcMS**4*muG**2*q_cut**5)/
             mbkin**14 + (825600*mcMS**6*muG**2*q_cut**5)/mbkin**16 - 
            (2256528*mcMS**8*muG**2*q_cut**5)/mbkin**18 - (15292416*mcMS**10*muG**2*
              q_cut**5)/mbkin**20 - (16239504*mcMS**12*muG**2*q_cut**5)/mbkin**22 - 
            (2731920*mcMS**14*muG**2*q_cut**5)/mbkin**24 + 
            (1086624*mcMS**16*muG**2*q_cut**5)/mbkin**26 + 
            (156240*mcMS**18*muG**2*q_cut**5)/mbkin**28 + (6720*mcMS**2*muG*mupi*
              q_cut**5)/mbkin**12 - (16128*mcMS**4*muG*mupi*q_cut**5)/mbkin**14 - 
            (317952*mcMS**6*muG*mupi*q_cut**5)/mbkin**16 + 
            (462672*mcMS**8*muG*mupi*q_cut**5)/mbkin**18 + 
            (3914496*mcMS**10*muG*mupi*q_cut**5)/mbkin**20 + 
            (3555792*mcMS**12*muG*mupi*q_cut**5)/mbkin**22 + 
            (480720*mcMS**14*muG*mupi*q_cut**5)/mbkin**24 - 
            (223776*mcMS**16*muG*mupi*q_cut**5)/mbkin**26 - 
            (31248*mcMS**18*muG*mupi*q_cut**5)/mbkin**28 - 
            (33600*mcMS**2*muG**2*q_cut**6)/mbkin**14 - (30240*mcMS**4*muG**2*q_cut**6)/
             mbkin**16 + (2204544*mcMS**6*muG**2*q_cut**6)/mbkin**18 + 
            (9292416*mcMS**8*muG**2*q_cut**6)/mbkin**20 + (13925232*mcMS**10*muG**2*
              q_cut**6)/mbkin**22 + (8644032*mcMS**12*muG**2*q_cut**6)/mbkin**24 + 
            (2204496*mcMS**14*muG**2*q_cut**6)/mbkin**26 + 
            (141120*mcMS**16*muG**2*q_cut**6)/mbkin**28 + (6720*mcMS**2*muG*mupi*
              q_cut**6)/mbkin**14 + (3360*mcMS**4*muG*mupi*q_cut**6)/mbkin**16 - 
            (436224*mcMS**6*muG*mupi*q_cut**6)/mbkin**18 - 
            (1708800*mcMS**8*muG*mupi*q_cut**6)/mbkin**20 - 
            (2377776*mcMS**10*muG*mupi*q_cut**6)/mbkin**22 - 
            (1596096*mcMS**12*muG*mupi*q_cut**6)/mbkin**24 - 
            (431760*mcMS**14*muG*mupi*q_cut**6)/mbkin**26 - 
            (28224*mcMS**16*muG*mupi*q_cut**6)/mbkin**28 + 
            (33600*mcMS**2*muG**2*q_cut**7)/mbkin**16 - (14640*mcMS**4*muG**2*q_cut**7)/
             mbkin**18 - (2029344*mcMS**6*muG**2*q_cut**7)/mbkin**20 - 
            (5885232*mcMS**8*muG**2*q_cut**7)/mbkin**22 - (5477280*mcMS**10*muG**2*
              q_cut**7)/mbkin**24 - (1841328*mcMS**12*muG**2*q_cut**7)/mbkin**26 - 
            (162000*mcMS**14*muG**2*q_cut**7)/mbkin**28 - (6720*mcMS**2*muG*mupi*
              q_cut**7)/mbkin**16 + (9840*mcMS**4*muG*mupi*q_cut**7)/mbkin**18 + 
            (378144*mcMS**6*muG*mupi*q_cut**7)/mbkin**20 + 
            (1128048*mcMS**8*muG*mupi*q_cut**7)/mbkin**22 + 
            (1040928*mcMS**10*muG*mupi*q_cut**7)/mbkin**24 + 
            (353904*mcMS**12*muG*mupi*q_cut**7)/mbkin**26 + 
            (32400*mcMS**14*muG*mupi*q_cut**7)/mbkin**28 - (8400*mcMS**2*muG**2*q_cut**8)/
             mbkin**18 + (17964*mcMS**4*muG**2*q_cut**8)/mbkin**20 + 
            (495648*mcMS**6*muG**2*q_cut**8)/mbkin**22 + (865080*mcMS**8*muG**2*q_cut**8)/
             mbkin**24 + (354816*mcMS**10*muG**2*q_cut**8)/mbkin**26 + 
            (75780*mcMS**12*muG**2*q_cut**8)/mbkin**28 + (1680*mcMS**2*muG*mupi*q_cut**8)/
             mbkin**18 - (12924*mcMS**4*muG*mupi*q_cut**8)/mbkin**20 - 
            (67104*mcMS**6*muG*mupi*q_cut**8)/mbkin**22 - 
            (100824*mcMS**8*muG*mupi*q_cut**8)/mbkin**24 - 
            (57600*mcMS**10*muG*mupi*q_cut**8)/mbkin**26 - 
            (15156*mcMS**12*muG*mupi*q_cut**8)/mbkin**28 - (4032*mcMS**4*muG**2*q_cut**9)/
             mbkin**22 + (31608*mcMS**6*muG**2*q_cut**9)/mbkin**24 + 
            (72432*mcMS**8*muG**2*q_cut**9)/mbkin**26 - (17640*mcMS**10*muG**2*q_cut**9)/
             mbkin**28 + (4032*mcMS**4*muG*mupi*q_cut**9)/mbkin**22 - 
            (15000*mcMS**6*muG*mupi*q_cut**9)/mbkin**24 - 
            (13488*mcMS**8*muG*mupi*q_cut**9)/mbkin**26 + 
            (3528*mcMS**10*muG*mupi*q_cut**9)/mbkin**28 - 
            (21600*mcMS**4*muG**2*q_cut**10)/mbkin**24 - (60840*mcMS**6*muG**2*q_cut**10)/
             mbkin**26 - (21600*mcMS**8*muG**2*q_cut**10)/mbkin**28 + 
            (4320*mcMS**4*muG*mupi*q_cut**10)/mbkin**24 + 
            (9480*mcMS**6*muG*mupi*q_cut**10)/mbkin**26 + 
            (4320*mcMS**8*muG*mupi*q_cut**10)/mbkin**28 + 
            (23400*mcMS**4*muG**2*q_cut**11)/mbkin**26 + (23400*mcMS**6*muG**2*q_cut**11)/
             mbkin**28 - (4680*mcMS**4*muG*mupi*q_cut**11)/mbkin**26 - 
            (4680*mcMS**6*muG*mupi*q_cut**11)/mbkin**28 - (6300*mcMS**4*muG**2*q_cut**12)/
             mbkin**28 + (1260*mcMS**4*muG*mupi*q_cut**12)/mbkin**28 - 
            24*mcMS**2*muG*((-1 + mcMS**2/mbkin**2)**2*(-140 + (1051*mcMS**2)/
                 mbkin**2 + (14862*mcMS**4)/mbkin**4 - (95997*mcMS**6)/mbkin**6 - 
                (145656*mcMS**8)/mbkin**8 + (1346196*mcMS**10)/mbkin**10 + 
                (2750088*mcMS**12)/mbkin**12 + (1327128*mcMS**14)/mbkin**14 - 
                (125268*mcMS**16)/mbkin**16 - (84199*mcMS**18)/mbkin**18 + 
                (914*mcMS**20)/mbkin**20 + (621*mcMS**22)/mbkin**22) - 
              (2*(-280 + (1760*mcMS**2)/mbkin**2 + (23113*mcMS**4)/mbkin**4 - 
                 (125460*mcMS**6)/mbkin**6 - (365415*mcMS**8)/mbkin**8 + 
                 (1341438*mcMS**10)/mbkin**10 + (4098540*mcMS**12)/mbkin**12 + 
                 (4153686*mcMS**14)/mbkin**14 + (1275780*mcMS**16)/mbkin**16 - 
                 (307910*mcMS**18)/mbkin**18 - (120893*mcMS**20)/mbkin**20 + 
                 (3590*mcMS**22)/mbkin**22 + (1251*mcMS**24)/mbkin**24)*q_cut)/mbkin**
                2 + (2*(-280 + (840*mcMS**2)/mbkin**2 + (17019*mcMS**4)/
                  mbkin**4 - (28092*mcMS**6)/mbkin**6 - (277950*mcMS**8)/
                  mbkin**8 - (212412*mcMS**10)/mbkin**10 + (119184*mcMS**12)/
                  mbkin**12 - (152520*mcMS**14)/mbkin**14 - (252834*mcMS**16)/
                  mbkin**16 - (53156*mcMS**18)/mbkin**18 + (7341*mcMS**20)/
                  mbkin**20 + (1260*mcMS**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
              (2*(-280 + (1279*mcMS**2)/mbkin**2 + (25697*mcMS**4)/mbkin**4 - 
                 (76926*mcMS**6)/mbkin**6 - (667752*mcMS**8)/mbkin**8 - 
                 (1237082*mcMS**10)/mbkin**10 - (1231936*mcMS**12)/mbkin**12 - 
                 (620422*mcMS**14)/mbkin**14 - (83844*mcMS**16)/mbkin**16 + 
                 (9215*mcMS**18)/mbkin**18 + (1251*mcMS**20)/mbkin**20)*q_cut**3)/
               mbkin**6 + ((1400 - (2959*mcMS**2)/mbkin**2 - (89604*mcMS**4)/
                  mbkin**4 + (73740*mcMS**6)/mbkin**6 + (1383144*mcMS**8)/
                  mbkin**8 + (2344226*mcMS**10)/mbkin**10 + (1246684*mcMS**12)/
                  mbkin**12 + (102124*mcMS**14)/mbkin**14 - (56264*mcMS**16)/
                  mbkin**16 - (6291*mcMS**18)/mbkin**18)*q_cut**4)/mbkin**8 + 
              (4*(-140 + (336*mcMS**2)/mbkin**2 + (6624*mcMS**4)/mbkin**4 - 
                 (9639*mcMS**6)/mbkin**6 - (81552*mcMS**8)/mbkin**8 - 
                 (74079*mcMS**10)/mbkin**10 - (10015*mcMS**12)/mbkin**12 + 
                 (4662*mcMS**14)/mbkin**14 + (651*mcMS**16)/mbkin**16)*q_cut**5)/
               mbkin**10 + (4*(-140 - (70*mcMS**2)/mbkin**2 + (9088*mcMS**4)/
                  mbkin**4 + (35600*mcMS**6)/mbkin**6 + (49537*mcMS**8)/mbkin**8 + 
                 (33252*mcMS**10)/mbkin**10 + (8995*mcMS**12)/mbkin**12 + 
                 (588*mcMS**14)/mbkin**14)*q_cut**6)/mbkin**12 - 
              (4*(-140 + (205*mcMS**2)/mbkin**2 + (7878*mcMS**4)/mbkin**4 + 
                 (23501*mcMS**6)/mbkin**6 + (21686*mcMS**8)/mbkin**8 + 
                 (7373*mcMS**10)/mbkin**10 + (675*mcMS**12)/mbkin**12)*q_cut**7)/
               mbkin**14 + ((-140 + (1077*mcMS**2)/mbkin**2 + (5592*mcMS**4)/
                  mbkin**4 + (8402*mcMS**6)/mbkin**6 + (4800*mcMS**8)/mbkin**8 + 
                 (1263*mcMS**10)/mbkin**10)*q_cut**8)/mbkin**16 - (2*mcMS**2*
                (168 - (625*mcMS**2)/mbkin**2 - (562*mcMS**4)/mbkin**4 + 
                 (147*mcMS**6)/mbkin**6)*q_cut**9)/mbkin**20 - (10*mcMS**2*
                (36 + (79*mcMS**2)/mbkin**2 + (36*mcMS**4)/mbkin**4)*q_cut**10)/mbkin**
                22 + (390*mcMS**2*(mbkin**2 + mcMS**2)*q_cut**11)/mbkin**26 - 
              (105*mcMS**2*q_cut**12)/mbkin**26) - 8*(-2*(-1 + mcMS**2/mbkin**2)**2*(
                -172 - (687*mcMS**2)/mbkin**2 + (38656*mcMS**4)/mbkin**4 - 
                (19884*mcMS**6)/mbkin**6 - (1069059*mcMS**8)/mbkin**8 - 
                (221682*mcMS**10)/mbkin**10 + (3433464*mcMS**12)/mbkin**12 + 
                (3472584*mcMS**14)/mbkin**14 + (660450*mcMS**16)/mbkin**16 - 
                (237695*mcMS**18)/mbkin**18 - (70200*mcMS**20)/mbkin**20 + 
                (1124*mcMS**22)/mbkin**22 + (621*mcMS**24)/mbkin**24) + 
              ((-1388 - (9012*mcMS**2)/mbkin**2 + (230491*mcMS**4)/mbkin**4 + 
                 (355108*mcMS**6)/mbkin**6 - (4608237*mcMS**8)/mbkin**8 - 
                 (5679876*mcMS**10)/mbkin**10 + (12773730*mcMS**12)/mbkin**12 + 
                 (27482208*mcMS**14)/mbkin**14 + (17583222*mcMS**16)/mbkin**16 + 
                 (1783916*mcMS**18)/mbkin**18 - (1637757*mcMS**20)/mbkin**20 - 
                 (392644*mcMS**22)/mbkin**22 + (15395*mcMS**24)/mbkin**24 + 
                 (5004*mcMS**26)/mbkin**26)*q_cut)/mbkin**2 + 
              ((1400 + (15400*mcMS**2)/mbkin**2 - (105765*mcMS**4)/mbkin**4 - 
                 (531075*mcMS**6)/mbkin**6 + (81882*mcMS**8)/mbkin**8 + 
                 (912366*mcMS**10)/mbkin**10 - (562032*mcMS**12)/mbkin**12 + 
                 (573936*mcMS**14)/mbkin**14 + (2274534*mcMS**16)/mbkin**16 + 
                 (1220434*mcMS**18)/mbkin**18 + (147341*mcMS**20)/mbkin**20 - 
                 (31701*mcMS**22)/mbkin**22 - (5040*mcMS**24)/mbkin**24)*q_cut**2)/
               mbkin**4 - (2*(-694 - (7282*mcMS**2)/mbkin**2 + (88915*mcMS**4)/
                  mbkin**4 + (527698*mcMS**6)/mbkin**6 + (227625*mcMS**8)/
                  mbkin**8 - (1022684*mcMS**10)/mbkin**10 - (2173541*mcMS**12)/
                  mbkin**12 - (2343456*mcMS**14)/mbkin**14 - (1034603*mcMS**16)/
                  mbkin**16 - (133978*mcMS**18)/mbkin**18 + (15034*mcMS**20)/
                  mbkin**20 + (2502*mcMS**22)/mbkin**22)*q_cut**3)/mbkin**6 + 
              (2*(-1747 - (24294*mcMS**2)/mbkin**2 + (68590*mcMS**4)/mbkin**4 + 
                 (610257*mcMS**6)/mbkin**6 + (258216*mcMS**8)/mbkin**8 - 
                 (1948666*mcMS**10)/mbkin**10 - (2583439*mcMS**12)/mbkin**12 - 
                 (1037823*mcMS**14)/mbkin**14 - (42243*mcMS**16)/mbkin**16 + 
                 (52154*mcMS**18)/mbkin**18 + (6291*mcMS**20)/mbkin**20)*q_cut**4)/
               mbkin**8 + (2*(728 + (9492*mcMS**2)/mbkin**2 - (19180*mcMS**4)/
                  mbkin**4 - (127832*mcMS**6)/mbkin**6 + (53767*mcMS**8)/
                  mbkin**8 + (381596*mcMS**10)/mbkin**10 + (214536*mcMS**12)/
                  mbkin**12 + (1364*mcMS**14)/mbkin**14 - (19243*mcMS**16)/
                  mbkin**16 - (2604*mcMS**18)/mbkin**18)*q_cut**5)/mbkin**10 - 
              (2*(-644 - (13636*mcMS**2)/mbkin**2 - (41804*mcMS**4)/mbkin**4 - 
                 (4064*mcMS**6)/mbkin**6 + (108175*mcMS**8)/mbkin**8 + 
                 (179041*mcMS**10)/mbkin**10 + (116257*mcMS**12)/mbkin**12 + 
                 (30835*mcMS**14)/mbkin**14 + (2352*mcMS**16)/mbkin**16)*q_cut**6)/
               mbkin**12 + (2*(-760 - (12180*mcMS**2)/mbkin**2 - (26033*mcMS**4)/
                  mbkin**4 + (24832*mcMS**6)/mbkin**6 + (105795*mcMS**8)/
                  mbkin**8 + (84468*mcMS**10)/mbkin**10 + (25418*mcMS**12)/
                  mbkin**12 + (2700*mcMS**14)/mbkin**14)*q_cut**7)/mbkin**14 - 
              (2*(-386 - (2343*mcMS**2)/mbkin**2 - (587*mcMS**4)/mbkin**4 + 
                 (9444*mcMS**6)/mbkin**6 + (9577*mcMS**8)/mbkin**8 + 
                 (3676*mcMS**10)/mbkin**10 + (1263*mcMS**12)/mbkin**12)*q_cut**8)/
               mbkin**16 + ((-196 + (588*mcMS**2)/mbkin**2 + (1549*mcMS**4)/
                  mbkin**4 - (2212*mcMS**6)/mbkin**6 - (2053*mcMS**8)/mbkin**8 + 
                 (588*mcMS**10)/mbkin**10)*q_cut**9)/mbkin**18 + 
              (5*(-48 - (96*mcMS**2)/mbkin**2 + (41*mcMS**4)/mbkin**4 + 
                 (247*mcMS**6)/mbkin**6 + (144*mcMS**8)/mbkin**8)*q_cut**10)/mbkin**
                20 - (20*(-13 - (13*mcMS**2)/mbkin**2 + (34*mcMS**4)/mbkin**4 + 
                 (39*mcMS**6)/mbkin**6)*q_cut**11)/mbkin**22 - 
              (70*(mbkin**4 - 3*mcMS**4)*q_cut**12)/mbkin**28)*rG + 
            8*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*(201 - (2594*mcMS**2)/mbkin**2 - 
                 (18020*mcMS**4)/mbkin**4 + (686586*mcMS**6)/mbkin**6 - 
                 (484473*mcMS**8)/mbkin**8 - (24524568*mcMS**10)/mbkin**10 - 
                 (51017352*mcMS**12)/mbkin**12 - (29833656*mcMS**14)/mbkin**14 - 
                 (1099137*mcMS**16)/mbkin**16 + (793962*mcMS**18)/mbkin**18 - 
                 (274084*mcMS**20)/mbkin**20 - (9490*mcMS**22)/mbkin**22 + 
                 (3105*mcMS**24)/mbkin**24)) + (2*(411 - (4441*mcMS**2)/
                  mbkin**2 - (47413*mcMS**4)/mbkin**4 + (998110*mcMS**6)/
                  mbkin**6 + (1955322*mcMS**8)/mbkin**8 - (24020691*mcMS**10)/
                  mbkin**10 - (77841426*mcMS**12)/mbkin**12 - (84248520*mcMS**14)/
                  mbkin**14 - (31464921*mcMS**16)/mbkin**16 + (2471757*mcMS**18)/
                  mbkin**18 + (1072279*mcMS**20)/mbkin**20 - (424934*mcMS**22)/
                  mbkin**22 - (10828*mcMS**24)/mbkin**24 + (6255*mcMS**26)/
                  mbkin**26)*q_cut)/mbkin**2 - (2*(420 - (2800*mcMS**2)/mbkin**2 - 
                 (64131*mcMS**4)/mbkin**4 + (406035*mcMS**6)/mbkin**6 + 
                 (2915250*mcMS**8)/mbkin**8 + (3976458*mcMS**10)/mbkin**10 + 
                 (2350824*mcMS**12)/mbkin**12 + (4693344*mcMS**14)/mbkin**14 + 
                 (3558342*mcMS**16)/mbkin**16 + (48462*mcMS**18)/mbkin**18 - 
                 (268445*mcMS**20)/mbkin**20 + (9861*mcMS**22)/mbkin**22 + 
                 (6300*mcMS**24)/mbkin**24)*q_cut**2)/mbkin**4 - 
              (2*(411 - (2797*mcMS**2)/mbkin**2 - (61931*mcMS**4)/mbkin**4 + 
                 (744527*mcMS**6)/mbkin**6 + (5541108*mcMS**8)/mbkin**8 + 
                 (10489326*mcMS**10)/mbkin**10 + (10608076*mcMS**12)/mbkin**12 + 
                 (5830586*mcMS**14)/mbkin**14 + (187913*mcMS**16)/mbkin**16 - 
                 (376185*mcMS**18)/mbkin**18 + (8423*mcMS**20)/mbkin**20 + 
                 (6255*mcMS**22)/mbkin**22)*q_cut**3)/mbkin**6 + 
              ((2091 - (7688*mcMS**2)/mbkin**2 - (351965*mcMS**4)/mbkin**4 + 
                 (922140*mcMS**6)/mbkin**6 + (11699682*mcMS**8)/mbkin**8 + 
                 (22377552*mcMS**10)/mbkin**10 + (11839006*mcMS**12)/mbkin**12 - 
                 (747556*mcMS**14)/mbkin**14 - (969997*mcMS**16)/mbkin**16 + 
                 (118544*mcMS**18)/mbkin**18 + (31455*mcMS**20)/mbkin**20)*q_cut**4)/
               mbkin**8 - (4*(231 - (1001*mcMS**2)/mbkin**2 - (36378*mcMS**4)/
                  mbkin**4 + (32226*mcMS**6)/mbkin**6 + (485535*mcMS**8)/
                  mbkin**8 + (473268*mcMS**10)/mbkin**10 - (63132*mcMS**12)/
                  mbkin**12 - (76864*mcMS**14)/mbkin**14 + (10500*mcMS**16)/
                  mbkin**16 + (3255*mcMS**18)/mbkin**18)*q_cut**5)/mbkin**10 - 
              (4*(168 + (532*mcMS**2)/mbkin**2 - (37666*mcMS**4)/mbkin**4 - 
                 (173612*mcMS**6)/mbkin**6 - (257329*mcMS**8)/mbkin**8 - 
                 (157955*mcMS**10)/mbkin**10 + (6303*mcMS**12)/mbkin**12 + 
                 (24871*mcMS**14)/mbkin**14 + (2940*mcMS**16)/mbkin**16)*q_cut**6)/
               mbkin**12 + (4*(255 - (245*mcMS**2)/mbkin**2 - (39116*mcMS**4)/
                  mbkin**4 - (138249*mcMS**6)/mbkin**6 - (129466*mcMS**8)/
                  mbkin**8 - (11635*mcMS**10)/mbkin**10 + (18713*mcMS**12)/
                  mbkin**12 + (3375*mcMS**14)/mbkin**14)*q_cut**7)/mbkin**14 + 
              ((-843 + (2036*mcMS**2)/mbkin**2 + (42441*mcMS**4)/mbkin**4 + 
                 (95556*mcMS**6)/mbkin**6 + (47845*mcMS**8)/mbkin**8 - 
                 (5364*mcMS**10)/mbkin**10 - (6315*mcMS**12)/mbkin**12)*q_cut**8)/
               mbkin**16 + (2*(147 - (441*mcMS**2)/mbkin**2 - (111*mcMS**4)/
                  mbkin**4 - (2794*mcMS**6)/mbkin**6 - (2452*mcMS**8)/mbkin**8 + 
                 (735*mcMS**10)/mbkin**10)*q_cut**9)/mbkin**18 + 
              (10*(36 + (72*mcMS**2)/mbkin**2 + (93*mcMS**4)/mbkin**4 + 
                 (223*mcMS**6)/mbkin**6 + (180*mcMS**8)/mbkin**8)*q_cut**10)/mbkin**
                20 - (30*(13 + (13*mcMS**2)/mbkin**2 + (45*mcMS**4)/mbkin**4 + 
                 (65*mcMS**6)/mbkin**6)*q_cut**11)/mbkin**22 + (105*(mbkin**4 + 
                 5*mcMS**4)*q_cut**12)/mbkin**28)*rhoD + 1680*sB + 
            (30800*mcMS**2*sB)/mbkin**2 - (731384*mcMS**4*sB)/mbkin**4 - 
            (2179088*mcMS**6*sB)/mbkin**6 + (34731536*mcMS**8*sB)/mbkin**8 + 
            (58508064*mcMS**10*sB)/mbkin**10 - (80713224*mcMS**12*sB)/mbkin**12 - 
            (84871392*mcMS**14*sB)/mbkin**14 + (32328240*mcMS**16*sB)/mbkin**16 + 
            (28088784*mcMS**18*sB)/mbkin**18 + (16871416*mcMS**20*sB)/mbkin**20 + 
            (472304*mcMS**22*sB)/mbkin**22 - (2513104*mcMS**24*sB)/mbkin**24 - 
            (49472*mcMS**26*sB)/mbkin**26 + (24840*mcMS**28*sB)/mbkin**28 - 
            (6720*q_cut*sB)/mbkin**2 - (156800*mcMS**2*q_cut*sB)/mbkin**4 + 
            (1797304*mcMS**4*q_cut*sB)/mbkin**6 + (14779760*mcMS**6*q_cut*sB)/
             mbkin**8 - (39450888*mcMS**8*q_cut*sB)/mbkin**10 - 
            (307704144*mcMS**10*q_cut*sB)/mbkin**12 - (510425808*mcMS**12*q_cut*sB)/
             mbkin**14 - (358901184*mcMS**14*q_cut*sB)/mbkin**16 - 
            (115602576*mcMS**16*q_cut*sB)/mbkin**18 - (17262528*mcMS**18*q_cut*sB)/
             mbkin**20 + (16659608*mcMS**20*q_cut*sB)/mbkin**22 + 
            (7228112*mcMS**22*q_cut*sB)/mbkin**24 - (125096*mcMS**24*q_cut*sB)/
             mbkin**26 - (100080*mcMS**26*q_cut*sB)/mbkin**28 + (6720*q_cut**2*sB)/
             mbkin**4 + (190400*mcMS**2*q_cut**2*sB)/mbkin**6 - 
            (435336*mcMS**4*q_cut**2*sB)/mbkin**8 - (10175112*mcMS**6*q_cut**2*sB)/
             mbkin**10 - (23688624*mcMS**8*q_cut**2*sB)/mbkin**12 - 
            (9373872*mcMS**10*q_cut**2*sB)/mbkin**14 - (7295808*mcMS**12*q_cut**2*sB)/
             mbkin**16 - (21152640*mcMS**14*q_cut**2*sB)/mbkin**18 - 
            (20304528*mcMS**16*q_cut**2*sB)/mbkin**20 - (14239056*mcMS**18*q_cut**2*sB)/
             mbkin**22 - (3173624*mcMS**20*q_cut**2*sB)/mbkin**24 + 
            (434760*mcMS**22*q_cut**2*sB)/mbkin**26 + (100800*mcMS**24*q_cut**2*sB)/
             mbkin**28 + (6720*q_cut**3*sB)/mbkin**6 + (183680*mcMS**2*q_cut**3*sB)/
             mbkin**8 - (1007072*mcMS**4*q_cut**3*sB)/mbkin**10 - 
            (18628912*mcMS**6*q_cut**3*sB)/mbkin**12 - (56964144*mcMS**8*q_cut**3*sB)/
             mbkin**14 - (82334208*mcMS**10*q_cut**3*sB)/mbkin**16 - 
            (88682576*mcMS**12*q_cut**3*sB)/mbkin**18 - (62325760*mcMS**14*q_cut**3*sB)/
             mbkin**20 - (28664272*mcMS**16*q_cut**3*sB)/mbkin**22 - 
            (5259072*mcMS**18*q_cut**3*sB)/mbkin**24 + (468464*mcMS**20*q_cut**3*sB)/
             mbkin**26 + (100080*mcMS**22*q_cut**3*sB)/mbkin**28 - 
            (16800*q_cut**4*sB)/mbkin**8 - (523040*mcMS**2*q_cut**4*sB)/mbkin**10 - 
            (545576*mcMS**4*q_cut**4*sB)/mbkin**12 + (17882352*mcMS**6*q_cut**4*sB)/
             mbkin**14 + (65571072*mcMS**8*q_cut**4*sB)/mbkin**16 + 
            (87175392*mcMS**10*q_cut**4*sB)/mbkin**18 + (63439936*mcMS**12*q_cut**4*sB)/
             mbkin**20 + (28625648*mcMS**14*q_cut**4*sB)/mbkin**22 + 
            (3146624*mcMS**16*q_cut**4*sB)/mbkin**24 - (1684576*mcMS**18*q_cut**4*sB)/
             mbkin**26 - (251640*mcMS**20*q_cut**4*sB)/mbkin**28 + 
            (6720*q_cut**5*sB)/mbkin**10 + (210560*mcMS**2*q_cut**5*sB)/mbkin**12 + 
            (422592*mcMS**4*q_cut**5*sB)/mbkin**14 - (3356544*mcMS**6*q_cut**5*sB)/
             mbkin**16 - (9863856*mcMS**8*q_cut**5*sB)/mbkin**18 - 
            (10081536*mcMS**10*q_cut**5*sB)/mbkin**20 - (6059808*mcMS**12*q_cut**5*sB)/
             mbkin**22 - (832736*mcMS**14*q_cut**5*sB)/mbkin**24 + 
            (583632*mcMS**16*q_cut**5*sB)/mbkin**26 + (104160*mcMS**18*q_cut**5*sB)/
             mbkin**28 + (6720*q_cut**6*sB)/mbkin**12 + (244160*mcMS**2*q_cut**6*sB)/
             mbkin**14 + (1627264*mcMS**4*q_cut**6*sB)/mbkin**16 + 
            (4060544*mcMS**6*q_cut**6*sB)/mbkin**18 + (5920336*mcMS**8*q_cut**6*sB)/
             mbkin**20 + (5762768*mcMS**10*q_cut**6*sB)/mbkin**22 + 
            (3621168*mcMS**12*q_cut**6*sB)/mbkin**24 + (1114736*mcMS**14*q_cut**6*sB)/
             mbkin**26 + (94080*mcMS**16*q_cut**6*sB)/mbkin**28 - 
            (6720*q_cut**7*sB)/mbkin**14 - (237440*mcMS**2*q_cut**7*sB)/mbkin**16 - 
            (1432496*mcMS**4*q_cut**7*sB)/mbkin**18 - (3091392*mcMS**6*q_cut**7*sB)/
             mbkin**20 - (3537712*mcMS**8*q_cut**7*sB)/mbkin**22 - 
            (2458816*mcMS**10*q_cut**7*sB)/mbkin**24 - (888640*mcMS**12*q_cut**7*sB)/
             mbkin**26 - (108000*mcMS**14*q_cut**7*sB)/mbkin**28 + 
            (1680*q_cut**8*sB)/mbkin**16 + (57680*mcMS**2*q_cut**8*sB)/mbkin**18 + 
            (320520*mcMS**4*q_cut**8*sB)/mbkin**20 + (406176*mcMS**6*q_cut**8*sB)/
             mbkin**22 + (206992*mcMS**8*q_cut**8*sB)/mbkin**24 + 
            (115296*mcMS**10*q_cut**8*sB)/mbkin**26 + (50520*mcMS**12*q_cut**8*sB)/
             mbkin**28 - (17016*mcMS**4*q_cut**9*sB)/mbkin**22 + 
            (42256*mcMS**6*q_cut**9*sB)/mbkin**24 + (42136*mcMS**8*q_cut**9*sB)/
             mbkin**26 - (11760*mcMS**10*q_cut**9*sB)/mbkin**28 - 
            (7800*mcMS**4*q_cut**10*sB)/mbkin**24 - (24440*mcMS**6*q_cut**10*sB)/
             mbkin**26 - (14400*mcMS**8*q_cut**10*sB)/mbkin**28 + 
            (13200*mcMS**4*q_cut**11*sB)/mbkin**26 + (15600*mcMS**6*q_cut**11*sB)/
             mbkin**28 - (4200*mcMS**4*q_cut**12*sB)/mbkin**28 - 1656*sE - 
            (17024*mcMS**2*sE)/mbkin**2 + (524648*mcMS**4*sE)/mbkin**4 + 
            (493472*mcMS**6*sE)/mbkin**6 - (19796072*mcMS**8*sE)/mbkin**8 - 
            (6705504*mcMS**10*sE)/mbkin**10 + (53596152*mcMS**12*sE)/mbkin**12 - 
            (3650880*mcMS**14*sE)/mbkin**14 - (33802440*mcMS**16*sE)/mbkin**16 + 
            (10860096*mcMS**18*sE)/mbkin**18 - (1923880*mcMS**20*sE)/mbkin**20 - 
            (1014368*mcMS**22*sE)/mbkin**22 + (1418152*mcMS**24*sE)/mbkin**24 + 
            (34208*mcMS**26*sE)/mbkin**26 - (14904*mcMS**28*sE)/mbkin**28 + 
            (6672*q_cut*sE)/mbkin**2 + (98768*mcMS**2*q_cut*sE)/mbkin**4 - 
            (1390912*mcMS**4*q_cut*sE)/mbkin**6 - (7268672*mcMS**6*q_cut*sE)/mbkin**8 + 
            (30554160*mcMS**8*q_cut*sE)/mbkin**10 + (130632048*mcMS**10*q_cut*sE)/
             mbkin**12 + (113878272*mcMS**12*q_cut*sE)/mbkin**14 + 
            (4777344*mcMS**14*q_cut*sE)/mbkin**16 - (22997904*mcMS**16*q_cut*sE)/
             mbkin**18 + (1645872*mcMS**18*q_cut*sE)/mbkin**20 - 
            (7863488*mcMS**20*q_cut*sE)/mbkin**22 - (4152128*mcMS**22*q_cut*sE)/
             mbkin**24 + (69200*mcMS**24*q_cut*sE)/mbkin**26 + 
            (60048*mcMS**26*q_cut*sE)/mbkin**28 - (6720*q_cut**2*sE)/mbkin**4 - 
            (129920*mcMS**2*q_cut**2*sE)/mbkin**6 + (497856*mcMS**4*q_cut**2*sE)/
             mbkin**8 + (5814624*mcMS**6*q_cut**2*sE)/mbkin**10 + 
            (7556928*mcMS**8*q_cut**2*sE)/mbkin**12 - (7261248*mcMS**10*q_cut**2*sE)/
             mbkin**14 - (2213760*mcMS**12*q_cut**2*sE)/mbkin**16 + 
            (16592640*mcMS**14*q_cut**2*sE)/mbkin**18 + (18923328*mcMS**16*q_cut**2*sE)/
             mbkin**20 + (10571712*mcMS**18*q_cut**2*sE)/mbkin**22 + 
            (1788608*mcMS**20*q_cut**2*sE)/mbkin**24 - (302688*mcMS**22*q_cut**2*sE)/
             mbkin**26 - (60480*mcMS**24*q_cut**2*sE)/mbkin**28 - 
            (6672*q_cut**3*sE)/mbkin**6 - (125456*mcMS**2*q_cut**3*sE)/mbkin**8 + 
            (890288*mcMS**4*q_cut**3*sE)/mbkin**10 + (10654768*mcMS**6*q_cut**3*sE)/
             mbkin**12 + (21660768*mcMS**8*q_cut**3*sE)/mbkin**14 + 
            (16149600*mcMS**10*q_cut**3*sE)/mbkin**16 + (6608480*mcMS**12*q_cut**3*sE)/
             mbkin**18 + (3438688*mcMS**14*q_cut**3*sE)/mbkin**20 + 
            (9145264*mcMS**16*q_cut**3*sE)/mbkin**22 + (3380400*mcMS**18*q_cut**3*sE)/
             mbkin**24 - (160016*mcMS**20*q_cut**3*sE)/mbkin**26 - 
            (60048*mcMS**22*q_cut**3*sE)/mbkin**28 + (16776*q_cut**4*sE)/mbkin**8 + 
            (372992*mcMS**2*q_cut**4*sE)/mbkin**10 - (156280*mcMS**4*q_cut**4*sE)/
             mbkin**12 - (11175168*mcMS**6*q_cut**4*sE)/mbkin**14 - 
            (26991504*mcMS**8*q_cut**4*sE)/mbkin**16 - (20178240*mcMS**10*q_cut**4*sE)/
             mbkin**18 - (13035280*mcMS**12*q_cut**4*sE)/mbkin**20 - 
            (11467520*mcMS**14*q_cut**4*sE)/mbkin**22 - (2075576*mcMS**16*q_cut**4*sE)/
             mbkin**24 + (917056*mcMS**18*q_cut**4*sE)/mbkin**26 + 
            (150984*mcMS**20*q_cut**4*sE)/mbkin**28 - (6944*q_cut**5*sE)/mbkin**10 - 
            (147616*mcMS**2*q_cut**5*sE)/mbkin**12 - (60352*mcMS**4*q_cut**5*sE)/
             mbkin**14 + (2193472*mcMS**6*q_cut**5*sE)/mbkin**16 + 
            (3579328*mcMS**8*q_cut**5*sE)/mbkin**18 + (1392704*mcMS**10*q_cut**5*sE)/
             mbkin**20 + (1054912*mcMS**12*q_cut**5*sE)/mbkin**22 - 
            (59328*mcMS**14*q_cut**5*sE)/mbkin**24 - (400288*mcMS**16*q_cut**5*sE)/
             mbkin**26 - (62496*mcMS**18*q_cut**5*sE)/mbkin**28 - 
            (6272*q_cut**6*sE)/mbkin**12 - (186368*mcMS**2*q_cut**6*sE)/mbkin**14 - 
            (920512*mcMS**4*q_cut**6*sE)/mbkin**16 - (1492864*mcMS**6*q_cut**6*sE)/
             mbkin**18 - (1181056*mcMS**8*q_cut**6*sE)/mbkin**20 - 
            (1039424*mcMS**10*q_cut**6*sE)/mbkin**22 - (1224128*mcMS**12*q_cut**6*sE)/
             mbkin**24 - (539840*mcMS**14*q_cut**6*sE)/mbkin**26 - 
            (56448*mcMS**16*q_cut**6*sE)/mbkin**28 + (7200*q_cut**7*sE)/mbkin**14 + 
            (173600*mcMS**2*q_cut**7*sE)/mbkin**16 + (748640*mcMS**4*q_cut**7*sE)/
             mbkin**18 + (1101408*mcMS**6*q_cut**7*sE)/mbkin**20 + 
            (1050976*mcMS**8*q_cut**7*sE)/mbkin**22 + (997216*mcMS**10*q_cut**7*sE)/
             mbkin**24 + (448288*mcMS**12*q_cut**7*sE)/mbkin**26 + 
            (64800*mcMS**14*q_cut**7*sE)/mbkin**28 - (3368*q_cut**8*sE)/mbkin**16 - 
            (37504*mcMS**2*q_cut**8*sE)/mbkin**18 - (137032*mcMS**4*q_cut**8*sE)/
             mbkin**20 - (121760*mcMS**6*q_cut**8*sE)/mbkin**22 - 
            (55176*mcMS**8*q_cut**8*sE)/mbkin**24 - (47584*mcMS**10*q_cut**8*sE)/
             mbkin**26 - (30312*mcMS**12*q_cut**8*sE)/mbkin**28 + 
            (784*q_cut**9*sE)/mbkin**18 - (2352*mcMS**2*q_cut**9*sE)/mbkin**20 + 
            (5376*mcMS**4*q_cut**9*sE)/mbkin**22 - (19200*mcMS**6*q_cut**9*sE)/
             mbkin**24 - (20656*mcMS**8*q_cut**9*sE)/mbkin**26 + 
            (7056*mcMS**10*q_cut**9*sE)/mbkin**28 + (960*q_cut**10*sE)/mbkin**20 + 
            (1920*mcMS**2*q_cut**10*sE)/mbkin**22 + (1920*mcMS**4*q_cut**10*sE)/
             mbkin**24 + (8480*mcMS**6*q_cut**10*sE)/mbkin**26 + 
            (8640*mcMS**8*q_cut**10*sE)/mbkin**28 - (1040*q_cut**11*sE)/mbkin**22 - 
            (1040*mcMS**2*q_cut**11*sE)/mbkin**24 - (6160*mcMS**4*q_cut**11*sE)/
             mbkin**26 - (9360*mcMS**6*q_cut**11*sE)/mbkin**28 + (280*q_cut**12*sE)/
             mbkin**24 + (2520*mcMS**4*q_cut**12*sE)/mbkin**28 - 213*sqB - 
            (3332*mcMS**2*sqB)/mbkin**2 + (92843*mcMS**4*sqB)/mbkin**4 + 
            (287792*mcMS**6*sqB)/mbkin**6 - (4445411*mcMS**8*sqB)/mbkin**8 - 
            (11891340*mcMS**10*sqB)/mbkin**10 + (6455133*mcMS**12*sqB)/
             mbkin**12 + (22235424*mcMS**14*sqB)/mbkin**14 + 
            (1168293*mcMS**16*sqB)/mbkin**16 - (10937628*mcMS**18*sqB)/
             mbkin**18 - (3379867*mcMS**20*sqB)/mbkin**20 + (311344*mcMS**22*sqB)/
             mbkin**22 + (109843*mcMS**24*sqB)/mbkin**24 - (2260*mcMS**26*sqB)/
             mbkin**26 - (621*mcMS**28*sqB)/mbkin**28 + (846*q_cut*sqB)/mbkin**2 + 
            (17894*mcMS**2*q_cut*sqB)/mbkin**4 - (224902*mcMS**4*q_cut*sqB)/mbkin**6 - 
            (1976060*mcMS**6*q_cut*sqB)/mbkin**8 + (4145544*mcMS**8*q_cut*sqB)/
             mbkin**10 + (46430562*mcMS**10*q_cut*sqB)/mbkin**12 + 
            (100628196*mcMS**12*q_cut*sqB)/mbkin**14 + (93478512*mcMS**14*q_cut*sqB)/
             mbkin**16 + (37222638*mcMS**16*q_cut*sqB)/mbkin**18 + 
            (1993746*mcMS**18*q_cut*sqB)/mbkin**20 - (2040638*mcMS**20*q_cut*sqB)/
             mbkin**22 - (280340*mcMS**22*q_cut*sqB)/mbkin**24 + 
            (19100*mcMS**24*q_cut*sqB)/mbkin**26 + (2502*mcMS**26*q_cut*sqB)/
             mbkin**28 - (840*q_cut**2*sqB)/mbkin**4 - (22400*mcMS**2*q_cut**2*sqB)/
             mbkin**6 + (43242*mcMS**4*q_cut**2*sqB)/mbkin**8 + 
            (1339974*mcMS**6*q_cut**2*sqB)/mbkin**10 + (4027188*mcMS**8*q_cut**2*sqB)/
             mbkin**12 + (3553428*mcMS**10*q_cut**2*sqB)/mbkin**14 + 
            (2532720*mcMS**12*q_cut**2*sqB)/mbkin**16 + (5403648*mcMS**14*q_cut**2*sqB)/
             mbkin**18 + (4888284*mcMS**16*q_cut**2*sqB)/mbkin**20 + 
            (1486908*mcMS**18*q_cut**2*sqB)/mbkin**22 + (65846*mcMS**20*q_cut**2*sqB)/
             mbkin**24 - (30678*mcMS**22*q_cut**2*sqB)/mbkin**26 - 
            (2520*mcMS**24*q_cut**2*sqB)/mbkin**28 - (846*q_cut**3*sqB)/mbkin**6 - 
            (21278*mcMS**2*q_cut**3*sqB)/mbkin**8 + (133430*mcMS**4*q_cut**3*sqB)/
             mbkin**10 + (2501194*mcMS**6*q_cut**3*sqB)/mbkin**12 + 
            (8789568*mcMS**8*q_cut**3*sqB)/mbkin**14 + (13410612*mcMS**10*q_cut**3*sqB)/
             mbkin**16 + (12845168*mcMS**12*q_cut**3*sqB)/mbkin**18 + 
            (7934668*mcMS**14*q_cut**3*sqB)/mbkin**20 + (2365342*mcMS**16*q_cut**3*sqB)/
             mbkin**22 + (190170*mcMS**18*q_cut**3*sqB)/mbkin**24 - 
            (23606*mcMS**20*q_cut**3*sqB)/mbkin**26 - (2502*mcMS**22*q_cut**3*sqB)/
             mbkin**28 + (2103*q_cut**4*sqB)/mbkin**8 + (61736*mcMS**2*q_cut**4*sqB)/
             mbkin**10 + (83855*mcMS**4*q_cut**4*sqB)/mbkin**12 - 
            (2321052*mcMS**6*q_cut**4*sqB)/mbkin**14 - (9988566*mcMS**8*q_cut**4*sqB)/
             mbkin**16 - (15003696*mcMS**10*q_cut**4*sqB)/mbkin**18 - 
            (9525682*mcMS**12*q_cut**4*sqB)/mbkin**20 - (2254172*mcMS**14*q_cut**4*sqB)/
             mbkin**22 + (63919*mcMS**16*q_cut**4*sqB)/mbkin**24 + 
            (85744*mcMS**18*q_cut**4*sqB)/mbkin**26 + (6291*mcMS**20*q_cut**4*sqB)/
             mbkin**28 - (812*q_cut**5*sqB)/mbkin**10 - (25228*mcMS**2*q_cut**5*sqB)/
             mbkin**12 - (67192*mcMS**4*q_cut**5*sqB)/mbkin**14 + 
            (391288*mcMS**6*q_cut**5*sqB)/mbkin**16 + (1434028*mcMS**8*q_cut**5*sqB)/
             mbkin**18 + (1387040*mcMS**10*q_cut**5*sqB)/mbkin**20 + 
            (315616*mcMS**12*q_cut**5*sqB)/mbkin**22 - (90672*mcMS**14*q_cut**5*sqB)/
             mbkin**24 - (37240*mcMS**16*q_cut**5*sqB)/mbkin**26 - 
            (2604*mcMS**18*q_cut**5*sqB)/mbkin**28 - (896*q_cut**6*sqB)/mbkin**12 - 
            (28784*mcMS**2*q_cut**6*sqB)/mbkin**14 - (208312*mcMS**4*q_cut**6*sqB)/
             mbkin**16 - (551920*mcMS**6*q_cut**6*sqB)/mbkin**18 - 
            (725956*mcMS**8*q_cut**6*sqB)/mbkin**20 - (551756*mcMS**10*q_cut**6*sqB)/
             mbkin**22 - (228404*mcMS**12*q_cut**6*sqB)/mbkin**24 - 
            (42980*mcMS**14*q_cut**6*sqB)/mbkin**26 - (2352*mcMS**16*q_cut**6*sqB)/
             mbkin**28 + (780*q_cut**7*sqB)/mbkin**14 + (28700*mcMS**2*q_cut**7*sqB)/
             mbkin**16 + (188264*mcMS**4*q_cut**7*sqB)/mbkin**18 + 
            (443676*mcMS**6*q_cut**7*sqB)/mbkin**20 + (462304*mcMS**8*q_cut**7*sqB)/
             mbkin**22 + (209860*mcMS**10*q_cut**7*sqB)/mbkin**24 + 
            (40228*mcMS**12*q_cut**7*sqB)/mbkin**26 + (2700*mcMS**14*q_cut**7*sqB)/
             mbkin**28 + (q_cut**8*sqB)/mbkin**16 - (7492*mcMS**2*q_cut**8*sqB)/
             mbkin**18 - (42259*mcMS**4*q_cut**8*sqB)/mbkin**20 - 
            (69524*mcMS**6*q_cut**8*sqB)/mbkin**22 - (39519*mcMS**8*q_cut**8*sqB)/
             mbkin**24 - (7564*mcMS**10*q_cut**8*sqB)/mbkin**26 - 
            (1263*mcMS**12*q_cut**8*sqB)/mbkin**28 - (98*q_cut**9*sqB)/mbkin**18 + 
            (294*mcMS**2*q_cut**9*sqB)/mbkin**20 + (1566*mcMS**4*q_cut**9*sqB)/
             mbkin**22 - (348*mcMS**6*q_cut**9*sqB)/mbkin**24 - 
            (724*mcMS**8*q_cut**9*sqB)/mbkin**26 + (294*mcMS**10*q_cut**9*sqB)/
             mbkin**28 - (120*q_cut**10*sqB)/mbkin**20 - (240*mcMS**2*q_cut**10*sqB)/
             mbkin**22 - (450*mcMS**4*q_cut**10*sqB)/mbkin**24 + 
            (170*mcMS**6*q_cut**10*sqB)/mbkin**26 + (360*mcMS**8*q_cut**10*sqB)/
             mbkin**28 + (130*q_cut**11*sqB)/mbkin**22 + (130*mcMS**2*q_cut**11*sqB)/
             mbkin**24 - (190*mcMS**4*q_cut**11*sqB)/mbkin**26 - 
            (390*mcMS**6*q_cut**11*sqB)/mbkin**28 - (35*q_cut**12*sqB)/mbkin**24 + 
            (105*mcMS**4*q_cut**12*sqB)/mbkin**28))*
         np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                 mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**
                4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
             np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
        (72*mcMS**4*(-16*(-((-1 + mcMS**2/mbkin**2)**4*(-767 - (3711*mcMS**2)/
                 mbkin**2 + (98861*mcMS**4)/mbkin**4 + (43717*mcMS**6)/mbkin**6 - 
                (841083*mcMS**8)/mbkin**8 - (744595*mcMS**10)/mbkin**10 - 
                (69221*mcMS**12)/mbkin**12 + (27747*mcMS**14)/mbkin**14 + 
                (26246*mcMS**16)/mbkin**16 + (3726*mcMS**18)/mbkin**18)) + 
             ((-1 + mcMS**2/mbkin**2)**2*(-3841 - (25194*mcMS**2)/mbkin**2 + 
                (398290*mcMS**4)/mbkin**4 + (568154*mcMS**6)/mbkin**6 - 
                (3222018*mcMS**8)/mbkin**8 - (6339314*mcMS**10)/mbkin**10 - 
                (3173788*mcMS**12)/mbkin**12 - (222702*mcMS**14)/mbkin**14 + 
                (192499*mcMS**16)/mbkin**16 + (136536*mcMS**18)/mbkin**18 + 
                (18738*mcMS**20)/mbkin**20)*q_cut)/mbkin**2 + 
             ((6921 + (51033*mcMS**2)/mbkin**2 - (589962*mcMS**4)/mbkin**4 - 
                (750018*mcMS**6)/mbkin**6 + (4107426*mcMS**8)/mbkin**8 + 
                (7988514*mcMS**10)/mbkin**10 + (8361852*mcMS**12)/mbkin**12 + 
                (4423164*mcMS**14)/mbkin**14 + (352365*mcMS**16)/mbkin**16 - 
                (329619*mcMS**18)/mbkin**18 - (242538*mcMS**20)/mbkin**20 - 
                (33858*mcMS**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
             ((-3847 - (32624*mcMS**2)/mbkin**2 + (269714*mcMS**4)/mbkin**4 + 
                (509592*mcMS**6)/mbkin**6 - (2290278*mcMS**8)/mbkin**8 - 
                (4886864*mcMS**10)/mbkin**10 - (2102320*mcMS**12)/mbkin**12 + 
                (331856*mcMS**14)/mbkin**14 + (480765*mcMS**16)/mbkin**16 + 
                (172880*mcMS**18)/mbkin**18 + (18846*mcMS**20)/mbkin**20)*q_cut**3)/
              mbkin**6 + (2*(-1925 - (25865*mcMS**2)/mbkin**2 + (1041*mcMS**4)/
                 mbkin**4 + (219837*mcMS**6)/mbkin**6 + (528849*mcMS**8)/
                 mbkin**8 + (535197*mcMS**10)/mbkin**10 + (298475*mcMS**12)/
                 mbkin**12 + (155807*mcMS**14)/mbkin**14 + (72450*mcMS**16)/
                 mbkin**16 + (9450*mcMS**18)/mbkin**18)*q_cut**4)/mbkin**8 - 
             (2*(-3479 - (44856*mcMS**2)/mbkin**2 + (21765*mcMS**4)/mbkin**4 + 
                (384640*mcMS**6)/mbkin**6 + (691899*mcMS**8)/mbkin**8 + 
                (706050*mcMS**10)/mbkin**10 + (414721*mcMS**12)/mbkin**12 + 
                (147084*mcMS**14)/mbkin**14 + (17262*mcMS**16)/mbkin**16)*q_cut**5)/
              mbkin**10 + (2*(-1967 - (24199*mcMS**2)/mbkin**2 + (22930*mcMS**4)/
                 mbkin**4 + (237662*mcMS**6)/mbkin**6 + (338587*mcMS**8)/
                 mbkin**8 + (219507*mcMS**10)/mbkin**10 + (81130*mcMS**12)/
                 mbkin**12 + (10206*mcMS**14)/mbkin**14)*q_cut**6)/mbkin**12 - 
             (2*(-397 - (4456*mcMS**2)/mbkin**2 + (9588*mcMS**4)/mbkin**4 + 
                (42868*mcMS**6)/mbkin**6 + (33001*mcMS**8)/mbkin**8 + 
                (17016*mcMS**10)/mbkin**10 + (2106*mcMS**12)/mbkin**12)*q_cut**7)/
              mbkin**14 + ((187 + (159*mcMS**2)/mbkin**2 - (1587*mcMS**4)/
                 mbkin**4 - (1751*mcMS**6)/mbkin**6 + (1530*mcMS**8)/mbkin**8 - 
                (3366*mcMS**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
             (5*(-57 - (72*mcMS**2)/mbkin**2 + (449*mcMS**4)/mbkin**4 + 
                (692*mcMS**6)/mbkin**6 + (1026*mcMS**8)/mbkin**8)*q_cut**9)/
              mbkin**18 - (15*(-11 - (11*mcMS**2)/mbkin**2 + (134*mcMS**4)/
                 mbkin**4 + (198*mcMS**6)/mbkin**6)*q_cut**10)/mbkin**20 - 
             (35*(mbkin**4 - 18*mcMS**4)*q_cut**11)/mbkin**26)*rE + 
           ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
             q_cut**2/mbkin**4)*((-3360*mcMS**2*muG**2)/mbkin**2 - (60*mcMS**4*muG**2)/
              mbkin**4 + (15708*mcMS**6*muG**2)/mbkin**6 + (110856*mcMS**8*muG**2)/
              mbkin**8 + (1868664*mcMS**10*muG**2)/mbkin**10 - 
             (1429008*mcMS**12*muG**2)/mbkin**12 - (2985360*mcMS**14*muG**2)/
              mbkin**14 + (1970472*mcMS**16*muG**2)/mbkin**16 + 
             (636024*mcMS**18*muG**2)/mbkin**18 - (176484*mcMS**20*muG**2)/
              mbkin**20 - (7452*mcMS**22*muG**2)/mbkin**22 - 
             (3360*mcMS**2*muG*mupi)/mbkin**2 + (13500*mcMS**4*muG*mupi)/
              mbkin**4 + (179172*mcMS**6*muG*mupi)/mbkin**6 - 
             (732456*mcMS**8*muG*mupi)/mbkin**8 - (1583064*mcMS**10*muG*mupi)/
              mbkin**10 + (2410128*mcMS**12*muG*mupi)/mbkin**12 + 
             (1627920*mcMS**14*muG*mupi)/mbkin**14 - (1543752*mcMS**16*muG*mupi)/
              mbkin**16 - (451224*mcMS**18*muG*mupi)/mbkin**18 + 
             (75684*mcMS**20*muG*mupi)/mbkin**20 + (7452*mcMS**22*muG*mupi)/
              mbkin**22 - (3360*mcMS**2*muG**2*q_cut)/mbkin**4 + 
             (30300*mcMS**4*muG**2*q_cut)/mbkin**6 + (308712*mcMS**6*muG**2*q_cut)/
              mbkin**8 - (670080*mcMS**8*muG**2*q_cut)/mbkin**10 - 
             (4759032*mcMS**10*muG**2*q_cut)/mbkin**12 - (8785320*mcMS**12*muG**2*q_cut)/
              mbkin**14 - (6468600*mcMS**14*muG**2*q_cut)/mbkin**16 - 
             (210432*mcMS**16*muG**2*q_cut)/mbkin**18 + (576840*mcMS**18*muG**2*q_cut)/
              mbkin**20 + (22572*mcMS**20*muG**2*q_cut)/mbkin**22 + 
             (10080*mcMS**2*muG*mupi*q_cut)/mbkin**4 - (10140*mcMS**4*muG*mupi*q_cut)/
              mbkin**6 - (510312*mcMS**6*muG*mupi*q_cut)/mbkin**8 + 
             (283680*mcMS**8*muG*mupi*q_cut)/mbkin**10 + (5767032*mcMS**10*muG*mupi*
               q_cut)/mbkin**12 + (9349800*mcMS**12*muG*mupi*q_cut)/mbkin**14 + 
             (5003640*mcMS**14*muG*mupi*q_cut)/mbkin**16 + 
             (361632*mcMS**16*muG*mupi*q_cut)/mbkin**18 - (274440*mcMS**18*muG*mupi*
               q_cut)/mbkin**20 - (22572*mcMS**20*muG*mupi*q_cut)/mbkin**22 + 
             (6720*mcMS**2*muG**2*q_cut**2)/mbkin**6 + (20160*mcMS**4*muG**2*q_cut**2)/
              mbkin**8 - (235080*mcMS**6*muG**2*q_cut**2)/mbkin**10 - 
             (509736*mcMS**8*muG**2*q_cut**2)/mbkin**12 + (1261584*mcMS**10*muG**2*q_cut**
                2)/mbkin**14 + (1876464*mcMS**12*muG**2*q_cut**2)/mbkin**16 - 
             (348456*mcMS**14*muG**2*q_cut**2)/mbkin**18 - (393336*mcMS**16*muG**2*q_cut**
                2)/mbkin**20 - (15120*mcMS**18*muG**2*q_cut**2)/mbkin**22 - 
             (6720*mcMS**2*muG*mupi*q_cut**2)/mbkin**6 - (6720*mcMS**4*muG*mupi*q_cut**
                2)/mbkin**8 + (248520*mcMS**6*muG*mupi*q_cut**2)/mbkin**10 + 
             (119976*mcMS**8*muG*mupi*q_cut**2)/mbkin**12 - 
             (1214544*mcMS**10*muG*mupi*q_cut**2)/mbkin**14 - 
             (1123824*mcMS**12*muG*mupi*q_cut**2)/mbkin**16 + 
             (113256*mcMS**14*muG*mupi*q_cut**2)/mbkin**18 + 
             (191736*mcMS**16*muG*mupi*q_cut**2)/mbkin**20 + 
             (15120*mcMS**18*muG*mupi*q_cut**2)/mbkin**22 + 
             (20160*mcMS**2*muG**2*q_cut**3)/mbkin**8 + (78960*mcMS**4*muG**2*q_cut**3)/
              mbkin**10 - (519000*mcMS**6*muG**2*q_cut**3)/mbkin**12 - 
             (2174256*mcMS**8*muG**2*q_cut**3)/mbkin**14 - (3033984*mcMS**10*muG**2*q_cut**
                3)/mbkin**16 - (1941744*mcMS**12*muG**2*q_cut**3)/mbkin**18 - 
             (453816*mcMS**14*muG**2*q_cut**3)/mbkin**20 - (15120*mcMS**16*muG**2*q_cut**
                3)/mbkin**22 - (6720*mcMS**2*muG*mupi*q_cut**3)/mbkin**8 - 
             (11760*mcMS**4*muG*mupi*q_cut**3)/mbkin**10 + (290520*mcMS**6*muG*mupi*
               q_cut**3)/mbkin**12 + (931056*mcMS**8*muG*mupi*q_cut**3)/mbkin**14 + 
             (1233024*mcMS**10*muG*mupi*q_cut**3)/mbkin**16 + 
             (900144*mcMS**12*muG*mupi*q_cut**3)/mbkin**18 + 
             (252216*mcMS**14*muG*mupi*q_cut**3)/mbkin**20 + 
             (15120*mcMS**16*muG*mupi*q_cut**3)/mbkin**22 - 
             (30240*mcMS**2*muG**2*q_cut**4)/mbkin**10 - (98280*mcMS**4*muG**2*q_cut**4)/
              mbkin**12 + (837600*mcMS**6*muG**2*q_cut**4)/mbkin**14 + 
             (2690664*mcMS**8*muG**2*q_cut**4)/mbkin**16 + (2414472*mcMS**10*muG**2*q_cut**
                4)/mbkin**18 + (657504*mcMS**12*muG**2*q_cut**4)/mbkin**20 + 
             (22680*mcMS**14*muG**2*q_cut**4)/mbkin**22 + (10080*mcMS**2*muG*mupi*q_cut**
                4)/mbkin**10 + (17640*mcMS**4*muG*mupi*q_cut**4)/mbkin**12 - 
             (414240*mcMS**6*muG*mupi*q_cut**4)/mbkin**14 - 
             (1249224*mcMS**8*muG*mupi*q_cut**4)/mbkin**16 - 
             (1154472*mcMS**10*muG*mupi*q_cut**4)/mbkin**18 - 
             (355104*mcMS**12*muG*mupi*q_cut**4)/mbkin**20 - 
             (22680*mcMS**14*muG*mupi*q_cut**4)/mbkin**22 + 
             (10080*mcMS**2*muG**2*q_cut**5)/mbkin**12 + (22680*mcMS**4*muG**2*q_cut**5)/
              mbkin**14 - (301608*mcMS**6*muG**2*q_cut**5)/mbkin**16 - 
             (593472*mcMS**8*muG**2*q_cut**5)/mbkin**18 - (208872*mcMS**10*muG**2*q_cut**
                5)/mbkin**20 - (8568*mcMS**12*muG**2*q_cut**5)/mbkin**22 - 
             (3360*mcMS**2*muG*mupi*q_cut**5)/mbkin**12 - (2520*mcMS**4*muG*mupi*q_cut**
                5)/mbkin**14 + (140328*mcMS**6*muG*mupi*q_cut**5)/mbkin**16 + 
             (274272*mcMS**8*muG*mupi*q_cut**5)/mbkin**18 + 
             (108072*mcMS**10*muG*mupi*q_cut**5)/mbkin**20 + 
             (8568*mcMS**12*muG*mupi*q_cut**5)/mbkin**22 + (1128*mcMS**6*muG**2*q_cut**6)/
              mbkin**18 - (1896*mcMS**8*muG**2*q_cut**6)/mbkin**20 + 
             (1008*mcMS**10*muG**2*q_cut**6)/mbkin**22 - (1128*mcMS**6*muG*mupi*q_cut**6)/
              mbkin**18 + (1896*mcMS**8*muG*mupi*q_cut**6)/mbkin**20 - 
             (1008*mcMS**10*muG*mupi*q_cut**6)/mbkin**22 + (2160*mcMS**4*muG**2*q_cut**7)/
              mbkin**18 + (3480*mcMS**6*muG**2*q_cut**7)/mbkin**20 + 
             (2160*mcMS**8*muG**2*q_cut**7)/mbkin**22 - (2160*mcMS**4*muG*mupi*q_cut**7)/
              mbkin**18 - (3480*mcMS**6*muG*mupi*q_cut**7)/mbkin**20 - 
             (2160*mcMS**8*muG*mupi*q_cut**7)/mbkin**22 - (3420*mcMS**4*muG**2*q_cut**8)/
              mbkin**20 - (3420*mcMS**6*muG**2*q_cut**8)/mbkin**22 + 
             (3420*mcMS**4*muG*mupi*q_cut**8)/mbkin**20 + (3420*mcMS**6*muG*mupi*q_cut**
                8)/mbkin**22 + (1260*mcMS**4*muG**2*q_cut**9)/mbkin**22 - 
             (1260*mcMS**4*muG*mupi*q_cut**9)/mbkin**22 + 24*mcMS**2*muG*
              ((-1 + mcMS**2/mbkin**2)**2*(-280 + (565*mcMS**2)/mbkin**2 + 
                 (16341*mcMS**4)/mbkin**4 - (28921*mcMS**6)/mbkin**6 - 
                 (206105*mcMS**8)/mbkin**8 - (182445*mcMS**10)/mbkin**10 - 
                 (23125*mcMS**12)/mbkin**12 + (7549*mcMS**14)/mbkin**14 + 
                 (621*mcMS**16)/mbkin**16) + ((840 - (845*mcMS**2)/mbkin**2 - 
                  (42526*mcMS**4)/mbkin**4 + (23640*mcMS**6)/mbkin**6 + 
                  (480586*mcMS**8)/mbkin**8 + (779150*mcMS**10)/mbkin**10 + 
                  (416970*mcMS**12)/mbkin**12 + (30136*mcMS**14)/mbkin**14 - 
                  (22870*mcMS**16)/mbkin**16 - (1881*mcMS**18)/mbkin**18)*q_cut)/
                mbkin**2 + (2*(-280 - (280*mcMS**2)/mbkin**2 + (10355*mcMS**4)/
                   mbkin**4 + (4999*mcMS**6)/mbkin**6 - (50606*mcMS**8)/mbkin**8 - 
                  (46826*mcMS**10)/mbkin**10 + (4719*mcMS**12)/mbkin**12 + 
                  (7989*mcMS**14)/mbkin**14 + (630*mcMS**16)/mbkin**16)*q_cut**2)/
                mbkin**4 + (2*(-280 - (490*mcMS**2)/mbkin**2 + (12105*mcMS**4)/
                   mbkin**4 + (38794*mcMS**6)/mbkin**6 + (51376*mcMS**8)/
                   mbkin**8 + (37506*mcMS**10)/mbkin**10 + (10509*mcMS**12)/
                   mbkin**12 + (630*mcMS**14)/mbkin**14)*q_cut**3)/mbkin**6 - 
               (2*(-420 - (735*mcMS**2)/mbkin**2 + (17260*mcMS**4)/mbkin**4 + 
                  (52051*mcMS**6)/mbkin**6 + (48103*mcMS**8)/mbkin**8 + 
                  (14796*mcMS**10)/mbkin**10 + (945*mcMS**12)/mbkin**12)*q_cut**4)/
                mbkin**8 + (2*(-140 - (105*mcMS**2)/mbkin**2 + (5847*mcMS**4)/
                   mbkin**4 + (11428*mcMS**6)/mbkin**6 + (4503*mcMS**8)/mbkin**8 + 
                  (357*mcMS**10)/mbkin**10)*q_cut**5)/mbkin**10 - (2*mcMS**4*
                 (47 - (79*mcMS**2)/mbkin**2 + (42*mcMS**4)/mbkin**4)*q_cut**6)/
                mbkin**16 - (10*mcMS**2*(18 + (29*mcMS**2)/mbkin**2 + (18*mcMS**4)/
                   mbkin**4)*q_cut**7)/mbkin**16 + (285*mcMS**2*(mbkin**2 + mcMS**2)*
                 q_cut**8)/mbkin**20 - (105*mcMS**2*q_cut**9)/mbkin**20) + 
             16*(-((-1 + mcMS**2/mbkin**2)**2*(-137 - (3431*mcMS**2)/mbkin**2 + 
                  (22624*mcMS**4)/mbkin**4 + (94372*mcMS**6)/mbkin**6 - 
                  (119450*mcMS**8)/mbkin**8 - (345242*mcMS**10)/mbkin**10 - 
                  (152798*mcMS**12)/mbkin**12 - (2588*mcMS**14)/mbkin**14 + 
                  (7069*mcMS**16)/mbkin**16 + (621*mcMS**18)/mbkin**18)) + 
               ((-417 - (10988*mcMS**2)/mbkin**2 + (37288*mcMS**4)/mbkin**4 + 
                  (239041*mcMS**6)/mbkin**6 - (38168*mcMS**8)/mbkin**8 - 
                  (928862*mcMS**10)/mbkin**10 - (1005470*mcMS**12)/mbkin**12 - 
                  (325136*mcMS**14)/mbkin**14 + (13966*mcMS**16)/mbkin**16 + 
                  (21025*mcMS**18)/mbkin**18 + (1881*mcMS**20)/mbkin**20)*q_cut)/
                mbkin**2 - (2*(-140 - (3993*mcMS**2)/mbkin**2 + (1395*mcMS**4)/
                   mbkin**4 + (30919*mcMS**6)/mbkin**6 - (27489*mcMS**8)/
                   mbkin**8 - (89157*mcMS**10)/mbkin**10 - (21743*mcMS**12)/
                   mbkin**12 + (18639*mcMS**14)/mbkin**14 + (7779*mcMS**16)/
                   mbkin**16 + (630*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
               ((-280 - (8826*mcMS**2)/mbkin**2 - (20363*mcMS**4)/mbkin**4 + 
                  (14773*mcMS**6)/mbkin**6 + (58659*mcMS**8)/mbkin**8 + 
                  (91223*mcMS**10)/mbkin**10 + (63057*mcMS**12)/mbkin**12 + 
                  (16713*mcMS**14)/mbkin**14 + (1260*mcMS**16)/mbkin**16)*q_cut**3)/
                mbkin**6 + (2*(-210 - (6612*mcMS**2)/mbkin**2 - (14884*mcMS**4)/
                   mbkin**4 + (18675*mcMS**6)/mbkin**6 + (61863*mcMS**8)/
                   mbkin**8 + (46048*mcMS**10)/mbkin**10 + (12381*mcMS**12)/
                   mbkin**12 + (945*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
               ((168 + (4164*mcMS**2)/mbkin**2 + (5451*mcMS**4)/mbkin**4 - 
                  (18857*mcMS**6)/mbkin**6 - (24019*mcMS**8)/mbkin**8 - 
                  (7641*mcMS**10)/mbkin**10 - (714*mcMS**12)/mbkin**12)*q_cut**5)/
                mbkin**10 + (2*(-14 + (17*mcMS**2)/mbkin**2 - (60*mcMS**4)/
                   mbkin**4 + mcMS**6/mbkin**6 + (26*mcMS**8)/mbkin**8 + 
                  (42*mcMS**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
               (5*(-12 - (10*mcMS**2)/mbkin**2 + (5*mcMS**4)/mbkin**4 + 
                  (25*mcMS**6)/mbkin**6 + (36*mcMS**8)/mbkin**8)*q_cut**7)/mbkin**14 - 
               (5*(-19 - (19*mcMS**2)/mbkin**2 + (45*mcMS**4)/mbkin**4 + 
                  (57*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 - (35*(mbkin**4 - 
                  3*mcMS**4)*q_cut**9)/mbkin**22)*rG - 8*mbkin*
              (-((-1 + mcMS**2/mbkin**2)**2*(-219 - (3427*mcMS**2)/mbkin**2 - 
                  (25220*mcMS**4)/mbkin**4 + (690492*mcMS**6)/mbkin**6 + 
                  (3385394*mcMS**8)/mbkin**8 + (3853498*mcMS**10)/mbkin**10 + 
                  (969876*mcMS**12)/mbkin**12 - (66556*mcMS**14)/mbkin**14 + 
                  (8017*mcMS**16)/mbkin**16 + (3105*mcMS**18)/mbkin**18)) + 
               ((-639 - (12606*mcMS**2)/mbkin**2 - (137707*mcMS**4)/mbkin**4 + 
                  (1200772*mcMS**6)/mbkin**6 + (8153946*mcMS**8)/mbkin**8 + 
                  (14790212*mcMS**10)/mbkin**10 + (9837754*mcMS**12)/mbkin**12 + 
                  (1656240*mcMS**14)/mbkin**14 - (260839*mcMS**16)/mbkin**16 + 
                  (23302*mcMS**18)/mbkin**18 + (9405*mcMS**20)/mbkin**20)*q_cut)/
                mbkin**2 - (2*(-210 - (4891*mcMS**2)/mbkin**2 - (70825*mcMS**4)/
                   mbkin**4 + (10961*mcMS**6)/mbkin**6 + (726865*mcMS**8)/
                   mbkin**8 + (829165*mcMS**10)/mbkin**10 + (50797*mcMS**12)/
                   mbkin**12 - (87657*mcMS**14)/mbkin**14 + (11805*mcMS**16)/
                   mbkin**16 + (3150*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
               (2*(210 + (5521*mcMS**2)/mbkin**2 + (90958*mcMS**4)/mbkin**4 + 
                  (297789*mcMS**6)/mbkin**6 + (370334*mcMS**8)/mbkin**8 + 
                  (299309*mcMS**10)/mbkin**10 + (83742*mcMS**12)/mbkin**12 - 
                  (15585*mcMS**14)/mbkin**14 - (3150*mcMS**16)/mbkin**16)*q_cut**3)/
                mbkin**6 + (2*(-315 - (8304*mcMS**2)/mbkin**2 - (138468*mcMS**4)/
                   mbkin**4 - (460673*mcMS**6)/mbkin**6 - (479693*mcMS**8)/
                   mbkin**8 - (125948*mcMS**10)/mbkin**10 + (23580*mcMS**12)/
                   mbkin**12 + (4725*mcMS**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
               ((126 + (5638*mcMS**2)/mbkin**2 + (85014*mcMS**4)/mbkin**4 + 
                  (204504*mcMS**6)/mbkin**6 + (84214*mcMS**8)/mbkin**8 - 
                  (13278*mcMS**10)/mbkin**10 - (3570*mcMS**12)/mbkin**12)*q_cut**5)/
                mbkin**10 + (2*(42 - (51*mcMS**2)/mbkin**2 + (159*mcMS**4)/
                   mbkin**4 - (541*mcMS**6)/mbkin**6 - (31*mcMS**8)/mbkin**8 + 
                  (210*mcMS**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
               (10*(18 + (15*mcMS**2)/mbkin**2 + (78*mcMS**4)/mbkin**4 + 
                  (17*mcMS**6)/mbkin**6 + (90*mcMS**8)/mbkin**8)*q_cut**7)/mbkin**14 - 
               (15*(19 + (19*mcMS**2)/mbkin**2 + (47*mcMS**4)/mbkin**4 + 
                  (95*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + (105*(mbkin**4 + 
                  5*mcMS**4)*q_cut**9)/mbkin**22)*rhoD - 3360*sB - 
             (85120*mcMS**2*sB)/mbkin**2 + (548568*mcMS**4*sB)/mbkin**4 + 
             (4992264*mcMS**6*sB)/mbkin**6 + (848880*mcMS**8*sB)/mbkin**8 - 
             (11177712*mcMS**10*sB)/mbkin**10 - (463680*mcMS**12*sB)/mbkin**12 + 
             (3575712*mcMS**14*sB)/mbkin**14 + (1418256*mcMS**16*sB)/mbkin**16 + 
             (526800*mcMS**18*sB)/mbkin**18 - (155768*mcMS**20*sB)/mbkin**20 - 
             (24840*mcMS**22*sB)/mbkin**22 + (10080*q_cut*sB)/mbkin**2 + 
             (299040*mcMS**2*q_cut*sB)/mbkin**4 - (169448*mcMS**4*q_cut*sB)/mbkin**6 - 
             (12282208*mcMS**6*q_cut*sB)/mbkin**8 - (36742752*mcMS**8*q_cut*sB)/
              mbkin**10 - (39176624*mcMS**10*q_cut*sB)/mbkin**12 - 
             (17580784*mcMS**12*q_cut*sB)/mbkin**14 - (4143696*mcMS**14*q_cut*sB)/
              mbkin**16 + (304*mcMS**16*q_cut*sB)/mbkin**18 + 
             (604928*mcMS**18*q_cut*sB)/mbkin**20 + (75240*mcMS**20*q_cut*sB)/
              mbkin**22 - (6720*q_cut**2*sB)/mbkin**4 - (217280*mcMS**2*q_cut**2*sB)/
              mbkin**6 - (600320*mcMS**4*q_cut**2*sB)/mbkin**8 + 
             (2179984*mcMS**6*q_cut**2*sB)/mbkin**10 + (5879888*mcMS**8*q_cut**2*sB)/
              mbkin**12 + (2959328*mcMS**10*q_cut**2*sB)/mbkin**14 + 
             (38624*mcMS**12*q_cut**2*sB)/mbkin**16 - (639984*mcMS**14*q_cut**2*sB)/
              mbkin**18 - (450960*mcMS**16*q_cut**2*sB)/mbkin**20 - 
             (50400*mcMS**18*q_cut**2*sB)/mbkin**22 - (6720*q_cut**3*sB)/mbkin**6 - 
             (237440*mcMS**2*q_cut**3*sB)/mbkin**8 - (1361360*mcMS**4*q_cut**3*sB)/
              mbkin**10 - (2607456*mcMS**6*q_cut**3*sB)/mbkin**12 - 
             (3283120*mcMS**8*q_cut**3*sB)/mbkin**14 - (3012208*mcMS**10*q_cut**3*sB)/
              mbkin**16 - (1675344*mcMS**12*q_cut**3*sB)/mbkin**18 - 
             (546720*mcMS**14*q_cut**3*sB)/mbkin**20 - (50400*mcMS**16*q_cut**3*sB)/
              mbkin**22 + (10080*q_cut**4*sB)/mbkin**8 + (356160*mcMS**2*q_cut**4*sB)/
              mbkin**10 + (2078160*mcMS**4*q_cut**4*sB)/mbkin**12 + 
             (4039744*mcMS**6*q_cut**4*sB)/mbkin**14 + (3825136*mcMS**8*q_cut**4*sB)/
              mbkin**16 + (2284336*mcMS**10*q_cut**4*sB)/mbkin**18 + 
             (790560*mcMS**12*q_cut**4*sB)/mbkin**20 + (75600*mcMS**14*q_cut**4*sB)/
              mbkin**22 - (3360*q_cut**5*sB)/mbkin**10 - (115360*mcMS**2*q_cut**5*sB)/
              mbkin**12 - (580608*mcMS**4*q_cut**5*sB)/mbkin**14 - 
             (801120*mcMS**6*q_cut**5*sB)/mbkin**16 - (572464*mcMS**8*q_cut**5*sB)/
              mbkin**18 - (238272*mcMS**10*q_cut**5*sB)/mbkin**20 - 
             (28560*mcMS**12*q_cut**5*sB)/mbkin**22 - (672*mcMS**4*q_cut**6*sB)/
              mbkin**16 - (208*mcMS**6*q_cut**6*sB)/mbkin**18 - 
             (1168*mcMS**8*q_cut**6*sB)/mbkin**20 + (3360*mcMS**10*q_cut**6*sB)/
              mbkin**22 + (2640*mcMS**4*q_cut**7*sB)/mbkin**18 + 
             (5920*mcMS**6*q_cut**7*sB)/mbkin**20 + (7200*mcMS**8*q_cut**7*sB)/
              mbkin**22 - (8520*mcMS**4*q_cut**8*sB)/mbkin**20 - 
             (11400*mcMS**6*q_cut**8*sB)/mbkin**22 + (4200*mcMS**4*q_cut**9*sB)/
              mbkin**22 + 1656*sE + (57736*mcMS**2*sE)/mbkin**2 - 
             (366072*mcMS**4*sE)/mbkin**4 - (2250888*mcMS**6*sE)/mbkin**6 + 
             (2070480*mcMS**8*sE)/mbkin**8 + (4465776*mcMS**10*sE)/mbkin**10 - 
             (4114320*mcMS**12*sE)/mbkin**12 - (461808*mcMS**14*sE)/mbkin**14 + 
             (747768*mcMS**16*sE)/mbkin**16 - (241080*mcMS**18*sE)/mbkin**18 + 
             (75848*mcMS**20*sE)/mbkin**20 + (14904*mcMS**22*sE)/mbkin**22 - 
             (5016*q_cut*sE)/mbkin**2 - (192624*mcMS**2*q_cut*sE)/mbkin**4 + 
             (233864*mcMS**4*q_cut*sE)/mbkin**6 + (6234112*mcMS**6*q_cut*sE)/mbkin**8 + 
             (12579408*mcMS**8*q_cut*sE)/mbkin**10 + (5003168*mcMS**10*q_cut*sE)/
              mbkin**12 - (2810288*mcMS**12*q_cut*sE)/mbkin**14 - 
             (826176*mcMS**14*q_cut*sE)/mbkin**16 - (11704*mcMS**16*q_cut*sE)/
              mbkin**18 - (322160*mcMS**18*q_cut*sE)/mbkin**20 - 
             (45144*mcMS**20*q_cut*sE)/mbkin**22 + (3360*q_cut**2*sE)/mbkin**4 + 
             (136688*mcMS**2*q_cut**2*sE)/mbkin**6 + (228560*mcMS**4*q_cut**2*sE)/
              mbkin**8 - (1394704*mcMS**6*q_cut**2*sE)/mbkin**10 - 
             (1962128*mcMS**8*q_cut**2*sE)/mbkin**12 + (667504*mcMS**10*q_cut**2*sE)/
              mbkin**14 + (1108336*mcMS**12*q_cut**2*sE)/mbkin**16 + 
             (576912*mcMS**14*q_cut**2*sE)/mbkin**18 + (282672*mcMS**16*q_cut**2*sE)/
              mbkin**20 + (30240*mcMS**18*q_cut**2*sE)/mbkin**22 + 
             (3360*q_cut**3*sE)/mbkin**6 + (146768*mcMS**2*q_cut**3*sE)/mbkin**8 + 
             (662144*mcMS**4*q_cut**3*sE)/mbkin**10 + (764016*mcMS**6*q_cut**3*sE)/
              mbkin**12 + (452416*mcMS**8*q_cut**3*sE)/mbkin**14 + 
             (42352*mcMS**10*q_cut**3*sE)/mbkin**16 + (158208*mcMS**12*q_cut**3*sE)/
              mbkin**18 + (225552*mcMS**14*q_cut**3*sE)/mbkin**20 + 
             (30240*mcMS**16*q_cut**3*sE)/mbkin**22 - (5040*q_cut**4*sE)/mbkin**8 - 
             (220032*mcMS**2*q_cut**4*sE)/mbkin**10 - (1006464*mcMS**4*q_cut**4*sE)/
              mbkin**12 - (1220560*mcMS**6*q_cut**4*sE)/mbkin**14 - 
             (631504*mcMS**8*q_cut**4*sE)/mbkin**16 - (590464*mcMS**10*q_cut**4*sE)/
              mbkin**18 - (380928*mcMS**12*q_cut**4*sE)/mbkin**20 - 
             (45360*mcMS**14*q_cut**4*sE)/mbkin**22 + (1904*q_cut**5*sE)/mbkin**10 + 
             (70832*mcMS**2*q_cut**5*sE)/mbkin**12 + (271248*mcMS**4*q_cut**5*sE)/
              mbkin**14 + (203488*mcMS**6*q_cut**5*sE)/mbkin**16 + 
             (145296*mcMS**8*q_cut**5*sE)/mbkin**18 + (118704*mcMS**10*q_cut**5*sE)/
              mbkin**20 + (17136*mcMS**12*q_cut**5*sE)/mbkin**22 - 
             (224*q_cut**6*sE)/mbkin**12 + (272*mcMS**2*q_cut**6*sE)/mbkin**14 - 
             (400*mcMS**4*q_cut**6*sE)/mbkin**16 + (3632*mcMS**6*q_cut**6*sE)/mbkin**18 - 
             (5168*mcMS**8*q_cut**6*sE)/mbkin**20 - (2016*mcMS**10*q_cut**6*sE)/
              mbkin**22 - (480*q_cut**7*sE)/mbkin**14 - (400*mcMS**2*q_cut**7*sE)/
              mbkin**16 - (960*mcMS**4*q_cut**7*sE)/mbkin**18 + 
             (2480*mcMS**6*q_cut**7*sE)/mbkin**20 - (4320*mcMS**8*q_cut**7*sE)/
              mbkin**22 + (760*q_cut**8*sE)/mbkin**16 + (760*mcMS**2*q_cut**8*sE)/
              mbkin**18 + (3000*mcMS**4*q_cut**8*sE)/mbkin**20 + 
             (6840*mcMS**6*q_cut**8*sE)/mbkin**22 - (280*q_cut**9*sE)/mbkin**18 - 
             (2520*mcMS**4*q_cut**9*sE)/mbkin**22 + 633*sqB + (9583*mcMS**2*sqB)/
              mbkin**2 - (76293*mcMS**4*sqB)/mbkin**4 - (712899*mcMS**6*sqB)/
              mbkin**6 - (625266*mcMS**8*sqB)/mbkin**8 + (1581930*mcMS**10*sqB)/
              mbkin**10 + (1241226*mcMS**12*sqB)/mbkin**12 - 
             (793362*mcMS**14*sqB)/mbkin**14 - (595791*mcMS**16*sqB)/mbkin**16 - 
             (42609*mcMS**18*sqB)/mbkin**18 + (12227*mcMS**20*sqB)/mbkin**20 + 
             (621*mcMS**22*sqB)/mbkin**22 - (1893*q_cut*sqB)/mbkin**2 - 
             (37242*mcMS**2*q_cut*sqB)/mbkin**4 + (26039*mcMS**4*q_cut*sqB)/mbkin**6 + 
             (1713892*mcMS**6*q_cut*sqB)/mbkin**8 + (6214110*mcMS**8*q_cut*sqB)/
              mbkin**10 + (8766044*mcMS**10*q_cut*sqB)/mbkin**12 + 
             (5443102*mcMS**12*q_cut*sqB)/mbkin**14 + (1253136*mcMS**14*q_cut*sqB)/
              mbkin**16 - (48277*mcMS**16*q_cut*sqB)/mbkin**18 - 
             (42230*mcMS**18*q_cut*sqB)/mbkin**20 - (1881*mcMS**20*q_cut*sqB)/
              mbkin**22 + (1260*q_cut**2*sqB)/mbkin**4 + (28274*mcMS**2*q_cut**2*sqB)/
              mbkin**6 + (89030*mcMS**4*q_cut**2*sqB)/mbkin**8 - 
             (243478*mcMS**6*q_cut**2*sqB)/mbkin**10 - (1014998*mcMS**8*q_cut**2*sqB)/
              mbkin**12 - (873326*mcMS**10*q_cut**2*sqB)/mbkin**14 - 
             (85358*mcMS**12*q_cut**2*sqB)/mbkin**16 + (124998*mcMS**14*q_cut**2*sqB)/
              mbkin**18 + (31938*mcMS**16*q_cut**2*sqB)/mbkin**20 + 
             (1260*mcMS**18*q_cut**2*sqB)/mbkin**22 + (1260*q_cut**3*sqB)/mbkin**6 + 
             (32054*mcMS**2*q_cut**3*sqB)/mbkin**8 + (190652*mcMS**4*q_cut**3*sqB)/
              mbkin**10 + (411942*mcMS**6*q_cut**3*sqB)/mbkin**12 + 
             (470956*mcMS**8*q_cut**3*sqB)/mbkin**14 + (374566*mcMS**10*q_cut**3*sqB)/
              mbkin**16 + (168732*mcMS**12*q_cut**3*sqB)/mbkin**18 + 
             (30258*mcMS**14*q_cut**3*sqB)/mbkin**20 + (1260*mcMS**16*q_cut**3*sqB)/
              mbkin**22 - (1890*q_cut**4*sqB)/mbkin**8 - (48096*mcMS**2*q_cut**4*sqB)/
              mbkin**10 - (293352*mcMS**4*q_cut**4*sqB)/mbkin**12 - 
             (657526*mcMS**6*q_cut**4*sqB)/mbkin**14 - (644398*mcMS**8*q_cut**4*sqB)/
              mbkin**16 - (282136*mcMS**10*q_cut**4*sqB)/mbkin**18 - 
             (48072*mcMS**12*q_cut**4*sqB)/mbkin**20 - (1890*mcMS**14*q_cut**4*sqB)/
              mbkin**22 + (602*q_cut**5*sqB)/mbkin**10 + (15506*mcMS**2*q_cut**5*sqB)/
              mbkin**12 + (82266*mcMS**4*q_cut**5*sqB)/mbkin**14 + 
             (138448*mcMS**6*q_cut**5*sqB)/mbkin**16 + (82458*mcMS**8*q_cut**5*sqB)/
              mbkin**18 + (15726*mcMS**10*q_cut**5*sqB)/mbkin**20 + 
             (714*mcMS**12*q_cut**5*sqB)/mbkin**22 + (28*q_cut**6*sqB)/mbkin**12 - 
             (34*mcMS**2*q_cut**6*sqB)/mbkin**14 + (218*mcMS**4*q_cut**6*sqB)/mbkin**16 - 
             (286*mcMS**6*q_cut**6*sqB)/mbkin**18 - (122*mcMS**8*q_cut**6*sqB)/
              mbkin**20 - (84*mcMS**10*q_cut**6*sqB)/mbkin**22 + (60*q_cut**7*sqB)/
              mbkin**14 + (50*mcMS**2*q_cut**7*sqB)/mbkin**16 + 
             (540*mcMS**4*q_cut**7*sqB)/mbkin**18 + (230*mcMS**6*q_cut**7*sqB)/
              mbkin**20 - (180*mcMS**8*q_cut**7*sqB)/mbkin**22 - (95*q_cut**8*sqB)/
              mbkin**16 - (95*mcMS**2*q_cut**8*sqB)/mbkin**18 + (45*mcMS**4*q_cut**8*sqB)/
              mbkin**20 + (285*mcMS**6*q_cut**8*sqB)/mbkin**22 + (35*q_cut**9*sqB)/
              mbkin**18 - (105*mcMS**4*q_cut**9*sqB)/mbkin**22))*
          np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 - 
        (60480*mcMS**8*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
             2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(48*mcMS**2*muG + 
           (144*mcMS**4*muG)/mbkin**2 - (1152*mcMS**6*muG)/mbkin**4 - 
           (2280*mcMS**8*muG)/mbkin**6 - (720*mcMS**10*muG)/mbkin**8 - 
           (24*mcMS**2*muG**2)/mbkin**2 - (72*mcMS**4*muG**2)/mbkin**4 + 
           (576*mcMS**6*muG**2)/mbkin**6 + (1140*mcMS**8*muG**2)/mbkin**8 + 
           (360*mcMS**10*muG**2)/mbkin**10 + (24*mcMS**2*muG*mupi)/mbkin**2 + 
           (72*mcMS**4*muG*mupi)/mbkin**4 - (576*mcMS**6*muG*mupi)/mbkin**6 - 
           (1140*mcMS**8*muG*mupi)/mbkin**8 - (360*mcMS**10*muG*mupi)/mbkin**10 - 
           16*(-4 - (52*mcMS**2)/mbkin**2 + (135*mcMS**4)/mbkin**4 + 
             (412*mcMS**6)/mbkin**6 + (88*mcMS**8)/mbkin**8)*rE + 
           8*(1 - (37*mcMS**2)/mbkin**2 - (42*mcMS**4)/mbkin**4 + 
             (224*mcMS**6)/mbkin**6 + (211*mcMS**8)/mbkin**8 + (39*mcMS**10)/
              mbkin**10)*rG + 48*mbkin*rhoD + (592*mcMS**2*rhoD)/mbkin + 
           (5592*mcMS**4*rhoD)/mbkin**3 + (13536*mcMS**6*rhoD)/mbkin**5 + 
           (7592*mcMS**8*rhoD)/mbkin**7 + (624*mcMS**10*rhoD)/mbkin**9 + 24*sB + 
           (824*mcMS**2*sB)/mbkin**2 + (3984*mcMS**4*sB)/mbkin**4 + 
           (4320*mcMS**6*sB)/mbkin**6 + (1360*mcMS**8*sB)/mbkin**8 + 
           (312*mcMS**10*sB)/mbkin**10 - (416*mcMS**2*sE)/mbkin**2 - 
           (1656*mcMS**4*sE)/mbkin**4 - (480*mcMS**6*sE)/mbkin**6 + 
           (584*mcMS**8*sE)/mbkin**8 - 6*sqB - (122*mcMS**2*sqB)/mbkin**2 - 
           (609*mcMS**4*sqB)/mbkin**4 - (972*mcMS**6*sqB)/mbkin**6 - 
           (523*mcMS**8*sqB)/mbkin**8 - (78*mcMS**10*sqB)/mbkin**10)*
          np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8))/
      (1260*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
          2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
            7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
            mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                  2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**3) + 
     (api4*(((18*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
              7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
              mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)**2*(3*mbkin**14 - 
             141*mbkin**12*mcMS**2 - 7785*mbkin**10*mcMS**4 - 33657*mbkin**8*
              mcMS**6 - 33657*mbkin**6*mcMS**8 - 7785*mbkin**4*mcMS**10 - 
             141*mbkin**2*mcMS**12 + 3*mcMS**14 + 3*mbkin**12*q_cut - 
             132*mbkin**10*mcMS**2*q_cut - 3153*mbkin**8*mcMS**4*q_cut - 
             7296*mbkin**6*mcMS**6*q_cut - 3153*mbkin**4*mcMS**8*q_cut - 
             132*mbkin**2*mcMS**10*q_cut + 3*mcMS**12*q_cut + 3*mbkin**10*q_cut**2 - 
             117*mbkin**8*mcMS**2*q_cut**2 - 1272*mbkin**6*mcMS**4*q_cut**2 - 
             1272*mbkin**4*mcMS**6*q_cut**2 - 117*mbkin**2*mcMS**8*q_cut**2 + 
             3*mcMS**10*q_cut**2 + 3*mbkin**8*q_cut**3 - 96*mbkin**6*mcMS**2*q_cut**3 - 
             408*mbkin**4*mcMS**4*q_cut**3 - 96*mbkin**2*mcMS**6*q_cut**3 + 
             3*mcMS**8*q_cut**3 + 3*mbkin**6*q_cut**4 - 69*mbkin**4*mcMS**2*q_cut**4 - 
             69*mbkin**2*mcMS**4*q_cut**4 + 3*mcMS**6*q_cut**4 - 25*mbkin**4*q_cut**5 + 
             20*mbkin**2*mcMS**2*q_cut**5 - 25*mcMS**4*q_cut**5 - 25*mbkin**2*q_cut**6 - 
             25*mcMS**2*q_cut**6 + 35*q_cut**7))/mbkin**34 - 
          (432*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**
                4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
            (108*mbkin**20 - 792*mbkin**18*mcMS**2 - 13329*mbkin**16*mcMS**4 + 
             40518*mbkin**14*mcMS**6 + 387441*mbkin**12*mcMS**8 + 
             668988*mbkin**10*mcMS**10 + 387441*mbkin**8*mcMS**12 + 
             40518*mbkin**6*mcMS**14 - 13329*mbkin**4*mcMS**16 - 
             792*mbkin**2*mcMS**18 + 108*mcMS**20 - 210*mbkin**18*q_cut - 
             222*mbkin**16*mcMS**2*q_cut + 15402*mbkin**14*mcMS**4*q_cut + 
             81210*mbkin**12*mcMS**6*q_cut + 153300*mbkin**10*mcMS**8*q_cut + 
             153300*mbkin**8*mcMS**10*q_cut + 81210*mbkin**6*mcMS**12*q_cut + 
             15402*mbkin**4*mcMS**14*q_cut - 222*mbkin**2*mcMS**16*q_cut - 
             210*mcMS**18*q_cut - 108*mbkin**16*q_cut**2 + 552*mbkin**14*mcMS**2*q_cut**2 + 
             22257*mbkin**12*mcMS**4*q_cut**2 + 101844*mbkin**10*mcMS**6*q_cut**2 + 
             158394*mbkin**8*mcMS**8*q_cut**2 + 101844*mbkin**6*mcMS**10*q_cut**2 + 
             22257*mbkin**4*mcMS**12*q_cut**2 + 552*mbkin**2*mcMS**14*q_cut**2 - 
             108*mcMS**16*q_cut**2 + 420*mbkin**14*q_cut**3 + 2088*mbkin**12*mcMS**2*
              q_cut**3 - 8028*mbkin**10*mcMS**4*q_cut**3 - 43584*mbkin**8*mcMS**6*q_cut**3 - 
             43584*mbkin**6*mcMS**8*q_cut**3 - 8028*mbkin**4*mcMS**10*q_cut**3 + 
             2088*mbkin**2*mcMS**12*q_cut**3 + 420*mcMS**14*q_cut**3 - 
             105*mbkin**12*q_cut**4 - 642*mbkin**10*mcMS**2*q_cut**4 - 
             966*mbkin**8*mcMS**4*q_cut**4 - 2118*mbkin**6*mcMS**6*q_cut**4 - 
             966*mbkin**4*mcMS**8*q_cut**4 - 642*mbkin**2*mcMS**10*q_cut**4 - 
             105*mcMS**12*q_cut**4 - 238*mbkin**10*q_cut**5 - 1650*mbkin**8*mcMS**2*
              q_cut**5 - 5522*mbkin**6*mcMS**4*q_cut**5 - 5522*mbkin**4*mcMS**6*q_cut**5 - 
             1650*mbkin**2*mcMS**8*q_cut**5 - 238*mcMS**10*q_cut**5 + 105*mbkin**8*q_cut**6 + 
             940*mbkin**6*mcMS**2*q_cut**6 + 1705*mbkin**4*mcMS**4*q_cut**6 + 
             940*mbkin**2*mcMS**6*q_cut**6 + 105*mcMS**8*q_cut**6 + 88*mbkin**6*q_cut**7 - 
             284*mbkin**4*mcMS**2*q_cut**7 - 284*mbkin**2*mcMS**4*q_cut**7 + 
             88*mcMS**6*q_cut**7 - 35*mbkin**4*q_cut**8 + 70*mbkin**2*mcMS**2*q_cut**8 - 
             35*mcMS**4*q_cut**8 - 60*mbkin**2*q_cut**9 - 60*mcMS**2*q_cut**9 + 35*q_cut**10)*
            np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                  mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**28 + 
          (2592*mcMS**8*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
              2*mcMS**2*q_cut + q_cut**2)**2*(423*mbkin**14 + 279*mbkin**12*mcMS**2 - 
             27945*mbkin**10*mcMS**4 - 97497*mbkin**8*mcMS**6 - 
             97497*mbkin**6*mcMS**8 - 27945*mbkin**4*mcMS**10 + 
             279*mbkin**2*mcMS**12 + 423*mcMS**14 - 417*mbkin**12*q_cut - 
             3492*mbkin**10*mcMS**2*q_cut - 9873*mbkin**8*mcMS**4*q_cut - 
             14016*mbkin**6*mcMS**6*q_cut - 9873*mbkin**4*mcMS**8*q_cut - 
             3492*mbkin**2*mcMS**10*q_cut - 417*mcMS**12*q_cut - 417*mbkin**10*q_cut**2 - 
             3897*mbkin**8*mcMS**2*q_cut**2 - 10932*mbkin**6*mcMS**4*q_cut**2 - 
             10932*mbkin**4*mcMS**6*q_cut**2 - 3897*mbkin**2*mcMS**8*q_cut**2 - 
             417*mcMS**10*q_cut**2 + 423*mbkin**8*q_cut**3 + 3264*mbkin**6*mcMS**2*q_cut**3 + 
             5892*mbkin**4*mcMS**4*q_cut**3 + 3264*mbkin**2*mcMS**6*q_cut**3 + 
             423*mcMS**8*q_cut**3 + 3*mbkin**6*q_cut**4 - 69*mbkin**4*mcMS**2*q_cut**4 - 
             69*mbkin**2*mcMS**4*q_cut**4 + 3*mcMS**6*q_cut**4 - 25*mbkin**4*q_cut**5 + 
             20*mbkin**2*mcMS**2*q_cut**5 - 25*mcMS**4*q_cut**5 - 25*mbkin**2*q_cut**6 - 
             25*mcMS**2*q_cut**6 + 35*q_cut**7)*np.log((mbkin**2 + mcMS**2 - q_cut - 
                mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                    2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                    mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/
           mbkin**26 - (6531840*mcMS**12*(mbkin**8 + 8*mbkin**6*mcMS**2 + 
             15*mbkin**4*mcMS**4 + 8*mbkin**2*mcMS**6 + mcMS**8)*
            ((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                q_cut + q_cut**2)/mbkin**4)**(3/2)*np.log((mbkin**2 + mcMS**2 - q_cut - 
                mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                    2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                    mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/
           mbkin**16)*(((8*mbkin**2*(3 + 8*mbkin))/
             (9*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)) - 
            (3*mbkin**4*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                 (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + (4*q_cut*
                 (6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                  12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**6)))/(2*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
              ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
               q_cut**2/mbkin**4)))/((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                  2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 
                7*mbkin**4*mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - 
                mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
             (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                      2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                      2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
            3 - (3*mbkin**4*((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                  2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 
                7*mbkin**4*mcMS**2 - 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - 
                mcMS**4*q_cut - mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3)*(
                (-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                  (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                   9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                   12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                 (9*mbkin**6)))/(2*mbkin**6*((-1 + mcMS**2/mbkin**2)**2 - 
                (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)) + 
             np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*((-4*(3 + 8*mbkin)*q_cut**3)/
                (3*mbkin**8) - (14*mcMS**2*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) - (28*mcMS**4*
                 (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**6) + (2*mcMS**6*(-6 - 16*mbkin + 12*mbkin**2 + 
                  9*mbkin**2*np.log(mu0**2/mcMS**2)))/(3*mbkin**8) + (2*q_cut**2*
                 (12*mbkin**2 + 32*mbkin**3 + 18*mcMS**2 + 48*mbkin*mcMS**2 - 
                  12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
                (9*mbkin**8) + (4*q_cut*(3*mbkin**4 + 8*mbkin**5 + 9*mcMS**4 + 
                  24*mbkin*mcMS**4 - 12*mbkin**2*mcMS**4 - 9*mbkin**2*mcMS**4*
                   np.log(mu0**2/mcMS**2)))/(9*mbkin**8)) - 
             12*((mcMS**4*(16/3 + 4*np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - 
                    q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                   (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4))))/mbkin**4 + mcMS**4*
                ((-8*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 6*mbkin**2*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) + 16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                     np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                         q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                    9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                        mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                        2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                         q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                     mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                        2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                     np.log(mu0**2/mcMS**2)))/(9*mbkin**4*(mbkin**4 - 2*mbkin**2*
                     mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)*
                   (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                         mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                       mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + mbkin**2*
                     np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                         q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                 (8*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                      mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                          q_cut**2)/mbkin**4))))/(9*mbkin**6)))))/
           (((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                q_cut + q_cut**2)/mbkin**4)**(3/2)*
            ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 
                 7*mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - 
                 mbkin**2*q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
              (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                       2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                       2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**
                4)**4)) + (mbkin**4*(18*mbkin**4*((-1 + mcMS**2/mbkin**2)**2*(1 - 
                (7*mcMS**2)/mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/mbkin**6) + 
              ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + (14*mcMS**6)/
                  mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/mbkin**2 + 
              (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*mcMS**4 + mcMS**6)*
                q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/
               mbkin**10 - (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + 
              q_cut**5/mbkin**10)**2*(3*(1 - (47*mcMS**2)/mbkin**2 - (2595*mcMS**4)/
                mbkin**4 - (11219*mcMS**6)/mbkin**6 - (11219*mcMS**8)/mbkin**8 - 
               (2595*mcMS**10)/mbkin**10 - (47*mcMS**12)/mbkin**12 + mcMS**14/
                mbkin**14) + (3*(mbkin**12 - 44*mbkin**10*mcMS**2 - 1051*mbkin**8*
                 mcMS**4 - 2432*mbkin**6*mcMS**6 - 1051*mbkin**4*mcMS**8 - 
                44*mbkin**2*mcMS**10 + mcMS**12)*q_cut)/mbkin**14 + 
             (3*(mbkin**10 - 39*mbkin**8*mcMS**2 - 424*mbkin**6*mcMS**4 - 
                424*mbkin**4*mcMS**6 - 39*mbkin**2*mcMS**8 + mcMS**10)*q_cut**2)/
              mbkin**14 + (3*(mbkin**8 - 32*mbkin**6*mcMS**2 - 136*mbkin**4*
                 mcMS**4 - 32*mbkin**2*mcMS**6 + mcMS**8)*q_cut**3)/mbkin**14 + 
             (3*(mbkin**6 - 23*mbkin**4*mcMS**2 - 23*mbkin**2*mcMS**4 + mcMS**6)*q_cut**
                4)/mbkin**14 - (5*(5 - (4*mcMS**2)/mbkin**2 + (5*mcMS**4)/
                 mbkin**4)*q_cut**5)/mbkin**10 - (25*(mbkin**2 + mcMS**2)*q_cut**6)/
              mbkin**14 + (35*q_cut**7)/mbkin**14)*((-8*(3 + 8*mbkin)*q_cut**2)/
              (9*mbkin**6) + (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 
                12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
             (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*mcMS**2 - 
                12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/
              (9*mbkin**6)) + ((-1 + mcMS**2/mbkin**2)**2 - 
             (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
            ((64*mbkin*(-4*(-1 + mcMS**2/mbkin**2)**4*(1 + mcMS**2/mbkin**2)**2*
                 (1179 - (27994*mcMS**2)/mbkin**2 + (151119*mcMS**4)/mbkin**4 + 
                  (982305*mcMS**6)/mbkin**6 - (6893010*mcMS**8)/mbkin**8 - 
                  (8620596*mcMS**10)/mbkin**10 + (1136781*mcMS**12)/mbkin**12 + 
                  (171093*mcMS**14)/mbkin**14 - (137043*mcMS**16)/mbkin**16 + 
                  (13572*mcMS**18)/mbkin**18 + (154*mcMS**20)/mbkin**20) + 
                ((-1 + mcMS**2/mbkin**2)**2*(26163 - (491315*mcMS**2)/mbkin**2 + 
                   (609844*mcMS**4)/mbkin**4 + (29190880*mcMS**6)/mbkin**6 - 
                   (45147167*mcMS**8)/mbkin**8 - (424453401*mcMS**10)/mbkin**10 - 
                   (729940536*mcMS**12)/mbkin**12 - (481708704*mcMS**14)/
                    mbkin**14 - (67588491*mcMS**16)/mbkin**16 + (33128619*
                     mcMS**18)/mbkin**18 - (4511740*mcMS**20)/mbkin**20 - 
                   (1905472*mcMS**22)/mbkin**22 + (315767*mcMS**24)/mbkin**24 + 
                   (3233*mcMS**26)/mbkin**26)*q_cut)/mbkin**2 + 
                (4*(-12321 + (196421*mcMS**2)/mbkin**2 + (301638*mcMS**4)/
                    mbkin**4 - (12526887*mcMS**6)/mbkin**6 + (1399932*mcMS**8)/
                    mbkin**8 + (127033574*mcMS**10)/mbkin**10 + (284550615*
                     mcMS**12)/mbkin**12 + (300700218*mcMS**14)/mbkin**14 + 
                   (150895899*mcMS**16)/mbkin**16 + (5517729*mcMS**18)/
                    mbkin**18 - (15244856*mcMS**20)/mbkin**20 + 
                   (2907981*mcMS**22)/mbkin**22 + (671466*mcMS**24)/mbkin**24 - 
                   (153900*mcMS**26)/mbkin**26 - (1349*mcMS**28)/mbkin**28)*q_cut**2)/
                 mbkin**4 + ((14763 - (64787*mcMS**2)/mbkin**2 - 
                   (2819568*mcMS**4)/mbkin**4 + (7404252*mcMS**6)/mbkin**6 + 
                   (43250901*mcMS**8)/mbkin**8 - (6393169*mcMS**10)/mbkin**10 - 
                   (79307576*mcMS**12)/mbkin**12 - (4041216*mcMS**14)/mbkin**14 + 
                   (48737885*mcMS**16)/mbkin**16 + (556627*mcMS**18)/mbkin**18 - 
                   (8219224*mcMS**20)/mbkin**20 + (656612*mcMS**22)/mbkin**22 + 
                   (224419*mcMS**24)/mbkin**24 + (81*mcMS**26)/mbkin**26)*q_cut**3)/
                 mbkin**6 + (4*(18273 - (235702*mcMS**2)/mbkin**2 - 
                   (1086112*mcMS**4)/mbkin**4 + (11266643*mcMS**6)/mbkin**6 + 
                   (36598184*mcMS**8)/mbkin**8 + (45843340*mcMS**10)/mbkin**10 + 
                   (44974590*mcMS**12)/mbkin**12 + (36547042*mcMS**14)/
                    mbkin**14 + (10361895*mcMS**16)/mbkin**16 - 
                   (3866986*mcMS**18)/mbkin**18 - (424058*mcMS**20)/mbkin**20 + 
                   (224799*mcMS**22)/mbkin**22 + (2444*mcMS**24)/mbkin**24)*q_cut**4)/
                 mbkin**8 - ((98721 - (917479*mcMS**2)/mbkin**2 - 
                   (9760513*mcMS**4)/mbkin**4 + (30195627*mcMS**6)/mbkin**6 + 
                   (170463350*mcMS**8)/mbkin**8 + (259417590*mcMS**10)/
                    mbkin**10 + (167078226*mcMS**12)/mbkin**12 + 
                   (15469354*mcMS**14)/mbkin**14 - (21555727*mcMS**16)/
                    mbkin**16 + (973161*mcMS**18)/mbkin**18 + (1273159*mcMS**20)/
                    mbkin**20 + (10643*mcMS**22)/mbkin**22)*q_cut**5)/mbkin**10 + 
                (4*(5934 - (43901*mcMS**2)/mbkin**2 - (695976*mcMS**4)/mbkin**4 + 
                   (2309184*mcMS**6)/mbkin**6 + (11870802*mcMS**8)/mbkin**8 + 
                   (11318680*mcMS**10)/mbkin**10 + (724070*mcMS**12)/mbkin**12 - 
                   (1728188*mcMS**14)/mbkin**14 + (119948*mcMS**16)/mbkin**16 + 
                   (74017*mcMS**18)/mbkin**18 + (526*mcMS**20)/mbkin**20)*q_cut**6)/
                 mbkin**12 + ((44175 - (243147*mcMS**2)/mbkin**2 - 
                   (5579352*mcMS**4)/mbkin**4 - (17016744*mcMS**6)/mbkin**6 - 
                   (22924990*mcMS**8)/mbkin**8 - (14689794*mcMS**10)/mbkin**10 - 
                   (2030576*mcMS**12)/mbkin**12 + (2336816*mcMS**14)/mbkin**14 + 
                   (572671*mcMS**16)/mbkin**16 + (6717*mcMS**18)/mbkin**18)*q_cut**7)/
                 mbkin**14 - (4*(11646 - (53280*mcMS**2)/mbkin**2 - 
                   (1304004*mcMS**4)/mbkin**4 - (3516303*mcMS**6)/mbkin**6 - 
                   (3269188*mcMS**8)/mbkin**8 - (647651*mcMS**10)/mbkin**10 + 
                   (523431*mcMS**12)/mbkin**12 + (121214*mcMS**14)/mbkin**14 + 
                   (3663*mcMS**16)/mbkin**16)*q_cut**8)/mbkin**16 + 
                ((24633 - (76675*mcMS**2)/mbkin**2 - (1532353*mcMS**4)/mbkin**4 - 
                   (3216569*mcMS**6)/mbkin**6 - (1176415*mcMS**8)/mbkin**8 + 
                   (448897*mcMS**10)/mbkin**10 + (145151*mcMS**12)/mbkin**12 + 
                   (15283*mcMS**14)/mbkin**14)*q_cut**9)/mbkin**18 - 
                (4*(1197 - (4127*mcMS**2)/mbkin**2 - (21706*mcMS**4)/mbkin**4 - 
                   (39039*mcMS**6)/mbkin**6 - (14018*mcMS**8)/mbkin**8 + 
                   (2910*mcMS**10)/mbkin**10 + (449*mcMS**12)/mbkin**12)*q_cut**10)/
                 mbkin**20 - ((8559 + (20945*mcMS**2)/mbkin**2 + (29208*mcMS**4)/
                    mbkin**4 + (54804*mcMS**6)/mbkin**6 + (34199*mcMS**8)/
                    mbkin**8 + (11197*mcMS**10)/mbkin**10)*q_cut**11)/mbkin**22 + 
                (4*(2121 + (4502*mcMS**2)/mbkin**2 + (8820*mcMS**4)/mbkin**4 + 
                   (9231*mcMS**6)/mbkin**6 + (2526*mcMS**8)/mbkin**8)*q_cut**12)/
                 mbkin**24 - ((2379 + (5003*mcMS**2)/mbkin**2 + (11693*mcMS**4)/
                    mbkin**4 + (3329*mcMS**6)/mbkin**6)*q_cut**13)/mbkin**26 + 
                (4*mcMS**2*(137 + (104*mcMS**2)/mbkin**2)*q_cut**14)/mbkin**30 - 
                (5*(15 + (29*mcMS**2)/mbkin**2)*q_cut**15)/mbkin**30 + (60*q_cut**16)/
                 mbkin**32))/9 + 18*(mbkin**4*((-1 + mcMS**2/mbkin**2)**2*
                   (1 - (7*mcMS**2)/mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/
                     mbkin**6) + ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/
                      mbkin**4 + (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*q_cut)/
                   mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*
                      mcMS**4 + mcMS**6)*q_cut**2)/mbkin**10 + (2*(mbkin**4 + 
                     mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/mbkin**10 - 
                  (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2*
                ((-980*(3 + 8*mbkin)*q_cut**7)/(9*mbkin**16) - 
                 (2*(47*mbkin**12*mcMS**2 + 5190*mbkin**10*mcMS**4 + 33657*
                     mbkin**8*mcMS**6 + 44876*mbkin**6*mcMS**8 + 12975*mbkin**4*
                     mcMS**10 + 282*mbkin**2*mcMS**12 - 7*mcMS**14)*(-6 - 
                    16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                  (3*mbkin**16) + (50*q_cut**6*(36*mbkin**2 + 96*mbkin**3 + 
                    42*mcMS**2 + 112*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 
                    9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**16) - 
                 5*q_cut**5*((-20*(3 + 8*mbkin)*(5 - (4*mcMS**2)/mbkin**2 + 
                      (5*mcMS**4)/mbkin**4))/(9*mbkin**12) - 
                   (4*(2*mbkin**2*mcMS**2 - 5*mcMS**4)*(-6 - 16*mbkin + 
                      12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                    (9*mbkin**16)) + 3*q_cut**4*((-16*(3 + 8*mbkin)*(1 - 
                      (23*mcMS**2)/mbkin**2 - (23*mcMS**4)/mbkin**4 + mcMS**6/
                       mbkin**6))/(9*mbkin**10) - (2*(23*mbkin**4*mcMS**2 + 
                      46*mbkin**2*mcMS**4 - 3*mcMS**6)*(-6 - 16*mbkin + 
                      12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                    (9*mbkin**16)) + 3*q_cut**3*((-4*(3 + 8*mbkin)*(1 - 
                      (32*mcMS**2)/mbkin**2 - (136*mcMS**4)/mbkin**4 - 
                      (32*mcMS**6)/mbkin**6 + mcMS**8/mbkin**8))/(3*mbkin**8) - 
                   (8*(8*mbkin**6*mcMS**2 + 68*mbkin**4*mcMS**4 + 24*mbkin**2*
                       mcMS**6 - mcMS**8)*(-6 - 16*mbkin + 12*mbkin**2 + 
                      9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**16)) + 
                 3*q_cut**2*((-8*(3 + 8*mbkin)*(1 - (39*mcMS**2)/mbkin**2 - 
                      (424*mcMS**4)/mbkin**4 - (424*mcMS**6)/mbkin**6 - 
                      (39*mcMS**8)/mbkin**8 + mcMS**10/mbkin**10))/(9*mbkin**6) - 
                   (2*(39*mbkin**8*mcMS**2 + 848*mbkin**6*mcMS**4 + 1272*mbkin**4*
                       mcMS**6 + 156*mbkin**2*mcMS**8 - 5*mcMS**10)*(-6 - 
                      16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                    (9*mbkin**16)) + 3*q_cut*((-4*(3 + 8*mbkin)*(1 - (44*mcMS**2)/
                       mbkin**2 - (1051*mcMS**4)/mbkin**4 - (2432*mcMS**6)/
                       mbkin**6 - (1051*mcMS**8)/mbkin**8 - (44*mcMS**10)/
                       mbkin**10 + mcMS**12/mbkin**12))/(9*mbkin**4) - 
                   (4*(22*mbkin**10*mcMS**2 + 1051*mbkin**8*mcMS**4 + 3648*
                       mbkin**6*mcMS**6 + 2102*mbkin**4*mcMS**8 + 110*mbkin**2*
                       mcMS**10 - 3*mcMS**12)*(-6 - 16*mbkin + 12*mbkin**2 + 
                      9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**16))) + 
               (3*(1 - (47*mcMS**2)/mbkin**2 - (2595*mcMS**4)/mbkin**4 - 
                   (11219*mcMS**6)/mbkin**6 - (11219*mcMS**8)/mbkin**8 - 
                   (2595*mcMS**10)/mbkin**10 - (47*mcMS**12)/mbkin**12 + 
                   mcMS**14/mbkin**14) + (3*(mbkin**12 - 44*mbkin**10*mcMS**2 - 
                    1051*mbkin**8*mcMS**4 - 2432*mbkin**6*mcMS**6 - 1051*mbkin**4*
                     mcMS**8 - 44*mbkin**2*mcMS**10 + mcMS**12)*q_cut)/mbkin**14 + 
                 (3*(mbkin**10 - 39*mbkin**8*mcMS**2 - 424*mbkin**6*mcMS**4 - 
                    424*mbkin**4*mcMS**6 - 39*mbkin**2*mcMS**8 + mcMS**10)*q_cut**2)/
                  mbkin**14 + (3*(mbkin**8 - 32*mbkin**6*mcMS**2 - 136*mbkin**4*
                     mcMS**4 - 32*mbkin**2*mcMS**6 + mcMS**8)*q_cut**3)/mbkin**14 + 
                 (3*(mbkin**6 - 23*mbkin**4*mcMS**2 - 23*mbkin**2*mcMS**4 + 
                    mcMS**6)*q_cut**4)/mbkin**14 - (5*(5 - (4*mcMS**2)/mbkin**2 + 
                    (5*mcMS**4)/mbkin**4)*q_cut**5)/mbkin**10 - 
                 (25*(mbkin**2 + mcMS**2)*q_cut**6)/mbkin**14 + (35*q_cut**7)/mbkin**14)*
                ((8*mbkin**2*(3 + 8*mbkin)*((-1 + mcMS**2/mbkin**2)**2*
                      (1 - (7*mcMS**2)/mbkin**2 - (7*mcMS**4)/mbkin**4 + 
                       mcMS**6/mbkin**6) + ((-3 + (14*mcMS**2)/mbkin**2 + 
                        (26*mcMS**4)/mbkin**4 + (14*mcMS**6)/mbkin**6 - 
                        (3*mcMS**8)/mbkin**8)*q_cut)/mbkin**2 + (2*(mbkin**6 - 
                        2*mbkin**4*mcMS**2 - 2*mbkin**2*mcMS**4 + mcMS**6)*q_cut**2)/
                      mbkin**10 + (2*(mbkin**4 + mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/
                      mbkin**10 - (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + 
                     q_cut**5/mbkin**10)**2)/9 + 2*mbkin**4*((-1 + mcMS**2/mbkin**2)**2*
                    (1 - (7*mcMS**2)/mbkin**2 - (7*mcMS**4)/mbkin**4 + mcMS**6/
                      mbkin**6) + ((-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/
                       mbkin**4 + (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8)*
                     q_cut)/mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mcMS**2 - 2*mbkin**2*
                       mcMS**4 + mcMS**6)*q_cut**2)/mbkin**10 + (2*(mbkin**4 + 
                      mbkin**2*mcMS**2 + mcMS**4)*q_cut**3)/mbkin**10 - 
                   (3*(mbkin**2 + mcMS**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)*
                  ((-20*(3 + 8*mbkin)*q_cut**5)/(9*mbkin**12) - 
                   (2*(mbkin**2 - mcMS**2)*(9*mbkin**6*mcMS**2 - 7*mbkin**4*
                       mcMS**4 - 31*mbkin**2*mcMS**6 + 5*mcMS**8)*(-6 - 
                      16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                    (9*mbkin**12) + (2*q_cut**4*(24*mbkin**2 + 64*mbkin**3 + 
                      30*mcMS**2 + 80*mbkin*mcMS**2 - 12*mbkin**2*mcMS**2 - 
                      9*mbkin**2*mcMS**2*np.log(mu0**2/mcMS**2)))/(3*mbkin**12) + 
                   2*q_cut**3*((-4*(3 + 8*mbkin)*(1 + mcMS**2/mbkin**2 + mcMS**4/
                         mbkin**4))/(3*mbkin**8) + (2*(mbkin**2*mcMS**2 + 
                        2*mcMS**4)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                         np.log(mu0**2/mcMS**2)))/(9*mbkin**12)) + 2*q_cut**2*
                    ((-8*(3 + 8*mbkin)*(1 - (2*mcMS**2)/mbkin**2 - (2*mcMS**4)/
                         mbkin**4 + mcMS**6/mbkin**6))/(9*mbkin**6) - 
                     (2*(2*mbkin**4*mcMS**2 + 4*mbkin**2*mcMS**4 - 3*mcMS**6)*
                       (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**12)) + q_cut*((-4*(3 + 8*mbkin)*
                       (-3 + (14*mcMS**2)/mbkin**2 + (26*mcMS**4)/mbkin**4 + 
                        (14*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8))/
                      (9*mbkin**4) + (4*(7*mbkin**6*mcMS**2 + 26*mbkin**4*
                         mcMS**4 + 21*mbkin**2*mcMS**6 - 6*mcMS**8)*(-6 - 
                        16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                      (9*mbkin**12)))))) - 
           6*(np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(72*mcMS**4*
                (9*(-1 + mcMS**4/mbkin**4)**2*(12 - (112*mcMS**2)/mbkin**2 - 
                   (1269*mcMS**4)/mbkin**4 + (7152*mcMS**6)/mbkin**6 + 
                   (30014*mcMS**8)/mbkin**8 + (7152*mcMS**10)/mbkin**10 - 
                   (1269*mcMS**12)/mbkin**12 - (112*mcMS**14)/mbkin**14 + 
                   (12*mcMS**16)/mbkin**16) - (6*(71 - (261*mcMS**2)/mbkin**2 - 
                    (7313*mcMS**4)/mbkin**4 + (699*mcMS**6)/mbkin**6 + 
                    (141606*mcMS**8)/mbkin**8 + (364158*mcMS**10)/mbkin**10 + 
                    (364158*mcMS**12)/mbkin**12 + (141606*mcMS**14)/mbkin**14 + 
                    (699*mcMS**16)/mbkin**16 - (7313*mcMS**18)/mbkin**18 - 
                    (261*mcMS**20)/mbkin**20 + (71*mcMS**22)/mbkin**22)*q_cut)/
                  mbkin**2 + (12*(35 + (70*mcMS**2)/mbkin**2 - (1887*mcMS**4)/
                     mbkin**4 - (7902*mcMS**6)/mbkin**6 - (8718*mcMS**8)/
                     mbkin**8 - (4776*mcMS**10)/mbkin**10 - (8718*mcMS**12)/
                     mbkin**12 - (7902*mcMS**14)/mbkin**14 - (1887*mcMS**16)/
                     mbkin**16 + (70*mcMS**18)/mbkin**18 + (35*mcMS**20)/
                     mbkin**20)*q_cut**2)/mbkin**4 + (6*(71 + (23*mcMS**2)/mbkin**2 - 
                    (7000*mcMS**4)/mbkin**4 - (32072*mcMS**6)/mbkin**6 - 
                    (55270*mcMS**8)/mbkin**8 - (55270*mcMS**10)/mbkin**10 - 
                    (32072*mcMS**12)/mbkin**12 - (7000*mcMS**14)/mbkin**14 + 
                    (23*mcMS**16)/mbkin**16 + (71*mcMS**18)/mbkin**18)*q_cut**3)/
                  mbkin**6 - (3*(351 + (1632*mcMS**2)/mbkin**2 - (11450*mcMS**4)/
                     mbkin**4 - (68080*mcMS**6)/mbkin**6 - (111678*mcMS**8)/
                     mbkin**8 - (68080*mcMS**10)/mbkin**10 - (11450*mcMS**12)/
                     mbkin**12 + (1632*mcMS**14)/mbkin**14 + (351*mcMS**16)/
                     mbkin**16)*q_cut**4)/mbkin**8 + (8*(49 + (301*mcMS**2)/
                     mbkin**2 - (909*mcMS**4)/mbkin**4 - (4193*mcMS**6)/mbkin**6 - 
                    (4193*mcMS**8)/mbkin**8 - (909*mcMS**10)/mbkin**10 + 
                    (301*mcMS**12)/mbkin**12 + (49*mcMS**14)/mbkin**14)*q_cut**5)/
                  mbkin**10 + (4*(119 + (966*mcMS**2)/mbkin**2 + (3327*mcMS**4)/
                     mbkin**4 + (4610*mcMS**6)/mbkin**6 + (3327*mcMS**8)/
                     mbkin**8 + (966*mcMS**10)/mbkin**10 + (119*mcMS**12)/
                     mbkin**12)*q_cut**6)/mbkin**12 - (120*(3 + (35*mcMS**2)/
                     mbkin**2 + (87*mcMS**4)/mbkin**4 + (87*mcMS**6)/mbkin**6 + 
                    (35*mcMS**8)/mbkin**8 + (3*mcMS**10)/mbkin**10)*q_cut**7)/
                  mbkin**14 + ((-106 + (1472*mcMS**2)/mbkin**2 + (2631*mcMS**4)/
                     mbkin**4 + (1472*mcMS**6)/mbkin**6 - (106*mcMS**8)/mbkin**8)*
                   q_cut**8)/mbkin**16 + (98*(mbkin**6 - 3*mbkin**4*mcMS**2 - 
                    3*mbkin**2*mcMS**4 + mcMS**6)*q_cut**9)/mbkin**24 + 
                 (120*(mbkin**2 + mcMS**2)**2*q_cut**10)/mbkin**24 - 
                 (130*(mbkin**2 + mcMS**2)*q_cut**11)/mbkin**24 + (35*q_cut**12)/
                  mbkin**24)*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
                 (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 
                    12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                 (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                     mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                     np.log(mu0**2/mcMS**2)))/(9*mbkin**6)) + 
               ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/mbkin**4 + 
                 q_cut**2/mbkin**4)*((-64*mbkin*(-((-1 + mcMS**2/mbkin**2)**2*
                      (201 - (2594*mcMS**2)/mbkin**2 - (18020*mcMS**4)/mbkin**4 + 
                       (686586*mcMS**6)/mbkin**6 - (484473*mcMS**8)/mbkin**8 - 
                       (24524568*mcMS**10)/mbkin**10 - (51017352*mcMS**12)/
                        mbkin**12 - (29833656*mcMS**14)/mbkin**14 - 
                       (1099137*mcMS**16)/mbkin**16 + (793962*mcMS**18)/
                        mbkin**18 - (274084*mcMS**20)/mbkin**20 - (9490*mcMS**22)/
                        mbkin**22 + (3105*mcMS**24)/mbkin**24)) + 
                    (2*(411 - (4441*mcMS**2)/mbkin**2 - (47413*mcMS**4)/
                        mbkin**4 + (998110*mcMS**6)/mbkin**6 + (1955322*mcMS**8)/
                        mbkin**8 - (24020691*mcMS**10)/mbkin**10 - (77841426*
                         mcMS**12)/mbkin**12 - (84248520*mcMS**14)/mbkin**14 - 
                       (31464921*mcMS**16)/mbkin**16 + (2471757*mcMS**18)/
                        mbkin**18 + (1072279*mcMS**20)/mbkin**20 - 
                       (424934*mcMS**22)/mbkin**22 - (10828*mcMS**24)/mbkin**24 + 
                       (6255*mcMS**26)/mbkin**26)*q_cut)/mbkin**2 - 
                    (2*(420 - (2800*mcMS**2)/mbkin**2 - (64131*mcMS**4)/
                        mbkin**4 + (406035*mcMS**6)/mbkin**6 + (2915250*mcMS**8)/
                        mbkin**8 + (3976458*mcMS**10)/mbkin**10 + (2350824*
                         mcMS**12)/mbkin**12 + (4693344*mcMS**14)/mbkin**14 + 
                       (3558342*mcMS**16)/mbkin**16 + (48462*mcMS**18)/mbkin**
                         18 - (268445*mcMS**20)/mbkin**20 + (9861*mcMS**22)/
                        mbkin**22 + (6300*mcMS**24)/mbkin**24)*q_cut**2)/mbkin**4 - 
                    (2*(411 - (2797*mcMS**2)/mbkin**2 - (61931*mcMS**4)/
                        mbkin**4 + (744527*mcMS**6)/mbkin**6 + (5541108*mcMS**8)/
                        mbkin**8 + (10489326*mcMS**10)/mbkin**10 + (10608076*
                         mcMS**12)/mbkin**12 + (5830586*mcMS**14)/mbkin**14 + 
                       (187913*mcMS**16)/mbkin**16 - (376185*mcMS**18)/mbkin**
                         18 + (8423*mcMS**20)/mbkin**20 + (6255*mcMS**22)/
                        mbkin**22)*q_cut**3)/mbkin**6 + ((2091 - (7688*mcMS**2)/
                        mbkin**2 - (351965*mcMS**4)/mbkin**4 + (922140*mcMS**6)/
                        mbkin**6 + (11699682*mcMS**8)/mbkin**8 + (22377552*
                         mcMS**10)/mbkin**10 + (11839006*mcMS**12)/mbkin**12 - 
                       (747556*mcMS**14)/mbkin**14 - (969997*mcMS**16)/mbkin**
                         16 + (118544*mcMS**18)/mbkin**18 + (31455*mcMS**20)/
                        mbkin**20)*q_cut**4)/mbkin**8 - (4*(231 - (1001*mcMS**2)/
                        mbkin**2 - (36378*mcMS**4)/mbkin**4 + (32226*mcMS**6)/
                        mbkin**6 + (485535*mcMS**8)/mbkin**8 + (473268*mcMS**10)/
                        mbkin**10 - (63132*mcMS**12)/mbkin**12 - (76864*mcMS**14)/
                        mbkin**14 + (10500*mcMS**16)/mbkin**16 + (3255*mcMS**18)/
                        mbkin**18)*q_cut**5)/mbkin**10 - (4*(168 + (532*mcMS**2)/
                        mbkin**2 - (37666*mcMS**4)/mbkin**4 - (173612*mcMS**6)/
                        mbkin**6 - (257329*mcMS**8)/mbkin**8 - (157955*mcMS**10)/
                        mbkin**10 + (6303*mcMS**12)/mbkin**12 + (24871*mcMS**14)/
                        mbkin**14 + (2940*mcMS**16)/mbkin**16)*q_cut**6)/mbkin**12 + 
                    (4*(255 - (245*mcMS**2)/mbkin**2 - (39116*mcMS**4)/mbkin**4 - 
                       (138249*mcMS**6)/mbkin**6 - (129466*mcMS**8)/mbkin**8 - 
                       (11635*mcMS**10)/mbkin**10 + (18713*mcMS**12)/mbkin**12 + 
                       (3375*mcMS**14)/mbkin**14)*q_cut**7)/mbkin**14 + 
                    ((-843 + (2036*mcMS**2)/mbkin**2 + (42441*mcMS**4)/mbkin**4 + 
                       (95556*mcMS**6)/mbkin**6 + (47845*mcMS**8)/mbkin**8 - 
                       (5364*mcMS**10)/mbkin**10 - (6315*mcMS**12)/mbkin**12)*
                      q_cut**8)/mbkin**16 + (2*(147 - (441*mcMS**2)/mbkin**2 - 
                       (111*mcMS**4)/mbkin**4 - (2794*mcMS**6)/mbkin**6 - 
                       (2452*mcMS**8)/mbkin**8 + (735*mcMS**10)/mbkin**10)*q_cut**9)/
                     mbkin**18 + (10*(36 + (72*mcMS**2)/mbkin**2 + (93*mcMS**4)/
                        mbkin**4 + (223*mcMS**6)/mbkin**6 + (180*mcMS**8)/
                        mbkin**8)*q_cut**10)/mbkin**20 - (30*(13 + (13*mcMS**2)/
                        mbkin**2 + (45*mcMS**4)/mbkin**4 + (65*mcMS**6)/mbkin**6)*
                      q_cut**11)/mbkin**22 + (105*(mbkin**4 + 5*mcMS**4)*q_cut**12)/
                     mbkin**28))/9 + 72*(mcMS**4*(9*(-1 + mcMS**4/mbkin**4)**2*
                      (12 - (112*mcMS**2)/mbkin**2 - (1269*mcMS**4)/mbkin**4 + 
                       (7152*mcMS**6)/mbkin**6 + (30014*mcMS**8)/mbkin**8 + 
                       (7152*mcMS**10)/mbkin**10 - (1269*mcMS**12)/mbkin**12 - 
                       (112*mcMS**14)/mbkin**14 + (12*mcMS**16)/mbkin**16) - 
                     (6*(71 - (261*mcMS**2)/mbkin**2 - (7313*mcMS**4)/mbkin**4 + 
                        (699*mcMS**6)/mbkin**6 + (141606*mcMS**8)/mbkin**8 + 
                        (364158*mcMS**10)/mbkin**10 + (364158*mcMS**12)/
                         mbkin**12 + (141606*mcMS**14)/mbkin**14 + (699*mcMS**16)/
                         mbkin**16 - (7313*mcMS**18)/mbkin**18 - (261*mcMS**20)/
                         mbkin**20 + (71*mcMS**22)/mbkin**22)*q_cut)/mbkin**2 + 
                     (12*(35 + (70*mcMS**2)/mbkin**2 - (1887*mcMS**4)/mbkin**4 - 
                        (7902*mcMS**6)/mbkin**6 - (8718*mcMS**8)/mbkin**8 - 
                        (4776*mcMS**10)/mbkin**10 - (8718*mcMS**12)/mbkin**12 - 
                        (7902*mcMS**14)/mbkin**14 - (1887*mcMS**16)/mbkin**16 + 
                        (70*mcMS**18)/mbkin**18 + (35*mcMS**20)/mbkin**20)*q_cut**2)/
                      mbkin**4 + (6*(71 + (23*mcMS**2)/mbkin**2 - (7000*mcMS**4)/
                         mbkin**4 - (32072*mcMS**6)/mbkin**6 - (55270*mcMS**8)/
                         mbkin**8 - (55270*mcMS**10)/mbkin**10 - (32072*mcMS**12)/
                         mbkin**12 - (7000*mcMS**14)/mbkin**14 + (23*mcMS**16)/
                         mbkin**16 + (71*mcMS**18)/mbkin**18)*q_cut**3)/mbkin**6 - 
                     (3*(351 + (1632*mcMS**2)/mbkin**2 - (11450*mcMS**4)/
                         mbkin**4 - (68080*mcMS**6)/mbkin**6 - (111678*mcMS**8)/
                         mbkin**8 - (68080*mcMS**10)/mbkin**10 - (11450*mcMS**12)/
                         mbkin**12 + (1632*mcMS**14)/mbkin**14 + (351*mcMS**16)/
                         mbkin**16)*q_cut**4)/mbkin**8 + (8*(49 + (301*mcMS**2)/
                         mbkin**2 - (909*mcMS**4)/mbkin**4 - (4193*mcMS**6)/
                         mbkin**6 - (4193*mcMS**8)/mbkin**8 - (909*mcMS**10)/
                         mbkin**10 + (301*mcMS**12)/mbkin**12 + (49*mcMS**14)/
                         mbkin**14)*q_cut**5)/mbkin**10 + (4*(119 + (966*mcMS**2)/
                         mbkin**2 + (3327*mcMS**4)/mbkin**4 + (4610*mcMS**6)/
                         mbkin**6 + (3327*mcMS**8)/mbkin**8 + (966*mcMS**10)/
                         mbkin**10 + (119*mcMS**12)/mbkin**12)*q_cut**6)/mbkin**12 - 
                     (120*(3 + (35*mcMS**2)/mbkin**2 + (87*mcMS**4)/mbkin**4 + 
                        (87*mcMS**6)/mbkin**6 + (35*mcMS**8)/mbkin**8 + 
                        (3*mcMS**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
                     ((-106 + (1472*mcMS**2)/mbkin**2 + (2631*mcMS**4)/mbkin**4 + 
                        (1472*mcMS**6)/mbkin**6 - (106*mcMS**8)/mbkin**8)*q_cut**8)/
                      mbkin**16 + (98*(mbkin**6 - 3*mbkin**4*mcMS**2 - 3*mbkin**2*
                         mcMS**4 + mcMS**6)*q_cut**9)/mbkin**24 + 
                     (120*(mbkin**2 + mcMS**2)**2*q_cut**10)/mbkin**24 - 
                     (130*(mbkin**2 + mcMS**2)*q_cut**11)/mbkin**24 + (35*q_cut**12)/
                      mbkin**24)*(16/3 + 4*np.log(mu0**2/mcMS**2)) + mcMS**4*
                    ((-560*(3 + 8*mbkin)*q_cut**12)/(3*mbkin**26) - 
                     (4*(mbkin**4 - mcMS**4)*(56*mbkin**18*mcMS**2 + 1293*
                         mbkin**16*mcMS**4 - 11008*mbkin**14*mcMS**6 - 63835*
                         mbkin**12*mcMS**8 + 7152*mbkin**10*mcMS**10 + 123863*
                         mbkin**8*mcMS**12 + 32576*mbkin**6*mcMS**14 - 6393*
                         mbkin**4*mcMS**16 - 616*mbkin**2*mcMS**18 + 72*mcMS**20)*
                       (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/mbkin**26 - (160*(mbkin**2 + mcMS**2)*q_cut**10*
                       (30*mbkin**2 + 80*mbkin**3 + 36*mcMS**2 + 96*mbkin*
                         mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                         np.log(mu0**2/mcMS**2)))/(3*mbkin**26) + (260*q_cut**11*
                       (66*mbkin**2 + 176*mbkin**3 + 72*mcMS**2 + 192*mbkin*
                         mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                         np.log(mu0**2/mcMS**2)))/(9*mbkin**26) + 98*q_cut**9*
                      ((-4*(3 + 8*mbkin)*(1 - (3*mcMS**2)/mbkin**2 - (3*mcMS**4)/
                          mbkin**4 + mcMS**6/mbkin**6))/mbkin**20 - 
                       (2*(mbkin**4*mcMS**2 + 2*mbkin**2*mcMS**4 - mcMS**6)*
                         (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/(3*mbkin**26)) + q_cut**8*((-32*(3 + 8*mbkin)*
                         (-106 + (1472*mcMS**2)/mbkin**2 + (2631*mcMS**4)/
                          mbkin**4 + (1472*mcMS**6)/mbkin**6 - (106*mcMS**8)/
                          mbkin**8))/(9*mbkin**18) + (4*(736*mbkin**6*mcMS**2 + 
                          2631*mbkin**4*mcMS**4 + 2208*mbkin**2*mcMS**6 - 
                          212*mcMS**8)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**26)) - 120*q_cut**7*
                      ((-28*(3 + 8*mbkin)*(3 + (35*mcMS**2)/mbkin**2 + 
                          (87*mcMS**4)/mbkin**4 + (87*mcMS**6)/mbkin**6 + 
                          (35*mcMS**8)/mbkin**8 + (3*mcMS**10)/mbkin**10))/
                        (9*mbkin**16) + (2*(35*mbkin**8*mcMS**2 + 174*mbkin**6*
                          mcMS**4 + 261*mbkin**4*mcMS**6 + 140*mbkin**2*mcMS**8 + 
                          15*mcMS**10)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**26)) + 4*q_cut**6*
                      ((-8*(3 + 8*mbkin)*(119 + (966*mcMS**2)/mbkin**2 + 
                          (3327*mcMS**4)/mbkin**4 + (4610*mcMS**6)/mbkin**6 + 
                          (3327*mcMS**8)/mbkin**8 + (966*mcMS**10)/mbkin**10 + 
                          (119*mcMS**12)/mbkin**12))/(3*mbkin**14) + 
                       (4*(161*mbkin**10*mcMS**2 + 1109*mbkin**8*mcMS**4 + 
                          2305*mbkin**6*mcMS**6 + 2218*mbkin**4*mcMS**8 + 
                          805*mbkin**2*mcMS**10 + 119*mcMS**12)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                        (3*mbkin**26)) + 8*q_cut**5*((-20*(3 + 8*mbkin)*(49 + 
                          (301*mcMS**2)/mbkin**2 - (909*mcMS**4)/mbkin**4 - 
                          (4193*mcMS**6)/mbkin**6 - (4193*mcMS**8)/mbkin**8 - 
                          (909*mcMS**10)/mbkin**10 + (301*mcMS**12)/mbkin**12 + 
                          (49*mcMS**14)/mbkin**14))/(9*mbkin**12) + 
                       (2*(301*mbkin**12*mcMS**2 - 1818*mbkin**10*mcMS**4 - 
                          12579*mbkin**8*mcMS**6 - 16772*mbkin**6*mcMS**8 - 
                          4545*mbkin**4*mcMS**10 + 1806*mbkin**2*mcMS**12 + 
                          343*mcMS**14)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**26)) - 
                     3*q_cut**4*((-16*(3 + 8*mbkin)*(351 + (1632*mcMS**2)/mbkin**
                          2 - (11450*mcMS**4)/mbkin**4 - (68080*mcMS**6)/
                          mbkin**6 - (111678*mcMS**8)/mbkin**8 - (68080*mcMS**10)/
                          mbkin**10 - (11450*mcMS**12)/mbkin**12 + (1632*
                          mcMS**14)/mbkin**14 + (351*mcMS**16)/mbkin**16))/
                        (9*mbkin**10) + (8*(408*mbkin**14*mcMS**2 - 5725*
                          mbkin**12*mcMS**4 - 51060*mbkin**10*mcMS**6 - 111678*
                          mbkin**8*mcMS**8 - 85100*mbkin**6*mcMS**10 - 17175*
                          mbkin**4*mcMS**12 + 2856*mbkin**2*mcMS**14 + 702*
                          mcMS**16)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**26)) + 6*q_cut**3*
                      ((-4*(3 + 8*mbkin)*(71 + (23*mcMS**2)/mbkin**2 - 
                          (7000*mcMS**4)/mbkin**4 - (32072*mcMS**6)/mbkin**6 - 
                          (55270*mcMS**8)/mbkin**8 - (55270*mcMS**10)/mbkin**10 - 
                          (32072*mcMS**12)/mbkin**12 - (7000*mcMS**14)/mbkin**
                          14 + (23*mcMS**16)/mbkin**16 + (71*mcMS**18)/mbkin**
                          18))/(3*mbkin**8) + (2*(23*mbkin**16*mcMS**2 - 
                          14000*mbkin**14*mcMS**4 - 96216*mbkin**12*mcMS**6 - 
                          221080*mbkin**10*mcMS**8 - 276350*mbkin**8*mcMS**10 - 
                          192432*mbkin**6*mcMS**12 - 49000*mbkin**4*mcMS**14 + 
                          184*mbkin**2*mcMS**16 + 639*mcMS**18)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                        (9*mbkin**26)) + 12*q_cut**2*((-8*(3 + 8*mbkin)*(35 + 
                          (70*mcMS**2)/mbkin**2 - (1887*mcMS**4)/mbkin**4 - 
                          (7902*mcMS**6)/mbkin**6 - (8718*mcMS**8)/mbkin**8 - 
                          (4776*mcMS**10)/mbkin**10 - (8718*mcMS**12)/mbkin**12 - 
                          (7902*mcMS**14)/mbkin**14 - (1887*mcMS**16)/mbkin**16 + 
                          (70*mcMS**18)/mbkin**18 + (35*mcMS**20)/mbkin**20))/
                        (9*mbkin**6) + (4*(35*mbkin**18*mcMS**2 - 1887*mbkin**16*
                          mcMS**4 - 11853*mbkin**14*mcMS**6 - 17436*mbkin**12*
                          mcMS**8 - 11940*mbkin**10*mcMS**10 - 26154*mbkin**8*
                          mcMS**12 - 27657*mbkin**6*mcMS**14 - 7548*mbkin**4*
                          mcMS**16 + 315*mbkin**2*mcMS**18 + 175*mcMS**20)*
                         (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**26)) - 6*q_cut*((-4*(3 + 8*mbkin)*
                         (71 - (261*mcMS**2)/mbkin**2 - (7313*mcMS**4)/mbkin**4 + 
                          (699*mcMS**6)/mbkin**6 + (141606*mcMS**8)/mbkin**8 + 
                          (364158*mcMS**10)/mbkin**10 + (364158*mcMS**12)/
                          mbkin**12 + (141606*mcMS**14)/mbkin**14 + 
                          (699*mcMS**16)/mbkin**16 - (7313*mcMS**18)/mbkin**18 - 
                          (261*mcMS**20)/mbkin**20 + (71*mcMS**22)/mbkin**22))/
                        (9*mbkin**4) - (2*(261*mbkin**20*mcMS**2 + 14626*
                          mbkin**18*mcMS**4 - 2097*mbkin**16*mcMS**6 - 566424*
                          mbkin**14*mcMS**8 - 1820790*mbkin**12*mcMS**10 - 
                          2184948*mbkin**10*mcMS**12 - 991242*mbkin**8*mcMS**14 - 
                          5592*mbkin**6*mcMS**16 + 65817*mbkin**4*mcMS**18 + 
                          2610*mbkin**2*mcMS**20 - 781*mcMS**22)*(-6 - 
                          16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**
                          2)))/(9*mbkin**26))))))*np.log((mbkin**2 + mcMS**2 - q_cut - 
                 mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                     2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                      mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                    mbkin**4))) + (72*mcMS**4*(mbkin**4 - 2*mbkin**2*mcMS**2 + 
                 mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)**2*(
                108*mbkin**20 - 792*mbkin**18*mcMS**2 - 13329*mbkin**16*mcMS**4 + 
                40518*mbkin**14*mcMS**6 + 387441*mbkin**12*mcMS**8 + 
                668988*mbkin**10*mcMS**10 + 387441*mbkin**8*mcMS**12 + 
                40518*mbkin**6*mcMS**14 - 13329*mbkin**4*mcMS**16 - 792*mbkin**2*
                 mcMS**18 + 108*mcMS**20 - 210*mbkin**18*q_cut - 222*mbkin**16*
                 mcMS**2*q_cut + 15402*mbkin**14*mcMS**4*q_cut + 81210*mbkin**12*mcMS**6*
                 q_cut + 153300*mbkin**10*mcMS**8*q_cut + 153300*mbkin**8*mcMS**10*q_cut + 
                81210*mbkin**6*mcMS**12*q_cut + 15402*mbkin**4*mcMS**14*q_cut - 
                222*mbkin**2*mcMS**16*q_cut - 210*mcMS**18*q_cut - 108*mbkin**16*q_cut**2 + 
                552*mbkin**14*mcMS**2*q_cut**2 + 22257*mbkin**12*mcMS**4*q_cut**2 + 
                101844*mbkin**10*mcMS**6*q_cut**2 + 158394*mbkin**8*mcMS**8*q_cut**2 + 
                101844*mbkin**6*mcMS**10*q_cut**2 + 22257*mbkin**4*mcMS**12*q_cut**2 + 
                552*mbkin**2*mcMS**14*q_cut**2 - 108*mcMS**16*q_cut**2 + 420*mbkin**14*
                 q_cut**3 + 2088*mbkin**12*mcMS**2*q_cut**3 - 8028*mbkin**10*mcMS**4*
                 q_cut**3 - 43584*mbkin**8*mcMS**6*q_cut**3 - 43584*mbkin**6*mcMS**8*
                 q_cut**3 - 8028*mbkin**4*mcMS**10*q_cut**3 + 2088*mbkin**2*mcMS**12*
                 q_cut**3 + 420*mcMS**14*q_cut**3 - 105*mbkin**12*q_cut**4 - 642*mbkin**10*
                 mcMS**2*q_cut**4 - 966*mbkin**8*mcMS**4*q_cut**4 - 2118*mbkin**6*mcMS**6*
                 q_cut**4 - 966*mbkin**4*mcMS**8*q_cut**4 - 642*mbkin**2*mcMS**10*q_cut**4 - 
                105*mcMS**12*q_cut**4 - 238*mbkin**10*q_cut**5 - 1650*mbkin**8*mcMS**2*
                 q_cut**5 - 5522*mbkin**6*mcMS**4*q_cut**5 - 5522*mbkin**4*mcMS**6*q_cut**5 - 
                1650*mbkin**2*mcMS**8*q_cut**5 - 238*mcMS**10*q_cut**5 + 105*mbkin**8*
                 q_cut**6 + 940*mbkin**6*mcMS**2*q_cut**6 + 1705*mbkin**4*mcMS**4*q_cut**6 + 
                940*mbkin**2*mcMS**6*q_cut**6 + 105*mcMS**8*q_cut**6 + 88*mbkin**6*q_cut**7 - 
                284*mbkin**4*mcMS**2*q_cut**7 - 284*mbkin**2*mcMS**4*q_cut**7 + 
                88*mcMS**6*q_cut**7 - 35*mbkin**4*q_cut**8 + 70*mbkin**2*mcMS**2*q_cut**8 - 
                35*mcMS**4*q_cut**8 - 60*mbkin**2*q_cut**9 - 60*mcMS**2*q_cut**9 + 35*q_cut**10)*
               ((-8*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                      q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(-6*mbkin**4*mcMS**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 16*mbkin**5*mcMS**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 12*mbkin**6*mcMS**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 6*mbkin**2*mcMS**4*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 16*mbkin**3*mcMS**4*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*mcMS**4*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 6*mbkin**2*mcMS**2*
                    q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*
                    q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*mcMS**2*
                    q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 9*mbkin**6*mcMS**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 
                   9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                       mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                    np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**2*q_cut*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                       2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2)))/
                 (9*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                   2*mcMS**2*q_cut + q_cut**2)*(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - 
                   mcMS**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                       mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**
                       4))) + (np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                     2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                  ((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                     (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                      9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                   (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                       mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                       np.log(mu0**2/mcMS**2)))/(9*mbkin**6))*np.log((mbkin**2 + 
                     mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                        mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/
                 (2*((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/
                    mbkin**4 + q_cut**2/mbkin**4))))/mbkin**28) - 
           72*((mcMS**4*(-36*mcMS**4*(9*(-1 + mcMS**2/mbkin**2)**2*(47 + 
                    (31*mcMS**2)/mbkin**2 - (3105*mcMS**4)/mbkin**4 - 
                    (10833*mcMS**6)/mbkin**6 - (10833*mcMS**8)/mbkin**8 - 
                    (3105*mcMS**10)/mbkin**10 + (31*mcMS**12)/mbkin**12 + 
                    (47*mcMS**14)/mbkin**14) - (3*(421 + (1354*mcMS**2)/
                      mbkin**2 - (17342*mcMS**4)/mbkin**4 - (84374*mcMS**6)/
                      mbkin**6 - (132758*mcMS**8)/mbkin**8 - (84374*mcMS**10)/
                      mbkin**10 - (17342*mcMS**12)/mbkin**12 + (1354*mcMS**14)/
                      mbkin**14 + (421*mcMS**16)/mbkin**16)*q_cut)/mbkin**2 + 
                  (6*(140 + (839*mcMS**2)/mbkin**2 - (795*mcMS**4)/mbkin**4 - 
                     (7114*mcMS**6)/mbkin**6 - (7114*mcMS**8)/mbkin**8 - 
                     (795*mcMS**10)/mbkin**10 + (839*mcMS**12)/mbkin**12 + 
                     (140*mcMS**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
                  (6*(140 + (1259*mcMS**2)/mbkin**2 + (3262*mcMS**4)/mbkin**4 + 
                     (4076*mcMS**6)/mbkin**6 + (3262*mcMS**8)/mbkin**8 + 
                     (1259*mcMS**10)/mbkin**10 + (140*mcMS**12)/mbkin**12)*q_cut**3)/
                   mbkin**6 - (6*(210 + (1891*mcMS**2)/mbkin**2 + (4862*mcMS**4)/
                      mbkin**4 + (4862*mcMS**6)/mbkin**6 + (1891*mcMS**8)/
                      mbkin**8 + (210*mcMS**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
                  ((392 + (3466*mcMS**2)/mbkin**2 + (6078*mcMS**4)/mbkin**4 + 
                     (3466*mcMS**6)/mbkin**6 + (392*mcMS**8)/mbkin**8)*q_cut**5)/
                   mbkin**10 + ((28 - (34*mcMS**2)/mbkin**2 - (34*mcMS**4)/
                      mbkin**4 + (28*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
                  (10*(6 + (5*mcMS**2)/mbkin**2 + (6*mcMS**4)/mbkin**4)*q_cut**7)/
                   mbkin**14 - (95*(mbkin**2 + mcMS**2)*q_cut**8)/mbkin**18 + 
                  (35*q_cut**9)/mbkin**18)*((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + 
                  (4*mcMS**2*(-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 
                     12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                  (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                      mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                      np.log(mu0**2/mcMS**2)))/(9*mbkin**6)) + 
                ((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*q_cut)/
                   mbkin**4 + q_cut**2/mbkin**4)*((64*mbkin*
                    (-((-1 + mcMS**2/mbkin**2)**2*(-219 - (3427*mcMS**2)/
                         mbkin**2 - (25220*mcMS**4)/mbkin**4 + (690492*mcMS**6)/
                         mbkin**6 + (3385394*mcMS**8)/mbkin**8 + (3853498*
                          mcMS**10)/mbkin**10 + (969876*mcMS**12)/mbkin**12 - 
                        (66556*mcMS**14)/mbkin**14 + (8017*mcMS**16)/mbkin**16 + 
                        (3105*mcMS**18)/mbkin**18)) + ((-639 - (12606*mcMS**2)/
                         mbkin**2 - (137707*mcMS**4)/mbkin**4 + (1200772*mcMS**6)/
                         mbkin**6 + (8153946*mcMS**8)/mbkin**8 + (14790212*
                          mcMS**10)/mbkin**10 + (9837754*mcMS**12)/mbkin**12 + 
                        (1656240*mcMS**14)/mbkin**14 - (260839*mcMS**16)/
                         mbkin**16 + (23302*mcMS**18)/mbkin**18 + (9405*mcMS**20)/
                         mbkin**20)*q_cut)/mbkin**2 - (2*(-210 - (4891*mcMS**2)/
                         mbkin**2 - (70825*mcMS**4)/mbkin**4 + (10961*mcMS**6)/
                         mbkin**6 + (726865*mcMS**8)/mbkin**8 + (829165*mcMS**10)/
                         mbkin**10 + (50797*mcMS**12)/mbkin**12 - (87657*
                          mcMS**14)/mbkin**14 + (11805*mcMS**16)/mbkin**16 + 
                        (3150*mcMS**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
                     (2*(210 + (5521*mcMS**2)/mbkin**2 + (90958*mcMS**4)/
                         mbkin**4 + (297789*mcMS**6)/mbkin**6 + (370334*mcMS**8)/
                         mbkin**8 + (299309*mcMS**10)/mbkin**10 + (83742*
                          mcMS**12)/mbkin**12 - (15585*mcMS**14)/mbkin**14 - 
                        (3150*mcMS**16)/mbkin**16)*q_cut**3)/mbkin**6 + 
                     (2*(-315 - (8304*mcMS**2)/mbkin**2 - (138468*mcMS**4)/
                         mbkin**4 - (460673*mcMS**6)/mbkin**6 - (479693*mcMS**8)/
                         mbkin**8 - (125948*mcMS**10)/mbkin**10 + (23580*
                          mcMS**12)/mbkin**12 + (4725*mcMS**14)/mbkin**14)*q_cut**4)/
                      mbkin**8 + ((126 + (5638*mcMS**2)/mbkin**2 + 
                        (85014*mcMS**4)/mbkin**4 + (204504*mcMS**6)/mbkin**6 + 
                        (84214*mcMS**8)/mbkin**8 - (13278*mcMS**10)/mbkin**10 - 
                        (3570*mcMS**12)/mbkin**12)*q_cut**5)/mbkin**10 + 
                     (2*(42 - (51*mcMS**2)/mbkin**2 + (159*mcMS**4)/mbkin**4 - 
                        (541*mcMS**6)/mbkin**6 - (31*mcMS**8)/mbkin**8 + 
                        (210*mcMS**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
                     (10*(18 + (15*mcMS**2)/mbkin**2 + (78*mcMS**4)/mbkin**4 + 
                        (17*mcMS**6)/mbkin**6 + (90*mcMS**8)/mbkin**8)*q_cut**7)/
                      mbkin**14 - (15*(19 + (19*mcMS**2)/mbkin**2 + (47*mcMS**4)/
                         mbkin**4 + (95*mcMS**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
                     (105*(mbkin**4 + 5*mcMS**4)*q_cut**9)/mbkin**22))/9 - 
                  36*(mcMS**4*(9*(-1 + mcMS**2/mbkin**2)**2*(47 + (31*mcMS**2)/
                         mbkin**2 - (3105*mcMS**4)/mbkin**4 - (10833*mcMS**6)/
                         mbkin**6 - (10833*mcMS**8)/mbkin**8 - (3105*mcMS**10)/
                         mbkin**10 + (31*mcMS**12)/mbkin**12 + (47*mcMS**14)/
                         mbkin**14) - (3*(421 + (1354*mcMS**2)/mbkin**2 - 
                         (17342*mcMS**4)/mbkin**4 - (84374*mcMS**6)/mbkin**6 - 
                         (132758*mcMS**8)/mbkin**8 - (84374*mcMS**10)/mbkin**10 - 
                         (17342*mcMS**12)/mbkin**12 + (1354*mcMS**14)/mbkin**14 + 
                         (421*mcMS**16)/mbkin**16)*q_cut)/mbkin**2 + 
                      (6*(140 + (839*mcMS**2)/mbkin**2 - (795*mcMS**4)/mbkin**4 - 
                         (7114*mcMS**6)/mbkin**6 - (7114*mcMS**8)/mbkin**8 - 
                         (795*mcMS**10)/mbkin**10 + (839*mcMS**12)/mbkin**12 + 
                         (140*mcMS**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
                      (6*(140 + (1259*mcMS**2)/mbkin**2 + (3262*mcMS**4)/
                          mbkin**4 + (4076*mcMS**6)/mbkin**6 + (3262*mcMS**8)/
                          mbkin**8 + (1259*mcMS**10)/mbkin**10 + (140*mcMS**12)/
                          mbkin**12)*q_cut**3)/mbkin**6 - (6*(210 + (1891*mcMS**2)/
                          mbkin**2 + (4862*mcMS**4)/mbkin**4 + (4862*mcMS**6)/
                          mbkin**6 + (1891*mcMS**8)/mbkin**8 + (210*mcMS**10)/
                          mbkin**10)*q_cut**4)/mbkin**8 + ((392 + (3466*mcMS**2)/
                          mbkin**2 + (6078*mcMS**4)/mbkin**4 + (3466*mcMS**6)/
                          mbkin**6 + (392*mcMS**8)/mbkin**8)*q_cut**5)/mbkin**10 + 
                      ((28 - (34*mcMS**2)/mbkin**2 - (34*mcMS**4)/mbkin**4 + 
                         (28*mcMS**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
                      (10*(6 + (5*mcMS**2)/mbkin**2 + (6*mcMS**4)/mbkin**4)*q_cut**7)/
                       mbkin**14 - (95*(mbkin**2 + mcMS**2)*q_cut**8)/mbkin**18 + 
                      (35*q_cut**9)/mbkin**18)*(16/3 + 4*np.log(mu0**2/mcMS**2)) + 
                    mcMS**4*((-140*(3 + 8*mbkin)*q_cut**9)/mbkin**20 - 
                      (6*(mbkin**2 - mcMS**2)*(21*mbkin**14*mcMS**2 + 2101*
                          mbkin**12*mcMS**4 + 6693*mbkin**10*mcMS**6 - 3611*
                          mbkin**8*mcMS**8 - 16491*mbkin**6*mcMS**10 - 7307*
                          mbkin**4*mcMS**12 - 27*mbkin**2*mcMS**14 + 141*mcMS**16)*
                        (-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/
                          mcMS**2)))/mbkin**20 + (190*q_cut**8*(48*mbkin**2 + 
                         128*mbkin**3 + 54*mcMS**2 + 144*mbkin*mcMS**2 - 
                         12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*np.log(mu0**2/
                          mcMS**2)))/(9*mbkin**20) + 10*q_cut**7*
                       ((-28*(3 + 8*mbkin)*(6 + (5*mcMS**2)/mbkin**2 + 
                          (6*mcMS**4)/mbkin**4))/(9*mbkin**16) + 
                        (2*(5*mbkin**2*mcMS**2 + 12*mcMS**4)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                         (9*mbkin**20)) + q_cut**6*((-8*(3 + 8*mbkin)*(28 - 
                          (34*mcMS**2)/mbkin**2 - (34*mcMS**4)/mbkin**4 + 
                          (28*mcMS**6)/mbkin**6))/(3*mbkin**14) - 
                        (4*(17*mbkin**4*mcMS**2 + 34*mbkin**2*mcMS**4 - 
                          42*mcMS**6)*(-6 - 16*mbkin + 12*mbkin**2 + 9*mbkin**2*
                          np.log(mu0**2/mcMS**2)))/(9*mbkin**20)) + q_cut**5*
                       ((-20*(3 + 8*mbkin)*(392 + (3466*mcMS**2)/mbkin**2 + 
                          (6078*mcMS**4)/mbkin**4 + (3466*mcMS**6)/mbkin**6 + 
                          (392*mcMS**8)/mbkin**8))/(9*mbkin**12) + 
                        (4*(1733*mbkin**6*mcMS**2 + 6078*mbkin**4*mcMS**4 + 
                          5199*mbkin**2*mcMS**6 + 784*mcMS**8)*(-6 - 16*mbkin + 
                          12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/
                         (9*mbkin**20)) - 6*q_cut**4*((-16*(3 + 8*mbkin)*(210 + 
                          (1891*mcMS**2)/mbkin**2 + (4862*mcMS**4)/mbkin**4 + 
                          (4862*mcMS**6)/mbkin**6 + (1891*mcMS**8)/mbkin**8 + 
                          (210*mcMS**10)/mbkin**10))/(9*mbkin**10) + 
                        (2*(1891*mbkin**8*mcMS**2 + 9724*mbkin**6*mcMS**4 + 
                          14586*mbkin**4*mcMS**6 + 7564*mbkin**2*mcMS**8 + 
                          1050*mcMS**10)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**20)) + 
                      6*q_cut**3*((-4*(3 + 8*mbkin)*(140 + (1259*mcMS**2)/mbkin**
                          2 + (3262*mcMS**4)/mbkin**4 + (4076*mcMS**6)/mbkin**6 + 
                          (3262*mcMS**8)/mbkin**8 + (1259*mcMS**10)/mbkin**10 + 
                          (140*mcMS**12)/mbkin**12))/(3*mbkin**8) + 
                        (2*(1259*mbkin**10*mcMS**2 + 6524*mbkin**8*mcMS**4 + 
                          12228*mbkin**6*mcMS**6 + 13048*mbkin**4*mcMS**8 + 
                          6295*mbkin**2*mcMS**10 + 840*mcMS**12)*(-6 - 
                          16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**
                          2)))/(9*mbkin**20)) + 6*q_cut**2*((-8*(3 + 8*mbkin)*
                          (140 + (839*mcMS**2)/mbkin**2 - (795*mcMS**4)/mbkin**
                          4 - (7114*mcMS**6)/mbkin**6 - (7114*mcMS**8)/mbkin**8 - 
                          (795*mcMS**10)/mbkin**10 + (839*mcMS**12)/mbkin**12 + 
                          (140*mcMS**14)/mbkin**14))/(9*mbkin**6) + 
                        (2*(839*mbkin**12*mcMS**2 - 1590*mbkin**10*mcMS**4 - 
                          21342*mbkin**8*mcMS**6 - 28456*mbkin**6*mcMS**8 - 
                          3975*mbkin**4*mcMS**10 + 5034*mbkin**2*mcMS**12 + 
                          980*mcMS**14)*(-6 - 16*mbkin + 12*mbkin**2 + 
                          9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**20)) - 
                      3*q_cut*((-4*(3 + 8*mbkin)*(421 + (1354*mcMS**2)/mbkin**2 - 
                          (17342*mcMS**4)/mbkin**4 - (84374*mcMS**6)/mbkin**6 - 
                          (132758*mcMS**8)/mbkin**8 - (84374*mcMS**10)/mbkin**
                          10 - (17342*mcMS**12)/mbkin**12 + (1354*mcMS**14)/
                          mbkin**14 + (421*mcMS**16)/mbkin**16))/(9*mbkin**4) + 
                        (4*(677*mbkin**14*mcMS**2 - 17342*mbkin**12*mcMS**4 - 
                          126561*mbkin**10*mcMS**6 - 265516*mbkin**8*mcMS**8 - 
                          210935*mbkin**6*mcMS**10 - 52026*mbkin**4*mcMS**12 + 
                          4739*mbkin**2*mcMS**14 + 1684*mcMS**16)*(-6 - 
                          16*mbkin + 12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**
                          2)))/(9*mbkin**20))))))*np.log((mbkin**2 + mcMS**2 - q_cut - 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                       2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                  (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                        mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                      mbkin**4)))**2)/mbkin**4 - (36*mcMS**4*(mbkin**4 - 
                 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                 q_cut**2)**2*(423*mbkin**14 + 279*mbkin**12*mcMS**2 - 27945*mbkin**10*
                 mcMS**4 - 97497*mbkin**8*mcMS**6 - 97497*mbkin**6*mcMS**8 - 
                27945*mbkin**4*mcMS**10 + 279*mbkin**2*mcMS**12 + 423*mcMS**14 - 
                417*mbkin**12*q_cut - 3492*mbkin**10*mcMS**2*q_cut - 9873*mbkin**8*
                 mcMS**4*q_cut - 14016*mbkin**6*mcMS**6*q_cut - 9873*mbkin**4*mcMS**8*
                 q_cut - 3492*mbkin**2*mcMS**10*q_cut - 417*mcMS**12*q_cut - 
                417*mbkin**10*q_cut**2 - 3897*mbkin**8*mcMS**2*q_cut**2 - 10932*mbkin**6*
                 mcMS**4*q_cut**2 - 10932*mbkin**4*mcMS**6*q_cut**2 - 3897*mbkin**2*
                 mcMS**8*q_cut**2 - 417*mcMS**10*q_cut**2 + 423*mbkin**8*q_cut**3 + 
                3264*mbkin**6*mcMS**2*q_cut**3 + 5892*mbkin**4*mcMS**4*q_cut**3 + 
                3264*mbkin**2*mcMS**6*q_cut**3 + 423*mcMS**8*q_cut**3 + 3*mbkin**6*q_cut**4 - 
                69*mbkin**4*mcMS**2*q_cut**4 - 69*mbkin**2*mcMS**4*q_cut**4 + 
                3*mcMS**6*q_cut**4 - 25*mbkin**4*q_cut**5 + 20*mbkin**2*mcMS**2*q_cut**5 - 
                25*mcMS**4*q_cut**5 - 25*mbkin**2*q_cut**6 - 25*mcMS**2*q_cut**6 + 35*q_cut**7)*(
                (mcMS**4*(16/3 + 4*np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - 
                      q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                          mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                         mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 + 
                mcMS**4*((-16*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                          mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                         q_cut**2)/mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) + 12*mbkin**6*mcMS**2*np.sqrt(0j + 
                       (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                         2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 6*mbkin**2*mcMS**4*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 16*mbkin**3*
                      mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 
                     12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                        mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*mcMS**2*q_cut*
                      np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 12*mbkin**4*
                      mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                     9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                         mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                      np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                      mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                      np.log(mu0**2/mcMS**2))*np.log((mbkin**2 + mcMS**2 - q_cut - 
                       mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                      (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))))/(9*mbkin**4*(mbkin**4 - 
                     2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + 
                     q_cut**2)*(mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + 
                     mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                  (8*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                         np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                       (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)))**2)/(9*mbkin**6))))/mbkin**22) - 
           60480*((mcMS**8*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                   q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*((-128*mbkin)/3 - 
                (4736*mcMS**2)/(9*mbkin) - (14912*mcMS**4)/(3*mbkin**3) - 
                (12032*mcMS**6)/mbkin**5 - (60736*mcMS**8)/(9*mbkin**7) - 
                (1664*mcMS**10)/(3*mbkin**9) + 144*mcMS**4*(4 + 
                  3*np.log(mu0**2/mcMS**2)) + (1440*mcMS**8*(-3 - 8*mbkin + 
                   12*mbkin**2 + 9*mbkin**2*np.log(mu0**2/mcMS**2)))/mbkin**6 + 
                (576*mcMS**10*(-6 - 16*mbkin + 20*mbkin**2 + 15*mbkin**2*
                    np.log(mu0**2/mcMS**2)))/mbkin**8 + (48*mcMS**12*(-12 - 
                   32*mbkin + 36*mbkin**2 + 27*mbkin**2*np.log(mu0**2/mcMS**2)))/
                 mbkin**10 + (192*mcMS**6*(-6 - 16*mbkin + 36*mbkin**2 + 
                   27*mbkin**2*np.log(mu0**2/mcMS**2)))/mbkin**4)*
               np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                        mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                      mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + mbkin**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                        q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8 + 
             (108*mcMS**4 + (864*mcMS**6)/mbkin**2 + (1620*mcMS**8)/mbkin**4 + 
               (864*mcMS**10)/mbkin**6 + (108*mcMS**12)/mbkin**8)*
              ((3*mcMS**8*((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                     q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
                 ((-8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6) + (4*mcMS**2*
                    (-1 + mcMS**2/mbkin**2)*(-6 - 16*mbkin + 12*mbkin**2 + 
                     9*mbkin**2*np.log(mu0**2/mcMS**2)))/(9*mbkin**4) + 
                  (4*q_cut*(6*mbkin**2 + 16*mbkin**3 + 12*mcMS**2 + 32*mbkin*
                      mcMS**2 - 12*mbkin**2*mcMS**2 - 9*mbkin**2*mcMS**2*
                      np.log(mu0**2/mcMS**2)))/(9*mbkin**6))*
                 np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                         2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                     mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                         2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/
                (2*mbkin**8*((-1 + mcMS**2/mbkin**2)**2 - (2*(mbkin**2 + mcMS**2)*
                    q_cut)/mbkin**4 + q_cut**2/mbkin**4)) + ((mbkin**4 - 2*mbkin**2*
                    mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                  mbkin**4)**(3/2)*((mcMS**8*(32/3 + 8*np.log(mu0**2/mcMS**2))*
                   np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                       mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**3)/
                  mbkin**8 + mcMS**8*((-8*(-6*mbkin**4*mcMS**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) - 16*mbkin**5*mcMS**2*np.sqrt(0j + 
                        (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                          2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 12*mbkin**6*mcMS**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 6*mbkin**2*
                       mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) + 
                      16*mbkin**3*mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                          mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                         mbkin**4) - 12*mbkin**4*mcMS**4*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4) - 6*mbkin**2*mcMS**2*q_cut*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 16*mbkin**3*
                       mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4) - 
                      12*mbkin**4*mcMS**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                          mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                         mbkin**4) + 9*mbkin**6*mcMS**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2) - 9*mbkin**4*
                       mcMS**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*
                       np.log(mu0**2/mcMS**2) - 9*mbkin**4*mcMS**2*q_cut*np.sqrt(0j + 
                        (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                          2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*np.log(mu0**2/mcMS**2))*
                     np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                         mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)))**2)/
                    (3*mbkin**8*(mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                      2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)*(mbkin**2 + mcMS**2 - 
                      q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + 
                          mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/
                         mbkin**4))*(-mbkin**2 - mcMS**2 + q_cut + mbkin**2*
                       np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*
                          q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))) - 
                   (16*(3 + 8*mbkin)*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*
                          np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                          2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/
                        (mbkin**2 + mcMS**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 
                          2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 2*mcMS**2*
                          q_cut + q_cut**2)/mbkin**4)))**3)/(9*mbkin**10)))))))/
         (((mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
             2*mcMS**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                 2*mcMS**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mcMS**2 - 7*
                mbkin**2*mcMS**4 + mcMS**6 - mbkin**4*q_cut - mcMS**4*q_cut - mbkin**2*
                q_cut**2 - mcMS**2*q_cut**2 + q_cut**3))/mbkin**6 - 
            (12*mcMS**4*np.log((mbkin**2 + mcMS**2 - q_cut - mbkin**2*np.sqrt(0j + 
                   (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 2*mbkin**2*q_cut - 
                     2*mcMS**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mcMS**2 - q_cut + 
                 mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mcMS**2 + mcMS**4 - 
                     2*mbkin**2*q_cut - 2*mcMS**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
           3)))/1260 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

