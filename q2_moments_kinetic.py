import numpy as np
import warnings 

def q2moment1Kin(q_cut, mbkin, mckin, mus, api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
 res = ( 
    ((18*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
           2*mckin**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mckin**2 - 
           7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
           mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)**2*(3*mbkin**8 - 
          42*mbkin**6*mckin**2 - 282*mbkin**4*mckin**4 - 42*mbkin**2*mckin**6 + 
          3*mckin**8 + 3*mbkin**6*q_cut - 33*mbkin**4*mckin**2*q_cut - 
          33*mbkin**2*mckin**4*q_cut + 3*mckin**6*q_cut - 7*mbkin**4*q_cut**2 + 
          2*mbkin**2*mckin**2*q_cut**2 - 7*mckin**4*q_cut**2 - 7*mbkin**2*q_cut**3 - 
          7*mckin**2*q_cut**3 + 8*q_cut**4))/mbkin**28 - 
       (216*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
           2*mckin**2*q_cut + q_cut**2)**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
            mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*
         (21*mbkin**14 - 321*mbkin**12*mckin**2 + 297*mbkin**10*mckin**4 + 
          6483*mbkin**8*mckin**6 + 6483*mbkin**6*mckin**8 + 
          297*mbkin**4*mckin**10 - 321*mbkin**2*mckin**12 + 21*mckin**14 - 
          30*mbkin**12*q_cut + 156*mbkin**10*mckin**2*q_cut + 1302*mbkin**8*mckin**4*
           q_cut + 1464*mbkin**6*mckin**6*q_cut + 1302*mbkin**4*mckin**8*q_cut + 
          156*mbkin**2*mckin**10*q_cut - 30*mckin**12*q_cut - 41*mbkin**10*q_cut**2 + 
          411*mbkin**8*mckin**2*q_cut**2 + 1394*mbkin**6*mckin**4*q_cut**2 + 
          1394*mbkin**4*mckin**6*q_cut**2 + 411*mbkin**2*mckin**8*q_cut**2 - 
          41*mckin**10*q_cut**2 + 60*mbkin**8*q_cut**3 - 64*mbkin**6*mckin**2*q_cut**3 - 
          568*mbkin**4*mckin**4*q_cut**3 - 64*mbkin**2*mckin**6*q_cut**3 + 
          60*mckin**8*q_cut**3 + 35*mbkin**6*q_cut**4 - 139*mbkin**4*mckin**2*q_cut**4 - 
          139*mbkin**2*mckin**4*q_cut**4 + 35*mckin**6*q_cut**4 - 46*mbkin**4*q_cut**5 - 
          28*mbkin**2*mckin**2*q_cut**5 - 46*mckin**4*q_cut**5 - 15*mbkin**2*q_cut**6 - 
          15*mckin**2*q_cut**6 + 16*q_cut**7)*np.log((mbkin**2 + mckin**2 - q_cut - 
            mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/
           (mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                 mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
               mbkin**4))))/mbkin**22 + 
       (2592*mckin**8*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
           2*mckin**2*q_cut + q_cut**2)**2*(33*mbkin**8 - 222*mbkin**6*mckin**2 - 
          702*mbkin**4*mckin**4 - 222*mbkin**2*mckin**6 + 33*mckin**8 - 
          27*mbkin**6*q_cut - 63*mbkin**4*mckin**2*q_cut - 63*mbkin**2*mckin**4*q_cut - 
          27*mckin**6*q_cut - 37*mbkin**4*q_cut**2 - 58*mbkin**2*mckin**2*q_cut**2 - 
          37*mckin**4*q_cut**2 + 23*mbkin**2*q_cut**3 + 23*mckin**2*q_cut**3 + 8*q_cut**4)*
         np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**20 - 
       (466560*mckin**12*(mbkin**2 + mckin**2)*
         ((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
         np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**10)/
      (180*mbkin**2*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
          2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
        3) + (((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
          mbkin**4 + q_cut**2/mbkin**4)*(-24*mbkin**2*muG*
          ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)**2*(6 - (41*mckin**2)/mbkin**2 + (20*mckin**4)/
            mbkin**4 - (800*mckin**6)/mbkin**6 + (3184*mckin**8)/mbkin**8 - 
           (8374*mckin**10)/mbkin**10 - (7172*mckin**12)/mbkin**12 + 
           (4912*mckin**14)/mbkin**14 - (358*mckin**16)/mbkin**16 - 
           (17*mckin**18)/mbkin**18 + ((-15 + (95*mckin**2)/mbkin**2 + 
              (173*mckin**4)/mbkin**4 - (3465*mckin**6)/mbkin**6 + 
              (395*mckin**8)/mbkin**8 + (2825*mckin**10)/mbkin**10 - 
              (5061*mckin**12)/mbkin**12 + (689*mckin**14)/mbkin**14 + 
              (44*mckin**16)/mbkin**16)*q_cut)/mbkin**2 + 
           ((-6 + (31*mckin**2)/mbkin**2 + (180*mckin**4)/mbkin**4 - 
              (3195*mckin**6)/mbkin**6 - (10526*mckin**8)/mbkin**8 - 
              (3891*mckin**10)/mbkin**10 + (552*mckin**12)/mbkin**12 + 
              (7*mckin**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
           ((45 - (263*mckin**2)/mbkin**2 - (1114*mckin**4)/mbkin**4 + 
              (3670*mckin**6)/mbkin**6 + (5017*mckin**8)/mbkin**8 - 
              (1723*mckin**10)/mbkin**10 - (104*mckin**12)/mbkin**12)*q_cut**3)/
            mbkin**6 + ((-30 + (221*mckin**2)/mbkin**2 + (80*mckin**4)/mbkin**4 - 
              (2094*mckin**6)/mbkin**6 + (302*mckin**8)/mbkin**8 + 
              (65*mckin**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
           ((-21 + (37*mckin**2)/mbkin**2 + (1245*mckin**4)/mbkin**4 + 
              (1095*mckin**6)/mbkin**6 + (52*mckin**8)/mbkin**8)*q_cut**5)/mbkin**10 - 
           ((-30 + (119*mckin**2)/mbkin**2 + (592*mckin**4)/mbkin**4 + 
              (59*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
           ((-9 + (35*mckin**2)/mbkin**2 + (8*mckin**4)/mbkin**4)*q_cut**7)/
            mbkin**14 + (4*mckin**2*q_cut**8)/mbkin**18) - 12*muG*mupi*
          ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)**2*(6 - (41*mckin**2)/mbkin**2 + (20*mckin**4)/
            mbkin**4 - (800*mckin**6)/mbkin**6 + (3184*mckin**8)/mbkin**8 - 
           (8374*mckin**10)/mbkin**10 - (7172*mckin**12)/mbkin**12 + 
           (4912*mckin**14)/mbkin**14 - (358*mckin**16)/mbkin**16 - 
           (17*mckin**18)/mbkin**18 + ((-15 + (95*mckin**2)/mbkin**2 + 
              (173*mckin**4)/mbkin**4 - (3465*mckin**6)/mbkin**6 + 
              (395*mckin**8)/mbkin**8 + (2825*mckin**10)/mbkin**10 - 
              (5061*mckin**12)/mbkin**12 + (689*mckin**14)/mbkin**14 + 
              (44*mckin**16)/mbkin**16)*q_cut)/mbkin**2 + 
           ((-6 + (31*mckin**2)/mbkin**2 + (180*mckin**4)/mbkin**4 - 
              (3195*mckin**6)/mbkin**6 - (10526*mckin**8)/mbkin**8 - 
              (3891*mckin**10)/mbkin**10 + (552*mckin**12)/mbkin**12 + 
              (7*mckin**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
           ((45 - (263*mckin**2)/mbkin**2 - (1114*mckin**4)/mbkin**4 + 
              (3670*mckin**6)/mbkin**6 + (5017*mckin**8)/mbkin**8 - 
              (1723*mckin**10)/mbkin**10 - (104*mckin**12)/mbkin**12)*q_cut**3)/
            mbkin**6 + ((-30 + (221*mckin**2)/mbkin**2 + (80*mckin**4)/mbkin**4 - 
              (2094*mckin**6)/mbkin**6 + (302*mckin**8)/mbkin**8 + 
              (65*mckin**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
           ((-21 + (37*mckin**2)/mbkin**2 + (1245*mckin**4)/mbkin**4 + 
              (1095*mckin**6)/mbkin**6 + (52*mckin**8)/mbkin**8)*q_cut**5)/mbkin**10 - 
           ((-30 + (119*mckin**2)/mbkin**2 + (592*mckin**4)/mbkin**4 + 
              (59*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
           ((-9 + (35*mckin**2)/mbkin**2 + (8*mckin**4)/mbkin**4)*q_cut**7)/
            mbkin**14 + (4*mckin**2*q_cut**8)/mbkin**18) + 
         12*muG**2*((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
             mbkin**4 + q_cut**2/mbkin**4)**2*(-18 + (27*mckin**2)/mbkin**2 - 
           (316*mckin**4)/mbkin**4 + (1304*mckin**6)/mbkin**6 - 
           (7248*mckin**8)/mbkin**8 + (15050*mckin**10)/mbkin**10 - 
           (31540*mckin**12)/mbkin**12 + (16248*mckin**14)/mbkin**14 - 
           (2062*mckin**16)/mbkin**16 - (85*mckin**18)/mbkin**18 + 
           ((-3 - (101*mckin**2)/mbkin**2 - (1279*mckin**4)/mbkin**4 + 
              (5419*mckin**6)/mbkin**6 - (22225*mckin**8)/mbkin**8 + 
              (31533*mckin**10)/mbkin**10 - (21625*mckin**12)/mbkin**12 + 
              (3741*mckin**14)/mbkin**14 + (220*mckin**16)/mbkin**16)*q_cut)/
            mbkin**2 + ((42 - (341*mckin**2)/mbkin**2 - (3860*mckin**4)/mbkin**4 + 
              (8857*mckin**6)/mbkin**6 - (24270*mckin**8)/mbkin**8 - 
              (6319*mckin**10)/mbkin**10 + (3248*mckin**12)/mbkin**12 + 
              (35*mckin**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
           ((57 + (133*mckin**2)/mbkin**2 - (4242*mckin**4)/mbkin**4 + 
              (894*mckin**6)/mbkin**6 + (19333*mckin**8)/mbkin**8 - 
              (9167*mckin**10)/mbkin**10 - (520*mckin**12)/mbkin**12)*q_cut**3)/
            mbkin**6 + ((-78 + (601*mckin**2)/mbkin**2 + (1040*mckin**4)/mbkin**
                4 - (13694*mckin**6)/mbkin**6 + (1358*mckin**8)/mbkin**8 + 
              (325*mckin**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
           ((-105 + (81*mckin**2)/mbkin**2 + (6697*mckin**4)/mbkin**4 + 
              (5635*mckin**6)/mbkin**6 + (260*mckin**8)/mbkin**8)*q_cut**5)/
            mbkin**10 - ((-150 + (595*mckin**2)/mbkin**2 + (2928*mckin**4)/mbkin**
                4 + (295*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
           (5*(-9 + (35*mckin**2)/mbkin**2 + (8*mckin**4)/mbkin**4)*q_cut**7)/
            mbkin**14 + (20*mckin**2*q_cut**8)/mbkin**18) - 
         8*mbkin*(-((-1 + mckin**2/mbkin**2)**4*(1 + mckin**2/mbkin**2)**2*
             (351 - (5815*mckin**2)/mbkin**2 + (36915*mckin**4)/mbkin**4 - 
              (109923*mckin**6)/mbkin**6 + (69257*mckin**8)/mbkin**8 - 
              (22209*mckin**10)/mbkin**10 + (1077*mckin**12)/mbkin**12 + 
              (107*mckin**14)/mbkin**14)) + (2*(-1 + mckin**2/mbkin**2)**2*
             (1089 - (13483*mckin**2)/mbkin**2 + (53852*mckin**4)/mbkin**4 - 
              (45772*mckin**6)/mbkin**6 - (288518*mckin**8)/mbkin**8 - 
              (239134*mckin**10)/mbkin**10 + (43532*mckin**12)/mbkin**12 + 
              (52492*mckin**14)/mbkin**14 - (52219*mckin**16)/mbkin**16 + 
              (3977*mckin**18)/mbkin**18 + (344*mckin**20)/mbkin**20)*q_cut)/
            mbkin**2 + ((-4929 + (52403*mckin**2)/mbkin**2 - (176271*mckin**4)/
               mbkin**4 + (115749*mckin**6)/mbkin**6 + (605090*mckin**8)/mbkin**
                8 + (898826*mckin**10)/mbkin**10 + (657918*mckin**12)/mbkin**12 - 
              (174202*mckin**14)/mbkin**14 - (222817*mckin**16)/mbkin**16 + 
              (202947*mckin**18)/mbkin**18 - (17775*mckin**20)/mbkin**20 - 
              (1579*mckin**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
           (2*(1659 - (10934*mckin**2)/mbkin**2 + (14621*mckin**4)/mbkin**4 + 
              (22408*mckin**6)/mbkin**6 - (61810*mckin**8)/mbkin**8 + 
              (4308*mckin**10)/mbkin**10 + (92322*mckin**12)/mbkin**12 - 
              (26296*mckin**14)/mbkin**14 - (44537*mckin**16)/mbkin**16 + 
              (7826*mckin**18)/mbkin**18 + (433*mckin**20)/mbkin**20)*q_cut**3)/
            mbkin**6 + (4*(1353 - (9240*mckin**2)/mbkin**2 + (13738*mckin**4)/
               mbkin**4 + (9254*mckin**6)/mbkin**6 - (21795*mckin**8)/mbkin**8 - 
              (9861*mckin**10)/mbkin**10 - (9696*mckin**12)/mbkin**12 - 
              (33052*mckin**14)/mbkin**14 + (4192*mckin**16)/mbkin**16 + 
              (675*mckin**18)/mbkin**18)*q_cut**4)/mbkin**8 - 
           (2*(6003 - (25356*mckin**2)/mbkin**2 + (10882*mckin**4)/mbkin**4 + 
              (16908*mckin**6)/mbkin**6 - (30624*mckin**8)/mbkin**8 - 
              (109320*mckin**10)/mbkin**10 - (93822*mckin**12)/mbkin**12 + 
              (22600*mckin**14)/mbkin**14 + (2793*mckin**16)/mbkin**16)*q_cut**5)/
            mbkin**10 + (2*(3483 - (12003*mckin**2)/mbkin**2 - (3121*mckin**4)/
               mbkin**4 - (19139*mckin**6)/mbkin**6 - (74989*mckin**8)/mbkin**8 - 
              (52799*mckin**10)/mbkin**10 + (11627*mckin**12)/mbkin**12 + 
              (1677*mckin**14)/mbkin**14)*q_cut**6)/mbkin**12 + 
           (4*(765 + (1989*mckin**2)/mbkin**2 + (3559*mckin**4)/mbkin**4 + 
              (13202*mckin**6)/mbkin**6 + (14387*mckin**8)/mbkin**8 + 
              (5569*mckin**10)/mbkin**10 + (417*mckin**12)/mbkin**12)*q_cut**7)/
            mbkin**14 - ((6399 + (6327*mckin**2)/mbkin**2 + (19022*mckin**4)/
               mbkin**4 + (37806*mckin**6)/mbkin**6 + (34907*mckin**8)/mbkin**8 + 
              (3747*mckin**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
           (2*(1719 + (1825*mckin**2)/mbkin**2 + (6279*mckin**4)/mbkin**4 + 
              (8751*mckin**6)/mbkin**6 + (1150*mckin**8)/mbkin**8)*q_cut**9)/
            mbkin**18 - ((693 + (829*mckin**2)/mbkin**2 + (3367*mckin**4)/mbkin**
                4 + (607*mckin**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
           ((6 + (88*mckin**2)/mbkin**2 + (58*mckin**4)/mbkin**4)*q_cut**11)/
            mbkin**22 - (2*(3 + (7*mckin**2)/mbkin**2)*q_cut**12)/mbkin**24 + 
           (6*q_cut**13)/mbkin**26)*rhoD + ((mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)*
           (-16*(-((-1 + mckin**2/mbkin**2)**4*(-317 + (2200*mckin**2)/mbkin**2 + 
                 (1589*mckin**4)/mbkin**4 - (2096*mckin**6)/mbkin**6 - 
                 (2611*mckin**8)/mbkin**8 + (2536*mckin**10)/mbkin**10 + 
                 (139*mckin**12)/mbkin**12)) + ((-1 + mckin**2/mbkin**2)**2*
                (-1535 + (7527*mckin**2)/mbkin**2 + (11043*mckin**4)/mbkin**4 + 
                 (3449*mckin**6)/mbkin**6 - (13681*mckin**8)/mbkin**8 - 
                 (8751*mckin**10)/mbkin**10 + (12717*mckin**12)/mbkin**12 + 
                 (751*mckin**14)/mbkin**14)*q_cut)/mbkin**2 - 
              ((-2610 + (10073*mckin**2)/mbkin**2 + (8271*mckin**4)/mbkin**4 + 
                 (9663*mckin**6)/mbkin**6 + (16889*mckin**8)/mbkin**8 - 
                 (24081*mckin**10)/mbkin**10 - (19599*mckin**12)/mbkin**12 + 
                 (23033*mckin**14)/mbkin**14 + (1401*mckin**16)/mbkin**16)*q_cut**2)/
               mbkin**4 + ((-1137 + (2623*mckin**2)/mbkin**2 + (347*mckin**4)/
                  mbkin**4 - (3231*mckin**6)/mbkin**6 - (18327*mckin**8)/
                  mbkin**8 + (13433*mckin**10)/mbkin**10 + (16165*mckin**12)/
                  mbkin**12 + (447*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
              ((-1962 - (917*mckin**2)/mbkin**2 + (8227*mckin**4)/mbkin**4 + 
                 (11800*mckin**6)/mbkin**6 - (1490*mckin**8)/mbkin**8 + 
                 (10753*mckin**10)/mbkin**10 + (2205*mckin**12)/mbkin**12)*q_cut**4)/
               mbkin**8 - ((-3015 - (1561*mckin**2)/mbkin**2 + (11820*mckin**4)/
                  mbkin**4 + (20886*mckin**6)/mbkin**6 + (25655*mckin**8)/
                  mbkin**8 + (3759*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
              ((-1710 + (271*mckin**2)/mbkin**2 + (11609*mckin**4)/mbkin**4 + 
                 (16225*mckin**6)/mbkin**6 + (2709*mckin**8)/mbkin**8)*q_cut**6)/
               mbkin**12 - ((-465 + (689*mckin**2)/mbkin**2 + (3797*mckin**4)/
                  mbkin**4 + (939*mckin**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
              ((-63 + (115*mckin**2)/mbkin**2 + (138*mckin**4)/mbkin**4)*q_cut**8)/
               mbkin**16 - (4*(2 + (5*mckin**2)/mbkin**2)*q_cut**9)/mbkin**18 + 
              (8*q_cut**10)/mbkin**20)*rE - ((-1 + mckin**2/mbkin**2)**2 - 
              (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
             (-4*(-((-1 + mckin**2/mbkin**2)**2*(-731 + (4718*mckin**2)/mbkin**2 - 
                   (4699*mckin**4)/mbkin**4 - (3384*mckin**6)/mbkin**6 - 
                   (2189*mckin**8)/mbkin**8 + (1882*mckin**10)/mbkin**10 + 
                   (83*mckin**12)/mbkin**12)) + ((-2179 + (9001*mckin**2)/
                    mbkin**2 - (4197*mckin**4)/mbkin**4 - (9313*mckin**6)/
                    mbkin**6 - (10973*mckin**8)/mbkin**8 - (5289*mckin**10)/
                    mbkin**10 + (5381*mckin**12)/mbkin**12 + (289*mckin**14)/
                    mbkin**14)*q_cut)/mbkin**2 - (2*(-694 + (1014*mckin**2)/
                    mbkin**2 - (1881*mckin**4)/mbkin**4 - (304*mckin**6)/
                    mbkin**6 + (612*mckin**8)/mbkin**8 + (1866*mckin**10)/
                    mbkin**10 + (107*mckin**12)/mbkin**12)*q_cut**2)/mbkin**4 - 
                (2*(-856 - (348*mckin**2)/mbkin**2 + (691*mckin**4)/mbkin**4 + 
                   (2447*mckin**6)/mbkin**6 + (2013*mckin**8)/mbkin**8 + 
                   (181*mckin**10)/mbkin**10)*q_cut**3)/mbkin**6 + 
                ((-2641 - (1196*mckin**2)/mbkin**2 + (6054*mckin**4)/mbkin**4 + 
                   (6404*mckin**6)/mbkin**6 + (715*mckin**8)/mbkin**8)*q_cut**4)/
                 mbkin**8 - ((-1103 + (405*mckin**2)/mbkin**2 + (2415*mckin**4)/
                    mbkin**4 + (427*mckin**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
                (4*(-28 + (27*mckin**2)/mbkin**2 + (19*mckin**4)/mbkin**4)*q_cut**6)/
                 mbkin**12 + (4*(-1 + mckin**2/mbkin**2)*q_cut**7)/mbkin**14 + 
                (2*q_cut**8)/mbkin**16)*rG - 4*(-((-1 + mckin**2/mbkin**2)**2*
                  (351 - (1796*mckin**2)/mbkin**2 - (5235*mckin**4)/mbkin**4 - 
                   (5200*mckin**6)/mbkin**6 - (7135*mckin**8)/mbkin**8 + 
                   (4404*mckin**10)/mbkin**10 + (211*mckin**12)/mbkin**12)) + 
                ((909 - (1687*mckin**2)/mbkin**2 - (18237*mckin**4)/mbkin**4 - 
                   (23505*mckin**6)/mbkin**6 - (8845*mckin**8)/mbkin**8 - 
                   (19449*mckin**10)/mbkin**10 + (12477*mckin**12)/mbkin**12 + 
                   (737*mckin**14)/mbkin**14)*q_cut)/mbkin**2 - 
                (6*(57 + (175*mckin**2)/mbkin**2 - (475*mckin**4)/mbkin**4 + 
                   (270*mckin**6)/mbkin**6 - (725*mckin**8)/mbkin**8 + 
                   (1403*mckin**10)/mbkin**10 + (95*mckin**12)/mbkin**12)*q_cut**2)/
                 mbkin**4 - (2*(390 + (1352*mckin**2)/mbkin**2 + (3467*mckin**4)/
                    mbkin**4 + (6483*mckin**6)/mbkin**6 + (5015*mckin**8)/
                    mbkin**8 + (413*mckin**10)/mbkin**10)*q_cut**3)/mbkin**6 + 
                ((561 + (4034*mckin**2)/mbkin**2 + (13860*mckin**4)/mbkin**4 + 
                   (15570*mckin**6)/mbkin**6 + (1655*mckin**8)/mbkin**8)*q_cut**4)/
                 mbkin**8 - (3*(-49 + (463*mckin**2)/mbkin**2 + (1961*mckin**4)/
                    mbkin**4 + (313*mckin**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
                (2*(-69 + (139*mckin**2)/mbkin**2 + (64*mckin**4)/mbkin**4)*q_cut**6)/
                 mbkin**12 + (4*(-3 + (5*mckin**2)/mbkin**2)*q_cut**7)/mbkin**14 + 
                (6*q_cut**8)/mbkin**16)*sB - 1968*sE + (14432*mckin**2*sE)/mbkin**
                2 - (26560*mckin**4*sE)/mbkin**4 + (35616*mckin**6*sE)/mbkin**6 - 
              (28000*mckin**8*sE)/mbkin**8 - (15712*mckin**10*sE)/mbkin**10 + 
              (32448*mckin**12*sE)/mbkin**12 - (9760*mckin**14*sE)/mbkin**14 - 
              (496*mckin**16*sE)/mbkin**16 + (5760*q_cut*sE)/mbkin**2 - 
              (18112*mckin**2*q_cut*sE)/mbkin**4 - (21504*mckin**4*q_cut*sE)/mbkin**6 - 
              (21120*mckin**6*q_cut*sE)/mbkin**8 - (44800*mckin**8*q_cut*sE)/mbkin**
                10 - (25920*mckin**10*q_cut*sE)/mbkin**12 + (31872*mckin**12*q_cut*
                sE)/mbkin**14 + (1664*mckin**14*q_cut*sE)/mbkin**16 - 
              (3424*q_cut**2*sE)/mbkin**4 + (2160*mckin**2*q_cut**2*sE)/mbkin**6 + 
              (1488*mckin**4*q_cut**2*sE)/mbkin**8 + (12992*mckin**6*q_cut**2*sE)/mbkin**
                10 - (28704*mckin**8*q_cut**2*sE)/mbkin**12 - (25776*mckin**10*q_cut**2*
                sE)/mbkin**14 - (976*mckin**12*q_cut**2*sE)/mbkin**16 - 
              (4656*q_cut**3*sE)/mbkin**6 - (7456*mckin**2*q_cut**3*sE)/mbkin**8 - 
              (9952*mckin**4*q_cut**3*sE)/mbkin**10 + (6432*mckin**6*q_cut**3*sE)/mbkin**
                12 - (16816*mckin**8*q_cut**3*sE)/mbkin**14 - (2752*mckin**10*q_cut**3*
                sE)/mbkin**16 + (6336*q_cut**4*sE)/mbkin**8 + (11504*mckin**2*q_cut**4*
                sE)/mbkin**10 + (18864*mckin**4*q_cut**4*sE)/mbkin**12 + 
              (32784*mckin**6*q_cut**4*sE)/mbkin**14 + (4880*mckin**8*q_cut**4*sE)/mbkin**
                16 - (1936*q_cut**5*sE)/mbkin**10 - (2976*mckin**2*q_cut**5*sE)/mbkin**
                12 - (12816*mckin**4*q_cut**5*sE)/mbkin**14 - (2944*mckin**6*q_cut**5*
                sE)/mbkin**16 - (144*q_cut**6*sE)/mbkin**12 + (512*mckin**2*q_cut**6*
                sE)/mbkin**14 + (656*mckin**4*q_cut**6*sE)/mbkin**16 - 
              (64*mckin**2*q_cut**7*sE)/mbkin**16 + (32*q_cut**8*sE)/mbkin**16 - 
              57*sqB + (338*mckin**2*sqB)/mbkin**2 + (5756*mckin**4*sqB)/mbkin**
                4 - (10002*mckin**6*sqB)/mbkin**6 + (3590*mckin**8*sqB)/mbkin**
                8 - (2098*mckin**10*sqB)/mbkin**10 + (3012*mckin**12*sqB)/mbkin**
                12 - (526*mckin**14*sqB)/mbkin**14 - (13*mckin**16*sqB)/mbkin**
                16 + (75*q_cut*sqB)/mbkin**2 + (713*mckin**2*q_cut*sqB)/mbkin**4 - 
              (13683*mckin**4*q_cut*sqB)/mbkin**6 - (18057*mckin**6*q_cut*sqB)/mbkin**
                8 - (7867*mckin**8*q_cut*sqB)/mbkin**10 - (3177*mckin**10*q_cut*sqB)/
               mbkin**12 + (1635*mckin**12*q_cut*sqB)/mbkin**14 + (41*mckin**14*q_cut*
                sqB)/mbkin**16 + (116*q_cut**2*sqB)/mbkin**4 - (1062*mckin**2*q_cut**2*
                sqB)/mbkin**6 + (2688*mckin**4*q_cut**2*sqB)/mbkin**8 - 
              (2668*mckin**6*q_cut**2*sqB)/mbkin**10 - (1236*mckin**8*q_cut**2*sqB)/
               mbkin**12 - (1182*mckin**10*q_cut**2*sqB)/mbkin**14 - 
              (16*mckin**12*q_cut**2*sqB)/mbkin**16 - (138*q_cut**3*sqB)/mbkin**6 - 
              (868*mckin**2*q_cut**3*sqB)/mbkin**8 + (1346*mckin**4*q_cut**3*sqB)/mbkin**
                10 - (18*mckin**6*q_cut**3*sqB)/mbkin**12 - (1024*mckin**8*q_cut**3*sqB)/
               mbkin**14 - (82*mckin**10*q_cut**3*sqB)/mbkin**16 - (171*q_cut**4*sqB)/
               mbkin**8 + (1028*mckin**2*q_cut**4*sqB)/mbkin**10 + (1686*mckin**4*
                q_cut**4*sqB)/mbkin**12 + (1716*mckin**6*q_cut**4*sqB)/mbkin**14 + 
              (125*mckin**8*q_cut**4*sqB)/mbkin**16 + (251*q_cut**5*sqB)/mbkin**10 - 
              (201*mckin**2*q_cut**5*sqB)/mbkin**12 - (663*mckin**4*q_cut**5*sqB)/mbkin**
                14 - (67*mckin**6*q_cut**5*sqB)/mbkin**16 - (66*q_cut**6*sqB)/mbkin**
                12 + (56*mckin**2*q_cut**6*sqB)/mbkin**14 + (14*mckin**4*q_cut**6*sqB)/
               mbkin**16 - (12*q_cut**7*sqB)/mbkin**14 - (4*mckin**2*q_cut**7*sqB)/mbkin**
                16 + (2*q_cut**8*sqB)/mbkin**16)))/mbkin**6) - 
       6*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
           2*mckin**2*q_cut + q_cut**2)/mbkin**4)*
        (32*(-((-1 + mckin**2/mbkin**2)**4*(-11 + (131*mckin**2)/mbkin**2 + 
              (343*mckin**4)/mbkin**4 - (4901*mckin**6)/mbkin**6 - 
              (3931*mckin**8)/mbkin**8 + (4513*mckin**10)/mbkin**10 + 
              (4201*mckin**12)/mbkin**12 - (3931*mckin**14)/mbkin**14 - 
              (842*mckin**16)/mbkin**16 + (108*mckin**18)/mbkin**18)) + 
           ((-1 + mckin**2/mbkin**2)**2*(-69 + (686*mckin**2)/mbkin**2 + 
              (1998*mckin**4)/mbkin**4 - (17384*mckin**6)/mbkin**6 - 
              (28414*mckin**8)/mbkin**8 - (7896*mckin**10)/mbkin**10 + 
              (25876*mckin**12)/mbkin**12 + (14624*mckin**14)/mbkin**14 - 
              (19869*mckin**16)/mbkin**16 - (4814*mckin**18)/mbkin**18 + 
              (702*mckin**20)/mbkin**20)*q_cut)/mbkin**2 - 
           (2*(-80 + (683*mckin**2)/mbkin**2 + (1558*mckin**4)/mbkin**4 - 
              (10930*mckin**6)/mbkin**6 - (12701*mckin**8)/mbkin**8 - 
              (15107*mckin**10)/mbkin**10 - (14557*mckin**12)/mbkin**12 + 
              (20503*mckin**14)/mbkin**14 + (17401*mckin**16)/mbkin**16 - 
              (16868*mckin**18)/mbkin**18 - (5317*mckin**20)/mbkin**20 + 
              (855*mckin**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
           (2*(-60 + (337*mckin**2)/mbkin**2 + (688*mckin**4)/mbkin**4 - 
              (2600*mckin**6)/mbkin**6 - (1620*mckin**8)/mbkin**8 + 
              (3788*mckin**10)/mbkin**10 + (14332*mckin**12)/mbkin**12 - 
              (9532*mckin**14)/mbkin**14 - (15876*mckin**16)/mbkin**16 - 
              (2417*mckin**18)/mbkin**18 + (720*mckin**20)/mbkin**20)*q_cut**3)/
            mbkin**6 + (2*(-75 + (292*mckin**2)/mbkin**2 + (3006*mckin**4)/mbkin**
                4 + (1155*mckin**6)/mbkin**6 - (7341*mckin**8)/mbkin**8 - 
              (10120*mckin**10)/mbkin**10 - (1458*mckin**12)/mbkin**12 - 
              (8299*mckin**14)/mbkin**14 - (3536*mckin**16)/mbkin**16 + 
              (720*mckin**18)/mbkin**18)*q_cut**4)/mbkin**8 + 
           ((378 - (886*mckin**2)/mbkin**2 - (8966*mckin**4)/mbkin**4 - 
              (4238*mckin**6)/mbkin**6 + (22414*mckin**8)/mbkin**8 + 
              (43870*mckin**10)/mbkin**10 + (48834*mckin**12)/mbkin**12 + 
              (10862*mckin**14)/mbkin**14 - (4284*mckin**16)/mbkin**16)*q_cut**5)/
            mbkin**10 + (2*(-126 + (313*mckin**2)/mbkin**2 + (2593*mckin**4)/
               mbkin**4 - (837*mckin**6)/mbkin**6 - (12889*mckin**8)/mbkin**8 - 
              (16866*mckin**10)/mbkin**10 - (3746*mckin**12)/mbkin**12 + 
              (1638*mckin**14)/mbkin**14)*q_cut**6)/mbkin**12 + 
           (2*(-30 - (317*mckin**2)/mbkin**2 - (672*mckin**4)/mbkin**4 + 
              (2136*mckin**6)/mbkin**6 + (6062*mckin**8)/mbkin**8 + 
              (2953*mckin**10)/mbkin**10 + (180*mckin**12)/mbkin**12)*q_cut**7)/
            mbkin**14 - ((-195 - (551*mckin**2)/mbkin**2 + (491*mckin**4)/mbkin**
                4 + (3933*mckin**6)/mbkin**6 + (5026*mckin**8)/mbkin**8 + 
              (2340*mckin**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
           ((-125 - (234*mckin**2)/mbkin**2 + (913*mckin**4)/mbkin**4 + 
              (2412*mckin**6)/mbkin**6 + (1710*mckin**8)/mbkin**8)*q_cut**9)/
            mbkin**18 + ((36 + (36*mckin**2)/mbkin**2 - (446*mckin**4)/mbkin**4 - 
              (558*mckin**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
           ((-4 + (72*mckin**4)/mbkin**4)*q_cut**11)/mbkin**22)*rE + 
         ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
           q_cut**2/mbkin**4)*((180*mckin**2*muG**2)/mbkin**2 - (1452*mckin**4*muG**2)/
            mbkin**4 + (4512*mckin**6*muG**2)/mbkin**6 + (1584*mckin**8*muG**2)/
            mbkin**8 - (38424*mckin**10*muG**2)/mbkin**10 + 
           (160680*mckin**12*muG**2)/mbkin**12 - (250128*mckin**14*muG**2)/
            mbkin**14 + (135888*mckin**16*muG**2)/mbkin**16 - 
           (924*mckin**18*muG**2)/mbkin**18 - (14076*mckin**20*muG**2)/mbkin**20 + 
           (2160*mckin**22*muG**2)/mbkin**22 + (60*mckin**2*muG*mupi)/mbkin**2 - 
           (996*mckin**4*muG*mupi)/mbkin**4 + (3360*mckin**6*muG*mupi)/mbkin**6 + 
           (1488*mckin**8*muG*mupi)/mbkin**8 - (25224*mckin**10*muG*mupi)/
            mbkin**10 - (24648*mckin**12*muG*mupi)/mbkin**12 + 
           (124176*mckin**14*muG*mupi)/mbkin**14 - (90960*mckin**16*muG*mupi)/
            mbkin**16 + (8652*mckin**18*muG*mupi)/mbkin**18 + 
           (4524*mckin**20*muG*mupi)/mbkin**20 - (432*mckin**22*muG*mupi)/
            mbkin**22 - (240*mckin**2*muG**2*q_cut)/mbkin**4 + 
           (1032*mckin**4*muG**2*q_cut)/mbkin**6 - (1344*mckin**6*muG**2*q_cut)/
            mbkin**8 + (16128*mckin**8*muG**2*q_cut)/mbkin**10 - 
           (66624*mckin**10*muG**2*q_cut)/mbkin**12 - (32784*mckin**12*muG**2*q_cut)/
            mbkin**14 - (183936*mckin**14*muG**2*q_cut)/mbkin**16 + 
           (23040*mckin**16*muG**2*q_cut)/mbkin**18 + (47088*mckin**18*muG**2*q_cut)/
            mbkin**20 - (9720*mckin**20*muG**2*q_cut)/mbkin**22 - 
           (240*mckin**2*muG*mupi*q_cut)/mbkin**4 + (3288*mckin**4*muG*mupi*q_cut)/
            mbkin**6 - (7680*mckin**6*muG*mupi*q_cut)/mbkin**8 - 
           (24000*mckin**8*muG*mupi*q_cut)/mbkin**10 + (73920*mckin**10*muG*mupi*
             q_cut)/mbkin**12 + (84624*mckin**12*muG*mupi*q_cut)/mbkin**14 + 
           (129216*mckin**14*muG*mupi*q_cut)/mbkin**16 - 
           (38976*mckin**16*muG*mupi*q_cut)/mbkin**18 - 
           (14736*mckin**18*muG*mupi*q_cut)/mbkin**20 + 
           (1944*mckin**20*muG*mupi*q_cut)/mbkin**22 - (240*mckin**2*muG**2*q_cut**2)/
            mbkin**6 + (3000*mckin**4*muG**2*q_cut**2)/mbkin**8 - 
           (5208*mckin**6*muG**2*q_cut**2)/mbkin**10 - (20472*mckin**8*muG**2*q_cut**2)/
            mbkin**12 - (9768*mckin**10*muG**2*q_cut**2)/mbkin**14 + 
           (34152*mckin**12*muG**2*q_cut**2)/mbkin**16 + (22008*mckin**14*muG**2*q_cut**2)/
            mbkin**18 - (53352*mckin**16*muG**2*q_cut**2)/mbkin**20 + 
           (12600*mckin**18*muG**2*q_cut**2)/mbkin**22 + (240*mckin**2*muG*mupi*q_cut**2)/
            mbkin**6 - (2520*mckin**4*muG*mupi*q_cut**2)/mbkin**8 + 
           (3384*mckin**6*muG*mupi*q_cut**2)/mbkin**10 + 
           (16152*mckin**8*muG*mupi*q_cut**2)/mbkin**12 + 
           (15048*mckin**10*muG*mupi*q_cut**2)/mbkin**14 - 
           (46152*mckin**12*muG*mupi*q_cut**2)/mbkin**16 + 
           (19752*mckin**14*muG*mupi*q_cut**2)/mbkin**18 + 
           (13896*mckin**16*muG*mupi*q_cut**2)/mbkin**20 - 
           (2520*mckin**18*muG*mupi*q_cut**2)/mbkin**22 - (240*mckin**2*muG**2*q_cut**3)/
            mbkin**8 - (1080*mckin**4*muG**2*q_cut**3)/mbkin**10 + 
           (4224*mckin**6*muG**2*q_cut**3)/mbkin**12 - (96840*mckin**8*muG**2*q_cut**3)/
            mbkin**14 - (119760*mckin**10*muG**2*q_cut**3)/mbkin**16 - 
           (151272*mckin**12*muG**2*q_cut**3)/mbkin**18 + 
           (20160*mckin**14*muG**2*q_cut**3)/mbkin**20 + (6120*mckin**16*muG**2*q_cut**3)/
            mbkin**22 + (240*mckin**2*muG*mupi*q_cut**3)/mbkin**8 - 
           (3624*mckin**4*muG*mupi*q_cut**3)/mbkin**10 + 
           (7104*mckin**6*muG*mupi*q_cut**3)/mbkin**12 + 
           (66216*mckin**8*muG*mupi*q_cut**3)/mbkin**14 + 
           (105168*mckin**10*muG*mupi*q_cut**3)/mbkin**16 + 
           (71112*mckin**12*muG*mupi*q_cut**3)/mbkin**18 + 
           (1536*mckin**14*muG*mupi*q_cut**3)/mbkin**20 - 
           (1224*mckin**16*muG*mupi*q_cut**3)/mbkin**22 + (1560*mckin**2*muG**2*q_cut**4)/
            mbkin**10 - (8400*mckin**4*muG**2*q_cut**4)/mbkin**12 - 
           (7416*mckin**6*muG**2*q_cut**4)/mbkin**14 + (114336*mckin**8*muG**2*q_cut**4)/
            mbkin**16 + (167928*mckin**10*muG**2*q_cut**4)/mbkin**18 + 
           (12960*mckin**12*muG**2*q_cut**4)/mbkin**20 - (29160*mckin**14*muG**2*q_cut**4)/
            mbkin**22 - (600*mckin**2*muG*mupi*q_cut**4)/mbkin**10 + 
           (8112*mckin**4*muG*mupi*q_cut**4)/mbkin**12 - 
           (12072*mckin**6*muG*mupi*q_cut**4)/mbkin**14 - 
           (95712*mckin**8*muG*mupi*q_cut**4)/mbkin**16 - 
           (93048*mckin**10*muG*mupi*q_cut**4)/mbkin**18 - 
           (10560*mckin**12*muG*mupi*q_cut**4)/mbkin**20 + 
           (5832*mckin**14*muG*mupi*q_cut**4)/mbkin**22 - (720*mckin**2*muG**2*q_cut**5)/
            mbkin**12 + (7032*mckin**4*muG**2*q_cut**5)/mbkin**14 - 
           (15648*mckin**6*muG**2*q_cut**5)/mbkin**16 - (98928*mckin**8*muG**2*q_cut**5)/
            mbkin**18 - (46224*mckin**10*muG**2*q_cut**5)/mbkin**20 + 
           (21240*mckin**12*muG**2*q_cut**5)/mbkin**22 + (240*mckin**2*muG*mupi*q_cut**5)/
            mbkin**12 - (4824*mckin**4*muG*mupi*q_cut**5)/mbkin**14 + 
           (11808*mckin**6*muG*mupi*q_cut**5)/mbkin**16 + 
           (40560*mckin**8*muG*mupi*q_cut**5)/mbkin**18 + 
           (11376*mckin**10*muG*mupi*q_cut**5)/mbkin**20 - 
           (4248*mckin**12*muG*mupi*q_cut**5)/mbkin**22 - (1200*mckin**2*muG**2*q_cut**6)/
            mbkin**14 + (10728*mckin**4*muG**2*q_cut**6)/mbkin**16 + 
           (57048*mckin**6*muG**2*q_cut**6)/mbkin**18 + (64872*mckin**8*muG**2*q_cut**6)/
            mbkin**20 + (6120*mckin**10*muG**2*q_cut**6)/mbkin**22 + 
           (240*mckin**2*muG*mupi*q_cut**6)/mbkin**14 - (1608*mckin**4*muG*mupi*
             q_cut**6)/mbkin**16 - (11832*mckin**6*muG*mupi*q_cut**6)/mbkin**18 - 
           (11784*mckin**8*muG*mupi*q_cut**6)/mbkin**20 - 
           (1224*mckin**10*muG*mupi*q_cut**6)/mbkin**22 + (1200*mckin**2*muG**2*q_cut**7)/
            mbkin**16 - (18600*mckin**4*muG**2*q_cut**7)/mbkin**18 - 
           (40608*mckin**6*muG**2*q_cut**7)/mbkin**20 - (16200*mckin**8*muG**2*q_cut**7)/
            mbkin**22 - (240*mckin**2*muG*mupi*q_cut**7)/mbkin**16 + 
           (3720*mckin**4*muG*mupi*q_cut**7)/mbkin**18 + 
           (7584*mckin**6*muG*mupi*q_cut**7)/mbkin**20 + 
           (3240*mckin**8*muG*mupi*q_cut**7)/mbkin**22 - (300*mckin**2*muG**2*q_cut**8)/
            mbkin**18 + (9180*mckin**4*muG**2*q_cut**8)/mbkin**20 + 
           (8280*mckin**6*muG**2*q_cut**8)/mbkin**22 + (60*mckin**2*muG*mupi*q_cut**8)/
            mbkin**18 - (1836*mckin**4*muG*mupi*q_cut**8)/mbkin**20 - 
           (1656*mckin**6*muG*mupi*q_cut**8)/mbkin**22 - (1440*mckin**4*muG**2*q_cut**9)/
            mbkin**22 + (288*mckin**4*muG*mupi*q_cut**9)/mbkin**22 - 
           24*mckin**2*muG*((-1 + mckin**2/mbkin**2)**2*(-5 + (73*mckin**2)/
                mbkin**2 - (129*mckin**4)/mbkin**4 - (455*mckin**6)/mbkin**6 + 
               (1321*mckin**8)/mbkin**8 + (5151*mckin**10)/mbkin**10 - 
               (1367*mckin**12)/mbkin**12 - (305*mckin**14)/mbkin**14 + 
               (36*mckin**16)/mbkin**16) - (2*(-10 + (137*mckin**2)/mbkin**2 - 
                (320*mckin**4)/mbkin**4 - (1000*mckin**6)/mbkin**6 + 
                (3080*mckin**8)/mbkin**8 + (3526*mckin**10)/mbkin**10 + 
                (5384*mckin**12)/mbkin**12 - (1624*mckin**14)/mbkin**14 - 
                (614*mckin**16)/mbkin**16 + (81*mckin**18)/mbkin**18)*q_cut)/
              mbkin**2 + (2*(-10 + (105*mckin**2)/mbkin**2 - (141*mckin**4)/
                 mbkin**4 - (673*mckin**6)/mbkin**6 - (627*mckin**8)/mbkin**8 + 
                (1923*mckin**10)/mbkin**10 - (823*mckin**12)/mbkin**12 - 
                (579*mckin**14)/mbkin**14 + (105*mckin**16)/mbkin**16)*q_cut**2)/
              mbkin**4 + (2*(-10 + (151*mckin**2)/mbkin**2 - (296*mckin**4)/
                 mbkin**4 - (2759*mckin**6)/mbkin**6 - (4382*mckin**8)/mbkin**8 - 
                (2963*mckin**10)/mbkin**10 - (64*mckin**12)/mbkin**12 + 
                (51*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
             ((50 - (676*mckin**2)/mbkin**2 + (1006*mckin**4)/mbkin**4 + 
                (7976*mckin**6)/mbkin**6 + (7754*mckin**8)/mbkin**8 + 
                (880*mckin**10)/mbkin**10 - (486*mckin**12)/mbkin**12)*q_cut**4)/
              mbkin**8 + ((-20 + (402*mckin**2)/mbkin**2 - (984*mckin**4)/
                 mbkin**4 - (3380*mckin**6)/mbkin**6 - (948*mckin**8)/mbkin**8 + 
                (354*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
             (2*(-10 + (67*mckin**2)/mbkin**2 + (493*mckin**4)/mbkin**4 + 
                (491*mckin**6)/mbkin**6 + (51*mckin**8)/mbkin**8)*q_cut**6)/
              mbkin**12 - (2*(-10 + (155*mckin**2)/mbkin**2 + (316*mckin**4)/
                 mbkin**4 + (135*mckin**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
             ((-5 + (153*mckin**2)/mbkin**2 + (138*mckin**4)/mbkin**4)*q_cut**8)/
              mbkin**16 - (24*mckin**2*q_cut**9)/mbkin**20) - 
           4*(-((-1 + mckin**2/mbkin**2)**2*(-43 + (438*mckin**2)/mbkin**2 + 
                (1390*mckin**4)/mbkin**4 - (15994*mckin**6)/mbkin**6 + 
                (14652*mckin**8)/mbkin**8 + (24394*mckin**10)/mbkin**10 + 
                (8738*mckin**12)/mbkin**12 - (6774*mckin**14)/mbkin**14 - 
                (1025*mckin**16)/mbkin**16 + (144*mckin**18)/mbkin**18)) + 
             (4*(-49 + (417*mckin**2)/mbkin**2 + (935*mckin**4)/mbkin**4 - 
                (7247*mckin**6)/mbkin**6 + (2361*mckin**8)/mbkin**8 + 
                (13385*mckin**10)/mbkin**10 + (16711*mckin**12)/mbkin**12 + 
                (4863*mckin**14)/mbkin**14 - (4600*mckin**16)/mbkin**16 - 
                (1018*mckin**18)/mbkin**18 + (162*mckin**20)/mbkin**20)*q_cut)/
              mbkin**2 - (4*(-65 + (360*mckin**2)/mbkin**2 + (625*mckin**4)/
                 mbkin**4 - (1220*mckin**6)/mbkin**6 + (2577*mckin**8)/mbkin**8 + 
                (1258*mckin**10)/mbkin**10 - (1589*mckin**12)/mbkin**12 - 
                (3504*mckin**14)/mbkin**14 - (812*mckin**16)/mbkin**16 + 
                (210*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
             (4*(-29 + (101*mckin**2)/mbkin**2 + (1989*mckin**4)/mbkin**4 + 
                (173*mckin**6)/mbkin**6 - (2343*mckin**8)/mbkin**8 - 
                (5835*mckin**10)/mbkin**10 - (4455*mckin**12)/mbkin**12 - 
                (359*mckin**14)/mbkin**14 + (102*mckin**16)/mbkin**16)*q_cut**3)/
              mbkin**6 + (2*(-299 + (600*mckin**2)/mbkin**2 + (5761*mckin**4)/
                 mbkin**4 + (944*mckin**6)/mbkin**6 - (14463*mckin**8)/mbkin**8 - 
                (14012*mckin**10)/mbkin**10 - (1655*mckin**12)/mbkin**12 + 
                (972*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 - 
             (4*(-113 + (249*mckin**2)/mbkin**2 + (1309*mckin**4)/mbkin**4 - 
                (833*mckin**6)/mbkin**6 - (3006*mckin**8)/mbkin**8 - 
                (728*mckin**10)/mbkin**10 + (354*mckin**12)/mbkin**12)*q_cut**5)/
              mbkin**10 - (4*(-29 - (268*mckin**2)/mbkin**2 - (177*mckin**4)/
                 mbkin**4 + (714*mckin**6)/mbkin**6 + (802*mckin**8)/mbkin**8 + 
                (102*mckin**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
             (4*(-85 - (195*mckin**2)/mbkin**2 + (167*mckin**4)/mbkin**4 + 
                (563*mckin**6)/mbkin**6 + (270*mckin**8)/mbkin**8)*q_cut**7)/
              mbkin**14 + ((179 + (204*mckin**2)/mbkin**2 - (571*mckin**4)/
                 mbkin**4 - (552*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 - 
             (32*(mbkin**4 - 3*mckin**4)*q_cut**9)/mbkin**22)*rG + 
           8*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(21 - (236*mckin**2)/mbkin**2 + 
                (280*mckin**4)/mbkin**4 + (5400*mckin**6)/mbkin**6 - 
                (43634*mckin**8)/mbkin**8 - (31160*mckin**10)/mbkin**10 + 
                (14712*mckin**12)/mbkin**12 - (4424*mckin**14)/mbkin**14 - 
                (1619*mckin**16)/mbkin**16 + (180*mckin**18)/mbkin**18)) + 
             (2*(51 - (508*mckin**2)/mbkin**2 + (671*mckin**4)/mbkin**4 + 
                (7066*mckin**6)/mbkin**6 - (40432*mckin**8)/mbkin**8 - 
                (66010*mckin**10)/mbkin**10 - (36356*mckin**12)/mbkin**12 + 
                (22310*mckin**14)/mbkin**14 - (4915*mckin**16)/mbkin**16 - 
                (3242*mckin**18)/mbkin**18 + (405*mckin**20)/mbkin**20)*q_cut)/
              mbkin**2 - (2*(75 - (535*mckin**2)/mbkin**2 + (234*mckin**4)/
                 mbkin**4 + (3612*mckin**6)/mbkin**6 - (2692*mckin**8)/mbkin**8 + 
                (6882*mckin**10)/mbkin**10 + (9870*mckin**12)/mbkin**12 - 
                (5092*mckin**14)/mbkin**14 - (2799*mckin**16)/mbkin**16 + 
                (525*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
             (2*(-21 + (124*mckin**2)/mbkin**2 + (650*mckin**4)/mbkin**4 - 
                (4312*mckin**6)/mbkin**6 + (1928*mckin**8)/mbkin**8 + 
                (728*mckin**10)/mbkin**10 + (7154*mckin**12)/mbkin**12 + 
                (1396*mckin**14)/mbkin**14 - (255*mckin**16)/mbkin**16)*q_cut**3)/
              mbkin**6 + (2*(168 - (595*mckin**2)/mbkin**2 - (1396*mckin**4)/
                 mbkin**4 + (3013*mckin**6)/mbkin**6 - (1930*mckin**8)/mbkin**8 - 
                (13117*mckin**10)/mbkin**10 - (3590*mckin**12)/mbkin**12 + 
                (1215*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
             (2*(-147 + (436*mckin**2)/mbkin**2 + (723*mckin**4)/mbkin**4 + 
                (726*mckin**6)/mbkin**6 + (6157*mckin**8)/mbkin**8 + 
                (2478*mckin**10)/mbkin**10 - (885*mckin**12)/mbkin**12)*q_cut**5)/
              mbkin**10 - (2*(21 + (227*mckin**2)/mbkin**2 + (112*mckin**4)/
                 mbkin**4 + (1090*mckin**6)/mbkin**6 + (1463*mckin**8)/mbkin**8 + 
                (255*mckin**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
             (2*(105 + (140*mckin**2)/mbkin**2 + (248*mckin**4)/mbkin**4 + 
                (856*mckin**6)/mbkin**6 + (675*mckin**8)/mbkin**8)*q_cut**7)/
              mbkin**14 - ((123 + (88*mckin**2)/mbkin**2 + (447*mckin**4)/
                 mbkin**4 + (690*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
             (24*(mbkin**4 + 5*mckin**4)*q_cut**9)/mbkin**22)*rhoD + 60*sB - 
           (520*mckin**2*sB)/mbkin**2 - (8228*mckin**4*sB)/mbkin**4 + 
           (63424*mckin**6*sB)/mbkin**6 + (35656*mckin**8*sB)/mbkin**8 - 
           (82832*mckin**10*sB)/mbkin**10 - (57080*mckin**12*sB)/mbkin**12 - 
           (98912*mckin**14*sB)/mbkin**14 + (191356*mckin**16*sB)/mbkin**16 - 
           (30056*mckin**18*sB)/mbkin**18 - (14308*mckin**20*sB)/mbkin**20 + 
           (1440*mckin**22*sB)/mbkin**22 - (240*q_cut*sB)/mbkin**2 + 
           (880*mckin**2*q_cut*sB)/mbkin**4 + (28672*mckin**4*q_cut*sB)/mbkin**6 - 
           (57136*mckin**6*q_cut*sB)/mbkin**8 - (436016*mckin**8*q_cut*sB)/mbkin**10 - 
           (582320*mckin**10*q_cut*sB)/mbkin**12 - (280240*mckin**12*q_cut*sB)/
            mbkin**14 - (229712*mckin**14*q_cut*sB)/mbkin**16 + 
           (134176*mckin**16*q_cut*sB)/mbkin**18 + (46016*mckin**18*q_cut*sB)/
            mbkin**20 - (6480*mckin**20*q_cut*sB)/mbkin**22 + (240*q_cut**2*sB)/
            mbkin**4 + (320*mckin**2*q_cut**2*sB)/mbkin**6 - (16896*mckin**4*q_cut**2*sB)/
            mbkin**8 - (22800*mckin**6*q_cut**2*sB)/mbkin**10 + 
           (41120*mckin**8*q_cut**2*sB)/mbkin**12 - (18960*mckin**10*q_cut**2*sB)/
            mbkin**14 + (28800*mckin**12*q_cut**2*sB)/mbkin**16 - 
           (95344*mckin**14*q_cut**2*sB)/mbkin**18 - (40080*mckin**16*q_cut**2*sB)/
            mbkin**20 + (8400*mckin**18*q_cut**2*sB)/mbkin**22 + 
           (240*q_cut**3*sB)/mbkin**6 + (80*mckin**2*q_cut**3*sB)/mbkin**8 - 
           (23840*mckin**4*q_cut**3*sB)/mbkin**10 - (80912*mckin**6*q_cut**3*sB)/
            mbkin**12 - (158912*mckin**8*q_cut**3*sB)/mbkin**14 - 
           (228848*mckin**10*q_cut**3*sB)/mbkin**16 - (174080*mckin**12*q_cut**3*sB)/
            mbkin**18 - (13648*mckin**14*q_cut**3*sB)/mbkin**20 + 
           (4080*mckin**16*q_cut**3*sB)/mbkin**22 - (600*q_cut**4*sB)/mbkin**8 - 
           (2480*mckin**2*q_cut**4*sB)/mbkin**10 + (20488*mckin**4*q_cut**4*sB)/
            mbkin**12 + (113552*mckin**6*q_cut**4*sB)/mbkin**14 + 
           (253960*mckin**8*q_cut**4*sB)/mbkin**16 + (257200*mckin**10*q_cut**4*sB)/
            mbkin**18 + (41000*mckin**12*q_cut**4*sB)/mbkin**20 - 
           (19440*mckin**14*q_cut**4*sB)/mbkin**22 + (240*q_cut**5*sB)/mbkin**10 + 
           (1040*mckin**2*q_cut**5*sB)/mbkin**12 + (1056*mckin**4*q_cut**5*sB)/
            mbkin**14 - (39792*mckin**6*q_cut**5*sB)/mbkin**16 - 
           (110944*mckin**8*q_cut**5*sB)/mbkin**18 - (35232*mckin**10*q_cut**5*sB)/
            mbkin**20 + (14160*mckin**12*q_cut**5*sB)/mbkin**22 + 
           (240*q_cut**6*sB)/mbkin**12 + (2240*mckin**2*q_cut**6*sB)/mbkin**14 + 
           (5056*mckin**4*q_cut**6*sB)/mbkin**16 + (29104*mckin**6*q_cut**6*sB)/
            mbkin**18 + (31664*mckin**8*q_cut**6*sB)/mbkin**20 + 
           (4080*mckin**10*q_cut**6*sB)/mbkin**22 - (240*q_cut**7*sB)/mbkin**14 - 
           (2000*mckin**2*q_cut**7*sB)/mbkin**16 - (10496*mckin**4*q_cut**7*sB)/
            mbkin**18 - (20560*mckin**6*q_cut**7*sB)/mbkin**20 - 
           (10800*mckin**8*q_cut**7*sB)/mbkin**22 + (60*q_cut**8*sB)/mbkin**16 + 
           (440*mckin**2*q_cut**8*sB)/mbkin**18 + (5148*mckin**4*q_cut**8*sB)/mbkin**20 + 
           (5520*mckin**6*q_cut**8*sB)/mbkin**22 - (960*mckin**4*q_cut**9*sB)/mbkin**22 - 
           96*sE + (928*mckin**2*sE)/mbkin**2 + (5792*mckin**4*sE)/mbkin**4 - 
           (54880*mckin**6*sE)/mbkin**6 + (55616*mckin**8*sE)/mbkin**8 - 
           (48448*mckin**10*sE)/mbkin**10 + (78656*mckin**12*sE)/mbkin**12 + 
           (33728*mckin**14*sE)/mbkin**14 - (99040*mckin**16*sE)/mbkin**16 + 
           (20384*mckin**18*sE)/mbkin**18 + (8224*mckin**20*sE)/mbkin**20 - 
           (864*mckin**22*sE)/mbkin**22 + (432*q_cut*sE)/mbkin**2 - 
           (2656*mckin**2*q_cut*sE)/mbkin**4 - (22480*mckin**4*q_cut*sE)/mbkin**6 + 
           (67264*mckin**6*q_cut*sE)/mbkin**8 + (180512*mckin**8*q_cut*sE)/mbkin**10 + 
           (197120*mckin**10*q_cut*sE)/mbkin**12 + (166816*mckin**12*q_cut*sE)/
            mbkin**14 + (84800*mckin**14*q_cut*sE)/mbkin**16 - 
           (97744*mckin**16*q_cut*sE)/mbkin**18 - (24992*mckin**18*q_cut*sE)/
            mbkin**20 + (3888*mckin**20*q_cut*sE)/mbkin**22 - (560*q_cut**2*sE)/
            mbkin**4 + (2000*mckin**2*q_cut**2*sE)/mbkin**6 + 
           (14336*mckin**4*q_cut**2*sE)/mbkin**8 + (1792*mckin**6*q_cut**2*sE)/
            mbkin**10 - (20512*mckin**8*q_cut**2*sE)/mbkin**12 - 
           (23968*mckin**10*q_cut**2*sE)/mbkin**14 + (98176*mckin**12*q_cut**2*sE)/
            mbkin**16 + (101504*mckin**14*q_cut**2*sE)/mbkin**18 + 
           (16592*mckin**16*q_cut**2*sE)/mbkin**20 - (5040*mckin**18*q_cut**2*sE)/
            mbkin**22 - (272*q_cut**3*sE)/mbkin**6 - (32*mckin**2*q_cut**3*sE)/mbkin**8 + 
           (25472*mckin**4*q_cut**3*sE)/mbkin**10 + (46368*mckin**6*q_cut**3*sE)/
            mbkin**12 + (54944*mckin**8*q_cut**3*sE)/mbkin**14 - 
           (1120*mckin**10*q_cut**3*sE)/mbkin**16 + (51072*mckin**12*q_cut**3*sE)/
            mbkin**18 + (18016*mckin**14*q_cut**3*sE)/mbkin**20 - 
           (2448*mckin**16*q_cut**3*sE)/mbkin**22 + (1296*q_cut**4*sE)/mbkin**8 + 
           (80*mckin**2*q_cut**4*sE)/mbkin**10 - (32080*mckin**4*q_cut**4*sE)/mbkin**12 - 
           (69200*mckin**6*q_cut**4*sE)/mbkin**14 - (89296*mckin**8*q_cut**4*sE)/
            mbkin**16 - (119440*mckin**10*q_cut**4*sE)/mbkin**18 - 
           (28400*mckin**12*q_cut**4*sE)/mbkin**20 + (11664*mckin**14*q_cut**4*sE)/
            mbkin**22 - (944*q_cut**5*sE)/mbkin**10 + (992*mckin**2*q_cut**5*sE)/
            mbkin**12 + (9584*mckin**4*q_cut**5*sE)/mbkin**14 + 
           (19072*mckin**6*q_cut**5*sE)/mbkin**16 + (49520*mckin**8*q_cut**5*sE)/
            mbkin**18 + (15776*mckin**10*q_cut**5*sE)/mbkin**20 - 
           (8496*mckin**12*q_cut**5*sE)/mbkin**22 - (272*q_cut**6*sE)/mbkin**12 - 
           (3344*mckin**2*q_cut**6*sE)/mbkin**14 - (3424*mckin**4*q_cut**6*sE)/
            mbkin**16 - (9312*mckin**6*q_cut**6*sE)/mbkin**18 - 
           (11408*mckin**8*q_cut**6*sE)/mbkin**20 - (2448*mckin**10*q_cut**6*sE)/
            mbkin**22 + (720*q_cut**7*sE)/mbkin**14 + (2720*mckin**2*q_cut**7*sE)/
            mbkin**16 + (4640*mckin**4*q_cut**7*sE)/mbkin**18 + 
           (8608*mckin**6*q_cut**7*sE)/mbkin**20 + (6480*mckin**8*q_cut**7*sE)/
            mbkin**22 - (368*q_cut**8*sE)/mbkin**16 - (688*mckin**2*q_cut**8*sE)/
            mbkin**18 - (2416*mckin**4*q_cut**8*sE)/mbkin**20 - 
           (3312*mckin**6*q_cut**8*sE)/mbkin**22 + (64*q_cut**9*sE)/mbkin**18 + 
           (576*mckin**4*q_cut**9*sE)/mbkin**22 - 3*sqB + (34*mckin**2*sqB)/
            mbkin**2 + (893*mckin**4*sqB)/mbkin**4 - (6652*mckin**6*sqB)/
            mbkin**6 - (19834*mckin**8*sqB)/mbkin**8 + (34484*mckin**10*sqB)/
            mbkin**10 + (5222*mckin**12*sqB)/mbkin**12 - (4768*mckin**14*sqB)/
            mbkin**14 - (11203*mckin**16*sqB)/mbkin**16 + (1514*mckin**18*sqB)/
            mbkin**18 + (349*mckin**20*sqB)/mbkin**20 - (36*mckin**22*sqB)/
            mbkin**22 + (6*q_cut*sqB)/mbkin**2 + (32*mckin**2*q_cut*sqB)/mbkin**4 - 
           (3202*mckin**4*q_cut*sqB)/mbkin**6 + (4780*mckin**6*q_cut*sqB)/mbkin**8 + 
           (72536*mckin**8*q_cut*sqB)/mbkin**10 + (107300*mckin**10*q_cut*sqB)/
            mbkin**12 + (57856*mckin**12*q_cut*sqB)/mbkin**14 + 
           (9620*mckin**14*q_cut*sqB)/mbkin**16 - (6142*mckin**16*q_cut*sqB)/
            mbkin**18 - (1028*mckin**18*q_cut*sqB)/mbkin**20 + 
           (162*mckin**20*q_cut*sqB)/mbkin**22 + (10*q_cut**2*sqB)/mbkin**4 - 
           (250*mckin**2*q_cut**2*sqB)/mbkin**6 + (1892*mckin**4*q_cut**2*sqB)/mbkin**8 + 
           (3880*mckin**6*q_cut**2*sqB)/mbkin**10 - (7312*mckin**8*q_cut**2*sqB)/
            mbkin**12 + (7772*mckin**10*q_cut**2*sqB)/mbkin**14 + 
           (9004*mckin**12*q_cut**2*sqB)/mbkin**16 + (4712*mckin**14*q_cut**2*sqB)/
            mbkin**18 + (662*mckin**16*q_cut**2*sqB)/mbkin**20 - 
           (210*mckin**18*q_cut**2*sqB)/mbkin**22 - (26*q_cut**3*sqB)/mbkin**6 + 
           (64*mckin**2*q_cut**3*sqB)/mbkin**8 + (2588*mckin**4*q_cut**3*sqB)/mbkin**10 + 
           (8856*mckin**6*q_cut**3*sqB)/mbkin**12 + (3320*mckin**8*q_cut**3*sqB)/
            mbkin**14 + (4664*mckin**10*q_cut**3*sqB)/mbkin**16 + 
           (5580*mckin**12*q_cut**3*sqB)/mbkin**18 + (592*mckin**14*q_cut**3*sqB)/
            mbkin**20 - (102*mckin**16*q_cut**3*sqB)/mbkin**22 - 
           (12*q_cut**4*sqB)/mbkin**8 + (410*mckin**2*q_cut**4*sqB)/mbkin**10 - 
           (2188*mckin**4*q_cut**4*sqB)/mbkin**12 - (10862*mckin**6*q_cut**4*sqB)/
            mbkin**14 - (12784*mckin**8*q_cut**4*sqB)/mbkin**16 - 
           (8026*mckin**10*q_cut**4*sqB)/mbkin**18 - (800*mckin**12*q_cut**4*sqB)/
            mbkin**20 + (486*mckin**14*q_cut**4*sqB)/mbkin**22 + 
           (58*q_cut**5*sqB)/mbkin**10 - (304*mckin**2*q_cut**5*sqB)/mbkin**12 - 
           (202*mckin**4*q_cut**5*sqB)/mbkin**14 + (1540*mckin**6*q_cut**5*sqB)/
            mbkin**16 + (2402*mckin**8*q_cut**5*sqB)/mbkin**18 + 
           (476*mckin**10*q_cut**5*sqB)/mbkin**20 - (354*mckin**12*q_cut**5*sqB)/
            mbkin**22 - (26*q_cut**6*sqB)/mbkin**12 - (62*mckin**2*q_cut**6*sqB)/
            mbkin**14 + (8*mckin**4*q_cut**6*sqB)/mbkin**16 - (372*mckin**6*q_cut**6*sqB)/
            mbkin**18 - (662*mckin**8*q_cut**6*sqB)/mbkin**20 - 
           (102*mckin**10*q_cut**6*sqB)/mbkin**22 - (30*q_cut**7*sqB)/mbkin**14 + 
           (80*mckin**2*q_cut**7*sqB)/mbkin**16 + (344*mckin**4*q_cut**7*sqB)/mbkin**18 + 
           (568*mckin**6*q_cut**7*sqB)/mbkin**20 + (270*mckin**8*q_cut**7*sqB)/
            mbkin**22 + (31*q_cut**8*sqB)/mbkin**16 - (4*mckin**2*q_cut**8*sqB)/
            mbkin**18 - (157*mckin**4*q_cut**8*sqB)/mbkin**20 - 
           (138*mckin**6*q_cut**8*sqB)/mbkin**22 - (8*q_cut**9*sqB)/mbkin**18 + 
           (24*mckin**4*q_cut**9*sqB)/mbkin**22))*
        np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
              mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
           mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))) - 
       (144*mckin**4*(-16*(-((-1 + mckin**2/mbkin**2)**4*(-16 + (84*mckin**2)/
                mbkin**2 + (397*mckin**4)/mbkin**4 + (7*mckin**6)/mbkin**6 - 
               (273*mckin**8)/mbkin**8 + (53*mckin**10)/mbkin**10 + 
               (108*mckin**12)/mbkin**12)) + ((-1 + mckin**2/mbkin**2)**2*
              (-83 + (309*mckin**2)/mbkin**2 + (1688*mckin**4)/mbkin**4 + 
               (1592*mckin**6)/mbkin**6 - (423*mckin**8)/mbkin**8 - 
               (1055*mckin**10)/mbkin**10 + (258*mckin**12)/mbkin**12 + 
               (594*mckin**14)/mbkin**14)*q_cut)/mbkin**2 - 
            ((-158 + (492*mckin**2)/mbkin**2 + (2013*mckin**4)/mbkin**4 + 
               (1738*mckin**6)/mbkin**6 + (2412*mckin**8)/mbkin**8 + 
               (120*mckin**10)/mbkin**10 - (2075*mckin**12)/mbkin**12 - 
               (6*mckin**14)/mbkin**14 + (1224*mckin**16)/mbkin**16)*q_cut**2)/
             mbkin**4 + ((-101 + (215*mckin**2)/mbkin**2 + (929*mckin**4)/
                mbkin**4 + (641*mckin**6)/mbkin**6 - (688*mckin**8)/mbkin**8 - 
               (784*mckin**10)/mbkin**10 + (1330*mckin**12)/mbkin**12 + 
               (918*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
            ((-80 - (260*mckin**2)/mbkin**2 + (156*mckin**4)/mbkin**4 + 
               (895*mckin**6)/mbkin**6 + (742*mckin**8)/mbkin**8 - (45*mckin**10)/
                mbkin**10 + (540*mckin**12)/mbkin**12)*q_cut**4)/mbkin**8 - 
            ((-179 - (393*mckin**2)/mbkin**2 + (414*mckin**4)/mbkin**4 + 
               (1540*mckin**6)/mbkin**6 + (1914*mckin**8)/mbkin**8 + 
               (1602*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
            (2*(-61 - (110*mckin**2)/mbkin**2 + (307*mckin**4)/mbkin**4 + 
               (770*mckin**6)/mbkin**6 + (648*mckin**8)/mbkin**8)*q_cut**6)/
             mbkin**12 + ((37 + (37*mckin**2)/mbkin**2 - (366*mckin**4)/mbkin**4 - 
               (486*mckin**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
            ((-4 + (72*mckin**4)/mbkin**4)*q_cut**8)/mbkin**16)*rE + 
          ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)*((-60*mckin**2*muG**2)/mbkin**2 + (384*mckin**4*muG**2)/
             mbkin**4 - (1704*mckin**6*muG**2)/mbkin**6 + (6120*mckin**8*muG**2)/
             mbkin**8 - (8700*mckin**10*muG**2)/mbkin**10 + (4080*mckin**12*muG**2)/
             mbkin**12 + (96*mckin**14*muG**2)/mbkin**14 - (216*mckin**16*muG**2)/
             mbkin**16 - (60*mckin**2*muG*mupi)/mbkin**2 + 
            (576*mckin**4*muG*mupi)/mbkin**4 - (1296*mckin**6*muG*mupi)/
             mbkin**6 - (1320*mckin**8*muG*mupi)/mbkin**8 + 
            (4500*mckin**10*muG*mupi)/mbkin**10 - (2160*mckin**12*muG*mupi)/
             mbkin**12 - (456*mckin**14*muG*mupi)/mbkin**14 + 
            (216*mckin**16*muG*mupi)/mbkin**16 - (60*mckin**2*muG**2*q_cut)/
             mbkin**4 + (876*mckin**4*muG**2*q_cut)/mbkin**6 - 
            (1128*mckin**6*muG**2*q_cut)/mbkin**8 - (2388*mckin**8*muG**2*q_cut)/
             mbkin**10 - (7248*mckin**10*muG**2*q_cut)/mbkin**12 + 
            (552*mckin**12*muG**2*q_cut)/mbkin**14 + (756*mckin**14*muG**2*q_cut)/
             mbkin**16 + (180*mckin**2*muG*mupi*q_cut)/mbkin**4 - 
            (1236*mckin**4*muG*mupi*q_cut)/mbkin**6 + (408*mckin**6*muG*mupi*q_cut)/
             mbkin**8 + (5508*mckin**8*muG*mupi*q_cut)/mbkin**10 + 
            (4008*mckin**10*muG*mupi*q_cut)/mbkin**12 + (528*mckin**12*muG*mupi*q_cut)/
             mbkin**14 - (756*mckin**14*muG*mupi*q_cut)/mbkin**16 + 
            (120*mckin**2*muG**2*q_cut**2)/mbkin**6 - (360*mckin**4*muG**2*q_cut**2)/
             mbkin**8 - (1188*mckin**6*muG**2*q_cut**2)/mbkin**10 + 
            (2976*mckin**8*muG**2*q_cut**2)/mbkin**12 - (108*mckin**10*muG**2*q_cut**2)/
             mbkin**14 - (720*mckin**12*muG**2*q_cut**2)/mbkin**16 - 
            (120*mckin**2*muG*mupi*q_cut**2)/mbkin**6 + (600*mckin**4*muG*mupi*q_cut**2)/
             mbkin**8 - (12*mckin**6*muG*mupi*q_cut**2)/mbkin**10 - 
            (1296*mckin**8*muG*mupi*q_cut**2)/mbkin**12 - 
            (612*mckin**10*muG*mupi*q_cut**2)/mbkin**14 + 
            (720*mckin**12*muG*mupi*q_cut**2)/mbkin**16 + (360*mckin**2*muG**2*q_cut**3)/
             mbkin**8 - (1020*mckin**4*muG**2*q_cut**3)/mbkin**10 - 
            (2868*mckin**6*muG**2*q_cut**3)/mbkin**12 - (3048*mckin**8*muG**2*q_cut**3)/
             mbkin**14 - (360*mckin**10*muG**2*q_cut**3)/mbkin**16 - 
            (120*mckin**2*muG*mupi*q_cut**3)/mbkin**8 + (780*mckin**4*muG*mupi*q_cut**3)/
             mbkin**10 + (1668*mckin**6*muG*mupi*q_cut**3)/mbkin**12 + 
            (2328*mckin**8*muG*mupi*q_cut**3)/mbkin**14 + 
            (360*mckin**10*muG*mupi*q_cut**3)/mbkin**16 - (540*mckin**2*muG**2*q_cut**4)/
             mbkin**10 + (2160*mckin**4*muG**2*q_cut**4)/mbkin**12 + 
            (3732*mckin**6*muG**2*q_cut**4)/mbkin**14 + (1080*mckin**8*muG**2*q_cut**4)/
             mbkin**16 + (180*mckin**2*muG*mupi*q_cut**4)/mbkin**10 - 
            (1440*mckin**4*muG*mupi*q_cut**4)/mbkin**12 - 
            (2652*mckin**6*muG*mupi*q_cut**4)/mbkin**14 - 
            (1080*mckin**8*muG*mupi*q_cut**4)/mbkin**16 + (180*mckin**2*muG**2*q_cut**5)/
             mbkin**12 - (1224*mckin**4*muG**2*q_cut**5)/mbkin**14 - 
            (684*mckin**6*muG**2*q_cut**5)/mbkin**16 - (60*mckin**2*muG*mupi*q_cut**5)/
             mbkin**12 + (864*mckin**4*muG*mupi*q_cut**5)/mbkin**14 + 
            (684*mckin**6*muG*mupi*q_cut**5)/mbkin**16 + (144*mckin**4*muG**2*q_cut**6)/
             mbkin**16 - (144*mckin**4*muG*mupi*q_cut**6)/mbkin**16 + 
            24*mckin**2*muG*((-1 + mckin**2/mbkin**2)**2*(-5 + (38*mckin**2)/
                 mbkin**2 - (27*mckin**4)/mbkin**4 - (202*mckin**6)/mbkin**6 - 
                (2*mckin**8)/mbkin**8 + (18*mckin**10)/mbkin**10) + 
              ((15 - (103*mckin**2)/mbkin**2 + (34*mckin**4)/mbkin**4 + 
                 (459*mckin**6)/mbkin**6 + (334*mckin**8)/mbkin**8 + 
                 (44*mckin**10)/mbkin**10 - (63*mckin**12)/mbkin**12)*q_cut)/mbkin**
                2 + ((-10 + (50*mckin**2)/mbkin**2 - mckin**4/mbkin**4 - 
                 (108*mckin**6)/mbkin**6 - (51*mckin**8)/mbkin**8 + (60*mckin**10)/
                  mbkin**10)*q_cut**2)/mbkin**4 + ((-10 + (65*mckin**2)/mbkin**2 + 
                 (139*mckin**4)/mbkin**4 + (194*mckin**6)/mbkin**6 + (30*mckin**8)/
                  mbkin**8)*q_cut**3)/mbkin**6 - ((-15 + (120*mckin**2)/mbkin**2 + 
                 (221*mckin**4)/mbkin**4 + (90*mckin**6)/mbkin**6)*q_cut**4)/mbkin**
                8 + ((-5 + (72*mckin**2)/mbkin**2 + (57*mckin**4)/mbkin**4)*q_cut**5)/
               mbkin**10 - (12*mckin**2*q_cut**6)/mbkin**14) + 
            4*(-((-1 + mckin**2/mbkin**2)**2*(-19 + (41*mckin**2)/mbkin**2 + 
                 (383*mckin**4)/mbkin**4 - (917*mckin**6)/mbkin**6 - 
                 (752*mckin**8)/mbkin**8 + (112*mckin**10)/mbkin**10 + 
                 (72*mckin**12)/mbkin**12)) + ((-69 + (122*mckin**2)/mbkin**2 + 
                 (539*mckin**4)/mbkin**4 - (814*mckin**6)/mbkin**6 - 
                 (3029*mckin**8)/mbkin**8 - (1520*mckin**10)/mbkin**10 + 
                 (199*mckin**12)/mbkin**12 + (252*mckin**14)/mbkin**14)*q_cut)/mbkin**
                2 + ((70 - (68*mckin**2)/mbkin**2 - (99*mckin**4)/mbkin**4 + 
                 (581*mckin**6)/mbkin**6 + (377*mckin**8)/mbkin**8 - 
                 (261*mckin**10)/mbkin**10 - (240*mckin**12)/mbkin**12)*q_cut**2)/
               mbkin**4 - (2*(-15 - (91*mckin**2)/mbkin**2 + (79*mckin**4)/
                  mbkin**4 + (143*mckin**6)/mbkin**6 + (238*mckin**8)/mbkin**8 + 
                 (60*mckin**10)/mbkin**10)*q_cut**3)/mbkin**6 + 
              ((-105 - (253*mckin**2)/mbkin**2 + (295*mckin**4)/mbkin**4 + 
                 (779*mckin**6)/mbkin**6 + (360*mckin**8)/mbkin**8)*q_cut**4)/mbkin**
                8 + ((71 + (96*mckin**2)/mbkin**2 - (273*mckin**4)/mbkin**4 - 
                 (228*mckin**6)/mbkin**6)*q_cut**5)/mbkin**10 - 
              (16*(mbkin**4 - 3*mckin**4)*q_cut**6)/mbkin**16)*rG - 
            8*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(3 - (17*mckin**2)/mbkin**2 + 
                 (443*mckin**4)/mbkin**4 + (1923*mckin**6)/mbkin**6 + 
                 (188*mckin**8)/mbkin**8 - (110*mckin**10)/mbkin**10 + 
                 (90*mckin**12)/mbkin**12)) + ((18 - (129*mckin**2)/mbkin**2 + 
                 (1036*mckin**4)/mbkin**4 + (4331*mckin**6)/mbkin**6 + 
                 (4596*mckin**8)/mbkin**8 + (443*mckin**10)/mbkin**10 - 
                 (530*mckin**12)/mbkin**12 + (315*mckin**14)/mbkin**14)*q_cut)/mbkin**
                2 + ((-30 + (136*mckin**2)/mbkin**2 - (287*mckin**4)/mbkin**4 - 
                 (779*mckin**6)/mbkin**6 + (165*mckin**8)/mbkin**8 + 
                 (255*mckin**10)/mbkin**10 - (300*mckin**12)/mbkin**12)*q_cut**2)/
               mbkin**4 - (2*mckin**2*(-8 - (38*mckin**2)/mbkin**2 + (95*mckin**4)/
                  mbkin**4 + (100*mckin**6)/mbkin**6 + (75*mckin**8)/mbkin**8)*
                q_cut**3)/mbkin**8 + ((45 - (39*mckin**2)/mbkin**2 - (45*mckin**4)/
                  mbkin**4 + (305*mckin**6)/mbkin**6 + (450*mckin**8)/mbkin**8)*
                q_cut**4)/mbkin**8 - ((42 + (7*mckin**2)/mbkin**2 + (120*mckin**4)/
                  mbkin**4 + (285*mckin**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
              (12*(mbkin**4 + 5*mckin**4)*q_cut**6)/mbkin**16)*rhoD - 60*sB + 
            (100*mckin**2*sB)/mbkin**2 + (4440*mckin**4*sB)/mbkin**4 - 
            (2040*mckin**6*sB)/mbkin**6 - (5180*mckin**8*sB)/mbkin**8 - 
            (1740*mckin**10*sB)/mbkin**10 + (4080*mckin**12*sB)/mbkin**12 + 
            (1120*mckin**14*sB)/mbkin**14 - (720*mckin**16*sB)/mbkin**16 + 
            (180*q_cut*sB)/mbkin**2 + (480*mckin**2*q_cut*sB)/mbkin**4 - 
            (8980*mckin**4*q_cut*sB)/mbkin**6 - (25160*mckin**6*q_cut*sB)/mbkin**8 - 
            (19140*mckin**8*q_cut*sB)/mbkin**10 - (6800*mckin**10*q_cut*sB)/mbkin**12 - 
            (700*mckin**12*q_cut*sB)/mbkin**14 + (2520*mckin**14*q_cut*sB)/mbkin**16 - 
            (120*q_cut**2*sB)/mbkin**4 - (640*mckin**2*q_cut**2*sB)/mbkin**6 + 
            (2540*mckin**4*q_cut**2*sB)/mbkin**8 + (4100*mckin**6*q_cut**2*sB)/
             mbkin**10 + (1260*mckin**8*q_cut**2*sB)/mbkin**12 + 
            (60*mckin**10*q_cut**2*sB)/mbkin**14 - (2400*mckin**12*q_cut**2*sB)/
             mbkin**16 - (120*q_cut**3*sB)/mbkin**6 - (1000*mckin**2*q_cut**3*sB)/
             mbkin**8 - (2320*mckin**4*q_cut**3*sB)/mbkin**10 - 
            (3920*mckin**6*q_cut**3*sB)/mbkin**12 - (4720*mckin**8*q_cut**3*sB)/
             mbkin**14 - (1200*mckin**10*q_cut**3*sB)/mbkin**16 + 
            (180*q_cut**4*sB)/mbkin**8 + (1500*mckin**2*q_cut**4*sB)/mbkin**10 + 
            (4500*mckin**4*q_cut**4*sB)/mbkin**12 + (6460*mckin**6*q_cut**4*sB)/
             mbkin**14 + (3600*mckin**8*q_cut**4*sB)/mbkin**16 - 
            (60*q_cut**5*sB)/mbkin**10 - (440*mckin**2*q_cut**5*sB)/mbkin**12 - 
            (2220*mckin**4*q_cut**5*sB)/mbkin**14 - (2280*mckin**6*q_cut**5*sB)/
             mbkin**16 + (480*mckin**4*q_cut**6*sB)/mbkin**16 + 48*sE + 
            (32*mckin**2*sE)/mbkin**2 - (2928*mckin**4*sE)/mbkin**4 + 
            (2928*mckin**6*sE)/mbkin**6 + (1280*mckin**8*sE)/mbkin**8 + 
            (192*mckin**10*sE)/mbkin**10 - (1392*mckin**12*sE)/mbkin**12 - 
            (592*mckin**14*sE)/mbkin**14 + (432*mckin**16*sE)/mbkin**16 - 
            (168*q_cut*sE)/mbkin**2 - (456*mckin**2*q_cut*sE)/mbkin**4 + 
            (5272*mckin**4*q_cut*sE)/mbkin**6 + (11720*mckin**6*q_cut*sE)/mbkin**8 + 
            (6120*mckin**8*q_cut*sE)/mbkin**10 + (2408*mckin**10*q_cut*sE)/mbkin**12 - 
            (344*mckin**12*q_cut*sE)/mbkin**14 - (1512*mckin**14*q_cut*sE)/mbkin**16 + 
            (160*q_cut**2*sE)/mbkin**4 + (384*mckin**2*q_cut**2*sE)/mbkin**6 - 
            (1568*mckin**4*q_cut**2*sE)/mbkin**8 - (2160*mckin**6*q_cut**2*sE)/
             mbkin**10 + (448*mckin**8*q_cut**2*sE)/mbkin**12 + 
            (2256*mckin**10*q_cut**2*sE)/mbkin**14 + (1440*mckin**12*q_cut**2*sE)/
             mbkin**16 + (80*q_cut**3*sE)/mbkin**6 + (944*mckin**2*q_cut**3*sE)/
             mbkin**8 + (944*mckin**4*q_cut**3*sE)/mbkin**10 + (736*mckin**6*q_cut**3*sE)/
             mbkin**12 - (224*mckin**8*q_cut**3*sE)/mbkin**14 + 
            (720*mckin**10*q_cut**3*sE)/mbkin**16 - (240*q_cut**4*sE)/mbkin**8 - 
            (1376*mckin**2*q_cut**4*sE)/mbkin**10 - (1920*mckin**4*q_cut**4*sE)/
             mbkin**12 - (1984*mckin**6*q_cut**4*sE)/mbkin**14 - 
            (2160*mckin**8*q_cut**4*sE)/mbkin**16 + (152*q_cut**5*sE)/mbkin**10 + 
            (472*mckin**2*q_cut**5*sE)/mbkin**12 + (888*mckin**4*q_cut**5*sE)/mbkin**14 + 
            (1368*mckin**6*q_cut**5*sE)/mbkin**16 - (32*q_cut**6*sE)/mbkin**12 - 
            (288*mckin**4*q_cut**6*sE)/mbkin**16 + 9*sqB - (49*mckin**2*sqB)/
             mbkin**2 - (552*mckin**4*sqB)/mbkin**4 - (108*mckin**6*sqB)/
             mbkin**6 + (1415*mckin**8*sqB)/mbkin**8 - (69*mckin**10*sqB)/
             mbkin**10 - (666*mckin**12*sqB)/mbkin**12 + (2*mckin**14*sqB)/
             mbkin**14 + (18*mckin**16*sqB)/mbkin**16 - (24*q_cut*sqB)/mbkin**2 - 
            (3*mckin**2*q_cut*sqB)/mbkin**4 + (1294*mckin**4*q_cut*sqB)/mbkin**6 + 
            (3941*mckin**6*q_cut*sqB)/mbkin**8 + (3966*mckin**8*q_cut*sqB)/mbkin**10 + 
            (1085*mckin**10*q_cut*sqB)/mbkin**12 - (116*mckin**12*q_cut*sqB)/
             mbkin**14 - (63*mckin**14*q_cut*sqB)/mbkin**16 + (10*q_cut**2*sqB)/
             mbkin**4 + (72*mckin**2*q_cut**2*sqB)/mbkin**6 - (359*mckin**4*q_cut**2*sqB)/
             mbkin**8 - (699*mckin**6*q_cut**2*sqB)/mbkin**10 - 
            (83*mckin**8*q_cut**2*sqB)/mbkin**12 + (159*mckin**10*q_cut**2*sqB)/
             mbkin**14 + (60*mckin**12*q_cut**2*sqB)/mbkin**16 + (20*q_cut**3*sqB)/
             mbkin**6 + (92*mckin**2*q_cut**3*sqB)/mbkin**8 + (152*mckin**4*q_cut**3*sqB)/
             mbkin**10 - (26*mckin**6*q_cut**3*sqB)/mbkin**12 + 
            (124*mckin**8*q_cut**3*sqB)/mbkin**14 + (30*mckin**10*q_cut**3*sqB)/
             mbkin**16 - (15*q_cut**4*sqB)/mbkin**8 - (143*mckin**2*q_cut**4*sqB)/
             mbkin**10 - (285*mckin**4*q_cut**4*sqB)/mbkin**12 - 
            (271*mckin**6*q_cut**4*sqB)/mbkin**14 - (90*mckin**8*q_cut**4*sqB)/
             mbkin**16 - (4*q_cut**5*sqB)/mbkin**10 + (31*mckin**2*q_cut**5*sqB)/
             mbkin**12 + (102*mckin**4*q_cut**5*sqB)/mbkin**14 + 
            (57*mckin**6*q_cut**5*sqB)/mbkin**16 + (4*q_cut**6*sqB)/mbkin**12 - 
            (12*mckin**4*q_cut**6*sqB)/mbkin**16))*
         np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 - 
       (4320*mckin**8*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(24*mckin**2*muG - 
          (72*mckin**4*muG)/mbkin**2 - (12*mckin**2*muG**2)/mbkin**2 + 
          (36*mckin**4*muG**2)/mbkin**4 + (12*mckin**2*muG*mupi)/mbkin**2 - 
          (36*mckin**4*muG*mupi)/mbkin**4 + 32*(1 + mckin**2/mbkin**2)*rE + 
          4*(1 - (4*mckin**2)/mbkin**2 + (15*mckin**4)/mbkin**4)*rG + 
          24*mbkin*rhoD + (80*mckin**2*rhoD)/mbkin + (120*mckin**4*rhoD)/
           mbkin**3 + 12*sB + (88*mckin**2*sB)/mbkin**2 + (60*mckin**4*sB)/
           mbkin**4 - (64*mckin**2*sE)/mbkin**2 - 3*sqB - (10*mckin**2*sqB)/
           mbkin**2 - (15*mckin**4*sqB)/mbkin**4)*
         np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8)/
      (180*mbkin**2*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
          2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
        3) + 
     (api4*((18*mbkin**4*((-8*(mbkin - mckin)**2*(mbkin + mckin)*
              (3*mbkin + 3*mckin + 8*mbkin*mckin))/(9*mbkin**6) + 
            (16*(4*mbkin**3 - 4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*
              q_cut)/(9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*
           (3*(1 - (14*mckin**2)/mbkin**2 - (94*mckin**4)/mbkin**4 - 
              (14*mckin**6)/mbkin**6 + mckin**8/mbkin**8) + 
            (3*(mbkin**6 - 11*mbkin**4*mckin**2 - 11*mbkin**2*mckin**4 + mckin**6)*
              q_cut)/mbkin**8 + ((-7 + (2*mckin**2)/mbkin**2 - (7*mckin**4)/mbkin**4)*
              q_cut**2)/mbkin**4 - (7*(mbkin**2 + mckin**2)*q_cut**3)/mbkin**8 + 
            (8*q_cut**4)/mbkin**8)*((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/
                mbkin**2 - (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6) + 
             ((-3 + (14*mckin**2)/mbkin**2 + (26*mckin**4)/mbkin**4 + 
                (14*mckin**6)/mbkin**6 - (3*mckin**8)/mbkin**8)*q_cut)/mbkin**2 + 
             (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 2*mbkin**2*mckin**4 + mckin**6)*
               q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*q_cut**
                3)/mbkin**10 - (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + 
             q_cut**5/mbkin**10)**2 + ((-1 + mckin**2/mbkin**2)**2 - 
            (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
           ((64*mbkin*(-((-1 + mckin**2/mbkin**2)**4*(1 + mckin**2/mbkin**2)**2*
                 (351 - (5815*mckin**2)/mbkin**2 + (36915*mckin**4)/mbkin**4 - 
                  (109923*mckin**6)/mbkin**6 + (69257*mckin**8)/mbkin**8 - 
                  (22209*mckin**10)/mbkin**10 + (1077*mckin**12)/mbkin**12 + 
                  (107*mckin**14)/mbkin**14)) + (2*(-1 + mckin**2/mbkin**2)**2*
                 (1089 - (13483*mckin**2)/mbkin**2 + (53852*mckin**4)/mbkin**4 - 
                  (45772*mckin**6)/mbkin**6 - (288518*mckin**8)/mbkin**8 - 
                  (239134*mckin**10)/mbkin**10 + (43532*mckin**12)/mbkin**12 + 
                  (52492*mckin**14)/mbkin**14 - (52219*mckin**16)/mbkin**16 + 
                  (3977*mckin**18)/mbkin**18 + (344*mckin**20)/mbkin**20)*q_cut)/
                mbkin**2 + ((-4929 + (52403*mckin**2)/mbkin**2 - 
                  (176271*mckin**4)/mbkin**4 + (115749*mckin**6)/mbkin**6 + 
                  (605090*mckin**8)/mbkin**8 + (898826*mckin**10)/mbkin**10 + 
                  (657918*mckin**12)/mbkin**12 - (174202*mckin**14)/mbkin**14 - 
                  (222817*mckin**16)/mbkin**16 + (202947*mckin**18)/mbkin**18 - 
                  (17775*mckin**20)/mbkin**20 - (1579*mckin**22)/mbkin**22)*q_cut**2)/
                mbkin**4 + (2*(1659 - (10934*mckin**2)/mbkin**2 + 
                  (14621*mckin**4)/mbkin**4 + (22408*mckin**6)/mbkin**6 - 
                  (61810*mckin**8)/mbkin**8 + (4308*mckin**10)/mbkin**10 + 
                  (92322*mckin**12)/mbkin**12 - (26296*mckin**14)/mbkin**14 - 
                  (44537*mckin**16)/mbkin**16 + (7826*mckin**18)/mbkin**18 + 
                  (433*mckin**20)/mbkin**20)*q_cut**3)/mbkin**6 + 
               (4*(1353 - (9240*mckin**2)/mbkin**2 + (13738*mckin**4)/mbkin**4 + 
                  (9254*mckin**6)/mbkin**6 - (21795*mckin**8)/mbkin**8 - 
                  (9861*mckin**10)/mbkin**10 - (9696*mckin**12)/mbkin**12 - 
                  (33052*mckin**14)/mbkin**14 + (4192*mckin**16)/mbkin**16 + 
                  (675*mckin**18)/mbkin**18)*q_cut**4)/mbkin**8 - 
               (2*(6003 - (25356*mckin**2)/mbkin**2 + (10882*mckin**4)/mbkin**4 + 
                  (16908*mckin**6)/mbkin**6 - (30624*mckin**8)/mbkin**8 - 
                  (109320*mckin**10)/mbkin**10 - (93822*mckin**12)/mbkin**12 + 
                  (22600*mckin**14)/mbkin**14 + (2793*mckin**16)/mbkin**16)*q_cut**5)/
                mbkin**10 + (2*(3483 - (12003*mckin**2)/mbkin**2 - 
                  (3121*mckin**4)/mbkin**4 - (19139*mckin**6)/mbkin**6 - 
                  (74989*mckin**8)/mbkin**8 - (52799*mckin**10)/mbkin**10 + 
                  (11627*mckin**12)/mbkin**12 + (1677*mckin**14)/mbkin**14)*q_cut**6)/
                mbkin**12 + (4*(765 + (1989*mckin**2)/mbkin**2 + (3559*mckin**4)/
                   mbkin**4 + (13202*mckin**6)/mbkin**6 + (14387*mckin**8)/
                   mbkin**8 + (5569*mckin**10)/mbkin**10 + (417*mckin**12)/
                   mbkin**12)*q_cut**7)/mbkin**14 - ((6399 + (6327*mckin**2)/
                   mbkin**2 + (19022*mckin**4)/mbkin**4 + (37806*mckin**6)/
                   mbkin**6 + (34907*mckin**8)/mbkin**8 + (3747*mckin**10)/
                   mbkin**10)*q_cut**8)/mbkin**16 + (2*(1719 + (1825*mckin**2)/
                   mbkin**2 + (6279*mckin**4)/mbkin**4 + (8751*mckin**6)/
                   mbkin**6 + (1150*mckin**8)/mbkin**8)*q_cut**9)/mbkin**18 - 
               ((693 + (829*mckin**2)/mbkin**2 + (3367*mckin**4)/mbkin**4 + 
                  (607*mckin**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
               ((6 + (88*mckin**2)/mbkin**2 + (58*mckin**4)/mbkin**4)*q_cut**11)/
                mbkin**22 - (2*(3 + (7*mckin**2)/mbkin**2)*q_cut**12)/mbkin**24 + 
               (6*q_cut**13)/mbkin**26))/9 + 18*(2*mbkin**4*(3*(1 - (14*mckin**2)/
                   mbkin**2 - (94*mckin**4)/mbkin**4 - (14*mckin**6)/mbkin**6 + 
                  mckin**8/mbkin**8) + (3*(mbkin**6 - 11*mbkin**4*mckin**2 - 
                   11*mbkin**2*mckin**4 + mckin**6)*q_cut)/mbkin**8 + 
                ((-7 + (2*mckin**2)/mbkin**2 - (7*mckin**4)/mbkin**4)*q_cut**2)/
                 mbkin**4 - (7*(mbkin**2 + mckin**2)*q_cut**3)/mbkin**8 + 
                (8*q_cut**4)/mbkin**8)*((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/
                   mbkin**2 - (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6) + 
                ((-3 + (14*mckin**2)/mbkin**2 + (26*mckin**4)/mbkin**4 + 
                   (14*mckin**6)/mbkin**6 - (3*mckin**8)/mbkin**8)*q_cut)/mbkin**2 + 
                (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 2*mbkin**2*mckin**4 + 
                   mckin**6)*q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mckin**2 + 
                   mckin**4)*q_cut**3)/mbkin**10 - (3*(mbkin**2 + mckin**2)*q_cut**4)/
                 mbkin**10 + q_cut**5/mbkin**10)*((-4*(mbkin - mckin)**2*
                  (mbkin + mckin)*(27*mbkin**7 + 27*mbkin**6*mckin + 
                   72*mbkin**7*mckin - 21*mbkin**5*mckin**2 - 21*mbkin**4*
                    mckin**3 - 56*mbkin**5*mckin**3 - 93*mbkin**3*mckin**4 - 
                   93*mbkin**2*mckin**5 - 248*mbkin**3*mckin**5 + 15*mbkin*
                    mckin**6 + 15*mckin**7 + 40*mbkin*mckin**7))/(9*mbkin**12) + 
                (4*(51*mbkin**8 + 24*mbkin**9 + 112*mbkin**8*mckin + 72*mbkin**6*
                    mckin**2 - 224*mbkin**7*mckin**2 + 416*mbkin**6*mckin**3 - 
                   108*mbkin**4*mckin**4 - 624*mbkin**5*mckin**4 + 336*mbkin**4*
                    mckin**5 - 204*mbkin**2*mckin**6 - 448*mbkin**3*mckin**6 - 
                   96*mbkin**2*mckin**7 + 45*mckin**8 + 120*mbkin*mckin**8)*q_cut)/
                 (9*mbkin**12) - (8*(12*mbkin**6 + 16*mbkin**7 + 16*mbkin**6*
                    mckin - 6*mbkin**4*mckin**2 - 48*mbkin**5*mckin**2 + 
                   32*mbkin**4*mckin**3 - 33*mbkin**2*mckin**4 - 64*mbkin**3*
                    mckin**4 - 24*mbkin**2*mckin**5 + 15*mckin**6 + 40*mbkin*
                    mckin**6)*q_cut**2)/(9*mbkin**12) - (8*(6*mbkin**4 + 
                   24*mbkin**5 - 8*mbkin**4*mckin + 6*mbkin**2*mckin**2 + 
                   32*mbkin**3*mckin**2 - 16*mbkin**2*mckin**3 + 15*mckin**4 + 
                   40*mbkin*mckin**4)*q_cut**3)/(9*mbkin**12) + 
                (4*(9*mbkin**2 + 32*mbkin**3 - 8*mbkin**2*mckin + 15*mckin**2 + 
                   40*mbkin*mckin**2)*q_cut**4)/(3*mbkin**12) - (20*(3 + 8*mbkin)*
                  q_cut**5)/(9*mbkin**12)) + ((-1 + mckin**2/mbkin**2)**2*
                  (1 - (7*mckin**2)/mbkin**2 - (7*mckin**4)/mbkin**4 + 
                   mckin**6/mbkin**6) + ((-3 + (14*mckin**2)/mbkin**2 + 
                    (26*mckin**4)/mbkin**4 + (14*mckin**6)/mbkin**6 - (3*mckin**8)/
                     mbkin**8)*q_cut)/mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 
                    2*mbkin**2*mckin**4 + mckin**6)*q_cut**2)/mbkin**10 + 
                 (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*q_cut**3)/mbkin**10 - 
                 (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2*(
                (8*mbkin**2*(3 + 8*mbkin)*(3*(1 - (14*mckin**2)/mbkin**2 - 
                     (94*mckin**4)/mbkin**4 - (14*mckin**6)/mbkin**6 + mckin**8/
                      mbkin**8) + (3*(mbkin**6 - 11*mbkin**4*mckin**2 - 
                      11*mbkin**2*mckin**4 + mckin**6)*q_cut)/mbkin**8 + 
                   ((-7 + (2*mckin**2)/mbkin**2 - (7*mckin**4)/mbkin**4)*q_cut**2)/
                    mbkin**4 - (7*(mbkin**2 + mckin**2)*q_cut**3)/mbkin**8 + 
                   (8*q_cut**4)/mbkin**8))/9 + mbkin**4*((-8*(21*mbkin**8 + 
                     56*mbkin**8*mckin + 261*mbkin**6*mckin**2 - 56*mbkin**7*
                      mckin**2 + 752*mbkin**6*mckin**3 - 219*mbkin**4*mckin**4 - 
                     752*mbkin**5*mckin**4 + 168*mbkin**4*mckin**5 - 69*mbkin**2*
                      mckin**6 - 168*mbkin**3*mckin**6 - 16*mbkin**2*mckin**7 + 
                     6*mckin**8 + 16*mbkin*mckin**8))/(3*mbkin**10) - 
                  (16*(9*mbkin**6 + 2*mbkin**7 + 22*mbkin**6*mckin - 44*mbkin**5*
                      mckin**2 + 44*mbkin**4*mckin**3 - 27*mbkin**2*mckin**4 - 
                     66*mbkin**3*mckin**4 - 6*mbkin**2*mckin**5 + 3*mckin**6 + 
                     8*mbkin*mckin**6)*q_cut)/(3*mbkin**10) + (16*(12*mbkin**4 + 
                     28*mbkin**5 + 4*mbkin**4*mckin - 15*mbkin**2*mckin**2 - 
                     12*mbkin**3*mckin**2 - 28*mbkin**2*mckin**3 + 21*mckin**4 + 
                     56*mbkin*mckin**4)*q_cut**2)/(9*mbkin**10) + 
                  (56*(3*mbkin**2 + 12*mbkin**3 - 4*mbkin**2*mckin + 6*mckin**2 + 
                     16*mbkin*mckin**2)*q_cut**3)/(9*mbkin**10) - 
                  (128*(3 + 8*mbkin)*q_cut**4)/(9*mbkin**10))))) - 
          6*((-64*mckin**4*(mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)**2*
              (mbkin**2 + 2*mbkin*mckin + mckin**2 - q_cut)**2*(3*mbkin**4 + 8*
                mbkin**4*mckin - 6*mbkin**2*mckin**2 - 8*mbkin**3*mckin**2 - 8*
                mbkin**2*mckin**3 + 3*mckin**4 + 8*mbkin*mckin**4 - 3*mbkin**2*
                q_cut - 8*mbkin**2*mckin*q_cut - 3*mckin**2*q_cut - 8*mbkin*mckin**2*q_cut)*
              (21*mbkin**14 - 321*mbkin**12*mckin**2 + 297*mbkin**10*mckin**4 + 
               6483*mbkin**8*mckin**6 + 6483*mbkin**6*mckin**8 + 297*mbkin**4*
                mckin**10 - 321*mbkin**2*mckin**12 + 21*mckin**14 - 30*mbkin**12*
                q_cut + 156*mbkin**10*mckin**2*q_cut + 1302*mbkin**8*mckin**4*q_cut + 1464*
                mbkin**6*mckin**6*q_cut + 1302*mbkin**4*mckin**8*q_cut + 156*mbkin**2*
                mckin**10*q_cut - 30*mckin**12*q_cut - 41*mbkin**10*q_cut**2 + 411*mbkin**8*
                mckin**2*q_cut**2 + 1394*mbkin**6*mckin**4*q_cut**2 + 1394*mbkin**4*
                mckin**6*q_cut**2 + 411*mbkin**2*mckin**8*q_cut**2 - 41*mckin**10*q_cut**2 + 
               60*mbkin**8*q_cut**3 - 64*mbkin**6*mckin**2*q_cut**3 - 568*mbkin**4*
                mckin**4*q_cut**3 - 64*mbkin**2*mckin**6*q_cut**3 + 60*mckin**8*q_cut**3 + 35*
                mbkin**6*q_cut**4 - 139*mbkin**4*mckin**2*q_cut**4 - 139*mbkin**2*mckin**4*
                q_cut**4 + 35*mckin**6*q_cut**4 - 46*mbkin**4*q_cut**5 - 28*mbkin**2*mckin**2*
                q_cut**5 - 46*mckin**4*q_cut**5 - 15*mbkin**2*q_cut**6 - 15*mckin**2*q_cut**6 + 
               16*q_cut**7))/(mbkin**24*(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mckin**2 + q_cut + 
               mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))) + 
            ((-16*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)*np.sqrt(0j + (mbkin**4 - 
                   2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                    q_cut + q_cut**2)/mbkin**4)*(3*mbkin**4 + 8*mbkin**4*mckin - 
                 6*mbkin**2*mckin**2 - 8*mbkin**3*mckin**2 - 8*mbkin**2*mckin**3 + 
                 3*mckin**4 + 8*mbkin*mckin**4 - 8*mbkin**3*q_cut + 8*mbkin**2*mckin*
                  q_cut - 6*mckin**2*q_cut - 16*mbkin*mckin**2*q_cut + 3*q_cut**2 + 
                 8*mbkin*q_cut**2)*(21*mbkin**14 - 321*mbkin**12*mckin**2 + 
                 297*mbkin**10*mckin**4 + 6483*mbkin**8*mckin**6 + 6483*mbkin**6*
                  mckin**8 + 297*mbkin**4*mckin**10 - 321*mbkin**2*mckin**12 + 
                 21*mckin**14 - 30*mbkin**12*q_cut + 156*mbkin**10*mckin**2*q_cut + 
                 1302*mbkin**8*mckin**4*q_cut + 1464*mbkin**6*mckin**6*q_cut + 
                 1302*mbkin**4*mckin**8*q_cut + 156*mbkin**2*mckin**10*q_cut - 
                 30*mckin**12*q_cut - 41*mbkin**10*q_cut**2 + 411*mbkin**8*mckin**2*
                  q_cut**2 + 1394*mbkin**6*mckin**4*q_cut**2 + 1394*mbkin**4*mckin**6*
                  q_cut**2 + 411*mbkin**2*mckin**8*q_cut**2 - 41*mckin**10*q_cut**2 + 
                 60*mbkin**8*q_cut**3 - 64*mbkin**6*mckin**2*q_cut**3 - 568*mbkin**4*
                  mckin**4*q_cut**3 - 64*mbkin**2*mckin**6*q_cut**3 + 60*mckin**8*q_cut**3 + 
                 35*mbkin**6*q_cut**4 - 139*mbkin**4*mckin**2*q_cut**4 - 139*mbkin**2*
                  mckin**4*q_cut**4 + 35*mckin**6*q_cut**4 - 46*mbkin**4*q_cut**5 - 
                 28*mbkin**2*mckin**2*q_cut**5 - 46*mckin**4*q_cut**5 - 15*mbkin**2*
                  q_cut**6 - 15*mckin**2*q_cut**6 + 16*q_cut**7))/mbkin**24 + 
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(36*mckin**4*
                 ((-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 3*mckin + 
                     8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 
                     4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/
                   (9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*
                 (3*(-1 + mckin**2/mbkin**2)**2*(7 - (107*mckin**2)/mbkin**2 + 
                    (99*mckin**4)/mbkin**4 + (2161*mckin**6)/mbkin**6 + 
                    (2161*mckin**8)/mbkin**8 + (99*mckin**10)/mbkin**10 - 
                    (107*mckin**12)/mbkin**12 + (7*mckin**14)/mbkin**14) - 
                  (24*(3 - (34*mckin**2)/mbkin**2 - (42*mckin**4)/mbkin**4 + 
                     (606*mckin**6)/mbkin**6 + (1094*mckin**8)/mbkin**8 + 
                     (606*mckin**10)/mbkin**10 - (42*mckin**12)/mbkin**12 - 
                     (34*mckin**14)/mbkin**14 + (3*mckin**16)/mbkin**16)*q_cut)/
                   mbkin**2 + (8*(5 - (10*mckin**2)/mbkin**2 - (261*mckin**4)/
                      mbkin**4 - (4*mckin**6)/mbkin**6 - (4*mckin**8)/mbkin**8 - 
                     (261*mckin**10)/mbkin**10 - (10*mckin**12)/mbkin**12 + 
                     (5*mckin**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
                  (16*(7 - (48*mckin**2)/mbkin**2 - (168*mckin**4)/mbkin**4 - 
                     (194*mckin**6)/mbkin**6 - (168*mckin**8)/mbkin**8 - 
                     (48*mckin**10)/mbkin**10 + (7*mckin**12)/mbkin**12)*q_cut**3)/
                   mbkin**6 - (6*(21 - (35*mckin**2)/mbkin**2 - (472*mckin**4)/
                      mbkin**4 - (472*mckin**6)/mbkin**6 - (35*mckin**8)/
                      mbkin**8 + (21*mckin**10)/mbkin**10)*q_cut**4)/mbkin**8 - 
                  (8*(7 - (26*mckin**2)/mbkin**2 + (6*mckin**4)/mbkin**4 - 
                     (26*mckin**6)/mbkin**6 + (7*mckin**8)/mbkin**8)*q_cut**5)/
                   mbkin**10 + (8*(14 + (3*mckin**2)/mbkin**2 + (3*mckin**4)/
                      mbkin**4 + (14*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 - 
                  (47*(mbkin**2 + mckin**2)*q_cut**8)/mbkin**18 + (16*q_cut**9)/
                   mbkin**18) + ((-1 + mckin**2/mbkin**2)**2 - 
                  (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
                 ((-64*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(21 - (236*mckin**2)/
                         mbkin**2 + (280*mckin**4)/mbkin**4 + (5400*mckin**6)/
                         mbkin**6 - (43634*mckin**8)/mbkin**8 - (31160*mckin**10)/
                         mbkin**10 + (14712*mckin**12)/mbkin**12 - 
                        (4424*mckin**14)/mbkin**14 - (1619*mckin**16)/mbkin**16 + 
                        (180*mckin**18)/mbkin**18)) + (2*(51 - (508*mckin**2)/
                         mbkin**2 + (671*mckin**4)/mbkin**4 + (7066*mckin**6)/
                         mbkin**6 - (40432*mckin**8)/mbkin**8 - (66010*mckin**10)/
                         mbkin**10 - (36356*mckin**12)/mbkin**12 + 
                        (22310*mckin**14)/mbkin**14 - (4915*mckin**16)/mbkin**
                          16 - (3242*mckin**18)/mbkin**18 + (405*mckin**20)/
                         mbkin**20)*q_cut)/mbkin**2 - (2*(75 - (535*mckin**2)/
                         mbkin**2 + (234*mckin**4)/mbkin**4 + (3612*mckin**6)/
                         mbkin**6 - (2692*mckin**8)/mbkin**8 + (6882*mckin**10)/
                         mbkin**10 + (9870*mckin**12)/mbkin**12 - 
                        (5092*mckin**14)/mbkin**14 - (2799*mckin**16)/mbkin**16 + 
                        (525*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
                     (2*(-21 + (124*mckin**2)/mbkin**2 + (650*mckin**4)/
                         mbkin**4 - (4312*mckin**6)/mbkin**6 + (1928*mckin**8)/
                         mbkin**8 + (728*mckin**10)/mbkin**10 + (7154*mckin**12)/
                         mbkin**12 + (1396*mckin**14)/mbkin**14 - (255*mckin**16)/
                         mbkin**16)*q_cut**3)/mbkin**6 + (2*(168 - (595*mckin**2)/
                         mbkin**2 - (1396*mckin**4)/mbkin**4 + (3013*mckin**6)/
                         mbkin**6 - (1930*mckin**8)/mbkin**8 - (13117*mckin**10)/
                         mbkin**10 - (3590*mckin**12)/mbkin**12 + 
                        (1215*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
                     (2*(-147 + (436*mckin**2)/mbkin**2 + (723*mckin**4)/
                         mbkin**4 + (726*mckin**6)/mbkin**6 + (6157*mckin**8)/
                         mbkin**8 + (2478*mckin**10)/mbkin**10 - (885*mckin**12)/
                         mbkin**12)*q_cut**5)/mbkin**10 - (2*(21 + (227*mckin**2)/
                         mbkin**2 + (112*mckin**4)/mbkin**4 + (1090*mckin**6)/
                         mbkin**6 + (1463*mckin**8)/mbkin**8 + (255*mckin**10)/
                         mbkin**10)*q_cut**6)/mbkin**12 + (2*(105 + (140*mckin**2)/
                         mbkin**2 + (248*mckin**4)/mbkin**4 + (856*mckin**6)/
                         mbkin**6 + (675*mckin**8)/mbkin**8)*q_cut**7)/mbkin**14 - 
                     ((123 + (88*mckin**2)/mbkin**2 + (447*mckin**4)/mbkin**4 + 
                        (690*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
                     (24*(mbkin**4 + 5*mckin**4)*q_cut**9)/mbkin**22))/9 + 
                  36*((8*mckin**2*(3 + 8*mckin)*(3*(-1 + mckin**2/mbkin**2)**2*
                        (7 - (107*mckin**2)/mbkin**2 + (99*mckin**4)/mbkin**4 + 
                         (2161*mckin**6)/mbkin**6 + (2161*mckin**8)/mbkin**8 + 
                         (99*mckin**10)/mbkin**10 - (107*mckin**12)/mbkin**12 + 
                         (7*mckin**14)/mbkin**14) - (24*(3 - (34*mckin**2)/
                          mbkin**2 - (42*mckin**4)/mbkin**4 + (606*mckin**6)/
                          mbkin**6 + (1094*mckin**8)/mbkin**8 + (606*mckin**10)/
                          mbkin**10 - (42*mckin**12)/mbkin**12 - (34*mckin**14)/
                          mbkin**14 + (3*mckin**16)/mbkin**16)*q_cut)/mbkin**2 + 
                       (8*(5 - (10*mckin**2)/mbkin**2 - (261*mckin**4)/mbkin**4 - 
                          (4*mckin**6)/mbkin**6 - (4*mckin**8)/mbkin**8 - 
                          (261*mckin**10)/mbkin**10 - (10*mckin**12)/mbkin**12 + 
                          (5*mckin**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
                       (16*(7 - (48*mckin**2)/mbkin**2 - (168*mckin**4)/mbkin**
                          4 - (194*mckin**6)/mbkin**6 - (168*mckin**8)/mbkin**8 - 
                          (48*mckin**10)/mbkin**10 + (7*mckin**12)/mbkin**12)*
                         q_cut**3)/mbkin**6 - (6*(21 - (35*mckin**2)/mbkin**2 - 
                          (472*mckin**4)/mbkin**4 - (472*mckin**6)/mbkin**6 - 
                          (35*mckin**8)/mbkin**8 + (21*mckin**10)/mbkin**10)*
                         q_cut**4)/mbkin**8 - (8*(7 - (26*mckin**2)/mbkin**2 + 
                          (6*mckin**4)/mbkin**4 - (26*mckin**6)/mbkin**6 + 
                          (7*mckin**8)/mbkin**8)*q_cut**5)/mbkin**10 + 
                       (8*(14 + (3*mckin**2)/mbkin**2 + (3*mckin**4)/mbkin**4 + 
                          (14*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 - 
                       (47*(mbkin**2 + mckin**2)*q_cut**8)/mbkin**18 + (16*q_cut**9)/
                        mbkin**18))/9 + mckin**4*((-4*(mbkin - mckin)**2*
                        (mbkin + mckin)*(363*mbkin**15 + 363*mbkin**14*mckin + 
                         968*mbkin**15*mckin - 1557*mbkin**13*mckin**2 - 
                         1557*mbkin**12*mckin**3 - 4152*mbkin**13*mckin**3 - 
                         18261*mbkin**11*mckin**4 - 18261*mbkin**10*mckin**5 - 
                         48696*mbkin**11*mckin**5 + 6483*mbkin**9*mckin**6 + 
                         6483*mbkin**8*mckin**7 + 17288*mbkin**9*mckin**7 + 
                         37413*mbkin**7*mckin**8 + 37413*mbkin**6*mckin**9 + 
                         99768*mbkin**7*mckin**9 + 4005*mbkin**5*mckin**10 + 
                         4005*mbkin**4*mckin**11 + 10680*mbkin**5*mckin**11 - 
                         2715*mbkin**3*mckin**12 - 2715*mbkin**2*mckin**13 - 
                         7240*mbkin**3*mckin**13 + 189*mbkin*mckin**14 + 
                         189*mckin**15 + 504*mbkin*mckin**15))/(3*mbkin**20) + 
                      (32*(111*mbkin**16 + 24*mbkin**17 + 272*mbkin**16*mckin + 
                         48*mbkin**14*mckin**2 - 544*mbkin**15*mckin**2 + 
                         672*mbkin**14*mckin**3 - 5832*mbkin**12*mckin**4 - 
                         1008*mbkin**13*mckin**4 - 14544*mbkin**12*mckin**5 - 
                         5856*mbkin**10*mckin**6 + 19392*mbkin**11*mckin**6 - 
                         35008*mbkin**10*mckin**7 + 7320*mbkin**8*mckin**8 + 
                         43760*mbkin**9*mckin**8 - 24240*mbkin**8*mckin**9 + 
                         11664*mbkin**6*mckin**10 + 29088*mbkin**7*mckin**10 + 
                         2016*mbkin**6*mckin**11 - 168*mbkin**4*mckin**12 - 
                         2352*mbkin**5*mckin**12 + 1904*mbkin**4*mckin**13 - 
                         888*mbkin**2*mckin**14 - 2176*mbkin**3*mckin**14 - 
                         192*mbkin**2*mckin**15 + 81*mckin**16 + 216*mbkin*
                          mckin**16)*q_cut)/(3*mbkin**20) - (32*(60*mbkin**14 + 
                         80*mbkin**15 + 80*mbkin**14*mckin + 1476*mbkin**12*
                          mckin**2 - 240*mbkin**13*mckin**2 + 4176*mbkin**12*
                          mckin**3 - 3096*mbkin**10*mckin**4 - 8352*mbkin**11*
                          mckin**4 + 96*mbkin**10*mckin**5 - 12*mbkin**8*mckin**
                          6 - 160*mbkin**9*mckin**6 + 128*mbkin**8*mckin**7 + 
                         3843*mbkin**6*mckin**8 - 192*mbkin**7*mckin**8 + 
                         10440*mbkin**6*mckin**9 - 5301*mbkin**4*mckin**10 - 
                         14616*mbkin**5*mckin**10 + 480*mbkin**4*mckin**11 - 
                         345*mbkin**2*mckin**12 - 640*mbkin**3*mckin**12 - 
                         280*mbkin**2*mckin**13 + 135*mckin**14 + 360*mbkin*
                          mckin**14)*q_cut**2)/(9*mbkin**20) - (64*(69*mbkin**12 + 
                         56*mbkin**13 + 128*mbkin**12*mckin + 144*mbkin**10*
                          mckin**2 - 512*mbkin**11*mckin**2 + 896*mbkin**10*
                          mckin**3 - 258*mbkin**8*mckin**4 - 2240*mbkin**9*
                          mckin**4 + 1552*mbkin**8*mckin**5 - 492*mbkin**6*
                          mckin**6 - 3104*mbkin**7*mckin**6 + 1792*mbkin**6*
                          mckin**7 - 936*mbkin**4*mckin**8 - 3136*mbkin**5*
                          mckin**8 + 640*mbkin**4*mckin**9 - 426*mbkin**2*
                          mckin**10 - 1024*mbkin**3*mckin**10 - 112*mbkin**2*
                          mckin**11 + 63*mckin**12 + 168*mbkin*mckin**12)*q_cut**3)/
                       (3*mbkin**20) + (8*(357*mbkin**10 + 672*mbkin**11 + 
                         280*mbkin**10*mckin + 2307*mbkin**8*mckin**2 - 
                         1400*mbkin**9*mckin**2 + 7552*mbkin**8*mckin**3 - 
                         4248*mbkin**6*mckin**4 - 22656*mbkin**7*mckin**4 + 
                         11328*mbkin**6*mckin**5 - 9492*mbkin**4*mckin**6 - 
                         26432*mbkin**5*mckin**6 + 1120*mbkin**4*mckin**7 - 
                         1155*mbkin**2*mckin**8 - 2240*mbkin**3*mckin**8 - 
                         840*mbkin**2*mckin**9 + 567*mckin**10 + 1512*mbkin*
                          mckin**10)*q_cut**4)/(3*mbkin**20) + (32*(183*mbkin**8 + 
                         280*mbkin**9 + 208*mbkin**8*mckin - 504*mbkin**6*
                          mckin**2 - 1248*mbkin**7*mckin**2 - 96*mbkin**6*
                          mckin**3 + 360*mbkin**4*mckin**4 + 336*mbkin**5*
                          mckin**4 + 624*mbkin**4*mckin**5 - 708*mbkin**2*
                          mckin**6 - 1664*mbkin**3*mckin**6 - 224*mbkin**2*
                          mckin**7 + 189*mckin**8 + 504*mbkin*mckin**8)*q_cut**5)/
                       (9*mbkin**20) - (32*(81*mbkin**6 + 224*mbkin**7 - 
                         8*mbkin**6*mckin + 15*mbkin**4*mckin**2 + 56*mbkin**5*
                          mckin**2 - 16*mbkin**4*mckin**3 - 18*mbkin**2*mckin**4 + 
                         64*mbkin**3*mckin**4 - 112*mbkin**2*mckin**5 + 
                         126*mckin**6 + 336*mbkin*mckin**6)*q_cut**6)/(3*mbkin**
                         20) + (188*(21*mbkin**2 + 64*mbkin**3 - 8*mbkin**2*
                          mckin + 27*mckin**2 + 72*mbkin*mckin**2)*q_cut**8)/
                       (9*mbkin**20) - (64*(3 + 8*mbkin)*q_cut**9)/mbkin**20)))))*
             np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)))) - 
          144*((64*mckin**8*(mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)*
              (mbkin**2 + 2*mbkin*mckin + mckin**2 - q_cut)*(33*mbkin**8 - 222*
                mbkin**6*mckin**2 - 702*mbkin**4*mckin**4 - 222*mbkin**2*mckin**6 + 
               33*mckin**8 - 27*mbkin**6*q_cut - 63*mbkin**4*mckin**2*q_cut - 63*
                mbkin**2*mckin**4*q_cut - 27*mckin**6*q_cut - 37*mbkin**4*q_cut**2 - 58*
                mbkin**2*mckin**2*q_cut**2 - 37*mckin**4*q_cut**2 + 23*mbkin**2*q_cut**3 + 23*
                mckin**2*q_cut**3 + 8*q_cut**4)*(3*mbkin**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                  mbkin**4) + 8*mbkin**4*mckin*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                  mbkin**4) - 6*mbkin**2*mckin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                  mbkin**4) - 8*mbkin**3*mckin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                  mbkin**4) - 8*mbkin**2*mckin**3*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                  mbkin**4) + 3*mckin**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                   mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 8*
                mbkin*mckin**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mbkin**2*
                q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                    q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**2*mckin*q_cut*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mckin**2*q_cut*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin*mckin**2*q_cut*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))*np.log((mbkin**2 + mckin**2 - 
                 q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                     2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/
                (mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                      mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                    mbkin**4))))/(mbkin**18*(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mckin**2 + q_cut + 
               mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))) + 
            ((-16*mckin**6*(3*mbkin**2 + 8*mbkin**2*mckin - 3*mckin**2 - 
                 8*mbkin*mckin**2)*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*(33*mbkin**8 - 
                 222*mbkin**6*mckin**2 - 702*mbkin**4*mckin**4 - 222*mbkin**2*
                  mckin**6 + 33*mckin**8 - 27*mbkin**6*q_cut - 63*mbkin**4*mckin**2*
                  q_cut - 63*mbkin**2*mckin**4*q_cut - 27*mckin**6*q_cut - 37*mbkin**4*
                  q_cut**2 - 58*mbkin**2*mckin**2*q_cut**2 - 37*mckin**4*q_cut**2 + 
                 23*mbkin**2*q_cut**3 + 23*mckin**2*q_cut**3 + 8*q_cut**4))/mbkin**22 + 
              (mckin**4*(-18*mckin**4*((-8*(mbkin - mckin)**2*(mbkin + mckin)*
                     (3*mbkin + 3*mckin + 8*mbkin*mckin))/(9*mbkin**6) + 
                   (16*(4*mbkin**3 - 4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*
                       mckin**2)*q_cut)/(9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/
                    (9*mbkin**6))*(3*(-1 + mckin**2/mbkin**2)**2*(11 - 
                     (74*mckin**2)/mbkin**2 - (234*mckin**4)/mbkin**4 - 
                     (74*mckin**6)/mbkin**6 + (11*mckin**8)/mbkin**8) + 
                   ((-93 + (369*mckin**2)/mbkin**2 + (1884*mckin**4)/mbkin**4 + 
                      (1884*mckin**6)/mbkin**6 + (369*mckin**8)/mbkin**8 - 
                      (93*mckin**10)/mbkin**10)*q_cut)/mbkin**2 + 
                   ((50 - (26*mckin**2)/mbkin**2 - (408*mckin**4)/mbkin**4 - 
                      (26*mckin**6)/mbkin**6 + (50*mckin**8)/mbkin**8)*q_cut**2)/
                    mbkin**4 + (2*(35 + (52*mckin**2)/mbkin**2 + (52*mckin**4)/
                       mbkin**4 + (35*mckin**6)/mbkin**6)*q_cut**3)/mbkin**6 - 
                   ((75 + (166*mckin**2)/mbkin**2 + (75*mckin**4)/mbkin**4)*q_cut**4)/
                    mbkin**8 + (7*(mbkin**2 + mckin**2)*q_cut**5)/mbkin**12 + 
                   (8*q_cut**6)/mbkin**12) + ((-1 + mckin**2/mbkin**2)**2 - 
                   (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
                  ((64*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(3 - (17*mckin**2)/
                          mbkin**2 + (443*mckin**4)/mbkin**4 + (1923*mckin**6)/
                          mbkin**6 + (188*mckin**8)/mbkin**8 - (110*mckin**10)/
                          mbkin**10 + (90*mckin**12)/mbkin**12)) + 
                      ((18 - (129*mckin**2)/mbkin**2 + (1036*mckin**4)/mbkin**4 + 
                         (4331*mckin**6)/mbkin**6 + (4596*mckin**8)/mbkin**8 + 
                         (443*mckin**10)/mbkin**10 - (530*mckin**12)/mbkin**12 + 
                         (315*mckin**14)/mbkin**14)*q_cut)/mbkin**2 + 
                      ((-30 + (136*mckin**2)/mbkin**2 - (287*mckin**4)/mbkin**4 - 
                         (779*mckin**6)/mbkin**6 + (165*mckin**8)/mbkin**8 + 
                         (255*mckin**10)/mbkin**10 - (300*mckin**12)/mbkin**12)*
                        q_cut**2)/mbkin**4 - (2*mckin**2*(-8 - (38*mckin**2)/
                          mbkin**2 + (95*mckin**4)/mbkin**4 + (100*mckin**6)/
                          mbkin**6 + (75*mckin**8)/mbkin**8)*q_cut**3)/mbkin**8 + 
                      ((45 - (39*mckin**2)/mbkin**2 - (45*mckin**4)/mbkin**4 + 
                         (305*mckin**6)/mbkin**6 + (450*mckin**8)/mbkin**8)*q_cut**4)/
                       mbkin**8 - ((42 + (7*mckin**2)/mbkin**2 + (120*mckin**4)/
                          mbkin**4 + (285*mckin**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
                      (12*(mbkin**4 + 5*mckin**4)*q_cut**6)/mbkin**16))/9 - 
                   18*((8*mckin**2*(3 + 8*mckin)*(3*(-1 + mckin**2/mbkin**2)**2*
                         (11 - (74*mckin**2)/mbkin**2 - (234*mckin**4)/mbkin**4 - 
                          (74*mckin**6)/mbkin**6 + (11*mckin**8)/mbkin**8) + 
                        ((-93 + (369*mckin**2)/mbkin**2 + (1884*mckin**4)/
                          mbkin**4 + (1884*mckin**6)/mbkin**6 + (369*mckin**8)/
                          mbkin**8 - (93*mckin**10)/mbkin**10)*q_cut)/mbkin**2 + 
                        ((50 - (26*mckin**2)/mbkin**2 - (408*mckin**4)/mbkin**4 - 
                          (26*mckin**6)/mbkin**6 + (50*mckin**8)/mbkin**8)*q_cut**2)/
                         mbkin**4 + (2*(35 + (52*mckin**2)/mbkin**2 + 
                          (52*mckin**4)/mbkin**4 + (35*mckin**6)/mbkin**6)*q_cut**3)/
                         mbkin**6 - ((75 + (166*mckin**2)/mbkin**2 + 
                          (75*mckin**4)/mbkin**4)*q_cut**4)/mbkin**8 + 
                        (7*(mbkin**2 + mckin**2)*q_cut**5)/mbkin**12 + (8*q_cut**6)/
                         mbkin**12))/9 + mckin**4*((-8*(mbkin - mckin)**2*
                         (mbkin + mckin)*(48*mbkin**9 + 48*mbkin**8*mckin + 
                          128*mbkin**9*mckin + 123*mbkin**7*mckin**2 + 
                          123*mbkin**6*mckin**3 + 328*mbkin**7*mckin**3 - 
                          357*mbkin**5*mckin**4 - 357*mbkin**4*mckin**5 - 
                          952*mbkin**5*mckin**5 - 207*mbkin**3*mckin**6 - 
                          207*mbkin**2*mckin**7 - 552*mbkin**3*mckin**7 + 
                          33*mbkin*mckin**8 + 33*mckin**9 + 88*mbkin*mckin**9))/
                        mbkin**14 + (8*(231*mbkin**10 + 124*mbkin**11 + 
                          492*mbkin**10*mckin + 1515*mbkin**8*mckin**2 - 
                          984*mbkin**9*mckin**2 + 5024*mbkin**8*mckin**3 - 
                          7536*mbkin**7*mckin**4 + 7536*mbkin**6*mckin**5 - 
                          3030*mbkin**4*mckin**6 - 10048*mbkin**5*mckin**6 + 
                          1968*mbkin**4*mckin**7 - 1155*mbkin**2*mckin**8 - 
                          2460*mbkin**3*mckin**8 - 620*mbkin**2*mckin**9 + 
                          279*mckin**10 + 744*mbkin*mckin**10)*q_cut)/(3*mbkin**
                          14) - (8*(189*mbkin**8 + 400*mbkin**9 + 104*mbkin**8*
                          mckin + 1107*mbkin**6*mckin**2 - 312*mbkin**7*mckin**
                          2 + 3264*mbkin**6*mckin**3 - 2331*mbkin**4*mckin**4 - 
                          6528*mbkin**5*mckin**4 + 312*mbkin**4*mckin**5 - 
                          495*mbkin**2*mckin**6 - 520*mbkin**3*mckin**6 - 
                          800*mbkin**2*mckin**7 + 450*mckin**8 + 1200*mbkin*
                          mckin**8)*q_cut**2)/(9*mbkin**14) - (8*(159*mbkin**6 + 
                          840*mbkin**7 - 416*mbkin**6*mckin + 312*mbkin**4*
                          mckin**2 + 1664*mbkin**5*mckin**2 - 832*mbkin**4*
                          mckin**3 + 465*mbkin**2*mckin**4 + 2080*mbkin**3*
                          mckin**4 - 840*mbkin**2*mckin**5 + 630*mckin**6 + 
                          1680*mbkin*mckin**6)*q_cut**3)/(9*mbkin**14) + 
                       (8*(201*mbkin**4 + 1200*mbkin**5 - 664*mbkin**4*mckin + 
                          1020*mbkin**2*mckin**2 + 3320*mbkin**3*mckin**2 - 
                          600*mbkin**2*mckin**3 + 675*mckin**4 + 1800*mbkin*
                          mckin**4)*q_cut**4)/(9*mbkin**14) - (56*(6*mbkin**2 + 
                          20*mbkin**3 - 4*mbkin**2*mckin + 9*mckin**2 + 24*mbkin*
                          mckin**2)*q_cut**5)/(9*mbkin**14) - (64*(3 + 8*mbkin)*
                         q_cut**6)/(3*mbkin**14))))))/mbkin**4)*
             np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                      mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                    mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                  np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                      q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2) - 
          4320*((-576*mckin**12*(mbkin**2 + mckin**2)*(mbkin**2 - 2*mbkin*mckin + 
               mckin**2 - q_cut)*(mbkin**2 + 2*mbkin*mckin + mckin**2 - q_cut)*
              (3*mbkin**4 + 8*mbkin**4*mckin - 6*mbkin**2*mckin**2 - 8*mbkin**3*
                mckin**2 - 8*mbkin**2*mckin**3 + 3*mckin**4 + 8*mbkin*mckin**4 - 3*
                mbkin**2*q_cut - 8*mbkin**2*mckin*q_cut - 3*mckin**2*q_cut - 8*mbkin*
                mckin**2*q_cut)*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                  mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                      mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                     mbkin**4)))**2)/(mbkin**16*(mbkin**2 + mckin**2 - q_cut + 
               mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))*
              (-mbkin**2 - mckin**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                  mbkin**4))) + ((-16*mckin**8*(12*mbkin**5 + 40*mbkin**3*
                  mckin**2 - 162*mbkin**4*mckin**2 - 432*mbkin**4*mckin**3 + 
                 60*mbkin*mckin**4 - 243*mbkin**2*mckin**4 - 648*mbkin**2*
                  mckin**5 + 81*mckin**6 + 216*mbkin*mckin**6)*
                ((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2))/(9*mbkin**12) - 
              (48*mckin**10*(mbkin**2 + mckin**2)*((mbkin**4 - 2*mbkin**2*
                    mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                  mbkin**4)**(3/2)*(-12*mbkin**6 - 32*mbkin**6*mckin + 
                 45*mbkin**4*mckin**2 + 32*mbkin**5*mckin**2 + 88*mbkin**4*
                  mckin**3 - 54*mbkin**2*mckin**4 - 88*mbkin**3*mckin**4 - 
                 56*mbkin**2*mckin**5 + 21*mckin**6 + 56*mbkin*mckin**6 + 
                 24*mbkin**4*q_cut + 64*mbkin**4*mckin*q_cut - 88*mbkin**3*mckin**2*
                  q_cut + 88*mbkin**2*mckin**3*q_cut - 42*mckin**4*q_cut - 112*mbkin*
                  mckin**4*q_cut - 12*mbkin**2*q_cut**2 - 32*mbkin**2*mckin*q_cut**2 + 
                 21*mckin**2*q_cut**2 + 56*mbkin*mckin**2*q_cut**2))/(mbkin**12*
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)))*np.log((mbkin**2 + mckin**2 - q_cut - 
                 mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                     2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/
                (mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                      mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                    mbkin**4)))**3))/(mbkin**2*
          ((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
             2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
               7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
               mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
            (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*
                  np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                      q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                 mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                     mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                    mbkin**4))))/mbkin**4)**3) + 
        ((18*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mckin**2 - 
              7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
              mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)**2*(3*mbkin**8 - 
             42*mbkin**6*mckin**2 - 282*mbkin**4*mckin**4 - 42*mbkin**2*mckin**6 + 
             3*mckin**8 + 3*mbkin**6*q_cut - 33*mbkin**4*mckin**2*q_cut - 
             33*mbkin**2*mckin**4*q_cut + 3*mckin**6*q_cut - 7*mbkin**4*q_cut**2 + 
             2*mbkin**2*mckin**2*q_cut**2 - 7*mckin**4*q_cut**2 - 7*mbkin**2*q_cut**3 - 
             7*mckin**2*q_cut**3 + 8*q_cut**4))/mbkin**28 - 
          (216*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
              2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*
            np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)*(21*mbkin**14 - 321*mbkin**12*
              mckin**2 + 297*mbkin**10*mckin**4 + 6483*mbkin**8*mckin**6 + 
             6483*mbkin**6*mckin**8 + 297*mbkin**4*mckin**10 - 
             321*mbkin**2*mckin**12 + 21*mckin**14 - 30*mbkin**12*q_cut + 
             156*mbkin**10*mckin**2*q_cut + 1302*mbkin**8*mckin**4*q_cut + 
             1464*mbkin**6*mckin**6*q_cut + 1302*mbkin**4*mckin**8*q_cut + 
             156*mbkin**2*mckin**10*q_cut - 30*mckin**12*q_cut - 41*mbkin**10*q_cut**2 + 
             411*mbkin**8*mckin**2*q_cut**2 + 1394*mbkin**6*mckin**4*q_cut**2 + 
             1394*mbkin**4*mckin**6*q_cut**2 + 411*mbkin**2*mckin**8*q_cut**2 - 
             41*mckin**10*q_cut**2 + 60*mbkin**8*q_cut**3 - 64*mbkin**6*mckin**2*q_cut**3 - 
             568*mbkin**4*mckin**4*q_cut**3 - 64*mbkin**2*mckin**6*q_cut**3 + 
             60*mckin**8*q_cut**3 + 35*mbkin**6*q_cut**4 - 139*mbkin**4*mckin**2*q_cut**4 - 
             139*mbkin**2*mckin**4*q_cut**4 + 35*mckin**6*q_cut**4 - 46*mbkin**4*q_cut**5 - 
             28*mbkin**2*mckin**2*q_cut**5 - 46*mckin**4*q_cut**5 - 15*mbkin**2*q_cut**6 - 
             15*mckin**2*q_cut**6 + 16*q_cut**7)*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
               mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**22 + 
          (2592*mckin**8*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
              2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*(33*mbkin**8 - 
             222*mbkin**6*mckin**2 - 702*mbkin**4*mckin**4 - 222*mbkin**2*
              mckin**6 + 33*mckin**8 - 27*mbkin**6*q_cut - 63*mbkin**4*mckin**2*q_cut - 
             63*mbkin**2*mckin**4*q_cut - 27*mckin**6*q_cut - 37*mbkin**4*q_cut**2 - 
             58*mbkin**2*mckin**2*q_cut**2 - 37*mckin**4*q_cut**2 + 23*mbkin**2*q_cut**3 + 
             23*mckin**2*q_cut**3 + 8*q_cut**4)*np.log((mbkin**2 + mckin**2 - q_cut - 
                mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**
             2)/mbkin**20 - (466560*mckin**12*(mbkin**2 + mckin**2)*
            ((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**10)*
         (((-4*(3 + 8*mbkin))/(9*mbkin**4*((mbkin**4 - 2*mbkin**2*mckin**2 + 
                 mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/
                2)) - (3*((-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 
                  3*mckin + 8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 
                  4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/
                (9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6)))/
             (2*mbkin**2*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
              ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
                mbkin**4 + q_cut**2/mbkin**4)))/
           ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*
                 mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
             (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                      2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - 
                  q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/
              mbkin**4)**3 - (3*((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(
                (-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 3*mckin + 
                   8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 4*mbkin**2*
                    mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/(9*mbkin**6) - 
                (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*(mbkin**6 - 7*mbkin**4*
                 mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/
              (2*mbkin**6*((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*
                  q_cut)/mbkin**4 + q_cut**2/mbkin**4)) - 
             (4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(21*mbkin**6 + 56*mbkin**6*
                 mckin + 21*mbkin**4*mckin**2 - 56*mbkin**5*mckin**2 + 
                112*mbkin**4*mckin**3 - 51*mbkin**2*mckin**4 - 112*mbkin**3*
                 mckin**4 - 24*mbkin**2*mckin**5 + 9*mckin**6 + 24*mbkin*
                 mckin**6 - 3*mbkin**4*q_cut - 8*mbkin**5*q_cut + 6*mbkin**2*mckin**2*
                 q_cut + 16*mbkin**2*mckin**3*q_cut - 9*mckin**4*q_cut - 24*mbkin*mckin**4*
                 q_cut - 3*mbkin**2*q_cut**2 - 16*mbkin**3*q_cut**2 + 8*mbkin**2*mckin*
                 q_cut**2 - 9*mckin**2*q_cut**2 - 24*mbkin*mckin**2*q_cut**2 + 9*q_cut**3 + 
                24*mbkin*q_cut**3))/(9*mbkin**8) - 12*((-16*mckin**4*
                 (3*mbkin**6*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 
                  8*mbkin**6*mckin*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                      mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                     mbkin**4) - 6*mbkin**4*mckin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4) - 8*mbkin**5*mckin**2*np.sqrt(0j + (mbkin**4 - 
                      2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                       q_cut + q_cut**2)/mbkin**4) - 8*mbkin**4*mckin**3*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                      2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 3*mbkin**2*mckin**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 8*mbkin**3*mckin**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mbkin**4*q_cut*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**4*mckin*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mbkin**2*mckin**2*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**3*mckin**2*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))/(9*mbkin**4*
                 (mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)*(mbkin**2 + 
                  2*mbkin*mckin + mckin**2 - q_cut)*(mbkin**2 + mckin**2 - q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))*
                 (-mbkin**2 - mckin**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4))) + ((-8*(3 + 8*mbkin)*mckin**4)/
                  (9*mbkin**6) + (8*mckin**2*(3 + 8*mckin))/(9*mbkin**4))*
                np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                       2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                        q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                       2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))))/
           (mbkin**2*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
               2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*
                  mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                 mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
              (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                       2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - 
                   q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                       2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**
                4)**4))))/180 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

def q2moment2Kin(q_cut, mbkin, mckin, mus, api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
 res = ( 
    ((72*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
           2*mckin**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mckin**2 - 
           7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
           mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)**2*(mbkin**10 - 
          23*mbkin**8*mckin**2 - 398*mbkin**6*mckin**4 - 398*mbkin**4*mckin**6 - 
          23*mbkin**2*mckin**8 + mckin**10 + mbkin**8*q_cut - 20*mbkin**6*mckin**2*
           q_cut - 102*mbkin**4*mckin**4*q_cut - 20*mbkin**2*mckin**6*q_cut + mckin**8*q_cut + 
          mbkin**6*q_cut**2 - 15*mbkin**4*mckin**2*q_cut**2 - 15*mbkin**2*mckin**4*q_cut**2 + 
          mckin**6*q_cut**2 - 4*mbkin**4*q_cut**3 + 2*mbkin**2*mckin**2*q_cut**3 - 
          4*mckin**4*q_cut**3 - 4*mbkin**2*q_cut**4 - 4*mckin**2*q_cut**4 + 5*q_cut**5))/
        mbkin**30 - (864*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
           2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*
         np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(17*mbkin**16 - 
          230*mbkin**14*mckin**2 - 508*mbkin**12*mckin**4 + 
          7790*mbkin**10*mckin**6 + 16102*mbkin**8*mckin**8 + 
          7790*mbkin**6*mckin**10 - 508*mbkin**4*mckin**12 - 
          230*mbkin**2*mckin**14 + 17*mckin**16 - 30*mbkin**14*q_cut + 
          122*mbkin**12*mckin**2*q_cut + 1566*mbkin**10*mckin**4*q_cut + 
          3382*mbkin**8*mckin**6*q_cut + 3382*mbkin**6*mckin**8*q_cut + 
          1566*mbkin**4*mckin**10*q_cut + 122*mbkin**2*mckin**12*q_cut - 
          30*mckin**14*q_cut - 17*mbkin**12*q_cut**2 + 180*mbkin**10*mckin**2*q_cut**2 + 
          2125*mbkin**8*mckin**4*q_cut**2 + 3656*mbkin**6*mckin**6*q_cut**2 + 
          2125*mbkin**4*mckin**8*q_cut**2 + 180*mbkin**2*mckin**10*q_cut**2 - 
          17*mckin**12*q_cut**2 + 50*mbkin**10*q_cut**3 + 62*mbkin**8*mckin**2*q_cut**3 - 
          1104*mbkin**6*mckin**4*q_cut**3 - 1104*mbkin**4*mckin**6*q_cut**3 + 
          62*mbkin**2*mckin**8*q_cut**3 + 50*mckin**10*q_cut**3 - 15*mbkin**8*q_cut**4 + 
          22*mbkin**6*mckin**2*q_cut**4 + 34*mbkin**4*mckin**4*q_cut**4 + 
          22*mbkin**2*mckin**6*q_cut**4 - 15*mckin**8*q_cut**4 - 2*mbkin**6*q_cut**5 - 
          198*mbkin**4*mckin**2*q_cut**5 - 198*mbkin**2*mckin**4*q_cut**5 - 
          2*mckin**6*q_cut**5 + 5*mbkin**4*q_cut**6 + 60*mbkin**2*mckin**2*q_cut**6 + 
          5*mckin**4*q_cut**6 - 18*mbkin**2*q_cut**7 - 18*mckin**2*q_cut**7 + 10*q_cut**8)*
         np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                 mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
               mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
             np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**24 + 
       (10368*mckin**8*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
           2*mckin**2*q_cut + q_cut**2)**2*(31*mbkin**10 - 153*mbkin**8*mckin**2 - 
          1138*mbkin**6*mckin**4 - 1138*mbkin**4*mckin**6 - 153*mbkin**2*mckin**8 + 
          31*mckin**10 - 29*mbkin**8*q_cut - 100*mbkin**6*mckin**2*q_cut - 
          162*mbkin**4*mckin**4*q_cut - 100*mbkin**2*mckin**6*q_cut - 29*mckin**8*q_cut - 
          29*mbkin**6*q_cut**2 - 125*mbkin**4*mckin**2*q_cut**2 - 125*mbkin**2*mckin**4*
           q_cut**2 - 29*mckin**6*q_cut**2 + 26*mbkin**4*q_cut**3 + 82*mbkin**2*mckin**2*
           q_cut**3 + 26*mckin**4*q_cut**3 - 4*mbkin**2*q_cut**4 - 4*mckin**2*q_cut**4 + 5*q_cut**5)*
         np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**22 - 
       (622080*mckin**12*(3*mbkin**4 + 8*mbkin**2*mckin**2 + 3*mckin**4)*
         ((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
         np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**12)/
      (540*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
          2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
        3) + (((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
          mbkin**4 + q_cut**2/mbkin**4)*(-72*mbkin**2*muG*
          ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)**2*(-((1 + mckin**2/mbkin**2)**2*
             (-6 - (33*mckin**2)/mbkin**2 + (1163*mckin**4)/mbkin**4 - 
              (5343*mckin**6)/mbkin**6 + (6489*mckin**8)/mbkin**8 + 
              (22085*mckin**10)/mbkin**10 - (10023*mckin**12)/mbkin**12 + 
              (771*mckin**14)/mbkin**14 + (17*mckin**16)/mbkin**16)) + 
           ((-10 - (115*mckin**2)/mbkin**2 + (1918*mckin**4)/mbkin**4 - 
              (8212*mckin**6)/mbkin**6 - (6930*mckin**8)/mbkin**8 - 
              (1330*mckin**10)/mbkin**10 - (11526*mckin**12)/mbkin**12 - 
              (5492*mckin**14)/mbkin**14 + (1428*mckin**16)/mbkin**16 + 
              (29*mckin**18)/mbkin**18)*q_cut)/mbkin**2 + 
           (2*(-8 + (5*mckin**2)/mbkin**2 + (578*mckin**4)/mbkin**4 - 
              (3580*mckin**6)/mbkin**6 - (14183*mckin**8)/mbkin**8 - 
              (13151*mckin**10)/mbkin**10 - (4092*mckin**12)/mbkin**12 + 
              (642*mckin**14)/mbkin**14 + (21*mckin**16)/mbkin**16)*q_cut**2)/
            mbkin**4 + ((30 + (192*mckin**2)/mbkin**2 - (3912*mckin**4)/mbkin**4 + 
              (2716*mckin**6)/mbkin**6 + (10742*mckin**8)/mbkin**8 + 
              (2184*mckin**10)/mbkin**10 - (3172*mckin**12)/mbkin**12 - 
              (84*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 - 
           (2*(-10 + (102*mckin**2)/mbkin**2 - (253*mckin**4)/mbkin**4 + 
              (327*mckin**6)/mbkin**6 - (299*mckin**8)/mbkin**8 + 
              (69*mckin**10)/mbkin**10 + (20*mckin**12)/mbkin**12)*q_cut**4)/
            mbkin**8 + (2*(-23 + (43*mckin**2)/mbkin**2 + (1111*mckin**4)/mbkin**
                4 + (1490*mckin**6)/mbkin**6 + (1026*mckin**8)/mbkin**8 + 
              (51*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
           (2*mckin**2*(23 - (218*mckin**2)/mbkin**2 - (158*mckin**4)/mbkin**4 + 
              (3*mckin**6)/mbkin**6)*q_cut**6)/mbkin**14 - 
           (2*(-13 + (52*mckin**2)/mbkin**2 + (186*mckin**4)/mbkin**4 + 
              (26*mckin**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
           ((-10 + (39*mckin**2)/mbkin**2 + (9*mckin**4)/mbkin**4)*q_cut**8)/
            mbkin**16 + (5*mckin**2*q_cut**9)/mbkin**20) - 36*muG*mupi*
          ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)**2*(-((1 + mckin**2/mbkin**2)**2*
             (-6 - (33*mckin**2)/mbkin**2 + (1163*mckin**4)/mbkin**4 - 
              (5343*mckin**6)/mbkin**6 + (6489*mckin**8)/mbkin**8 + 
              (22085*mckin**10)/mbkin**10 - (10023*mckin**12)/mbkin**12 + 
              (771*mckin**14)/mbkin**14 + (17*mckin**16)/mbkin**16)) + 
           ((-10 - (115*mckin**2)/mbkin**2 + (1918*mckin**4)/mbkin**4 - 
              (8212*mckin**6)/mbkin**6 - (6930*mckin**8)/mbkin**8 - 
              (1330*mckin**10)/mbkin**10 - (11526*mckin**12)/mbkin**12 - 
              (5492*mckin**14)/mbkin**14 + (1428*mckin**16)/mbkin**16 + 
              (29*mckin**18)/mbkin**18)*q_cut)/mbkin**2 + 
           (2*(-8 + (5*mckin**2)/mbkin**2 + (578*mckin**4)/mbkin**4 - 
              (3580*mckin**6)/mbkin**6 - (14183*mckin**8)/mbkin**8 - 
              (13151*mckin**10)/mbkin**10 - (4092*mckin**12)/mbkin**12 + 
              (642*mckin**14)/mbkin**14 + (21*mckin**16)/mbkin**16)*q_cut**2)/
            mbkin**4 + ((30 + (192*mckin**2)/mbkin**2 - (3912*mckin**4)/mbkin**4 + 
              (2716*mckin**6)/mbkin**6 + (10742*mckin**8)/mbkin**8 + 
              (2184*mckin**10)/mbkin**10 - (3172*mckin**12)/mbkin**12 - 
              (84*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 - 
           (2*(-10 + (102*mckin**2)/mbkin**2 - (253*mckin**4)/mbkin**4 + 
              (327*mckin**6)/mbkin**6 - (299*mckin**8)/mbkin**8 + 
              (69*mckin**10)/mbkin**10 + (20*mckin**12)/mbkin**12)*q_cut**4)/
            mbkin**8 + (2*(-23 + (43*mckin**2)/mbkin**2 + (1111*mckin**4)/mbkin**
                4 + (1490*mckin**6)/mbkin**6 + (1026*mckin**8)/mbkin**8 + 
              (51*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
           (2*mckin**2*(23 - (218*mckin**2)/mbkin**2 - (158*mckin**4)/mbkin**4 + 
              (3*mckin**6)/mbkin**6)*q_cut**6)/mbkin**14 - 
           (2*(-13 + (52*mckin**2)/mbkin**2 + (186*mckin**4)/mbkin**4 + 
              (26*mckin**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
           ((-10 + (39*mckin**2)/mbkin**2 + (9*mckin**4)/mbkin**4)*q_cut**8)/
            mbkin**16 + (5*mckin**2*q_cut**9)/mbkin**20) + 
         36*muG**2*((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
             mbkin**4 + q_cut**2/mbkin**4)**2*(-18 - (231*mckin**2)/mbkin**2 + 
           (1641*mckin**4)/mbkin**4 - (5542*mckin**6)/mbkin**6 + 
           (26*mckin**8)/mbkin**8 + (672*mckin**10)/mbkin**10 - 
           (58660*mckin**12)/mbkin**12 - (20842*mckin**14)/mbkin**14 + 
           (26856*mckin**16)/mbkin**16 - (4297*mckin**18)/mbkin**18 - 
           (85*mckin**20)/mbkin**20 + ((-18 - (239*mckin**2)/mbkin**2 - 
              (530*mckin**4)/mbkin**4 + (1020*mckin**6)/mbkin**6 - 
              (37266*mckin**8)/mbkin**8 + (34518*mckin**10)/mbkin**10 - 
              (10102*mckin**12)/mbkin**12 - (24964*mckin**14)/mbkin**14 + 
              (7196*mckin**16)/mbkin**16 + (145*mckin**18)/mbkin**18)*q_cut)/
            mbkin**2 + (2*(16 - (67*mckin**2)/mbkin**2 - (4098*mckin**4)/mbkin**
                4 + (6356*mckin**6)/mbkin**6 - (27619*mckin**8)/mbkin**8 - 
              (24327*mckin**10)/mbkin**10 - (7848*mckin**12)/mbkin**12 + 
              (3554*mckin**14)/mbkin**14 + (105*mckin**16)/mbkin**16)*q_cut**2)/
            mbkin**4 - (2*(-51 - (400*mckin**2)/mbkin**2 + (6112*mckin**4)/mbkin**
                4 + (474*mckin**6)/mbkin**6 - (16791*mckin**8)/mbkin**8 - 
              (5188*mckin**10)/mbkin**10 + (7926*mckin**12)/mbkin**12 + 
              (210*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 - 
           (2*(14 - (94*mckin**2)/mbkin**2 - (1493*mckin**4)/mbkin**4 + 
              (11407*mckin**6)/mbkin**6 + (4013*mckin**8)/mbkin**8 + 
              (681*mckin**10)/mbkin**10 + (100*mckin**12)/mbkin**12)*q_cut**4)/
            mbkin**8 + (2*(-75 - (97*mckin**2)/mbkin**2 + (6039*mckin**4)/mbkin**
                4 + (6826*mckin**6)/mbkin**6 + (5110*mckin**8)/mbkin**8 + 
              (255*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
           (2*mckin**2*(55 - (814*mckin**2)/mbkin**2 - (694*mckin**4)/mbkin**4 + 
              (15*mckin**6)/mbkin**6)*q_cut**6)/mbkin**14 + 
           (130*(mbkin**6 - 4*mbkin**4*mckin**2 - 14*mbkin**2*mckin**4 - 
              2*mckin**6)*q_cut**7)/mbkin**20 + (5*(-10 + (39*mckin**2)/mbkin**2 + 
              (9*mckin**4)/mbkin**4)*q_cut**8)/mbkin**16 + (25*mckin**2*q_cut**9)/
            mbkin**20) - 24*mbkin*(-((-1 + mckin**2/mbkin**2)**4*
             (1 + mckin**2/mbkin**2)**2*(503 - (9464*mckin**2)/mbkin**2 + 
              (69322*mckin**4)/mbkin**4 - (179128*mckin**6)/mbkin**6 - 
              (217124*mckin**8)/mbkin**8 + (134968*mckin**10)/mbkin**10 - 
              (44170*mckin**12)/mbkin**12 + (3064*mckin**14)/mbkin**14 + 
              (109*mckin**16)/mbkin**16)) + ((-1 + mckin**2/mbkin**2)**2*
             (2903 - (41547*mckin**2)/mbkin**2 + (196111*mckin**4)/mbkin**4 + 
              (84389*mckin**6)/mbkin**6 - (2226350*mckin**8)/mbkin**8 - 
              (4060522*mckin**10)/mbkin**10 - (2007262*mckin**12)/mbkin**12 + 
              (332278*mckin**14)/mbkin**14 + (144215*mckin**16)/mbkin**16 - 
              (185931*mckin**18)/mbkin**18 + (19663*mckin**20)/mbkin**20 + 
              (613*mckin**22)/mbkin**22)*q_cut)/mbkin**2 - 
           (2*(2870 - (34643*mckin**2)/mbkin**2 + (127322*mckin**4)/mbkin**4 + 
              (190963*mckin**6)/mbkin**6 - (1214040*mckin**8)/mbkin**8 - 
              (2984930*mckin**10)/mbkin**10 - (3081596*mckin**12)/mbkin**12 - 
              (1121550*mckin**14)/mbkin**14 + (431494*mckin**16)/mbkin**16 + 
              (81077*mckin**18)/mbkin**18 - (159838*mckin**20)/mbkin**20 + 
              (20891*mckin**22)/mbkin**22 + (540*mckin**24)/mbkin**24)*q_cut**2)/
            mbkin**4 - (2*(-969 + (3860*mckin**2)/mbkin**2 + (20318*mckin**4)/
               mbkin**4 - (143553*mckin**6)/mbkin**6 + (43662*mckin**8)/mbkin**8 + 
              (243236*mckin**10)/mbkin**10 - (166416*mckin**12)/mbkin**12 - 
              (159266*mckin**14)/mbkin**14 + (150619*mckin**16)/mbkin**16 + 
              (20136*mckin**18)/mbkin**18 - (11694*mckin**20)/mbkin**20 + 
              (67*mckin**22)/mbkin**22)*q_cut**3)/mbkin**6 + 
           ((8985 - (76768*mckin**2)/mbkin**2 + (136327*mckin**4)/mbkin**4 + 
              (661168*mckin**6)/mbkin**6 + (524386*mckin**8)/mbkin**8 + 
              (437592*mckin**10)/mbkin**10 + (491062*mckin**12)/mbkin**12 - 
              (151632*mckin**14)/mbkin**14 - (284475*mckin**16)/mbkin**16 + 
              (57128*mckin**18)/mbkin**18 + (2563*mckin**20)/mbkin**20)*q_cut**4)/
            mbkin**8 - ((12407 - (70063*mckin**2)/mbkin**2 + (19312*mckin**4)/
               mbkin**4 + (726800*mckin**6)/mbkin**6 + (1038918*mckin**8)/mbkin**
                8 + (304322*mckin**10)/mbkin**10 - (490544*mckin**12)/mbkin**12 - 
              (209264*mckin**14)/mbkin**14 + (102019*mckin**16)/mbkin**16 + 
              (1997*mckin**18)/mbkin**18)*q_cut**5)/mbkin**10 - 
           (4*(-354 + (2543*mckin**2)/mbkin**2 - (2380*mckin**4)/mbkin**4 - 
              (59099*mckin**6)/mbkin**6 - (24326*mckin**8)/mbkin**8 + 
              (44915*mckin**10)/mbkin**10 + (6624*mckin**12)/mbkin**12 - 
              (8999*mckin**14)/mbkin**14 + (564*mckin**16)/mbkin**16)*q_cut**6)/
            mbkin**12 + (4*(2103 - (3452*mckin**2)/mbkin**2 - (14684*mckin**4)/
               mbkin**4 - (17263*mckin**6)/mbkin**6 - (5615*mckin**8)/mbkin**8 + 
              (7734*mckin**10)/mbkin**10 + (11390*mckin**12)/mbkin**12 + 
              (1035*mckin**14)/mbkin**14)*q_cut**7)/mbkin**14 + 
           ((-5829 + (9694*mckin**2)/mbkin**2 + (45611*mckin**4)/mbkin**4 + 
              (25144*mckin**6)/mbkin**6 - (44863*mckin**8)/mbkin**8 - 
              (44134*mckin**10)/mbkin**10 - (1047*mckin**12)/mbkin**12)*q_cut**8)/
            mbkin**16 - ((343 + (6467*mckin**2)/mbkin**2 + (16652*mckin**4)/mbkin**
                4 - (4820*mckin**6)/mbkin**6 - (5767*mckin**8)/mbkin**8 + 
              (2021*mckin**10)/mbkin**10)*q_cut**9)/mbkin**18 + 
           (2*(830 + (1771*mckin**2)/mbkin**2 + (3242*mckin**4)/mbkin**4 + 
              (3405*mckin**6)/mbkin**6 + (928*mckin**8)/mbkin**8)*q_cut**10)/
            mbkin**20 - (2*(247 + (420*mckin**2)/mbkin**2 + (1194*mckin**4)/mbkin**
                4 + (291*mckin**6)/mbkin**6)*q_cut**11)/mbkin**22 + 
           ((3 + (92*mckin**2)/mbkin**2 + (65*mckin**4)/mbkin**4)*q_cut**12)/
            mbkin**24 - ((9 + (19*mckin**2)/mbkin**2)*q_cut**13)/mbkin**26 + 
           (8*q_cut**14)/mbkin**28)*rhoD + ((mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)*
           (-16*(-((-1 + mckin**2/mbkin**2)**4*(-2158 + (14281*mckin**2)/
                  mbkin**2 + (16728*mckin**4)/mbkin**4 - (3957*mckin**6)/
                  mbkin**6 - (23262*mckin**8)/mbkin**8 - (4629*mckin**10)/
                  mbkin**10 + (14812*mckin**12)/mbkin**12 + (425*mckin**14)/
                  mbkin**14)) + (4*(-1 + mckin**2/mbkin**2)**2*(-2507 + 
                 (11037*mckin**2)/mbkin**2 + (27788*mckin**4)/mbkin**4 + 
                 (14892*mckin**6)/mbkin**6 - (16983*mckin**8)/mbkin**8 - 
                 (31701*mckin**10)/mbkin**10 + (3740*mckin**12)/mbkin**12 + 
                 (17724*mckin**14)/mbkin**14 + (490*mckin**16)/mbkin**16)*q_cut)/
               mbkin**2 - ((-15790 + (48321*mckin**2)/mbkin**2 + 
                 (127224*mckin**4)/mbkin**4 + (95584*mckin**6)/mbkin**6 + 
                 (114300*mckin**8)/mbkin**8 - (90522*mckin**10)/mbkin**10 - 
                 (246488*mckin**12)/mbkin**12 + (43344*mckin**14)/mbkin**14 + 
                 (117234*mckin**16)/mbkin**16 + (2633*mckin**18)/mbkin**18)*q_cut**2)/
               mbkin**4 + ((-4746 + (2616*mckin**2)/mbkin**2 + (37008*mckin**4)/
                  mbkin**4 - (24212*mckin**6)/mbkin**6 - (142016*mckin**8)/
                  mbkin**8 + (4768*mckin**10)/mbkin**10 + (173592*mckin**12)/
                  mbkin**12 + (47004*mckin**14)/mbkin**14 - (894*mckin**16)/
                  mbkin**16)*q_cut**3)/mbkin**6 + (4*(-3324 - (3576*mckin**2)/
                  mbkin**2 + (6375*mckin**4)/mbkin**4 + (22309*mckin**6)/
                  mbkin**6 + (6100*mckin**8)/mbkin**8 + (17121*mckin**10)/
                  mbkin**10 + (24243*mckin**12)/mbkin**12 + (1140*mckin**14)/
                  mbkin**14)*q_cut**4)/mbkin**8 - (2*(-7473 - (14868*mckin**2)/
                  mbkin**2 + (13473*mckin**4)/mbkin**4 + (68584*mckin**6)/
                  mbkin**6 + (112941*mckin**8)/mbkin**8 + (61452*mckin**10)/
                  mbkin**10 + (483*mckin**12)/mbkin**12)*q_cut**5)/mbkin**10 - 
              (12*(347 + (965*mckin**2)/mbkin**2 - (2210*mckin**4)/mbkin**4 - 
                 (7196*mckin**6)/mbkin**6 - (2977*mckin**8)/mbkin**8 + 
                 (476*mckin**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
              (6*(-265 + (350*mckin**2)/mbkin**2 + (1064*mckin**4)/mbkin**4 + 
                 (2354*mckin**6)/mbkin**6 + (1073*mckin**8)/mbkin**8)*q_cut**7)/mbkin**
                14 - (3*(-382 + (609*mckin**2)/mbkin**2 + (2628*mckin**4)/
                  mbkin**4 + (917*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
              ((-214 + (368*mckin**2)/mbkin**2 + (470*mckin**4)/mbkin**4)*q_cut**9)/
               mbkin**18 - ((34 + (79*mckin**2)/mbkin**2)*q_cut**10)/mbkin**20 + 
              (32*q_cut**11)/mbkin**22)*rE - ((-1 + mckin**2/mbkin**2)**2 - 
              (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
             (-4*(-((-1 + mckin**2/mbkin**2)**2*(-3991 + (22369*mckin**2)/
                    mbkin**2 + (9981*mckin**4)/mbkin**4 - (62043*mckin**6)/
                    mbkin**6 - (57033*mckin**8)/mbkin**8 - (897*mckin**10)/
                    mbkin**10 + (10723*mckin**12)/mbkin**12 + (251*mckin**14)/
                    mbkin**14)) + (6*(-1763 + (4604*mckin**2)/mbkin**2 + 
                   (13346*mckin**4)/mbkin**4 - (19732*mckin**6)/mbkin**6 - 
                   (33036*mckin**8)/mbkin**8 - (24348*mckin**10)/mbkin**10 + 
                   (2270*mckin**12)/mbkin**12 + (4788*mckin**14)/mbkin**14 + 
                   (111*mckin**16)/mbkin**16)*q_cut)/mbkin**2 - 
                (12*(-322 - (930*mckin**2)/mbkin**2 + (741*mckin**4)/mbkin**4 - 
                   (1797*mckin**6)/mbkin**6 + (282*mckin**8)/mbkin**8 + 
                   (3186*mckin**10)/mbkin**10 + (1079*mckin**12)/mbkin**12 + 
                   mckin**14/mbkin**14)*q_cut**2)/mbkin**4 - 
                (4*(-2673 - (2970*mckin**2)/mbkin**2 + (618*mckin**4)/mbkin**4 + 
                   (9080*mckin**6)/mbkin**6 + (13851*mckin**8)/mbkin**8 + 
                   (6882*mckin**10)/mbkin**10 + (300*mckin**12)/mbkin**12)*q_cut**3)/
                 mbkin**6 + (12*(-748 - (1798*mckin**2)/mbkin**2 + 
                   (2525*mckin**4)/mbkin**4 + (5355*mckin**6)/mbkin**6 + 
                   (2166*mckin**8)/mbkin**8 + (36*mckin**10)/mbkin**10)*q_cut**4)/
                 mbkin**8 + (6*(-169 + (308*mckin**2)/mbkin**2 - (834*mckin**4)/
                    mbkin**4 + (68*mckin**6)/mbkin**6 + (211*mckin**8)/mbkin**8)*
                  q_cut**5)/mbkin**10 - (24*(-100 + (40*mckin**2)/mbkin**2 + 
                   (199*mckin**4)/mbkin**4 + (49*mckin**6)/mbkin**6)*q_cut**6)/
                 mbkin**12 + (12*(-31 + (30*mckin**2)/mbkin**2 + (21*mckin**4)/
                    mbkin**4)*q_cut**7)/mbkin**14 + (15*(-1 + mckin**2/mbkin**2)*
                  q_cut**8)/mbkin**16 + (8*q_cut**9)/mbkin**18)*rG - 
              12*(-((-1 + mckin**2/mbkin**2)**2*(773 - (1157*mckin**2)/mbkin**2 - 
                   (30827*mckin**4)/mbkin**4 - (37389*mckin**6)/mbkin**6 - 
                   (23869*mckin**8)/mbkin**8 - (7027*mckin**10)/mbkin**10 + 
                   (8563*mckin**12)/mbkin**12 + (213*mckin**14)/mbkin**14)) + 
                (2*(959 + (2370*mckin**2)/mbkin**2 - (35340*mckin**4)/mbkin**4 - 
                   (73922*mckin**6)/mbkin**6 - (52234*mckin**8)/mbkin**8 - 
                   (30418*mckin**10)/mbkin**10 - (4644*mckin**12)/mbkin**12 + 
                   (11506*mckin**14)/mbkin**14 + (283*mckin**16)/mbkin**16)*q_cut)/
                 mbkin**2 - (4*(138 + (1678*mckin**2)/mbkin**2 - (129*mckin**4)/
                    mbkin**4 - (321*mckin**6)/mbkin**6 - (426*mckin**8)/mbkin**8 + 
                   (3978*mckin**10)/mbkin**10 + (2637*mckin**12)/mbkin**12 + 
                   (5*mckin**14)/mbkin**14)*q_cut**2)/mbkin**4 - 
                (4*(456 + (2987*mckin**2)/mbkin**2 + (4858*mckin**4)/mbkin**4 + 
                   (9938*mckin**6)/mbkin**6 + (9260*mckin**8)/mbkin**8 + 
                   (5515*mckin**10)/mbkin**10 + (250*mckin**12)/mbkin**12)*q_cut**3)/
                 mbkin**6 + (4*(337 + (3220*mckin**2)/mbkin**2 + (8155*mckin**4)/
                    mbkin**4 + (10439*mckin**6)/mbkin**6 + (5285*mckin**8)/
                    mbkin**8 + (98*mckin**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
                (2*(-63 - (414*mckin**2)/mbkin**2 - (1068*mckin**4)/mbkin**4 + 
                   (66*mckin**6)/mbkin**6 + (479*mckin**8)/mbkin**8)*q_cut**5)/
                 mbkin**10 - (8*(-21 + (146*mckin**2)/mbkin**2 + (479*mckin**4)/
                    mbkin**4 + (107*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
                (4*(-38 + (77*mckin**2)/mbkin**2 + (35*mckin**4)/mbkin**4)*q_cut**7)/
                 mbkin**14 + (5*(-3 + (5*mckin**2)/mbkin**2)*q_cut**8)/mbkin**16 + 
                (8*q_cut**9)/mbkin**18)*sB - 10496*sE + (55248*mckin**2*sE)/mbkin**
                2 + (72576*mckin**4*sE)/mbkin**4 - (154048*mckin**6*sE)/mbkin**
                6 + (21120*mckin**8*sE)/mbkin**8 - (52704*mckin**10*sE)/mbkin**
                10 - (24448*mckin**12*sE)/mbkin**12 + (153024*mckin**14*sE)/
               mbkin**14 - (58752*mckin**16*sE)/mbkin**16 - (1520*mckin**18*sE)/
               mbkin**18 + (27264*q_cut*sE)/mbkin**2 - (4128*mckin**2*q_cut*sE)/mbkin**
                4 - (502944*mckin**4*q_cut*sE)/mbkin**6 - (430944*mckin**6*q_cut*sE)/
               mbkin**8 - (400608*mckin**8*q_cut*sE)/mbkin**10 - (306912*mckin**10*
                q_cut*sE)/mbkin**12 + (16800*mckin**12*q_cut*sE)/mbkin**14 + 
              (169056*mckin**14*q_cut*sE)/mbkin**16 + (3936*mckin**16*q_cut*sE)/mbkin**
                18 - (8832*q_cut**2*sE)/mbkin**4 - (62592*mckin**2*q_cut**2*sE)/mbkin**
                6 + (51840*mckin**4*q_cut**2*sE)/mbkin**8 + (108864*mckin**6*q_cut**2*
                sE)/mbkin**10 - (150912*mckin**8*q_cut**2*sE)/mbkin**12 - 
              (296064*mckin**10*q_cut**2*sE)/mbkin**14 - (84096*mckin**12*q_cut**2*sE)/
               mbkin**16 + (192*mckin**14*q_cut**2*sE)/mbkin**18 - (28032*q_cut**3*sE)/
               mbkin**6 - (89760*mckin**2*q_cut**3*sE)/mbkin**8 - (92448*mckin**4*
                q_cut**3*sE)/mbkin**10 - (12992*mckin**6*q_cut**3*sE)/mbkin**12 - 
              (88896*mckin**8*q_cut**3*sE)/mbkin**14 - (155808*mckin**10*q_cut**3*sE)/
               mbkin**16 - (7200*mckin**12*q_cut**3*sE)/mbkin**18 + 
              (21600*q_cut**4*sE)/mbkin**8 + (117600*mckin**2*q_cut**4*sE)/mbkin**10 + 
              (148800*mckin**4*q_cut**4*sE)/mbkin**12 + (251520*mckin**6*q_cut**4*sE)/
               mbkin**14 + (160992*mckin**8*q_cut**4*sE)/mbkin**16 + 
              (1632*mckin**10*q_cut**4*sE)/mbkin**18 + (2784*q_cut**5*sE)/mbkin**10 - 
              (10752*mckin**2*q_cut**5*sE)/mbkin**12 - (28800*mckin**4*q_cut**5*sE)/
               mbkin**14 - (9216*mckin**6*q_cut**5*sE)/mbkin**16 + (9120*mckin**8*
                q_cut**5*sE)/mbkin**18 - (3936*q_cut**6*sE)/mbkin**12 - 
              (7104*mckin**2*q_cut**6*sE)/mbkin**14 - (23904*mckin**4*q_cut**6*sE)/mbkin**
                16 - (8256*mckin**6*q_cut**6*sE)/mbkin**18 - (480*q_cut**7*sE)/mbkin**
                14 + (1728*mckin**2*q_cut**7*sE)/mbkin**16 + (2208*mckin**4*q_cut**7*
                sE)/mbkin**18 - (240*mckin**2*q_cut**8*sE)/mbkin**18 + 
              (128*q_cut**9*sE)/mbkin**18 - 803*sqB + (1899*mckin**2*sqB)/mbkin**2 + 
              (58452*mckin**4*sqB)/mbkin**4 - (23284*mckin**6*sqB)/mbkin**6 - 
              (87534*mckin**8*sqB)/mbkin**8 + (10878*mckin**10*sqB)/mbkin**10 + 
              (32876*mckin**12*sqB)/mbkin**12 + (10548*mckin**14*sqB)/mbkin**14 - 
              (2991*mckin**16*sqB)/mbkin**16 - (41*mckin**18*sqB)/mbkin**18 + 
              (1914*q_cut*sqB)/mbkin**2 + (8832*mckin**2*q_cut*sqB)/mbkin**4 - 
              (110436*mckin**4*q_cut*sqB)/mbkin**6 - (338112*mckin**6*q_cut*sqB)/mbkin**
                8 - (279840*mckin**8*q_cut*sqB)/mbkin**10 - (102144*mckin**10*q_cut*
                sqB)/mbkin**12 + (5028*mckin**12*q_cut*sqB)/mbkin**14 + 
              (8256*mckin**14*q_cut*sqB)/mbkin**16 + (102*mckin**16*q_cut*sqB)/mbkin**
                18 - (552*q_cut**2*sqB)/mbkin**4 - (8088*mckin**2*q_cut**2*sqB)/mbkin**
                6 - (5796*mckin**4*q_cut**2*sqB)/mbkin**8 - (2556*mckin**6*q_cut**2*sqB)/
               mbkin**10 - (24408*mckin**8*q_cut**2*sqB)/mbkin**12 - 
              (22248*mckin**10*q_cut**2*sqB)/mbkin**14 - (3564*mckin**12*q_cut**2*sqB)/
               mbkin**16 + (12*mckin**14*q_cut**2*sqB)/mbkin**18 - (1548*q_cut**3*sqB)/
               mbkin**6 - (16692*mckin**2*q_cut**3*sqB)/mbkin**8 - (11460*mckin**4*
                q_cut**3*sqB)/mbkin**10 - (14288*mckin**6*q_cut**3*sqB)/mbkin**12 - 
              (17340*mckin**8*q_cut**3*sqB)/mbkin**14 - (7932*mckin**10*q_cut**3*sqB)/
               mbkin**16 - (180*mckin**12*q_cut**3*sqB)/mbkin**18 + 
              (1164*q_cut**4*sqB)/mbkin**8 + (16356*mckin**2*q_cut**4*sqB)/mbkin**10 + 
              (40188*mckin**4*q_cut**4*sqB)/mbkin**12 + (29892*mckin**6*q_cut**4*sqB)/
               mbkin**14 + (7428*mckin**8*q_cut**4*sqB)/mbkin**16 + (12*mckin**10*
                q_cut**4*sqB)/mbkin**18 - (510*q_cut**5*sqB)/mbkin**10 - 
              (2076*mckin**2*q_cut**5*sqB)/mbkin**12 - (5256*mckin**4*q_cut**5*sqB)/
               mbkin**14 - (36*mckin**6*q_cut**5*sqB)/mbkin**16 + (246*mckin**8*q_cut**5*
                sqB)/mbkin**18 + (588*q_cut**6*sqB)/mbkin**12 - (408*mckin**2*q_cut**6*
                sqB)/mbkin**14 - (1308*mckin**4*q_cut**6*sqB)/mbkin**16 - 
              (192*mckin**6*q_cut**6*sqB)/mbkin**18 - (216*q_cut**7*sqB)/mbkin**14 + 
              (192*mckin**2*q_cut**7*sqB)/mbkin**16 + (48*mckin**4*q_cut**7*sqB)/mbkin**
                18 - (45*q_cut**8*sqB)/mbkin**16 - (15*mckin**2*q_cut**8*sqB)/mbkin**
                18 + (8*q_cut**9*sqB)/mbkin**18)))/mbkin**6) - 
       12*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
           2*mckin**2*q_cut + q_cut**2)/mbkin**4)*
        (16*(-((-1 + mckin**2/mbkin**2)**4*(-58 + (585*mckin**2)/mbkin**2 + 
              (4376*mckin**4)/mbkin**4 - (36483*mckin**6)/mbkin**6 - 
              (43458*mckin**8)/mbkin**8 + (11933*mckin**10)/mbkin**10 + 
              (47124*mckin**12)/mbkin**12 + (8583*mckin**14)/mbkin**14 - 
              (26848*mckin**16)/mbkin**16 - (2978*mckin**18)/mbkin**18 + 
              (504*mckin**20)/mbkin**20)) + (2*(-1 + mckin**2/mbkin**2)**2*
             (-176 + (1403*mckin**2)/mbkin**2 + (11699*mckin**4)/mbkin**4 - 
              (60791*mckin**6)/mbkin**6 - (146769*mckin**8)/mbkin**8 - 
              (73597*mckin**10)/mbkin**10 + (82981*mckin**12)/mbkin**12 + 
              (120123*mckin**14)/mbkin**14 - (8527*mckin**16)/mbkin**16 - 
              (67374*mckin**18)/mbkin**18 - (7400*mckin**20)/mbkin**20 + 
              (1548*mckin**22)/mbkin**22)*q_cut)/mbkin**2 + 
           ((770 - (4873*mckin**2)/mbkin**2 - (39796*mckin**4)/mbkin**4 + 
              (144026*mckin**6)/mbkin**6 + (339380*mckin**8)/mbkin**8 + 
              (299362*mckin**10)/mbkin**10 + (220952*mckin**12)/mbkin**12 - 
              (216028*mckin**14)/mbkin**14 - (441230*mckin**16)/mbkin**16 + 
              (32151*mckin**18)/mbkin**18 + (233484*mckin**20)/mbkin**20 + 
              (26162*mckin**22)/mbkin**22 - (6840*mckin**24)/mbkin**24)*q_cut**2)/
            mbkin**4 + (2*(-235 + (591*mckin**2)/mbkin**2 + (7800*mckin**4)/mbkin**
                4 - (8734*mckin**6)/mbkin**6 - (52158*mckin**8)/mbkin**8 + 
              (25714*mckin**10)/mbkin**10 + (129232*mckin**12)/mbkin**12 + 
              (2040*mckin**14)/mbkin**14 - (152887*mckin**16)/mbkin**16 - 
              (63857*mckin**18)/mbkin**18 + (984*mckin**20)/mbkin**20 + 
              (2070*mckin**22)/mbkin**22)*q_cut**3)/mbkin**6 + 
           (2*(-435 + (981*mckin**2)/mbkin**2 + (22494*mckin**4)/mbkin**4 + 
              (23775*mckin**6)/mbkin**6 - (20508*mckin**8)/mbkin**8 - 
              (76853*mckin**10)/mbkin**10 - (40824*mckin**12)/mbkin**12 - 
              (68001*mckin**14)/mbkin**14 - (91813*mckin**16)/mbkin**16 - 
              (9258*mckin**18)/mbkin**18 + (4050*mckin**20)/mbkin**20)*q_cut**4)/
            mbkin**8 - (4*(-426 + (6*mckin**2)/mbkin**2 + (14202*mckin**4)/mbkin**
                4 + (22767*mckin**6)/mbkin**6 - (11589*mckin**8)/mbkin**8 - 
              (74018*mckin**10)/mbkin**10 - (112690*mckin**12)/mbkin**12 - 
              (63853*mckin**14)/mbkin**14 + (2397*mckin**16)/mbkin**16 + 
              (3888*mckin**18)/mbkin**18)*q_cut**5)/mbkin**10 + 
           (2*(-378 - (159*mckin**2)/mbkin**2 + (11412*mckin**4)/mbkin**4 + 
              (16246*mckin**6)/mbkin**6 - (37738*mckin**8)/mbkin**8 - 
              (96055*mckin**10)/mbkin**10 - (44068*mckin**12)/mbkin**12 + 
              (11250*mckin**14)/mbkin**14 + (3024*mckin**16)/mbkin**16)*q_cut**6)/
            mbkin**12 + (4*(-165 - (738*mckin**2)/mbkin**2 - (591*mckin**4)/mbkin**
                4 + (940*mckin**6)/mbkin**6 + (4641*mckin**8)/mbkin**8 + 
              (2194*mckin**10)/mbkin**10 + (1377*mckin**12)/mbkin**12 + 
              (1890*mckin**14)/mbkin**14)*q_cut**7)/mbkin**14 - 
           ((-870 - (3543*mckin**2)/mbkin**2 + (1320*mckin**4)/mbkin**4 + 
              (10451*mckin**6)/mbkin**6 + (14996*mckin**8)/mbkin**8 + 
              (19998*mckin**10)/mbkin**10 + (8640*mckin**12)/mbkin**12)*q_cut**8)/
            mbkin**16 + (2*(-140 - (631*mckin**2)/mbkin**2 + (631*mckin**4)/mbkin**
                4 + (3208*mckin**6)/mbkin**6 + (3818*mckin**8)/mbkin**8 + 
              (900*mckin**10)/mbkin**10)*q_cut**9)/mbkin**18 + 
           ((-62 - (9*mckin**2)/mbkin**2 + (476*mckin**4)/mbkin**4 + 
              (1114*mckin**6)/mbkin**6 + (1656*mckin**8)/mbkin**8)*q_cut**10)/
            mbkin**20 + ((58 + (58*mckin**2)/mbkin**2 - (788*mckin**4)/mbkin**4 - 
              (1044*mckin**6)/mbkin**6)*q_cut**11)/mbkin**22 - 
           (10*(mbkin**4 - 18*mckin**4)*q_cut**12)/mbkin**28)*rE + 
         ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
           q_cut**2/mbkin**4)*((540*mckin**2*muG**2)/mbkin**2 - (4248*mckin**4*muG**2)/
            mbkin**4 + (7200*mckin**6*muG**2)/mbkin**6 + (41904*mckin**8*muG**2)/
            mbkin**8 - (183960*mckin**10*muG**2)/mbkin**10 + 
           (687024*mckin**12*muG**2)/mbkin**12 - (719712*mckin**14*muG**2)/
            mbkin**14 - (252000*mckin**16*muG**2)/mbkin**16 + 
           (549756*mckin**18*muG**2)/mbkin**18 - (109080*mckin**20*muG**2)/
            mbkin**20 - (22464*mckin**22*muG**2)/mbkin**22 + 
           (5040*mckin**24*muG**2)/mbkin**24 + (180*mckin**2*muG*mupi)/mbkin**2 - 
           (2664*mckin**4*muG*mupi)/mbkin**4 + (2736*mckin**6*muG*mupi)/
            mbkin**6 + (68112*mckin**8*muG*mupi)/mbkin**8 - 
           (257544*mckin**10*muG*mupi)/mbkin**10 - (90288*mckin**12*muG*mupi)/
            mbkin**12 + (528192*mckin**14*muG*mupi)/mbkin**14 - 
           (27936*mckin**16*muG*mupi)/mbkin**16 - (281772*mckin**18*muG*mupi)/
            mbkin**18 + (53784*mckin**20*muG*mupi)/mbkin**20 + 
           (8208*mckin**22*muG*mupi)/mbkin**22 - (1008*mckin**24*muG*mupi)/
            mbkin**24 - (720*mckin**2*muG**2*q_cut)/mbkin**4 + 
           (3168*mckin**4*muG**2*q_cut)/mbkin**6 - (6768*mckin**6*muG**2*q_cut)/
            mbkin**8 + (114336*mckin**8*muG**2*q_cut)/mbkin**10 - 
           (356544*mckin**10*muG**2*q_cut)/mbkin**12 - (417312*mckin**12*muG**2*q_cut)/
            mbkin**14 - (1368000*mckin**14*muG**2*q_cut)/mbkin**16 - 
           (565344*mckin**16*muG**2*q_cut)/mbkin**18 + (387792*mckin**18*muG**2*q_cut)/
            mbkin**20 + (52992*mckin**20*muG**2*q_cut)/mbkin**22 - 
           (20880*mckin**22*muG**2*q_cut)/mbkin**24 - (720*mckin**2*muG*mupi*q_cut)/
            mbkin**4 + (8352*mckin**4*muG*mupi*q_cut)/mbkin**6 + 
           (1872*mckin**6*muG*mupi*q_cut)/mbkin**8 - (204480*mckin**8*muG*mupi*q_cut)/
            mbkin**10 + (266400*mckin**10*muG*mupi*q_cut)/mbkin**12 + 
           (936576*mckin**12*muG*mupi*q_cut)/mbkin**14 + 
           (1141920*mckin**14*muG*mupi*q_cut)/mbkin**16 + 
           (230400*mckin**16*muG*mupi*q_cut)/mbkin**18 - 
           (186768*mckin**18*muG*mupi*q_cut)/mbkin**20 - 
           (20448*mckin**20*muG*mupi*q_cut)/mbkin**22 + 
           (4176*mckin**22*muG*mupi*q_cut)/mbkin**24 - (720*mckin**2*muG**2*q_cut**2)/
            mbkin**6 + (7200*mckin**4*muG**2*q_cut**2)/mbkin**8 + 
           (5328*mckin**6*muG**2*q_cut**2)/mbkin**10 - (113760*mckin**8*muG**2*q_cut**2)/
            mbkin**12 - (187920*mckin**10*muG**2*q_cut**2)/mbkin**14 + 
           (251136*mckin**12*muG**2*q_cut**2)/mbkin**16 + 
           (101520*mckin**14*muG**2*q_cut**2)/mbkin**18 - 
           (235296*mckin**16*muG**2*q_cut**2)/mbkin**20 - 
           (30528*mckin**18*muG**2*q_cut**2)/mbkin**22 + (21600*mckin**20*muG**2*q_cut**2)/
            mbkin**24 + (720*mckin**2*muG*mupi*q_cut**2)/mbkin**6 - 
           (5760*mckin**4*muG*mupi*q_cut**2)/mbkin**8 - (8784*mckin**6*muG*mupi*
             q_cut**2)/mbkin**10 + (100224*mckin**8*muG*mupi*q_cut**2)/mbkin**12 + 
           (92880*mckin**10*muG*mupi*q_cut**2)/mbkin**14 - 
           (143136*mckin**12*muG*mupi*q_cut**2)/mbkin**16 + 
           (42480*mckin**14*muG*mupi*q_cut**2)/mbkin**18 + 
           (99072*mckin**16*muG*mupi*q_cut**2)/mbkin**20 + 
           (8064*mckin**18*muG*mupi*q_cut**2)/mbkin**22 - 
           (4320*mckin**20*muG*mupi*q_cut**2)/mbkin**24 - (720*mckin**2*muG**2*q_cut**3)/
            mbkin**8 - (1080*mckin**4*muG**2*q_cut**3)/mbkin**10 + 
           (34776*mckin**6*muG**2*q_cut**3)/mbkin**12 - (323064*mckin**8*muG**2*q_cut**3)/
            mbkin**14 - (826920*mckin**10*muG**2*q_cut**3)/mbkin**16 - 
           (943848*mckin**12*muG**2*q_cut**3)/mbkin**18 - 
           (345528*mckin**14*muG**2*q_cut**3)/mbkin**20 + 
           (37080*mckin**16*muG**2*q_cut**3)/mbkin**22 + (22680*mckin**18*muG**2*q_cut**3)/
            mbkin**24 + (720*mckin**2*muG*mupi*q_cut**3)/mbkin**8 - 
           (8136*mckin**4*muG*mupi*q_cut**3)/mbkin**10 - 
           (8280*mckin**6*muG*mupi*q_cut**3)/mbkin**12 + 
           (221400*mckin**8*muG*mupi*q_cut**3)/mbkin**14 + 
           (500616*mckin**10*muG*mupi*q_cut**3)/mbkin**16 + 
           (469800*mckin**12*muG*mupi*q_cut**3)/mbkin**18 + 
           (200952*mckin**14*muG*mupi*q_cut**3)/mbkin**20 + 
           (6408*mckin**16*muG*mupi*q_cut**3)/mbkin**22 - 
           (4536*mckin**18*muG*mupi*q_cut**3)/mbkin**24 + (4680*mckin**2*muG**2*q_cut**4)/
            mbkin**10 - (21240*mckin**4*muG**2*q_cut**4)/mbkin**12 - 
           (100224*mckin**6*muG**2*q_cut**4)/mbkin**14 + (330840*mckin**8*muG**2*q_cut**4)/
            mbkin**16 + (918216*mckin**10*muG**2*q_cut**4)/mbkin**18 + 
           (432792*mckin**12*muG**2*q_cut**4)/mbkin**20 - 
           (87840*mckin**14*muG**2*q_cut**4)/mbkin**22 - (57240*mckin**16*muG**2*q_cut**4)/
            mbkin**24 - (1800*mckin**2*muG*mupi*q_cut**4)/mbkin**10 + 
           (14328*mckin**4*muG*mupi*q_cut**4)/mbkin**12 + 
           (28800*mckin**6*muG*mupi*q_cut**4)/mbkin**14 - 
           (246456*mckin**8*muG*mupi*q_cut**4)/mbkin**16 - 
           (465480*mckin**10*muG*mupi*q_cut**4)/mbkin**18 - 
           (205272*mckin**12*muG*mupi*q_cut**4)/mbkin**20 + 
           (8928*mckin**14*muG*mupi*q_cut**4)/mbkin**22 + 
           (11448*mckin**16*muG*mupi*q_cut**4)/mbkin**24 - 
           (2160*mckin**2*muG**2*q_cut**5)/mbkin**12 + (15048*mckin**4*muG**2*q_cut**5)/
            mbkin**14 + (7416*mckin**6*muG**2*q_cut**5)/mbkin**16 - 
           (256464*mckin**8*muG**2*q_cut**5)/mbkin**18 - (269568*mckin**10*muG**2*
             q_cut**5)/mbkin**20 + (1224*mckin**12*muG**2*q_cut**5)/mbkin**22 + 
           (18360*mckin**14*muG**2*q_cut**5)/mbkin**24 + (720*mckin**2*muG*mupi*q_cut**5)/
            mbkin**12 - (3816*mckin**4*muG*mupi*q_cut**5)/mbkin**14 - 
           (12024*mckin**6*muG*mupi*q_cut**5)/mbkin**16 + 
           (52272*mckin**8*muG*mupi*q_cut**5)/mbkin**18 + 
           (39744*mckin**10*muG*mupi*q_cut**5)/mbkin**20 - 
           (6408*mckin**12*muG*mupi*q_cut**5)/mbkin**22 - 
           (3672*mckin**14*muG*mupi*q_cut**5)/mbkin**24 - (3600*mckin**2*muG**2*q_cut**6)/
            mbkin**14 + (21672*mckin**4*muG**2*q_cut**6)/mbkin**16 + 
           (154224*mckin**6*muG**2*q_cut**6)/mbkin**18 + (236880*mckin**8*muG**2*q_cut**6)/
            mbkin**20 + (119808*mckin**10*muG**2*q_cut**6)/mbkin**22 + 
           (33480*mckin**12*muG**2*q_cut**6)/mbkin**24 + (720*mckin**2*muG*mupi*q_cut**6)/
            mbkin**14 - (7272*mckin**4*muG*mupi*q_cut**6)/mbkin**16 - 
           (19440*mckin**6*muG*mupi*q_cut**6)/mbkin**18 - 
           (18864*mckin**8*muG*mupi*q_cut**6)/mbkin**20 - 
           (18432*mckin**10*muG*mupi*q_cut**6)/mbkin**22 - 
           (6696*mckin**12*muG*mupi*q_cut**6)/mbkin**24 + (3600*mckin**2*muG**2*q_cut**7)/
            mbkin**16 - (26280*mckin**4*muG**2*q_cut**7)/mbkin**18 - 
           (98712*mckin**6*muG**2*q_cut**7)/mbkin**20 - (78552*mckin**8*muG**2*q_cut**7)/
            mbkin**22 - (27000*mckin**10*muG**2*q_cut**7)/mbkin**24 - 
           (720*mckin**2*muG*mupi*q_cut**7)/mbkin**16 + (6120*mckin**4*muG*mupi*
             q_cut**7)/mbkin**18 + (17496*mckin**6*muG*mupi*q_cut**7)/mbkin**20 + 
           (16056*mckin**8*muG*mupi*q_cut**7)/mbkin**22 + 
           (5400*mckin**10*muG*mupi*q_cut**7)/mbkin**24 - (900*mckin**2*muG**2*q_cut**8)/
            mbkin**18 + (720*mckin**4*muG**2*q_cut**8)/mbkin**20 + 
           (1440*mckin**6*muG**2*q_cut**8)/mbkin**22 - (1080*mckin**8*muG**2*q_cut**8)/
            mbkin**24 + (180*mckin**2*muG*mupi*q_cut**8)/mbkin**18 - 
           (144*mckin**4*muG*mupi*q_cut**8)/mbkin**20 - (1008*mckin**6*muG*mupi*
             q_cut**8)/mbkin**22 + (216*mckin**8*muG*mupi*q_cut**8)/mbkin**24 + 
           (6840*mckin**4*muG**2*q_cut**9)/mbkin**22 + (6840*mckin**6*muG**2*q_cut**9)/
            mbkin**24 - (1368*mckin**4*muG*mupi*q_cut**9)/mbkin**22 - 
           (1368*mckin**6*muG*mupi*q_cut**9)/mbkin**24 - (1800*mckin**4*muG**2*q_cut**10)/
            mbkin**24 + (360*mckin**4*muG*mupi*q_cut**10)/mbkin**24 - 
           72*mckin**2*muG*((-1 + mckin**2/mbkin**2)**2*(-5 + (64*mckin**2)/
                mbkin**2 + (57*mckin**4)/mbkin**4 - (1842*mckin**6)/mbkin**6 + 
               (3413*mckin**8)/mbkin**8 + (11176*mckin**10)/mbkin**10 + 
               (4267*mckin**12)/mbkin**12 - (1866*mckin**14)/mbkin**14 - 
               (172*mckin**16)/mbkin**16 + (28*mckin**18)/mbkin**18) - 
             (4*(-5 + (58*mckin**2)/mbkin**2 + (13*mckin**4)/mbkin**4 - 
                (1420*mckin**6)/mbkin**6 + (1850*mckin**8)/mbkin**8 + 
                (6504*mckin**10)/mbkin**10 + (7930*mckin**12)/mbkin**12 + 
                (1600*mckin**14)/mbkin**14 - (1297*mckin**16)/mbkin**16 - 
                (142*mckin**18)/mbkin**18 + (29*mckin**20)/mbkin**20)*q_cut)/
              mbkin**2 + (4*(-5 + (40*mckin**2)/mbkin**2 + (61*mckin**4)/
                 mbkin**4 - (696*mckin**6)/mbkin**6 - (645*mckin**8)/mbkin**8 + 
                (994*mckin**10)/mbkin**10 - (295*mckin**12)/mbkin**12 - 
                (688*mckin**14)/mbkin**14 - (56*mckin**16)/mbkin**16 + 
                (30*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
             (2*(-10 + (113*mckin**2)/mbkin**2 + (115*mckin**4)/mbkin**4 - 
                (3075*mckin**6)/mbkin**6 - (6953*mckin**8)/mbkin**8 - 
                (6525*mckin**10)/mbkin**10 - (2791*mckin**12)/mbkin**12 - 
                (89*mckin**14)/mbkin**14 + (63*mckin**16)/mbkin**16)*q_cut**3)/
              mbkin**6 + ((50 - (398*mckin**2)/mbkin**2 - (800*mckin**4)/
                 mbkin**4 + (6846*mckin**6)/mbkin**6 + (12930*mckin**8)/mbkin**8 + 
                (5702*mckin**10)/mbkin**10 - (248*mckin**12)/mbkin**12 - 
                (318*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
             (2*(-10 + (53*mckin**2)/mbkin**2 + (167*mckin**4)/mbkin**4 - 
                (726*mckin**6)/mbkin**6 - (552*mckin**8)/mbkin**8 + (89*mckin**10)/
                 mbkin**10 + (51*mckin**12)/mbkin**12)*q_cut**5)/mbkin**10 + 
             (2*(-10 + (101*mckin**2)/mbkin**2 + (270*mckin**4)/mbkin**4 + 
                (262*mckin**6)/mbkin**6 + (256*mckin**8)/mbkin**8 + (93*mckin**10)/
                 mbkin**10)*q_cut**6)/mbkin**12 - (2*(-10 + (85*mckin**2)/mbkin**2 + 
                (243*mckin**4)/mbkin**4 + (223*mckin**6)/mbkin**6 + (75*mckin**8)/
                 mbkin**8)*q_cut**7)/mbkin**14 + ((-5 + (4*mckin**2)/mbkin**2 + 
                (28*mckin**4)/mbkin**4 - (6*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
             (38*mckin**2*(mbkin**2 + mckin**2)*q_cut**9)/mbkin**22 - 
             (10*mckin**2*q_cut**10)/mbkin**22) - 
           4*(-((-1 + mckin**2/mbkin**2)**2*(-97 + (735*mckin**2)/mbkin**2 + 
                (7763*mckin**4)/mbkin**4 - (49677*mckin**6)/mbkin**6 - 
                (29811*mckin**8)/mbkin**8 + (166397*mckin**10)/mbkin**10 + 
                (165981*mckin**12)/mbkin**12 + (5133*mckin**14)/mbkin**14 - 
                (23212*mckin**16)/mbkin**16 - (1628*mckin**18)/mbkin**18 + 
                (336*mckin**20)/mbkin**20)) + (4*(-101 + (552*mckin**2)/
                 mbkin**2 + (5871*mckin**4)/mbkin**4 - (19052*mckin**6)/mbkin**6 - 
                (43986*mckin**8)/mbkin**8 + (71160*mckin**10)/mbkin**10 + 
                (151138*mckin**12)/mbkin**12 + (98832*mckin**14)/mbkin**14 - 
                (5145*mckin**16)/mbkin**16 - (16416*mckin**18)/mbkin**18 - 
                (1281*mckin**20)/mbkin**20 + (348*mckin**22)/mbkin**22)*q_cut)/
              mbkin**2 - (12*(-35 + (45*mckin**2)/mbkin**2 + (1125*mckin**4)/
                 mbkin**4 + (885*mckin**6)/mbkin**6 - (2003*mckin**8)/mbkin**8 + 
                (3409*mckin**10)/mbkin**10 - (399*mckin**12)/mbkin**12 - 
                (6723*mckin**14)/mbkin**14 - (3108*mckin**16)/mbkin**16 - 
                (36*mckin**18)/mbkin**18 + (120*mckin**20)/mbkin**20)*q_cut**2)/
              mbkin**4 - (4*(-111 + (258*mckin**2)/mbkin**2 + (6843*mckin**4)/
                 mbkin**4 + (6336*mckin**6)/mbkin**6 - (7977*mckin**8)/mbkin**8 - 
                (27428*mckin**10)/mbkin**10 - (35087*mckin**12)/mbkin**12 - 
                (15128*mckin**14)/mbkin**14 - (660*mckin**16)/mbkin**16 + 
                (378*mckin**18)/mbkin**18)*q_cut**3)/mbkin**6 + 
             (2*(-561 - (651*mckin**2)/mbkin**2 + (15192*mckin**4)/mbkin**4 + 
                (20940*mckin**6)/mbkin**6 - (48597*mckin**8)/mbkin**8 - 
                (82731*mckin**10)/mbkin**10 - (30908*mckin**12)/mbkin**12 + 
                (2496*mckin**14)/mbkin**14 + (1908*mckin**16)/mbkin**16)*q_cut**4)/
              mbkin**8 - (12*(-29 - (102*mckin**2)/mbkin**2 + (461*mckin**4)/
                 mbkin**4 + (182*mckin**6)/mbkin**6 - (2061*mckin**8)/mbkin**8 - 
                (718*mckin**10)/mbkin**10 + (405*mckin**12)/mbkin**12 + 
                (102*mckin**14)/mbkin**14)*q_cut**5)/mbkin**10 - 
             (4*(-171 - (513*mckin**2)/mbkin**2 + (393*mckin**4)/mbkin**4 + 
                (1009*mckin**6)/mbkin**6 + (1000*mckin**8)/mbkin**8 + 
                (1164*mckin**10)/mbkin**10 + (558*mckin**12)/mbkin**12)*q_cut**6)/
              mbkin**12 + (4*(-135 - (510*mckin**2)/mbkin**2 + (273*mckin**4)/
                 mbkin**4 + (1460*mckin**6)/mbkin**6 + (1302*mckin**8)/mbkin**8 + 
                (450*mckin**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
             (3*(-13 + (69*mckin**2)/mbkin**2 - (172*mckin**4)/mbkin**4 - 
                (180*mckin**6)/mbkin**6 + (24*mckin**8)/mbkin**8)*q_cut**8)/
              mbkin**16 - (8*(-19 - (19*mckin**2)/mbkin**2 + (49*mckin**4)/
                 mbkin**4 + (57*mckin**6)/mbkin**6)*q_cut**9)/mbkin**18 - 
             (40*(mbkin**4 - 3*mckin**4)*q_cut**10)/mbkin**24)*rG + 
           24*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(13 - (155*mckin**2)/mbkin**2 + 
                (233*mckin**4)/mbkin**4 + (9069*mckin**6)/mbkin**6 - 
                (59845*mckin**8)/mbkin**8 - (147269*mckin**10)/mbkin**10 - 
                (55861*mckin**12)/mbkin**12 + (18451*mckin**14)/mbkin**14 - 
                (5640*mckin**16)/mbkin**16 - (1056*mckin**18)/mbkin**18 + 
                (140*mckin**20)/mbkin**20)) + (4*(14 - (143*mckin**2)/mbkin**2 + 
                (150*mckin**4)/mbkin**4 + (6469*mckin**6)/mbkin**6 - 
                (22516*mckin**8)/mbkin**8 - (106182*mckin**10)/mbkin**10 - 
                (107880*mckin**12)/mbkin**12 - (20198*mckin**14)/mbkin**14 + 
                (12998*mckin**16)/mbkin**16 - (3867*mckin**18)/mbkin**18 - 
                (910*mckin**20)/mbkin**20 + (145*mckin**22)/mbkin**22)*q_cut)/
              mbkin**2 - (4*(15 - (95*mckin**2)/mbkin**2 - (89*mckin**4)/
                 mbkin**4 + (2989*mckin**6)/mbkin**6 + (4773*mckin**8)/mbkin**8 + 
                (2337*mckin**10)/mbkin**10 + (10585*mckin**12)/mbkin**12 + 
                (2765*mckin**14)/mbkin**14 - (2854*mckin**16)/mbkin**16 - 
                (416*mckin**18)/mbkin**18 + (150*mckin**20)/mbkin**20)*q_cut**2)/
              mbkin**4 - (2*(33 - (229*mckin**2)/mbkin**2 - (526*mckin**4)/
                 mbkin**4 + (11552*mckin**6)/mbkin**6 + (19468*mckin**8)/
                 mbkin**8 + (17570*mckin**10)/mbkin**10 + (6430*mckin**12)/
                 mbkin**12 - (8984*mckin**14)/mbkin**14 - (1277*mckin**16)/
                 mbkin**16 + (315*mckin**18)/mbkin**18)*q_cut**3)/mbkin**6 + 
             (2*(84 - (301*mckin**2)/mbkin**2 - (1280*mckin**4)/mbkin**4 + 
                (11016*mckin**6)/mbkin**6 + (26043*mckin**8)/mbkin**8 + 
                (1603*mckin**10)/mbkin**10 - (11796*mckin**12)/mbkin**12 - 
                (692*mckin**14)/mbkin**14 + (795*mckin**16)/mbkin**16)*q_cut**4)/
              mbkin**8 - (2*(21 - (37*mckin**2)/mbkin**2 - (145*mckin**4)/
                 mbkin**4 + (2777*mckin**6)/mbkin**6 + (927*mckin**8)/mbkin**8 - 
                (2211*mckin**10)/mbkin**10 + (413*mckin**12)/mbkin**12 + 
                (255*mckin**14)/mbkin**14)*q_cut**5)/mbkin**10 + 
             ((-126 + (92*mckin**2)/mbkin**2 + (1574*mckin**4)/mbkin**4 + 
                (2084*mckin**6)/mbkin**6 + (2298*mckin**8)/mbkin**8 - 
                (560*mckin**10)/mbkin**10 - (930*mckin**12)/mbkin**12)*q_cut**6)/
              mbkin**12 + (2*(45 - (25*mckin**2)/mbkin**2 - (476*mckin**4)/
                 mbkin**4 - (338*mckin**6)/mbkin**6 + (523*mckin**8)/mbkin**8 + 
                (375*mckin**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
             ((21 + (77*mckin**2)/mbkin**2 + (272*mckin**4)/mbkin**4 - 
                (64*mckin**6)/mbkin**6 + (30*mckin**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
             (2*(19 + (19*mckin**2)/mbkin**2 + (63*mckin**4)/mbkin**4 + 
                (95*mckin**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
             (10*(mbkin**4 + 5*mckin**4)*q_cut**10)/mbkin**24)*rhoD + 180*sB - 
           (300*mckin**2*sB)/mbkin**2 - (44160*mckin**4*sB)/mbkin**4 + 
           (177168*mckin**6*sB)/mbkin**6 + (875160*mckin**8*sB)/mbkin**8 - 
           (698568*mckin**10*sB)/mbkin**10 - (945120*mckin**12*sB)/mbkin**12 + 
           (23040*mckin**14*sB)/mbkin**14 + (261012*mckin**16*sB)/mbkin**16 + 
           (525300*mckin**18*sB)/mbkin**18 - (150432*mckin**20*sB)/mbkin**20 - 
           (26640*mckin**22*sB)/mbkin**22 + (3360*mckin**24*sB)/mbkin**24 - 
           (720*q_cut*sB)/mbkin**2 - (2400*mckin**2*q_cut*sB)/mbkin**4 + 
           (134064*mckin**4*q_cut*sB)/mbkin**6 + (54816*mckin**6*q_cut*sB)/mbkin**8 - 
           (2573472*mckin**8*q_cut*sB)/mbkin**10 - (5234112*mckin**10*q_cut*sB)/
            mbkin**12 - (4081632*mckin**12*q_cut*sB)/mbkin**14 - 
           (1732416*mckin**14*q_cut*sB)/mbkin**16 - (211728*mckin**16*q_cut*sB)/
            mbkin**18 + (531360*mckin**18*q_cut*sB)/mbkin**20 + 
           (66480*mckin**20*q_cut*sB)/mbkin**22 - (13920*mckin**22*q_cut*sB)/
            mbkin**24 + (720*q_cut**2*sB)/mbkin**4 + (6000*mckin**2*q_cut**2*sB)/
            mbkin**6 - (66864*mckin**4*q_cut**2*sB)/mbkin**8 - 
           (258384*mckin**6*q_cut**2*sB)/mbkin**10 - (20016*mckin**8*q_cut**2*sB)/
            mbkin**12 + (37296*mckin**10*q_cut**2*sB)/mbkin**14 - 
           (112944*mckin**12*q_cut**2*sB)/mbkin**16 - (362832*mckin**14*q_cut**2*sB)/
            mbkin**18 - (303936*mckin**16*q_cut**2*sB)/mbkin**20 - 
           (22080*mckin**18*q_cut**2*sB)/mbkin**22 + (14400*mckin**20*q_cut**2*sB)/
            mbkin**24 + (720*q_cut**3*sB)/mbkin**6 + (5280*mckin**2*q_cut**3*sB)/
            mbkin**8 - (106176*mckin**4*q_cut**3*sB)/mbkin**10 - 
           (524784*mckin**6*q_cut**3*sB)/mbkin**12 - (866112*mckin**8*q_cut**3*sB)/
            mbkin**14 - (1283568*mckin**10*q_cut**3*sB)/mbkin**16 - 
           (1067328*mckin**12*q_cut**3*sB)/mbkin**18 - (541008*mckin**14*q_cut**3*sB)/
            mbkin**20 - (35088*mckin**16*q_cut**3*sB)/mbkin**22 + 
           (15120*mckin**18*q_cut**3*sB)/mbkin**24 - (1800*q_cut**4*sB)/mbkin**8 - 
           (20040*mckin**2*q_cut**4*sB)/mbkin**10 + (86640*mckin**4*q_cut**4*sB)/
            mbkin**12 + (603360*mckin**6*q_cut**4*sB)/mbkin**14 + 
           (1157832*mckin**8*q_cut**4*sB)/mbkin**16 + (1188024*mckin**10*q_cut**4*sB)/
            mbkin**18 + (561840*mckin**12*q_cut**4*sB)/mbkin**20 - 
           (17472*mckin**14*q_cut**4*sB)/mbkin**22 - (38160*mckin**16*q_cut**4*sB)/
            mbkin**24 + (720*q_cut**5*sB)/mbkin**10 + (8160*mckin**2*q_cut**5*sB)/
            mbkin**12 - (16608*mckin**4*q_cut**5*sB)/mbkin**14 - 
           (95088*mckin**6*q_cut**5*sB)/mbkin**16 - (156240*mckin**8*q_cut**5*sB)/
            mbkin**18 - (82368*mckin**10*q_cut**5*sB)/mbkin**20 + 
           (31200*mckin**12*q_cut**5*sB)/mbkin**22 + (12240*mckin**14*q_cut**5*sB)/
            mbkin**24 + (720*q_cut**6*sB)/mbkin**12 + (11760*mckin**2*q_cut**6*sB)/
            mbkin**14 + (45888*mckin**4*q_cut**6*sB)/mbkin**16 + 
           (59088*mckin**6*q_cut**6*sB)/mbkin**18 + (32640*mckin**8*q_cut**6*sB)/
            mbkin**20 + (39360*mckin**10*q_cut**6*sB)/mbkin**22 + 
           (22320*mckin**12*q_cut**6*sB)/mbkin**24 - (720*q_cut**7*sB)/mbkin**14 - 
           (11040*mckin**2*q_cut**7*sB)/mbkin**16 - (40224*mckin**4*q_cut**7*sB)/
            mbkin**18 - (52944*mckin**6*q_cut**7*sB)/mbkin**20 - 
           (42960*mckin**8*q_cut**7*sB)/mbkin**22 - (18000*mckin**10*q_cut**7*sB)/
            mbkin**24 + (180*q_cut**8*sB)/mbkin**16 + (2580*mckin**2*q_cut**8*sB)/
            mbkin**18 + (4848*mckin**4*q_cut**8*sB)/mbkin**20 + 
           (3408*mckin**6*q_cut**8*sB)/mbkin**22 - (720*mckin**8*q_cut**8*sB)/mbkin**24 + 
           (3792*mckin**4*q_cut**9*sB)/mbkin**22 + (4560*mckin**6*q_cut**9*sB)/
            mbkin**24 - (1200*mckin**4*q_cut**10*sB)/mbkin**24 - 224*sE + 
           (1168*mckin**2*sE)/mbkin**2 + (30464*mckin**4*sE)/mbkin**4 - 
           (154784*mckin**6*sE)/mbkin**6 - (282656*mckin**8*sE)/mbkin**8 + 
           (506464*mckin**10*sE)/mbkin**10 + (114688*mckin**12*sE)/mbkin**12 - 
           (93248*mckin**14*sE)/mbkin**14 + (51424*mckin**16*sE)/mbkin**16 - 
           (275184*mckin**18*sE)/mbkin**18 + (88320*mckin**20*sE)/mbkin**20 + 
           (15584*mckin**22*sE)/mbkin**22 - (2016*mckin**24*sE)/mbkin**24 + 
           (928*q_cut*sE)/mbkin**2 - (1056*mckin**2*q_cut*sE)/mbkin**4 - 
           (98592*mckin**4*q_cut*sE)/mbkin**6 + (53920*mckin**6*q_cut*sE)/mbkin**8 + 
           (1396416*mckin**8*q_cut*sE)/mbkin**10 + (1653312*mckin**10*q_cut*sE)/
            mbkin**12 + (1115200*mckin**12*q_cut*sE)/mbkin**14 + 
           (512448*mckin**14*q_cut*sE)/mbkin**16 + (6816*mckin**16*q_cut*sE)/
            mbkin**18 - (325920*mckin**18*q_cut*sE)/mbkin**20 - 
           (36384*mckin**20*q_cut*sE)/mbkin**22 + (8352*mckin**22*q_cut*sE)/mbkin**24 - 
           (960*q_cut**2*sE)/mbkin**4 - (2880*mckin**2*q_cut**2*sE)/mbkin**6 + 
           (53184*mckin**4*q_cut**2*sE)/mbkin**8 + (146112*mckin**6*q_cut**2*sE)/
            mbkin**10 - (116352*mckin**8*q_cut**2*sE)/mbkin**12 - 
           (159168*mckin**10*q_cut**2*sE)/mbkin**14 + (324864*mckin**12*q_cut**2*sE)/
            mbkin**16 + (547008*mckin**14*q_cut**2*sE)/mbkin**18 + 
           (216384*mckin**16*q_cut**2*sE)/mbkin**20 + (2688*mckin**18*q_cut**2*sE)/
            mbkin**22 - (8640*mckin**20*q_cut**2*sE)/mbkin**24 - 
           (1008*q_cut**3*sE)/mbkin**6 - (1776*mckin**2*q_cut**3*sE)/mbkin**8 + 
           (89088*mckin**4*q_cut**3*sE)/mbkin**10 + (285888*mckin**6*q_cut**3*sE)/
            mbkin**12 + (308256*mckin**8*q_cut**3*sE)/mbkin**14 + 
           (170720*mckin**10*q_cut**3*sE)/mbkin**16 + (181376*mckin**12*q_cut**3*sE)/
            mbkin**18 + (290240*mckin**14*q_cut**3*sE)/mbkin**20 + 
           (33360*mckin**16*q_cut**3*sE)/mbkin**22 - (9072*mckin**18*q_cut**3*sE)/
            mbkin**24 + (2544*q_cut**4*sE)/mbkin**8 + (13344*mckin**2*q_cut**4*sE)/
            mbkin**10 - (85824*mckin**4*q_cut**4*sE)/mbkin**12 - 
           (367680*mckin**6*q_cut**4*sE)/mbkin**14 - (424416*mckin**8*q_cut**4*sE)/
            mbkin**16 - (497376*mckin**10*q_cut**4*sE)/mbkin**18 - 
           (324160*mckin**12*q_cut**4*sE)/mbkin**20 + (11712*mckin**14*q_cut**4*sE)/
            mbkin**22 + (22896*mckin**16*q_cut**4*sE)/mbkin**24 - 
           (816*q_cut**5*sE)/mbkin**10 - (6768*mckin**2*q_cut**5*sE)/mbkin**12 + 
           (13680*mckin**4*q_cut**5*sE)/mbkin**14 + (54576*mckin**6*q_cut**5*sE)/
            mbkin**16 + (68016*mckin**8*q_cut**5*sE)/mbkin**18 + 
           (26736*mckin**10*q_cut**5*sE)/mbkin**20 - (35184*mckin**12*q_cut**5*sE)/
            mbkin**22 - (7344*mckin**14*q_cut**5*sE)/mbkin**24 - 
           (1488*q_cut**6*sE)/mbkin**12 - (9024*mckin**2*q_cut**6*sE)/mbkin**14 - 
           (17904*mckin**4*q_cut**6*sE)/mbkin**16 - (13888*mckin**6*q_cut**6*sE)/
            mbkin**18 + (10256*mckin**8*q_cut**6*sE)/mbkin**20 - 
           (9216*mckin**10*q_cut**6*sE)/mbkin**22 - (13392*mckin**12*q_cut**6*sE)/
            mbkin**24 + (1200*q_cut**7*sE)/mbkin**14 + (8880*mckin**2*q_cut**7*sE)/
            mbkin**16 + (19680*mckin**4*q_cut**7*sE)/mbkin**18 + 
           (21152*mckin**6*q_cut**7*sE)/mbkin**20 + (22512*mckin**8*q_cut**7*sE)/
            mbkin**22 + (10800*mckin**10*q_cut**7*sE)/mbkin**24 + 
           (48*q_cut**8*sE)/mbkin**16 - (1584*mckin**2*q_cut**8*sE)/mbkin**18 - 
           (2784*mckin**4*q_cut**8*sE)/mbkin**20 - (3360*mckin**6*q_cut**8*sE)/
            mbkin**22 + (432*mckin**8*q_cut**8*sE)/mbkin**24 - (304*q_cut**9*sE)/
            mbkin**18 - (304*mckin**2*q_cut**9*sE)/mbkin**20 - 
           (1712*mckin**4*q_cut**9*sE)/mbkin**22 - (2736*mckin**6*q_cut**9*sE)/
            mbkin**24 + (80*q_cut**10*sE)/mbkin**20 + (720*mckin**4*q_cut**10*sE)/
            mbkin**24 - 17*sqB + (49*mckin**2*sqB)/mbkin**2 + 
           (5132*mckin**4*sqB)/mbkin**4 - (20288*mckin**6*sqB)/mbkin**6 - 
           (146474*mckin**8*sqB)/mbkin**8 + (48502*mckin**10*sqB)/mbkin**10 + 
           (253600*mckin**12*sqB)/mbkin**12 - (4568*mckin**14*sqB)/mbkin**14 - 
           (119537*mckin**16*sqB)/mbkin**16 - (24231*mckin**18*sqB)/mbkin**18 + 
           (7380*mckin**20*sqB)/mbkin**20 + (536*mckin**22*sqB)/mbkin**22 - 
           (84*mckin**24*sqB)/mbkin**24 + (64*q_cut*sqB)/mbkin**2 + 
           (252*mckin**2*q_cut*sqB)/mbkin**4 - (15648*mckin**4*q_cut*sqB)/mbkin**6 - 
           (12404*mckin**6*q_cut*sqB)/mbkin**8 + (354816*mckin**8*q_cut*sqB)/
            mbkin**10 + (956568*mckin**10*q_cut*sqB)/mbkin**12 + 
           (871600*mckin**12*q_cut*sqB)/mbkin**14 + (308568*mckin**14*q_cut*sqB)/
            mbkin**16 - (19776*mckin**16*q_cut*sqB)/mbkin**18 - 
           (24276*mckin**18*q_cut*sqB)/mbkin**20 - (912*mckin**20*q_cut*sqB)/
            mbkin**22 + (348*mckin**22*q_cut*sqB)/mbkin**24 - (60*q_cut**2*sqB)/
            mbkin**4 - (660*mckin**2*q_cut**2*sqB)/mbkin**6 + 
           (7428*mckin**4*q_cut**2*sqB)/mbkin**8 + (34908*mckin**6*q_cut**2*sqB)/
            mbkin**10 + (21180*mckin**8*q_cut**2*sqB)/mbkin**12 + 
           (4812*mckin**10*q_cut**2*sqB)/mbkin**14 + (61356*mckin**12*q_cut**2*sqB)/
            mbkin**16 + (60252*mckin**14*q_cut**2*sqB)/mbkin**18 + 
           (13416*mckin**16*q_cut**2*sqB)/mbkin**20 - (672*mckin**18*q_cut**2*sqB)/
            mbkin**22 - (360*mckin**20*q_cut**2*sqB)/mbkin**24 - 
           (54*q_cut**3*sqB)/mbkin**6 - (618*mckin**2*q_cut**3*sqB)/mbkin**8 + 
           (12876*mckin**4*q_cut**3*sqB)/mbkin**10 + (70464*mckin**6*q_cut**3*sqB)/
            mbkin**12 + (94992*mckin**8*q_cut**3*sqB)/mbkin**14 + 
           (94100*mckin**10*q_cut**3*sqB)/mbkin**16 + (69044*mckin**12*q_cut**3*sqB)/
            mbkin**18 + (21776*mckin**14*q_cut**3*sqB)/mbkin**20 + 
           (678*mckin**16*q_cut**3*sqB)/mbkin**22 - (378*mckin**18*q_cut**3*sqB)/
            mbkin**24 + (132*q_cut**4*sqB)/mbkin**8 + (2142*mckin**2*q_cut**4*sqB)/
            mbkin**10 - (10176*mckin**4*q_cut**4*sqB)/mbkin**12 - 
           (80640*mckin**6*q_cut**4*sqB)/mbkin**14 - (148506*mckin**8*q_cut**4*sqB)/
            mbkin**16 - (99042*mckin**10*q_cut**4*sqB)/mbkin**18 - 
           (20104*mckin**12*q_cut**4*sqB)/mbkin**20 + (2760*mckin**14*q_cut**4*sqB)/
            mbkin**22 + (954*mckin**16*q_cut**4*sqB)/mbkin**24 - 
           (78*q_cut**5*sqB)/mbkin**10 - (714*mckin**2*q_cut**5*sqB)/mbkin**12 + 
           (1782*mckin**4*q_cut**5*sqB)/mbkin**14 + (14034*mckin**6*q_cut**5*sqB)/
            mbkin**16 + (18006*mckin**8*q_cut**5*sqB)/mbkin**18 + 
           (1194*mckin**10*q_cut**5*sqB)/mbkin**20 - (2526*mckin**12*q_cut**5*sqB)/
            mbkin**22 - (306*mckin**14*q_cut**5*sqB)/mbkin**24 + 
           (6*q_cut**6*sqB)/mbkin**12 - (1332*mckin**2*q_cut**6*sqB)/mbkin**14 - 
           (4638*mckin**4*q_cut**6*sqB)/mbkin**16 - (5260*mckin**6*q_cut**6*sqB)/
            mbkin**18 - (3130*mckin**8*q_cut**6*sqB)/mbkin**20 - 
           (1488*mckin**10*q_cut**6*sqB)/mbkin**22 - (558*mckin**12*q_cut**6*sqB)/
            mbkin**24 + (30*q_cut**7*sqB)/mbkin**14 + (1170*mckin**2*q_cut**7*sqB)/
            mbkin**16 + (4176*mckin**4*q_cut**7*sqB)/mbkin**18 + 
           (4628*mckin**6*q_cut**7*sqB)/mbkin**20 + (2202*mckin**8*q_cut**7*sqB)/
            mbkin**22 + (450*mckin**10*q_cut**7*sqB)/mbkin**24 - 
           (51*q_cut**8*sqB)/mbkin**16 - (327*mckin**2*q_cut**8*sqB)/mbkin**18 - 
           (912*mckin**4*q_cut**8*sqB)/mbkin**20 - (528*mckin**6*q_cut**8*sqB)/
            mbkin**22 + (18*mckin**8*q_cut**8*sqB)/mbkin**24 + (38*q_cut**9*sqB)/
            mbkin**18 + (38*mckin**2*q_cut**9*sqB)/mbkin**20 - (50*mckin**4*q_cut**9*sqB)/
            mbkin**22 - (114*mckin**6*q_cut**9*sqB)/mbkin**24 - (10*q_cut**10*sqB)/
            mbkin**20 + (30*mckin**4*q_cut**10*sqB)/mbkin**24))*
        np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
              mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
           mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))) - 
       (144*mckin**4*(-16*(-((-1 + mckin**2/mbkin**2)**4*(-88 + (274*mckin**2)/
                mbkin**2 + (4140*mckin**4)/mbkin**4 + (919*mckin**6)/mbkin**6 - 
               (2041*mckin**8)/mbkin**8 - (1623*mckin**10)/mbkin**10 + 
               (975*mckin**12)/mbkin**12 + (504*mckin**14)/mbkin**14)) + 
            ((-1 + mckin**2/mbkin**2)**2*(-444 + (658*mckin**2)/mbkin**2 + 
               (16802*mckin**4)/mbkin**4 + (19797*mckin**6)/mbkin**6 - 
               (2506*mckin**8)/mbkin**8 - (12332*mckin**10)/mbkin**10 - 
               (5424*mckin**12)/mbkin**12 + (5337*mckin**14)/mbkin**14 + 
               (2592*mckin**16)/mbkin**16)*q_cut)/mbkin**2 - 
            (3*(-268 + (194*mckin**2)/mbkin**2 + (7472*mckin**4)/mbkin**4 + 
               (7533*mckin**6)/mbkin**6 + (6459*mckin**8)/mbkin**8 + 
               (256*mckin**10)/mbkin**10 - (6086*mckin**12)/mbkin**12 - 
               (3839*mckin**14)/mbkin**14 + (3015*mckin**16)/mbkin**16 + 
               (1584*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
            ((-438 - (218*mckin**2)/mbkin**2 + (10234*mckin**4)/mbkin**4 + 
               (12285*mckin**6)/mbkin**6 - (8678*mckin**8)/mbkin**8 - 
               (11348*mckin**10)/mbkin**10 + (5918*mckin**12)/mbkin**12 + 
               (11121*mckin**14)/mbkin**14 + (2484*mckin**16)/mbkin**16)*q_cut**3)/
             mbkin**6 + ((-480 - (2270*mckin**2)/mbkin**2 - (1602*mckin**4)/
                mbkin**4 + (4773*mckin**6)/mbkin**6 + (7609*mckin**8)/mbkin**8 + 
               (2511*mckin**10)/mbkin**10 + (5895*mckin**12)/mbkin**12 + 
               (3240*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 - 
            (3*(-274 - (1282*mckin**2)/mbkin**2 - (488*mckin**4)/mbkin**4 + 
               (2601*mckin**6)/mbkin**6 + (5406*mckin**8)/mbkin**8 + 
               (5847*mckin**10)/mbkin**10 + (1692*mckin**12)/mbkin**12)*q_cut**5)/
             mbkin**10 + ((-396 - (1946*mckin**2)/mbkin**2 + (166*mckin**4)/
                mbkin**4 + (7217*mckin**6)/mbkin**6 + (9357*mckin**8)/mbkin**8 + 
               (1728*mckin**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
            ((6 + (202*mckin**2)/mbkin**2 + (24*mckin**4)/mbkin**4 - 
               (405*mckin**6)/mbkin**6 + (972*mckin**8)/mbkin**8)*q_cut**7)/
             mbkin**14 + (48*(mbkin**6 + mbkin**4*mckin**2 - 12*mbkin**2*mckin**4 - 
               18*mckin**6)*q_cut**8)/mbkin**22 - (10*(mbkin**4 - 18*mckin**4)*q_cut**9)/
             mbkin**22)*rE + ((-1 + mckin**2/mbkin**2)**2 - 
            (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
           ((-360*mckin**2*muG**2)/mbkin**2 + (1872*mckin**4*muG**2)/mbkin**4 - 
            (8100*mckin**6*muG**2)/mbkin**6 + (34668*mckin**8*muG**2)/mbkin**8 - 
            (15480*mckin**10*muG**2)/mbkin**10 - (54720*mckin**12*muG**2)/
             mbkin**12 + (47988*mckin**14*muG**2)/mbkin**14 - 
            (4860*mckin**16*muG**2)/mbkin**16 - (1008*mckin**18*muG**2)/mbkin**18 - 
            (360*mckin**2*muG*mupi)/mbkin**2 + (3168*mckin**4*muG*mupi)/
             mbkin**4 - (1980*mckin**6*muG*mupi)/mbkin**6 - 
            (34668*mckin**8*muG*mupi)/mbkin**8 + (40680*mckin**10*muG*mupi)/
             mbkin**10 + (19440*mckin**12*muG*mupi)/mbkin**12 - 
            (27828*mckin**14*muG*mupi)/mbkin**14 + (540*mckin**16*muG*mupi)/
             mbkin**16 + (1008*mckin**18*muG*mupi)/mbkin**18 - 
            (360*mckin**2*muG**2*q_cut)/mbkin**4 + (4968*mckin**4*muG**2*q_cut)/
             mbkin**6 + (2124*mckin**6*muG**2*q_cut)/mbkin**8 - 
            (43848*mckin**8*muG**2*q_cut)/mbkin**10 - (91728*mckin**10*muG**2*q_cut)/
             mbkin**12 - (75888*mckin**12*muG**2*q_cut)/mbkin**14 + 
            (20124*mckin**14*muG**2*q_cut)/mbkin**16 + (3168*mckin**16*muG**2*q_cut)/
             mbkin**18 + (1080*mckin**2*muG*mupi*q_cut)/mbkin**4 - 
            (6408*mckin**4*muG*mupi*q_cut)/mbkin**6 - (10764*mckin**6*muG*mupi*q_cut)/
             mbkin**8 + (58248*mckin**8*muG*mupi*q_cut)/mbkin**10 + 
            (103968*mckin**10*muG*mupi*q_cut)/mbkin**12 + 
            (45648*mckin**12*muG*mupi*q_cut)/mbkin**14 - 
            (7164*mckin**14*muG*mupi*q_cut)/mbkin**16 - (3168*mckin**16*muG*mupi*
              q_cut)/mbkin**18 + (720*mckin**2*muG**2*q_cut**2)/mbkin**6 - 
            (1440*mckin**4*muG**2*q_cut**2)/mbkin**8 - (12384*mckin**6*muG**2*q_cut**2)/
             mbkin**10 + (14904*mckin**8*muG**2*q_cut**2)/mbkin**12 + 
            (30024*mckin**10*muG**2*q_cut**2)/mbkin**14 - (14544*mckin**12*muG**2*
              q_cut**2)/mbkin**16 - (2160*mckin**14*muG**2*q_cut**2)/mbkin**18 - 
            (720*mckin**2*muG*mupi*q_cut**2)/mbkin**6 + (2880*mckin**4*muG*mupi*
              q_cut**2)/mbkin**8 + (6624*mckin**6*muG*mupi*q_cut**2)/mbkin**10 - 
            (16344*mckin**8*muG*mupi*q_cut**2)/mbkin**12 - 
            (15624*mckin**10*muG*mupi*q_cut**2)/mbkin**14 + 
            (5904*mckin**12*muG*mupi*q_cut**2)/mbkin**16 + 
            (2160*mckin**14*muG*mupi*q_cut**2)/mbkin**18 + 
            (2160*mckin**2*muG**2*q_cut**3)/mbkin**8 - (2880*mckin**4*muG**2*q_cut**3)/
             mbkin**10 - (27864*mckin**6*muG**2*q_cut**3)/mbkin**12 - 
            (42048*mckin**8*muG**2*q_cut**3)/mbkin**14 - (22104*mckin**10*muG**2*q_cut**3)/
             mbkin**16 - (2520*mckin**12*muG**2*q_cut**3)/mbkin**18 - 
            (720*mckin**2*muG*mupi*q_cut**3)/mbkin**8 + (2880*mckin**4*muG*mupi*
              q_cut**3)/mbkin**10 + (14904*mckin**6*muG*mupi*q_cut**3)/mbkin**12 + 
            (21888*mckin**8*muG*mupi*q_cut**3)/mbkin**14 + 
            (13464*mckin**10*muG*mupi*q_cut**3)/mbkin**16 + 
            (2520*mckin**12*muG*mupi*q_cut**3)/mbkin**18 - 
            (3240*mckin**2*muG**2*q_cut**4)/mbkin**10 + (6480*mckin**4*muG**2*q_cut**4)/
             mbkin**12 + (38196*mckin**6*muG**2*q_cut**4)/mbkin**14 + 
            (30636*mckin**8*muG**2*q_cut**4)/mbkin**16 + (3600*mckin**10*muG**2*q_cut**4)/
             mbkin**18 + (1080*mckin**2*muG*mupi*q_cut**4)/mbkin**10 - 
            (4320*mckin**4*muG*mupi*q_cut**4)/mbkin**12 - 
            (20916*mckin**6*muG*mupi*q_cut**4)/mbkin**14 - 
            (17676*mckin**8*muG*mupi*q_cut**4)/mbkin**16 - 
            (3600*mckin**10*muG*mupi*q_cut**4)/mbkin**18 + 
            (1080*mckin**2*muG**2*q_cut**5)/mbkin**12 - (2592*mckin**4*muG**2*q_cut**5)/
             mbkin**14 - (8244*mckin**6*muG**2*q_cut**5)/mbkin**16 - 
            (432*mckin**8*muG**2*q_cut**5)/mbkin**18 - (360*mckin**2*muG*mupi*q_cut**5)/
             mbkin**12 + (1152*mckin**4*muG*mupi*q_cut**5)/mbkin**14 + 
            (3924*mckin**6*muG*mupi*q_cut**5)/mbkin**16 + 
            (432*mckin**8*muG*mupi*q_cut**5)/mbkin**18 - (1008*mckin**4*muG**2*q_cut**6)/
             mbkin**16 - (1008*mckin**6*muG**2*q_cut**6)/mbkin**18 + 
            (1008*mckin**4*muG*mupi*q_cut**6)/mbkin**16 + 
            (1008*mckin**6*muG*mupi*q_cut**6)/mbkin**18 + (360*mckin**4*muG**2*q_cut**7)/
             mbkin**18 - (360*mckin**4*muG*mupi*q_cut**7)/mbkin**18 + 
            72*mckin**2*muG*((-1 + mckin**2/mbkin**2)**2*(-10 + (68*mckin**2)/
                 mbkin**2 + (91*mckin**4)/mbkin**4 - (849*mckin**6)/mbkin**6 - 
                (659*mckin**8)/mbkin**8 + (71*mckin**10)/mbkin**10 + 
                (28*mckin**12)/mbkin**12) + ((30 - (178*mckin**2)/mbkin**2 - 
                 (299*mckin**4)/mbkin**4 + (1618*mckin**6)/mbkin**6 + 
                 (2888*mckin**8)/mbkin**8 + (1268*mckin**10)/mbkin**10 - 
                 (199*mckin**12)/mbkin**12 - (88*mckin**14)/mbkin**14)*q_cut)/mbkin**
                2 + (2*(-10 + (40*mckin**2)/mbkin**2 + (92*mckin**4)/mbkin**4 - 
                 (227*mckin**6)/mbkin**6 - (217*mckin**8)/mbkin**8 + 
                 (82*mckin**10)/mbkin**10 + (30*mckin**12)/mbkin**12)*q_cut**2)/mbkin**
                4 + ((-20 + (80*mckin**2)/mbkin**2 + (414*mckin**4)/mbkin**4 + 
                 (608*mckin**6)/mbkin**6 + (374*mckin**8)/mbkin**8 + 
                 (70*mckin**10)/mbkin**10)*q_cut**3)/mbkin**6 - 
              ((-30 + (120*mckin**2)/mbkin**2 + (581*mckin**4)/mbkin**4 + 
                 (491*mckin**6)/mbkin**6 + (100*mckin**8)/mbkin**8)*q_cut**4)/mbkin**
                8 + ((-10 + (32*mckin**2)/mbkin**2 + (109*mckin**4)/mbkin**4 + 
                 (12*mckin**6)/mbkin**6)*q_cut**5)/mbkin**10 + (28*mckin**2*
                (mbkin**2 + mckin**2)*q_cut**6)/mbkin**16 - (10*mckin**2*q_cut**7)/mbkin**
                16) + 4*(-((-1 + mckin**2/mbkin**2)**2*(-82 - (224*mckin**2)/
                  mbkin**2 + (4995*mckin**4)/mbkin**4 - (3584*mckin**6)/mbkin**6 - 
                 (17674*mckin**8)/mbkin**8 - (5232*mckin**10)/mbkin**10 + 
                 (1305*mckin**12)/mbkin**12 + (336*mckin**14)/mbkin**14)) + 
              ((-262 - (926*mckin**2)/mbkin**2 + (9471*mckin**4)/mbkin**4 + 
                 (3031*mckin**6)/mbkin**6 - (41278*mckin**8)/mbkin**8 - 
                 (45366*mckin**10)/mbkin**10 - (9987*mckin**12)/mbkin**12 + 
                 (3621*mckin**14)/mbkin**14 + (1056*mckin**16)/mbkin**16)*q_cut)/
               mbkin**2 + ((180 + (996*mckin**2)/mbkin**2 - (3076*mckin**4)/
                  mbkin**4 + (2042*mckin**6)/mbkin**6 + (9992*mckin**8)/mbkin**8 + 
                 (594*mckin**10)/mbkin**10 - (3288*mckin**12)/mbkin**12 - 
                 (720*mckin**14)/mbkin**14)*q_cut**2)/mbkin**4 - 
              (2*(-110 - (688*mckin**2)/mbkin**2 + (59*mckin**4)/mbkin**4 + 
                 (1248*mckin**6)/mbkin**6 + (2690*mckin**8)/mbkin**8 + 
                 (1869*mckin**10)/mbkin**10 + (420*mckin**12)/mbkin**12)*q_cut**3)/
               mbkin**6 + ((-310 - (2224*mckin**2)/mbkin**2 + (641*mckin**4)/
                  mbkin**4 + (7288*mckin**6)/mbkin**6 + (5757*mckin**8)/mbkin**8 + 
                 (1200*mckin**10)/mbkin**10)*q_cut**4)/mbkin**8 - 
              (3*(-6 - (202*mckin**2)/mbkin**2 + (331*mckin**4)/mbkin**4 + 
                 (485*mckin**6)/mbkin**6 + (48*mckin**8)/mbkin**8)*q_cut**5)/mbkin**
                10 - (8*(-14 - (14*mckin**2)/mbkin**2 + (33*mckin**4)/mbkin**4 + 
                 (42*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 - 
              (40*(mbkin**4 - 3*mckin**4)*q_cut**7)/mbkin**18)*rG - 
            24*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(-2 - (44*mckin**2)/mbkin**2 + 
                 (729*mckin**4)/mbkin**4 + (9244*mckin**6)/mbkin**6 + 
                 (10134*mckin**8)/mbkin**8 + (36*mckin**10)/mbkin**10 - 
                 (77*mckin**12)/mbkin**12 + (140*mckin**14)/mbkin**14)) + 
              ((-2 - (226*mckin**2)/mbkin**2 + (1025*mckin**4)/mbkin**4 + 
                 (19673*mckin**6)/mbkin**6 + (39598*mckin**8)/mbkin**8 + 
                 (21350*mckin**10)/mbkin**10 - (861*mckin**12)/mbkin**12 - 
                 (357*mckin**14)/mbkin**14 + (440*mckin**16)/mbkin**16)*q_cut)/mbkin**
                2 - (2*mckin**2*(-88 - (232*mckin**2)/mbkin**2 + (2115*mckin**4)/
                  mbkin**4 + (1978*mckin**6)/mbkin**6 - (633*mckin**8)/mbkin**8 + 
                 (70*mckin**10)/mbkin**10 + (150*mckin**12)/mbkin**12)*q_cut**2)/
               mbkin**6 + ((-10 + (216*mckin**2)/mbkin**2 + (1252*mckin**4)/
                  mbkin**4 + (972*mckin**6)/mbkin**6 + (866*mckin**8)/mbkin**8 - 
                 (370*mckin**10)/mbkin**10 - (350*mckin**12)/mbkin**12)*q_cut**3)/
               mbkin**6 + ((10 - (284*mckin**2)/mbkin**2 - (2029*mckin**4)/
                  mbkin**4 - (1764*mckin**6)/mbkin**6 + (735*mckin**8)/mbkin**8 + 
                 (500*mckin**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
              ((18 + (106*mckin**2)/mbkin**2 + (601*mckin**4)/mbkin**4 - 
                 (157*mckin**6)/mbkin**6 - (60*mckin**8)/mbkin**8)*q_cut**5)/mbkin**
                10 - (4*(7 + (7*mckin**2)/mbkin**2 + (17*mckin**4)/mbkin**4 + 
                 (35*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
              (10*(mbkin**4 + 5*mckin**4)*q_cut**7)/mbkin**18)*rhoD - 360*sB - 
            (1920*mckin**2*sB)/mbkin**2 + (45204*mckin**4*sB)/mbkin**4 + 
            (44976*mckin**6*sB)/mbkin**6 - (128580*mckin**8*sB)/mbkin**8 - 
            (16800*mckin**10*sB)/mbkin**10 + (19980*mckin**12*sB)/mbkin**12 + 
            (41616*mckin**14*sB)/mbkin**14 - (756*mckin**16*sB)/mbkin**16 - 
            (3360*mckin**18*sB)/mbkin**18 + (1080*q_cut*sB)/mbkin**2 + 
            (10440*mckin**2*q_cut*sB)/mbkin**4 - (74364*mckin**4*q_cut*sB)/mbkin**6 - 
            (348228*mckin**6*q_cut*sB)/mbkin**8 - (436248*mckin**8*q_cut*sB)/
             mbkin**10 - (214728*mckin**10*q_cut*sB)/mbkin**12 - 
            (55908*mckin**12*q_cut*sB)/mbkin**14 + (18756*mckin**14*q_cut*sB)/
             mbkin**16 + (10560*mckin**16*q_cut*sB)/mbkin**18 - (720*q_cut**2*sB)/
             mbkin**4 - (8880*mckin**2*q_cut**2*sB)/mbkin**6 + 
            (13200*mckin**4*q_cut**2*sB)/mbkin**8 + (67512*mckin**6*q_cut**2*sB)/
             mbkin**10 + (40176*mckin**8*q_cut**2*sB)/mbkin**12 + 
            (5832*mckin**10*q_cut**2*sB)/mbkin**14 - (19200*mckin**12*q_cut**2*sB)/
             mbkin**16 - (7200*mckin**14*q_cut**2*sB)/mbkin**18 - 
            (720*q_cut**3*sB)/mbkin**6 - (11040*mckin**2*q_cut**3*sB)/mbkin**8 - 
            (26280*mckin**4*q_cut**3*sB)/mbkin**10 - (41328*mckin**6*q_cut**3*sB)/
             mbkin**12 - (50448*mckin**8*q_cut**3*sB)/mbkin**14 - 
            (30120*mckin**10*q_cut**3*sB)/mbkin**16 - (8400*mckin**12*q_cut**3*sB)/
             mbkin**18 + (1080*q_cut**4*sB)/mbkin**8 + (16560*mckin**2*q_cut**4*sB)/
             mbkin**10 + (43980*mckin**4*q_cut**4*sB)/mbkin**12 + 
            (60792*mckin**6*q_cut**4*sB)/mbkin**14 + (44100*mckin**8*q_cut**4*sB)/
             mbkin**16 + (12000*mckin**10*q_cut**4*sB)/mbkin**18 - 
            (360*q_cut**5*sB)/mbkin**10 - (5160*mckin**2*q_cut**5*sB)/mbkin**12 - 
            (9804*mckin**4*q_cut**5*sB)/mbkin**14 - (10284*mckin**6*q_cut**5*sB)/
             mbkin**16 - (1440*mckin**8*q_cut**5*sB)/mbkin**18 - 
            (2496*mckin**4*q_cut**6*sB)/mbkin**16 - (3360*mckin**6*q_cut**6*sB)/
             mbkin**18 + (1200*mckin**4*q_cut**7*sB)/mbkin**18 + 224*sE + 
            (1920*mckin**2*sE)/mbkin**2 - (28320*mckin**4*sE)/mbkin**4 - 
            (3568*mckin**6*sE)/mbkin**6 + (70704*mckin**8*sE)/mbkin**8 - 
            (36384*mckin**10*sE)/mbkin**10 + (7520*mckin**12*sE)/mbkin**12 - 
            (14160*mckin**14*sE)/mbkin**14 + (48*mckin**16*sE)/mbkin**16 + 
            (2016*mckin**18*sE)/mbkin**18 - (704*q_cut*sE)/mbkin**2 - 
            (8032*mckin**2*q_cut*sE)/mbkin**4 + (45120*mckin**4*q_cut*sE)/mbkin**6 + 
            (159920*mckin**6*q_cut*sE)/mbkin**8 + (127264*mckin**8*q_cut*sE)/
             mbkin**10 + (38208*mckin**10*q_cut*sE)/mbkin**12 + 
            (13536*mckin**12*q_cut*sE)/mbkin**14 - (11856*mckin**14*q_cut*sE)/
             mbkin**16 - (6336*mckin**16*q_cut*sE)/mbkin**18 + (480*q_cut**2*sE)/
             mbkin**4 + (6432*mckin**2*q_cut**2*sE)/mbkin**6 - 
            (10592*mckin**4*q_cut**2*sE)/mbkin**8 - (34400*mckin**6*q_cut**2*sE)/
             mbkin**10 - (4544*mckin**8*q_cut**2*sE)/mbkin**12 + 
            (17856*mckin**10*q_cut**2*sE)/mbkin**14 + (17568*mckin**12*q_cut**2*sE)/
             mbkin**16 + (4320*mckin**14*q_cut**2*sE)/mbkin**18 + 
            (560*q_cut**3*sE)/mbkin**6 + (7552*mckin**2*q_cut**3*sE)/mbkin**8 + 
            (12304*mckin**4*q_cut**3*sE)/mbkin**10 + (10944*mckin**6*q_cut**3*sE)/
             mbkin**12 + (1840*mckin**8*q_cut**3*sE)/mbkin**14 + 
            (8448*mckin**10*q_cut**3*sE)/mbkin**16 + (5040*mckin**12*q_cut**3*sE)/
             mbkin**18 - (800*q_cut**4*sE)/mbkin**8 - (11648*mckin**2*q_cut**4*sE)/
             mbkin**10 - (20288*mckin**4*q_cut**4*sE)/mbkin**12 - 
            (18928*mckin**6*q_cut**4*sE)/mbkin**14 - (21552*mckin**8*q_cut**4*sE)/
             mbkin**16 - (7200*mckin**10*q_cut**4*sE)/mbkin**18 + 
            (96*q_cut**5*sE)/mbkin**10 + (3552*mckin**2*q_cut**5*sE)/mbkin**12 + 
            (4032*mckin**4*q_cut**5*sE)/mbkin**14 + (6480*mckin**6*q_cut**5*sE)/
             mbkin**16 + (864*mckin**8*q_cut**5*sE)/mbkin**18 + (224*q_cut**6*sE)/
             mbkin**12 + (224*mckin**2*q_cut**6*sE)/mbkin**14 + 
            (864*mckin**4*q_cut**6*sE)/mbkin**16 + (2016*mckin**6*q_cut**6*sE)/
             mbkin**18 - (80*q_cut**7*sE)/mbkin**14 - (720*mckin**4*q_cut**7*sE)/
             mbkin**18 + 62*sqB - (5685*mckin**4*sqB)/mbkin**4 - 
            (10018*mckin**6*sqB)/mbkin**6 + (16491*mckin**8*sqB)/mbkin**8 + 
            (15108*mckin**10*sqB)/mbkin**10 - (11035*mckin**12*sqB)/mbkin**12 - 
            (5430*mckin**14*sqB)/mbkin**14 + (423*mckin**16*sqB)/mbkin**16 + 
            (84*mckin**18*sqB)/mbkin**18 - (182*q_cut*sqB)/mbkin**2 - 
            (886*mckin**2*q_cut*sqB)/mbkin**4 + (9915*mckin**4*q_cut*sqB)/mbkin**6 + 
            (53819*mckin**6*q_cut*sqB)/mbkin**8 + (84730*mckin**8*q_cut*sqB)/
             mbkin**10 + (49362*mckin**10*q_cut*sqB)/mbkin**12 + 
            (7041*mckin**12*q_cut*sqB)/mbkin**14 - (1935*mckin**14*q_cut*sqB)/
             mbkin**16 - (264*mckin**16*q_cut*sqB)/mbkin**18 + (120*q_cut**2*sqB)/
             mbkin**4 + (936*mckin**2*q_cut**2*sqB)/mbkin**6 - 
            (1256*mckin**4*q_cut**2*sqB)/mbkin**8 - (10490*mckin**6*q_cut**2*sqB)/
             mbkin**10 - (8804*mckin**8*q_cut**2*sqB)/mbkin**12 + 
            (702*mckin**10*q_cut**2*sqB)/mbkin**14 + (1812*mckin**12*q_cut**2*sqB)/
             mbkin**16 + (180*mckin**14*q_cut**2*sqB)/mbkin**18 + 
            (110*q_cut**3*sqB)/mbkin**6 + (1336*mckin**2*q_cut**3*sqB)/mbkin**8 + 
            (3232*mckin**4*q_cut**3*sqB)/mbkin**10 + (3012*mckin**6*q_cut**3*sqB)/
             mbkin**12 + (2938*mckin**8*q_cut**3*sqB)/mbkin**14 + 
            (1482*mckin**10*q_cut**3*sqB)/mbkin**16 + (210*mckin**12*q_cut**3*sqB)/
             mbkin**18 - (170*q_cut**4*sqB)/mbkin**8 - (1964*mckin**2*q_cut**4*sqB)/
             mbkin**10 - (5759*mckin**4*q_cut**4*sqB)/mbkin**12 - 
            (6484*mckin**6*q_cut**4*sqB)/mbkin**14 - (2763*mckin**8*q_cut**4*sqB)/
             mbkin**16 - (300*mckin**10*q_cut**4*sqB)/mbkin**18 + 
            (78*q_cut**5*sqB)/mbkin**10 + (606*mckin**2*q_cut**5*sqB)/mbkin**12 + 
            (1611*mckin**4*q_cut**5*sqB)/mbkin**14 + (969*mckin**6*q_cut**5*sqB)/
             mbkin**16 + (36*mckin**8*q_cut**5*sqB)/mbkin**18 - (28*q_cut**6*sqB)/
             mbkin**12 - (28*mckin**2*q_cut**6*sqB)/mbkin**14 + 
            (12*mckin**4*q_cut**6*sqB)/mbkin**16 + (84*mckin**6*q_cut**6*sqB)/mbkin**18 + 
            (10*q_cut**7*sqB)/mbkin**14 - (30*mckin**4*q_cut**7*sqB)/mbkin**18))*
         np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 - 
       (8640*mckin**8*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(72*mckin**2*muG - 
          (144*mckin**4*muG)/mbkin**2 - (432*mckin**6*muG)/mbkin**4 - 
          (36*mckin**2*muG**2)/mbkin**2 + (72*mckin**4*muG**2)/mbkin**4 + 
          (216*mckin**6*muG**2)/mbkin**6 + (36*mckin**2*muG*mupi)/mbkin**2 - 
          (72*mckin**4*muG*mupi)/mbkin**4 - (216*mckin**6*muG*mupi)/mbkin**6 + 
          (96 + (368*mckin**2)/mbkin**2 - (192*mckin**4)/mbkin**4)*rE + 
          4*(3 - (35*mckin**2)/mbkin**2 + (78*mckin**4)/mbkin**4 + 
            (66*mckin**6)/mbkin**6)*rG + 72*mbkin*rhoD + (408*mckin**2*rhoD)/
           mbkin + (1680*mckin**4*rhoD)/mbkin**3 + (528*mckin**6*rhoD)/mbkin**5 + 
          36*sB + (516*mckin**2*sB)/mbkin**2 + (696*mckin**4*sB)/mbkin**4 + 
          (264*mckin**6*sB)/mbkin**6 - (304*mckin**2*sE)/mbkin**2 - 
          (192*mckin**4*sE)/mbkin**4 - 9*sqB - (67*mckin**2*sqB)/mbkin**2 - 
          (138*mckin**4*sqB)/mbkin**4 - (66*mckin**6*sqB)/mbkin**6)*
         np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8)/
      (540*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
          2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
        3) + 
     (api4*((72*mbkin**4*((-8*(mbkin - mckin)**2*(mbkin + mckin)*
              (3*mbkin + 3*mckin + 8*mbkin*mckin))/(9*mbkin**6) + 
            (16*(4*mbkin**3 - 4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*
              q_cut)/(9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*
           ((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/mbkin**2 - (7*mckin**4)/
                mbkin**4 + mckin**6/mbkin**6) + ((-3 + (14*mckin**2)/mbkin**2 + 
                (26*mckin**4)/mbkin**4 + (14*mckin**6)/mbkin**6 - (3*mckin**8)/
                 mbkin**8)*q_cut)/mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 
                2*mbkin**2*mckin**4 + mckin**6)*q_cut**2)/mbkin**10 + 
             (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*q_cut**3)/mbkin**10 - 
             (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2*
           (1 - (23*mckin**2)/mbkin**2 - (398*mckin**4)/mbkin**4 - 
            (398*mckin**6)/mbkin**6 - (23*mckin**8)/mbkin**8 + 
            mckin**10/mbkin**10 + ((mbkin**8 - 20*mbkin**6*mckin**2 - 102*mbkin**4*
                mckin**4 - 20*mbkin**2*mckin**6 + mckin**8)*q_cut)/mbkin**10 + 
            ((mbkin**6 - 15*mbkin**4*mckin**2 - 15*mbkin**2*mckin**4 + mckin**6)*
              q_cut**2)/mbkin**10 + ((-4 + (2*mckin**2)/mbkin**2 - (4*mckin**4)/
                mbkin**4)*q_cut**3)/mbkin**6 - (4*(mbkin**2 + mckin**2)*q_cut**4)/
             mbkin**10 + (5*q_cut**5)/mbkin**10) + ((-1 + mckin**2/mbkin**2)**2 - 
            (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
           ((64*mbkin*(-((-1 + mckin**2/mbkin**2)**4*(1 + mckin**2/mbkin**2)**2*
                 (503 - (9464*mckin**2)/mbkin**2 + (69322*mckin**4)/mbkin**4 - 
                  (179128*mckin**6)/mbkin**6 - (217124*mckin**8)/mbkin**8 + 
                  (134968*mckin**10)/mbkin**10 - (44170*mckin**12)/mbkin**12 + 
                  (3064*mckin**14)/mbkin**14 + (109*mckin**16)/mbkin**16)) + 
               ((-1 + mckin**2/mbkin**2)**2*(2903 - (41547*mckin**2)/mbkin**2 + 
                  (196111*mckin**4)/mbkin**4 + (84389*mckin**6)/mbkin**6 - 
                  (2226350*mckin**8)/mbkin**8 - (4060522*mckin**10)/mbkin**10 - 
                  (2007262*mckin**12)/mbkin**12 + (332278*mckin**14)/mbkin**14 + 
                  (144215*mckin**16)/mbkin**16 - (185931*mckin**18)/mbkin**18 + 
                  (19663*mckin**20)/mbkin**20 + (613*mckin**22)/mbkin**22)*q_cut)/
                mbkin**2 - (2*(2870 - (34643*mckin**2)/mbkin**2 + 
                  (127322*mckin**4)/mbkin**4 + (190963*mckin**6)/mbkin**6 - 
                  (1214040*mckin**8)/mbkin**8 - (2984930*mckin**10)/mbkin**10 - 
                  (3081596*mckin**12)/mbkin**12 - (1121550*mckin**14)/mbkin**14 + 
                  (431494*mckin**16)/mbkin**16 + (81077*mckin**18)/mbkin**18 - 
                  (159838*mckin**20)/mbkin**20 + (20891*mckin**22)/mbkin**22 + 
                  (540*mckin**24)/mbkin**24)*q_cut**2)/mbkin**4 - 
               (2*(-969 + (3860*mckin**2)/mbkin**2 + (20318*mckin**4)/mbkin**4 - 
                  (143553*mckin**6)/mbkin**6 + (43662*mckin**8)/mbkin**8 + 
                  (243236*mckin**10)/mbkin**10 - (166416*mckin**12)/mbkin**12 - 
                  (159266*mckin**14)/mbkin**14 + (150619*mckin**16)/mbkin**16 + 
                  (20136*mckin**18)/mbkin**18 - (11694*mckin**20)/mbkin**20 + 
                  (67*mckin**22)/mbkin**22)*q_cut**3)/mbkin**6 + 
               ((8985 - (76768*mckin**2)/mbkin**2 + (136327*mckin**4)/mbkin**4 + 
                  (661168*mckin**6)/mbkin**6 + (524386*mckin**8)/mbkin**8 + 
                  (437592*mckin**10)/mbkin**10 + (491062*mckin**12)/mbkin**12 - 
                  (151632*mckin**14)/mbkin**14 - (284475*mckin**16)/mbkin**16 + 
                  (57128*mckin**18)/mbkin**18 + (2563*mckin**20)/mbkin**20)*q_cut**4)/
                mbkin**8 - ((12407 - (70063*mckin**2)/mbkin**2 + (19312*mckin**4)/
                   mbkin**4 + (726800*mckin**6)/mbkin**6 + (1038918*mckin**8)/
                   mbkin**8 + (304322*mckin**10)/mbkin**10 - (490544*mckin**12)/
                   mbkin**12 - (209264*mckin**14)/mbkin**14 + (102019*mckin**16)/
                   mbkin**16 + (1997*mckin**18)/mbkin**18)*q_cut**5)/mbkin**10 - 
               (4*(-354 + (2543*mckin**2)/mbkin**2 - (2380*mckin**4)/mbkin**4 - 
                  (59099*mckin**6)/mbkin**6 - (24326*mckin**8)/mbkin**8 + 
                  (44915*mckin**10)/mbkin**10 + (6624*mckin**12)/mbkin**12 - 
                  (8999*mckin**14)/mbkin**14 + (564*mckin**16)/mbkin**16)*q_cut**6)/
                mbkin**12 + (4*(2103 - (3452*mckin**2)/mbkin**2 - 
                  (14684*mckin**4)/mbkin**4 - (17263*mckin**6)/mbkin**6 - 
                  (5615*mckin**8)/mbkin**8 + (7734*mckin**10)/mbkin**10 + 
                  (11390*mckin**12)/mbkin**12 + (1035*mckin**14)/mbkin**14)*q_cut**7)/
                mbkin**14 + ((-5829 + (9694*mckin**2)/mbkin**2 + (45611*mckin**4)/
                   mbkin**4 + (25144*mckin**6)/mbkin**6 - (44863*mckin**8)/
                   mbkin**8 - (44134*mckin**10)/mbkin**10 - (1047*mckin**12)/
                   mbkin**12)*q_cut**8)/mbkin**16 - ((343 + (6467*mckin**2)/
                   mbkin**2 + (16652*mckin**4)/mbkin**4 - (4820*mckin**6)/
                   mbkin**6 - (5767*mckin**8)/mbkin**8 + (2021*mckin**10)/
                   mbkin**10)*q_cut**9)/mbkin**18 + (2*(830 + (1771*mckin**2)/
                   mbkin**2 + (3242*mckin**4)/mbkin**4 + (3405*mckin**6)/
                   mbkin**6 + (928*mckin**8)/mbkin**8)*q_cut**10)/mbkin**20 - 
               (2*(247 + (420*mckin**2)/mbkin**2 + (1194*mckin**4)/mbkin**4 + 
                  (291*mckin**6)/mbkin**6)*q_cut**11)/mbkin**22 + 
               ((3 + (92*mckin**2)/mbkin**2 + (65*mckin**4)/mbkin**4)*q_cut**12)/
                mbkin**24 - ((9 + (19*mckin**2)/mbkin**2)*q_cut**13)/mbkin**26 + 
               (8*q_cut**14)/mbkin**28))/3 + 72*((-4*((-1 + mckin**2/mbkin**2)**2*
                   (1 - (7*mckin**2)/mbkin**2 - (7*mckin**4)/mbkin**4 + 
                    mckin**6/mbkin**6) + ((-3 + (14*mckin**2)/mbkin**2 + 
                     (26*mckin**4)/mbkin**4 + (14*mckin**6)/mbkin**6 - 
                     (3*mckin**8)/mbkin**8)*q_cut)/mbkin**2 + (2*(mbkin**6 - 
                     2*mbkin**4*mckin**2 - 2*mbkin**2*mckin**4 + mckin**6)*q_cut**2)/
                   mbkin**10 + (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*q_cut**3)/
                   mbkin**10 - (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + 
                  q_cut**5/mbkin**10)**2*(69*mbkin**10 + 184*mbkin**10*mckin + 
                 2319*mbkin**8*mckin**2 - 184*mbkin**9*mckin**2 + 6368*mbkin**8*
                  mckin**3 + 1194*mbkin**6*mckin**4 - 6368*mbkin**7*mckin**4 + 
                 9552*mbkin**6*mckin**5 - 3306*mbkin**4*mckin**6 - 9552*mbkin**5*
                  mckin**6 + 736*mbkin**4*mckin**7 - 291*mbkin**2*mckin**8 - 
                 736*mbkin**3*mckin**8 - 40*mbkin**2*mckin**9 + 15*mckin**10 + 
                 40*mbkin*mckin**10 + 63*mbkin**8*q_cut + 8*mbkin**9*q_cut + 
                 160*mbkin**8*mckin*q_cut + 492*mbkin**6*mckin**2*q_cut - 320*mbkin**7*
                  mckin**2*q_cut + 1632*mbkin**6*mckin**3*q_cut - 738*mbkin**4*mckin**4*
                  q_cut - 2448*mbkin**5*mckin**4*q_cut + 480*mbkin**4*mckin**5*q_cut - 
                 252*mbkin**2*mckin**6*q_cut - 640*mbkin**3*mckin**6*q_cut - 
                 32*mbkin**2*mckin**7*q_cut + 15*mckin**8*q_cut + 40*mbkin*mckin**8*
                  q_cut + 51*mbkin**6*q_cut**2 + 16*mbkin**7*q_cut**2 + 120*mbkin**6*mckin*
                  q_cut**2 - 45*mbkin**4*mckin**2*q_cut**2 - 360*mbkin**5*mckin**2*q_cut**2 + 
                 240*mbkin**4*mckin**3*q_cut**2 - 189*mbkin**2*mckin**4*q_cut**2 - 
                 480*mbkin**3*mckin**4*q_cut**2 - 24*mbkin**2*mckin**5*q_cut**2 + 
                 15*mckin**6*q_cut**2 + 40*mbkin*mckin**6*q_cut**2 - 42*mbkin**4*q_cut**3 - 
                 96*mbkin**5*q_cut**3 - 16*mbkin**4*mckin*q_cut**3 + 48*mbkin**2*mckin**2*
                  q_cut**3 + 64*mbkin**3*mckin**2*q_cut**3 + 64*mbkin**2*mckin**3*q_cut**3 - 
                 60*mckin**4*q_cut**3 - 160*mbkin*mckin**4*q_cut**3 - 36*mbkin**2*q_cut**4 - 
                 128*mbkin**3*q_cut**4 + 32*mbkin**2*mckin*q_cut**4 - 60*mckin**2*q_cut**4 - 
                 160*mbkin*mckin**2*q_cut**4 + 75*q_cut**5 + 200*mbkin*q_cut**5))/(9*
                mbkin**8) + (1 - (23*mckin**2)/mbkin**2 - (398*mckin**4)/
                 mbkin**4 - (398*mckin**6)/mbkin**6 - (23*mckin**8)/mbkin**8 + 
                mckin**10/mbkin**10 + ((mbkin**8 - 20*mbkin**6*mckin**2 - 
                   102*mbkin**4*mckin**4 - 20*mbkin**2*mckin**6 + mckin**8)*q_cut)/
                 mbkin**10 + ((mbkin**6 - 15*mbkin**4*mckin**2 - 15*mbkin**2*
                    mckin**4 + mckin**6)*q_cut**2)/mbkin**10 + 
                ((-4 + (2*mckin**2)/mbkin**2 - (4*mckin**4)/mbkin**4)*q_cut**3)/
                 mbkin**6 - (4*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + 
                (5*q_cut**5)/mbkin**10)*((8*mbkin**2*(3 + 8*mbkin)*
                  ((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/mbkin**2 - 
                      (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6) + 
                    ((-3 + (14*mckin**2)/mbkin**2 + (26*mckin**4)/mbkin**4 + 
                       (14*mckin**6)/mbkin**6 - (3*mckin**8)/mbkin**8)*q_cut)/
                     mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 2*mbkin**2*
                        mckin**4 + mckin**6)*q_cut**2)/mbkin**10 + (2*(mbkin**4 + 
                       mbkin**2*mckin**2 + mckin**4)*q_cut**3)/mbkin**10 - 
                    (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2)/
                 9 + 2*mbkin**4*((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/
                     mbkin**2 - (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6) + 
                  ((-3 + (14*mckin**2)/mbkin**2 + (26*mckin**4)/mbkin**4 + 
                     (14*mckin**6)/mbkin**6 - (3*mckin**8)/mbkin**8)*q_cut)/
                   mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 2*mbkin**2*
                      mckin**4 + mckin**6)*q_cut**2)/mbkin**10 + 
                  (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*q_cut**3)/mbkin**10 - 
                  (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)*
                 ((-4*(mbkin - mckin)**2*(mbkin + mckin)*(27*mbkin**7 + 
                     27*mbkin**6*mckin + 72*mbkin**7*mckin - 21*mbkin**5*
                      mckin**2 - 21*mbkin**4*mckin**3 - 56*mbkin**5*mckin**3 - 
                     93*mbkin**3*mckin**4 - 93*mbkin**2*mckin**5 - 248*mbkin**3*
                      mckin**5 + 15*mbkin*mckin**6 + 15*mckin**7 + 40*mbkin*
                      mckin**7))/(9*mbkin**12) + (4*(51*mbkin**8 + 24*mbkin**9 + 
                     112*mbkin**8*mckin + 72*mbkin**6*mckin**2 - 224*mbkin**7*
                      mckin**2 + 416*mbkin**6*mckin**3 - 108*mbkin**4*mckin**4 - 
                     624*mbkin**5*mckin**4 + 336*mbkin**4*mckin**5 - 204*mbkin**2*
                      mckin**6 - 448*mbkin**3*mckin**6 - 96*mbkin**2*mckin**7 + 
                     45*mckin**8 + 120*mbkin*mckin**8)*q_cut)/(9*mbkin**12) - 
                  (8*(12*mbkin**6 + 16*mbkin**7 + 16*mbkin**6*mckin - 6*mbkin**4*
                      mckin**2 - 48*mbkin**5*mckin**2 + 32*mbkin**4*mckin**3 - 
                     33*mbkin**2*mckin**4 - 64*mbkin**3*mckin**4 - 24*mbkin**2*
                      mckin**5 + 15*mckin**6 + 40*mbkin*mckin**6)*q_cut**2)/
                   (9*mbkin**12) - (8*(6*mbkin**4 + 24*mbkin**5 - 8*mbkin**4*
                      mckin + 6*mbkin**2*mckin**2 + 32*mbkin**3*mckin**2 - 
                     16*mbkin**2*mckin**3 + 15*mckin**4 + 40*mbkin*mckin**4)*
                    q_cut**3)/(9*mbkin**12) + (4*(9*mbkin**2 + 32*mbkin**3 - 
                     8*mbkin**2*mckin + 15*mckin**2 + 40*mbkin*mckin**2)*q_cut**4)/
                   (3*mbkin**12) - (20*(3 + 8*mbkin)*q_cut**5)/(9*mbkin**12))))) - 
          12*((-128*mckin**4*(mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)**2*
              (mbkin**2 + 2*mbkin*mckin + mckin**2 - q_cut)**2*(3*mbkin**4 + 8*
                mbkin**4*mckin - 6*mbkin**2*mckin**2 - 8*mbkin**3*mckin**2 - 8*
                mbkin**2*mckin**3 + 3*mckin**4 + 8*mbkin*mckin**4 - 3*mbkin**2*
                q_cut - 8*mbkin**2*mckin*q_cut - 3*mckin**2*q_cut - 8*mbkin*mckin**2*q_cut)*
              (17*mbkin**16 - 230*mbkin**14*mckin**2 - 508*mbkin**12*mckin**4 + 
               7790*mbkin**10*mckin**6 + 16102*mbkin**8*mckin**8 + 7790*mbkin**6*
                mckin**10 - 508*mbkin**4*mckin**12 - 230*mbkin**2*mckin**14 + 17*
                mckin**16 - 30*mbkin**14*q_cut + 122*mbkin**12*mckin**2*q_cut + 1566*
                mbkin**10*mckin**4*q_cut + 3382*mbkin**8*mckin**6*q_cut + 3382*mbkin**6*
                mckin**8*q_cut + 1566*mbkin**4*mckin**10*q_cut + 122*mbkin**2*mckin**12*
                q_cut - 30*mckin**14*q_cut - 17*mbkin**12*q_cut**2 + 180*mbkin**10*mckin**2*
                q_cut**2 + 2125*mbkin**8*mckin**4*q_cut**2 + 3656*mbkin**6*mckin**6*
                q_cut**2 + 2125*mbkin**4*mckin**8*q_cut**2 + 180*mbkin**2*mckin**10*
                q_cut**2 - 17*mckin**12*q_cut**2 + 50*mbkin**10*q_cut**3 + 62*mbkin**8*
                mckin**2*q_cut**3 - 1104*mbkin**6*mckin**4*q_cut**3 - 1104*mbkin**4*
                mckin**6*q_cut**3 + 62*mbkin**2*mckin**8*q_cut**3 + 50*mckin**10*q_cut**3 - 
               15*mbkin**8*q_cut**4 + 22*mbkin**6*mckin**2*q_cut**4 + 34*mbkin**4*mckin**4*
                q_cut**4 + 22*mbkin**2*mckin**6*q_cut**4 - 15*mckin**8*q_cut**4 - 2*mbkin**6*
                q_cut**5 - 198*mbkin**4*mckin**2*q_cut**5 - 198*mbkin**2*mckin**4*q_cut**5 - 
               2*mckin**6*q_cut**5 + 5*mbkin**4*q_cut**6 + 60*mbkin**2*mckin**2*q_cut**6 + 5*
                mckin**4*q_cut**6 - 18*mbkin**2*q_cut**7 - 18*mckin**2*q_cut**7 + 10*q_cut**8))/
             (mbkin**26*(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                 (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mckin**2 + q_cut + 
               mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))) + 
            ((-32*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)*np.sqrt(0j + (mbkin**4 - 
                   2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                    q_cut + q_cut**2)/mbkin**4)*(3*mbkin**4 + 8*mbkin**4*mckin - 
                 6*mbkin**2*mckin**2 - 8*mbkin**3*mckin**2 - 8*mbkin**2*mckin**3 + 
                 3*mckin**4 + 8*mbkin*mckin**4 - 8*mbkin**3*q_cut + 8*mbkin**2*mckin*
                  q_cut - 6*mckin**2*q_cut - 16*mbkin*mckin**2*q_cut + 3*q_cut**2 + 
                 8*mbkin*q_cut**2)*(17*mbkin**16 - 230*mbkin**14*mckin**2 - 
                 508*mbkin**12*mckin**4 + 7790*mbkin**10*mckin**6 + 16102*mbkin**8*
                  mckin**8 + 7790*mbkin**6*mckin**10 - 508*mbkin**4*mckin**12 - 
                 230*mbkin**2*mckin**14 + 17*mckin**16 - 30*mbkin**14*q_cut + 
                 122*mbkin**12*mckin**2*q_cut + 1566*mbkin**10*mckin**4*q_cut + 
                 3382*mbkin**8*mckin**6*q_cut + 3382*mbkin**6*mckin**8*q_cut + 
                 1566*mbkin**4*mckin**10*q_cut + 122*mbkin**2*mckin**12*q_cut - 
                 30*mckin**14*q_cut - 17*mbkin**12*q_cut**2 + 180*mbkin**10*mckin**2*
                  q_cut**2 + 2125*mbkin**8*mckin**4*q_cut**2 + 3656*mbkin**6*mckin**6*
                  q_cut**2 + 2125*mbkin**4*mckin**8*q_cut**2 + 180*mbkin**2*mckin**10*
                  q_cut**2 - 17*mckin**12*q_cut**2 + 50*mbkin**10*q_cut**3 + 62*mbkin**8*
                  mckin**2*q_cut**3 - 1104*mbkin**6*mckin**4*q_cut**3 - 1104*mbkin**4*
                  mckin**6*q_cut**3 + 62*mbkin**2*mckin**8*q_cut**3 + 50*mckin**10*q_cut**3 - 
                 15*mbkin**8*q_cut**4 + 22*mbkin**6*mckin**2*q_cut**4 + 34*mbkin**4*
                  mckin**4*q_cut**4 + 22*mbkin**2*mckin**6*q_cut**4 - 15*mckin**8*q_cut**4 - 
                 2*mbkin**6*q_cut**5 - 198*mbkin**4*mckin**2*q_cut**5 - 198*mbkin**2*
                  mckin**4*q_cut**5 - 2*mckin**6*q_cut**5 + 5*mbkin**4*q_cut**6 + 
                 60*mbkin**2*mckin**2*q_cut**6 + 5*mckin**4*q_cut**6 - 18*mbkin**2*q_cut**7 - 
                 18*mckin**2*q_cut**7 + 10*q_cut**8))/mbkin**26 + 
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(72*mckin**4*
                 ((-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 3*mckin + 
                     8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 
                     4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/
                   (9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*
                 ((-1 + mckin**4/mbkin**4)**2*(17 - (264*mckin**2)/mbkin**2 + 
                    (3*mckin**4)/mbkin**4 + (8048*mckin**6)/mbkin**6 + 
                    (3*mckin**8)/mbkin**8 - (264*mckin**10)/mbkin**10 + 
                    (17*mckin**12)/mbkin**12) - (16*(4 - (38*mckin**2)/mbkin**2 - 
                     (173*mckin**4)/mbkin**4 + (887*mckin**6)/mbkin**6 + 
                     (3100*mckin**8)/mbkin**8 + (3100*mckin**10)/mbkin**10 + 
                     (887*mckin**12)/mbkin**12 - (173*mckin**14)/mbkin**14 - 
                     (38*mckin**16)/mbkin**16 + (4*mckin**18)/mbkin**18)*q_cut)/
                   mbkin**2 + (4*(15 - (50*mckin**2)/mbkin**2 - (534*mckin**4)/
                      mbkin**4 - (630*mckin**6)/mbkin**6 - (122*mckin**8)/
                      mbkin**8 - (630*mckin**10)/mbkin**10 - (534*mckin**12)/
                      mbkin**12 - (50*mckin**14)/mbkin**14 + (15*mckin**16)/
                      mbkin**16)*q_cut**2)/mbkin**4 + ((54 - (242*mckin**2)/
                      mbkin**2 - (4222*mckin**4)/mbkin**4 - (7014*mckin**6)/
                      mbkin**6 - (7014*mckin**8)/mbkin**8 - (4222*mckin**10)/
                      mbkin**10 - (242*mckin**12)/mbkin**12 + (54*mckin**14)/
                      mbkin**14)*q_cut**3)/mbkin**6 + ((-132 + (8*mckin**2)/
                      mbkin**2 + (4184*mckin**4)/mbkin**4 + (8048*mckin**6)/
                      mbkin**6 + (4184*mckin**8)/mbkin**8 + (8*mckin**10)/
                      mbkin**10 - (132*mckin**12)/mbkin**12)*q_cut**4)/mbkin**8 + 
                  (2*(39 - (73*mckin**2)/mbkin**2 - (510*mckin**4)/mbkin**4 - 
                     (510*mckin**6)/mbkin**6 - (73*mckin**8)/mbkin**8 + 
                     (39*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
                  ((-6 + (472*mckin**2)/mbkin**2 + (716*mckin**4)/mbkin**4 + 
                     (472*mckin**6)/mbkin**6 - (6*mckin**8)/mbkin**8)*q_cut**6)/
                   mbkin**12 - (10*(3 + (31*mckin**2)/mbkin**2 + (31*mckin**4)/
                      mbkin**4 + (3*mckin**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
                  ((51 + (112*mckin**2)/mbkin**2 + (51*mckin**4)/mbkin**4)*q_cut**8)/
                   mbkin**16 - (38*(mbkin**2 + mckin**2)*q_cut**9)/mbkin**20 + 
                  (10*q_cut**10)/mbkin**20) + ((-1 + mckin**2/mbkin**2)**2 - 
                  (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
                 ((-64*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(13 - (155*mckin**2)/
                         mbkin**2 + (233*mckin**4)/mbkin**4 + (9069*mckin**6)/
                         mbkin**6 - (59845*mckin**8)/mbkin**8 - (147269*mckin**
                          10)/mbkin**10 - (55861*mckin**12)/mbkin**12 + 
                        (18451*mckin**14)/mbkin**14 - (5640*mckin**16)/mbkin**
                          16 - (1056*mckin**18)/mbkin**18 + (140*mckin**20)/
                         mbkin**20)) + (4*(14 - (143*mckin**2)/mbkin**2 + 
                        (150*mckin**4)/mbkin**4 + (6469*mckin**6)/mbkin**6 - 
                        (22516*mckin**8)/mbkin**8 - (106182*mckin**10)/mbkin**
                          10 - (107880*mckin**12)/mbkin**12 - (20198*mckin**14)/
                         mbkin**14 + (12998*mckin**16)/mbkin**16 - 
                        (3867*mckin**18)/mbkin**18 - (910*mckin**20)/mbkin**20 + 
                        (145*mckin**22)/mbkin**22)*q_cut)/mbkin**2 - 
                     (4*(15 - (95*mckin**2)/mbkin**2 - (89*mckin**4)/mbkin**4 + 
                        (2989*mckin**6)/mbkin**6 + (4773*mckin**8)/mbkin**8 + 
                        (2337*mckin**10)/mbkin**10 + (10585*mckin**12)/mbkin**
                          12 + (2765*mckin**14)/mbkin**14 - (2854*mckin**16)/
                         mbkin**16 - (416*mckin**18)/mbkin**18 + (150*mckin**20)/
                         mbkin**20)*q_cut**2)/mbkin**4 - (2*(33 - (229*mckin**2)/
                         mbkin**2 - (526*mckin**4)/mbkin**4 + (11552*mckin**6)/
                         mbkin**6 + (19468*mckin**8)/mbkin**8 + (17570*mckin**10)/
                         mbkin**10 + (6430*mckin**12)/mbkin**12 - 
                        (8984*mckin**14)/mbkin**14 - (1277*mckin**16)/mbkin**16 + 
                        (315*mckin**18)/mbkin**18)*q_cut**3)/mbkin**6 + 
                     (2*(84 - (301*mckin**2)/mbkin**2 - (1280*mckin**4)/
                         mbkin**4 + (11016*mckin**6)/mbkin**6 + (26043*mckin**8)/
                         mbkin**8 + (1603*mckin**10)/mbkin**10 - (11796*mckin**
                          12)/mbkin**12 - (692*mckin**14)/mbkin**14 + 
                        (795*mckin**16)/mbkin**16)*q_cut**4)/mbkin**8 - 
                     (2*(21 - (37*mckin**2)/mbkin**2 - (145*mckin**4)/mbkin**4 + 
                        (2777*mckin**6)/mbkin**6 + (927*mckin**8)/mbkin**8 - 
                        (2211*mckin**10)/mbkin**10 + (413*mckin**12)/mbkin**12 + 
                        (255*mckin**14)/mbkin**14)*q_cut**5)/mbkin**10 + 
                     ((-126 + (92*mckin**2)/mbkin**2 + (1574*mckin**4)/mbkin**4 + 
                        (2084*mckin**6)/mbkin**6 + (2298*mckin**8)/mbkin**8 - 
                        (560*mckin**10)/mbkin**10 - (930*mckin**12)/mbkin**12)*
                       q_cut**6)/mbkin**12 + (2*(45 - (25*mckin**2)/mbkin**2 - 
                        (476*mckin**4)/mbkin**4 - (338*mckin**6)/mbkin**6 + 
                        (523*mckin**8)/mbkin**8 + (375*mckin**10)/mbkin**10)*
                       q_cut**7)/mbkin**14 + ((21 + (77*mckin**2)/mbkin**2 + 
                        (272*mckin**4)/mbkin**4 - (64*mckin**6)/mbkin**6 + 
                        (30*mckin**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
                     (2*(19 + (19*mckin**2)/mbkin**2 + (63*mckin**4)/mbkin**4 + 
                        (95*mckin**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
                     (10*(mbkin**4 + 5*mckin**4)*q_cut**10)/mbkin**24))/3 + 
                  72*((8*mckin**2*(3 + 8*mckin)*((-1 + mckin**4/mbkin**4)**2*
                        (17 - (264*mckin**2)/mbkin**2 + (3*mckin**4)/mbkin**4 + 
                         (8048*mckin**6)/mbkin**6 + (3*mckin**8)/mbkin**8 - 
                         (264*mckin**10)/mbkin**10 + (17*mckin**12)/mbkin**12) - 
                       (16*(4 - (38*mckin**2)/mbkin**2 - (173*mckin**4)/mbkin**
                          4 + (887*mckin**6)/mbkin**6 + (3100*mckin**8)/mbkin**
                          8 + (3100*mckin**10)/mbkin**10 + (887*mckin**12)/
                          mbkin**12 - (173*mckin**14)/mbkin**14 - (38*mckin**16)/
                          mbkin**16 + (4*mckin**18)/mbkin**18)*q_cut)/mbkin**2 + 
                       (4*(15 - (50*mckin**2)/mbkin**2 - (534*mckin**4)/mbkin**
                          4 - (630*mckin**6)/mbkin**6 - (122*mckin**8)/mbkin**8 - 
                          (630*mckin**10)/mbkin**10 - (534*mckin**12)/mbkin**12 - 
                          (50*mckin**14)/mbkin**14 + (15*mckin**16)/mbkin**16)*
                         q_cut**2)/mbkin**4 + ((54 - (242*mckin**2)/mbkin**2 - 
                          (4222*mckin**4)/mbkin**4 - (7014*mckin**6)/mbkin**6 - 
                          (7014*mckin**8)/mbkin**8 - (4222*mckin**10)/mbkin**10 - 
                          (242*mckin**12)/mbkin**12 + (54*mckin**14)/mbkin**14)*
                         q_cut**3)/mbkin**6 + ((-132 + (8*mckin**2)/mbkin**2 + 
                          (4184*mckin**4)/mbkin**4 + (8048*mckin**6)/mbkin**6 + 
                          (4184*mckin**8)/mbkin**8 + (8*mckin**10)/mbkin**10 - 
                          (132*mckin**12)/mbkin**12)*q_cut**4)/mbkin**8 + 
                       (2*(39 - (73*mckin**2)/mbkin**2 - (510*mckin**4)/mbkin**
                          4 - (510*mckin**6)/mbkin**6 - (73*mckin**8)/mbkin**8 + 
                          (39*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
                       ((-6 + (472*mckin**2)/mbkin**2 + (716*mckin**4)/mbkin**4 + 
                          (472*mckin**6)/mbkin**6 - (6*mckin**8)/mbkin**8)*q_cut**6)/
                        mbkin**12 - (10*(3 + (31*mckin**2)/mbkin**2 + 
                          (31*mckin**4)/mbkin**4 + (3*mckin**6)/mbkin**6)*q_cut**7)/
                        mbkin**14 + ((51 + (112*mckin**2)/mbkin**2 + 
                          (51*mckin**4)/mbkin**4)*q_cut**8)/mbkin**16 - 
                       (38*(mbkin**2 + mckin**2)*q_cut**9)/mbkin**20 + (10*q_cut**10)/
                        mbkin**20))/9 + mckin**4*((-8*(mbkin - mckin)**2*
                        (mbkin**3 + mbkin**2*mckin + mbkin*mckin**2 + mckin**3)*
                        (396*mbkin**15 + 396*mbkin**14*mckin + 1056*mbkin**15*
                          mckin + 93*mbkin**13*mckin**2 + 93*mbkin**12*mckin**3 + 
                         248*mbkin**13*mckin**3 - 38196*mbkin**11*mckin**4 - 
                         38196*mbkin**10*mckin**5 - 101856*mbkin**11*mckin**5 + 
                         9*mbkin**9*mckin**6 + 9*mbkin**8*mckin**7 + 24*mbkin**9*
                          mckin**7 + 86484*mbkin**7*mckin**8 + 86484*mbkin**6*
                          mckin**9 + 230624*mbkin**7*mckin**9 - 117*mbkin**5*
                          mckin**10 - 117*mbkin**4*mckin**11 - 312*mbkin**5*
                          mckin**11 - 3564*mbkin**3*mckin**12 - 3564*mbkin**2*
                          mckin**13 - 9504*mbkin**3*mckin**13 + 255*mbkin*
                          mckin**14 + 255*mckin**15 + 680*mbkin*mckin**15))/
                       (9*mbkin**22) + (128*(63*mbkin**18 + 16*mbkin**19 + 
                         152*mbkin**18*mckin + 405*mbkin**16*mckin**2 - 
                         304*mbkin**17*mckin**2 + 1384*mbkin**16*mckin**3 - 
                         4770*mbkin**14*mckin**4 - 2076*mbkin**15*mckin**4 - 
                         10644*mbkin**14*mckin**5 - 13278*mbkin**12*mckin**6 + 
                         14192*mbkin**13*mckin**6 - 49600*mbkin**12*mckin**7 + 
                         62000*mbkin**11*mckin**8 - 62000*mbkin**10*mckin**9 + 
                         19917*mbkin**8*mckin**10 + 74400*mbkin**9*mckin**10 - 
                         21288*mbkin**8*mckin**11 + 11130*mbkin**6*mckin**12 + 
                         24836*mbkin**7*mckin**12 + 4844*mbkin**6*mckin**13 - 
                         1620*mbkin**4*mckin**14 - 5536*mbkin**5*mckin**14 + 
                         1216*mbkin**4*mckin**15 - 567*mbkin**2*mckin**16 - 
                         1368*mbkin**3*mckin**16 - 144*mbkin**2*mckin**17 + 
                         60*mckin**18 + 160*mbkin*mckin**18)*q_cut)/(9*mbkin**22) - 
                      (32*(120*mbkin**16 + 120*mbkin**17 + 200*mbkin**16*mckin + 
                         1377*mbkin**14*mckin**2 - 600*mbkin**15*mckin**2 + 
                         4272*mbkin**14*mckin**3 - 369*mbkin**12*mckin**4 - 
                         8544*mbkin**13*mckin**4 + 7560*mbkin**12*mckin**5 - 
                         3993*mbkin**10*mckin**6 - 12600*mbkin**11*mckin**6 + 
                         1952*mbkin**10*mckin**7 + 3627*mbkin**8*mckin**8 - 
                         2928*mbkin**9*mckin**8 + 12600*mbkin**8*mckin**9 - 
                         1809*mbkin**6*mckin**10 - 17640*mbkin**7*mckin**10 + 
                         12816*mbkin**6*mckin**11 - 5883*mbkin**4*mckin**12 - 
                         17088*mbkin**5*mckin**12 + 1400*mbkin**4*mckin**13 - 
                         855*mbkin**2*mckin**14 - 1800*mbkin**3*mckin**14 - 
                         480*mbkin**2*mckin**15 + 225*mckin**16 + 600*mbkin*
                          mckin**16)*q_cut**2)/(9*mbkin**22) - (16*(303*mbkin**14 + 
                         324*mbkin**15 + 484*mbkin**14*mckin + 5607*mbkin**12*
                          mckin**2 - 1936*mbkin**13*mckin**2 + 16888*mbkin**12*
                          mckin**3 - 51*mbkin**10*mckin**4 - 42220*mbkin**11*
                          mckin**4 + 42084*mbkin**10*mckin**5 - 10521*mbkin**8*
                          mckin**6 - 84168*mbkin**9*mckin**6 + 56112*mbkin**8*
                          mckin**7 - 20991*mbkin**6*mckin**8 - 98196*mbkin**7*
                          mckin**8 + 42220*mbkin**6*mckin**9 - 24243*mbkin**4*
                          mckin**10 - 67552*mbkin**5*mckin**10 + 2904*mbkin**4*
                          mckin**11 - 1917*mbkin**2*mckin**12 - 4356*mbkin**3*
                          mckin**12 - 756*mbkin**2*mckin**13 + 405*mckin**14 + 
                         1080*mbkin*mckin**14)*q_cut**3)/(9*mbkin**22) + 
                      (32*(201*mbkin**12 + 528*mbkin**13 + 8*mbkin**12*mckin + 
                         3123*mbkin**10*mckin**2 - 40*mbkin**11*mckin**2 + 
                         8368*mbkin**10*mckin**3 - 360*mbkin**8*mckin**4 - 
                         25104*mbkin**9*mckin**4 + 24144*mbkin**8*mckin**5 - 
                         14850*mbkin**6*mckin**6 - 56336*mbkin**7*mckin**6 + 
                         16736*mbkin**6*mckin**7 - 12537*mbkin**4*mckin**8 - 
                         33472*mbkin**5*mckin**8 + 40*mbkin**4*mckin**9 - 
                         324*mbkin**2*mckin**10 - 72*mbkin**3*mckin**10 - 
                         792*mbkin**2*mckin**11 + 495*mckin**12 + 1320*mbkin*
                          mckin**12)*q_cut**4)/(9*mbkin**22) - (16*(402*mbkin**10 + 
                         780*mbkin**11 + 292*mbkin**10*mckin + 873*mbkin**8*
                          mckin**2 - 1752*mbkin**9*mckin**2 + 4080*mbkin**8*
                          mckin**3 - 3060*mbkin**6*mckin**4 - 14280*mbkin**7*
                          mckin**4 + 6120*mbkin**6*mckin**5 - 5682*mbkin**4*
                          mckin**6 - 16320*mbkin**5*mckin**6 + 1168*mbkin**4*
                          mckin**7 - 1278*mbkin**2*mckin**8 - 2628*mbkin**3*
                          mckin**8 - 780*mbkin**2*mckin**9 + 585*mckin**10 + 
                         1560*mbkin*mckin**10)*q_cut**5)/(9*mbkin**22) + 
                      (16*(381*mbkin**8 + 72*mbkin**9 + 944*mbkin**8*mckin - 
                         1404*mbkin**6*mckin**2 - 6608*mbkin**7*mckin**2 + 
                         2864*mbkin**6*mckin**3 - 3234*mbkin**4*mckin**4 - 
                         11456*mbkin**5*mckin**4 + 2832*mbkin**4*mckin**5 - 
                         3204*mbkin**2*mckin**6 - 8496*mbkin**3*mckin**6 - 
                         48*mbkin**2*mckin**7 + 45*mckin**8 + 120*mbkin*mckin**8)*
                        q_cut**6)/(9*mbkin**22) + (80*(-15*mbkin**6 + 84*mbkin**7 - 
                         124*mbkin**6*mckin + 279*mbkin**4*mckin**2 + 
                         992*mbkin**5*mckin**2 - 248*mbkin**4*mckin**3 + 
                         405*mbkin**2*mckin**4 + 1116*mbkin**3*mckin**4 - 
                         36*mbkin**2*mckin**5 + 45*mckin**6 + 120*mbkin*mckin**6)*
                        q_cut**7)/(9*mbkin**22) - (8*(444*mbkin**4 + 1632*mbkin**5 - 
                         448*mbkin**4*mckin + 1359*mbkin**2*mckin**2 + 4032*
                          mbkin**3*mckin**2 - 408*mbkin**2*mckin**3 + 765*
                          mckin**4 + 2040*mbkin*mckin**4)*q_cut**8)/(9*mbkin**22) + 
                      (304*(12*mbkin**2 + 36*mbkin**3 - 4*mbkin**2*mckin + 
                         15*mckin**2 + 40*mbkin*mckin**2)*q_cut**9)/(9*mbkin**22) - 
                      (400*(3 + 8*mbkin)*q_cut**10)/(9*mbkin**22))))))*
             np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)))) - 
          144*((256*mckin**8*(mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)*
              (mbkin**2 + 2*mbkin*mckin + mckin**2 - q_cut)*(31*mbkin**10 - 153*
                mbkin**8*mckin**2 - 1138*mbkin**6*mckin**4 - 1138*mbkin**4*
                mckin**6 - 153*mbkin**2*mckin**8 + 31*mckin**10 - 29*mbkin**8*q_cut - 
               100*mbkin**6*mckin**2*q_cut - 162*mbkin**4*mckin**4*q_cut - 100*mbkin**2*
                mckin**6*q_cut - 29*mckin**8*q_cut - 29*mbkin**6*q_cut**2 - 125*mbkin**4*
                mckin**2*q_cut**2 - 125*mbkin**2*mckin**4*q_cut**2 - 29*mckin**6*q_cut**2 + 
               26*mbkin**4*q_cut**3 + 82*mbkin**2*mckin**2*q_cut**3 + 26*mckin**4*q_cut**3 - 
               4*mbkin**2*q_cut**4 - 4*mckin**2*q_cut**4 + 5*q_cut**5)*(3*mbkin**4*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 8*mbkin**4*mckin*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 6*mbkin**2*mckin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**3*mckin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**2*mckin**3*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 3*mckin**4*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 8*mbkin*mckin**4*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mbkin**2*q_cut*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**2*mckin*q_cut*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mckin**2*q_cut*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin*mckin**2*q_cut*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))*np.log((mbkin**2 + mckin**2 - 
                 q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                     2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/
                (mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                      mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                    mbkin**4))))/(mbkin**20*(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mckin**2 + q_cut + 
               mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))) + 
            ((-64*mckin**6*(3*mbkin**2 + 8*mbkin**2*mckin - 3*mckin**2 - 
                 8*mbkin*mckin**2)*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*(31*mbkin**10 - 
                 153*mbkin**8*mckin**2 - 1138*mbkin**6*mckin**4 - 1138*mbkin**4*
                  mckin**6 - 153*mbkin**2*mckin**8 + 31*mckin**10 - 29*mbkin**8*
                  q_cut - 100*mbkin**6*mckin**2*q_cut - 162*mbkin**4*mckin**4*q_cut - 
                 100*mbkin**2*mckin**6*q_cut - 29*mckin**8*q_cut - 29*mbkin**6*q_cut**2 - 
                 125*mbkin**4*mckin**2*q_cut**2 - 125*mbkin**2*mckin**4*q_cut**2 - 
                 29*mckin**6*q_cut**2 + 26*mbkin**4*q_cut**3 + 82*mbkin**2*mckin**2*
                  q_cut**3 + 26*mckin**4*q_cut**3 - 4*mbkin**2*q_cut**4 - 4*mckin**2*q_cut**4 + 
                 5*q_cut**5))/mbkin**24 + (mckin**4*(-72*mckin**4*
                  ((-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 3*mckin + 
                      8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 
                      4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/
                    (9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*
                  ((-1 + mckin**2/mbkin**2)**2*(31 - (153*mckin**2)/mbkin**2 - 
                     (1138*mckin**4)/mbkin**4 - (1138*mckin**6)/mbkin**6 - 
                     (153*mckin**8)/mbkin**8 + (31*mckin**10)/mbkin**10) + 
                   ((-91 + (202*mckin**2)/mbkin**2 + (2591*mckin**4)/mbkin**4 + 
                      (4676*mckin**6)/mbkin**6 + (2591*mckin**8)/mbkin**8 + 
                      (202*mckin**10)/mbkin**10 - (91*mckin**12)/mbkin**12)*q_cut)/
                    mbkin**2 + ((60 + (38*mckin**2)/mbkin**2 - (518*mckin**4)/
                       mbkin**4 - (518*mckin**6)/mbkin**6 + (38*mckin**8)/
                       mbkin**8 + (60*mckin**10)/mbkin**10)*q_cut**2)/mbkin**4 + 
                   ((55 + (238*mckin**2)/mbkin**2 + (226*mckin**4)/mbkin**4 + 
                      (238*mckin**6)/mbkin**6 + (55*mckin**8)/mbkin**8)*q_cut**3)/
                    mbkin**6 - ((85 + (337*mckin**2)/mbkin**2 + (337*mckin**4)/
                       mbkin**4 + (85*mckin**6)/mbkin**6)*q_cut**4)/mbkin**8 + 
                   ((39 + (88*mckin**2)/mbkin**2 + (39*mckin**4)/mbkin**4)*q_cut**5)/
                    mbkin**10 - (14*(mbkin**2 + mckin**2)*q_cut**6)/mbkin**14 + 
                   (5*q_cut**7)/mbkin**14) + ((-1 + mckin**2/mbkin**2)**2 - 
                   (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
                  ((64*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(-2 - (44*mckin**2)/
                          mbkin**2 + (729*mckin**4)/mbkin**4 + (9244*mckin**6)/
                          mbkin**6 + (10134*mckin**8)/mbkin**8 + (36*mckin**10)/
                          mbkin**10 - (77*mckin**12)/mbkin**12 + (140*mckin**14)/
                          mbkin**14)) + ((-2 - (226*mckin**2)/mbkin**2 + 
                         (1025*mckin**4)/mbkin**4 + (19673*mckin**6)/mbkin**6 + 
                         (39598*mckin**8)/mbkin**8 + (21350*mckin**10)/mbkin**
                          10 - (861*mckin**12)/mbkin**12 - (357*mckin**14)/
                          mbkin**14 + (440*mckin**16)/mbkin**16)*q_cut)/mbkin**2 - 
                      (2*mckin**2*(-88 - (232*mckin**2)/mbkin**2 + 
                         (2115*mckin**4)/mbkin**4 + (1978*mckin**6)/mbkin**6 - 
                         (633*mckin**8)/mbkin**8 + (70*mckin**10)/mbkin**10 + 
                         (150*mckin**12)/mbkin**12)*q_cut**2)/mbkin**6 + 
                      ((-10 + (216*mckin**2)/mbkin**2 + (1252*mckin**4)/mbkin**
                          4 + (972*mckin**6)/mbkin**6 + (866*mckin**8)/mbkin**8 - 
                         (370*mckin**10)/mbkin**10 - (350*mckin**12)/mbkin**12)*
                        q_cut**3)/mbkin**6 + ((10 - (284*mckin**2)/mbkin**2 - 
                         (2029*mckin**4)/mbkin**4 - (1764*mckin**6)/mbkin**6 + 
                         (735*mckin**8)/mbkin**8 + (500*mckin**10)/mbkin**10)*
                        q_cut**4)/mbkin**8 + ((18 + (106*mckin**2)/mbkin**2 + 
                         (601*mckin**4)/mbkin**4 - (157*mckin**6)/mbkin**6 - 
                         (60*mckin**8)/mbkin**8)*q_cut**5)/mbkin**10 - 
                      (4*(7 + (7*mckin**2)/mbkin**2 + (17*mckin**4)/mbkin**4 + 
                         (35*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
                      (10*(mbkin**4 + 5*mckin**4)*q_cut**7)/mbkin**18))/3 - 
                   72*((8*mckin**2*(3 + 8*mckin)*((-1 + mckin**2/mbkin**2)**2*
                         (31 - (153*mckin**2)/mbkin**2 - (1138*mckin**4)/
                          mbkin**4 - (1138*mckin**6)/mbkin**6 - (153*mckin**8)/
                          mbkin**8 + (31*mckin**10)/mbkin**10) + 
                        ((-91 + (202*mckin**2)/mbkin**2 + (2591*mckin**4)/
                          mbkin**4 + (4676*mckin**6)/mbkin**6 + (2591*mckin**8)/
                          mbkin**8 + (202*mckin**10)/mbkin**10 - (91*mckin**12)/
                          mbkin**12)*q_cut)/mbkin**2 + ((60 + (38*mckin**2)/mbkin**
                          2 - (518*mckin**4)/mbkin**4 - (518*mckin**6)/mbkin**6 + 
                          (38*mckin**8)/mbkin**8 + (60*mckin**10)/mbkin**10)*
                          q_cut**2)/mbkin**4 + ((55 + (238*mckin**2)/mbkin**2 + 
                          (226*mckin**4)/mbkin**4 + (238*mckin**6)/mbkin**6 + 
                          (55*mckin**8)/mbkin**8)*q_cut**3)/mbkin**6 - 
                        ((85 + (337*mckin**2)/mbkin**2 + (337*mckin**4)/mbkin**
                          4 + (85*mckin**6)/mbkin**6)*q_cut**4)/mbkin**8 + 
                        ((39 + (88*mckin**2)/mbkin**2 + (39*mckin**4)/mbkin**4)*
                          q_cut**5)/mbkin**10 - (14*(mbkin**2 + mckin**2)*q_cut**6)/
                         mbkin**14 + (5*q_cut**7)/mbkin**14))/9 + mckin**4*
                      ((-4*(mbkin - mckin)**2*(mbkin + mckin)*(645*mbkin**11 + 
                          645*mbkin**10*mckin + 1720*mbkin**11*mckin + 
                          5451*mbkin**9*mckin**2 + 5451*mbkin**8*mckin**3 + 
                          14536*mbkin**9*mckin**3 - 3414*mbkin**7*mckin**4 - 
                          3414*mbkin**6*mckin**5 - 9104*mbkin**7*mckin**5 - 
                          15234*mbkin**5*mckin**6 - 15234*mbkin**4*mckin**7 - 
                          40624*mbkin**5*mckin**7 - 3219*mbkin**3*mckin**8 - 
                          3219*mbkin**2*mckin**9 - 8584*mbkin**3*mckin**9 + 
                          651*mbkin*mckin**10 + 651*mckin**11 + 1736*mbkin*
                          mckin**11))/(9*mbkin**16) + (4*(879*mbkin**12 + 
                          728*mbkin**13 + 1616*mbkin**12*mckin + 14334*mbkin**10*
                          mckin**2 - 3232*mbkin**11*mckin**2 + 41456*mbkin**10*
                          mckin**3 + 18765*mbkin**8*mckin**4 - 62184*mbkin**9*
                          mckin**4 + 112224*mbkin**8*mckin**5 - 25020*mbkin**6*
                          mckin**6 - 149632*mbkin**7*mckin**6 + 82912*mbkin**6*
                          mckin**7 - 35835*mbkin**4*mckin**8 - 103640*mbkin**5*
                          mckin**8 + 8080*mbkin**4*mckin**9 - 5274*mbkin**2*
                          mckin**10 - 9696*mbkin**3*mckin**10 - 4368*mbkin**2*
                          mckin**11 + 1911*mckin**12 + 5096*mbkin*mckin**12)*q_cut)/
                        (9*mbkin**16) - (8*(123*mbkin**10 + 480*mbkin**11 - 
                          152*mbkin**10*mckin + 1725*mbkin**8*mckin**2 + 
                          456*mbkin**9*mckin**2 + 4144*mbkin**8*mckin**3 - 
                          777*mbkin**6*mckin**4 - 8288*mbkin**7*mckin**4 + 
                          6216*mbkin**6*mckin**5 - 4113*mbkin**4*mckin**6 - 
                          10360*mbkin**5*mckin**6 - 608*mbkin**4*mckin**7 - 
                          108*mbkin**2*mckin**8 + 912*mbkin**3*mckin**8 - 
                          1200*mbkin**2*mckin**9 + 630*mckin**10 + 1680*mbkin*
                          mckin**10)*q_cut**2)/(9*mbkin**16) - (4*(-219*mbkin**8 + 
                          1320*mbkin**9 - 1904*mbkin**8*mckin + 1500*mbkin**6*
                          mckin**2 + 7616*mbkin**7*mckin**2 - 3616*mbkin**6*
                          mckin**3 + 1248*mbkin**4*mckin**4 + 9040*mbkin**5*
                          mckin**4 - 5712*mbkin**4*mckin**5 + 3624*mbkin**2*
                          mckin**6 + 11424*mbkin**3*mckin**6 - 1760*mbkin**2*
                          mckin**7 + 1155*mckin**8 + 3080*mbkin*mckin**8)*q_cut**3)/
                        (9*mbkin**16) + (4*(9*mbkin**6 + 2720*mbkin**7 - 
                          2696*mbkin**6*mckin + 3033*mbkin**4*mckin**2 + 
                          13480*mbkin**5*mckin**2 - 5392*mbkin**4*mckin**3 + 
                          5301*mbkin**2*mckin**4 + 16176*mbkin**3*mckin**4 - 
                          2040*mbkin**2*mckin**5 + 1785*mckin**6 + 4760*mbkin*
                          mckin**6)*q_cut**4)/(9*mbkin**16) - (4*(321*mbkin**4 + 
                          1560*mbkin**5 - 704*mbkin**4*mckin + 1350*mbkin**2*
                          mckin**2 + 4224*mbkin**3*mckin**2 - 624*mbkin**2*
                          mckin**3 + 819*mckin**4 + 2184*mbkin*mckin**4)*q_cut**5)/
                        (9*mbkin**16) + (56*(15*mbkin**2 + 48*mbkin**3 - 
                          8*mbkin**2*mckin + 21*mckin**2 + 56*mbkin*mckin**2)*
                         q_cut**6)/(9*mbkin**16) - (140*(3 + 8*mbkin)*q_cut**7)/
                        (9*mbkin**16))))))/mbkin**4)*
             np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                      mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                    mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                  np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                      q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2) - 
          8640*((-384*mckin**12*(3*mbkin**4 + 8*mbkin**2*mckin**2 + 3*mckin**4)*
              (mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)*(mbkin**2 + 2*mbkin*
                mckin + mckin**2 - q_cut)*(3*mbkin**4 + 8*mbkin**4*mckin - 6*
                mbkin**2*mckin**2 - 8*mbkin**3*mckin**2 - 8*mbkin**2*mckin**3 + 3*
                mckin**4 + 8*mbkin*mckin**4 - 3*mbkin**2*q_cut - 8*mbkin**2*mckin*
                q_cut - 3*mckin**2*q_cut - 8*mbkin*mckin**2*q_cut)*
              np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/
             (mbkin**18*(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                 (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mckin**2 + q_cut + 
               mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))) + 
            ((-64*mckin**8*(3*mbkin**7 + 17*mbkin**5*mckin**2 - 27*mbkin**6*
                  mckin**2 - 72*mbkin**6*mckin**3 + 70*mbkin**3*mckin**4 - 
                 108*mbkin**4*mckin**4 - 288*mbkin**4*mckin**5 + 22*mbkin*
                  mckin**6 - 18*mbkin**2*mckin**6 + 96*mbkin**3*mckin**6 - 
                 144*mbkin**2*mckin**7 + 27*mckin**8 + 72*mbkin*mckin**8)*
                ((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2))/(3*mbkin**14) - 
              (32*mckin**10*(3*mbkin**4 + 8*mbkin**2*mckin**2 + 3*mckin**4)*
                ((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*(-12*mbkin**6 - 
                 32*mbkin**6*mckin + 45*mbkin**4*mckin**2 + 32*mbkin**5*mckin**2 + 
                 88*mbkin**4*mckin**3 - 54*mbkin**2*mckin**4 - 88*mbkin**3*
                  mckin**4 - 56*mbkin**2*mckin**5 + 21*mckin**6 + 56*mbkin*
                  mckin**6 + 24*mbkin**4*q_cut + 64*mbkin**4*mckin*q_cut - 
                 88*mbkin**3*mckin**2*q_cut + 88*mbkin**2*mckin**3*q_cut - 42*mckin**4*
                  q_cut - 112*mbkin*mckin**4*q_cut - 12*mbkin**2*q_cut**2 - 32*mbkin**2*
                  mckin*q_cut**2 + 21*mckin**2*q_cut**2 + 56*mbkin*mckin**2*q_cut**2))/(
                mbkin**14*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)))*
             np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                      mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                    mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                  np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                      q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3))/
         (((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
             2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
               7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
               mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
            (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*
                  np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                      q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                 mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                     mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                    mbkin**4))))/mbkin**4)**3) + 
        ((72*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mckin**2 - 
              7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
              mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)**2*(mbkin**10 - 
             23*mbkin**8*mckin**2 - 398*mbkin**6*mckin**4 - 398*mbkin**4*mckin**6 - 
             23*mbkin**2*mckin**8 + mckin**10 + mbkin**8*q_cut - 20*mbkin**6*mckin**2*
              q_cut - 102*mbkin**4*mckin**4*q_cut - 20*mbkin**2*mckin**6*q_cut + 
             mckin**8*q_cut + mbkin**6*q_cut**2 - 15*mbkin**4*mckin**2*q_cut**2 - 
             15*mbkin**2*mckin**4*q_cut**2 + mckin**6*q_cut**2 - 4*mbkin**4*q_cut**3 + 
             2*mbkin**2*mckin**2*q_cut**3 - 4*mckin**4*q_cut**3 - 4*mbkin**2*q_cut**4 - 
             4*mckin**2*q_cut**4 + 5*q_cut**5))/mbkin**30 - 
          (864*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
              2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*
            np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)*(17*mbkin**16 - 230*mbkin**14*
              mckin**2 - 508*mbkin**12*mckin**4 + 7790*mbkin**10*mckin**6 + 
             16102*mbkin**8*mckin**8 + 7790*mbkin**6*mckin**10 - 
             508*mbkin**4*mckin**12 - 230*mbkin**2*mckin**14 + 17*mckin**16 - 
             30*mbkin**14*q_cut + 122*mbkin**12*mckin**2*q_cut + 1566*mbkin**10*mckin**4*
              q_cut + 3382*mbkin**8*mckin**6*q_cut + 3382*mbkin**6*mckin**8*q_cut + 
             1566*mbkin**4*mckin**10*q_cut + 122*mbkin**2*mckin**12*q_cut - 
             30*mckin**14*q_cut - 17*mbkin**12*q_cut**2 + 180*mbkin**10*mckin**2*q_cut**2 + 
             2125*mbkin**8*mckin**4*q_cut**2 + 3656*mbkin**6*mckin**6*q_cut**2 + 
             2125*mbkin**4*mckin**8*q_cut**2 + 180*mbkin**2*mckin**10*q_cut**2 - 
             17*mckin**12*q_cut**2 + 50*mbkin**10*q_cut**3 + 62*mbkin**8*mckin**2*q_cut**3 - 
             1104*mbkin**6*mckin**4*q_cut**3 - 1104*mbkin**4*mckin**6*q_cut**3 + 
             62*mbkin**2*mckin**8*q_cut**3 + 50*mckin**10*q_cut**3 - 15*mbkin**8*q_cut**4 + 
             22*mbkin**6*mckin**2*q_cut**4 + 34*mbkin**4*mckin**4*q_cut**4 + 
             22*mbkin**2*mckin**6*q_cut**4 - 15*mckin**8*q_cut**4 - 2*mbkin**6*q_cut**5 - 
             198*mbkin**4*mckin**2*q_cut**5 - 198*mbkin**2*mckin**4*q_cut**5 - 
             2*mckin**6*q_cut**5 + 5*mbkin**4*q_cut**6 + 60*mbkin**2*mckin**2*q_cut**6 + 
             5*mckin**4*q_cut**6 - 18*mbkin**2*q_cut**7 - 18*mckin**2*q_cut**7 + 10*q_cut**8)*
            np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                  mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**24 + 
          (10368*mckin**8*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
              2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*(31*mbkin**10 - 
             153*mbkin**8*mckin**2 - 1138*mbkin**6*mckin**4 - 1138*mbkin**4*
              mckin**6 - 153*mbkin**2*mckin**8 + 31*mckin**10 - 29*mbkin**8*q_cut - 
             100*mbkin**6*mckin**2*q_cut - 162*mbkin**4*mckin**4*q_cut - 
             100*mbkin**2*mckin**6*q_cut - 29*mckin**8*q_cut - 29*mbkin**6*q_cut**2 - 
             125*mbkin**4*mckin**2*q_cut**2 - 125*mbkin**2*mckin**4*q_cut**2 - 
             29*mckin**6*q_cut**2 + 26*mbkin**4*q_cut**3 + 82*mbkin**2*mckin**2*q_cut**3 + 
             26*mckin**4*q_cut**3 - 4*mbkin**2*q_cut**4 - 4*mckin**2*q_cut**4 + 5*q_cut**5)*
            np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**22 - 
          (622080*mckin**12*(3*mbkin**4 + 8*mbkin**2*mckin**2 + 3*mckin**4)*
            ((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**12)*
         ((-3*((-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 3*mckin + 
                8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 4*mbkin**2*
                 mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/(9*mbkin**6) - 
             (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6)))/
           (2*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*((-1 + mckin**2/mbkin**2)**2 - 
             (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
            ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*
                  mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                 mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
              (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                       2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - 
                   q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                       2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**
                4)**3) - (3*((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(
                (-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 3*mckin + 
                   8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 4*mbkin**2*
                    mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/(9*mbkin**6) - 
                (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*(mbkin**6 - 7*mbkin**4*
                 mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/
              (2*mbkin**6*((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*
                  q_cut)/mbkin**4 + q_cut**2/mbkin**4)) - 
             (4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(21*mbkin**6 + 56*mbkin**6*
                 mckin + 21*mbkin**4*mckin**2 - 56*mbkin**5*mckin**2 + 
                112*mbkin**4*mckin**3 - 51*mbkin**2*mckin**4 - 112*mbkin**3*
                 mckin**4 - 24*mbkin**2*mckin**5 + 9*mckin**6 + 24*mbkin*
                 mckin**6 - 3*mbkin**4*q_cut - 8*mbkin**5*q_cut + 6*mbkin**2*mckin**2*
                 q_cut + 16*mbkin**2*mckin**3*q_cut - 9*mckin**4*q_cut - 24*mbkin*mckin**4*
                 q_cut - 3*mbkin**2*q_cut**2 - 16*mbkin**3*q_cut**2 + 8*mbkin**2*mckin*
                 q_cut**2 - 9*mckin**2*q_cut**2 - 24*mbkin*mckin**2*q_cut**2 + 9*q_cut**3 + 
                24*mbkin*q_cut**3))/(9*mbkin**8) - 12*((-16*mckin**4*
                 (3*mbkin**6*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 
                  8*mbkin**6*mckin*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                      mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                     mbkin**4) - 6*mbkin**4*mckin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4) - 8*mbkin**5*mckin**2*np.sqrt(0j + (mbkin**4 - 
                      2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                       q_cut + q_cut**2)/mbkin**4) - 8*mbkin**4*mckin**3*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                      2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 3*mbkin**2*mckin**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 8*mbkin**3*mckin**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mbkin**4*q_cut*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**4*mckin*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mbkin**2*mckin**2*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**3*mckin**2*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))/(9*mbkin**4*
                 (mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)*(mbkin**2 + 
                  2*mbkin*mckin + mckin**2 - q_cut)*(mbkin**2 + mckin**2 - q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))*
                 (-mbkin**2 - mckin**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4))) + ((-8*(3 + 8*mbkin)*mckin**4)/
                  (9*mbkin**6) + (8*mckin**2*(3 + 8*mckin))/(9*mbkin**4))*
                np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                       2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                        q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                       2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))))/
           (((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*
                  mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                 mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
              (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                       2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - 
                   q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                       2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**
                4)**4))))/540 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

def q2moment3Kin(q_cut, mbkin, mckin, mus, api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
 res = ( 
    (mbkin**2*((180*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)**2*(mbkin**12 - 
           34*mbkin**10*mckin**2 - 1133*mbkin**8*mckin**4 - 2708*mbkin**6*
            mckin**6 - 1133*mbkin**4*mckin**8 - 34*mbkin**2*mckin**10 + mckin**12 + 
           mbkin**10*q_cut - 31*mbkin**8*mckin**2*q_cut - 390*mbkin**6*mckin**4*q_cut - 
           390*mbkin**4*mckin**6*q_cut - 31*mbkin**2*mckin**8*q_cut + mckin**10*q_cut + 
           mbkin**8*q_cut**2 - 26*mbkin**6*mckin**2*q_cut**2 - 118*mbkin**4*mckin**4*
            q_cut**2 - 26*mbkin**2*mckin**6*q_cut**2 + mckin**8*q_cut**2 + mbkin**6*q_cut**3 - 
           19*mbkin**4*mckin**2*q_cut**3 - 19*mbkin**2*mckin**4*q_cut**3 + mckin**6*q_cut**3 - 
           6*mbkin**4*q_cut**4 + 4*mbkin**2*mckin**2*q_cut**4 - 6*mckin**4*q_cut**4 - 
           6*mbkin**2*q_cut**5 - 6*mckin**2*q_cut**5 + 8*q_cut**6))/mbkin**32 - 
        (2160*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
             mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*
          (37*mbkin**18 - 397*mbkin**16*mckin**2 - 2854*mbkin**14*mckin**4 + 
           18134*mbkin**12*mckin**6 + 75800*mbkin**10*mckin**8 + 
           75800*mbkin**8*mckin**10 + 18134*mbkin**6*mckin**12 - 
           2854*mbkin**4*mckin**14 - 397*mbkin**2*mckin**16 + 37*mckin**18 - 
           70*mbkin**16*q_cut + 132*mbkin**14*mckin**2*q_cut + 4424*mbkin**12*mckin**4*
            q_cut + 15500*mbkin**10*mckin**6*q_cut + 20508*mbkin**8*mckin**8*q_cut + 
           15500*mbkin**6*mckin**10*q_cut + 4424*mbkin**4*mckin**12*q_cut + 
           132*mbkin**2*mckin**14*q_cut - 70*mckin**16*q_cut - 37*mbkin**14*q_cut**2 + 
           307*mbkin**12*mckin**2*q_cut**2 + 6201*mbkin**10*mckin**4*q_cut**2 + 
           18225*mbkin**8*mckin**6*q_cut**2 + 18225*mbkin**6*mckin**8*q_cut**2 + 
           6201*mbkin**4*mckin**10*q_cut**2 + 307*mbkin**2*mckin**12*q_cut**2 - 
           37*mckin**14*q_cut**2 + 140*mbkin**12*q_cut**3 + 272*mbkin**10*mckin**2*q_cut**3 - 
           2796*mbkin**8*mckin**4*q_cut**3 - 7136*mbkin**6*mckin**6*q_cut**3 - 
           2796*mbkin**4*mckin**8*q_cut**3 + 272*mbkin**2*mckin**10*q_cut**3 + 
           140*mckin**12*q_cut**3 - 49*mbkin**10*q_cut**4 + 13*mbkin**8*mckin**2*q_cut**4 - 
           300*mbkin**6*mckin**4*q_cut**4 - 300*mbkin**4*mckin**6*q_cut**4 + 
           13*mbkin**2*mckin**8*q_cut**4 - 49*mckin**10*q_cut**4 - 70*mbkin**8*q_cut**5 - 
           372*mbkin**6*mckin**2*q_cut**5 - 668*mbkin**4*mckin**4*q_cut**5 - 
           372*mbkin**2*mckin**6*q_cut**5 - 70*mckin**8*q_cut**5 + 77*mbkin**6*q_cut**6 + 
           41*mbkin**4*mckin**2*q_cut**6 + 41*mbkin**2*mckin**4*q_cut**6 + 
           77*mckin**6*q_cut**6 - 16*mbkin**4*q_cut**7 + 32*mbkin**2*mckin**2*q_cut**7 - 
           16*mckin**4*q_cut**7 - 28*mbkin**2*q_cut**8 - 28*mckin**2*q_cut**8 + 16*q_cut**9)*
          np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**26 + 
        (25920*mckin**8*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
            2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*(71*mbkin**12 - 
           174*mbkin**10*mckin**2 - 3723*mbkin**8*mckin**4 - 
           7468*mbkin**6*mckin**6 - 3723*mbkin**4*mckin**8 - 
           174*mbkin**2*mckin**10 + 71*mckin**12 - 69*mbkin**10*q_cut - 
           381*mbkin**8*mckin**2*q_cut - 810*mbkin**6*mckin**4*q_cut - 
           810*mbkin**4*mckin**6*q_cut - 381*mbkin**2*mckin**8*q_cut - 69*mckin**10*q_cut - 
           69*mbkin**8*q_cut**2 - 446*mbkin**6*mckin**2*q_cut**2 - 818*mbkin**4*mckin**4*
            q_cut**2 - 446*mbkin**2*mckin**6*q_cut**2 - 69*mckin**8*q_cut**2 + 
           71*mbkin**6*q_cut**3 + 331*mbkin**4*mckin**2*q_cut**3 + 331*mbkin**2*mckin**4*
            q_cut**3 + 71*mckin**6*q_cut**3 - 6*mbkin**4*q_cut**4 + 4*mbkin**2*mckin**2*
            q_cut**4 - 6*mckin**4*q_cut**4 - 6*mbkin**2*q_cut**5 - 6*mckin**2*q_cut**5 + 8*q_cut**6)*
          np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**24 - 
        (10886400*mckin**12*(mbkin**6 + 5*mbkin**4*mckin**2 + 5*mbkin**2*mckin**4 + 
           mckin**6)*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
             2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**14))/
      (2520*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
          2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
        3) + (mbkin**2*(((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
           mbkin**4 + q_cut**2/mbkin**4)*(-72*mbkin**2*muG*
           ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
             q_cut**2/mbkin**4)**2*(25 + (789*mckin**2)/mbkin**2 - 
            (11707*mckin**4)/mbkin**4 + (28953*mckin**6)/mbkin**6 + 
            (73278*mckin**8)/mbkin**8 - (325002*mckin**10)/mbkin**10 - 
            (894522*mckin**12)/mbkin**12 - (477522*mckin**14)/mbkin**14 + 
            (113193*mckin**16)/mbkin**16 + (47093*mckin**18)/mbkin**18 - 
            (6027*mckin**20)/mbkin**20 - (71*mckin**22)/mbkin**22 + 
            ((-35 - (1504*mckin**2)/mbkin**2 + (15413*mckin**4)/mbkin**4 - 
               (30432*mckin**6)/mbkin**6 - (180506*mckin**8)/mbkin**8 - 
               (158416*mckin**10)/mbkin**10 - (166746*mckin**12)/mbkin**12 - 
               (202976*mckin**14)/mbkin**14 - (10627*mckin**16)/mbkin**16 + 
               (9968*mckin**18)/mbkin**18 + (101*mckin**20)/mbkin**20)*q_cut)/
             mbkin**2 + (2*(-30 - (544*mckin**2)/mbkin**2 + (7709*mckin**4)/
                mbkin**4 - (22753*mckin**6)/mbkin**6 - (158279*mckin**8)/
                mbkin**8 - (280433*mckin**10)/mbkin**10 - (161069*mckin**12)/
                mbkin**12 - (18783*mckin**14)/mbkin**14 + (5109*mckin**16)/
                mbkin**16 + (81*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
            (2*(-35 - (1674*mckin**2)/mbkin**2 + (14879*mckin**4)/mbkin**4 + 
               (17828*mckin**6)/mbkin**6 - (66689*mckin**8)/mbkin**8 - 
               (71234*mckin**10)/mbkin**10 + (21933*mckin**12)/mbkin**12 + 
               (10784*mckin**14)/mbkin**14 + (96*mckin**16)/mbkin**16)*q_cut**3)/
             mbkin**6 - (2*(-35 + (183*mckin**2)/mbkin**2 + (448*mckin**4)/
                mbkin**4 + (902*mckin**6)/mbkin**6 + (14355*mckin**8)/mbkin**8 - 
               (503*mckin**10)/mbkin**10 + (1208*mckin**12)/mbkin**12 + 
               (98*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
            (2*mckin**2*(-1184 + (8011*mckin**2)/mbkin**2 + (29884*mckin**4)/
                mbkin**4 + (29424*mckin**6)/mbkin**6 + (6156*mckin**8)/mbkin**8 + 
               (21*mckin**10)/mbkin**10)*q_cut**5)/mbkin**12 + 
            (2*(-70 + (692*mckin**2)/mbkin**2 - (1269*mckin**4)/mbkin**4 - 
               (6871*mckin**6)/mbkin**6 - (389*mckin**8)/mbkin**8 + 
               (147*mckin**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
            (2*(5 + (14*mckin**2)/mbkin**2 - (339*mckin**4)/mbkin**4 - 
               (284*mckin**6)/mbkin**6 + (4*mckin**8)/mbkin**8)*q_cut**7)/mbkin**14 - 
            (3*(-35 + (141*mckin**2)/mbkin**2 + (439*mckin**4)/mbkin**4 + 
               (71*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
            ((-45 + (176*mckin**2)/mbkin**2 + (41*mckin**4)/mbkin**4)*q_cut**9)/
             mbkin**18 + (24*mckin**2*q_cut**10)/mbkin**22) - 
          36*muG*mupi*((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
              mbkin**4 + q_cut**2/mbkin**4)**2*(25 + (789*mckin**2)/mbkin**2 - 
            (11707*mckin**4)/mbkin**4 + (28953*mckin**6)/mbkin**6 + 
            (73278*mckin**8)/mbkin**8 - (325002*mckin**10)/mbkin**10 - 
            (894522*mckin**12)/mbkin**12 - (477522*mckin**14)/mbkin**14 + 
            (113193*mckin**16)/mbkin**16 + (47093*mckin**18)/mbkin**18 - 
            (6027*mckin**20)/mbkin**20 - (71*mckin**22)/mbkin**22 + 
            ((-35 - (1504*mckin**2)/mbkin**2 + (15413*mckin**4)/mbkin**4 - 
               (30432*mckin**6)/mbkin**6 - (180506*mckin**8)/mbkin**8 - 
               (158416*mckin**10)/mbkin**10 - (166746*mckin**12)/mbkin**12 - 
               (202976*mckin**14)/mbkin**14 - (10627*mckin**16)/mbkin**16 + 
               (9968*mckin**18)/mbkin**18 + (101*mckin**20)/mbkin**20)*q_cut)/
             mbkin**2 + (2*(-30 - (544*mckin**2)/mbkin**2 + (7709*mckin**4)/
                mbkin**4 - (22753*mckin**6)/mbkin**6 - (158279*mckin**8)/
                mbkin**8 - (280433*mckin**10)/mbkin**10 - (161069*mckin**12)/
                mbkin**12 - (18783*mckin**14)/mbkin**14 + (5109*mckin**16)/
                mbkin**16 + (81*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
            (2*(-35 - (1674*mckin**2)/mbkin**2 + (14879*mckin**4)/mbkin**4 + 
               (17828*mckin**6)/mbkin**6 - (66689*mckin**8)/mbkin**8 - 
               (71234*mckin**10)/mbkin**10 + (21933*mckin**12)/mbkin**12 + 
               (10784*mckin**14)/mbkin**14 + (96*mckin**16)/mbkin**16)*q_cut**3)/
             mbkin**6 - (2*(-35 + (183*mckin**2)/mbkin**2 + (448*mckin**4)/
                mbkin**4 + (902*mckin**6)/mbkin**6 + (14355*mckin**8)/mbkin**8 - 
               (503*mckin**10)/mbkin**10 + (1208*mckin**12)/mbkin**12 + 
               (98*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
            (2*mckin**2*(-1184 + (8011*mckin**2)/mbkin**2 + (29884*mckin**4)/
                mbkin**4 + (29424*mckin**6)/mbkin**6 + (6156*mckin**8)/mbkin**8 + 
               (21*mckin**10)/mbkin**10)*q_cut**5)/mbkin**12 + 
            (2*(-70 + (692*mckin**2)/mbkin**2 - (1269*mckin**4)/mbkin**4 - 
               (6871*mckin**6)/mbkin**6 - (389*mckin**8)/mbkin**8 + 
               (147*mckin**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
            (2*(5 + (14*mckin**2)/mbkin**2 - (339*mckin**4)/mbkin**4 - 
               (284*mckin**6)/mbkin**6 + (4*mckin**8)/mbkin**8)*q_cut**7)/mbkin**14 - 
            (3*(-35 + (141*mckin**2)/mbkin**2 + (439*mckin**4)/mbkin**4 + 
               (71*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
            ((-45 + (176*mckin**2)/mbkin**2 + (41*mckin**4)/mbkin**4)*q_cut**9)/
             mbkin**18 + (24*mckin**2*q_cut**10)/mbkin**22) + 
          36*muG**2*((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
              mbkin**4 + q_cut**2/mbkin**4)**2*(-75 - (2767*mckin**2)/mbkin**2 + 
            (18697*mckin**4)/mbkin**4 - (48675*mckin**6)/mbkin**6 - 
            (55770*mckin**8)/mbkin**8 - (17058*mckin**10)/mbkin**10 - 
            (650082*mckin**12)/mbkin**12 - (1112826*mckin**14)/mbkin**14 + 
            (320421*mckin**16)/mbkin**16 + (128241*mckin**18)/mbkin**18 - 
            (31271*mckin**20)/mbkin**20 - (355*mckin**22)/mbkin**22 + 
            ((-95 - (2840*mckin**2)/mbkin**2 + (9705*mckin**4)/mbkin**4 - 
               (48896*mckin**6)/mbkin**6 - (295810*mckin**8)/mbkin**8 - 
               (193920*mckin**10)/mbkin**10 + (406750*mckin**12)/mbkin**12 - 
               (605248*mckin**14)/mbkin**14 - (45663*mckin**16)/mbkin**16 + 
               (49752*mckin**18)/mbkin**18 + (505*mckin**20)/mbkin**20)*q_cut)/
             mbkin**2 + (2*(30 + (436*mckin**2)/mbkin**2 - (26411*mckin**4)/
                mbkin**4 - (7345*mckin**6)/mbkin**6 - (184695*mckin**8)/mbkin**8 - 
               (575529*mckin**10)/mbkin**10 - (334469*mckin**12)/mbkin**12 - 
               (11887*mckin**14)/mbkin**14 + (26633*mckin**16)/mbkin**16 + 
               (405*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
            (2*(-215 - (5310*mckin**2)/mbkin**2 + (48067*mckin**4)/mbkin**4 + 
               (66136*mckin**6)/mbkin**6 - (122925*mckin**8)/mbkin**8 - 
               (294958*mckin**10)/mbkin**10 + (100777*mckin**12)/mbkin**12 + 
               (53196*mckin**14)/mbkin**14 + (480*mckin**16)/mbkin**16)*q_cut**3)/
             mbkin**6 - (2*(-75 + (535*mckin**2)/mbkin**2 - (8184*mckin**4)/
                mbkin**4 + (51670*mckin**6)/mbkin**6 + (174771*mckin**8)/
                mbkin**8 + (22097*mckin**10)/mbkin**10 + (6520*mckin**12)/
                mbkin**12 + (490*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
            ((-480 - (7288*mckin**2)/mbkin**2 + (79886*mckin**4)/mbkin**4 + 
               (239560*mckin**6)/mbkin**6 + (261440*mckin**8)/mbkin**8 + 
               (58976*mckin**10)/mbkin**10 + (210*mckin**12)/mbkin**12)*q_cut**5)/
             mbkin**10 + (2*(-170 + (2000*mckin**2)/mbkin**2 - (3941*mckin**4)/
                mbkin**4 - (35663*mckin**6)/mbkin**6 - (1921*mckin**8)/mbkin**8 + 
               (735*mckin**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
            (2*(25 - (206*mckin**2)/mbkin**2 - (415*mckin**4)/mbkin**4 - 
               (968*mckin**6)/mbkin**6 + (20*mckin**8)/mbkin**8)*q_cut**7)/mbkin**14 - 
            (3*(-175 + (705*mckin**2)/mbkin**2 + (2131*mckin**4)/mbkin**4 + 
               (355*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
            (5*(-45 + (176*mckin**2)/mbkin**2 + (41*mckin**4)/mbkin**4)*q_cut**9)/
             mbkin**18 + (120*mckin**2*q_cut**10)/mbkin**22) - 
          24*mbkin*(-((-1 + mckin**2/mbkin**2)**4*(1 + mckin**2/mbkin**2)**2*
              (2803 - (59389*mckin**2)/mbkin**2 + (425638*mckin**4)/mbkin**4 - 
               (142434*mckin**6)/mbkin**6 - (7007992*mckin**8)/mbkin**8 - 
               (411584*mckin**10)/mbkin**10 + (753258*mckin**12)/mbkin**12 - 
               (298894*mckin**14)/mbkin**14 + (24853*mckin**16)/mbkin**16 + 
               (461*mckin**18)/mbkin**18)) + ((-1 + mckin**2/mbkin**2)**2*
              (15765 - (259196*mckin**2)/mbkin**2 + (1031280*mckin**4)/mbkin**4 + 
               (5842740*mckin**6)/mbkin**6 - (24509183*mckin**8)/mbkin**8 - 
               (87162472*mckin**10)/mbkin**10 - (89842656*mckin**12)/mbkin**12 - 
               (25831304*mckin**14)/mbkin**14 + (7046591*mckin**16)/mbkin**16 - 
               (151836*mckin**18)/mbkin**18 - (1156752*mckin**20)/mbkin**20 + 
               (149588*mckin**22)/mbkin**22 + (2475*mckin**24)/mbkin**24)*q_cut)/
             mbkin**2 - (2*(15112 - (210025*mckin**2)/mbkin**2 + (472499*
                 mckin**4)/mbkin**4 + (5851497*mckin**6)/mbkin**6 - (11357051*
                 mckin**8)/mbkin**8 - (56972502*mckin**10)/mbkin**10 - 
               (85340102*mckin**12)/mbkin**12 - (61750930*mckin**14)/mbkin**14 - 
               (11562978*mckin**16)/mbkin**16 + (7576839*mckin**18)/mbkin**18 - 
               (785517*mckin**20)/mbkin**20 - (914135*mckin**22)/mbkin**22 + 
               (150229*mckin**24)/mbkin**24 + (2104*mckin**26)/mbkin**26)*q_cut**2)/
             mbkin**4 + (2*(4935 - (22515*mckin**2)/mbkin**2 - (355240*mckin**4)/
                mbkin**4 + (1715991*mckin**6)/mbkin**6 + (2655127*mckin**8)/
                mbkin**8 - (4426866*mckin**10)/mbkin**10 - (3311288*mckin**12)/
                mbkin**12 + (4104698*mckin**14)/mbkin**14 + (1043701*mckin**16)/
                mbkin**16 - (1487323*mckin**18)/mbkin**18 + (15840*mckin**20)/
                mbkin**20 + (62895*mckin**22)/mbkin**22 + (45*mckin**24)/
                mbkin**24)*q_cut**3)/mbkin**6 + ((43537 - (458971*mckin**2)/
                mbkin**2 - (93925*mckin**4)/mbkin**4 + (12415735*mckin**6)/
                mbkin**6 + (21452530*mckin**8)/mbkin**8 + (19866834*mckin**10)/
                mbkin**10 + (19989798*mckin**12)/mbkin**12 + (10268870*mckin**14)/
                mbkin**14 - (3574435*mckin**16)/mbkin**16 - (1439255*mckin**18)/
                mbkin**18 + (436447*mckin**20)/mbkin**20 + (7139*mckin**22)/
                mbkin**22)*q_cut**4)/mbkin**8 - ((58149 - (424380*mckin**2)/
                mbkin**2 - (1661061*mckin**4)/mbkin**4 + (11127360*mckin**6)/
                mbkin**6 + (29799650*mckin**8)/mbkin**8 + (27800000*mckin**10)/
                mbkin**10 + (5553270*mckin**12)/mbkin**12 - (6210720*mckin**14)/
                mbkin**14 - (461543*mckin**16)/mbkin**16 + (648988*mckin**18)/
                mbkin**18 + (6063*mckin**20)/mbkin**20)*q_cut**5)/mbkin**10 + 
            (4*(2930 - (20127*mckin**2)/mbkin**2 - (145371*mckin**4)/mbkin**4 + 
               (861647*mckin**6)/mbkin**6 + (1702217*mckin**8)/mbkin**8 + 
               (293611*mckin**10)/mbkin**10 - (560795*mckin**12)/mbkin**12 + 
               (4239*mckin**14)/mbkin**14 + (35411*mckin**16)/mbkin**16 + 
               (110*mckin**18)/mbkin**18)*q_cut**6)/mbkin**12 - 
            (4*(-5667 + (16545*mckin**2)/mbkin**2 + (282306*mckin**4)/mbkin**4 + 
               (576507*mckin**6)/mbkin**6 + (446228*mckin**8)/mbkin**8 + 
               (21045*mckin**10)/mbkin**10 - (206250*mckin**12)/mbkin**12 - 
               (83609*mckin**14)/mbkin**14 + (951*mckin**16)/mbkin**16)*q_cut**7)/
             mbkin**14 + ((-13425 + (43605*mckin**2)/mbkin**2 + (1070385*
                 mckin**4)/mbkin**4 + (1555763*mckin**6)/mbkin**6 + (247781*
                 mckin**8)/mbkin**8 - (859585*mckin**10)/mbkin**10 - 
               (271765*mckin**12)/mbkin**12 + (8265*mckin**14)/mbkin**14)*q_cut**8)/
             mbkin**16 + ((3211 - (5958*mckin**2)/mbkin**2 - (298825*mckin**4)/
                mbkin**4 - (133880*mckin**6)/mbkin**6 + (266305*mckin**8)/
                mbkin**8 + (71726*mckin**10)/mbkin**10 - (643*mckin**12)/
                mbkin**12)*q_cut**9)/mbkin**18 - (2*(3444 + (8075*mckin**2)/
                mbkin**2 + (7199*mckin**4)/mbkin**4 + (24877*mckin**6)/mbkin**6 + 
               (14553*mckin**8)/mbkin**8 + (4308*mckin**10)/mbkin**10)*q_cut**10)/
             mbkin**20 + (2*(3295 + (6917*mckin**2)/mbkin**2 + (13436*mckin**4)/
                mbkin**4 + (14159*mckin**6)/mbkin**6 + (3805*mckin**8)/mbkin**8)*
              q_cut**11)/mbkin**22 - ((1869 + (3645*mckin**2)/mbkin**2 + 
               (9103*mckin**4)/mbkin**4 + (2463*mckin**6)/mbkin**6)*q_cut**12)/
             mbkin**24 + (5*(mbkin**4 + 80*mbkin**2*mckin**2 + 59*mckin**4)*q_cut**13)/
             mbkin**30 - (48*(mbkin**2 + 2*mckin**2)*q_cut**14)/mbkin**30 + 
            (40*q_cut**15)/mbkin**30)*rhoD + ((mbkin**6 - 7*mbkin**4*mckin**2 - 
             7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
             mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)*
            (-16*(-((-1 + mckin**2/mbkin**2)**4*(-16529 + (91326*mckin**2)/
                   mbkin**2 + (305370*mckin**4)/mbkin**4 - (252698*mckin**6)/
                   mbkin**6 - (524676*mckin**8)/mbkin**8 - (176622*mckin**10)/
                   mbkin**10 + (103798*mckin**12)/mbkin**12 + (105354*mckin**14)/
                   mbkin**14 + (1797*mckin**16)/mbkin**16)) + 
               (2*(-1 + mckin**2/mbkin**2)**2*(-37623 + (116641*mckin**2)/
                   mbkin**2 + (808206*mckin**4)/mbkin**4 - (28274*mckin**6)/
                   mbkin**6 - (1319312*mckin**8)/mbkin**8 - (1291200*mckin**10)/
                   mbkin**10 - (352094*mckin**12)/mbkin**12 + (401986*mckin**14)/
                   mbkin**14 + (246231*mckin**16)/mbkin**16 + (3919*mckin**18)/
                   mbkin**18)*q_cut)/mbkin**2 + ((114885 - (167094*mckin**2)/
                   mbkin**2 - (2139307*mckin**4)/mbkin**4 + (43388*mckin**6)/
                   mbkin**6 + (2018354*mckin**8)/mbkin**8 + (2751296*mckin**10)/
                   mbkin**10 + (4252250*mckin**12)/mbkin**12 + 
                  (1266308*mckin**14)/mbkin**14 - (1541527*mckin**16)/mbkin**16 - 
                  (782218*mckin**18)/mbkin**18 - (10255*mckin**20)/mbkin**20)*
                 q_cut**2)/mbkin**4 - (2*(15345 + (35887*mckin**2)/mbkin**2 - 
                  (319224*mckin**4)/mbkin**4 + (127134*mckin**6)/mbkin**6 + 
                  (1286022*mckin**8)/mbkin**8 + (745104*mckin**10)/mbkin**10 - 
                  (914648*mckin**12)/mbkin**12 - (813694*mckin**14)/mbkin**14 - 
                  (131991*mckin**16)/mbkin**16 + (305*mckin**18)/mbkin**18)*q_cut**3)/
                mbkin**6 + (2*(-47995 - (123687*mckin**2)/mbkin**2 + 
                  (174111*mckin**4)/mbkin**4 + (564235*mckin**6)/mbkin**6 + 
                  (577645*mckin**8)/mbkin**8 + (446275*mckin**10)/mbkin**10 + 
                  (745753*mckin**12)/mbkin**12 + (310881*mckin**14)/mbkin**14 + 
                  (4830*mckin**16)/mbkin**16)*q_cut**4)/mbkin**8 - 
               (2*(-51671 - (201155*mckin**2)/mbkin**2 + (163523*mckin**4)/
                   mbkin**4 + (897993*mckin**6)/mbkin**6 + (1593699*mckin**8)/
                   mbkin**8 + (1338455*mckin**10)/mbkin**10 + (348873*mckin**12)/
                   mbkin**12 + (2523*mckin**14)/mbkin**14)*q_cut**5)/mbkin**10 + 
               (2*(-17689 - (72181*mckin**2)/mbkin**2 + (150624*mckin**4)/
                   mbkin**4 + (551050*mckin**6)/mbkin**6 + (492577*mckin**8)/
                   mbkin**8 + (109095*mckin**10)/mbkin**10 + (4452*mckin**12)/
                   mbkin**12)*q_cut**6)/mbkin**12 - (2*(-3065 - (177*mckin**2)/
                   mbkin**2 + (31314*mckin**4)/mbkin**4 + (22874*mckin**6)/
                   mbkin**6 + (17991*mckin**8)/mbkin**8 + (12615*mckin**10)/
                   mbkin**10)*q_cut**7)/mbkin**14 + ((-7235 + (7924*mckin**2)/
                   mbkin**2 + (34922*mckin**4)/mbkin**4 + (61668*mckin**6)/
                   mbkin**6 + (26385*mckin**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
               (16*(-290 + (471*mckin**2)/mbkin**2 + (1857*mckin**4)/mbkin**4 + 
                  (735*mckin**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
               ((-971 + (1612*mckin**2)/mbkin**2 + (2143*mckin**4)/mbkin**4)*
                 q_cut**10)/mbkin**20 - (8*(22 + (49*mckin**2)/mbkin**2)*q_cut**11)/
                mbkin**22 + (160*q_cut**12)/mbkin**24)*rE - 
             ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
               q_cut**2/mbkin**4)*(-4*(-((-1 + mckin**2/mbkin**2)**2*(-27053 + 
                    (86226*mckin**2)/mbkin**2 + (575046*mckin**4)/mbkin**4 - 
                    (609158*mckin**6)/mbkin**6 - (1436880*mckin**8)/mbkin**8 - 
                    (602802*mckin**10)/mbkin**10 + (123994*mckin**12)/mbkin**12 + 
                    (75174*mckin**14)/mbkin**14 + (1053*mckin**16)/mbkin**16)) + 
                 (4*(-17144 - (6564*mckin**2)/mbkin**2 + (379329*mckin**4)/
                     mbkin**4 - (45999*mckin**6)/mbkin**6 - (941649*mckin**8)/
                     mbkin**8 - (1045641*mckin**10)/mbkin**10 - 
                    (306441*mckin**12)/mbkin**12 + (120231*mckin**14)/mbkin**14 + 
                    (48849*mckin**16)/mbkin**16 + (629*mckin**18)/mbkin**18)*q_cut)/
                  mbkin**2 - (6*(-3605 - (29460*mckin**2)/mbkin**2 + 
                    (11114*mckin**4)/mbkin**4 - (6348*mckin**6)/mbkin**6 - 
                    (33800*mckin**8)/mbkin**8 + (77508*mckin**10)/mbkin**10 + 
                    (72846*mckin**12)/mbkin**12 + (12540*mckin**14)/mbkin**14 + 
                    (5*mckin**16)/mbkin**16)*q_cut**2)/mbkin**4 - 
                 (2*(-32263 - (142675*mckin**2)/mbkin**2 + (7835*mckin**4)/
                     mbkin**4 + (190079*mckin**6)/mbkin**6 + (442691*mckin**8)/
                     mbkin**8 + (369935*mckin**10)/mbkin**10 + (90345*mckin**12)/
                     mbkin**12 + (1413*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
                 ((-47518 - (330552*mckin**2)/mbkin**2 + (106806*mckin**4)/
                     mbkin**4 + (962320*mckin**6)/mbkin**6 + (758274*mckin**8)/
                     mbkin**8 + (148632*mckin**10)/mbkin**10 - (42*mckin**12)/
                     mbkin**12)*q_cut**4)/mbkin**8 + (24*(53 + (1383*mckin**2)/
                     mbkin**2 - (2122*mckin**4)/mbkin**4 - (3098*mckin**6)/
                     mbkin**6 - (303*mckin**8)/mbkin**8 + (7*mckin**10)/mbkin**10)*
                   q_cut**5)/mbkin**10 + ((-5854 + (3896*mckin**2)/mbkin**2 - 
                    (7780*mckin**4)/mbkin**4 + (8184*mckin**6)/mbkin**6 + 
                    (4914*mckin**8)/mbkin**8)*q_cut**6)/mbkin**12 - 
                 (10*(-917 + (375*mckin**2)/mbkin**2 + (1749*mckin**4)/mbkin**4 + 
                    (489*mckin**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
                 (3*(-557 + (540*mckin**2)/mbkin**2 + (377*mckin**4)/mbkin**4)*
                   q_cut**8)/mbkin**16 + (72*(-1 + mckin**2/mbkin**2)*q_cut**9)/
                  mbkin**18 + (40*q_cut**10)/mbkin**20)*rG - 12*
                (-((-1 + mckin**2/mbkin**2)**2*(5857 + (22438*mckin**2)/mbkin**2 - 
                    (394686*mckin**4)/mbkin**4 - (890034*mckin**6)/mbkin**6 - 
                    (634464*mckin**8)/mbkin**8 - (262246*mckin**10)/mbkin**10 + 
                    (35198*mckin**12)/mbkin**12 + (60722*mckin**14)/mbkin**14 + 
                    (895*mckin**16)/mbkin**16)) + (4*(3566 + (30072*mckin**2)/
                     mbkin**2 - (173049*mckin**4)/mbkin**4 - (699873*mckin**6)/
                     mbkin**6 - (769467*mckin**8)/mbkin**8 - (377943*mckin**10)/
                     mbkin**10 - (159039*mckin**12)/mbkin**12 + (49257*mckin**14)/
                     mbkin**14 + (39621*mckin**16)/mbkin**16 + (535*mckin**18)/
                     mbkin**18)*q_cut)/mbkin**2 - (2*(1905 + (41030*mckin**2)/
                     mbkin**2 + (64112*mckin**4)/mbkin**4 - (1502*mckin**6)/
                     mbkin**6 + (24078*mckin**8)/mbkin**8 + (60282*mckin**10)/
                     mbkin**10 + (121880*mckin**12)/mbkin**12 + (30910*mckin**14)/
                     mbkin**14 + (25*mckin**16)/mbkin**16)*q_cut**2)/mbkin**4 - 
                 (2*(6707 + (83567*mckin**2)/mbkin**2 + (176017*mckin**4)/
                     mbkin**4 + (278501*mckin**6)/mbkin**6 + (332897*mckin**8)/
                     mbkin**8 + (246037*mckin**10)/mbkin**10 + (73947*mckin**12)/
                     mbkin**12 + (1175*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
                 ((9642 + (159460*mckin**2)/mbkin**2 + (478626*mckin**4)/
                     mbkin**4 + (641576*mckin**6)/mbkin**6 + (516490*mckin**8)/
                     mbkin**8 + (123972*mckin**10)/mbkin**10 - (70*mckin**12)/
                     mbkin**12)*q_cut**4)/mbkin**8 + (8*(-81 - (1917*mckin**2)/
                     mbkin**2 - (4518*mckin**4)/mbkin**4 - (7162*mckin**6)/
                     mbkin**6 - (901*mckin**8)/mbkin**8 + (35*mckin**10)/
                     mbkin**10)*q_cut**5)/mbkin**10 + (2*(-107 - (654*mckin**2)/
                     mbkin**2 - (184*mckin**4)/mbkin**4 + (3018*mckin**6)/
                     mbkin**6 + (1855*mckin**8)/mbkin**8)*q_cut**6)/mbkin**12 - 
                 (10*(-75 + (469*mckin**2)/mbkin**2 + (1395*mckin**4)/mbkin**4 + 
                    (355*mckin**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
                 ((-681 + (1384*mckin**2)/mbkin**2 + (625*mckin**4)/mbkin**4)*
                   q_cut**8)/mbkin**16 + (24*(-3 + (5*mckin**2)/mbkin**2)*q_cut**9)/
                  mbkin**18 + (40*q_cut**10)/mbkin**20)*sB - 70480*sE + 
               (105488*mckin**2*sE)/mbkin**2 + (2685536*mckin**4*sE)/mbkin**4 - 
               (2646400*mckin**6*sE)/mbkin**6 - (1847776*mckin**8*sE)/mbkin**8 + 
               (1481312*mckin**10*sE)/mbkin**10 - (685888*mckin**12*sE)/
                mbkin**12 + (827264*mckin**14*sE)/mbkin**14 + (580400*mckin**16*
                 sE)/mbkin**16 - (423024*mckin**18*sE)/mbkin**18 - 
               (6432*mckin**20*sE)/mbkin**20 + (175840*q_cut*sE)/mbkin**2 + 
               (763584*mckin**2*q_cut*sE)/mbkin**4 - (5905152*mckin**4*q_cut*sE)/
                mbkin**6 - (11687616*mckin**6*q_cut*sE)/mbkin**8 - (6505152*mckin**8*
                 q_cut*sE)/mbkin**10 - (2639808*mckin**10*q_cut*sE)/mbkin**12 - 
               (2264832*mckin**12*q_cut*sE)/mbkin**14 + (1738944*mckin**14*q_cut*sE)/
                mbkin**16 + (1149408*mckin**16*q_cut*sE)/mbkin**18 + (15104*
                 mckin**18*q_cut*sE)/mbkin**20 - (50880*q_cut**2*sE)/mbkin**4 - 
               (734880*mckin**2*q_cut**2*sE)/mbkin**6 - (354720*mckin**4*q_cut**2*sE)/
                mbkin**8 + (1468896*mckin**6*q_cut**2*sE)/mbkin**10 - (348576*
                 mckin**8*q_cut**2*sE)/mbkin**12 - (3542496*mckin**10*q_cut**2*sE)/
                mbkin**14 - (2898144*mckin**12*q_cut**2*sE)/mbkin**16 - 
               (474720*mckin**14*q_cut**2*sE)/mbkin**18 + (480*mckin**16*q_cut**2*sE)/
                mbkin**20 - (165760*q_cut**3*sE)/mbkin**6 - (1377664*mckin**2*q_cut**3*
                 sE)/mbkin**8 - (1563904*mckin**4*q_cut**3*sE)/mbkin**10 - 
               (872704*mckin**6*q_cut**3*sE)/mbkin**12 - (531712*mckin**8*q_cut**3*sE)/
                mbkin**14 - (2099584*mckin**10*q_cut**3*sE)/mbkin**16 - (1037184*
                 mckin**12*q_cut**3*sE)/mbkin**18 - (17664*mckin**14*q_cut**3*sE)/
                mbkin**20 + (116000*q_cut**4*sE)/mbkin**8 + (1421952*mckin**2*q_cut**4*
                 sE)/mbkin**10 + (2406240*mckin**4*q_cut**4*sE)/mbkin**12 + 
               (2765056*mckin**6*q_cut**4*sE)/mbkin**14 + (3099936*mckin**8*q_cut**4*sE)/
                mbkin**16 + (872640*mckin**10*q_cut**4*sE)/mbkin**18 + (672*mckin**12*
                 q_cut**4*sE)/mbkin**20 - (3840*q_cut**5*sE)/mbkin**10 - (137088*
                 mckin**2*q_cut**5*sE)/mbkin**12 - (262656*mckin**4*q_cut**5*sE)/
                mbkin**14 - (336384*mckin**6*q_cut**5*sE)/mbkin**16 - (13056*mckin**8*
                 q_cut**5*sE)/mbkin**18 - (2688*mckin**10*q_cut**5*sE)/mbkin**20 + 
               (15200*q_cut**6*sE)/mbkin**12 - (20224*mckin**2*q_cut**6*sE)/mbkin**14 - 
               (55168*mckin**4*q_cut**6*sE)/mbkin**16 + (3072*mckin**6*q_cut**6*sE)/
                mbkin**18 + (35616*mckin**8*q_cut**6*sE)/mbkin**20 - (14560*q_cut**7*sE)/
                mbkin**14 - (27840*mckin**2*q_cut**7*sE)/mbkin**16 - (84960*mckin**4*
                 q_cut**7*sE)/mbkin**18 - (34560*mckin**6*q_cut**7*sE)/mbkin**20 - 
               (2160*q_cut**8*sE)/mbkin**16 + (7824*mckin**2*q_cut**8*sE)/mbkin**18 + 
               (9984*mckin**4*q_cut**8*sE)/mbkin**20 - (1152*mckin**2*q_cut**9*sE)/
                mbkin**20 + (640*q_cut**10*sE)/mbkin**20 - 7651*sqB - 
               (17644*mckin**2*sqB)/mbkin**2 + (688991*mckin**4*sqB)/mbkin**4 + 
               (773432*mckin**6*sqB)/mbkin**6 - (1727530*mckin**8*sqB)/mbkin**8 - 
               (969472*mckin**10*sqB)/mbkin**10 + (709394*mckin**12*sqB)/
                mbkin**12 + (562184*mckin**14*sqB)/mbkin**14 + (9293*mckin**16*
                 sqB)/mbkin**16 - (20820*mckin**18*sqB)/mbkin**18 - 
               (177*mckin**20*sqB)/mbkin**20 + (18532*q_cut*sqB)/mbkin**2 + 
               (170328*mckin**2*q_cut*sqB)/mbkin**4 - (941220*mckin**4*q_cut*sqB)/
                mbkin**6 - (5610852*mckin**6*q_cut*sqB)/mbkin**8 - (8028036*mckin**8*
                 q_cut*sqB)/mbkin**10 - (4624524*mckin**10*q_cut*sqB)/mbkin**12 - 
               (851580*mckin**12*q_cut*sqB)/mbkin**14 + (216708*mckin**14*q_cut*sqB)/
                mbkin**16 + (54720*mckin**16*q_cut*sqB)/mbkin**18 + (404*mckin**18*
                 q_cut*sqB)/mbkin**20 - (5070*q_cut**2*sqB)/mbkin**4 - (109200*mckin**2*
                 q_cut**2*sqB)/mbkin**6 - (277932*mckin**4*q_cut**2*sqB)/mbkin**8 - 
               (147432*mckin**6*q_cut**2*sqB)/mbkin**10 - (336624*mckin**8*q_cut**2*sqB)/
                mbkin**12 - (512448*mckin**10*q_cut**2*sqB)/mbkin**14 - 
               (224244*mckin**12*q_cut**2*sqB)/mbkin**16 - (20040*mckin**14*q_cut**2*
                 sqB)/mbkin**18 + (30*mckin**16*q_cut**2*sqB)/mbkin**20 - 
               (17422*q_cut**3*sqB)/mbkin**6 - (229786*mckin**2*q_cut**3*sqB)/mbkin**8 - 
               (532666*mckin**4*q_cut**3*sqB)/mbkin**10 - (503470*mckin**6*q_cut**3*sqB)/
                mbkin**12 - (496234*mckin**8*q_cut**3*sqB)/mbkin**14 - (294526*
                 mckin**10*q_cut**3*sqB)/mbkin**16 - (50286*mckin**12*q_cut**3*sqB)/
                mbkin**18 - (474*mckin**14*q_cut**3*sqB)/mbkin**20 + (13586*q_cut**4*
                 sqB)/mbkin**8 + (212652*mckin**2*q_cut**4*sqB)/mbkin**10 + 
               (757458*mckin**4*q_cut**4*sqB)/mbkin**12 + (806704*mckin**6*q_cut**4*sqB)/
                mbkin**14 + (343986*mckin**8*q_cut**4*sqB)/mbkin**16 + 
               (39636*mckin**10*q_cut**4*sqB)/mbkin**18 + (42*mckin**12*q_cut**4*sqB)/
                mbkin**20 - (1464*q_cut**5*sqB)/mbkin**10 - (19656*mckin**2*q_cut**5*
                 sqB)/mbkin**12 - (54288*mckin**4*q_cut**5*sqB)/mbkin**14 - 
               (34992*mckin**6*q_cut**5*sqB)/mbkin**16 - (600*mckin**8*q_cut**5*sqB)/
                mbkin**18 - (168*mckin**10*q_cut**5*sqB)/mbkin**20 - (1702*q_cut**6*sqB)/
                mbkin**12 - (6028*mckin**2*q_cut**6*sqB)/mbkin**14 - (13792*mckin**4*
                 q_cut**6*sqB)/mbkin**16 + (1500*mckin**6*q_cut**6*sqB)/mbkin**18 + 
               (966*mckin**8*q_cut**6*sqB)/mbkin**20 + (2330*q_cut**7*sqB)/mbkin**14 - 
               (1470*mckin**2*q_cut**7*sqB)/mbkin**16 - (4770*mckin**4*q_cut**7*sqB)/
                mbkin**18 - (810*mckin**6*q_cut**7*sqB)/mbkin**20 - (963*q_cut**8*sqB)/
                mbkin**16 + (876*mckin**2*q_cut**8*sqB)/mbkin**18 + (219*mckin**4*
                 q_cut**8*sqB)/mbkin**20 - (216*q_cut**9*sqB)/mbkin**18 - (72*mckin**2*
                 q_cut**9*sqB)/mbkin**20 + (40*q_cut**10*sqB)/mbkin**20)))/mbkin**6) - 
        12*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)/mbkin**4)*
         (16*(-((-1 + mckin**2/mbkin**2)**4*(-380 + (2665*mckin**2)/mbkin**2 + 
               (51248*mckin**4)/mbkin**4 - (302302*mckin**6)/mbkin**6 - 
               (716770*mckin**8)/mbkin**8 + (773636*mckin**10)/mbkin**10 + 
               (1352942*mckin**12)/mbkin**12 + (338714*mckin**14)/mbkin**14 - 
               (183346*mckin**16)/mbkin**16 - (220253*mckin**18)/mbkin**18 - 
               (10574*mckin**20)/mbkin**20 + (3060*mckin**22)/mbkin**22)) + 
            (2*(-1 + mckin**2/mbkin**2)**2*(-1145 + (5525*mckin**2)/mbkin**2 + 
               (132776*mckin**4)/mbkin**4 - (475932*mckin**6)/mbkin**6 - 
               (2047932*mckin**8)/mbkin**8 + (363678*mckin**10)/mbkin**10 + 
               (3806364*mckin**12)/mbkin**12 + (3201960*mckin**14)/mbkin**14 + 
               (640143*mckin**16)/mbkin**16 - (713347*mckin**18)/mbkin**18 - 
               (545492*mckin**20)/mbkin**20 - (21308*mckin**22)/mbkin**22 + 
               (9270*mckin**24)/mbkin**24)*q_cut)/mbkin**2 - 
            ((-4980 + (15565*mckin**2)/mbkin**2 + (460790*mckin**4)/mbkin**4 - 
               (1082073*mckin**6)/mbkin**6 - (5400314*mckin**8)/mbkin**8 + 
               (329986*mckin**10)/mbkin**10 + (6336868*mckin**12)/mbkin**12 + 
               (9252922*mckin**14)/mbkin**14 + (9659440*mckin**16)/mbkin**16 + 
               (2356861*mckin**18)/mbkin**18 - (2620314*mckin**20)/mbkin**20 - 
               (1870561*mckin**22)/mbkin**22 - (56450*mckin**24)/mbkin**24 + 
               (40500*mckin**26)/mbkin**26)*q_cut**2)/mbkin**4 + 
            (4*(-770 - (425*mckin**2)/mbkin**2 + (42860*mckin**4)/mbkin**4 - 
               (6553*mckin**6)/mbkin**6 - (460604*mckin**8)/mbkin**8 + 
               (213734*mckin**10)/mbkin**10 + (1527392*mckin**12)/mbkin**12 + 
               (766954*mckin**14)/mbkin**14 - (799798*mckin**16)/mbkin**16 - 
               (839309*mckin**18)/mbkin**18 - (191076*mckin**20)/mbkin**20 + 
               (13455*mckin**22)/mbkin**22 + (6300*mckin**24)/mbkin**24)*q_cut**3)/
             mbkin**6 + (2*(-2665 - (3660*mckin**2)/mbkin**2 + (216220*mckin**4)/
                mbkin**4 + (333804*mckin**6)/mbkin**6 - (482042*mckin**8)/
                mbkin**8 - (1290220*mckin**10)/mbkin**10 - (1494860*mckin**12)/
                mbkin**12 - (1281730*mckin**14)/mbkin**14 - (1412601*mckin**16)/
                mbkin**16 - (676392*mckin**18)/mbkin**18 + (22620*mckin**20)/
                mbkin**20 + (21510*mckin**22)/mbkin**22)*q_cut**4)/mbkin**8 - 
            (4*(-2655 - (9270*mckin**2)/mbkin**2 + (135271*mckin**4)/mbkin**4 + 
               (293165*mckin**6)/mbkin**6 - (273693*mckin**8)/mbkin**8 - 
               (1178103*mckin**10)/mbkin**10 - (1794639*mckin**12)/mbkin**12 - 
               (1361095*mckin**14)/mbkin**14 - (355238*mckin**16)/mbkin**16 + 
               (69807*mckin**18)/mbkin**18 + (21330*mckin**20)/mbkin**20)*q_cut**5)/
             mbkin**10 + (2*(-2640 - (8715*mckin**2)/mbkin**2 + (123148*mckin**4)/
                mbkin**4 + (214717*mckin**6)/mbkin**6 - (539178*mckin**8)/
                mbkin**8 - (1417455*mckin**10)/mbkin**10 - (1141044*mckin**12)/
                mbkin**12 - (236615*mckin**14)/mbkin**14 + (69426*mckin**16)/
                mbkin**16 + (21060*mckin**18)/mbkin**18)*q_cut**6)/mbkin**12 + 
            (4*(-690 - (8535*mckin**2)/mbkin**2 - (20320*mckin**4)/mbkin**4 + 
               (23073*mckin**6)/mbkin**6 + (142274*mckin**8)/mbkin**8 + 
               (174489*mckin**10)/mbkin**10 + (129296*mckin**12)/mbkin**12 + 
               (60345*mckin**14)/mbkin**14 + (4860*mckin**16)/mbkin**16)*q_cut**7)/
             mbkin**14 - ((-4320 - (43275*mckin**2)/mbkin**2 - (35240*mckin**4)/
                mbkin**4 + (192068*mckin**6)/mbkin**6 + (508934*mckin**8)/
                mbkin**8 + (602131*mckin**10)/mbkin**10 + (312570*mckin**12)/
                mbkin**12 + (28620*mckin**14)/mbkin**14)*q_cut**8)/mbkin**16 + 
            (2*(-985 - (9315*mckin**2)/mbkin**2 - (1095*mckin**4)/mbkin**4 + 
               (62681*mckin**6)/mbkin**6 + (118342*mckin**8)/mbkin**8 + 
               (67430*mckin**10)/mbkin**10 + (6390*mckin**12)/mbkin**12)*q_cut**9)/
             mbkin**18 - ((-700 - (3555*mckin**2)/mbkin**2 + (4658*mckin**4)/
                mbkin**4 + (29699*mckin**6)/mbkin**6 + (28886*mckin**8)/mbkin**8 + 
               (8820*mckin**10)/mbkin**10)*q_cut**10)/mbkin**20 + 
            (32*(-15 - (25*mckin**2)/mbkin**2 + (131*mckin**4)/mbkin**4 + 
               (317*mckin**6)/mbkin**6 + (270*mckin**8)/mbkin**8)*q_cut**11)/
             mbkin**22 - (10*(-23 - (23*mckin**2)/mbkin**2 + (318*mckin**4)/
                mbkin**4 + (414*mckin**6)/mbkin**6)*q_cut**12)/mbkin**24 - 
            (40*(mbkin**4 - 18*mckin**4)*q_cut**13)/mbkin**30)*rE + 
          ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)*((3780*mckin**2*muG**2)/mbkin**2 - 
            (23760*mckin**4*muG**2)/mbkin**4 - (39348*mckin**6*muG**2)/mbkin**6 + 
            (673560*mckin**8*muG**2)/mbkin**8 - (1433376*mckin**10*muG**2)/
             mbkin**10 + (4650048*mckin**12*muG**2)/mbkin**12 + 
            (2922696*mckin**14*muG**2)/mbkin**14 - (16610832*mckin**16*muG**2)/
             mbkin**16 + (7950420*mckin**18*muG**2)/mbkin**18 + 
            (3388176*mckin**20*muG**2)/mbkin**20 - (1472148*mckin**22*muG**2)/
             mbkin**22 - (39816*mckin**24*muG**2)/mbkin**24 + 
            (30600*mckin**26*muG**2)/mbkin**26 + (1260*mckin**2*muG*mupi)/
             mbkin**2 - (15840*mckin**4*muG*mupi)/mbkin**4 - 
            (41148*mckin**6*muG*mupi)/mbkin**6 + (855864*mckin**8*muG*mupi)/
             mbkin**8 - (2008800*mckin**10*muG*mupi)/mbkin**10 - 
            (4847904*mckin**12*muG*mupi)/mbkin**12 + (5623128*mckin**14*muG*
              mupi)/mbkin**14 + (7036848*mckin**16*muG*mupi)/mbkin**16 - 
            (5620644*mckin**18*muG*mupi)/mbkin**18 - (1584000*mckin**20*muG*
              mupi)/mbkin**20 + (577764*mckin**22*muG*mupi)/mbkin**22 + 
            (29592*mckin**24*muG*mupi)/mbkin**24 - (6120*mckin**26*muG*mupi)/
             mbkin**26 - (5040*mckin**2*muG**2*q_cut)/mbkin**4 + 
            (13320*mckin**4*muG**2*q_cut)/mbkin**6 - (7776*mckin**6*muG**2*q_cut)/
             mbkin**8 + (1056600*mckin**8*muG**2*q_cut)/mbkin**10 - 
            (2260800*mckin**10*muG**2*q_cut)/mbkin**12 - (9882000*mckin**12*muG**2*
              q_cut)/mbkin**14 - (20960928*mckin**14*muG**2*q_cut)/mbkin**16 - 
            (25217712*mckin**16*muG**2*q_cut)/mbkin**18 + 
            (700272*mckin**18*muG**2*q_cut)/mbkin**20 + (4524552*mckin**20*muG**2*q_cut)/
             mbkin**22 - (91008*mckin**22*muG**2*q_cut)/mbkin**24 - 
            (124200*mckin**24*muG**2*q_cut)/mbkin**26 - (5040*mckin**2*muG*mupi*q_cut)/
             mbkin**4 + (47160*mckin**4*muG*mupi*q_cut)/mbkin**6 + 
            (197856*mckin**6*muG*mupi*q_cut)/mbkin**8 - (2047608*mckin**8*muG*mupi*
              q_cut)/mbkin**10 - (561024*mckin**10*muG*mupi*q_cut)/mbkin**12 + 
            (15349392*mckin**12*muG*mupi*q_cut)/mbkin**14 + 
            (25799328*mckin**14*muG*mupi*q_cut)/mbkin**16 + 
            (16085232*mckin**16*muG*mupi*q_cut)/mbkin**18 - 
            (815184*mckin**18*muG*mupi*q_cut)/mbkin**20 - 
            (1783368*mckin**20*muG*mupi*q_cut)/mbkin**22 - 
            (36864*mckin**22*muG*mupi*q_cut)/mbkin**24 + (24840*mckin**24*muG*mupi*
              q_cut)/mbkin**26 - (5040*mckin**2*muG**2*q_cut**2)/mbkin**6 + 
            (40320*mckin**4*muG**2*q_cut**2)/mbkin**8 + (177840*mckin**6*muG**2*q_cut**2)/
             mbkin**10 - (877248*mckin**8*muG**2*q_cut**2)/mbkin**12 - 
            (3926304*mckin**10*muG**2*q_cut**2)/mbkin**14 + 
            (735264*mckin**12*muG**2*q_cut**2)/mbkin**16 + 
            (4323744*mckin**14*muG**2*q_cut**2)/mbkin**18 - 
            (2464704*mckin**16*muG**2*q_cut**2)/mbkin**20 - 
            (2724912*mckin**18*muG**2*q_cut**2)/mbkin**22 + 
            (240480*mckin**20*muG**2*q_cut**2)/mbkin**24 + 
            (126000*mckin**22*muG**2*q_cut**2)/mbkin**26 + 
            (5040*mckin**2*muG*mupi*q_cut**2)/mbkin**6 - (30240*mckin**4*muG*mupi*
              q_cut**2)/mbkin**8 - (169200*mckin**6*muG*mupi*q_cut**2)/mbkin**10 + 
            (720000*mckin**8*muG*mupi*q_cut**2)/mbkin**12 + 
            (2313504*mckin**10*muG*mupi*q_cut**2)/mbkin**14 - 
            (372384*mckin**12*muG*mupi*q_cut**2)/mbkin**16 - 
            (815904*mckin**14*muG*mupi*q_cut**2)/mbkin**18 + 
            (1880064*mckin**16*muG*mupi*q_cut**2)/mbkin**20 + 
            (886320*mckin**18*muG*mupi*q_cut**2)/mbkin**22 - 
            (37440*mckin**20*muG*mupi*q_cut**2)/mbkin**24 - 
            (25200*mckin**22*muG*mupi*q_cut**2)/mbkin**26 - 
            (5040*mckin**2*muG**2*q_cut**3)/mbkin**8 - (11880*mckin**4*muG**2*q_cut**3)/
             mbkin**10 + (407808*mckin**6*muG**2*q_cut**3)/mbkin**12 - 
            (2325168*mckin**8*muG**2*q_cut**3)/mbkin**14 - (12438432*mckin**10*muG**2*
              q_cut**3)/mbkin**16 - (17994816*mckin**12*muG**2*q_cut**3)/mbkin**18 - 
            (13595904*mckin**14*muG**2*q_cut**3)/mbkin**20 - 
            (2472912*mckin**16*muG**2*q_cut**3)/mbkin**22 + 
            (702288*mckin**18*muG**2*q_cut**3)/mbkin**24 + 
            (124200*mckin**20*muG**2*q_cut**3)/mbkin**26 + 
            (5040*mckin**2*muG*mupi*q_cut**3)/mbkin**8 - (39960*mckin**4*muG*mupi*
              q_cut**3)/mbkin**10 - (266976*mckin**6*muG*mupi*q_cut**3)/mbkin**12 + 
            (1733904*mckin**8*muG*mupi*q_cut**3)/mbkin**14 + 
            (6554304*mckin**10*muG*mupi*q_cut**3)/mbkin**16 + 
            (8608032*mckin**12*muG*mupi*q_cut**3)/mbkin**18 + 
            (6371424*mckin**14*muG*mupi*q_cut**3)/mbkin**20 + 
            (1519344*mckin**16*muG*mupi*q_cut**3)/mbkin**22 - 
            (74736*mckin**18*muG*mupi*q_cut**3)/mbkin**24 - 
            (24840*mckin**20*muG*mupi*q_cut**3)/mbkin**26 + 
            (32760*mckin**2*muG**2*q_cut**4)/mbkin**10 - (81000*mckin**4*muG**2*q_cut**4)/
             mbkin**12 - (1250064*mckin**6*muG**2*q_cut**4)/mbkin**14 + 
            (1785384*mckin**8*muG**2*q_cut**4)/mbkin**16 + (13305744*mckin**10*muG**2*
              q_cut**4)/mbkin**18 + (14415336*mckin**12*muG**2*q_cut**4)/mbkin**20 + 
            (2808000*mckin**14*muG**2*q_cut**4)/mbkin**22 - 
            (1709064*mckin**16*muG**2*q_cut**4)/mbkin**24 - 
            (307800*mckin**18*muG**2*q_cut**4)/mbkin**26 - 
            (12600*mckin**2*muG*mupi*q_cut**4)/mbkin**10 + 
            (66600*mckin**4*muG*mupi*q_cut**4)/mbkin**12 + 
            (499248*mckin**6*muG*mupi*q_cut**4)/mbkin**14 - 
            (1575720*mckin**8*muG*mupi*q_cut**4)/mbkin**16 - 
            (6865776*mckin**10*muG*mupi*q_cut**4)/mbkin**18 - 
            (6544296*mckin**12*muG*mupi*q_cut**4)/mbkin**20 - 
            (1386720*mckin**14*muG*mupi*q_cut**4)/mbkin**22 + 
            (305928*mckin**16*muG*mupi*q_cut**4)/mbkin**24 + 
            (61560*mckin**18*muG*mupi*q_cut**4)/mbkin**26 - 
            (15120*mckin**2*muG**2*q_cut**5)/mbkin**12 + (68040*mckin**4*muG**2*q_cut**5)/
             mbkin**14 + (291168*mckin**6*muG**2*q_cut**5)/mbkin**16 - 
            (2070936*mckin**8*muG**2*q_cut**5)/mbkin**18 - 
            (5521680*mckin**10*muG**2*q_cut**5)/mbkin**20 - 
            (2596392*mckin**12*muG**2*q_cut**5)/mbkin**22 + 
            (455616*mckin**14*muG**2*q_cut**5)/mbkin**24 + 
            (113400*mckin**16*muG**2*q_cut**5)/mbkin**26 + 
            (5040*mckin**2*muG*mupi*q_cut**5)/mbkin**12 - 
            (27720*mckin**4*muG*mupi*q_cut**5)/mbkin**14 - 
            (120672*mckin**6*muG*mupi*q_cut**5)/mbkin**16 + 
            (514296*mckin**8*muG*mupi*q_cut**5)/mbkin**18 + 
            (1383984*mckin**10*muG*mupi*q_cut**5)/mbkin**20 + 
            (487656*mckin**12*muG*mupi*q_cut**5)/mbkin**22 - 
            (104832*mckin**14*muG*mupi*q_cut**5)/mbkin**24 - 
            (22680*mckin**16*muG*mupi*q_cut**5)/mbkin**26 - 
            (25200*mckin**2*muG**2*q_cut**6)/mbkin**14 + (68040*mckin**4*muG**2*q_cut**6)/
             mbkin**16 + (1398456*mckin**6*muG**2*q_cut**6)/mbkin**18 + 
            (3709584*mckin**8*muG**2*q_cut**6)/mbkin**20 + 
            (3680928*mckin**10*muG**2*q_cut**6)/mbkin**22 + 
            (1273608*mckin**12*muG**2*q_cut**6)/mbkin**24 + 
            (113400*mckin**14*muG**2*q_cut**6)/mbkin**26 + 
            (5040*mckin**2*muG*mupi*q_cut**6)/mbkin**14 - 
            (7560*mckin**4*muG*mupi*q_cut**6)/mbkin**16 - (302616*mckin**6*muG*mupi*
              q_cut**6)/mbkin**18 - (713232*mckin**8*muG*mupi*q_cut**6)/mbkin**20 - 
            (725472*mckin**10*muG*mupi*q_cut**6)/mbkin**22 - 
            (261576*mckin**12*muG*mupi*q_cut**6)/mbkin**24 - 
            (22680*mckin**14*muG*mupi*q_cut**6)/mbkin**26 + 
            (25200*mckin**2*muG**2*q_cut**7)/mbkin**16 - (94680*mckin**4*muG**2*q_cut**7)/
             mbkin**18 - (1141920*mckin**6*muG**2*q_cut**7)/mbkin**20 - 
            (2010528*mckin**8*muG**2*q_cut**7)/mbkin**22 - 
            (1093680*mckin**10*muG**2*q_cut**7)/mbkin**24 - 
            (81000*mckin**12*muG**2*q_cut**7)/mbkin**26 - (5040*mckin**2*muG*mupi*
              q_cut**7)/mbkin**16 + (8280*mckin**4*muG*mupi*q_cut**7)/mbkin**18 + 
            (264960*mckin**6*muG*mupi*q_cut**7)/mbkin**20 + 
            (486720*mckin**8*muG*mupi*q_cut**7)/mbkin**22 + 
            (234000*mckin**10*muG*mupi*q_cut**7)/mbkin**24 + 
            (16200*mckin**12*muG*mupi*q_cut**7)/mbkin**26 - 
            (6300*mckin**2*muG**2*q_cut**8)/mbkin**18 + (27000*mckin**4*muG**2*q_cut**8)/
             mbkin**20 + (288612*mckin**6*muG**2*q_cut**8)/mbkin**22 + 
            (305712*mckin**8*muG**2*q_cut**8)/mbkin**24 + (10800*mckin**10*muG**2*
              q_cut**8)/mbkin**26 + (1260*mckin**2*muG*mupi*q_cut**8)/mbkin**18 - 
            (1800*mckin**4*muG*mupi*q_cut**8)/mbkin**20 - 
            (67284*mckin**6*muG*mupi*q_cut**8)/mbkin**22 - 
            (59904*mckin**8*muG*mupi*q_cut**8)/mbkin**24 - 
            (2160*mckin**10*muG*mupi*q_cut**8)/mbkin**26 - 
            (25200*mckin**4*muG**2*q_cut**9)/mbkin**22 - (71136*mckin**6*muG**2*q_cut**9)/
             mbkin**24 - (25200*mckin**8*muG**2*q_cut**9)/mbkin**26 + 
            (5040*mckin**4*muG*mupi*q_cut**9)/mbkin**22 + 
            (11232*mckin**6*muG*mupi*q_cut**9)/mbkin**24 + 
            (5040*mckin**8*muG*mupi*q_cut**9)/mbkin**26 + 
            (27000*mckin**4*muG**2*q_cut**10)/mbkin**24 + (27000*mckin**6*muG**2*
              q_cut**10)/mbkin**26 - (5400*mckin**4*muG*mupi*q_cut**10)/mbkin**24 - 
            (5400*mckin**6*muG*mupi*q_cut**10)/mbkin**26 - 
            (7200*mckin**4*muG**2*q_cut**11)/mbkin**26 + (1440*mckin**4*muG*mupi*
              q_cut**11)/mbkin**26 - 72*mckin**2*muG*((-1 + mckin**2/mbkin**2)**2*(
                -35 + (370*mckin**2)/mbkin**2 + (1918*mckin**4)/mbkin**4 - 
                (20308*mckin**6)/mbkin**6 + (13266*mckin**8)/mbkin**8 + 
                (181504*mckin**10)/mbkin**10 + (193544*mckin**12)/mbkin**12 + 
                (10116*mckin**14)/mbkin**14 - (17183*mckin**16)/mbkin**16 - 
                (482*mckin**18)/mbkin**18 + (170*mckin**20)/mbkin**20) - 
              (2*(-70 + (655*mckin**2)/mbkin**2 + (2748*mckin**4)/mbkin**4 - 
                 (28439*mckin**6)/mbkin**6 - (7792*mckin**8)/mbkin**8 + 
                 (213186*mckin**10)/mbkin**10 + (358324*mckin**12)/mbkin**12 + 
                 (223406*mckin**14)/mbkin**14 - (11322*mckin**16)/mbkin**16 - 
                 (24769*mckin**18)/mbkin**18 - (512*mckin**20)/mbkin**20 + 
                 (345*mckin**22)/mbkin**22)*q_cut)/mbkin**2 + 
              (4*(-35 + (210*mckin**2)/mbkin**2 + (1175*mckin**4)/mbkin**4 - 
                 (5000*mckin**6)/mbkin**6 - (16066*mckin**8)/mbkin**8 + 
                 (2586*mckin**10)/mbkin**10 + (5666*mckin**12)/mbkin**12 - 
                 (13056*mckin**14)/mbkin**14 - (6155*mckin**16)/mbkin**16 + 
                 (260*mckin**18)/mbkin**18 + (175*mckin**20)/mbkin**20)*q_cut**2)/
               mbkin**4 + (2*(-70 + (555*mckin**2)/mbkin**2 + (3708*mckin**4)/
                  mbkin**4 - (24082*mckin**6)/mbkin**6 - (91032*mckin**8)/
                  mbkin**8 - (119556*mckin**10)/mbkin**10 - (88492*mckin**12)/
                  mbkin**12 - (21102*mckin**14)/mbkin**14 + (1038*mckin**16)/
                  mbkin**16 + (345*mckin**18)/mbkin**18)*q_cut**3)/mbkin**6 - 
              (2*(-175 + (925*mckin**2)/mbkin**2 + (6934*mckin**4)/mbkin**4 - 
                 (21885*mckin**6)/mbkin**6 - (95358*mckin**8)/mbkin**8 - 
                 (90893*mckin**10)/mbkin**10 - (19260*mckin**12)/mbkin**12 + 
                 (4249*mckin**14)/mbkin**14 + (855*mckin**16)/mbkin**16)*q_cut**4)/
               mbkin**8 + (2*(-70 + (385*mckin**2)/mbkin**2 + (1676*mckin**4)/
                  mbkin**4 - (7143*mckin**6)/mbkin**6 - (19222*mckin**8)/
                  mbkin**8 - (6773*mckin**10)/mbkin**10 + (1456*mckin**12)/
                  mbkin**12 + (315*mckin**14)/mbkin**14)*q_cut**5)/mbkin**10 + 
              (2*(-70 + (105*mckin**2)/mbkin**2 + (4203*mckin**4)/mbkin**4 + 
                 (9906*mckin**6)/mbkin**6 + (10076*mckin**8)/mbkin**8 + 
                 (3633*mckin**10)/mbkin**10 + (315*mckin**12)/mbkin**12)*q_cut**6)/
               mbkin**12 - (10*(-14 + (23*mckin**2)/mbkin**2 + (736*mckin**4)/
                  mbkin**4 + (1352*mckin**6)/mbkin**6 + (650*mckin**8)/mbkin**8 + 
                 (45*mckin**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
              ((-35 + (50*mckin**2)/mbkin**2 + (1869*mckin**4)/mbkin**4 + 
                 (1664*mckin**6)/mbkin**6 + (60*mckin**8)/mbkin**8)*q_cut**8)/mbkin**
                16 - (4*mckin**2*(35 + (78*mckin**2)/mbkin**2 + (35*mckin**4)/
                  mbkin**4)*q_cut**9)/mbkin**20 + (150*mckin**2*(mbkin**2 + mckin**2)*
                q_cut**10)/mbkin**24 - (40*mckin**2*q_cut**11)/mbkin**24) - 
            4*(-((-1 + mckin**2/mbkin**2)**2*(-575 + (1630*mckin**2)/mbkin**2 + 
                 (85106*mckin**4)/mbkin**4 - (282442*mckin**6)/mbkin**6 - 
                 (1378372*mckin**8)/mbkin**8 + (1535906*mckin**10)/mbkin**10 + 
                 (4125290*mckin**12)/mbkin**12 + (1796594*mckin**14)/mbkin**14 - 
                 (249373*mckin**16)/mbkin**16 - (188768*mckin**18)/mbkin**18 - 
                 (3836*mckin**20)/mbkin**20 + (2040*mckin**22)/mbkin**22)) + 
              (4*(-585 + (285*mckin**2)/mbkin**2 + (64813*mckin**4)/mbkin**4 - 
                 (59927*mckin**6)/mbkin**6 - (921308*mckin**8)/mbkin**8 + 
                 (129898*mckin**10)/mbkin**10 + (2676538*mckin**12)/mbkin**12 + 
                 (3044602*mckin**14)/mbkin**14 + (904567*mckin**16)/mbkin**16 - 
                 (261887*mckin**18)/mbkin**18 - (134543*mckin**20)/mbkin**20 - 
                 (1323*mckin**22)/mbkin**22 + (2070*mckin**24)/mbkin**24)*q_cut)/
               mbkin**2 - (4*(-595 - (2310*mckin**2)/mbkin**2 + (33975*mckin**4)/
                  mbkin**4 + (77820*mckin**6)/mbkin**6 - (95082*mckin**8)/
                  mbkin**8 - (7416*mckin**10)/mbkin**10 + (117360*mckin**12)/
                  mbkin**12 - (247224*mckin**14)/mbkin**14 - (273243*mckin**16)/
                  mbkin**16 - (64490*mckin**18)/mbkin**18 + (5505*mckin**20)/
                  mbkin**20 + (2100*mckin**22)/mbkin**22)*q_cut**2)/mbkin**4 - 
              (4*(-585 - (2055*mckin**2)/mbkin**2 + (59033*mckin**4)/mbkin**4 + 
                 (170055*mckin**6)/mbkin**6 - (69365*mckin**8)/mbkin**8 - 
                 (373829*mckin**10)/mbkin**10 - (644871*mckin**12)/mbkin**12 - 
                 (469615*mckin**14)/mbkin**14 - (106890*mckin**16)/mbkin**16 + 
                 (4692*mckin**18)/mbkin**18 + (2070*mckin**20)/mbkin**20)*q_cut**3)/
               mbkin**6 + (2*(-2895 - (21000*mckin**2)/mbkin**2 + 
                 (119453*mckin**4)/mbkin**4 + (428252*mckin**6)/mbkin**6 - 
                 (312001*mckin**8)/mbkin**8 - (1414660*mckin**10)/mbkin**10 - 
                 (1024579*mckin**12)/mbkin**12 - (162412*mckin**14)/mbkin**14 + 
                 (50862*mckin**16)/mbkin**16 + (10260*mckin**18)/mbkin**18)*q_cut**4)/
               mbkin**8 - (4*(-525 - (4515*mckin**2)/mbkin**2 + (17451*mckin**4)/
                  mbkin**4 + (35671*mckin**6)/mbkin**6 - (71799*mckin**8)/
                  mbkin**8 - (98271*mckin**10)/mbkin**10 - (16201*mckin**12)/
                  mbkin**12 + (10059*mckin**14)/mbkin**14 + (1890*mckin**16)/
                  mbkin**16)*q_cut**5)/mbkin**10 - (4*(-525 - (7560*mckin**2)/
                  mbkin**2 - (15337*mckin**4)/mbkin**4 + (10568*mckin**6)/
                  mbkin**6 + (44285*mckin**8)/mbkin**8 + (50502*mckin**10)/
                  mbkin**10 + (20097*mckin**12)/mbkin**12 + (1890*mckin**14)/
                  mbkin**14)*q_cut**6)/mbkin**12 + (20*(-69 - (1443*mckin**2)/
                  mbkin**2 - (1933*mckin**4)/mbkin**4 + (4085*mckin**6)/mbkin**6 + 
                 (7964*mckin**8)/mbkin**8 + (3702*mckin**10)/mbkin**10 + 
                 (270*mckin**12)/mbkin**12)*q_cut**7)/mbkin**14 + 
              ((135 + (7260*mckin**2)/mbkin**2 + (2877*mckin**4)/mbkin**4 - 
                 (24560*mckin**6)/mbkin**6 - (18312*mckin**8)/mbkin**8 - 
                 (720*mckin**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
              (16*(-35 - (70*mckin**2)/mbkin**2 + (29*mckin**4)/mbkin**4 + 
                 (181*mckin**6)/mbkin**6 + (105*mckin**8)/mbkin**8)*q_cut**9)/mbkin**
                18 - (120*(-5 - (5*mckin**2)/mbkin**2 + (13*mckin**4)/mbkin**4 + 
                 (15*mckin**6)/mbkin**6)*q_cut**10)/mbkin**20 - 
              (160*(mbkin**4 - 3*mckin**4)*q_cut**11)/mbkin**26)*rG + 
            24*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(65 - (810*mckin**2)/
                  mbkin**2 - (516*mckin**4)/mbkin**4 + (110224*mckin**6)/
                  mbkin**6 - (410120*mckin**8)/mbkin**8 - (2939462*mckin**10)/
                  mbkin**10 - (3104774*mckin**12)/mbkin**12 - (475058*mckin**14)/
                  mbkin**14 + (163447*mckin**16)/mbkin**16 - (52384*mckin**18)/
                  mbkin**18 - (4742*mckin**20)/mbkin**20 + (850*mckin**22)/
                  mbkin**22)) + (2*(135 - (1420*mckin**2)/mbkin**2 - 
                 (3196*mckin**4)/mbkin**4 + (162656*mckin**6)/mbkin**6 - 
                 (138305*mckin**8)/mbkin**8 - (3423984*mckin**10)/mbkin**10 - 
                 (6202792*mckin**12)/mbkin**12 - (3867608*mckin**14)/mbkin**14 - 
                 (95475*mckin**16)/mbkin**16 + (226796*mckin**18)/mbkin**18 - 
                 (77740*mckin**20)/mbkin**20 - (7352*mckin**22)/mbkin**22 + 
                 (1725*mckin**24)/mbkin**24)*q_cut)/mbkin**2 - 
              (4*(70 - (455*mckin**2)/mbkin**2 - (3470*mckin**4)/mbkin**4 + 
                 (37665*mckin**6)/mbkin**6 + (138922*mckin**8)/mbkin**8 + 
                 (83452*mckin**10)/mbkin**10 + (121324*mckin**12)/mbkin**12 + 
                 (190868*mckin**14)/mbkin**14 + (16424*mckin**16)/mbkin**16 - 
                 (25445*mckin**18)/mbkin**18 - (790*mckin**20)/mbkin**20 + 
                 (875*mckin**22)/mbkin**22)*q_cut**2)/mbkin**4 - 
              (2*(135 - (880*mckin**2)/mbkin**2 - (7621*mckin**4)/mbkin**4 + 
                 (133182*mckin**6)/mbkin**6 + (543892*mckin**8)/mbkin**8 + 
                 (658370*mckin**10)/mbkin**10 + (525528*mckin**12)/mbkin**12 + 
                 (55122*mckin**14)/mbkin**14 - (76843*mckin**16)/mbkin**16 - 
                 (2082*mckin**18)/mbkin**18 + (1725*mckin**20)/mbkin**20)*q_cut**3)/
               mbkin**6 + (2*(330 - (1015*mckin**2)/mbkin**2 - (21973*mckin**4)/
                  mbkin**4 + (116514*mckin**6)/mbkin**6 + (621711*mckin**8)/
                  mbkin**8 + (591448*mckin**10)/mbkin**10 - (14853*mckin**12)/
                  mbkin**12 - (96558*mckin**14)/mbkin**14 + (6769*mckin**16)/
                  mbkin**16 + (4275*mckin**18)/mbkin**18)*q_cut**4)/mbkin**8 - 
              (2*(105 - (280*mckin**2)/mbkin**2 - (8822*mckin**4)/mbkin**4 + 
                 (25052*mckin**6)/mbkin**6 + (79356*mckin**8)/mbkin**8 - 
                 (7796*mckin**10)/mbkin**10 - (25990*mckin**12)/mbkin**12 + 
                 (3136*mckin**14)/mbkin**14 + (1575*mckin**16)/mbkin**16)*q_cut**5)/
               mbkin**10 - (2*(105 + (455*mckin**2)/mbkin**2 - (8521*mckin**4)/
                  mbkin**4 - (25699*mckin**6)/mbkin**6 - (20961*mckin**8)/
                  mbkin**8 + (4285*mckin**10)/mbkin**10 + (11193*mckin**12)/
                  mbkin**12 + (1575*mckin**14)/mbkin**14)*q_cut**6)/mbkin**12 + 
              (10*(3 + (88*mckin**2)/mbkin**2 - (1667*mckin**4)/mbkin**4 - 
                 (3298*mckin**6)/mbkin**6 + (671*mckin**8)/mbkin**8 + 
                 (2090*mckin**10)/mbkin**10 + (225*mckin**12)/mbkin**12)*q_cut**7)/
               mbkin**14 + ((45 - (310*mckin**2)/mbkin**2 + (4303*mckin**4)/
                  mbkin**4 + (694*mckin**6)/mbkin**6 - (5204*mckin**8)/mbkin**8 - 
                 (300*mckin**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
              (4*(35 + (70*mckin**2)/mbkin**2 + (94*mckin**4)/mbkin**4 + 
                 (218*mckin**6)/mbkin**6 + (175*mckin**8)/mbkin**8)*q_cut**9)/mbkin**
                18 - (30*(5 + (5*mckin**2)/mbkin**2 + (17*mckin**4)/mbkin**4 + 
                 (25*mckin**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
              (40*(mbkin**4 + 5*mckin**4)*q_cut**11)/mbkin**26)*rhoD + 1260*sB + 
            (9240*mckin**2*sB)/mbkin**2 - (444348*mckin**4*sB)/mbkin**4 + 
            (379944*mckin**6*sB)/mbkin**6 + (15149640*mckin**8*sB)/mbkin**8 + 
            (2327616*mckin**10*sB)/mbkin**10 - (26470152*mckin**12*sB)/
             mbkin**12 - (6409872*mckin**14*sB)/mbkin**14 + 
            (7311276*mckin**16*sB)/mbkin**16 + (7117560*mckin**18*sB)/mbkin**18 + 
            (2594244*mckin**20*sB)/mbkin**20 - (1478808*mckin**22*sB)/mbkin**22 - 
            (108000*mckin**24*sB)/mbkin**24 + (20400*mckin**26*sB)/mbkin**26 - 
            (5040*q_cut*sB)/mbkin**2 - (62160*mckin**2*q_cut*sB)/mbkin**4 + 
            (1228416*mckin**4*q_cut*sB)/mbkin**6 + (4170192*mckin**6*q_cut*sB)/
             mbkin**8 - (27720144*mckin**8*q_cut*sB)/mbkin**10 - 
            (98929248*mckin**10*q_cut*sB)/mbkin**12 - (111450432*mckin**12*q_cut*sB)/
             mbkin**14 - (55878048*mckin**14*q_cut*sB)/mbkin**16 - 
            (16085904*mckin**16*q_cut*sB)/mbkin**18 + (3898032*mckin**18*q_cut*sB)/
             mbkin**20 + (4635456*mckin**20*q_cut*sB)/mbkin**22 + 
            (171600*mckin**22*q_cut*sB)/mbkin**24 - (82800*mckin**24*q_cut*sB)/
             mbkin**26 + (5040*q_cut**2*sB)/mbkin**4 + (87360*mckin**2*q_cut**2*sB)/
             mbkin**6 - (505200*mckin**4*q_cut**2*sB)/mbkin**8 - 
            (4070880*mckin**6*q_cut**2*sB)/mbkin**10 - (4525536*mckin**8*q_cut**2*sB)/
             mbkin**12 + (142656*mckin**10*q_cut**2*sB)/mbkin**14 - 
            (2771904*mckin**12*q_cut**2*sB)/mbkin**16 - (4946496*mckin**14*q_cut**2*sB)/
             mbkin**18 - (5925840*mckin**16*q_cut**2*sB)/mbkin**20 - 
            (2336640*mckin**18*q_cut**2*sB)/mbkin**22 + (87600*mckin**20*q_cut**2*sB)/
             mbkin**24 + (84000*mckin**22*q_cut**2*sB)/mbkin**26 + 
            (5040*q_cut**3*sB)/mbkin**6 + (82320*mckin**2*q_cut**3*sB)/mbkin**8 - 
            (854496*mckin**4*q_cut**3*sB)/mbkin**10 - (7703856*mckin**6*q_cut**3*sB)/
             mbkin**12 - (16016976*mckin**8*q_cut**3*sB)/mbkin**14 - 
            (21826608*mckin**10*q_cut**3*sB)/mbkin**16 - (21509136*mckin**12*q_cut**3*
              sB)/mbkin**18 - (13371216*mckin**14*q_cut**3*sB)/mbkin**20 - 
            (3961056*mckin**16*q_cut**3*sB)/mbkin**22 + (110880*mckin**18*q_cut**3*sB)/
             mbkin**24 + (82800*mckin**20*q_cut**3*sB)/mbkin**26 - 
            (12600*q_cut**4*sB)/mbkin**8 - (253680*mckin**2*q_cut**4*sB)/mbkin**10 + 
            (404088*mckin**4*q_cut**4*sB)/mbkin**12 + (8097600*mckin**6*q_cut**4*sB)/
             mbkin**14 + (19759944*mckin**8*q_cut**4*sB)/mbkin**16 + 
            (21465504*mckin**10*q_cut**4*sB)/mbkin**18 + (14271960*mckin**12*q_cut**4*
              sB)/mbkin**20 + (3432288*mckin**14*q_cut**4*sB)/mbkin**22 - 
            (769440*mckin**16*q_cut**4*sB)/mbkin**24 - (205200*mckin**18*q_cut**4*sB)/
             mbkin**26 + (5040*q_cut**5*sB)/mbkin**10 + (102480*mckin**2*q_cut**5*sB)/
             mbkin**12 - (16512*mckin**4*q_cut**5*sB)/mbkin**14 - 
            (1479984*mckin**6*q_cut**5*sB)/mbkin**16 - (2775456*mckin**8*q_cut**5*sB)/
             mbkin**18 - (2796144*mckin**10*q_cut**5*sB)/mbkin**20 - 
            (875232*mckin**12*q_cut**5*sB)/mbkin**22 + (290640*mckin**14*q_cut**5*sB)/
             mbkin**24 + (75600*mckin**16*q_cut**5*sB)/mbkin**26 + 
            (5040*q_cut**6*sB)/mbkin**12 + (127680*mckin**2*q_cut**6*sB)/mbkin**14 + 
            (597024*mckin**4*q_cut**6*sB)/mbkin**16 + (1262448*mckin**6*q_cut**6*sB)/
             mbkin**18 + (1846128*mckin**8*q_cut**6*sB)/mbkin**20 + 
            (1700544*mckin**10*q_cut**6*sB)/mbkin**22 + (725760*mckin**12*q_cut**6*sB)/
             mbkin**24 + (75600*mckin**14*q_cut**6*sB)/mbkin**26 - 
            (5040*q_cut**7*sB)/mbkin**14 - (122640*mckin**2*q_cut**7*sB)/mbkin**16 - 
            (494400*mckin**4*q_cut**7*sB)/mbkin**18 - (1044720*mckin**6*q_cut**7*sB)/
             mbkin**20 - (1306080*mckin**8*q_cut**7*sB)/mbkin**22 - 
            (661440*mckin**10*q_cut**7*sB)/mbkin**24 - (54000*mckin**12*q_cut**7*sB)/
             mbkin**26 + (1260*q_cut**8*sB)/mbkin**16 + (29400*mckin**2*q_cut**8*sB)/
             mbkin**18 + (84036*mckin**4*q_cut**8*sB)/mbkin**20 + 
            (198456*mckin**6*q_cut**8*sB)/mbkin**22 + (166080*mckin**8*q_cut**8*sB)/
             mbkin**24 + (7200*mckin**10*q_cut**8*sB)/mbkin**26 - 
            (8928*mckin**4*q_cut**9*sB)/mbkin**22 - (28800*mckin**6*q_cut**9*sB)/
             mbkin**24 - (16800*mckin**8*q_cut**9*sB)/mbkin**26 + 
            (15120*mckin**4*q_cut**10*sB)/mbkin**24 + (18000*mckin**6*q_cut**10*sB)/
             mbkin**26 - (4800*mckin**4*q_cut**11*sB)/mbkin**26 - 1360*sE - 
            (2000*mckin**2*sE)/mbkin**2 + (309680*mckin**4*sE)/mbkin**4 - 
            (670736*mckin**6*sE)/mbkin**6 - (7412592*mckin**8*sE)/mbkin**8 + 
            (6194640*mckin**10*sE)/mbkin**10 + (8632992*mckin**12*sE)/mbkin**12 - 
            (5378784*mckin**14*sE)/mbkin**14 - (700464*mckin**16*sE)/mbkin**16 - 
            (312368*mckin**18*sE)/mbkin**18 - (1548880*mckin**20*sE)/mbkin**20 + 
            (836848*mckin**22*sE)/mbkin**22 + (65264*mckin**24*sE)/mbkin**24 - 
            (12240*mckin**26*sE)/mbkin**26 + (5520*q_cut*sE)/mbkin**2 + 
            (32160*mckin**2*q_cut*sE)/mbkin**4 - (906080*mckin**4*q_cut*sE)/mbkin**6 - 
            (1643168*mckin**6*q_cut*sE)/mbkin**8 + (17160304*mckin**8*q_cut*sE)/
             mbkin**10 + (36106432*mckin**10*q_cut*sE)/mbkin**12 + 
            (22402624*mckin**12*q_cut*sE)/mbkin**14 + (4453696*mckin**14*q_cut*sE)/
             mbkin**16 + (2931184*mckin**16*q_cut*sE)/mbkin**18 - 
            (2327648*mckin**18*q_cut*sE)/mbkin**20 - (2691296*mckin**20*q_cut*sE)/
             mbkin**22 - (94368*mckin**22*q_cut*sE)/mbkin**24 + 
            (49680*mckin**24*q_cut*sE)/mbkin**26 - (5600*q_cut**2*sE)/mbkin**4 - 
            (57120*mckin**2*q_cut**2*sE)/mbkin**6 + (427680*mckin**4*q_cut**2*sE)/
             mbkin**8 + (2262240*mckin**6*q_cut**2*sE)/mbkin**10 + 
            (581760*mckin**8*q_cut**2*sE)/mbkin**12 - (2877312*mckin**10*q_cut**2*sE)/
             mbkin**14 + (1913472*mckin**12*q_cut**2*sE)/mbkin**16 + 
            (6809472*mckin**14*q_cut**2*sE)/mbkin**18 + (5638368*mckin**16*q_cut**2*sE)/
             mbkin**20 + (1425440*mckin**18*q_cut**2*sE)/mbkin**22 - 
            (101280*mckin**20*q_cut**2*sE)/mbkin**24 - (50400*mckin**22*q_cut**2*sE)/
             mbkin**26 - (5520*q_cut**3*sE)/mbkin**6 - (54240*mckin**2*q_cut**3*sE)/
             mbkin**8 + (691120*mckin**4*q_cut**3*sE)/mbkin**10 + 
            (4345728*mckin**6*q_cut**3*sE)/mbkin**12 + (5522528*mckin**8*q_cut**3*sE)/
             mbkin**14 + (4002368*mckin**10*q_cut**3*sE)/mbkin**16 + 
            (2024544*mckin**12*q_cut**3*sE)/mbkin**18 + (3545728*mckin**14*q_cut**3*sE)/
             mbkin**20 + (2277168*mckin**16*q_cut**3*sE)/mbkin**22 + 
            (21408*mckin**18*q_cut**3*sE)/mbkin**24 - (49680*mckin**20*q_cut**3*sE)/
             mbkin**26 + (13680*q_cut**4*sE)/mbkin**8 + (186480*mckin**2*q_cut**4*sE)/
             mbkin**10 - (511520*mckin**4*q_cut**4*sE)/mbkin**12 - 
            (4919264*mckin**6*q_cut**4*sE)/mbkin**14 - (7467680*mckin**8*q_cut**4*sE)/
             mbkin**16 - (6060512*mckin**10*q_cut**4*sE)/mbkin**18 - 
            (5694752*mckin**12*q_cut**4*sE)/mbkin**20 - (1869920*mckin**14*q_cut**4*sE)/
             mbkin**22 + (413616*mckin**16*q_cut**4*sE)/mbkin**24 + 
            (123120*mckin**18*q_cut**4*sE)/mbkin**26 - (5040*q_cut**5*sE)/mbkin**10 - 
            (77280*mckin**2*q_cut**5*sE)/mbkin**12 + (108480*mckin**4*q_cut**5*sE)/
             mbkin**14 + (874016*mckin**6*q_cut**5*sE)/mbkin**16 + 
            (894432*mckin**8*q_cut**5*sE)/mbkin**18 + (642528*mckin**10*q_cut**5*sE)/
             mbkin**20 + (23872*mckin**12*q_cut**5*sE)/mbkin**22 - 
            (213024*mckin**14*q_cut**5*sE)/mbkin**24 - (45360*mckin**16*q_cut**5*sE)/
             mbkin**26 - (5040*q_cut**6*sE)/mbkin**12 - (105840*mckin**2*q_cut**6*sE)/
             mbkin**14 - (365840*mckin**4*q_cut**6*sE)/mbkin**16 - 
            (419792*mckin**6*q_cut**6*sE)/mbkin**18 - (359504*mckin**8*q_cut**6*sE)/
             mbkin**20 - (464784*mckin**10*q_cut**6*sE)/mbkin**22 - 
            (371952*mckin**12*q_cut**6*sE)/mbkin**24 - (45360*mckin**14*q_cut**6*sE)/
             mbkin**26 + (3600*q_cut**7*sE)/mbkin**14 + (102240*mckin**2*q_cut**7*sE)/
             mbkin**16 + (292720*mckin**4*q_cut**7*sE)/mbkin**18 + 
            (389440*mckin**6*q_cut**7*sE)/mbkin**20 + (544240*mckin**8*q_cut**7*sE)/
             mbkin**22 + (363360*mckin**10*q_cut**7*sE)/mbkin**24 + 
            (32400*mckin**12*q_cut**7*sE)/mbkin**26 - (480*q_cut**8*sE)/mbkin**16 - 
            (25440*mckin**2*q_cut**8*sE)/mbkin**18 - (44400*mckin**4*q_cut**8*sE)/
             mbkin**20 - (82928*mckin**6*q_cut**8*sE)/mbkin**22 - 
            (85728*mckin**8*q_cut**8*sE)/mbkin**24 - (4320*mckin**10*q_cut**8*sE)/
             mbkin**26 + (1120*q_cut**9*sE)/mbkin**18 + (2240*mckin**2*q_cut**9*sE)/
             mbkin**20 + (2240*mckin**4*q_cut**9*sE)/mbkin**22 + 
            (9664*mckin**6*q_cut**9*sE)/mbkin**24 + (10080*mckin**8*q_cut**9*sE)/
             mbkin**26 - (1200*q_cut**10*sE)/mbkin**20 - (1200*mckin**2*q_cut**10*sE)/
             mbkin**22 - (6960*mckin**4*q_cut**10*sE)/mbkin**24 - 
            (10800*mckin**6*q_cut**10*sE)/mbkin**26 + (320*q_cut**11*sE)/mbkin**22 + 
            (2880*mckin**4*q_cut**11*sE)/mbkin**26 - 145*sqB - (800*mckin**2*sqB)/
             mbkin**2 + (54167*mckin**4*sqB)/mbkin**4 - (39482*mckin**6*sqB)/
             mbkin**6 - (2079312*mckin**8*sqB)/mbkin**8 - (1917294*mckin**10*sqB)/
             mbkin**10 + (4531890*mckin**12*sqB)/mbkin**12 + 
            (3302724*mckin**14*sqB)/mbkin**14 - (2155053*mckin**16*sqB)/
             mbkin**16 - (1741748*mckin**18*sqB)/mbkin**18 - 
            (24841*mckin**20*sqB)/mbkin**20 + (69430*mckin**22*sqB)/mbkin**22 + 
            (974*mckin**24*sqB)/mbkin**24 - (510*mckin**26*sqB)/mbkin**26 + 
            (570*q_cut*sqB)/mbkin**2 + (6480*mckin**2*q_cut*sqB)/mbkin**4 - 
            (148544*mckin**4*q_cut*sqB)/mbkin**6 - (579896*mckin**6*q_cut*sqB)/
             mbkin**8 + (3426370*mckin**8*q_cut*sqB)/mbkin**10 + 
            (16467184*mckin**10*q_cut*sqB)/mbkin**12 + (23592832*mckin**12*q_cut*sqB)/
             mbkin**14 + (14304928*mckin**14*q_cut*sqB)/mbkin**16 + 
            (2600590*mckin**16*q_cut*sqB)/mbkin**18 - (688736*mckin**18*q_cut*sqB)/
             mbkin**20 - (201440*mckin**20*q_cut*sqB)/mbkin**22 + 
            (4152*mckin**22*q_cut*sqB)/mbkin**24 + (2070*mckin**24*q_cut*sqB)/
             mbkin**26 - (560*q_cut**2*sqB)/mbkin**4 - (9660*mckin**2*q_cut**2*sqB)/
             mbkin**6 + (55200*mckin**4*q_cut**2*sqB)/mbkin**8 + 
            (549540*mckin**6*q_cut**2*sqB)/mbkin**10 + (928824*mckin**8*q_cut**2*sqB)/
             mbkin**12 + (375504*mckin**10*q_cut**2*sqB)/mbkin**14 + 
            (785808*mckin**12*q_cut**2*sqB)/mbkin**16 + (1426416*mckin**14*q_cut**2*sqB)/
             mbkin**18 + (718968*mckin**16*q_cut**2*sqB)/mbkin**20 + 
            (84620*mckin**18*q_cut**2*sqB)/mbkin**22 - (13680*mckin**20*q_cut**2*sqB)/
             mbkin**24 - (2100*mckin**22*q_cut**2*sqB)/mbkin**26 - 
            (570*q_cut**3*sqB)/mbkin**6 - (8760*mckin**2*q_cut**3*sqB)/mbkin**8 + 
            (108574*mckin**4*q_cut**3*sqB)/mbkin**10 + (1034772*mckin**6*q_cut**3*sqB)/
             mbkin**12 + (2407232*mckin**8*q_cut**3*sqB)/mbkin**14 + 
            (2729180*mckin**10*q_cut**3*sqB)/mbkin**16 + (2234328*mckin**12*q_cut**3*
              sqB)/mbkin**18 + (1038412*mckin**14*q_cut**3*sqB)/mbkin**20 + 
            (159642*mckin**16*q_cut**3*sqB)/mbkin**22 - (7812*mckin**18*q_cut**3*sqB)/
             mbkin**24 - (2070*mckin**20*q_cut**3*sqB)/mbkin**26 + 
            (1440*q_cut**4*sqB)/mbkin**8 + (27510*mckin**2*q_cut**4*sqB)/mbkin**10 - 
            (43082*mckin**4*q_cut**4*sqB)/mbkin**12 - (1078244*mckin**6*q_cut**4*sqB)/
             mbkin**14 - (3002066*mckin**8*q_cut**4*sqB)/mbkin**16 - 
            (2994488*mckin**10*q_cut**4*sqB)/mbkin**18 - (1173962*mckin**12*q_cut**4*
              sqB)/mbkin**20 - (79652*mckin**14*q_cut**4*sqB)/mbkin**22 + 
            (42966*mckin**16*q_cut**4*sqB)/mbkin**24 + (5130*mckin**18*q_cut**4*sqB)/
             mbkin**26 - (630*q_cut**5*sqB)/mbkin**10 - (10920*mckin**2*q_cut**5*sqB)/
             mbkin**12 - (5652*mckin**4*q_cut**5*sqB)/mbkin**14 + 
            (184832*mckin**6*q_cut**5*sqB)/mbkin**16 + (365376*mckin**8*q_cut**5*sqB)/
             mbkin**18 + (177024*mckin**10*q_cut**5*sqB)/mbkin**20 - 
            (14900*mckin**12*q_cut**5*sqB)/mbkin**22 - (20664*mckin**14*q_cut**5*sqB)/
             mbkin**24 - (1890*mckin**16*q_cut**5*sqB)/mbkin**26 - 
            (630*q_cut**6*sqB)/mbkin**12 - (13650*mckin**2*q_cut**6*sqB)/mbkin**14 - 
            (70346*mckin**4*q_cut**6*sqB)/mbkin**16 - (131414*mckin**6*q_cut**6*sqB)/
             mbkin**18 - (124586*mckin**8*q_cut**6*sqB)/mbkin**20 - 
            (86790*mckin**10*q_cut**6*sqB)/mbkin**22 - (26502*mckin**12*q_cut**6*sqB)/
             mbkin**24 - (1890*mckin**14*q_cut**6*sqB)/mbkin**26 + 
            (810*q_cut**7*sqB)/mbkin**14 + (12840*mckin**2*q_cut**7*sqB)/mbkin**16 + 
            (61750*mckin**4*q_cut**7*sqB)/mbkin**18 + (109420*mckin**6*q_cut**7*sqB)/
             mbkin**20 + (85210*mckin**8*q_cut**7*sqB)/mbkin**22 + 
            (27180*mckin**10*q_cut**7*sqB)/mbkin**24 + (1350*mckin**12*q_cut**7*sqB)/
             mbkin**26 - (255*q_cut**8*sqB)/mbkin**16 - (2910*mckin**2*q_cut**8*sqB)/
             mbkin**18 - (11409*mckin**4*q_cut**8*sqB)/mbkin**20 - 
            (15662*mckin**6*q_cut**8*sqB)/mbkin**22 - (6588*mckin**8*q_cut**8*sqB)/
             mbkin**24 - (180*mckin**10*q_cut**8*sqB)/mbkin**26 - 
            (140*q_cut**9*sqB)/mbkin**18 - (280*mckin**2*q_cut**9*sqB)/mbkin**20 - 
            (568*mckin**4*q_cut**9*sqB)/mbkin**22 + (184*mckin**6*q_cut**9*sqB)/
             mbkin**24 + (420*mckin**8*q_cut**9*sqB)/mbkin**26 + (150*q_cut**10*sqB)/
             mbkin**20 + (150*mckin**2*q_cut**10*sqB)/mbkin**22 - 
            (210*mckin**4*q_cut**10*sqB)/mbkin**24 - (450*mckin**6*q_cut**10*sqB)/
             mbkin**26 - (40*q_cut**11*sqB)/mbkin**22 + (120*mckin**4*q_cut**11*sqB)/
             mbkin**26))*np.log((mbkin**2 + mckin**2 - q_cut - 
            mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/
           (mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                 mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
               mbkin**4))) - (144*mckin**4*
          (-16*(-((-1 + mckin**2/mbkin**2)**4*(-590 - (170*mckin**2)/mbkin**2 + 
                (47529*mckin**4)/mbkin**4 + (10177*mckin**6)/mbkin**6 - 
                (120338*mckin**8)/mbkin**8 - (39978*mckin**10)/mbkin**10 - 
                (3053*mckin**12)/mbkin**12 + (12643*mckin**14)/mbkin**14 + 
                (3060*mckin**16)/mbkin**16)) + ((-1 + mckin**2/mbkin**2)**2*(
                -2960 - (5850*mckin**2)/mbkin**2 + (190377*mckin**4)/mbkin**4 + 
                (219388*mckin**6)/mbkin**6 - (442203*mckin**8)/mbkin**8 - 
                (597078*mckin**10)/mbkin**10 - (182993*mckin**12)/mbkin**12 + 
                (13020*mckin**14)/mbkin**14 + (67059*mckin**16)/mbkin**16 + 
                (15480*mckin**18)/mbkin**18)*q_cut)/mbkin**2 - 
             (3*(-1780 - (4930*mckin**2)/mbkin**2 + (89849*mckin**4)/mbkin**4 + 
                (83367*mckin**6)/mbkin**6 - (134199*mckin**8)/mbkin**8 - 
                (202701*mckin**10)/mbkin**10 - (265421*mckin**12)/mbkin**12 - 
                (100835*mckin**14)/mbkin**14 + (4207*mckin**16)/mbkin**16 + 
                (39243*mckin**18)/mbkin**18 + (9360*mckin**20)/mbkin**20)*q_cut**2)/
              mbkin**4 + ((-2970 - (11480*mckin**2)/mbkin**2 + (121169*mckin**4)/
                 mbkin**4 + (162838*mckin**6)/mbkin**6 - (396629*mckin**8)/
                 mbkin**8 - (498572*mckin**10)/mbkin**10 - (54133*mckin**12)/
                 mbkin**12 + (158794*mckin**14)/mbkin**14 + (97083*mckin**16)/
                 mbkin**16 + (15660*mckin**18)/mbkin**18)*q_cut**3)/mbkin**6 + 
             ((-2940 - (26390*mckin**2)/mbkin**2 - (12875*mckin**4)/mbkin**4 + 
                (74925*mckin**6)/mbkin**6 + (184456*mckin**8)/mbkin**8 + 
                (125662*mckin**10)/mbkin**10 + (82895*mckin**12)/mbkin**12 + 
                (75915*mckin**14)/mbkin**14 + (15120*mckin**16)/mbkin**16)*q_cut**4)/
              mbkin**8 - (3*(-1750 - (14980*mckin**2)/mbkin**2 - (1663*mckin**4)/
                 mbkin**4 + (43998*mckin**6)/mbkin**6 + (91212*mckin**8)/
                 mbkin**8 + (97050*mckin**10)/mbkin**10 + (54369*mckin**12)/
                 mbkin**12 + (8820*mckin**14)/mbkin**14)*q_cut**5)/mbkin**10 + 
             ((-2940 - (23870*mckin**2)/mbkin**2 + (9833*mckin**4)/mbkin**4 + 
                (105581*mckin**6)/mbkin**6 + (151243*mckin**8)/mbkin**8 + 
                (84441*mckin**10)/mbkin**10 + (15120*mckin**12)/mbkin**12)*q_cut**6)/
              mbkin**12 - ((-810 - (4360*mckin**2)/mbkin**2 + (9023*mckin**4)/
                 mbkin**4 + (21544*mckin**6)/mbkin**6 + (14031*mckin**8)/
                 mbkin**8 + (7020*mckin**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
             (6*(-55 - (70*mckin**2)/mbkin**2 + (431*mckin**4)/mbkin**4 + 
                (672*mckin**6)/mbkin**6 + (990*mckin**8)/mbkin**8)*q_cut**8)/
              mbkin**16 - (10*(-19 - (19*mckin**2)/mbkin**2 + (230*mckin**4)/
                 mbkin**4 + (342*mckin**6)/mbkin**6)*q_cut**9)/mbkin**18 - 
             (40*(mbkin**4 - 18*mckin**4)*q_cut**10)/mbkin**24)*rE + 
           ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
             q_cut**2/mbkin**4)*((-2520*mckin**2*muG**2)/mbkin**2 + 
             (7740*mckin**4*muG**2)/mbkin**4 - (25884*mckin**6*muG**2)/mbkin**6 + 
             (165564*mckin**8*muG**2)/mbkin**8 + (515340*mckin**10*muG**2)/
              mbkin**10 - (1157940*mckin**12*muG**2)/mbkin**12 + 
             (117180*mckin**14*muG**2)/mbkin**14 + (465444*mckin**16*muG**2)/
              mbkin**16 - (78804*mckin**18*muG**2)/mbkin**18 - 
             (6120*mckin**20*muG**2)/mbkin**20 - (2520*mckin**2*muG*mupi)/
              mbkin**2 + (17460*mckin**4*muG*mupi)/mbkin**4 + 
             (51084*mckin**6*muG*mupi)/mbkin**6 - (442764*mckin**8*muG*mupi)/
              mbkin**8 - (36540*mckin**10*muG*mupi)/mbkin**10 + 
             (910980*mckin**12*muG*mupi)/mbkin**12 - (243180*mckin**14*muG*mupi)/
              mbkin**14 - (289044*mckin**16*muG*mupi)/mbkin**16 + 
             (28404*mckin**18*muG*mupi)/mbkin**18 + (6120*mckin**20*muG*mupi)/
              mbkin**20 - (2520*mckin**2*muG**2*q_cut)/mbkin**4 + 
             (30060*mckin**4*muG**2*q_cut)/mbkin**6 + (108504*mckin**6*muG**2*q_cut)/
              mbkin**8 - (486972*mckin**8*muG**2*q_cut)/mbkin**10 - 
             (1597032*mckin**10*muG**2*q_cut)/mbkin**12 - (2135052*mckin**12*muG**
                2*q_cut)/mbkin**14 - (563832*mckin**14*muG**2*q_cut)/mbkin**16 + 
             (273564*mckin**16*muG**2*q_cut)/mbkin**18 + (18720*mckin**18*muG**2*q_cut)/
              mbkin**20 + (7560*mckin**2*muG*mupi*q_cut)/mbkin**4 - 
             (30060*mckin**4*muG*mupi*q_cut)/mbkin**6 - (214344*mckin**6*muG*mupi*
               q_cut)/mbkin**8 + (476892*mckin**8*muG*mupi*q_cut)/mbkin**10 + 
             (2035512*mckin**10*muG*mupi*q_cut)/mbkin**12 + 
             (1832652*mckin**12*muG*mupi*q_cut)/mbkin**14 + 
             (387432*mckin**14*muG*mupi*q_cut)/mbkin**16 - 
             (122364*mckin**16*muG*mupi*q_cut)/mbkin**18 - 
             (18720*mckin**18*muG*mupi*q_cut)/mbkin**20 + (5040*mckin**2*muG**2*q_cut**
                2)/mbkin**6 - (131400*mckin**6*muG**2*q_cut**2)/mbkin**10 - 
             (42912*mckin**8*muG**2*q_cut**2)/mbkin**12 + (603144*mckin**10*muG**2*q_cut**
                2)/mbkin**14 + (130968*mckin**12*muG**2*q_cut**2)/mbkin**16 - 
             (189360*mckin**14*muG**2*q_cut**2)/mbkin**18 - (12600*mckin**16*muG**2*q_cut**
                2)/mbkin**20 - (5040*mckin**2*muG*mupi*q_cut**2)/mbkin**6 + 
             (10080*mckin**4*muG*mupi*q_cut**2)/mbkin**8 + (111240*mckin**6*muG*mupi*
               q_cut**2)/mbkin**10 - (98208*mckin**8*muG*mupi*q_cut**2)/mbkin**12 - 
             (401544*mckin**10*muG*mupi*q_cut**2)/mbkin**14 - 
             (80568*mckin**12*muG*mupi*q_cut**2)/mbkin**16 + 
             (88560*mckin**14*muG*mupi*q_cut**2)/mbkin**18 + 
             (12600*mckin**16*muG*mupi*q_cut**2)/mbkin**20 + 
             (15120*mckin**2*muG**2*q_cut**3)/mbkin**8 + (12600*mckin**4*muG**2*q_cut**3)/
              mbkin**10 - (302760*mckin**6*muG**2*q_cut**3)/mbkin**12 - 
             (719352*mckin**8*muG**2*q_cut**3)/mbkin**14 - (676512*mckin**10*muG**2*q_cut**
                3)/mbkin**16 - (239760*mckin**12*muG**2*q_cut**3)/mbkin**18 - 
             (12600*mckin**14*muG**2*q_cut**3)/mbkin**20 - (5040*mckin**2*muG*mupi*q_cut**
                3)/mbkin**8 + (7560*mckin**4*muG*mupi*q_cut**3)/mbkin**10 + 
             (161640*mckin**6*muG*mupi*q_cut**3)/mbkin**12 + 
             (316152*mckin**8*muG*mupi*q_cut**3)/mbkin**14 + 
             (323712*mckin**10*muG*mupi*q_cut**3)/mbkin**16 + 
             (138960*mckin**12*muG*mupi*q_cut**3)/mbkin**18 + 
             (12600*mckin**14*muG*mupi*q_cut**3)/mbkin**20 - 
             (22680*mckin**2*muG**2*q_cut**4)/mbkin**10 - (3780*mckin**4*muG**2*q_cut**4)/
              mbkin**12 + (454500*mckin**6*muG**2*q_cut**4)/mbkin**14 + 
             (754668*mckin**8*muG**2*q_cut**4)/mbkin**16 + (343620*mckin**10*muG**2*q_cut**
                4)/mbkin**18 + (17640*mckin**12*muG**2*q_cut**4)/mbkin**20 + 
             (7560*mckin**2*muG*mupi*q_cut**4)/mbkin**10 - (11340*mckin**4*muG*mupi*
               q_cut**4)/mbkin**12 - (227700*mckin**6*muG*mupi*q_cut**4)/mbkin**14 - 
             (376668*mckin**8*muG*mupi*q_cut**4)/mbkin**16 - 
             (192420*mckin**10*muG*mupi*q_cut**4)/mbkin**18 - 
             (17640*mckin**12*muG*mupi*q_cut**4)/mbkin**20 + 
             (7560*mckin**2*muG**2*q_cut**5)/mbkin**12 - (6300*mckin**4*muG**2*q_cut**5)/
              mbkin**14 - (143496*mckin**6*muG**2*q_cut**5)/mbkin**16 - 
             (109476*mckin**8*muG**2*q_cut**5)/mbkin**18 - (5040*mckin**10*muG**2*q_cut**
                5)/mbkin**20 - (2520*mckin**2*muG*mupi*q_cut**5)/mbkin**12 + 
             (6300*mckin**4*muG*mupi*q_cut**5)/mbkin**14 + (67896*mckin**6*muG*mupi*
               q_cut**5)/mbkin**16 + (59076*mckin**8*muG*mupi*q_cut**5)/mbkin**18 + 
             (5040*mckin**10*muG*mupi*q_cut**5)/mbkin**20 + 
             (2520*mckin**4*muG**2*q_cut**6)/mbkin**16 + (4176*mckin**6*muG**2*q_cut**6)/
              mbkin**18 + (2520*mckin**8*muG**2*q_cut**6)/mbkin**20 - 
             (2520*mckin**4*muG*mupi*q_cut**6)/mbkin**16 - (4176*mckin**6*muG*mupi*
               q_cut**6)/mbkin**18 - (2520*mckin**8*muG*mupi*q_cut**6)/mbkin**20 - 
             (3960*mckin**4*muG**2*q_cut**7)/mbkin**18 - (3960*mckin**6*muG**2*q_cut**7)/
              mbkin**20 + (3960*mckin**4*muG*mupi*q_cut**7)/mbkin**18 + 
             (3960*mckin**6*muG*mupi*q_cut**7)/mbkin**20 + (1440*mckin**4*muG**2*q_cut**
                8)/mbkin**20 - (1440*mckin**4*muG*mupi*q_cut**8)/mbkin**20 + 
             72*mckin**2*muG*((-1 + mckin**2/mbkin**2)**2*(-70 + (345*mckin**2)/
                  mbkin**2 + (2179*mckin**4)/mbkin**4 - (8286*mckin**6)/mbkin**6 - 
                 (19766*mckin**8)/mbkin**8 - (5941*mckin**10)/mbkin**10 + 
                 (1129*mckin**12)/mbkin**12 + (170*mckin**14)/mbkin**14) + 
               ((210 - (835*mckin**2)/mbkin**2 - (5954*mckin**4)/mbkin**4 + 
                  (13247*mckin**6)/mbkin**6 + (56542*mckin**8)/mbkin**8 + 
                  (50907*mckin**10)/mbkin**10 + (10762*mckin**12)/mbkin**12 - 
                  (3399*mckin**14)/mbkin**14 - (520*mckin**16)/mbkin**16)*q_cut)/
                mbkin**2 + (2*(-70 + (140*mckin**2)/mbkin**2 + (1545*mckin**4)/
                   mbkin**4 - (1364*mckin**6)/mbkin**6 - (5577*mckin**8)/
                   mbkin**8 - (1119*mckin**10)/mbkin**10 + (1230*mckin**12)/
                   mbkin**12 + (175*mckin**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
               (2*(-70 + (105*mckin**2)/mbkin**2 + (2245*mckin**4)/mbkin**4 + 
                  (4391*mckin**6)/mbkin**6 + (4496*mckin**8)/mbkin**8 + 
                  (1930*mckin**10)/mbkin**10 + (175*mckin**12)/mbkin**12)*q_cut**3)/
                mbkin**6 - ((-210 + (315*mckin**2)/mbkin**2 + (6325*mckin**4)/
                   mbkin**4 + (10463*mckin**6)/mbkin**6 + (5345*mckin**8)/
                   mbkin**8 + (490*mckin**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
               ((-70 + (175*mckin**2)/mbkin**2 + (1886*mckin**4)/mbkin**4 + 
                  (1641*mckin**6)/mbkin**6 + (140*mckin**8)/mbkin**8)*q_cut**5)/
                mbkin**10 - (2*mckin**2*(35 + (58*mckin**2)/mbkin**2 + 
                  (35*mckin**4)/mbkin**4)*q_cut**6)/mbkin**14 + (110*mckin**2*
                 (mbkin**2 + mckin**2)*q_cut**7)/mbkin**18 - (40*mckin**2*q_cut**8)/
                mbkin**18) + 4*(-((-1 + mckin**2/mbkin**2)**2*(-470 - 
                  (5510*mckin**2)/mbkin**2 + (55113*mckin**4)/mbkin**4 + 
                  (61525*mckin**6)/mbkin**6 - (282350*mckin**8)/mbkin**8 - 
                  (267510*mckin**10)/mbkin**10 - (31085*mckin**12)/mbkin**12 + 
                  (14647*mckin**14)/mbkin**14 + (2040*mckin**16)/mbkin**16)) + 
               ((-1450 - (18580*mckin**2)/mbkin**2 + (103993*mckin**4)/mbkin**4 + 
                  (231998*mckin**6)/mbkin**6 - (425575*mckin**8)/mbkin**8 - 
                  (1094180*mckin**10)/mbkin**10 - (618145*mckin**12)/mbkin**12 - 
                  (41638*mckin**14)/mbkin**14 + (42937*mckin**16)/mbkin**16 + 
                  (6240*mckin**18)/mbkin**18)*q_cut)/mbkin**2 - (2*(-490 - 
                  (7300*mckin**2)/mbkin**2 + (13660*mckin**4)/mbkin**4 + 
                  (22907*mckin**6)/mbkin**6 - (77531*mckin**8)/mbkin**8 - 
                  (62779*mckin**10)/mbkin**10 + (16973*mckin**12)/mbkin**12 + 
                  (16860*mckin**14)/mbkin**14 + (2100*mckin**16)/mbkin**16)*q_cut**2)/
                mbkin**4 - (2*(-490 - (8770*mckin**2)/mbkin**2 - (7155*mckin**4)/
                   mbkin**4 + (17517*mckin**6)/mbkin**6 + (38840*mckin**8)/
                   mbkin**8 + (45853*mckin**10)/mbkin**10 + (18785*mckin**12)/
                   mbkin**12 + (2100*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
               ((-1330 - (26770*mckin**2)/mbkin**2 - (18885*mckin**4)/mbkin**4 + 
                  (89471*mckin**6)/mbkin**6 + (131639*mckin**8)/mbkin**8 + 
                  (55915*mckin**10)/mbkin**10 + (5880*mckin**12)/mbkin**12)*q_cut**4)/
                mbkin**8 - ((-350 - (8440*mckin**2)/mbkin**2 + (3047*mckin**4)/
                   mbkin**4 + (29380*mckin**6)/mbkin**6 + (16563*mckin**8)/
                   mbkin**8 + (1680*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
               (8*(-35 - (30*mckin**2)/mbkin**2 + (14*mckin**4)/mbkin**4 + 
                  (76*mckin**6)/mbkin**6 + (105*mckin**8)/mbkin**8)*q_cut**6)/
                mbkin**12 - (40*(-11 - (11*mckin**2)/mbkin**2 + (26*mckin**4)/
                   mbkin**4 + (33*mckin**6)/mbkin**6)*q_cut**7)/mbkin**14 - 
               (160*(mbkin**4 - 3*mckin**4)*q_cut**8)/mbkin**20)*rG - 
             24*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(-40 - (530*mckin**2)/
                   mbkin**2 + (1237*mckin**4)/mbkin**4 + (117101*mckin**6)/
                   mbkin**6 + (301726*mckin**8)/mbkin**8 + (144226*mckin**10)/
                   mbkin**10 - (5749*mckin**12)/mbkin**12 + (619*mckin**14)/
                   mbkin**14 + (850*mckin**16)/mbkin**16)) + 
               ((-110 - (2180*mckin**2)/mbkin**2 - (8043*mckin**4)/mbkin**4 + 
                  (231598*mckin**6)/mbkin**6 + (844217*mckin**8)/mbkin**8 + 
                  (919012*mckin**10)/mbkin**10 + (277427*mckin**12)/mbkin**12 - 
                  (28230*mckin**14)/mbkin**14 + (1469*mckin**16)/mbkin**16 + 
                  (2600*mckin**18)/mbkin**18)*q_cut)/mbkin**2 - 
               (2*(-35 - (870*mckin**2)/mbkin**2 - (7720*mckin**4)/mbkin**4 + 
                  (19403*mckin**6)/mbkin**6 + (68795*mckin**8)/mbkin**8 + 
                  (21911*mckin**10)/mbkin**10 - (10719*mckin**12)/mbkin**12 + 
                  (1600*mckin**14)/mbkin**14 + (875*mckin**16)/mbkin**16)*q_cut**2)/
                mbkin**4 + (2*(35 + (975*mckin**2)/mbkin**2 + (11590*mckin**4)/
                   mbkin**4 + (21792*mckin**6)/mbkin**6 + (19921*mckin**8)/
                   mbkin**8 + (9944*mckin**10)/mbkin**10 - (2650*mckin**12)/
                   mbkin**12 - (875*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 + 
               ((-140 - (2810*mckin**2)/mbkin**2 - (35835*mckin**4)/mbkin**4 - 
                  (69801*mckin**6)/mbkin**6 - (30887*mckin**8)/mbkin**8 + 
                  (8455*mckin**10)/mbkin**10 + (2450*mckin**12)/mbkin**12)*q_cut**4)/
                mbkin**8 + ((70 + (900*mckin**2)/mbkin**2 + (10637*mckin**4)/
                   mbkin**4 + (9736*mckin**6)/mbkin**6 - (2311*mckin**8)/
                   mbkin**8 - (700*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
               ((70 + (60*mckin**2)/mbkin**2 + (308*mckin**4)/mbkin**4 + 
                  (76*mckin**6)/mbkin**6 + (350*mckin**8)/mbkin**8)*q_cut**6)/
                mbkin**12 - (10*(11 + (11*mckin**2)/mbkin**2 + (27*mckin**4)/
                   mbkin**4 + (55*mckin**6)/mbkin**6)*q_cut**7)/mbkin**14 + 
               (40*(mbkin**4 + 5*mckin**4)*q_cut**8)/mbkin**20)*rhoD - 2520*sB - 
             (36120*mckin**2*sB)/mbkin**2 + (411444*mckin**4*sB)/mbkin**4 + 
             (1483596*mckin**6*sB)/mbkin**6 - (1465380*mckin**8*sB)/mbkin**8 - 
             (1936620*mckin**10*sB)/mbkin**10 + (711900*mckin**12*sB)/mbkin**12 + 
             (534996*mckin**14*sB)/mbkin**14 + (377244*mckin**16*sB)/mbkin**16 - 
             (58140*mckin**18*sB)/mbkin**18 - (20400*mckin**20*sB)/mbkin**20 + 
             (7560*q_cut*sB)/mbkin**2 + (141120*mckin**2*q_cut*sB)/mbkin**4 - 
             (497964*mckin**4*q_cut*sB)/mbkin**6 - (5101848*mckin**6*q_cut*sB)/
              mbkin**8 - (9831468*mckin**8*q_cut*sB)/mbkin**10 - 
             (7031328*mckin**10*q_cut*sB)/mbkin**12 - (2379828*mckin**12*q_cut*sB)/
              mbkin**14 - (327624*mckin**14*q_cut*sB)/mbkin**16 + 
             (283140*mckin**16*q_cut*sB)/mbkin**18 + (62400*mckin**18*q_cut*sB)/
              mbkin**20 - (5040*q_cut**2*sB)/mbkin**4 - (107520*mckin**2*q_cut**2*sB)/
              mbkin**6 - (50400*mckin**4*q_cut**2*sB)/mbkin**8 + 
             (1033512*mckin**6*q_cut**2*sB)/mbkin**10 + (1250856*mckin**8*q_cut**2*sB)/
              mbkin**12 + (328632*mckin**10*q_cut**2*sB)/mbkin**14 - 
             (123720*mckin**12*q_cut**2*sB)/mbkin**16 - (228000*mckin**14*q_cut**2*sB)/
              mbkin**18 - (42000*mckin**16*q_cut**2*sB)/mbkin**20 - 
             (5040*q_cut**3*sB)/mbkin**6 - (122640*mckin**2*q_cut**3*sB)/mbkin**8 - 
             (457800*mckin**4*q_cut**3*sB)/mbkin**10 - (691848*mckin**6*q_cut**3*sB)/
              mbkin**12 - (888528*mckin**8*q_cut**3*sB)/mbkin**14 - 
             (681720*mckin**10*q_cut**3*sB)/mbkin**16 - (307800*mckin**12*q_cut**3*sB)/
              mbkin**18 - (42000*mckin**14*q_cut**3*sB)/mbkin**20 + 
             (7560*q_cut**4*sB)/mbkin**8 + (183960*mckin**2*q_cut**4*sB)/mbkin**10 + 
             (716940*mckin**4*q_cut**4*sB)/mbkin**12 + (1069212*mckin**6*q_cut**4*sB)/
              mbkin**14 + (894780*mckin**8*q_cut**4*sB)/mbkin**16 + 
             (445260*mckin**10*q_cut**4*sB)/mbkin**18 + (58800*mckin**12*q_cut**4*sB)/
              mbkin**20 - (2520*q_cut**5*sB)/mbkin**10 - (58800*mckin**2*q_cut**5*sB)/
              mbkin**12 - (185724*mckin**4*q_cut**5*sB)/mbkin**14 - 
             (197424*mckin**6*q_cut**5*sB)/mbkin**16 - (131820*mckin**8*q_cut**5*sB)/
              mbkin**18 - (16800*mckin**10*q_cut**5*sB)/mbkin**20 + 
             (3024*mckin**4*q_cut**6*sB)/mbkin**16 + (7200*mckin**6*q_cut**6*sB)/
              mbkin**18 + (8400*mckin**8*q_cut**6*sB)/mbkin**20 - 
             (9840*mckin**4*q_cut**7*sB)/mbkin**18 - (13200*mckin**6*q_cut**7*sB)/
              mbkin**20 + (4800*mckin**4*q_cut**8*sB)/mbkin**20 + 1360*sE + 
             (27200*mckin**2*sE)/mbkin**2 - (259360*mckin**4*sE)/mbkin**4 - 
             (553312*mckin**6*sE)/mbkin**6 + (1278032*mckin**8*sE)/mbkin**8 + 
             (59360*mckin**10*sE)/mbkin**10 - (668080*mckin**12*sE)/mbkin**12 + 
             (218720*mckin**14*sE)/mbkin**14 - (143008*mckin**16*sE)/mbkin**16 + 
             (26848*mckin**18*sE)/mbkin**18 + (12240*mckin**20*sE)/mbkin**20 - 
             (4160*q_cut*sE)/mbkin**2 - (96800*mckin**2*q_cut*sE)/mbkin**4 + 
             (332240*mckin**4*q_cut*sE)/mbkin**6 + (2431312*mckin**6*q_cut*sE)/
              mbkin**8 + (3031744*mckin**8*q_cut*sE)/mbkin**10 + 
             (707744*mckin**10*q_cut*sE)/mbkin**12 + (12784*mckin**12*q_cut*sE)/
              mbkin**14 + (68464*mckin**14*q_cut*sE)/mbkin**16 - 
             (155968*mckin**16*q_cut*sE)/mbkin**18 - (37440*mckin**18*q_cut*sE)/
              mbkin**20 + (2800*q_cut**2*sE)/mbkin**4 + (71200*mckin**2*q_cut**2*sE)/
              mbkin**6 - (16000*mckin**4*q_cut**2*sE)/mbkin**8 - 
             (560480*mckin**6*q_cut**2*sE)/mbkin**10 - (296416*mckin**8*q_cut**2*sE)/
              mbkin**12 + (240832*mckin**10*q_cut**2*sE)/mbkin**14 + 
             (251584*mckin**12*q_cut**2*sE)/mbkin**16 + (160320*mckin**14*q_cut**2*sE)/
              mbkin**18 + (25200*mckin**16*q_cut**2*sE)/mbkin**20 + 
             (2800*q_cut**3*sE)/mbkin**6 + (79600*mckin**2*q_cut**3*sE)/mbkin**8 + 
             (217200*mckin**4*q_cut**3*sE)/mbkin**10 + (182640*mckin**6*q_cut**3*sE)/
              mbkin**12 + (108784*mckin**8*q_cut**3*sE)/mbkin**14 + 
             (46544*mckin**10*q_cut**3*sE)/mbkin**16 + (112720*mckin**12*q_cut**3*sE)/
              mbkin**18 + (25200*mckin**14*q_cut**3*sE)/mbkin**20 - 
             (3920*q_cut**4*sE)/mbkin**8 - (120320*mckin**2*q_cut**4*sE)/mbkin**10 - 
             (337440*mckin**4*q_cut**4*sE)/mbkin**12 - (306080*mckin**6*q_cut**4*sE)/
              mbkin**14 - (255776*mckin**8*q_cut**4*sE)/mbkin**16 - 
             (207040*mckin**10*q_cut**4*sE)/mbkin**18 - (35280*mckin**12*q_cut**4*sE)/
              mbkin**20 + (1120*q_cut**5*sE)/mbkin**10 + (38720*mckin**2*q_cut**5*sE)/
              mbkin**12 + (80720*mckin**4*q_cut**5*sE)/mbkin**14 + 
             (62992*mckin**6*q_cut**5*sE)/mbkin**16 + (56832*mckin**8*q_cut**5*sE)/
              mbkin**18 + (10080*mckin**10*q_cut**5*sE)/mbkin**20 - 
             (560*q_cut**6*sE)/mbkin**12 - (480*mckin**2*q_cut**6*sE)/mbkin**14 - 
             (1120*mckin**4*q_cut**6*sE)/mbkin**16 + (2848*mckin**6*q_cut**6*sE)/
              mbkin**18 - (5040*mckin**8*q_cut**6*sE)/mbkin**20 + (880*q_cut**7*sE)/
              mbkin**14 + (880*mckin**2*q_cut**7*sE)/mbkin**16 + 
             (3440*mckin**4*q_cut**7*sE)/mbkin**18 + (7920*mckin**6*q_cut**7*sE)/
              mbkin**20 - (320*q_cut**8*sE)/mbkin**16 - (2880*mckin**4*q_cut**8*sE)/
              mbkin**20 + 460*sqB + (3110*mckin**2*sqB)/mbkin**2 - 
             (53971*mckin**4*sqB)/mbkin**4 - (232651*mckin**6*sqB)/mbkin**6 + 
             (64337*mckin**8*sqB)/mbkin**8 + (481775*mckin**10*sqB)/mbkin**10 - 
             (5635*mckin**12*sqB)/mbkin**12 - (222829*mckin**14*sqB)/mbkin**14 - 
             (41029*mckin**16*sqB)/mbkin**16 + (5923*mckin**18*sqB)/mbkin**18 + 
             (510*mckin**20*sqB)/mbkin**20 - (1370*q_cut*sqB)/mbkin**2 - 
             (15620*mckin**2*q_cut*sqB)/mbkin**4 + (66491*mckin**4*q_cut*sqB)/
              mbkin**6 + (746374*mckin**6*q_cut*sqB)/mbkin**8 + 
             (1788331*mckin**8*q_cut*sqB)/mbkin**10 + (1693796*mckin**10*q_cut*sqB)/
              mbkin**12 + (617581*mckin**12*q_cut*sqB)/mbkin**14 + 
             (27010*mckin**14*q_cut*sqB)/mbkin**16 - (22153*mckin**16*q_cut*sqB)/
              mbkin**18 - (1560*mckin**18*q_cut*sqB)/mbkin**20 + (910*q_cut**2*sqB)/
              mbkin**4 + (12940*mckin**2*q_cut**2*sqB)/mbkin**6 + 
             (12080*mckin**4*q_cut**2*sqB)/mbkin**8 - (138098*mckin**6*q_cut**2*sqB)/
              mbkin**10 - (257050*mckin**8*q_cut**2*sqB)/mbkin**12 - 
             (92426*mckin**10*q_cut**2*sqB)/mbkin**14 + (34474*mckin**12*q_cut**2*sqB)/
              mbkin**16 + (17880*mckin**14*q_cut**2*sqB)/mbkin**18 + 
             (1050*mckin**16*q_cut**2*sqB)/mbkin**20 + (910*q_cut**3*sqB)/mbkin**6 + 
             (15670*mckin**2*q_cut**3*sqB)/mbkin**8 + (63360*mckin**4*q_cut**3*sqB)/
              mbkin**10 + (93252*mckin**6*q_cut**3*sqB)/mbkin**12 + 
             (86746*mckin**8*q_cut**3*sqB)/mbkin**14 + (58364*mckin**10*q_cut**3*sqB)/
              mbkin**16 + (16480*mckin**12*q_cut**3*sqB)/mbkin**18 + 
             (1050*mckin**14*q_cut**3*sqB)/mbkin**20 - (1400*q_cut**4*sqB)/mbkin**8 - 
             (23390*mckin**2*q_cut**4*sqB)/mbkin**10 - (101145*mckin**4*q_cut**4*sqB)/
              mbkin**12 - (159083*mckin**6*q_cut**4*sqB)/mbkin**14 - 
             (104861*mckin**8*q_cut**4*sqB)/mbkin**16 - (27235*mckin**10*q_cut**4*sqB)/
              mbkin**18 - (1470*mckin**12*q_cut**4*sqB)/mbkin**20 + 
             (490*q_cut**5*sqB)/mbkin**10 + (7340*mckin**2*q_cut**5*sqB)/mbkin**12 + 
             (26891*mckin**4*q_cut**5*sqB)/mbkin**14 + (27988*mckin**6*q_cut**5*sqB)/
              mbkin**16 + (8787*mckin**8*q_cut**5*sqB)/mbkin**18 + 
             (420*mckin**10*q_cut**5*sqB)/mbkin**20 + (70*q_cut**6*sqB)/mbkin**12 + 
             (60*mckin**2*q_cut**6*sqB)/mbkin**14 + (644*mckin**4*q_cut**6*sqB)/
              mbkin**16 + (268*mckin**6*q_cut**6*sqB)/mbkin**18 - 
             (210*mckin**8*q_cut**6*sqB)/mbkin**20 - (110*q_cut**7*sqB)/mbkin**14 - 
             (110*mckin**2*q_cut**7*sqB)/mbkin**16 + (50*mckin**4*q_cut**7*sqB)/
              mbkin**18 + (330*mckin**6*q_cut**7*sqB)/mbkin**20 + (40*q_cut**8*sqB)/
              mbkin**16 - (120*mckin**4*q_cut**8*sqB)/mbkin**20))*
          np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 - 
        (60480*mckin**8*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
             2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          (72*mckin**2*muG - (1080*mckin**6*muG)/mbkin**4 - (720*mckin**8*muG)/
            mbkin**6 - (36*mckin**2*muG**2)/mbkin**2 + (540*mckin**6*muG**2)/
            mbkin**6 + (360*mckin**8*muG**2)/mbkin**8 + (36*mckin**2*muG*mupi)/
            mbkin**2 - (540*mckin**6*muG*mupi)/mbkin**6 - (360*mckin**8*muG*mupi)/
            mbkin**8 - 16*(-6 - (47*mckin**2)/mbkin**2 + (70*mckin**4)/mbkin**4 + 
             (55*mckin**6)/mbkin**6)*rE + 4*(3 - (68*mckin**2)/mbkin**2 + 
             (55*mckin**4)/mbkin**4 + (280*mckin**6)/mbkin**6 + 
             (90*mckin**8)/mbkin**8)*rG + 72*mbkin*rhoD + (624*mckin**2*rhoD)/
            mbkin + (4200*mckin**4*rhoD)/mbkin**3 + (5040*mckin**6*rhoD)/
            mbkin**5 + (720*mckin**8*rhoD)/mbkin**7 + 36*sB + 
           (840*mckin**2*sB)/mbkin**2 + (2340*mckin**4*sB)/mbkin**4 + 
           (1320*mckin**6*sB)/mbkin**6 + (360*mckin**8*sB)/mbkin**8 - 
           (448*mckin**2*sE)/mbkin**2 - (880*mckin**4*sE)/mbkin**4 + 
           (80*mckin**6*sE)/mbkin**6 - 9*sqB - (118*mckin**2*sqB)/mbkin**2 - 
           (385*mckin**4*sqB)/mbkin**4 - (370*mckin**6*sqB)/mbkin**6 - 
           (90*mckin**8*sqB)/mbkin**8)*np.log((mbkin**2 + mckin**2 - q_cut - 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/
             (mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                 mbkin**4)))**3)/mbkin**8))/
      (2520*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
          2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
        3) + 
     (api4*((mbkin**2*(180*mbkin**4*((-8*(mbkin - mckin)**2*(mbkin + mckin)*(
                3*mbkin + 3*mckin + 8*mbkin*mckin))/(9*mbkin**6) + 
             (16*(4*mbkin**3 - 4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*
               q_cut)/(9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*
            ((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/mbkin**2 - 
                (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6) + 
              ((-3 + (14*mckin**2)/mbkin**2 + (26*mckin**4)/mbkin**4 + 
                 (14*mckin**6)/mbkin**6 - (3*mckin**8)/mbkin**8)*q_cut)/mbkin**2 + 
              (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 2*mbkin**2*mckin**4 + mckin**6)*
                q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*
                q_cut**3)/mbkin**10 - (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + 
              q_cut**5/mbkin**10)**2*(1 - (34*mckin**2)/mbkin**2 - (1133*mckin**4)/
              mbkin**4 - (2708*mckin**6)/mbkin**6 - (1133*mckin**8)/mbkin**8 - 
             (34*mckin**10)/mbkin**10 + mckin**12/mbkin**12 + 
             ((mbkin**10 - 31*mbkin**8*mckin**2 - 390*mbkin**6*mckin**4 - 
                390*mbkin**4*mckin**6 - 31*mbkin**2*mckin**8 + mckin**10)*q_cut)/
              mbkin**12 + ((mbkin**8 - 26*mbkin**6*mckin**2 - 118*mbkin**4*
                 mckin**4 - 26*mbkin**2*mckin**6 + mckin**8)*q_cut**2)/mbkin**12 + 
             ((mbkin**6 - 19*mbkin**4*mckin**2 - 19*mbkin**2*mckin**4 + mckin**6)*
               q_cut**3)/mbkin**12 + ((-6 + (4*mckin**2)/mbkin**2 - (6*mckin**4)/
                 mbkin**4)*q_cut**4)/mbkin**8 - (6*(mbkin**2 + mckin**2)*q_cut**5)/
              mbkin**12 + (8*q_cut**6)/mbkin**12) + ((-1 + mckin**2/mbkin**2)**2 - 
             (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
            ((64*mbkin*(-((-1 + mckin**2/mbkin**2)**4*(1 + mckin**2/mbkin**2)**2*
                  (2803 - (59389*mckin**2)/mbkin**2 + (425638*mckin**4)/
                    mbkin**4 - (142434*mckin**6)/mbkin**6 - (7007992*mckin**8)/
                    mbkin**8 - (411584*mckin**10)/mbkin**10 + (753258*mckin**12)/
                    mbkin**12 - (298894*mckin**14)/mbkin**14 + (24853*mckin**16)/
                    mbkin**16 + (461*mckin**18)/mbkin**18)) + 
                ((-1 + mckin**2/mbkin**2)**2*(15765 - (259196*mckin**2)/mbkin**2 + 
                   (1031280*mckin**4)/mbkin**4 + (5842740*mckin**6)/mbkin**6 - 
                   (24509183*mckin**8)/mbkin**8 - (87162472*mckin**10)/
                    mbkin**10 - (89842656*mckin**12)/mbkin**12 - 
                   (25831304*mckin**14)/mbkin**14 + (7046591*mckin**16)/
                    mbkin**16 - (151836*mckin**18)/mbkin**18 - (1156752*
                     mckin**20)/mbkin**20 + (149588*mckin**22)/mbkin**22 + 
                   (2475*mckin**24)/mbkin**24)*q_cut)/mbkin**2 - 
                (2*(15112 - (210025*mckin**2)/mbkin**2 + (472499*mckin**4)/
                    mbkin**4 + (5851497*mckin**6)/mbkin**6 - (11357051*mckin**8)/
                    mbkin**8 - (56972502*mckin**10)/mbkin**10 - (85340102*
                     mckin**12)/mbkin**12 - (61750930*mckin**14)/mbkin**14 - 
                   (11562978*mckin**16)/mbkin**16 + (7576839*mckin**18)/
                    mbkin**18 - (785517*mckin**20)/mbkin**20 - (914135*mckin**22)/
                    mbkin**22 + (150229*mckin**24)/mbkin**24 + (2104*mckin**26)/
                    mbkin**26)*q_cut**2)/mbkin**4 + (2*(4935 - (22515*mckin**2)/
                    mbkin**2 - (355240*mckin**4)/mbkin**4 + (1715991*mckin**6)/
                    mbkin**6 + (2655127*mckin**8)/mbkin**8 - (4426866*mckin**10)/
                    mbkin**10 - (3311288*mckin**12)/mbkin**12 + 
                   (4104698*mckin**14)/mbkin**14 + (1043701*mckin**16)/
                    mbkin**16 - (1487323*mckin**18)/mbkin**18 + (15840*mckin**20)/
                    mbkin**20 + (62895*mckin**22)/mbkin**22 + (45*mckin**24)/
                    mbkin**24)*q_cut**3)/mbkin**6 + ((43537 - (458971*mckin**2)/
                    mbkin**2 - (93925*mckin**4)/mbkin**4 + (12415735*mckin**6)/
                    mbkin**6 + (21452530*mckin**8)/mbkin**8 + (19866834*
                     mckin**10)/mbkin**10 + (19989798*mckin**12)/mbkin**12 + 
                   (10268870*mckin**14)/mbkin**14 - (3574435*mckin**16)/
                    mbkin**16 - (1439255*mckin**18)/mbkin**18 + 
                   (436447*mckin**20)/mbkin**20 + (7139*mckin**22)/mbkin**22)*
                  q_cut**4)/mbkin**8 - ((58149 - (424380*mckin**2)/mbkin**2 - 
                   (1661061*mckin**4)/mbkin**4 + (11127360*mckin**6)/mbkin**6 + 
                   (29799650*mckin**8)/mbkin**8 + (27800000*mckin**10)/
                    mbkin**10 + (5553270*mckin**12)/mbkin**12 - 
                   (6210720*mckin**14)/mbkin**14 - (461543*mckin**16)/mbkin**16 + 
                   (648988*mckin**18)/mbkin**18 + (6063*mckin**20)/mbkin**20)*
                  q_cut**5)/mbkin**10 + (4*(2930 - (20127*mckin**2)/mbkin**2 - 
                   (145371*mckin**4)/mbkin**4 + (861647*mckin**6)/mbkin**6 + 
                   (1702217*mckin**8)/mbkin**8 + (293611*mckin**10)/mbkin**10 - 
                   (560795*mckin**12)/mbkin**12 + (4239*mckin**14)/mbkin**14 + 
                   (35411*mckin**16)/mbkin**16 + (110*mckin**18)/mbkin**18)*q_cut**6)/
                 mbkin**12 - (4*(-5667 + (16545*mckin**2)/mbkin**2 + 
                   (282306*mckin**4)/mbkin**4 + (576507*mckin**6)/mbkin**6 + 
                   (446228*mckin**8)/mbkin**8 + (21045*mckin**10)/mbkin**10 - 
                   (206250*mckin**12)/mbkin**12 - (83609*mckin**14)/mbkin**14 + 
                   (951*mckin**16)/mbkin**16)*q_cut**7)/mbkin**14 + 
                ((-13425 + (43605*mckin**2)/mbkin**2 + (1070385*mckin**4)/
                    mbkin**4 + (1555763*mckin**6)/mbkin**6 + (247781*mckin**8)/
                    mbkin**8 - (859585*mckin**10)/mbkin**10 - (271765*mckin**12)/
                    mbkin**12 + (8265*mckin**14)/mbkin**14)*q_cut**8)/mbkin**16 + 
                ((3211 - (5958*mckin**2)/mbkin**2 - (298825*mckin**4)/mbkin**4 - 
                   (133880*mckin**6)/mbkin**6 + (266305*mckin**8)/mbkin**8 + 
                   (71726*mckin**10)/mbkin**10 - (643*mckin**12)/mbkin**12)*q_cut**9)/
                 mbkin**18 - (2*(3444 + (8075*mckin**2)/mbkin**2 + 
                   (7199*mckin**4)/mbkin**4 + (24877*mckin**6)/mbkin**6 + 
                   (14553*mckin**8)/mbkin**8 + (4308*mckin**10)/mbkin**10)*q_cut**10)/
                 mbkin**20 + (2*(3295 + (6917*mckin**2)/mbkin**2 + 
                   (13436*mckin**4)/mbkin**4 + (14159*mckin**6)/mbkin**6 + 
                   (3805*mckin**8)/mbkin**8)*q_cut**11)/mbkin**22 - 
                ((1869 + (3645*mckin**2)/mbkin**2 + (9103*mckin**4)/mbkin**4 + 
                   (2463*mckin**6)/mbkin**6)*q_cut**12)/mbkin**24 + 
                (5*(mbkin**4 + 80*mbkin**2*mckin**2 + 59*mckin**4)*q_cut**13)/
                 mbkin**30 - (48*(mbkin**2 + 2*mckin**2)*q_cut**14)/mbkin**30 + 
                (40*q_cut**15)/mbkin**30))/3 + 180*(mbkin**4*
                ((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/mbkin**2 - 
                    (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6) + 
                  ((-3 + (14*mckin**2)/mbkin**2 + (26*mckin**4)/mbkin**4 + 
                     (14*mckin**6)/mbkin**6 - (3*mckin**8)/mbkin**8)*q_cut)/
                   mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 2*mbkin**2*
                      mckin**4 + mckin**6)*q_cut**2)/mbkin**10 + 
                  (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*q_cut**3)/mbkin**10 - 
                  (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2*
                ((-8*(3 + 8*mbkin)*mckin**12)/(3*mbkin**14) + 
                 (8*mckin**10*(3 + 8*mckin))/(3*mbkin**12) - 
                 (136*(3*mbkin**2 + 8*mbkin**2*mckin - 3*mckin**2 - 8*mbkin*
                     mckin**2))/(9*mbkin**4) - 1133*((-8*(3 + 8*mbkin)*mckin**4)/
                    (9*mbkin**6) + (8*mckin**2*(3 + 8*mckin))/(9*mbkin**4)) - 
                 2708*((-4*(3 + 8*mbkin)*mckin**6)/(3*mbkin**8) + 
                   (4*mckin**4*(3 + 8*mckin))/(3*mbkin**6)) - 1133*
                  ((-16*(3 + 8*mbkin)*mckin**8)/(9*mbkin**10) + (16*mckin**6*
                     (3 + 8*mckin))/(9*mbkin**8)) - 34*((-20*(3 + 8*mbkin)*
                     mckin**10)/(9*mbkin**12) + (20*mckin**8*(3 + 8*mckin))/
                    (9*mbkin**10)) - (8*(48*mbkin**10 + 4*mbkin**11 + 
                    124*mbkin**10*mckin + 1077*mbkin**8*mckin**2 - 248*mbkin**9*
                     mckin**2 + 3120*mbkin**8*mckin**3 - 4680*mbkin**7*mckin**4 + 
                    4680*mbkin**6*mckin**5 - 2154*mbkin**4*mckin**6 - 
                    6240*mbkin**5*mckin**6 + 496*mbkin**4*mckin**7 - 240*mbkin**2*
                     mckin**8 - 620*mbkin**3*mckin**8 - 20*mbkin**2*mckin**9 + 
                    9*mckin**10 + 24*mbkin*mckin**10)*q_cut)/(9*mbkin**14) - 
                 (8*(42*mbkin**8 + 8*mbkin**9 + 104*mbkin**8*mckin + 237*mbkin**6*
                     mckin**2 - 312*mbkin**7*mckin**2 + 944*mbkin**6*mckin**3 - 
                    591*mbkin**4*mckin**4 - 1888*mbkin**5*mckin**4 + 312*mbkin**4*
                     mckin**5 - 201*mbkin**2*mckin**6 - 520*mbkin**3*mckin**6 - 
                    16*mbkin**2*mckin**7 + 9*mckin**8 + 24*mbkin*mckin**8)*q_cut**2)/
                  (9*mbkin**14) - (8*(33*mbkin**6 + 12*mbkin**7 + 76*mbkin**6*
                     mckin - 57*mbkin**4*mckin**2 - 304*mbkin**5*mckin**2 + 
                    152*mbkin**4*mckin**3 - 147*mbkin**2*mckin**4 - 380*mbkin**3*
                     mckin**4 - 12*mbkin**2*mckin**5 + 9*mckin**6 + 24*mbkin*
                     mckin**6)*q_cut**3)/(9*mbkin**14) + (16*(21*mbkin**4 + 
                    48*mbkin**5 + 8*mbkin**4*mckin - 24*mbkin**2*mckin**2 - 
                    40*mbkin**3*mckin**2 - 24*mbkin**2*mckin**3 + 27*mckin**4 + 
                    72*mbkin*mckin**4)*q_cut**4)/(9*mbkin**14) + 
                 (16*(6*mbkin**2 + 20*mbkin**3 - 4*mbkin**2*mckin + 9*mckin**2 + 
                    24*mbkin*mckin**2)*q_cut**5)/(3*mbkin**14) - (64*(3 + 8*mbkin)*
                   q_cut**6)/(3*mbkin**14)) + (1 - (34*mckin**2)/mbkin**2 - 
                 (1133*mckin**4)/mbkin**4 - (2708*mckin**6)/mbkin**6 - 
                 (1133*mckin**8)/mbkin**8 - (34*mckin**10)/mbkin**10 + 
                 mckin**12/mbkin**12 + ((mbkin**10 - 31*mbkin**8*mckin**2 - 
                    390*mbkin**6*mckin**4 - 390*mbkin**4*mckin**6 - 31*mbkin**2*
                     mckin**8 + mckin**10)*q_cut)/mbkin**12 + 
                 ((mbkin**8 - 26*mbkin**6*mckin**2 - 118*mbkin**4*mckin**4 - 
                    26*mbkin**2*mckin**6 + mckin**8)*q_cut**2)/mbkin**12 + 
                 ((mbkin**6 - 19*mbkin**4*mckin**2 - 19*mbkin**2*mckin**4 + 
                    mckin**6)*q_cut**3)/mbkin**12 + ((-6 + (4*mckin**2)/mbkin**2 - 
                    (6*mckin**4)/mbkin**4)*q_cut**4)/mbkin**8 - 
                 (6*(mbkin**2 + mckin**2)*q_cut**5)/mbkin**12 + (8*q_cut**6)/mbkin**12)*
                ((8*mbkin**2*(3 + 8*mbkin)*((-1 + mckin**2/mbkin**2)**2*
                      (1 - (7*mckin**2)/mbkin**2 - (7*mckin**4)/mbkin**4 + 
                       mckin**6/mbkin**6) + ((-3 + (14*mckin**2)/mbkin**2 + 
                        (26*mckin**4)/mbkin**4 + (14*mckin**6)/mbkin**6 - 
                        (3*mckin**8)/mbkin**8)*q_cut)/mbkin**2 + (2*(mbkin**6 - 
                        2*mbkin**4*mckin**2 - 2*mbkin**2*mckin**4 + mckin**6)*
                       q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mckin**2 + 
                        mckin**4)*q_cut**3)/mbkin**10 - (3*(mbkin**2 + mckin**2)*
                       q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2)/9 + 2*mbkin**4*
                  ((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/mbkin**2 - 
                     (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6) + 
                   ((-3 + (14*mckin**2)/mbkin**2 + (26*mckin**4)/mbkin**4 + 
                      (14*mckin**6)/mbkin**6 - (3*mckin**8)/mbkin**8)*q_cut)/
                    mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 2*mbkin**2*
                       mckin**4 + mckin**6)*q_cut**2)/mbkin**10 + 
                   (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*q_cut**3)/mbkin**10 - 
                   (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)*
                  ((-4*(mbkin - mckin)**2*(mbkin + mckin)*(27*mbkin**7 + 
                      27*mbkin**6*mckin + 72*mbkin**7*mckin - 21*mbkin**5*
                       mckin**2 - 21*mbkin**4*mckin**3 - 56*mbkin**5*mckin**3 - 
                      93*mbkin**3*mckin**4 - 93*mbkin**2*mckin**5 - 248*mbkin**3*
                       mckin**5 + 15*mbkin*mckin**6 + 15*mckin**7 + 40*mbkin*
                       mckin**7))/(9*mbkin**12) + (4*(51*mbkin**8 + 24*mbkin**9 + 
                      112*mbkin**8*mckin + 72*mbkin**6*mckin**2 - 224*mbkin**7*
                       mckin**2 + 416*mbkin**6*mckin**3 - 108*mbkin**4*mckin**4 - 
                      624*mbkin**5*mckin**4 + 336*mbkin**4*mckin**5 - 204*mbkin**2*
                       mckin**6 - 448*mbkin**3*mckin**6 - 96*mbkin**2*mckin**7 + 
                      45*mckin**8 + 120*mbkin*mckin**8)*q_cut)/(9*mbkin**12) - 
                   (8*(12*mbkin**6 + 16*mbkin**7 + 16*mbkin**6*mckin - 6*mbkin**4*
                       mckin**2 - 48*mbkin**5*mckin**2 + 32*mbkin**4*mckin**3 - 
                      33*mbkin**2*mckin**4 - 64*mbkin**3*mckin**4 - 24*mbkin**2*
                       mckin**5 + 15*mckin**6 + 40*mbkin*mckin**6)*q_cut**2)/
                    (9*mbkin**12) - (8*(6*mbkin**4 + 24*mbkin**5 - 8*mbkin**4*
                       mckin + 6*mbkin**2*mckin**2 + 32*mbkin**3*mckin**2 - 
                      16*mbkin**2*mckin**3 + 15*mckin**4 + 40*mbkin*mckin**4)*
                     q_cut**3)/(9*mbkin**12) + (4*(9*mbkin**2 + 32*mbkin**3 - 
                      8*mbkin**2*mckin + 15*mckin**2 + 40*mbkin*mckin**2)*q_cut**4)/
                    (3*mbkin**12) - (20*(3 + 8*mbkin)*q_cut**5)/(9*mbkin**12))))) - 
           12*((-320*mckin**4*(mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)**2*
               (mbkin**2 + 2*mbkin*mckin + mckin**2 - q_cut)**2*(3*mbkin**4 + 
                8*mbkin**4*mckin - 6*mbkin**2*mckin**2 - 8*mbkin**3*mckin**2 - 
                8*mbkin**2*mckin**3 + 3*mckin**4 + 8*mbkin*mckin**4 - 
                3*mbkin**2*q_cut - 8*mbkin**2*mckin*q_cut - 3*mckin**2*q_cut - 
                8*mbkin*mckin**2*q_cut)*(37*mbkin**18 - 397*mbkin**16*mckin**2 - 
                2854*mbkin**14*mckin**4 + 18134*mbkin**12*mckin**6 + 
                75800*mbkin**10*mckin**8 + 75800*mbkin**8*mckin**10 + 
                18134*mbkin**6*mckin**12 - 2854*mbkin**4*mckin**14 - 
                397*mbkin**2*mckin**16 + 37*mckin**18 - 70*mbkin**16*q_cut + 
                132*mbkin**14*mckin**2*q_cut + 4424*mbkin**12*mckin**4*q_cut + 
                15500*mbkin**10*mckin**6*q_cut + 20508*mbkin**8*mckin**8*q_cut + 
                15500*mbkin**6*mckin**10*q_cut + 4424*mbkin**4*mckin**12*q_cut + 
                132*mbkin**2*mckin**14*q_cut - 70*mckin**16*q_cut - 37*mbkin**14*q_cut**2 + 
                307*mbkin**12*mckin**2*q_cut**2 + 6201*mbkin**10*mckin**4*q_cut**2 + 
                18225*mbkin**8*mckin**6*q_cut**2 + 18225*mbkin**6*mckin**8*q_cut**2 + 
                6201*mbkin**4*mckin**10*q_cut**2 + 307*mbkin**2*mckin**12*q_cut**2 - 
                37*mckin**14*q_cut**2 + 140*mbkin**12*q_cut**3 + 272*mbkin**10*mckin**2*
                 q_cut**3 - 2796*mbkin**8*mckin**4*q_cut**3 - 7136*mbkin**6*mckin**6*
                 q_cut**3 - 2796*mbkin**4*mckin**8*q_cut**3 + 272*mbkin**2*mckin**10*
                 q_cut**3 + 140*mckin**12*q_cut**3 - 49*mbkin**10*q_cut**4 + 13*mbkin**8*
                 mckin**2*q_cut**4 - 300*mbkin**6*mckin**4*q_cut**4 - 300*mbkin**4*
                 mckin**6*q_cut**4 + 13*mbkin**2*mckin**8*q_cut**4 - 49*mckin**10*q_cut**4 - 
                70*mbkin**8*q_cut**5 - 372*mbkin**6*mckin**2*q_cut**5 - 668*mbkin**4*
                 mckin**4*q_cut**5 - 372*mbkin**2*mckin**6*q_cut**5 - 70*mckin**8*q_cut**5 + 
                77*mbkin**6*q_cut**6 + 41*mbkin**4*mckin**2*q_cut**6 + 41*mbkin**2*
                 mckin**4*q_cut**6 + 77*mckin**6*q_cut**6 - 16*mbkin**4*q_cut**7 + 
                32*mbkin**2*mckin**2*q_cut**7 - 16*mckin**4*q_cut**7 - 28*mbkin**2*q_cut**8 - 
                28*mckin**2*q_cut**8 + 16*q_cut**9))/(mbkin**28*(mbkin**2 + mckin**2 - 
                q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - 
                mckin**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))) + ((-80*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + 
                  mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(3*mbkin**4 + 8*mbkin**4*
                   mckin - 6*mbkin**2*mckin**2 - 8*mbkin**3*mckin**2 - 
                  8*mbkin**2*mckin**3 + 3*mckin**4 + 8*mbkin*mckin**4 - 
                  8*mbkin**3*q_cut + 8*mbkin**2*mckin*q_cut - 6*mckin**2*q_cut - 
                  16*mbkin*mckin**2*q_cut + 3*q_cut**2 + 8*mbkin*q_cut**2)*(37*mbkin**18 - 
                  397*mbkin**16*mckin**2 - 2854*mbkin**14*mckin**4 + 
                  18134*mbkin**12*mckin**6 + 75800*mbkin**10*mckin**8 + 
                  75800*mbkin**8*mckin**10 + 18134*mbkin**6*mckin**12 - 
                  2854*mbkin**4*mckin**14 - 397*mbkin**2*mckin**16 + 
                  37*mckin**18 - 70*mbkin**16*q_cut + 132*mbkin**14*mckin**2*q_cut + 
                  4424*mbkin**12*mckin**4*q_cut + 15500*mbkin**10*mckin**6*q_cut + 
                  20508*mbkin**8*mckin**8*q_cut + 15500*mbkin**6*mckin**10*q_cut + 
                  4424*mbkin**4*mckin**12*q_cut + 132*mbkin**2*mckin**14*q_cut - 
                  70*mckin**16*q_cut - 37*mbkin**14*q_cut**2 + 307*mbkin**12*mckin**2*
                   q_cut**2 + 6201*mbkin**10*mckin**4*q_cut**2 + 18225*mbkin**8*mckin**6*
                   q_cut**2 + 18225*mbkin**6*mckin**8*q_cut**2 + 6201*mbkin**4*mckin**10*
                   q_cut**2 + 307*mbkin**2*mckin**12*q_cut**2 - 37*mckin**14*q_cut**2 + 
                  140*mbkin**12*q_cut**3 + 272*mbkin**10*mckin**2*q_cut**3 - 
                  2796*mbkin**8*mckin**4*q_cut**3 - 7136*mbkin**6*mckin**6*q_cut**3 - 
                  2796*mbkin**4*mckin**8*q_cut**3 + 272*mbkin**2*mckin**10*q_cut**3 + 
                  140*mckin**12*q_cut**3 - 49*mbkin**10*q_cut**4 + 13*mbkin**8*mckin**2*
                   q_cut**4 - 300*mbkin**6*mckin**4*q_cut**4 - 300*mbkin**4*mckin**6*
                   q_cut**4 + 13*mbkin**2*mckin**8*q_cut**4 - 49*mckin**10*q_cut**4 - 
                  70*mbkin**8*q_cut**5 - 372*mbkin**6*mckin**2*q_cut**5 - 668*mbkin**4*
                   mckin**4*q_cut**5 - 372*mbkin**2*mckin**6*q_cut**5 - 70*mckin**8*
                   q_cut**5 + 77*mbkin**6*q_cut**6 + 41*mbkin**4*mckin**2*q_cut**6 + 
                  41*mbkin**2*mckin**4*q_cut**6 + 77*mckin**6*q_cut**6 - 16*mbkin**4*
                   q_cut**7 + 32*mbkin**2*mckin**2*q_cut**7 - 16*mckin**4*q_cut**7 - 
                  28*mbkin**2*q_cut**8 - 28*mckin**2*q_cut**8 + 16*q_cut**9))/mbkin**28 + 
               np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(180*mckin**4*
                  ((-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 3*mckin + 
                      8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 
                      4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/
                    (9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*
                  ((-1 + mckin**2/mbkin**2)**2*(37 - (397*mckin**2)/mbkin**2 - 
                     (2854*mckin**4)/mbkin**4 + (18134*mckin**6)/mbkin**6 + 
                     (75800*mckin**8)/mbkin**8 + (75800*mckin**10)/mbkin**10 + 
                     (18134*mckin**12)/mbkin**12 - (2854*mckin**14)/mbkin**14 - 
                     (397*mckin**16)/mbkin**16 + (37*mckin**18)/mbkin**18) - 
                   (16*(9 - (62*mckin**2)/mbkin**2 - (662*mckin**4)/mbkin**4 + 
                      (1486*mckin**6)/mbkin**6 + (12121*mckin**8)/mbkin**8 + 
                      (19576*mckin**10)/mbkin**10 + (12121*mckin**12)/mbkin**12 + 
                      (1486*mckin**14)/mbkin**14 - (662*mckin**16)/mbkin**16 - 
                      (62*mckin**18)/mbkin**18 + (9*mckin**20)/mbkin**20)*q_cut)/
                    mbkin**2 + (4*(35 - (35*mckin**2)/mbkin**2 - (1604*mckin**4)/
                       mbkin**4 - (3896*mckin**6)/mbkin**6 - (2060*mckin**8)/
                       mbkin**8 - (2060*mckin**10)/mbkin**10 - (3896*mckin**12)/
                       mbkin**12 - (1604*mckin**14)/mbkin**14 - (35*mckin**16)/
                       mbkin**16 + (35*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
                   (16*(9 - (26*mckin**2)/mbkin**2 - (737*mckin**4)/mbkin**4 - 
                      (2164*mckin**6)/mbkin**6 - (2732*mckin**8)/mbkin**8 - 
                      (2164*mckin**10)/mbkin**10 - (737*mckin**12)/mbkin**12 - 
                      (26*mckin**14)/mbkin**14 + (9*mckin**16)/mbkin**16)*q_cut**3)/
                    mbkin**6 - (2*(183 + (203*mckin**2)/mbkin**2 - 
                      (5437*mckin**4)/mbkin**4 - (19201*mckin**6)/mbkin**6 - 
                      (19201*mckin**8)/mbkin**8 - (5437*mckin**10)/mbkin**10 + 
                      (203*mckin**12)/mbkin**12 + (183*mckin**14)/mbkin**14)*
                     q_cut**4)/mbkin**8 + (8*(21 + (14*mckin**2)/mbkin**2 - 
                      (277*mckin**4)/mbkin**4 - (668*mckin**6)/mbkin**6 - 
                      (277*mckin**8)/mbkin**8 + (14*mckin**10)/mbkin**10 + 
                      (21*mckin**12)/mbkin**12)*q_cut**5)/mbkin**10 + 
                   (8*(21 + (98*mckin**2)/mbkin**2 + (227*mckin**4)/mbkin**4 + 
                      (227*mckin**6)/mbkin**6 + (98*mckin**8)/mbkin**8 + 
                      (21*mckin**10)/mbkin**10)*q_cut**6)/mbkin**12 - 
                   (16*(15 + (34*mckin**2)/mbkin**2 + (58*mckin**4)/mbkin**4 + 
                      (34*mckin**6)/mbkin**6 + (15*mckin**8)/mbkin**8)*q_cut**7)/
                    mbkin**14 + ((81 + (37*mckin**2)/mbkin**2 + (37*mckin**4)/
                       mbkin**4 + (81*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
                   (56*(mbkin**2 + mckin**2)**2*q_cut**9)/mbkin**22 - 
                   (60*(mbkin**2 + mckin**2)*q_cut**10)/mbkin**22 + (16*q_cut**11)/
                    mbkin**22) + ((-1 + mckin**2/mbkin**2)**2 - 
                   (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
                  ((-64*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(65 - (810*mckin**2)/
                          mbkin**2 - (516*mckin**4)/mbkin**4 + (110224*mckin**6)/
                          mbkin**6 - (410120*mckin**8)/mbkin**8 - (2939462*
                          mckin**10)/mbkin**10 - (3104774*mckin**12)/mbkin**12 - 
                         (475058*mckin**14)/mbkin**14 + (163447*mckin**16)/
                          mbkin**16 - (52384*mckin**18)/mbkin**18 - 
                         (4742*mckin**20)/mbkin**20 + (850*mckin**22)/mbkin**
                          22)) + (2*(135 - (1420*mckin**2)/mbkin**2 - 
                         (3196*mckin**4)/mbkin**4 + (162656*mckin**6)/mbkin**6 - 
                         (138305*mckin**8)/mbkin**8 - (3423984*mckin**10)/
                          mbkin**10 - (6202792*mckin**12)/mbkin**12 - 
                         (3867608*mckin**14)/mbkin**14 - (95475*mckin**16)/
                          mbkin**16 + (226796*mckin**18)/mbkin**18 - 
                         (77740*mckin**20)/mbkin**20 - (7352*mckin**22)/mbkin**
                          22 + (1725*mckin**24)/mbkin**24)*q_cut)/mbkin**2 - 
                      (4*(70 - (455*mckin**2)/mbkin**2 - (3470*mckin**4)/
                          mbkin**4 + (37665*mckin**6)/mbkin**6 + (138922*
                          mckin**8)/mbkin**8 + (83452*mckin**10)/mbkin**10 + 
                         (121324*mckin**12)/mbkin**12 + (190868*mckin**14)/
                          mbkin**14 + (16424*mckin**16)/mbkin**16 - 
                         (25445*mckin**18)/mbkin**18 - (790*mckin**20)/mbkin**
                          20 + (875*mckin**22)/mbkin**22)*q_cut**2)/mbkin**4 - 
                      (2*(135 - (880*mckin**2)/mbkin**2 - (7621*mckin**4)/
                          mbkin**4 + (133182*mckin**6)/mbkin**6 + (543892*
                          mckin**8)/mbkin**8 + (658370*mckin**10)/mbkin**10 + 
                         (525528*mckin**12)/mbkin**12 + (55122*mckin**14)/
                          mbkin**14 - (76843*mckin**16)/mbkin**16 - 
                         (2082*mckin**18)/mbkin**18 + (1725*mckin**20)/mbkin**20)*
                        q_cut**3)/mbkin**6 + (2*(330 - (1015*mckin**2)/mbkin**2 - 
                         (21973*mckin**4)/mbkin**4 + (116514*mckin**6)/mbkin**6 + 
                         (621711*mckin**8)/mbkin**8 + (591448*mckin**10)/
                          mbkin**10 - (14853*mckin**12)/mbkin**12 - 
                         (96558*mckin**14)/mbkin**14 + (6769*mckin**16)/mbkin**
                          16 + (4275*mckin**18)/mbkin**18)*q_cut**4)/mbkin**8 - 
                      (2*(105 - (280*mckin**2)/mbkin**2 - (8822*mckin**4)/
                          mbkin**4 + (25052*mckin**6)/mbkin**6 + (79356*mckin**8)/
                          mbkin**8 - (7796*mckin**10)/mbkin**10 - (25990*
                          mckin**12)/mbkin**12 + (3136*mckin**14)/mbkin**14 + 
                         (1575*mckin**16)/mbkin**16)*q_cut**5)/mbkin**10 - 
                      (2*(105 + (455*mckin**2)/mbkin**2 - (8521*mckin**4)/
                          mbkin**4 - (25699*mckin**6)/mbkin**6 - (20961*mckin**8)/
                          mbkin**8 + (4285*mckin**10)/mbkin**10 + (11193*
                          mckin**12)/mbkin**12 + (1575*mckin**14)/mbkin**14)*
                        q_cut**6)/mbkin**12 + (10*(3 + (88*mckin**2)/mbkin**2 - 
                         (1667*mckin**4)/mbkin**4 - (3298*mckin**6)/mbkin**6 + 
                         (671*mckin**8)/mbkin**8 + (2090*mckin**10)/mbkin**10 + 
                         (225*mckin**12)/mbkin**12)*q_cut**7)/mbkin**14 + 
                      ((45 - (310*mckin**2)/mbkin**2 + (4303*mckin**4)/mbkin**4 + 
                         (694*mckin**6)/mbkin**6 - (5204*mckin**8)/mbkin**8 - 
                         (300*mckin**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
                      (4*(35 + (70*mckin**2)/mbkin**2 + (94*mckin**4)/mbkin**4 + 
                         (218*mckin**6)/mbkin**6 + (175*mckin**8)/mbkin**8)*q_cut**9)/
                       mbkin**18 - (30*(5 + (5*mckin**2)/mbkin**2 + (17*mckin**4)/
                          mbkin**4 + (25*mckin**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
                      (40*(mbkin**4 + 5*mckin**4)*q_cut**11)/mbkin**26))/3 + 
                   180*((8*mckin**2*(3 + 8*mckin)*((-1 + mckin**2/mbkin**2)**2*
                         (37 - (397*mckin**2)/mbkin**2 - (2854*mckin**4)/
                          mbkin**4 + (18134*mckin**6)/mbkin**6 + (75800*mckin**8)/
                          mbkin**8 + (75800*mckin**10)/mbkin**10 + (18134*
                          mckin**12)/mbkin**12 - (2854*mckin**14)/mbkin**14 - 
                          (397*mckin**16)/mbkin**16 + (37*mckin**18)/mbkin**18) - 
                        (16*(9 - (62*mckin**2)/mbkin**2 - (662*mckin**4)/mbkin**
                          4 + (1486*mckin**6)/mbkin**6 + (12121*mckin**8)/
                          mbkin**8 + (19576*mckin**10)/mbkin**10 + (12121*
                          mckin**12)/mbkin**12 + (1486*mckin**14)/mbkin**14 - 
                          (662*mckin**16)/mbkin**16 - (62*mckin**18)/mbkin**18 + 
                          (9*mckin**20)/mbkin**20)*q_cut)/mbkin**2 + 
                        (4*(35 - (35*mckin**2)/mbkin**2 - (1604*mckin**4)/
                          mbkin**4 - (3896*mckin**6)/mbkin**6 - (2060*mckin**8)/
                          mbkin**8 - (2060*mckin**10)/mbkin**10 - (3896*mckin**
                          12)/mbkin**12 - (1604*mckin**14)/mbkin**14 - 
                          (35*mckin**16)/mbkin**16 + (35*mckin**18)/mbkin**18)*
                          q_cut**2)/mbkin**4 + (16*(9 - (26*mckin**2)/mbkin**2 - 
                          (737*mckin**4)/mbkin**4 - (2164*mckin**6)/mbkin**6 - 
                          (2732*mckin**8)/mbkin**8 - (2164*mckin**10)/mbkin**10 - 
                          (737*mckin**12)/mbkin**12 - (26*mckin**14)/mbkin**14 + 
                          (9*mckin**16)/mbkin**16)*q_cut**3)/mbkin**6 - 
                        (2*(183 + (203*mckin**2)/mbkin**2 - (5437*mckin**4)/
                          mbkin**4 - (19201*mckin**6)/mbkin**6 - (19201*mckin**8)/
                          mbkin**8 - (5437*mckin**10)/mbkin**10 + (203*mckin**12)/
                          mbkin**12 + (183*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
                        (8*(21 + (14*mckin**2)/mbkin**2 - (277*mckin**4)/mbkin**
                          4 - (668*mckin**6)/mbkin**6 - (277*mckin**8)/mbkin**8 + 
                          (14*mckin**10)/mbkin**10 + (21*mckin**12)/mbkin**12)*
                          q_cut**5)/mbkin**10 + (8*(21 + (98*mckin**2)/mbkin**2 + 
                          (227*mckin**4)/mbkin**4 + (227*mckin**6)/mbkin**6 + 
                          (98*mckin**8)/mbkin**8 + (21*mckin**10)/mbkin**10)*
                          q_cut**6)/mbkin**12 - (16*(15 + (34*mckin**2)/mbkin**2 + 
                          (58*mckin**4)/mbkin**4 + (34*mckin**6)/mbkin**6 + 
                          (15*mckin**8)/mbkin**8)*q_cut**7)/mbkin**14 + 
                        ((81 + (37*mckin**2)/mbkin**2 + (37*mckin**4)/mbkin**4 + 
                          (81*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
                        (56*(mbkin**2 + mckin**2)**2*q_cut**9)/mbkin**22 - 
                        (60*(mbkin**2 + mckin**2)*q_cut**10)/mbkin**22 + (16*q_cut**11)/
                         mbkin**22))/9 + mckin**4*((-4*(mbkin - mckin)**2*
                         (mbkin + mckin)*(1413*mbkin**19 + 1413*mbkin**18*
                          mckin + 3768*mbkin**19*mckin + 13551*mbkin**17*
                          mckin**2 + 13551*mbkin**16*mckin**3 + 36136*mbkin**17*
                          mckin**3 - 197454*mbkin**15*mckin**4 - 197454*mbkin**14*
                          mckin**5 - 526544*mbkin**15*mckin**5 - 637590*mbkin**13*
                          mckin**6 - 637590*mbkin**12*mckin**7 - 1700240*
                          mbkin**13*mckin**7 + 227400*mbkin**11*mckin**8 + 
                          227400*mbkin**10*mckin**9 + 606400*mbkin**11*mckin**9 + 
                          1265388*mbkin**9*mckin**10 + 1265388*mbkin**8*mckin**
                          11 + 3374368*mbkin**9*mckin**11 + 495150*mbkin**7*
                          mckin**12 + 495150*mbkin**6*mckin**13 + 1320400*
                          mbkin**7*mckin**13 - 67530*mbkin**5*mckin**14 - 
                          67530*mbkin**4*mckin**15 - 180080*mbkin**5*mckin**15 - 
                          12909*mbkin**3*mckin**16 - 12909*mbkin**2*mckin**17 - 
                          34424*mbkin**3*mckin**17 + 1221*mbkin*mckin**18 + 
                          1221*mckin**19 + 3256*mbkin*mckin**19))/(9*mbkin**
                          24) + (64*(213*mbkin**20 + 72*mbkin**21 + 496*
                          mbkin**20*mckin + 3600*mbkin**18*mckin**2 - 992*
                          mbkin**19*mckin**2 + 10592*mbkin**18*mckin**3 - 
                          19332*mbkin**16*mckin**4 - 15888*mbkin**17*mckin**4 - 
                          35664*mbkin**16*mckin**5 - 127620*mbkin**14*mckin**6 + 
                          47552*mbkin**15*mckin**6 - 387872*mbkin**14*mckin**7 - 
                          111825*mbkin**12*mckin**8 + 484840*mbkin**13*mckin**8 - 
                          783040*mbkin**12*mckin**9 + 134190*mbkin**10*mckin**
                          10 + 939648*mbkin**11*mckin**10 - 581808*mbkin**10*
                          mckin**11 + 223335*mbkin**8*mckin**12 + 678776*mbkin**9*
                          mckin**12 - 83216*mbkin**8*mckin**13 + 51552*mbkin**6*
                          mckin**14 + 95104*mbkin**7*mckin**14 + 42368*mbkin**6*
                          mckin**15 - 16200*mbkin**4*mckin**16 - 47664*mbkin**5*
                          mckin**16 + 4464*mbkin**4*mckin**17 - 2130*mbkin**2*
                          mckin**18 - 4960*mbkin**3*mckin**18 - 720*mbkin**2*
                          mckin**19 + 297*mckin**20 + 792*mbkin*mckin**20)*q_cut)/
                        (9*mbkin**24) - (16*(315*mbkin**18 + 560*mbkin**19 + 
                          280*mbkin**18*mckin + 9309*mbkin**16*mckin**2 - 
                          840*mbkin**17*mckin**2 + 25664*mbkin**16*mckin**3 + 
                          15816*mbkin**14*mckin**4 - 51328*mbkin**15*mckin**4 + 
                          93504*mbkin**14*mckin**5 - 33720*mbkin**12*mckin**6 - 
                          155840*mbkin**13*mckin**6 + 65920*mbkin**12*mckin**7 - 
                          6180*mbkin**10*mckin**8 - 98880*mbkin**11*mckin**8 + 
                          82400*mbkin**10*mckin**9 + 26868*mbkin**8*mckin**10 - 
                          115360*mbkin**9*mckin**10 + 187008*mbkin**8*mckin**11 - 
                          59820*mbkin**6*mckin**12 - 249344*mbkin**7*mckin**12 + 
                          89824*mbkin**6*mckin**13 - 42468*mbkin**4*mckin**14 - 
                          115488*mbkin**5*mckin**14 + 2240*mbkin**4*mckin**15 - 
                          1995*mbkin**2*mckin**16 - 2800*mbkin**3*mckin**16 - 
                          2520*mbkin**2*mckin**17 + 1155*mckin**18 + 3080*mbkin*
                          mckin**18)*q_cut**2)/(9*mbkin**24) - (64*(159*mbkin**16 + 
                          216*mbkin**17 + 208*mbkin**16*mckin + 4110*mbkin**14*
                          mckin**2 - 832*mbkin**15*mckin**2 + 11792*mbkin**14*
                          mckin**3 + 8421*mbkin**12*mckin**4 - 29480*mbkin**13*
                          mckin**4 + 51936*mbkin**12*mckin**5 - 6168*mbkin**10*
                          mckin**6 - 103872*mbkin**11*mckin**6 + 87424*mbkin**10*
                          mckin**7 - 24912*mbkin**8*mckin**8 - 152992*mbkin**9*
                          mckin**8 + 86560*mbkin**8*mckin**9 - 38670*mbkin**6*
                          mckin**10 - 138496*mbkin**7*mckin**10 + 35376*mbkin**6*
                          mckin**11 - 19353*mbkin**4*mckin**12 - 53064*mbkin**5*
                          mckin**12 + 1456*mbkin**4*mckin**13 - 996*mbkin**2*
                          mckin**14 - 2080*mbkin**3*mckin**14 - 576*mbkin**2*
                          mckin**15 + 297*mckin**16 + 792*mbkin*mckin**16)*q_cut**3)/
                        (9*mbkin**24) + (8*(1587*mbkin**14 + 5856*mbkin**15 - 
                          1624*mbkin**14*mckin + 35667*mbkin**12*mckin**2 + 
                          8120*mbkin**13*mckin**2 + 86992*mbkin**12*mckin**3 + 
                          74943*mbkin**10*mckin**4 - 260976*mbkin**11*mckin**4 + 
                          460824*mbkin**10*mckin**5 - 172809*mbkin**8*mckin**6 - 
                          1075256*mbkin**9*mckin**6 + 614432*mbkin**8*mckin**7 - 
                          379269*mbkin**6*mckin**8 - 1228864*mbkin**7*mckin**8 + 
                          217480*mbkin**6*mckin**9 - 150453*mbkin**4*mckin**10 - 
                          391464*mbkin**5*mckin**10 - 9744*mbkin**4*mckin**11 + 
                          2247*mbkin**2*mckin**12 + 16240*mbkin**3*mckin**12 - 
                          10248*mbkin**2*mckin**13 + 6039*mckin**14 + 16104*
                          mbkin*mckin**14)*q_cut**4)/(9*mbkin**24) - 
                       (32*(273*mbkin**12 + 840*mbkin**13 - 112*mbkin**12*
                          mckin + 1914*mbkin**10*mckin**2 + 672*mbkin**11*
                          mckin**2 + 4432*mbkin**10*mckin**3 + 195*mbkin**8*
                          mckin**4 - 15512*mbkin**9*mckin**4 + 16032*mbkin**8*
                          mckin**5 - 12708*mbkin**6*mckin**6 - 42752*mbkin**7*
                          mckin**6 + 8864*mbkin**6*mckin**7 - 7689*mbkin**4*
                          mckin**8 - 19944*mbkin**5*mckin**8 - 560*mbkin**4*
                          mckin**9 + 42*mbkin**2*mckin**10 + 1120*mbkin**3*
                          mckin**10 - 1008*mbkin**2*mckin**11 + 693*mckin**12 + 
                          1848*mbkin*mckin**12)*q_cut**5)/(9*mbkin**24) - 
                       (32*(84*mbkin**10 + 1008*mbkin**11 - 784*mbkin**10*
                          mckin + 696*mbkin**8*mckin**2 + 5488*mbkin**9*mckin**
                          2 - 3632*mbkin**8*mckin**3 + 3405*mbkin**6*mckin**4 + 
                          14528*mbkin**7*mckin**4 - 5448*mbkin**6*mckin**5 + 
                          4953*mbkin**4*mckin**6 + 16344*mbkin**5*mckin**6 - 
                          3136*mbkin**4*mckin**7 + 2625*mbkin**2*mckin**8 + 
                          7840*mbkin**3*mckin**8 - 840*mbkin**2*mckin**9 + 
                          693*mckin**10 + 1848*mbkin*mckin**10)*q_cut**6)/
                        (9*mbkin**24) + (64*(213*mbkin**8 + 840*mbkin**9 - 
                          272*mbkin**8*mckin + 468*mbkin**6*mckin**2 + 2176*
                          mbkin**7*mckin**2 - 928*mbkin**6*mckin**3 + 1260*
                          mbkin**4*mckin**4 + 4176*mbkin**5*mckin**4 - 816*
                          mbkin**4*mckin**5 + 840*mbkin**2*mckin**6 + 2720*
                          mbkin**3*mckin**6 - 480*mbkin**2*mckin**7 + 495*
                          mckin**8 + 1320*mbkin*mckin**8)*q_cut**7)/(9*mbkin**24) - 
                       (4*(1833*mbkin**6 + 5184*mbkin**7 - 296*mbkin**6*mckin + 
                          777*mbkin**4*mckin**2 + 2664*mbkin**5*mckin**2 - 
                          592*mbkin**4*mckin**3 + 381*mbkin**2*mckin**4 + 
                          2960*mbkin**3*mckin**4 - 1944*mbkin**2*mckin**5 + 
                          2673*mckin**6 + 7128*mbkin*mckin**6)*q_cut**8)/
                        (9*mbkin**24) - (224*(mbkin**2 + mckin**2)*(21*mbkin**2 + 
                          72*mbkin**3 - 16*mbkin**2*mckin + 33*mckin**2 + 
                          88*mbkin*mckin**2)*q_cut**9)/(9*mbkin**24) + 
                       (80*(27*mbkin**2 + 80*mbkin**3 - 8*mbkin**2*mckin + 
                          33*mckin**2 + 88*mbkin*mckin**2)*q_cut**10)/(3*mbkin**
                          24) - (704*(3 + 8*mbkin)*q_cut**11)/(9*mbkin**24))))))*
              np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                     2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                      q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
                 mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                     2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))) - 
           144*((640*mckin**8*(mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)*(
                mbkin**2 + 2*mbkin*mckin + mckin**2 - q_cut)*(71*mbkin**12 - 
                174*mbkin**10*mckin**2 - 3723*mbkin**8*mckin**4 - 7468*mbkin**6*
                 mckin**6 - 3723*mbkin**4*mckin**8 - 174*mbkin**2*mckin**10 + 
                71*mckin**12 - 69*mbkin**10*q_cut - 381*mbkin**8*mckin**2*q_cut - 
                810*mbkin**6*mckin**4*q_cut - 810*mbkin**4*mckin**6*q_cut - 
                381*mbkin**2*mckin**8*q_cut - 69*mckin**10*q_cut - 69*mbkin**8*q_cut**2 - 
                446*mbkin**6*mckin**2*q_cut**2 - 818*mbkin**4*mckin**4*q_cut**2 - 
                446*mbkin**2*mckin**6*q_cut**2 - 69*mckin**8*q_cut**2 + 71*mbkin**6*
                 q_cut**3 + 331*mbkin**4*mckin**2*q_cut**3 + 331*mbkin**2*mckin**4*q_cut**3 + 
                71*mckin**6*q_cut**3 - 6*mbkin**4*q_cut**4 + 4*mbkin**2*mckin**2*q_cut**4 - 
                6*mckin**4*q_cut**4 - 6*mbkin**2*q_cut**5 - 6*mckin**2*q_cut**5 + 8*q_cut**6)*(
                3*mbkin**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 
                8*mbkin**4*mckin*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                6*mbkin**2*mckin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                8*mbkin**3*mckin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                8*mbkin**2*mckin**3*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 
                3*mckin**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 
                8*mbkin*mckin**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                3*mbkin**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                8*mbkin**2*mckin*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                3*mckin**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                8*mbkin*mckin**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))*
               np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                      2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                       q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/
              (mbkin**22*(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                  (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mckin**2 + q_cut + 
                mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))) + 
             ((-160*mckin**6*(3*mbkin**2 + 8*mbkin**2*mckin - 3*mckin**2 - 
                  8*mbkin*mckin**2)*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*(71*mbkin**12 - 
                  174*mbkin**10*mckin**2 - 3723*mbkin**8*mckin**4 - 7468*mbkin**6*
                   mckin**6 - 3723*mbkin**4*mckin**8 - 174*mbkin**2*mckin**10 + 
                  71*mckin**12 - 69*mbkin**10*q_cut - 381*mbkin**8*mckin**2*q_cut - 
                  810*mbkin**6*mckin**4*q_cut - 810*mbkin**4*mckin**6*q_cut - 
                  381*mbkin**2*mckin**8*q_cut - 69*mckin**10*q_cut - 69*mbkin**8*q_cut**2 - 
                  446*mbkin**6*mckin**2*q_cut**2 - 818*mbkin**4*mckin**4*q_cut**2 - 
                  446*mbkin**2*mckin**6*q_cut**2 - 69*mckin**8*q_cut**2 + 71*mbkin**6*
                   q_cut**3 + 331*mbkin**4*mckin**2*q_cut**3 + 331*mbkin**2*mckin**4*
                   q_cut**3 + 71*mckin**6*q_cut**3 - 6*mbkin**4*q_cut**4 + 4*mbkin**2*
                   mckin**2*q_cut**4 - 6*mckin**4*q_cut**4 - 6*mbkin**2*q_cut**5 - 
                  6*mckin**2*q_cut**5 + 8*q_cut**6))/mbkin**26 + (mckin**4*
                 (-180*mckin**4*((-8*(mbkin - mckin)**2*(mbkin + mckin)*
                      (3*mbkin + 3*mckin + 8*mbkin*mckin))/(9*mbkin**6) + 
                    (16*(4*mbkin**3 - 4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*
                        mckin**2)*q_cut)/(9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/
                     (9*mbkin**6))*((-1 + mckin**2/mbkin**2)**2*(71 - 
                      (174*mckin**2)/mbkin**2 - (3723*mckin**4)/mbkin**4 - 
                      (7468*mckin**6)/mbkin**6 - (3723*mckin**8)/mbkin**8 - 
                      (174*mckin**10)/mbkin**10 + (71*mckin**12)/mbkin**12) + 
                    ((-211 - (37*mckin**2)/mbkin**2 + (7677*mckin**4)/mbkin**4 + 
                       (22811*mckin**6)/mbkin**6 + (22811*mckin**8)/mbkin**8 + 
                       (7677*mckin**10)/mbkin**10 - (37*mckin**12)/mbkin**12 - 
                       (211*mckin**14)/mbkin**14)*q_cut)/mbkin**2 + 
                    (2*(70 + (209*mckin**2)/mbkin**2 - (668*mckin**4)/mbkin**4 - 
                       (1742*mckin**6)/mbkin**6 - (668*mckin**8)/mbkin**8 + 
                       (209*mckin**10)/mbkin**10 + (70*mckin**12)/mbkin**12)*
                      q_cut**2)/mbkin**4 + (2*(70 + (419*mckin**2)/mbkin**2 + 
                       (729*mckin**4)/mbkin**4 + (729*mckin**6)/mbkin**6 + 
                       (419*mckin**8)/mbkin**8 + (70*mckin**10)/mbkin**10)*q_cut**3)/
                     mbkin**6 - ((217 + (1234*mckin**2)/mbkin**2 + 
                       (2162*mckin**4)/mbkin**4 + (1234*mckin**6)/mbkin**6 + 
                       (217*mckin**8)/mbkin**8)*q_cut**4)/mbkin**8 + 
                    (11*(7 + (31*mckin**2)/mbkin**2 + (31*mckin**4)/mbkin**4 + 
                       (7*mckin**6)/mbkin**6)*q_cut**5)/mbkin**10 + 
                    (2*(7 + (6*mckin**2)/mbkin**2 + (7*mckin**4)/mbkin**4)*q_cut**6)/
                     mbkin**12 - (22*(mbkin**2 + mckin**2)*q_cut**7)/mbkin**16 + 
                    (8*q_cut**8)/mbkin**16) + ((-1 + mckin**2/mbkin**2)**2 - 
                    (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
                   ((64*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(-40 - 
                          (530*mckin**2)/mbkin**2 + (1237*mckin**4)/mbkin**4 + 
                          (117101*mckin**6)/mbkin**6 + (301726*mckin**8)/
                          mbkin**8 + (144226*mckin**10)/mbkin**10 - 
                          (5749*mckin**12)/mbkin**12 + (619*mckin**14)/mbkin**
                          14 + (850*mckin**16)/mbkin**16)) + ((-110 - 
                          (2180*mckin**2)/mbkin**2 - (8043*mckin**4)/mbkin**4 + 
                          (231598*mckin**6)/mbkin**6 + (844217*mckin**8)/
                          mbkin**8 + (919012*mckin**10)/mbkin**10 + (277427*
                          mckin**12)/mbkin**12 - (28230*mckin**14)/mbkin**14 + 
                          (1469*mckin**16)/mbkin**16 + (2600*mckin**18)/mbkin**
                          18)*q_cut)/mbkin**2 - (2*(-35 - (870*mckin**2)/mbkin**2 - 
                          (7720*mckin**4)/mbkin**4 + (19403*mckin**6)/mbkin**6 + 
                          (68795*mckin**8)/mbkin**8 + (21911*mckin**10)/mbkin**
                          10 - (10719*mckin**12)/mbkin**12 + (1600*mckin**14)/
                          mbkin**14 + (875*mckin**16)/mbkin**16)*q_cut**2)/mbkin**4 + 
                       (2*(35 + (975*mckin**2)/mbkin**2 + (11590*mckin**4)/
                          mbkin**4 + (21792*mckin**6)/mbkin**6 + (19921*mckin**8)/
                          mbkin**8 + (9944*mckin**10)/mbkin**10 - (2650*mckin**
                          12)/mbkin**12 - (875*mckin**14)/mbkin**14)*q_cut**3)/
                        mbkin**6 + ((-140 - (2810*mckin**2)/mbkin**2 - 
                          (35835*mckin**4)/mbkin**4 - (69801*mckin**6)/mbkin**6 - 
                          (30887*mckin**8)/mbkin**8 + (8455*mckin**10)/mbkin**
                          10 + (2450*mckin**12)/mbkin**12)*q_cut**4)/mbkin**8 + 
                       ((70 + (900*mckin**2)/mbkin**2 + (10637*mckin**4)/
                          mbkin**4 + (9736*mckin**6)/mbkin**6 - (2311*mckin**8)/
                          mbkin**8 - (700*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 + 
                       ((70 + (60*mckin**2)/mbkin**2 + (308*mckin**4)/mbkin**4 + 
                          (76*mckin**6)/mbkin**6 + (350*mckin**8)/mbkin**8)*q_cut**6)/
                        mbkin**12 - (10*(11 + (11*mckin**2)/mbkin**2 + 
                          (27*mckin**4)/mbkin**4 + (55*mckin**6)/mbkin**6)*q_cut**7)/
                        mbkin**14 + (40*(mbkin**4 + 5*mckin**4)*q_cut**8)/mbkin**20))/
                     3 - 180*((8*mckin**2*(3 + 8*mckin)*((-1 + mckin**2/mbkin**
                          2)**2*(71 - (174*mckin**2)/mbkin**2 - (3723*mckin**4)/
                          mbkin**4 - (7468*mckin**6)/mbkin**6 - (3723*mckin**8)/
                          mbkin**8 - (174*mckin**10)/mbkin**10 + (71*mckin**12)/
                          mbkin**12) + ((-211 - (37*mckin**2)/mbkin**2 + 
                          (7677*mckin**4)/mbkin**4 + (22811*mckin**6)/mbkin**6 + 
                          (22811*mckin**8)/mbkin**8 + (7677*mckin**10)/mbkin**
                          10 - (37*mckin**12)/mbkin**12 - (211*mckin**14)/
                          mbkin**14)*q_cut)/mbkin**2 + (2*(70 + (209*mckin**2)/
                          mbkin**2 - (668*mckin**4)/mbkin**4 - (1742*mckin**6)/
                          mbkin**6 - (668*mckin**8)/mbkin**8 + (209*mckin**10)/
                          mbkin**10 + (70*mckin**12)/mbkin**12)*q_cut**2)/mbkin**4 + 
                         (2*(70 + (419*mckin**2)/mbkin**2 + (729*mckin**4)/
                          mbkin**4 + (729*mckin**6)/mbkin**6 + (419*mckin**8)/
                          mbkin**8 + (70*mckin**10)/mbkin**10)*q_cut**3)/mbkin**6 - 
                         ((217 + (1234*mckin**2)/mbkin**2 + (2162*mckin**4)/
                          mbkin**4 + (1234*mckin**6)/mbkin**6 + (217*mckin**8)/
                          mbkin**8)*q_cut**4)/mbkin**8 + (11*(7 + (31*mckin**2)/
                          mbkin**2 + (31*mckin**4)/mbkin**4 + (7*mckin**6)/
                          mbkin**6)*q_cut**5)/mbkin**10 + (2*(7 + (6*mckin**2)/
                          mbkin**2 + (7*mckin**4)/mbkin**4)*q_cut**6)/mbkin**12 - 
                         (22*(mbkin**2 + mckin**2)*q_cut**7)/mbkin**16 + (8*q_cut**8)/
                          mbkin**16))/9 + mckin**4*((-16*(mbkin - mckin)**2*
                          (mbkin + mckin)*(237*mbkin**13 + 237*mbkin**12*
                          mckin + 632*mbkin**13*mckin + 5193*mbkin**11*mckin**
                          2 + 5193*mbkin**10*mckin**3 + 13848*mbkin**11*mckin**
                          3 + 5634*mbkin**9*mckin**4 + 5634*mbkin**8*mckin**5 + 
                          15024*mbkin**9*mckin**5 - 16836*mbkin**7*mckin**6 - 
                          16836*mbkin**6*mckin**7 - 44896*mbkin**7*mckin**7 - 
                          16101*mbkin**5*mckin**8 - 16101*mbkin**4*mckin**9 - 
                          42936*mbkin**5*mckin**9 - 1233*mbkin**3*mckin**10 - 
                          1233*mbkin**2*mckin**11 - 3288*mbkin**3*mckin**11 + 
                          426*mbkin*mckin**12 + 426*mckin**13 + 1136*mbkin*
                          mckin**13))/(9*mbkin**18) + (8*(261*mbkin**14 + 
                          844*mbkin**15 - 148*mbkin**14*mckin + 23142*mbkin**12*
                          mckin**2 + 296*mbkin**13*mckin**2 + 61416*mbkin**12*
                          mckin**3 + 68103*mbkin**10*mckin**4 - 92124*mbkin**11*
                          mckin**4 + 273732*mbkin**10*mckin**5 - 364976*mbkin**9*
                          mckin**6 + 364976*mbkin**8*mckin**7 - 113505*mbkin**6*
                          mckin**8 - 456220*mbkin**7*mckin**8 + 153540*mbkin**6*
                          mckin**9 - 69426*mbkin**4*mckin**10 - 184248*mbkin**5*
                          mckin**10 - 888*mbkin**4*mckin**11 - 1827*mbkin**2*
                          mckin**12 + 1036*mbkin**3*mckin**12 - 5908*mbkin**2*
                          mckin**13 + 2532*mckin**14 + 6752*mbkin*mckin**14)*q_cut)/
                         (9*mbkin**18) - (8*(-207*mbkin**12 + 1120*mbkin**13 - 
                          1672*mbkin**12*mckin + 5889*mbkin**10*mckin**2 + 
                          5016*mbkin**11*mckin**2 + 10688*mbkin**10*mckin**3 + 
                          7662*mbkin**8*mckin**4 - 21376*mbkin**9*mckin**4 + 
                          41808*mbkin**8*mckin**5 - 18114*mbkin**6*mckin**6 - 
                          69680*mbkin**7*mckin**6 + 21376*mbkin**6*mckin**7 - 
                          15159*mbkin**4*mckin**8 - 32064*mbkin**5*mckin**8 - 
                          8360*mbkin**4*mckin**9 + 3129*mbkin**2*mckin**10 + 
                          11704*mbkin**3*mckin**10 - 3360*mbkin**2*mckin**11 + 
                          1680*mckin**12 + 4480*mbkin*mckin**12)*q_cut**2)/
                         (9*mbkin**18) - (8*(-627*mbkin**10 + 1680*mbkin**11 - 
                          3352*mbkin**10*mckin + 654*mbkin**8*mckin**2 + 
                          13408*mbkin**9*mckin**2 - 11664*mbkin**8*mckin**3 + 
                          4374*mbkin**6*mckin**4 + 29160*mbkin**7*mckin**4 - 
                          17496*mbkin**6*mckin**5 + 8094*mbkin**4*mckin**6 + 
                          34992*mbkin**5*mckin**6 - 13408*mbkin**4*mckin**7 + 
                          7749*mbkin**2*mckin**8 + 23464*mbkin**3*mckin**8 - 
                          2800*mbkin**2*mckin**9 + 1680*mckin**10 + 4480*mbkin*
                          mckin**10)*q_cut**3)/(9*mbkin**18) + (8*(-549*mbkin**8 + 
                          3472*mbkin**9 - 4936*mbkin**8*mckin + 2769*mbkin**6*
                          mckin**2 + 24680*mbkin**7*mckin**2 - 17296*mbkin**6*
                          mckin**3 + 13905*mbkin**4*mckin**4 + 51888*mbkin**5*
                          mckin**4 - 14808*mbkin**4*mckin**5 + 11655*mbkin**2*
                          mckin**6 + 34552*mbkin**3*mckin**6 - 3472*mbkin**2*
                          mckin**7 + 2604*mckin**8 + 6944*mbkin*mckin**8)*q_cut**4)/
                         (9*mbkin**18) - (176*(3*mbkin**6 + 70*mbkin**7 - 
                          62*mbkin**6*mckin + 93*mbkin**4*mckin**2 + 372*mbkin**5*
                          mckin**2 - 124*mbkin**4*mckin**3 + 147*mbkin**2*mckin**
                          4 + 434*mbkin**3*mckin**4 - 42*mbkin**2*mckin**5 + 
                          42*mckin**6 + 112*mbkin*mckin**6)*q_cut**5)/(9*mbkin**
                          18) - (32*(27*mbkin**4 + 84*mbkin**5 - 12*mbkin**4*
                          mckin + 21*mbkin**2*mckin**2 + 84*mbkin**3*mckin**2 - 
                          28*mbkin**2*mckin**3 + 42*mckin**4 + 112*mbkin*mckin**
                          4)*q_cut**6)/(9*mbkin**18) + (176*(9*mbkin**2 + 
                          28*mbkin**3 - 4*mbkin**2*mckin + 12*mckin**2 + 
                          32*mbkin*mckin**2)*q_cut**7)/(9*mbkin**18) - 
                        (256*(3 + 8*mbkin)*q_cut**8)/(9*mbkin**18))))))/mbkin**4)*
              np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2) - 
           60480*((-960*mckin**12*(mbkin**6 + 5*mbkin**4*mckin**2 + 5*mbkin**2*
                 mckin**4 + mckin**6)*(mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)*(
                mbkin**2 + 2*mbkin*mckin + mckin**2 - q_cut)*(3*mbkin**4 + 
                8*mbkin**4*mckin - 6*mbkin**2*mckin**2 - 8*mbkin**3*mckin**2 - 
                8*mbkin**2*mckin**3 + 3*mckin**4 + 8*mbkin*mckin**4 - 
                3*mbkin**2*q_cut - 8*mbkin**2*mckin*q_cut - 3*mckin**2*q_cut - 
                8*mbkin*mckin**2*q_cut)*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                        q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                   mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                       mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                      mbkin**4)))**2)/(mbkin**20*(mbkin**2 + mckin**2 - q_cut + 
                mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - 
                mckin**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))) + ((-16*mckin**8*(12*mbkin**9 + 104*mbkin**7*
                   mckin**2 - 90*mbkin**8*mckin**2 - 240*mbkin**8*mckin**3 + 
                  700*mbkin**5*mckin**4 - 675*mbkin**6*mckin**4 - 1800*mbkin**6*
                   mckin**5 + 840*mbkin**3*mckin**6 - 675*mbkin**4*mckin**6 + 
                  600*mbkin**5*mckin**6 - 2400*mbkin**4*mckin**7 + 120*mbkin*
                   mckin**8 + 225*mbkin**2*mckin**8 + 1200*mbkin**3*mckin**8 - 
                  600*mbkin**2*mckin**9 + 135*mckin**10 + 360*mbkin*mckin**10)*
                 ((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2))/(3*mbkin**16) - 
               (80*mckin**10*(mbkin**6 + 5*mbkin**4*mckin**2 + 5*mbkin**2*
                   mckin**4 + mckin**6)*((mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**
                  (3/2)*(-12*mbkin**6 - 32*mbkin**6*mckin + 45*mbkin**4*
                   mckin**2 + 32*mbkin**5*mckin**2 + 88*mbkin**4*mckin**3 - 
                  54*mbkin**2*mckin**4 - 88*mbkin**3*mckin**4 - 56*mbkin**2*
                   mckin**5 + 21*mckin**6 + 56*mbkin*mckin**6 + 24*mbkin**4*q_cut + 
                  64*mbkin**4*mckin*q_cut - 88*mbkin**3*mckin**2*q_cut + 88*mbkin**2*
                   mckin**3*q_cut - 42*mckin**4*q_cut - 112*mbkin*mckin**4*q_cut - 
                  12*mbkin**2*q_cut**2 - 32*mbkin**2*mckin*q_cut**2 + 21*mckin**2*q_cut**2 + 
                  56*mbkin*mckin**2*q_cut**2))/(mbkin**16*(mbkin**4 - 2*mbkin**2*
                   mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)))*
              np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)))/
         (((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
             2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
               7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
               mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
            (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*
                  np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                      q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                 mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                     mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                    mbkin**4))))/mbkin**4)**3) + 
        ((180*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mckin**2 - 
              7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
              mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)**2*(mbkin**12 - 
             34*mbkin**10*mckin**2 - 1133*mbkin**8*mckin**4 - 2708*mbkin**6*
              mckin**6 - 1133*mbkin**4*mckin**8 - 34*mbkin**2*mckin**10 + 
             mckin**12 + mbkin**10*q_cut - 31*mbkin**8*mckin**2*q_cut - 
             390*mbkin**6*mckin**4*q_cut - 390*mbkin**4*mckin**6*q_cut - 
             31*mbkin**2*mckin**8*q_cut + mckin**10*q_cut + mbkin**8*q_cut**2 - 
             26*mbkin**6*mckin**2*q_cut**2 - 118*mbkin**4*mckin**4*q_cut**2 - 
             26*mbkin**2*mckin**6*q_cut**2 + mckin**8*q_cut**2 + mbkin**6*q_cut**3 - 
             19*mbkin**4*mckin**2*q_cut**3 - 19*mbkin**2*mckin**4*q_cut**3 + 
             mckin**6*q_cut**3 - 6*mbkin**4*q_cut**4 + 4*mbkin**2*mckin**2*q_cut**4 - 
             6*mckin**4*q_cut**4 - 6*mbkin**2*q_cut**5 - 6*mckin**2*q_cut**5 + 8*q_cut**6))/
           mbkin**32 - (2160*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
              2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*
            np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)*(37*mbkin**18 - 397*mbkin**16*
              mckin**2 - 2854*mbkin**14*mckin**4 + 18134*mbkin**12*mckin**6 + 
             75800*mbkin**10*mckin**8 + 75800*mbkin**8*mckin**10 + 
             18134*mbkin**6*mckin**12 - 2854*mbkin**4*mckin**14 - 
             397*mbkin**2*mckin**16 + 37*mckin**18 - 70*mbkin**16*q_cut + 
             132*mbkin**14*mckin**2*q_cut + 4424*mbkin**12*mckin**4*q_cut + 
             15500*mbkin**10*mckin**6*q_cut + 20508*mbkin**8*mckin**8*q_cut + 
             15500*mbkin**6*mckin**10*q_cut + 4424*mbkin**4*mckin**12*q_cut + 
             132*mbkin**2*mckin**14*q_cut - 70*mckin**16*q_cut - 37*mbkin**14*q_cut**2 + 
             307*mbkin**12*mckin**2*q_cut**2 + 6201*mbkin**10*mckin**4*q_cut**2 + 
             18225*mbkin**8*mckin**6*q_cut**2 + 18225*mbkin**6*mckin**8*q_cut**2 + 
             6201*mbkin**4*mckin**10*q_cut**2 + 307*mbkin**2*mckin**12*q_cut**2 - 
             37*mckin**14*q_cut**2 + 140*mbkin**12*q_cut**3 + 272*mbkin**10*mckin**2*
              q_cut**3 - 2796*mbkin**8*mckin**4*q_cut**3 - 7136*mbkin**6*mckin**6*q_cut**3 - 
             2796*mbkin**4*mckin**8*q_cut**3 + 272*mbkin**2*mckin**10*q_cut**3 + 
             140*mckin**12*q_cut**3 - 49*mbkin**10*q_cut**4 + 13*mbkin**8*mckin**2*q_cut**4 - 
             300*mbkin**6*mckin**4*q_cut**4 - 300*mbkin**4*mckin**6*q_cut**4 + 
             13*mbkin**2*mckin**8*q_cut**4 - 49*mckin**10*q_cut**4 - 70*mbkin**8*q_cut**5 - 
             372*mbkin**6*mckin**2*q_cut**5 - 668*mbkin**4*mckin**4*q_cut**5 - 
             372*mbkin**2*mckin**6*q_cut**5 - 70*mckin**8*q_cut**5 + 77*mbkin**6*q_cut**6 + 
             41*mbkin**4*mckin**2*q_cut**6 + 41*mbkin**2*mckin**4*q_cut**6 + 
             77*mckin**6*q_cut**6 - 16*mbkin**4*q_cut**7 + 32*mbkin**2*mckin**2*q_cut**7 - 
             16*mckin**4*q_cut**7 - 28*mbkin**2*q_cut**8 - 28*mckin**2*q_cut**8 + 16*q_cut**9)*
            np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                    mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                  mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**26 + 
          (25920*mckin**8*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
              2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*(71*mbkin**12 - 
             174*mbkin**10*mckin**2 - 3723*mbkin**8*mckin**4 - 7468*mbkin**6*
              mckin**6 - 3723*mbkin**4*mckin**8 - 174*mbkin**2*mckin**10 + 
             71*mckin**12 - 69*mbkin**10*q_cut - 381*mbkin**8*mckin**2*q_cut - 
             810*mbkin**6*mckin**4*q_cut - 810*mbkin**4*mckin**6*q_cut - 
             381*mbkin**2*mckin**8*q_cut - 69*mckin**10*q_cut - 69*mbkin**8*q_cut**2 - 
             446*mbkin**6*mckin**2*q_cut**2 - 818*mbkin**4*mckin**4*q_cut**2 - 
             446*mbkin**2*mckin**6*q_cut**2 - 69*mckin**8*q_cut**2 + 71*mbkin**6*q_cut**3 + 
             331*mbkin**4*mckin**2*q_cut**3 + 331*mbkin**2*mckin**4*q_cut**3 + 
             71*mckin**6*q_cut**3 - 6*mbkin**4*q_cut**4 + 4*mbkin**2*mckin**2*q_cut**4 - 
             6*mckin**4*q_cut**4 - 6*mbkin**2*q_cut**5 - 6*mckin**2*q_cut**5 + 8*q_cut**6)*
            np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**24 - 
          (10886400*mckin**12*(mbkin**6 + 5*mbkin**4*mckin**2 + 
             5*mbkin**2*mckin**4 + mckin**6)*((mbkin**4 - 2*mbkin**2*mckin**2 + 
               mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**14)*
         (((4*(3 + 8*mbkin))/(9*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)) - 
            (3*mbkin**2*((-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 
                  3*mckin + 8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 
                  4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/
                (9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6)))/
             (2*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*((-1 + mckin**2/mbkin**2)**
                2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)))/
           ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*
                 mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
             (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                      2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - 
                  q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/
              mbkin**4)**3 - (3*mbkin**2*((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                  mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(
                (-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 3*mckin + 
                   8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 4*mbkin**2*
                    mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/(9*mbkin**6) - 
                (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*(mbkin**6 - 7*mbkin**4*
                 mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/
              (2*mbkin**6*((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*
                  q_cut)/mbkin**4 + q_cut**2/mbkin**4)) - 
             (4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(21*mbkin**6 + 56*mbkin**6*
                 mckin + 21*mbkin**4*mckin**2 - 56*mbkin**5*mckin**2 + 
                112*mbkin**4*mckin**3 - 51*mbkin**2*mckin**4 - 112*mbkin**3*
                 mckin**4 - 24*mbkin**2*mckin**5 + 9*mckin**6 + 24*mbkin*
                 mckin**6 - 3*mbkin**4*q_cut - 8*mbkin**5*q_cut + 6*mbkin**2*mckin**2*
                 q_cut + 16*mbkin**2*mckin**3*q_cut - 9*mckin**4*q_cut - 24*mbkin*mckin**4*
                 q_cut - 3*mbkin**2*q_cut**2 - 16*mbkin**3*q_cut**2 + 8*mbkin**2*mckin*
                 q_cut**2 - 9*mckin**2*q_cut**2 - 24*mbkin*mckin**2*q_cut**2 + 9*q_cut**3 + 
                24*mbkin*q_cut**3))/(9*mbkin**8) - 12*((-16*mckin**4*
                 (3*mbkin**6*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 
                  8*mbkin**6*mckin*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                      mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                     mbkin**4) - 6*mbkin**4*mckin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4) - 8*mbkin**5*mckin**2*np.sqrt(0j + (mbkin**4 - 
                      2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                       q_cut + q_cut**2)/mbkin**4) - 8*mbkin**4*mckin**3*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                      2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 3*mbkin**2*mckin**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 8*mbkin**3*mckin**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mbkin**4*q_cut*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**4*mckin*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mbkin**2*mckin**2*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**3*mckin**2*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))/(9*mbkin**4*
                 (mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)*(mbkin**2 + 
                  2*mbkin*mckin + mckin**2 - q_cut)*(mbkin**2 + mckin**2 - q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))*
                 (-mbkin**2 - mckin**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4))) + ((-8*(3 + 8*mbkin)*mckin**4)/
                  (9*mbkin**6) + (8*mckin**2*(3 + 8*mckin))/(9*mbkin**4))*
                np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                       2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                        q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                       2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))))/
           (((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*
                  mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                 mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
              (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                       2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - 
                   q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                       2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**
                4)**4))))/2520 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

def q2moment4Kin(q_cut, mbkin, mckin, mus, api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
 res = ( 
    (mbkin**4*((18*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)**2*(3*mbkin**14 - 
           141*mbkin**12*mckin**2 - 7785*mbkin**10*mckin**4 - 
           33657*mbkin**8*mckin**6 - 33657*mbkin**6*mckin**8 - 
           7785*mbkin**4*mckin**10 - 141*mbkin**2*mckin**12 + 3*mckin**14 + 
           3*mbkin**12*q_cut - 132*mbkin**10*mckin**2*q_cut - 3153*mbkin**8*mckin**4*
            q_cut - 7296*mbkin**6*mckin**6*q_cut - 3153*mbkin**4*mckin**8*q_cut - 
           132*mbkin**2*mckin**10*q_cut + 3*mckin**12*q_cut + 3*mbkin**10*q_cut**2 - 
           117*mbkin**8*mckin**2*q_cut**2 - 1272*mbkin**6*mckin**4*q_cut**2 - 
           1272*mbkin**4*mckin**6*q_cut**2 - 117*mbkin**2*mckin**8*q_cut**2 + 
           3*mckin**10*q_cut**2 + 3*mbkin**8*q_cut**3 - 96*mbkin**6*mckin**2*q_cut**3 - 
           408*mbkin**4*mckin**4*q_cut**3 - 96*mbkin**2*mckin**6*q_cut**3 + 
           3*mckin**8*q_cut**3 + 3*mbkin**6*q_cut**4 - 69*mbkin**4*mckin**2*q_cut**4 - 
           69*mbkin**2*mckin**4*q_cut**4 + 3*mckin**6*q_cut**4 - 25*mbkin**4*q_cut**5 + 
           20*mbkin**2*mckin**2*q_cut**5 - 25*mckin**4*q_cut**5 - 25*mbkin**2*q_cut**6 - 
           25*mckin**2*q_cut**6 + 35*q_cut**7))/mbkin**34 - 
        (432*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
             mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*
          (108*mbkin**20 - 792*mbkin**18*mckin**2 - 13329*mbkin**16*mckin**4 + 
           40518*mbkin**14*mckin**6 + 387441*mbkin**12*mckin**8 + 
           668988*mbkin**10*mckin**10 + 387441*mbkin**8*mckin**12 + 
           40518*mbkin**6*mckin**14 - 13329*mbkin**4*mckin**16 - 
           792*mbkin**2*mckin**18 + 108*mckin**20 - 210*mbkin**18*q_cut - 
           222*mbkin**16*mckin**2*q_cut + 15402*mbkin**14*mckin**4*q_cut + 
           81210*mbkin**12*mckin**6*q_cut + 153300*mbkin**10*mckin**8*q_cut + 
           153300*mbkin**8*mckin**10*q_cut + 81210*mbkin**6*mckin**12*q_cut + 
           15402*mbkin**4*mckin**14*q_cut - 222*mbkin**2*mckin**16*q_cut - 
           210*mckin**18*q_cut - 108*mbkin**16*q_cut**2 + 552*mbkin**14*mckin**2*q_cut**2 + 
           22257*mbkin**12*mckin**4*q_cut**2 + 101844*mbkin**10*mckin**6*q_cut**2 + 
           158394*mbkin**8*mckin**8*q_cut**2 + 101844*mbkin**6*mckin**10*q_cut**2 + 
           22257*mbkin**4*mckin**12*q_cut**2 + 552*mbkin**2*mckin**14*q_cut**2 - 
           108*mckin**16*q_cut**2 + 420*mbkin**14*q_cut**3 + 2088*mbkin**12*mckin**2*
            q_cut**3 - 8028*mbkin**10*mckin**4*q_cut**3 - 43584*mbkin**8*mckin**6*q_cut**3 - 
           43584*mbkin**6*mckin**8*q_cut**3 - 8028*mbkin**4*mckin**10*q_cut**3 + 
           2088*mbkin**2*mckin**12*q_cut**3 + 420*mckin**14*q_cut**3 - 
           105*mbkin**12*q_cut**4 - 642*mbkin**10*mckin**2*q_cut**4 - 
           966*mbkin**8*mckin**4*q_cut**4 - 2118*mbkin**6*mckin**6*q_cut**4 - 
           966*mbkin**4*mckin**8*q_cut**4 - 642*mbkin**2*mckin**10*q_cut**4 - 
           105*mckin**12*q_cut**4 - 238*mbkin**10*q_cut**5 - 1650*mbkin**8*mckin**2*
            q_cut**5 - 5522*mbkin**6*mckin**4*q_cut**5 - 5522*mbkin**4*mckin**6*q_cut**5 - 
           1650*mbkin**2*mckin**8*q_cut**5 - 238*mckin**10*q_cut**5 + 105*mbkin**8*q_cut**6 + 
           940*mbkin**6*mckin**2*q_cut**6 + 1705*mbkin**4*mckin**4*q_cut**6 + 
           940*mbkin**2*mckin**6*q_cut**6 + 105*mckin**8*q_cut**6 + 88*mbkin**6*q_cut**7 - 
           284*mbkin**4*mckin**2*q_cut**7 - 284*mbkin**2*mckin**4*q_cut**7 + 
           88*mckin**6*q_cut**7 - 35*mbkin**4*q_cut**8 + 70*mbkin**2*mckin**2*q_cut**8 - 
           35*mckin**4*q_cut**8 - 60*mbkin**2*q_cut**9 - 60*mckin**2*q_cut**9 + 35*q_cut**10)*
          np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                  mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
              np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**28 + 
        (2592*mckin**8*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)**2*(423*mbkin**14 + 279*mbkin**12*mckin**2 - 
           27945*mbkin**10*mckin**4 - 97497*mbkin**8*mckin**6 - 
           97497*mbkin**6*mckin**8 - 27945*mbkin**4*mckin**10 + 
           279*mbkin**2*mckin**12 + 423*mckin**14 - 417*mbkin**12*q_cut - 
           3492*mbkin**10*mckin**2*q_cut - 9873*mbkin**8*mckin**4*q_cut - 
           14016*mbkin**6*mckin**6*q_cut - 9873*mbkin**4*mckin**8*q_cut - 
           3492*mbkin**2*mckin**10*q_cut - 417*mckin**12*q_cut - 417*mbkin**10*q_cut**2 - 
           3897*mbkin**8*mckin**2*q_cut**2 - 10932*mbkin**6*mckin**4*q_cut**2 - 
           10932*mbkin**4*mckin**6*q_cut**2 - 3897*mbkin**2*mckin**8*q_cut**2 - 
           417*mckin**10*q_cut**2 + 423*mbkin**8*q_cut**3 + 3264*mbkin**6*mckin**2*q_cut**3 + 
           5892*mbkin**4*mckin**4*q_cut**3 + 3264*mbkin**2*mckin**6*q_cut**3 + 
           423*mckin**8*q_cut**3 + 3*mbkin**6*q_cut**4 - 69*mbkin**4*mckin**2*q_cut**4 - 
           69*mbkin**2*mckin**4*q_cut**4 + 3*mckin**6*q_cut**4 - 25*mbkin**4*q_cut**5 + 
           20*mbkin**2*mckin**2*q_cut**5 - 25*mckin**4*q_cut**5 - 25*mbkin**2*q_cut**6 - 
           25*mckin**2*q_cut**6 + 35*q_cut**7)*np.log((mbkin**2 + mckin**2 - q_cut - 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/
             (mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                 mbkin**4)))**2)/mbkin**26 - 
        (6531840*mckin**12*(mbkin**8 + 8*mbkin**6*mckin**2 + 15*mbkin**4*mckin**4 + 
           8*mbkin**2*mckin**6 + mckin**8)*((mbkin**4 - 2*mbkin**2*mckin**2 + 
             mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**16))/
      (1260*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
          2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
        3) + (mbkin**4*(((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
           mbkin**4 + q_cut**2/mbkin**4)*(-24*mbkin**2*muG*
           ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
             q_cut**2/mbkin**4)**2*(-((1 + mckin**2/mbkin**2)**2*(-33 - (2131*mckin**2)/
                mbkin**2 + (30294*mckin**4)/mbkin**4 - (78660*mckin**6)/mbkin**6 - 
               (303336*mckin**8)/mbkin**8 + (1430424*mckin**10)/mbkin**10 + 
               (1908216*mckin**12)/mbkin**12 - (417984*mckin**14)/mbkin**14 - 
               (84555*mckin**16)/mbkin**16 + (12471*mckin**18)/mbkin**18 + 
               (94*mckin**20)/mbkin**20)) + ((-42 - (3763*mckin**2)/mbkin**2 + 
               (26558*mckin**4)/mbkin**4 + (44403*mckin**6)/mbkin**6 - 
               (669144*mckin**8)/mbkin**8 - (1235460*mckin**10)/mbkin**10 - 
               (1259364*mckin**12)/mbkin**12 - (1346004*mckin**14)/mbkin**14 - 
               (612774*mckin**16)/mbkin**16 + (45903*mckin**18)/mbkin**18 + 
               (19966*mckin**20)/mbkin**20 + (121*mckin**22)/mbkin**22)*q_cut)/
             mbkin**2 + ((-75 - (3521*mckin**2)/mbkin**2 + (35220*mckin**4)/
                mbkin**4 - (13530*mckin**6)/mbkin**6 - (1006434*mckin**8)/
                mbkin**8 - (2777886*mckin**10)/mbkin**10 - (2724486*mckin**12)/
                mbkin**12 - (998682*mckin**14)/mbkin**14 - (17175*mckin**16)/
                mbkin**16 + (21963*mckin**18)/mbkin**18 + (206*mckin**20)/
                mbkin**20)*q_cut**2)/mbkin**4 + ((84 + (8007*mckin**2)/mbkin**2 - 
               (44916*mckin**4)/mbkin**4 - (259494*mckin**6)/mbkin**6 + 
               (280412*mckin**8)/mbkin**8 + (873934*mckin**10)/mbkin**10 + 
               (232148*mckin**12)/mbkin**12 - (222734*mckin**14)/mbkin**14 - 
               (42208*mckin**16)/mbkin**16 - (233*mckin**18)/mbkin**18)*q_cut**3)/
             mbkin**6 - (2*(-21 - (375*mckin**2)/mbkin**2 + (4350*mckin**4)/
                mbkin**4 + (858*mckin**6)/mbkin**6 + (40570*mckin**8)/mbkin**8 + 
               (22642*mckin**10)/mbkin**10 + (7617*mckin**12)/mbkin**12 + 
               (3239*mckin**14)/mbkin**14 + (56*mckin**16)/mbkin**16)*q_cut**4)/
             mbkin**8 + (2*mckin**2*(-2439 + (10338*mckin**2)/mbkin**2 + 
               (103080*mckin**4)/mbkin**4 + (158590*mckin**6)/mbkin**6 + 
               (93216*mckin**8)/mbkin**8 + (11622*mckin**10)/mbkin**10 - 
               (7*mckin**12)/mbkin**12)*q_cut**5)/mbkin**12 - 
            (2*(-21 + (3*mckin**2)/mbkin**2 - (369*mckin**4)/mbkin**4 + 
               (16671*mckin**6)/mbkin**6 + (12001*mckin**8)/mbkin**8 + 
               (1097*mckin**10)/mbkin**10 + (28*mckin**12)/mbkin**12)*q_cut**6)/
             mbkin**12 + (2*(-90 + (771*mckin**2)/mbkin**2 - (786*mckin**4)/
                mbkin**4 - (5817*mckin**6)/mbkin**6 + (112*mckin**8)/mbkin**8 + 
               (187*mckin**10)/mbkin**10)*q_cut**7)/mbkin**14 - 
            ((-21 + (51*mckin**2)/mbkin**2 + (453*mckin**4)/mbkin**4 + 
               (495*mckin**6)/mbkin**6 + (2*mckin**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
            ((-138 + (559*mckin**2)/mbkin**2 + (1610*mckin**4)/mbkin**4 + 
               (283*mckin**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
            ((-63 + (247*mckin**2)/mbkin**2 + (58*mckin**4)/mbkin**4)*q_cut**10)/
             mbkin**20 + (35*mckin**2*q_cut**11)/mbkin**24) - 
          12*muG*mupi*((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
              mbkin**4 + q_cut**2/mbkin**4)**2*(-((1 + mckin**2/mbkin**2)**2*
              (-33 - (2131*mckin**2)/mbkin**2 + (30294*mckin**4)/mbkin**4 - 
               (78660*mckin**6)/mbkin**6 - (303336*mckin**8)/mbkin**8 + 
               (1430424*mckin**10)/mbkin**10 + (1908216*mckin**12)/mbkin**12 - 
               (417984*mckin**14)/mbkin**14 - (84555*mckin**16)/mbkin**16 + 
               (12471*mckin**18)/mbkin**18 + (94*mckin**20)/mbkin**20)) + 
            ((-42 - (3763*mckin**2)/mbkin**2 + (26558*mckin**4)/mbkin**4 + 
               (44403*mckin**6)/mbkin**6 - (669144*mckin**8)/mbkin**8 - 
               (1235460*mckin**10)/mbkin**10 - (1259364*mckin**12)/mbkin**12 - 
               (1346004*mckin**14)/mbkin**14 - (612774*mckin**16)/mbkin**16 + 
               (45903*mckin**18)/mbkin**18 + (19966*mckin**20)/mbkin**20 + 
               (121*mckin**22)/mbkin**22)*q_cut)/mbkin**2 + 
            ((-75 - (3521*mckin**2)/mbkin**2 + (35220*mckin**4)/mbkin**4 - 
               (13530*mckin**6)/mbkin**6 - (1006434*mckin**8)/mbkin**8 - 
               (2777886*mckin**10)/mbkin**10 - (2724486*mckin**12)/mbkin**12 - 
               (998682*mckin**14)/mbkin**14 - (17175*mckin**16)/mbkin**16 + 
               (21963*mckin**18)/mbkin**18 + (206*mckin**20)/mbkin**20)*q_cut**2)/
             mbkin**4 + ((84 + (8007*mckin**2)/mbkin**2 - (44916*mckin**4)/
                mbkin**4 - (259494*mckin**6)/mbkin**6 + (280412*mckin**8)/
                mbkin**8 + (873934*mckin**10)/mbkin**10 + (232148*mckin**12)/
                mbkin**12 - (222734*mckin**14)/mbkin**14 - (42208*mckin**16)/
                mbkin**16 - (233*mckin**18)/mbkin**18)*q_cut**3)/mbkin**6 - 
            (2*(-21 - (375*mckin**2)/mbkin**2 + (4350*mckin**4)/mbkin**4 + 
               (858*mckin**6)/mbkin**6 + (40570*mckin**8)/mbkin**8 + 
               (22642*mckin**10)/mbkin**10 + (7617*mckin**12)/mbkin**12 + 
               (3239*mckin**14)/mbkin**14 + (56*mckin**16)/mbkin**16)*q_cut**4)/
             mbkin**8 + (2*mckin**2*(-2439 + (10338*mckin**2)/mbkin**2 + 
               (103080*mckin**4)/mbkin**4 + (158590*mckin**6)/mbkin**6 + 
               (93216*mckin**8)/mbkin**8 + (11622*mckin**10)/mbkin**10 - 
               (7*mckin**12)/mbkin**12)*q_cut**5)/mbkin**12 - 
            (2*(-21 + (3*mckin**2)/mbkin**2 - (369*mckin**4)/mbkin**4 + 
               (16671*mckin**6)/mbkin**6 + (12001*mckin**8)/mbkin**8 + 
               (1097*mckin**10)/mbkin**10 + (28*mckin**12)/mbkin**12)*q_cut**6)/
             mbkin**12 + (2*(-90 + (771*mckin**2)/mbkin**2 - (786*mckin**4)/
                mbkin**4 - (5817*mckin**6)/mbkin**6 + (112*mckin**8)/mbkin**8 + 
               (187*mckin**10)/mbkin**10)*q_cut**7)/mbkin**14 - 
            ((-21 + (51*mckin**2)/mbkin**2 + (453*mckin**4)/mbkin**4 + 
               (495*mckin**6)/mbkin**6 + (2*mckin**8)/mbkin**8)*q_cut**8)/mbkin**16 - 
            ((-138 + (559*mckin**2)/mbkin**2 + (1610*mckin**4)/mbkin**4 + 
               (283*mckin**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
            ((-63 + (247*mckin**2)/mbkin**2 + (58*mckin**4)/mbkin**4)*q_cut**10)/
             mbkin**20 + (35*mckin**2*q_cut**11)/mbkin**24) + 
          12*muG**2*((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
              mbkin**4 + q_cut**2/mbkin**4)**2*(-99 - (7119*mckin**2)/mbkin**2 + 
            (37829*mckin**4)/mbkin**4 - (17113*mckin**6)/mbkin**6 - 
            (532974*mckin**8)/mbkin**8 - (15204*mckin**10)/mbkin**10 - 
            (2224272*mckin**12)/mbkin**12 - (6676824*mckin**14)/mbkin**14 - 
            (2350377*mckin**16)/mbkin**16 + (1791459*mckin**18)/mbkin**18 + 
            (80763*mckin**20)/mbkin**20 - (64799*mckin**22)/mbkin**22 - 
            (470*mckin**24)/mbkin**24 + ((-138 - (7727*mckin**2)/mbkin**2 + 
               (32390*mckin**4)/mbkin**4 - (100577*mckin**6)/mbkin**6 - 
               (1118888*mckin**8)/mbkin**8 - (1883828*mckin**10)/mbkin**10 + 
               (1076956*mckin**12)/mbkin**12 - (1208036*mckin**14)/mbkin**14 - 
               (2113334*mckin**16)/mbkin**16 + (233467*mckin**18)/mbkin**18 + 
               (99510*mckin**20)/mbkin**20 + (605*mckin**22)/mbkin**22)*q_cut)/
             mbkin**2 + ((33 + (2803*mckin**2)/mbkin**2 - (82700*mckin**4)/
                mbkin**4 - (325442*mckin**6)/mbkin**6 - (916186*mckin**8)/
                mbkin**8 - (5230838*mckin**10)/mbkin**10 - (5833550*mckin**12)/
                mbkin**12 - (2100802*mckin**14)/mbkin**14 + (236197*mckin**16)/
                mbkin**16 + (112255*mckin**18)/mbkin**18 + (1030*mckin**20)/
                mbkin**20)*q_cut**2)/mbkin**4 + ((492 + (28915*mckin**2)/mbkin**2 - 
               (164660*mckin**4)/mbkin**4 - (809006*mckin**6)/mbkin**6 + 
               (248124*mckin**8)/mbkin**8 + (2883926*mckin**10)/mbkin**10 + 
               (1025940*mckin**12)/mbkin**12 - (1069270*mckin**14)/mbkin**14 - 
               (209496*mckin**16)/mbkin**16 - (1165*mckin**18)/mbkin**18)*q_cut**3)/
             mbkin**6 - (2*(-141 + (393*mckin**2)/mbkin**2 - (5790*mckin**4)/
                mbkin**4 + (26550*mckin**6)/mbkin**6 + (712286*mckin**8)/
                mbkin**8 + (454454*mckin**10)/mbkin**10 + (76233*mckin**12)/
                mbkin**12 + (15871*mckin**14)/mbkin**14 + (280*mckin**16)/
                mbkin**16)*q_cut**4)/mbkin**8 + (2*(-132 - (11435*mckin**2)/
                mbkin**2 + (60198*mckin**4)/mbkin**4 + (474024*mckin**6)/
                mbkin**6 + (702514*mckin**8)/mbkin**8 + (444840*mckin**10)/
                mbkin**10 + (57426*mckin**12)/mbkin**12 - (35*mckin**14)/
                mbkin**14)*q_cut**5)/mbkin**10 - (2*(195 - (2837*mckin**2)/mbkin**2 - 
               (1149*mckin**4)/mbkin**4 + (116747*mckin**6)/mbkin**6 + 
               (78417*mckin**8)/mbkin**8 + (7121*mckin**10)/mbkin**10 + 
               (140*mckin**12)/mbkin**12)*q_cut**6)/mbkin**12 + 
            (2*(-198 + (1767*mckin**2)/mbkin**2 - (386*mckin**4)/mbkin**4 - 
               (29909*mckin**6)/mbkin**6 + (684*mckin**8)/mbkin**8 + 
               (935*mckin**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
            ((105 - (1039*mckin**2)/mbkin**2 + (1391*mckin**4)/mbkin**4 - 
               (1171*mckin**6)/mbkin**6 - (10*mckin**8)/mbkin**8)*q_cut**8)/
             mbkin**16 - (5*(-138 + (559*mckin**2)/mbkin**2 + (1554*mckin**4)/
                mbkin**4 + (283*mckin**6)/mbkin**6)*q_cut**9)/mbkin**18 + 
            (5*(-63 + (247*mckin**2)/mbkin**2 + (58*mckin**4)/mbkin**4)*q_cut**10)/
             mbkin**20 + (175*mckin**2*q_cut**11)/mbkin**24) - 
          8*mbkin*(-4*(-1 + mckin**2/mbkin**2)**4*(1 + mckin**2/mbkin**2)**2*
             (1179 - (27994*mckin**2)/mbkin**2 + (151119*mckin**4)/mbkin**4 + 
              (982305*mckin**6)/mbkin**6 - (6893010*mckin**8)/mbkin**8 - 
              (8620596*mckin**10)/mbkin**10 + (1136781*mckin**12)/mbkin**12 + 
              (171093*mckin**14)/mbkin**14 - (137043*mckin**16)/mbkin**16 + 
              (13572*mckin**18)/mbkin**18 + (154*mckin**20)/mbkin**20) + 
            ((-1 + mckin**2/mbkin**2)**2*(26163 - (491315*mckin**2)/mbkin**2 + 
               (609844*mckin**4)/mbkin**4 + (29190880*mckin**6)/mbkin**6 - 
               (45147167*mckin**8)/mbkin**8 - (424453401*mckin**10)/mbkin**10 - 
               (729940536*mckin**12)/mbkin**12 - (481708704*mckin**14)/
                mbkin**14 - (67588491*mckin**16)/mbkin**16 + (33128619*mckin**18)/
                mbkin**18 - (4511740*mckin**20)/mbkin**20 - (1905472*mckin**22)/
                mbkin**22 + (315767*mckin**24)/mbkin**24 + (3233*mckin**26)/
                mbkin**26)*q_cut)/mbkin**2 + (4*(-12321 + (196421*mckin**2)/
                mbkin**2 + (301638*mckin**4)/mbkin**4 - (12526887*mckin**6)/
                mbkin**6 + (1399932*mckin**8)/mbkin**8 + (127033574*mckin**10)/
                mbkin**10 + (284550615*mckin**12)/mbkin**12 + (300700218*
                 mckin**14)/mbkin**14 + (150895899*mckin**16)/mbkin**16 + 
               (5517729*mckin**18)/mbkin**18 - (15244856*mckin**20)/mbkin**20 + 
               (2907981*mckin**22)/mbkin**22 + (671466*mckin**24)/mbkin**24 - 
               (153900*mckin**26)/mbkin**26 - (1349*mckin**28)/mbkin**28)*q_cut**2)/
             mbkin**4 + ((14763 - (64787*mckin**2)/mbkin**2 - (2819568*mckin**4)/
                mbkin**4 + (7404252*mckin**6)/mbkin**6 + (43250901*mckin**8)/
                mbkin**8 - (6393169*mckin**10)/mbkin**10 - (79307576*mckin**12)/
                mbkin**12 - (4041216*mckin**14)/mbkin**14 + (48737885*mckin**16)/
                mbkin**16 + (556627*mckin**18)/mbkin**18 - (8219224*mckin**20)/
                mbkin**20 + (656612*mckin**22)/mbkin**22 + (224419*mckin**24)/
                mbkin**24 + (81*mckin**26)/mbkin**26)*q_cut**3)/mbkin**6 + 
            (4*(18273 - (235702*mckin**2)/mbkin**2 - (1086112*mckin**4)/
                mbkin**4 + (11266643*mckin**6)/mbkin**6 + (36598184*mckin**8)/
                mbkin**8 + (45843340*mckin**10)/mbkin**10 + (44974590*mckin**12)/
                mbkin**12 + (36547042*mckin**14)/mbkin**14 + (10361895*mckin**16)/
                mbkin**16 - (3866986*mckin**18)/mbkin**18 - (424058*mckin**20)/
                mbkin**20 + (224799*mckin**22)/mbkin**22 + (2444*mckin**24)/
                mbkin**24)*q_cut**4)/mbkin**8 - ((98721 - (917479*mckin**2)/
                mbkin**2 - (9760513*mckin**4)/mbkin**4 + (30195627*mckin**6)/
                mbkin**6 + (170463350*mckin**8)/mbkin**8 + (259417590*mckin**10)/
                mbkin**10 + (167078226*mckin**12)/mbkin**12 + (15469354*
                 mckin**14)/mbkin**14 - (21555727*mckin**16)/mbkin**16 + 
               (973161*mckin**18)/mbkin**18 + (1273159*mckin**20)/mbkin**20 + 
               (10643*mckin**22)/mbkin**22)*q_cut**5)/mbkin**10 + 
            (4*(5934 - (43901*mckin**2)/mbkin**2 - (695976*mckin**4)/mbkin**4 + 
               (2309184*mckin**6)/mbkin**6 + (11870802*mckin**8)/mbkin**8 + 
               (11318680*mckin**10)/mbkin**10 + (724070*mckin**12)/mbkin**12 - 
               (1728188*mckin**14)/mbkin**14 + (119948*mckin**16)/mbkin**16 + 
               (74017*mckin**18)/mbkin**18 + (526*mckin**20)/mbkin**20)*q_cut**6)/
             mbkin**12 + ((44175 - (243147*mckin**2)/mbkin**2 - (5579352*
                 mckin**4)/mbkin**4 - (17016744*mckin**6)/mbkin**6 - (22924990*
                 mckin**8)/mbkin**8 - (14689794*mckin**10)/mbkin**10 - 
               (2030576*mckin**12)/mbkin**12 + (2336816*mckin**14)/mbkin**14 + 
               (572671*mckin**16)/mbkin**16 + (6717*mckin**18)/mbkin**18)*q_cut**7)/
             mbkin**14 - (4*(11646 - (53280*mckin**2)/mbkin**2 - (1304004*
                 mckin**4)/mbkin**4 - (3516303*mckin**6)/mbkin**6 - (3269188*
                 mckin**8)/mbkin**8 - (647651*mckin**10)/mbkin**10 + 
               (523431*mckin**12)/mbkin**12 + (121214*mckin**14)/mbkin**14 + 
               (3663*mckin**16)/mbkin**16)*q_cut**8)/mbkin**16 + 
            ((24633 - (76675*mckin**2)/mbkin**2 - (1532353*mckin**4)/mbkin**4 - 
               (3216569*mckin**6)/mbkin**6 - (1176415*mckin**8)/mbkin**8 + 
               (448897*mckin**10)/mbkin**10 + (145151*mckin**12)/mbkin**12 + 
               (15283*mckin**14)/mbkin**14)*q_cut**9)/mbkin**18 - 
            (4*(1197 - (4127*mckin**2)/mbkin**2 - (21706*mckin**4)/mbkin**4 - 
               (39039*mckin**6)/mbkin**6 - (14018*mckin**8)/mbkin**8 + 
               (2910*mckin**10)/mbkin**10 + (449*mckin**12)/mbkin**12)*q_cut**10)/
             mbkin**20 - ((8559 + (20945*mckin**2)/mbkin**2 + (29208*mckin**4)/
                mbkin**4 + (54804*mckin**6)/mbkin**6 + (34199*mckin**8)/mbkin**8 + 
               (11197*mckin**10)/mbkin**10)*q_cut**11)/mbkin**22 + 
            (4*(2121 + (4502*mckin**2)/mbkin**2 + (8820*mckin**4)/mbkin**4 + 
               (9231*mckin**6)/mbkin**6 + (2526*mckin**8)/mbkin**8)*q_cut**12)/
             mbkin**24 - ((2379 + (5003*mckin**2)/mbkin**2 + (11693*mckin**4)/
                mbkin**4 + (3329*mckin**6)/mbkin**6)*q_cut**13)/mbkin**26 + 
            (4*mckin**2*(137 + (104*mckin**2)/mbkin**2)*q_cut**14)/mbkin**30 - 
            (5*(15 + (29*mckin**2)/mbkin**2)*q_cut**15)/mbkin**30 + 
            (60*q_cut**16)/mbkin**32)*rhoD + ((mbkin**6 - 7*mbkin**4*mckin**2 - 
             7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
             mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)*
            (-16*(-((-1 + mckin**2/mbkin**2)**4*(-11807 + (43798*mckin**2)/
                   mbkin**2 + (479384*mckin**4)/mbkin**4 - (709662*mckin**6)/
                   mbkin**6 - (2037632*mckin**8)/mbkin**8 - (998650*mckin**10)/
                   mbkin**10 + (38820*mckin**12)/mbkin**12 + (204634*mckin**14)/
                   mbkin**14 + (72155*mckin**16)/mbkin**16 + (800*mckin**18)/
                   mbkin**18)) + ((-1 + mckin**2/mbkin**2)**2*(-53075 + 
                  (56430*mckin**2)/mbkin**2 + (2219469*mckin**4)/mbkin**4 - 
                  (1103324*mckin**6)/mbkin**6 - (11042008*mckin**8)/mbkin**8 - 
                  (11699532*mckin**10)/mbkin**10 - (3837272*mckin**12)/
                   mbkin**12 + (583124*mckin**14)/mbkin**14 + (1195347*mckin**16)/
                   mbkin**16 + (332166*mckin**18)/mbkin**18 + (3395*mckin**20)/
                   mbkin**20)*q_cut)/mbkin**2 + ((79585 + (71237*mckin**2)/
                   mbkin**2 - (3051995*mckin**4)/mbkin**4 + (927717*mckin**6)/
                   mbkin**6 + (13296168*mckin**8)/mbkin**8 + (18284376*mckin**10)/
                   mbkin**10 + (15027276*mckin**12)/mbkin**12 + 
                  (5791152*mckin**14)/mbkin**14 - (1097145*mckin**16)/mbkin**16 - 
                  (2117941*mckin**18)/mbkin**18 - (515489*mckin**20)/mbkin**20 - 
                  (4381*mckin**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
               ((-19567 - (124096*mckin**2)/mbkin**2 + (851901*mckin**4)/
                   mbkin**4 + (142210*mckin**6)/mbkin**6 - (6133506*mckin**8)/
                   mbkin**8 - (6567734*mckin**10)/mbkin**10 + (601502*mckin**12)/
                   mbkin**12 + (3352138*mckin**14)/mbkin**14 + 
                  (1456325*mckin**16)/mbkin**16 + (156130*mckin**18)/mbkin**18 - 
                  (183*mckin**20)/mbkin**20)*q_cut**3)/mbkin**6 + 
               ((-67452 - (313401*mckin**2)/mbkin**2 + (755046*mckin**4)/
                   mbkin**4 + (2414684*mckin**6)/mbkin**6 + (3075992*mckin**8)/
                   mbkin**8 + (2747000*mckin**10)/mbkin**10 + (2478366*mckin**12)/
                   mbkin**12 + (1825168*mckin**14)/mbkin**14 + (418744*mckin**16)/
                   mbkin**16 + (4885*mckin**18)/mbkin**18)*q_cut**4)/mbkin**8 - 
               (2*(-35815 - (226435*mckin**2)/mbkin**2 + (358442*mckin**4)/
                   mbkin**4 + (1835272*mckin**6)/mbkin**6 + (2982792*mckin**8)/
                   mbkin**8 + (2754841*mckin**10)/mbkin**10 + (1366980*mckin**12)/
                   mbkin**12 + (235624*mckin**14)/mbkin**14 + (2391*mckin**16)/
                   mbkin**16)*q_cut**5)/mbkin**10 + (2*(-12395 - (83570*mckin**2)/
                   mbkin**2 + (199368*mckin**4)/mbkin**4 + (884692*mckin**6)/
                   mbkin**6 + (991494*mckin**8)/mbkin**8 + (506815*mckin**10)/
                   mbkin**10 + (88823*mckin**12)/mbkin**12 + (1353*mckin**14)/
                   mbkin**14)*q_cut**6)/mbkin**12 + (2*(715 + (5343*mckin**2)/
                   mbkin**2 - (21471*mckin**4)/mbkin**4 - (34137*mckin**6)/
                   mbkin**6 - (33187*mckin**8)/mbkin**8 - (13378*mckin**10)/
                   mbkin**10 + (1443*mckin**12)/mbkin**12)*q_cut**7)/mbkin**14 - 
               ((-2087 + (2386*mckin**2)/mbkin**2 + (18520*mckin**4)/mbkin**4 + 
                  (2730*mckin**6)/mbkin**6 + (10355*mckin**8)/mbkin**8 + 
                  (10962*mckin**10)/mbkin**10)*q_cut**8)/mbkin**16 + 
               ((-3243 + (3494*mckin**2)/mbkin**2 + (16102*mckin**4)/mbkin**4 + 
                  (26896*mckin**6)/mbkin**6 + (11675*mckin**8)/mbkin**8)*q_cut**9)/
                mbkin**18 - ((-2053 + (3361*mckin**2)/mbkin**2 + (12637*mckin**4)/
                   mbkin**4 + (5333*mckin**6)/mbkin**6)*q_cut**10)/mbkin**20 + 
               ((-455 + (738*mckin**2)/mbkin**2 + (1009*mckin**4)/mbkin**4)*
                 q_cut**11)/mbkin**22 - (15*(6 + (13*mckin**2)/mbkin**2)*q_cut**12)/
                mbkin**24 + (80*q_cut**13)/mbkin**26)*rE - ((-1 + mckin**2/mbkin**2)**
                2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
              (-4*(-2*(-1 + mckin**2/mbkin**2)**2*(-8941 - (6827*mckin**2)/
                    mbkin**2 + (409306*mckin**4)/mbkin**4 + (95050*mckin**6)/
                    mbkin**6 - (1229126*mckin**8)/mbkin**8 - (1188302*mckin**10)/
                    mbkin**10 - (194666*mckin**12)/mbkin**12 + (101926*mckin**14)/
                    mbkin**14 + (25507*mckin**16)/mbkin**16 + (233*mckin**18)/
                    mbkin**18) + ((-44221 - (216428*mckin**2)/mbkin**2 + 
                    (1611477*mckin**4)/mbkin**4 + (2327952*mckin**6)/mbkin**6 - 
                    (4524756*mckin**8)/mbkin**8 - (9498816*mckin**10)/mbkin**10 - 
                    (5978964*mckin**12)/mbkin**12 - (429264*mckin**14)/
                     mbkin**14 + (655281*mckin**16)/mbkin**16 + 
                    (129964*mckin**18)/mbkin**18 + (1055*mckin**20)/mbkin**20)*
                   q_cut)/mbkin**2 - (3*(-4221 - (64891*mckin**2)/mbkin**2 - 
                    (48868*mckin**4)/mbkin**4 + (76692*mckin**6)/mbkin**6 - 
                    (88296*mckin**8)/mbkin**8 + (102712*mckin**10)/mbkin**10 + 
                    (313380*mckin**12)/mbkin**12 + (141964*mckin**14)/mbkin**14 + 
                    (15045*mckin**16)/mbkin**16 + (3*mckin**18)/mbkin**18)*q_cut**2)/
                  mbkin**4 - (2*(-20817 - (183792*mckin**2)/mbkin**2 - 
                    (135051*mckin**4)/mbkin**4 + (238636*mckin**6)/mbkin**6 + 
                    (639405*mckin**8)/mbkin**8 + (822464*mckin**10)/mbkin**10 + 
                    (396761*mckin**12)/mbkin**12 + (60252*mckin**14)/mbkin**14 + 
                    (574*mckin**16)/mbkin**16)*q_cut**3)/mbkin**6 + 
                 (2*(-15621 - (180327*mckin**2)/mbkin**2 - (164400*mckin**4)/
                     mbkin**4 + (580364*mckin**6)/mbkin**6 + (870223*mckin**8)/
                     mbkin**8 + (378225*mckin**10)/mbkin**10 + (48724*mckin**12)/
                     mbkin**12 + (284*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 - 
                 (2*(-1756 - (14964*mckin**2)/mbkin**2 - (2592*mckin**4)/
                     mbkin**4 + (68024*mckin**6)/mbkin**6 + (36849*mckin**8)/
                     mbkin**8 + (5232*mckin**10)/mbkin**10 + (287*mckin**12)/
                     mbkin**12)*q_cut**5)/mbkin**10 + (10*(-66 + (654*mckin**2)/
                     mbkin**2 - (1140*mckin**4)/mbkin**4 - (1204*mckin**6)/
                     mbkin**6 + (117*mckin**8)/mbkin**8 + (7*mckin**10)/mbkin**10)*
                   q_cut**6)/mbkin**12 + (2*(-1347 + (648*mckin**2)/mbkin**2 - 
                    (945*mckin**4)/mbkin**4 + (2092*mckin**6)/mbkin**6 + 
                    (1052*mckin**8)/mbkin**8)*q_cut**7)/mbkin**14 - 
                 (2*(-1960 + (810*mckin**2)/mbkin**2 + (3651*mckin**4)/mbkin**4 + 
                    (1091*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 + 
                 ((-779 + (756*mckin**2)/mbkin**2 + (527*mckin**4)/mbkin**4)*
                   q_cut**9)/mbkin**18 + (35*(-1 + mckin**2/mbkin**2)*q_cut**10)/
                  mbkin**20 + (20*q_cut**11)/mbkin**22)*rG - 4*
                (-2*(-1 + mckin**2/mbkin**2)**2*(6246 + (67217*mckin**2)/
                    mbkin**2 - (529254*mckin**4)/mbkin**4 - (2490332*mckin**6)/
                    mbkin**6 - (2668594*mckin**8)/mbkin**8 - (1145520*mckin**10)/
                    mbkin**10 - (280166*mckin**12)/mbkin**12 + (158480*mckin**14)/
                    mbkin**14 + (62208*mckin**16)/mbkin**16 + (595*mckin**18)/
                    mbkin**18) + ((30051 + (480428*mckin**2)/mbkin**2 - 
                    (1230867*mckin**4)/mbkin**4 - (12753936*mckin**6)/mbkin**6 - 
                    (21644916*mckin**8)/mbkin**8 - (14619072*mckin**10)/
                     mbkin**10 - (5141604*mckin**12)/mbkin**12 - 
                    (1089024*mckin**14)/mbkin**14 + (1095105*mckin**16)/
                     mbkin**16 + (318180*mckin**18)/mbkin**18 + (2695*mckin**20)/
                     mbkin**20)*q_cut)/mbkin**2 - (3*(2529 + (82713*mckin**2)/
                     mbkin**2 + (305552*mckin**4)/mbkin**4 + (155956*mckin**6)/
                     mbkin**6 + (105796*mckin**8)/mbkin**8 + (230008*mckin**10)/
                     mbkin**10 + (330632*mckin**12)/mbkin**12 + 
                    (264908*mckin**14)/mbkin**14 + (37251*mckin**16)/mbkin**16 + 
                    (15*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
                 (2*(14181 + (280842*mckin**2)/mbkin**2 + (943095*mckin**4)/
                     mbkin**4 + (1413102*mckin**6)/mbkin**6 + (1811119*mckin**8)/
                     mbkin**8 + (1516738*mckin**10)/mbkin**10 + 
                    (804843*mckin**12)/mbkin**12 + (148334*mckin**14)/mbkin**14 + 
                    (1442*mckin**16)/mbkin**16)*q_cut**3)/mbkin**6 + 
                 (2*(10191 + (248865*mckin**2)/mbkin**2 + (1087626*mckin**4)/
                     mbkin**4 + (1689732*mckin**6)/mbkin**6 + (1520309*mckin**8)/
                     mbkin**8 + (788349*mckin**10)/mbkin**10 + (120732*mckin**12)/
                     mbkin**12 + (712*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 - 
                 (6*(282 + (7474*mckin**2)/mbkin**2 + (28672*mckin**4)/mbkin**4 + 
                    (40064*mckin**6)/mbkin**6 + (27057*mckin**8)/mbkin**8 + 
                    (4094*mckin**10)/mbkin**10 + (245*mckin**12)/mbkin**12)*q_cut**5)/
                  mbkin**10 + (10*(-18 - (906*mckin**2)/mbkin**2 - 
                    (1926*mckin**4)/mbkin**4 - (3108*mckin**6)/mbkin**6 + 
                    (149*mckin**8)/mbkin**8 + (35*mckin**10)/mbkin**10)*q_cut**6)/
                  mbkin**12 + (2*(-81 - (462*mckin**2)/mbkin**2 + (1233*mckin**4)/
                     mbkin**4 + (4734*mckin**6)/mbkin**6 + (2380*mckin**8)/
                     mbkin**8)*q_cut**7)/mbkin**14 - (6*(-173 + (1028*mckin**2)/
                     mbkin**2 + (2902*mckin**4)/mbkin**4 + (791*mckin**6)/
                     mbkin**6)*q_cut**8)/mbkin**16 + ((-951 + (1936*mckin**2)/
                     mbkin**2 + (871*mckin**4)/mbkin**4)*q_cut**9)/mbkin**18 + 
                 (35*(-3 + (5*mckin**2)/mbkin**2)*q_cut**10)/mbkin**20 + 
                 (60*q_cut**11)/mbkin**22)*sB - 46416*sE - (180016*mckin**2*sE)/
                mbkin**2 + (3327728*mckin**4*sE)/mbkin**4 + (1108272*mckin**6*sE)/
                mbkin**6 - (9079776*mckin**8*sE)/mbkin**8 + (1599360*mckin**10*
                 sE)/mbkin**10 + (4680480*mckin**12*sE)/mbkin**12 - (2416128*
                 mckin**14*sE)/mbkin**14 + (1409328*mckin**16*sE)/mbkin**16 - 
               (108624*mckin**18*sE)/mbkin**18 - (291344*mckin**20*sE)/
                mbkin**20 - (2864*mckin**22*sE)/mbkin**22 + (113376*q_cut*sE)/
                mbkin**2 + (1176992*mckin**2*q_cut*sE)/mbkin**4 - (4539072*mckin**4*
                 q_cut*sE)/mbkin**6 - (22322496*mckin**6*q_cut*sE)/mbkin**8 - 
               (17816832*mckin**8*q_cut*sE)/mbkin**10 - (449664*mckin**10*q_cut*sE)/
                mbkin**12 + (1554240*mckin**12*q_cut*sE)/mbkin**14 - (806976*
                 mckin**14*q_cut*sE)/mbkin**16 + (2644128*mckin**16*q_cut*sE)/
                mbkin**18 + (765024*mckin**18*q_cut*sE)/mbkin**20 + (6400*mckin**20*
                 q_cut*sE)/mbkin**22 - (30384*q_cut**2*sE)/mbkin**4 - (719568*mckin**2*
                 q_cut**2*sE)/mbkin**6 - (1578336*mckin**4*q_cut**2*sE)/mbkin**8 + 
               (1362048*mckin**6*q_cut**2*sE)/mbkin**10 + (1248288*mckin**8*q_cut**2*sE)/
                mbkin**12 - (3449280*mckin**10*q_cut**2*sE)/mbkin**14 - (5232480*
                 mckin**12*q_cut**2*sE)/mbkin**16 - (2609664*mckin**14*q_cut**2*sE)/
                mbkin**18 - (280368*mckin**16*q_cut**2*sE)/mbkin**20 + (144*mckin**18*
                 q_cut**2*sE)/mbkin**22 - (106992*q_cut**3*sE)/mbkin**6 - (1520496*
                 mckin**2*q_cut**3*sE)/mbkin**8 - (3246960*mckin**4*q_cut**3*sE)/
                mbkin**10 - (1996656*mckin**6*q_cut**3*sE)/mbkin**12 - (667760*
                 mckin**8*q_cut**3*sE)/mbkin**14 - (1415408*mckin**10*q_cut**3*sE)/
                mbkin**16 - (2723568*mckin**12*q_cut**3*sE)/mbkin**18 - 
               (698608*mckin**14*q_cut**3*sE)/mbkin**20 - (7168*mckin**16*q_cut**3*sE)/
                mbkin**22 + (78384*q_cut**4*sE)/mbkin**8 + (1397952*mckin**2*q_cut**4*
                 sE)/mbkin**10 + (4097136*mckin**4*q_cut**4*sE)/mbkin**12 + 
               (3924288*mckin**6*q_cut**4*sE)/mbkin**14 + (4126192*mckin**8*q_cut**4*sE)/
                mbkin**16 + (3239904*mckin**10*q_cut**4*sE)/mbkin**18 + 
               (575280*mckin**12*q_cut**4*sE)/mbkin**20 + (3488*mckin**14*q_cut**4*sE)/
                mbkin**22 - (9328*q_cut**5*sE)/mbkin**10 - (115056*mckin**2*q_cut**5*sE)/
                mbkin**12 - (386496*mckin**4*q_cut**5*sE)/mbkin**14 - (386944*
                 mckin**6*q_cut**5*sE)/mbkin**16 - (361392*mckin**8*q_cut**5*sE)/
                mbkin**18 - (63792*mckin**10*q_cut**5*sE)/mbkin**20 - (3136*mckin**12*
                 q_cut**5*sE)/mbkin**22 + (1200*q_cut**6*sE)/mbkin**12 - (24960*mckin**2*
                 q_cut**6*sE)/mbkin**14 - (55200*mckin**4*q_cut**6*sE)/mbkin**16 - 
               (51840*mckin**6*q_cut**6*sE)/mbkin**18 + (19760*mckin**8*q_cut**6*sE)/
                mbkin**20 - (1120*mckin**10*q_cut**6*sE)/mbkin**22 + (6960*q_cut**7*sE)/
                mbkin**14 - (5904*mckin**2*q_cut**7*sE)/mbkin**16 - (15888*mckin**4*
                 q_cut**7*sE)/mbkin**18 + (5232*mckin**6*q_cut**7*sE)/mbkin**20 + 
               (15296*mckin**8*q_cut**7*sE)/mbkin**22 - (6112*q_cut**8*sE)/mbkin**16 - 
               (12048*mckin**2*q_cut**8*sE)/mbkin**18 - (34848*mckin**4*q_cut**8*sE)/
                mbkin**20 - (15472*mckin**6*q_cut**8*sE)/mbkin**22 - (1008*q_cut**9*sE)/
                mbkin**18 + (3664*mckin**2*q_cut**9*sE)/mbkin**20 + (4672*mckin**4*
                 q_cut**9*sE)/mbkin**22 - (560*mckin**2*q_cut**10*sE)/mbkin**22 + 
               (320*q_cut**11*sE)/mbkin**22 - 6084*sqB - (51628*mckin**2*sqB)/
                mbkin**2 + (647540*mckin**4*sqB)/mbkin**4 + (2185704*mckin**6*
                 sqB)/mbkin**6 - (1160436*mckin**8*sqB)/mbkin**8 - (3871392*
                 mckin**10*sqB)/mbkin**10 - (5292*mckin**12*sqB)/mbkin**12 + 
               (1785960*mckin**14*sqB)/mbkin**14 + (538296*mckin**16*sqB)/
                mbkin**16 - (48564*mckin**18*sqB)/mbkin**18 - (14024*mckin**20*
                 sqB)/mbkin**20 - (80*mckin**22*sqB)/mbkin**22 + (14649*q_cut*sqB)/
                mbkin**2 + (230462*mckin**2*q_cut*sqB)/mbkin**4 - (478107*mckin**4*
                 q_cut*sqB)/mbkin**6 - (7544880*mckin**6*q_cut*sqB)/mbkin**8 - 
               (17208996*mckin**8*q_cut*sqB)/mbkin**10 - (15625344*mckin**10*q_cut*
                 sqB)/mbkin**12 - (5968452*mckin**12*q_cut*sqB)/mbkin**14 - 
               (316608*mckin**14*q_cut*sqB)/mbkin**16 + (291675*mckin**16*q_cut*sqB)/
                mbkin**18 + (35826*mckin**18*q_cut*sqB)/mbkin**20 + (175*mckin**20*
                 q_cut*sqB)/mbkin**22 - (3789*q_cut**2*sqB)/mbkin**4 - (117945*mckin**2*
                 q_cut**2*sqB)/mbkin**6 - (543048*mckin**4*q_cut**2*sqB)/mbkin**8 - 
               (622812*mckin**6*q_cut**2*sqB)/mbkin**10 - (569172*mckin**8*q_cut**2*sqB)/
                mbkin**12 - (1021296*mckin**10*q_cut**2*sqB)/mbkin**14 - 
               (789072*mckin**12*q_cut**2*sqB)/mbkin**16 - (201876*mckin**14*q_cut**2*
                 sqB)/mbkin**18 - (11799*mckin**16*q_cut**2*sqB)/mbkin**20 + 
               (9*mckin**18*q_cut**2*sqB)/mbkin**22 - (13812*q_cut**3*sqB)/mbkin**6 - 
               (270534*mckin**2*q_cut**3*sqB)/mbkin**8 - (1041540*mckin**4*q_cut**3*sqB)/
                mbkin**10 - (1476162*mckin**6*q_cut**3*sqB)/mbkin**12 - (1406192*
                 mckin**8*q_cut**3*sqB)/mbkin**14 - (991022*mckin**10*q_cut**3*sqB)/
                mbkin**16 - (329844*mckin**12*q_cut**3*sqB)/mbkin**18 - 
               (33178*mckin**14*q_cut**3*sqB)/mbkin**20 - (196*mckin**16*q_cut**3*sqB)/
                mbkin**22 + (9996*q_cut**4*sqB)/mbkin**8 + (239034*mckin**2*q_cut**4*
                 sqB)/mbkin**10 + (1181778*mckin**4*q_cut**4*sqB)/mbkin**12 + 
               (1946208*mckin**6*q_cut**4*sqB)/mbkin**14 + (1272532*mckin**8*q_cut**4*
                 sqB)/mbkin**16 + (332286*mckin**10*q_cut**4*sqB)/mbkin**18 + 
               (26154*mckin**12*q_cut**4*sqB)/mbkin**20 + (92*mckin**14*q_cut**4*sqB)/
                mbkin**22 - (442*q_cut**5*sqB)/mbkin**10 - (22842*mckin**2*q_cut**5*sqB)/
                mbkin**12 - (87144*mckin**4*q_cut**5*sqB)/mbkin**14 - (111472*
                 mckin**6*q_cut**5*sqB)/mbkin**16 - (31992*mckin**8*q_cut**5*sqB)/
                mbkin**18 - (2910*mckin**10*q_cut**5*sqB)/mbkin**20 - (70*mckin**12*
                 q_cut**5*sqB)/mbkin**22 - (330*q_cut**6*sqB)/mbkin**12 - (4020*mckin**2*
                 q_cut**6*sqB)/mbkin**14 - (10620*mckin**4*q_cut**6*sqB)/mbkin**16 - 
               (5640*mckin**6*q_cut**6*sqB)/mbkin**18 + (800*mckin**8*q_cut**6*sqB)/
                mbkin**20 - (70*mckin**10*q_cut**6*sqB)/mbkin**22 - (672*q_cut**7*sqB)/
                mbkin**14 - (2298*mckin**2*q_cut**7*sqB)/mbkin**16 - (4992*mckin**4*
                 q_cut**7*sqB)/mbkin**18 + (810*mckin**6*q_cut**7*sqB)/mbkin**20 + 
               (416*mckin**8*q_cut**7*sqB)/mbkin**22 + (1016*q_cut**8*sqB)/mbkin**16 - 
               (606*mckin**2*q_cut**8*sqB)/mbkin**18 - (1986*mckin**4*q_cut**8*sqB)/
                mbkin**20 - (364*mckin**6*q_cut**8*sqB)/mbkin**22 - (447*q_cut**9*sqB)/
                mbkin**18 + (412*mckin**2*q_cut**9*sqB)/mbkin**20 + (103*mckin**4*
                 q_cut**9*sqB)/mbkin**22 - (105*q_cut**10*sqB)/mbkin**20 - (35*mckin**2*
                 q_cut**10*sqB)/mbkin**22 + (20*q_cut**11*sqB)/mbkin**22)))/mbkin**6) - 
        6*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
            2*mckin**2*q_cut + q_cut**2)/mbkin**4)*
         (16*(-((-1 + mckin**2/mbkin**2)**4*(-487 + (1378*mckin**2)/mbkin**2 + 
               (103211*mckin**4)/mbkin**4 - (451836*mckin**6)/mbkin**6 - 
               (2360958*mckin**8)/mbkin**8 + (4610472*mckin**10)/mbkin**10 + 
               (11668476*mckin**12)/mbkin**12 + (5391216*mckin**14)/mbkin**14 - 
               (326961*mckin**16)/mbkin**16 - (795314*mckin**18)/mbkin**18 - 
               (332527*mckin**20)/mbkin**20 - (1436*mckin**22)/mbkin**22 + 
               (3726*mckin**24)/mbkin**24)) + (2*(-1 + mckin**2/mbkin**2)**2*
              (-1464 + (878*mckin**2)/mbkin**2 + (264747*mckin**4)/mbkin**4 - 
               (654866*mckin**6)/mbkin**6 - (6024249*mckin**8)/mbkin**8 + 
               (5029272*mckin**10)/mbkin**10 + (32640138*mckin**12)/mbkin**12 + 
               (33450960*mckin**14)/mbkin**14 + (10105578*mckin**16)/mbkin**16 - 
               (1628862*mckin**18)/mbkin**18 - (2359709*mckin**20)/mbkin**20 - 
               (806502*mckin**22)/mbkin**22 + (8687*mckin**24)/mbkin**24 + 
               (11232*mckin**26)/mbkin**26)*q_cut)/mbkin**2 + 
            ((6355 + (7208*mckin**2)/mbkin**2 - (936656*mckin**4)/mbkin**4 + 
               (1373882*mckin**6)/mbkin**6 + (17299235*mckin**8)/mbkin**8 - 
               (10086138*mckin**10)/mbkin**10 - (79195350*mckin**12)/mbkin**12 - 
               (110084676*mckin**14)/mbkin**14 - (87026223*mckin**16)/mbkin**16 - 
               (28155028*mckin**18)/mbkin**18 + (5642380*mckin**20)/mbkin**20 + 
               (8425514*mckin**22)/mbkin**22 + (2715769*mckin**24)/mbkin**24 - 
               (80762*mckin**26)/mbkin**26 - (48870*mckin**28)/mbkin**28)*q_cut**2)/
             mbkin**4 + (2*(-1960 - (9526*mckin**2)/mbkin**2 + (177979*mckin**4)/
                mbkin**4 + (103540*mckin**6)/mbkin**6 - (2959273*mckin**8)/
                mbkin**8 + (220386*mckin**10)/mbkin**10 + (17116804*mckin**12)/
                mbkin**12 + (16837442*mckin**14)/mbkin**14 - (700248*mckin**16)/
                mbkin**16 - (6799396*mckin**18)/mbkin**18 - (3305847*mckin**20)/
                mbkin**20 - (420862*mckin**22)/mbkin**22 + (77361*mckin**24)/
                mbkin**24 + (15120*mckin**26)/mbkin**26)*q_cut**3)/mbkin**6 + 
            ((-6845 - (37942*mckin**2)/mbkin**2 + (791885*mckin**4)/mbkin**4 + 
               (1728634*mckin**6)/mbkin**6 - (5957916*mckin**8)/mbkin**8 - 
               (15444608*mckin**10)/mbkin**10 - (18941330*mckin**12)/mbkin**12 - 
               (18044424*mckin**14)/mbkin**14 - (12783065*mckin**16)/mbkin**16 - 
               (7164330*mckin**18)/mbkin**18 - (1748003*mckin**20)/mbkin**20 + 
               (205406*mckin**22)/mbkin**22 + (52650*mckin**24)/mbkin**24)*q_cut**4)/
             mbkin**8 - (4*(-3434 - (26020*mckin**2)/mbkin**2 + (254961*mckin**4)/
                mbkin**4 + (749581*mckin**6)/mbkin**6 - (1690441*mckin**8)/
                mbkin**8 - (6219248*mckin**10)/mbkin**10 - (8487200*mckin**12)/
                mbkin**12 - (6481863*mckin**14)/mbkin**14 - (2731315*mckin**16)/
                mbkin**16 - (318150*mckin**18)/mbkin**18 + (155525*mckin**20)/
                mbkin**20 + (26532*mckin**22)/mbkin**22)*q_cut**5)/mbkin**10 + 
            ((-6969 - (50096*mckin**2)/mbkin**2 + (479318*mckin**4)/mbkin**4 + 
               (1199560*mckin**6)/mbkin**6 - (3991626*mckin**8)/mbkin**8 - 
               (11924636*mckin**10)/mbkin**10 - (10756192*mckin**12)/mbkin**12 - 
               (4517112*mckin**14)/mbkin**14 - (393005*mckin**16)/mbkin**16 + 
               (317212*mckin**18)/mbkin**18 + (54882*mckin**20)/mbkin**20)*q_cut**6)/
             mbkin**12 + (4*(-960 - (14642*mckin**2)/mbkin**2 - (31541*mckin**4)/
                mbkin**4 + (94173*mckin**6)/mbkin**6 + (501370*mckin**8)/
                mbkin**8 + (725465*mckin**10)/mbkin**10 + (578564*mckin**12)/
                mbkin**12 + (295434*mckin**14)/mbkin**14 + (75253*mckin**16)/
                mbkin**16 + (7200*mckin**18)/mbkin**18)*q_cut**7)/mbkin**14 - 
            ((-6645 - (78502*mckin**2)/mbkin**2 - (8337*mckin**4)/mbkin**4 + 
               (664936*mckin**6)/mbkin**6 + (1551613*mckin**8)/mbkin**8 + 
               (1896026*mckin**10)/mbkin**10 + (1248611*mckin**12)/mbkin**12 + 
               (395792*mckin**14)/mbkin**14 + (54090*mckin**16)/mbkin**16)*q_cut**8)/
             mbkin**16 + (2*(-1760 - (16554*mckin**2)/mbkin**2 + (17955*mckin**4)/
                mbkin**4 + (167540*mckin**6)/mbkin**6 + (275800*mckin**8)/
                mbkin**8 + (197764*mckin**10)/mbkin**10 + (71587*mckin**12)/
                mbkin**12 + (16560*mckin**14)/mbkin**14)*q_cut**9)/mbkin**18 - 
            ((-777 - (4296*mckin**2)/mbkin**2 + (18684*mckin**4)/mbkin**4 + 
               (39826*mckin**6)/mbkin**6 + (6697*mckin**8)/mbkin**8 + 
               (6210*mckin**10)/mbkin**10 + (8946*mckin**12)/mbkin**12)*q_cut**10)/
             mbkin**20 - (2*(-136 - (442*mckin**2)/mbkin**2 - (79*mckin**4)/
                mbkin**4 + (5286*mckin**6)/mbkin**6 + (5739*mckin**8)/mbkin**8 + 
               (2448*mckin**10)/mbkin**10)*q_cut**11)/mbkin**22 + 
            (5*(-83 - (138*mckin**2)/mbkin**2 + (735*mckin**4)/mbkin**4 + 
               (1766*mckin**6)/mbkin**6 + (1494*mckin**8)/mbkin**8)*q_cut**12)/
             mbkin**24 + (200*(mbkin**6 + mbkin**4*mckin**2 - 14*mbkin**2*
                mckin**4 - 18*mckin**6)*q_cut**13)/mbkin**32 - 
            (35*(mbkin**4 - 18*mckin**4)*q_cut**14)/mbkin**32)*rE + 
          ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
            q_cut**2/mbkin**4)*((5040*mckin**2*muG**2)/mbkin**2 - 
            (17868*mckin**4*muG**2)/mbkin**4 - (215088*mckin**6*muG**2)/mbkin**6 + 
            (1321848*mckin**8*muG**2)/mbkin**8 - (532800*mckin**10*muG**2)/
             mbkin**10 + (2895012*mckin**12*muG**2)/mbkin**12 + 
            (26575200*mckin**14*muG**2)/mbkin**14 - (40119408*mckin**16*muG**2)/
             mbkin**16 - (19286928*mckin**18*muG**2)/mbkin**18 + 
            (31501980*mckin**20*muG**2)/mbkin**20 + (485328*mckin**22*muG**2)/
             mbkin**22 - (2758152*mckin**24*muG**2)/mbkin**24 + 
            (108576*mckin**26*muG**2)/mbkin**26 + (37260*mckin**28*muG**2)/
             mbkin**28 + (1680*mckin**2*muG*mupi)/mbkin**2 - 
            (15972*mckin**4*muG*mupi)/mbkin**4 - (151440*mckin**6*muG*mupi)/
             mbkin**6 + (1496040*mckin**8*muG*mupi)/mbkin**8 - 
            (734400*mckin**10*muG*mupi)/mbkin**10 - (18498132*mckin**12*muG*
              mupi)/mbkin**12 + (1055520*mckin**14*muG*mupi)/mbkin**14 + 
            (33922224*mckin**16*muG*mupi)/mbkin**16 + 
            (353232*mckin**18*muG*mupi)/mbkin**18 - (17921580*mckin**20*muG*
              mupi)/mbkin**20 - (528528*mckin**22*muG*mupi)/mbkin**22 + 
            (1024872*mckin**24*muG*mupi)/mbkin**24 + (3936*mckin**26*muG*mupi)/
             mbkin**26 - (7452*mckin**28*muG*mupi)/mbkin**28 - 
            (6720*mckin**2*muG**2*q_cut)/mbkin**4 - (1920*mckin**4*muG**2*q_cut)/
             mbkin**6 + (70728*mckin**6*muG**2*q_cut)/mbkin**8 + 
            (1722336*mckin**8*muG**2*q_cut)/mbkin**10 - (391224*mckin**10*muG**2*q_cut)/
             mbkin**12 - (29557776*mckin**12*muG**2*q_cut)/mbkin**14 - 
            (66936864*mckin**14*muG**2*q_cut)/mbkin**16 - (114608208*mckin**16*muG**2*
              q_cut)/mbkin**18 - (53646816*mckin**18*muG**2*q_cut)/mbkin**20 + 
            (17158224*mckin**20*muG**2*q_cut)/mbkin**22 + 
            (7600536*mckin**22*muG**2*q_cut)/mbkin**24 - (752976*mckin**24*muG**2*q_cut)/
             mbkin**26 - (150120*mckin**26*muG**2*q_cut)/mbkin**28 - 
            (6720*mckin**2*muG*mupi*q_cut)/mbkin**4 + (42240*mckin**4*muG*mupi*q_cut)/
             mbkin**6 + (554712*mckin**6*muG*mupi*q_cut)/mbkin**8 - 
            (3011040*mckin**8*muG*mupi*q_cut)/mbkin**10 - 
            (8769960*mckin**10*muG*mupi*q_cut)/mbkin**12 + 
            (32194512*mckin**12*muG*mupi*q_cut)/mbkin**14 + 
            (98364960*mckin**14*muG*mupi*q_cut)/mbkin**16 + 
            (99688464*mckin**16*muG*mupi*q_cut)/mbkin**18 + 
            (30618720*mckin**18*muG*mupi*q_cut)/mbkin**20 - 
            (7389840*mckin**20*muG*mupi*q_cut)/mbkin**22 - 
            (2901432*mckin**22*muG*mupi*q_cut)/mbkin**24 + 
            (86160*mckin**24*muG*mupi*q_cut)/mbkin**26 + (30024*mckin**26*muG*mupi*
              q_cut)/mbkin**28 - (6720*mckin**2*muG**2*q_cut**2)/mbkin**6 + 
            (33600*mckin**4*muG**2*q_cut**2)/mbkin**8 + (469416*mckin**6*muG**2*q_cut**2)/
             mbkin**10 - (866592*mckin**8*muG**2*q_cut**2)/mbkin**12 - 
            (10653072*mckin**10*muG**2*q_cut**2)/mbkin**14 - 
            (9608352*mckin**12*muG**2*q_cut**2)/mbkin**16 + 
            (14150016*mckin**14*muG**2*q_cut**2)/mbkin**18 + 
            (2844480*mckin**16*muG**2*q_cut**2)/mbkin**20 - 
            (13364592*mckin**18*muG**2*q_cut**2)/mbkin**22 - 
            (4050912*mckin**20*muG**2*q_cut**2)/mbkin**24 + 
            (943128*mckin**22*muG**2*q_cut**2)/mbkin**26 + 
            (151200*mckin**24*muG**2*q_cut**2)/mbkin**28 + 
            (6720*mckin**2*muG*mupi*q_cut**2)/mbkin**6 - (20160*mckin**4*muG*mupi*
              q_cut**2)/mbkin**8 - (408456*mckin**6*muG*mupi*q_cut**2)/mbkin**10 + 
            (674208*mckin**8*muG*mupi*q_cut**2)/mbkin**12 + 
            (6670800*mckin**10*muG*mupi*q_cut**2)/mbkin**14 + 
            (5097888*mckin**12*muG*mupi*q_cut**2)/mbkin**16 - 
            (2860416*mckin**14*muG*mupi*q_cut**2)/mbkin**18 + 
            (3660480*mckin**16*muG*mupi*q_cut**2)/mbkin**20 + 
            (6068016*mckin**18*muG*mupi*q_cut**2)/mbkin**22 + 
            (1275744*mckin**20*muG*mupi*q_cut**2)/mbkin**24 - 
            (176184*mckin**22*muG*mupi*q_cut**2)/mbkin**26 - 
            (30240*mckin**24*muG*mupi*q_cut**2)/mbkin**28 - 
            (6720*mckin**2*muG**2*q_cut**3)/mbkin**8 - (29304*mckin**4*muG**2*q_cut**3)/
             mbkin**10 + (741624*mckin**6*muG**2*q_cut**3)/mbkin**12 - 
            (2216208*mckin**8*muG**2*q_cut**3)/mbkin**14 - (30858432*mckin**10*muG**2*
              q_cut**3)/mbkin**16 - (65947824*mckin**12*muG**2*q_cut**3)/mbkin**18 - 
            (68148864*mckin**14*muG**2*q_cut**3)/mbkin**20 - 
            (32704464*mckin**16*muG**2*q_cut**3)/mbkin**22 - 
            (2054112*mckin**18*muG**2*q_cut**3)/mbkin**24 + 
            (1490184*mckin**20*muG**2*q_cut**3)/mbkin**26 + 
            (150120*mckin**22*muG**2*q_cut**3)/mbkin**28 + 
            (6720*mckin**2*muG*mupi*q_cut**3)/mbkin**8 - (30696*mckin**4*muG*mupi*
              q_cut**3)/mbkin**10 - (616728*mckin**6*muG*mupi*q_cut**3)/mbkin**12 + 
            (1846224*mckin**8*muG*mupi*q_cut**3)/mbkin**14 + 
            (16026048*mckin**10*muG*mupi*q_cut**3)/mbkin**16 + 
            (29689968*mckin**12*muG*mupi*q_cut**3)/mbkin**18 + 
            (29566464*mckin**14*muG*mupi*q_cut**3)/mbkin**20 + 
            (14890128*mckin**16*muG*mupi*q_cut**3)/mbkin**22 + 
            (2012256*mckin**18*muG*mupi*q_cut**3)/mbkin**24 - 
            (221160*mckin**20*muG*mupi*q_cut**3)/mbkin**26 - 
            (30024*mckin**22*muG*mupi*q_cut**3)/mbkin**28 + 
            (43680*mckin**2*muG**2*q_cut**4)/mbkin**10 + (21372*mckin**4*muG**2*q_cut**4)/
             mbkin**12 - (2449776*mckin**6*muG**2*q_cut**4)/mbkin**14 - 
            (764592*mckin**8*muG**2*q_cut**4)/mbkin**16 + (31572960*mckin**10*muG**2*
              q_cut**4)/mbkin**18 + (62188536*mckin**12*muG**2*q_cut**4)/mbkin**20 + 
            (35488272*mckin**14*muG**2*q_cut**4)/mbkin**22 - 
            (47280*mckin**16*muG**2*q_cut**4)/mbkin**24 - (3628512*mckin**18*muG**2*
              q_cut**4)/mbkin**26 - (377460*mckin**20*muG**2*q_cut**4)/mbkin**28 - 
            (16800*mckin**2*muG*mupi*q_cut**4)/mbkin**10 + 
            (35508*mckin**4*muG*mupi*q_cut**4)/mbkin**12 + 
            (1075248*mckin**6*muG*mupi*q_cut**4)/mbkin**14 - 
            (884880*mckin**8*muG*mupi*q_cut**4)/mbkin**16 - 
            (16597728*mckin**10*muG*mupi*q_cut**4)/mbkin**18 - 
            (28130712*mckin**12*muG*mupi*q_cut**4)/mbkin**20 - 
            (14960208*mckin**14*muG*mupi*q_cut**4)/mbkin**22 - 
            (1225488*mckin**16*muG*mupi*q_cut**4)/mbkin**24 + 
            (675168*mckin**18*muG*mupi*q_cut**4)/mbkin**26 + 
            (75492*mckin**20*muG*mupi*q_cut**4)/mbkin**28 - 
            (20160*mckin**2*muG**2*q_cut**5)/mbkin**12 + (29568*mckin**4*muG**2*q_cut**5)/
             mbkin**14 + (825600*mckin**6*muG**2*q_cut**5)/mbkin**16 - 
            (2256528*mckin**8*muG**2*q_cut**5)/mbkin**18 - (15292416*mckin**10*muG**2*
              q_cut**5)/mbkin**20 - (16239504*mckin**12*muG**2*q_cut**5)/mbkin**22 - 
            (2731920*mckin**14*muG**2*q_cut**5)/mbkin**24 + 
            (1086624*mckin**16*muG**2*q_cut**5)/mbkin**26 + 
            (156240*mckin**18*muG**2*q_cut**5)/mbkin**28 + 
            (6720*mckin**2*muG*mupi*q_cut**5)/mbkin**12 - 
            (16128*mckin**4*muG*mupi*q_cut**5)/mbkin**14 - 
            (317952*mckin**6*muG*mupi*q_cut**5)/mbkin**16 + 
            (462672*mckin**8*muG*mupi*q_cut**5)/mbkin**18 + 
            (3914496*mckin**10*muG*mupi*q_cut**5)/mbkin**20 + 
            (3555792*mckin**12*muG*mupi*q_cut**5)/mbkin**22 + 
            (480720*mckin**14*muG*mupi*q_cut**5)/mbkin**24 - 
            (223776*mckin**16*muG*mupi*q_cut**5)/mbkin**26 - 
            (31248*mckin**18*muG*mupi*q_cut**5)/mbkin**28 - 
            (33600*mckin**2*muG**2*q_cut**6)/mbkin**14 - (30240*mckin**4*muG**2*q_cut**6)/
             mbkin**16 + (2204544*mckin**6*muG**2*q_cut**6)/mbkin**18 + 
            (9292416*mckin**8*muG**2*q_cut**6)/mbkin**20 + (13925232*mckin**10*muG**2*
              q_cut**6)/mbkin**22 + (8644032*mckin**12*muG**2*q_cut**6)/mbkin**24 + 
            (2204496*mckin**14*muG**2*q_cut**6)/mbkin**26 + 
            (141120*mckin**16*muG**2*q_cut**6)/mbkin**28 + 
            (6720*mckin**2*muG*mupi*q_cut**6)/mbkin**14 + 
            (3360*mckin**4*muG*mupi*q_cut**6)/mbkin**16 - (436224*mckin**6*muG*mupi*
              q_cut**6)/mbkin**18 - (1708800*mckin**8*muG*mupi*q_cut**6)/mbkin**20 - 
            (2377776*mckin**10*muG*mupi*q_cut**6)/mbkin**22 - 
            (1596096*mckin**12*muG*mupi*q_cut**6)/mbkin**24 - 
            (431760*mckin**14*muG*mupi*q_cut**6)/mbkin**26 - 
            (28224*mckin**16*muG*mupi*q_cut**6)/mbkin**28 + 
            (33600*mckin**2*muG**2*q_cut**7)/mbkin**16 - (14640*mckin**4*muG**2*q_cut**7)/
             mbkin**18 - (2029344*mckin**6*muG**2*q_cut**7)/mbkin**20 - 
            (5885232*mckin**8*muG**2*q_cut**7)/mbkin**22 - 
            (5477280*mckin**10*muG**2*q_cut**7)/mbkin**24 - 
            (1841328*mckin**12*muG**2*q_cut**7)/mbkin**26 - 
            (162000*mckin**14*muG**2*q_cut**7)/mbkin**28 - 
            (6720*mckin**2*muG*mupi*q_cut**7)/mbkin**16 + 
            (9840*mckin**4*muG*mupi*q_cut**7)/mbkin**18 + (378144*mckin**6*muG*mupi*
              q_cut**7)/mbkin**20 + (1128048*mckin**8*muG*mupi*q_cut**7)/mbkin**22 + 
            (1040928*mckin**10*muG*mupi*q_cut**7)/mbkin**24 + 
            (353904*mckin**12*muG*mupi*q_cut**7)/mbkin**26 + 
            (32400*mckin**14*muG*mupi*q_cut**7)/mbkin**28 - 
            (8400*mckin**2*muG**2*q_cut**8)/mbkin**18 + (17964*mckin**4*muG**2*q_cut**8)/
             mbkin**20 + (495648*mckin**6*muG**2*q_cut**8)/mbkin**22 + 
            (865080*mckin**8*muG**2*q_cut**8)/mbkin**24 + (354816*mckin**10*muG**2*
              q_cut**8)/mbkin**26 + (75780*mckin**12*muG**2*q_cut**8)/mbkin**28 + 
            (1680*mckin**2*muG*mupi*q_cut**8)/mbkin**18 - 
            (12924*mckin**4*muG*mupi*q_cut**8)/mbkin**20 - 
            (67104*mckin**6*muG*mupi*q_cut**8)/mbkin**22 - 
            (100824*mckin**8*muG*mupi*q_cut**8)/mbkin**24 - 
            (57600*mckin**10*muG*mupi*q_cut**8)/mbkin**26 - 
            (15156*mckin**12*muG*mupi*q_cut**8)/mbkin**28 - 
            (4032*mckin**4*muG**2*q_cut**9)/mbkin**22 + (31608*mckin**6*muG**2*q_cut**9)/
             mbkin**24 + (72432*mckin**8*muG**2*q_cut**9)/mbkin**26 - 
            (17640*mckin**10*muG**2*q_cut**9)/mbkin**28 + (4032*mckin**4*muG*mupi*
              q_cut**9)/mbkin**22 - (15000*mckin**6*muG*mupi*q_cut**9)/mbkin**24 - 
            (13488*mckin**8*muG*mupi*q_cut**9)/mbkin**26 + 
            (3528*mckin**10*muG*mupi*q_cut**9)/mbkin**28 - 
            (21600*mckin**4*muG**2*q_cut**10)/mbkin**24 - (60840*mckin**6*muG**2*
              q_cut**10)/mbkin**26 - (21600*mckin**8*muG**2*q_cut**10)/mbkin**28 + 
            (4320*mckin**4*muG*mupi*q_cut**10)/mbkin**24 + 
            (9480*mckin**6*muG*mupi*q_cut**10)/mbkin**26 + 
            (4320*mckin**8*muG*mupi*q_cut**10)/mbkin**28 + 
            (23400*mckin**4*muG**2*q_cut**11)/mbkin**26 + (23400*mckin**6*muG**2*
              q_cut**11)/mbkin**28 - (4680*mckin**4*muG*mupi*q_cut**11)/mbkin**26 - 
            (4680*mckin**6*muG*mupi*q_cut**11)/mbkin**28 - 
            (6300*mckin**4*muG**2*q_cut**12)/mbkin**28 + (1260*mckin**4*muG*mupi*
              q_cut**12)/mbkin**28 - 24*mckin**2*muG*((-1 + mckin**2/mbkin**2)**2*(
                -140 + (1051*mckin**2)/mbkin**2 + (14862*mckin**4)/mbkin**4 - 
                (95997*mckin**6)/mbkin**6 - (145656*mckin**8)/mbkin**8 + 
                (1346196*mckin**10)/mbkin**10 + (2750088*mckin**12)/mbkin**12 + 
                (1327128*mckin**14)/mbkin**14 - (125268*mckin**16)/mbkin**16 - 
                (84199*mckin**18)/mbkin**18 + (914*mckin**20)/mbkin**20 + 
                (621*mckin**22)/mbkin**22) - (2*(-280 + (1760*mckin**2)/
                  mbkin**2 + (23113*mckin**4)/mbkin**4 - (125460*mckin**6)/
                  mbkin**6 - (365415*mckin**8)/mbkin**8 + (1341438*mckin**10)/
                  mbkin**10 + (4098540*mckin**12)/mbkin**12 + (4153686*mckin**14)/
                  mbkin**14 + (1275780*mckin**16)/mbkin**16 - (307910*mckin**18)/
                  mbkin**18 - (120893*mckin**20)/mbkin**20 + (3590*mckin**22)/
                  mbkin**22 + (1251*mckin**24)/mbkin**24)*q_cut)/mbkin**2 + 
              (2*(-280 + (840*mckin**2)/mbkin**2 + (17019*mckin**4)/mbkin**4 - 
                 (28092*mckin**6)/mbkin**6 - (277950*mckin**8)/mbkin**8 - 
                 (212412*mckin**10)/mbkin**10 + (119184*mckin**12)/mbkin**12 - 
                 (152520*mckin**14)/mbkin**14 - (252834*mckin**16)/mbkin**16 - 
                 (53156*mckin**18)/mbkin**18 + (7341*mckin**20)/mbkin**20 + 
                 (1260*mckin**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
              (2*(-280 + (1279*mckin**2)/mbkin**2 + (25697*mckin**4)/mbkin**4 - 
                 (76926*mckin**6)/mbkin**6 - (667752*mckin**8)/mbkin**8 - 
                 (1237082*mckin**10)/mbkin**10 - (1231936*mckin**12)/mbkin**12 - 
                 (620422*mckin**14)/mbkin**14 - (83844*mckin**16)/mbkin**16 + 
                 (9215*mckin**18)/mbkin**18 + (1251*mckin**20)/mbkin**20)*q_cut**3)/
               mbkin**6 + ((1400 - (2959*mckin**2)/mbkin**2 - (89604*mckin**4)/
                  mbkin**4 + (73740*mckin**6)/mbkin**6 + (1383144*mckin**8)/
                  mbkin**8 + (2344226*mckin**10)/mbkin**10 + (1246684*mckin**12)/
                  mbkin**12 + (102124*mckin**14)/mbkin**14 - (56264*mckin**16)/
                  mbkin**16 - (6291*mckin**18)/mbkin**18)*q_cut**4)/mbkin**8 + 
              (4*(-140 + (336*mckin**2)/mbkin**2 + (6624*mckin**4)/mbkin**4 - 
                 (9639*mckin**6)/mbkin**6 - (81552*mckin**8)/mbkin**8 - 
                 (74079*mckin**10)/mbkin**10 - (10015*mckin**12)/mbkin**12 + 
                 (4662*mckin**14)/mbkin**14 + (651*mckin**16)/mbkin**16)*q_cut**5)/
               mbkin**10 + (4*(-140 - (70*mckin**2)/mbkin**2 + (9088*mckin**4)/
                  mbkin**4 + (35600*mckin**6)/mbkin**6 + (49537*mckin**8)/
                  mbkin**8 + (33252*mckin**10)/mbkin**10 + (8995*mckin**12)/
                  mbkin**12 + (588*mckin**14)/mbkin**14)*q_cut**6)/mbkin**12 - 
              (4*(-140 + (205*mckin**2)/mbkin**2 + (7878*mckin**4)/mbkin**4 + 
                 (23501*mckin**6)/mbkin**6 + (21686*mckin**8)/mbkin**8 + 
                 (7373*mckin**10)/mbkin**10 + (675*mckin**12)/mbkin**12)*q_cut**7)/
               mbkin**14 + ((-140 + (1077*mckin**2)/mbkin**2 + (5592*mckin**4)/
                  mbkin**4 + (8402*mckin**6)/mbkin**6 + (4800*mckin**8)/mbkin**8 + 
                 (1263*mckin**10)/mbkin**10)*q_cut**8)/mbkin**16 - (2*mckin**2*
                (168 - (625*mckin**2)/mbkin**2 - (562*mckin**4)/mbkin**4 + 
                 (147*mckin**6)/mbkin**6)*q_cut**9)/mbkin**20 - (10*mckin**2*
                (36 + (79*mckin**2)/mbkin**2 + (36*mckin**4)/mbkin**4)*q_cut**10)/
               mbkin**22 + (390*mckin**2*(mbkin**2 + mckin**2)*q_cut**11)/mbkin**26 - 
              (105*mckin**2*q_cut**12)/mbkin**26) - 8*(-2*(-1 + mckin**2/mbkin**2)**2*(
                -172 - (687*mckin**2)/mbkin**2 + (38656*mckin**4)/mbkin**4 - 
                (19884*mckin**6)/mbkin**6 - (1069059*mckin**8)/mbkin**8 - 
                (221682*mckin**10)/mbkin**10 + (3433464*mckin**12)/mbkin**12 + 
                (3472584*mckin**14)/mbkin**14 + (660450*mckin**16)/mbkin**16 - 
                (237695*mckin**18)/mbkin**18 - (70200*mckin**20)/mbkin**20 + 
                (1124*mckin**22)/mbkin**22 + (621*mckin**24)/mbkin**24) + 
              ((-1388 - (9012*mckin**2)/mbkin**2 + (230491*mckin**4)/mbkin**4 + 
                 (355108*mckin**6)/mbkin**6 - (4608237*mckin**8)/mbkin**8 - 
                 (5679876*mckin**10)/mbkin**10 + (12773730*mckin**12)/mbkin**12 + 
                 (27482208*mckin**14)/mbkin**14 + (17583222*mckin**16)/
                  mbkin**16 + (1783916*mckin**18)/mbkin**18 - (1637757*mckin**20)/
                  mbkin**20 - (392644*mckin**22)/mbkin**22 + (15395*mckin**24)/
                  mbkin**24 + (5004*mckin**26)/mbkin**26)*q_cut)/mbkin**2 + 
              ((1400 + (15400*mckin**2)/mbkin**2 - (105765*mckin**4)/mbkin**4 - 
                 (531075*mckin**6)/mbkin**6 + (81882*mckin**8)/mbkin**8 + 
                 (912366*mckin**10)/mbkin**10 - (562032*mckin**12)/mbkin**12 + 
                 (573936*mckin**14)/mbkin**14 + (2274534*mckin**16)/mbkin**16 + 
                 (1220434*mckin**18)/mbkin**18 + (147341*mckin**20)/mbkin**20 - 
                 (31701*mckin**22)/mbkin**22 - (5040*mckin**24)/mbkin**24)*q_cut**2)/
               mbkin**4 - (2*(-694 - (7282*mckin**2)/mbkin**2 + (88915*mckin**4)/
                  mbkin**4 + (527698*mckin**6)/mbkin**6 + (227625*mckin**8)/
                  mbkin**8 - (1022684*mckin**10)/mbkin**10 - (2173541*mckin**12)/
                  mbkin**12 - (2343456*mckin**14)/mbkin**14 - (1034603*mckin**16)/
                  mbkin**16 - (133978*mckin**18)/mbkin**18 + (15034*mckin**20)/
                  mbkin**20 + (2502*mckin**22)/mbkin**22)*q_cut**3)/mbkin**6 + 
              (2*(-1747 - (24294*mckin**2)/mbkin**2 + (68590*mckin**4)/mbkin**4 + 
                 (610257*mckin**6)/mbkin**6 + (258216*mckin**8)/mbkin**8 - 
                 (1948666*mckin**10)/mbkin**10 - (2583439*mckin**12)/mbkin**12 - 
                 (1037823*mckin**14)/mbkin**14 - (42243*mckin**16)/mbkin**16 + 
                 (52154*mckin**18)/mbkin**18 + (6291*mckin**20)/mbkin**20)*q_cut**4)/
               mbkin**8 + (2*(728 + (9492*mckin**2)/mbkin**2 - (19180*mckin**4)/
                  mbkin**4 - (127832*mckin**6)/mbkin**6 + (53767*mckin**8)/
                  mbkin**8 + (381596*mckin**10)/mbkin**10 + (214536*mckin**12)/
                  mbkin**12 + (1364*mckin**14)/mbkin**14 - (19243*mckin**16)/
                  mbkin**16 - (2604*mckin**18)/mbkin**18)*q_cut**5)/mbkin**10 - 
              (2*(-644 - (13636*mckin**2)/mbkin**2 - (41804*mckin**4)/mbkin**4 - 
                 (4064*mckin**6)/mbkin**6 + (108175*mckin**8)/mbkin**8 + 
                 (179041*mckin**10)/mbkin**10 + (116257*mckin**12)/mbkin**12 + 
                 (30835*mckin**14)/mbkin**14 + (2352*mckin**16)/mbkin**16)*q_cut**6)/
               mbkin**12 + (2*(-760 - (12180*mckin**2)/mbkin**2 - 
                 (26033*mckin**4)/mbkin**4 + (24832*mckin**6)/mbkin**6 + 
                 (105795*mckin**8)/mbkin**8 + (84468*mckin**10)/mbkin**10 + 
                 (25418*mckin**12)/mbkin**12 + (2700*mckin**14)/mbkin**14)*q_cut**7)/
               mbkin**14 - (2*(-386 - (2343*mckin**2)/mbkin**2 - (587*mckin**4)/
                  mbkin**4 + (9444*mckin**6)/mbkin**6 + (9577*mckin**8)/mbkin**8 + 
                 (3676*mckin**10)/mbkin**10 + (1263*mckin**12)/mbkin**12)*q_cut**8)/
               mbkin**16 + ((-196 + (588*mckin**2)/mbkin**2 + (1549*mckin**4)/
                  mbkin**4 - (2212*mckin**6)/mbkin**6 - (2053*mckin**8)/mbkin**8 + 
                 (588*mckin**10)/mbkin**10)*q_cut**9)/mbkin**18 + 
              (5*(-48 - (96*mckin**2)/mbkin**2 + (41*mckin**4)/mbkin**4 + 
                 (247*mckin**6)/mbkin**6 + (144*mckin**8)/mbkin**8)*q_cut**10)/mbkin**
                20 - (20*(-13 - (13*mckin**2)/mbkin**2 + (34*mckin**4)/mbkin**4 + 
                 (39*mckin**6)/mbkin**6)*q_cut**11)/mbkin**22 - 
              (70*(mbkin**4 - 3*mckin**4)*q_cut**12)/mbkin**28)*rG + 
            8*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(201 - (2594*mckin**2)/
                  mbkin**2 - (18020*mckin**4)/mbkin**4 + (686586*mckin**6)/
                  mbkin**6 - (484473*mckin**8)/mbkin**8 - (24524568*mckin**10)/
                  mbkin**10 - (51017352*mckin**12)/mbkin**12 - 
                 (29833656*mckin**14)/mbkin**14 - (1099137*mckin**16)/mbkin**16 + 
                 (793962*mckin**18)/mbkin**18 - (274084*mckin**20)/mbkin**20 - 
                 (9490*mckin**22)/mbkin**22 + (3105*mckin**24)/mbkin**24)) + 
              (2*(411 - (4441*mckin**2)/mbkin**2 - (47413*mckin**4)/mbkin**4 + 
                 (998110*mckin**6)/mbkin**6 + (1955322*mckin**8)/mbkin**8 - 
                 (24020691*mckin**10)/mbkin**10 - (77841426*mckin**12)/
                  mbkin**12 - (84248520*mckin**14)/mbkin**14 - 
                 (31464921*mckin**16)/mbkin**16 + (2471757*mckin**18)/mbkin**18 + 
                 (1072279*mckin**20)/mbkin**20 - (424934*mckin**22)/mbkin**22 - 
                 (10828*mckin**24)/mbkin**24 + (6255*mckin**26)/mbkin**26)*q_cut)/
               mbkin**2 - (2*(420 - (2800*mckin**2)/mbkin**2 - (64131*mckin**4)/
                  mbkin**4 + (406035*mckin**6)/mbkin**6 + (2915250*mckin**8)/
                  mbkin**8 + (3976458*mckin**10)/mbkin**10 + (2350824*mckin**12)/
                  mbkin**12 + (4693344*mckin**14)/mbkin**14 + (3558342*mckin**16)/
                  mbkin**16 + (48462*mckin**18)/mbkin**18 - (268445*mckin**20)/
                  mbkin**20 + (9861*mckin**22)/mbkin**22 + (6300*mckin**24)/
                  mbkin**24)*q_cut**2)/mbkin**4 - (2*(411 - (2797*mckin**2)/
                  mbkin**2 - (61931*mckin**4)/mbkin**4 + (744527*mckin**6)/
                  mbkin**6 + (5541108*mckin**8)/mbkin**8 + (10489326*mckin**10)/
                  mbkin**10 + (10608076*mckin**12)/mbkin**12 + 
                 (5830586*mckin**14)/mbkin**14 + (187913*mckin**16)/mbkin**16 - 
                 (376185*mckin**18)/mbkin**18 + (8423*mckin**20)/mbkin**20 + 
                 (6255*mckin**22)/mbkin**22)*q_cut**3)/mbkin**6 + 
              ((2091 - (7688*mckin**2)/mbkin**2 - (351965*mckin**4)/mbkin**4 + 
                 (922140*mckin**6)/mbkin**6 + (11699682*mckin**8)/mbkin**8 + 
                 (22377552*mckin**10)/mbkin**10 + (11839006*mckin**12)/
                  mbkin**12 - (747556*mckin**14)/mbkin**14 - (969997*mckin**16)/
                  mbkin**16 + (118544*mckin**18)/mbkin**18 + (31455*mckin**20)/
                  mbkin**20)*q_cut**4)/mbkin**8 - (4*(231 - (1001*mckin**2)/
                  mbkin**2 - (36378*mckin**4)/mbkin**4 + (32226*mckin**6)/
                  mbkin**6 + (485535*mckin**8)/mbkin**8 + (473268*mckin**10)/
                  mbkin**10 - (63132*mckin**12)/mbkin**12 - (76864*mckin**14)/
                  mbkin**14 + (10500*mckin**16)/mbkin**16 + (3255*mckin**18)/
                  mbkin**18)*q_cut**5)/mbkin**10 - (4*(168 + (532*mckin**2)/
                  mbkin**2 - (37666*mckin**4)/mbkin**4 - (173612*mckin**6)/
                  mbkin**6 - (257329*mckin**8)/mbkin**8 - (157955*mckin**10)/
                  mbkin**10 + (6303*mckin**12)/mbkin**12 + (24871*mckin**14)/
                  mbkin**14 + (2940*mckin**16)/mbkin**16)*q_cut**6)/mbkin**12 + 
              (4*(255 - (245*mckin**2)/mbkin**2 - (39116*mckin**4)/mbkin**4 - 
                 (138249*mckin**6)/mbkin**6 - (129466*mckin**8)/mbkin**8 - 
                 (11635*mckin**10)/mbkin**10 + (18713*mckin**12)/mbkin**12 + 
                 (3375*mckin**14)/mbkin**14)*q_cut**7)/mbkin**14 + 
              ((-843 + (2036*mckin**2)/mbkin**2 + (42441*mckin**4)/mbkin**4 + 
                 (95556*mckin**6)/mbkin**6 + (47845*mckin**8)/mbkin**8 - 
                 (5364*mckin**10)/mbkin**10 - (6315*mckin**12)/mbkin**12)*q_cut**8)/
               mbkin**16 + (2*(147 - (441*mckin**2)/mbkin**2 - (111*mckin**4)/
                  mbkin**4 - (2794*mckin**6)/mbkin**6 - (2452*mckin**8)/mbkin**8 + 
                 (735*mckin**10)/mbkin**10)*q_cut**9)/mbkin**18 + 
              (10*(36 + (72*mckin**2)/mbkin**2 + (93*mckin**4)/mbkin**4 + 
                 (223*mckin**6)/mbkin**6 + (180*mckin**8)/mbkin**8)*q_cut**10)/mbkin**
                20 - (30*(13 + (13*mckin**2)/mbkin**2 + (45*mckin**4)/mbkin**4 + 
                 (65*mckin**6)/mbkin**6)*q_cut**11)/mbkin**22 + 
              (105*(mbkin**4 + 5*mckin**4)*q_cut**12)/mbkin**28)*rhoD + 1680*sB + 
            (30800*mckin**2*sB)/mbkin**2 - (731384*mckin**4*sB)/mbkin**4 - 
            (2179088*mckin**6*sB)/mbkin**6 + (34731536*mckin**8*sB)/mbkin**8 + 
            (58508064*mckin**10*sB)/mbkin**10 - (80713224*mckin**12*sB)/
             mbkin**12 - (84871392*mckin**14*sB)/mbkin**14 + 
            (32328240*mckin**16*sB)/mbkin**16 + (28088784*mckin**18*sB)/
             mbkin**18 + (16871416*mckin**20*sB)/mbkin**20 + 
            (472304*mckin**22*sB)/mbkin**22 - (2513104*mckin**24*sB)/mbkin**24 - 
            (49472*mckin**26*sB)/mbkin**26 + (24840*mckin**28*sB)/mbkin**28 - 
            (6720*q_cut*sB)/mbkin**2 - (156800*mckin**2*q_cut*sB)/mbkin**4 + 
            (1797304*mckin**4*q_cut*sB)/mbkin**6 + (14779760*mckin**6*q_cut*sB)/
             mbkin**8 - (39450888*mckin**8*q_cut*sB)/mbkin**10 - 
            (307704144*mckin**10*q_cut*sB)/mbkin**12 - (510425808*mckin**12*q_cut*sB)/
             mbkin**14 - (358901184*mckin**14*q_cut*sB)/mbkin**16 - 
            (115602576*mckin**16*q_cut*sB)/mbkin**18 - (17262528*mckin**18*q_cut*sB)/
             mbkin**20 + (16659608*mckin**20*q_cut*sB)/mbkin**22 + 
            (7228112*mckin**22*q_cut*sB)/mbkin**24 - (125096*mckin**24*q_cut*sB)/
             mbkin**26 - (100080*mckin**26*q_cut*sB)/mbkin**28 + 
            (6720*q_cut**2*sB)/mbkin**4 + (190400*mckin**2*q_cut**2*sB)/mbkin**6 - 
            (435336*mckin**4*q_cut**2*sB)/mbkin**8 - (10175112*mckin**6*q_cut**2*sB)/
             mbkin**10 - (23688624*mckin**8*q_cut**2*sB)/mbkin**12 - 
            (9373872*mckin**10*q_cut**2*sB)/mbkin**14 - (7295808*mckin**12*q_cut**2*sB)/
             mbkin**16 - (21152640*mckin**14*q_cut**2*sB)/mbkin**18 - 
            (20304528*mckin**16*q_cut**2*sB)/mbkin**20 - (14239056*mckin**18*q_cut**2*
              sB)/mbkin**22 - (3173624*mckin**20*q_cut**2*sB)/mbkin**24 + 
            (434760*mckin**22*q_cut**2*sB)/mbkin**26 + (100800*mckin**24*q_cut**2*sB)/
             mbkin**28 + (6720*q_cut**3*sB)/mbkin**6 + (183680*mckin**2*q_cut**3*sB)/
             mbkin**8 - (1007072*mckin**4*q_cut**3*sB)/mbkin**10 - 
            (18628912*mckin**6*q_cut**3*sB)/mbkin**12 - (56964144*mckin**8*q_cut**3*sB)/
             mbkin**14 - (82334208*mckin**10*q_cut**3*sB)/mbkin**16 - 
            (88682576*mckin**12*q_cut**3*sB)/mbkin**18 - (62325760*mckin**14*q_cut**3*
              sB)/mbkin**20 - (28664272*mckin**16*q_cut**3*sB)/mbkin**22 - 
            (5259072*mckin**18*q_cut**3*sB)/mbkin**24 + (468464*mckin**20*q_cut**3*sB)/
             mbkin**26 + (100080*mckin**22*q_cut**3*sB)/mbkin**28 - 
            (16800*q_cut**4*sB)/mbkin**8 - (523040*mckin**2*q_cut**4*sB)/mbkin**10 - 
            (545576*mckin**4*q_cut**4*sB)/mbkin**12 + (17882352*mckin**6*q_cut**4*sB)/
             mbkin**14 + (65571072*mckin**8*q_cut**4*sB)/mbkin**16 + 
            (87175392*mckin**10*q_cut**4*sB)/mbkin**18 + (63439936*mckin**12*q_cut**4*
              sB)/mbkin**20 + (28625648*mckin**14*q_cut**4*sB)/mbkin**22 + 
            (3146624*mckin**16*q_cut**4*sB)/mbkin**24 - (1684576*mckin**18*q_cut**4*sB)/
             mbkin**26 - (251640*mckin**20*q_cut**4*sB)/mbkin**28 + 
            (6720*q_cut**5*sB)/mbkin**10 + (210560*mckin**2*q_cut**5*sB)/mbkin**12 + 
            (422592*mckin**4*q_cut**5*sB)/mbkin**14 - (3356544*mckin**6*q_cut**5*sB)/
             mbkin**16 - (9863856*mckin**8*q_cut**5*sB)/mbkin**18 - 
            (10081536*mckin**10*q_cut**5*sB)/mbkin**20 - (6059808*mckin**12*q_cut**5*sB)/
             mbkin**22 - (832736*mckin**14*q_cut**5*sB)/mbkin**24 + 
            (583632*mckin**16*q_cut**5*sB)/mbkin**26 + (104160*mckin**18*q_cut**5*sB)/
             mbkin**28 + (6720*q_cut**6*sB)/mbkin**12 + (244160*mckin**2*q_cut**6*sB)/
             mbkin**14 + (1627264*mckin**4*q_cut**6*sB)/mbkin**16 + 
            (4060544*mckin**6*q_cut**6*sB)/mbkin**18 + (5920336*mckin**8*q_cut**6*sB)/
             mbkin**20 + (5762768*mckin**10*q_cut**6*sB)/mbkin**22 + 
            (3621168*mckin**12*q_cut**6*sB)/mbkin**24 + (1114736*mckin**14*q_cut**6*sB)/
             mbkin**26 + (94080*mckin**16*q_cut**6*sB)/mbkin**28 - 
            (6720*q_cut**7*sB)/mbkin**14 - (237440*mckin**2*q_cut**7*sB)/mbkin**16 - 
            (1432496*mckin**4*q_cut**7*sB)/mbkin**18 - (3091392*mckin**6*q_cut**7*sB)/
             mbkin**20 - (3537712*mckin**8*q_cut**7*sB)/mbkin**22 - 
            (2458816*mckin**10*q_cut**7*sB)/mbkin**24 - (888640*mckin**12*q_cut**7*sB)/
             mbkin**26 - (108000*mckin**14*q_cut**7*sB)/mbkin**28 + 
            (1680*q_cut**8*sB)/mbkin**16 + (57680*mckin**2*q_cut**8*sB)/mbkin**18 + 
            (320520*mckin**4*q_cut**8*sB)/mbkin**20 + (406176*mckin**6*q_cut**8*sB)/
             mbkin**22 + (206992*mckin**8*q_cut**8*sB)/mbkin**24 + 
            (115296*mckin**10*q_cut**8*sB)/mbkin**26 + (50520*mckin**12*q_cut**8*sB)/
             mbkin**28 - (17016*mckin**4*q_cut**9*sB)/mbkin**22 + 
            (42256*mckin**6*q_cut**9*sB)/mbkin**24 + (42136*mckin**8*q_cut**9*sB)/
             mbkin**26 - (11760*mckin**10*q_cut**9*sB)/mbkin**28 - 
            (7800*mckin**4*q_cut**10*sB)/mbkin**24 - (24440*mckin**6*q_cut**10*sB)/
             mbkin**26 - (14400*mckin**8*q_cut**10*sB)/mbkin**28 + 
            (13200*mckin**4*q_cut**11*sB)/mbkin**26 + (15600*mckin**6*q_cut**11*sB)/
             mbkin**28 - (4200*mckin**4*q_cut**12*sB)/mbkin**28 - 1656*sE - 
            (17024*mckin**2*sE)/mbkin**2 + (524648*mckin**4*sE)/mbkin**4 + 
            (493472*mckin**6*sE)/mbkin**6 - (19796072*mckin**8*sE)/mbkin**8 - 
            (6705504*mckin**10*sE)/mbkin**10 + (53596152*mckin**12*sE)/
             mbkin**12 - (3650880*mckin**14*sE)/mbkin**14 - 
            (33802440*mckin**16*sE)/mbkin**16 + (10860096*mckin**18*sE)/
             mbkin**18 - (1923880*mckin**20*sE)/mbkin**20 - 
            (1014368*mckin**22*sE)/mbkin**22 + (1418152*mckin**24*sE)/mbkin**24 + 
            (34208*mckin**26*sE)/mbkin**26 - (14904*mckin**28*sE)/mbkin**28 + 
            (6672*q_cut*sE)/mbkin**2 + (98768*mckin**2*q_cut*sE)/mbkin**4 - 
            (1390912*mckin**4*q_cut*sE)/mbkin**6 - (7268672*mckin**6*q_cut*sE)/
             mbkin**8 + (30554160*mckin**8*q_cut*sE)/mbkin**10 + 
            (130632048*mckin**10*q_cut*sE)/mbkin**12 + (113878272*mckin**12*q_cut*sE)/
             mbkin**14 + (4777344*mckin**14*q_cut*sE)/mbkin**16 - 
            (22997904*mckin**16*q_cut*sE)/mbkin**18 + (1645872*mckin**18*q_cut*sE)/
             mbkin**20 - (7863488*mckin**20*q_cut*sE)/mbkin**22 - 
            (4152128*mckin**22*q_cut*sE)/mbkin**24 + (69200*mckin**24*q_cut*sE)/
             mbkin**26 + (60048*mckin**26*q_cut*sE)/mbkin**28 - (6720*q_cut**2*sE)/
             mbkin**4 - (129920*mckin**2*q_cut**2*sE)/mbkin**6 + 
            (497856*mckin**4*q_cut**2*sE)/mbkin**8 + (5814624*mckin**6*q_cut**2*sE)/
             mbkin**10 + (7556928*mckin**8*q_cut**2*sE)/mbkin**12 - 
            (7261248*mckin**10*q_cut**2*sE)/mbkin**14 - (2213760*mckin**12*q_cut**2*sE)/
             mbkin**16 + (16592640*mckin**14*q_cut**2*sE)/mbkin**18 + 
            (18923328*mckin**16*q_cut**2*sE)/mbkin**20 + (10571712*mckin**18*q_cut**2*
              sE)/mbkin**22 + (1788608*mckin**20*q_cut**2*sE)/mbkin**24 - 
            (302688*mckin**22*q_cut**2*sE)/mbkin**26 - (60480*mckin**24*q_cut**2*sE)/
             mbkin**28 - (6672*q_cut**3*sE)/mbkin**6 - (125456*mckin**2*q_cut**3*sE)/
             mbkin**8 + (890288*mckin**4*q_cut**3*sE)/mbkin**10 + 
            (10654768*mckin**6*q_cut**3*sE)/mbkin**12 + (21660768*mckin**8*q_cut**3*sE)/
             mbkin**14 + (16149600*mckin**10*q_cut**3*sE)/mbkin**16 + 
            (6608480*mckin**12*q_cut**3*sE)/mbkin**18 + (3438688*mckin**14*q_cut**3*sE)/
             mbkin**20 + (9145264*mckin**16*q_cut**3*sE)/mbkin**22 + 
            (3380400*mckin**18*q_cut**3*sE)/mbkin**24 - (160016*mckin**20*q_cut**3*sE)/
             mbkin**26 - (60048*mckin**22*q_cut**3*sE)/mbkin**28 + 
            (16776*q_cut**4*sE)/mbkin**8 + (372992*mckin**2*q_cut**4*sE)/mbkin**10 - 
            (156280*mckin**4*q_cut**4*sE)/mbkin**12 - (11175168*mckin**6*q_cut**4*sE)/
             mbkin**14 - (26991504*mckin**8*q_cut**4*sE)/mbkin**16 - 
            (20178240*mckin**10*q_cut**4*sE)/mbkin**18 - (13035280*mckin**12*q_cut**4*
              sE)/mbkin**20 - (11467520*mckin**14*q_cut**4*sE)/mbkin**22 - 
            (2075576*mckin**16*q_cut**4*sE)/mbkin**24 + (917056*mckin**18*q_cut**4*sE)/
             mbkin**26 + (150984*mckin**20*q_cut**4*sE)/mbkin**28 - 
            (6944*q_cut**5*sE)/mbkin**10 - (147616*mckin**2*q_cut**5*sE)/mbkin**12 - 
            (60352*mckin**4*q_cut**5*sE)/mbkin**14 + (2193472*mckin**6*q_cut**5*sE)/
             mbkin**16 + (3579328*mckin**8*q_cut**5*sE)/mbkin**18 + 
            (1392704*mckin**10*q_cut**5*sE)/mbkin**20 + (1054912*mckin**12*q_cut**5*sE)/
             mbkin**22 - (59328*mckin**14*q_cut**5*sE)/mbkin**24 - 
            (400288*mckin**16*q_cut**5*sE)/mbkin**26 - (62496*mckin**18*q_cut**5*sE)/
             mbkin**28 - (6272*q_cut**6*sE)/mbkin**12 - (186368*mckin**2*q_cut**6*sE)/
             mbkin**14 - (920512*mckin**4*q_cut**6*sE)/mbkin**16 - 
            (1492864*mckin**6*q_cut**6*sE)/mbkin**18 - (1181056*mckin**8*q_cut**6*sE)/
             mbkin**20 - (1039424*mckin**10*q_cut**6*sE)/mbkin**22 - 
            (1224128*mckin**12*q_cut**6*sE)/mbkin**24 - (539840*mckin**14*q_cut**6*sE)/
             mbkin**26 - (56448*mckin**16*q_cut**6*sE)/mbkin**28 + 
            (7200*q_cut**7*sE)/mbkin**14 + (173600*mckin**2*q_cut**7*sE)/mbkin**16 + 
            (748640*mckin**4*q_cut**7*sE)/mbkin**18 + (1101408*mckin**6*q_cut**7*sE)/
             mbkin**20 + (1050976*mckin**8*q_cut**7*sE)/mbkin**22 + 
            (997216*mckin**10*q_cut**7*sE)/mbkin**24 + (448288*mckin**12*q_cut**7*sE)/
             mbkin**26 + (64800*mckin**14*q_cut**7*sE)/mbkin**28 - 
            (3368*q_cut**8*sE)/mbkin**16 - (37504*mckin**2*q_cut**8*sE)/mbkin**18 - 
            (137032*mckin**4*q_cut**8*sE)/mbkin**20 - (121760*mckin**6*q_cut**8*sE)/
             mbkin**22 - (55176*mckin**8*q_cut**8*sE)/mbkin**24 - 
            (47584*mckin**10*q_cut**8*sE)/mbkin**26 - (30312*mckin**12*q_cut**8*sE)/
             mbkin**28 + (784*q_cut**9*sE)/mbkin**18 - (2352*mckin**2*q_cut**9*sE)/
             mbkin**20 + (5376*mckin**4*q_cut**9*sE)/mbkin**22 - 
            (19200*mckin**6*q_cut**9*sE)/mbkin**24 - (20656*mckin**8*q_cut**9*sE)/
             mbkin**26 + (7056*mckin**10*q_cut**9*sE)/mbkin**28 + 
            (960*q_cut**10*sE)/mbkin**20 + (1920*mckin**2*q_cut**10*sE)/mbkin**22 + 
            (1920*mckin**4*q_cut**10*sE)/mbkin**24 + (8480*mckin**6*q_cut**10*sE)/
             mbkin**26 + (8640*mckin**8*q_cut**10*sE)/mbkin**28 - 
            (1040*q_cut**11*sE)/mbkin**22 - (1040*mckin**2*q_cut**11*sE)/mbkin**24 - 
            (6160*mckin**4*q_cut**11*sE)/mbkin**26 - (9360*mckin**6*q_cut**11*sE)/
             mbkin**28 + (280*q_cut**12*sE)/mbkin**24 + (2520*mckin**4*q_cut**12*sE)/
             mbkin**28 - 213*sqB - (3332*mckin**2*sqB)/mbkin**2 + 
            (92843*mckin**4*sqB)/mbkin**4 + (287792*mckin**6*sqB)/mbkin**6 - 
            (4445411*mckin**8*sqB)/mbkin**8 - (11891340*mckin**10*sqB)/
             mbkin**10 + (6455133*mckin**12*sqB)/mbkin**12 + 
            (22235424*mckin**14*sqB)/mbkin**14 + (1168293*mckin**16*sqB)/
             mbkin**16 - (10937628*mckin**18*sqB)/mbkin**18 - 
            (3379867*mckin**20*sqB)/mbkin**20 + (311344*mckin**22*sqB)/
             mbkin**22 + (109843*mckin**24*sqB)/mbkin**24 - (2260*mckin**26*sqB)/
             mbkin**26 - (621*mckin**28*sqB)/mbkin**28 + (846*q_cut*sqB)/mbkin**2 + 
            (17894*mckin**2*q_cut*sqB)/mbkin**4 - (224902*mckin**4*q_cut*sqB)/
             mbkin**6 - (1976060*mckin**6*q_cut*sqB)/mbkin**8 + 
            (4145544*mckin**8*q_cut*sqB)/mbkin**10 + (46430562*mckin**10*q_cut*sqB)/
             mbkin**12 + (100628196*mckin**12*q_cut*sqB)/mbkin**14 + 
            (93478512*mckin**14*q_cut*sqB)/mbkin**16 + (37222638*mckin**16*q_cut*sqB)/
             mbkin**18 + (1993746*mckin**18*q_cut*sqB)/mbkin**20 - 
            (2040638*mckin**20*q_cut*sqB)/mbkin**22 - (280340*mckin**22*q_cut*sqB)/
             mbkin**24 + (19100*mckin**24*q_cut*sqB)/mbkin**26 + 
            (2502*mckin**26*q_cut*sqB)/mbkin**28 - (840*q_cut**2*sqB)/mbkin**4 - 
            (22400*mckin**2*q_cut**2*sqB)/mbkin**6 + (43242*mckin**4*q_cut**2*sqB)/
             mbkin**8 + (1339974*mckin**6*q_cut**2*sqB)/mbkin**10 + 
            (4027188*mckin**8*q_cut**2*sqB)/mbkin**12 + (3553428*mckin**10*q_cut**2*sqB)/
             mbkin**14 + (2532720*mckin**12*q_cut**2*sqB)/mbkin**16 + 
            (5403648*mckin**14*q_cut**2*sqB)/mbkin**18 + (4888284*mckin**16*q_cut**2*
              sqB)/mbkin**20 + (1486908*mckin**18*q_cut**2*sqB)/mbkin**22 + 
            (65846*mckin**20*q_cut**2*sqB)/mbkin**24 - (30678*mckin**22*q_cut**2*sqB)/
             mbkin**26 - (2520*mckin**24*q_cut**2*sqB)/mbkin**28 - 
            (846*q_cut**3*sqB)/mbkin**6 - (21278*mckin**2*q_cut**3*sqB)/mbkin**8 + 
            (133430*mckin**4*q_cut**3*sqB)/mbkin**10 + (2501194*mckin**6*q_cut**3*sqB)/
             mbkin**12 + (8789568*mckin**8*q_cut**3*sqB)/mbkin**14 + 
            (13410612*mckin**10*q_cut**3*sqB)/mbkin**16 + 
            (12845168*mckin**12*q_cut**3*sqB)/mbkin**18 + 
            (7934668*mckin**14*q_cut**3*sqB)/mbkin**20 + (2365342*mckin**16*q_cut**3*
              sqB)/mbkin**22 + (190170*mckin**18*q_cut**3*sqB)/mbkin**24 - 
            (23606*mckin**20*q_cut**3*sqB)/mbkin**26 - (2502*mckin**22*q_cut**3*sqB)/
             mbkin**28 + (2103*q_cut**4*sqB)/mbkin**8 + (61736*mckin**2*q_cut**4*sqB)/
             mbkin**10 + (83855*mckin**4*q_cut**4*sqB)/mbkin**12 - 
            (2321052*mckin**6*q_cut**4*sqB)/mbkin**14 - (9988566*mckin**8*q_cut**4*sqB)/
             mbkin**16 - (15003696*mckin**10*q_cut**4*sqB)/mbkin**18 - 
            (9525682*mckin**12*q_cut**4*sqB)/mbkin**20 - (2254172*mckin**14*q_cut**4*
              sqB)/mbkin**22 + (63919*mckin**16*q_cut**4*sqB)/mbkin**24 + 
            (85744*mckin**18*q_cut**4*sqB)/mbkin**26 + (6291*mckin**20*q_cut**4*sqB)/
             mbkin**28 - (812*q_cut**5*sqB)/mbkin**10 - (25228*mckin**2*q_cut**5*sqB)/
             mbkin**12 - (67192*mckin**4*q_cut**5*sqB)/mbkin**14 + 
            (391288*mckin**6*q_cut**5*sqB)/mbkin**16 + (1434028*mckin**8*q_cut**5*sqB)/
             mbkin**18 + (1387040*mckin**10*q_cut**5*sqB)/mbkin**20 + 
            (315616*mckin**12*q_cut**5*sqB)/mbkin**22 - (90672*mckin**14*q_cut**5*sqB)/
             mbkin**24 - (37240*mckin**16*q_cut**5*sqB)/mbkin**26 - 
            (2604*mckin**18*q_cut**5*sqB)/mbkin**28 - (896*q_cut**6*sqB)/mbkin**12 - 
            (28784*mckin**2*q_cut**6*sqB)/mbkin**14 - (208312*mckin**4*q_cut**6*sqB)/
             mbkin**16 - (551920*mckin**6*q_cut**6*sqB)/mbkin**18 - 
            (725956*mckin**8*q_cut**6*sqB)/mbkin**20 - (551756*mckin**10*q_cut**6*sqB)/
             mbkin**22 - (228404*mckin**12*q_cut**6*sqB)/mbkin**24 - 
            (42980*mckin**14*q_cut**6*sqB)/mbkin**26 - (2352*mckin**16*q_cut**6*sqB)/
             mbkin**28 + (780*q_cut**7*sqB)/mbkin**14 + (28700*mckin**2*q_cut**7*sqB)/
             mbkin**16 + (188264*mckin**4*q_cut**7*sqB)/mbkin**18 + 
            (443676*mckin**6*q_cut**7*sqB)/mbkin**20 + (462304*mckin**8*q_cut**7*sqB)/
             mbkin**22 + (209860*mckin**10*q_cut**7*sqB)/mbkin**24 + 
            (40228*mckin**12*q_cut**7*sqB)/mbkin**26 + (2700*mckin**14*q_cut**7*sqB)/
             mbkin**28 + (q_cut**8*sqB)/mbkin**16 - (7492*mckin**2*q_cut**8*sqB)/
             mbkin**18 - (42259*mckin**4*q_cut**8*sqB)/mbkin**20 - 
            (69524*mckin**6*q_cut**8*sqB)/mbkin**22 - (39519*mckin**8*q_cut**8*sqB)/
             mbkin**24 - (7564*mckin**10*q_cut**8*sqB)/mbkin**26 - 
            (1263*mckin**12*q_cut**8*sqB)/mbkin**28 - (98*q_cut**9*sqB)/mbkin**18 + 
            (294*mckin**2*q_cut**9*sqB)/mbkin**20 + (1566*mckin**4*q_cut**9*sqB)/
             mbkin**22 - (348*mckin**6*q_cut**9*sqB)/mbkin**24 - 
            (724*mckin**8*q_cut**9*sqB)/mbkin**26 + (294*mckin**10*q_cut**9*sqB)/
             mbkin**28 - (120*q_cut**10*sqB)/mbkin**20 - (240*mckin**2*q_cut**10*sqB)/
             mbkin**22 - (450*mckin**4*q_cut**10*sqB)/mbkin**24 + 
            (170*mckin**6*q_cut**10*sqB)/mbkin**26 + (360*mckin**8*q_cut**10*sqB)/
             mbkin**28 + (130*q_cut**11*sqB)/mbkin**22 + (130*mckin**2*q_cut**11*sqB)/
             mbkin**24 - (190*mckin**4*q_cut**11*sqB)/mbkin**26 - 
            (390*mckin**6*q_cut**11*sqB)/mbkin**28 - (35*q_cut**12*sqB)/mbkin**24 + 
            (105*mckin**4*q_cut**12*sqB)/mbkin**28))*
         np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                 mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
               mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
             np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                2*mckin**2*q_cut + q_cut**2)/mbkin**4))) - 
        (72*mckin**4*(-16*(-((-1 + mckin**2/mbkin**2)**4*(-767 - (3711*mckin**2)/
                 mbkin**2 + (98861*mckin**4)/mbkin**4 + (43717*mckin**6)/
                 mbkin**6 - (841083*mckin**8)/mbkin**8 - (744595*mckin**10)/
                 mbkin**10 - (69221*mckin**12)/mbkin**12 + (27747*mckin**14)/
                 mbkin**14 + (26246*mckin**16)/mbkin**16 + (3726*mckin**18)/
                 mbkin**18)) + ((-1 + mckin**2/mbkin**2)**2*(-3841 - 
                (25194*mckin**2)/mbkin**2 + (398290*mckin**4)/mbkin**4 + 
                (568154*mckin**6)/mbkin**6 - (3222018*mckin**8)/mbkin**8 - 
                (6339314*mckin**10)/mbkin**10 - (3173788*mckin**12)/mbkin**12 - 
                (222702*mckin**14)/mbkin**14 + (192499*mckin**16)/mbkin**16 + 
                (136536*mckin**18)/mbkin**18 + (18738*mckin**20)/mbkin**20)*q_cut)/
              mbkin**2 + ((6921 + (51033*mckin**2)/mbkin**2 - (589962*mckin**4)/
                 mbkin**4 - (750018*mckin**6)/mbkin**6 + (4107426*mckin**8)/
                 mbkin**8 + (7988514*mckin**10)/mbkin**10 + (8361852*mckin**12)/
                 mbkin**12 + (4423164*mckin**14)/mbkin**14 + (352365*mckin**16)/
                 mbkin**16 - (329619*mckin**18)/mbkin**18 - (242538*mckin**20)/
                 mbkin**20 - (33858*mckin**22)/mbkin**22)*q_cut**2)/mbkin**4 + 
             ((-3847 - (32624*mckin**2)/mbkin**2 + (269714*mckin**4)/mbkin**4 + 
                (509592*mckin**6)/mbkin**6 - (2290278*mckin**8)/mbkin**8 - 
                (4886864*mckin**10)/mbkin**10 - (2102320*mckin**12)/mbkin**12 + 
                (331856*mckin**14)/mbkin**14 + (480765*mckin**16)/mbkin**16 + 
                (172880*mckin**18)/mbkin**18 + (18846*mckin**20)/mbkin**20)*q_cut**3)/
              mbkin**6 + (2*(-1925 - (25865*mckin**2)/mbkin**2 + (1041*mckin**4)/
                 mbkin**4 + (219837*mckin**6)/mbkin**6 + (528849*mckin**8)/
                 mbkin**8 + (535197*mckin**10)/mbkin**10 + (298475*mckin**12)/
                 mbkin**12 + (155807*mckin**14)/mbkin**14 + (72450*mckin**16)/
                 mbkin**16 + (9450*mckin**18)/mbkin**18)*q_cut**4)/mbkin**8 - 
             (2*(-3479 - (44856*mckin**2)/mbkin**2 + (21765*mckin**4)/mbkin**4 + 
                (384640*mckin**6)/mbkin**6 + (691899*mckin**8)/mbkin**8 + 
                (706050*mckin**10)/mbkin**10 + (414721*mckin**12)/mbkin**12 + 
                (147084*mckin**14)/mbkin**14 + (17262*mckin**16)/mbkin**16)*q_cut**5)/
              mbkin**10 + (2*(-1967 - (24199*mckin**2)/mbkin**2 + 
                (22930*mckin**4)/mbkin**4 + (237662*mckin**6)/mbkin**6 + 
                (338587*mckin**8)/mbkin**8 + (219507*mckin**10)/mbkin**10 + 
                (81130*mckin**12)/mbkin**12 + (10206*mckin**14)/mbkin**14)*q_cut**6)/
              mbkin**12 - (2*(-397 - (4456*mckin**2)/mbkin**2 + (9588*mckin**4)/
                 mbkin**4 + (42868*mckin**6)/mbkin**6 + (33001*mckin**8)/
                 mbkin**8 + (17016*mckin**10)/mbkin**10 + (2106*mckin**12)/
                 mbkin**12)*q_cut**7)/mbkin**14 + ((187 + (159*mckin**2)/mbkin**2 - 
                (1587*mckin**4)/mbkin**4 - (1751*mckin**6)/mbkin**6 + 
                (1530*mckin**8)/mbkin**8 - (3366*mckin**10)/mbkin**10)*q_cut**8)/
              mbkin**16 + (5*(-57 - (72*mckin**2)/mbkin**2 + (449*mckin**4)/
                 mbkin**4 + (692*mckin**6)/mbkin**6 + (1026*mckin**8)/mbkin**8)*q_cut**
                9)/mbkin**18 - (15*(-11 - (11*mckin**2)/mbkin**2 + (134*mckin**4)/
                 mbkin**4 + (198*mckin**6)/mbkin**6)*q_cut**10)/mbkin**20 - 
             (35*(mbkin**4 - 18*mckin**4)*q_cut**11)/mbkin**26)*rE + 
           ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + 
             q_cut**2/mbkin**4)*((-3360*mckin**2*muG**2)/mbkin**2 - 
             (60*mckin**4*muG**2)/mbkin**4 + (15708*mckin**6*muG**2)/mbkin**6 + 
             (110856*mckin**8*muG**2)/mbkin**8 + (1868664*mckin**10*muG**2)/
              mbkin**10 - (1429008*mckin**12*muG**2)/mbkin**12 - 
             (2985360*mckin**14*muG**2)/mbkin**14 + (1970472*mckin**16*muG**2)/
              mbkin**16 + (636024*mckin**18*muG**2)/mbkin**18 - 
             (176484*mckin**20*muG**2)/mbkin**20 - (7452*mckin**22*muG**2)/
              mbkin**22 - (3360*mckin**2*muG*mupi)/mbkin**2 + 
             (13500*mckin**4*muG*mupi)/mbkin**4 + (179172*mckin**6*muG*mupi)/
              mbkin**6 - (732456*mckin**8*muG*mupi)/mbkin**8 - 
             (1583064*mckin**10*muG*mupi)/mbkin**10 + (2410128*mckin**12*muG*
               mupi)/mbkin**12 + (1627920*mckin**14*muG*mupi)/mbkin**14 - 
             (1543752*mckin**16*muG*mupi)/mbkin**16 - (451224*mckin**18*muG*
               mupi)/mbkin**18 + (75684*mckin**20*muG*mupi)/mbkin**20 + 
             (7452*mckin**22*muG*mupi)/mbkin**22 - (3360*mckin**2*muG**2*q_cut)/
              mbkin**4 + (30300*mckin**4*muG**2*q_cut)/mbkin**6 + 
             (308712*mckin**6*muG**2*q_cut)/mbkin**8 - (670080*mckin**8*muG**2*q_cut)/
              mbkin**10 - (4759032*mckin**10*muG**2*q_cut)/mbkin**12 - 
             (8785320*mckin**12*muG**2*q_cut)/mbkin**14 - (6468600*mckin**14*muG**
                2*q_cut)/mbkin**16 - (210432*mckin**16*muG**2*q_cut)/mbkin**18 + 
             (576840*mckin**18*muG**2*q_cut)/mbkin**20 + (22572*mckin**20*muG**2*q_cut)/
              mbkin**22 + (10080*mckin**2*muG*mupi*q_cut)/mbkin**4 - 
             (10140*mckin**4*muG*mupi*q_cut)/mbkin**6 - (510312*mckin**6*muG*mupi*
               q_cut)/mbkin**8 + (283680*mckin**8*muG*mupi*q_cut)/mbkin**10 + 
             (5767032*mckin**10*muG*mupi*q_cut)/mbkin**12 + 
             (9349800*mckin**12*muG*mupi*q_cut)/mbkin**14 + 
             (5003640*mckin**14*muG*mupi*q_cut)/mbkin**16 + 
             (361632*mckin**16*muG*mupi*q_cut)/mbkin**18 - 
             (274440*mckin**18*muG*mupi*q_cut)/mbkin**20 - 
             (22572*mckin**20*muG*mupi*q_cut)/mbkin**22 + (6720*mckin**2*muG**2*q_cut**
                2)/mbkin**6 + (20160*mckin**4*muG**2*q_cut**2)/mbkin**8 - 
             (235080*mckin**6*muG**2*q_cut**2)/mbkin**10 - (509736*mckin**8*muG**2*q_cut**
                2)/mbkin**12 + (1261584*mckin**10*muG**2*q_cut**2)/mbkin**14 + 
             (1876464*mckin**12*muG**2*q_cut**2)/mbkin**16 - 
             (348456*mckin**14*muG**2*q_cut**2)/mbkin**18 - (393336*mckin**16*muG**
                2*q_cut**2)/mbkin**20 - (15120*mckin**18*muG**2*q_cut**2)/mbkin**22 - 
             (6720*mckin**2*muG*mupi*q_cut**2)/mbkin**6 - (6720*mckin**4*muG*mupi*q_cut**
                2)/mbkin**8 + (248520*mckin**6*muG*mupi*q_cut**2)/mbkin**10 + 
             (119976*mckin**8*muG*mupi*q_cut**2)/mbkin**12 - 
             (1214544*mckin**10*muG*mupi*q_cut**2)/mbkin**14 - 
             (1123824*mckin**12*muG*mupi*q_cut**2)/mbkin**16 + 
             (113256*mckin**14*muG*mupi*q_cut**2)/mbkin**18 + 
             (191736*mckin**16*muG*mupi*q_cut**2)/mbkin**20 + 
             (15120*mckin**18*muG*mupi*q_cut**2)/mbkin**22 + 
             (20160*mckin**2*muG**2*q_cut**3)/mbkin**8 + (78960*mckin**4*muG**2*q_cut**3)/
              mbkin**10 - (519000*mckin**6*muG**2*q_cut**3)/mbkin**12 - 
             (2174256*mckin**8*muG**2*q_cut**3)/mbkin**14 - (3033984*mckin**10*muG**
                2*q_cut**3)/mbkin**16 - (1941744*mckin**12*muG**2*q_cut**3)/mbkin**18 - 
             (453816*mckin**14*muG**2*q_cut**3)/mbkin**20 - (15120*mckin**16*muG**2*q_cut**
                3)/mbkin**22 - (6720*mckin**2*muG*mupi*q_cut**3)/mbkin**8 - 
             (11760*mckin**4*muG*mupi*q_cut**3)/mbkin**10 + 
             (290520*mckin**6*muG*mupi*q_cut**3)/mbkin**12 + 
             (931056*mckin**8*muG*mupi*q_cut**3)/mbkin**14 + 
             (1233024*mckin**10*muG*mupi*q_cut**3)/mbkin**16 + 
             (900144*mckin**12*muG*mupi*q_cut**3)/mbkin**18 + 
             (252216*mckin**14*muG*mupi*q_cut**3)/mbkin**20 + 
             (15120*mckin**16*muG*mupi*q_cut**3)/mbkin**22 - 
             (30240*mckin**2*muG**2*q_cut**4)/mbkin**10 - (98280*mckin**4*muG**2*q_cut**4)/
              mbkin**12 + (837600*mckin**6*muG**2*q_cut**4)/mbkin**14 + 
             (2690664*mckin**8*muG**2*q_cut**4)/mbkin**16 + (2414472*mckin**10*muG**
                2*q_cut**4)/mbkin**18 + (657504*mckin**12*muG**2*q_cut**4)/mbkin**20 + 
             (22680*mckin**14*muG**2*q_cut**4)/mbkin**22 + (10080*mckin**2*muG*mupi*
               q_cut**4)/mbkin**10 + (17640*mckin**4*muG*mupi*q_cut**4)/mbkin**12 - 
             (414240*mckin**6*muG*mupi*q_cut**4)/mbkin**14 - 
             (1249224*mckin**8*muG*mupi*q_cut**4)/mbkin**16 - 
             (1154472*mckin**10*muG*mupi*q_cut**4)/mbkin**18 - 
             (355104*mckin**12*muG*mupi*q_cut**4)/mbkin**20 - 
             (22680*mckin**14*muG*mupi*q_cut**4)/mbkin**22 + 
             (10080*mckin**2*muG**2*q_cut**5)/mbkin**12 + (22680*mckin**4*muG**2*q_cut**5)/
              mbkin**14 - (301608*mckin**6*muG**2*q_cut**5)/mbkin**16 - 
             (593472*mckin**8*muG**2*q_cut**5)/mbkin**18 - (208872*mckin**10*muG**2*q_cut**
                5)/mbkin**20 - (8568*mckin**12*muG**2*q_cut**5)/mbkin**22 - 
             (3360*mckin**2*muG*mupi*q_cut**5)/mbkin**12 - (2520*mckin**4*muG*mupi*
               q_cut**5)/mbkin**14 + (140328*mckin**6*muG*mupi*q_cut**5)/mbkin**16 + 
             (274272*mckin**8*muG*mupi*q_cut**5)/mbkin**18 + 
             (108072*mckin**10*muG*mupi*q_cut**5)/mbkin**20 + 
             (8568*mckin**12*muG*mupi*q_cut**5)/mbkin**22 + 
             (1128*mckin**6*muG**2*q_cut**6)/mbkin**18 - (1896*mckin**8*muG**2*q_cut**6)/
              mbkin**20 + (1008*mckin**10*muG**2*q_cut**6)/mbkin**22 - 
             (1128*mckin**6*muG*mupi*q_cut**6)/mbkin**18 + (1896*mckin**8*muG*mupi*
               q_cut**6)/mbkin**20 - (1008*mckin**10*muG*mupi*q_cut**6)/mbkin**22 + 
             (2160*mckin**4*muG**2*q_cut**7)/mbkin**18 + (3480*mckin**6*muG**2*q_cut**7)/
              mbkin**20 + (2160*mckin**8*muG**2*q_cut**7)/mbkin**22 - 
             (2160*mckin**4*muG*mupi*q_cut**7)/mbkin**18 - (3480*mckin**6*muG*mupi*
               q_cut**7)/mbkin**20 - (2160*mckin**8*muG*mupi*q_cut**7)/mbkin**22 - 
             (3420*mckin**4*muG**2*q_cut**8)/mbkin**20 - (3420*mckin**6*muG**2*q_cut**8)/
              mbkin**22 + (3420*mckin**4*muG*mupi*q_cut**8)/mbkin**20 + 
             (3420*mckin**6*muG*mupi*q_cut**8)/mbkin**22 + (1260*mckin**4*muG**2*q_cut**
                9)/mbkin**22 - (1260*mckin**4*muG*mupi*q_cut**9)/mbkin**22 + 
             24*mckin**2*muG*((-1 + mckin**2/mbkin**2)**2*(-280 + (565*mckin**2)/
                  mbkin**2 + (16341*mckin**4)/mbkin**4 - (28921*mckin**6)/
                  mbkin**6 - (206105*mckin**8)/mbkin**8 - (182445*mckin**10)/
                  mbkin**10 - (23125*mckin**12)/mbkin**12 + (7549*mckin**14)/
                  mbkin**14 + (621*mckin**16)/mbkin**16) + ((840 - (845*mckin**2)/
                   mbkin**2 - (42526*mckin**4)/mbkin**4 + (23640*mckin**6)/
                   mbkin**6 + (480586*mckin**8)/mbkin**8 + (779150*mckin**10)/
                   mbkin**10 + (416970*mckin**12)/mbkin**12 + (30136*mckin**14)/
                   mbkin**14 - (22870*mckin**16)/mbkin**16 - (1881*mckin**18)/
                   mbkin**18)*q_cut)/mbkin**2 + (2*(-280 - (280*mckin**2)/mbkin**2 + 
                  (10355*mckin**4)/mbkin**4 + (4999*mckin**6)/mbkin**6 - 
                  (50606*mckin**8)/mbkin**8 - (46826*mckin**10)/mbkin**10 + 
                  (4719*mckin**12)/mbkin**12 + (7989*mckin**14)/mbkin**14 + 
                  (630*mckin**16)/mbkin**16)*q_cut**2)/mbkin**4 + 
               (2*(-280 - (490*mckin**2)/mbkin**2 + (12105*mckin**4)/mbkin**4 + 
                  (38794*mckin**6)/mbkin**6 + (51376*mckin**8)/mbkin**8 + 
                  (37506*mckin**10)/mbkin**10 + (10509*mckin**12)/mbkin**12 + 
                  (630*mckin**14)/mbkin**14)*q_cut**3)/mbkin**6 - 
               (2*(-420 - (735*mckin**2)/mbkin**2 + (17260*mckin**4)/mbkin**4 + 
                  (52051*mckin**6)/mbkin**6 + (48103*mckin**8)/mbkin**8 + 
                  (14796*mckin**10)/mbkin**10 + (945*mckin**12)/mbkin**12)*q_cut**4)/
                mbkin**8 + (2*(-140 - (105*mckin**2)/mbkin**2 + (5847*mckin**4)/
                   mbkin**4 + (11428*mckin**6)/mbkin**6 + (4503*mckin**8)/
                   mbkin**8 + (357*mckin**10)/mbkin**10)*q_cut**5)/mbkin**10 - 
               (2*mckin**4*(47 - (79*mckin**2)/mbkin**2 + (42*mckin**4)/mbkin**4)*
                 q_cut**6)/mbkin**16 - (10*mckin**2*(18 + (29*mckin**2)/mbkin**2 + 
                  (18*mckin**4)/mbkin**4)*q_cut**7)/mbkin**16 + (285*mckin**2*
                 (mbkin**2 + mckin**2)*q_cut**8)/mbkin**20 - (105*mckin**2*q_cut**9)/
                mbkin**20) + 16*(-((-1 + mckin**2/mbkin**2)**2*(-137 - 
                  (3431*mckin**2)/mbkin**2 + (22624*mckin**4)/mbkin**4 + 
                  (94372*mckin**6)/mbkin**6 - (119450*mckin**8)/mbkin**8 - 
                  (345242*mckin**10)/mbkin**10 - (152798*mckin**12)/mbkin**12 - 
                  (2588*mckin**14)/mbkin**14 + (7069*mckin**16)/mbkin**16 + 
                  (621*mckin**18)/mbkin**18)) + ((-417 - (10988*mckin**2)/
                   mbkin**2 + (37288*mckin**4)/mbkin**4 + (239041*mckin**6)/
                   mbkin**6 - (38168*mckin**8)/mbkin**8 - (928862*mckin**10)/
                   mbkin**10 - (1005470*mckin**12)/mbkin**12 - (325136*mckin**14)/
                   mbkin**14 + (13966*mckin**16)/mbkin**16 + (21025*mckin**18)/
                   mbkin**18 + (1881*mckin**20)/mbkin**20)*q_cut)/mbkin**2 - 
               (2*(-140 - (3993*mckin**2)/mbkin**2 + (1395*mckin**4)/mbkin**4 + 
                  (30919*mckin**6)/mbkin**6 - (27489*mckin**8)/mbkin**8 - 
                  (89157*mckin**10)/mbkin**10 - (21743*mckin**12)/mbkin**12 + 
                  (18639*mckin**14)/mbkin**14 + (7779*mckin**16)/mbkin**16 + 
                  (630*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 - 
               ((-280 - (8826*mckin**2)/mbkin**2 - (20363*mckin**4)/mbkin**4 + 
                  (14773*mckin**6)/mbkin**6 + (58659*mckin**8)/mbkin**8 + 
                  (91223*mckin**10)/mbkin**10 + (63057*mckin**12)/mbkin**12 + 
                  (16713*mckin**14)/mbkin**14 + (1260*mckin**16)/mbkin**16)*q_cut**3)/
                mbkin**6 + (2*(-210 - (6612*mckin**2)/mbkin**2 - (14884*mckin**4)/
                   mbkin**4 + (18675*mckin**6)/mbkin**6 + (61863*mckin**8)/
                   mbkin**8 + (46048*mckin**10)/mbkin**10 + (12381*mckin**12)/
                   mbkin**12 + (945*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
               ((168 + (4164*mckin**2)/mbkin**2 + (5451*mckin**4)/mbkin**4 - 
                  (18857*mckin**6)/mbkin**6 - (24019*mckin**8)/mbkin**8 - 
                  (7641*mckin**10)/mbkin**10 - (714*mckin**12)/mbkin**12)*q_cut**5)/
                mbkin**10 + (2*(-14 + (17*mckin**2)/mbkin**2 - (60*mckin**4)/
                   mbkin**4 + mckin**6/mbkin**6 + (26*mckin**8)/mbkin**8 + 
                  (42*mckin**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
               (5*(-12 - (10*mckin**2)/mbkin**2 + (5*mckin**4)/mbkin**4 + 
                  (25*mckin**6)/mbkin**6 + (36*mckin**8)/mbkin**8)*q_cut**7)/
                mbkin**14 - (5*(-19 - (19*mckin**2)/mbkin**2 + (45*mckin**4)/
                   mbkin**4 + (57*mckin**6)/mbkin**6)*q_cut**8)/mbkin**16 - 
               (35*(mbkin**4 - 3*mckin**4)*q_cut**9)/mbkin**22)*rG - 
             8*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(-219 - (3427*mckin**2)/
                   mbkin**2 - (25220*mckin**4)/mbkin**4 + (690492*mckin**6)/
                   mbkin**6 + (3385394*mckin**8)/mbkin**8 + (3853498*mckin**10)/
                   mbkin**10 + (969876*mckin**12)/mbkin**12 - (66556*mckin**14)/
                   mbkin**14 + (8017*mckin**16)/mbkin**16 + (3105*mckin**18)/
                   mbkin**18)) + ((-639 - (12606*mckin**2)/mbkin**2 - 
                  (137707*mckin**4)/mbkin**4 + (1200772*mckin**6)/mbkin**6 + 
                  (8153946*mckin**8)/mbkin**8 + (14790212*mckin**10)/mbkin**10 + 
                  (9837754*mckin**12)/mbkin**12 + (1656240*mckin**14)/mbkin**14 - 
                  (260839*mckin**16)/mbkin**16 + (23302*mckin**18)/mbkin**18 + 
                  (9405*mckin**20)/mbkin**20)*q_cut)/mbkin**2 - (2*(-210 - 
                  (4891*mckin**2)/mbkin**2 - (70825*mckin**4)/mbkin**4 + 
                  (10961*mckin**6)/mbkin**6 + (726865*mckin**8)/mbkin**8 + 
                  (829165*mckin**10)/mbkin**10 + (50797*mckin**12)/mbkin**12 - 
                  (87657*mckin**14)/mbkin**14 + (11805*mckin**16)/mbkin**16 + 
                  (3150*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
               (2*(210 + (5521*mckin**2)/mbkin**2 + (90958*mckin**4)/mbkin**4 + 
                  (297789*mckin**6)/mbkin**6 + (370334*mckin**8)/mbkin**8 + 
                  (299309*mckin**10)/mbkin**10 + (83742*mckin**12)/mbkin**12 - 
                  (15585*mckin**14)/mbkin**14 - (3150*mckin**16)/mbkin**16)*q_cut**3)/
                mbkin**6 + (2*(-315 - (8304*mckin**2)/mbkin**2 - 
                  (138468*mckin**4)/mbkin**4 - (460673*mckin**6)/mbkin**6 - 
                  (479693*mckin**8)/mbkin**8 - (125948*mckin**10)/mbkin**10 + 
                  (23580*mckin**12)/mbkin**12 + (4725*mckin**14)/mbkin**14)*q_cut**4)/
                mbkin**8 + ((126 + (5638*mckin**2)/mbkin**2 + (85014*mckin**4)/
                   mbkin**4 + (204504*mckin**6)/mbkin**6 + (84214*mckin**8)/
                   mbkin**8 - (13278*mckin**10)/mbkin**10 - (3570*mckin**12)/
                   mbkin**12)*q_cut**5)/mbkin**10 + (2*(42 - (51*mckin**2)/mbkin**2 + 
                  (159*mckin**4)/mbkin**4 - (541*mckin**6)/mbkin**6 - 
                  (31*mckin**8)/mbkin**8 + (210*mckin**10)/mbkin**10)*q_cut**6)/
                mbkin**12 + (10*(18 + (15*mckin**2)/mbkin**2 + (78*mckin**4)/
                   mbkin**4 + (17*mckin**6)/mbkin**6 + (90*mckin**8)/mbkin**8)*
                 q_cut**7)/mbkin**14 - (15*(19 + (19*mckin**2)/mbkin**2 + 
                  (47*mckin**4)/mbkin**4 + (95*mckin**6)/mbkin**6)*q_cut**8)/
                mbkin**16 + (105*(mbkin**4 + 5*mckin**4)*q_cut**9)/mbkin**22)*rhoD - 
             3360*sB - (85120*mckin**2*sB)/mbkin**2 + (548568*mckin**4*sB)/
              mbkin**4 + (4992264*mckin**6*sB)/mbkin**6 + (848880*mckin**8*sB)/
              mbkin**8 - (11177712*mckin**10*sB)/mbkin**10 - 
             (463680*mckin**12*sB)/mbkin**12 + (3575712*mckin**14*sB)/mbkin**14 + 
             (1418256*mckin**16*sB)/mbkin**16 + (526800*mckin**18*sB)/mbkin**18 - 
             (155768*mckin**20*sB)/mbkin**20 - (24840*mckin**22*sB)/mbkin**22 + 
             (10080*q_cut*sB)/mbkin**2 + (299040*mckin**2*q_cut*sB)/mbkin**4 - 
             (169448*mckin**4*q_cut*sB)/mbkin**6 - (12282208*mckin**6*q_cut*sB)/
              mbkin**8 - (36742752*mckin**8*q_cut*sB)/mbkin**10 - 
             (39176624*mckin**10*q_cut*sB)/mbkin**12 - (17580784*mckin**12*q_cut*sB)/
              mbkin**14 - (4143696*mckin**14*q_cut*sB)/mbkin**16 + 
             (304*mckin**16*q_cut*sB)/mbkin**18 + (604928*mckin**18*q_cut*sB)/
              mbkin**20 + (75240*mckin**20*q_cut*sB)/mbkin**22 - (6720*q_cut**2*sB)/
              mbkin**4 - (217280*mckin**2*q_cut**2*sB)/mbkin**6 - 
             (600320*mckin**4*q_cut**2*sB)/mbkin**8 + (2179984*mckin**6*q_cut**2*sB)/
              mbkin**10 + (5879888*mckin**8*q_cut**2*sB)/mbkin**12 + 
             (2959328*mckin**10*q_cut**2*sB)/mbkin**14 + (38624*mckin**12*q_cut**2*sB)/
              mbkin**16 - (639984*mckin**14*q_cut**2*sB)/mbkin**18 - 
             (450960*mckin**16*q_cut**2*sB)/mbkin**20 - (50400*mckin**18*q_cut**2*sB)/
              mbkin**22 - (6720*q_cut**3*sB)/mbkin**6 - (237440*mckin**2*q_cut**3*sB)/
              mbkin**8 - (1361360*mckin**4*q_cut**3*sB)/mbkin**10 - 
             (2607456*mckin**6*q_cut**3*sB)/mbkin**12 - (3283120*mckin**8*q_cut**3*sB)/
              mbkin**14 - (3012208*mckin**10*q_cut**3*sB)/mbkin**16 - 
             (1675344*mckin**12*q_cut**3*sB)/mbkin**18 - (546720*mckin**14*q_cut**3*sB)/
              mbkin**20 - (50400*mckin**16*q_cut**3*sB)/mbkin**22 + 
             (10080*q_cut**4*sB)/mbkin**8 + (356160*mckin**2*q_cut**4*sB)/mbkin**10 + 
             (2078160*mckin**4*q_cut**4*sB)/mbkin**12 + (4039744*mckin**6*q_cut**4*sB)/
              mbkin**14 + (3825136*mckin**8*q_cut**4*sB)/mbkin**16 + 
             (2284336*mckin**10*q_cut**4*sB)/mbkin**18 + (790560*mckin**12*q_cut**4*sB)/
              mbkin**20 + (75600*mckin**14*q_cut**4*sB)/mbkin**22 - 
             (3360*q_cut**5*sB)/mbkin**10 - (115360*mckin**2*q_cut**5*sB)/mbkin**12 - 
             (580608*mckin**4*q_cut**5*sB)/mbkin**14 - (801120*mckin**6*q_cut**5*sB)/
              mbkin**16 - (572464*mckin**8*q_cut**5*sB)/mbkin**18 - 
             (238272*mckin**10*q_cut**5*sB)/mbkin**20 - (28560*mckin**12*q_cut**5*sB)/
              mbkin**22 - (672*mckin**4*q_cut**6*sB)/mbkin**16 - 
             (208*mckin**6*q_cut**6*sB)/mbkin**18 - (1168*mckin**8*q_cut**6*sB)/
              mbkin**20 + (3360*mckin**10*q_cut**6*sB)/mbkin**22 + 
             (2640*mckin**4*q_cut**7*sB)/mbkin**18 + (5920*mckin**6*q_cut**7*sB)/
              mbkin**20 + (7200*mckin**8*q_cut**7*sB)/mbkin**22 - 
             (8520*mckin**4*q_cut**8*sB)/mbkin**20 - (11400*mckin**6*q_cut**8*sB)/
              mbkin**22 + (4200*mckin**4*q_cut**9*sB)/mbkin**22 + 1656*sE + 
             (57736*mckin**2*sE)/mbkin**2 - (366072*mckin**4*sE)/mbkin**4 - 
             (2250888*mckin**6*sE)/mbkin**6 + (2070480*mckin**8*sE)/mbkin**8 + 
             (4465776*mckin**10*sE)/mbkin**10 - (4114320*mckin**12*sE)/
              mbkin**12 - (461808*mckin**14*sE)/mbkin**14 + (747768*mckin**16*sE)/
              mbkin**16 - (241080*mckin**18*sE)/mbkin**18 + (75848*mckin**20*sE)/
              mbkin**20 + (14904*mckin**22*sE)/mbkin**22 - (5016*q_cut*sE)/
              mbkin**2 - (192624*mckin**2*q_cut*sE)/mbkin**4 + 
             (233864*mckin**4*q_cut*sE)/mbkin**6 + (6234112*mckin**6*q_cut*sE)/
              mbkin**8 + (12579408*mckin**8*q_cut*sE)/mbkin**10 + 
             (5003168*mckin**10*q_cut*sE)/mbkin**12 - (2810288*mckin**12*q_cut*sE)/
              mbkin**14 - (826176*mckin**14*q_cut*sE)/mbkin**16 - 
             (11704*mckin**16*q_cut*sE)/mbkin**18 - (322160*mckin**18*q_cut*sE)/
              mbkin**20 - (45144*mckin**20*q_cut*sE)/mbkin**22 + (3360*q_cut**2*sE)/
              mbkin**4 + (136688*mckin**2*q_cut**2*sE)/mbkin**6 + 
             (228560*mckin**4*q_cut**2*sE)/mbkin**8 - (1394704*mckin**6*q_cut**2*sE)/
              mbkin**10 - (1962128*mckin**8*q_cut**2*sE)/mbkin**12 + 
             (667504*mckin**10*q_cut**2*sE)/mbkin**14 + (1108336*mckin**12*q_cut**2*sE)/
              mbkin**16 + (576912*mckin**14*q_cut**2*sE)/mbkin**18 + 
             (282672*mckin**16*q_cut**2*sE)/mbkin**20 + (30240*mckin**18*q_cut**2*sE)/
              mbkin**22 + (3360*q_cut**3*sE)/mbkin**6 + (146768*mckin**2*q_cut**3*sE)/
              mbkin**8 + (662144*mckin**4*q_cut**3*sE)/mbkin**10 + 
             (764016*mckin**6*q_cut**3*sE)/mbkin**12 + (452416*mckin**8*q_cut**3*sE)/
              mbkin**14 + (42352*mckin**10*q_cut**3*sE)/mbkin**16 + 
             (158208*mckin**12*q_cut**3*sE)/mbkin**18 + (225552*mckin**14*q_cut**3*sE)/
              mbkin**20 + (30240*mckin**16*q_cut**3*sE)/mbkin**22 - 
             (5040*q_cut**4*sE)/mbkin**8 - (220032*mckin**2*q_cut**4*sE)/mbkin**10 - 
             (1006464*mckin**4*q_cut**4*sE)/mbkin**12 - (1220560*mckin**6*q_cut**4*sE)/
              mbkin**14 - (631504*mckin**8*q_cut**4*sE)/mbkin**16 - 
             (590464*mckin**10*q_cut**4*sE)/mbkin**18 - (380928*mckin**12*q_cut**4*sE)/
              mbkin**20 - (45360*mckin**14*q_cut**4*sE)/mbkin**22 + 
             (1904*q_cut**5*sE)/mbkin**10 + (70832*mckin**2*q_cut**5*sE)/mbkin**12 + 
             (271248*mckin**4*q_cut**5*sE)/mbkin**14 + (203488*mckin**6*q_cut**5*sE)/
              mbkin**16 + (145296*mckin**8*q_cut**5*sE)/mbkin**18 + 
             (118704*mckin**10*q_cut**5*sE)/mbkin**20 + (17136*mckin**12*q_cut**5*sE)/
              mbkin**22 - (224*q_cut**6*sE)/mbkin**12 + (272*mckin**2*q_cut**6*sE)/
              mbkin**14 - (400*mckin**4*q_cut**6*sE)/mbkin**16 + 
             (3632*mckin**6*q_cut**6*sE)/mbkin**18 - (5168*mckin**8*q_cut**6*sE)/
              mbkin**20 - (2016*mckin**10*q_cut**6*sE)/mbkin**22 - 
             (480*q_cut**7*sE)/mbkin**14 - (400*mckin**2*q_cut**7*sE)/mbkin**16 - 
             (960*mckin**4*q_cut**7*sE)/mbkin**18 + (2480*mckin**6*q_cut**7*sE)/
              mbkin**20 - (4320*mckin**8*q_cut**7*sE)/mbkin**22 + (760*q_cut**8*sE)/
              mbkin**16 + (760*mckin**2*q_cut**8*sE)/mbkin**18 + 
             (3000*mckin**4*q_cut**8*sE)/mbkin**20 + (6840*mckin**6*q_cut**8*sE)/
              mbkin**22 - (280*q_cut**9*sE)/mbkin**18 - (2520*mckin**4*q_cut**9*sE)/
              mbkin**22 + 633*sqB + (9583*mckin**2*sqB)/mbkin**2 - 
             (76293*mckin**4*sqB)/mbkin**4 - (712899*mckin**6*sqB)/mbkin**6 - 
             (625266*mckin**8*sqB)/mbkin**8 + (1581930*mckin**10*sqB)/mbkin**10 + 
             (1241226*mckin**12*sqB)/mbkin**12 - (793362*mckin**14*sqB)/
              mbkin**14 - (595791*mckin**16*sqB)/mbkin**16 - 
             (42609*mckin**18*sqB)/mbkin**18 + (12227*mckin**20*sqB)/mbkin**20 + 
             (621*mckin**22*sqB)/mbkin**22 - (1893*q_cut*sqB)/mbkin**2 - 
             (37242*mckin**2*q_cut*sqB)/mbkin**4 + (26039*mckin**4*q_cut*sqB)/
              mbkin**6 + (1713892*mckin**6*q_cut*sqB)/mbkin**8 + 
             (6214110*mckin**8*q_cut*sqB)/mbkin**10 + (8766044*mckin**10*q_cut*sqB)/
              mbkin**12 + (5443102*mckin**12*q_cut*sqB)/mbkin**14 + 
             (1253136*mckin**14*q_cut*sqB)/mbkin**16 - (48277*mckin**16*q_cut*sqB)/
              mbkin**18 - (42230*mckin**18*q_cut*sqB)/mbkin**20 - 
             (1881*mckin**20*q_cut*sqB)/mbkin**22 + (1260*q_cut**2*sqB)/mbkin**4 + 
             (28274*mckin**2*q_cut**2*sqB)/mbkin**6 + (89030*mckin**4*q_cut**2*sqB)/
              mbkin**8 - (243478*mckin**6*q_cut**2*sqB)/mbkin**10 - 
             (1014998*mckin**8*q_cut**2*sqB)/mbkin**12 - (873326*mckin**10*q_cut**2*sqB)/
              mbkin**14 - (85358*mckin**12*q_cut**2*sqB)/mbkin**16 + 
             (124998*mckin**14*q_cut**2*sqB)/mbkin**18 + (31938*mckin**16*q_cut**2*sqB)/
              mbkin**20 + (1260*mckin**18*q_cut**2*sqB)/mbkin**22 + 
             (1260*q_cut**3*sqB)/mbkin**6 + (32054*mckin**2*q_cut**3*sqB)/mbkin**8 + 
             (190652*mckin**4*q_cut**3*sqB)/mbkin**10 + (411942*mckin**6*q_cut**3*sqB)/
              mbkin**12 + (470956*mckin**8*q_cut**3*sqB)/mbkin**14 + 
             (374566*mckin**10*q_cut**3*sqB)/mbkin**16 + (168732*mckin**12*q_cut**3*sqB)/
              mbkin**18 + (30258*mckin**14*q_cut**3*sqB)/mbkin**20 + 
             (1260*mckin**16*q_cut**3*sqB)/mbkin**22 - (1890*q_cut**4*sqB)/mbkin**8 - 
             (48096*mckin**2*q_cut**4*sqB)/mbkin**10 - (293352*mckin**4*q_cut**4*sqB)/
              mbkin**12 - (657526*mckin**6*q_cut**4*sqB)/mbkin**14 - 
             (644398*mckin**8*q_cut**4*sqB)/mbkin**16 - (282136*mckin**10*q_cut**4*sqB)/
              mbkin**18 - (48072*mckin**12*q_cut**4*sqB)/mbkin**20 - 
             (1890*mckin**14*q_cut**4*sqB)/mbkin**22 + (602*q_cut**5*sqB)/mbkin**10 + 
             (15506*mckin**2*q_cut**5*sqB)/mbkin**12 + (82266*mckin**4*q_cut**5*sqB)/
              mbkin**14 + (138448*mckin**6*q_cut**5*sqB)/mbkin**16 + 
             (82458*mckin**8*q_cut**5*sqB)/mbkin**18 + (15726*mckin**10*q_cut**5*sqB)/
              mbkin**20 + (714*mckin**12*q_cut**5*sqB)/mbkin**22 + 
             (28*q_cut**6*sqB)/mbkin**12 - (34*mckin**2*q_cut**6*sqB)/mbkin**14 + 
             (218*mckin**4*q_cut**6*sqB)/mbkin**16 - (286*mckin**6*q_cut**6*sqB)/
              mbkin**18 - (122*mckin**8*q_cut**6*sqB)/mbkin**20 - 
             (84*mckin**10*q_cut**6*sqB)/mbkin**22 + (60*q_cut**7*sqB)/mbkin**14 + 
             (50*mckin**2*q_cut**7*sqB)/mbkin**16 + (540*mckin**4*q_cut**7*sqB)/
              mbkin**18 + (230*mckin**6*q_cut**7*sqB)/mbkin**20 - 
             (180*mckin**8*q_cut**7*sqB)/mbkin**22 - (95*q_cut**8*sqB)/mbkin**16 - 
             (95*mckin**2*q_cut**8*sqB)/mbkin**18 + (45*mckin**4*q_cut**8*sqB)/
              mbkin**20 + (285*mckin**6*q_cut**8*sqB)/mbkin**22 + (35*q_cut**9*sqB)/
              mbkin**18 - (105*mckin**4*q_cut**9*sqB)/mbkin**22))*
          np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/mbkin**4 - 
        (60480*mckin**8*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
             2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          (48*mckin**2*muG + (144*mckin**4*muG)/mbkin**2 - (1152*mckin**6*muG)/
            mbkin**4 - (2280*mckin**8*muG)/mbkin**6 - (720*mckin**10*muG)/
            mbkin**8 - (24*mckin**2*muG**2)/mbkin**2 - (72*mckin**4*muG**2)/
            mbkin**4 + (576*mckin**6*muG**2)/mbkin**6 + (1140*mckin**8*muG**2)/
            mbkin**8 + (360*mckin**10*muG**2)/mbkin**10 + (24*mckin**2*muG*mupi)/
            mbkin**2 + (72*mckin**4*muG*mupi)/mbkin**4 - (576*mckin**6*muG*mupi)/
            mbkin**6 - (1140*mckin**8*muG*mupi)/mbkin**8 - 
           (360*mckin**10*muG*mupi)/mbkin**10 - 16*(-4 - (52*mckin**2)/mbkin**2 + 
             (135*mckin**4)/mbkin**4 + (412*mckin**6)/mbkin**6 + 
             (88*mckin**8)/mbkin**8)*rE + 8*(1 - (37*mckin**2)/mbkin**2 - 
             (42*mckin**4)/mbkin**4 + (224*mckin**6)/mbkin**6 + 
             (211*mckin**8)/mbkin**8 + (39*mckin**10)/mbkin**10)*rG + 
           48*mbkin*rhoD + (592*mckin**2*rhoD)/mbkin + (5592*mckin**4*rhoD)/
            mbkin**3 + (13536*mckin**6*rhoD)/mbkin**5 + (7592*mckin**8*rhoD)/
            mbkin**7 + (624*mckin**10*rhoD)/mbkin**9 + 24*sB + 
           (824*mckin**2*sB)/mbkin**2 + (3984*mckin**4*sB)/mbkin**4 + 
           (4320*mckin**6*sB)/mbkin**6 + (1360*mckin**8*sB)/mbkin**8 + 
           (312*mckin**10*sB)/mbkin**10 - (416*mckin**2*sE)/mbkin**2 - 
           (1656*mckin**4*sE)/mbkin**4 - (480*mckin**6*sE)/mbkin**6 + 
           (584*mckin**8*sE)/mbkin**8 - 6*sqB - (122*mckin**2*sqB)/mbkin**2 - 
           (609*mckin**4*sqB)/mbkin**4 - (972*mckin**6*sqB)/mbkin**6 - 
           (523*mckin**8*sqB)/mbkin**8 - (78*mckin**10*sqB)/mbkin**10)*
          np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                   mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                 mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**8))/
      (1260*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
          2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
       ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
            7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
            mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
         (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
              mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**4)**
        3) + 
     (api4*((mbkin**4*(18*mbkin**4*((-8*(mbkin - mckin)**2*(mbkin + mckin)*(
                3*mbkin + 3*mckin + 8*mbkin*mckin))/(9*mbkin**6) + 
             (16*(4*mbkin**3 - 4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*
               q_cut)/(9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*
            ((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/mbkin**2 - 
                (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6) + 
              ((-3 + (14*mckin**2)/mbkin**2 + (26*mckin**4)/mbkin**4 + 
                 (14*mckin**6)/mbkin**6 - (3*mckin**8)/mbkin**8)*q_cut)/mbkin**2 + 
              (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 2*mbkin**2*mckin**4 + mckin**6)*
                q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*
                q_cut**3)/mbkin**10 - (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + 
              q_cut**5/mbkin**10)**2*(3*(1 - (47*mckin**2)/mbkin**2 - (2595*mckin**4)/
                mbkin**4 - (11219*mckin**6)/mbkin**6 - (11219*mckin**8)/mbkin**8 - 
               (2595*mckin**10)/mbkin**10 - (47*mckin**12)/mbkin**12 + mckin**14/
                mbkin**14) + (3*(mbkin**12 - 44*mbkin**10*mckin**2 - 
                1051*mbkin**8*mckin**4 - 2432*mbkin**6*mckin**6 - 1051*mbkin**4*
                 mckin**8 - 44*mbkin**2*mckin**10 + mckin**12)*q_cut)/mbkin**14 + 
             (3*(mbkin**10 - 39*mbkin**8*mckin**2 - 424*mbkin**6*mckin**4 - 
                424*mbkin**4*mckin**6 - 39*mbkin**2*mckin**8 + mckin**10)*q_cut**2)/
              mbkin**14 + (3*(mbkin**8 - 32*mbkin**6*mckin**2 - 136*mbkin**4*
                 mckin**4 - 32*mbkin**2*mckin**6 + mckin**8)*q_cut**3)/mbkin**14 + 
             (3*(mbkin**6 - 23*mbkin**4*mckin**2 - 23*mbkin**2*mckin**4 + mckin**6)*
               q_cut**4)/mbkin**14 - (5*(5 - (4*mckin**2)/mbkin**2 + (5*mckin**4)/
                 mbkin**4)*q_cut**5)/mbkin**10 - (25*(mbkin**2 + mckin**2)*q_cut**6)/
              mbkin**14 + (35*q_cut**7)/mbkin**14) + ((-1 + mckin**2/mbkin**2)**2 - 
             (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
            ((64*mbkin*(-4*(-1 + mckin**2/mbkin**2)**4*(1 + mckin**2/mbkin**2)**2*
                 (1179 - (27994*mckin**2)/mbkin**2 + (151119*mckin**4)/mbkin**4 + 
                  (982305*mckin**6)/mbkin**6 - (6893010*mckin**8)/mbkin**8 - 
                  (8620596*mckin**10)/mbkin**10 + (1136781*mckin**12)/mbkin**12 + 
                  (171093*mckin**14)/mbkin**14 - (137043*mckin**16)/mbkin**16 + 
                  (13572*mckin**18)/mbkin**18 + (154*mckin**20)/mbkin**20) + 
                ((-1 + mckin**2/mbkin**2)**2*(26163 - (491315*mckin**2)/mbkin**2 + 
                   (609844*mckin**4)/mbkin**4 + (29190880*mckin**6)/mbkin**6 - 
                   (45147167*mckin**8)/mbkin**8 - (424453401*mckin**10)/
                    mbkin**10 - (729940536*mckin**12)/mbkin**12 - 
                   (481708704*mckin**14)/mbkin**14 - (67588491*mckin**16)/
                    mbkin**16 + (33128619*mckin**18)/mbkin**18 - 
                   (4511740*mckin**20)/mbkin**20 - (1905472*mckin**22)/
                    mbkin**22 + (315767*mckin**24)/mbkin**24 + (3233*mckin**26)/
                    mbkin**26)*q_cut)/mbkin**2 + (4*(-12321 + (196421*mckin**2)/
                    mbkin**2 + (301638*mckin**4)/mbkin**4 - (12526887*mckin**6)/
                    mbkin**6 + (1399932*mckin**8)/mbkin**8 + (127033574*
                     mckin**10)/mbkin**10 + (284550615*mckin**12)/mbkin**12 + 
                   (300700218*mckin**14)/mbkin**14 + (150895899*mckin**16)/
                    mbkin**16 + (5517729*mckin**18)/mbkin**18 - (15244856*
                     mckin**20)/mbkin**20 + (2907981*mckin**22)/mbkin**22 + 
                   (671466*mckin**24)/mbkin**24 - (153900*mckin**26)/mbkin**26 - 
                   (1349*mckin**28)/mbkin**28)*q_cut**2)/mbkin**4 + 
                ((14763 - (64787*mckin**2)/mbkin**2 - (2819568*mckin**4)/
                    mbkin**4 + (7404252*mckin**6)/mbkin**6 + (43250901*mckin**8)/
                    mbkin**8 - (6393169*mckin**10)/mbkin**10 - (79307576*
                     mckin**12)/mbkin**12 - (4041216*mckin**14)/mbkin**14 + 
                   (48737885*mckin**16)/mbkin**16 + (556627*mckin**18)/
                    mbkin**18 - (8219224*mckin**20)/mbkin**20 + 
                   (656612*mckin**22)/mbkin**22 + (224419*mckin**24)/mbkin**24 + 
                   (81*mckin**26)/mbkin**26)*q_cut**3)/mbkin**6 + 
                (4*(18273 - (235702*mckin**2)/mbkin**2 - (1086112*mckin**4)/
                    mbkin**4 + (11266643*mckin**6)/mbkin**6 + (36598184*mckin**8)/
                    mbkin**8 + (45843340*mckin**10)/mbkin**10 + (44974590*
                     mckin**12)/mbkin**12 + (36547042*mckin**14)/mbkin**14 + 
                   (10361895*mckin**16)/mbkin**16 - (3866986*mckin**18)/
                    mbkin**18 - (424058*mckin**20)/mbkin**20 + (224799*mckin**22)/
                    mbkin**22 + (2444*mckin**24)/mbkin**24)*q_cut**4)/mbkin**8 - 
                ((98721 - (917479*mckin**2)/mbkin**2 - (9760513*mckin**4)/
                    mbkin**4 + (30195627*mckin**6)/mbkin**6 + (170463350*
                     mckin**8)/mbkin**8 + (259417590*mckin**10)/mbkin**10 + 
                   (167078226*mckin**12)/mbkin**12 + (15469354*mckin**14)/
                    mbkin**14 - (21555727*mckin**16)/mbkin**16 + 
                   (973161*mckin**18)/mbkin**18 + (1273159*mckin**20)/mbkin**20 + 
                   (10643*mckin**22)/mbkin**22)*q_cut**5)/mbkin**10 + 
                (4*(5934 - (43901*mckin**2)/mbkin**2 - (695976*mckin**4)/
                    mbkin**4 + (2309184*mckin**6)/mbkin**6 + (11870802*mckin**8)/
                    mbkin**8 + (11318680*mckin**10)/mbkin**10 + 
                   (724070*mckin**12)/mbkin**12 - (1728188*mckin**14)/mbkin**14 + 
                   (119948*mckin**16)/mbkin**16 + (74017*mckin**18)/mbkin**18 + 
                   (526*mckin**20)/mbkin**20)*q_cut**6)/mbkin**12 + 
                ((44175 - (243147*mckin**2)/mbkin**2 - (5579352*mckin**4)/
                    mbkin**4 - (17016744*mckin**6)/mbkin**6 - (22924990*mckin**8)/
                    mbkin**8 - (14689794*mckin**10)/mbkin**10 - 
                   (2030576*mckin**12)/mbkin**12 + (2336816*mckin**14)/
                    mbkin**14 + (572671*mckin**16)/mbkin**16 + (6717*mckin**18)/
                    mbkin**18)*q_cut**7)/mbkin**14 - (4*(11646 - (53280*mckin**2)/
                    mbkin**2 - (1304004*mckin**4)/mbkin**4 - (3516303*mckin**6)/
                    mbkin**6 - (3269188*mckin**8)/mbkin**8 - (647651*mckin**10)/
                    mbkin**10 + (523431*mckin**12)/mbkin**12 + (121214*mckin**14)/
                    mbkin**14 + (3663*mckin**16)/mbkin**16)*q_cut**8)/mbkin**16 + 
                ((24633 - (76675*mckin**2)/mbkin**2 - (1532353*mckin**4)/
                    mbkin**4 - (3216569*mckin**6)/mbkin**6 - (1176415*mckin**8)/
                    mbkin**8 + (448897*mckin**10)/mbkin**10 + (145151*mckin**12)/
                    mbkin**12 + (15283*mckin**14)/mbkin**14)*q_cut**9)/mbkin**18 - 
                (4*(1197 - (4127*mckin**2)/mbkin**2 - (21706*mckin**4)/mbkin**4 - 
                   (39039*mckin**6)/mbkin**6 - (14018*mckin**8)/mbkin**8 + 
                   (2910*mckin**10)/mbkin**10 + (449*mckin**12)/mbkin**12)*q_cut**10)/
                 mbkin**20 - ((8559 + (20945*mckin**2)/mbkin**2 + 
                   (29208*mckin**4)/mbkin**4 + (54804*mckin**6)/mbkin**6 + 
                   (34199*mckin**8)/mbkin**8 + (11197*mckin**10)/mbkin**10)*
                  q_cut**11)/mbkin**22 + (4*(2121 + (4502*mckin**2)/mbkin**2 + 
                   (8820*mckin**4)/mbkin**4 + (9231*mckin**6)/mbkin**6 + 
                   (2526*mckin**8)/mbkin**8)*q_cut**12)/mbkin**24 - 
                ((2379 + (5003*mckin**2)/mbkin**2 + (11693*mckin**4)/mbkin**4 + 
                   (3329*mckin**6)/mbkin**6)*q_cut**13)/mbkin**26 + 
                (4*mckin**2*(137 + (104*mckin**2)/mbkin**2)*q_cut**14)/mbkin**30 - 
                (5*(15 + (29*mckin**2)/mbkin**2)*q_cut**15)/mbkin**30 + 
                (60*q_cut**16)/mbkin**32))/9 + 18*(mbkin**4*
                ((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/mbkin**2 - 
                    (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6) + 
                  ((-3 + (14*mckin**2)/mbkin**2 + (26*mckin**4)/mbkin**4 + 
                     (14*mckin**6)/mbkin**6 - (3*mckin**8)/mbkin**8)*q_cut)/
                   mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 2*mbkin**2*
                      mckin**4 + mckin**6)*q_cut**2)/mbkin**10 + 
                  (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*q_cut**3)/mbkin**10 - 
                  (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2*
                ((-4*(141*mbkin**14 + 376*mbkin**14*mckin + 15429*mbkin**12*
                     mckin**2 - 376*mbkin**13*mckin**2 + 41520*mbkin**12*
                     mckin**3 + 85401*mbkin**10*mckin**4 - 41520*mbkin**11*
                     mckin**4 + 269256*mbkin**10*mckin**5 + 33657*mbkin**8*
                     mckin**6 - 269256*mbkin**9*mckin**6 + 359008*mbkin**8*
                     mckin**7 - 95703*mbkin**6*mckin**8 - 359008*mbkin**7*
                     mckin**8 + 103800*mbkin**6*mckin**9 - 38079*mbkin**4*
                     mckin**10 - 103800*mbkin**5*mckin**10 + 2256*mbkin**4*
                     mckin**11 - 867*mbkin**2*mckin**12 - 2256*mbkin**3*
                     mckin**12 - 56*mbkin**2*mckin**13 + 21*mckin**14 + 
                    56*mbkin*mckin**14))/(3*mbkin**16) - (4*(135*mbkin**12 + 
                    8*mbkin**13 + 352*mbkin**12*mckin + 6042*mbkin**10*mckin**2 - 
                    704*mbkin**11*mckin**2 + 16816*mbkin**10*mckin**3 + 
                    12429*mbkin**8*mckin**4 - 25224*mbkin**9*mckin**4 + 
                    58368*mbkin**8*mckin**5 - 16572*mbkin**6*mckin**6 - 
                    77824*mbkin**7*mckin**6 + 33632*mbkin**6*mckin**7 - 
                    15105*mbkin**4*mckin**8 - 42040*mbkin**5*mckin**8 + 
                    1760*mbkin**4*mckin**9 - 810*mbkin**2*mckin**10 - 
                    2112*mbkin**3*mckin**10 - 48*mbkin**2*mckin**11 + 
                    21*mckin**12 + 56*mbkin*mckin**12)*q_cut)/(3*mbkin**16) - 
                 (4*(123*mbkin**10 + 16*mbkin**11 + 312*mbkin**10*mckin + 
                    2193*mbkin**8*mckin**2 - 936*mbkin**9*mckin**2 + 6784*mbkin**8*
                     mckin**3 - 1272*mbkin**6*mckin**4 - 13568*mbkin**7*mckin**4 + 
                    10176*mbkin**6*mckin**5 - 5892*mbkin**4*mckin**6 - 
                    16960*mbkin**5*mckin**6 + 1248*mbkin**4*mckin**7 - 
                    717*mbkin**2*mckin**8 - 1872*mbkin**3*mckin**8 - 40*mbkin**2*
                     mckin**9 + 21*mckin**10 + 56*mbkin*mckin**10)*q_cut**2)/
                  (3*mbkin**16) - (4*(105*mbkin**8 + 24*mbkin**9 + 256*mbkin**8*
                     mckin + 432*mbkin**6*mckin**2 - 1024*mbkin**7*mckin**2 + 
                    2176*mbkin**6*mckin**3 - 1752*mbkin**4*mckin**4 - 
                    5440*mbkin**5*mckin**4 + 768*mbkin**4*mckin**5 - 588*mbkin**2*
                     mckin**6 - 1536*mbkin**3*mckin**6 - 32*mbkin**2*mckin**7 + 
                    21*mckin**8 + 56*mbkin*mckin**8)*q_cut**3)/(3*mbkin**16) - 
                 (4*(81*mbkin**6 + 32*mbkin**7 + 184*mbkin**6*mckin - 
                    207*mbkin**4*mckin**2 - 920*mbkin**5*mckin**2 + 368*mbkin**4*
                     mckin**3 - 423*mbkin**2*mckin**4 - 1104*mbkin**3*mckin**4 - 
                    24*mbkin**2*mckin**5 + 21*mckin**6 + 56*mbkin*mckin**6)*q_cut**4)/
                  (3*mbkin**16) + (20*(87*mbkin**4 + 200*mbkin**5 + 32*mbkin**4*
                     mckin - 102*mbkin**2*mckin**2 - 192*mbkin**3*mckin**2 - 
                    80*mbkin**2*mckin**3 + 105*mckin**4 + 280*mbkin*mckin**4)*
                   q_cut**5)/(9*mbkin**16) + (100*(15*mbkin**2 + 48*mbkin**3 - 
                    8*mbkin**2*mckin + 21*mckin**2 + 56*mbkin*mckin**2)*q_cut**6)/
                  (9*mbkin**16) - (980*(3 + 8*mbkin)*q_cut**7)/(9*mbkin**16)) + 
               (3*(1 - (47*mckin**2)/mbkin**2 - (2595*mckin**4)/mbkin**4 - 
                   (11219*mckin**6)/mbkin**6 - (11219*mckin**8)/mbkin**8 - 
                   (2595*mckin**10)/mbkin**10 - (47*mckin**12)/mbkin**12 + 
                   mckin**14/mbkin**14) + (3*(mbkin**12 - 44*mbkin**10*mckin**2 - 
                    1051*mbkin**8*mckin**4 - 2432*mbkin**6*mckin**6 - 
                    1051*mbkin**4*mckin**8 - 44*mbkin**2*mckin**10 + mckin**12)*
                   q_cut)/mbkin**14 + (3*(mbkin**10 - 39*mbkin**8*mckin**2 - 
                    424*mbkin**6*mckin**4 - 424*mbkin**4*mckin**6 - 39*mbkin**2*
                     mckin**8 + mckin**10)*q_cut**2)/mbkin**14 + 
                 (3*(mbkin**8 - 32*mbkin**6*mckin**2 - 136*mbkin**4*mckin**4 - 
                    32*mbkin**2*mckin**6 + mckin**8)*q_cut**3)/mbkin**14 + 
                 (3*(mbkin**6 - 23*mbkin**4*mckin**2 - 23*mbkin**2*mckin**4 + 
                    mckin**6)*q_cut**4)/mbkin**14 - (5*(5 - (4*mckin**2)/mbkin**2 + 
                    (5*mckin**4)/mbkin**4)*q_cut**5)/mbkin**10 - 
                 (25*(mbkin**2 + mckin**2)*q_cut**6)/mbkin**14 + (35*q_cut**7)/mbkin**14)*
                ((8*mbkin**2*(3 + 8*mbkin)*((-1 + mckin**2/mbkin**2)**2*
                      (1 - (7*mckin**2)/mbkin**2 - (7*mckin**4)/mbkin**4 + 
                       mckin**6/mbkin**6) + ((-3 + (14*mckin**2)/mbkin**2 + 
                        (26*mckin**4)/mbkin**4 + (14*mckin**6)/mbkin**6 - 
                        (3*mckin**8)/mbkin**8)*q_cut)/mbkin**2 + (2*(mbkin**6 - 
                        2*mbkin**4*mckin**2 - 2*mbkin**2*mckin**4 + mckin**6)*
                       q_cut**2)/mbkin**10 + (2*(mbkin**4 + mbkin**2*mckin**2 + 
                        mckin**4)*q_cut**3)/mbkin**10 - (3*(mbkin**2 + mckin**2)*
                       q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)**2)/9 + 2*mbkin**4*
                  ((-1 + mckin**2/mbkin**2)**2*(1 - (7*mckin**2)/mbkin**2 - 
                     (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6) + 
                   ((-3 + (14*mckin**2)/mbkin**2 + (26*mckin**4)/mbkin**4 + 
                      (14*mckin**6)/mbkin**6 - (3*mckin**8)/mbkin**8)*q_cut)/
                    mbkin**2 + (2*(mbkin**6 - 2*mbkin**4*mckin**2 - 2*mbkin**2*
                       mckin**4 + mckin**6)*q_cut**2)/mbkin**10 + 
                   (2*(mbkin**4 + mbkin**2*mckin**2 + mckin**4)*q_cut**3)/mbkin**10 - 
                   (3*(mbkin**2 + mckin**2)*q_cut**4)/mbkin**10 + q_cut**5/mbkin**10)*
                  ((-4*(mbkin - mckin)**2*(mbkin + mckin)*(27*mbkin**7 + 
                      27*mbkin**6*mckin + 72*mbkin**7*mckin - 21*mbkin**5*
                       mckin**2 - 21*mbkin**4*mckin**3 - 56*mbkin**5*mckin**3 - 
                      93*mbkin**3*mckin**4 - 93*mbkin**2*mckin**5 - 248*mbkin**3*
                       mckin**5 + 15*mbkin*mckin**6 + 15*mckin**7 + 40*mbkin*
                       mckin**7))/(9*mbkin**12) + (4*(51*mbkin**8 + 24*mbkin**9 + 
                      112*mbkin**8*mckin + 72*mbkin**6*mckin**2 - 224*mbkin**7*
                       mckin**2 + 416*mbkin**6*mckin**3 - 108*mbkin**4*mckin**4 - 
                      624*mbkin**5*mckin**4 + 336*mbkin**4*mckin**5 - 204*mbkin**2*
                       mckin**6 - 448*mbkin**3*mckin**6 - 96*mbkin**2*mckin**7 + 
                      45*mckin**8 + 120*mbkin*mckin**8)*q_cut)/(9*mbkin**12) - 
                   (8*(12*mbkin**6 + 16*mbkin**7 + 16*mbkin**6*mckin - 6*mbkin**4*
                       mckin**2 - 48*mbkin**5*mckin**2 + 32*mbkin**4*mckin**3 - 
                      33*mbkin**2*mckin**4 - 64*mbkin**3*mckin**4 - 24*mbkin**2*
                       mckin**5 + 15*mckin**6 + 40*mbkin*mckin**6)*q_cut**2)/
                    (9*mbkin**12) - (8*(6*mbkin**4 + 24*mbkin**5 - 8*mbkin**4*
                       mckin + 6*mbkin**2*mckin**2 + 32*mbkin**3*mckin**2 - 
                      16*mbkin**2*mckin**3 + 15*mckin**4 + 40*mbkin*mckin**4)*
                     q_cut**3)/(9*mbkin**12) + (4*(9*mbkin**2 + 32*mbkin**3 - 
                      8*mbkin**2*mckin + 15*mckin**2 + 40*mbkin*mckin**2)*q_cut**4)/
                    (3*mbkin**12) - (20*(3 + 8*mbkin)*q_cut**5)/(9*mbkin**12))))) - 
           6*((-128*mckin**4*(mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)**2*
               (mbkin**2 + 2*mbkin*mckin + mckin**2 - q_cut)**2*(3*mbkin**4 + 
                8*mbkin**4*mckin - 6*mbkin**2*mckin**2 - 8*mbkin**3*mckin**2 - 
                8*mbkin**2*mckin**3 + 3*mckin**4 + 8*mbkin*mckin**4 - 
                3*mbkin**2*q_cut - 8*mbkin**2*mckin*q_cut - 3*mckin**2*q_cut - 
                8*mbkin*mckin**2*q_cut)*(108*mbkin**20 - 792*mbkin**18*mckin**2 - 
                13329*mbkin**16*mckin**4 + 40518*mbkin**14*mckin**6 + 
                387441*mbkin**12*mckin**8 + 668988*mbkin**10*mckin**10 + 
                387441*mbkin**8*mckin**12 + 40518*mbkin**6*mckin**14 - 
                13329*mbkin**4*mckin**16 - 792*mbkin**2*mckin**18 + 
                108*mckin**20 - 210*mbkin**18*q_cut - 222*mbkin**16*mckin**2*q_cut + 
                15402*mbkin**14*mckin**4*q_cut + 81210*mbkin**12*mckin**6*q_cut + 
                153300*mbkin**10*mckin**8*q_cut + 153300*mbkin**8*mckin**10*q_cut + 
                81210*mbkin**6*mckin**12*q_cut + 15402*mbkin**4*mckin**14*q_cut - 
                222*mbkin**2*mckin**16*q_cut - 210*mckin**18*q_cut - 108*mbkin**16*
                 q_cut**2 + 552*mbkin**14*mckin**2*q_cut**2 + 22257*mbkin**12*mckin**4*
                 q_cut**2 + 101844*mbkin**10*mckin**6*q_cut**2 + 158394*mbkin**8*mckin**8*
                 q_cut**2 + 101844*mbkin**6*mckin**10*q_cut**2 + 22257*mbkin**4*mckin**12*
                 q_cut**2 + 552*mbkin**2*mckin**14*q_cut**2 - 108*mckin**16*q_cut**2 + 
                420*mbkin**14*q_cut**3 + 2088*mbkin**12*mckin**2*q_cut**3 - 
                8028*mbkin**10*mckin**4*q_cut**3 - 43584*mbkin**8*mckin**6*q_cut**3 - 
                43584*mbkin**6*mckin**8*q_cut**3 - 8028*mbkin**4*mckin**10*q_cut**3 + 
                2088*mbkin**2*mckin**12*q_cut**3 + 420*mckin**14*q_cut**3 - 
                105*mbkin**12*q_cut**4 - 642*mbkin**10*mckin**2*q_cut**4 - 966*mbkin**8*
                 mckin**4*q_cut**4 - 2118*mbkin**6*mckin**6*q_cut**4 - 966*mbkin**4*
                 mckin**8*q_cut**4 - 642*mbkin**2*mckin**10*q_cut**4 - 105*mckin**12*
                 q_cut**4 - 238*mbkin**10*q_cut**5 - 1650*mbkin**8*mckin**2*q_cut**5 - 
                5522*mbkin**6*mckin**4*q_cut**5 - 5522*mbkin**4*mckin**6*q_cut**5 - 
                1650*mbkin**2*mckin**8*q_cut**5 - 238*mckin**10*q_cut**5 + 105*mbkin**8*
                 q_cut**6 + 940*mbkin**6*mckin**2*q_cut**6 + 1705*mbkin**4*mckin**4*
                 q_cut**6 + 940*mbkin**2*mckin**6*q_cut**6 + 105*mckin**8*q_cut**6 + 
                88*mbkin**6*q_cut**7 - 284*mbkin**4*mckin**2*q_cut**7 - 284*mbkin**2*
                 mckin**4*q_cut**7 + 88*mckin**6*q_cut**7 - 35*mbkin**4*q_cut**8 + 
                70*mbkin**2*mckin**2*q_cut**8 - 35*mckin**4*q_cut**8 - 60*mbkin**2*q_cut**9 - 
                60*mckin**2*q_cut**9 + 35*q_cut**10))/(mbkin**30*(mbkin**2 + mckin**2 - 
                q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - 
                mckin**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))) + ((-32*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + 
                  mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(3*mbkin**4 + 8*mbkin**4*
                   mckin - 6*mbkin**2*mckin**2 - 8*mbkin**3*mckin**2 - 
                  8*mbkin**2*mckin**3 + 3*mckin**4 + 8*mbkin*mckin**4 - 
                  8*mbkin**3*q_cut + 8*mbkin**2*mckin*q_cut - 6*mckin**2*q_cut - 
                  16*mbkin*mckin**2*q_cut + 3*q_cut**2 + 8*mbkin*q_cut**2)*
                 (108*mbkin**20 - 792*mbkin**18*mckin**2 - 13329*mbkin**16*
                   mckin**4 + 40518*mbkin**14*mckin**6 + 387441*mbkin**12*
                   mckin**8 + 668988*mbkin**10*mckin**10 + 387441*mbkin**8*
                   mckin**12 + 40518*mbkin**6*mckin**14 - 13329*mbkin**4*
                   mckin**16 - 792*mbkin**2*mckin**18 + 108*mckin**20 - 
                  210*mbkin**18*q_cut - 222*mbkin**16*mckin**2*q_cut + 15402*mbkin**14*
                   mckin**4*q_cut + 81210*mbkin**12*mckin**6*q_cut + 153300*mbkin**10*
                   mckin**8*q_cut + 153300*mbkin**8*mckin**10*q_cut + 81210*mbkin**6*
                   mckin**12*q_cut + 15402*mbkin**4*mckin**14*q_cut - 222*mbkin**2*
                   mckin**16*q_cut - 210*mckin**18*q_cut - 108*mbkin**16*q_cut**2 + 
                  552*mbkin**14*mckin**2*q_cut**2 + 22257*mbkin**12*mckin**4*q_cut**2 + 
                  101844*mbkin**10*mckin**6*q_cut**2 + 158394*mbkin**8*mckin**8*
                   q_cut**2 + 101844*mbkin**6*mckin**10*q_cut**2 + 22257*mbkin**4*
                   mckin**12*q_cut**2 + 552*mbkin**2*mckin**14*q_cut**2 - 108*mckin**16*
                   q_cut**2 + 420*mbkin**14*q_cut**3 + 2088*mbkin**12*mckin**2*q_cut**3 - 
                  8028*mbkin**10*mckin**4*q_cut**3 - 43584*mbkin**8*mckin**6*q_cut**3 - 
                  43584*mbkin**6*mckin**8*q_cut**3 - 8028*mbkin**4*mckin**10*q_cut**3 + 
                  2088*mbkin**2*mckin**12*q_cut**3 + 420*mckin**14*q_cut**3 - 
                  105*mbkin**12*q_cut**4 - 642*mbkin**10*mckin**2*q_cut**4 - 966*mbkin**8*
                   mckin**4*q_cut**4 - 2118*mbkin**6*mckin**6*q_cut**4 - 966*mbkin**4*
                   mckin**8*q_cut**4 - 642*mbkin**2*mckin**10*q_cut**4 - 105*mckin**12*
                   q_cut**4 - 238*mbkin**10*q_cut**5 - 1650*mbkin**8*mckin**2*q_cut**5 - 
                  5522*mbkin**6*mckin**4*q_cut**5 - 5522*mbkin**4*mckin**6*q_cut**5 - 
                  1650*mbkin**2*mckin**8*q_cut**5 - 238*mckin**10*q_cut**5 + 105*mbkin**8*
                   q_cut**6 + 940*mbkin**6*mckin**2*q_cut**6 + 1705*mbkin**4*mckin**4*
                   q_cut**6 + 940*mbkin**2*mckin**6*q_cut**6 + 105*mckin**8*q_cut**6 + 
                  88*mbkin**6*q_cut**7 - 284*mbkin**4*mckin**2*q_cut**7 - 284*mbkin**2*
                   mckin**4*q_cut**7 + 88*mckin**6*q_cut**7 - 35*mbkin**4*q_cut**8 + 
                  70*mbkin**2*mckin**2*q_cut**8 - 35*mckin**4*q_cut**8 - 60*mbkin**2*
                   q_cut**9 - 60*mckin**2*q_cut**9 + 35*q_cut**10))/mbkin**30 + 
               np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(72*mckin**4*
                  ((-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 3*mckin + 
                      8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 
                      4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/
                    (9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*
                  (9*(-1 + mckin**4/mbkin**4)**2*(12 - (112*mckin**2)/mbkin**2 - 
                     (1269*mckin**4)/mbkin**4 + (7152*mckin**6)/mbkin**6 + 
                     (30014*mckin**8)/mbkin**8 + (7152*mckin**10)/mbkin**10 - 
                     (1269*mckin**12)/mbkin**12 - (112*mckin**14)/mbkin**14 + 
                     (12*mckin**16)/mbkin**16) - (6*(71 - (261*mckin**2)/
                       mbkin**2 - (7313*mckin**4)/mbkin**4 + (699*mckin**6)/
                       mbkin**6 + (141606*mckin**8)/mbkin**8 + (364158*mckin**10)/
                       mbkin**10 + (364158*mckin**12)/mbkin**12 + 
                      (141606*mckin**14)/mbkin**14 + (699*mckin**16)/mbkin**16 - 
                      (7313*mckin**18)/mbkin**18 - (261*mckin**20)/mbkin**20 + 
                      (71*mckin**22)/mbkin**22)*q_cut)/mbkin**2 + 
                   (12*(35 + (70*mckin**2)/mbkin**2 - (1887*mckin**4)/mbkin**4 - 
                      (7902*mckin**6)/mbkin**6 - (8718*mckin**8)/mbkin**8 - 
                      (4776*mckin**10)/mbkin**10 - (8718*mckin**12)/mbkin**12 - 
                      (7902*mckin**14)/mbkin**14 - (1887*mckin**16)/mbkin**16 + 
                      (70*mckin**18)/mbkin**18 + (35*mckin**20)/mbkin**20)*q_cut**2)/
                    mbkin**4 + (6*(71 + (23*mckin**2)/mbkin**2 - (7000*mckin**4)/
                       mbkin**4 - (32072*mckin**6)/mbkin**6 - (55270*mckin**8)/
                       mbkin**8 - (55270*mckin**10)/mbkin**10 - (32072*mckin**12)/
                       mbkin**12 - (7000*mckin**14)/mbkin**14 + (23*mckin**16)/
                       mbkin**16 + (71*mckin**18)/mbkin**18)*q_cut**3)/mbkin**6 - 
                   (3*(351 + (1632*mckin**2)/mbkin**2 - (11450*mckin**4)/
                       mbkin**4 - (68080*mckin**6)/mbkin**6 - (111678*mckin**8)/
                       mbkin**8 - (68080*mckin**10)/mbkin**10 - (11450*mckin**12)/
                       mbkin**12 + (1632*mckin**14)/mbkin**14 + (351*mckin**16)/
                       mbkin**16)*q_cut**4)/mbkin**8 + (8*(49 + (301*mckin**2)/
                       mbkin**2 - (909*mckin**4)/mbkin**4 - (4193*mckin**6)/
                       mbkin**6 - (4193*mckin**8)/mbkin**8 - (909*mckin**10)/
                       mbkin**10 + (301*mckin**12)/mbkin**12 + (49*mckin**14)/
                       mbkin**14)*q_cut**5)/mbkin**10 + (4*(119 + (966*mckin**2)/
                       mbkin**2 + (3327*mckin**4)/mbkin**4 + (4610*mckin**6)/
                       mbkin**6 + (3327*mckin**8)/mbkin**8 + (966*mckin**10)/
                       mbkin**10 + (119*mckin**12)/mbkin**12)*q_cut**6)/mbkin**12 - 
                   (120*(3 + (35*mckin**2)/mbkin**2 + (87*mckin**4)/mbkin**4 + 
                      (87*mckin**6)/mbkin**6 + (35*mckin**8)/mbkin**8 + 
                      (3*mckin**10)/mbkin**10)*q_cut**7)/mbkin**14 + 
                   ((-106 + (1472*mckin**2)/mbkin**2 + (2631*mckin**4)/mbkin**4 + 
                      (1472*mckin**6)/mbkin**6 - (106*mckin**8)/mbkin**8)*q_cut**8)/
                    mbkin**16 + (98*(mbkin**6 - 3*mbkin**4*mckin**2 - 3*mbkin**2*
                       mckin**4 + mckin**6)*q_cut**9)/mbkin**24 + 
                   (120*(mbkin**2 + mckin**2)**2*q_cut**10)/mbkin**24 - 
                   (130*(mbkin**2 + mckin**2)*q_cut**11)/mbkin**24 + (35*q_cut**12)/
                    mbkin**24) + ((-1 + mckin**2/mbkin**2)**2 - 
                   (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
                  ((-64*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(201 - 
                         (2594*mckin**2)/mbkin**2 - (18020*mckin**4)/mbkin**4 + 
                         (686586*mckin**6)/mbkin**6 - (484473*mckin**8)/mbkin**
                          8 - (24524568*mckin**10)/mbkin**10 - (51017352*
                          mckin**12)/mbkin**12 - (29833656*mckin**14)/mbkin**14 - 
                         (1099137*mckin**16)/mbkin**16 + (793962*mckin**18)/
                          mbkin**18 - (274084*mckin**20)/mbkin**20 - 
                         (9490*mckin**22)/mbkin**22 + (3105*mckin**24)/mbkin**
                          24)) + (2*(411 - (4441*mckin**2)/mbkin**2 - 
                         (47413*mckin**4)/mbkin**4 + (998110*mckin**6)/mbkin**6 + 
                         (1955322*mckin**8)/mbkin**8 - (24020691*mckin**10)/
                          mbkin**10 - (77841426*mckin**12)/mbkin**12 - 
                         (84248520*mckin**14)/mbkin**14 - (31464921*mckin**16)/
                          mbkin**16 + (2471757*mckin**18)/mbkin**18 + 
                         (1072279*mckin**20)/mbkin**20 - (424934*mckin**22)/
                          mbkin**22 - (10828*mckin**24)/mbkin**24 + 
                         (6255*mckin**26)/mbkin**26)*q_cut)/mbkin**2 - 
                      (2*(420 - (2800*mckin**2)/mbkin**2 - (64131*mckin**4)/
                          mbkin**4 + (406035*mckin**6)/mbkin**6 + (2915250*
                          mckin**8)/mbkin**8 + (3976458*mckin**10)/mbkin**10 + 
                         (2350824*mckin**12)/mbkin**12 + (4693344*mckin**14)/
                          mbkin**14 + (3558342*mckin**16)/mbkin**16 + 
                         (48462*mckin**18)/mbkin**18 - (268445*mckin**20)/
                          mbkin**20 + (9861*mckin**22)/mbkin**22 + 
                         (6300*mckin**24)/mbkin**24)*q_cut**2)/mbkin**4 - 
                      (2*(411 - (2797*mckin**2)/mbkin**2 - (61931*mckin**4)/
                          mbkin**4 + (744527*mckin**6)/mbkin**6 + (5541108*
                          mckin**8)/mbkin**8 + (10489326*mckin**10)/mbkin**10 + 
                         (10608076*mckin**12)/mbkin**12 + (5830586*mckin**14)/
                          mbkin**14 + (187913*mckin**16)/mbkin**16 - 
                         (376185*mckin**18)/mbkin**18 + (8423*mckin**20)/
                          mbkin**20 + (6255*mckin**22)/mbkin**22)*q_cut**3)/
                       mbkin**6 + ((2091 - (7688*mckin**2)/mbkin**2 - 
                         (351965*mckin**4)/mbkin**4 + (922140*mckin**6)/mbkin**
                          6 + (11699682*mckin**8)/mbkin**8 + (22377552*mckin**
                          10)/mbkin**10 + (11839006*mckin**12)/mbkin**12 - 
                         (747556*mckin**14)/mbkin**14 - (969997*mckin**16)/
                          mbkin**16 + (118544*mckin**18)/mbkin**18 + 
                         (31455*mckin**20)/mbkin**20)*q_cut**4)/mbkin**8 - 
                      (4*(231 - (1001*mckin**2)/mbkin**2 - (36378*mckin**4)/
                          mbkin**4 + (32226*mckin**6)/mbkin**6 + (485535*
                          mckin**8)/mbkin**8 + (473268*mckin**10)/mbkin**10 - 
                         (63132*mckin**12)/mbkin**12 - (76864*mckin**14)/
                          mbkin**14 + (10500*mckin**16)/mbkin**16 + 
                         (3255*mckin**18)/mbkin**18)*q_cut**5)/mbkin**10 - 
                      (4*(168 + (532*mckin**2)/mbkin**2 - (37666*mckin**4)/
                          mbkin**4 - (173612*mckin**6)/mbkin**6 - (257329*
                          mckin**8)/mbkin**8 - (157955*mckin**10)/mbkin**10 + 
                         (6303*mckin**12)/mbkin**12 + (24871*mckin**14)/mbkin**
                          14 + (2940*mckin**16)/mbkin**16)*q_cut**6)/mbkin**12 + 
                      (4*(255 - (245*mckin**2)/mbkin**2 - (39116*mckin**4)/
                          mbkin**4 - (138249*mckin**6)/mbkin**6 - (129466*
                          mckin**8)/mbkin**8 - (11635*mckin**10)/mbkin**10 + 
                         (18713*mckin**12)/mbkin**12 + (3375*mckin**14)/mbkin**
                          14)*q_cut**7)/mbkin**14 + ((-843 + (2036*mckin**2)/
                          mbkin**2 + (42441*mckin**4)/mbkin**4 + (95556*mckin**6)/
                          mbkin**6 + (47845*mckin**8)/mbkin**8 - (5364*mckin**10)/
                          mbkin**10 - (6315*mckin**12)/mbkin**12)*q_cut**8)/
                       mbkin**16 + (2*(147 - (441*mckin**2)/mbkin**2 - 
                         (111*mckin**4)/mbkin**4 - (2794*mckin**6)/mbkin**6 - 
                         (2452*mckin**8)/mbkin**8 + (735*mckin**10)/mbkin**10)*
                        q_cut**9)/mbkin**18 + (10*(36 + (72*mckin**2)/mbkin**2 + 
                         (93*mckin**4)/mbkin**4 + (223*mckin**6)/mbkin**6 + 
                         (180*mckin**8)/mbkin**8)*q_cut**10)/mbkin**20 - 
                      (30*(13 + (13*mckin**2)/mbkin**2 + (45*mckin**4)/mbkin**4 + 
                         (65*mckin**6)/mbkin**6)*q_cut**11)/mbkin**22 + 
                      (105*(mbkin**4 + 5*mckin**4)*q_cut**12)/mbkin**28))/9 + 
                   72*((8*mckin**2*(3 + 8*mckin)*(9*(-1 + mckin**4/mbkin**4)**2*
                         (12 - (112*mckin**2)/mbkin**2 - (1269*mckin**4)/
                          mbkin**4 + (7152*mckin**6)/mbkin**6 + (30014*mckin**8)/
                          mbkin**8 + (7152*mckin**10)/mbkin**10 - (1269*mckin**
                          12)/mbkin**12 - (112*mckin**14)/mbkin**14 + 
                          (12*mckin**16)/mbkin**16) - (6*(71 - (261*mckin**2)/
                          mbkin**2 - (7313*mckin**4)/mbkin**4 + (699*mckin**6)/
                          mbkin**6 + (141606*mckin**8)/mbkin**8 + (364158*
                          mckin**10)/mbkin**10 + (364158*mckin**12)/mbkin**12 + 
                          (141606*mckin**14)/mbkin**14 + (699*mckin**16)/mbkin**
                          16 - (7313*mckin**18)/mbkin**18 - (261*mckin**20)/
                          mbkin**20 + (71*mckin**22)/mbkin**22)*q_cut)/mbkin**2 + 
                        (12*(35 + (70*mckin**2)/mbkin**2 - (1887*mckin**4)/
                          mbkin**4 - (7902*mckin**6)/mbkin**6 - (8718*mckin**8)/
                          mbkin**8 - (4776*mckin**10)/mbkin**10 - (8718*mckin**
                          12)/mbkin**12 - (7902*mckin**14)/mbkin**14 - 
                          (1887*mckin**16)/mbkin**16 + (70*mckin**18)/mbkin**18 + 
                          (35*mckin**20)/mbkin**20)*q_cut**2)/mbkin**4 + 
                        (6*(71 + (23*mckin**2)/mbkin**2 - (7000*mckin**4)/
                          mbkin**4 - (32072*mckin**6)/mbkin**6 - (55270*mckin**8)/
                          mbkin**8 - (55270*mckin**10)/mbkin**10 - (32072*
                          mckin**12)/mbkin**12 - (7000*mckin**14)/mbkin**14 + 
                          (23*mckin**16)/mbkin**16 + (71*mckin**18)/mbkin**18)*
                          q_cut**3)/mbkin**6 - (3*(351 + (1632*mckin**2)/mbkin**2 - 
                          (11450*mckin**4)/mbkin**4 - (68080*mckin**6)/mbkin**6 - 
                          (111678*mckin**8)/mbkin**8 - (68080*mckin**10)/mbkin**
                          10 - (11450*mckin**12)/mbkin**12 + (1632*mckin**14)/
                          mbkin**14 + (351*mckin**16)/mbkin**16)*q_cut**4)/mbkin**8 + 
                        (8*(49 + (301*mckin**2)/mbkin**2 - (909*mckin**4)/
                          mbkin**4 - (4193*mckin**6)/mbkin**6 - (4193*mckin**8)/
                          mbkin**8 - (909*mckin**10)/mbkin**10 + (301*mckin**12)/
                          mbkin**12 + (49*mckin**14)/mbkin**14)*q_cut**5)/mbkin**10 + 
                        (4*(119 + (966*mckin**2)/mbkin**2 + (3327*mckin**4)/
                          mbkin**4 + (4610*mckin**6)/mbkin**6 + (3327*mckin**8)/
                          mbkin**8 + (966*mckin**10)/mbkin**10 + (119*mckin**12)/
                          mbkin**12)*q_cut**6)/mbkin**12 - (120*(3 + (35*mckin**2)/
                          mbkin**2 + (87*mckin**4)/mbkin**4 + (87*mckin**6)/
                          mbkin**6 + (35*mckin**8)/mbkin**8 + (3*mckin**10)/
                          mbkin**10)*q_cut**7)/mbkin**14 + ((-106 + (1472*mckin**2)/
                          mbkin**2 + (2631*mckin**4)/mbkin**4 + (1472*mckin**6)/
                          mbkin**6 - (106*mckin**8)/mbkin**8)*q_cut**8)/mbkin**16 + 
                        (98*(mbkin**6 - 3*mbkin**4*mckin**2 - 3*mbkin**2*mckin**
                          4 + mckin**6)*q_cut**9)/mbkin**24 + (120*(mbkin**2 + 
                          mckin**2)**2*q_cut**10)/mbkin**24 - (130*(mbkin**2 + 
                          mckin**2)*q_cut**11)/mbkin**24 + (35*q_cut**12)/mbkin**24))/
                      9 + mckin**4*((-8*(mbkin - mckin)**2*(mbkin**3 + mbkin**2*
                          mckin + mbkin*mckin**2 + mckin**3)*(168*mbkin**19 + 
                          168*mbkin**18*mckin + 448*mbkin**19*mckin + 3879*
                          mbkin**17*mckin**2 + 3879*mbkin**16*mckin**3 + 10344*
                          mbkin**17*mckin**3 - 33024*mbkin**15*mckin**4 - 
                          33024*mbkin**14*mckin**5 - 88064*mbkin**15*mckin**5 - 
                          191505*mbkin**13*mckin**6 - 191505*mbkin**12*mckin**7 - 
                          510680*mbkin**13*mckin**7 + 21456*mbkin**11*mckin**8 + 
                          21456*mbkin**10*mckin**9 + 57216*mbkin**11*mckin**9 + 
                          371589*mbkin**9*mckin**10 + 371589*mbkin**8*mckin**11 + 
                          990904*mbkin**9*mckin**11 + 97728*mbkin**7*mckin**12 + 
                          97728*mbkin**6*mckin**13 + 260608*mbkin**7*mckin**13 - 
                          19179*mbkin**5*mckin**14 - 19179*mbkin**4*mckin**15 - 
                          51144*mbkin**5*mckin**15 - 1848*mbkin**3*mckin**16 - 
                          1848*mbkin**2*mckin**17 - 4928*mbkin**3*mckin**17 + 
                          216*mbkin*mckin**18 + 216*mckin**19 + 576*mbkin*
                          mckin**19))/mbkin**26 + (32*(249*mbkin**22 + 
                          142*mbkin**23 + 522*mbkin**22*mckin + 10578*mbkin**20*
                          mckin**2 - 1044*mbkin**21*mckin**2 + 29252*mbkin**20*
                          mckin**3 - 18027*mbkin**18*mckin**4 - 43878*mbkin**19*
                          mckin**4 - 4194*mbkin**18*mckin**5 - 422721*mbkin**16*
                          mckin**6 + 5592*mbkin**17*mckin**6 - 1132848*mbkin**16*
                          mckin**7 - 834570*mbkin**14*mckin**8 + 1416060*
                          mbkin**15*mckin**8 - 3641580*mbkin**14*mckin**9 + 
                          4369896*mbkin**13*mckin**10 - 4369896*mbkin**12*
                          mckin**11 + 1168398*mbkin**10*mckin**12 + 5098212*
                          mbkin**11*mckin**12 - 1982484*mbkin**10*mckin**13 + 
                          845442*mbkin**8*mckin**14 + 2265696*mbkin**9*mckin**
                          14 - 11184*mbkin**8*mckin**15 + 54081*mbkin**6*
                          mckin**16 + 12582*mbkin**7*mckin**16 + 131634*mbkin**6*
                          mckin**17 - 52890*mbkin**4*mckin**18 - 146260*mbkin**5*
                          mckin**18 + 5220*mbkin**4*mckin**19 - 2739*mbkin**2*
                          mckin**20 - 5742*mbkin**3*mckin**20 - 1562*mbkin**2*
                          mckin**21 + 639*mckin**22 + 1704*mbkin*mckin**22)*q_cut)/
                        (3*mbkin**26) - (32*(280*mbkin**21 - 280*mbkin**20*
                          mckin + 5976*mbkin**18*mckin**2 + 840*mbkin**19*
                          mckin**2 + 15096*mbkin**18*mckin**3 + 24237*mbkin**16*
                          mckin**4 - 30192*mbkin**17*mckin**4 + 94824*mbkin**16*
                          mckin**5 - 6957*mbkin**14*mckin**6 - 158040*mbkin**15*
                          mckin**6 + 139488*mbkin**14*mckin**7 - 42642*mbkin**12*
                          mckin**8 - 209232*mbkin**13*mckin**8 + 95520*mbkin**12*
                          mckin**9 + 28314*mbkin**10*mckin**10 - 133728*mbkin**11*
                          mckin**10 + 209232*mbkin**10*mckin**11 - 21645*mbkin**8*
                          mckin**12 - 278976*mbkin**9*mckin**12 + 221256*mbkin**8*
                          mckin**13 - 84033*mbkin**6*mckin**14 - 284472*mbkin**7*
                          mckin**14 + 60384*mbkin**6*mckin**15 - 29250*mbkin**4*
                          mckin**16 - 75480*mbkin**5*mckin**16 - 2520*mbkin**4*
                          mckin**17 + 630*mbkin**2*mckin**18 + 3080*mbkin**3*
                          mckin**18 - 1400*mbkin**2*mckin**19 + 630*mckin**20 + 
                          1680*mbkin*mckin**20)*q_cut**2)/(3*mbkin**26) - 
                       (16*(285*mbkin**18 + 852*mbkin**19 - 92*mbkin**18*mckin + 
                          21138*mbkin**16*mckin**2 + 368*mbkin**17*mckin**2 + 
                          56000*mbkin**16*mckin**3 + 91824*mbkin**14*mckin**4 - 
                          140000*mbkin**15*mckin**4 + 384864*mbkin**14*mckin**5 + 
                          42972*mbkin**12*mckin**6 - 769728*mbkin**13*mckin**6 + 
                          884320*mbkin**12*mckin**7 - 165810*mbkin**10*mckin**8 - 
                          1547560*mbkin**11*mckin**8 + 1105400*mbkin**10*
                          mckin**9 - 374592*mbkin**8*mckin**10 - 1768640*mbkin**9*
                          mckin**10 + 769728*mbkin**8*mckin**11 - 359472*mbkin**6*
                          mckin**12 - 1154592*mbkin**7*mckin**12 + 196000*
                          mbkin**6*mckin**13 - 105276*mbkin**4*mckin**14 - 
                          280000*mbkin**5*mckin**14 - 736*mbkin**4*mckin**15 - 
                          579*mbkin**2*mckin**16 + 1012*mbkin**3*mckin**16 - 
                          2556*mbkin**2*mckin**17 + 1278*mckin**18 + 3408*mbkin*
                          mckin**18)*q_cut**3)/(3*mbkin**26) + (16*(-171*mbkin**16 + 
                          2808*mbkin**17 - 3264*mbkin**16*mckin + 23295*
                          mbkin**14*mckin**2 + 16320*mbkin**15*mckin**2 + 
                          45800*mbkin**14*mckin**3 + 101655*mbkin**12*mckin**4 - 
                          137400*mbkin**13*mckin**4 + 408480*mbkin**12*mckin**5 - 
                          22386*mbkin**10*mckin**6 - 953120*mbkin**11*mckin**6 + 
                          893424*mbkin**10*mckin**7 - 414768*mbkin**8*mckin**8 - 
                          1786848*mbkin**9*mckin**8 + 680800*mbkin**8*mckin**9 - 
                          408015*mbkin**6*mckin**10 - 1225440*mbkin**7*mckin**
                          10 + 137400*mbkin**6*mckin**11 - 94443*mbkin**4*
                          mckin**12 - 229000*mbkin**5*mckin**12 - 22848*mbkin**4*
                          mckin**13 + 11358*mbkin**2*mckin**14 + 35904*mbkin**3*
                          mckin**14 - 5616*mbkin**2*mckin**15 + 3159*mckin**16 + 
                          8424*mbkin*mckin**16)*q_cut**4)/(3*mbkin**26) - 
                       (128*(-42*mbkin**14 + 490*mbkin**15 - 602*mbkin**14*
                          mckin + 2718*mbkin**12*mckin**2 + 3612*mbkin**13*
                          mckin**2 + 3636*mbkin**12*mckin**3 + 4662*mbkin**10*
                          mckin**4 - 12726*mbkin**11*mckin**4 + 25158*mbkin**10*
                          mckin**5 - 12579*mbkin**8*mckin**6 - 67088*mbkin**9*
                          mckin**6 + 33544*mbkin**8*mckin**7 - 24894*mbkin**6*
                          mckin**8 - 75474*mbkin**7*mckin**8 + 9090*mbkin**6*
                          mckin**9 - 8172*mbkin**4*mckin**10 - 18180*mbkin**5*
                          mckin**10 - 3612*mbkin**4*mckin**11 + 2226*mbkin**2*
                          mckin**12 + 6622*mbkin**3*mckin**12 - 686*mbkin**2*
                          mckin**13 + 441*mckin**14 + 1176*mbkin*mckin**14)*
                         q_cut**5)/(9*mbkin**26) - (32*(-126*mbkin**12 + 952*
                          mbkin**13 - 1288*mbkin**12*mckin + 54*mbkin**10*
                          mckin**2 + 9016*mbkin**11*mckin**2 - 8872*mbkin**10*
                          mckin**3 + 6393*mbkin**8*mckin**4 + 35488*mbkin**9*
                          mckin**4 - 18440*mbkin**8*mckin**5 + 14091*mbkin**6*
                          mckin**6 + 55320*mbkin**7*mckin**6 - 17744*mbkin**6*
                          mckin**7 + 14220*mbkin**4*mckin**8 + 44360*mbkin**5*
                          mckin**8 - 6440*mbkin**4*mckin**9 + 4956*mbkin**2*
                          mckin**10 + 14168*mbkin**3*mckin**10 - 952*mbkin**2*
                          mckin**11 + 714*mckin**12 + 1904*mbkin*mckin**12)*
                         q_cut**6)/(3*mbkin**26) + (320*(-21*mbkin**10 + 
                          84*mbkin**11 - 140*mbkin**10*mckin + 159*mbkin**8*
                          mckin**2 + 1120*mbkin**9*mckin**2 - 696*mbkin**8*
                          mckin**3 + 783*mbkin**6*mckin**4 + 3132*mbkin**7*
                          mckin**4 - 1044*mbkin**6*mckin**5 + 1095*mbkin**4*
                          mckin**6 + 3480*mbkin**5*mckin**6 - 560*mbkin**4*
                          mckin**7 + 555*mbkin**2*mckin**8 + 1540*mbkin**3*
                          mckin**8 - 60*mbkin**2*mckin**9 + 54*mckin**10 + 
                          144*mbkin*mckin**10)*q_cut**7)/(3*mbkin**26) + 
                       (8*(3480*mbkin**8 + 3392*mbkin**9 + 5888*mbkin**8*mckin - 
                          11979*mbkin**6*mckin**2 - 52992*mbkin**7*mckin**2 + 
                          21048*mbkin**6*mckin**3 - 32841*mbkin**4*mckin**4 - 
                          105240*mbkin**5*mckin**4 + 17664*mbkin**4*mckin**5 - 
                          24924*mbkin**2*mckin**6 - 64768*mbkin**3*mckin**6 - 
                          1696*mbkin**2*mckin**7 + 1908*mckin**8 + 5088*mbkin*
                          mckin**8)*q_cut**8)/(9*mbkin**26) - (1568*(3*mbkin**6 + 
                          6*mbkin**7 + 2*mbkin**6*mckin - 6*mbkin**4*mckin**2 - 
                          20*mbkin**5*mckin**2 + 4*mbkin**4*mckin**3 - 9*mbkin**2*
                          mckin**4 - 22*mbkin**3*mckin**4 - 2*mbkin**2*mckin**5 + 
                          3*mckin**6 + 8*mbkin*mckin**6)*q_cut**9)/(3*mbkin**26) - 
                       (640*(mbkin**2 + mckin**2)*(6*mbkin**2 + 20*mbkin**3 - 
                          4*mbkin**2*mckin + 9*mckin**2 + 24*mbkin*mckin**2)*
                         q_cut**10)/(3*mbkin**26) + (1040*(15*mbkin**2 + 
                          44*mbkin**3 - 4*mbkin**2*mckin + 18*mckin**2 + 
                          48*mbkin*mckin**2)*q_cut**11)/(9*mbkin**26) - 
                       (560*(3 + 8*mbkin)*q_cut**12)/(3*mbkin**26))))))*
              np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                     2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                      q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
                 mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                     2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))) - 
           72*((128*mckin**8*(mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)*(
                mbkin**2 + 2*mbkin*mckin + mckin**2 - q_cut)*(423*mbkin**14 + 
                279*mbkin**12*mckin**2 - 27945*mbkin**10*mckin**4 - 97497*mbkin**8*
                 mckin**6 - 97497*mbkin**6*mckin**8 - 27945*mbkin**4*mckin**10 + 
                279*mbkin**2*mckin**12 + 423*mckin**14 - 417*mbkin**12*q_cut - 
                3492*mbkin**10*mckin**2*q_cut - 9873*mbkin**8*mckin**4*q_cut - 
                14016*mbkin**6*mckin**6*q_cut - 9873*mbkin**4*mckin**8*q_cut - 
                3492*mbkin**2*mckin**10*q_cut - 417*mckin**12*q_cut - 417*mbkin**10*
                 q_cut**2 - 3897*mbkin**8*mckin**2*q_cut**2 - 10932*mbkin**6*mckin**4*
                 q_cut**2 - 10932*mbkin**4*mckin**6*q_cut**2 - 3897*mbkin**2*mckin**8*
                 q_cut**2 - 417*mckin**10*q_cut**2 + 423*mbkin**8*q_cut**3 + 3264*mbkin**6*
                 mckin**2*q_cut**3 + 5892*mbkin**4*mckin**4*q_cut**3 + 3264*mbkin**2*
                 mckin**6*q_cut**3 + 423*mckin**8*q_cut**3 + 3*mbkin**6*q_cut**4 - 
                69*mbkin**4*mckin**2*q_cut**4 - 69*mbkin**2*mckin**4*q_cut**4 + 
                3*mckin**6*q_cut**4 - 25*mbkin**4*q_cut**5 + 20*mbkin**2*mckin**2*q_cut**5 - 
                25*mckin**4*q_cut**5 - 25*mbkin**2*q_cut**6 - 25*mckin**2*q_cut**6 + 
                35*q_cut**7)*(3*mbkin**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 
                8*mbkin**4*mckin*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                6*mbkin**2*mckin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                8*mbkin**3*mckin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                8*mbkin**2*mckin**3*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 
                3*mckin**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 
                8*mbkin*mckin**4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                3*mbkin**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                8*mbkin**2*mckin*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                3*mckin**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 
                8*mbkin*mckin**2*q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))*
               np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                      2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                       q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/
              (mbkin**24*(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                  (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mckin**2 + q_cut + 
                mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))) + 
             ((-32*mckin**6*(3*mbkin**2 + 8*mbkin**2*mckin - 3*mckin**2 - 
                  8*mbkin*mckin**2)*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*(423*mbkin**14 + 
                  279*mbkin**12*mckin**2 - 27945*mbkin**10*mckin**4 - 
                  97497*mbkin**8*mckin**6 - 97497*mbkin**6*mckin**8 - 
                  27945*mbkin**4*mckin**10 + 279*mbkin**2*mckin**12 + 
                  423*mckin**14 - 417*mbkin**12*q_cut - 3492*mbkin**10*mckin**2*q_cut - 
                  9873*mbkin**8*mckin**4*q_cut - 14016*mbkin**6*mckin**6*q_cut - 
                  9873*mbkin**4*mckin**8*q_cut - 3492*mbkin**2*mckin**10*q_cut - 
                  417*mckin**12*q_cut - 417*mbkin**10*q_cut**2 - 3897*mbkin**8*mckin**2*
                   q_cut**2 - 10932*mbkin**6*mckin**4*q_cut**2 - 10932*mbkin**4*mckin**6*
                   q_cut**2 - 3897*mbkin**2*mckin**8*q_cut**2 - 417*mckin**10*q_cut**2 + 
                  423*mbkin**8*q_cut**3 + 3264*mbkin**6*mckin**2*q_cut**3 + 5892*mbkin**4*
                   mckin**4*q_cut**3 + 3264*mbkin**2*mckin**6*q_cut**3 + 423*mckin**8*
                   q_cut**3 + 3*mbkin**6*q_cut**4 - 69*mbkin**4*mckin**2*q_cut**4 - 
                  69*mbkin**2*mckin**4*q_cut**4 + 3*mckin**6*q_cut**4 - 25*mbkin**4*
                   q_cut**5 + 20*mbkin**2*mckin**2*q_cut**5 - 25*mckin**4*q_cut**5 - 
                  25*mbkin**2*q_cut**6 - 25*mckin**2*q_cut**6 + 35*q_cut**7))/mbkin**28 + 
               (mckin**4*(-36*mckin**4*((-8*(mbkin - mckin)**2*(mbkin + mckin)*
                      (3*mbkin + 3*mckin + 8*mbkin*mckin))/(9*mbkin**6) + 
                    (16*(4*mbkin**3 - 4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*
                        mckin**2)*q_cut)/(9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/
                     (9*mbkin**6))*(9*(-1 + mckin**2/mbkin**2)**2*(47 + 
                      (31*mckin**2)/mbkin**2 - (3105*mckin**4)/mbkin**4 - 
                      (10833*mckin**6)/mbkin**6 - (10833*mckin**8)/mbkin**8 - 
                      (3105*mckin**10)/mbkin**10 + (31*mckin**12)/mbkin**12 + 
                      (47*mckin**14)/mbkin**14) - (3*(421 + (1354*mckin**2)/
                        mbkin**2 - (17342*mckin**4)/mbkin**4 - (84374*mckin**6)/
                        mbkin**6 - (132758*mckin**8)/mbkin**8 - (84374*mckin**10)/
                        mbkin**10 - (17342*mckin**12)/mbkin**12 + 
                       (1354*mckin**14)/mbkin**14 + (421*mckin**16)/mbkin**16)*
                      q_cut)/mbkin**2 + (6*(140 + (839*mckin**2)/mbkin**2 - 
                       (795*mckin**4)/mbkin**4 - (7114*mckin**6)/mbkin**6 - 
                       (7114*mckin**8)/mbkin**8 - (795*mckin**10)/mbkin**10 + 
                       (839*mckin**12)/mbkin**12 + (140*mckin**14)/mbkin**14)*
                      q_cut**2)/mbkin**4 + (6*(140 + (1259*mckin**2)/mbkin**2 + 
                       (3262*mckin**4)/mbkin**4 + (4076*mckin**6)/mbkin**6 + 
                       (3262*mckin**8)/mbkin**8 + (1259*mckin**10)/mbkin**10 + 
                       (140*mckin**12)/mbkin**12)*q_cut**3)/mbkin**6 - 
                    (6*(210 + (1891*mckin**2)/mbkin**2 + (4862*mckin**4)/
                        mbkin**4 + (4862*mckin**6)/mbkin**6 + (1891*mckin**8)/
                        mbkin**8 + (210*mckin**10)/mbkin**10)*q_cut**4)/mbkin**8 + 
                    ((392 + (3466*mckin**2)/mbkin**2 + (6078*mckin**4)/mbkin**4 + 
                       (3466*mckin**6)/mbkin**6 + (392*mckin**8)/mbkin**8)*q_cut**5)/
                     mbkin**10 + ((28 - (34*mckin**2)/mbkin**2 - (34*mckin**4)/
                        mbkin**4 + (28*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
                    (10*(6 + (5*mckin**2)/mbkin**2 + (6*mckin**4)/mbkin**4)*q_cut**7)/
                     mbkin**14 - (95*(mbkin**2 + mckin**2)*q_cut**8)/mbkin**18 + 
                    (35*q_cut**9)/mbkin**18) + ((-1 + mckin**2/mbkin**2)**2 - 
                    (2*(mbkin**2 + mckin**2)*q_cut)/mbkin**4 + q_cut**2/mbkin**4)*
                   ((64*mbkin*(-((-1 + mckin**2/mbkin**2)**2*(-219 - 
                          (3427*mckin**2)/mbkin**2 - (25220*mckin**4)/mbkin**4 + 
                          (690492*mckin**6)/mbkin**6 + (3385394*mckin**8)/
                          mbkin**8 + (3853498*mckin**10)/mbkin**10 + 
                          (969876*mckin**12)/mbkin**12 - (66556*mckin**14)/
                          mbkin**14 + (8017*mckin**16)/mbkin**16 + (3105*mckin**
                          18)/mbkin**18)) + ((-639 - (12606*mckin**2)/mbkin**2 - 
                          (137707*mckin**4)/mbkin**4 + (1200772*mckin**6)/
                          mbkin**6 + (8153946*mckin**8)/mbkin**8 + (14790212*
                          mckin**10)/mbkin**10 + (9837754*mckin**12)/mbkin**12 + 
                          (1656240*mckin**14)/mbkin**14 - (260839*mckin**16)/
                          mbkin**16 + (23302*mckin**18)/mbkin**18 + 
                          (9405*mckin**20)/mbkin**20)*q_cut)/mbkin**2 - 
                       (2*(-210 - (4891*mckin**2)/mbkin**2 - (70825*mckin**4)/
                          mbkin**4 + (10961*mckin**6)/mbkin**6 + (726865*mckin**
                          8)/mbkin**8 + (829165*mckin**10)/mbkin**10 + 
                          (50797*mckin**12)/mbkin**12 - (87657*mckin**14)/
                          mbkin**14 + (11805*mckin**16)/mbkin**16 + 
                          (3150*mckin**18)/mbkin**18)*q_cut**2)/mbkin**4 + 
                       (2*(210 + (5521*mckin**2)/mbkin**2 + (90958*mckin**4)/
                          mbkin**4 + (297789*mckin**6)/mbkin**6 + (370334*
                          mckin**8)/mbkin**8 + (299309*mckin**10)/mbkin**10 + 
                          (83742*mckin**12)/mbkin**12 - (15585*mckin**14)/
                          mbkin**14 - (3150*mckin**16)/mbkin**16)*q_cut**3)/
                        mbkin**6 + (2*(-315 - (8304*mckin**2)/mbkin**2 - 
                          (138468*mckin**4)/mbkin**4 - (460673*mckin**6)/
                          mbkin**6 - (479693*mckin**8)/mbkin**8 - (125948*
                          mckin**10)/mbkin**10 + (23580*mckin**12)/mbkin**12 + 
                          (4725*mckin**14)/mbkin**14)*q_cut**4)/mbkin**8 + 
                       ((126 + (5638*mckin**2)/mbkin**2 + (85014*mckin**4)/
                          mbkin**4 + (204504*mckin**6)/mbkin**6 + (84214*mckin**
                          8)/mbkin**8 - (13278*mckin**10)/mbkin**10 - 
                          (3570*mckin**12)/mbkin**12)*q_cut**5)/mbkin**10 + 
                       (2*(42 - (51*mckin**2)/mbkin**2 + (159*mckin**4)/mbkin**
                          4 - (541*mckin**6)/mbkin**6 - (31*mckin**8)/mbkin**8 + 
                          (210*mckin**10)/mbkin**10)*q_cut**6)/mbkin**12 + 
                       (10*(18 + (15*mckin**2)/mbkin**2 + (78*mckin**4)/mbkin**
                          4 + (17*mckin**6)/mbkin**6 + (90*mckin**8)/mbkin**8)*
                         q_cut**7)/mbkin**14 - (15*(19 + (19*mckin**2)/mbkin**2 + 
                          (47*mckin**4)/mbkin**4 + (95*mckin**6)/mbkin**6)*q_cut**8)/
                        mbkin**16 + (105*(mbkin**4 + 5*mckin**4)*q_cut**9)/mbkin**
                         22))/9 - 36*((8*mckin**2*(3 + 8*mckin)*
                        (9*(-1 + mckin**2/mbkin**2)**2*(47 + (31*mckin**2)/
                          mbkin**2 - (3105*mckin**4)/mbkin**4 - (10833*mckin**6)/
                          mbkin**6 - (10833*mckin**8)/mbkin**8 - (3105*mckin**10)/
                          mbkin**10 + (31*mckin**12)/mbkin**12 + (47*mckin**14)/
                          mbkin**14) - (3*(421 + (1354*mckin**2)/mbkin**2 - 
                          (17342*mckin**4)/mbkin**4 - (84374*mckin**6)/mbkin**6 - 
                          (132758*mckin**8)/mbkin**8 - (84374*mckin**10)/mbkin**
                          10 - (17342*mckin**12)/mbkin**12 + (1354*mckin**14)/
                          mbkin**14 + (421*mckin**16)/mbkin**16)*q_cut)/mbkin**2 + 
                         (6*(140 + (839*mckin**2)/mbkin**2 - (795*mckin**4)/
                          mbkin**4 - (7114*mckin**6)/mbkin**6 - (7114*mckin**8)/
                          mbkin**8 - (795*mckin**10)/mbkin**10 + (839*mckin**12)/
                          mbkin**12 + (140*mckin**14)/mbkin**14)*q_cut**2)/mbkin**4 + 
                         (6*(140 + (1259*mckin**2)/mbkin**2 + (3262*mckin**4)/
                          mbkin**4 + (4076*mckin**6)/mbkin**6 + (3262*mckin**8)/
                          mbkin**8 + (1259*mckin**10)/mbkin**10 + (140*mckin**12)/
                          mbkin**12)*q_cut**3)/mbkin**6 - (6*(210 + (1891*mckin**2)/
                          mbkin**2 + (4862*mckin**4)/mbkin**4 + (4862*mckin**6)/
                          mbkin**6 + (1891*mckin**8)/mbkin**8 + (210*mckin**10)/
                          mbkin**10)*q_cut**4)/mbkin**8 + ((392 + (3466*mckin**2)/
                          mbkin**2 + (6078*mckin**4)/mbkin**4 + (3466*mckin**6)/
                          mbkin**6 + (392*mckin**8)/mbkin**8)*q_cut**5)/mbkin**10 + 
                         ((28 - (34*mckin**2)/mbkin**2 - (34*mckin**4)/mbkin**4 + 
                          (28*mckin**6)/mbkin**6)*q_cut**6)/mbkin**12 + 
                         (10*(6 + (5*mckin**2)/mbkin**2 + (6*mckin**4)/mbkin**4)*
                          q_cut**7)/mbkin**14 - (95*(mbkin**2 + mckin**2)*q_cut**8)/
                          mbkin**18 + (35*q_cut**9)/mbkin**18))/9 + mckin**4*
                       ((-12*(mbkin - mckin)**2*(mbkin + mckin)*(63*mbkin**15 + 
                          63*mbkin**14*mckin + 168*mbkin**15*mckin + 6303*
                          mbkin**13*mckin**2 + 6303*mbkin**12*mckin**3 + 16808*
                          mbkin**13*mckin**3 + 20079*mbkin**11*mckin**4 + 
                          20079*mbkin**10*mckin**5 + 53544*mbkin**11*mckin**5 - 
                          10833*mbkin**9*mckin**6 - 10833*mbkin**8*mckin**7 - 
                          28888*mbkin**9*mckin**7 - 49473*mbkin**7*mckin**8 - 
                          49473*mbkin**6*mckin**9 - 131928*mbkin**7*mckin**9 - 
                          21921*mbkin**5*mckin**10 - 21921*mbkin**4*mckin**11 - 
                          58456*mbkin**5*mckin**11 - 81*mbkin**3*mckin**12 - 
                          81*mbkin**2*mckin**13 - 216*mbkin**3*mckin**13 + 
                          423*mbkin*mckin**14 + 423*mckin**15 + 1128*mbkin*
                          mckin**15))/mbkin**20 + (4*(-2799*mbkin**16 + 3368*
                          mbkin**17 - 10832*mbkin**16*mckin + 112176*mbkin**14*
                          mckin**2 + 21664*mbkin**15*mckin**2 + 277472*mbkin**14*
                          mckin**3 + 603288*mbkin**12*mckin**4 - 416208*mbkin**13*
                          mckin**4 + 2024976*mbkin**12*mckin**5 + 580608*mbkin**
                          10*mckin**6 - 2699968*mbkin**11*mckin**6 + 4248256*
                          mbkin**10*mckin**7 - 725760*mbkin**8*mckin**8 - 5310320*
                          mbkin**9*mckin**8 + 3374960*mbkin**8*mckin**9 - 1206576*
                          mbkin**6*mckin**10 - 4049952*mbkin**7*mckin**10 + 
                          832416*mbkin**6*mckin**11 - 392616*mbkin**4*mckin**12 - 
                          971152*mbkin**5*mckin**12 - 75824*mbkin**4*mckin**13 + 
                          22392*mbkin**2*mckin**14 + 86656*mbkin**3*mckin**14 - 
                          26944*mbkin**2*mckin**15 + 11367*mckin**16 + 30312*
                          mbkin*mckin**16)*q_cut)/(3*mbkin**20) - 
                        (8*(-1677*mbkin**14 + 2240*mbkin**15 - 6712*mbkin**14*
                          mckin + 12321*mbkin**12*mckin**2 + 20136*mbkin**13*
                          mckin**2 + 12720*mbkin**12*mckin**3 + 54486*mbkin**10*
                          mckin**4 - 25440*mbkin**11*mckin**4 + 170736*mbkin**10*
                          mckin**5 - 21342*mbkin**8*mckin**6 - 284560*mbkin**9*
                          mckin**6 + 227648*mbkin**8*mckin**7 - 116127*mbkin**6*
                          mckin**8 - 341472*mbkin**7*mckin**8 + 31800*mbkin**6*
                          mckin**9 - 31797*mbkin**4*mckin**10 - 44520*mbkin**5*
                          mckin**10 - 40272*mbkin**4*mckin**11 + 17196*mbkin**2*
                          mckin**12 + 53696*mbkin**3*mckin**12 - 7840*mbkin**2*
                          mckin**13 + 3780*mckin**14 + 10080*mbkin*mckin**14)*
                          q_cut**2)/(3*mbkin**20) - (8*(-2517*mbkin**12 + 3360*
                          mbkin**13 - 10072*mbkin**12*mckin - 4464*mbkin**10*
                          mckin**2 + 40288*mbkin**11*mckin**2 - 52192*mbkin**10*
                          mckin**3 + 12246*mbkin**8*mckin**4 + 130480*mbkin**9*
                          mckin**4 - 97824*mbkin**8*mckin**5 + 34224*mbkin**6*
                          mckin**6 + 195648*mbkin**7*mckin**6 - 104384*mbkin**6*
                          mckin**7 + 49617*mbkin**4*mckin**8 + 182672*mbkin**5*
                          mckin**8 - 50360*mbkin**4*mckin**9 + 27696*mbkin**2*
                          mckin**10 + 80576*mbkin**3*mckin**10 - 6720*mbkin**2*
                          mckin**11 + 3780*mckin**12 + 10080*mbkin*mckin**12)*
                          q_cut**3)/(3*mbkin**20) + (8*(-3153*mbkin**10 + 6720*
                          mbkin**11 - 15128*mbkin**10*mckin - 807*mbkin**8*
                          mckin**2 + 75640*mbkin**9*mckin**2 - 77792*mbkin**8*
                          mckin**3 + 43758*mbkin**6*mckin**4 + 233376*mbkin**7*
                          mckin**4 - 116688*mbkin**6*mckin**5 + 79410*mbkin**4*
                          mckin**6 + 272272*mbkin**5*mckin**6 - 60512*mbkin**4*
                          mckin**7 + 42234*mbkin**2*mckin**8 + 121024*mbkin**3*
                          mckin**8 - 8400*mbkin**2*mckin**9 + 5670*mckin**10 + 
                          15120*mbkin*mckin**10)*q_cut**4)/(3*mbkin**20) - 
                        (8*(-2259*mbkin**8 + 7840*mbkin**9 - 13864*mbkin**8*
                          mckin + 12960*mbkin**6*mckin**2 + 83184*mbkin**7*
                          mckin**2 - 48624*mbkin**6*mckin**3 + 48222*mbkin**4*
                          mckin**4 + 170184*mbkin**5*mckin**4 - 41592*mbkin**4*
                          mckin**5 + 39240*mbkin**2*mckin**6 + 110912*mbkin**3*
                          mckin**6 - 6272*mbkin**2*mckin**7 + 5292*mckin**8 + 
                          14112*mbkin*mckin**8)*q_cut**5)/(9*mbkin**20) - 
                        (8*(303*mbkin**6 + 672*mbkin**7 + 136*mbkin**6*mckin - 
                          255*mbkin**4*mckin**2 - 952*mbkin**5*mckin**2 + 
                          272*mbkin**4*mckin**3 - 534*mbkin**2*mckin**4 - 
                          1088*mbkin**3*mckin**4 - 336*mbkin**2*mckin**5 + 
                          378*mckin**6 + 1008*mbkin*mckin**6)*q_cut**6)/
                         (9*mbkin**20) - (40*(111*mbkin**4 + 336*mbkin**5 - 
                          40*mbkin**4*mckin + 84*mbkin**2*mckin**2 + 320*mbkin**3*
                          mckin**2 - 96*mbkin**2*mckin**3 + 162*mckin**4 + 
                          432*mbkin*mckin**4)*q_cut**7)/(9*mbkin**20) + 
                        (380*(21*mbkin**2 + 64*mbkin**3 - 8*mbkin**2*mckin + 
                          27*mckin**2 + 72*mbkin*mckin**2)*q_cut**8)/(9*mbkin**20) - 
                        (140*(3 + 8*mbkin)*q_cut**9)/mbkin**20)))))/mbkin**4)*
              np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2) - 
           60480*((-576*mckin**12*(mbkin**8 + 8*mbkin**6*mckin**2 + 15*mbkin**4*
                 mckin**4 + 8*mbkin**2*mckin**6 + mckin**8)*(mbkin**2 - 
                2*mbkin*mckin + mckin**2 - q_cut)*(mbkin**2 + 2*mbkin*mckin + 
                mckin**2 - q_cut)*(3*mbkin**4 + 8*mbkin**4*mckin - 6*mbkin**2*
                 mckin**2 - 8*mbkin**3*mckin**2 - 8*mbkin**2*mckin**3 + 
                3*mckin**4 + 8*mbkin*mckin**4 - 3*mbkin**2*q_cut - 8*mbkin**2*mckin*
                 q_cut - 3*mckin**2*q_cut - 8*mbkin*mckin**2*q_cut)*np.log(
                 (mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                        mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                       q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                    np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                        q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**2)/
              (mbkin**22*(mbkin**2 + mckin**2 - q_cut + mbkin**2*np.sqrt(0j + 
                  (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4))*(-mbkin**2 - mckin**2 + q_cut + 
                mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))) + 
             ((-32*mckin**8*(12*mbkin**11 + 148*mbkin**9*mckin**2 - 81*mbkin**10*
                   mckin**2 - 216*mbkin**10*mckin**3 + 1398*mbkin**7*mckin**4 - 
                  972*mbkin**8*mckin**4 - 2592*mbkin**8*mckin**5 + 3384*mbkin**5*
                   mckin**6 - 2106*mbkin**6*mckin**6 + 864*mbkin**7*mckin**6 - 
                  6480*mbkin**6*mckin**7 + 1898*mbkin**3*mckin**8 - 405*mbkin**4*
                   mckin**8 + 3240*mbkin**5*mckin**8 - 4320*mbkin**4*mckin**9 + 
                  156*mbkin*mckin**10 + 729*mbkin**2*mckin**10 + 2592*mbkin**3*
                   mckin**10 - 648*mbkin**2*mckin**11 + 162*mckin**12 + 
                  432*mbkin*mckin**12)*((mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**
                  (3/2))/(9*mbkin**18) - (48*mckin**10*(mbkin**8 + 8*mbkin**6*
                   mckin**2 + 15*mbkin**4*mckin**4 + 8*mbkin**2*mckin**6 + 
                  mckin**8)*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
                 (-12*mbkin**6 - 32*mbkin**6*mckin + 45*mbkin**4*mckin**2 + 
                  32*mbkin**5*mckin**2 + 88*mbkin**4*mckin**3 - 54*mbkin**2*
                   mckin**4 - 88*mbkin**3*mckin**4 - 56*mbkin**2*mckin**5 + 
                  21*mckin**6 + 56*mbkin*mckin**6 + 24*mbkin**4*q_cut + 64*mbkin**4*
                   mckin*q_cut - 88*mbkin**3*mckin**2*q_cut + 88*mbkin**2*mckin**3*q_cut - 
                  42*mckin**4*q_cut - 112*mbkin*mckin**4*q_cut - 12*mbkin**2*q_cut**2 - 
                  32*mbkin**2*mckin*q_cut**2 + 21*mckin**2*q_cut**2 + 56*mbkin*mckin**2*
                   q_cut**2))/(mbkin**18*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                  2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)))*
              np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)))/
         (((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
             2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
          ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*mckin**2 - 
               7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
               mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
            (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*
                  np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                      q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                 mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                     mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                    mbkin**4))))/mbkin**4)**3) + 
        ((18*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
              2*mckin**2*q_cut + q_cut**2)**3*(mbkin**6 - 7*mbkin**4*mckin**2 - 
              7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - mckin**4*q_cut - 
              mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3)**2*(3*mbkin**14 - 
             141*mbkin**12*mckin**2 - 7785*mbkin**10*mckin**4 - 
             33657*mbkin**8*mckin**6 - 33657*mbkin**6*mckin**8 - 
             7785*mbkin**4*mckin**10 - 141*mbkin**2*mckin**12 + 3*mckin**14 + 
             3*mbkin**12*q_cut - 132*mbkin**10*mckin**2*q_cut - 3153*mbkin**8*mckin**4*
              q_cut - 7296*mbkin**6*mckin**6*q_cut - 3153*mbkin**4*mckin**8*q_cut - 
             132*mbkin**2*mckin**10*q_cut + 3*mckin**12*q_cut + 3*mbkin**10*q_cut**2 - 
             117*mbkin**8*mckin**2*q_cut**2 - 1272*mbkin**6*mckin**4*q_cut**2 - 
             1272*mbkin**4*mckin**6*q_cut**2 - 117*mbkin**2*mckin**8*q_cut**2 + 
             3*mckin**10*q_cut**2 + 3*mbkin**8*q_cut**3 - 96*mbkin**6*mckin**2*q_cut**3 - 
             408*mbkin**4*mckin**4*q_cut**3 - 96*mbkin**2*mckin**6*q_cut**3 + 
             3*mckin**8*q_cut**3 + 3*mbkin**6*q_cut**4 - 69*mbkin**4*mckin**2*q_cut**4 - 
             69*mbkin**2*mckin**4*q_cut**4 + 3*mckin**6*q_cut**4 - 25*mbkin**4*q_cut**5 + 
             20*mbkin**2*mckin**2*q_cut**5 - 25*mckin**4*q_cut**5 - 25*mbkin**2*q_cut**6 - 
             25*mckin**2*q_cut**6 + 35*q_cut**7))/mbkin**34 - 
          (432*mckin**4*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
              2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*
            np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)*(108*mbkin**20 - 
             792*mbkin**18*mckin**2 - 13329*mbkin**16*mckin**4 + 
             40518*mbkin**14*mckin**6 + 387441*mbkin**12*mckin**8 + 
             668988*mbkin**10*mckin**10 + 387441*mbkin**8*mckin**12 + 
             40518*mbkin**6*mckin**14 - 13329*mbkin**4*mckin**16 - 
             792*mbkin**2*mckin**18 + 108*mckin**20 - 210*mbkin**18*q_cut - 
             222*mbkin**16*mckin**2*q_cut + 15402*mbkin**14*mckin**4*q_cut + 
             81210*mbkin**12*mckin**6*q_cut + 153300*mbkin**10*mckin**8*q_cut + 
             153300*mbkin**8*mckin**10*q_cut + 81210*mbkin**6*mckin**12*q_cut + 
             15402*mbkin**4*mckin**14*q_cut - 222*mbkin**2*mckin**16*q_cut - 
             210*mckin**18*q_cut - 108*mbkin**16*q_cut**2 + 552*mbkin**14*mckin**2*
              q_cut**2 + 22257*mbkin**12*mckin**4*q_cut**2 + 101844*mbkin**10*mckin**6*
              q_cut**2 + 158394*mbkin**8*mckin**8*q_cut**2 + 101844*mbkin**6*mckin**10*
              q_cut**2 + 22257*mbkin**4*mckin**12*q_cut**2 + 552*mbkin**2*mckin**14*
              q_cut**2 - 108*mckin**16*q_cut**2 + 420*mbkin**14*q_cut**3 + 
             2088*mbkin**12*mckin**2*q_cut**3 - 8028*mbkin**10*mckin**4*q_cut**3 - 
             43584*mbkin**8*mckin**6*q_cut**3 - 43584*mbkin**6*mckin**8*q_cut**3 - 
             8028*mbkin**4*mckin**10*q_cut**3 + 2088*mbkin**2*mckin**12*q_cut**3 + 
             420*mckin**14*q_cut**3 - 105*mbkin**12*q_cut**4 - 642*mbkin**10*mckin**2*
              q_cut**4 - 966*mbkin**8*mckin**4*q_cut**4 - 2118*mbkin**6*mckin**6*q_cut**4 - 
             966*mbkin**4*mckin**8*q_cut**4 - 642*mbkin**2*mckin**10*q_cut**4 - 
             105*mckin**12*q_cut**4 - 238*mbkin**10*q_cut**5 - 1650*mbkin**8*mckin**2*
              q_cut**5 - 5522*mbkin**6*mckin**4*q_cut**5 - 5522*mbkin**4*mckin**6*q_cut**5 - 
             1650*mbkin**2*mckin**8*q_cut**5 - 238*mckin**10*q_cut**5 + 
             105*mbkin**8*q_cut**6 + 940*mbkin**6*mckin**2*q_cut**6 + 1705*mbkin**4*
              mckin**4*q_cut**6 + 940*mbkin**2*mckin**6*q_cut**6 + 105*mckin**8*q_cut**6 + 
             88*mbkin**6*q_cut**7 - 284*mbkin**4*mckin**2*q_cut**7 - 284*mbkin**2*mckin**4*
              q_cut**7 + 88*mckin**6*q_cut**7 - 35*mbkin**4*q_cut**8 + 70*mbkin**2*mckin**2*
              q_cut**8 - 35*mckin**4*q_cut**8 - 60*mbkin**2*q_cut**9 - 60*mckin**2*q_cut**9 + 
             35*q_cut**10)*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*
                np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
               mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                   2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**28 + 
          (2592*mckin**8*(mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
              2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)**2*(423*mbkin**14 + 
             279*mbkin**12*mckin**2 - 27945*mbkin**10*mckin**4 - 
             97497*mbkin**8*mckin**6 - 97497*mbkin**6*mckin**8 - 
             27945*mbkin**4*mckin**10 + 279*mbkin**2*mckin**12 + 423*mckin**14 - 
             417*mbkin**12*q_cut - 3492*mbkin**10*mckin**2*q_cut - 9873*mbkin**8*
              mckin**4*q_cut - 14016*mbkin**6*mckin**6*q_cut - 9873*mbkin**4*mckin**8*
              q_cut - 3492*mbkin**2*mckin**10*q_cut - 417*mckin**12*q_cut - 
             417*mbkin**10*q_cut**2 - 3897*mbkin**8*mckin**2*q_cut**2 - 
             10932*mbkin**6*mckin**4*q_cut**2 - 10932*mbkin**4*mckin**6*q_cut**2 - 
             3897*mbkin**2*mckin**8*q_cut**2 - 417*mckin**10*q_cut**2 + 
             423*mbkin**8*q_cut**3 + 3264*mbkin**6*mckin**2*q_cut**3 + 
             5892*mbkin**4*mckin**4*q_cut**3 + 3264*mbkin**2*mckin**6*q_cut**3 + 
             423*mckin**8*q_cut**3 + 3*mbkin**6*q_cut**4 - 69*mbkin**4*mckin**2*q_cut**4 - 
             69*mbkin**2*mckin**4*q_cut**4 + 3*mckin**6*q_cut**4 - 25*mbkin**4*q_cut**5 + 
             20*mbkin**2*mckin**2*q_cut**5 - 25*mckin**4*q_cut**5 - 25*mbkin**2*q_cut**6 - 
             25*mckin**2*q_cut**6 + 35*q_cut**7)*np.log((mbkin**2 + mckin**2 - q_cut - 
                mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                    2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + 
                mckin**2 - q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                    mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**
             2)/mbkin**26 - (6531840*mckin**12*(mbkin**8 + 8*mbkin**6*mckin**2 + 
             15*mbkin**4*mckin**4 + 8*mbkin**2*mckin**6 + mckin**8)*
            ((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                     mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                   mbkin**4))/(mbkin**2 + mckin**2 - q_cut + mbkin**2*
                 np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                    2*mckin**2*q_cut + q_cut**2)/mbkin**4)))**3)/mbkin**16)*
         (((8*mbkin**2*(3 + 8*mbkin))/(9*((mbkin**4 - 2*mbkin**2*mckin**2 + 
                 mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/
                2)) - (3*mbkin**4*((-8*(mbkin - mckin)**2*(mbkin + mckin)*
                 (3*mbkin + 3*mckin + 8*mbkin*mckin))/(9*mbkin**6) + 
               (16*(4*mbkin**3 - 4*mbkin**2*mckin + 3*mckin**2 + 8*mbkin*
                   mckin**2)*q_cut)/(9*mbkin**6) - (8*(3 + 8*mbkin)*q_cut**2)/
                (9*mbkin**6)))/(2*((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
              ((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*q_cut)/
                mbkin**4 + q_cut**2/mbkin**4)))/
           ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*
                 mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
             (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                      2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - 
                  q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/
              mbkin**4)**3 - (3*mbkin**4*((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                  mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(
                (-8*(mbkin - mckin)**2*(mbkin + mckin)*(3*mbkin + 3*mckin + 
                   8*mbkin*mckin))/(9*mbkin**6) + (16*(4*mbkin**3 - 4*mbkin**2*
                    mckin + 3*mckin**2 + 8*mbkin*mckin**2)*q_cut)/(9*mbkin**6) - 
                (8*(3 + 8*mbkin)*q_cut**2)/(9*mbkin**6))*(mbkin**6 - 7*mbkin**4*
                 mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/
              (2*mbkin**6*((-1 + mckin**2/mbkin**2)**2 - (2*(mbkin**2 + mckin**2)*
                  q_cut)/mbkin**4 + q_cut**2/mbkin**4)) - 
             (4*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                  2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(21*mbkin**6 + 56*mbkin**6*
                 mckin + 21*mbkin**4*mckin**2 - 56*mbkin**5*mckin**2 + 
                112*mbkin**4*mckin**3 - 51*mbkin**2*mckin**4 - 112*mbkin**3*
                 mckin**4 - 24*mbkin**2*mckin**5 + 9*mckin**6 + 24*mbkin*
                 mckin**6 - 3*mbkin**4*q_cut - 8*mbkin**5*q_cut + 6*mbkin**2*mckin**2*
                 q_cut + 16*mbkin**2*mckin**3*q_cut - 9*mckin**4*q_cut - 24*mbkin*mckin**4*
                 q_cut - 3*mbkin**2*q_cut**2 - 16*mbkin**3*q_cut**2 + 8*mbkin**2*mckin*
                 q_cut**2 - 9*mckin**2*q_cut**2 - 24*mbkin*mckin**2*q_cut**2 + 9*q_cut**3 + 
                24*mbkin*q_cut**3))/(9*mbkin**8) - 12*((-16*mckin**4*
                 (3*mbkin**6*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 
                  8*mbkin**6*mckin*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + 
                      mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/
                     mbkin**4) - 6*mbkin**4*mckin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4) - 8*mbkin**5*mckin**2*np.sqrt(0j + (mbkin**4 - 
                      2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                       q_cut + q_cut**2)/mbkin**4) - 8*mbkin**4*mckin**3*np.sqrt(0j + 
                    (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                      2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 3*mbkin**2*mckin**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) + 8*mbkin**3*mckin**4*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mbkin**4*q_cut*
                   np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**4*mckin*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 3*mbkin**2*mckin**2*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4) - 8*mbkin**3*mckin**2*
                   q_cut*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*
                       q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4)))/(9*mbkin**4*
                 (mbkin**2 - 2*mbkin*mckin + mckin**2 - q_cut)*(mbkin**2 + 
                  2*mbkin*mckin + mckin**2 - q_cut)*(mbkin**2 + mckin**2 - q_cut + 
                  mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                      2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))*
                 (-mbkin**2 - mckin**2 + q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*
                       mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*q_cut + 
                      q_cut**2)/mbkin**4))) + ((-8*(3 + 8*mbkin)*mckin**4)/
                  (9*mbkin**6) + (8*mckin**2*(3 + 8*mckin))/(9*mbkin**4))*
                np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + (mbkin**4 - 
                       2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*mckin**2*
                        q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - q_cut + 
                   mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                       2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))))/
           (((mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 2*
                mckin**2*q_cut + q_cut**2)/mbkin**4)**(3/2)*
            ((np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                   2*mckin**2*q_cut + q_cut**2)/mbkin**4)*(mbkin**6 - 7*mbkin**4*
                  mckin**2 - 7*mbkin**2*mckin**4 + mckin**6 - mbkin**4*q_cut - 
                 mckin**4*q_cut - mbkin**2*q_cut**2 - mckin**2*q_cut**2 + q_cut**3))/mbkin**6 - 
              (12*mckin**4*np.log((mbkin**2 + mckin**2 - q_cut - mbkin**2*np.sqrt(0j + 
                     (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 2*mbkin**2*q_cut - 
                       2*mckin**2*q_cut + q_cut**2)/mbkin**4))/(mbkin**2 + mckin**2 - 
                   q_cut + mbkin**2*np.sqrt(0j + (mbkin**4 - 2*mbkin**2*mckin**2 + mckin**4 - 
                       2*mbkin**2*q_cut - 2*mckin**2*q_cut + q_cut**2)/mbkin**4))))/mbkin**
                4)**4))))/1260 )
 if any(res.imag != 0): 
                 warnings.warn('You chose a value of q_cut outside of phase space.') 
 return np.where(res.imag != 0, 0, res.real)

