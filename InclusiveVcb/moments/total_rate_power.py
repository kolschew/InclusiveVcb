import numpy as np
from mpmath import fp, polylog


def total_rate_kin(mbkin, mckin, mus, api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
 res = ( 1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
     mckin**8/mbkin**8 + (api4**3*(256/9 + (256*mbkin**4)/(81*mckin**4) -
        (2048*mbkin**2)/(243*mckin**2) - (22528*mckin**2)/(243*mbkin**2) +
        (35072*mckin**4)/(243*mbkin**4) - (8192*mckin**6)/(81*mbkin**6) +
        (6400*mckin**8)/(243*mbkin**8) + (64*(-17 + (16*mckin**2)/mbkin**2 +
           (12*mckin**4)/mbkin**4 - (16*mckin**6)/mbkin**6 +
           (5*mckin**8)/mbkin**8 - 12*np.log(mckin**2/mbkin**2)))/243))/mbkin**7 -
     (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4 +
     (api4**3*(5896/81 + (4096*mbkin**3)/(243*mckin**3) -
        (17984*mbkin**2)/(243*mckin**2) - (8192*mbkin)/(729*mckin) +
        (8192*mckin)/(81*mbkin) - (25792*mckin**2)/(729*mbkin**2) -
        (32768*mckin**3)/(81*mbkin**3) + (7552*mckin**4)/(27*mbkin**4) +
        (339968*mckin**5)/(729*mbkin**5) - (93568*mckin**6)/(243*mbkin**6) -
        (40960*mckin**7)/(243*mbkin**7) + (103480*mckin**8)/(729*mbkin**8) +
        (1024*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
           (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
           12*np.log(mckin**2/mbkin**2)))/729 - (128*np.log(mckin**2/mbkin**2))/3 -
        (128*mbkin**2*np.log(mckin**2/mbkin**2))/(9*mckin**2) +
        (128*mckin**2*np.log(mckin**2/mbkin**2))/(9*mbkin**2) +
        (32*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4 +
        (80*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
           mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/27))/
      mbkin**6 + (api4*(-260123404758497280*(1 - mckin/mbkin)**5 +
        390185107137745920*(1 - mckin/mbkin)**6 - 388690823028777984*
         (1 - mckin/mbkin)**7 + 195377647247557632*(1 - mckin/mbkin)**8 -
        34216429928650112*(1 - mckin/mbkin)**9 - 6097700477192896*
         (1 - mckin/mbkin)**10 - 3945134197513600*(1 - mckin/mbkin)**11 -
        3335963793491680*(1 - mckin/mbkin)**12 - 2990240473256960*
         (1 - mckin/mbkin)**13 - 2738659018760960*(1 - mckin/mbkin)**14 -
        2537989658886624*(1 - mckin/mbkin)**15 - 2370644934651168*
         (1 - mckin/mbkin)**16 - 2227287774097176*(1 - mckin/mbkin)**17 -
        2102198391233484*(1 - mckin/mbkin)**18 - 1991562479412482*
         (1 - mckin/mbkin)**19 - 1892686294969085*(1 - mckin/mbkin)**20 +
        132126173845585920*(1 - mckin/mbkin)**7*(np.log(2) +
          np.log(1 - mckin/mbkin)) - 66063086922792960*(1 - mckin/mbkin)**8*
         (np.log(2) + np.log(1 - mckin/mbkin)) + 11010514487132160*
         (1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)) +
        5505257243566080*(1 - mckin/mbkin)**10*(np.log(2) +
          np.log(1 - mckin/mbkin)) + 4754540346716160*(1 - mckin/mbkin)**11*
         (np.log(2) + np.log(1 - mckin/mbkin)) + 4379181898291200*
         (1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)) +
        4076007766871040*(1 - mckin/mbkin)**13*(np.log(2) +
          np.log(1 - mckin/mbkin)) + 3808925793953280*(1 - mckin/mbkin)**14*
         (np.log(2) + np.log(1 - mckin/mbkin)) + 3570717547837440*
         (1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin)) +
        3357773812673280*(1 - mckin/mbkin)**16*(np.log(2) +
          np.log(1 - mckin/mbkin)) + 3167016139647360*(1 - mckin/mbkin)**17*
         (np.log(2) + np.log(1 - mckin/mbkin)) + 2995631463464640*
         (1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin)) +
        2841110280328320*(1 - mckin/mbkin)**19*(np.log(2) +
          np.log(1 - mckin/mbkin)) + 2701265736929760*(1 - mckin/mbkin)**20*
         (np.log(2) + np.log(1 - mckin/mbkin))))/20322140996757600 +
     (np.sqrt(0j + (-1 + mckin**2/mbkin**2)**2)*
        (18*mbkin**2*((-3 + (5*mckin**2)/mbkin**2 - (19*mckin**4)/mbkin**4 +
             (5*mckin**6)/mbkin**6)*muG - (1 - (7*mckin**2)/mbkin**2 -
             (7*mckin**4)/mbkin**4 + mckin**6/mbkin**6)*mupi) +
         32*(-2 - (2*mckin**2)/mbkin**2 - (11*mckin**4)/mbkin**4 +
           (9*mckin**6)/mbkin**6)*rE + 256*rG - (80*mckin**2*rG)/mbkin**2 +
         (64*mckin**4*rG)/mbkin**4 - (48*mckin**6*rG)/mbkin**6 +
         24*mbkin*(17 + mckin**2/mbkin**2 - (11*mckin**4)/mbkin**4 +
           (5*mckin**6)/mbkin**6)*rhoD - 24*sB - (72*mckin**2*sB)/mbkin**2 +
         (216*mckin**4*sB)/mbkin**4 - (120*mckin**6*sB)/mbkin**6 + 200*sE -
         (88*mckin**2*sE)/mbkin**2 - (88*mckin**4*sE)/mbkin**4 +
         (72*mckin**6*sE)/mbkin**6 - 25*sqB + (23*mckin**2*sqB)/mbkin**2 -
         (13*mckin**4*sqB)/mbkin**4 + (3*mckin**6*sqB)/mbkin**6) -
       12*((18*mckin**4*muG)/mbkin**2 - (18*mckin**4*mupi)/mbkin**2 + 16*rE -
         16*rG - 24*mbkin*rhoD - 8*sE + sqB)*
        np.log((1 + mckin**2/mbkin**2 - np.sqrt(0j + (-1 + mckin**2/mbkin**2)**2))/
          (1 + mckin**2/mbkin**2 + np.sqrt(0j + (-1 + mckin**2/mbkin**2)**2))))/
      (36*mbkin**4) + api4**2*((-9.854113931163779e19*(1 - mckin/mbkin)**5 -
         2.5903365411196256e20*(1 - mckin/mbkin)**6 - 1.5679296844106574e21*
          (1 - mckin/mbkin)**7 + 1.602586733095746e21*(1 - mckin/mbkin)**8 -
         3.731382058497059e20*(1 - mckin/mbkin)**9 - 1.442450583699921e20*
          (1 - mckin/mbkin)**10 - 1.0117940133228708e21*(1 - mckin/mbkin)**
           11 - 1.4567466646450717e21*(1 - mckin/mbkin)**12 -
         2.2050041791842524e21*(1 - mckin/mbkin)**13 - 3.10373436948258e21*
          (1 - mckin/mbkin)**14 - 4.17958779756945e21*(1 - mckin/mbkin)**15 -
         58963096098466560000*(1 - mckin/mbkin)**5*np.pi**2 +
         88444644147699840000*(1 - mckin/mbkin)**6*np.pi**2 -
         65172716793401817600*(1 - mckin/mbkin)**7*np.pi**2 -
         9108973844357587200*(1 - mckin/mbkin)**8*np.pi**2 + 27694301930794699200*
          (1 - mckin/mbkin)**9*np.pi**2 + 9354724595990373600*(1 - mckin/mbkin)**10*
          np.pi**2 + 99846396491298256800*(1 - mckin/mbkin)**11*np.pi**2 +
         145282820224320979200*(1 - mckin/mbkin)**12*np.pi**2 +
         221188319752082032350*(1 - mckin/mbkin)**13*np.pi**2 +
         312313023387573082875*(1 - mckin/mbkin)**14*np.pi**2 +
         421384671784085884875*(1 - mckin/mbkin)**15*np.pi**2 +
         23585238439386624000*(1 - mckin/mbkin)**5*np.pi**2*np.log(2) -
         35377857659079936000*(1 - mckin/mbkin)**6*np.pi**2*np.log(2) +
         17408152181452032000*(1 - mckin/mbkin)**7*np.pi**2*np.log(2) -
         2807766480879360000*(1 - mckin/mbkin)**8*np.pi**2*np.log(2) +
         1380485186432352000*(1 - mckin/mbkin)**9*np.pi**2*np.log(2) +
         690242593216176000*(1 - mckin/mbkin)**10*np.pi**2*np.log(2) +
         508375900704672000*(1 - mckin/mbkin)**11*np.pi**2*np.log(2) +
         417442554448920000*(1 - mckin/mbkin)**12*np.pi**2*np.log(2) +
         359888250273552000*(1 - mckin/mbkin)**13*np.pi**2*np.log(2) +
         319023467138376000*(1 - mckin/mbkin)**14*np.pi**2*np.log(2) +
         287976049320960000*(1 - mckin/mbkin)**15*np.pi**2*np.log(2) +
         2156493012354474393600*(1 - mckin/mbkin)**7*
          (np.log(2) + np.log(1 - mckin/mbkin)) - 1114185917132493004800*
          (1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)) +
         72609864602488012800*(1 - mckin/mbkin)**9*(np.log(2) +
           np.log(1 - mckin/mbkin)) + 63592262841345638400*(1 - mckin/mbkin)**10*
          (np.log(2) + np.log(1 - mckin/mbkin)) + 49568665585965926400*
          (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)) +
         44918542055279692800*(1 - mckin/mbkin)**12*(np.log(2) +
           np.log(1 - mckin/mbkin)) + 42007093034271805440*(1 - mckin/mbkin)**13*
          (np.log(2) + np.log(1 - mckin/mbkin)) + 39569178342479385600*
          (1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)) +
         37379682179716836960*(1 - mckin/mbkin)**15*(np.log(2) +
           np.log(1 - mckin/mbkin)) - 53909116432883712000*(1 - mckin/mbkin)**7*
          np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) + 26954558216441856000*
          (1 - mckin/mbkin)**8*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
         4492426369406976000*(1 - mckin/mbkin)**9*np.pi**2*
          (np.log(2) + np.log(1 - mckin/mbkin)) - 2246213184703488000*
          (1 - mckin/mbkin)**10*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
         1939911386789376000*(1 - mckin/mbkin)**11*np.pi**2*
          (np.log(2) + np.log(1 - mckin/mbkin)) - 1786760487832320000*
          (1 - mckin/mbkin)**12*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
         1663061684828544000*(1 - mckin/mbkin)**13*np.pi**2*
          (np.log(2) + np.log(1 - mckin/mbkin)) - 1554088929801408000*
          (1 - mckin/mbkin)**14*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
         1456897013155584000*(1 - mckin/mbkin)**15*np.pi**2*
          (np.log(2) + np.log(1 - mckin/mbkin)) - 242591023947976704000*
          (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
         121295511973988352000*(1 - mckin/mbkin)**8*
          (np.log(2) + np.log(1 - mckin/mbkin))**2 + 13061313703646208000*
          (1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
         6530656851823104000*(1 - mckin/mbkin)**10*
          (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7243470295117056000*
          (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
         7599877016764032000*(1 - mckin/mbkin)**12*
          (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7781163757509542400*
          (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
         7874890507804320000*(1 - mckin/mbkin)**14*
          (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7913392760156083200*
          (1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin))**2)/
        16583370777693720000 + ((-260123404758497280*(1 - mckin/mbkin)**5 +
          390185107137745920*(1 - mckin/mbkin)**6 - 388690823028777984*
           (1 - mckin/mbkin)**7 + 195377647247557632*(1 - mckin/mbkin)**8 -
          34216429928650112*(1 - mckin/mbkin)**9 - 6097700477192896*
           (1 - mckin/mbkin)**10 - 3945134197513600*(1 - mckin/mbkin)**11 -
          3335963793491680*(1 - mckin/mbkin)**12 - 2990240473256960*
           (1 - mckin/mbkin)**13 - 2738659018760960*(1 - mckin/mbkin)**14 -
          2537989658886624*(1 - mckin/mbkin)**15 - 2370644934651168*
           (1 - mckin/mbkin)**16 - 2227287774097176*(1 - mckin/mbkin)**17 -
          2102198391233484*(1 - mckin/mbkin)**18 - 1991562479412482*
           (1 - mckin/mbkin)**19 - 1892686294969085*(1 - mckin/mbkin)**20 +
          132126173845585920*(1 - mckin/mbkin)**7*(np.log(2) +
            np.log(1 - mckin/mbkin)) - 66063086922792960*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 11010514487132160*
           (1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)) +
          5505257243566080*(1 - mckin/mbkin)**10*(np.log(2) +
            np.log(1 - mckin/mbkin)) + 4754540346716160*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 4379181898291200*
           (1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)) +
          4076007766871040*(1 - mckin/mbkin)**13*(np.log(2) +
            np.log(1 - mckin/mbkin)) + 3808925793953280*(1 - mckin/mbkin)**14*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 3570717547837440*
           (1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin)) +
          3357773812673280*(1 - mckin/mbkin)**16*(np.log(2) +
            np.log(1 - mckin/mbkin)) + 3167016139647360*(1 - mckin/mbkin)**17*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 2995631463464640*
           (1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin)) +
          2841110280328320*(1 - mckin/mbkin)**19*(np.log(2) +
            np.log(1 - mckin/mbkin)) + 2701265736929760*(1 - mckin/mbkin)**20*
           (np.log(2) + np.log(1 - mckin/mbkin)))*np.log(mus**2/mbkin**2))/
        9754627678443648) + api4**3*(-56.76139445601858 +
       (149.45387889939457*mckin)/mbkin + (106.6760975426464*mckin**2)/
        mbkin**2 - (867.4602119154761*mckin**3)/mbkin**3 +
       (1369.663803681558*mckin**4)/mbkin**4 - (1087.359759730835*mckin**5)/
        mbkin**5 + (528.9881451748558*mckin**6)/mbkin**6 -
       (188.545666531547*mckin**7)/mbkin**7 + (50.04286990645883*mckin**8)/
        mbkin**8 - (6.740744276484015*mckin**9)/mbkin**9 +
       (0.4525459396358147*mckin**10)/mbkin**10 -
       (0.05787111672406232*mckin**11)/mbkin**11 +
       (0.0036502071398262868*mckin**12)/mbkin**12 - (8*mckin*(-53 + 12*np.pi**2))/
        (5*mbkin) + (4*mckin**2*(-53 + 12*np.pi**2))/mbkin**2 -
       (16*mckin**3*(-53 + 12*np.pi**2))/(3*mbkin**3) + (4*mckin**4*(-53 + 12*np.pi**2))/
        mbkin**4 - (8*mckin**5*(-53 + 12*np.pi**2))/(5*mbkin**5) +
       (4*mckin**6*(-53 + 12*np.pi**2))/(15*mbkin**6) + (8*mckin*(-25 + 12*np.pi**2))/
        (9*mbkin) - (16*mckin**2*(-25 + 12*np.pi**2))/(9*mbkin**2) +
       (16*mckin**3*(-25 + 12*np.pi**2))/(9*mbkin**3) - (8*mckin**4*(-25 + 12*np.pi**2))/
        (9*mbkin**4) + (8*mckin**5*(-25 + 12*np.pi**2))/(45*mbkin**5) -
       (32*mckin*(-425 + 42*np.pi**2))/(27*mbkin) + (64*mckin**2*(-425 + 42*np.pi**2))/
        (27*mbkin**2) - (64*mckin**3*(-425 + 42*np.pi**2))/(27*mbkin**3) +
       (32*mckin**4*(-425 + 42*np.pi**2))/(27*mbkin**4) -
       (32*mckin**5*(-425 + 42*np.pi**2))/(135*mbkin**5) +
       (8*(1 - mckin/mbkin)**8*(2.3602057966924995e7 - 9.032110830091551e6*
           np.log(1 - mckin/mbkin)))/8037225 +
       ((1 - mckin/mbkin)**8*(27777.593289938988 -
          107520*np.log(1 - mckin/mbkin)))/99225 +
       (2*(1 - mckin/mbkin)**7*(-6.295573325216855e6 +
          322560*np.log(1 - mckin/mbkin)))/297675 +
       (2*(1 - mckin/mbkin)**7*(-46444.24492436717 +
          322560*np.log(1 - mckin/mbkin)))/297675 +
       ((1 - mckin/mbkin)**9*(-4.369460867520098e6 +
          3870720*np.log(1 - mckin/mbkin)))/5358150 +
       ((1 - mckin/mbkin)**10*(-307526.4832868618 +
          3870720*np.log(1 - mckin/mbkin)))/10716300 +
       (32*(1 - mckin/mbkin)**7*(-2.2454205851842865e7 +
          4.5160554150457755e6*np.log(1 - mckin/mbkin)))/8037225 +
       ((1 - mckin/mbkin)**10*(-5.947216996175766e7 +
          56972160*np.log(1 - mckin/mbkin)))/642978000 +
       ((1 - mckin/mbkin)**9*(-8.339744456062602e9 +
          59149440*np.log(1 - mckin/mbkin)))/321489000 +
       ((1 - mckin/mbkin)**9*(-1.6761393013254833e8 +
          59149440*np.log(1 - mckin/mbkin)))/321489000 +
       ((1 - mckin/mbkin)**11*(-2.8093281821019053e7 +
          106444800*np.log(1 - mckin/mbkin)))/294698250 +
       ((1 - mckin/mbkin)**12*(-3.3513457817542505e8 +
          1277337600*np.log(1 - mckin/mbkin)))/3536379000 +
       ((1 - mckin/mbkin)**11*(-1.065338276125e9 + 28320171264*
           np.log(1 - mckin/mbkin)))/363068244000 +
       ((1 - mckin/mbkin)**12*(6.962026213725e10 + 678719897856*
           np.log(1 - mckin/mbkin)))/9439774344000 +
       ((1 - mckin/mbkin)**12*(5.45419807026e12 + 7566763842237600*
           np.log(1 - mckin/mbkin) - 773781496488000*np.pi**2*np.log(1 - mckin/mbkin)))/
        477888576165000 + ((1 - mckin/mbkin)**11*(1.5690035120251562e11 +
          117773082504000*np.log(1 - mckin/mbkin) - 12000054528000*np.pi**2*
           np.log(1 - mckin/mbkin)))/4084517745000 +
       ((1 - mckin/mbkin)**10*(1.8731763771455002e8 + 54731779200*
           np.log(1 - mckin/mbkin) - 5576497920*np.pi**2*np.log(1 - mckin/mbkin)))/
        1736040600 + ((1 - mckin/mbkin)**9*(1.1831679430663261e9 +
          23481521280*np.log(1 - mckin/mbkin) - 2418716160*np.pi**2*
           np.log(1 - mckin/mbkin)))/868020300 +
       ((1 - mckin/mbkin)**9*(1.6859200907665215e9 + 23481521280*
           np.log(1 - mckin/mbkin) - 2418716160*np.pi**2*np.log(1 - mckin/mbkin)))/
        868020300 + (4*(1 - mckin/mbkin)**7*(4.892181089024176e6 +
          3343200*np.log(1 - mckin/mbkin) - 443520*np.pi**2*np.log(1 - mckin/mbkin)))/
        893025 + (4*(1 - mckin/mbkin)**7*(7.607828682783567e6 +
          3343200*np.log(1 - mckin/mbkin) - 443520*np.pi**2*np.log(1 - mckin/mbkin)))/
        893025 + (4*(1 - mckin/mbkin)**8*(-1.7233637022910248e6 +
          3872400*np.log(1 - mckin/mbkin) - 332640*np.pi**2*np.log(1 - mckin/mbkin)))/
        893025 + (4*(1 - mckin/mbkin)**8*(-2.348464802626316e6 -
          6812400*np.log(1 - mckin/mbkin) + 776160*np.pi**2*np.log(1 - mckin/mbkin)))/
        893025 + ((1 - mckin/mbkin)**12*(-4.7368961117082525e14 +
          10232712960069600*np.log(1 - mckin/mbkin) - 493951289832000*np.pi**2*
           np.log(1 - mckin/mbkin) - 4078781650944000*np.log(2)*
           np.log(1 - mckin/mbkin) - 2039390825472000*np.log(1 - mckin/mbkin)**2))/
        1433665728495000 + ((1 - mckin/mbkin)**11*(-1.320924219396316e13 -
          113272940088000*np.log(1 - mckin/mbkin) + 25731566784000*np.pi**2*
           np.log(1 - mckin/mbkin) - 98354995200000*np.log(2)*
           np.log(1 - mckin/mbkin) - 49177497600000*np.log(1 - mckin/mbkin)**2))/
        36760659705000 + ((1 - mckin/mbkin)**10*(7.362472748438601e9 +
          1.944038289073944e9*np.log(1 - mckin/mbkin) -
          6604416000*np.log(1 - mckin/mbkin)**2))/7595177625 +
       (2*(1 - mckin/mbkin)**9*(-2.043347784041684e10 +
          1.5152870289073944e10*np.log(1 - mckin/mbkin) -
          6604416000*np.log(1 - mckin/mbkin)**2))/7595177625 +
       ((1 - mckin/mbkin)**10*(-7.816387617119295e9 +
          3844874880*np.log(1 - mckin/mbkin) + 1724486400*np.pi**2*
           np.log(1 - mckin/mbkin) - 10838016000*np.log(2)*np.log(1 - mckin/mbkin) -
          5419008000*np.log(1 - mckin/mbkin)**2))/4050761400 +
       (4*(1 - mckin/mbkin)**8*(2.959923218393065e8 + 3.9666592112035346e8*
           np.log(1 - mckin/mbkin) - 64033200*np.log(1 - mckin/mbkin)**2))/
        31255875 - (4*(1.4998040391086718e8 + 1.1006631214406794e7*
           np.log(1 - mckin/mbkin) + 1058400*np.log(1 - mckin/mbkin)**2))/31255875 +
       (32*mckin*(1.4998040391086718e8 + 1.1006631214406794e7*
           np.log(1 - mckin/mbkin) + 1058400*np.log(1 - mckin/mbkin)**2))/
        (31255875*mbkin) - (16*mckin**2*(1.4998040391086718e8 +
          1.1006631214406794e7*np.log(1 - mckin/mbkin) +
          1058400*np.log(1 - mckin/mbkin)**2))/(4465125*mbkin**2) +
       (32*mckin**3*(1.4998040391086718e8 + 1.1006631214406794e7*
           np.log(1 - mckin/mbkin) + 1058400*np.log(1 - mckin/mbkin)**2))/
        (4465125*mbkin**3) - (8*mckin**4*(1.4998040391086718e8 +
          1.1006631214406794e7*np.log(1 - mckin/mbkin) +
          1058400*np.log(1 - mckin/mbkin)**2))/(893025*mbkin**4) +
       (32*mckin**5*(1.4998040391086718e8 + 1.1006631214406794e7*
           np.log(1 - mckin/mbkin) + 1058400*np.log(1 - mckin/mbkin)**2))/
        (4465125*mbkin**5) - (16*mckin**6*(1.4998040391086718e8 +
          1.1006631214406794e7*np.log(1 - mckin/mbkin) +
          1058400*np.log(1 - mckin/mbkin)**2))/(4465125*mbkin**6) +
       (32*mckin**7*(1.4998040391086718e8 + 1.1006631214406794e7*
           np.log(1 - mckin/mbkin) + 1058400*np.log(1 - mckin/mbkin)**2))/
        (31255875*mbkin**7) - (4*mckin**8*(1.4998040391086718e8 +
          1.1006631214406794e7*np.log(1 - mckin/mbkin) +
          1058400*np.log(1 - mckin/mbkin)**2))/(31255875*mbkin**8) +
       (4*(3.571635547133767e8 + 6.858286242881359e7*np.log(1 - mckin/mbkin) +
          2116800*np.log(1 - mckin/mbkin)**2))/31255875 -
       (4*mckin*(3.571635547133767e8 + 6.858286242881359e7*
           np.log(1 - mckin/mbkin) + 2116800*np.log(1 - mckin/mbkin)**2))/
        (4465125*mbkin) + (4*mckin**2*(3.571635547133767e8 +
          6.858286242881359e7*np.log(1 - mckin/mbkin) +
          2116800*np.log(1 - mckin/mbkin)**2))/(1488375*mbkin**2) -
       (4*mckin**3*(3.571635547133767e8 + 6.858286242881359e7*
           np.log(1 - mckin/mbkin) + 2116800*np.log(1 - mckin/mbkin)**2))/
        (893025*mbkin**3) + (4*mckin**4*(3.571635547133767e8 +
          6.858286242881359e7*np.log(1 - mckin/mbkin) +
          2116800*np.log(1 - mckin/mbkin)**2))/(893025*mbkin**4) -
       (4*mckin**5*(3.571635547133767e8 + 6.858286242881359e7*
           np.log(1 - mckin/mbkin) + 2116800*np.log(1 - mckin/mbkin)**2))/
        (1488375*mbkin**5) + (4*mckin**6*(3.571635547133767e8 +
          6.858286242881359e7*np.log(1 - mckin/mbkin) +
          2116800*np.log(1 - mckin/mbkin)**2))/(4465125*mbkin**6) -
       (4*mckin**7*(3.571635547133767e8 + 6.858286242881359e7*
           np.log(1 - mckin/mbkin) + 2116800*np.log(1 - mckin/mbkin)**2))/
        (31255875*mbkin**7) + (8*(1 - mckin/mbkin)**7*(-3.8079727309483135e8 -
          5.2473232112035346e8*np.log(1 - mckin/mbkin) +
          64033200*np.log(1 - mckin/mbkin)**2))/31255875 +
       ((1 - mckin/mbkin)**8*(7.911582676799587e8 - 1215997440*
           np.log(1 - mckin/mbkin) + 22579200*np.pi**2*np.log(1 - mckin/mbkin) +
          248371200*np.log(2)*np.log(1 - mckin/mbkin) + 124185600*
           np.log(1 - mckin/mbkin)**2))/10418625 +
       ((1 - mckin/mbkin)**10*(2.383037195768547e9 - 90712621440*
           np.log(1 - mckin/mbkin) + 1625702400*np.pi**2*np.log(1 - mckin/mbkin) +
          17882726400*np.log(2)*np.log(1 - mckin/mbkin) + 8941363200*
           np.log(1 - mckin/mbkin)**2))/4500846000 +
       ((1 - mckin/mbkin)**11*(4.354794724084651e11 - 7321334904576*
           np.log(1 - mckin/mbkin) + 131139993600*np.pi**2*np.log(1 - mckin/mbkin) +
          1442539929600*np.log(2)*np.log(1 - mckin/mbkin) + 721269964800*
           np.log(1 - mckin/mbkin)**2))/363068244000 +
       ((1 - mckin/mbkin)**12*(1.7908257007533662e14 - 2768748983320320*
           np.log(1 - mckin/mbkin) + 48974826700800*np.pi**2*np.log(1 - mckin/mbkin) +
          538723093708800*np.log(2)*np.log(1 - mckin/mbkin) + 269361546854400*
           np.log(1 - mckin/mbkin)**2))/141596615160000 +
       (1 - mckin/mbkin)**8*(-22.325561696147247 +
         (1670656*(np.log(2) + np.log(1 - mckin/mbkin)))/99225 -
         (2048*(np.log(2) + np.log(1 - mckin/mbkin))**2)/945) +
       (1 - mckin/mbkin)**11*(-1.9445235887299503 +
         (2649344*(np.log(2) + np.log(1 - mckin/mbkin)))/893025 -
         (1024*(np.log(2) + np.log(1 - mckin/mbkin))**2)/2835) +
       (1 - mckin/mbkin)**10*(-1.806678800052964 +
         (2649344*(np.log(2) + np.log(1 - mckin/mbkin)))/893025 -
         (1024*(np.log(2) + np.log(1 - mckin/mbkin))**2)/2835) +
       (1 - mckin/mbkin)**12*(-1.9385065147452105 +
         (4678353472*(np.log(2) + np.log(1 - mckin/mbkin)))/1620840375 -
         (161792*(np.log(2) + np.log(1 - mckin/mbkin))**2)/467775) +
       6*(1 - mckin/mbkin)**7*(235.63371610028608 - 195.47800292021884*
          (np.log(2) + np.log(1 - mckin/mbkin)) +
         (5175584*(np.log(2) + np.log(1 - mckin/mbkin))**2)/99225 -
         (512*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/945 -
         (11264*(np.log(2) + np.log(1 - mckin/mbkin))**3)/2835) +
       12*(1 - mckin/mbkin)**8*(184.49777327012214 - 125.53246768203891*
          (np.log(2) + np.log(1 - mckin/mbkin)) +
         (3844444*(np.log(2) + np.log(1 - mckin/mbkin))**2)/99225 -
         (704*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/945 -
         (7744*(np.log(2) + np.log(1 - mckin/mbkin))**3)/2835) +
       (16*(1 - mckin/mbkin)**9*(37.48563761588568 - 51.513921703543545*
           (np.log(2) + np.log(1 - mckin/mbkin)) +
          (30410*(np.log(2) + np.log(1 - mckin/mbkin))**2)/1323 -
          (640*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/1701 -
          (3520*(np.log(2) + np.log(1 - mckin/mbkin))**3)/1701))/3 +
       (16*(1 - mckin/mbkin)**10*(3.456380558677994 - 15.177877114399013*
           (np.log(2) + np.log(1 - mckin/mbkin)) +
          (124525*(np.log(2) + np.log(1 - mckin/mbkin))**2)/11907 -
          (320*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/1701 -
          (1760*(np.log(2) + np.log(1 - mckin/mbkin))**3)/1701))/3 +
       (16*(1 - mckin/mbkin)**11*(7.7631058746446495 - 16.67843069242872*
           (np.log(2) + np.log(1 - mckin/mbkin)) +
          (114228047*(np.log(2) + np.log(1 - mckin/mbkin))**2)/10914750 -
          (512*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/2835 -
          (2816*(np.log(2) + np.log(1 - mckin/mbkin))**3)/2835))/3 +
       (16*(1 - mckin/mbkin)**12*(8.24248771286603 - 16.640444128266363*
           (np.log(2) + np.log(1 - mckin/mbkin)) +
          (2043951619*(np.log(2) + np.log(1 - mckin/mbkin))**2)/196465500 -
          (1504*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/8505 -
          (8272*(np.log(2) + np.log(1 - mckin/mbkin))**3)/8505))/3 +
       3*(1 - mckin/mbkin)**8*(25.303112811927434 -
         (197673088*(np.log(2) + np.log(1 - mckin/mbkin)))/10418625 +
         (512*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/2835 +
         (417664*(np.log(2) + np.log(1 - mckin/mbkin))**2)/99225 -
         (1024*(np.log(2) + np.log(1 - mckin/mbkin))**3)/2835) +
       6*(1 - mckin/mbkin)**9*(20.41115474942871 - 19.84841378705638*
          (np.log(2) + np.log(1 - mckin/mbkin)) +
         (4302248*(np.log(2) + np.log(1 - mckin/mbkin))**2)/893025 -
         (128*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/2835 -
         (2816*(np.log(2) + np.log(1 - mckin/mbkin))**3)/8505) +
       6*(1 - mckin/mbkin)**10*(4.718152899276642 - 9.9016319376199*
          (np.log(2) + np.log(1 - mckin/mbkin)) +
         (2151124*(np.log(2) + np.log(1 - mckin/mbkin))**2)/893025 -
         (64*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/2835 -
         (1408*(np.log(2) + np.log(1 - mckin/mbkin))**3)/8505) +
       6*(1 - mckin/mbkin)**11*(4.837117157409 - 9.364665308703108*
          (np.log(2) + np.log(1 - mckin/mbkin)) +
         (4273226*(np.log(2) + np.log(1 - mckin/mbkin))**2)/1964655 -
         (608*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/31185 -
         (1216*(np.log(2) + np.log(1 - mckin/mbkin))**3)/8505) +
       6*(1 - mckin/mbkin)**12*(5.027359184971152 - 9.098704574545069*
          (np.log(2) + np.log(1 - mckin/mbkin)) +
         (20218013*(np.log(2) + np.log(1 - mckin/mbkin))**2)/9823275 -
         (16*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/891 -
         (32*(np.log(2) + np.log(1 - mckin/mbkin))**3)/243) +
       3*(1 - mckin/mbkin)**12*(-0.882260576835832 +
         (649842210604*(np.log(2) + np.log(1 - mckin/mbkin)))/374414126625 -
         (32*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/2673 -
         (36540344*(np.log(2) + np.log(1 - mckin/mbkin))**2)/108056025 +
         (64*(np.log(2) + np.log(1 - mckin/mbkin))**3)/2673) +
       3*(1 - mckin/mbkin)**11*(-0.8387292813277862 +
         (95669990056*(np.log(2) + np.log(1 - mckin/mbkin)))/53487732375 -
         (1216*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/93555 -
         (38440112*(np.log(2) + np.log(1 - mckin/mbkin))**2)/108056025 +
         (2432*(np.log(2) + np.log(1 - mckin/mbkin))**3)/93555) +
       3*(1 - mckin/mbkin)**10*(-0.7947208789824312 +
         (532971728*(np.log(2) + np.log(1 - mckin/mbkin)))/281302875 -
         (128*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/8505 -
         (349088*(np.log(2) + np.log(1 - mckin/mbkin))**2)/893025 +
         (256*(np.log(2) + np.log(1 - mckin/mbkin))**3)/8505) +
       3*(1 - mckin/mbkin)**9*(-3.4503123188142952 +
         (1065943456*(np.log(2) + np.log(1 - mckin/mbkin)))/281302875 -
         (256*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/8505 -
         (698176*(np.log(2) + np.log(1 - mckin/mbkin))**2)/893025 +
         (512*(np.log(2) + np.log(1 - mckin/mbkin))**3)/8505) +
       12*(1 - mckin/mbkin)**12*(-6.628186181687086 + 11.882173553991642*
          (np.log(2) + np.log(1 - mckin/mbkin)) -
         (197730229*(np.log(2) + np.log(1 - mckin/mbkin))**2)/65488500 +
         (4*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/81 +
         (44*(np.log(2) + np.log(1 - mckin/mbkin))**3)/243) +
       (64*(1 - mckin/mbkin)**11*(-2.424142371531664 + 3.265917877575156*
           (np.log(2) + np.log(1 - mckin/mbkin)) -
          (24536128*(np.log(2) + np.log(1 - mckin/mbkin))**2)/15436575 +
          (512*(np.log(2) + np.log(1 - mckin/mbkin))**3)/2673))/27 +
       12*(1 - mckin/mbkin)**11*(-6.2833830996753735 + 12.199951856327948*
          (np.log(2) + np.log(1 - mckin/mbkin)) -
         (314971639*(np.log(2) + np.log(1 - mckin/mbkin))**2)/98232750 +
         (152*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/2835 +
         (1672*(np.log(2) + np.log(1 - mckin/mbkin))**3)/8505) +
       12*(1 - mckin/mbkin)**10*(-6.0861610995868 + 12.865076522568618*
          (np.log(2) + np.log(1 - mckin/mbkin)) -
         (355277*(np.log(2) + np.log(1 - mckin/mbkin))**2)/99225 +
         (176*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/2835 +
         (1936*(np.log(2) + np.log(1 - mckin/mbkin))**3)/8505) +
       (64*(1 - mckin/mbkin)**12*(-1.7351238787249756 + 3.3068321667917404*
           (np.log(2) + np.log(1 - mckin/mbkin)) -
          (1406552*(np.log(2) + np.log(1 - mckin/mbkin))**2)/735075 +
          (256*(np.log(2) + np.log(1 - mckin/mbkin))**3)/891))/27 +
       (8*(1 - mckin/mbkin)**12*(-2.867300487076906 + 6.096328539298564*
           (np.log(2) + np.log(1 - mckin/mbkin)) -
          (32431753*(np.log(2) + np.log(1 - mckin/mbkin))**2)/9823275 +
          (3008*(np.log(2) + np.log(1 - mckin/mbkin))**3)/8505))/3 +
       (8*(1 - mckin/mbkin)**11*(-2.67188877043403 + 6.124714883326739*
           (np.log(2) + np.log(1 - mckin/mbkin)) -
          (10846174*(np.log(2) + np.log(1 - mckin/mbkin))**2)/3274425 +
          (1024*(np.log(2) + np.log(1 - mckin/mbkin))**3)/2835))/3 +
       (8*(1 - mckin/mbkin)**10*(-0.9049885670426162 + 5.559745071498317*
           (np.log(2) + np.log(1 - mckin/mbkin)) -
          (4348*(np.log(2) + np.log(1 - mckin/mbkin))**2)/1323 +
          (640*(np.log(2) + np.log(1 - mckin/mbkin))**3)/1701))/3 +
       12*(1 - mckin/mbkin)**9*(-25.522085056088876 + 25.834562216213072*
          (np.log(2) + np.log(1 - mckin/mbkin)) -
         (710554*(np.log(2) + np.log(1 - mckin/mbkin))**2)/99225 +
         (352*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/2835 +
         (3872*(np.log(2) + np.log(1 - mckin/mbkin))**3)/8505) +
       3*(1 - mckin/mbkin)**7*(-41.02718494058007 +
         (395346176*(np.log(2) + np.log(1 - mckin/mbkin)))/10418625 -
         (1024*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/2835 -
         (835328*(np.log(2) + np.log(1 - mckin/mbkin))**2)/99225 +
         (2048*(np.log(2) + np.log(1 - mckin/mbkin))**3)/2835) +
       (8*(1 - mckin/mbkin)**9*(-14.34990784561841 + 19.651270701584956*
           (np.log(2) + np.log(1 - mckin/mbkin)) -
          (87224*(np.log(2) + np.log(1 - mckin/mbkin))**2)/11907 +
          (1280*(np.log(2) + np.log(1 - mckin/mbkin))**3)/1701))/3 +
       6*(1 - mckin/mbkin)**8*(-148.06385569660222 + 97.84058876169672*
          (np.log(2) + np.log(1 - mckin/mbkin)) -
         (2587792*(np.log(2) + np.log(1 - mckin/mbkin))**2)/99225 +
         (256*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/945 +
         (5632*(np.log(2) + np.log(1 - mckin/mbkin))**3)/2835) +
       12*(1 - mckin/mbkin)**7*(-281.1426896294166 + 250.81096711010957*
          (np.log(2) + np.log(1 - mckin/mbkin)) -
         (7688888*(np.log(2) + np.log(1 - mckin/mbkin))**2)/99225 +
         (1408*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin))**2)/945 +
         (15488*(np.log(2) + np.log(1 - mckin/mbkin))**3)/2835) -
       (23*(9.854113931163779e19*(1 - mckin/mbkin)**5 +
          2.5903365411196256e20*(1 - mckin/mbkin)**6 + 1.5679296844106574e21*
           (1 - mckin/mbkin)**7 - 1.602586733095746e21*(1 - mckin/mbkin)**8 +
          3.731382058497059e20*(1 - mckin/mbkin)**9 + 1.442450583699921e20*
           (1 - mckin/mbkin)**10 + 1.0117940133228708e21*(1 - mckin/mbkin)**
            11 + 1.4567466646450717e21*(1 - mckin/mbkin)**12 +
          2.2050041791842524e21*(1 - mckin/mbkin)**13 + 3.10373436948258e21*
           (1 - mckin/mbkin)**14 + 4.17958779756945e21*(1 - mckin/mbkin)**15 +
          58963096098466560000*(1 - mckin/mbkin)**5*np.pi**2 -
          88444644147699840000*(1 - mckin/mbkin)**6*np.pi**2 +
          65172716793401817600*(1 - mckin/mbkin)**7*np.pi**2 +
          9108973844357587200*(1 - mckin/mbkin)**8*np.pi**2 - 27694301930794699200*
           (1 - mckin/mbkin)**9*np.pi**2 - 9354724595990373600*(1 - mckin/mbkin)**
            10*np.pi**2 - 99846396491298256800*(1 - mckin/mbkin)**11*np.pi**2 -
          145282820224320979200*(1 - mckin/mbkin)**12*np.pi**2 -
          221188319752082032350*(1 - mckin/mbkin)**13*np.pi**2 -
          312313023387573082875*(1 - mckin/mbkin)**14*np.pi**2 -
          421384671784085884875*(1 - mckin/mbkin)**15*np.pi**2 -
          23585238439386624000*(1 - mckin/mbkin)**5*np.pi**2*np.log(2) +
          35377857659079936000*(1 - mckin/mbkin)**6*np.pi**2*np.log(2) -
          17408152181452032000*(1 - mckin/mbkin)**7*np.pi**2*np.log(2) +
          2807766480879360000*(1 - mckin/mbkin)**8*np.pi**2*np.log(2) -
          1380485186432352000*(1 - mckin/mbkin)**9*np.pi**2*np.log(2) -
          690242593216176000*(1 - mckin/mbkin)**10*np.pi**2*np.log(2) -
          508375900704672000*(1 - mckin/mbkin)**11*np.pi**2*np.log(2) -
          417442554448920000*(1 - mckin/mbkin)**12*np.pi**2*np.log(2) -
          359888250273552000*(1 - mckin/mbkin)**13*np.pi**2*np.log(2) -
          319023467138376000*(1 - mckin/mbkin)**14*np.pi**2*np.log(2) -
          287976049320960000*(1 - mckin/mbkin)**15*np.pi**2*np.log(2) -
          2156493012354474393600*(1 - mckin/mbkin)**7*(np.log(2) +
            np.log(1 - mckin/mbkin)) + 1114185917132493004800*
           (1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)) -
          72609864602488012800*(1 - mckin/mbkin)**9*(np.log(2) +
            np.log(1 - mckin/mbkin)) - 63592262841345638400*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin)) - 49568665585965926400*
           (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)) -
          44918542055279692800*(1 - mckin/mbkin)**12*(np.log(2) +
            np.log(1 - mckin/mbkin)) - 42007093034271805440*(1 - mckin/mbkin)**13*
           (np.log(2) + np.log(1 - mckin/mbkin)) - 39569178342479385600*
           (1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)) -
          37379682179716836960*(1 - mckin/mbkin)**15*(np.log(2) +
            np.log(1 - mckin/mbkin)) + 53909116432883712000*(1 - mckin/mbkin)**7*
           np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) - 26954558216441856000*
           (1 - mckin/mbkin)**8*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) +
          4492426369406976000*(1 - mckin/mbkin)**9*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 2246213184703488000*
           (1 - mckin/mbkin)**10*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) +
          1939911386789376000*(1 - mckin/mbkin)**11*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 1786760487832320000*
           (1 - mckin/mbkin)**12*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) +
          1663061684828544000*(1 - mckin/mbkin)**13*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 1554088929801408000*
           (1 - mckin/mbkin)**14*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) +
          1456897013155584000*(1 - mckin/mbkin)**15*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 242591023947976704000*
           (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin))**2 -
          121295511973988352000*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin))**2 - 13061313703646208000*
           (1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin))**2 -
          6530656851823104000*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin))**2 - 7243470295117056000*
           (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin))**2 -
          7599877016764032000*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin))**2 - 7781163757509542400*
           (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin))**2 -
          7874890507804320000*(1 - mckin/mbkin)**14*
           (np.log(2) + np.log(1 - mckin/mbkin))**2 - 7913392760156083200*
           (1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin))**2)*
         np.log(mus**2/mbkin**2))/99500224666162320000 +
       (29*((-64*(1 - mckin/mbkin)**5)/5 + (96*(1 - mckin/mbkin)**6)/5 -
          (32*(-9937 + 3360*np.log(2) + 3360*np.log(1 - mckin/mbkin)))/33075 +
          (256*mckin*(-9937 + 3360*np.log(2) + 3360*np.log(1 - mckin/mbkin)))/
           (33075*mbkin) - (128*mckin**2*(-9937 + 3360*np.log(2) +
             3360*np.log(1 - mckin/mbkin)))/(4725*mbkin**2) +
          (256*mckin**3*(-9937 + 3360*np.log(2) + 3360*np.log(1 - mckin/mbkin)))/
           (4725*mbkin**3) - (64*mckin**4*(-9937 + 3360*np.log(2) +
             3360*np.log(1 - mckin/mbkin)))/(945*mbkin**4) +
          (256*mckin**5*(-9937 + 3360*np.log(2) + 3360*np.log(1 - mckin/mbkin)))/
           (4725*mbkin**5) - (128*mckin**6*(-9937 + 3360*np.log(2) +
             3360*np.log(1 - mckin/mbkin)))/(4725*mbkin**6) +
          (256*mckin**7*(-9937 + 3360*np.log(2) + 3360*np.log(1 - mckin/mbkin)))/
           (33075*mbkin**7) - (32*mckin**8*(-9937 + 3360*np.log(2) +
             3360*np.log(1 - mckin/mbkin)))/(33075*mbkin**8) +
          (32*(-19769 + 6720*np.log(2) + 6720*np.log(1 - mckin/mbkin)))/33075 -
          (32*mckin*(-19769 + 6720*np.log(2) + 6720*np.log(1 - mckin/mbkin)))/
           (4725*mbkin) + (32*mckin**2*(-19769 + 6720*np.log(2) +
             6720*np.log(1 - mckin/mbkin)))/(1575*mbkin**2) -
          (32*mckin**3*(-19769 + 6720*np.log(2) + 6720*np.log(1 - mckin/mbkin)))/
           (945*mbkin**3) + (32*mckin**4*(-19769 + 6720*np.log(2) +
             6720*np.log(1 - mckin/mbkin)))/(945*mbkin**4) -
          (32*mckin**5*(-19769 + 6720*np.log(2) + 6720*np.log(1 - mckin/mbkin)))/
           (1575*mbkin**5) + (32*mckin**6*(-19769 + 6720*np.log(2) +
             6720*np.log(1 - mckin/mbkin)))/(4725*mbkin**6) -
          (32*mckin**7*(-19769 + 6720*np.log(2) + 6720*np.log(1 - mckin/mbkin)))/
           (33075*mbkin**7) + (4*(-125299 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/297675 -
          (4*mckin*(-125299 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (33075*mbkin) + (16*mckin**2*(-125299 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(33075*mbkin**2) -
          (16*mckin**3*(-125299 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (14175*mbkin**3) + (8*mckin**4*(-125299 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(4725*mbkin**4) -
          (8*mckin**5*(-125299 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (4725*mbkin**5) + (16*mckin**6*(-125299 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(14175*mbkin**6) -
          (16*mckin**7*(-125299 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (33075*mbkin**7) + (4*mckin**8*(-125299 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(33075*mbkin**8) -
          (4*mckin**9*(-125299 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (297675*mbkin**9) + (2*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/297675 -
          (4*mckin*(-44659 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (59535*mbkin) + (2*mckin**2*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(6615*mbkin**2) -
          (16*mckin**3*(-44659 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (19845*mbkin**3) + (4*mckin**4*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(2835*mbkin**4) -
          (8*mckin**5*(-44659 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (4725*mbkin**5) + (4*mckin**6*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(2835*mbkin**6) -
          (16*mckin**7*(-44659 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (19845*mbkin**7) + (2*mckin**8*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(6615*mbkin**8) -
          (4*mckin**9*(-44659 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (59535*mbkin**9) + (2*mckin**10*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(297675*mbkin**10) +
          (4*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/1029105 -
          (4*mckin*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (93555*mbkin) + (4*mckin**2*(-49945 + 60192*np.log(2) +
             60192*np.log(1 - mckin/mbkin)))/(18711*mbkin**2) -
          (4*mckin**3*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (6237*mbkin**3) + (8*mckin**4*(-49945 + 60192*np.log(2) +
             60192*np.log(1 - mckin/mbkin)))/(6237*mbkin**4) -
          (8*mckin**5*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (4455*mbkin**5) + (8*mckin**6*(-49945 + 60192*np.log(2) +
             60192*np.log(1 - mckin/mbkin)))/(4455*mbkin**6) -
          (8*mckin**7*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (6237*mbkin**7) + (4*mckin**8*(-49945 + 60192*np.log(2) +
             60192*np.log(1 - mckin/mbkin)))/(6237*mbkin**8) -
          (4*mckin**9*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (18711*mbkin**9) + (4*mckin**10*(-49945 + 60192*np.log(2) +
             60192*np.log(1 - mckin/mbkin)))/(93555*mbkin**10) -
          (4*mckin**11*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (1029105*mbkin**11) + (32*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/15810795 -
          (32*mckin*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (1216215*mbkin) + (64*mckin**2*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(405405*mbkin**2) -
          (64*mckin**3*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (110565*mbkin**3) + (32*mckin**4*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(22113*mbkin**4) -
          (32*mckin**5*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (12285*mbkin**5) + (128*mckin**6*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(36855*mbkin**6) -
          (128*mckin**7*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (36855*mbkin**7) + (32*mckin**8*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(12285*mbkin**8) -
          (32*mckin**9*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (22113*mbkin**9) + (64*mckin**10*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(110565*mbkin**10) -
          (64*mckin**11*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (405405*mbkin**11) + (32*mckin**12*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(1216215*mbkin**12) -
          (32*mckin**13*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (15810795*mbkin**13) + (-1182523 + 1552320*np.log(2) +
            1552320*np.log(1 - mckin/mbkin))/7203735 -
          (4*mckin*(-1182523 + 1552320*np.log(2) + 1552320*np.log(1 -
                mckin/mbkin)))/(2401245*mbkin) +
          (2*mckin**2*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(218295*mbkin**2) -
          (4*mckin**3*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(130977*mbkin**3) +
          (mckin**4*(-1182523 + 1552320*np.log(2) + 1552320*np.log(1 -
                mckin/mbkin)))/(14553*mbkin**4) -
          (8*mckin**5*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(72765*mbkin**5) +
          (4*mckin**6*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(31185*mbkin**6) -
          (8*mckin**7*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(72765*mbkin**7) +
          (mckin**8*(-1182523 + 1552320*np.log(2) + 1552320*np.log(1 -
                mckin/mbkin)))/(14553*mbkin**8) -
          (4*mckin**9*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(130977*mbkin**9) +
          (2*mckin**10*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(218295*mbkin**10) -
          (4*mckin**11*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(2401245*mbkin**11) +
          (mckin**12*(-1182523 + 1552320*np.log(2) + 1552320*np.log(1 -
                mckin/mbkin)))/(7203735*mbkin**12) +
          (8*(-20507983 + 28522494*np.log(2) + 28522494*np.log(1 - mckin/mbkin)))/
           1217431215 - (16*mckin*(-20507983 + 28522494*np.log(2) +
             28522494*np.log(1 - mckin/mbkin)))/(173918745*mbkin) +
          (8*mckin**2*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(13378365*mbkin**2) -
          (32*mckin**3*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(13378365*mbkin**3) +
          (8*mckin**4*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(1216215*mbkin**4) -
          (16*mckin**5*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(1216215*mbkin**5) +
          (8*mckin**6*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**6) -
          (64*mckin**7*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(2837835*mbkin**7) +
          (8*mckin**8*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**8) -
          (16*mckin**9*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(1216215*mbkin**9) +
          (8*mckin**10*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(1216215*mbkin**10) -
          (32*mckin**11*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(13378365*mbkin**11) +
          (8*mckin**12*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(13378365*mbkin**12) -
          (16*mckin**13*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(173918745*mbkin**13) +
          (8*mckin**14*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(1217431215*mbkin**14) +
          (-33813661 + 47893560*np.log(2) + 47893560*np.log(1 - mckin/mbkin))/
           289864575 - (16*mckin*(-33813661 + 47893560*np.log(2) +
             47893560*np.log(1 - mckin/mbkin)))/(289864575*mbkin) +
          (8*mckin**2*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(19324305*mbkin**2) -
          (16*mckin**3*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(8281845*mbkin**3) +
          (4*mckin**4*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(637065*mbkin**4) -
          (16*mckin**5*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(1061775*mbkin**5) +
          (8*mckin**6*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(289575*mbkin**6) -
          (16*mckin**7*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**7) +
          (2*mckin**8*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(45045*mbkin**8) -
          (16*mckin**9*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**9) +
          (8*mckin**10*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(289575*mbkin**10) -
          (16*mckin**11*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(1061775*mbkin**11) +
          (4*mckin**12*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(637065*mbkin**12) -
          (16*mckin**13*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(8281845*mbkin**13) +
          (8*mckin**14*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(19324305*mbkin**14) -
          (16*mckin**15*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(289864575*mbkin**15) +
          (mckin**16*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(289864575*mbkin**16) +
          (-253404061 + 356516160*np.log(2) + 356516160*np.log(1 - mckin/mbkin))/
           2029052025 - (mckin*(-253404061 + 356516160*np.log(2) +
             356516160*np.log(1 - mckin/mbkin)))/(135270135*mbkin) +
          (mckin**2*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(19324305*mbkin**2) -
          (mckin**3*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(4459455*mbkin**3) +
          (mckin**4*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(1486485*mbkin**4) -
          (mckin**5*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(675675*mbkin**5) +
          (mckin**6*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**6) -
          (mckin**7*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(315315*mbkin**7) +
          (mckin**8*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(315315*mbkin**8) -
          (mckin**9*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**9) +
          (mckin**10*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(675675*mbkin**10) -
          (mckin**11*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(1486485*mbkin**11) +
          (mckin**12*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(4459455*mbkin**12) -
          (mckin**13*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(19324305*mbkin**13) +
          (mckin**14*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(135270135*mbkin**14) -
          (mckin**15*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(2029052025*mbkin**15) +
          (-36724834687 + 52219630320*np.log(2) + 52219630320*
             np.log(1 - mckin/mbkin))/335083448700 -
          (mckin*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(19710791100*mbkin) +
          (2*mckin**2*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(4927697775*mbkin**2) -
          (2*mckin**3*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(985539555*mbkin**3) +
          (mckin**4*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(140791365*mbkin**4) -
          (mckin**5*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(54150525*mbkin**5) +
          (2*mckin**6*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(54150525*mbkin**6) -
          (2*mckin**7*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(34459425*mbkin**7) +
          (mckin**8*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(13783770*mbkin**8) -
          (mckin**9*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(13783770*mbkin**9) +
          (2*mckin**10*(-36724834687 + 52219630320*np.log(2) +
             52219630320*np.log(1 - mckin/mbkin)))/(34459425*mbkin**10) -
          (2*mckin**11*(-36724834687 + 52219630320*np.log(2) +
             52219630320*np.log(1 - mckin/mbkin)))/(54150525*mbkin**11) +
          (mckin**12*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(54150525*mbkin**12) -
          (mckin**13*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(140791365*mbkin**13) +
          (2*mckin**14*(-36724834687 + 52219630320*np.log(2) +
             52219630320*np.log(1 - mckin/mbkin)))/(985539555*mbkin**14) -
          (2*mckin**15*(-36724834687 + 52219630320*np.log(2) +
             52219630320*np.log(1 - mckin/mbkin)))/(4927697775*mbkin**15) +
          (mckin**16*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(19710791100*mbkin**16) -
          (mckin**17*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(335083448700*mbkin**17) +
          (-485272020137 + 691512341520*np.log(2) + 691512341520*
             np.log(1 - mckin/mbkin))/4691168281800 -
          (mckin*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(260620460100*mbkin) +
          (mckin**2*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(30661230600*mbkin**2) -
          (2*mckin**3*(-485272020137 + 691512341520*np.log(2) +
             691512341520*np.log(1 - mckin/mbkin)))/(11497961475*mbkin**3) +
          (mckin**4*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(1533061530*mbkin**4) -
          (mckin**5*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(547521975*mbkin**5) +
          (mckin**6*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(252702450*mbkin**6) -
          (2*mckin**7*(-485272020137 + 691512341520*np.log(2) +
             691512341520*np.log(1 - mckin/mbkin)))/(294819525*mbkin**7) +
          (mckin**8*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(107207100*mbkin**8) -
          (mckin**9*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(96486390*mbkin**9) +
          (mckin**10*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(107207100*mbkin**10) -
          (2*mckin**11*(-485272020137 + 691512341520*np.log(2) +
             691512341520*np.log(1 - mckin/mbkin)))/(294819525*mbkin**11) +
          (mckin**12*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(252702450*mbkin**12) -
          (mckin**13*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(547521975*mbkin**13) +
          (mckin**14*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(1533061530*mbkin**14) -
          (2*mckin**15*(-485272020137 + 691512341520*np.log(2) +
             691512341520*np.log(1 - mckin/mbkin)))/(11497961475*mbkin**15) +
          (mckin**16*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(30661230600*mbkin**16) -
          (mckin**17*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(260620460100*mbkin**17) +
          (mckin**18*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(4691168281800*mbkin**18) +
          (-4916068298621 + 7016274641376*np.log(2) + 7016274641376*
             np.log(1 - mckin/mbkin))/52784781809760 -
          (mckin*(-4916068298621 + 7016274641376*np.log(2) + 7016274641376*
              np.log(1 - mckin/mbkin)))/(2639239090488*mbkin) +
          (mckin**2*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(277814641104*mbkin**2) -
          (mckin**3*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(46302440184*mbkin**3) +
          (mckin**4*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(10894691808*mbkin**4) -
          (mckin**5*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(3404591190*mbkin**5) +
          (mckin**6*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(1361836476*mbkin**6) -
          (mckin**7*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(680918238*mbkin**7) +
          (mckin**8*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(419026608*mbkin**8) -
          (mckin**9*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(314269956*mbkin**9) +
          (mckin**10*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(285699960*mbkin**10) -
          (mckin**11*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(314269956*mbkin**11) +
          (mckin**12*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(419026608*mbkin**12) -
          (mckin**13*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(680918238*mbkin**13) +
          (mckin**14*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(1361836476*mbkin**14) -
          (mckin**15*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(3404591190*mbkin**15) +
          (mckin**16*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(10894691808*mbkin**16) -
          (mckin**17*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(46302440184*mbkin**17) +
          (mckin**18*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(277814641104*mbkin**18) -
          (mckin**19*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(2639239090488*mbkin**19) +
          (mckin**20*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(52784781809760*mbkin**20) +
          (-995781239706241 + 1420555140164160*np.log(2) + 1420555140164160*
             np.log(1 - mckin/mbkin))/10161070498378800 -
          (mckin*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(534793184125200*
            mbkin) + (mckin**2*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(59421464902800*
            mbkin**2) - (mckin**3*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(10486140865200*
            mbkin**3) + (mckin**4*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(2621535216300*
            mbkin**4) - (mckin**5*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(873845072100*mbkin**5) +
          (mckin**6*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(374505030900*mbkin**6) -
          (mckin**7*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(201656555100*mbkin**7) +
          (mckin**8*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(134437703400*mbkin**8) -
          (mckin**9*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(109994484600*mbkin**9) +
          (mckin**10*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(109994484600*
            mbkin**10) - (mckin**11*(-995781239706241 + 1420555140164160*
              np.log(2) + 1420555140164160*np.log(1 - mckin/mbkin)))/
           (134437703400*mbkin**11) + (mckin**12*(-995781239706241 +
             1420555140164160*np.log(2) + 1420555140164160*np.log(1 -
                mckin/mbkin)))/(201656555100*mbkin**12) -
          (mckin**13*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(374505030900*
            mbkin**13) + (mckin**14*(-995781239706241 + 1420555140164160*
              np.log(2) + 1420555140164160*np.log(1 - mckin/mbkin)))/
           (873845072100*mbkin**14) - (mckin**15*(-995781239706241 +
             1420555140164160*np.log(2) + 1420555140164160*np.log(1 -
                mckin/mbkin)))/(2621535216300*mbkin**15) +
          (mckin**16*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(10486140865200*
            mbkin**16) - (mckin**17*(-995781239706241 + 1420555140164160*
              np.log(2) + 1420555140164160*np.log(1 - mckin/mbkin)))/
           (59421464902800*mbkin**17) + (mckin**18*(-995781239706241 +
             1420555140164160*np.log(2) + 1420555140164160*np.log(1 -
                mckin/mbkin)))/(534793184125200*mbkin**18) -
          (mckin**19*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(10161070498378800*
            mbkin**19))*np.log(mus**2/mbkin**2))/12 +
       (529*((-64*(1 - mckin/mbkin)**5)/5 + (96*(1 - mckin/mbkin)**6)/5 -
          (32*(-9937 + 3360*np.log(2) + 3360*np.log(1 - mckin/mbkin)))/33075 +
          (256*mckin*(-9937 + 3360*np.log(2) + 3360*np.log(1 - mckin/mbkin)))/
           (33075*mbkin) - (128*mckin**2*(-9937 + 3360*np.log(2) +
             3360*np.log(1 - mckin/mbkin)))/(4725*mbkin**2) +
          (256*mckin**3*(-9937 + 3360*np.log(2) + 3360*np.log(1 - mckin/mbkin)))/
           (4725*mbkin**3) - (64*mckin**4*(-9937 + 3360*np.log(2) +
             3360*np.log(1 - mckin/mbkin)))/(945*mbkin**4) +
          (256*mckin**5*(-9937 + 3360*np.log(2) + 3360*np.log(1 - mckin/mbkin)))/
           (4725*mbkin**5) - (128*mckin**6*(-9937 + 3360*np.log(2) +
             3360*np.log(1 - mckin/mbkin)))/(4725*mbkin**6) +
          (256*mckin**7*(-9937 + 3360*np.log(2) + 3360*np.log(1 - mckin/mbkin)))/
           (33075*mbkin**7) - (32*mckin**8*(-9937 + 3360*np.log(2) +
             3360*np.log(1 - mckin/mbkin)))/(33075*mbkin**8) +
          (32*(-19769 + 6720*np.log(2) + 6720*np.log(1 - mckin/mbkin)))/33075 -
          (32*mckin*(-19769 + 6720*np.log(2) + 6720*np.log(1 - mckin/mbkin)))/
           (4725*mbkin) + (32*mckin**2*(-19769 + 6720*np.log(2) +
             6720*np.log(1 - mckin/mbkin)))/(1575*mbkin**2) -
          (32*mckin**3*(-19769 + 6720*np.log(2) + 6720*np.log(1 - mckin/mbkin)))/
           (945*mbkin**3) + (32*mckin**4*(-19769 + 6720*np.log(2) +
             6720*np.log(1 - mckin/mbkin)))/(945*mbkin**4) -
          (32*mckin**5*(-19769 + 6720*np.log(2) + 6720*np.log(1 - mckin/mbkin)))/
           (1575*mbkin**5) + (32*mckin**6*(-19769 + 6720*np.log(2) +
             6720*np.log(1 - mckin/mbkin)))/(4725*mbkin**6) -
          (32*mckin**7*(-19769 + 6720*np.log(2) + 6720*np.log(1 - mckin/mbkin)))/
           (33075*mbkin**7) + (4*(-125299 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/297675 -
          (4*mckin*(-125299 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (33075*mbkin) + (16*mckin**2*(-125299 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(33075*mbkin**2) -
          (16*mckin**3*(-125299 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (14175*mbkin**3) + (8*mckin**4*(-125299 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(4725*mbkin**4) -
          (8*mckin**5*(-125299 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (4725*mbkin**5) + (16*mckin**6*(-125299 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(14175*mbkin**6) -
          (16*mckin**7*(-125299 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (33075*mbkin**7) + (4*mckin**8*(-125299 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(33075*mbkin**8) -
          (4*mckin**9*(-125299 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (297675*mbkin**9) + (2*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/297675 -
          (4*mckin*(-44659 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (59535*mbkin) + (2*mckin**2*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(6615*mbkin**2) -
          (16*mckin**3*(-44659 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (19845*mbkin**3) + (4*mckin**4*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(2835*mbkin**4) -
          (8*mckin**5*(-44659 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (4725*mbkin**5) + (4*mckin**6*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(2835*mbkin**6) -
          (16*mckin**7*(-44659 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (19845*mbkin**7) + (2*mckin**8*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(6615*mbkin**8) -
          (4*mckin**9*(-44659 + 40320*np.log(2) + 40320*np.log(1 - mckin/mbkin)))/
           (59535*mbkin**9) + (2*mckin**10*(-44659 + 40320*np.log(2) +
             40320*np.log(1 - mckin/mbkin)))/(297675*mbkin**10) +
          (4*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/1029105 -
          (4*mckin*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (93555*mbkin) + (4*mckin**2*(-49945 + 60192*np.log(2) +
             60192*np.log(1 - mckin/mbkin)))/(18711*mbkin**2) -
          (4*mckin**3*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (6237*mbkin**3) + (8*mckin**4*(-49945 + 60192*np.log(2) +
             60192*np.log(1 - mckin/mbkin)))/(6237*mbkin**4) -
          (8*mckin**5*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (4455*mbkin**5) + (8*mckin**6*(-49945 + 60192*np.log(2) +
             60192*np.log(1 - mckin/mbkin)))/(4455*mbkin**6) -
          (8*mckin**7*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (6237*mbkin**7) + (4*mckin**8*(-49945 + 60192*np.log(2) +
             60192*np.log(1 - mckin/mbkin)))/(6237*mbkin**8) -
          (4*mckin**9*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (18711*mbkin**9) + (4*mckin**10*(-49945 + 60192*np.log(2) +
             60192*np.log(1 - mckin/mbkin)))/(93555*mbkin**10) -
          (4*mckin**11*(-49945 + 60192*np.log(2) + 60192*np.log(1 - mckin/mbkin)))/
           (1029105*mbkin**11) + (32*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/15810795 -
          (32*mckin*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (1216215*mbkin) + (64*mckin**2*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(405405*mbkin**2) -
          (64*mckin**3*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (110565*mbkin**3) + (32*mckin**4*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(22113*mbkin**4) -
          (32*mckin**5*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (12285*mbkin**5) + (128*mckin**6*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(36855*mbkin**6) -
          (128*mckin**7*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (36855*mbkin**7) + (32*mckin**8*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(12285*mbkin**8) -
          (32*mckin**9*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (22113*mbkin**9) + (64*mckin**10*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(110565*mbkin**10) -
          (64*mckin**11*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (405405*mbkin**11) + (32*mckin**12*(-72701 + 99099*np.log(2) +
             99099*np.log(1 - mckin/mbkin)))/(1216215*mbkin**12) -
          (32*mckin**13*(-72701 + 99099*np.log(2) + 99099*np.log(1 - mckin/mbkin)))/
           (15810795*mbkin**13) + (-1182523 + 1552320*np.log(2) +
            1552320*np.log(1 - mckin/mbkin))/7203735 -
          (4*mckin*(-1182523 + 1552320*np.log(2) + 1552320*np.log(1 -
                mckin/mbkin)))/(2401245*mbkin) +
          (2*mckin**2*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(218295*mbkin**2) -
          (4*mckin**3*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(130977*mbkin**3) +
          (mckin**4*(-1182523 + 1552320*np.log(2) + 1552320*np.log(1 -
                mckin/mbkin)))/(14553*mbkin**4) -
          (8*mckin**5*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(72765*mbkin**5) +
          (4*mckin**6*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(31185*mbkin**6) -
          (8*mckin**7*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(72765*mbkin**7) +
          (mckin**8*(-1182523 + 1552320*np.log(2) + 1552320*np.log(1 -
                mckin/mbkin)))/(14553*mbkin**8) -
          (4*mckin**9*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(130977*mbkin**9) +
          (2*mckin**10*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(218295*mbkin**10) -
          (4*mckin**11*(-1182523 + 1552320*np.log(2) + 1552320*
              np.log(1 - mckin/mbkin)))/(2401245*mbkin**11) +
          (mckin**12*(-1182523 + 1552320*np.log(2) + 1552320*np.log(1 -
                mckin/mbkin)))/(7203735*mbkin**12) +
          (8*(-20507983 + 28522494*np.log(2) + 28522494*np.log(1 - mckin/mbkin)))/
           1217431215 - (16*mckin*(-20507983 + 28522494*np.log(2) +
             28522494*np.log(1 - mckin/mbkin)))/(173918745*mbkin) +
          (8*mckin**2*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(13378365*mbkin**2) -
          (32*mckin**3*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(13378365*mbkin**3) +
          (8*mckin**4*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(1216215*mbkin**4) -
          (16*mckin**5*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(1216215*mbkin**5) +
          (8*mckin**6*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**6) -
          (64*mckin**7*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(2837835*mbkin**7) +
          (8*mckin**8*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**8) -
          (16*mckin**9*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(1216215*mbkin**9) +
          (8*mckin**10*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(1216215*mbkin**10) -
          (32*mckin**11*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(13378365*mbkin**11) +
          (8*mckin**12*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(13378365*mbkin**12) -
          (16*mckin**13*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(173918745*mbkin**13) +
          (8*mckin**14*(-20507983 + 28522494*np.log(2) + 28522494*
              np.log(1 - mckin/mbkin)))/(1217431215*mbkin**14) +
          (-33813661 + 47893560*np.log(2) + 47893560*np.log(1 - mckin/mbkin))/
           289864575 - (16*mckin*(-33813661 + 47893560*np.log(2) +
             47893560*np.log(1 - mckin/mbkin)))/(289864575*mbkin) +
          (8*mckin**2*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(19324305*mbkin**2) -
          (16*mckin**3*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(8281845*mbkin**3) +
          (4*mckin**4*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(637065*mbkin**4) -
          (16*mckin**5*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(1061775*mbkin**5) +
          (8*mckin**6*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(289575*mbkin**6) -
          (16*mckin**7*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**7) +
          (2*mckin**8*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(45045*mbkin**8) -
          (16*mckin**9*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**9) +
          (8*mckin**10*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(289575*mbkin**10) -
          (16*mckin**11*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(1061775*mbkin**11) +
          (4*mckin**12*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(637065*mbkin**12) -
          (16*mckin**13*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(8281845*mbkin**13) +
          (8*mckin**14*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(19324305*mbkin**14) -
          (16*mckin**15*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(289864575*mbkin**15) +
          (mckin**16*(-33813661 + 47893560*np.log(2) + 47893560*
              np.log(1 - mckin/mbkin)))/(289864575*mbkin**16) +
          (-253404061 + 356516160*np.log(2) + 356516160*np.log(1 - mckin/mbkin))/
           2029052025 - (mckin*(-253404061 + 356516160*np.log(2) +
             356516160*np.log(1 - mckin/mbkin)))/(135270135*mbkin) +
          (mckin**2*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(19324305*mbkin**2) -
          (mckin**3*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(4459455*mbkin**3) +
          (mckin**4*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(1486485*mbkin**4) -
          (mckin**5*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(675675*mbkin**5) +
          (mckin**6*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**6) -
          (mckin**7*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(315315*mbkin**7) +
          (mckin**8*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(315315*mbkin**8) -
          (mckin**9*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(405405*mbkin**9) +
          (mckin**10*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(675675*mbkin**10) -
          (mckin**11*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(1486485*mbkin**11) +
          (mckin**12*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(4459455*mbkin**12) -
          (mckin**13*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(19324305*mbkin**13) +
          (mckin**14*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(135270135*mbkin**14) -
          (mckin**15*(-253404061 + 356516160*np.log(2) + 356516160*
              np.log(1 - mckin/mbkin)))/(2029052025*mbkin**15) +
          (-36724834687 + 52219630320*np.log(2) + 52219630320*
             np.log(1 - mckin/mbkin))/335083448700 -
          (mckin*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(19710791100*mbkin) +
          (2*mckin**2*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(4927697775*mbkin**2) -
          (2*mckin**3*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(985539555*mbkin**3) +
          (mckin**4*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(140791365*mbkin**4) -
          (mckin**5*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(54150525*mbkin**5) +
          (2*mckin**6*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(54150525*mbkin**6) -
          (2*mckin**7*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(34459425*mbkin**7) +
          (mckin**8*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(13783770*mbkin**8) -
          (mckin**9*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(13783770*mbkin**9) +
          (2*mckin**10*(-36724834687 + 52219630320*np.log(2) +
             52219630320*np.log(1 - mckin/mbkin)))/(34459425*mbkin**10) -
          (2*mckin**11*(-36724834687 + 52219630320*np.log(2) +
             52219630320*np.log(1 - mckin/mbkin)))/(54150525*mbkin**11) +
          (mckin**12*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(54150525*mbkin**12) -
          (mckin**13*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(140791365*mbkin**13) +
          (2*mckin**14*(-36724834687 + 52219630320*np.log(2) +
             52219630320*np.log(1 - mckin/mbkin)))/(985539555*mbkin**14) -
          (2*mckin**15*(-36724834687 + 52219630320*np.log(2) +
             52219630320*np.log(1 - mckin/mbkin)))/(4927697775*mbkin**15) +
          (mckin**16*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(19710791100*mbkin**16) -
          (mckin**17*(-36724834687 + 52219630320*np.log(2) + 52219630320*
              np.log(1 - mckin/mbkin)))/(335083448700*mbkin**17) +
          (-485272020137 + 691512341520*np.log(2) + 691512341520*
             np.log(1 - mckin/mbkin))/4691168281800 -
          (mckin*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(260620460100*mbkin) +
          (mckin**2*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(30661230600*mbkin**2) -
          (2*mckin**3*(-485272020137 + 691512341520*np.log(2) +
             691512341520*np.log(1 - mckin/mbkin)))/(11497961475*mbkin**3) +
          (mckin**4*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(1533061530*mbkin**4) -
          (mckin**5*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(547521975*mbkin**5) +
          (mckin**6*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(252702450*mbkin**6) -
          (2*mckin**7*(-485272020137 + 691512341520*np.log(2) +
             691512341520*np.log(1 - mckin/mbkin)))/(294819525*mbkin**7) +
          (mckin**8*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(107207100*mbkin**8) -
          (mckin**9*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(96486390*mbkin**9) +
          (mckin**10*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(107207100*mbkin**10) -
          (2*mckin**11*(-485272020137 + 691512341520*np.log(2) +
             691512341520*np.log(1 - mckin/mbkin)))/(294819525*mbkin**11) +
          (mckin**12*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(252702450*mbkin**12) -
          (mckin**13*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(547521975*mbkin**13) +
          (mckin**14*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(1533061530*mbkin**14) -
          (2*mckin**15*(-485272020137 + 691512341520*np.log(2) +
             691512341520*np.log(1 - mckin/mbkin)))/(11497961475*mbkin**15) +
          (mckin**16*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(30661230600*mbkin**16) -
          (mckin**17*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(260620460100*mbkin**17) +
          (mckin**18*(-485272020137 + 691512341520*np.log(2) + 691512341520*
              np.log(1 - mckin/mbkin)))/(4691168281800*mbkin**18) +
          (-4916068298621 + 7016274641376*np.log(2) + 7016274641376*
             np.log(1 - mckin/mbkin))/52784781809760 -
          (mckin*(-4916068298621 + 7016274641376*np.log(2) + 7016274641376*
              np.log(1 - mckin/mbkin)))/(2639239090488*mbkin) +
          (mckin**2*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(277814641104*mbkin**2) -
          (mckin**3*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(46302440184*mbkin**3) +
          (mckin**4*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(10894691808*mbkin**4) -
          (mckin**5*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(3404591190*mbkin**5) +
          (mckin**6*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(1361836476*mbkin**6) -
          (mckin**7*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(680918238*mbkin**7) +
          (mckin**8*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(419026608*mbkin**8) -
          (mckin**9*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(314269956*mbkin**9) +
          (mckin**10*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(285699960*mbkin**10) -
          (mckin**11*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(314269956*mbkin**11) +
          (mckin**12*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(419026608*mbkin**12) -
          (mckin**13*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(680918238*mbkin**13) +
          (mckin**14*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(1361836476*mbkin**14) -
          (mckin**15*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(3404591190*mbkin**15) +
          (mckin**16*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(10894691808*mbkin**16) -
          (mckin**17*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(46302440184*mbkin**17) +
          (mckin**18*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(277814641104*mbkin**18) -
          (mckin**19*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(2639239090488*mbkin**19) +
          (mckin**20*(-4916068298621 + 7016274641376*np.log(2) +
             7016274641376*np.log(1 - mckin/mbkin)))/(52784781809760*mbkin**20) +
          (-995781239706241 + 1420555140164160*np.log(2) + 1420555140164160*
             np.log(1 - mckin/mbkin))/10161070498378800 -
          (mckin*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(534793184125200*
            mbkin) + (mckin**2*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(59421464902800*
            mbkin**2) - (mckin**3*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(10486140865200*
            mbkin**3) + (mckin**4*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(2621535216300*
            mbkin**4) - (mckin**5*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(873845072100*mbkin**5) +
          (mckin**6*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(374505030900*mbkin**6) -
          (mckin**7*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(201656555100*mbkin**7) +
          (mckin**8*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(134437703400*mbkin**8) -
          (mckin**9*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(109994484600*mbkin**9) +
          (mckin**10*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(109994484600*
            mbkin**10) - (mckin**11*(-995781239706241 + 1420555140164160*
              np.log(2) + 1420555140164160*np.log(1 - mckin/mbkin)))/
           (134437703400*mbkin**11) + (mckin**12*(-995781239706241 +
             1420555140164160*np.log(2) + 1420555140164160*np.log(1 -
                mckin/mbkin)))/(201656555100*mbkin**12) -
          (mckin**13*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(374505030900*
            mbkin**13) + (mckin**14*(-995781239706241 + 1420555140164160*
              np.log(2) + 1420555140164160*np.log(1 - mckin/mbkin)))/
           (873845072100*mbkin**14) - (mckin**15*(-995781239706241 +
             1420555140164160*np.log(2) + 1420555140164160*np.log(1 -
                mckin/mbkin)))/(2621535216300*mbkin**15) +
          (mckin**16*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(10486140865200*
            mbkin**16) - (mckin**17*(-995781239706241 + 1420555140164160*
              np.log(2) + 1420555140164160*np.log(1 - mckin/mbkin)))/
           (59421464902800*mbkin**17) + (mckin**18*(-995781239706241 +
             1420555140164160*np.log(2) + 1420555140164160*np.log(1 -
                mckin/mbkin)))/(534793184125200*mbkin**18) -
          (mckin**19*(-995781239706241 + 1420555140164160*np.log(2) +
             1420555140164160*np.log(1 - mckin/mbkin)))/(10161070498378800*
            mbkin**19))*np.log(mus**2/mbkin**2)**2)/144 +
       (np.log(mus**2/mbkin**2)*((-9.854113931163779e19*(1 - mckin/mbkin)**5 -
            2.5903365411196256e20*(1 - mckin/mbkin)**6 -
            1.5679296844106574e21*(1 - mckin/mbkin)**7 +
            1.602586733095746e21*(1 - mckin/mbkin)**8 - 3.731382058497059e20*
             (1 - mckin/mbkin)**9 - 1.442450583699921e20*(1 - mckin/mbkin)**
              10 - 1.0117940133228708e21*(1 - mckin/mbkin)**11 -
            1.4567466646450717e21*(1 - mckin/mbkin)**12 -
            2.2050041791842524e21*(1 - mckin/mbkin)**13 -
            3.10373436948258e21*(1 - mckin/mbkin)**14 - 4.17958779756945e21*
             (1 - mckin/mbkin)**15 - 58963096098466560000*(1 - mckin/mbkin)**5*
             np.pi**2 + 88444644147699840000*(1 - mckin/mbkin)**6*np.pi**2 -
            65172716793401817600*(1 - mckin/mbkin)**7*np.pi**2 -
            9108973844357587200*(1 - mckin/mbkin)**8*np.pi**2 +
            27694301930794699200*(1 - mckin/mbkin)**9*np.pi**2 +
            9354724595990373600*(1 - mckin/mbkin)**10*np.pi**2 +
            99846396491298256800*(1 - mckin/mbkin)**11*np.pi**2 +
            145282820224320979200*(1 - mckin/mbkin)**12*np.pi**2 +
            221188319752082032350*(1 - mckin/mbkin)**13*np.pi**2 +
            312313023387573082875*(1 - mckin/mbkin)**14*np.pi**2 +
            421384671784085884875*(1 - mckin/mbkin)**15*np.pi**2 +
            23585238439386624000*(1 - mckin/mbkin)**5*np.pi**2*np.log(2) -
            35377857659079936000*(1 - mckin/mbkin)**6*np.pi**2*np.log(2) +
            17408152181452032000*(1 - mckin/mbkin)**7*np.pi**2*np.log(2) -
            2807766480879360000*(1 - mckin/mbkin)**8*np.pi**2*np.log(2) +
            1380485186432352000*(1 - mckin/mbkin)**9*np.pi**2*np.log(2) +
            690242593216176000*(1 - mckin/mbkin)**10*np.pi**2*np.log(2) +
            508375900704672000*(1 - mckin/mbkin)**11*np.pi**2*np.log(2) +
            417442554448920000*(1 - mckin/mbkin)**12*np.pi**2*np.log(2) +
            359888250273552000*(1 - mckin/mbkin)**13*np.pi**2*np.log(2) +
            319023467138376000*(1 - mckin/mbkin)**14*np.pi**2*np.log(2) +
            287976049320960000*(1 - mckin/mbkin)**15*np.pi**2*np.log(2) +
            2156493012354474393600*(1 - mckin/mbkin)**7*(np.log(2) +
              np.log(1 - mckin/mbkin)) - 1114185917132493004800*
             (1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)) +
            72609864602488012800*(1 - mckin/mbkin)**9*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 63592262841345638400*(1 - mckin/mbkin)**
              10*(np.log(2) + np.log(1 - mckin/mbkin)) + 49568665585965926400*
             (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)) +
            44918542055279692800*(1 - mckin/mbkin)**12*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 42007093034271805440*(1 - mckin/mbkin)**
              13*(np.log(2) + np.log(1 - mckin/mbkin)) + 39569178342479385600*
             (1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)) +
            37379682179716836960*(1 - mckin/mbkin)**15*(np.log(2) +
              np.log(1 - mckin/mbkin)) - 53909116432883712000*(1 - mckin/mbkin)**
              7*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) + 26954558216441856000*
             (1 - mckin/mbkin)**8*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
            4492426369406976000*(1 - mckin/mbkin)**9*np.pi**2*
             (np.log(2) + np.log(1 - mckin/mbkin)) - 2246213184703488000*
             (1 - mckin/mbkin)**10*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
            1939911386789376000*(1 - mckin/mbkin)**11*np.pi**2*
             (np.log(2) + np.log(1 - mckin/mbkin)) - 1786760487832320000*
             (1 - mckin/mbkin)**12*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
            1663061684828544000*(1 - mckin/mbkin)**13*np.pi**2*
             (np.log(2) + np.log(1 - mckin/mbkin)) - 1554088929801408000*
             (1 - mckin/mbkin)**14*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
            1456897013155584000*(1 - mckin/mbkin)**15*np.pi**2*
             (np.log(2) + np.log(1 - mckin/mbkin)) - 242591023947976704000*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
            121295511973988352000*(1 - mckin/mbkin)**8*
             (np.log(2) + np.log(1 - mckin/mbkin))**2 + 13061313703646208000*
             (1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
            6530656851823104000*(1 - mckin/mbkin)**10*
             (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7243470295117056000*
             (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
            7599877016764032000*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7781163757509542400*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
            7874890507804320000*(1 - mckin/mbkin)**14*
             (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7913392760156083200*
             (1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin))**2)/
           16583370777693720000 + (23*(-260123404758497280*(1 - mckin/mbkin)**
               5 + 390185107137745920*(1 - mckin/mbkin)**6 -
             388690823028777984*(1 - mckin/mbkin)**7 + 195377647247557632*
              (1 - mckin/mbkin)**8 - 34216429928650112*(1 - mckin/mbkin)**9 -
             6097700477192896*(1 - mckin/mbkin)**10 - 3945134197513600*
              (1 - mckin/mbkin)**11 - 3335963793491680*(1 - mckin/mbkin)**12 -
             2990240473256960*(1 - mckin/mbkin)**13 - 2738659018760960*
              (1 - mckin/mbkin)**14 - 2537989658886624*(1 - mckin/mbkin)**15 -
             2370644934651168*(1 - mckin/mbkin)**16 - 2227287774097176*
              (1 - mckin/mbkin)**17 - 2102198391233484*(1 - mckin/mbkin)**18 -
             1991562479412482*(1 - mckin/mbkin)**19 - 1892686294969085*
              (1 - mckin/mbkin)**20 + 132126173845585920*(1 - mckin/mbkin)**7*
              (np.log(2) + np.log(1 - mckin/mbkin)) - 66063086922792960*
              (1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)) +
             11010514487132160*(1 - mckin/mbkin)**9*(np.log(2) + np.log(
                1 - mckin/mbkin)) + 5505257243566080*(1 - mckin/mbkin)**10*
              (np.log(2) + np.log(1 - mckin/mbkin)) + 4754540346716160*
              (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)) +
             4379181898291200*(1 - mckin/mbkin)**12*(np.log(2) + np.log(
                1 - mckin/mbkin)) + 4076007766871040*(1 - mckin/mbkin)**13*
              (np.log(2) + np.log(1 - mckin/mbkin)) + 3808925793953280*
              (1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)) +
             3570717547837440*(1 - mckin/mbkin)**15*(np.log(2) + np.log(
                1 - mckin/mbkin)) + 3357773812673280*(1 - mckin/mbkin)**16*
              (np.log(2) + np.log(1 - mckin/mbkin)) + 3167016139647360*
              (1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin)) +
             2995631463464640*(1 - mckin/mbkin)**18*(np.log(2) + np.log(
                1 - mckin/mbkin)) + 2841110280328320*(1 - mckin/mbkin)**19*
              (np.log(2) + np.log(1 - mckin/mbkin)) + 2701265736929760*
              (1 - mckin/mbkin)**20*(np.log(2) + np.log(1 - mckin/mbkin)))*
            np.log(mus**2/mbkin**2))/243865691961091200))/3 +
       ((-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
           (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
          195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
           (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
          3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
           (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
          2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
           (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
          2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
           (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
          1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
           (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
          66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
            np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
           (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
          4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
            np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
           (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
          3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
            np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
           (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
          3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
            np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
           (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
           (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
          2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
            np.log(1 - mckin/mbkin)))*(21 + 57*np.log(mus**2/mbkin**2) +
          2*np.log(mus**2/mbkin**2)**2))/1463194151766547200) +
     (api4**2*(1792/81 - (256*mbkin**2)/(27*mckin**2) + (512*mckin**2)/
          (81*mbkin**2) - (512*mckin**4)/(9*mbkin**4) + (4352*mckin**6)/
          (81*mbkin**6) - (1280*mckin**8)/(81*mbkin**8) +
         (64*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2)))/81) +
       api4**3*(762496/2187 - (87296*mbkin**2)/(729*mckin**2) -
         (282112*mbkin)/(729*mckin) + (23552*mckin)/(81*mbkin) +
         (948224*mckin**2)/(2187*mbkin**2) - (131072*mckin**3)/(729*mbkin**3) -
         (95744*mckin**4)/(81*mbkin**4) + (533504*mckin**5)/(729*mbkin**5) +
         (1124096*mckin**6)/(2187*mbkin**6) - (332288*mckin**7)/(729*mbkin**7) +
         (12160*mckin**8)/(2187*mbkin**8) - (896*np.pi**2)/81 +
         (128*mbkin**2*np.pi**2)/(27*mckin**2) - (256*mckin**2*np.pi**2)/(81*mbkin**2) +
         (256*mckin**4*np.pi**2)/(9*mbkin**4) - (2176*mckin**6*np.pi**2)/(81*mbkin**6) +
         (640*mckin**8*np.pi**2)/(81*mbkin**8) +
         (26560*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2)))/2187 +
         (32*mbkin**2*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2)))/(243*mckin**2) -
         (32*np.pi**2*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2)))/81 - (512*np.log(mckin**2/mbkin**2))/9 -
         (1024*mbkin*np.log(mckin**2/mbkin**2))/(9*mckin) -
         (2048*mckin*np.log(mckin**2/mbkin**2))/(9*mbkin) +
         (1024*mckin**2*np.log(mckin**2/mbkin**2))/(27*mbkin**2) +
         (1024*mckin**3*np.log(mckin**2/mbkin**2))/(27*mbkin**3) +
         (7168*mckin**4*np.log(mckin**2/mbkin**2))/(27*mbkin**4) +
         (640*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/
          27 - (896*np.log(2/mus))/9 + (128*mbkin**2*np.log(2/mus))/(3*mckin**2) -
         (256*mckin**2*np.log(2/mus))/(9*mbkin**2) + (256*mckin**4*np.log(2/mus))/
          mbkin**4 - (2176*mckin**6*np.log(2/mus))/(9*mbkin**6) +
         (640*mckin**8*np.log(2/mus))/(9*mbkin**8) -
         (32*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2))*np.log(2/mus))/9 -
         (448*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/243 +
         (64*mbkin**2*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(81*mckin**2) -
         (128*mckin**2*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(243*mbkin**2) +
         (128*mckin**4*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(27*mbkin**4) -
         (1088*mckin**6*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(243*mbkin**6) +
         (320*mckin**8*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(243*mbkin**8) +
         (8*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2))*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/81 -
         (20*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2))*((2*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/
             27 + (4*np.log(mus**2/mckin**2))/27))/9 - (1792*np.log(mus**2/mckin**2))/
          243 + (256*mbkin**2*np.log(mus**2/mckin**2))/(81*mckin**2) -
         (512*mckin**2*np.log(mus**2/mckin**2))/(243*mbkin**2) +
         (512*mckin**4*np.log(mus**2/mckin**2))/(27*mbkin**4) -
         (4352*mckin**6*np.log(mus**2/mckin**2))/(243*mbkin**6) +
         (1280*mckin**8*np.log(mus**2/mckin**2))/(243*mbkin**8) +
         (16*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2))*np.log(mus**2/mckin**2))/243))/mbkin**5 +
     (api4**2*(-3172/81 - (32*mbkin**2)/(9*mckin**2) - (2048*mbkin)/(81*mckin) +
         (8192*mckin)/(243*mbkin) + (15136*mckin**2)/(243*mbkin**2) +
         (4096*mckin**3)/(81*mbkin**3) - (3808*mckin**4)/(81*mbkin**4) -
         (8192*mckin**5)/(81*mbkin**5) + (5312*mckin**6)/(81*mbkin**6) +
         (10240*mckin**7)/(243*mbkin**7) - (9268*mckin**8)/(243*mbkin**8) +
         (512*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2)))/243 - 32*np.log(mckin**2/mbkin**2) -
         (128*mckin**2*np.log(mckin**2/mbkin**2))/(3*mbkin**2) +
         (176*mckin**4*np.log(mckin**2/mbkin**2))/(3*mbkin**4) +
         (40*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/
          9) + api4**3*(-4744846074154/2703645945 - (169257152*mbkin**2)/
          (2380833*mckin**2) - (16787220263141639*mbkin)/(27712010450124*
           mckin) + (123118992743852681*mckin)/(127013381229735*mbkin) -
         (5524185012272723*mckin**2)/(1415629016802*mbkin**2) +
         (8902370578115365121*mckin**3)/(240656932856340*mbkin**3) -
         (257390915317679791*mckin**4)/(2359381694670*mbkin**4) +
         (801159859495809656*mckin**5)/(3539072542005*mbkin**5) -
         (60176144758969102*mckin**6)/(168527263905*mbkin**6) +
         (4207031612527687*mckin**7)/(10213773570*mbkin**7) -
         (727645098577498*mckin**8)/(2326806405*mbkin**8) +
         (1711171532658442*mckin**9)/(24748759035*mbkin**9) +
         (1699632416233267*mckin**10)/(8249586345*mbkin**10) -
         (18717598749855497*mckin**11)/(49497518070*mbkin**11) +
         (1382707304952858293*mckin**12)/(3539072542005*mbkin**12) -
         (341943151195082464*mckin**13)/(1179690847335*mbkin**13) +
         (82642474657474222*mckin**14)/(505581791715*mbkin**14) -
         (37183537305838351*mckin**15)/(524307043260*mbkin**15) +
         (156712962234200096*mckin**16)/(6684914801565*mbkin**16) -
         (6766223855258947*mckin**17)/(1179690847335*mbkin**17) +
         (204090269190441313*mckin**18)/(207840078375930*mbkin**18) -
         (496567508421713*mckin**19)/(4718763389340*mbkin**19) +
         (31596616134481*mckin**20)/(5938287953598*mbkin**20) +
         (1024*(1 - mckin/mbkin)**3)/9 - (512*mbkin**2*(1 - mckin/mbkin)**3)/
          (9*mckin**2) - (512*mckin**2*(1 - mckin/mbkin)**3)/(9*mbkin**2) -
         256*(1 - mckin/mbkin)**4 + (128*mbkin**2*(1 - mckin/mbkin)**4)/
          mckin**2 - (256*mbkin*(1 - mckin/mbkin)**4)/(9*mckin) +
         (256*mckin*(1 - mckin/mbkin)**4)/(9*mbkin) +
         (128*mckin**2*(1 - mckin/mbkin)**4)/mbkin**2 +
         (242944*(1 - mckin/mbkin)**5)/675 -
         (121472*mbkin**2*(1 - mckin/mbkin)**5)/(675*mckin**2) +
         (256*mbkin*(1 - mckin/mbkin)**5)/(5*mckin) -
         (256*mckin*(1 - mckin/mbkin)**5)/(5*mbkin) -
         (121472*mckin**2*(1 - mckin/mbkin)**5)/(675*mbkin**2) -
         (10236928*(1 - mckin/mbkin)**6)/42525 +
         (5118464*mbkin**2*(1 - mckin/mbkin)**6)/(42525*mckin**2) -
         (343936*mbkin*(1 - mckin/mbkin)**6)/(6075*mckin) +
         (343936*mckin*(1 - mckin/mbkin)**6)/(6075*mbkin) +
         (5118464*mckin**2*(1 - mckin/mbkin)**6)/(42525*mbkin**2) +
         (5369984*(1 - mckin/mbkin)**7)/99225 -
         (2684992*mbkin**2*(1 - mckin/mbkin)**7)/(99225*mckin**2) +
         (9745408*mbkin*(1 - mckin/mbkin)**7)/(297675*mckin) -
         (9745408*mckin*(1 - mckin/mbkin)**7)/(297675*mbkin) -
         (2684992*mckin**2*(1 - mckin/mbkin)**7)/(99225*mbkin**2) +
         (721712*(1 - mckin/mbkin)**8)/59535 -
         (360856*mbkin**2*(1 - mckin/mbkin)**8)/(59535*mckin**2) -
         (644368*mbkin*(1 - mckin/mbkin)**8)/(99225*mckin) +
         (644368*mckin*(1 - mckin/mbkin)**8)/(99225*mbkin) -
         (360856*mckin**2*(1 - mckin/mbkin)**8)/(59535*mbkin**2) +
         (734432*(1 - mckin/mbkin)**9)/76545 -
         (367216*mbkin**2*(1 - mckin/mbkin)**9)/(76545*mckin**2) -
         (650032*mbkin*(1 - mckin/mbkin)**9)/(535815*mckin) +
         (650032*mckin*(1 - mckin/mbkin)**9)/(535815*mbkin) -
         (367216*mckin**2*(1 - mckin/mbkin)**9)/(76545*mbkin**2) +
         (19108528*(1 - mckin/mbkin)**10)/1964655 -
         (9554264*mbkin**2*(1 - mckin/mbkin)**10)/(1964655*mckin**2) -
         (64688*mbkin*(1 - mckin/mbkin)**10)/(76545*mckin) +
         (64688*mckin*(1 - mckin/mbkin)**10)/(76545*mbkin) -
         (9554264*mckin**2*(1 - mckin/mbkin)**10)/(1964655*mbkin**2) +
         (577664*(1 - mckin/mbkin)**11)/56133 -
         (288832*mbkin**2*(1 - mckin/mbkin)**11)/(56133*mckin**2) -
         (16850608*mbkin*(1 - mckin/mbkin)**11)/(21611205*mckin) +
         (16850608*mckin*(1 - mckin/mbkin)**11)/(21611205*mbkin) -
         (288832*mckin**2*(1 - mckin/mbkin)**11)/(56133*mbkin**2) +
         (37786880*(1 - mckin/mbkin)**12)/3440151 -
         (18893440*mbkin**2*(1 - mckin/mbkin)**12)/(3440151*mckin**2) -
         (640768*mbkin*(1 - mckin/mbkin)**12)/(841995*mckin) +
         (640768*mckin*(1 - mckin/mbkin)**12)/(841995*mbkin) -
         (18893440*mckin**2*(1 - mckin/mbkin)**12)/(3440151*mbkin**2) +
         (680271352*(1 - mckin/mbkin)**13)/57972915 -
         (340135676*mbkin**2*(1 - mckin/mbkin)**13)/(57972915*mckin**2) -
         (168874624*mbkin*(1 - mckin/mbkin)**13)/(223609815*mckin) +
         (168874624*mckin*(1 - mckin/mbkin)**13)/(223609815*mbkin) -
         (340135676*mckin**2*(1 - mckin/mbkin)**13)/(57972915*mbkin**2) +
         (435369184*(1 - mckin/mbkin)**14)/34783749 -
         (217684592*mbkin**2*(1 - mckin/mbkin)**14)/(34783749*mckin**2) -
         (306181756*mbkin*(1 - mckin/mbkin)**14)/(405810405*mckin) +
         (306181756*mckin*(1 - mckin/mbkin)**14)/(405810405*mbkin) -
         (217684592*mckin**2*(1 - mckin/mbkin)**14)/(34783749*mbkin**2) +
         (34745241056*(1 - mckin/mbkin)**15)/2608781175 -
         (17372620528*mbkin**2*(1 - mckin/mbkin)**15)/(2608781175*mckin**2) -
         (1972500064*mbkin*(1 - mckin/mbkin)**15)/(2608781175*mckin) +
         (1972500064*mckin*(1 - mckin/mbkin)**15)/(2608781175*mbkin) -
         (17372620528*mckin**2*(1 - mckin/mbkin)**15)/(2608781175*mbkin**2) +
         (1462595593171*(1 - mckin/mbkin)**16)/103481653275 -
         (1462595593171*mbkin**2*(1 - mckin/mbkin)**16)/(206963306550*
           mckin**2) - (1979593631*mbkin*(1 - mckin/mbkin)**16)/
          (2608781175*mckin) + (1979593631*mckin*(1 - mckin/mbkin)**16)/
          (2608781175*mbkin) - (1462595593171*mckin**2*(1 - mckin/mbkin)**16)/
          (206963306550*mbkin**2) + (17542717714153*(1 - mckin/mbkin)**17)/
          1172792070450 - (17542717714153*mbkin**2*(1 - mckin/mbkin)**17)/
          (2345584140900*mckin**2) - (1340564003491*mbkin*(1 - mckin/mbkin)**
            17)/(1759188105675*mckin) + (1340564003491*mckin*
           (1 - mckin/mbkin)**17)/(1759188105675*mbkin) -
         (17542717714153*mckin**2*(1 - mckin/mbkin)**17)/(2345584140900*
           mbkin**2) + (24672660896281*(1 - mckin/mbkin)**18)/1562707356210 -
         (24672660896281*mbkin**2*(1 - mckin/mbkin)**18)/(3125414712420*
           mckin**2) - (48474483141979*mbkin*(1 - mckin/mbkin)**18)/
          (63330771804300*mckin) + (48474483141979*mckin*(1 - mckin/mbkin)**
            18)/(63330771804300*mbkin) - (24672660896281*mckin**2*
           (1 - mckin/mbkin)**18)/(3125414712420*mbkin**2) -
         (22826272832761*mbkin*(1 - mckin/mbkin)**19)/(29691439767990*mckin) +
         (22826272832761*mckin*(1 - mckin/mbkin)**19)/(29691439767990*mbkin) +
         (3962*np.pi**2)/81 + (32*mbkin**2*np.pi**2)/(9*mckin**2) +
         (1024*mbkin*np.pi**2)/(81*mckin) - (4096*mckin*np.pi**2)/(243*mbkin) -
         (20960*mckin**2*np.pi**2)/(243*mbkin**2) - (2048*mckin**3*np.pi**2)/
          (81*mbkin**3) + (3056*mckin**4*np.pi**2)/(81*mbkin**4) +
         (4096*mckin**5*np.pi**2)/(81*mbkin**5) - (1792*mckin**6*np.pi**2)/
          (81*mbkin**6) - (5120*mckin**7*np.pi**2)/(243*mbkin**7) +
         (4418*mckin**8*np.pi**2)/(243*mbkin**8) +
         (7808*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2)))/243 +
         (256*mbkin*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2)))/(729*mckin) -
         (256*np.pi**2*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2)))/243 - (30848*np.log(mckin**2/mbkin**2))/27 -
         (8192*mckin*np.log(mckin**2/mbkin**2))/(27*mbkin) -
         (30464*mckin**2*np.log(mckin**2/mbkin**2))/(27*mbkin**2) +
         (8192*mckin**3*np.log(mckin**2/mbkin**2))/(81*mbkin**3) +
         (177328*mckin**4*np.log(mckin**2/mbkin**2))/(81*mbkin**4) +
         32*np.pi**2*np.log(mckin**2/mbkin**2) + (32*mckin**2*np.pi**2*
           np.log(mckin**2/mbkin**2))/mbkin**2 - (56*mckin**4*np.pi**2*
           np.log(mckin**2/mbkin**2))/mbkin**4 +
         (14480*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/
          81 + (8*mbkin**2*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/
          (9*mckin**2) - (40*np.pi**2*(1 - (8*mckin**2)/mbkin**2 +
            (8*mckin**6)/mbkin**6 - mckin**8/mbkin**8 -
            (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/9 -
         (16384*(1 - mckin/mbkin)**5*(np.log(2) + np.log(1 - mckin/mbkin)))/135 +
         (8192*mbkin**2*(1 - mckin/mbkin)**5*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (135*mckin**2) + (8192*mckin**2*(1 - mckin/mbkin)**5*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(135*mbkin**2) +
         (32768*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin)))/405 -
         (16384*mbkin**2*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (405*mckin**2) + (8192*mbkin*(1 - mckin/mbkin)**6*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(405*mckin) -
         (8192*mckin*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (405*mbkin) - (16384*mckin**2*(1 - mckin/mbkin)**6*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(405*mbkin**2) -
         (16384*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)))/945 +
         (8192*mbkin**2*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (945*mckin**2) - (32768*mbkin*(1 - mckin/mbkin)**7*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(2835*mckin) +
         (32768*mckin*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (2835*mbkin) + (8192*mckin**2*(1 - mckin/mbkin)**7*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(945*mbkin**2) -
         (2048*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)))/189 +
         (1024*mbkin**2*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (189*mckin**2) + (2048*mbkin*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(945*mckin) -
         (2048*mckin*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (945*mbkin) + (1024*mckin**2*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(189*mbkin**2) -
         (19456*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)))/1701 +
         (9728*mbkin**2*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1701*mckin**2) + (2048*mbkin*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1701*mckin) -
         (2048*mckin*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1701*mbkin) + (9728*mckin**2*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1701*mbkin**2) -
         (1024*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)))/81 +
         (512*mbkin**2*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (81*mckin**2) + (9728*mbkin*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(8505*mckin) -
         (9728*mckin*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (8505*mbkin) + (512*mckin**2*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(81*mbkin**2) -
         (5632*(1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)))/405 +
         (2816*mbkin**2*(1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (405*mckin**2) + (1024*mbkin*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(891*mckin) -
         (1024*mckin*(1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (891*mbkin) + (2816*mckin**2*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(405*mbkin**2) -
         (202624*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)))/
          13365 + (101312*mbkin**2*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(13365*mckin**2) +
         (1408*mbkin*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1215*mckin) - (1408*mckin*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1215*mbkin) +
         (101312*mckin**2*(1 - mckin/mbkin)**12*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(13365*mbkin**2) -
         (189952*(1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)))/
          11583 + (94976*mbkin**2*(1 - mckin/mbkin)**13*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(11583*mckin**2) +
         (202624*mbkin*(1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (173745*mckin) - (202624*mckin*(1 - mckin/mbkin)**13*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(173745*mbkin) +
         (94976*mckin**2*(1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (11583*mbkin**2) - (1428992*(1 - mckin/mbkin)**14*
           (np.log(2) + np.log(1 - mckin/mbkin)))/81081 +
         (714496*mbkin**2*(1 - mckin/mbkin)**14*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(81081*mckin**2) +
         (13568*mbkin*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (11583*mckin) - (13568*mckin*(1 - mckin/mbkin)**14*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(11583*mbkin) +
         (714496*mckin**2*(1 - mckin/mbkin)**14*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(81081*mbkin**2) -
         (22912768*(1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1216215 + (11456384*mbkin**2*(1 - mckin/mbkin)**15*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1216215*mckin**2) +
         (1428992*mbkin*(1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1216215*mckin) - (1428992*mckin*(1 - mckin/mbkin)**15*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1216215*mbkin) +
         (11456384*mckin**2*(1 - mckin/mbkin)**15*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(1216215*mbkin**2) -
         (2709104*(1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)))/
          135135 + (1354552*mbkin**2*(1 - mckin/mbkin)**16*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(135135*mckin**2) +
         (1432048*mbkin*(1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1216215*mckin) - (1432048*mckin*(1 - mckin/mbkin)**16*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1216215*mbkin) +
         (1354552*mckin**2*(1 - mckin/mbkin)**16*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(135135*mbkin**2) -
         (6973984*(1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin)))/
          328185 + (3486992*mbkin**2*(1 - mckin/mbkin)**17*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(328185*mckin**2) +
         (2709104*mbkin*(1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (2297295*mckin) - (2709104*mckin*(1 - mckin/mbkin)**17*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(2297295*mbkin) +
         (3486992*mckin**2*(1 - mckin/mbkin)**17*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(328185*mbkin**2) -
         (1205584*(1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin)))/
          53703 + (602792*mbkin**2*(1 - mckin/mbkin)**18*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(53703*mckin**2) +
         (3486992*mbkin*(1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (2953665*mckin) - (3486992*mckin*(1 - mckin/mbkin)**18*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(2953665*mbkin) +
         (602792*mckin**2*(1 - mckin/mbkin)**18*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(53703*mbkin**2) +
         (1205584*mbkin*(1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1020357*mckin) - (1205584*mckin*(1 - mckin/mbkin)**19*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1020357*mbkin) +
         (16384*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/189 -
         (8192*mbkin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(567*mckin) -
         (16384*mckin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(81*mbkin) +
         (16384*mckin**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (81*mbkin**2) - (16384*mckin**4*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(81*mbkin**4) +
         (16384*mckin**5*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (81*mbkin**5) - (16384*mckin**6*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(189*mbkin**6) +
         (8192*mckin**7*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (567*mbkin**7) - (4096*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/81 +
         (4096*mbkin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/(567*mckin) +
         (81920*mckin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/(567*mbkin) -
         (16384*mckin**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (81*mbkin**2) + (8192*mckin**3*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(81*mbkin**3) +
         (8192*mckin**4*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (81*mbkin**4) - (16384*mckin**5*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(81*mbkin**5) +
         (81920*mckin**6*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (567*mbkin**6) - (4096*mckin**7*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(81*mbkin**7) +
         (4096*mckin**8*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (567*mbkin**8) + (16384*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          1701 - (2048*mbkin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (1701*mckin) - (2048*mckin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (63*mbkin) + (32768*mckin**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(567*mbkin**2) -
         (4096*mckin**3*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (81*mbkin**3) + (4096*mckin**5*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(81*mbkin**5) -
         (32768*mckin**6*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (567*mbkin**6) + (2048*mckin**7*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(63*mbkin**7) -
         (16384*mckin**8*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**8) + (2048*mckin**9*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(1701*mbkin**9) +
         (1024*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/189 -
         (1024*mbkin*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (1701*mckin) - (5120*mckin*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(243*mbkin) +
         (25600*mckin**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (567*mbkin**2) - (10240*mckin**3*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(189*mbkin**3) +
         (2048*mckin**4*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (81*mbkin**4) + (2048*mckin**5*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(81*mbkin**5) -
         (10240*mckin**6*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (189*mbkin**6) + (25600*mckin**7*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(567*mbkin**7) -
         (5120*mckin**8*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (243*mbkin**8) + (1024*mckin**9*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(189*mbkin**9) -
         (1024*mckin**10*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**10) + (97280*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          18711 - (9728*mbkin*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (18711*mckin) - (38912*mckin*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(1701*mbkin) +
         (97280*mckin**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**2) - (48640*mckin**3*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(567*mbkin**3) +
         (38912*mckin**4*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (567*mbkin**4) - (38912*mckin**6*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(567*mbkin**6) +
         (48640*mckin**7*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (567*mbkin**7) - (97280*mckin**8*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(1701*mbkin**8) +
         (38912*mckin**9*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**9) - (97280*mckin**10*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(18711*mbkin**10) +
         (9728*mckin**11*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (18711*mbkin**11) + (1280*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          243 - (1280*mbkin*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (2673*mckin) - (2560*mckin*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(99*mbkin) +
         (17920*mckin**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (243*mbkin**2) - (32000*mckin**3*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(243*mbkin**3) +
         (1280*mckin**4*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (9*mbkin**4) - (5120*mckin**5*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**5) -
         (5120*mckin**6*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (81*mbkin**6) + (1280*mckin**7*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(9*mbkin**7) -
         (32000*mckin**8*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (243*mbkin**8) + (17920*mckin**9*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(243*mbkin**9) -
         (2560*mckin**10*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (99*mbkin**10) + (1280*mckin**11*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(243*mbkin**11) -
         (1280*mckin**12*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**12) + (5632*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          1053 - (1408*mbkin*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (3159*mckin) - (7040*mckin*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(243*mbkin) +
         (22528*mckin**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (243*mbkin**2) - (15488*mckin**3*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(81*mbkin**3) +
         (61952*mckin**4*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (243*mbkin**4) - (15488*mckin**5*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(81*mbkin**5) +
         (15488*mckin**7*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (81*mbkin**7) - (61952*mckin**8*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(243*mbkin**8) +
         (15488*mckin**9*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (81*mbkin**9) - (22528*mckin**10*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(243*mbkin**10) +
         (7040*mckin**11*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (243*mbkin**11) - (5632*mckin**12*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(1053*mbkin**12) +
         (1408*mckin**13*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (3159*mbkin**13) + (101312*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/18711 -
         (101312*mbkin*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (243243*mckin) - (101312*mckin*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(3159*mbkin) +
         (101312*mckin**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (891*mbkin**2) - (709184*mckin**3*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(2673*mbkin**3) +
         (101312*mckin**4*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (243*mbkin**4) - (101312*mckin**5*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(243*mbkin**5) +
         (101312*mckin**6*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (567*mbkin**6) + (101312*mckin**7*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(567*mbkin**7) -
         (101312*mckin**8*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (243*mbkin**8) + (101312*mckin**9*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(243*mbkin**9) -
         (709184*mckin**10*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**10) + (101312*mckin**11*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(891*mbkin**11) -
         (101312*mckin**12*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (3159*mbkin**12) + (101312*mckin**13*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(18711*mbkin**13) -
         (101312*mckin**14*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (243243*mbkin**14) + (189952*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/34749 -
         (13568*mbkin*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (34749*mckin) - (135680*mckin*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(3861*mbkin) +
         (4748800*mckin**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**2) - (949760*mckin**3*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(2673*mbkin**3) +
         (189952*mckin**4*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (297*mbkin**4) - (189952*mckin**5*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(243*mbkin**5) +
         (135680*mckin**6*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (243*mbkin**6) - (135680*mckin**8*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(243*mbkin**8) +
         (189952*mckin**9*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (243*mbkin**9) - (189952*mckin**10*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(297*mbkin**10) +
         (949760*mckin**11*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**11) - (4748800*mckin**12*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(34749*mbkin**12) +
         (135680*mckin**13*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (3861*mbkin**13) - (189952*mckin**14*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(34749*mbkin**14) +
         (13568*mckin**15*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**15) + (446560*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/81081 -
         (89312*mbkin*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (243243*mckin) - (714496*mckin*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(18711*mbkin) +
         (3572480*mckin**2*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (22113*mbkin**2) - (1786240*mckin**3*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(3861*mbkin**3) +
         (2500736*mckin**4*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**4) - (3572480*mckin**5*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(2673*mbkin**5) +
         (714496*mckin**6*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (567*mbkin**6) - (893120*mckin**7*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(1701*mbkin**7) -
         (893120*mckin**8*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**8) + (714496*mckin**9*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(567*mbkin**9) -
         (3572480*mckin**10*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**10) + (2500736*mckin**11*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(2673*mbkin**11) -
         (1786240*mckin**12*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (3861*mbkin**12) + (3572480*mckin**13*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(22113*mbkin**13) -
         (714496*mckin**14*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (18711*mbkin**14) + (446560*mckin**15*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(81081*mbkin**15) -
         (89312*mckin**16*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (243243*mbkin**16) + (22912768*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/4135131 -
         (1432048*mbkin*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (4135131*mckin) - (1432048*mckin*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(34749*mbkin) +
         (45825536*mckin**2*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (243243*mbkin**2) - (143204800*mckin**3*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(243243*mbkin**3) +
         (45825536*mckin**4*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**4) - (5728192*mckin**5*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(2673*mbkin**5) +
         (45825536*mckin**6*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (18711*mbkin**6) - (2864096*mckin**7*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(1701*mbkin**7) +
         (2864096*mckin**9*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**9) - (45825536*mckin**10*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(18711*mbkin**10) +
         (5728192*mckin**11*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**11) - (45825536*mckin**12*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(34749*mbkin**12) +
         (143204800*mckin**13*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (243243*mbkin**13) - (45825536*mckin**14*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(243243*mbkin**14) +
         (1432048*mckin**15*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**15) - (22912768*mckin**16*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(4135131*mbkin**16) +
         (1432048*mckin**17*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (4135131*mbkin**17) + (1354552*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/243243 -
         (1354552*mbkin*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (4135131*mckin) - (6772760*mckin*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(153153*mbkin) +
         (1354552*mckin**2*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (6237*mbkin**2) - (5418208*mckin**3*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(7371*mbkin**3) +
         (5418208*mckin**4*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (3003*mbkin**4) - (37927456*mckin**5*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(11583*mbkin**5) +
         (27091040*mckin**6*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (6237*mbkin**6) - (2709104*mckin**7*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(693*mbkin**7) +
         (2709104*mckin**8*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**8) + (2709104*mckin**9*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(1701*mbkin**9) -
         (2709104*mckin**10*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (693*mbkin**10) + (27091040*mckin**11*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(6237*mbkin**11) -
         (37927456*mckin**12*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (11583*mbkin**12) + (5418208*mckin**13*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(3003*mbkin**13) -
         (5418208*mckin**14*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (7371*mbkin**14) + (1354552*mckin**15*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(6237*mbkin**15) -
         (6772760*mckin**16*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (153153*mbkin**16) + (1354552*mckin**17*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(243243*mbkin**17) -
         (1354552*mckin**18*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (4135131*mbkin**18) + (6973984*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/1247103 -
         (3486992*mbkin*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (11223927*mckin) - (27895936*mckin*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(590733*mbkin) +
         (48817888*mckin**2*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (196911*mbkin**2) - (3486992*mckin**3*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(3861*mbkin**3) +
         (27895936*mckin**4*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (11583*mbkin**4) - (55791872*mckin**5*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(11583*mbkin**5) +
         (27895936*mckin**6*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (3861*mbkin**6) - (6973984*mckin**7*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(891*mbkin**7) +
         (13947968*mckin**8*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**8) - (13947968*mckin**10*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(2673*mbkin**10) +
         (6973984*mckin**11*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (891*mbkin**11) - (27895936*mckin**12*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(3861*mbkin**12) +
         (55791872*mckin**13*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (11583*mbkin**13) - (27895936*mckin**14*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(11583*mbkin**14) +
         (3486992*mckin**15*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (3861*mbkin**15) - (48817888*mckin**16*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(196911*mbkin**16) +
         (27895936*mckin**17*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (590733*mbkin**17) - (6973984*mckin**18*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(1247103*mbkin**18) +
         (3486992*mckin**19*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (11223927*mbkin**19) + (301396*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/53703 -
         (301396*mbkin*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1020357*mckin) - (3013960*mckin*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(60021*mbkin) +
         (15069800*mckin**2*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (53703*mbkin**2) - (1506980*mckin**3*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1377*mbkin**3) +
         (3315356*mckin**4*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1053*mbkin**4) - (2411168*mckin**5*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(351*mbkin**5) +
         (12055840*mckin**6*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1053*mbkin**6) - (15069800*mckin**7*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1053*mbkin**7) +
         (3013960*mckin**8*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (243*mbkin**8) - (1205584*mckin**9*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(243*mbkin**9) -
         (1205584*mckin**10*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (243*mbkin**10) + (3013960*mckin**11*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(243*mbkin**11) -
         (15069800*mckin**12*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1053*mbkin**12) + (12055840*mckin**13*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1053*mbkin**13) -
         (2411168*mckin**14*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (351*mbkin**14) + (3315356*mckin**15*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1053*mbkin**15) -
         (1506980*mckin**16*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1377*mbkin**16) + (15069800*mckin**17*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(53703*mbkin**17) -
         (3013960*mckin**18*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (60021*mbkin**18) + (301396*mckin**19*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(53703*mbkin**19) -
         (301396*mckin**20*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1020357*mbkin**20) + (-260123404758497280*(1 - mckin/mbkin)**5 +
           390185107137745920*(1 - mckin/mbkin)**6 - 388690823028777984*
            (1 - mckin/mbkin)**7 + 195377647247557632*(1 - mckin/mbkin)**8 -
           34216429928650112*(1 - mckin/mbkin)**9 - 6097700477192896*
            (1 - mckin/mbkin)**10 - 3945134197513600*(1 - mckin/mbkin)**11 -
           3335963793491680*(1 - mckin/mbkin)**12 - 2990240473256960*
            (1 - mckin/mbkin)**13 - 2738659018760960*(1 - mckin/mbkin)**14 -
           2537989658886624*(1 - mckin/mbkin)**15 - 2370644934651168*
            (1 - mckin/mbkin)**16 - 2227287774097176*(1 - mckin/mbkin)**17 -
           2102198391233484*(1 - mckin/mbkin)**18 - 1991562479412482*
            (1 - mckin/mbkin)**19 - 1892686294969085*(1 - mckin/mbkin)**20 +
           132126173845585920*(1 - mckin/mbkin)**7*(np.log(2) +
             np.log(1 - mckin/mbkin)) - 66063086922792960*(1 - mckin/mbkin)**8*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 11010514487132160*
            (1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)) +
           5505257243566080*(1 - mckin/mbkin)**10*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 4754540346716160*(1 - mckin/mbkin)**11*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 4379181898291200*
            (1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)) +
           4076007766871040*(1 - mckin/mbkin)**13*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 3808925793953280*(1 - mckin/mbkin)**14*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 3570717547837440*
            (1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin)) +
           3357773812673280*(1 - mckin/mbkin)**16*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 3167016139647360*(1 - mckin/mbkin)**17*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 2995631463464640*
            (1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin)) +
           2841110280328320*(1 - mckin/mbkin)**19*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 2701265736929760*(1 - mckin/mbkin)**20*
            (np.log(2) + np.log(1 - mckin/mbkin)))/4572481724270460 +
         (3962*np.log(2/mus))/9 + (32*mbkin**2*np.log(2/mus))/mckin**2 +
         (1024*mbkin*np.log(2/mus))/(9*mckin) - (4096*mckin*np.log(2/mus))/
          (27*mbkin) - (20960*mckin**2*np.log(2/mus))/(27*mbkin**2) -
         (2048*mckin**3*np.log(2/mus))/(9*mbkin**3) + (3056*mckin**4*np.log(2/mus))/
          (9*mbkin**4) + (4096*mckin**5*np.log(2/mus))/(9*mbkin**5) -
         (1792*mckin**6*np.log(2/mus))/(9*mbkin**6) - (5120*mckin**7*np.log(2/mus))/
          (27*mbkin**7) + (4418*mckin**8*np.log(2/mus))/(27*mbkin**8) -
         (256*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2))*np.log(2/mus))/27 +
         288*np.log(mckin**2/mbkin**2)*np.log(2/mus) +
         (288*mckin**2*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**2 -
         (504*mckin**4*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**4 -
         40*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
           mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
          np.log(2/mus) - (512*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/243 +
         (512*mbkin*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(243*mckin) -
         (2048*mckin*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(729*mbkin) +
         (2048*mckin**2*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(729*mbkin**2) -
         (1024*mckin**3*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(243*mbkin**3) +
         (1024*mckin**4*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(243*mbkin**4) +
         (2048*mckin**5*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(243*mbkin**5) -
         (2048*mckin**6*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(243*mbkin**6) -
         (2560*mckin**7*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(729*mbkin**7) +
         (2560*mckin**8*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/(729*mbkin**8) +
         (64*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2))*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/243 +
         (32*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/81 +
         (16*mckin**2*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/(81*mbkin**2) -
         (16*mckin**4*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/(9*mbkin**4) +
         (112*mckin**6*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/(81*mbkin**6) -
         (16*mckin**8*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/(81*mbkin**8) +
         (32*mckin**2*np.log(mckin**2/mbkin**2)*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/
          (27*mbkin**2) - (32*mckin**4*np.log(mckin**2/mbkin**2)*(-234 + 9*np.pi**2 +
            81*np.log(2/mus)))/(27*mbkin**4) +
         (4*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           (-234 + 9*np.pi**2 + 81*np.log(2/mus)))/81 -
         (160*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2))*((2*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/
             27 + (4*np.log(mus**2/mckin**2))/27))/27 -
         (5*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           ((2*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/27 + (2*np.log(mus**2/mckin**2))/
             9))/3 + (3226*np.log(mus**2/mckin**2))/243 +
         (32*mbkin**2*np.log(mus**2/mckin**2))/(27*mckin**2) +
         (2048*mbkin*np.log(mus**2/mckin**2))/(243*mckin) -
         (8192*mckin*np.log(mus**2/mckin**2))/(729*mbkin) -
         (16432*mckin**2*np.log(mus**2/mckin**2))/(729*mbkin**2) -
         (4096*mckin**3*np.log(mus**2/mckin**2))/(243*mbkin**3) +
         (3808*mckin**4*np.log(mus**2/mckin**2))/(243*mbkin**4) +
         (8192*mckin**5*np.log(mus**2/mckin**2))/(243*mbkin**5) -
         (4880*mckin**6*np.log(mus**2/mckin**2))/(243*mbkin**6) -
         (10240*mckin**7*np.log(mus**2/mckin**2))/(729*mbkin**7) +
         (9106*mckin**8*np.log(mus**2/mckin**2))/(729*mbkin**8) +
         (128*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2))*np.log(mus**2/mckin**2))/729 +
         (32*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/3 +
         (128*mckin**2*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/(9*mbkin**2) -
         (200*mckin**4*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/(9*mbkin**4) -
         (4*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(mus**2/mckin**2))/3))/mbkin**4 +
     (api4*(80/9 - (256*mckin)/(9*mbkin) - (128*mckin**2)/(3*mbkin**2) -
         (128*mckin**3)/(3*mbkin**3) + (128*mckin**4)/(3*mbkin**4) +
         (256*mckin**5)/(3*mbkin**5) - (128*mckin**6)/(9*mbkin**6) -
         (128*mckin**7)/(9*mbkin**7) + (16*mckin**8)/(3*mbkin**8) -
         (256*mckin**3*np.log(mckin**2/mbkin**2))/(3*mbkin**3) -
         (64*mckin**4*np.log(mckin**2/mbkin**2))/(3*mbkin**4)) +
       api4**2*(2649349310527609/34640013062655 - (121060300623525443*mckin)/
          (103920039187965*mbkin) + (1440839856915754643*mckin**2)/
          (300821166070425*mbkin**2) - (134827643679333109*mckin**3)/
          (6684914801565*mbkin**3) + (108855982869738326*mckin**4)/
          (1608669337275*mbkin**4) - (631608607657799036*mckin**5)/
          (3539072542005*mbkin**5) + (280484771740852*mckin**6)/
          (729555255*mbkin**6) - (33457308112150388*mckin**7)/
          (49566842325*mbkin**7) + (4857430359237682*mckin**8)/
          (4986013725*mbkin**8) - (3706850196725426*mckin**9)/
          (3172917825*mbkin**9) + (28907674909392014*mckin**10)/
          (24748759035*mbkin**10) - (24087722462611706*mckin**11)/
          (24748759035*mbkin**11) + (3088054860268528*mckin**12)/
          (4583103525*mbkin**12) - (139041722784092188*mckin**13)/
          (361129851225*mbkin**13) + (187024937060038172*mckin**14)/
          (1040903688825*mbkin**14) - (2408676010351724*mckin**15)/
          (35748207495*mbkin**15) + (7792880276783387*mckin**16)/
          (393230282445*mbkin**16) - (1911703888198739*mckin**17)/
          (434085376725*mbkin**17) + (69728083745549401*mckin**18)/
          (100273722023475*mbkin**18) - (397457102548235579*mckin**19)/
          (5715602155338075*mbkin**19) + (9832136597242*mckin**20)/
          (2969143976799*mbkin**20) + (128*mckin*np.pi**2)/(9*mbkin) -
         (128*mckin**2*np.pi**2)/(9*mbkin**2) + (64*mckin**3*np.pi**2)/(3*mbkin**3) -
         (64*mckin**4*np.pi**2)/(3*mbkin**4) - (128*mckin**5*np.pi**2)/(3*mbkin**5) +
         (128*mckin**6*np.pi**2)/(3*mbkin**6) + (64*mckin**7*np.pi**2)/(9*mbkin**7) -
         (64*mckin**8*np.pi**2)/(9*mbkin**8) - (3904*mckin**3*np.log(mckin**2/mbkin**2))/
          (3*mbkin**3) + (3904*mckin**4*np.log(mckin**2/mbkin**2))/(3*mbkin**4) +
         (128*mckin**3*np.pi**2*np.log(mckin**2/mbkin**2))/(3*mbkin**3) -
         (128*mckin**4*np.pi**2*np.log(mckin**2/mbkin**2))/(3*mbkin**4) +
         (1220*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/9 -
         (40*np.pi**2*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/9 -
         (32768*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/2835 +
         (32768*mckin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(405*mbkin) -
         (32768*mckin**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (135*mbkin**2) + (32768*mckin**3*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(81*mbkin**3) -
         (32768*mckin**4*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (81*mbkin**4) + (32768*mckin**5*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(135*mbkin**5) -
         (32768*mckin**6*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (405*mbkin**6) + (32768*mckin**7*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(2835*mbkin**7) +
         (16384*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/2835 -
         (131072*mckin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (2835*mbkin) + (65536*mckin**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(405*mbkin**2) -
         (131072*mckin**3*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (405*mbkin**3) + (32768*mckin**4*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(81*mbkin**4) -
         (131072*mckin**5*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (405*mbkin**5) + (65536*mckin**6*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(405*mbkin**6) -
         (131072*mckin**7*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**7) + (16384*mckin**8*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(2835*mbkin**8) -
         (8192*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/8505 +
         (8192*mckin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/(945*mbkin) -
         (32768*mckin**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (945*mbkin**2) + (32768*mckin**3*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(405*mbkin**3) -
         (16384*mckin**4*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (135*mbkin**4) + (16384*mckin**5*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(135*mbkin**5) -
         (32768*mckin**6*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (405*mbkin**6) + (32768*mckin**7*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(945*mbkin**7) -
         (8192*mckin**8*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (945*mbkin**8) + (8192*mckin**9*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(8505*mbkin**9) -
         (4096*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/8505 +
         (8192*mckin*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (1701*mbkin) - (4096*mckin**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(189*mbkin**2) +
         (32768*mckin**3*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (567*mbkin**3) - (8192*mckin**4*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(81*mbkin**4) +
         (16384*mckin**5*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) - (8192*mckin**6*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(81*mbkin**6) +
         (32768*mckin**7*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (567*mbkin**7) - (4096*mckin**8*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(189*mbkin**8) +
         (8192*mckin**9*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**9) - (4096*mckin**10*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(8505*mbkin**10) -
         (38912*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/93555 +
         (38912*mckin*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (8505*mbkin) - (38912*mckin**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(1701*mbkin**2) +
         (38912*mckin**3*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (567*mbkin**3) - (77824*mckin**4*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(567*mbkin**4) +
         (77824*mckin**5*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (405*mbkin**5) - (77824*mckin**6*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(405*mbkin**6) +
         (77824*mckin**7*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (567*mbkin**7) - (38912*mckin**8*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(567*mbkin**8) +
         (38912*mckin**9*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**9) - (38912*mckin**10*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(8505*mbkin**10) +
         (38912*mckin**11*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (93555*mbkin**11) - (1024*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          2673 + (4096*mckin*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (891*mbkin) - (2048*mckin**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**2) +
         (20480*mckin**3*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (243*mbkin**3) - (5120*mckin**4*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(27*mbkin**4) +
         (8192*mckin**5*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (27*mbkin**5) - (28672*mckin**6*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**6) +
         (8192*mckin**7*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (27*mbkin**7) - (5120*mckin**8*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(27*mbkin**8) +
         (20480*mckin**9*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (243*mbkin**9) - (2048*mckin**10*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**10) +
         (4096*mckin**11*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (891*mbkin**11) - (1024*mckin**12*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(2673*mbkin**12) -
         (5632*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/15795 +
         (5632*mckin*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (1215*mbkin) - (11264*mckin**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(405*mbkin**2) +
         (123904*mckin**3*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (1215*mbkin**3) - (61952*mckin**4*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(243*mbkin**4) +
         (61952*mckin**5*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) - (247808*mckin**6*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(405*mbkin**6) +
         (247808*mckin**7*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (405*mbkin**7) - (61952*mckin**8*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(135*mbkin**8) +
         (61952*mckin**9*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (243*mbkin**9) - (123904*mckin**10*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(1215*mbkin**10) +
         (11264*mckin**11*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (405*mbkin**11) - (5632*mckin**12*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(1215*mbkin**12) +
         (5632*mckin**13*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (15795*mbkin**13) - (405248*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/1216215 +
         (810496*mckin*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (173745*mbkin) - (405248*mckin**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(13365*mbkin**2) +
         (1620992*mckin**3*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (13365*mbkin**3) - (405248*mckin**4*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(1215*mbkin**4) +
         (810496*mckin**5*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (1215*mbkin**5) - (405248*mckin**6*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(405*mbkin**6) +
         (3241984*mckin**7*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**7) - (405248*mckin**8*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(405*mbkin**8) +
         (810496*mckin**9*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (1215*mbkin**9) - (405248*mckin**10*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(1215*mbkin**10) +
         (1620992*mckin**11*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (13365*mbkin**11) - (405248*mckin**12*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(13365*mbkin**12) +
         (810496*mckin**13*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (173745*mbkin**13) - (405248*mckin**14*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(1216215*mbkin**14) -
         (54272*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/173745 +
         (54272*mckin*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (11583*mbkin) - (379904*mckin**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(11583*mbkin**2) +
         (379904*mckin**3*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**3) - (379904*mckin**4*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(891*mbkin**4) +
         (379904*mckin**5*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (405*mbkin**5) - (379904*mckin**6*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(243*mbkin**6) +
         (54272*mckin**7*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (27*mbkin**7) - (54272*mckin**8*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(27*mbkin**8) +
         (379904*mckin**9*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (243*mbkin**9) - (379904*mckin**10*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(405*mbkin**10) +
         (379904*mckin**11*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (891*mbkin**11) - (379904*mckin**12*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(2673*mbkin**12) +
         (379904*mckin**13*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (11583*mbkin**13) - (54272*mckin**14*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(11583*mbkin**14) +
         (54272*mckin**15*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (173745*mbkin**15) - (357248*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/1216215 +
         (5715968*mckin*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (1216215*mbkin) - (2857984*mckin**2*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(81081*mbkin**2) +
         (5715968*mckin**3*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**3) - (1428992*mckin**4*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(2673*mbkin**4) +
         (5715968*mckin**5*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (4455*mbkin**5) - (2857984*mckin**6*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(1215*mbkin**6) +
         (5715968*mckin**7*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**7) - (714496*mckin**8*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(189*mbkin**8) +
         (5715968*mckin**9*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**9) - (2857984*mckin**10*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(1215*mbkin**10) +
         (5715968*mckin**11*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (4455*mbkin**11) - (1428992*mckin**12*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(2673*mbkin**12) +
         (5715968*mckin**13*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**13) - (2857984*mckin**14*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(81081*mbkin**14) +
         (5715968*mckin**15*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (1216215*mbkin**15) - (357248*mckin**16*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(1216215*mbkin**16) -
         (5728192*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/20675655 +
         (5728192*mckin*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (1216215*mbkin) - (45825536*mckin**2*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(1216215*mbkin**2) +
         (45825536*mckin**3*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (243243*mbkin**3) - (22912768*mckin**4*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(34749*mbkin**4) +
         (22912768*mckin**5*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (13365*mbkin**5) - (45825536*mckin**6*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(13365*mbkin**6) +
         (45825536*mckin**7*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (8505*mbkin**7) - (11456384*mckin**8*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(1701*mbkin**8) +
         (11456384*mckin**9*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**9) - (45825536*mckin**10*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(8505*mbkin**10) +
         (45825536*mckin**11*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (13365*mbkin**11) - (22912768*mckin**12*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(13365*mbkin**12) +
         (22912768*mckin**13*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**13) - (45825536*mckin**14*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(243243*mbkin**14) +
         (45825536*mckin**15*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (1216215*mbkin**15) - (5728192*mckin**16*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(1216215*mbkin**16) +
         (5728192*mckin**17*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (20675655*mbkin**17) - (5418208*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/20675655 +
         (10836416*mckin*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (2297295*mbkin) - (5418208*mckin**2*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(135135*mbkin**2) +
         (86691328*mckin**3*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (405405*mbkin**3) - (21672832*mckin**4*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(27027*mbkin**4) +
         (43345664*mckin**5*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (19305*mbkin**5) - (21672832*mckin**6*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(4455*mbkin**6) +
         (86691328*mckin**7*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (10395*mbkin**7) - (10836416*mckin**8*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(945*mbkin**8) +
         (21672832*mckin**9*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**9) - (10836416*mckin**10*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(945*mbkin**10) +
         (86691328*mckin**11*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (10395*mbkin**11) - (21672832*mckin**12*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(4455*mbkin**12) +
         (43345664*mckin**13*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (19305*mbkin**13) - (21672832*mckin**14*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(27027*mbkin**14) +
         (86691328*mckin**15*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (405405*mbkin**15) - (5418208*mckin**16*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(135135*mbkin**16) +
         (10836416*mckin**17*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (2297295*mbkin**17) - (5418208*mckin**18*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(20675655*mbkin**18) -
         (13947968*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/56119635 +
         (13947968*mckin*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (2953665*mbkin) - (13947968*mckin**2*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(328185*mbkin**2) +
         (13947968*mckin**3*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (57915*mbkin**3) - (55791872*mckin**4*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(57915*mbkin**4) +
         (55791872*mckin**5*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (19305*mbkin**5) - (390543104*mckin**6*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(57915*mbkin**6) +
         (55791872*mckin**7*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (4455*mbkin**7) - (27895936*mckin**8*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(1485*mbkin**8) +
         (27895936*mckin**9*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (1215*mbkin**9) - (27895936*mckin**10*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(1215*mbkin**10) +
         (27895936*mckin**11*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (1485*mbkin**11) - (55791872*mckin**12*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(4455*mbkin**12) +
         (390543104*mckin**13*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (57915*mbkin**13) - (55791872*mckin**14*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(19305*mbkin**14) +
         (55791872*mckin**15*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (57915*mbkin**15) - (13947968*mckin**16*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(57915*mbkin**16) +
         (13947968*mckin**17*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (328185*mbkin**17) - (13947968*mckin**18*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(2953665*mbkin**18) +
         (13947968*mckin**19*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (56119635*mbkin**19) - (1205584*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/5101785 +
         (4822336*mckin*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1020357*mbkin) - (2411168*mckin**2*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(53703*mbkin**2) +
         (4822336*mckin**3*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (17901*mbkin**3) - (1205584*mckin**4*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1053*mbkin**4) +
         (19289344*mckin**5*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (5265*mbkin**5) - (9644672*mckin**6*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1053*mbkin**6) +
         (19289344*mckin**7*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1053*mbkin**7) - (2411168*mckin**8*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(81*mbkin**8) +
         (9644672*mckin**9*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (243*mbkin**9) - (53045696*mckin**10*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1215*mbkin**10) +
         (9644672*mckin**11*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (243*mbkin**11) - (2411168*mckin**12*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(81*mbkin**12) +
         (19289344*mckin**13*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1053*mbkin**13) - (9644672*mckin**14*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1053*mbkin**14) +
         (19289344*mckin**15*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (5265*mbkin**15) - (1205584*mckin**16*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1053*mbkin**16) +
         (4822336*mckin**17*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (17901*mbkin**17) - (2411168*mckin**18*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(53703*mbkin**18) +
         (4822336*mckin**19*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1020357*mbkin**19) - (1205584*mckin**20*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(5101785*mbkin**20) +
         (-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
            (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
           195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
            (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
           3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
            (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
           2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
            (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
           2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
            (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
           1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
            (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
           66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
            (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
           4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
            (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
           3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
            (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
           3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
            (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
           2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
             np.log(1 - mckin/mbkin)))/2286240862135230 + (128*mckin*np.log(2/mus))/
          mbkin - (128*mckin**2*np.log(2/mus))/mbkin**2 + (192*mckin**3*np.log(2/mus))/
          mbkin**3 - (192*mckin**4*np.log(2/mus))/mbkin**4 -
         (384*mckin**5*np.log(2/mus))/mbkin**5 + (384*mckin**6*np.log(2/mus))/
          mbkin**6 + (64*mckin**7*np.log(2/mus))/mbkin**7 - (64*mckin**8*np.log(2/mus))/
          mbkin**8 + (384*mckin**3*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**3 -
         (384*mckin**4*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**4 -
         40*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
           mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
          np.log(2/mus) + (128*mckin*np.log(mus**2/mckin**2))/(27*mbkin) -
         (128*mckin**2*np.log(mus**2/mckin**2))/(27*mbkin**2) +
         (64*mckin**3*np.log(mus**2/mckin**2))/(9*mbkin**3) -
         (64*mckin**4*np.log(mus**2/mckin**2))/(9*mbkin**4) -
         (128*mckin**5*np.log(mus**2/mckin**2))/(9*mbkin**5) +
         (128*mckin**6*np.log(mus**2/mckin**2))/(9*mbkin**6) +
         (64*mckin**7*np.log(mus**2/mckin**2))/(27*mbkin**7) -
         (64*mckin**8*np.log(mus**2/mckin**2))/(27*mbkin**8) +
         (128*mckin**3*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/(9*mbkin**3) -
         (128*mckin**4*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/(9*mbkin**4) -
         (40*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(mus**2/mckin**2))/27) + api4**3*(604.6652786744438 -
         (252199.58207605133*mckin)/mbkin + (1.5750809669713078e6*mckin**2)/
          mbkin**2 - (6.269469518269783e6*mckin**3)/mbkin**3 +
         (1.7392000922279537e7*mckin**4)/mbkin**4 -
         (3.551015910572836e7*mckin**5)/mbkin**5 +
         (5.519348934597813e7*mckin**6)/mbkin**6 -
         (6.641260105733509e7*mckin**7)/mbkin**7 +
         (6.232961661959774e7*mckin**8)/mbkin**8 -
         (4.5640930269306965e7*mckin**9)/mbkin**9 +
         (2.5861632376156766e7*mckin**10)/mbkin**10 -
         (1.1133023142903041e7*mckin**11)/mbkin**11 +
         (3.5237410254216627e6*mckin**12)/mbkin**12 -
         (773985.2832044418*mckin**13)/mbkin**13 +
         (105472.14087760425*mckin**14)/mbkin**14 -
         (6720.9300265884995*mckin**15)/mbkin**15 + (15616*(1 - mckin/mbkin)**4)/
          9 - (15616*mckin*(1 - mckin/mbkin)**4)/(9*mbkin) -
         (15616*(1 - mckin/mbkin)**5)/5 + (15616*mckin*(1 - mckin/mbkin)**5)/
          (5*mbkin) + (20980096*(1 - mckin/mbkin)**6)/6075 -
         (20980096*mckin*(1 - mckin/mbkin)**6)/(6075*mbkin) -
         (594469888*(1 - mckin/mbkin)**7)/297675 +
         (594469888*mckin*(1 - mckin/mbkin)**7)/(297675*mbkin) +
         (39306448*(1 - mckin/mbkin)**8)/99225 -
         (39306448*mckin*(1 - mckin/mbkin)**8)/(99225*mbkin) +
         (39651952*(1 - mckin/mbkin)**9)/535815 -
         (39651952*mckin*(1 - mckin/mbkin)**9)/(535815*mbkin) +
         (3945968*(1 - mckin/mbkin)**10)/76545 -
         (3945968*mckin*(1 - mckin/mbkin)**10)/(76545*mbkin) +
         (1027887088*(1 - mckin/mbkin)**11)/21611205 -
         (1027887088*mckin*(1 - mckin/mbkin)**11)/(21611205*mbkin) +
         (39086848*(1 - mckin/mbkin)**12)/841995 -
         (39086848*mckin*(1 - mckin/mbkin)**12)/(841995*mbkin) +
         (10301352064*(1 - mckin/mbkin)**13)/223609815 -
         (10301352064*mckin*(1 - mckin/mbkin)**13)/(223609815*mbkin) +
         (18677087116*(1 - mckin/mbkin)**14)/405810405 -
         (18677087116*mckin*(1 - mckin/mbkin)**14)/(405810405*mbkin) +
         (120322503904*(1 - mckin/mbkin)**15)/2608781175 -
         (120322503904*mckin*(1 - mckin/mbkin)**15)/(2608781175*mbkin) +
         (120755211491*(1 - mckin/mbkin)**16)/2608781175 -
         (120755211491*mckin*(1 - mckin/mbkin)**16)/(2608781175*mbkin) +
         (81774404212951*(1 - mckin/mbkin)**17)/1759188105675 -
         (81774404212951*mckin*(1 - mckin/mbkin)**17)/(1759188105675*mbkin) +
         (2956943471660719*(1 - mckin/mbkin)**18)/63330771804300 -
         (2956943471660719*mckin*(1 - mckin/mbkin)**18)/(63330771804300*
           mbkin) + (1392402642798421*(1 - mckin/mbkin)**19)/29691439767990 -
         (1392402642798421*mckin*(1 - mckin/mbkin)**19)/(29691439767990*
           mbkin) + (9812998290947413*mckin*np.pi**2)/(394447713660*mbkin) -
         (367551879097253*mckin**2*np.pi**2)/(2328498900*mbkin**2) +
         (95651015223803623*mckin**3*np.pi**2)/(151710659100*mbkin**3) -
         (265574999571993821*mckin**4*np.pi**2)/(151710659100*mbkin**4) +
         (1409000557876237*mckin**5*np.pi**2)/(394053660*mbkin**5) -
         (243325215712279*mckin**6*np.pi**2)/(43783740*mbkin**6) +
         (30742968422992271*mckin**7*np.pi**2)/(4597292700*mbkin**7) -
         (739994098365497*mckin**8*np.pi**2)/(117879300*mbkin**8) +
         (274519133724491*mckin**9*np.pi**2)/(59705100*mbkin**9) -
         (7187882048892731*mckin**10*np.pi**2)/(2758375620*mbkin**10) +
         (34042318494414169*mckin**11*np.pi**2)/(30342131820*mbkin**11) -
         (7697312041067713*mckin**12*np.pi**2)/(21672951300*mbkin**12) +
         (285475794361297*mckin**13*np.pi**2)/(3659069700*mbkin**13) -
         (599152523841059*mckin**14*np.pi**2)/(56349673380*mbkin**14) +
         (2545509950573*mckin**15*np.pi**2)/(3756644892*mbkin**15) -
         (512*(1 - mckin/mbkin)**4*np.pi**2)/9 + (512*mckin*(1 - mckin/mbkin)**4*
           np.pi**2)/(9*mbkin) + (512*(1 - mckin/mbkin)**5*np.pi**2)/5 -
         (512*mckin*(1 - mckin/mbkin)**5*np.pi**2)/(5*mbkin) -
         (687872*(1 - mckin/mbkin)**6*np.pi**2)/6075 +
         (687872*mckin*(1 - mckin/mbkin)**6*np.pi**2)/(6075*mbkin) +
         (19490816*(1 - mckin/mbkin)**7*np.pi**2)/297675 -
         (19490816*mckin*(1 - mckin/mbkin)**7*np.pi**2)/(297675*mbkin) -
         (1288736*(1 - mckin/mbkin)**8*np.pi**2)/99225 +
         (1288736*mckin*(1 - mckin/mbkin)**8*np.pi**2)/(99225*mbkin) -
         (1300064*(1 - mckin/mbkin)**9*np.pi**2)/535815 +
         (1300064*mckin*(1 - mckin/mbkin)**9*np.pi**2)/(535815*mbkin) -
         (129376*(1 - mckin/mbkin)**10*np.pi**2)/76545 +
         (129376*mckin*(1 - mckin/mbkin)**10*np.pi**2)/(76545*mbkin) -
         (33701216*(1 - mckin/mbkin)**11*np.pi**2)/21611205 +
         (33701216*mckin*(1 - mckin/mbkin)**11*np.pi**2)/(21611205*mbkin) -
         (1281536*(1 - mckin/mbkin)**12*np.pi**2)/841995 +
         (1281536*mckin*(1 - mckin/mbkin)**12*np.pi**2)/(841995*mbkin) -
         (337749248*(1 - mckin/mbkin)**13*np.pi**2)/223609815 +
         (337749248*mckin*(1 - mckin/mbkin)**13*np.pi**2)/(223609815*mbkin) -
         (612363512*(1 - mckin/mbkin)**14*np.pi**2)/405810405 +
         (612363512*mckin*(1 - mckin/mbkin)**14*np.pi**2)/(405810405*mbkin) -
         (3945000128*(1 - mckin/mbkin)**15*np.pi**2)/2608781175 +
         (3945000128*mckin*(1 - mckin/mbkin)**15*np.pi**2)/(2608781175*mbkin) -
         (3959187262*(1 - mckin/mbkin)**16*np.pi**2)/2608781175 +
         (3959187262*mckin*(1 - mckin/mbkin)**16*np.pi**2)/(2608781175*mbkin) -
         (2681128006982*(1 - mckin/mbkin)**17*np.pi**2)/1759188105675 +
         (2681128006982*mckin*(1 - mckin/mbkin)**17*np.pi**2)/
          (1759188105675*mbkin) - (48474483141979*(1 - mckin/mbkin)**18*np.pi**2)/
          31665385902150 + (48474483141979*mckin*(1 - mckin/mbkin)**18*np.pi**2)/
          (31665385902150*mbkin) - (22826272832761*(1 - mckin/mbkin)**19*np.pi**2)/
          14845719883995 + (22826272832761*mckin*(1 - mckin/mbkin)**19*np.pi**2)/
          (14845719883995*mbkin) - (32*mckin*np.pi**4)/(3*mbkin) +
         (32*mckin**2*np.pi**4)/(3*mbkin**2) - (16*mckin**3*np.pi**4)/mbkin**3 +
         (16*mckin**4*np.pi**4)/mbkin**4 + (32*mckin**5*np.pi**4)/mbkin**5 -
         (32*mckin**6*np.pi**4)/mbkin**6 - (16*mckin**7*np.pi**4)/(3*mbkin**7) +
         (16*mckin**8*np.pi**4)/(3*mbkin**8) + (3413216*mckin*np.pi**2*np.log(2))/
          (66339*mbkin) - (995971616*mckin**2*np.pi**2*np.log(2))/(3648645*mbkin**2) +
         (2664448*mckin**3*np.pi**2*np.log(2))/(2835*mbkin**3) -
         (57852416*mckin**4*np.pi**2*np.log(2))/(25515*mbkin**4) +
         (20753056*mckin**5*np.pi**2*np.log(2))/(5103*mbkin**5) -
         (48120416*mckin**6*np.pi**2*np.log(2))/(8505*mbkin**6) +
         (53049856*mckin**7*np.pi**2*np.log(2))/(8505*mbkin**7) -
         (46529536*mckin**8*np.pi**2*np.log(2))/(8505*mbkin**8) +
         (32274016*mckin**9*np.pi**2*np.log(2))/(8505*mbkin**9) -
         (52461856*mckin**10*np.pi**2*np.log(2))/(25515*mbkin**10) +
         (239280128*mckin**11*np.pi**2*np.log(2))/(280665*mbkin**11) -
         (740864*mckin**12*np.pi**2*np.log(2))/(2835*mbkin**12) +
         (15668896*mckin**13*np.pi**2*np.log(2))/(280665*mbkin**13) -
         (2462816*mckin**14*np.pi**2*np.log(2))/(331695*mbkin**14) +
         (10240*mckin**15*np.pi**2*np.log(2))/(22113*mbkin**15) -
         (21956.324891380977*mckin**3*np.log(mckin**2/mbkin**2))/mbkin**3 +
         (21956.324891380977*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4 +
         (4384*mckin**3*np.pi**2*np.log(mckin**2/mbkin**2))/(3*mbkin**3) -
         (4384*mckin**4*np.pi**2*np.log(mckin**2/mbkin**2))/(3*mbkin**4) -
         (32*mckin**3*np.pi**4*np.log(mckin**2/mbkin**2))/mbkin**3 +
         (32*mckin**4*np.pi**4*np.log(mckin**2/mbkin**2))/mbkin**4 +
         1109.4410319108126*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
           mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4) -
         (499712*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin)))/405 +
         (499712*mckin*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (405*mbkin) + (1998848*(1 - mckin/mbkin)**7*(np.log(2) +
            np.log(1 - mckin/mbkin)))/2835 - (1998848*mckin*(1 - mckin/mbkin)**7*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(2835*mbkin) -
         (124928*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)))/945 +
         (124928*mckin*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (945*mbkin) - (124928*(1 - mckin/mbkin)**9*(np.log(2) +
            np.log(1 - mckin/mbkin)))/1701 + (124928*mckin*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1701*mbkin) -
         (593408*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)))/8505 +
         (593408*mckin*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (8505*mbkin) - (62464*(1 - mckin/mbkin)**11*(np.log(2) +
            np.log(1 - mckin/mbkin)))/891 + (62464*mckin*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(891*mbkin) -
         (85888*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)))/1215 +
         (85888*mckin*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1215*mbkin) - (12360064*(1 - mckin/mbkin)**13*
           (np.log(2) + np.log(1 - mckin/mbkin)))/173745 +
         (12360064*mckin*(1 - mckin/mbkin)**13*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(173745*mbkin) -
         (827648*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)))/
          11583 + (827648*mckin*(1 - mckin/mbkin)**14*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(11583*mbkin) -
         (87168512*(1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1216215 + (87168512*mckin*(1 - mckin/mbkin)**15*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1216215*mbkin) -
         (87354928*(1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1216215 + (87354928*mckin*(1 - mckin/mbkin)**16*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1216215*mbkin) -
         (165255344*(1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin)))/
          2297295 + (165255344*mckin*(1 - mckin/mbkin)**17*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(2297295*mbkin) -
         (212706512*(1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin)))/
          2953665 + (212706512*mckin*(1 - mckin/mbkin)**18*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(2953665*mbkin) -
         (73540624*(1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1020357 + (73540624*mckin*(1 - mckin/mbkin)**19*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1020357*mbkin) +
         (16384*(1 - mckin/mbkin)**6*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          405 - (16384*mckin*(1 - mckin/mbkin)**6*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(405*mbkin) -
         (65536*(1 - mckin/mbkin)**7*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          2835 + (65536*mckin*(1 - mckin/mbkin)**7*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(2835*mbkin) +
         (4096*(1 - mckin/mbkin)**8*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          945 - (4096*mckin*(1 - mckin/mbkin)**8*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(945*mbkin) +
         (4096*(1 - mckin/mbkin)**9*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1701 - (4096*mckin*(1 - mckin/mbkin)**9*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1701*mbkin) +
         (19456*(1 - mckin/mbkin)**10*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          8505 - (19456*mckin*(1 - mckin/mbkin)**10*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(8505*mbkin) +
         (2048*(1 - mckin/mbkin)**11*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          891 - (2048*mckin*(1 - mckin/mbkin)**11*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(891*mbkin) +
         (2816*(1 - mckin/mbkin)**12*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1215 - (2816*mckin*(1 - mckin/mbkin)**12*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1215*mbkin) +
         (405248*(1 - mckin/mbkin)**13*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          173745 - (405248*mckin*(1 - mckin/mbkin)**13*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(173745*mbkin) +
         (27136*(1 - mckin/mbkin)**14*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          11583 - (27136*mckin*(1 - mckin/mbkin)**14*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(11583*mbkin) +
         (2857984*(1 - mckin/mbkin)**15*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1216215 - (2857984*mckin*(1 - mckin/mbkin)**15*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1216215*mbkin) +
         (2864096*(1 - mckin/mbkin)**16*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1216215 - (2864096*mckin*(1 - mckin/mbkin)**16*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1216215*mbkin) +
         (5418208*(1 - mckin/mbkin)**17*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          2297295 - (5418208*mckin*(1 - mckin/mbkin)**17*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(2297295*mbkin) +
         (6973984*(1 - mckin/mbkin)**18*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          2953665 - (6973984*mckin*(1 - mckin/mbkin)**18*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(2953665*mbkin) +
         (2411168*(1 - mckin/mbkin)**19*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1020357 - (2411168*mckin*(1 - mckin/mbkin)**19*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1020357*mbkin) -
         (8192*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/315 +
         (65536*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(315*mbkin) -
         (32768*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(45*mbkin**2) +
         (65536*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(45*mbkin**3) -
         (16384*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(9*mbkin**4) +
         (65536*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(45*mbkin**5) -
         (32768*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(45*mbkin**6) +
         (65536*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(315*mbkin**7) -
         (8192*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(315*mbkin**8) -
         (321536*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/229635 +
         (643072*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(45927*mbkin) -
         (321536*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(5103*mbkin**2) +
         (2572288*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 5*np.log(2) + 5*np.log(1 - mckin/mbkin)))/(15309*mbkin**3) -
         (643072*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(2187*mbkin**4) +
         (1286144*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 5*np.log(2) + 5*np.log(1 - mckin/mbkin)))/(3645*mbkin**5) -
         (643072*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(2187*mbkin**6) +
         (2572288*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 5*np.log(2) + 5*np.log(1 - mckin/mbkin)))/(15309*mbkin**7) -
         (321536*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(5103*mbkin**8) +
         (643072*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(45927*mbkin**9) -
         (321536*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 5*np.log(2) + 5*np.log(1 - mckin/mbkin)))/(229635*mbkin**10) -
         (4115968*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 6*np.log(2) +
            6*np.log(1 - mckin/mbkin)))/2525985 +
         (16463872*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 6*np.log(2) +
            6*np.log(1 - mckin/mbkin)))/(841995*mbkin) -
         (8231936*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(76545*mbkin**2) +
         (16463872*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(45927*mbkin**3) -
         (4115968*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(5103*mbkin**4) +
         (32927744*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(25515*mbkin**5) -
         (16463872*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(10935*mbkin**6) +
         (32927744*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(25515*mbkin**7) -
         (4115968*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(5103*mbkin**8) +
         (16463872*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(45927*mbkin**9) -
         (8231936*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(76545*mbkin**10) +
         (16463872*mckin**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(841995*mbkin**11) -
         (4115968*mckin**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(2525985*mbkin**12) -
         (68816896*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/297675 +
         (68816896*mckin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (42525*mbkin) - (68816896*mckin**2*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(14175*mbkin**2) +
         (68816896*mckin**3*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (8505*mbkin**3) - (68816896*mckin**4*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(8505*mbkin**4) +
         (68816896*mckin**5*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (14175*mbkin**5) - (68816896*mckin**6*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(42525*mbkin**6) +
         (68816896*mckin**7*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (297675*mbkin**7) + (16384*np.pi**2*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/2835 - (16384*mckin*np.pi**2*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(405*mbkin) +
         (16384*mckin**2*np.pi**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (135*mbkin**2) - (16384*mckin**3*np.pi**2*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(81*mbkin**3) +
         (16384*mckin**4*np.pi**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (81*mbkin**4) - (16384*mckin**5*np.pi**2*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(135*mbkin**5) +
         (16384*mckin**6*np.pi**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (405*mbkin**6) - (16384*mckin**7*np.pi**2*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(2835*mbkin**7) -
         (3696256*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/2189187 +
         (7392512*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(312741*mbkin) -
         (3696256*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(24057*mbkin**2) +
         (14785024*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(24057*mbkin**3) -
         (3696256*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(2187*mbkin**4) +
         (7392512*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(2187*mbkin**5) -
         (3696256*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(729*mbkin**6) +
         (29570048*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(5103*mbkin**7) -
         (3696256*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(729*mbkin**8) +
         (7392512*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(2187*mbkin**9) -
         (3696256*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(2187*mbkin**10) +
         (14785024*mckin**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(24057*mbkin**11) -
         (3696256*mckin**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(24057*mbkin**12) +
         (7392512*mckin**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(312741*mbkin**13) -
         (3696256*mckin**14*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(2189187*mbkin**14) +
         (8192*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/315 -
         (8192*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(45*mbkin) +
         (8192*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(15*mbkin**2) -
         (8192*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(9*mbkin**3) +
         (8192*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(9*mbkin**4) -
         (8192*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(15*mbkin**5) +
         (8192*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(45*mbkin**6) -
         (8192*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(315*mbkin**7) +
         (1316864*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/11025 -
         (10534912*mckin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (11025*mbkin) + (5267456*mckin**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(1575*mbkin**2) -
         (10534912*mckin**3*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (1575*mbkin**3) + (2633728*mckin**4*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(315*mbkin**4) -
         (10534912*mckin**5*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (1575*mbkin**5) + (5267456*mckin**6*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(1575*mbkin**6) -
         (10534912*mckin**7*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (11025*mbkin**7) + (1316864*mckin**8*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(11025*mbkin**8) -
         (8192*np.pi**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/2835 +
         (65536*mckin*np.pi**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (2835*mbkin) - (32768*mckin**2*np.pi**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(405*mbkin**2) +
         (65536*mckin**3*np.pi**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (405*mbkin**3) - (16384*mckin**4*np.pi**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(81*mbkin**4) +
         (65536*mckin**5*np.pi**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (405*mbkin**5) - (32768*mckin**6*np.pi**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(405*mbkin**6) +
         (65536*mckin**7*np.pi**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**7) - (8192*mckin**8*np.pi**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(2835*mbkin**8) -
         (563052544*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/72335025 +
         (563052544*mckin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (8037225*mbkin) - (2252210176*mckin**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(8037225*mbkin**2) +
         (2252210176*mckin**3*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (3444525*mbkin**3) - (1126105088*mckin**4*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(1148175*mbkin**4) +
         (1126105088*mckin**5*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (1148175*mbkin**5) - (2252210176*mckin**6*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(3444525*mbkin**6) +
         (2252210176*mckin**7*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (8037225*mbkin**7) - (563052544*mckin**8*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(8037225*mbkin**8) +
         (563052544*mckin**9*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (72335025*mbkin**9) + (4096*np.pi**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/8505 -
         (4096*mckin*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (945*mbkin) + (16384*mckin**2*np.pi**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(945*mbkin**2) -
         (16384*mckin**3*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (405*mbkin**3) + (8192*mckin**4*np.pi**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(135*mbkin**4) -
         (8192*mckin**5*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) + (16384*mckin**6*np.pi**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(405*mbkin**6) -
         (16384*mckin**7*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (945*mbkin**7) + (4096*mckin**8*np.pi**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(945*mbkin**8) -
         (4096*mckin**9*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (8505*mbkin**9) - (321536*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/229635 +
         (321536*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(25515*mbkin) -
         (1286144*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/(25515*mbkin**2) +
         (1286144*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/(10935*mbkin**3) -
         (643072*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(3645*mbkin**4) +
         (643072*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(3645*mbkin**5) -
         (1286144*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/(10935*mbkin**6) +
         (1286144*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/(25515*mbkin**7) -
         (321536*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(25515*mbkin**8) +
         (321536*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(229635*mbkin**9) -
         (493125632*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/72335025 +
         (986251264*mckin*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (14467005*mbkin) - (493125632*mckin**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(1607445*mbkin**2) +
         (3945005056*mckin**3*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (4822335*mbkin**3) - (986251264*mckin**4*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(688905*mbkin**4) +
         (1972502528*mckin**5*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (1148175*mbkin**5) - (986251264*mckin**6*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(688905*mbkin**6) +
         (3945005056*mckin**7*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (4822335*mbkin**7) - (493125632*mckin**8*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(1607445*mbkin**8) +
         (986251264*mckin**9*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (14467005*mbkin**9) - (493125632*mckin**10*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(72335025*mbkin**10) +
         (2048*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/8505 -
         (4096*mckin*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (1701*mbkin) + (2048*mckin**2*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(189*mbkin**2) -
         (16384*mckin**3*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (567*mbkin**3) + (4096*mckin**4*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(81*mbkin**4) -
         (8192*mckin**5*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) + (4096*mckin**6*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(81*mbkin**6) -
         (16384*mckin**7*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (567*mbkin**7) + (2048*mckin**8*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(189*mbkin**8) -
         (4096*mckin**9*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**9) + (2048*mckin**10*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(8505*mbkin**10) -
         (15503317504*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/2917512675 +
         (15503317504*mckin*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (265228425*mbkin) - (15503317504*mckin**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(53045685*mbkin**2) +
         (15503317504*mckin**3*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (17681895*mbkin**3) - (31006635008*mckin**4*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(17681895*mbkin**4) +
         (31006635008*mckin**5*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (12629925*mbkin**5) - (31006635008*mckin**6*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(12629925*mbkin**6) +
         (31006635008*mckin**7*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (17681895*mbkin**7) - (15503317504*mckin**8*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(17681895*mbkin**8) +
         (15503317504*mckin**9*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (53045685*mbkin**9) - (15503317504*mckin**10*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(265228425*mbkin**10) +
         (15503317504*mckin**11*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (2917512675*mbkin**11) + (19456*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/93555 -
         (19456*mckin*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (8505*mbkin) + (19456*mckin**2*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(1701*mbkin**2) -
         (19456*mckin**3*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (567*mbkin**3) + (38912*mckin**4*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(567*mbkin**4) -
         (38912*mckin**5*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (405*mbkin**5) + (38912*mckin**6*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(405*mbkin**6) -
         (38912*mckin**7*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (567*mbkin**7) + (19456*mckin**8*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(567*mbkin**8) -
         (19456*mckin**9*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**9) + (19456*mckin**10*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(8505*mbkin**10) -
         (19456*mckin**11*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (93555*mbkin**11) - (653824*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/841995 +
         (653824*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(76545*mbkin) -
         (653824*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(15309*mbkin**2) +
         (653824*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(5103*mbkin**3) -
         (1307648*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(5103*mbkin**4) +
         (1307648*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(3645*mbkin**5) -
         (1307648*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(3645*mbkin**6) +
         (1307648*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(5103*mbkin**7) -
         (653824*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(5103*mbkin**8) +
         (653824*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(15309*mbkin**9) -
         (653824*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(76545*mbkin**10) +
         (653824*mckin**11*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(841995*mbkin**11) -
         (42146772224*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/8752538025 +
         (168587088896*mckin*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (2917512675*mbkin) - (84293544448*mckin**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(265228425*mbkin**2) +
         (168587088896*mckin**3*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (159137055*mbkin**3) - (42146772224*mckin**4*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(17681895*mbkin**4) +
         (337174177792*mckin**5*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (88409475*mbkin**5) - (168587088896*mckin**6*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(37889775*mbkin**6) +
         (337174177792*mckin**7*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (88409475*mbkin**7) - (42146772224*mckin**8*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(17681895*mbkin**8) +
         (168587088896*mckin**9*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (159137055*mbkin**9) - (84293544448*mckin**10*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(265228425*mbkin**10) +
         (168587088896*mckin**11*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (2917512675*mbkin**11) - (42146772224*mckin**12*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(8752538025*mbkin**12) +
         (512*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/2673 -
         (2048*mckin*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (891*mbkin) + (1024*mckin**2*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**2) -
         (10240*mckin**3*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (243*mbkin**3) + (2560*mckin**4*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(27*mbkin**4) -
         (4096*mckin**5*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (27*mbkin**5) + (14336*mckin**6*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**6) -
         (4096*mckin**7*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (27*mbkin**7) + (2560*mckin**8*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(27*mbkin**8) -
         (10240*mckin**9*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (243*mbkin**9) + (1024*mckin**10*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**10) -
         (2048*mckin**11*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (891*mbkin**11) + (512*mckin**12*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(2673*mbkin**12) -
         (4757951009792*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          1056556375875 + (4757951009792*mckin*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(81273567375*mbkin) -
         (9515902019584*mckin**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (27091189125*mbkin**2) + (9515902019584*mckin**3*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(7388506125*mbkin**3) -
         (4757951009792*mckin**4*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (1477701225*mbkin**4) + (4757951009792*mckin**5*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(820945125*mbkin**5) -
         (19031804039168*mckin**6*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (2462835375*mbkin**6) + (19031804039168*mckin**7*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(2462835375*mbkin**7) -
         (4757951009792*mckin**8*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (820945125*mbkin**8) + (4757951009792*mckin**9*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(1477701225*mbkin**9) -
         (9515902019584*mckin**10*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (7388506125*mbkin**10) + (9515902019584*mckin**11*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(27091189125*mbkin**11) -
         (4757951009792*mckin**12*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (81273567375*mbkin**12) + (4757951009792*mckin**13*
           (1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(1056556375875*
           mbkin**13) + (2816*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          15795 - (2816*mckin*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (1215*mbkin) + (5632*mckin**2*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(405*mbkin**2) -
         (61952*mckin**3*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (1215*mbkin**3) + (30976*mckin**4*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(243*mbkin**4) -
         (30976*mckin**5*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) + (123904*mckin**6*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(405*mbkin**6) -
         (123904*mckin**7*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (405*mbkin**7) + (30976*mckin**8*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(135*mbkin**8) -
         (30976*mckin**9*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (243*mbkin**9) + (61952*mckin**10*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(1215*mbkin**10) -
         (5632*mckin**11*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (405*mbkin**11) + (2816*mckin**12*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(1215*mbkin**12) -
         (2816*mckin**13*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (15795*mbkin**13) - (19565696*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/23455575 +
         (19565696*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(1804275*mbkin) -
         (39131392*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(601425*mbkin**2) +
         (39131392*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(164025*mbkin**3) -
         (19565696*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(32805*mbkin**4) +
         (19565696*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(18225*mbkin**5) -
         (78262784*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(54675*mbkin**6) +
         (78262784*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(54675*mbkin**7) -
         (19565696*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(18225*mbkin**8) +
         (19565696*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(32805*mbkin**9) -
         (39131392*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(164025*mbkin**10) +
         (39131392*mckin**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(601425*mbkin**11) -
         (19565696*mckin**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(1804275*mbkin**12) +
         (19565696*mckin**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(23455575*mbkin**13) -
         (2091515803904*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          493059642075 + (4183031607808*mckin*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(70437091725*mbkin) -
         (2091515803904*mckin**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (5418237825*mbkin**2) + (8366063215616*mckin**3*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(5418237825*mbkin**3) -
         (2091515803904*mckin**4*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (492567075*mbkin**4) + (4183031607808*mckin**5*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(492567075*mbkin**5) -
         (2091515803904*mckin**6*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (164189025*mbkin**6) + (16732126431232*mckin**7*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(1149323175*mbkin**7) -
         (2091515803904*mckin**8*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (164189025*mbkin**8) + (4183031607808*mckin**9*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(492567075*mbkin**9) -
         (2091515803904*mckin**10*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (492567075*mbkin**10) + (8366063215616*mckin**11*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(5418237825*mbkin**11) -
         (2091515803904*mckin**12*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (5418237825*mbkin**12) + (4183031607808*mckin**13*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(70437091725*mbkin**13) -
         (2091515803904*mckin**14*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (493059642075*mbkin**14) + (202624*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/1216215 -
         (405248*mckin*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (173745*mbkin) + (202624*mckin**2*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(13365*mbkin**2) -
         (810496*mckin**3*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (13365*mbkin**3) + (202624*mckin**4*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(1215*mbkin**4) -
         (405248*mckin**5*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (1215*mbkin**5) + (202624*mckin**6*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(405*mbkin**6) -
         (1620992*mckin**7*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**7) + (202624*mckin**8*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(405*mbkin**8) -
         (405248*mckin**9*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (1215*mbkin**9) + (202624*mckin**10*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(1215*mbkin**10) -
         (810496*mckin**11*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (13365*mbkin**11) + (202624*mckin**12*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(13365*mbkin**12) -
         (405248*mckin**13*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (173745*mbkin**13) + (202624*mckin**14*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(1216215*mbkin**14) -
         (29636777650696*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          7395894631125 + (29636777650696*mckin*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(493059642075*mbkin) -
         (29636777650696*mckin**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (70437091725*mbkin**2) + (29636777650696*mckin**3*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(16254713475*mbkin**3) -
         (29636777650696*mckin**4*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (5418237825*mbkin**4) + (29636777650696*mckin**5*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(2462835375*mbkin**5) -
         (29636777650696*mckin**6*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (1477701225*mbkin**6) + (29636777650696*mckin**7*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(1149323175*mbkin**7) -
         (29636777650696*mckin**8*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (1149323175*mbkin**8) + (29636777650696*mckin**9*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(1477701225*mbkin**9) -
         (29636777650696*mckin**10*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (2462835375*mbkin**10) + (29636777650696*mckin**11*
           (1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(5418237825*mbkin**11) -
         (29636777650696*mckin**12*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (16254713475*mbkin**12) + (29636777650696*mckin**13*
           (1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(70437091725*
           mbkin**13) - (29636777650696*mckin**14*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(493059642075*mbkin**14) +
         (29636777650696*mckin**15*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (7395894631125*mbkin**15) + (27136*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/173745 -
         (27136*mckin*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (11583*mbkin) + (189952*mckin**2*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(11583*mbkin**2) -
         (189952*mckin**3*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**3) + (189952*mckin**4*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(891*mbkin**4) -
         (189952*mckin**5*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (405*mbkin**5) + (189952*mckin**6*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(243*mbkin**6) -
         (27136*mckin**7*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (27*mbkin**7) + (27136*mckin**8*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(27*mbkin**8) -
         (189952*mckin**9*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (243*mbkin**9) + (189952*mckin**10*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(405*mbkin**10) -
         (189952*mckin**11*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (891*mbkin**11) + (189952*mckin**12*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(2673*mbkin**12) -
         (189952*mckin**13*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (11583*mbkin**13) + (27136*mckin**14*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(11583*mbkin**14) -
         (27136*mckin**15*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (173745*mbkin**15) - (139287296*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/164189025 +
         (139287296*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(10945935*mbkin) -
         (139287296*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(1563705*mbkin**2) +
         (139287296*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(360855*mbkin**3) -
         (139287296*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(120285*mbkin**4) +
         (139287296*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(54675*mbkin**5) -
         (139287296*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(32805*mbkin**6) +
         (139287296*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(25515*mbkin**7) -
         (139287296*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(25515*mbkin**8) +
         (139287296*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(32805*mbkin**9) -
         (139287296*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(54675*mbkin**10) +
         (139287296*mckin**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(120285*mbkin**11) -
         (139287296*mckin**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(360855*mbkin**12) +
         (139287296*mckin**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(1563705*mbkin**13) -
         (139287296*mckin**14*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(10945935*mbkin**14) +
         (139287296*mckin**15*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(164189025*mbkin**15) +
         (173*(-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin))))/27434890345622760 -
         (np.pi**2*(-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin))))/4572481724270460 +
         (-9.854113931163779e19*(1 - mckin/mbkin)**5 - 2.5903365411196256e20*
            (1 - mckin/mbkin)**6 - 1.5679296844106574e21*(1 - mckin/mbkin)**
             7 + 1.602586733095746e21*(1 - mckin/mbkin)**8 -
           3.731382058497059e20*(1 - mckin/mbkin)**9 - 1.442450583699921e20*
            (1 - mckin/mbkin)**10 - 1.0117940133228708e21*(1 - mckin/mbkin)**
             11 - 1.4567466646450717e21*(1 - mckin/mbkin)**12 -
           2.2050041791842524e21*(1 - mckin/mbkin)**13 - 3.10373436948258e21*
            (1 - mckin/mbkin)**14 - 4.17958779756945e21*(1 - mckin/mbkin)**
             15 - 58963096098466560000*(1 - mckin/mbkin)**5*np.pi**2 +
           88444644147699840000*(1 - mckin/mbkin)**6*np.pi**2 -
           65172716793401817600*(1 - mckin/mbkin)**7*np.pi**2 -
           9108973844357587200*(1 - mckin/mbkin)**8*np.pi**2 +
           27694301930794699200*(1 - mckin/mbkin)**9*np.pi**2 +
           9354724595990373600*(1 - mckin/mbkin)**10*np.pi**2 +
           99846396491298256800*(1 - mckin/mbkin)**11*np.pi**2 +
           145282820224320979200*(1 - mckin/mbkin)**12*np.pi**2 +
           221188319752082032350*(1 - mckin/mbkin)**13*np.pi**2 +
           312313023387573082875*(1 - mckin/mbkin)**14*np.pi**2 +
           421384671784085884875*(1 - mckin/mbkin)**15*np.pi**2 +
           23585238439386624000*(1 - mckin/mbkin)**5*np.pi**2*np.log(2) -
           35377857659079936000*(1 - mckin/mbkin)**6*np.pi**2*np.log(2) +
           17408152181452032000*(1 - mckin/mbkin)**7*np.pi**2*np.log(2) -
           2807766480879360000*(1 - mckin/mbkin)**8*np.pi**2*np.log(2) +
           1380485186432352000*(1 - mckin/mbkin)**9*np.pi**2*np.log(2) +
           690242593216176000*(1 - mckin/mbkin)**10*np.pi**2*np.log(2) +
           508375900704672000*(1 - mckin/mbkin)**11*np.pi**2*np.log(2) +
           417442554448920000*(1 - mckin/mbkin)**12*np.pi**2*np.log(2) +
           359888250273552000*(1 - mckin/mbkin)**13*np.pi**2*np.log(2) +
           319023467138376000*(1 - mckin/mbkin)**14*np.pi**2*np.log(2) +
           287976049320960000*(1 - mckin/mbkin)**15*np.pi**2*np.log(2) +
           2156493012354474393600*(1 - mckin/mbkin)**7*(np.log(2) +
             np.log(1 - mckin/mbkin)) - 1114185917132493004800*
            (1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)) +
           72609864602488012800*(1 - mckin/mbkin)**9*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 63592262841345638400*(1 - mckin/mbkin)**
             10*(np.log(2) + np.log(1 - mckin/mbkin)) + 49568665585965926400*
            (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)) +
           44918542055279692800*(1 - mckin/mbkin)**12*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 42007093034271805440*(1 - mckin/mbkin)**
             13*(np.log(2) + np.log(1 - mckin/mbkin)) + 39569178342479385600*
            (1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)) +
           37379682179716836960*(1 - mckin/mbkin)**15*(np.log(2) +
             np.log(1 - mckin/mbkin)) - 53909116432883712000*(1 - mckin/mbkin)**7*
            np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) + 26954558216441856000*
            (1 - mckin/mbkin)**8*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
           4492426369406976000*(1 - mckin/mbkin)**9*np.pi**2*
            (np.log(2) + np.log(1 - mckin/mbkin)) - 2246213184703488000*
            (1 - mckin/mbkin)**10*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
           1939911386789376000*(1 - mckin/mbkin)**11*np.pi**2*
            (np.log(2) + np.log(1 - mckin/mbkin)) - 1786760487832320000*
            (1 - mckin/mbkin)**12*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
           1663061684828544000*(1 - mckin/mbkin)**13*np.pi**2*
            (np.log(2) + np.log(1 - mckin/mbkin)) - 1554088929801408000*
            (1 - mckin/mbkin)**14*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
           1456897013155584000*(1 - mckin/mbkin)**15*np.pi**2*
            (np.log(2) + np.log(1 - mckin/mbkin)) - 242591023947976704000*
            (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
           121295511973988352000*(1 - mckin/mbkin)**8*
            (np.log(2) + np.log(1 - mckin/mbkin))**2 + 13061313703646208000*
            (1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
           6530656851823104000*(1 - mckin/mbkin)**10*
            (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7243470295117056000*
            (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
           7599877016764032000*(1 - mckin/mbkin)**12*
            (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7781163757509542400*
            (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
           7874890507804320000*(1 - mckin/mbkin)**14*
            (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7913392760156083200*
            (1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin))**2)/
          1865629212490543500 + (37184*mckin*np.log(2/mus))/(9*mbkin) -
         (37184*mckin**2*np.log(2/mus))/(9*mbkin**2) + (18592*mckin**3*np.log(2/mus))/
          (3*mbkin**3) - (18592*mckin**4*np.log(2/mus))/(3*mbkin**4) -
         (37184*mckin**5*np.log(2/mus))/(3*mbkin**5) + (37184*mckin**6*np.log(2/mus))/
          (3*mbkin**6) + (18592*mckin**7*np.log(2/mus))/(9*mbkin**7) -
         (18592*mckin**8*np.log(2/mus))/(9*mbkin**8) - 512*(1 - mckin/mbkin)**4*
          np.log(2/mus) + (512*mckin*(1 - mckin/mbkin)**4*np.log(2/mus))/mbkin +
         (4608*(1 - mckin/mbkin)**5*np.log(2/mus))/5 -
         (4608*mckin*(1 - mckin/mbkin)**5*np.log(2/mus))/(5*mbkin) -
         (687872*(1 - mckin/mbkin)**6*np.log(2/mus))/675 +
         (687872*mckin*(1 - mckin/mbkin)**6*np.log(2/mus))/(675*mbkin) +
         (19490816*(1 - mckin/mbkin)**7*np.log(2/mus))/33075 -
         (19490816*mckin*(1 - mckin/mbkin)**7*np.log(2/mus))/(33075*mbkin) -
         (1288736*(1 - mckin/mbkin)**8*np.log(2/mus))/11025 +
         (1288736*mckin*(1 - mckin/mbkin)**8*np.log(2/mus))/(11025*mbkin) -
         (1300064*(1 - mckin/mbkin)**9*np.log(2/mus))/59535 +
         (1300064*mckin*(1 - mckin/mbkin)**9*np.log(2/mus))/(59535*mbkin) -
         (129376*(1 - mckin/mbkin)**10*np.log(2/mus))/8505 +
         (129376*mckin*(1 - mckin/mbkin)**10*np.log(2/mus))/(8505*mbkin) -
         (33701216*(1 - mckin/mbkin)**11*np.log(2/mus))/2401245 +
         (33701216*mckin*(1 - mckin/mbkin)**11*np.log(2/mus))/(2401245*mbkin) -
         (1281536*(1 - mckin/mbkin)**12*np.log(2/mus))/93555 +
         (1281536*mckin*(1 - mckin/mbkin)**12*np.log(2/mus))/(93555*mbkin) -
         (337749248*(1 - mckin/mbkin)**13*np.log(2/mus))/24845535 +
         (337749248*mckin*(1 - mckin/mbkin)**13*np.log(2/mus))/(24845535*mbkin) -
         (612363512*(1 - mckin/mbkin)**14*np.log(2/mus))/45090045 +
         (612363512*mckin*(1 - mckin/mbkin)**14*np.log(2/mus))/(45090045*mbkin) -
         (3945000128*(1 - mckin/mbkin)**15*np.log(2/mus))/289864575 +
         (3945000128*mckin*(1 - mckin/mbkin)**15*np.log(2/mus))/
          (289864575*mbkin) - (3959187262*(1 - mckin/mbkin)**16*np.log(2/mus))/
          289864575 + (3959187262*mckin*(1 - mckin/mbkin)**16*np.log(2/mus))/
          (289864575*mbkin) - (2681128006982*(1 - mckin/mbkin)**17*np.log(2/mus))/
          195465345075 + (2681128006982*mckin*(1 - mckin/mbkin)**17*
           np.log(2/mus))/(195465345075*mbkin) -
         (48474483141979*(1 - mckin/mbkin)**18*np.log(2/mus))/3518376211350 +
         (48474483141979*mckin*(1 - mckin/mbkin)**18*np.log(2/mus))/
          (3518376211350*mbkin) - (22826272832761*(1 - mckin/mbkin)**19*
           np.log(2/mus))/1649524431555 + (22826272832761*mckin*
           (1 - mckin/mbkin)**19*np.log(2/mus))/(1649524431555*mbkin) -
         (128*mckin*np.pi**2*np.log(2/mus))/mbkin + (128*mckin**2*np.pi**2*np.log(2/mus))/
          mbkin**2 - (192*mckin**3*np.pi**2*np.log(2/mus))/mbkin**3 +
         (192*mckin**4*np.pi**2*np.log(2/mus))/mbkin**4 +
         (384*mckin**5*np.pi**2*np.log(2/mus))/mbkin**5 -
         (384*mckin**6*np.pi**2*np.log(2/mus))/mbkin**6 - (64*mckin**7*np.pi**2*np.log(2/mus))/
          mbkin**7 + (64*mckin**8*np.pi**2*np.log(2/mus))/mbkin**8 +
         (37184*mckin**3*np.log(mckin**2/mbkin**2)*np.log(2/mus))/(3*mbkin**3) -
         (37184*mckin**4*np.log(mckin**2/mbkin**2)*np.log(2/mus))/(3*mbkin**4) -
         (384*mckin**3*np.pi**2*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**3 +
         (384*mckin**4*np.pi**2*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**4 -
         (11620*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(2/mus))/9 + 40*np.pi**2*(1 - (8*mckin**2)/mbkin**2 +
           (8*mckin**6)/mbkin**6 - mckin**8/mbkin**8 -
           (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*np.log(2/mus) +
         (16384*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/45 - (16384*mckin*(1 - mckin/mbkin)**6*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(45*mbkin) -
         (65536*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/315 + (65536*mckin*(1 - mckin/mbkin)**7*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(315*mbkin) +
         (4096*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/105 - (4096*mckin*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(105*mbkin) +
         (4096*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/189 - (4096*mckin*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(189*mbkin) +
         (19456*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/945 - (19456*mckin*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(945*mbkin) +
         (2048*(1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/99 - (2048*mckin*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(99*mbkin) +
         (2816*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/135 - (2816*mckin*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(135*mbkin) +
         (405248*(1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/19305 - (405248*mckin*(1 - mckin/mbkin)**13*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(19305*mbkin) +
         (27136*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/1287 - (27136*mckin*(1 - mckin/mbkin)**14*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(1287*mbkin) +
         (2857984*(1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/135135 - (2857984*mckin*(1 - mckin/mbkin)**15*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(135135*mbkin) +
         (2864096*(1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/135135 - (2864096*mckin*(1 - mckin/mbkin)**16*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(135135*mbkin) +
         (5418208*(1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/255255 - (5418208*mckin*(1 - mckin/mbkin)**17*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(255255*mbkin) +
         (6973984*(1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/328185 - (6973984*mckin*(1 - mckin/mbkin)**18*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(328185*mbkin) +
         (2411168*(1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/113373 - (2411168*mckin*(1 - mckin/mbkin)**19*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(113373*mbkin) -
         ((-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin)))*np.log(2/mus))/508053524918940 -
         (576*mckin*np.log(2/mus)**2)/mbkin + (576*mckin**2*np.log(2/mus)**2)/
          mbkin**2 - (864*mckin**3*np.log(2/mus)**2)/mbkin**3 +
         (864*mckin**4*np.log(2/mus)**2)/mbkin**4 + (1728*mckin**5*np.log(2/mus)**2)/
          mbkin**5 - (1728*mckin**6*np.log(2/mus)**2)/mbkin**6 -
         (288*mckin**7*np.log(2/mus)**2)/mbkin**7 + (288*mckin**8*np.log(2/mus)**2)/
          mbkin**8 - (1728*mckin**3*np.log(mckin**2/mbkin**2)*np.log(2/mus)**2)/
          mbkin**3 + (1728*mckin**4*np.log(mckin**2/mbkin**2)*np.log(2/mus)**2)/
          mbkin**4 + 180*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
           mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
          np.log(2/mus)**2 + (13246746552638045*np.log(mus**2/mbkin**2))/
          83136031350372 - (379910484789952015*mckin*np.log(mus**2/mbkin**2))/
          (249408094051116*mbkin) + (1310350319989205843*mckin**2*
           np.log(mus**2/mbkin**2))/(144394159713804*mbkin**2) -
         (652389962242240745*mckin**3*np.log(mus**2/mbkin**2))/
          (16043795523756*mbkin**3) + (53904637677142363*mckin**4*
           np.log(mus**2/mbkin**2))/(386080640946*mbkin**4) -
         (795267650907243595*mckin**5*np.log(mus**2/mbkin**2))/
          (2123443525203*mbkin**5) + (351792707890865*mckin**6*
           np.log(mus**2/mbkin**2))/(437733153*mbkin**6) -
         (8361639403698197*mckin**7*np.log(mus**2/mbkin**2))/(5948021079*mbkin**7) +
         (2428174474130441*mckin**8*np.log(mus**2/mbkin**2))/(1196643294*mbkin**8) -
         (1853425098362713*mckin**9*np.log(mus**2/mbkin**2))/(761500278*mbkin**9) +
         (72269187273480035*mckin**10*np.log(mus**2/mbkin**2))/
          (29698510842*mbkin**10) - (60219306156529265*mckin**11*
           np.log(mus**2/mbkin**2))/(29698510842*mbkin**11) +
         (772013715067132*mckin**12*np.log(mus**2/mbkin**2))/(549972423*mbkin**12) -
         (34760430696023047*mckin**13*np.log(mus**2/mbkin**2))/
          (43335582147*mbkin**13) + (46756234265009543*mckin**14*
           np.log(mus**2/mbkin**2))/(124908442659*mbkin**14) -
         (3010845012939655*mckin**15*np.log(mus**2/mbkin**2))/
          (21448924497*mbkin**15) + (38964401383916935*mckin**16*
           np.log(mus**2/mbkin**2))/(943752677868*mbkin**16) -
         (1911703888198739*mckin**17*np.log(mus**2/mbkin**2))/
          (208360980828*mbkin**17) + (69728083745549401*mckin**18*
           np.log(mus**2/mbkin**2))/(48131386571268*mbkin**18) -
         (397457102548235579*mckin**19*np.log(mus**2/mbkin**2))/
          (2743489034562276*mbkin**19) + (122901707465525*mckin**20*
           np.log(mus**2/mbkin**2))/(17814863860794*mbkin**20) -
         (40960*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          1701 + (40960*mckin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin) -
         (40960*mckin**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**2) +
         (204800*mckin**3*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**3) -
         (204800*mckin**4*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**4) +
         (40960*mckin**5*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) -
         (40960*mckin**6*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**6) +
         (40960*mckin**7*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**7) +
         (20480*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          1701 - (163840*mckin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin) +
         (81920*mckin**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**2) -
         (163840*mckin**3*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**3) +
         (204800*mckin**4*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**4) -
         (163840*mckin**5*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**5) +
         (81920*mckin**6*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**6) -
         (163840*mckin**7*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**7) +
         (20480*mckin**8*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**8) -
         (10240*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          5103 + (10240*mckin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin) -
         (40960*mckin**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**2) +
         (40960*mckin**3*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**3) -
         (20480*mckin**4*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**4) +
         (20480*mckin**5*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) -
         (40960*mckin**6*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**6) +
         (40960*mckin**7*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**7) -
         (10240*mckin**8*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**8) +
         (10240*mckin**9*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**9) -
         (5120*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          5103 + (51200*mckin*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin) -
         (25600*mckin**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**2) +
         (204800*mckin**3*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**3) -
         (51200*mckin**4*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**4) +
         (20480*mckin**5*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) -
         (51200*mckin**6*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**6) +
         (204800*mckin**7*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**7) -
         (25600*mckin**8*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**8) +
         (51200*mckin**9*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**9) -
         (5120*mckin**10*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**10) -
         (48640*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          56133 + (48640*mckin*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin) -
         (243200*mckin**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**2) +
         (243200*mckin**3*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**3) -
         (486400*mckin**4*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**4) +
         (97280*mckin**5*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**5) -
         (97280*mckin**6*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**6) +
         (486400*mckin**7*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**7) -
         (243200*mckin**8*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**8) +
         (243200*mckin**9*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**9) -
         (48640*mckin**10*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**10) +
         (48640*mckin**11*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(56133*mbkin**11) -
         (6400*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          8019 + (25600*mckin*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin) -
         (12800*mckin**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**2) +
         (128000*mckin**3*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**3) -
         (32000*mckin**4*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**4) +
         (51200*mckin**5*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) -
         (179200*mckin**6*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**6) +
         (51200*mckin**7*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**7) -
         (32000*mckin**8*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**8) +
         (128000*mckin**9*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**9) -
         (12800*mckin**10*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**10) +
         (25600*mckin**11*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**11) -
         (6400*mckin**12*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**12) -
         (7040*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          9477 + (7040*mckin*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin) -
         (14080*mckin**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**2) +
         (154880*mckin**3*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**3) -
         (387200*mckin**4*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**4) +
         (77440*mckin**5*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) -
         (309760*mckin**6*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**6) +
         (309760*mckin**7*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**7) -
         (77440*mckin**8*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**8) +
         (387200*mckin**9*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**9) -
         (154880*mckin**10*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**10) +
         (14080*mckin**11*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**11) -
         (7040*mckin**12*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**12) +
         (7040*mckin**13*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(9477*mbkin**13) -
         (506560*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/729729 + (1013120*mckin*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(104247*mbkin) -
         (506560*mckin**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**2) +
         (2026240*mckin**3*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**3) -
         (506560*mckin**4*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**4) +
         (1013120*mckin**5*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**5) -
         (506560*mckin**6*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**6) +
         (4052480*mckin**7*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**7) -
         (506560*mckin**8*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**8) +
         (1013120*mckin**9*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**9) -
         (506560*mckin**10*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**10) +
         (2026240*mckin**11*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**11) -
         (506560*mckin**12*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**12) +
         (1013120*mckin**13*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(104247*mbkin**13) -
         (506560*mckin**14*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729729*mbkin**14) -
         (67840*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          104247 + (339200*mckin*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin) -
         (2374400*mckin**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**2) +
         (2374400*mckin**3*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**3) -
         (2374400*mckin**4*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**4) +
         (474880*mckin**5*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**5) -
         (2374400*mckin**6*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**6) +
         (339200*mckin**7*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**7) -
         (339200*mckin**8*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**8) +
         (2374400*mckin**9*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**9) -
         (474880*mckin**10*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**10) +
         (2374400*mckin**11*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**11) -
         (2374400*mckin**12*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**12) +
         (2374400*mckin**13*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**13) -
         (339200*mckin**14*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**14) +
         (67840*mckin**15*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(104247*mbkin**15) -
         (446560*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/729729 + (7144960*mckin*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(729729*mbkin) -
         (17862400*mckin**2*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**2) +
         (35724800*mckin**3*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(104247*mbkin**3) -
         (8931200*mckin**4*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**4) +
         (7144960*mckin**5*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**5) -
         (3572480*mckin**6*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**6) +
         (35724800*mckin**7*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**7) -
         (4465600*mckin**8*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**8) +
         (35724800*mckin**9*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**9) -
         (3572480*mckin**10*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**10) +
         (7144960*mckin**11*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**11) -
         (8931200*mckin**12*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**12) +
         (35724800*mckin**13*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(104247*mbkin**13) -
         (17862400*mckin**14*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**14) +
         (7144960*mckin**15*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729729*mbkin**15) -
         (446560*mckin**16*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729729*mbkin**16) -
         (7160240*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/12405393 +
         (7160240*mckin*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729729*mbkin) -
         (57281920*mckin**2*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729729*mbkin**2) +
         (286409600*mckin**3*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729729*mbkin**3) -
         (143204800*mckin**4*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(104247*mbkin**4) +
         (28640960*mckin**5*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**5) -
         (57281920*mckin**6*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**6) +
         (57281920*mckin**7*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**7) -
         (71602400*mckin**8*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**8) +
         (71602400*mckin**9*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**9) -
         (57281920*mckin**10*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**10) +
         (57281920*mckin**11*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**11) -
         (28640960*mckin**12*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(8019*mbkin**12) +
         (143204800*mckin**13*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(104247*mbkin**13) -
         (286409600*mckin**14*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729729*mbkin**14) +
         (57281920*mckin**15*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729729*mbkin**15) -
         (7160240*mckin**16*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729729*mbkin**16) +
         (7160240*mckin**17*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(12405393*mbkin**17) -
         (6772760*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/12405393 + (13545520*mckin*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(1378377*mbkin) -
         (6772760*mckin**2*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81081*mbkin**2) +
         (108364160*mckin**3*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**3) -
         (135455200*mckin**4*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81081*mbkin**4) +
         (54182080*mckin**5*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11583*mbkin**5) -
         (27091040*mckin**6*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**6) +
         (108364160*mckin**7*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(6237*mbkin**7) -
         (13545520*mckin**8*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**8) +
         (135455200*mckin**9*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(5103*mbkin**9) -
         (13545520*mckin**10*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**10) +
         (108364160*mckin**11*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(6237*mbkin**11) -
         (27091040*mckin**12*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**12) +
         (54182080*mckin**13*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11583*mbkin**13) -
         (135455200*mckin**14*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81081*mbkin**14) +
         (108364160*mckin**15*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**15) -
         (6772760*mckin**16*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81081*mbkin**16) +
         (13545520*mckin**17*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1378377*mbkin**17) -
         (6772760*mckin**18*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(12405393*mbkin**18) -
         (17434960*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/33671781 + (17434960*mckin*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(1772199*mbkin) -
         (17434960*mckin**2*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(196911*mbkin**2) +
         (17434960*mckin**3*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**3) -
         (69739840*mckin**4*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**4) +
         (69739840*mckin**5*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11583*mbkin**5) -
         (488178880*mckin**6*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**6) +
         (69739840*mckin**7*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**7) -
         (34869920*mckin**8*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(891*mbkin**8) +
         (34869920*mckin**9*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**9) -
         (34869920*mckin**10*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**10) +
         (34869920*mckin**11*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(891*mbkin**11) -
         (69739840*mckin**12*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**12) +
         (488178880*mckin**13*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**13) -
         (69739840*mckin**14*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11583*mbkin**14) +
         (69739840*mckin**15*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**15) -
         (17434960*mckin**16*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**16) +
         (17434960*mckin**17*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(196911*mbkin**17) -
         (17434960*mckin**18*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1772199*mbkin**18) +
         (17434960*mckin**19*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(33671781*mbkin**19) -
         (1506980*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/3061071 + (30139600*mckin*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(3061071*mbkin) -
         (15069800*mckin**2*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(161109*mbkin**2) +
         (30139600*mckin**3*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(53703*mbkin**3) -
         (7534900*mckin**4*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin**4) +
         (24111680*mckin**5*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin**5) -
         (60279200*mckin**6*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin**6) +
         (120558400*mckin**7*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin**7) -
         (15069800*mckin**8*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**8) +
         (60279200*mckin**9*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**9) -
         (66307120*mckin**10*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**10) +
         (60279200*mckin**11*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(729*mbkin**11) -
         (15069800*mckin**12*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**12) +
         (120558400*mckin**13*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin**13) -
         (60279200*mckin**14*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin**14) +
         (24111680*mckin**15*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin**15) -
         (7534900*mckin**16*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin**16) +
         (30139600*mckin**17*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(53703*mbkin**17) -
         (15069800*mckin**18*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(161109*mbkin**18) +
         (30139600*mckin**19*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3061071*mbkin**19) -
         (1506980*mckin**20*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3061071*mbkin**20) +
         (5*(-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin)))*np.log(mus**2/mbkin**2))/5486978069124552 +
         (1504*mckin*np.log(mus**2/mckin**2))/(9*mbkin) -
         (1504*mckin**2*np.log(mus**2/mckin**2))/(9*mbkin**2) +
         (752*mckin**3*np.log(mus**2/mckin**2))/(3*mbkin**3) -
         (752*mckin**4*np.log(mus**2/mckin**2))/(3*mbkin**4) -
         (1504*mckin**5*np.log(mus**2/mckin**2))/(3*mbkin**5) +
         (1504*mckin**6*np.log(mus**2/mckin**2))/(3*mbkin**6) +
         (752*mckin**7*np.log(mus**2/mckin**2))/(9*mbkin**7) -
         (752*mckin**8*np.log(mus**2/mckin**2))/(9*mbkin**8) -
         (512*(1 - mckin/mbkin)**4*np.log(mus**2/mckin**2))/27 +
         (512*mckin*(1 - mckin/mbkin)**4*np.log(mus**2/mckin**2))/(27*mbkin) +
         (512*(1 - mckin/mbkin)**5*np.log(mus**2/mckin**2))/15 -
         (512*mckin*(1 - mckin/mbkin)**5*np.log(mus**2/mckin**2))/(15*mbkin) -
         (687872*(1 - mckin/mbkin)**6*np.log(mus**2/mckin**2))/18225 +
         (687872*mckin*(1 - mckin/mbkin)**6*np.log(mus**2/mckin**2))/
          (18225*mbkin) + (19490816*(1 - mckin/mbkin)**7*np.log(mus**2/mckin**2))/
          893025 - (19490816*mckin*(1 - mckin/mbkin)**7*np.log(mus**2/mckin**2))/
          (893025*mbkin) - (1288736*(1 - mckin/mbkin)**8*np.log(mus**2/mckin**2))/
          297675 + (1288736*mckin*(1 - mckin/mbkin)**8*np.log(mus**2/mckin**2))/
          (297675*mbkin) - (1300064*(1 - mckin/mbkin)**9*np.log(mus**2/mckin**2))/
          1607445 + (1300064*mckin*(1 - mckin/mbkin)**9*np.log(mus**2/mckin**2))/
          (1607445*mbkin) - (129376*(1 - mckin/mbkin)**10*np.log(mus**2/mckin**2))/
          229635 + (129376*mckin*(1 - mckin/mbkin)**10*np.log(mus**2/mckin**2))/
          (229635*mbkin) - (33701216*(1 - mckin/mbkin)**11*np.log(mus**2/mckin**2))/
          64833615 + (33701216*mckin*(1 - mckin/mbkin)**11*np.log(mus**2/mckin**2))/
          (64833615*mbkin) - (1281536*(1 - mckin/mbkin)**12*
           np.log(mus**2/mckin**2))/2525985 + (1281536*mckin*(1 - mckin/mbkin)**12*
           np.log(mus**2/mckin**2))/(2525985*mbkin) -
         (337749248*(1 - mckin/mbkin)**13*np.log(mus**2/mckin**2))/670829445 +
         (337749248*mckin*(1 - mckin/mbkin)**13*np.log(mus**2/mckin**2))/
          (670829445*mbkin) - (612363512*(1 - mckin/mbkin)**14*
           np.log(mus**2/mckin**2))/1217431215 +
         (612363512*mckin*(1 - mckin/mbkin)**14*np.log(mus**2/mckin**2))/
          (1217431215*mbkin) - (3945000128*(1 - mckin/mbkin)**15*
           np.log(mus**2/mckin**2))/7826343525 + (3945000128*mckin*
           (1 - mckin/mbkin)**15*np.log(mus**2/mckin**2))/(7826343525*mbkin) -
         (3959187262*(1 - mckin/mbkin)**16*np.log(mus**2/mckin**2))/7826343525 +
         (3959187262*mckin*(1 - mckin/mbkin)**16*np.log(mus**2/mckin**2))/
          (7826343525*mbkin) - (2681128006982*(1 - mckin/mbkin)**17*
           np.log(mus**2/mckin**2))/5277564317025 +
         (2681128006982*mckin*(1 - mckin/mbkin)**17*np.log(mus**2/mckin**2))/
          (5277564317025*mbkin) - (48474483141979*(1 - mckin/mbkin)**18*
           np.log(mus**2/mckin**2))/94996157706450 +
         (48474483141979*mckin*(1 - mckin/mbkin)**18*np.log(mus**2/mckin**2))/
          (94996157706450*mbkin) - (22826272832761*(1 - mckin/mbkin)**19*
           np.log(mus**2/mckin**2))/44537159651985 +
         (22826272832761*mckin*(1 - mckin/mbkin)**19*np.log(mus**2/mckin**2))/
          (44537159651985*mbkin) - (128*mckin*np.pi**2*np.log(mus**2/mckin**2))/
          (27*mbkin) + (128*mckin**2*np.pi**2*np.log(mus**2/mckin**2))/(27*mbkin**2) -
         (64*mckin**3*np.pi**2*np.log(mus**2/mckin**2))/(9*mbkin**3) +
         (64*mckin**4*np.pi**2*np.log(mus**2/mckin**2))/(9*mbkin**4) +
         (128*mckin**5*np.pi**2*np.log(mus**2/mckin**2))/(9*mbkin**5) -
         (128*mckin**6*np.pi**2*np.log(mus**2/mckin**2))/(9*mbkin**6) -
         (64*mckin**7*np.pi**2*np.log(mus**2/mckin**2))/(27*mbkin**7) +
         (64*mckin**8*np.pi**2*np.log(mus**2/mckin**2))/(27*mbkin**8) +
         (1504*mckin**3*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/(3*mbkin**3) -
         (1504*mckin**4*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/(3*mbkin**4) -
         (128*mckin**3*np.pi**2*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/
          (9*mbkin**3) + (128*mckin**4*np.pi**2*np.log(mckin**2/mbkin**2)*
           np.log(mus**2/mckin**2))/(9*mbkin**4) -
         (470*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(mus**2/mckin**2))/9 + (40*np.pi**2*(1 - (8*mckin**2)/mbkin**2 +
            (8*mckin**6)/mbkin**6 - mckin**8/mbkin**8 -
            (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*np.log(mus**2/mckin**2))/
          27 + (16384*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/1215 - (16384*mckin*(1 - mckin/mbkin)**6*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/(1215*mbkin) -
         (65536*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/8505 + (65536*mckin*(1 - mckin/mbkin)**7*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/(8505*mbkin) +
         (4096*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/2835 - (4096*mckin*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/(2835*mbkin) +
         (4096*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/5103 - (4096*mckin*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/(5103*mbkin) +
         (19456*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/25515 - (19456*mckin*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/
          (25515*mbkin) + (2048*(1 - mckin/mbkin)**11*(np.log(2) +
            np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/2673 -
         (2048*mckin*(1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(2673*mbkin) + (2816*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/3645 -
         (2816*mckin*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(3645*mbkin) + (405248*(1 - mckin/mbkin)**13*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/521235 -
         (405248*mckin*(1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(521235*mbkin) + (27136*(1 - mckin/mbkin)**14*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/34749 -
         (27136*mckin*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(34749*mbkin) + (2857984*(1 - mckin/mbkin)**15*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/3648645 -
         (2857984*mckin*(1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(3648645*mbkin) +
         (2864096*(1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/3648645 - (2864096*mckin*(1 - mckin/mbkin)**16*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/
          (3648645*mbkin) + (5418208*(1 - mckin/mbkin)**17*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/6891885 -
         (5418208*mckin*(1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(6891885*mbkin) +
         (6973984*(1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/8860995 - (6973984*mckin*(1 - mckin/mbkin)**18*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/
          (8860995*mbkin) + (2411168*(1 - mckin/mbkin)**19*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/3061071 -
         (2411168*mckin*(1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(3061071*mbkin) -
         ((-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin)))*np.log(mus**2/mckin**2))/13717445172811380 -
         (128*mckin*np.log(2/mus)*np.log(mus**2/mckin**2))/(3*mbkin) +
         (128*mckin**2*np.log(2/mus)*np.log(mus**2/mckin**2))/(3*mbkin**2) -
         (64*mckin**3*np.log(2/mus)*np.log(mus**2/mckin**2))/mbkin**3 +
         (64*mckin**4*np.log(2/mus)*np.log(mus**2/mckin**2))/mbkin**4 +
         (128*mckin**5*np.log(2/mus)*np.log(mus**2/mckin**2))/mbkin**5 -
         (128*mckin**6*np.log(2/mus)*np.log(mus**2/mckin**2))/mbkin**6 -
         (64*mckin**7*np.log(2/mus)*np.log(mus**2/mckin**2))/(3*mbkin**7) +
         (64*mckin**8*np.log(2/mus)*np.log(mus**2/mckin**2))/(3*mbkin**8) -
         (128*mckin**3*np.log(mckin**2/mbkin**2)*np.log(2/mus)*np.log(mus**2/mckin**2))/
          mbkin**3 + (128*mckin**4*np.log(mckin**2/mbkin**2)*np.log(2/mus)*
           np.log(mus**2/mckin**2))/mbkin**4 + (40*(1 - (8*mckin**2)/mbkin**2 +
            (8*mckin**6)/mbkin**6 - mckin**8/mbkin**8 -
            (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*np.log(2/mus)*
           np.log(mus**2/mckin**2))/3 - (64*mckin*np.log(mus**2/mckin**2)**2)/
          (81*mbkin) + (64*mckin**2*np.log(mus**2/mckin**2)**2)/(81*mbkin**2) -
         (32*mckin**3*np.log(mus**2/mckin**2)**2)/(27*mbkin**3) +
         (32*mckin**4*np.log(mus**2/mckin**2)**2)/(27*mbkin**4) +
         (64*mckin**5*np.log(mus**2/mckin**2)**2)/(27*mbkin**5) -
         (64*mckin**6*np.log(mus**2/mckin**2)**2)/(27*mbkin**6) -
         (32*mckin**7*np.log(mus**2/mckin**2)**2)/(81*mbkin**7) +
         (32*mckin**8*np.log(mus**2/mckin**2)**2)/(81*mbkin**8) -
         (64*mckin**3*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2)**2)/
          (27*mbkin**3) + (64*mckin**4*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2)**
            2)/(27*mbkin**4) + (20*(1 - (8*mckin**2)/mbkin**2 +
            (8*mckin**6)/mbkin**6 - mckin**8/mbkin**8 -
            (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*np.log(mus**2/mckin**2)**2)/
          81))/mbkin +
     ((16*api4*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
          (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
          12*np.log(mckin**2/mbkin**2)))/27 +
       api4**2*(-160/3 - (512*mbkin)/(27*mckin) - (7424*mckin)/(27*mbkin) +
         (5632*mckin**2)/(27*mbkin**2) + (1024*mckin**3)/(3*mbkin**3) -
         (1792*mckin**5)/(27*mbkin**5) - (512*mckin**6)/(3*mbkin**6) +
         (512*mckin**7)/(27*mbkin**7) + (416*mckin**8)/(27*mbkin**8) -
         (512*mckin*np.log(mckin**2/mbkin**2))/(3*mbkin) -
         (512*mckin**2*np.log(mckin**2/mbkin**2))/(9*mbkin**2) -
         (1024*mckin**3*np.log(mckin**2/mbkin**2))/(9*mbkin**3) +
         (896*mckin**4*np.log(mckin**2/mbkin**2))/(3*mbkin**4) +
         (640*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/
          27 - (2*(-17 + (16*mckin**2)/mbkin**2 + (12*mckin**4)/mbkin**4 -
            (16*mckin**6)/mbkin**6 + (5*mckin**8)/mbkin**8 -
            12*np.log(mckin**2/mbkin**2))*((2*(-147 + 6*np.pi**2 + 54*np.log(2/mus)))/
             27 + (4*np.log(mus**2/mckin**2))/27))/3) +
       api4**3*(-27662269794518822/5668365773889 - (3468446855826574*mbkin)/
          (20784007837593*mckin) + (4110509221189662098*mckin)/
          (489908756171835*mbkin) - (1501036004462738746*mckin**2)/
          (25784671377465*mbkin**2) + (35020129787948225056*mckin**3)/
          (180492699642255*mbkin**3) - (3888781756705260304*mckin**4)/
          (10617217626015*mbkin**4) + (1008820610922418312*mckin**5)/
          (2123443525203*mbkin**5) - (6157823873143784*mckin**6)/
          (26609567985*mbkin**6) - (344604737327598092*mckin**7)/
          (505581791715*mbkin**7) + (46674744486128036*mckin**8)/
          (20941257645*mbkin**8) - (17008858485396524*mckin**9)/
          (4367428065*mbkin**9) + (72945772648415812*mckin**10)/
          (14849255421*mbkin**10) - (27482213033422984*mckin**11)/
          (5711252085*mbkin**11) + (7979274371490235496*mckin**12)/
          (2123443525203*mbkin**12) - (25040295951005181368*mckin**13)/
          (10617217626015*mbkin**13) + (12601283303434314184*mckin**14)/
          (10617217626015*mbkin**14) - (111745692602997850*mckin**15)/
          (235938169467*mbkin**15) + (196221967275624242*mckin**16)/
          (1336982960313*mbkin**16) - (2051517369018204898*mckin**17)/
          (60164233214085*mbkin**17) + (19187042723999125454*mckin**18)/
          (3429361293202845*mbkin**18) - (1984159514437379368*mckin**19)/
          (3429361293202845*mbkin**19) + (252772929075848*mckin**20)/
          (8907431930397*mbkin**20) + (8192*(1 - mckin/mbkin)**3)/27 -
         (8192*mbkin*(1 - mckin/mbkin)**3)/(27*mckin) +
         (8192*mckin*(1 - mckin/mbkin)**3)/(27*mbkin) -
         (8192*mckin**2*(1 - mckin/mbkin)**3)/(27*mbkin**2) -
         (20480*(1 - mckin/mbkin)**4)/27 + (16384*mbkin*(1 - mckin/mbkin)**4)/
          (27*mckin) - (14336*mckin*(1 - mckin/mbkin)**4)/(27*mbkin) +
         (2048*mckin**2*(1 - mckin/mbkin)**4)/(3*mbkin**2) +
         (2220032*(1 - mckin/mbkin)**5)/2025 -
         (1667072*mbkin*(1 - mckin/mbkin)**5)/(2025*mckin) +
         (1390592*mckin*(1 - mckin/mbkin)**5)/(2025*mbkin) -
         (1943552*mckin**2*(1 - mckin/mbkin)**5)/(2025*mbkin**2) -
         (20231168*(1 - mckin/mbkin)**6)/25515 +
         (20878336*mbkin*(1 - mckin/mbkin)**6)/(42525*mckin) -
         (43374592*mckin*(1 - mckin/mbkin)**6)/(127575*mbkin) +
         (81895424*mckin**2*(1 - mckin/mbkin)**6)/(127575*mbkin**2) +
         (41368576*(1 - mckin/mbkin)**7)/178605 -
         (50916352*mbkin*(1 - mckin/mbkin)**7)/(893025*mckin) -
         (27046912*mckin*(1 - mckin/mbkin)**7)/(893025*mbkin) -
         (42959872*mckin**2*(1 - mckin/mbkin)**7)/(297675*mbkin**2) +
         (13403648*(1 - mckin/mbkin)**8)/893025 -
         (44333312*mbkin*(1 - mckin/mbkin)**8)/(893025*mckin) +
         (8542592*mckin*(1 - mckin/mbkin)**8)/(127575*mbkin) -
         (5773696*mckin**2*(1 - mckin/mbkin)**8)/(178605*mbkin**2) +
         (35927936*(1 - mckin/mbkin)**9)/1607445 -
         (15442816*mbkin*(1 - mckin/mbkin)**9)/(535815*mckin) +
         (51528704*mckin*(1 - mckin/mbkin)**9)/(1607445*mbkin) -
         (5875456*mckin**2*(1 - mckin/mbkin)**9)/(229635*mbkin**2) +
         (418756864*(1 - mckin/mbkin)**10)/17681895 -
         (99690496*mbkin*(1 - mckin/mbkin)**10)/(3536379*mckin) +
         (538300288*mckin*(1 - mckin/mbkin)**10)/(17681895*mbkin) -
         (152868224*mckin**2*(1 - mckin/mbkin)**10)/(5893965*mbkin**2) +
         (1644400256*(1 - mckin/mbkin)**11)/64833615 -
         (212667776*mbkin*(1 - mckin/mbkin)**11)/(7203735*mckin) +
         (2048814848*mckin*(1 - mckin/mbkin)**11)/(64833615*mbkin) -
         (4621312*mckin**2*(1 - mckin/mbkin)**11)/(168399*mbkin**2) +
         (9847287808*(1 - mckin/mbkin)**12)/361215855 -
         (3771121664*mbkin*(1 - mckin/mbkin)**12)/(120405285*mckin) +
         (12046403584*mckin*(1 - mckin/mbkin)**12)/(361215855*mbkin) -
         (302295040*mckin**2*(1 - mckin/mbkin)**12)/(10320453*mbkin**2) +
         (137481633088*(1 - mckin/mbkin)**13)/4695806115 -
         (156395590976*mbkin*(1 - mckin/mbkin)**13)/(4695806115*mckin) +
         (33170513984*mckin*(1 - mckin/mbkin)**13)/(939161223*mbkin) -
         (5442170816*mckin**2*(1 - mckin/mbkin)**13)/(173918745*mbkin**2) +
         (114555009376*(1 - mckin/mbkin)**14)/3652293645 -
         (129251733664*mbkin*(1 - mckin/mbkin)**14)/(3652293645*mckin) +
         (12418190528*mckin*(1 - mckin/mbkin)**14)/(332026695*mbkin) -
         (3482953472*mckin**2*(1 - mckin/mbkin)**14)/(104351247*mbkin**2) +
         (262181927936*(1 - mckin/mbkin)**15)/7826343525 -
         (19582795264*mbkin*(1 - mckin/mbkin)**15)/(521756235*mckin) +
         (44217418496*mckin*(1 - mckin/mbkin)**15)/(1118049075*mbkin) -
         (277961928448*mckin**2*(1 - mckin/mbkin)**15)/(7826343525*mbkin**2) +
         (33217721099392*(1 - mckin/mbkin)**16)/931334879475 -
         (36986867372816*mbkin*(1 - mckin/mbkin)**16)/(931334879475*mckin) +
         (3533767319048*mckin*(1 - mckin/mbkin)**16)/(84666807225*mbkin) -
         (11700764745368*mckin**2*(1 - mckin/mbkin)**16)/(310444959825*
           mbkin**2) + (199788100541908*(1 - mckin/mbkin)**17)/5277564317025 -
         (20112465872524*mbkin*(1 - mckin/mbkin)**17)/(479778574275*mckin) +
         (231961636625692*mckin*(1 - mckin/mbkin)**17)/(5277564317025*mbkin) -
         (70170870856612*mckin**2*(1 - mckin/mbkin)**17)/(1759188105675*
           mbkin**2) + (36153867420877538*(1 - mckin/mbkin)**18)/
          902463498211275 - (781135845875842*mbkin*(1 - mckin/mbkin)**18)/
          (17695362710025*mckin) + (3206150653774088*mckin*
           (1 - mckin/mbkin)**18)/(69420269093175*mbkin) -
         (98690643585124*mckin**2*(1 - mckin/mbkin)**18)/(2344061034315*
           mbkin**2) - (91305091331044*(1 - mckin/mbkin)**19)/44537159651985 -
         (91305091331044*mbkin*(1 - mckin/mbkin)**19)/(44537159651985*mckin) +
         (182610182662088*mckin*(1 - mckin/mbkin)**19)/(44537159651985*
           mbkin) + (496*np.pi**2)/9 + (512*mbkin*np.pi**2)/(27*mckin) +
         (7168*mckin*np.pi**2)/(27*mbkin) - (640*mckin**2*np.pi**2)/(3*mbkin**2) -
         (3200*mckin**3*np.pi**2)/(9*mbkin**3) + (128*mckin**4*np.pi**2)/(9*mbkin**4) +
         (2560*mckin**5*np.pi**2)/(27*mbkin**5) + (1408*mckin**6*np.pi**2)/(9*mbkin**6) -
         (640*mckin**7*np.pi**2)/(27*mbkin**7) - (112*mckin**8*np.pi**2)/(9*mbkin**8) -
         (1246336*mckin*np.log(mckin**2/mbkin**2))/(243*mbkin) -
         (161408*mckin**2*np.log(mckin**2/mbkin**2))/(81*mbkin**2) -
         (66304*mckin**3*np.log(mckin**2/mbkin**2))/(27*mbkin**3) +
         (2169184*mckin**4*np.log(mckin**2/mbkin**2))/(243*mbkin**4) +
         (512*mckin*np.pi**2*np.log(mckin**2/mbkin**2))/(3*mbkin) +
         (512*mckin**2*np.pi**2*np.log(mckin**2/mbkin**2))/(9*mbkin**2) +
         (256*mckin**3*np.pi**2*np.log(mckin**2/mbkin**2))/(3*mbkin**3) -
         (2624*mckin**4*np.pi**2*np.log(mckin**2/mbkin**2))/(9*mbkin**4) +
         (529120*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/
          729 + (160*mbkin**2*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/
          (81*mckin**2) + (64*mbkin*(1 - (8*mckin**2)/mbkin**2 +
            (8*mckin**6)/mbkin**6 - mckin**8/mbkin**8 -
            (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/(27*mckin) -
         (640*np.pi**2*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/
          27 - (131072*(1 - mckin/mbkin)**5*(np.log(2) + np.log(1 - mckin/mbkin)))/
          405 + (131072*mbkin*(1 - mckin/mbkin)**5*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(405*mckin) -
         (131072*mckin*(1 - mckin/mbkin)**5*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (405*mbkin) + (131072*mckin**2*(1 - mckin/mbkin)**5*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(405*mbkin**2) +
         (65536*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin)))/243 -
         (65536*mbkin*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (405*mckin) + (131072*mckin*(1 - mckin/mbkin)**6*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1215*mbkin) -
         (262144*mckin**2*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1215*mbkin**2) - (131072*(1 - mckin/mbkin)**7*
           (np.log(2) + np.log(1 - mckin/mbkin)))/1701 +
         (131072*mbkin*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (8505*mckin) + (131072*mckin*(1 - mckin/mbkin)**7*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(8505*mbkin) +
         (131072*mckin**2*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (2835*mbkin**2) - (65536*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin)))/2835 +
         (32768*mbkin*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (945*mckin) - (16384*mckin*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(405*mbkin) +
         (16384*mckin**2*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (567*mbkin**2) - (139264*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin)))/5103 +
         (8192*mbkin*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (243*mckin) - (188416*mckin*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(5103*mbkin) +
         (155648*mckin**2*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (5103*mbkin**2) - (782336*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin)))/25515 +
         (937984*mbkin*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (25515*mckin) - (1015808*mckin*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(25515*mbkin) +
         (8192*mckin**2*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (243*mbkin**2) - (151552*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin)))/4455 +
         (536576*mbkin*(1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (13365*mckin) - (192512*mckin*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(4455*mbkin) +
         (45056*mckin**2*(1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1215*mbkin**2) - (1497088*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin)))/40095 +
         (581632*mbkin*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (13365*mckin) - (373760*mckin*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(8019*mbkin) +
         (1620992*mckin**2*(1 - mckin/mbkin)**12*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(40095*mbkin**2) -
         (21173248*(1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)))/
          521235 + (24415232*mbkin*(1 - mckin/mbkin)**13*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(521235*mckin) -
         (26036224*mckin*(1 - mckin/mbkin)**13*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(521235*mbkin) +
         (1519616*mckin**2*(1 - mckin/mbkin)**13*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(34749*mbkin**2) -
         (395264*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)))/9009 +
         (12191744*mbkin*(1 - mckin/mbkin)**14*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(243243*mckin) -
         (4317184*mckin*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (81081*mbkin) + (11431936*mckin**2*(1 - mckin/mbkin)**14*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(243243*mbkin**2) -
         (171870208*(1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin)))/
          3648645 + (4327424*mbkin*(1 - mckin/mbkin)**15*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(81081*mckin) -
         (29452288*mckin*(1 - mckin/mbkin)**15*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(521235*mbkin) +
         (183302144*mckin**2*(1 - mckin/mbkin)**15*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(3648645*mbkin**2) -
         (14123008*(1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)))/
          280665 + (29501696*mbkin*(1 - mckin/mbkin)**16*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(521235*mckin) -
         (19815296*mckin*(1 - mckin/mbkin)**16*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(331695*mbkin) +
         (21672832*mckin**2*(1 - mckin/mbkin)**16*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(405405*mbkin**2) -
         (368870272*(1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin)))/
          6891885 + (12491392*mbkin*(1 - mckin/mbkin)**17*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(208845*mckin) -
         (433888768*mckin*(1 - mckin/mbkin)**17*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(6891885*mbkin) +
         (55791872*mckin**2*(1 - mckin/mbkin)**17*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(984555*mbkin**2) -
         (502561024*(1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin)))/
          8860995 + (10948096*mbkin*(1 - mckin/mbkin)**18*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(173745*mckin) -
         (45096064*mckin*(1 - mckin/mbkin)**18*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(681615*mbkin) +
         (9644672*mckin**2*(1 - mckin/mbkin)**18*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(161109*mbkin**2) +
         (9644672*(1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)))/
          3061071 + (9644672*mbkin*(1 - mckin/mbkin)**19*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(3061071*mckin) -
         (19289344*mckin*(1 - mckin/mbkin)**19*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(3061071*mbkin) +
         (327680*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/1701 -
         (65536*mbkin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(1701*mckin) -
         (65536*mckin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(243*mbkin) -
         (65536*mckin**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (243*mbkin**2) + (327680*mckin**3*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(243*mbkin**3) -
         (458752*mckin**4*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (243*mbkin**4) + (327680*mckin**5*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(243*mbkin**5) -
         (851968*mckin**6*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**6) + (131072*mckin**7*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(1701*mbkin**7) -
         (65536*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/567 +
         (32768*mbkin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/(1701*mckin) +
         (131072*mckin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/(567*mbkin) -
         (65536*mckin**3*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (81*mbkin**3) + (131072*mckin**4*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(81*mbkin**4) -
         (131072*mckin**5*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (81*mbkin**5) + (524288*mckin**6*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(567*mbkin**6) -
         (163840*mckin**7*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (567*mbkin**7) + (65536*mckin**8*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(1701*mbkin**8) +
         (16384*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/729 -
         (16384*mbkin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/(5103*mckin) -
         (32768*mckin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/(567*mbkin) +
         (65536*mckin**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**2) + (32768*mckin**3*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(243*mbkin**3) -
         (32768*mckin**4*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (81*mbkin**4) + (131072*mckin**5*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(243*mbkin**5) -
         (720896*mckin**6*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**6) + (16384*mckin**7*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(81*mbkin**7) -
         (278528*mckin**8*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (5103*mbkin**8) + (32768*mckin**9*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(5103*mbkin**9) +
         (65536*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/5103 -
         (8192*mbkin*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (5103*mckin) - (204800*mckin*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(5103*mbkin) +
         (81920*mckin**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**2) + (81920*mckin**3*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(1701*mbkin**3) -
         (65536*mckin**4*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (243*mbkin**4) + (114688*mckin**5*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(243*mbkin**5) -
         (819200*mckin**6*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**6) + (532480*mckin**7*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(1701*mbkin**7) -
         (655360*mckin**8*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (5103*mbkin**8) + (155648*mckin**9*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(5103*mbkin**9) -
         (16384*mckin**10*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (5103*mbkin**10) + (77824*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          6237 - (77824*mbkin*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (56133*mckin) - (77824*mckin*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(1701*mbkin) +
         (389120*mckin**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (5103*mbkin**2) - (155648*mckin**4*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(567*mbkin**4) +
         (155648*mckin**5*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (243*mbkin**5) - (155648*mckin**6*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(189*mbkin**6) +
         (389120*mckin**7*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (567*mbkin**7) - (1945600*mckin**8*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(5103*mbkin**8) +
         (77824*mckin**9*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (567*mbkin**9) - (77824*mckin**10*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(2673*mbkin**10) +
         (155648*mckin**11*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (56133*mbkin**11) + (102400*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/8019 -
         (10240*mbkin*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (8019*mckin) - (143360*mckin*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(2673*mbkin) +
         (81920*mckin**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (729*mbkin**2) - (51200*mckin**3*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(729*mbkin**3) -
         (20480*mckin**4*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (81*mbkin**4) + (204800*mckin**5*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(243*mbkin**5) -
         (327680*mckin**6*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (243*mbkin**6) + (112640*mckin**7*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**7) -
         (716800*mckin**8*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (729*mbkin**8) + (348160*mckin**9*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(729*mbkin**9) -
         (409600*mckin**10*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**10) + (235520*mckin**11*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(8019*mbkin**11) -
         (20480*mckin**12*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (8019*mbkin**12) + (123904*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/9477 -
         (11264*mbkin*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (9477*mckin) - (45056*mckin*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(729*mbkin) +
         (112640*mckin**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (729*mbkin**2) - (123904*mckin**3*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(729*mbkin**3) -
         (123904*mckin**4*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (729*mbkin**4) + (247808*mckin**5*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(243*mbkin**5) -
         (495616*mckin**6*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (243*mbkin**6) + (619520*mckin**7*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(243*mbkin**7) -
         (1610752*mckin**8*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (729*mbkin**8) + (991232*mckin**9*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(729*mbkin**9) -
         (428032*mckin**10*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (729*mbkin**10) + (123904*mckin**11*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(729*mbkin**11) -
         (281600*mckin**12*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (9477*mbkin**12) + (22528*mckin**13*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(9477*mbkin**13) +
         (3241984*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/243243 -
         (810496*mbkin*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (729729*mckin) - (810496*mckin*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(11583*mbkin) +
         (1620992*mckin**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (8019*mbkin**2) - (810496*mckin**3*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(2673*mbkin**3) +
         (810496*mckin**5*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (729*mbkin**5) - (1620992*mckin**6*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(567*mbkin**6) +
         (810496*mckin**7*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (189*mbkin**7) - (3241984*mckin**8*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(729*mbkin**8) +
         (810496*mckin**9*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (243*mbkin**9) - (1620992*mckin**10*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(891*mbkin**10) +
         (5673472*mckin**11*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (8019*mbkin**11) - (6483968*mckin**12*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(34749*mbkin**12) +
         (810496*mckin**13*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (27027*mbkin**13) - (1620992*mckin**14*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(729729*mbkin**14) +
         (108544*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/8019 -
         (108544*mbkin*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (104247*mckin) - (2713600*mckin*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(34749*mbkin) +
         (26593280*mckin**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (104247*mbkin**2) - (3799040*mckin**3*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(8019*mbkin**3) +
         (759808*mckin**4*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**4) + (759808*mckin**5*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(729*mbkin**5) -
         (2713600*mckin**6*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (729*mbkin**6) + (542720*mckin**7*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(81*mbkin**7) -
         (5969920*mckin**8*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (729*mbkin**8) + (5318656*mckin**9*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(729*mbkin**9) -
         (12916736*mckin**10*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**10) + (18995200*mckin**11*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(8019*mbkin**11) -
         (87377920*mckin**12*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (104247*mbkin**12) + (542720*mckin**13*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(2673*mbkin**13) -
         (3147776*mckin**14*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (104247*mbkin**14) + (217088*mckin**15*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(104247*mbkin**15) +
         (1428992*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/104247 -
         (714496*mbkin*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (729729*mckin) - (5715968*mckin*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(66339*mbkin) +
         (228638720*mckin**2*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (729729*mbkin**2) - (71449600*mckin**3*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(104247*mbkin**3) +
         (5715968*mckin**4*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (8019*mbkin**4) + (5715968*mckin**5*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(8019*mbkin**5) -
         (22863872*mckin**6*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (5103*mbkin**6) + (7144960*mckin**7*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(729*mbkin**7) -
         (71449600*mckin**8*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (5103*mbkin**8) + (74307584*mckin**9*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(5103*mbkin**9) -
         (91455488*mckin**10*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (8019*mbkin**10) + (54301696*mckin**11*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(8019*mbkin**11) -
         (28579840*mckin**12*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (9477*mbkin**12) + (714496000*mckin**13*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(729729*mbkin**13) -
         (22863872*mckin**14*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (104247*mbkin**14) + (22149376*mckin**15*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(729729*mbkin**15) -
         (1428992*mckin**16*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (729729*mbkin**16) + (57281920*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/4135131 -
         (11456384*mbkin*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (12405393*mckin) - (22912768*mckin*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(243243*mbkin) +
         (91651072*mckin**2*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (243243*mbkin**2) - (229127680*mckin**3*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(243243*mbkin**3) +
         (45825536*mckin**4*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**4) - (91651072*mckin**6*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(18711*mbkin**6) +
         (22912768*mckin**7*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**7) - (114563840*mckin**8*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(5103*mbkin**8) +
         (45825536*mckin**9*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (1701*mbkin**9) - (458255360*mckin**10*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(18711*mbkin**10) +
         (45825536*mckin**11*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**11) - (320778752*mckin**12*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(34749*mbkin**12) +
         (916510720*mckin**13*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (243243*mbkin**13) - (91651072*mckin**14*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(81081*mbkin**14) +
         (57281920*mckin**15*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (243243*mbkin**15) - (11456384*mckin**16*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(375921*mbkin**16) +
         (22912768*mckin**17*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (12405393*mbkin**17) + (173382656*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/12405393 -
         (10836416*mbkin*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (12405393*mckin) - (10836416*mckin*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(106029*mbkin) +
         (108364160*mckin**2*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (243243*mbkin**2) - (43345664*mckin**3*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(34749*mbkin**3) +
         (173382656*mckin**4*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (81081*mbkin**4) - (43345664*mckin**5*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(34749*mbkin**5) -
         (86691328*mckin**6*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (18711*mbkin**6) + (108364160*mckin**7*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(6237*mbkin**7) -
         (173382656*mckin**8*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (5103*mbkin**8) + (238401152*mckin**9*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(5103*mbkin**9) -
         (43345664*mckin**10*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (891*mbkin**10) + (736876288*mckin**11*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(18711*mbkin**11) -
         (866913280*mckin**12*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**12) + (996950272*mckin**13*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(81081*mbkin**13) -
         (86691328*mckin**14*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (18711*mbkin**14) + (314256064*mckin**15*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(243243*mbkin**15) -
         (346765312*mckin**16*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (1378377*mbkin**16) + (54182080*mckin**17*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(1772199*mbkin**17) -
         (21672832*mckin**18*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (12405393*mbkin**18) + (27895936*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/1980693 -
         (27895936*mbkin*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (33671781*mckin) - (195271552*mckin*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(1772199*mbkin) +
         (27895936*mckin**2*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (53703*mbkin**2) - (55791872*mckin**3*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(34749*mbkin**3) +
         (111583744*mckin**4*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**4) - (111583744*mckin**5*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(34749*mbkin**5) -
         (111583744*mckin**6*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**6) + (55791872*mckin**7*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(2673*mbkin**7) -
         (390543104*mckin**8*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (8019*mbkin**8) + (55791872*mckin**9*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(729*mbkin**9) -
         (725294336*mckin**10*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (8019*mbkin**10) + (223167488*mckin**11*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(2673*mbkin**11) -
         (2120091136*mckin**12*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**12) + (111583744*mckin**13*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(3159*mbkin**13) -
         (557918720*mckin**14*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**14) + (195271552*mckin**15*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(34749*mbkin**15) -
         (864774016*mckin**16*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (590733*mbkin**16) + (27895936*mckin**17*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(104247*mbkin**17) -
         (1032149632*mckin**18*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (33671781*mbkin**18) + (55791872*mckin**19*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(33671781*mbkin**19) +
         (4822336*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/340119 -
         (2411168*mbkin*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (3061071*mckin) - (120558400*mckin*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1020357*mbkin) +
         (96446720*mckin**2*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (161109*mbkin**2) - (12055840*mckin**3*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(5967*mbkin**3) +
         (4822336*mckin**4*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1053*mbkin**4) - (19289344*mckin**5*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(3159*mbkin**5) +
         (24111680*mckin**7*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1053*mbkin**7) - (48223360*mckin**8*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(729*mbkin**8) +
         (9644672*mckin**9*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (81*mbkin**9) - (38578688*mckin**10*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(243*mbkin**10) +
         (120558400*mckin**11*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (729*mbkin**11) - (48223360*mckin**12*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(351*mbkin**12) +
         (96446720*mckin**13*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1053*mbkin**13) - (154314752*mckin**14*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(3159*mbkin**14) +
         (2411168*mckin**15*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (117*mbkin**15) - (120558400*mckin**16*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(17901*mbkin**16) +
         (265228480*mckin**17*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (161109*mbkin**17) - (96446720*mckin**18*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(340119*mbkin**18) +
         (2411168*mckin**19*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (78489*mbkin**19) - (4822336*mckin**20*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(3061071*mbkin**20) +
         (4*(-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin))))/3429361293202845 + 496*np.log(2/mus) +
         (512*mbkin*np.log(2/mus))/(3*mckin) + (7168*mckin*np.log(2/mus))/
          (3*mbkin) - (1920*mckin**2*np.log(2/mus))/mbkin**2 -
         (3200*mckin**3*np.log(2/mus))/mbkin**3 + (128*mckin**4*np.log(2/mus))/
          mbkin**4 + (2560*mckin**5*np.log(2/mus))/(3*mbkin**5) +
         (1408*mckin**6*np.log(2/mus))/mbkin**6 - (640*mckin**7*np.log(2/mus))/
          (3*mbkin**7) - (112*mckin**8*np.log(2/mus))/mbkin**8 +
         (1536*mckin*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin +
         (512*mckin**2*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**2 +
         (768*mckin**3*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**3 -
         (2624*mckin**4*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**4 -
         (640*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(2/mus))/3 + (256*mckin*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/
          (243*mbkin) - (256*mckin**2*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/
          (243*mbkin**2) + (128*mckin**3*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/
          (81*mbkin**3) - (128*mckin**4*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/
          (81*mbkin**4) - (256*mckin**5*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/
          (81*mbkin**5) + (256*mckin**6*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/
          (81*mbkin**6) + (128*mckin**7*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/
          (243*mbkin**7) - (128*mckin**8*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/
          (243*mbkin**8) + (256*mckin**3*np.log(mckin**2/mbkin**2)*
           (-234 + 9*np.pi**2 + 81*np.log(2/mus)))/(81*mbkin**3) -
         (256*mckin**4*np.log(mckin**2/mbkin**2)*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/
          (81*mbkin**4) + (32*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           (-234 + 9*np.pi**2 + 81*np.log(2/mus)))/243 -
         (40*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           ((2*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/27 + (2*np.log(mus**2/mckin**2))/
             9))/9 + (496*np.log(mus**2/mckin**2))/27 +
         (512*mbkin*np.log(mus**2/mckin**2))/(81*mckin) +
         (7424*mckin*np.log(mus**2/mckin**2))/(81*mbkin) -
         (6016*mckin**2*np.log(mus**2/mckin**2))/(81*mbkin**2) -
         (1024*mckin**3*np.log(mus**2/mckin**2))/(9*mbkin**3) +
         (1792*mckin**5*np.log(mus**2/mckin**2))/(81*mbkin**5) +
         (1664*mckin**6*np.log(mus**2/mckin**2))/(27*mbkin**6) -
         (512*mckin**7*np.log(mus**2/mckin**2))/(81*mbkin**7) -
         (464*mckin**8*np.log(mus**2/mckin**2))/(81*mbkin**8) +
         (512*mckin*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/(9*mbkin) +
         (512*mckin**2*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/(27*mbkin**2) +
         (1024*mckin**3*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/
          (27*mbkin**3) - (320*mckin**4*np.log(mckin**2/mbkin**2)*
           np.log(mus**2/mckin**2))/(3*mbkin**4) -
         (608*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(mus**2/mckin**2))/81 - (2*(-17 + (16*mckin**2)/mbkin**2 +
            (12*mckin**4)/mbkin**4 - (16*mckin**6)/mbkin**6 +
            (5*mckin**8)/mbkin**8 - 12*np.log(mckin**2/mbkin**2))*
           ((-126695.04872806392 + 306504*np.log(2/mus) - 11664*np.pi**2*np.log(
                2/mus) - 52488*np.log(2/mus)**2)/2916 -
            (2*(-147 + 6*np.pi**2 + 54*np.log(2/mus))*np.log(mus**2/mckin**2))/81 +
            (21 + 57*np.log(mus**2/mckin**2) - 2*np.log(mus**2/mckin**2)**2)/81))/3))/
      mbkin**3 + (api4*(-20/3 - (112*mckin**2)/(3*mbkin**2) +
         (48*mckin**4)/mbkin**4 - (16*mckin**6)/(3*mbkin**6) +
         (4*mckin**8)/(3*mbkin**8) - (32*mckin**2*np.log(mckin**2/mbkin**2))/
          mbkin**2 - (16*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4) +
       api4**2*(-6273112628327/15320660355 + (2649349310527609*mbkin)/
          (92373368167080*mckin) + (511262019577991257*mckin)/
          (423377937432450*mbkin) - (8110322966684879*mckin**2)/
          (1387871585100*mbkin**2) + (14471367571252478657*mckin**3)/
          (802189776187800*mbkin**3) - (324339584899681999*mckin**4)/
          (7864605648900*mbkin**4) + (90948914036368172*mckin**5)/
          (1179690847335*mbkin**5) - (30666968564167367*mckin**6)/
          (280878773175*mbkin**6) + (47194525665047*mckin**7)/
          (420319900*mbkin**7) - (282219995801063*mckin**8)/
          (3878010675*mbkin**8) - (3597890666443*mckin**9)/
          (41247931725*mbkin**9) + (401662703898359*mckin**10)/
          (5499724230*mbkin**10) - (18530565542904137*mckin**11)/
          (164991726900*mbkin**11) + (1277483849769067349*mckin**12)/
          (11796908473350*mbkin**12) - (151400853599994512*mckin**13)/
          (1966151412225*mbkin**13) + (35484451873152599*mckin**14)/
          (842636319525*mbkin**14) - (18702555837085577*mckin**15)/
          (1048614086520*mbkin**15) + (64399203016910624*mckin**16)/
          (11141524669275*mbkin**16) - (5468757565122931*mckin**17)/
          (3932302824450*mbkin**17) + (162592894134003649*mckin**18)/
          (692800261253100*mbkin**18) - (22978828361473*mckin**19)/
          (925247723400*mbkin**19) + (4916068298621*mckin**20)/
          (3958858635732*mbkin**20) + (16*np.pi**2)/3 + (8*mckin**2*np.pi**2)/
          (3*mbkin**2) - (24*mckin**4*np.pi**2)/mbkin**4 + (56*mckin**6*np.pi**2)/
          (3*mbkin**6) - (8*mckin**8*np.pi**2)/(3*mbkin**8) -
         (5792*mckin**2*np.log(mckin**2/mbkin**2))/(9*mbkin**2) -
         (4096*mckin**3*np.log(mckin**2/mbkin**2))/(27*mbkin**3) +
         (21472*mckin**4*np.log(mckin**2/mbkin**2))/(27*mbkin**4) +
         (16*mckin**2*np.pi**2*np.log(mckin**2/mbkin**2))/mbkin**2 -
         (16*mckin**4*np.pi**2*np.log(mckin**2/mbkin**2))/mbkin**4 +
         (6070*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/
          81 - (5*np.pi**2*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/3 +
         (8192*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/315 -
         (4096*mbkin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(945*mckin) -
         (8192*mckin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(135*mbkin) +
         (8192*mckin**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (135*mbkin**2) - (8192*mckin**4*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(135*mbkin**4) +
         (8192*mckin**5*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) - (8192*mckin**6*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(315*mbkin**6) +
         (4096*mckin**7*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (945*mbkin**7) - (2048*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          135 + (2048*mbkin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (945*mckin) + (8192*mckin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (189*mbkin) - (8192*mckin**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(135*mbkin**2) +
         (4096*mckin**3*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (135*mbkin**3) + (4096*mckin**4*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(135*mbkin**4) -
         (8192*mckin**5*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) + (8192*mckin**6*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(189*mbkin**6) -
         (2048*mckin**7*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (135*mbkin**7) + (2048*mckin**8*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(945*mbkin**8) +
         (8192*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/2835 -
         (1024*mbkin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/(2835*mckin) -
         (1024*mckin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/(105*mbkin) +
         (16384*mckin**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (945*mbkin**2) - (2048*mckin**3*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(135*mbkin**3) +
         (2048*mckin**5*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) - (16384*mckin**6*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(945*mbkin**6) +
         (1024*mckin**7*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (105*mbkin**7) - (8192*mckin**8*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(2835*mbkin**8) +
         (1024*mckin**9*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**9) + (512*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          315 - (512*mbkin*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (2835*mckin) - (512*mckin*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(81*mbkin) +
         (2560*mckin**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (189*mbkin**2) - (1024*mckin**3*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(63*mbkin**3) +
         (1024*mckin**4*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (135*mbkin**4) + (1024*mckin**5*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(135*mbkin**5) -
         (1024*mckin**6*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (63*mbkin**6) + (2560*mckin**7*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(189*mbkin**7) -
         (512*mckin**8*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (81*mbkin**8) + (512*mckin**9*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(315*mbkin**9) -
         (512*mckin**10*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**10) + (9728*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          6237 - (4864*mbkin*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (31185*mckin) - (19456*mckin*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(2835*mbkin) +
         (9728*mckin**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (567*mbkin**2) - (4864*mckin**3*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(189*mbkin**3) +
         (19456*mckin**4*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (945*mbkin**4) - (19456*mckin**6*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(945*mbkin**6) +
         (4864*mckin**7*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (189*mbkin**7) - (9728*mckin**8*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(567*mbkin**8) +
         (19456*mckin**9*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**9) - (9728*mckin**10*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(6237*mbkin**10) +
         (4864*mckin**11*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (31185*mbkin**11) + (128*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          81 - (128*mbkin*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (891*mckin) - (256*mckin*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (33*mbkin) + (1792*mckin**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**2) -
         (3200*mckin**3*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (81*mbkin**3) + (128*mckin**4*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(3*mbkin**4) -
         (512*mckin**5*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (27*mbkin**5) - (512*mckin**6*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(27*mbkin**6) +
         (128*mckin**7*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (3*mbkin**7) - (3200*mckin**8*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**8) +
         (1792*mckin**9*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (81*mbkin**9) - (256*mckin**10*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(33*mbkin**10) +
         (128*mckin**11*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (81*mbkin**11) - (128*mckin**12*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(891*mbkin**12) +
         (2816*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/1755 -
         (704*mbkin*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(5265*mckin) -
         (704*mckin*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(81*mbkin) +
         (11264*mckin**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (405*mbkin**2) - (7744*mckin**3*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(135*mbkin**3) +
         (30976*mckin**4*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (405*mbkin**4) - (7744*mckin**5*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(135*mbkin**5) +
         (7744*mckin**7*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (135*mbkin**7) - (30976*mckin**8*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(405*mbkin**8) +
         (7744*mckin**9*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (135*mbkin**9) - (11264*mckin**10*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(405*mbkin**10) +
         (704*mckin**11*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (81*mbkin**11) - (2816*mckin**12*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(1755*mbkin**12) +
         (704*mckin**13*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (5265*mbkin**13) + (50656*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          31185 - (50656*mbkin*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (405405*mckin) - (50656*mckin*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(5265*mbkin) +
         (50656*mckin**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (1485*mbkin**2) - (354592*mckin**3*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(4455*mbkin**3) +
         (50656*mckin**4*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (405*mbkin**4) - (50656*mckin**5*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(405*mbkin**5) +
         (50656*mckin**6*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (945*mbkin**6) + (50656*mckin**7*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(945*mbkin**7) -
         (50656*mckin**8*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (405*mbkin**8) + (50656*mckin**9*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(405*mbkin**9) -
         (354592*mckin**10*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (4455*mbkin**10) + (50656*mckin**11*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(1485*mbkin**11) -
         (50656*mckin**12*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (5265*mbkin**12) + (50656*mckin**13*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(31185*mbkin**13) -
         (50656*mckin**14*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (405405*mbkin**14) + (94976*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/57915 -
         (6784*mbkin*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (57915*mckin) - (13568*mckin*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(1287*mbkin) +
         (474880*mckin**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (11583*mbkin**2) - (94976*mckin**3*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(891*mbkin**3) +
         (94976*mckin**4*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (495*mbkin**4) - (94976*mckin**5*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(405*mbkin**5) +
         (13568*mckin**6*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (81*mbkin**6) - (13568*mckin**8*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(81*mbkin**8) +
         (94976*mckin**9*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (405*mbkin**9) - (94976*mckin**10*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(495*mbkin**10) +
         (94976*mckin**11*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (891*mbkin**11) - (474880*mckin**12*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(11583*mbkin**12) +
         (13568*mckin**13*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (1287*mbkin**13) - (94976*mckin**14*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(57915*mbkin**14) +
         (6784*mckin**15*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (57915*mbkin**15) + (44656*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/27027 -
         (44656*mbkin*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (405405*mckin) - (357248*mckin*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(31185*mbkin) +
         (357248*mckin**2*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (7371*mbkin**2) - (178624*mckin**3*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(1287*mbkin**3) +
         (1250368*mckin**4*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (4455*mbkin**4) - (357248*mckin**5*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(891*mbkin**5) +
         (357248*mckin**6*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (945*mbkin**6) - (89312*mckin**7*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(567*mbkin**7) -
         (89312*mckin**8*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (567*mbkin**8) + (357248*mckin**9*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(945*mbkin**9) -
         (357248*mckin**10*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (891*mbkin**10) + (1250368*mckin**11*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(4455*mbkin**11) -
         (178624*mckin**12*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (1287*mbkin**12) + (357248*mckin**13*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(7371*mbkin**13) -
         (357248*mckin**14*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (31185*mbkin**14) + (44656*mckin**15*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(27027*mbkin**15) -
         (44656*mckin**16*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (405405*mbkin**16) + (11456384*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/6891885 -
         (716024*mbkin*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (6891885*mckin) - (716024*mckin*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(57915*mbkin) +
         (22912768*mckin**2*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (405405*mbkin**2) - (14320480*mckin**3*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(81081*mbkin**3) +
         (22912768*mckin**4*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (57915*mbkin**4) - (2864096*mckin**5*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(4455*mbkin**5) +
         (22912768*mckin**6*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (31185*mbkin**6) - (1432048*mckin**7*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(2835*mbkin**7) +
         (1432048*mckin**9*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**9) - (22912768*mckin**10*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(31185*mbkin**10) +
         (2864096*mckin**11*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (4455*mbkin**11) - (22912768*mckin**12*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(57915*mbkin**12) +
         (14320480*mckin**13*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (81081*mbkin**13) - (22912768*mckin**14*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(405405*mbkin**14) +
         (716024*mckin**15*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (57915*mbkin**15) - (11456384*mckin**16*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(6891885*mbkin**16) +
         (716024*mckin**17*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (6891885*mbkin**17) + (677276*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/405405 -
         (677276*mbkin*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (6891885*mckin) - (677276*mckin*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(51051*mbkin) +
         (677276*mckin**2*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (10395*mbkin**2) - (2709104*mckin**3*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(12285*mbkin**3) +
         (2709104*mckin**4*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (5005*mbkin**4) - (18963728*mckin**5*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(19305*mbkin**5) +
         (2709104*mckin**6*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (2079*mbkin**6) - (1354552*mckin**7*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(1155*mbkin**7) +
         (1354552*mckin**8*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**8) + (1354552*mckin**9*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(2835*mbkin**9) -
         (1354552*mckin**10*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (1155*mbkin**10) + (2709104*mckin**11*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(2079*mbkin**11) -
         (18963728*mckin**12*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (19305*mbkin**12) + (2709104*mckin**13*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(5005*mbkin**13) -
         (2709104*mckin**14*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (12285*mbkin**14) + (677276*mckin**15*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(10395*mbkin**15) -
         (677276*mckin**16*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (51051*mbkin**16) + (677276*mckin**17*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(405405*mbkin**17) -
         (677276*mckin**18*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (6891885*mbkin**18) + (3486992*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/2078505 -
         (1743496*mbkin*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (18706545*mckin) - (13947968*mckin*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(984555*mbkin) +
         (24408944*mckin**2*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (328185*mbkin**2) - (1743496*mckin**3*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(6435*mbkin**3) +
         (13947968*mckin**4*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (19305*mbkin**4) - (27895936*mckin**5*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(19305*mbkin**5) +
         (13947968*mckin**6*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (6435*mbkin**6) - (3486992*mckin**7*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(1485*mbkin**7) +
         (6973984*mckin**8*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (4455*mbkin**8) - (6973984*mckin**10*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(4455*mbkin**10) +
         (3486992*mckin**11*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (1485*mbkin**11) - (13947968*mckin**12*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(6435*mbkin**12) +
         (27895936*mckin**13*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (19305*mbkin**13) - (13947968*mckin**14*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(19305*mbkin**14) +
         (1743496*mckin**15*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (6435*mbkin**15) - (24408944*mckin**16*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(328185*mbkin**16) +
         (13947968*mckin**17*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (984555*mbkin**17) - (3486992*mckin**18*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(2078505*mbkin**18) +
         (1743496*mckin**19*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (18706545*mbkin**19) + (150698*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/89505 -
         (150698*mbkin*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1700595*mckin) - (301396*mckin*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(20007*mbkin) +
         (1506980*mckin**2*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (17901*mbkin**2) - (150698*mckin**3*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(459*mbkin**3) +
         (1657678*mckin**4*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1755*mbkin**4) - (1205584*mckin**5*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(585*mbkin**5) +
         (1205584*mckin**6*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (351*mbkin**6) - (1506980*mckin**7*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(351*mbkin**7) +
         (301396*mckin**8*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (81*mbkin**8) - (602792*mckin**9*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(405*mbkin**9) -
         (602792*mckin**10*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (405*mbkin**10) + (301396*mckin**11*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(81*mbkin**11) -
         (1506980*mckin**12*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (351*mbkin**12) + (1205584*mckin**13*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(351*mbkin**13) -
         (1205584*mckin**14*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (585*mbkin**14) + (1657678*mckin**15*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(1755*mbkin**15) -
         (150698*mckin**16*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (459*mbkin**16) + (1506980*mckin**17*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(17901*mbkin**17) -
         (301396*mckin**18*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (20007*mbkin**18) + (150698*mckin**19*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(89505*mbkin**19) -
         (150698*mckin**20*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (1700595*mbkin**20) + (-260123404758497280*(1 - mckin/mbkin)**5 +
           390185107137745920*(1 - mckin/mbkin)**6 - 388690823028777984*
            (1 - mckin/mbkin)**7 + 195377647247557632*(1 - mckin/mbkin)**8 -
           34216429928650112*(1 - mckin/mbkin)**9 - 6097700477192896*
            (1 - mckin/mbkin)**10 - 3945134197513600*(1 - mckin/mbkin)**11 -
           3335963793491680*(1 - mckin/mbkin)**12 - 2990240473256960*
            (1 - mckin/mbkin)**13 - 2738659018760960*(1 - mckin/mbkin)**14 -
           2537989658886624*(1 - mckin/mbkin)**15 - 2370644934651168*
            (1 - mckin/mbkin)**16 - 2227287774097176*(1 - mckin/mbkin)**17 -
           2102198391233484*(1 - mckin/mbkin)**18 - 1991562479412482*
            (1 - mckin/mbkin)**19 - 1892686294969085*(1 - mckin/mbkin)**20 +
           132126173845585920*(1 - mckin/mbkin)**7*(np.log(2) +
             np.log(1 - mckin/mbkin)) - 66063086922792960*(1 - mckin/mbkin)**8*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 11010514487132160*
            (1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)) +
           5505257243566080*(1 - mckin/mbkin)**10*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 4754540346716160*(1 - mckin/mbkin)**11*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 4379181898291200*
            (1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)) +
           4076007766871040*(1 - mckin/mbkin)**13*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 3808925793953280*(1 - mckin/mbkin)**14*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 3570717547837440*
            (1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin)) +
           3357773812673280*(1 - mckin/mbkin)**16*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 3167016139647360*(1 - mckin/mbkin)**17*
            (np.log(2) + np.log(1 - mckin/mbkin)) + 2995631463464640*
            (1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin)) +
           2841110280328320*(1 - mckin/mbkin)**19*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 2701265736929760*(1 - mckin/mbkin)**20*
            (np.log(2) + np.log(1 - mckin/mbkin)))/6096642299027280 +
         48*np.log(2/mus) + (24*mckin**2*np.log(2/mus))/mbkin**2 -
         (216*mckin**4*np.log(2/mus))/mbkin**4 + (168*mckin**6*np.log(2/mus))/
          mbkin**6 - (24*mckin**8*np.log(2/mus))/mbkin**8 +
         (144*mckin**2*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**2 -
         (144*mckin**4*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**4 -
         15*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
           mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
          np.log(2/mus) + ((-1 + (8*mckin**2)/mbkin**2 - (8*mckin**6)/mbkin**6 +
            mckin**8/mbkin**8 + (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           ((2*(-234 + 9*np.pi**2 + 81*np.log(2/mus)))/27 + (2*np.log(mus**2/mckin**2))/
             9))/2 + (16*np.log(mus**2/mckin**2))/9 +
         (8*mckin**2*np.log(mus**2/mckin**2))/(9*mbkin**2) -
         (8*mckin**4*np.log(mus**2/mckin**2))/mbkin**4 +
         (56*mckin**6*np.log(mus**2/mckin**2))/(9*mbkin**6) -
         (8*mckin**8*np.log(mus**2/mckin**2))/(9*mbkin**8) +
         (16*mckin**2*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/(3*mbkin**2) -
         (16*mckin**4*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/(3*mbkin**4) -
         (5*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(mus**2/mckin**2))/9) + api4**3*(-1574.5411209646584 +
         (6770.809337838554*mbkin)/mckin + (483107.34321066807*mckin)/mbkin -
         (1.7125850847397905e6*mckin**2)/mbkin**2 +
         (3.9544315954681076e6*mckin**3)/mbkin**3 -
         (6.020437996649014e6*mckin**4)/mbkin**4 +
         (5.325616723988416e6*mckin**5)/mbkin**5 +
         (188752.77996823582*mckin**6)/mbkin**6 -
         (9.232874444767239e6*mckin**7)/mbkin**7 +
         (1.7381818640896317e7*mckin**8)/mbkin**8 -
         (2.0762039306942314e7*mckin**9)/mbkin**9 +
         (1.8866704929548886e7*mckin**10)/mbkin**10 -
         (1.397279835925316e7*mckin**11)/mbkin**11 +
         (8.729181067137234e6*mckin**12)/mbkin**12 -
         (4.649635684877882e6*mckin**13)/mbkin**13 +
         (2.0899179867034564e6*mckin**14)/mbkin**14 -
         (772371.9214149766*mckin**15)/mbkin**15 +
         (160272384242815408*mckin**16)/(707814508401*mbkin**16) -
         (13105819130875664*mckin**17)/(260451226035*mbkin**17) +
         (1434087851588410768*mckin**18)/(180492699642255*mbkin**18) -
         (8174493464928796592*mckin**19)/(10288083879608535*mbkin**19) +
         (1011091716303392*mckin**20)/(26722295791191*mbkin**20) -
         (32768*(1 - mckin/mbkin)**3)/81 + (65536*mckin*(1 - mckin/mbkin)**3)/
          (81*mbkin) - (32768*mckin**2*(1 - mckin/mbkin)**3)/(81*mbkin**2) +
         (57344*(1 - mckin/mbkin)**4)/81 + (1664*mbkin*(1 - mckin/mbkin)**4)/
          (3*mckin) - (176000*mckin*(1 - mckin/mbkin)**4)/(81*mbkin) +
         (8192*mckin**2*(1 - mckin/mbkin)**4)/(9*mbkin**2) -
         (5562368*(1 - mckin/mbkin)**5)/6075 -
         (4992*mbkin*(1 - mckin/mbkin)**5)/(5*mckin) +
         (19401856*mckin*(1 - mckin/mbkin)**5)/(6075*mbkin) -
         (7774208*mckin**2*(1 - mckin/mbkin)**5)/(6075*mbkin**2) +
         (173498368*(1 - mckin/mbkin)**6)/382725 +
         (2235584*mbkin*(1 - mckin/mbkin)**6)/(2025*mckin) -
         (61573696*mckin*(1 - mckin/mbkin)**6)/(25515*mbkin) +
         (327581696*mckin**2*(1 - mckin/mbkin)**6)/(382725*mbkin**2) +
         (108187648*(1 - mckin/mbkin)**7)/2679075 -
         (63345152*mbkin*(1 - mckin/mbkin)**7)/(99225*mckin) +
         (423529984*mckin*(1 - mckin/mbkin)**7)/(535815*mbkin) -
         (171839488*mckin**2*(1 - mckin/mbkin)**7)/(893025*mbkin**2) -
         (34170368*(1 - mckin/mbkin)**8)/382725 +
         (4188392*mbkin*(1 - mckin/mbkin)**8)/(33075*mckin) +
         (15406744*mckin*(1 - mckin/mbkin)**8)/(2679075*mbkin) -
         (23094784*mckin**2*(1 - mckin/mbkin)**8)/(535815*mbkin**2) -
         (206114816*(1 - mckin/mbkin)**9)/4822335 +
         (4225208*mbkin*(1 - mckin/mbkin)**9)/(178605*mckin) +
         (85515656*mckin*(1 - mckin/mbkin)**9)/(1607445*mbkin) -
         (23501824*mckin**2*(1 - mckin/mbkin)**9)/(688905*mbkin**2) -
         (2153201152*(1 - mckin/mbkin)**10)/53045685 +
         (420472*mbkin*(1 - mckin/mbkin)**10)/(25515*mckin) +
         (3113458552*mckin*(1 - mckin/mbkin)**10)/(53045685*mbkin) -
         (611472896*mckin**2*(1 - mckin/mbkin)**10)/(17681895*mbkin**2) -
         (8195259392*(1 - mckin/mbkin)**11)/194500845 +
         (109528952*mbkin*(1 - mckin/mbkin)**11)/(7203735*mckin) +
         (1372755352*mckin*(1 - mckin/mbkin)**11)/(21611205*mbkin) -
         (18485248*mckin**2*(1 - mckin/mbkin)**11)/(505197*mbkin**2) -
         (48185614336*(1 - mckin/mbkin)**12)/1083647565 +
         (4164992*mbkin*(1 - mckin/mbkin)**12)/(280665*mckin) +
         (24808628608*mckin*(1 - mckin/mbkin)**12)/(361215855*mbkin) -
         (1209180160*mckin**2*(1 - mckin/mbkin)**12)/(30961359*mbkin**2) -
         (132682055936*(1 - mckin/mbkin)**13)/2817483669 +
         (84437312*mbkin*(1 - mckin/mbkin)**13)/(5733585*mckin) +
         (1043702252224*mckin*(1 - mckin/mbkin)**13)/(14087418345*mbkin) -
         (21768683264*mckin**2*(1 - mckin/mbkin)**13)/(521756235*mbkin**2) -
         (49672762112*(1 - mckin/mbkin)**14)/996080085 +
         (153090878*mbkin*(1 - mckin/mbkin)**14)/(10405395*mckin) +
         (872809174778*mckin*(1 - mckin/mbkin)**14)/(10956880935*mbkin) -
         (13931813888*mckin**2*(1 - mckin/mbkin)**14)/(313053741*mbkin**2) -
         (176869673984*(1 - mckin/mbkin)**15)/3354147225 +
         (986250032*mbkin*(1 - mckin/mbkin)**15)/(66891825*mckin) +
         (667920556816*mckin*(1 - mckin/mbkin)**15)/(7826343525*mbkin) -
         (1111847713792*mckin**2*(1 - mckin/mbkin)**15)/(23479030575*mbkin**2) -
         (14135069276192*(1 - mckin/mbkin)**16)/254000421675 +
         (1979593631*mbkin*(1 - mckin/mbkin)**16)/(133783650*mckin) +
         (509104231591817*mckin*(1 - mckin/mbkin)**16)/(5588009276850*mbkin) -
         (46803058981472*mckin**2*(1 - mckin/mbkin)**16)/(931334879475*
           mbkin**2) - (927846546502768*(1 - mckin/mbkin)**17)/15832692951075 +
         (1340564003491*mbkin*(1 - mckin/mbkin)**17)/(90214774650*mckin) +
         (3069256028338883*mckin*(1 - mckin/mbkin)**17)/(31665385902150*
           mbkin) - (280683483426448*mckin**2*(1 - mckin/mbkin)**17)/
          (5277564317025*mbkin**2) - (12824602615096352*(1 - mckin/mbkin)**18)/
          208260807279525 + (48474483141979*mbkin*(1 - mckin/mbkin)**18)/
          (3247731887400*mckin) + (742117024288296779*mckin*
           (1 - mckin/mbkin)**18)/(7219707985690200*mbkin) -
         (394762574340496*mckin**2*(1 - mckin/mbkin)**18)/(7032183102945*
           mbkin**2) - (730440730648352*(1 - mckin/mbkin)**19)/
          133611478955955 + (22826272832761*mbkin*(1 - mckin/mbkin)**19)/
          (1522637936820*mckin) - (5090258841705703*mckin*(1 - mckin/mbkin)**
            19)/(534445915823820*mbkin) - (232006051154951*mbkin*np.pi**2)/
          (350620189920*mckin) - (43575599069928971*mckin*np.pi**2)/
          (876550474800*mbkin) + (58284316282323413*mckin**2*np.pi**2)/
          (328706428050*mbkin**2) - (12144003187406957*mckin**3*np.pi**2)/
          (28897268400*mbkin**3) + (1330601589944303*mckin**4*np.pi**2)/
          (1945008450*mbkin**4) - (390383485900897*mckin**5*np.pi**2)/
          (525404880*mbkin**5) + (162395799073693*mckin**6*np.pi**2)/
          (383107725*mbkin**6) + (8402985200737*mckin**7*np.pi**2)/
          (54729675*mbkin**7) - (241306387804093*mckin**8*np.pi**2)/
          (383107725*mbkin**8) + (13737254822946883*mckin**9*np.pi**2)/
          (18389170800*mbkin**9) - (44667047662109*mckin**10*np.pi**2)/
          (80270190*mbkin**10) + (58165204092298427*mckin**11*np.pi**2)/
          (202280878800*mbkin**11) - (195208550920151*mckin**12*np.pi**2)/
          (1878322446*mbkin**12) + (9492936773307287*mckin**13*np.pi**2)/
          (375664489200*mbkin**13) - (17530308580702*mckin**14*np.pi**2)/
          (4695806115*mbkin**14) + (2545509950573*mckin**15*np.pi**2)/
          (10017719712*mbkin**15) - (64*mbkin*(1 - mckin/mbkin)**4*np.pi**2)/
          (3*mckin) + (64*mckin*(1 - mckin/mbkin)**4*np.pi**2)/(3*mbkin) +
         (192*mbkin*(1 - mckin/mbkin)**5*np.pi**2)/(5*mckin) -
         (192*mckin*(1 - mckin/mbkin)**5*np.pi**2)/(5*mbkin) -
         (85984*mbkin*(1 - mckin/mbkin)**6*np.pi**2)/(2025*mckin) +
         (85984*mckin*(1 - mckin/mbkin)**6*np.pi**2)/(2025*mbkin) +
         (2436352*mbkin*(1 - mckin/mbkin)**7*np.pi**2)/(99225*mckin) -
         (2436352*mckin*(1 - mckin/mbkin)**7*np.pi**2)/(99225*mbkin) -
         (161092*mbkin*(1 - mckin/mbkin)**8*np.pi**2)/(33075*mckin) +
         (161092*mckin*(1 - mckin/mbkin)**8*np.pi**2)/(33075*mbkin) -
         (162508*mbkin*(1 - mckin/mbkin)**9*np.pi**2)/(178605*mckin) +
         (162508*mckin*(1 - mckin/mbkin)**9*np.pi**2)/(178605*mbkin) -
         (16172*mbkin*(1 - mckin/mbkin)**10*np.pi**2)/(25515*mckin) +
         (16172*mckin*(1 - mckin/mbkin)**10*np.pi**2)/(25515*mbkin) -
         (4212652*mbkin*(1 - mckin/mbkin)**11*np.pi**2)/(7203735*mckin) +
         (4212652*mckin*(1 - mckin/mbkin)**11*np.pi**2)/(7203735*mbkin) -
         (160192*mbkin*(1 - mckin/mbkin)**12*np.pi**2)/(280665*mckin) +
         (160192*mckin*(1 - mckin/mbkin)**12*np.pi**2)/(280665*mbkin) -
         (42218656*mbkin*(1 - mckin/mbkin)**13*np.pi**2)/(74536605*mckin) +
         (42218656*mckin*(1 - mckin/mbkin)**13*np.pi**2)/(74536605*mbkin) -
         (76545439*mbkin*(1 - mckin/mbkin)**14*np.pi**2)/(135270135*mckin) +
         (76545439*mckin*(1 - mckin/mbkin)**14*np.pi**2)/(135270135*mbkin) -
         (493125016*mbkin*(1 - mckin/mbkin)**15*np.pi**2)/(869593725*mckin) +
         (493125016*mckin*(1 - mckin/mbkin)**15*np.pi**2)/(869593725*mbkin) -
         (1979593631*mbkin*(1 - mckin/mbkin)**16*np.pi**2)/(3478374900*mckin) +
         (1979593631*mckin*(1 - mckin/mbkin)**16*np.pi**2)/(3478374900*mbkin) -
         (1340564003491*mbkin*(1 - mckin/mbkin)**17*np.pi**2)/
          (2345584140900*mckin) + (1340564003491*mckin*(1 - mckin/mbkin)**17*
           np.pi**2)/(2345584140900*mbkin) - (48474483141979*mbkin*
           (1 - mckin/mbkin)**18*np.pi**2)/(84441029072400*mckin) +
         (48474483141979*mckin*(1 - mckin/mbkin)**18*np.pi**2)/
          (84441029072400*mbkin) - (22826272832761*mbkin*(1 - mckin/mbkin)**19*
           np.pi**2)/(39588586357320*mckin) + (22826272832761*mckin*
           (1 - mckin/mbkin)**19*np.pi**2)/(39588586357320*mbkin) -
         (2*mckin**2*np.pi**4)/mbkin**2 + (18*mckin**4*np.pi**4)/mbkin**4 -
         (14*mckin**6*np.pi**4)/mbkin**6 + (2*mckin**8*np.pi**4)/mbkin**8 -
         (60736*mbkin*np.pi**2*np.log(2))/(31185*mckin) -
         (2590528*mckin*np.pi**2*np.log(2))/(31185*mbkin) +
         (60829324*mckin**2*np.pi**2*np.log(2))/(243243*mbkin**2) -
         (604864*mckin**3*np.pi**2*np.log(2))/(1215*mbkin**3) +
         (1913036*mckin**4*np.pi**2*np.log(2))/(2835*mbkin**4) -
         (724928*mckin**5*np.pi**2*np.log(2))/(1215*mbkin**5) +
         (123236*mckin**6*np.pi**2*np.log(2))/(567*mbkin**6) +
         (18112*mckin**7*np.pi**2*np.log(2))/(63*mbkin**7) -
         (118796*mckin**8*np.pi**2*np.log(2))/(189*mbkin**8) +
         (5545024*mckin**9*np.pi**2*np.log(2))/(8505*mbkin**9) -
         (2010716*mckin**10*np.pi**2*np.log(2))/(4455*mbkin**10) +
         (20741824*mckin**11*np.pi**2*np.log(2))/(93555*mbkin**11) -
         (205988*mckin**12*np.pi**2*np.log(2))/(2673*mbkin**12) +
         (7358528*mckin**13*np.pi**2*np.log(2))/(405405*mbkin**13) -
         (3172*mckin**14*np.pi**2*np.log(2))/(1215*mbkin**14) +
         (1280*mckin**15*np.pi**2*np.log(2))/(7371*mbkin**15) -
         (12576.63417994688*mckin**2*np.log(mckin**2/mbkin**2))/mbkin**2 -
         (370688*mckin**3*np.log(mckin**2/mbkin**2))/(81*mbkin**3) +
         (17153.02924167527*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4 +
         (6332*mckin**2*np.pi**2*np.log(mckin**2/mbkin**2))/(9*mbkin**2) +
         (4096*mckin**3*np.pi**2*np.log(mckin**2/mbkin**2))/(27*mbkin**3) -
         (23092*mckin**4*np.pi**2*np.log(mckin**2/mbkin**2))/(27*mbkin**4) -
         (12*mckin**2*np.pi**4*np.log(mckin**2/mbkin**2))/mbkin**2 +
         (12*mckin**4*np.pi**4*np.log(mckin**2/mbkin**2))/mbkin**4 +
         1545.7862250201865*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
           mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4) +
         (1280*mbkin*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/
          (243*mckin) - (26305*np.pi**2*(1 - (8*mckin**2)/mbkin**2 +
            (8*mckin**6)/mbkin**6 - mckin**8/mbkin**8 -
            (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/324 +
         (5*np.pi**4*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4))/4 +
         (524288*(1 - mckin/mbkin)**5*(np.log(2) + np.log(1 - mckin/mbkin)))/1215 -
         (1048576*mckin*(1 - mckin/mbkin)**5*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1215*mbkin) + (524288*mckin**2*(1 - mckin/mbkin)**5*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(1215*mbkin**2) -
         (524288*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin)))/3645 -
         (53248*mbkin*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (135*mckin) + (200704*mckin*(1 - mckin/mbkin)**6*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(243*mbkin) -
         (1048576*mckin**2*(1 - mckin/mbkin)**6*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(3645*mbkin**2) -
         (524288*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)))/25515 +
         (212992*mbkin*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (945*mckin) - (1359872*mckin*(1 - mckin/mbkin)**7*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(5103*mbkin) +
         (524288*mckin**2*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (8505*mbkin**2) + (65536*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin)))/1215 -
         (13312*mbkin*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (315*mckin) - (142336*mckin*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(2835*mbkin) +
         (65536*mckin**2*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (1701*mbkin**2) + (753664*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin)))/15309 -
         (13312*mbkin*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (567*mckin) - (338944*mckin*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(5103*mbkin) +
         (622592*mckin**2*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (15309*mbkin**2) + (4063232*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin)))/76545 -
         (63232*mbkin*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (2835*mckin) - (5796608*mckin*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(76545*mbkin) +
         (32768*mckin**2*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (729*mbkin**2) + (770048*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin)))/13365 -
         (6656*mbkin*(1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (297*mckin) - (3394048*mckin*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(40095*mbkin) +
         (180224*mckin**2*(1 - mckin/mbkin)**11*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(3645*mbkin**2) +
         (1495040*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)))/
          24057 - (9152*mbkin*(1 - mckin/mbkin)**12*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(405*mckin) -
         (3747008*mckin*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (40095*mbkin) + (6483968*mckin**2*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(120285*mbkin**2) +
         (104144896*(1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1563705 - (101312*mbkin*(1 - mckin/mbkin)**13*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(4455*mckin) -
         (159761344*mckin*(1 - mckin/mbkin)**13*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(1563705*mbkin) +
         (6078464*mckin**2*(1 - mckin/mbkin)**13*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(104247*mbkin**2) +
         (17268736*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)))/
          243243 - (6784*mbkin*(1 - mckin/mbkin)**14*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(297*mckin) -
         (7351424*mckin*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (66339*mbkin) + (45727744*mckin**2*(1 - mckin/mbkin)**14*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(729729*mbkin**2) +
         (117809152*(1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin)))/
          1563705 - (714496*mbkin*(1 - mckin/mbkin)**15*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(31185*mckin) -
         (145231616*mckin*(1 - mckin/mbkin)**15*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(1216215*mbkin) +
         (733208576*mckin**2*(1 - mckin/mbkin)**15*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(10945935*mbkin**2) +
         (79261184*(1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)))/
          995085 - (716024*mbkin*(1 - mckin/mbkin)**16*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(31185*mckin) -
         (1400770552*mckin*(1 - mckin/mbkin)**16*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(10945935*mbkin) +
         (86691328*mckin**2*(1 - mckin/mbkin)**16*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(1216215*mbkin**2) +
         (1735555072*(1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin)))/
          20675655 - (1354552*mbkin*(1 - mckin/mbkin)**17*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(58905*mckin) -
         (940759912*mckin*(1 - mckin/mbkin)**17*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(6891885*mbkin) +
         (223167488*mckin**2*(1 - mckin/mbkin)**17*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(2953665*mbkin**2) +
         (180384256*(1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin)))/
          2044845 - (1743496*mbkin*(1 - mckin/mbkin)**18*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(75735*mckin) -
         (1284952024*mckin*(1 - mckin/mbkin)**18*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(8860995*mbkin) +
         (38578688*mckin**2*(1 - mckin/mbkin)**18*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(483327*mbkin**2) +
         (77157376*(1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)))/
          9183213 - (602792*mbkin*(1 - mckin/mbkin)**19*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(26163*mckin) +
         (134422616*mckin*(1 - mckin/mbkin)**19*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(9183213*mbkin) +
         (2048*mbkin*(1 - mckin/mbkin)**6*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(135*mckin) -
         (2048*mckin*(1 - mckin/mbkin)**6*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(135*mbkin) -
         (8192*mbkin*(1 - mckin/mbkin)**7*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(945*mckin) +
         (8192*mckin*(1 - mckin/mbkin)**7*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(945*mbkin) +
         (512*mbkin*(1 - mckin/mbkin)**8*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (315*mckin) - (512*mckin*(1 - mckin/mbkin)**8*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(315*mbkin) +
         (512*mbkin*(1 - mckin/mbkin)**9*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)))/
          (567*mckin) - (512*mckin*(1 - mckin/mbkin)**9*np.pi**2*
           (np.log(2) + np.log(1 - mckin/mbkin)))/(567*mbkin) +
         (2432*mbkin*(1 - mckin/mbkin)**10*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(2835*mckin) -
         (2432*mckin*(1 - mckin/mbkin)**10*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(2835*mbkin) +
         (256*mbkin*(1 - mckin/mbkin)**11*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(297*mckin) -
         (256*mckin*(1 - mckin/mbkin)**11*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(297*mbkin) +
         (352*mbkin*(1 - mckin/mbkin)**12*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(405*mckin) -
         (352*mckin*(1 - mckin/mbkin)**12*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(405*mbkin) +
         (50656*mbkin*(1 - mckin/mbkin)**13*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(57915*mckin) -
         (50656*mckin*(1 - mckin/mbkin)**13*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(57915*mbkin) +
         (3392*mbkin*(1 - mckin/mbkin)**14*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(3861*mckin) -
         (3392*mckin*(1 - mckin/mbkin)**14*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(3861*mbkin) +
         (357248*mbkin*(1 - mckin/mbkin)**15*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(405405*mckin) -
         (357248*mckin*(1 - mckin/mbkin)**15*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(405405*mbkin) +
         (358012*mbkin*(1 - mckin/mbkin)**16*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(405405*mckin) -
         (358012*mckin*(1 - mckin/mbkin)**16*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(405405*mbkin) +
         (677276*mbkin*(1 - mckin/mbkin)**17*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(765765*mckin) -
         (677276*mckin*(1 - mckin/mbkin)**17*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(765765*mbkin) +
         (871748*mbkin*(1 - mckin/mbkin)**18*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(984555*mckin) -
         (871748*mckin*(1 - mckin/mbkin)**18*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(984555*mbkin) +
         (301396*mbkin*(1 - mckin/mbkin)**19*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(340119*mckin) -
         (301396*mckin*(1 - mckin/mbkin)**19*np.pi**2*(np.log(2) +
            np.log(1 - mckin/mbkin)))/(340119*mbkin) +
         (1024*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/15 -
         (1024*mbkin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(105*mckin) -
         (4096*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(21*mbkin) +
         (4096*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(15*mbkin**2) -
         (2048*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(15*mbkin**3) -
         (2048*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(15*mbkin**4) +
         (4096*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(15*mbkin**5) -
         (4096*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(21*mbkin**6) +
         (1024*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(15*mbkin**7) -
         (1024*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 4*np.log(2) +
            4*np.log(1 - mckin/mbkin)))/(105*mbkin**8) +
         (40192*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/8505 -
         (40192*mbkin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(76545*mckin) -
         (40192*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(2187*mbkin) +
         (200960*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(5103*mbkin**2) -
         (80384*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(1701*mbkin**3) +
         (80384*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(3645*mbkin**4) +
         (80384*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(3645*mbkin**5) -
         (80384*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(1701*mbkin**6) +
         (200960*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(5103*mbkin**7) -
         (40192*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(2187*mbkin**8) +
         (40192*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(8505*mbkin**9) -
         (40192*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 5*np.log(2) +
            5*np.log(1 - mckin/mbkin)))/(76545*mbkin**10) +
         (514496*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 6*np.log(2) +
            6*np.log(1 - mckin/mbkin)))/76545 -
         (514496*mbkin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 6*np.log(2) +
            6*np.log(1 - mckin/mbkin)))/(841995*mckin) -
         (1028992*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 6*np.log(2) +
            6*np.log(1 - mckin/mbkin)))/(31185*mbkin) +
         (1028992*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(10935*mbkin**2) -
         (2572480*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(15309*mbkin**3) +
         (514496*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 6*np.log(2) +
            6*np.log(1 - mckin/mbkin)))/(2835*mbkin**4) -
         (2057984*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(25515*mbkin**5) -
         (2057984*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(25515*mbkin**6) +
         (514496*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 6*np.log(2) +
            6*np.log(1 - mckin/mbkin)))/(2835*mbkin**7) -
         (2572480*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(15309*mbkin**8) +
         (1028992*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(10935*mbkin**9) -
         (1028992*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(31185*mbkin**10) +
         (514496*mckin**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(76545*mbkin**11) -
         (514496*mckin**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 6*np.log(2) + 6*np.log(1 - mckin/mbkin)))/(841995*mbkin**12) +
         (372763648*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/893025 -
         (8602112*mbkin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (99225*mckin) - (63087616*mckin*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(127575*mbkin) -
         (40137728*mckin**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (42525*mbkin**2) + (2621440*mckin**3*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(729*mbkin**3) -
         (613590016*mckin**4*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (127575*mbkin**4) + (143363072*mckin**5*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(42525*mbkin**5) -
         (1106766848*mckin**6*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (893025*mbkin**6) + (169169408*mckin**7*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(893025*mbkin**7) -
         (4096*np.pi**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/315 +
         (2048*mbkin*np.pi**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (945*mckin) + (4096*mckin*np.pi**2*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(135*mbkin) -
         (4096*mckin**2*np.pi**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (135*mbkin**2) + (4096*mckin**4*np.pi**2*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(135*mbkin**4) -
         (4096*mckin**5*np.pi**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) + (4096*mckin**6*np.pi**2*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(315*mbkin**6) -
         (2048*mckin**7*np.pi**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/
          (945*mbkin**7) + (462032*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/56133 -
         (462032*mbkin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(729729*mckin) -
         (462032*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(9477*mbkin) +
         (462032*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(2673*mbkin**2) -
         (3234224*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(8019*mbkin**3) +
         (462032*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(729*mbkin**4) -
         (462032*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(729*mbkin**5) +
         (462032*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(1701*mbkin**6) +
         (462032*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(1701*mbkin**7) -
         (462032*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(729*mbkin**8) +
         (462032*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(729*mbkin**9) -
         (3234224*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(8019*mbkin**10) +
         (462032*mckin**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(2673*mbkin**11) -
         (462032*mckin**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(9477*mbkin**12) +
         (462032*mckin**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(56133*mbkin**13) -
         (462032*mckin**14*(np.log(2) + np.log(1 - mckin/mbkin))*
           (1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin)))/(729729*mbkin**14) -
         (2048*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/35 +
         (1024*mbkin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(105*mckin) +
         (2048*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(15*mbkin) -
         (2048*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(15*mbkin**2) +
         (2048*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(15*mbkin**4) -
         (2048*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(15*mbkin**5) +
         (2048*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(35*mbkin**6) -
         (1024*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin)))/(105*mbkin**7) -
         (33446144*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/127575 +
         (164608*mbkin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (3675*mckin) + (86598656*mckin*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(178605*mbkin) +
         (23501824*mckin**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (127575*mbkin**2) - (287002112*mckin**3*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(127575*mbkin**3) +
         (538751488*mckin**4*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (127575*mbkin**4) - (527000576*mckin**5*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(127575*mbkin**5) +
         (416900096*mckin**6*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (178605*mbkin**6) - (92428544*mckin**7*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(127575*mbkin**7) +
         (85874944*mckin**8*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (893025*mbkin**8) + (1024*np.pi**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/135 - (1024*mbkin*np.pi**2*
           (1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/(945*mckin) -
         (4096*mckin*np.pi**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (189*mbkin) + (4096*mckin**2*np.pi**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(135*mbkin**2) -
         (2048*mckin**3*np.pi**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (135*mbkin**3) - (2048*mckin**4*np.pi**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(135*mbkin**4) +
         (4096*mckin**5*np.pi**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) - (4096*mckin**6*np.pi**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(189*mbkin**6) +
         (1024*mckin**7*np.pi**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin)))/
          (135*mbkin**7) - (1024*mckin**8*np.pi**2*(1 + 8*np.log(2) +
            8*np.log(1 - mckin/mbkin)))/(945*mbkin**8) +
         (356614144*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/24111675 -
         (70381568*mbkin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (24111675*mckin) - (1568768*mckin*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(893025*mbkin) -
         (1351155712*mckin**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (8037225*mbkin**2) + (684990464*mckin**3*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(1148175*mbkin**3) -
         (262144*mckin**4*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (243*mbkin**4) + (1379393536*mckin**5*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(1148175*mbkin**5) -
         (6906380288*mckin**6*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (8037225*mbkin**6) + (345632768*mckin**7*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(893025*mbkin**7) -
         (2420998144*mckin**8*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (24111675*mbkin**8) + (276819968*mckin**9*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(24111675*mbkin**9) -
         (4096*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/2835 +
         (512*mbkin*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (2835*mckin) + (512*mckin*np.pi**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(105*mbkin) -
         (8192*mckin**2*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (945*mbkin**2) + (1024*mckin**3*np.pi**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(135*mbkin**3) -
         (1024*mckin**5*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (135*mbkin**5) + (8192*mckin**6*np.pi**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(945*mbkin**6) -
         (512*mckin**7*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (105*mbkin**7) + (4096*mckin**8*np.pi**2*(1 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(2835*mbkin**8) -
         (512*mckin**9*np.pi**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**9) + (321536*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin)))/76545 -
         (40192*mbkin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(76545*mckin) -
         (40192*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(2835*mbkin) +
         (643072*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(25515*mbkin**2) -
         (80384*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(3645*mbkin**3) +
         (80384*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(3645*mbkin**5) -
         (643072*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(25515*mbkin**6) +
         (40192*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(2835*mbkin**7) -
         (321536*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(76545*mbkin**8) +
         (40192*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 9*np.log(2) +
            9*np.log(1 - mckin/mbkin)))/(76545*mbkin**9) +
         (5574656*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/297675 -
         (61640704*mbkin*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (24111675*mckin) - (32149504*mckin*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(688905*mbkin) -
         (290816*mckin**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (321489*mbkin**2) + (151969792*mckin**3*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(535815*mbkin**3) -
         (908910592*mckin**4*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (1148175*mbkin**4) + (1361911808*mckin**5*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(1148175*mbkin**5) -
         (604971008*mckin**6*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (535815*mbkin**6) + (226791424*mckin**7*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(321489*mbkin**7) -
         (194351104*mckin**8*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (688905*mbkin**8) + (176328704*mckin**9*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(2679075*mbkin**9) -
         (164859904*mckin**10*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (24111675*mbkin**10) - (256*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/315 +
         (256*mbkin*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (2835*mckin) + (256*mckin*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(81*mbkin) -
         (1280*mckin**2*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (189*mbkin**2) + (512*mckin**3*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(63*mbkin**3) -
         (512*mckin**4*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (135*mbkin**4) - (512*mckin**5*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(135*mbkin**5) +
         (512*mckin**6*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (63*mbkin**6) - (1280*mckin**7*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(189*mbkin**7) +
         (256*mckin**8*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (81*mbkin**8) - (256*mckin**9*np.pi**2*(1 + 10*np.log(2) +
            10*np.log(1 - mckin/mbkin)))/(315*mbkin**9) +
         (256*mckin**10*np.pi**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**10) + (3156735616*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/194500845 -
         (1937914688*mbkin*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (972504225*mckin) - (4156189952*mckin*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(88409475*mbkin) +
         (280360576*mckin**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (17681895*mbkin**2) + (1657554112*mckin**3*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(5893965*mbkin**3) -
         (28203029248*mckin**4*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (29469825*mbkin**4) + (1245184*mckin**5*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(729*mbkin**5) -
         (58088221952*mckin**6*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (29469825*mbkin**6) + (9128852288*mckin**7*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(5893965*mbkin**7) -
         (14662235776*mckin**8*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (17681895*mbkin**8) + (25729002752*mckin**9*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(88409475*mbkin**9) -
         (11785860736*mckin**10*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (194500845*mbkin**10) + (5533383488*mckin**11*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(972504225*mbkin**11) -
         (4864*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/6237 +
         (2432*mbkin*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (31185*mckin) + (9728*mckin*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(2835*mbkin) -
         (4864*mckin**2*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (567*mbkin**2) + (2432*mckin**3*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(189*mbkin**3) -
         (9728*mckin**4*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (945*mbkin**4) + (9728*mckin**6*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(945*mbkin**6) -
         (2432*mckin**7*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (189*mbkin**7) + (4864*mckin**8*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(567*mbkin**8) -
         (9728*mckin**9*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (2835*mbkin**9) + (4864*mckin**10*np.pi**2*(1 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(6237*mbkin**10) -
         (2432*mckin**11*np.pi**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/
          (31185*mbkin**11) + (163456*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin)))/56133 -
         (81728*mbkin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(280665*mckin) -
         (326912*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(25515*mbkin) +
         (163456*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(5103*mbkin**2) -
         (81728*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(1701*mbkin**3) +
         (326912*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(8505*mbkin**4) -
         (326912*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(8505*mbkin**6) +
         (81728*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(1701*mbkin**7) -
         (163456*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(5103*mbkin**8) +
         (326912*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(25515*mbkin**9) -
         (163456*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(56133*mbkin**10) +
         (81728*mckin**11*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 11*np.log(2) +
            11*np.log(1 - mckin/mbkin)))/(280665*mbkin**11) +
         (4365178528*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/265228425 -
         (5268346528*mbkin*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (2917512675*mckin) - (6121205056*mckin*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(108056025*mbkin) +
         (2021109056*mckin**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (37889775*mbkin**2) + (2679531872*mckin**3*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(10609137*mbkin**3) -
         (11289733472*mckin**4*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (9823275*mbkin**4) + (217362965888*mckin**5*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(88409475*mbkin**5) -
         (299249130112*mckin**6*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (88409475*mbkin**6) + (31761274528*mckin**7*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(9823275*mbkin**7) -
         (23151072928*mckin**8*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (10609137*mbkin**8) + (38921973056*mckin**9*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(37889775*mbkin**9) -
         (11607292352*mckin**10*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (36018675*mbkin**10) + (16106362528*mckin**11*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(265228425*mbkin**11) -
         (15203194528*mckin**12*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (2917512675*mbkin**12) - (64*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/81 +
         (64*mbkin*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (891*mckin) + (128*mckin*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(33*mbkin) -
         (896*mckin**2*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (81*mbkin**2) + (1600*mckin**3*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**3) -
         (64*mckin**4*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (3*mbkin**4) + (256*mckin**5*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(27*mbkin**5) +
         (256*mckin**6*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (27*mbkin**6) - (64*mckin**7*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(3*mbkin**7) +
         (1600*mckin**8*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (81*mbkin**8) - (896*mckin**9*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**9) +
         (128*mckin**10*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (33*mbkin**10) - (64*mckin**11*np.pi**2*(1 + 12*np.log(2) +
            12*np.log(1 - mckin/mbkin)))/(81*mbkin**11) +
         (64*mckin**12*np.pi**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin)))/
          (891*mbkin**12) + (2006891792896*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/117395152875 -
         (594743876224*mbkin*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (352185458625*mckin) - (371493649024*mckin*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(5418237825*mbkin) +
         (2818395203584*mckin**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (27091189125*mbkin**2) + (149423547776*mckin**3*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(820945125*mbkin**3) -
         (3202280175104*mckin**4*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (2462835375*mbkin**4) + (2754009531776*mckin**5*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(820945125*mbkin**5) -
         (3964928*mckin**6*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (729*mbkin**6) + (5059748420224*mckin**7*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(820945125*mbkin**7) -
         (12425235728896*mckin**8*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (2462835375*mbkin**8) + (2455162436224*mckin**9*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(820945125*mbkin**9) -
         (34073427011584*mckin**10*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (27091189125*mbkin**10) + (1934245239424*mckin**11*
           (1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(5418237825*mbkin**11) -
         (7216063760896*mckin**12*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (117395152875*mbkin**12) + (1710995012224*mckin**13*
           (1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(352185458625*
           mbkin**13) - (1408*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          1755 + (352*mbkin*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (5265*mckin) + (352*mckin*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(81*mbkin) -
         (5632*mckin**2*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (405*mbkin**2) + (3872*mckin**3*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(135*mbkin**3) -
         (15488*mckin**4*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (405*mbkin**4) + (3872*mckin**5*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(135*mbkin**5) -
         (3872*mckin**7*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (135*mbkin**7) + (15488*mckin**8*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(405*mbkin**8) -
         (3872*mckin**9*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (135*mbkin**9) + (5632*mckin**10*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(405*mbkin**10) -
         (352*mckin**11*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (81*mbkin**11) + (1408*mckin**12*np.pi**2*(1 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(1755*mbkin**12) -
         (352*mckin**13*np.pi**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/
          (5265*mbkin**13) + (9782848*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/2606175 -
         (2445712*mbkin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(7818525*mckin) -
         (2445712*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(120285*mbkin) +
         (39131392*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(601425*mbkin**2) -
         (2445712*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(18225*mbkin**3) +
         (9782848*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(54675*mbkin**4) -
         (2445712*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(18225*mbkin**5) +
         (2445712*mckin**7*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(18225*mbkin**7) -
         (9782848*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(54675*mbkin**8) +
         (2445712*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 13*np.log(2) +
            13*np.log(1 - mckin/mbkin)))/(18225*mbkin**9) -
         (39131392*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(601425*mbkin**10) +
         (2445712*mckin**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(120285*mbkin**11) -
         (9782848*mckin**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(2606175*mbkin**12) +
         (2445712*mckin**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin)))/(7818525*mbkin**13) +
         (223994560288*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          12642554925 - (261439475488*mbkin*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(164353214025*mckin) -
         (15721210208*mckin*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (194041575*mbkin) + (33059392096*mckin**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(200675475*mbkin**2) +
         (16722751712*mckin**3*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (258011325*mbkin**3) - (225344422112*mckin**4*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(164189025*mbkin**4) +
         (712128319712*mckin**5*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (164189025*mbkin**5) - (3146047807712*mckin**6*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(383107725*mbkin**6) +
         (4155710656288*mckin**7*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (383107725*mbkin**7) - (1721791168288*mckin**8*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(164189025*mbkin**8) +
         (1235007270688*mckin**9*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (164189025*mbkin**9) - (1026385600288*mckin**10*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(258011325*mbkin**10) +
         (910484672288*mckin**11*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (602026425*mbkin**11) - (836729536288*mckin**12*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(2134457325*mbkin**12) +
         (60436022176*mckin**13*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (972504225*mbkin**13) - (748223373088*mckin**14*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(164353214025*mbkin**14) -
         (25328*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/31185 +
         (25328*mbkin*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (405405*mckin) + (25328*mckin*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(5265*mbkin) -
         (25328*mckin**2*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (1485*mbkin**2) + (177296*mckin**3*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(4455*mbkin**3) -
         (25328*mckin**4*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (405*mbkin**4) + (25328*mckin**5*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(405*mbkin**5) -
         (25328*mckin**6*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (945*mbkin**6) - (25328*mckin**7*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(945*mbkin**7) +
         (25328*mckin**8*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (405*mbkin**8) - (25328*mckin**9*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(405*mbkin**9) +
         (177296*mckin**10*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (4455*mbkin**10) - (25328*mckin**11*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(1485*mbkin**11) +
         (25328*mckin**12*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (5265*mbkin**12) - (25328*mckin**13*np.pi**2*(1 + 14*np.log(2) +
            14*np.log(1 - mckin/mbkin)))/(31185*mbkin**13) +
         (25328*mckin**14*np.pi**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin)))/
          (405405*mbkin**14) + (6431321516674*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/352185458625 -
         (3704597206337*mbkin*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (2465298210375*mckin) - (5127490988674*mckin*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(54784404675*mbkin) +
         (3302128249474*mckin**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (14087418345*mbkin**2) - (564084140674*mckin**3*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(5418237825*mbkin**3) -
         (3999322707326*mckin**4*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (3010132125*mbkin**4) + (13126136403326*mckin**5*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(2462835375*mbkin**5) -
         (40506577491326*mckin**6*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (3447969525*mbkin**6) + (4341760*mckin**7*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(243*mbkin**7) -
         (69015186860674*mckin**8*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (3447969525*mbkin**8) + (41634745772674*mckin**9*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(2462835375*mbkin**9) -
         (32507932076674*mckin**10*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (3010132125*mbkin**10) + (27944525228674*mckin**11*
           (1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(5418237825*mbkin**11) -
         (25206481119874*mckin**12*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (14087418345*mbkin**12) + (23381118380674*mckin**13*
           (1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(54784404675*
           mbkin**13) - (22077287852674*mckin**14*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(352185458625*mbkin**14) +
         (10549707478337*mckin**15*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (2465298210375*mbkin**15) - (47488*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/57915 +
         (3392*mbkin*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (57915*mckin) + (6784*mckin*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(1287*mbkin) -
         (237440*mckin**2*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (11583*mbkin**2) + (47488*mckin**3*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(891*mbkin**3) -
         (47488*mckin**4*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (495*mbkin**4) + (47488*mckin**5*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(405*mbkin**5) -
         (6784*mckin**6*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (81*mbkin**6) + (6784*mckin**8*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(81*mbkin**8) -
         (47488*mckin**9*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (405*mbkin**9) + (47488*mckin**10*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(495*mbkin**10) -
         (47488*mckin**11*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (891*mbkin**11) + (237440*mckin**12*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(11583*mbkin**12) -
         (6784*mckin**13*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (1287*mbkin**13) + (47488*mckin**14*np.pi**2*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(57915*mbkin**14) -
         (3392*mckin**15*np.pi**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/
          (57915*mbkin**15) + (34821824*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/7818525 -
         (17410912*mbkin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(54729675*mckin) -
         (34821824*mckin*(np.log(2) + np.log(1 - mckin/mbkin))*(2 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin)))/(1216215*mbkin) +
         (34821824*mckin**2*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(312741*mbkin**2) -
         (34821824*mckin**3*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(120285*mbkin**3) +
         (34821824*mckin**4*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(66825*mbkin**4) -
         (34821824*mckin**5*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(54675*mbkin**5) +
         (34821824*mckin**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(76545*mbkin**6) -
         (34821824*mckin**8*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(76545*mbkin**8) +
         (34821824*mckin**9*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(54675*mbkin**9) -
         (34821824*mckin**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(66825*mbkin**10) +
         (34821824*mckin**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(120285*mbkin**11) -
         (34821824*mckin**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(312741*mbkin**12) +
         (34821824*mckin**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(1216215*mbkin**13) -
         (34821824*mckin**14*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(7818525*mbkin**14) +
         (17410912*mckin**15*(np.log(2) + np.log(1 - mckin/mbkin))*
           (2 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin)))/(54729675*mbkin**15) -
         (5715968*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/2189187 +
         (91455488*mckin*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (2189187*mbkin) - (228638720*mckin**2*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(729729*mbkin**2) +
         (457277440*mckin**3*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (312741*mbkin**3) - (114319360*mckin**4*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(24057*mbkin**4) +
         (91455488*mckin**5*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (8019*mbkin**5) - (45727744*mckin**6*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(2187*mbkin**6) +
         (457277440*mckin**7*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (15309*mbkin**7) - (57159680*mckin**8*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(1701*mbkin**8) +
         (457277440*mckin**9*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (15309*mbkin**9) - (45727744*mckin**10*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(2187*mbkin**10) +
         (91455488*mckin**11*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (8019*mbkin**11) - (114319360*mckin**12*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(24057*mbkin**12) +
         (457277440*mckin**13*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (312741*mbkin**13) - (228638720*mckin**14*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(729729*mbkin**14) +
         (91455488*mckin**15*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin)))/
          (2189187*mbkin**15) - (5715968*mckin**16*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin)))/(2189187*mbkin**16) -
         (91651072*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/37216179 +
         (91651072*mckin*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (2189187*mbkin) - (733208576*mckin**2*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(2189187*mbkin**2) +
         (3666042880*mckin**3*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (2189187*mbkin**3) - (1833021440*mckin**4*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(312741*mbkin**4) +
         (366604288*mckin**5*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (24057*mbkin**5) - (733208576*mckin**6*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(24057*mbkin**6) +
         (733208576*mckin**7*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (15309*mbkin**7) - (916510720*mckin**8*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(15309*mbkin**8) +
         (916510720*mckin**9*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (15309*mbkin**9) - (733208576*mckin**10*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(15309*mbkin**10) +
         (733208576*mckin**11*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (24057*mbkin**11) - (366604288*mckin**12*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(24057*mbkin**12) +
         (1833021440*mckin**13*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (312741*mbkin**13) - (3666042880*mckin**14*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(2189187*mbkin**14) +
         (733208576*mckin**15*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (2189187*mbkin**15) - (91651072*mckin**16*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin)))/(2189187*mbkin**16) +
         (91651072*mckin**17*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin)))/
          (37216179*mbkin**17) - (86691328*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/37216179 +
         (173382656*mckin*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (4135131*mbkin) - (86691328*mckin**2*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(243243*mbkin**2) +
         (1387061248*mckin**3*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (729729*mbkin**3) - (1733826560*mckin**4*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(243243*mbkin**4) +
         (693530624*mckin**5*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**5) - (346765312*mckin**6*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(8019*mbkin**6) +
         (1387061248*mckin**7*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (18711*mbkin**7) - (173382656*mckin**8*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(1701*mbkin**8) +
         (1733826560*mckin**9*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (15309*mbkin**9) - (173382656*mckin**10*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(1701*mbkin**10) +
         (1387061248*mckin**11*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (18711*mbkin**11) - (346765312*mckin**12*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(8019*mbkin**12) +
         (693530624*mckin**13*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**13) - (1733826560*mckin**14*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(243243*mbkin**14) +
         (1387061248*mckin**15*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (729729*mbkin**15) - (86691328*mckin**16*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(243243*mbkin**16) +
         (173382656*mckin**17*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin)))/
          (4135131*mbkin**17) - (86691328*mckin**18*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin)))/(37216179*mbkin**18) -
         (223167488*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/101015343 +
         (223167488*mckin*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (5316597*mbkin) - (223167488*mckin**2*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(590733*mbkin**2) +
         (223167488*mckin**3*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (104247*mbkin**3) - (892669952*mckin**4*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(104247*mbkin**4) +
         (892669952*mckin**5*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (34749*mbkin**5) - (6248689664*mckin**6*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(104247*mbkin**6) +
         (892669952*mckin**7*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (8019*mbkin**7) - (446334976*mckin**8*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(2673*mbkin**8) +
         (446334976*mckin**9*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (2187*mbkin**9) - (446334976*mckin**10*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(2187*mbkin**10) +
         (446334976*mckin**11*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (2673*mbkin**11) - (892669952*mckin**12*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(8019*mbkin**12) +
         (6248689664*mckin**13*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (104247*mbkin**13) - (892669952*mckin**14*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(34749*mbkin**14) +
         (892669952*mckin**15*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (104247*mbkin**15) - (223167488*mckin**16*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(104247*mbkin**16) +
         (223167488*mckin**17*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (590733*mbkin**17) - (223167488*mckin**18*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin)))/(5316597*mbkin**18) +
         (223167488*mckin**19*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin)))/
          (101015343*mbkin**19) - (19289344*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/9183213 +
         (385786880*mckin*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (9183213*mbkin) - (192893440*mckin**2*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(483327*mbkin**2) +
         (385786880*mckin**3*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (161109*mbkin**3) - (96446720*mckin**4*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(9477*mbkin**4) +
         (308629504*mckin**5*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (9477*mbkin**5) - (771573760*mckin**6*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(9477*mbkin**6) +
         (1543147520*mckin**7*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (9477*mbkin**7) - (192893440*mckin**8*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(729*mbkin**8) +
         (771573760*mckin**9*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (2187*mbkin**9) - (848731136*mckin**10*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(2187*mbkin**10) +
         (771573760*mckin**11*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (2187*mbkin**11) - (192893440*mckin**12*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(729*mbkin**12) +
         (1543147520*mckin**13*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (9477*mbkin**13) - (771573760*mckin**14*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(9477*mbkin**14) +
         (308629504*mckin**15*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (9477*mbkin**15) - (96446720*mckin**16*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(9477*mbkin**16) +
         (385786880*mckin**17*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (161109*mbkin**17) - (192893440*mckin**18*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(483327*mbkin**18) +
         (385786880*mckin**19*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin)))/
          (9183213*mbkin**19) - (19289344*mckin**20*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin)))/(9183213*mbkin**20) +
         (167*(-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin))))/47031240592496160 -
         (np.pi**2*(-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin))))/12193284598054560 +
         (-9.854113931163779e19*(1 - mckin/mbkin)**5 - 2.5903365411196256e20*
            (1 - mckin/mbkin)**6 - 1.5679296844106574e21*(1 - mckin/mbkin)**
             7 + 1.602586733095746e21*(1 - mckin/mbkin)**8 -
           3.731382058497059e20*(1 - mckin/mbkin)**9 - 1.442450583699921e20*
            (1 - mckin/mbkin)**10 - 1.0117940133228708e21*(1 - mckin/mbkin)**
             11 - 1.4567466646450717e21*(1 - mckin/mbkin)**12 -
           2.2050041791842524e21*(1 - mckin/mbkin)**13 - 3.10373436948258e21*
            (1 - mckin/mbkin)**14 - 4.17958779756945e21*(1 - mckin/mbkin)**
             15 - 58963096098466560000*(1 - mckin/mbkin)**5*np.pi**2 +
           88444644147699840000*(1 - mckin/mbkin)**6*np.pi**2 -
           65172716793401817600*(1 - mckin/mbkin)**7*np.pi**2 -
           9108973844357587200*(1 - mckin/mbkin)**8*np.pi**2 +
           27694301930794699200*(1 - mckin/mbkin)**9*np.pi**2 +
           9354724595990373600*(1 - mckin/mbkin)**10*np.pi**2 +
           99846396491298256800*(1 - mckin/mbkin)**11*np.pi**2 +
           145282820224320979200*(1 - mckin/mbkin)**12*np.pi**2 +
           221188319752082032350*(1 - mckin/mbkin)**13*np.pi**2 +
           312313023387573082875*(1 - mckin/mbkin)**14*np.pi**2 +
           421384671784085884875*(1 - mckin/mbkin)**15*np.pi**2 +
           23585238439386624000*(1 - mckin/mbkin)**5*np.pi**2*np.log(2) -
           35377857659079936000*(1 - mckin/mbkin)**6*np.pi**2*np.log(2) +
           17408152181452032000*(1 - mckin/mbkin)**7*np.pi**2*np.log(2) -
           2807766480879360000*(1 - mckin/mbkin)**8*np.pi**2*np.log(2) +
           1380485186432352000*(1 - mckin/mbkin)**9*np.pi**2*np.log(2) +
           690242593216176000*(1 - mckin/mbkin)**10*np.pi**2*np.log(2) +
           508375900704672000*(1 - mckin/mbkin)**11*np.pi**2*np.log(2) +
           417442554448920000*(1 - mckin/mbkin)**12*np.pi**2*np.log(2) +
           359888250273552000*(1 - mckin/mbkin)**13*np.pi**2*np.log(2) +
           319023467138376000*(1 - mckin/mbkin)**14*np.pi**2*np.log(2) +
           287976049320960000*(1 - mckin/mbkin)**15*np.pi**2*np.log(2) +
           2156493012354474393600*(1 - mckin/mbkin)**7*(np.log(2) +
             np.log(1 - mckin/mbkin)) - 1114185917132493004800*
            (1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin)) +
           72609864602488012800*(1 - mckin/mbkin)**9*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 63592262841345638400*(1 - mckin/mbkin)**
             10*(np.log(2) + np.log(1 - mckin/mbkin)) + 49568665585965926400*
            (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin)) +
           44918542055279692800*(1 - mckin/mbkin)**12*(np.log(2) +
             np.log(1 - mckin/mbkin)) + 42007093034271805440*(1 - mckin/mbkin)**
             13*(np.log(2) + np.log(1 - mckin/mbkin)) + 39569178342479385600*
            (1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin)) +
           37379682179716836960*(1 - mckin/mbkin)**15*(np.log(2) +
             np.log(1 - mckin/mbkin)) - 53909116432883712000*(1 - mckin/mbkin)**7*
            np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) + 26954558216441856000*
            (1 - mckin/mbkin)**8*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
           4492426369406976000*(1 - mckin/mbkin)**9*np.pi**2*
            (np.log(2) + np.log(1 - mckin/mbkin)) - 2246213184703488000*
            (1 - mckin/mbkin)**10*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
           1939911386789376000*(1 - mckin/mbkin)**11*np.pi**2*
            (np.log(2) + np.log(1 - mckin/mbkin)) - 1786760487832320000*
            (1 - mckin/mbkin)**12*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
           1663061684828544000*(1 - mckin/mbkin)**13*np.pi**2*
            (np.log(2) + np.log(1 - mckin/mbkin)) - 1554088929801408000*
            (1 - mckin/mbkin)**14*np.pi**2*(np.log(2) + np.log(1 - mckin/mbkin)) -
           1456897013155584000*(1 - mckin/mbkin)**15*np.pi**2*
            (np.log(2) + np.log(1 - mckin/mbkin)) - 242591023947976704000*
            (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
           121295511973988352000*(1 - mckin/mbkin)**8*
            (np.log(2) + np.log(1 - mckin/mbkin))**2 + 13061313703646208000*
            (1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
           6530656851823104000*(1 - mckin/mbkin)**10*
            (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7243470295117056000*
            (1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
           7599877016764032000*(1 - mckin/mbkin)**12*
            (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7781163757509542400*
            (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin))**2 +
           7874890507804320000*(1 - mckin/mbkin)**14*
            (np.log(2) + np.log(1 - mckin/mbkin))**2 + 7913392760156083200*
            (1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin))**2)/
          4975011233308116000 + (14048*np.log(2/mus))/9 +
         (4096*mckin*np.log(2/mus))/(3*mbkin) + (13168*mckin**2*np.log(2/mus))/
          (9*mbkin**2) - (2048*mckin**3*np.log(2/mus))/mbkin**3 -
         (29264*mckin**4*np.log(2/mus))/(3*mbkin**4) + (4096*mckin**5*np.log(2/mus))/
          (3*mbkin**5) + (67600*mckin**6*np.log(2/mus))/(9*mbkin**6) -
         (2048*mckin**7*np.log(2/mus))/(3*mbkin**7) - (7024*mckin**8*np.log(2/mus))/
          (9*mbkin**8) - (192*mbkin*(1 - mckin/mbkin)**4*np.log(2/mus))/mckin +
         (192*mckin*(1 - mckin/mbkin)**4*np.log(2/mus))/mbkin +
         (1728*mbkin*(1 - mckin/mbkin)**5*np.log(2/mus))/(5*mckin) -
         (1728*mckin*(1 - mckin/mbkin)**5*np.log(2/mus))/(5*mbkin) -
         (85984*mbkin*(1 - mckin/mbkin)**6*np.log(2/mus))/(225*mckin) +
         (85984*mckin*(1 - mckin/mbkin)**6*np.log(2/mus))/(225*mbkin) +
         (2436352*mbkin*(1 - mckin/mbkin)**7*np.log(2/mus))/(11025*mckin) -
         (2436352*mckin*(1 - mckin/mbkin)**7*np.log(2/mus))/(11025*mbkin) -
         (161092*mbkin*(1 - mckin/mbkin)**8*np.log(2/mus))/(3675*mckin) +
         (161092*mckin*(1 - mckin/mbkin)**8*np.log(2/mus))/(3675*mbkin) -
         (162508*mbkin*(1 - mckin/mbkin)**9*np.log(2/mus))/(19845*mckin) +
         (162508*mckin*(1 - mckin/mbkin)**9*np.log(2/mus))/(19845*mbkin) -
         (16172*mbkin*(1 - mckin/mbkin)**10*np.log(2/mus))/(2835*mckin) +
         (16172*mckin*(1 - mckin/mbkin)**10*np.log(2/mus))/(2835*mbkin) -
         (4212652*mbkin*(1 - mckin/mbkin)**11*np.log(2/mus))/(800415*mckin) +
         (4212652*mckin*(1 - mckin/mbkin)**11*np.log(2/mus))/(800415*mbkin) -
         (160192*mbkin*(1 - mckin/mbkin)**12*np.log(2/mus))/(31185*mckin) +
         (160192*mckin*(1 - mckin/mbkin)**12*np.log(2/mus))/(31185*mbkin) -
         (42218656*mbkin*(1 - mckin/mbkin)**13*np.log(2/mus))/(8281845*mckin) +
         (42218656*mckin*(1 - mckin/mbkin)**13*np.log(2/mus))/(8281845*mbkin) -
         (76545439*mbkin*(1 - mckin/mbkin)**14*np.log(2/mus))/(15030015*mckin) +
         (76545439*mckin*(1 - mckin/mbkin)**14*np.log(2/mus))/(15030015*mbkin) -
         (493125016*mbkin*(1 - mckin/mbkin)**15*np.log(2/mus))/(96621525*mckin) +
         (493125016*mckin*(1 - mckin/mbkin)**15*np.log(2/mus))/(96621525*mbkin) -
         (1979593631*mbkin*(1 - mckin/mbkin)**16*np.log(2/mus))/
          (386486100*mckin) + (1979593631*mckin*(1 - mckin/mbkin)**16*
           np.log(2/mus))/(386486100*mbkin) - (1340564003491*mbkin*
           (1 - mckin/mbkin)**17*np.log(2/mus))/(260620460100*mckin) +
         (1340564003491*mckin*(1 - mckin/mbkin)**17*np.log(2/mus))/
          (260620460100*mbkin) - (48474483141979*mbkin*(1 - mckin/mbkin)**18*
           np.log(2/mus))/(9382336563600*mckin) +
         (48474483141979*mckin*(1 - mckin/mbkin)**18*np.log(2/mus))/
          (9382336563600*mbkin) - (22826272832761*mbkin*(1 - mckin/mbkin)**19*
           np.log(2/mus))/(4398731817480*mckin) +
         (22826272832761*mckin*(1 - mckin/mbkin)**19*np.log(2/mus))/
          (4398731817480*mbkin) - 48*np.pi**2*np.log(2/mus) -
         (24*mckin**2*np.pi**2*np.log(2/mus))/mbkin**2 + (216*mckin**4*np.pi**2*np.log(2/mus))/
          mbkin**4 - (168*mckin**6*np.pi**2*np.log(2/mus))/mbkin**6 +
         (24*mckin**8*np.pi**2*np.log(2/mus))/mbkin**8 +
         (6048*mckin**2*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**2 +
         (4096*mckin**3*np.log(mckin**2/mbkin**2)*np.log(2/mus))/(3*mbkin**3) -
         (22240*mckin**4*np.log(mckin**2/mbkin**2)*np.log(2/mus))/(3*mbkin**4) -
         (144*mckin**2*np.pi**2*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**2 +
         (144*mckin**4*np.pi**2*np.log(mckin**2/mbkin**2)*np.log(2/mus))/mbkin**4 -
         (6310*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(2/mus))/9 + 15*np.pi**2*(1 - (8*mckin**2)/mbkin**2 +
           (8*mckin**6)/mbkin**6 - mckin**8/mbkin**8 -
           (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*np.log(2/mus) +
         (2048*mbkin*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(15*mckin) - (2048*mckin*(1 - mckin/mbkin)**6*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(15*mbkin) -
         (8192*mbkin*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(105*mckin) + (8192*mckin*(1 - mckin/mbkin)**7*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(105*mbkin) +
         (512*mbkin*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(35*mckin) - (512*mckin*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(35*mbkin) +
         (512*mbkin*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(63*mckin) - (512*mckin*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(63*mbkin) +
         (2432*mbkin*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(315*mckin) - (2432*mckin*(1 - mckin/mbkin)**10*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(315*mbkin) +
         (256*mbkin*(1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(33*mckin) - (256*mckin*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(33*mbkin) +
         (352*mbkin*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(45*mckin) - (352*mckin*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(45*mbkin) +
         (50656*mbkin*(1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(6435*mckin) - (50656*mckin*(1 - mckin/mbkin)**13*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(6435*mbkin) +
         (3392*mbkin*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(429*mckin) - (3392*mckin*(1 - mckin/mbkin)**14*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(429*mbkin) +
         (357248*mbkin*(1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(45045*mckin) - (357248*mckin*(1 - mckin/mbkin)**15*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(45045*mbkin) +
         (358012*mbkin*(1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(45045*mckin) - (358012*mckin*(1 - mckin/mbkin)**16*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(45045*mbkin) +
         (677276*mbkin*(1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(85085*mckin) - (677276*mckin*(1 - mckin/mbkin)**17*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(85085*mbkin) +
         (871748*mbkin*(1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(109395*mckin) - (871748*mckin*(1 - mckin/mbkin)**18*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(109395*mbkin) +
         (301396*mbkin*(1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(2/mus))/(37791*mckin) - (301396*mckin*(1 - mckin/mbkin)**19*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(2/mus))/(37791*mbkin) -
         ((-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin)))*np.log(2/mus))/1354809399783840 -
         216*np.log(2/mus)**2 - (108*mckin**2*np.log(2/mus)**2)/mbkin**2 +
         (972*mckin**4*np.log(2/mus)**2)/mbkin**4 - (756*mckin**6*np.log(2/mus)**2)/
          mbkin**6 + (108*mckin**8*np.log(2/mus)**2)/mbkin**8 -
         (648*mckin**2*np.log(mckin**2/mbkin**2)*np.log(2/mus)**2)/mbkin**2 +
         (648*mckin**4*np.log(mckin**2/mbkin**2)*np.log(2/mus)**2)/mbkin**4 +
         (135*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(2/mus)**2)/2 - (18806404529635*np.log(mus**2/mbkin**2))/
          36769584852 + (13246746552638045*mbkin*np.log(mus**2/mbkin**2))/
          (221696083600992*mckin) + (575490020752928857*mckin*
           np.log(mus**2/mbkin**2))/(203221409967576*mbkin) -
         (7891279135278479*mckin**2*np.log(mus**2/mbkin**2))/(666178360848*
           mbkin**2) + (14288824831071077057*mckin**3*np.log(mus**2/mbkin**2))/
          (385051092570144*mbkin**3) - (332528095788640399*mckin**4*
           np.log(mus**2/mbkin**2))/(3775010711472*mbkin**4) +
         (113909846883917815*mckin**5*np.log(mus**2/mbkin**2))/
          (707814508401*mbkin**5) - (30441877170039767*mckin**6*
           np.log(mus**2/mbkin**2))/(134821811124*mbkin**6) +
         (1273391377801069*mckin**7*np.log(mus**2/mbkin**2))/(5447345904*mbkin**7) -
         (282537896923063*mckin**8*np.log(mus**2/mbkin**2))/(1861445124*mbkin**8) -
         (3597890666443*mckin**9*np.log(mus**2/mbkin**2))/(19799007228*mbkin**9) +
         (2008313519491795*mckin**10*np.log(mus**2/mbkin**2))/
          (13199338152*mbkin**10) - (18530565542904137*mckin**11*
           np.log(mus**2/mbkin**2))/(79196028912*mbkin**11) +
         (1277483849769067349*mckin**12*np.log(mus**2/mbkin**2))/
          (5662516067208*mbkin**12) - (37850213399998628*mckin**13*
           np.log(mus**2/mbkin**2))/(235938169467*mbkin**13) +
         (35484451873152599*mckin**14*np.log(mus**2/mbkin**2))/
          (404465433372*mbkin**14) - (93512779185427885*mckin**15*
           np.log(mus**2/mbkin**2))/(2516673807648*mbkin**15) +
         (16099800754227656*mckin**16*np.log(mus**2/mbkin**2))/
          (1336982960313*mbkin**16) - (5468757565122931*mckin**17*
           np.log(mus**2/mbkin**2))/(1887505355736*mbkin**17) +
         (162592894134003649*mckin**18*np.log(mus**2/mbkin**2))/
          (332544125401488*mbkin**18) - (22978828361473*mckin**19*
           np.log(mus**2/mbkin**2))/(444118907232*mbkin**19) +
         (122901707465525*mckin**20*np.log(mus**2/mbkin**2))/(47506303628784*
           mbkin**20) + (10240*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/189 - (5120*mbkin*(1 + 7*np.log(2) +
            7*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(567*mckin) -
         (10240*mckin*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin) +
         (10240*mckin**2*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**2) -
         (10240*mckin**4*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**4) +
         (10240*mckin**5*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) -
         (10240*mckin**6*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(189*mbkin**6) +
         (5120*mckin**7*(1 + 7*np.log(2) + 7*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**7) -
         (2560*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          81 + (2560*mbkin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mckin) +
         (51200*mckin*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin) -
         (10240*mckin**2*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**2) +
         (5120*mckin**3*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**3) +
         (5120*mckin**4*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**4) -
         (10240*mckin**5*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) +
         (51200*mckin**6*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**6) -
         (2560*mckin**7*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**7) +
         (2560*mckin**8*(1 + 8*np.log(2) + 8*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**8) +
         (10240*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          1701 - (1280*mbkin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mckin) -
         (1280*mckin*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(63*mbkin) +
         (20480*mckin**2*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**2) -
         (2560*mckin**3*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**3) +
         (2560*mckin**5*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) -
         (20480*mckin**6*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**6) +
         (1280*mckin**7*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(63*mbkin**7) -
         (10240*mckin**8*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**8) +
         (1280*mckin**9*(1 + 9*np.log(2) + 9*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**9) +
         (640*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          189 - (640*mbkin*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mckin) -
         (3200*mckin*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin) +
         (16000*mckin**2*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**2) -
         (6400*mckin**3*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(189*mbkin**3) +
         (1280*mckin**4*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**4) +
         (1280*mckin**5*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) -
         (6400*mckin**6*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(189*mbkin**6) +
         (16000*mckin**7*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**7) -
         (3200*mckin**8*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**8) +
         (640*mckin**9*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(189*mbkin**9) -
         (640*mckin**10*(1 + 10*np.log(2) + 10*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**10) +
         (60800*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          18711 - (6080*mbkin*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(18711*mckin) -
         (24320*mckin*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin) +
         (60800*mckin**2*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**2) -
         (30400*mckin**3*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**3) +
         (24320*mckin**4*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**4) -
         (24320*mckin**6*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**6) +
         (30400*mckin**7*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**7) -
         (60800*mckin**8*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**8) +
         (24320*mckin**9*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**9) -
         (60800*mckin**10*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(18711*mbkin**10) +
         (6080*mckin**11*(1 + 11*np.log(2) + 11*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(18711*mbkin**11) +
         (800*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          243 - (800*mbkin*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mckin) -
         (1600*mckin*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(99*mbkin) +
         (11200*mckin**2*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**2) -
         (20000*mckin**3*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**3) +
         (800*mckin**4*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(9*mbkin**4) -
         (3200*mckin**5*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) -
         (3200*mckin**6*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**6) +
         (800*mckin**7*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(9*mbkin**7) -
         (20000*mckin**8*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**8) +
         (11200*mckin**9*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**9) -
         (1600*mckin**10*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(99*mbkin**10) +
         (800*mckin**11*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**11) -
         (800*mckin**12*(1 + 12*np.log(2) + 12*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**12) +
         (3520*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          1053 - (880*mbkin*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mckin) -
         (4400*mckin*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin) +
         (14080*mckin**2*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**2) -
         (9680*mckin**3*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**3) +
         (38720*mckin**4*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**4) -
         (9680*mckin**5*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**5) +
         (9680*mckin**7*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**7) -
         (38720*mckin**8*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**8) +
         (9680*mckin**9*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81*mbkin**9) -
         (14080*mckin**10*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**10) +
         (4400*mckin**11*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**11) -
         (3520*mckin**12*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1053*mbkin**12) +
         (880*mckin**13*(1 + 13*np.log(2) + 13*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin**13) +
         (63320*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/
          18711 - (63320*mbkin*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mckin) -
         (63320*mckin*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin) +
         (63320*mckin**2*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(891*mbkin**2) -
         (443240*mckin**3*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**3) +
         (63320*mckin**4*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**4) -
         (63320*mckin**5*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**5) +
         (63320*mckin**6*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**6) +
         (63320*mckin**7*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**7) -
         (63320*mckin**8*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**8) +
         (63320*mckin**9*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**9) -
         (443240*mckin**10*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**10) +
         (63320*mckin**11*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(891*mbkin**11) -
         (63320*mckin**12*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3159*mbkin**12) +
         (63320*mckin**13*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(18711*mbkin**13) -
         (63320*mckin**14*(1 + 14*np.log(2) + 14*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**14) +
         (118720*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/34749 - (8480*mbkin*(1 + 15*np.log(2) +
            15*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(34749*mckin) -
         (84800*mckin*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3861*mbkin) +
         (2968000*mckin**2*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**2) -
         (593600*mckin**3*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**3) +
         (118720*mckin**4*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(297*mbkin**4) -
         (118720*mckin**5*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**5) +
         (84800*mckin**6*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**6) -
         (84800*mckin**8*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**8) +
         (118720*mckin**9*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**9) -
         (118720*mckin**10*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(297*mbkin**10) +
         (593600*mckin**11*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**11) -
         (2968000*mckin**12*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**12) +
         (84800*mckin**13*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3861*mbkin**13) -
         (118720*mckin**14*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**14) +
         (8480*mckin**15*(1 + 15*np.log(2) + 15*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**15) +
         (279100*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/81081 - (55820*mbkin*(1 + 16*np.log(2) +
            16*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(243243*mckin) -
         (446560*mckin*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(18711*mbkin) +
         (2232800*mckin**2*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(22113*mbkin**2) -
         (1116400*mckin**3*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3861*mbkin**3) +
         (1562960*mckin**4*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**4) -
         (2232800*mckin**5*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**5) +
         (446560*mckin**6*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**6) -
         (558200*mckin**7*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**7) -
         (558200*mckin**8*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**8) +
         (446560*mckin**9*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(567*mbkin**9) -
         (2232800*mckin**10*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**10) +
         (1562960*mckin**11*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**11) -
         (1116400*mckin**12*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3861*mbkin**12) +
         (2232800*mckin**13*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(22113*mbkin**13) -
         (446560*mckin**14*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(18711*mbkin**14) +
         (279100*mckin**15*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(81081*mbkin**15) -
         (55820*mckin**16*(1 + 16*np.log(2) + 16*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**16) +
         (14320480*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/4135131 - (895030*mbkin*(1 + 17*np.log(2) +
            17*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(4135131*mckin) -
         (895030*mckin*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin) +
         (28640960*mckin**2*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**2) -
         (89503000*mckin**3*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**3) +
         (28640960*mckin**4*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**4) -
         (3580120*mckin**5*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**5) +
         (28640960*mckin**6*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(18711*mbkin**6) -
         (1790060*mckin**7*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**7) +
         (1790060*mckin**9*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**9) -
         (28640960*mckin**10*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(18711*mbkin**10) +
         (3580120*mckin**11*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**11) -
         (28640960*mckin**12*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**12) +
         (89503000*mckin**13*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**13) -
         (28640960*mckin**14*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**14) +
         (895030*mckin**15*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(34749*mbkin**15) -
         (14320480*mckin**16*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(4135131*mbkin**16) +
         (895030*mckin**17*(1 + 17*np.log(2) + 17*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(4135131*mbkin**17) +
         (846595*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/243243 - (846595*mbkin*(1 + 18*np.log(2) +
            18*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(4135131*mckin) -
         (4232975*mckin*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(153153*mbkin) +
         (846595*mckin**2*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(6237*mbkin**2) -
         (3386380*mckin**3*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(7371*mbkin**3) +
         (3386380*mckin**4*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3003*mbkin**4) -
         (23704660*mckin**5*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11583*mbkin**5) +
         (16931900*mckin**6*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(6237*mbkin**6) -
         (1693190*mckin**7*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(693*mbkin**7) +
         (1693190*mckin**8*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**8) +
         (1693190*mckin**9*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1701*mbkin**9) -
         (1693190*mckin**10*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(693*mbkin**10) +
         (16931900*mckin**11*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(6237*mbkin**11) -
         (23704660*mckin**12*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11583*mbkin**12) +
         (3386380*mckin**13*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3003*mbkin**13) -
         (3386380*mckin**14*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(7371*mbkin**14) +
         (846595*mckin**15*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(6237*mbkin**15) -
         (4232975*mckin**16*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(153153*mbkin**16) +
         (846595*mckin**17*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243243*mbkin**17) -
         (846595*mckin**18*(1 + 18*np.log(2) + 18*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(4135131*mbkin**18) +
         (4358740*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/1247103 - (2179370*mbkin*(1 + 19*np.log(2) +
            19*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(11223927*mckin) -
         (17434960*mckin*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(590733*mbkin) +
         (30511180*mckin**2*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(196911*mbkin**2) -
         (2179370*mckin**3*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3861*mbkin**3) +
         (17434960*mckin**4*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11583*mbkin**4) -
         (34869920*mckin**5*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11583*mbkin**5) +
         (17434960*mckin**6*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3861*mbkin**6) -
         (4358740*mckin**7*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(891*mbkin**7) +
         (8717480*mckin**8*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**8) -
         (8717480*mckin**10*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2673*mbkin**10) +
         (4358740*mckin**11*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(891*mbkin**11) -
         (17434960*mckin**12*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3861*mbkin**12) +
         (34869920*mckin**13*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11583*mbkin**13) -
         (17434960*mckin**14*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11583*mbkin**14) +
         (2179370*mckin**15*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(3861*mbkin**15) -
         (30511180*mckin**16*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(196911*mbkin**16) +
         (17434960*mckin**17*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(590733*mbkin**17) -
         (4358740*mckin**18*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1247103*mbkin**18) +
         (2179370*mckin**19*(1 + 19*np.log(2) + 19*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(11223927*mbkin**19) +
         (376745*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/107406 - (376745*mbkin*(1 + 20*np.log(2) +
            20*np.log(1 - mckin/mbkin))*np.log(mus**2/mbkin**2))/(2040714*mckin) -
         (1883725*mckin*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(60021*mbkin) +
         (9418625*mckin**2*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(53703*mbkin**2) -
         (1883725*mckin**3*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2754*mbkin**3) +
         (4144195*mckin**4*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2106*mbkin**4) -
         (1506980*mckin**5*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(351*mbkin**5) +
         (7534900*mckin**6*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1053*mbkin**6) -
         (9418625*mckin**7*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1053*mbkin**7) +
         (1883725*mckin**8*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**8) -
         (753490*mckin**9*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**9) -
         (753490*mckin**10*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**10) +
         (1883725*mckin**11*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(243*mbkin**11) -
         (9418625*mckin**12*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1053*mbkin**12) +
         (7534900*mckin**13*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(1053*mbkin**13) -
         (1506980*mckin**14*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(351*mbkin**14) +
         (4144195*mckin**15*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2106*mbkin**15) -
         (1883725*mckin**16*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2754*mbkin**16) +
         (9418625*mckin**17*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(53703*mbkin**17) -
         (1883725*mckin**18*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(60021*mbkin**18) +
         (376745*mckin**19*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(107406*mbkin**19) -
         (376745*mckin**20*(1 + 20*np.log(2) + 20*np.log(1 - mckin/mbkin))*
           np.log(mus**2/mbkin**2))/(2040714*mbkin**20) +
         (5*(-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin)))*np.log(mus**2/mbkin**2))/14631941517665472 +
         (15332*np.log(mus**2/mckin**2))/243 + (4096*mckin*np.log(mus**2/mckin**2))/
          (81*mbkin) + (13810*mckin**2*np.log(mus**2/mckin**2))/(243*mbkin**2) -
         (2048*mckin**3*np.log(mus**2/mckin**2))/(27*mbkin**3) -
         (31190*mckin**4*np.log(mus**2/mckin**2))/(81*mbkin**4) +
         (4096*mckin**5*np.log(mus**2/mckin**2))/(81*mbkin**5) +
         (72094*mckin**6*np.log(mus**2/mckin**2))/(243*mbkin**6) -
         (2048*mckin**7*np.log(mus**2/mckin**2))/(81*mbkin**7) -
         (7666*mckin**8*np.log(mus**2/mckin**2))/(243*mbkin**8) -
         (64*mbkin*(1 - mckin/mbkin)**4*np.log(mus**2/mckin**2))/(9*mckin) +
         (64*mckin*(1 - mckin/mbkin)**4*np.log(mus**2/mckin**2))/(9*mbkin) +
         (64*mbkin*(1 - mckin/mbkin)**5*np.log(mus**2/mckin**2))/(5*mckin) -
         (64*mckin*(1 - mckin/mbkin)**5*np.log(mus**2/mckin**2))/(5*mbkin) -
         (85984*mbkin*(1 - mckin/mbkin)**6*np.log(mus**2/mckin**2))/(6075*mckin) +
         (85984*mckin*(1 - mckin/mbkin)**6*np.log(mus**2/mckin**2))/(6075*mbkin) +
         (2436352*mbkin*(1 - mckin/mbkin)**7*np.log(mus**2/mckin**2))/
          (297675*mckin) - (2436352*mckin*(1 - mckin/mbkin)**7*
           np.log(mus**2/mckin**2))/(297675*mbkin) -
         (161092*mbkin*(1 - mckin/mbkin)**8*np.log(mus**2/mckin**2))/
          (99225*mckin) + (161092*mckin*(1 - mckin/mbkin)**8*
           np.log(mus**2/mckin**2))/(99225*mbkin) -
         (162508*mbkin*(1 - mckin/mbkin)**9*np.log(mus**2/mckin**2))/
          (535815*mckin) + (162508*mckin*(1 - mckin/mbkin)**9*
           np.log(mus**2/mckin**2))/(535815*mbkin) -
         (16172*mbkin*(1 - mckin/mbkin)**10*np.log(mus**2/mckin**2))/
          (76545*mckin) + (16172*mckin*(1 - mckin/mbkin)**10*
           np.log(mus**2/mckin**2))/(76545*mbkin) -
         (4212652*mbkin*(1 - mckin/mbkin)**11*np.log(mus**2/mckin**2))/
          (21611205*mckin) + (4212652*mckin*(1 - mckin/mbkin)**11*
           np.log(mus**2/mckin**2))/(21611205*mbkin) -
         (160192*mbkin*(1 - mckin/mbkin)**12*np.log(mus**2/mckin**2))/
          (841995*mckin) + (160192*mckin*(1 - mckin/mbkin)**12*
           np.log(mus**2/mckin**2))/(841995*mbkin) -
         (42218656*mbkin*(1 - mckin/mbkin)**13*np.log(mus**2/mckin**2))/
          (223609815*mckin) + (42218656*mckin*(1 - mckin/mbkin)**13*
           np.log(mus**2/mckin**2))/(223609815*mbkin) -
         (76545439*mbkin*(1 - mckin/mbkin)**14*np.log(mus**2/mckin**2))/
          (405810405*mckin) + (76545439*mckin*(1 - mckin/mbkin)**14*
           np.log(mus**2/mckin**2))/(405810405*mbkin) -
         (493125016*mbkin*(1 - mckin/mbkin)**15*np.log(mus**2/mckin**2))/
          (2608781175*mckin) + (493125016*mckin*(1 - mckin/mbkin)**15*
           np.log(mus**2/mckin**2))/(2608781175*mbkin) -
         (1979593631*mbkin*(1 - mckin/mbkin)**16*np.log(mus**2/mckin**2))/
          (10435124700*mckin) + (1979593631*mckin*(1 - mckin/mbkin)**16*
           np.log(mus**2/mckin**2))/(10435124700*mbkin) -
         (1340564003491*mbkin*(1 - mckin/mbkin)**17*np.log(mus**2/mckin**2))/
          (7036752422700*mckin) + (1340564003491*mckin*(1 - mckin/mbkin)**17*
           np.log(mus**2/mckin**2))/(7036752422700*mbkin) -
         (48474483141979*mbkin*(1 - mckin/mbkin)**18*np.log(mus**2/mckin**2))/
          (253323087217200*mckin) + (48474483141979*mckin*(1 - mckin/mbkin)**
            18*np.log(mus**2/mckin**2))/(253323087217200*mbkin) -
         (22826272832761*mbkin*(1 - mckin/mbkin)**19*np.log(mus**2/mckin**2))/
          (118765759071960*mckin) + (22826272832761*mckin*(1 - mckin/mbkin)**
            19*np.log(mus**2/mckin**2))/(118765759071960*mbkin) -
         (16*np.pi**2*np.log(mus**2/mckin**2))/9 - (8*mckin**2*np.pi**2*np.log(mus**2/mckin**2))/
          (9*mbkin**2) + (8*mckin**4*np.pi**2*np.log(mus**2/mckin**2))/mbkin**4 -
         (56*mckin**6*np.pi**2*np.log(mus**2/mckin**2))/(9*mbkin**6) +
         (8*mckin**8*np.pi**2*np.log(mus**2/mckin**2))/(9*mbkin**8) +
         (6476*mckin**2*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/
          (27*mbkin**2) + (4096*mckin**3*np.log(mckin**2/mbkin**2)*
           np.log(mus**2/mckin**2))/(81*mbkin**3) -
         (23524*mckin**4*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/
          (81*mbkin**4) - (16*mckin**2*np.pi**2*np.log(mckin**2/mbkin**2)*
           np.log(mus**2/mckin**2))/(3*mbkin**2) +
         (16*mckin**4*np.pi**2*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2))/
          (3*mbkin**4) - (26845*(1 - (8*mckin**2)/mbkin**2 +
            (8*mckin**6)/mbkin**6 - mckin**8/mbkin**8 -
            (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*np.log(mus**2/mckin**2))/
          972 + (5*np.pi**2*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(mus**2/mckin**2))/9 + (2048*mbkin*(1 - mckin/mbkin)**6*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/(405*mckin) -
         (2048*mckin*(1 - mckin/mbkin)**6*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(405*mbkin) - (8192*mbkin*(1 - mckin/mbkin)**7*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/(2835*mckin) +
         (8192*mckin*(1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(2835*mbkin) + (512*mbkin*(1 - mckin/mbkin)**8*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/(945*mckin) -
         (512*mckin*(1 - mckin/mbkin)**8*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(945*mbkin) + (512*mbkin*(1 - mckin/mbkin)**9*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/(1701*mckin) -
         (512*mckin*(1 - mckin/mbkin)**9*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(1701*mbkin) +
         (2432*mbkin*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(8505*mckin) -
         (2432*mckin*(1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(8505*mbkin) + (256*mbkin*(1 - mckin/mbkin)**11*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/(891*mckin) -
         (256*mckin*(1 - mckin/mbkin)**11*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(891*mbkin) + (352*mbkin*(1 - mckin/mbkin)**12*
           (np.log(2) + np.log(1 - mckin/mbkin))*np.log(mus**2/mckin**2))/(1215*mckin) -
         (352*mckin*(1 - mckin/mbkin)**12*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(1215*mbkin) +
         (50656*mbkin*(1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(173745*mckin) -
         (50656*mckin*(1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(173745*mbkin) +
         (3392*mbkin*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(11583*mckin) -
         (3392*mckin*(1 - mckin/mbkin)**14*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(11583*mbkin) +
         (357248*mbkin*(1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(1216215*mckin) -
         (357248*mckin*(1 - mckin/mbkin)**15*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(1216215*mbkin) +
         (358012*mbkin*(1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(1216215*mckin) -
         (358012*mckin*(1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(1216215*mbkin) +
         (677276*mbkin*(1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(2297295*mckin) -
         (677276*mckin*(1 - mckin/mbkin)**17*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(2297295*mbkin) +
         (871748*mbkin*(1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(2953665*mckin) -
         (871748*mckin*(1 - mckin/mbkin)**18*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(2953665*mbkin) +
         (301396*mbkin*(1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(1020357*mckin) -
         (301396*mckin*(1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin))*
           np.log(mus**2/mckin**2))/(1020357*mbkin) -
         ((-260123404758497280*(1 - mckin/mbkin)**5 + 390185107137745920*
             (1 - mckin/mbkin)**6 - 388690823028777984*(1 - mckin/mbkin)**7 +
            195377647247557632*(1 - mckin/mbkin)**8 - 34216429928650112*
             (1 - mckin/mbkin)**9 - 6097700477192896*(1 - mckin/mbkin)**10 -
            3945134197513600*(1 - mckin/mbkin)**11 - 3335963793491680*
             (1 - mckin/mbkin)**12 - 2990240473256960*(1 - mckin/mbkin)**13 -
            2738659018760960*(1 - mckin/mbkin)**14 - 2537989658886624*
             (1 - mckin/mbkin)**15 - 2370644934651168*(1 - mckin/mbkin)**16 -
            2227287774097176*(1 - mckin/mbkin)**17 - 2102198391233484*
             (1 - mckin/mbkin)**18 - 1991562479412482*(1 - mckin/mbkin)**19 -
            1892686294969085*(1 - mckin/mbkin)**20 + 132126173845585920*
             (1 - mckin/mbkin)**7*(np.log(2) + np.log(1 - mckin/mbkin)) -
            66063086922792960*(1 - mckin/mbkin)**8*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 11010514487132160*(1 - mckin/mbkin)**9*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 5505257243566080*
             (1 - mckin/mbkin)**10*(np.log(2) + np.log(1 - mckin/mbkin)) +
            4754540346716160*(1 - mckin/mbkin)**11*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 4379181898291200*(1 - mckin/mbkin)**12*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 4076007766871040*
             (1 - mckin/mbkin)**13*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3808925793953280*(1 - mckin/mbkin)**14*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 3570717547837440*(1 - mckin/mbkin)**15*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 3357773812673280*
             (1 - mckin/mbkin)**16*(np.log(2) + np.log(1 - mckin/mbkin)) +
            3167016139647360*(1 - mckin/mbkin)**17*(np.log(2) +
              np.log(1 - mckin/mbkin)) + 2995631463464640*(1 - mckin/mbkin)**18*
             (np.log(2) + np.log(1 - mckin/mbkin)) + 2841110280328320*
             (1 - mckin/mbkin)**19*(np.log(2) + np.log(1 - mckin/mbkin)) +
            2701265736929760*(1 - mckin/mbkin)**20*(np.log(2) +
              np.log(1 - mckin/mbkin)))*np.log(mus**2/mckin**2))/36579853794163680 -
         16*np.log(2/mus)*np.log(mus**2/mckin**2) - (8*mckin**2*np.log(2/mus)*
           np.log(mus**2/mckin**2))/mbkin**2 + (72*mckin**4*np.log(2/mus)*
           np.log(mus**2/mckin**2))/mbkin**4 - (56*mckin**6*np.log(2/mus)*
           np.log(mus**2/mckin**2))/mbkin**6 + (8*mckin**8*np.log(2/mus)*
           np.log(mus**2/mckin**2))/mbkin**8 - (48*mckin**2*np.log(mckin**2/mbkin**2)*
           np.log(2/mus)*np.log(mus**2/mckin**2))/mbkin**2 +
         (48*mckin**4*np.log(mckin**2/mbkin**2)*np.log(2/mus)*np.log(mus**2/mckin**2))/
          mbkin**4 + 5*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
           mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
          np.log(2/mus)*np.log(mus**2/mckin**2) - (8*np.log(mus**2/mckin**2)**2)/27 -
         (4*mckin**2*np.log(mus**2/mckin**2)**2)/(27*mbkin**2) +
         (4*mckin**4*np.log(mus**2/mckin**2)**2)/(3*mbkin**4) -
         (28*mckin**6*np.log(mus**2/mckin**2)**2)/(27*mbkin**6) +
         (4*mckin**8*np.log(mus**2/mckin**2)**2)/(27*mbkin**8) -
         (8*mckin**2*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2)**2)/(9*mbkin**2) +
         (8*mckin**4*np.log(mckin**2/mbkin**2)*np.log(mus**2/mckin**2)**2)/(9*mbkin**4) +
         (5*(1 - (8*mckin**2)/mbkin**2 + (8*mckin**6)/mbkin**6 -
            mckin**8/mbkin**8 - (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           np.log(mus**2/mckin**2)**2)/54 + ((-1 + (8*mckin**2)/mbkin**2 -
            (8*mckin**6)/mbkin**6 + mckin**8/mbkin**8 +
            (12*mckin**4*np.log(mckin**2/mbkin**2))/mbkin**4)*
           ((-167340.53777234623 + 324000*np.log(2/mus) - 11664*np.pi**2*np.log(
                2/mus) - 52488*np.log(2/mus)**2)/1944 -
            (2*(-234 + 9*np.pi**2 + 81*np.log(2/mus))*np.log(mus**2/mckin**2))/81 +
            (21 + 57*np.log(mus**2/mckin**2) - 2*np.log(mus**2/mckin**2)**2)/54))/2))/
      mbkin**2 )
 return res.real

def total_rate_MS(mbkin, mcMS, mus, mu0, api4, muG, sB, rE, sqB, sE, rG, rhoD, mupi):

    res = (1 - (8*mcMS**2)/mbkin**2 + (8*mcMS**6)/mbkin**6 - mcMS**8/mbkin**8 - 
        (3*muG)/(2*mbkin**2) + (4*mcMS**2*muG)/mbkin**4 - (12*mcMS**4*muG)/mbkin**6 + 
        (12*mcMS**6*muG)/mbkin**8 - (5*mcMS**8*muG)/(2*mbkin**10) - 
        mupi/(2*mbkin**2) + (4*mcMS**2*mupi)/mbkin**4 - (4*mcMS**6*mupi)/mbkin**8 + 
        (mcMS**8*mupi)/(2*mbkin**10) + (34*rhoD)/(3*mbkin**3) - 
        (32*mcMS**2*rhoD)/(3*mbkin**5) - (8*mcMS**4*rhoD)/mbkin**7 + 
        (32*mcMS**6*rhoD)/(3*mbkin**9) - (10*mcMS**8*rhoD)/(3*mbkin**11) - 
        (12*mcMS**4*np.log(mcMS**2/mbkin**2))/mbkin**4 + 
        ((-6*mcMS**4*muG)/mbkin**6 + (6*mcMS**4*mupi)/mbkin**6 + (8*rhoD)/mbkin**3)*
        np.log(mcMS**2/mbkin**2) + 
        ((2*(-1 - (2*mcMS**2)/mbkin**2 + (12*mcMS**4)/mbkin**4 - 
            (14*mcMS**6)/mbkin**6 + (5*mcMS**8)/mbkin**8)*sB)/3 + 
        (sqB*(-25 + (48*mcMS**2)/mbkin**2 - (36*mcMS**4)/mbkin**4 + 
            (16*mcMS**6)/mbkin**6 - (3*mcMS**8)/mbkin**8 - 12*np.log(mcMS**2/mbkin**2)))/
            36 - (2*sE*(-25 + (36*mcMS**2)/mbkin**2 - (20*mcMS**6)/mbkin**6 + 
            (9*mcMS**8)/mbkin**8 - 12*np.log(mcMS**2/mbkin**2)))/9 - 
        (8*rE*(2 + (9*mcMS**4)/mbkin**4 - (20*mcMS**6)/mbkin**6 + 
            (9*mcMS**8)/mbkin**8 + 6*np.log(mcMS**2/mbkin**2)))/9 + 
        (4*rG*(16 - (21*mcMS**2)/mbkin**2 + (9*mcMS**4)/mbkin**4 - 
            (7*mcMS**6)/mbkin**6 + (3*mcMS**8)/mbkin**8 + 12*np.log(mcMS**2/mbkin**2)))/
            9)/mbkin**4 + api4*(-1425252426638329/223320230733600 - 
        272/(27*mbkin**3) + 4/mbkin**2 + 80/(9*mbkin) + (2649349310527609*mcMS)/
            (61582245444720*mbkin) + (256*mcMS**2)/(27*mbkin**5) - 
        (64*mcMS**2)/(3*mbkin**4) - (128*mcMS**2)/(3*mbkin**3) - 
        (4197068578247*mcMS**2)/(20427547140*mbkin**2) + 
        (371136550584956041*mcMS**3)/(534793184125200*mbkin**3) + 
        (64*mcMS**4)/(9*mbkin**7) + (16*mcMS**4)/mbkin**6 + 
        (128*mcMS**4)/(3*mbkin**5) - (47319152313660041*mcMS**4)/
            (20972281730400*mbkin**4) + (45300689655370427*mcMS**5)/
            (7864605648900*mbkin**5) - (256*mcMS**6)/(27*mbkin**9) - 
        (128*mcMS**6)/(9*mbkin**7) - (40818484420001*mcMS**6)/
            (3404591190*mbkin**6) + (328364278022567*mcMS**7)/
            (15888092220*mbkin**7) + (80*mcMS**8)/(27*mbkin**11) + 
        (4*mcMS**8)/(3*mbkin**10) + (16*mcMS**8)/(3*mbkin**9) - 
        (3384604682760869*mcMS**8)/(115232317200*mbkin**8) + 
        (546447454894529*mcMS**9)/(15713497800*mbkin**9) - 
        (157739121574127*mcMS**10)/(4583103525*mbkin**10) + 
        (6882359619137911*mcMS**11)/(241987866120*mbkin**11) - 
        (860250268694657*mcMS**12)/(43997793840*mbkin**12) + 
        (29086506506933*mcMS**13)/(2618916300*mbkin**13) - 
        (20276892020748997*mcMS**14)/(3932302824450*mbkin**14) + 
        (5046733464982007*mcMS**15)/(2621535216300*mbkin**15) - 
        (5929043471749*mcMS**16)/(10512421920*mbkin**16) + 
        (14993717402449*mcMS**17)/(120043363440*mbkin**17) - 
        (238966675564711*mcMS**18)/(12154390548300*mbkin**18) + 
        (22978828361473*mcMS**19)/(11719804496400*mbkin**19) - 
        (4916068298621*mcMS**20)/(52784781809760*mbkin**20) + 
        (6511301/1119195 - (14120272*mcMS)/(264537*mbkin) + 
            (94132*mcMS**2)/(351*mbkin**2) - (1128048*mcMS**3)/(1105*mbkin**3) + 
            (143624077*mcMS**4)/(45045*mbkin**4) - (955456*mcMS**5)/(117*mbkin**5) + 
            (1998832*mcMS**6)/(117*mbkin**6) - (2671680*mcMS**7)/(91*mbkin**7) + 
            (62053406*mcMS**8)/(1485*mbkin**8) - (6688928*mcMS**9)/(135*mbkin**9) + 
            (2208008*mcMS**10)/(45*mbkin**10) - (84324832*mcMS**11)/
            (2079*mbkin**11) + (27890*mcMS**12)/mbkin**12 - (713024*mcMS**13)/
            (45*mbkin**13) + (30128528*mcMS**14)/(4095*mbkin**14) - 
            (41246272*mcMS**15)/(15015*mbkin**15) + (10463*mcMS**16)/
            (13*mbkin**16) - (1063600*mcMS**17)/(5967*mbkin**17) + 
            (5859964*mcMS**18)/(208845*mbkin**18) - (342128*mcMS**19)/
            (122265*mbkin**19) + (75349*mcMS**20)/(566865*mbkin**20))*np.log(2) + 
        (-64/(9*mbkin**3) + (16*mcMS**2)/mbkin**2 - (16*mcMS**4)/mbkin**6 - 
            (64*mcMS**4)/(3*mbkin**5) - (40*mcMS**4)/mbkin**4 - 
            (48*mcMS**6)/mbkin**6 + (8*mcMS**8)/mbkin**8)*np.log(mcMS**2/mbkin**2) + 
        (48*mcMS**4*np.log(mcMS**2/mbkin**2)**2)/mbkin**4 + 
        (6511301/1119195 - (14120272*mcMS)/(264537*mbkin) + 
            (94132*mcMS**2)/(351*mbkin**2) - (1128048*mcMS**3)/(1105*mbkin**3) + 
            (143624077*mcMS**4)/(45045*mbkin**4) - (955456*mcMS**5)/(117*mbkin**5) + 
            (1998832*mcMS**6)/(117*mbkin**6) - (2671680*mcMS**7)/(91*mbkin**7) + 
            (62053406*mcMS**8)/(1485*mbkin**8) - (6688928*mcMS**9)/(135*mbkin**9) + 
            (2208008*mcMS**10)/(45*mbkin**10) - (84324832*mcMS**11)/
            (2079*mbkin**11) + (27890*mcMS**12)/mbkin**12 - (713024*mcMS**13)/
            (45*mbkin**13) + (30128528*mcMS**14)/(4095*mbkin**14) - 
            (41246272*mcMS**15)/(15015*mbkin**15) + (10463*mcMS**16)/
            (13*mbkin**16) - (1063600*mcMS**17)/(5967*mbkin**17) + 
            (5859964*mcMS**18)/(208845*mbkin**18) - (342128*mcMS**19)/
            (122265*mbkin**19) + (75349*mcMS**20)/(566865*mbkin**20))*
            np.log(1 - mcMS/mbkin) + ((-16*mcMS**2)/mbkin**2 - (24*mcMS**4)/mbkin**4 + 
            (48*mcMS**6)/mbkin**6 - (8*mcMS**8)/mbkin**8 - 
            (48*mcMS**4*np.log(mcMS**2/mbkin**2))/mbkin**4)*np.log(mu0**2/mus**2) + 
        ((-16*mcMS**2)/mbkin**2 - (24*mcMS**4)/mbkin**4 + (48*mcMS**6)/mbkin**6 - 
            (8*mcMS**8)/mbkin**8 - (48*mcMS**4*np.log(mcMS**2/mbkin**2))/mbkin**4)*
            np.log(mus**2/mbkin**2)) + api4**2*(-62.10998262078874 - 320/(81*mbkin**5) - 
        1156/(243*mbkin**4) - 283901/(3645*mbkin**3) + 3242416099682471/
            (55830057683400*mbkin**2) + 1980381092049071/(25123525957530*mbkin) - 
        742/(243*mbkin*mcMS**2) + (13605022399426541*mcMS)/
            (92373368167080*mbkin**3) + (10955673088898932*mcMS)/
            (34640013062655*mbkin**2) + (10205.818638681507*mcMS)/mbkin - 
        (128*mcMS**2)/(9*mbkin**6) - (531068*mcMS**2)/(3645*mbkin**5) - 
        (2041561284304*mcMS**2)/(2188665765*mbkin**4) - (28052281501292*mcMS**2)/
            (15320660355*mbkin**3) - (64363.065126779795*mcMS**2)/mbkin**2 + 
        (410465753831809561*mcMS**3)/(267396592062600*mbkin**5) + 
        (860260710910472642*mcMS**3)/(300821166070425*mbkin**4) + 
        (253907.4263127173*mcMS**3)/mbkin**3 - (256*mcMS**4)/(27*mbkin**9) - 
        (752*mcMS**4)/(81*mbkin**8) + (37729*mcMS**4)/(135*mbkin**7) - 
        (50500082823777401*mcMS**4)/(15729211297800*mbkin**6) - 
        (2828505259240019*mcMS**4)/(620889919650*mbkin**5) - 
        (702666.1726900433*mcMS**4)/mbkin**4 + (64083119032201547*mcMS**5)/
            (11796908473350*mbkin**7) + (151625581504*mcMS**5)/(35712495*mbkin**6) + 
        (1.4373563603290329e6*mcMS**5)/mbkin**5 + (2048*mcMS**6)/(81*mbkin**11) + 
        (14656*mcMS**6)/(243*mbkin**10) - (449594*mcMS**6)/(3645*mbkin**9) - 
        (45554398472*mcMS**6)/(11904165*mbkin**8) + (165849320616424*mcMS**6)/
            (15320660355*mbkin**7) - (2.250814338172405e6*mcMS**6)/mbkin**6 - 
        (49578014567677*mcMS**7)/(7944046110*mbkin**9) - 
        (1908393286902392*mcMS**7)/(35748207495*mbkin**8) + 
        (2.7610452413924173e6*mcMS**7)/mbkin**7 - (320*mcMS**8)/(27*mbkin**13) - 
        (2596*mcMS**8)/(81*mbkin**12) + (17134*mcMS**8)/(243*mbkin**11) + 
        (2344310422569749*mcMS**8)/(86424237900*mbkin**10) + 
        (2692926361803289*mcMS**8)/(21606059475*mbkin**9) - 
        (2.698286812886065e6*mcMS**8)/mbkin**8 - (4638841376579*mcMS**9)/
            (86337900*mbkin**11) - (213276605964776*mcMS**9)/
            (1039863825*mbkin**10) + (2.139651583717923e6*mcMS**9)/mbkin**9 + 
        (53962991221684*mcMS**10)/(723647925*mbkin**12) + 
        (429048024435280*mcMS**10)/(1649917269*mbkin**11) - 
        (1.4131946205782248e6*mcMS**10)/mbkin**10 - (5756976638199079*mcMS**11)/
            (72596359836*mbkin**13) - (23778161873422204*mcMS**11)/
            (90745449795*mbkin**12) + (809769.8111957415*mcMS**11)/mbkin**11 + 
        (733335667127057*mcMS**12)/(10999448460*mbkin**14) + 
        (107352332070551*mcMS**12)/(505076715*mbkin**13) - 
        (423281.18928453606*mcMS**12)/mbkin**12 - (175890613358531*mcMS**13)/
            (3928374450*mbkin**15) - (819908479461856*mcMS**13)/
            (5892561675*mbkin**14) + (207616.17233609466*mcMS**13)/mbkin**13 + 
        (141521855980373936*mcMS**14)/(5898454236675*mbkin**16) + 
        (143821109334331496*mcMS**14)/(1966151412225*mbkin**15) - 
        (93205.70920230761*mcMS**14)/mbkin**14 - (4436494901407367*mcMS**15)/
            (436922536050*mbkin**17) - (2116479415418744*mcMS**15)/
            (69393579255*mbkin**16) + (35960.80809967587*mcMS**15)/mbkin**15 + 
        (498145915917851*mcMS**16)/(149802012360*mbkin**18) + 
        (9164823618173*mcMS**16)/(928524870*mbkin**17) - 
        (209025348451886*mcMS**16)/(18725251545*mbkin**16) - 
        (146581850591723*mcMS**17)/(180065045160*mbkin**19) - 
        (53858522664724*mcMS**17)/(22508130645*mbkin**18) + 
        (13914362059201*mcMS**17)/(5296030740*mbkin**17) + 
        (426492755283692*mcMS**18)/(3038597637075*mbkin**20) + 
        (860899448389804*mcMS**18)/(2103644517975*mbkin**19) - 
        (443639620565602*mcMS**18)/(1012865879025*mbkin**18) - 
        (349791545733281*mcMS**19)/(22988847281400*mbkin**21) - 
        (705418596668242*mcMS**19)/(16010090070975*mbkin**20) + 
        (362691083547521*mcMS**19)/(7864605648900*mbkin**19) + 
        (4414905824237*mcMS**20)/(5655512336760*mbkin**22) + 
        (22241583279313*mcMS**20)/(9897146589330*mbkin**21) - 
        (22826272832761*mcMS**20)/(9897146589330*mbkin**20) + 
        (1568*np.pi**2)/(243*mbkin**3) - (2*np.pi**2)/mbkin**2 - (40*np.pi**2)/(9*mbkin) - 
        (232006051154951*mcMS*np.pi**2)/(233746793280*mbkin) - 
        (124*mcMS**2*np.pi**2)/(27*mbkin**5) + (32*mcMS**2*np.pi**2)/(3*mbkin**4) + 
        (64*mcMS**2*np.pi**2)/(3*mbkin**3) + (3982634073533*mcMS**2*np.pi**2)/
            (626107482*mbkin**2) - (973375607721467*mcMS**3*np.pi**2)/
            (38529691200*mbkin**3) - (200*mcMS**4*np.pi**2)/(81*mbkin**7) - 
        (8*mcMS**4*np.pi**2)/mbkin**6 - (64*mcMS**4*np.pi**2)/(3*mbkin**5) + 
        (4693435138416001*mcMS**4*np.pi**2)/(67426959600*mbkin**4) - 
        (494454213273613*mcMS**5*np.pi**2)/(3502699200*mbkin**5) - 
        (40*mcMS**6*np.pi**2)/(81*mbkin**9) + (64*mcMS**6*np.pi**2)/(9*mbkin**7) + 
        (2383218266311*mcMS**6*np.pi**2)/(10945935*mbkin**6) - 
        (425126865602629*mcMS**7*np.pi**2)/(1634592960*mbkin**7) - 
        (176*mcMS**8*np.pi**2)/(81*mbkin**11) - (2*mcMS**8*np.pi**2)/(3*mbkin**10) - 
        (8*mcMS**8*np.pi**2)/(3*mbkin**9) + (15021137549651*mcMS**8*np.pi**2)/
            (61916400*mbkin**8) - (206289550942129*mcMS**9*np.pi**2)/
            (1167566400*mbkin**9) + (1817350800766*mcMS**10*np.pi**2)/
            (18243225*mbkin**10) - (2302843096433723*mcMS**11*np.pi**2)/
            (53941567680*mbkin**11) + (25925727481081*mcMS**12*np.pi**2)/
            (1926484560*mbkin**12) - (1475137445608273*mcMS**13*np.pi**2)/
            (500885985600*mbkin**13) + (8765154290351*mcMS**14*np.pi**2)/
            (21913761870*mbkin**14) - (2545509950573*mcMS**15*np.pi**2)/
            (100177197120*mbkin**15) - (172*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/
            (243*mbkin**3) + (364*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(9*mbkin**7) + 
        (440*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(9*mbkin**3) + 
        (-9946516/2606175 + (1790032*mcMS)/(675675*mbkin) + 
            (111076996*mcMS**2)/(868725*mbkin**2) - (993662224*mcMS**3)/
            (1403325*mbkin**3) + (565574404*mcMS**4)/(280665*mbkin**4) - 
            (2022928*mcMS**5)/(525*mbkin**5) + (100165252*mcMS**6)/
            (18225*mbkin**6) - (260772688*mcMS**7)/(42525*mbkin**7) + 
            (76993732*mcMS**8)/(14175*mbkin**8) - (2773168*mcMS**9)/(729*mbkin**9) + 
            (17673212*mcMS**10)/(8505*mbkin**10) - (405443344*mcMS**11)/
            (467775*mbkin**11) + (374535772*mcMS**12)/(1403325*mbkin**12) - 
            (149130896*mcMS**13)/(2606175*mbkin**13) + (1031444*mcMS**14)/
            (135135*mbkin**14) - (8705456*mcMS**15)/(18243225*mbkin**15))*
            np.log(2)**2 + (-1016/(81*mbkin**3) - (308*mcMS**2)/(81*mbkin**5) - 
            (98*mcMS**2)/(3*mbkin**2) + (1028*mcMS**4)/(243*mbkin**7) + 
            (184*mcMS**4)/(3*mbkin**6) + (736*mcMS**4)/(9*mbkin**5) + 
            (1043*mcMS**4)/(3*mbkin**4) - (1016*mcMS**6)/(81*mbkin**9) + 
            (194*mcMS**6)/mbkin**6 - (536*mcMS**8)/(81*mbkin**11) - 
            (121*mcMS**8)/(3*mbkin**8))*np.log(mcMS**2/mbkin**2)**2 - 
        (146*mcMS**4*np.log(mcMS**2/mbkin**2)**3)/mbkin**4 + 
        (27590512342093/328706428050 + 26045204/(1119195*mbkin**2) + 
            104180816/(2014551*mbkin) - (141202720*mcMS)/(793611*mbkin**3) - 
            (903697408*mcMS)/(2380833*mbkin**2) - (121608408827531279*mcMS)/
            (176953627100250*mbkin) + (753056*mcMS**2)/(1053*mbkin**4) + 
            (1506112*mcMS**2)/(1053*mbkin**3) + (20558323897999*mcMS**2)/
            (7114857750*mbkin**2) - (2256096*mcMS**3)/(1105*mbkin**5) - 
            (12032512*mcMS**3)/(3315*mbkin**4) - (20219961322255187*mcMS**3)/
            (2149234337250*mbkin**3) + (574496308*mcMS**4)/(135135*mbkin**6) + 
            (2297985232*mcMS**4)/(405405*mbkin**5) + (54098349407369*mcMS**4)/
            (1945008450*mbkin**4) - (1910912*mcMS**5)/(351*mbkin**7) - 
            (10395050718469*mcMS**5)/(141891750*mbkin**5) - (31981312*mcMS**6)/
            (1053*mbkin**7) + (268534093404133*mcMS**6)/(1641890250*mbkin**6) + 
            (1781120*mcMS**7)/(91*mbkin**9) + (28497920*mcMS**7)/(273*mbkin**8) - 
            (1166183774476367*mcMS**7)/(3831077250*mbkin**7) - 
            (248213624*mcMS**8)/(4455*mbkin**10) - (992854496*mcMS**8)/
            (4455*mbkin**9) + (603566981146373*mcMS**8)/(1277025750*mbkin**8) + 
            (13377856*mcMS**9)/(135*mbkin**11) + (428091392*mcMS**9)/
            (1215*mbkin**10) - (201375863874881*mcMS**9)/(328378050*mbkin**9) - 
            (17664064*mcMS**10)/(135*mbkin**12) - (35328128*mcMS**10)/
            (81*mbkin**11) + (2545068527118631*mcMS**10)/(3831077250*mbkin**10) + 
            (843248320*mcMS**11)/(6237*mbkin**13) + (2698394624*mcMS**11)/
            (6237*mbkin**12) - (25245604156713251*mcMS**11)/
            (42141849750*mbkin**11) - (111560*mcMS**12)/mbkin**14 - 
            (3123680*mcMS**12)/(9*mbkin**13) + (56577754794125923*mcMS**12)/
            (126425549250*mbkin**12) + (9982336*mcMS**13)/(135*mbkin**15) + 
            (91267072*mcMS**13)/(405*mbkin**14) - (5868026480761229*mcMS**13)/
            (21344573250*mbkin**13) - (482056448*mcMS**14)/(12285*mbkin**16) - 
            (482056448*mcMS**14)/(4095*mbkin**15) + (37165257525661*mcMS**14)/
            (270540270*mbkin**14) + (82492544*mcMS**15)/(5005*mbkin**17) + 
            (1319880704*mcMS**15)/(27027*mbkin**16) - (90299507058070337*mcMS**15)/
            (1643532140250*mbkin**15) - (209260*mcMS**16)/(39*mbkin**18) - 
            (1841488*mcMS**16)/(117*mbkin**17) + (669632*mcMS**16)/(39*mbkin**16) + 
            (23399200*mcMS**17)/(17901*mbkin**19) + (68070400*mcMS**17)/
            (17901*mbkin**18) - (4254400*mcMS**17)/(1053*mbkin**17) - 
            (46879712*mcMS**18)/(208845*mbkin**20) - (93759424*mcMS**18)/
            (144585*mbkin**19) + (46879712*mcMS**18)/(69615*mbkin**18) + 
            (684256*mcMS**19)/(28215*mbkin**21) + (76636672*mcMS**19)/
            (1100385*mbkin**20) - (1368512*mcMS**19)/(19305*mbkin**19) - 
            (2109772*mcMS**20)/(1700595*mbkin**22) - (1205584*mcMS**20)/
            (340119*mbkin**21) + (1205584*mcMS**20)/(340119*mbkin**20) - 
            (343016*np.pi**2)/135135 + (180704*mcMS*np.pi**2)/(9009*mbkin) - 
            (300088*mcMS**2*np.pi**2)/(3861*mbkin**2) + (308192*mcMS**3*np.pi**2)/
            (1485*mbkin**3) - (508136*mcMS**4*np.pi**2)/(1155*mbkin**4) + 
            (103328*mcMS**5*np.pi**2)/(135*mbkin**5) - (29032*mcMS**6*np.pi**2)/
            (27*mbkin**6) + (25184*mcMS**7*np.pi**2)/(21*mbkin**7) - 
            (66488*mcMS**8*np.pi**2)/(63*mbkin**8) + (98656*mcMS**9*np.pi**2)/
            (135*mbkin**9) - (17768*mcMS**10*np.pi**2)/(45*mbkin**10) + 
            (339424*mcMS**11*np.pi**2)/(2079*mbkin**11) - (14824*mcMS**12*np.pi**2)/
            (297*mbkin**12) + (22816*mcMS**13*np.pi**2)/(2145*mbkin**13) - 
            (190744*mcMS**14*np.pi**2)/(135135*mbkin**14) + (1696*mcMS**15*np.pi**2)/
            (19305*mbkin**15) + ((14120272*mcMS)/(264537*mbkin) - 
            (188264*mcMS**2)/(351*mbkin**2) + (3384144*mcMS**3)/(1105*mbkin**3) - 
            (574496308*mcMS**4)/(45045*mbkin**4) + (4777280*mcMS**5)/
                (117*mbkin**5) - (3997664*mcMS**6)/(39*mbkin**6) + 
            (2671680*mcMS**7)/(13*mbkin**7) - (496427248*mcMS**8)/
                (1485*mbkin**8) + (6688928*mcMS**9)/(15*mbkin**9) - 
            (4416016*mcMS**10)/(9*mbkin**10) + (84324832*mcMS**11)/
                (189*mbkin**11) - (334680*mcMS**12)/mbkin**12 + (9269312*mcMS**13)/
                (45*mbkin**13) - (60257056*mcMS**14)/(585*mbkin**14) + 
            (41246272*mcMS**15)/(1001*mbkin**15) - (167408*mcMS**16)/
                (13*mbkin**16) + (1063600*mcMS**17)/(351*mbkin**17) - 
            (11719928*mcMS**18)/(23205*mbkin**18) + (342128*mcMS**19)/
                (6435*mbkin**19) - (301396*mcMS**20)/(113373*mbkin**20) + 
            (344*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**3) - 
            (728*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(9*mbkin**7) - 
            (880*(mcMS**2/mbkin**2)**(3/2))/(9*mbkin**3))*np.log(mcMS**2/mbkin**2))*
            np.log(1 - mcMS/mbkin) + (-9946516/2606175 + (1790032*mcMS)/
            (675675*mbkin) + (111076996*mcMS**2)/(868725*mbkin**2) - 
            (993662224*mcMS**3)/(1403325*mbkin**3) + (565574404*mcMS**4)/
            (280665*mbkin**4) - (2022928*mcMS**5)/(525*mbkin**5) + 
            (100165252*mcMS**6)/(18225*mbkin**6) - (260772688*mcMS**7)/
            (42525*mbkin**7) + (76993732*mcMS**8)/(14175*mbkin**8) - 
            (2773168*mcMS**9)/(729*mbkin**9) + (17673212*mcMS**10)/
            (8505*mbkin**10) - (405443344*mcMS**11)/(467775*mbkin**11) + 
            (374535772*mcMS**12)/(1403325*mbkin**12) - (149130896*mcMS**13)/
            (2606175*mbkin**13) + (1031444*mcMS**14)/(135135*mbkin**14) - 
            (8705456*mcMS**15)/(18243225*mbkin**15))*np.log(1 - mcMS/mbkin)**2 + 
        (27538/(243*mbkin**3) - (742*mbkin)/(243*mcMS**4) + 
            44924/(3645*mbkin*mcMS**2) - (27016*mcMS**2)/(243*mbkin**5) - 
            (40870*mcMS**4)/(729*mbkin**7) + (28484*mcMS**6)/(243*mbkin**9) - 
            (88178*mcMS**8)/(1215*mbkin**11))*np.log(1 - mcMS**2/mbkin**2) + 
        np.log(mcMS**2/mbkin**2)*(-256/(27*mbkin**5) - 2048/(81*mbkin**4) - 
            196/(243*mbkin**3) + 2/(3*mbkin**2) + 40/(27*mbkin) - 
            (2291073463739113*mcMS)/(61582245444720*mbkin) - 
            (7696*mcMS**2)/(81*mbkin**5) + (352*mcMS**2)/(9*mbkin**4) + 
            (704*mcMS**2)/(9*mbkin**3) + (4435350068993*mcMS**2)/
            (10213773570*mbkin**2) - (331807347338102521*mcMS**3)/
            (178264394708400*mbkin**3) + (16*mcMS**4)/(3*mbkin**8) - 
            (10061*mcMS**4)/(729*mbkin**7) - (968*mcMS**4)/(3*mbkin**6) - 
            (560*mcMS**4)/mbkin**5 + (7951.298981667*mcMS**4)/mbkin**4 - 
            (41544203780004203*mcMS**5)/(1572921129780*mbkin**5) - 
            (3052*mcMS**6)/(81*mbkin**9) + (2240*mcMS**6)/(27*mbkin**7) + 
            (37419150707083*mcMS**6)/(567431865*mbkin**6) - 
            (100900938658973*mcMS**7)/(756575820*mbkin**7) - 
            (15607*mcMS**8)/(1215*mbkin**11) - (94*mcMS**8)/(9*mbkin**10) - 
            (376*mcMS**8)/(9*mbkin**9) + (446439552967277*mcMS**8)/
            (2057719950*mbkin**8) - (45909984092659*mcMS**9)/
            (158722200*mbkin**9) + (291816629210152*mcMS**10)/
            (916620705*mbkin**10) - (374754229442903*mcMS**11)/
            (1294052760*mbkin**11) + (796792967910857*mcMS**12)/
            (3666482820*mbkin**12) - (50058521771447*mcMS**13)/
            (374130900*mbkin**13) + (2685542939845591*mcMS**14)/
            (40125539025*mbkin**14) - (4680590326837223*mcMS**15)/
            (174769014420*mbkin**15) + (104512674225943*mcMS**16)/
            (12483501030*mbkin**16) - (13914362059201*mcMS**17)/
            (7061374320*mbkin**17) + (221819810282801*mcMS**18)/
            (675243919350*mbkin**18) - (362691083547521*mcMS**19)/
            (10486140865200*mbkin**19) + (22826272832761*mcMS**20)/
            (13196195452440*mbkin**20) + (32*np.pi**2)/(9*mbkin**3) + 
            (8*mcMS**4*np.pi**2)/mbkin**6 + (32*mcMS**4*np.pi**2)/(3*mbkin**5) - 
            (8*mcMS**4*np.pi**2)/mbkin**4 + ((-344*np.sqrt(0j + mcMS**2/mbkin**2))/
                (243*mbkin**3) + (728*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(9*mbkin**7) + 
            (880*(mcMS**2/mbkin**2)**(3/2))/(9*mbkin**3))*np.log(1 + mcMS/mbkin) + 
            (3416/(81*mbkin**3) + (88*mcMS**2)/(9*mbkin**5) + (26344*mcMS**4)/
                (243*mbkin**7) + (1520*mcMS**6)/(81*mbkin**9) + 
            (704*mcMS**8)/(81*mbkin**11))*np.log(1 - mcMS**2/mbkin**2)) + 
        ((2*mcMS**2)/(3*mbkin**2) - (71*mcMS**4)/mbkin**4 + (94*mcMS**6)/mbkin**6 - 
            (71*mcMS**8)/(3*mbkin**8) - (46*mcMS**4*np.log(mcMS**2/mbkin**2))/mbkin**4)*
            np.log(mu0**2/mus**2)**2 + (136/(3*mbkin**3) - 18/mbkin**2 - 40/mbkin - 
            (128*mcMS**2)/(3*mbkin**5) + (96*mcMS**2)/mbkin**4 + 
            (192*mcMS**2)/mbkin**3 - (32*mcMS**4)/mbkin**7 - (72*mcMS**4)/mbkin**6 - 
            (192*mcMS**4)/mbkin**5 + (128*mcMS**6)/(3*mbkin**9) + 
            (64*mcMS**6)/mbkin**7 - (40*mcMS**8)/(3*mbkin**11) - 
            (6*mcMS**8)/mbkin**10 - (24*mcMS**8)/mbkin**9 + 
            (32/mbkin**3 + (72*mcMS**4)/mbkin**6 + (96*mcMS**4)/mbkin**5)*
            np.log(mcMS**2/mbkin**2))*np.log(2/mus) + 
        (-299900853349184039/24386569196109120 - 1016/(81*mbkin**3) - 
            2/(3*mbkin**2) - 40/(27*mbkin) + (442920953991272333*mcMS)/
            (4064428199351520*mbkin) + (1408*mcMS**2)/(81*mbkin**5) - 
            (352*mcMS**2)/(9*mbkin**4) - (704*mcMS**2)/(9*mbkin**3) - 
            (898023654678739*mcMS**2)/(1348218111240*mbkin**2) + 
            (7995489995126581427*mcMS**3)/(3208759104751200*mbkin**3) + 
            (736*mcMS**4)/(27*mbkin**7) + (88*mcMS**4)/(3*mbkin**6) + 
            (1088*mcMS**4)/(9*mbkin**5) - (73920603124772561*mcMS**4)/
            (7401981787200*mbkin**4) + (278268617534974331*mcMS**5)/
            (9437526778680*mbkin**5) - (4480*mcMS**6)/(81*mbkin**9) - 
            (2240*mcMS**6)/(27*mbkin**7) - (724867378999853*mcMS**6)/
            (10213773570*mbkin**6) + (4453971834522761*mcMS**7)/
            (31776184440*mbkin**7) + (1880*mcMS**8)/(81*mbkin**11) + 
            (94*mcMS**8)/(9*mbkin**10) + (376*mcMS**8)/(9*mbkin**9) - 
            (1087902473036581843*mcMS**8)/(4839757322400*mbkin**8) + 
            (1474052461641281*mcMS**9)/(4962157200*mbkin**9) - 
            (3576812751457991*mcMS**10)/(10999448460*mbkin**10) + 
            (428005474988275969*mcMS**11)/(1451927196720*mbkin**11) - 
            (153882080327111*mcMS**12)/(697203936*mbkin**12) + 
            (304876087601503571*mcMS**13)/(2247030185400*mbkin**13) - 
            (3264218263360811*mcMS**14)/(48150646830*mbkin**14) + 
            (85260725591574083*mcMS**15)/(3145842259560*mbkin**15) - 
            (10145868551653759*mcMS**16)/(1198416098880*mbkin**16) + 
            (1434258647440951*mcMS**17)/(720260180640*mbkin**17) - 
            (24195506186107219*mcMS**18)/(72926343289800*mbkin**18) + 
            (8347484721312487*mcMS**19)/(239084011726560*mbkin**19) - 
            (10049938477907*mcMS**20)/(5758339833792*mbkin**20) + 
            (32/(27*mbkin**3) + (196*mcMS**2)/(3*mbkin**2) - (184*mcMS**4)/
                (3*mbkin**6) - (736*mcMS**4)/(9*mbkin**5) - (680*mcMS**4)/
                (3*mbkin**4) - (388*mcMS**6)/mbkin**6 + (242*mcMS**8)/(3*mbkin**8))*
            np.log(mcMS**2/mbkin**2) + (292*mcMS**4*np.log(mcMS**2/mbkin**2)**2)/mbkin**4 + 
            (400315229/37413090 - (1214855480*mcMS)/(8729721*mbkin) + 
            (10181450*mcMS**2)/(11583*mbkin**2) - (793211480*mcMS**3)/
                (196911*mbkin**3) + (578057131*mcMS**4)/(38610*mbkin**4) - 
            (237591056*mcMS**5)/(5265*mbkin**5) + (115275460*mcMS**6)/
                (1053*mbkin**6) - (176088368*mcMS**7)/(819*mbkin**7) + 
            (10768320109*mcMS**8)/(31185*mbkin**8) - (185080064*mcMS**9)/
                (405*mbkin**9) + (22516192*mcMS**10)/(45*mbkin**10) - 
            (104769824*mcMS**11)/(231*mbkin**11) + (302512021*mcMS**12)/
                (891*mbkin**12) - (4028303216*mcMS**13)/(19305*mbkin**13) + 
            (42257454076*mcMS**14)/(405405*mbkin**14) - (2412926416*mcMS**15)/
                (57915*mbkin**15) + (1014911*mcMS**16)/(78*mbkin**16) - 
            (54775400*mcMS**17)/(17901*mbkin**17) + (319368038*mcMS**18)/
                (626535*mbkin**18) - (3934472*mcMS**19)/(73359*mbkin**19) + 
            (9117229*mcMS**20)/(3401190*mbkin**20))*np.log(1 - mcMS/mbkin))*
            np.log(mus**2/mbkin**2) + ((-98*mcMS**2)/(3*mbkin**2) - 
            (121*mcMS**4)/mbkin**4 + (194*mcMS**6)/mbkin**6 - 
            (121*mcMS**8)/(3*mbkin**8) - (146*mcMS**4*np.log(mcMS**2/mbkin**2))/mbkin**4)*
            np.log(mus**2/mbkin**2)**2 + np.log(2)*(27590512342093/328706428050 + 
            26045204/(1119195*mbkin**2) + 104180816/(2014551*mbkin) - 
            (141202720*mcMS)/(793611*mbkin**3) - (903697408*mcMS)/
            (2380833*mbkin**2) - (121608408827531279*mcMS)/(176953627100250*
            mbkin) + (753056*mcMS**2)/(1053*mbkin**4) + (1506112*mcMS**2)/
            (1053*mbkin**3) + (20558323897999*mcMS**2)/(7114857750*mbkin**2) - 
            (2256096*mcMS**3)/(1105*mbkin**5) - (12032512*mcMS**3)/(3315*mbkin**4) - 
            (20219961322255187*mcMS**3)/(2149234337250*mbkin**3) + 
            (574496308*mcMS**4)/(135135*mbkin**6) + (2297985232*mcMS**4)/
            (405405*mbkin**5) + (54098349407369*mcMS**4)/(1945008450*mbkin**4) - 
            (1910912*mcMS**5)/(351*mbkin**7) - (10395050718469*mcMS**5)/
            (141891750*mbkin**5) - (31981312*mcMS**6)/(1053*mbkin**7) + 
            (268534093404133*mcMS**6)/(1641890250*mbkin**6) + 
            (1781120*mcMS**7)/(91*mbkin**9) + (28497920*mcMS**7)/(273*mbkin**8) - 
            (1166183774476367*mcMS**7)/(3831077250*mbkin**7) - 
            (248213624*mcMS**8)/(4455*mbkin**10) - (992854496*mcMS**8)/
            (4455*mbkin**9) + (603566981146373*mcMS**8)/(1277025750*mbkin**8) + 
            (13377856*mcMS**9)/(135*mbkin**11) + (428091392*mcMS**9)/
            (1215*mbkin**10) - (201375863874881*mcMS**9)/(328378050*mbkin**9) - 
            (17664064*mcMS**10)/(135*mbkin**12) - (35328128*mcMS**10)/
            (81*mbkin**11) + (2545068527118631*mcMS**10)/(3831077250*mbkin**10) + 
            (843248320*mcMS**11)/(6237*mbkin**13) + (2698394624*mcMS**11)/
            (6237*mbkin**12) - (25245604156713251*mcMS**11)/
            (42141849750*mbkin**11) - (111560*mcMS**12)/mbkin**14 - 
            (3123680*mcMS**12)/(9*mbkin**13) + (56577754794125923*mcMS**12)/
            (126425549250*mbkin**12) + (9982336*mcMS**13)/(135*mbkin**15) + 
            (91267072*mcMS**13)/(405*mbkin**14) - (5868026480761229*mcMS**13)/
            (21344573250*mbkin**13) - (482056448*mcMS**14)/(12285*mbkin**16) - 
            (482056448*mcMS**14)/(4095*mbkin**15) + (37165257525661*mcMS**14)/
            (270540270*mbkin**14) + (82492544*mcMS**15)/(5005*mbkin**17) + 
            (1319880704*mcMS**15)/(27027*mbkin**16) - (90299507058070337*mcMS**15)/
            (1643532140250*mbkin**15) - (209260*mcMS**16)/(39*mbkin**18) - 
            (1841488*mcMS**16)/(117*mbkin**17) + (669632*mcMS**16)/(39*mbkin**16) + 
            (23399200*mcMS**17)/(17901*mbkin**19) + (68070400*mcMS**17)/
            (17901*mbkin**18) - (4254400*mcMS**17)/(1053*mbkin**17) - 
            (46879712*mcMS**18)/(208845*mbkin**20) - (93759424*mcMS**18)/
            (144585*mbkin**19) + (46879712*mcMS**18)/(69615*mbkin**18) + 
            (684256*mcMS**19)/(28215*mbkin**21) + (76636672*mcMS**19)/
            (1100385*mbkin**20) - (1368512*mcMS**19)/(19305*mbkin**19) - 
            (2109772*mcMS**20)/(1700595*mbkin**22) - (1205584*mcMS**20)/
            (340119*mbkin**21) + (1205584*mcMS**20)/(340119*mbkin**20) - 
            (863516*np.pi**2)/405405 + (2315776*mcMS*np.pi**2)/(135135*mbkin) - 
            (26955671*mcMS**2*np.pi**2)/(405405*mbkin**2) + (147040*mcMS**3*np.pi**2)/
            (891*mbkin**3) - (3558496*mcMS**4*np.pi**2)/(10395*mbkin**4) + 
            (239168*mcMS**5*np.pi**2)/(405*mbkin**5) - (473089*mcMS**6*np.pi**2)/
            (567*mbkin**6) + (42592*mcMS**7*np.pi**2)/(45*mbkin**7) - 
            (158540*mcMS**8*np.pi**2)/(189*mbkin**8) + (1650304*mcMS**9*np.pi**2)/
            (2835*mbkin**9) - (42551*mcMS**10*np.pi**2)/(135*mbkin**10) + 
            (4070624*mcMS**11*np.pi**2)/(31185*mbkin**11) - (177992*mcMS**12*np.pi**2)/
            (4455*mbkin**12) + (230336*mcMS**13*np.pi**2)/(27027*mbkin**13) - 
            (458833*mcMS**14*np.pi**2)/(405405*mbkin**14) + (28576*mcMS**15*np.pi**2)/
            (405405*mbkin**15) + ((14120272*mcMS)/(264537*mbkin) - 
            (188264*mcMS**2)/(351*mbkin**2) + (3384144*mcMS**3)/(1105*mbkin**3) - 
            (574496308*mcMS**4)/(45045*mbkin**4) + (4777280*mcMS**5)/
                (117*mbkin**5) - (3997664*mcMS**6)/(39*mbkin**6) + 
            (2671680*mcMS**7)/(13*mbkin**7) - (496427248*mcMS**8)/
                (1485*mbkin**8) + (6688928*mcMS**9)/(15*mbkin**9) - 
            (4416016*mcMS**10)/(9*mbkin**10) + (84324832*mcMS**11)/
                (189*mbkin**11) - (334680*mcMS**12)/mbkin**12 + (9269312*mcMS**13)/
                (45*mbkin**13) - (60257056*mcMS**14)/(585*mbkin**14) + 
            (41246272*mcMS**15)/(1001*mbkin**15) - (167408*mcMS**16)/
                (13*mbkin**16) + (1063600*mcMS**17)/(351*mbkin**17) - 
            (11719928*mcMS**18)/(23205*mbkin**18) + (342128*mcMS**19)/
                (6435*mbkin**19) - (301396*mcMS**20)/(113373*mbkin**20) - 
            (16*mcMS**4*np.pi**2)/(3*mbkin**4))*np.log(mcMS**2/mbkin**2) + 
            (-19893032/2606175 + (3580064*mcMS)/(675675*mbkin) + 
            (222153992*mcMS**2)/(868725*mbkin**2) - (1987324448*mcMS**3)/
                (1403325*mbkin**3) + (1131148808*mcMS**4)/(280665*mbkin**4) - 
            (4045856*mcMS**5)/(525*mbkin**5) + (200330504*mcMS**6)/
                (18225*mbkin**6) - (521545376*mcMS**7)/(42525*mbkin**7) + 
            (153987464*mcMS**8)/(14175*mbkin**8) - (5546336*mcMS**9)/
                (729*mbkin**9) + (35346424*mcMS**10)/(8505*mbkin**10) - 
            (810886688*mcMS**11)/(467775*mbkin**11) + (749071544*mcMS**12)/
                (1403325*mbkin**12) - (298261792*mcMS**13)/(2606175*mbkin**13) + 
            (2062888*mcMS**14)/(135135*mbkin**14) - (17410912*mcMS**15)/
                (18243225*mbkin**15))*np.log(1 - mcMS/mbkin) + 
            ((-14120272*mcMS)/(264537*mbkin) + (188264*mcMS**2)/(351*mbkin**2) - 
            (3384144*mcMS**3)/(1105*mbkin**3) + (574496308*mcMS**4)/
                (45045*mbkin**4) - (4777280*mcMS**5)/(117*mbkin**5) + 
            (3997664*mcMS**6)/(39*mbkin**6) - (2671680*mcMS**7)/(13*mbkin**7) + 
            (496427248*mcMS**8)/(1485*mbkin**8) - (6688928*mcMS**9)/
                (15*mbkin**9) + (4416016*mcMS**10)/(9*mbkin**10) - 
            (84324832*mcMS**11)/(189*mbkin**11) + (334680*mcMS**12)/mbkin**12 - 
            (9269312*mcMS**13)/(45*mbkin**13) + (60257056*mcMS**14)/
                (585*mbkin**14) - (41246272*mcMS**15)/(1001*mbkin**15) + 
            (167408*mcMS**16)/(13*mbkin**16) - (1063600*mcMS**17)/
                (351*mbkin**17) + (11719928*mcMS**18)/(23205*mbkin**18) - 
            (342128*mcMS**19)/(6435*mbkin**19) + (301396*mcMS**20)/
                (113373*mbkin**20))*np.log(mu0**2/mus**2) + 
            (400315229/37413090 - (1214855480*mcMS)/(8729721*mbkin) + 
            (10181450*mcMS**2)/(11583*mbkin**2) - (793211480*mcMS**3)/
                (196911*mbkin**3) + (578057131*mcMS**4)/(38610*mbkin**4) - 
            (237591056*mcMS**5)/(5265*mbkin**5) + (115275460*mcMS**6)/
                (1053*mbkin**6) - (176088368*mcMS**7)/(819*mbkin**7) + 
            (10768320109*mcMS**8)/(31185*mbkin**8) - (185080064*mcMS**9)/
                (405*mbkin**9) + (22516192*mcMS**10)/(45*mbkin**10) - 
            (104769824*mcMS**11)/(231*mbkin**11) + (302512021*mcMS**12)/
                (891*mbkin**12) - (4028303216*mcMS**13)/(19305*mbkin**13) + 
            (42257454076*mcMS**14)/(405405*mbkin**14) - (2412926416*mcMS**15)/
                (57915*mbkin**15) + (1014911*mcMS**16)/(78*mbkin**16) - 
            (54775400*mcMS**17)/(17901*mbkin**17) + (319368038*mcMS**18)/
                (626535*mbkin**18) - (3934472*mcMS**19)/(73359*mbkin**19) + 
            (9117229*mcMS**20)/(3401190*mbkin**20))*np.log(mus**2/mbkin**2)) + 
        np.log(mu0**2/mus**2)*(-128/(9*mbkin**3) + (2291073463739113*mcMS)/
            (61582245444720*mbkin) + (512*mcMS**2)/(27*mbkin**5) - 
            (128*mcMS**2)/(3*mbkin**4) - (256*mcMS**2)/(3*mbkin**3) - 
            (3981404576993*mcMS**2)/(10213773570*mbkin**2) + 
            (331807347338102521*mcMS**3)/(178264394708400*mbkin**3) + 
            (256*mcMS**4)/(9*mbkin**7) + (32*mcMS**4)/mbkin**6 + 
            (128*mcMS**4)/mbkin**5 - (44002551759513161*mcMS**4)/
            (5243070432600*mbkin**4) + (41544203780004203*mcMS**5)/
            (1572921129780*mbkin**5) - (512*mcMS**6)/(9*mbkin**9) - 
            (256*mcMS**6)/(3*mbkin**7) - (37494808289083*mcMS**6)/
            (567431865*mbkin**6) + (100900938658973*mcMS**7)/
            (756575820*mbkin**7) + (640*mcMS**8)/(27*mbkin**11) + 
            (32*mcMS**8)/(3*mbkin**10) + (128*mcMS**8)/(3*mbkin**9) - 
            (446393825857277*mcMS**8)/(2057719950*mbkin**8) + 
            (45909984092659*mcMS**9)/(158722200*mbkin**9) - 
            (291816629210152*mcMS**10)/(916620705*mbkin**10) + 
            (374754229442903*mcMS**11)/(1294052760*mbkin**11) - 
            (796792967910857*mcMS**12)/(3666482820*mbkin**12) + 
            (50058521771447*mcMS**13)/(374130900*mbkin**13) - 
            (2685542939845591*mcMS**14)/(40125539025*mbkin**14) + 
            (4680590326837223*mcMS**15)/(174769014420*mbkin**15) - 
            (104512674225943*mcMS**16)/(12483501030*mbkin**16) + 
            (13914362059201*mcMS**17)/(7061374320*mbkin**17) - 
            (221819810282801*mcMS**18)/(675243919350*mbkin**18) + 
            (362691083547521*mcMS**19)/(10486140865200*mbkin**19) - 
            (22826272832761*mcMS**20)/(13196195452440*mbkin**20) + 
            ((32*mcMS**2)/mbkin**2 - (64*mcMS**4)/mbkin**6 - (256*mcMS**4)/
                (3*mbkin**5) - (430*mcMS**4)/(3*mbkin**4) - (288*mcMS**6)/mbkin**6 + 
            (64*mcMS**8)/mbkin**8)*np.log(mcMS**2/mbkin**2) + 
            (192*mcMS**4*np.log(mcMS**2/mbkin**2)**2)/mbkin**4 + 
            ((-14120272*mcMS)/(264537*mbkin) + (188264*mcMS**2)/(351*mbkin**2) - 
            (3384144*mcMS**3)/(1105*mbkin**3) + (574496308*mcMS**4)/
                (45045*mbkin**4) - (4777280*mcMS**5)/(117*mbkin**5) + 
            (3997664*mcMS**6)/(39*mbkin**6) - (2671680*mcMS**7)/(13*mbkin**7) + 
            (496427248*mcMS**8)/(1485*mbkin**8) - (6688928*mcMS**9)/
                (15*mbkin**9) + (4416016*mcMS**10)/(9*mbkin**10) - 
            (84324832*mcMS**11)/(189*mbkin**11) + (334680*mcMS**12)/mbkin**12 - 
            (9269312*mcMS**13)/(45*mbkin**13) + (60257056*mcMS**14)/
                (585*mbkin**14) - (41246272*mcMS**15)/(1001*mbkin**15) + 
            (167408*mcMS**16)/(13*mbkin**16) - (1063600*mcMS**17)/
                (351*mbkin**17) + (11719928*mcMS**18)/(23205*mbkin**18) - 
            (342128*mcMS**19)/(6435*mbkin**19) + (301396*mcMS**20)/
                (113373*mbkin**20))*np.log(1 - mcMS/mbkin) + 
            ((-32*mcMS**2)/mbkin**2 - (192*mcMS**4)/mbkin**4 + (288*mcMS**6)/
                mbkin**6 - (64*mcMS**8)/mbkin**8 - (192*mcMS**4*np.log(mcMS**2/mbkin**2))/
                mbkin**4)*np.log(mus**2/mbkin**2)) + 
        (-3640/(81*mbkin**3) - (472*mcMS**2)/(81*mbkin**5) + 
            (3080*mcMS**4)/(81*mbkin**7) + (848*mcMS**6)/(81*mbkin**9) - 
            (224*mcMS**8)/(27*mbkin**11))*fp.polylog(2, 1 - mbkin**2/mcMS**2) + 
        (-304/(81*mbkin**3) - (160*mcMS**2)/(9*mbkin**5) + 
            (464*mcMS**4)/(27*mbkin**7) + (4400*mcMS**6)/(81*mbkin**9) - 
            (32*mcMS**8)/(3*mbkin**11) - (1688*np.sqrt(0j + mcMS**2/mbkin**2))/
            (243*mbkin**3) + (2824*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (24976*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3))*
            fp.polylog(2, -(mcMS/mbkin)) + (-304/(81*mbkin**3) - 
            (160*mcMS**2)/(9*mbkin**5) + (464*mcMS**4)/(27*mbkin**7) + 
            (4400*mcMS**6)/(81*mbkin**9) - (32*mcMS**8)/(3*mbkin**11) - 
            (104*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**3) - 
            (5912*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) - 
            (70064*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3))*
            fp.polylog(2, mcMS/mbkin) + (-536/(81*mbkin**3) + (8*mcMS**2)/mbkin**5 - 
            (136*mcMS**4)/(9*mbkin**7) + (344*mcMS**6)/(81*mbkin**9) + 
            (256*mcMS**8)/(27*mbkin**11) + (500*np.sqrt(0j + mcMS**2/mbkin**2))/
            (243*mbkin**3) + (772*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (11272*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3))*
            fp.polylog(2, mcMS**2/mbkin**2)) + api4*((-3*muG)/mbkin**4 - 
        (8*muG)/mbkin**3 - (56*muG)/(27*mbkin**2) + (8*mcMS**2*muG)/(3*mbkin**6) + 
        (64*mcMS**2*muG)/(9*mbkin**5) + (2531*mcMS**2*muG)/(648*mbkin**4) + 
        (16*mcMS**4*muG)/mbkin**8 + (128*mcMS**4*muG)/(3*mbkin**7) - 
        (73481*mcMS**4*muG)/(648*mbkin**6) - (24*mcMS**6*muG)/mbkin**10 - 
        (64*mcMS**6*muG)/mbkin**9 + (32351*mcMS**6*muG)/(216*mbkin**8) + 
        (25*mcMS**8*muG)/(3*mbkin**12) + (200*mcMS**8*muG)/(9*mbkin**11) - 
        (917*mcMS**8*muG)/(24*mbkin**10) - mupi/mbkin**4 - (8*mupi)/(3*mbkin**3) + 
        (1425252426638329*mupi)/(446640461467200*mbkin**2) - 
        (2649349310527609*mcMS*mupi)/(123164490889440*mbkin**3) + 
        (8*mcMS**2*mupi)/(3*mbkin**6) + (64*mcMS**2*mupi)/(9*mbkin**5) + 
        (4197068578247*mcMS**2*mupi)/(40855094280*mbkin**4) - 
        (371136550584956041*mcMS**3*mupi)/(1069586368250400*mbkin**5) - 
        (8*mcMS**4*mupi)/mbkin**8 - (64*mcMS**4*mupi)/(3*mbkin**7) + 
        (47319152313660041*mcMS**4*mupi)/(41944563460800*mbkin**6) - 
        (45300689655370427*mcMS**5*mupi)/(15729211297800*mbkin**7) + 
        (8*mcMS**6*mupi)/mbkin**10 + (64*mcMS**6*mupi)/(3*mbkin**9) + 
        (40818484420001*mcMS**6*mupi)/(6809182380*mbkin**8) - 
        (328364278022567*mcMS**7*mupi)/(31776184440*mbkin**9) - 
        (5*mcMS**8*mupi)/(3*mbkin**12) - (40*mcMS**8*mupi)/(9*mbkin**11) + 
        (3384604682760869*mcMS**8*mupi)/(230464634400*mbkin**10) - 
        (546447454894529*mcMS**9*mupi)/(31426995600*mbkin**11) + 
        (157739121574127*mcMS**10*mupi)/(9166207050*mbkin**12) - 
        (6882359619137911*mcMS**11*mupi)/(483975732240*mbkin**13) + 
        (860250268694657*mcMS**12*mupi)/(87995587680*mbkin**14) - 
        (29086506506933*mcMS**13*mupi)/(5237832600*mbkin**15) + 
        (20276892020748997*mcMS**14*mupi)/(7864605648900*mbkin**16) - 
        (5046733464982007*mcMS**15*mupi)/(5243070432600*mbkin**17) + 
        (5929043471749*mcMS**16*mupi)/(21024843840*mbkin**18) - 
        (14993717402449*mcMS**17*mupi)/(240086726880*mbkin**19) + 
        (238966675564711*mcMS**18*mupi)/(24308781096600*mbkin**20) - 
        (22978828361473*mcMS**19*mupi)/(23439608992800*mbkin**21) + 
        (4916068298621*mcMS**20*mupi)/(105569563619520*mbkin**22) - 
        (19*muG*np.pi**2)/(54*mbkin**2) - (5*mcMS**2*muG*np.pi**2)/(3*mbkin**4) - 
        (4*mcMS**4*muG*np.pi**2)/(3*mbkin**6) + (275*mcMS**6*muG*np.pi**2)/(54*mbkin**8) - 
        (5*mcMS**8*muG*np.pi**2)/(3*mbkin**10) + (28*np.sqrt(0j + mcMS**2/mbkin**2)*muG*np.pi**2)/
            (9*mbkin**2) + (140*(mcMS**2/mbkin**2)**(3/2)*muG*np.pi**2)/(27*mbkin**2) + 
        (40*rhoD)/(9*mbkin**5) + (320*rhoD)/(27*mbkin**4) - 
        (66559*rhoD)/(3240*mbkin**3) + (371*rhoD)/(108*mbkin*mcMS**2) + 
        (178127*mcMS**2*rhoD)/(810*mbkin**5) + (32*mcMS**4*rhoD)/(3*mbkin**9) + 
        (256*mcMS**4*rhoD)/(9*mbkin**8) - (18289*mcMS**4*rhoD)/(120*mbkin**7) - 
        (256*mcMS**6*rhoD)/(9*mbkin**11) - (2048*mcMS**6*rhoD)/(27*mbkin**10) - 
        (4163*mcMS**6*rhoD)/(1620*mbkin**9) + (40*mcMS**8*rhoD)/(3*mbkin**13) + 
        (320*mcMS**8*rhoD)/(9*mbkin**12) - (5165*mcMS**8*rhoD)/(108*mbkin**11) - 
        (43*np.pi**2*rhoD)/(27*mbkin**3) - (mcMS**2*np.pi**2*rhoD)/(6*mbkin**5) - 
        (11*mcMS**4*np.pi**2*rhoD)/(9*mbkin**7) + (53*mcMS**6*np.pi**2*rhoD)/
            (9*mbkin**9) + (7*mcMS**8*np.pi**2*rhoD)/(9*mbkin**11) + 
        (43*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2*rhoD)/(54*mbkin**3) - 
        (91*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2*rhoD)/(2*mbkin**7) - 
        (55*(mcMS**2/mbkin**2)**(3/2)*np.pi**2*rhoD)/mbkin**3 + 
        ((-6511301*mupi)/(2238390*mbkin**2) + (7060136*mcMS*mupi)/
            (264537*mbkin**3) - (47066*mcMS**2*mupi)/(351*mbkin**4) + 
            (564024*mcMS**3*mupi)/(1105*mbkin**5) - (143624077*mcMS**4*mupi)/
            (90090*mbkin**6) + (477728*mcMS**5*mupi)/(117*mbkin**7) - 
            (999416*mcMS**6*mupi)/(117*mbkin**8) + (1335840*mcMS**7*mupi)/
            (91*mbkin**9) - (31026703*mcMS**8*mupi)/(1485*mbkin**10) + 
            (3344464*mcMS**9*mupi)/(135*mbkin**11) - (1104004*mcMS**10*mupi)/
            (45*mbkin**12) + (42162416*mcMS**11*mupi)/(2079*mbkin**13) - 
            (13945*mcMS**12*mupi)/mbkin**14 + (356512*mcMS**13*mupi)/
            (45*mbkin**15) - (15064264*mcMS**14*mupi)/(4095*mbkin**16) + 
            (20623136*mcMS**15*mupi)/(15015*mbkin**17) - (10463*mcMS**16*mupi)/
            (26*mbkin**18) + (531800*mcMS**17*mupi)/(5967*mbkin**19) - 
            (2929982*mcMS**18*mupi)/(208845*mbkin**20) + (171064*mcMS**19*mupi)/
            (122265*mbkin**21) - (75349*mcMS**20*mupi)/(1133730*mbkin**22))*
            np.log(2) + ((-7*mcMS**2*muG)/(6*mbkin**4) + (293*mcMS**4*muG)/
            (18*mbkin**6) + (269*mcMS**6*muG)/(54*mbkin**8) - 
            (5*mcMS**8*muG)/(3*mbkin**10) - (24*mcMS**4*mupi)/mbkin**6 + 
            (115*rhoD)/(9*mbkin**3) + (77*mcMS**2*rhoD)/(18*mbkin**5) - 
            (257*mcMS**4*rhoD)/(54*mbkin**7) + (127*mcMS**6*rhoD)/(9*mbkin**9) + 
            (67*mcMS**8*rhoD)/(9*mbkin**11))*np.log(mcMS**2/mbkin**2)**2 + 
        ((-6511301*mupi)/(2238390*mbkin**2) + (7060136*mcMS*mupi)/
            (264537*mbkin**3) - (47066*mcMS**2*mupi)/(351*mbkin**4) + 
            (564024*mcMS**3*mupi)/(1105*mbkin**5) - (143624077*mcMS**4*mupi)/
            (90090*mbkin**6) + (477728*mcMS**5*mupi)/(117*mbkin**7) - 
            (999416*mcMS**6*mupi)/(117*mbkin**8) + (1335840*mcMS**7*mupi)/
            (91*mbkin**9) - (31026703*mcMS**8*mupi)/(1485*mbkin**10) + 
            (3344464*mcMS**9*mupi)/(135*mbkin**11) - (1104004*mcMS**10*mupi)/
            (45*mbkin**12) + (42162416*mcMS**11*mupi)/(2079*mbkin**13) - 
            (13945*mcMS**12*mupi)/mbkin**14 + (356512*mcMS**13*mupi)/
            (45*mbkin**15) - (15064264*mcMS**14*mupi)/(4095*mbkin**16) + 
            (20623136*mcMS**15*mupi)/(15015*mbkin**17) - (10463*mcMS**16*mupi)/
            (26*mbkin**18) + (531800*mcMS**17*mupi)/(5967*mbkin**19) - 
            (2929982*mcMS**18*mupi)/(208845*mbkin**20) + (171064*mcMS**19*mupi)/
            (122265*mbkin**21) - (75349*mcMS**20*mupi)/(1133730*mbkin**22))*
            np.log(1 - mcMS/mbkin) + ((79*muG)/(324*mbkin**2) - 
            (61*muG)/(108*mcMS**2) - (50*mcMS**2*muG)/(9*mbkin**4) + 
            (976*mcMS**4*muG)/(27*mbkin**6) - (11905*mcMS**6*muG)/(324*mbkin**8) + 
            (233*mcMS**8*muG)/(36*mbkin**10) - (13769*rhoD)/(108*mbkin**3) + 
            (371*mbkin*rhoD)/(108*mcMS**4) - (11231*rhoD)/(810*mbkin*mcMS**2) + 
            (3377*mcMS**2*rhoD)/(27*mbkin**5) + (20435*mcMS**4*rhoD)/
            (324*mbkin**7) - (7121*mcMS**6*rhoD)/(54*mbkin**9) + 
            (44089*mcMS**8*rhoD)/(540*mbkin**11))*np.log(1 - mcMS**2/mbkin**2) + 
        np.log(mcMS**2/mbkin**2)*((-611*mcMS**2*muG)/(54*mbkin**4) + 
            (4*mcMS**4*muG)/mbkin**8 + (32*mcMS**4*muG)/(3*mbkin**7) - 
            (434*mcMS**4*muG)/(27*mbkin**6) - (13835*mcMS**6*muG)/(324*mbkin**8) + 
            (487*mcMS**8*muG)/(36*mbkin**10) - (8*mcMS**2*mupi)/mbkin**4 - 
            (4*mcMS**4*mupi)/mbkin**8 - (32*mcMS**4*mupi)/(3*mbkin**7) + 
            (20*mcMS**4*mupi)/mbkin**6 + (24*mcMS**6*mupi)/mbkin**8 - 
            (4*mcMS**8*mupi)/mbkin**10 + (32*rhoD)/(3*mbkin**5) + 
            (256*rhoD)/(9*mbkin**4) - (5345*rhoD)/(54*mbkin**3) + 
            (326*mcMS**2*rhoD)/(3*mbkin**5) + (21293*mcMS**4*rhoD)/(648*mbkin**7) + 
            (731*mcMS**6*rhoD)/(18*mbkin**9) + (16207*mcMS**8*rhoD)/
            (1080*mbkin**11) + ((-56*np.sqrt(0j + mcMS**2/mbkin**2)*muG)/(9*mbkin**2) - 
            (280*(mcMS**2/mbkin**2)**(3/2)*muG)/(27*mbkin**2) - 
            (43*np.sqrt(0j + mcMS**2/mbkin**2)*rhoD)/(27*mbkin**3) + 
            (91*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*rhoD)/mbkin**7 + 
            (110*(mcMS**2/mbkin**2)**(3/2)*rhoD)/mbkin**3)*np.log(1 - mcMS/mbkin) + 
            ((56*np.sqrt(0j + mcMS**2/mbkin**2)*muG)/(9*mbkin**2) + 
            (280*(mcMS**2/mbkin**2)**(3/2)*muG)/(27*mbkin**2) + 
            (43*np.sqrt(0j + mcMS**2/mbkin**2)*rhoD)/(27*mbkin**3) - 
            (91*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*rhoD)/mbkin**7 - 
            (110*(mcMS**2/mbkin**2)**(3/2)*rhoD)/mbkin**3)*np.log(1 + mcMS/mbkin) + 
            ((35*muG)/(27*mbkin**2) + (5*mcMS**2*muG)/mbkin**4 + 
            (215*mcMS**4*muG)/(9*mbkin**6) - (547*mcMS**6*muG)/(27*mbkin**8) + 
            (20*mcMS**8*muG)/(3*mbkin**10) - (427*rhoD)/(9*mbkin**3) - 
            (11*mcMS**2*rhoD)/mbkin**5 - (3293*mcMS**4*rhoD)/(27*mbkin**7) - 
            (190*mcMS**6*rhoD)/(9*mbkin**9) - (88*mcMS**8*rhoD)/(9*mbkin**11))*
            np.log(1 - mcMS**2/mbkin**2)) + ((8*mcMS**2*muG)/mbkin**4 - 
            (60*mcMS**4*muG)/mbkin**6 + (72*mcMS**6*muG)/mbkin**8 - 
            (20*mcMS**8*muG)/mbkin**10 + (8*mcMS**2*mupi)/mbkin**4 + 
            (12*mcMS**4*mupi)/mbkin**6 - (24*mcMS**6*mupi)/mbkin**8 + 
            (4*mcMS**8*mupi)/mbkin**10 + (16*rhoD)/mbkin**3 - 
            (64*mcMS**2*rhoD)/(3*mbkin**5) - (32*mcMS**4*rhoD)/mbkin**7 + 
            (64*mcMS**6*rhoD)/mbkin**9 - (80*mcMS**8*rhoD)/(3*mbkin**11) + 
            ((-24*mcMS**4*muG)/mbkin**6 + (24*mcMS**4*mupi)/mbkin**6)*
            np.log(mcMS**2/mbkin**2))*np.log(mu0**2/mus**2) + 
        ((-9*muG)/(4*mbkin**2) + (6*mcMS**2*muG)/mbkin**4 - 
            (18*mcMS**4*muG)/mbkin**6 + (18*mcMS**6*muG)/mbkin**8 - 
            (15*mcMS**8*muG)/(4*mbkin**10) - (9*rhoD)/(4*mbkin**3) + 
            (6*mcMS**2*rhoD)/mbkin**5 - (18*mcMS**4*rhoD)/mbkin**7 + 
            (18*mcMS**6*rhoD)/mbkin**9 - (15*mcMS**8*rhoD)/(4*mbkin**11) + 
            ((-9*mcMS**4*muG)/mbkin**6 - (9*mcMS**4*rhoD)/mbkin**7)*
            np.log(mcMS**2/mbkin**2))*np.log(mus/mbkin) + 
        ((8*mcMS**2*muG)/mbkin**4 - (60*mcMS**4*muG)/mbkin**6 + 
            (72*mcMS**6*muG)/mbkin**8 - (20*mcMS**8*muG)/mbkin**10 + 
            (8*mcMS**2*mupi)/mbkin**4 + (12*mcMS**4*mupi)/mbkin**6 - 
            (24*mcMS**6*mupi)/mbkin**8 + (4*mcMS**8*mupi)/mbkin**10 + 
            (16*rhoD)/mbkin**3 - (64*mcMS**2*rhoD)/(3*mbkin**5) - 
            (32*mcMS**4*rhoD)/mbkin**7 + (64*mcMS**6*rhoD)/mbkin**9 - 
            (80*mcMS**8*rhoD)/(3*mbkin**11) + ((-24*mcMS**4*muG)/mbkin**6 + 
            (24*mcMS**4*mupi)/mbkin**6)*np.log(mcMS**2/mbkin**2))*
            np.log(mus**2/mbkin**2) + ((455*rhoD)/(9*mbkin**3) + (59*mcMS**2*rhoD)/
            (9*mbkin**5) - (385*mcMS**4*rhoD)/(9*mbkin**7) - 
            (106*mcMS**6*rhoD)/(9*mbkin**9) + (28*mcMS**8*rhoD)/(3*mbkin**11))*
            fp.polylog(2, 1 - mbkin**2/mcMS**2) + ((38*muG)/(9*mbkin**2) + 
            (20*mcMS**2*muG)/mbkin**4 + (16*mcMS**4*muG)/mbkin**6 - 
            (550*mcMS**6*muG)/(9*mbkin**8) + (20*mcMS**8*muG)/mbkin**10 + 
            (112*np.sqrt(0j + mcMS**2/mbkin**2)*muG)/(9*mbkin**2) + 
            (560*(mcMS**2/mbkin**2)**(3/2)*muG)/(27*mbkin**2) + 
            (38*rhoD)/(9*mbkin**3) + (20*mcMS**2*rhoD)/mbkin**5 - 
            (58*mcMS**4*rhoD)/(3*mbkin**7) - (550*mcMS**6*rhoD)/(9*mbkin**9) + 
            (12*mcMS**8*rhoD)/mbkin**11 + (211*np.sqrt(0j + mcMS**2/mbkin**2)*rhoD)/
            (27*mbkin**3) - (353*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*rhoD)/(3*mbkin**7) - 
            (3122*(mcMS**2/mbkin**2)**(3/2)*rhoD)/(27*mbkin**3))*
            fp.polylog(2, -(mcMS/mbkin)) + ((38*muG)/(9*mbkin**2) + 
            (20*mcMS**2*muG)/mbkin**4 + (16*mcMS**4*muG)/mbkin**6 - 
            (550*mcMS**6*muG)/(9*mbkin**8) + (20*mcMS**8*muG)/mbkin**10 - 
            (112*np.sqrt(0j + mcMS**2/mbkin**2)*muG)/(9*mbkin**2) - 
            (560*(mcMS**2/mbkin**2)**(3/2)*muG)/(27*mbkin**2) + 
            (38*rhoD)/(9*mbkin**3) + (20*mcMS**2*rhoD)/mbkin**5 - 
            (58*mcMS**4*rhoD)/(3*mbkin**7) - (550*mcMS**6*rhoD)/(9*mbkin**9) + 
            (12*mcMS**8*rhoD)/mbkin**11 + (13*np.sqrt(0j + mcMS**2/mbkin**2)*rhoD)/
            (9*mbkin**3) + (739*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*rhoD)/(3*mbkin**7) + 
            (8758*(mcMS**2/mbkin**2)**(3/2)*rhoD)/(27*mbkin**3))*
            fp.polylog(2, mcMS/mbkin) + ((67*rhoD)/(9*mbkin**3) - 
            (9*mcMS**2*rhoD)/mbkin**5 + (17*mcMS**4*rhoD)/mbkin**7 - 
            (43*mcMS**6*rhoD)/(9*mbkin**9) - (32*mcMS**8*rhoD)/(3*mbkin**11) - 
            (125*np.sqrt(0j + mcMS**2/mbkin**2)*rhoD)/(54*mbkin**3) - 
            (193*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*rhoD)/(6*mbkin**7) - 
            (1409*(mcMS**2/mbkin**2)**(3/2)*rhoD)/(27*mbkin**3))*
            fp.polylog(2, mcMS**2/mbkin**2)) + api4**3*(-184950.44420212752 + 
        1216/(243*mbkin**7) + 22264/(729*mbkin**6) - 134936/(1215*mbkin**5) - 
        35682230739408427/(115954735188600*mbkin**4) - 557.6112919538557/
            mbkin**3 - 1939.9941735508244/mbkin**2 - 5389.774145633451/mbkin - 
        5936/(729*mbkin**3*mcMS**2) - 47488/(2187*mbkin**2*mcMS**2) - 
        42665/(1458*mbkin*mcMS**2) + (2956442893489177*mcMS)/
            (19794293178660*mbkin**5) + (23822150694003176*mcMS)/
            (34640013062655*mbkin**4) + (37021.61545685546*mcMS)/mbkin**3 + 
        (78388.70801171157*mcMS)/mbkin**2 - (9358.145632576896*mcMS)/mbkin - 
        (64*mcMS**2)/(27*mbkin**8) + (7136*mcMS**2)/(81*mbkin**7) - 
        (3288214427347*mcMS**2)/(9192396213*mbkin**6) - 
        (4391.631981159506*mcMS**2)/mbkin**5 - (189491.5505261303*mcMS**2)/
            mbkin**4 - (382856.57863412646*mcMS**2)/mbkin**3 - 
        (41517.03524694126*mcMS**2)/mbkin**2 - (281211089367011689*mcMS**3)/
            (240656932856340*mbkin**7) - (234407880272854196*mcMS**3)/
            (36098539928451*mbkin**6) + (540971.5533639141*mcMS**3)/mbkin**5 + 
        (1.013058676530211e6*mcMS**3)/mbkin**4 + (596614.4806078207*mcMS**3)/
            mbkin**3 + (256*mcMS**4)/(27*mbkin**11) + (3136*mcMS**4)/(81*mbkin**10) - 
        (4687112*mcMS**4)/(10935*mbkin**9) + (351272943059250449*mcMS**4)/
            (38608064094600*mbkin**8) + (56526.49268336005*mcMS**4)/mbkin**7 - 
        (1.0151475895039897e6*mcMS**4)/mbkin**6 - (1.625405859436086e6*mcMS**4)/
            mbkin**5 - (2.854148163798943e6*mcMS**4)/mbkin**4 - 
        (737433567890477179*mcMS**5)/(17695362710025*mbkin**9) - 
        (11224385486776356992*mcMS**5)/(53086088130075*mbkin**8) + 
        (1.1264036180687936e6*mcMS**5)/mbkin**7 + 
        (2732437565404654865393*mcMS**5)/(2388873965853375*mbkin**6) + 
        (8.1524835473563485e6*mcMS**5)/mbkin**5 - (10240*mcMS**6)/
            (243*mbkin**13) - (155200*mcMS**6)/(729*mbkin**12) + 
        (5732192*mcMS**6)/(10935*mbkin**11) + (17019811258504622*mcMS**6)/
            (137885943195*mbkin**10) + (609482.2588054772*mcMS**6)/mbkin**9 - 
        (8562077579317878554*mcMS**6)/(31851652878045*mbkin**8) + 
        (1.2675380131069066e6*mcMS**6)/mbkin**7 - 
        (1.6035346707621306e7*mcMS**6)/mbkin**6 - (9586294363501019*mcMS**7)/
            (35748207495*mbkin**11) - (144310094793850448*mcMS**7)/
            (107244622485*mbkin**10) - (1.4887670056813075e6*mcMS**7)/mbkin**9 - 
        (4.459180020379724e6*mcMS**7)/mbkin**8 + (2.3063497421273895e7*mcMS**7)/
            mbkin**7 + (2240*mcMS**8)/(81*mbkin**15) + (35480*mcMS**8)/
            (243*mbkin**14) - (1052272*mcMS**8)/(3645*mbkin**13) + 
        (72593407777125481*mcMS**8)/(155563628220*mbkin**12) + 
        (2.344574359246224e6*mcMS**8)/mbkin**11 + 
        (3.2123413245152263e6*mcMS**8)/mbkin**10 + 
        (5.668262022837839e6*mcMS**8)/mbkin**9 - (2.4941208894620407e7*mcMS**8)/
            mbkin**8 - (23319274886243689*mcMS**9)/(35355370050*mbkin**13) - 
        (174931214847990896*mcMS**9)/(53033055075*mbkin**12) - 
        (3.908746005812783e6*mcMS**9)/mbkin**11 - 
        (3.3864788670419026e6*mcMS**9)/mbkin**10 + (2.04804406819528e7*mcMS**9)/
            mbkin**9 + (1841912917164548*mcMS**10)/(2426348925*mbkin**14) + 
        (156484202822082976*mcMS**10)/(41247931725*mbkin**13) + 
        (3.4356955309533565e6*mcMS**10)/mbkin**12 - (801132.8616024388*mcMS**10)/
            mbkin**11 - (1.2690854980564993e7*mcMS**10)/mbkin**10 - 
        (390379529238350627*mcMS**11)/(544472698770*mbkin**15) - 
        (417901197959024072*mcMS**11)/(116672721165*mbkin**14) - 
        (2.410262386901205e6*mcMS**11)/mbkin**13 + 
        (3.919124303220365e6*mcMS**11)/mbkin**12 + 
        (5.771791137892302e6*mcMS**11)/mbkin**11 + (846467766572027*mcMS**12)/
            (1523000556*mbkin**16) + (206120089691034394*mcMS**12)/
            (74246277105*mbkin**15) + (1.4560504030793202e6*mcMS**12)/mbkin**14 - 
        (4.5060883170757415e6*mcMS**12)/mbkin**13 - 
        (1.7829972333634205e6*mcMS**12)/mbkin**12 - (414993058959041*mcMS**13)/
            (1178512335*mbkin**17) - (230327730768064*mcMS**13)/
            (130945815*mbkin**16) - (796039.3687588704*mcMS**13)/mbkin**15 + 
        (3.304216223480193e6*mcMS**13)/mbkin**14 + (271330.00689752307*mcMS**13)/
            mbkin**13 + (3198040036461786454*mcMS**14)/(17695362710025*mbkin**18) + 
        (47931000070706648432*mcMS**14)/(53086088130075*mbkin**17) + 
        (391559.1449260058*mcMS**14)/mbkin**16 - (1.767535228397757e6*mcMS**14)/
            mbkin**15 + (52656.25176854513*mcMS**14)/mbkin**14 - 
        (5670900751504007*mcMS**15)/(76603301775*mbkin**19) - 
        (6546000489792038032*mcMS**15)/(17695362710025*mbkin**18) - 
        (164100.7637367768*mcMS**15)/mbkin**17 + (719255.8226277549*mcMS**15)/
            mbkin**16 - (49339.79768400003*mcMS**15)/mbkin**15 + 
        (10626401697487309*mcMS**16)/(449406037080*mbkin**20) + 
        (19918556085249223*mcMS**16)/(168527263905*mbkin**19) + 
        (147939217797606043*mcMS**16)/(2696436222480*mbkin**18) - 
        (302247115159911569*mcMS**16)/(1348218111240*mbkin**17) + 
        (261014626447576501*mcMS**16)/(14380993186560*mbkin**16) - 
        (1254711243948307*mcMS**17)/(220988919060*mbkin**21) - 
        (51759160683342248*mcMS**17)/(1823158582245*mbkin**20) - 
        (602779880445452767*mcMS**17)/(43755805973880*mbkin**19) + 
        (76458363312156827*mcMS**17)/(1458526865796*mbkin**18) - 
        (1442254413489287*mcMS**17)/(317502446976*mbkin**17) + 
        (5274955447780621*mcMS**18)/(5469475746735*mbkin**22) + 
        (2261654762030168*mcMS**18)/(468812206863*mbkin**21) + 
        (47813095251824449*mcMS**18)/(19690112688246*mbkin**20) - 
        (713912912589144041*mcMS**18)/(82042136201025*mbkin**19) + 
        (33436032246808409*mcMS**18)/(41672196165600*mbkin**18) - 
        (46331567952815701*mcMS**19)/(448282521987300*mbkin**23) - 
        (6440290782567676*mcMS**19)/(12452292277425*mbkin**22) - 
        (2166181719446920759*mcMS**19)/(8069085395771400*mbkin**21) + 
        (8595684879073057*mcMS**19)/(9404528433300*mbkin**20) - 
        (60946383895204079*mcMS**19)/(683097176361600*mbkin**19) + 
        (624247152931711*mcMS**20)/(118765759071960*mbkin**24) + 
        (1171937419428241*mcMS**20)/(44537159651985*mbkin**23) + 
        (158571553853231*mcMS**20)/(11311024673520*mbkin**22) - 
        (1808151255223099*mcMS**20)/(39588586357320*mbkin**21) + 
        (850401221316739*mcMS**20)/(180976394776320*mbkin**20) - 
        (3820703038853*mcMS**2)/(45961981065*mbkin**6*(1 - mcMS/mbkin)**2) - 
        (61131248621648*mcMS**2)/(137885943195*mbkin**5*(1 - mcMS/mbkin)**2) - 
        (15282812155412*mcMS**2)/(59093975655*mbkin**4*(1 - mcMS/mbkin)**2) + 
        (122262497243296*mcMS**2)/(137885943195*mbkin**3*(1 - mcMS/mbkin)**2) - 
        (15282812155412*mcMS**2)/(45961981065*mbkin**2*(1 - mcMS/mbkin)**2) + 
        (1324623229305471443*mcMS**3)/(1203284664281700*mbkin**7*
            (1 - mcMS/mbkin)**2) + (5298492917221885772*mcMS**3)/
            (902463498211275*mbkin**6*(1 - mcMS/mbkin)**2) + 
        (1324623229305471443*mcMS**3)/(386770070661975*mbkin**5*
            (1 - mcMS/mbkin)**2) - (10596985834443771544*mcMS**3)/
            (902463498211275*mbkin**4*(1 - mcMS/mbkin)**2) + 
        (1324623229305471443*mcMS**3)/(300821166070425*mbkin**3*
            (1 - mcMS/mbkin)**2) - (5514257196652051*mcMS**4)/
            (695942547300*mbkin**8*(1 - mcMS/mbkin)**2) - 
        (22057028786608204*mcMS**4)/(521956910475*mbkin**7*(1 - mcMS/mbkin)**2) - 
        (5514257196652051*mcMS**4)/(223695818775*mbkin**6*(1 - mcMS/mbkin)**2) + 
        (44114057573216408*mcMS**4)/(521956910475*mbkin**5*(1 - mcMS/mbkin)**2) - 
        (5514257196652051*mcMS**4)/(173985636825*mbkin**4*(1 - mcMS/mbkin)**2) + 
        (2445534345846504631*mcMS**5)/(63330771804300*mbkin**9*
            (1 - mcMS/mbkin)**2) + (9782137383386018524*mcMS**5)/
            (47498078853225*mbkin**8*(1 - mcMS/mbkin)**2) + 
        (2445534345846504631*mcMS**5)/(20356319508525*mbkin**7*
            (1 - mcMS/mbkin)**2) - (19564274766772037048*mcMS**5)/
            (47498078853225*mbkin**6*(1 - mcMS/mbkin)**2) + 
        (2445534345846504631*mcMS**5)/(15832692951075*mbkin**5*
            (1 - mcMS/mbkin)**2) - (73546477272645373*mcMS**6)/
            (532191359700*mbkin**10*(1 - mcMS/mbkin)**2) - 
        (294185909090581492*mcMS**6)/(399143519775*mbkin**9*
            (1 - mcMS/mbkin)**2) - (73546477272645373*mcMS**6)/
            (171061508475*mbkin**8*(1 - mcMS/mbkin)**2) + 
        (588371818181162984*mcMS**6)/(399143519775*mbkin**7*
            (1 - mcMS/mbkin)**2) - (73546477272645373*mcMS**6)/
            (133047839925*mbkin**6*(1 - mcMS/mbkin)**2) + 
        (70971441129844033*mcMS**7)/(186266975895*mbkin**11*
            (1 - mcMS/mbkin)**2) + (1135543058077504528*mcMS**7)/
            (558800927685*mbkin**10*(1 - mcMS/mbkin)**2) + 
        (283885764519376132*mcMS**7)/(239486111865*mbkin**9*
            (1 - mcMS/mbkin)**2) - (2271086116155009056*mcMS**7)/
            (558800927685*mbkin**8*(1 - mcMS/mbkin)**2) + 
        (283885764519376132*mcMS**7)/(186266975895*mbkin**7*
            (1 - mcMS/mbkin)**2) - (5292307230847823*mcMS**8)/
            (6335611425*mbkin**12*(1 - mcMS/mbkin)**2) - (84676915693565168*mcMS**8)/
            (19006834275*mbkin**11*(1 - mcMS/mbkin)**2) - 
        (148184602463739044*mcMS**8)/(57020502825*mbkin**10*
            (1 - mcMS/mbkin)**2) + (169353831387130336*mcMS**8)/
            (19006834275*mbkin**9*(1 - mcMS/mbkin)**2) - (21169228923391292*mcMS**8)/
            (6335611425*mbkin**8*(1 - mcMS/mbkin)**2) + (1883705490346937*mcMS**9)/
            (1267122285*mbkin**13*(1 - mcMS/mbkin)**2) + (30139287845550992*mcMS**9)/
            (3801366855*mbkin**12*(1 - mcMS/mbkin)**2) + (52743753729714236*mcMS**9)/
            (11404100565*mbkin**11*(1 - mcMS/mbkin)**2) - 
        (60278575691101984*mcMS**9)/(3801366855*mbkin**10*(1 - mcMS/mbkin)**2) + 
        (7534821961387748*mcMS**9)/(1267122285*mbkin**9*(1 - mcMS/mbkin)**2) - 
        (3996921465647711*mcMS**10)/(1836952425*mbkin**14*(1 - mcMS/mbkin)**2) - 
        (63950743450363376*mcMS**10)/(5510857275*mbkin**13*(1 - mcMS/mbkin)**2) - 
        (15987685862590844*mcMS**10)/(2361795975*mbkin**12*(1 - mcMS/mbkin)**2) + 
        (127901486900726752*mcMS**10)/(5510857275*mbkin**11*
            (1 - mcMS/mbkin)**2) - (15987685862590844*mcMS**10)/
            (1836952425*mbkin**10*(1 - mcMS/mbkin)**2) + 
        (34379222836281097*mcMS**11)/(13025662650*mbkin**15*
            (1 - mcMS/mbkin)**2) + (275033782690248776*mcMS**11)/
            (19538493975*mbkin**14*(1 - mcMS/mbkin)**2) + 
        (68758445672562194*mcMS**11)/(8373640275*mbkin**13*(1 - mcMS/mbkin)**2) - 
        (550067565380497552*mcMS**11)/(19538493975*mbkin**12*
            (1 - mcMS/mbkin)**2) + (68758445672562194*mcMS**11)/
            (6512831325*mbkin**11*(1 - mcMS/mbkin)**2) - (991151981540419*mcMS**12)/
            (372161790*mbkin**16*(1 - mcMS/mbkin)**2) - (7929215852323352*mcMS**12)/
            (558242685*mbkin**15*(1 - mcMS/mbkin)**2) - (1982303963080838*mcMS**12)/
            (239246865*mbkin**14*(1 - mcMS/mbkin)**2) + (15858431704646704*mcMS**12)/
            (558242685*mbkin**13*(1 - mcMS/mbkin)**2) - (1982303963080838*mcMS**12)/
            (186080895*mbkin**12*(1 - mcMS/mbkin)**2) + (2648132431926677*mcMS**13)/
            (1184151150*mbkin**17*(1 - mcMS/mbkin)**2) + 
        (21185059455413416*mcMS**13)/(1776226725*mbkin**16*(1 - mcMS/mbkin)**2) + 
        (5296264863853354*mcMS**13)/(761240025*mbkin**15*(1 - mcMS/mbkin)**2) - 
        (42370118910826832*mcMS**13)/(1776226725*mbkin**14*(1 - mcMS/mbkin)**2) + 
        (5296264863853354*mcMS**13)/(592075575*mbkin**13*(1 - mcMS/mbkin)**2) - 
        (223279079747087713*mcMS**14)/(143282289150*mbkin**18*
            (1 - mcMS/mbkin)**2) - (1786232637976701704*mcMS**14)/
            (214923433725*mbkin**17*(1 - mcMS/mbkin)**2) - 
        (446558159494175426*mcMS**14)/(92110043025*mbkin**16*
            (1 - mcMS/mbkin)**2) + (3572465275953403408*mcMS**14)/
            (214923433725*mbkin**15*(1 - mcMS/mbkin)**2) - 
        (446558159494175426*mcMS**14)/(71641144575*mbkin**14*
            (1 - mcMS/mbkin)**2) + (4765673910687463*mcMS**15)/
            (5321913597*mbkin**19*(1 - mcMS/mbkin)**2) + 
        (76250782570999408*mcMS**15)/(15965740791*mbkin**18*
            (1 - mcMS/mbkin)**2) + (19062695642749852*mcMS**15)/
            (6842460339*mbkin**17*(1 - mcMS/mbkin)**2) - 
        (152501565141998816*mcMS**15)/(15965740791*mbkin**16*
            (1 - mcMS/mbkin)**2) + (19062695642749852*mcMS**15)/
            (5321913597*mbkin**15*(1 - mcMS/mbkin)**2) - 
        (55872539918685143*mcMS**16)/(133047839925*mbkin**20*
            (1 - mcMS/mbkin)**2) - (893960638698962288*mcMS**16)/
            (399143519775*mbkin**19*(1 - mcMS/mbkin)**2) - 
        (223490159674740572*mcMS**16)/(171061508475*mbkin**18*
            (1 - mcMS/mbkin)**2) + (1787921277397924576*mcMS**16)/
            (399143519775*mbkin**17*(1 - mcMS/mbkin)**2) - 
        (223490159674740572*mcMS**16)/(133047839925*mbkin**16*
            (1 - mcMS/mbkin)**2) + (251792791028627*mcMS**17)/
            (1592025435*mbkin**21*(1 - mcMS/mbkin)**2) + (4028684656458032*mcMS**17)/
            (4776076305*mbkin**20*(1 - mcMS/mbkin)**2) + (1007171164114508*mcMS**17)/
            (2046889845*mbkin**19*(1 - mcMS/mbkin)**2) - (8057369312916064*mcMS**17)/
            (4776076305*mbkin**18*(1 - mcMS/mbkin)**2) + (1007171164114508*mcMS**17)/
            (1592025435*mbkin**17*(1 - mcMS/mbkin)**2) - (4832108312706614*mcMS**18)/
            (103481653275*mbkin**22*(1 - mcMS/mbkin)**2) - 
        (77313733003305824*mcMS**18)/(310444959825*mbkin**21*
            (1 - mcMS/mbkin)**2) - (19328433250826456*mcMS**18)/
            (133047839925*mbkin**20*(1 - mcMS/mbkin)**2) + 
        (154627466006611648*mcMS**18)/(310444959825*mbkin**19*
            (1 - mcMS/mbkin)**2) - (19328433250826456*mcMS**18)/
            (103481653275*mbkin**18*(1 - mcMS/mbkin)**2) + 
        (1436625073319083*mcMS**19)/(137975537700*mbkin**23*
            (1 - mcMS/mbkin)**2) + (5746500293276332*mcMS**19)/
            (103481653275*mbkin**22*(1 - mcMS/mbkin)**2) + 
        (1436625073319083*mcMS**19)/(44349279975*mbkin**21*(1 - mcMS/mbkin)**2) - 
        (11493000586552664*mcMS**19)/(103481653275*mbkin**20*
            (1 - mcMS/mbkin)**2) + (1436625073319083*mcMS**19)/
            (34493884425*mbkin**19*(1 - mcMS/mbkin)**2) - 
        (20887342577910533*mcMS**20)/(12666154360860*mbkin**24*
            (1 - mcMS/mbkin)**2) - (83549370311642132*mcMS**20)/
            (9499615770645*mbkin**23*(1 - mcMS/mbkin)**2) - 
        (20887342577910533*mcMS**20)/(4071263901705*mbkin**22*
            (1 - mcMS/mbkin)**2) + (167098740623284264*mcMS**20)/
            (9499615770645*mbkin**21*(1 - mcMS/mbkin)**2) - 
        (20887342577910533*mcMS**20)/(3166538590215*mbkin**20*
            (1 - mcMS/mbkin)**2) + (198978903088724189*mcMS**21)/
            (1203284664281700*mbkin**25*(1 - mcMS/mbkin)**2) + 
        (795915612354896756*mcMS**21)/(902463498211275*mbkin**24*
            (1 - mcMS/mbkin)**2) + (198978903088724189*mcMS**21)/
            (386770070661975*mbkin**23*(1 - mcMS/mbkin)**2) - 
        (1591831224709793512*mcMS**21)/(902463498211275*mbkin**22*
            (1 - mcMS/mbkin)**2) + (198978903088724189*mcMS**21)/
            (300821166070425*mbkin**21*(1 - mcMS/mbkin)**2) - 
        (24672660896281*mcMS**22)/(3125414712420*mbkin**26*(1 - mcMS/mbkin)**2) - 
        (98690643585124*mcMS**22)/(2344061034315*mbkin**25*(1 - mcMS/mbkin)**2) - 
        (24672660896281*mcMS**22)/(1004597586135*mbkin**24*(1 - mcMS/mbkin)**2) + 
        (197381287170248*mcMS**22)/(2344061034315*mbkin**23*
            (1 - mcMS/mbkin)**2) - (24672660896281*mcMS**22)/
            (781353678105*mbkin**22*(1 - mcMS/mbkin)**2) + (2291073463739113*mcMS)/
            (138560052250620*mbkin**5*(1 - mcMS/mbkin)) + (9164293854956452*mcMS)/
            (103920039187965*mbkin**4*(1 - mcMS/mbkin)) - (84769718158347181*mcMS)/
            (356297277215880*mbkin**3*(1 - mcMS/mbkin)) - 
        (455923619284083487*mcMS)/(415680156751860*mbkin**2*(1 - mcMS/mbkin)) + 
        (294.4372408189296*mcMS)/(mbkin*(1 - mcMS/mbkin)) - 
        (66120814157661443*mcMS**2)/(415680156751860*mbkin**6*
            (1 - mcMS/mbkin)) - (264483256630645772*mcMS**2)/
            (311760117563895*mbkin**5*(1 - mcMS/mbkin)) + 
        (2446470123833473391*mcMS**2)/(1068891831647640*mbkin**4*
            (1 - mcMS/mbkin)) + (13158042017374627157*mcMS**2)/
            (1247040470255580*mbkin**3*(1 - mcMS/mbkin)) - 
        (2832.5048482600646*mcMS**2)/(mbkin**2*(1 - mcMS/mbkin)) + 
        (1166928234496119443*mcMS**3)/(1203284664281700*mbkin**7*
            (1 - mcMS/mbkin)) + (4667712937984477772*mcMS**3)/
            (902463498211275*mbkin**6*(1 - mcMS/mbkin)) - 
        (43176344676356419391*mcMS**3)/(3094160565295800*mbkin**5*
            (1 - mcMS/mbkin)) - (232218718664727769157*mcMS**3)/
            (3609853992845100*mbkin**4*(1 - mcMS/mbkin)) + 
        (17269.013493475835*mcMS**3)/(mbkin**3*(1 - mcMS/mbkin)) - 
        (118345817852851093*mcMS**4)/(26739659206260*mbkin**8*
            (1 - mcMS/mbkin)) - (473383271411404372*mcMS**4)/
            (20054744404695*mbkin**7*(1 - mcMS/mbkin)) + 
        (4378795260555490441*mcMS**4)/(68759123673240*mbkin**6*
            (1 - mcMS/mbkin)) + (23550817752717367507*mcMS**4)/
            (80218977618780*mbkin**5*(1 - mcMS/mbkin)) - 
        (78811.35782320914*mcMS**4)/(mbkin**4*(1 - mcMS/mbkin)) + 
        (49345371782260483*mcMS**5)/(3217338674550*mbkin**9*(1 - mcMS/mbkin)) + 
        (394762974258083864*mcMS**5)/(4826008011825*mbkin**8*(1 - mcMS/mbkin)) - 
        (1825778755943637871*mcMS**5)/(8273156591700*mbkin**7*
            (1 - mcMS/mbkin)) - (9819728984669836117*mcMS**5)/
            (9652016023650*mbkin**6*(1 - mcMS/mbkin)) + (273112.3936058774*mcMS**5)/
            (mbkin**5*(1 - mcMS/mbkin)) - (146208599147473679*mcMS**6)/
            (3539072542005*mbkin**10*(1 - mcMS/mbkin)) - 
        (2339337586359578864*mcMS**6)/(10617217626015*mbkin**9*
            (1 - mcMS/mbkin)) + (5409718168456526123*mcMS**6)/
            (9100472250870*mbkin**8*(1 - mcMS/mbkin)) + 
        (29095511230347262121*mcMS**6)/(10617217626015*mbkin**7*
            (1 - mcMS/mbkin)) - (735656.7327332316*mcMS**6)/
            (mbkin**6*(1 - mcMS/mbkin)) + (64819098503293*mcMS**7)/
            (729555255*mbkin**11*(1 - mcMS/mbkin)) + (1037105576052688*mcMS**7)/
            (2188665765*mbkin**10*(1 - mcMS/mbkin)) - (16788146512352887*mcMS**7)/
            (13131994590*mbkin**9*(1 - mcMS/mbkin)) - (12899000602155307*mcMS**7)/
            (2188665765*mbkin**8*(1 - mcMS/mbkin)) + 
        (1.5821095759052273e6*mcMS**7)/(mbkin**7*(1 - mcMS/mbkin)) - 
        (7714866552402197*mcMS**8)/(49566842325*mbkin**12*(1 - mcMS/mbkin)) - 
        (123437864838435152*mcMS**8)/(148700526975*mbkin**11*(1 - mcMS/mbkin)) + 
        (285450062438881289*mcMS**8)/(127457594550*mbkin**10*(1 - mcMS/mbkin)) + 
        (1535258443928037203*mcMS**8)/(148700526975*mbkin**9*(1 - mcMS/mbkin)) - 
        (2.771589631470022e6*mcMS**8)/(mbkin**8*(1 - mcMS/mbkin)) + 
        (2242974842855321*mcMS**9)/(9972027450*mbkin**13*(1 - mcMS/mbkin)) + 
        (17943798742842568*mcMS**9)/(14958041175*mbkin**12*(1 - mcMS/mbkin)) - 
        (82990069185646877*mcMS**9)/(25642356300*mbkin**11*(1 - mcMS/mbkin)) - 
        (446351993728208879*mcMS**9)/(29916082350*mbkin**10*(1 - mcMS/mbkin)) + 
        (4.0052783198585245e6*mcMS**9)/(mbkin**9*(1 - mcMS/mbkin)) - 
        (1713682422600793*mcMS**10)/(6345835650*mbkin**14*(1 - mcMS/mbkin)) - 
        (13709459380806344*mcMS**10)/(9518753475*mbkin**13*(1 - mcMS/mbkin)) + 
        (63406249636229341*mcMS**10)/(16317863100*mbkin**12*(1 - mcMS/mbkin)) + 
        (341022802097557807*mcMS**10)/(19037506950*mbkin**11*(1 - mcMS/mbkin)) - 
        (4.808762016500472e6*mcMS**10)/(mbkin**10*(1 - mcMS/mbkin)) + 
        (13374421001572999*mcMS**11)/(49497518070*mbkin**15*(1 - mcMS/mbkin)) + 
        (106995368012583992*mcMS**11)/(74246277105*mbkin**14*(1 - mcMS/mbkin)) - 
        (494853577058200963*mcMS**11)/(127279332180*mbkin**13*
            (1 - mcMS/mbkin)) - (2661509779313026801*mcMS**11)/
            (148492554210*mbkin**12*(1 - mcMS/mbkin)) + 
        (4.811531457940194e6*mcMS**11)/(mbkin**11*(1 - mcMS/mbkin)) - 
        (11151579707994493*mcMS**12)/(49497518070*mbkin**16*(1 - mcMS/mbkin)) - 
        (89212637663955944*mcMS**12)/(74246277105*mbkin**15*(1 - mcMS/mbkin)) + 
        (412608449195796241*mcMS**12)/(127279332180*mbkin**14*
            (1 - mcMS/mbkin)) + (2219164361890904107*mcMS**12)/
            (148492554210*mbkin**13*(1 - mcMS/mbkin)) - 
        (4.0118504243609793e6*mcMS**12)/(mbkin**12*(1 - mcMS/mbkin)) + 
        (715203600706132*mcMS**13)/(4583103525*mbkin**17*(1 - mcMS/mbkin)) + 
        (11443257611298112*mcMS**13)/(13749310575*mbkin**16*(1 - mcMS/mbkin)) - 
        (13231266613063442*mcMS**13)/(5892561675*mbkin**15*(1 - mcMS/mbkin)) - 
        (142325516540520268*mcMS**13)/(13749310575*mbkin**14*(1 - mcMS/mbkin)) + 
        (2.778828775528237e6*mcMS**13)/(mbkin**13*(1 - mcMS/mbkin)) - 
        (32217277599333127*mcMS**14)/(361129851225*mbkin**18*(1 - mcMS/mbkin)) - 
        (515476441589330032*mcMS**14)/(1083389553675*mbkin**17*
            (1 - mcMS/mbkin)) + (8344274898227279893*mcMS**14)/
            (6500337322050*mbkin**16*(1 - mcMS/mbkin)) + 
        (6411238242267292273*mcMS**14)/(1083389553675*mbkin**15*
            (1 - mcMS/mbkin)) - (1.5886097102490827e6*mcMS**14)/
            (mbkin**14*(1 - mcMS/mbkin)) + (43352528289259463*mcMS**15)/
            (1040903688825*mbkin**19*(1 - mcMS/mbkin)) + 
        (693640452628151408*mcMS**15)/(3122711066475*mbkin**18*
            (1 - mcMS/mbkin)) - (1604043546702600131*mcMS**15)/
            (2676609485550*mbkin**17*(1 - mcMS/mbkin)) - 
        (8627153129562633137*mcMS**15)/(3122711066475*mbkin**16*
            (1 - mcMS/mbkin)) + (741644.2692842743*mcMS**15)/
            (mbkin**15*(1 - mcMS/mbkin)) - (111704868472735*mcMS**16)/
            (7149641499*mbkin**20*(1 - mcMS/mbkin)) - (1787277895563760*mcMS**16)/
            (21448924497*mbkin**19*(1 - mcMS/mbkin)) + (4133080133491195*mcMS**16)/
            (18384792426*mbkin**18*(1 - mcMS/mbkin)) + (22229268826074265*mcMS**16)/
            (21448924497*mbkin**17*(1 - mcMS/mbkin)) - 
        (278214.39593740494*mcMS**16)/(mbkin**16*(1 - mcMS/mbkin)) + 
        (7230231600513707*mcMS**17)/(1572921129780*mbkin**21*(1 - mcMS/mbkin)) + 
        (28920926402054828*mcMS**17)/(1179690847335*mbkin**20*
            (1 - mcMS/mbkin)) - (267518569219007159*mcMS**17)/
            (4044654333720*mbkin**19*(1 - mcMS/mbkin)) - 
        (1438816088502227693*mcMS**17)/(4718763389340*mbkin**18*
            (1 - mcMS/mbkin)) + (81853.45678088568*mcMS**17)/
            (mbkin**17*(1 - mcMS/mbkin)) - (1774149351078739*mcMS**18)/
            (1736341506900*mbkin**22*(1 - mcMS/mbkin)) - 
        (7096597404314956*mcMS**18)/(1302256130175*mbkin**21*(1 - mcMS/mbkin)) + 
        (65643525989913343*mcMS**18)/(4464878160600*mbkin**20*
            (1 - mcMS/mbkin)) + (353055720864669061*mcMS**18)/
            (5209024520700*mbkin**19*(1 - mcMS/mbkin)) - 
        (18194.779307777702*mcMS**18)/(mbkin**18*(1 - mcMS/mbkin)) + 
        (64726178334967321*mcMS**19)/(401094888093900*mbkin**23*
            (1 - mcMS/mbkin)) + (258904713339869284*mcMS**19)/
            (300821166070425*mbkin**22*(1 - mcMS/mbkin)) - 
        (2394868598393790877*mcMS**19)/(1031386855098600*mbkin**21*
            (1 - mcMS/mbkin)) - (12880509488658496879*mcMS**19)/
            (1203284664281700*mbkin**20*(1 - mcMS/mbkin)) + 
        (2873.5886595382344*mcMS**19)/(mbkin**19*(1 - mcMS/mbkin)) - 
        (369023890038773819*mcMS**20)/(22862408621352300*mbkin**24*
            (1 - mcMS/mbkin)) - (1476095560155095276*mcMS**20)/
            (17146806466014225*mbkin**23*(1 - mcMS/mbkin)) + 
        (13653883931434631303*mcMS**20)/(58789050740620200*mbkin**22*
            (1 - mcMS/mbkin)) + (73435754117715989981*mcMS**20)/
            (68587225864056900*mbkin**21*(1 - mcMS/mbkin)) - 
        (287.42481635517186*mcMS**20)/(mbkin**20*(1 - mcMS/mbkin)) + 
        (22826272832761*mcMS**21)/(29691439767990*mbkin**25*(1 - mcMS/mbkin)) + 
        (182610182662088*mcMS**21)/(44537159651985*mbkin**24*(1 - mcMS/mbkin)) - 
        (844572094812157*mcMS**21)/(76349416546260*mbkin**23*(1 - mcMS/mbkin)) - 
        (4542428293719439*mcMS**21)/(89074319303970*mbkin**22*
            (1 - mcMS/mbkin)) + (13.689749741627345*mcMS**21)/
            (mbkin**21*(1 - mcMS/mbkin)) + 179696/(10935*mbkin**3*
            (mbkin**2 - mcMS**2)) + 1437568/(32805*mbkin**2*(mbkin**2 - mcMS**2)) - 
        359392/(10935*mbkin*(mbkin**2 - mcMS**2)) - 
        23744/(2187*mcMS**2*(mbkin**2 - mcMS**2)) - 
        2968/(729*mbkin*mcMS**2*(mbkin**2 - mcMS**2)) + 
        (5936*mbkin)/(729*mcMS**2*(mbkin**2 - mcMS**2)) + 
        (110152*mcMS**2)/(729*mbkin**5*(mbkin**2 - mcMS**2)) + 
        (881216*mcMS**2)/(2187*mbkin**4*(mbkin**2 - mcMS**2)) - 
        (220304*mcMS**2)/(729*mbkin**3*(mbkin**2 - mcMS**2)) - 
        (108064*mcMS**4)/(729*mbkin**7*(mbkin**2 - mcMS**2)) - 
        (864512*mcMS**4)/(2187*mbkin**6*(mbkin**2 - mcMS**2)) + 
        (216128*mcMS**4)/(729*mbkin**5*(mbkin**2 - mcMS**2)) - 
        (163480*mcMS**6)/(2187*mbkin**9*(mbkin**2 - mcMS**2)) - 
        (1307840*mcMS**6)/(6561*mbkin**8*(mbkin**2 - mcMS**2)) + 
        (326960*mcMS**6)/(2187*mbkin**7*(mbkin**2 - mcMS**2)) + 
        (113936*mcMS**8)/(729*mbkin**11*(mbkin**2 - mcMS**2)) + 
        (911488*mcMS**8)/(2187*mbkin**10*(mbkin**2 - mcMS**2)) - 
        (227872*mcMS**8)/(729*mbkin**9*(mbkin**2 - mcMS**2)) - 
        (352712*mcMS**10)/(3645*mbkin**13*(mbkin**2 - mcMS**2)) - 
        (2821696*mcMS**10)/(10935*mbkin**12*(mbkin**2 - mcMS**2)) + 
        (705424*mcMS**10)/(3645*mbkin**11*(mbkin**2 - mcMS**2)) + 
        0.5174790616738993*(-47373355352/5893965 + (485636863424*mcMS)/
            (5893965*mbkin) - (50149010624*mcMS**2)/(127575*mbkin**2) + 
            (3089394228416*mcMS**3)/(2679075*mbkin**3) - (2063288071448*mcMS**4)/
            (893025*mbkin**4) + (990427992704*mcMS**5)/(297675*mbkin**5) - 
            (449810162944*mcMS**6)/(127575*mbkin**6) + (2470467195008*mcMS**7)/
            (893025*mbkin**7) - (474446489624*mcMS**8)/(297675*mbkin**8) + 
            (251471091392*mcMS**9)/(382725*mbkin**9) - (492385098176*mcMS**10)/
            (2679075*mbkin**10) + (307269158848*mcMS**11)/(9823275*mbkin**11) - 
            (14402054984*mcMS**12)/(5893965*mbkin**12)) + 
        (4256*np.pi**2)/(729*mbkin**5) + (21412*np.pi**2)/(2187*mbkin**4) + 
        (881201*np.pi**2)/(7290*mbkin**3) + (182237269436078191*np.pi**2)/
            (870948899861040*mbkin**2) + (201925015555159231*np.pi**2)/
            (391927004937468*mbkin) + (371*np.pi**2)/(243*mbkin*mcMS**2) - 
        (17747732067833644297*mcMS*np.pi**2)/(5225693399166240*mbkin**3) - 
        (294392289942030061*mcMS*np.pi**2)/(40584157316010*mbkin**2) + 
        (575170243214465203079*mcMS*np.pi**2)/(2733522655663800*mbkin) + 
        (128*mcMS**2*np.pi**2)/(9*mbkin**6) + (129589*mcMS**2*np.pi**2)/(3645*mbkin**5) + 
        (53887173095336743*mcMS**2*np.pi**2)/(3033490750290*mbkin**4) + 
        (53938617315161422*mcMS**2*np.pi**2)/(1516745375145*mbkin**3) - 
        (17563432288657679933*mcMS**2*np.pi**2)/(16136497377000*mbkin**2) - 
        (500124566882420850143*mcMS**3*np.pi**2)/(9626277314253600*mbkin**5) - 
        (251828986423715747416*mcMS**3*np.pi**2)/(2707390494633825*mbkin**4) + 
        (8339587769418224437*mcMS**3*np.pi**2)/(2450710647000*mbkin**3) + 
        (1952*mcMS**4*np.pi**2)/(243*mbkin**9) + (3952*mcMS**4*np.pi**2)/(729*mbkin**8) - 
        (488981*mcMS**4*np.pi**2)/(2430*mbkin**7) + 
        (82449430423427437397*mcMS**4*np.pi**2)/(849377410081200*mbkin**6) + 
        (12113368403642977613*mcMS**4*np.pi**2)/(91004722508700*mbkin**5) - 
        (64966687484575317757*mcMS**4*np.pi**2)/(9102639546000*mbkin**4) - 
        (179769700330922348623*mcMS**5*np.pi**2)/(1698754820162400*mbkin**7) - 
        (92442394874840566*mcMS**5*np.pi**2)/(3539072542005*mbkin**6) + 
        (669423083563284145811*mcMS**5*np.pi**2)/(63718476822000*mbkin**5) - 
        (2752*mcMS**6*np.pi**2)/(243*mbkin**11) - (16832*mcMS**6*np.pi**2)/
            (729*mbkin**10) + (97627*mcMS**6*np.pi**2)/(3645*mbkin**9) + 
        (4614431141833*mcMS**6*np.pi**2)/(189143955*mbkin**8) - 
        (137482530828505316*mcMS**6*np.pi**2)/(413657829585*mbkin**7) - 
        (384600921789467137*mcMS**6*np.pi**2)/(35010152100*mbkin**6) + 
        (1355412405576820307*mcMS**7*np.pi**2)/(10295483758560*mbkin**9) + 
        (321144679829513395*mcMS**7*np.pi**2)/(386080640946*mbkin**8) + 
        (507685974189815276761*mcMS**7*np.pi**2)/(63718476822000*mbkin**7) + 
        (1184*mcMS**8*np.pi**2)/(81*mbkin**13) + (9580*mcMS**8*np.pi**2)/(243*mbkin**12) - 
        (1801*mcMS**8*np.pi**2)/(27*mbkin**11) - (959574462101714579*mcMS**8*np.pi**2)/
            (3629817991800*mbkin**10) - (1054870061402601991*mcMS**8*np.pi**2)/
            (907454497950*mbkin**9) - (20849535010571303317*mcMS**8*np.pi**2)/
            (5792588802000*mbkin**8) + (53459135782152251*mcMS**9*np.pi**2)/
            (188561973600*mbkin**11) + (58391575439874692*mcMS**9*np.pi**2)/
            (53033055075*mbkin**10) + (3386492197074766799*mcMS**9*np.pi**2)/
            (6826979659500*mbkin**9) - (4866374988808478*mcMS**10*np.pi**2)/
            (24748759035*mbkin**12) - (54351339774837416*mcMS**10*np.pi**2)/
            (74246277105*mbkin**11) + (55662415966133493199*mcMS**10*np.pi**2)/
            (95577715233000*mbkin**10) + (446072947136541481*mcMS**11*np.pi**2)/
            (5226937908192*mbkin**13) + (107398266737325491*mcMS**11*np.pi**2)/
            (326683619262*mbkin**12) - (574699705987196701*mcMS**11*np.pi**2)/
            (1155335019300*mbkin**11) - (32012415866684791*mcMS**12*np.pi**2)/
            (2177890795080*mbkin**14) - (30344262420794467*mcMS**12*np.pi**2)/
            (376942637610*mbkin**13) + (1775022166803500557*mcMS**12*np.pi**2)/
            (8653126482000*mbkin**12) - (2056765056456933451*mcMS**13*np.pi**2)/
            (242679260023200*mbkin**15) - (170467912134828692*mcMS**13*np.pi**2)/
            (22751180627175*mbkin**14) - (1474778611823953*mcMS**13*np.pi**2)/
            (28897268400*mbkin**13) + (434277136208529359*mcMS**14*np.pi**2)/
            (53086088130075*mbkin**16) + (4497441339299036*mcMS**14*np.pi**2)/
            (272236349385*mbkin**15) + (7010465932780*mcMS**14*np.pi**2)/
            (939161223*mbkin**14) - (697963683819270523*mcMS**15*np.pi**2)/
            (188750535573600*mbkin**17) - (516280173594609343*mcMS**15*np.pi**2)/
            (63703305756090*mbkin**16) - (12724616134177*mcMS**15*np.pi**2)/
            (25044299280*mbkin**15) + (5929043471749*mcMS**16*np.pi**2)/
            (5256210960*mbkin**18) + (5929043471749*mcMS**16*np.pi**2)/
            (2365294932*mbkin**17) - (14993717402449*mcMS**17*np.pi**2)/
            (60021681720*mbkin**19) - (14993717402449*mcMS**17*np.pi**2)/
            (27009756774*mbkin**18) + (238966675564711*mcMS**18*np.pi**2)/
            (6077195274150*mbkin**20) + (477933351129422*mcMS**18*np.pi**2)/
            (5469475746735*mbkin**19) - (22978828361473*mcMS**19*np.pi**2)/
            (5859902248200*mbkin**21) - (22978828361473*mcMS**19*np.pi**2)/
            (2636956011690*mbkin**20) + (4916068298621*mcMS**20*np.pi**2)/
            (26392390904880*mbkin**22) + (4916068298621*mcMS**20*np.pi**2)/
            (11876575907196*mbkin**21) - (344*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/
            (729*mbkin**5) - (2752*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(2187*mbkin**4) - 
        (7009*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(729*mbkin**3) - 
        (728*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(9*mbkin**9) - 
        (5824*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(27*mbkin**8) + 
        (20657*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(27*mbkin**7) - 
        (880*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(27*mbkin**5) - 
        (7040*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(81*mbkin**4) + 
        (7150*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(9*mbkin**3) + 
        (2291073463739113*mcMS*np.pi**2)/(184746736334160*mbkin**3*
            (1 - mcMS/mbkin)) + (2291073463739113*mcMS*np.pi**2)/
            (69280026125310*mbkin**2*(1 - mcMS/mbkin)) + 
        (2291073463739113*mcMS*np.pi**2)/(369493472668320*mbkin*
            (1 - mcMS/mbkin)) - (66120814157661443*mcMS**2*np.pi**2)/
            (554240209002480*mbkin**4*(1 - mcMS/mbkin)) - 
        (66120814157661443*mcMS**2*np.pi**2)/(207840078375930*mbkin**3*
            (1 - mcMS/mbkin)) - (66120814157661443*mcMS**2*np.pi**2)/
            (1108480418004960*mbkin**2*(1 - mcMS/mbkin)) + 
        (1166928234496119443*mcMS**3*np.pi**2)/(1604379552375600*mbkin**5*
            (1 - mcMS/mbkin)) + (1166928234496119443*mcMS**3*np.pi**2)/
            (601642332140850*mbkin**4*(1 - mcMS/mbkin)) + 
        (1166928234496119443*mcMS**3*np.pi**2)/(3208759104751200*mbkin**3*
            (1 - mcMS/mbkin)) - (118345817852851093*mcMS**4*np.pi**2)/
            (35652878941680*mbkin**6*(1 - mcMS/mbkin)) - 
        (118345817852851093*mcMS**4*np.pi**2)/(13369829603130*mbkin**5*
            (1 - mcMS/mbkin)) - (118345817852851093*mcMS**4*np.pi**2)/
            (71305757883360*mbkin**4*(1 - mcMS/mbkin)) + 
        (49345371782260483*mcMS**5*np.pi**2)/(4289784899400*mbkin**7*
            (1 - mcMS/mbkin)) + (49345371782260483*mcMS**5*np.pi**2)/
            (1608669337275*mbkin**6*(1 - mcMS/mbkin)) + 
        (49345371782260483*mcMS**5*np.pi**2)/(8579569798800*mbkin**5*
            (1 - mcMS/mbkin)) - (146208599147473679*mcMS**6*np.pi**2)/
            (4718763389340*mbkin**8*(1 - mcMS/mbkin)) - 
        (292417198294947358*mcMS**6*np.pi**2)/(3539072542005*mbkin**7*
            (1 - mcMS/mbkin)) - (146208599147473679*mcMS**6*np.pi**2)/
            (9437526778680*mbkin**6*(1 - mcMS/mbkin)) + 
        (64819098503293*mcMS**7*np.pi**2)/(972740340*mbkin**9*(1 - mcMS/mbkin)) + 
        (129638197006586*mcMS**7*np.pi**2)/(729555255*mbkin**8*(1 - mcMS/mbkin)) + 
        (64819098503293*mcMS**7*np.pi**2)/(1945480680*mbkin**7*(1 - mcMS/mbkin)) - 
        (7714866552402197*mcMS**8*np.pi**2)/(66089123100*mbkin**10*
            (1 - mcMS/mbkin)) - (15429733104804394*mcMS**8*np.pi**2)/
            (49566842325*mbkin**9*(1 - mcMS/mbkin)) - 
        (7714866552402197*mcMS**8*np.pi**2)/(132178246200*mbkin**8*
            (1 - mcMS/mbkin)) + (2242974842855321*mcMS**9*np.pi**2)/
            (13296036600*mbkin**11*(1 - mcMS/mbkin)) + 
        (2242974842855321*mcMS**9*np.pi**2)/(4986013725*mbkin**10*
            (1 - mcMS/mbkin)) + (2242974842855321*mcMS**9*np.pi**2)/
            (26592073200*mbkin**9*(1 - mcMS/mbkin)) - 
        (1713682422600793*mcMS**10*np.pi**2)/(8461114200*mbkin**12*
            (1 - mcMS/mbkin)) - (1713682422600793*mcMS**10*np.pi**2)/
            (3172917825*mbkin**11*(1 - mcMS/mbkin)) - 
        (1713682422600793*mcMS**10*np.pi**2)/(16922228400*mbkin**10*
            (1 - mcMS/mbkin)) + (13374421001572999*mcMS**11*np.pi**2)/
            (65996690760*mbkin**13*(1 - mcMS/mbkin)) + 
        (13374421001572999*mcMS**11*np.pi**2)/(24748759035*mbkin**12*
            (1 - mcMS/mbkin)) + (13374421001572999*mcMS**11*np.pi**2)/
            (131993381520*mbkin**11*(1 - mcMS/mbkin)) - 
        (11151579707994493*mcMS**12*np.pi**2)/(65996690760*mbkin**14*
            (1 - mcMS/mbkin)) - (11151579707994493*mcMS**12*np.pi**2)/
            (24748759035*mbkin**13*(1 - mcMS/mbkin)) - 
        (11151579707994493*mcMS**12*np.pi**2)/(131993381520*mbkin**12*
            (1 - mcMS/mbkin)) + (178800900176533*mcMS**13*np.pi**2)/
            (1527701175*mbkin**15*(1 - mcMS/mbkin)) + 
        (1430407201412264*mcMS**13*np.pi**2)/(4583103525*mbkin**14*
            (1 - mcMS/mbkin)) + (178800900176533*mcMS**13*np.pi**2)/
            (3055402350*mbkin**13*(1 - mcMS/mbkin)) - 
        (32217277599333127*mcMS**14*np.pi**2)/(481506468300*mbkin**16*
            (1 - mcMS/mbkin)) - (64434555198666254*mcMS**14*np.pi**2)/
            (361129851225*mbkin**15*(1 - mcMS/mbkin)) - 
        (32217277599333127*mcMS**14*np.pi**2)/(963012936600*mbkin**14*
            (1 - mcMS/mbkin)) + (43352528289259463*mcMS**15*np.pi**2)/
            (1387871585100*mbkin**17*(1 - mcMS/mbkin)) + 
        (86705056578518926*mcMS**15*np.pi**2)/(1040903688825*mbkin**16*
            (1 - mcMS/mbkin)) + (43352528289259463*mcMS**15*np.pi**2)/
            (2775743170200*mbkin**15*(1 - mcMS/mbkin)) - 
        (111704868472735*mcMS**16*np.pi**2)/(9532855332*mbkin**18*
            (1 - mcMS/mbkin)) - (223409736945470*mcMS**16*np.pi**2)/
            (7149641499*mbkin**17*(1 - mcMS/mbkin)) - 
        (111704868472735*mcMS**16*np.pi**2)/(19065710664*mbkin**16*
            (1 - mcMS/mbkin)) + (7230231600513707*mcMS**17*np.pi**2)/
            (2097228173040*mbkin**19*(1 - mcMS/mbkin)) + 
        (7230231600513707*mcMS**17*np.pi**2)/(786460564890*mbkin**18*
            (1 - mcMS/mbkin)) + (7230231600513707*mcMS**17*np.pi**2)/
            (4194456346080*mbkin**17*(1 - mcMS/mbkin)) - 
        (1774149351078739*mcMS**18*np.pi**2)/(2315122009200*mbkin**20*
            (1 - mcMS/mbkin)) - (1774149351078739*mcMS**18*np.pi**2)/
            (868170753450*mbkin**19*(1 - mcMS/mbkin)) - 
        (1774149351078739*mcMS**18*np.pi**2)/(4630244018400*mbkin**18*
            (1 - mcMS/mbkin)) + (64726178334967321*mcMS**19*np.pi**2)/
            (534793184125200*mbkin**21*(1 - mcMS/mbkin)) + 
        (64726178334967321*mcMS**19*np.pi**2)/(200547444046950*mbkin**20*
            (1 - mcMS/mbkin)) + (64726178334967321*mcMS**19*np.pi**2)/
            (1069586368250400*mbkin**19*(1 - mcMS/mbkin)) - 
        (369023890038773819*mcMS**20*np.pi**2)/(30483211495136400*mbkin**22*
            (1 - mcMS/mbkin)) - (369023890038773819*mcMS**20*np.pi**2)/
            (11431204310676150*mbkin**21*(1 - mcMS/mbkin)) - 
        (369023890038773819*mcMS**20*np.pi**2)/(60966422990272800*mbkin**20*
            (1 - mcMS/mbkin)) + (22826272832761*mcMS**21*np.pi**2)/
            (39588586357320*mbkin**23*(1 - mcMS/mbkin)) + 
        (22826272832761*mcMS**21*np.pi**2)/(14845719883995*mbkin**22*
            (1 - mcMS/mbkin)) + (22826272832761*mcMS**21*np.pi**2)/
            (79177172714640*mbkin**21*(1 - mcMS/mbkin)) - 
        (1090*np.pi**4)/(243*mbkin**3) + (3*np.pi**4)/(2*mbkin**2) + 
        (10*np.pi**4)/(3*mbkin) + (2194530493961*mcMS*np.pi**4)/(9725042250*mbkin) + 
        (94*mcMS**2*np.pi**4)/(27*mbkin**5) - (8*mcMS**2*np.pi**4)/mbkin**4 - 
        (16*mcMS**2*np.pi**4)/mbkin**3 - (126068618713*mcMS**2*np.pi**4)/
            (126299250*mbkin**2) + (493975343077*mcMS**3*np.pi**4)/
            (176818950*mbkin**3) + (172*mcMS**4*np.pi**4)/(81*mbkin**7) + 
        (6*mcMS**4*np.pi**4)/mbkin**6 + (16*mcMS**4*np.pi**4)/mbkin**5 - 
        (9190679983*mcMS**4*np.pi**4)/(1683990*mbkin**4) + 
        (228831767977*mcMS**5*np.pi**4)/(29469825*mbkin**5) - 
        (76*mcMS**6*np.pi**4)/(81*mbkin**9) - (16*mcMS**6*np.pi**4)/(3*mbkin**7) - 
        (344686887737*mcMS**6*np.pi**4)/(42099750*mbkin**6) + 
        (944638716947*mcMS**7*np.pi**4)/(147349125*mbkin**7) + 
        (118*mcMS**8*np.pi**4)/(81*mbkin**11) + (mcMS**8*np.pi**4)/(2*mbkin**10) + 
        (2*mcMS**8*np.pi**4)/mbkin**9 - (19750002199*mcMS**8*np.pi**4)/
            (5358150*mbkin**8) + (2550828803*mcMS**9*np.pi**4)/(1683990*mbkin**9) - 
        (37330259221*mcMS**10*np.pi**4)/(88409475*mbkin**10) + 
        (99615610673*mcMS**11*np.pi**4)/(1389291750*mbkin**11) - 
        (108772480427*mcMS**12*np.pi**4)/(19450084500*mbkin**12) + 
        (86*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**4)/(243*mbkin**3) - 
        (182*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**4)/(9*mbkin**7) - 
        (220*(mcMS**2/mbkin**2)**(3/2)*np.pi**4)/(9*mbkin**3) + 
        (23931088/2525985 - (14708576*mcMS)/(2525985*mbkin) - 
            (8950336*mcMS**2)/(32805*mbkin**2) + (33425824*mcMS**3)/
            (25515*mbkin**3) - (233018864*mcMS**4)/(76545*mbkin**4) + 
            (67270976*mcMS**5)/(15309*mbkin**5) - (47392768*mcMS**6)/
            (10935*mbkin**6) + (235510208*mcMS**7)/(76545*mbkin**7) - 
            (123588112*mcMS**8)/(76545*mbkin**8) + (20426656*mcMS**9)/
            (32805*mbkin**9) - (38385728*mcMS**10)/(229635*mbkin**10) + 
            (69725024*mcMS**11)/(2525985*mbkin**11) - (1775344*mcMS**12)/
            (841995*mbkin**12))*np.log(2)**3 + (-508/(243*mbkin**3) + 
            (1694*mcMS**2)/(243*mbkin**5) + (1813*mcMS**2)/(27*mbkin**2) - 
            (11822*mcMS**4)/(729*mbkin**7) - (1660*mcMS**4)/(9*mbkin**6) - 
            (6640*mcMS**4)/(27*mbkin**5) - (4868*mcMS**4)/(3*mbkin**4) + 
            (17780*mcMS**6)/(243*mbkin**9) - (5917*mcMS**6)/(9*mbkin**6) + 
            (12596*mcMS**8)/(243*mbkin**11) + (8833*mcMS**8)/(54*mbkin**8))*
            np.log(mcMS**2/mbkin**2)**3 + (3577*mcMS**4*np.log(mcMS**2/mbkin**2)**4)/
            (9*mbkin**4) + np.log(2)**4*(-5921669419/17681895 + (60704607928*mcMS)/
            (17681895*mbkin) - (6268626328*mcMS**2)/(382725*mbkin**2) + 
            (386174278552*mcMS**3)/(8037225*mbkin**3) - (257911008931*mcMS**4)/
            (2679075*mbkin**4) + (123803499088*mcMS**5)/(893025*mbkin**5) - 
            (56226270368*mcMS**6)/(382725*mbkin**6) + (308808399376*mcMS**7)/
            (2679075*mbkin**7) - (59305811203*mcMS**8)/(893025*mbkin**8) + 
            (31433886424*mcMS**9)/(1148175*mbkin**9) - (61548137272*mcMS**10)/
            (8037225*mbkin**10) + (38408644856*mcMS**11)/(29469825*mbkin**11) - 
            (1800256873*mcMS**12)/(17681895*mbkin**12) + 
            (392*mcMS**4*np.log(mcMS**2/mbkin**2))/(27*mbkin**4)) + 
        (860.236869119249 - 608/(243*mbkin**5) + 2442426148/
            (90654795*mbkin**4) + 4943533216/(30218265*mbkin**3) + 
            42887214162286888/(53086088130075*mbkin**2) + 143017147230146764/
            (95554958634135*mbkin) - (56481088*mcMS)/(340119*mbkin**5) - 
            (1807394816*mcMS)/(2380833*mbkin**4) - (310449803443917791*mcMS)/
            (53086088130075*mbkin**3) - (9521780371527925408*mcMS)/
            (796291321951125*mbkin**2) - (10767.745948118347*mcMS)/mbkin - 
            (320*mcMS**2)/(27*mbkin**7) + (276688*mcMS**2)/(3159*mbkin**6) + 
            (3236864*mcMS**2)/(9477*mbkin**5) + (236347688805556*mcMS**2)/
            (10672286625*mbkin**4) + (80933419563736*mcMS**2)/
            (1524612375*mbkin**3) + (32579.81524008369*mcMS**2)/mbkin**2 + 
            (1504064*mcMS**3)/(663*mbkin**7) + (24065024*mcMS**3)/(1989*mbkin**6) - 
            (67140833722675667*mcMS**3)/(1074617168625*mbkin**5) - 
            (1834962771873232112*mcMS**3)/(9671554517625*mbkin**4) - 
            (62419.16538300422*mcMS**3)/mbkin**3 + (928*mcMS**4)/(81*mbkin**9) - 
            (1720111004*mcMS**4)/(110565*mbkin**8) - (4597297504*mcMS**4)/
            (57915*mbkin**7) + (5451292868690296*mcMS**4)/(37927664775*mbkin**6) + 
            (67431499870061452*mcMS**4)/(113782994325*mbkin**5) + 
            (102968.6732429856*mcMS**4)/mbkin**4 + (64971008*mcMS**5)/
            (1053*mbkin**9) + (978386944*mcMS**5)/(3159*mbkin**8) - 
            (43027179107*mcMS**5)/(165375*mbkin**7) - (28651812118912*mcMS**5)/
            (18243225*mbkin**6) - (175330.469026736*mcMS**5)/mbkin**5 + 
            (8800*mcMS**6)/(243*mbkin**11) - (1654117696*mcMS**6)/(9477*mbkin**10) - 
            (2750621632*mcMS**6)/(3159*mbkin**9) + (6251242037792*mcMS**6)/
            (18243225*mbkin**8) + (25486894372646296*mcMS**6)/
            (7388506125*mbkin**7) + (274814.73178371985*mcMS**6)/mbkin**6 + 
            (103304960*mcMS**7)/(273*mbkin**11) + (170987520*mcMS**7)/
            (91*mbkin**10) - (1528624525107313*mcMS**7)/(5746615875*mbkin**9) - 
            (107969763387927568*mcMS**7)/(17239847625*mbkin**8) - 
            (344959.21727236494*mcMS**7)/mbkin**7 - (64*mcMS**8)/(9*mbkin**13) - 
            (1737546056*mcMS**8)/(2673*mbkin**12) - (25814102848*mcMS**8)/
            (8019*mbkin**11) - (38836619987162*mcMS**8)/(638512875*mbkin**10) + 
            (54207343839673328*mcMS**8)/(5746615875*mbkin**9) + 
            (335759.41387939523*mcMS**8)/mbkin**8 + (1096984192*mcMS**9)/
            (1215*mbkin**13) + (16267472896*mcMS**9)/(3645*mbkin**12) + 
            (13635195701351*mcMS**9)/(23455575*mbkin**11) - 
            (17467877286240224*mcMS**9)/(1477701225*mbkin**10) - 
            (256666.41833722152*mcMS**9)/mbkin**9 - (415105504*mcMS**10)/
            (405*mbkin**14) - (2049031424*mcMS**10)/(405*mbkin**13) - 
            (6185635194938764*mcMS**10)/(5746615875*mbkin**12) + 
            (3272208865215496*mcMS**10)/(265228425*mbkin**11) + 
            (155910.77884184313*mcMS**10)/mbkin**10 + (17876864384*mcMS**11)/
            (18711*mbkin**15) + (37777524736*mcMS**11)/(8019*mbkin**14) + 
            (16492777551145283*mcMS**11)/(12642554925*mbkin**13) - 
            (677191509617510224*mcMS**11)/(63212774625*mbkin**12) - 
            (75364.08237712669*mcMS**11)/mbkin**11 - (6582040*mcMS**12)/
            (9*mbkin**16) - (97280320*mcMS**12)/(27*mbkin**15) - 
            (74839125943357106*mcMS**12)/(63212774625*mbkin**14) + 
            (626890258336895296*mcMS**12)/(81273567375*mbkin**13) + 
            (29205.97924280903*mcMS**12)/mbkin**12 + (37077248*mcMS**13)/
            (81*mbkin**17) + (182534144*mcMS**13)/(81*mbkin**16) + 
            (2435254597480993*mcMS**13)/(2910623625*mbkin**15) - 
            (439487617846053184*mcMS**13)/(96050579625*mbkin**14) - 
            (23348306427538*mcMS**13)/(2462835375*mbkin**13) - 
            (8556501952*mcMS**14)/(36855*mbkin**18) - (126298789376*mcMS**14)/
            (110565*mbkin**17) - (25479574691032376*mcMS**14)/
            (54784404675*mbkin**16) + (363735726365229992*mcMS**14)/
            (164353214025*mbkin**15) + (3175998276118*mcMS**14)/
            (1118049075*mbkin**14) + (164985088*mcMS**15)/(1755*mbkin**19) + 
            (187423059968*mcMS**15)/(405405*mbkin**18) + 
            (55426302030929857*mcMS**15)/(273922023375*mbkin**17) - 
            (254139874445295152*mcMS**15)/(295835785245*mbkin**16) - 
            (139299712876378*mcMS**15)/(164353214025*mbkin**15) - 
            (3473716*mcMS**16)/(117*mbkin**20) - (5691872*mcMS**16)/(39*mbkin**19) - 
            (71294882*mcMS**16)/(1053*mbkin**18) + (91697732*mcMS**16)/
            (351*mbkin**17) + (73241*mcMS**16)/(312*mbkin**16) + 
            (378641600*mcMS**17)/(53703*mbkin**21) + (5581772800*mcMS**17)/
            (161109*mbkin**20) + (2716434400*mcMS**17)/(161109*mbkin**19) - 
            (3220580800*mcMS**17)/(53703*mbkin**18) - (930650*mcMS**17)/
            (17901*mbkin**17) - (445357264*mcMS**18)/(375921*mbkin**22) - 
            (937594240*mcMS**18)/(161109*mbkin**21) - (9973658728*mcMS**18)/
            (3383289*mbkin**20) + (55107101456*mcMS**18)/(5638815*mbkin**19) + 
            (1464991*mcMS**18)/(179010*mbkin**18) + (138219712*mcMS**19)/
            (1100385*mbkin**23) + (678781952*mcMS**19)/(1100385*mbkin**22) + 
            (3201633824*mcMS**19)/(9903465*mbkin**21) - (23264704*mcMS**19)/
            (23085*mbkin**20) - (299362*mcMS**19)/(366795*mbkin**19) - 
            (32249372*mcMS**20)/(5101785*mbkin**24) - (475000096*mcMS**20)/
            (15305355*mbkin**23) - (1054886*mcMS**20)/(62985*mbkin**22) + 
            (9343276*mcMS**20)/(188955*mbkin**21) + (527443*mcMS**20)/
            (13604760*mbkin**20) - (208*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**5) - 
            (1664*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**4) + 
            (416*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**3) - 
            (11824*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**9) - 
            (94592*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**8) + 
            (23648*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**7) - 
            (140128*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**5) - 
            (1121024*(mcMS**2/mbkin**2)**(3/2))/(2187*mbkin**4) + 
            (280256*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**3) + 
            (376528*mcMS**2)/(3159*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (6024448*mcMS**2)/(9477*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (10542784*mcMS**2)/(28431*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (12048896*mcMS**2)/(9477*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (1506112*mcMS**2)/(3159*mbkin**2*(1 - mcMS/mbkin)**2) - 
            (429497312*mcMS**3)/(268515*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (6871956992*mcMS**3)/(805545*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (12025924736*mcMS**3)/(2416635*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (13743913984*mcMS**3)/(805545*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (1717989248*mcMS**3)/(268515*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (234545329016*mcMS**4)/(20675655*mbkin**8*(1 - mcMS/mbkin)**2) + 
            (3752725264256*mcMS**4)/(62026965*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (938181316064*mcMS**4)/(26582985*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (7505450528512*mcMS**4)/(62026965*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (938181316064*mcMS**4)/(20675655*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (75343294960*mcMS**5)/(1378377*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (1205492719360*mcMS**5)/(4135131*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (301373179840*mcMS**5)/(1772199*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (2410985438720*mcMS**5)/(4135131*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (301373179840*mcMS**5)/(1378377*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (79048041848*mcMS**6)/(405405*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (1264768669568*mcMS**6)/(1216215*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (316192167392*mcMS**6)/(521235*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (2529537339136*mcMS**6)/(1216215*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (316192167392*mcMS**6)/(405405*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (566619520*mcMS**7)/(1053*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (9065912320*mcMS**7)/(3159*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (15865346560*mcMS**7)/(9477*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (18131824640*mcMS**7)/(3159*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (2266478080*mcMS**7)/(1053*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (205356871136*mcMS**8)/(173745*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (3285709938176*mcMS**8)/(521235*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (5749992391808*mcMS**8)/(1563705*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (6571419876352*mcMS**8)/(521235*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (821427484544*mcMS**8)/(173745*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (366047261248*mcMS**9)/(173745*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (5856756179968*mcMS**9)/(521235*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (10249323314944*mcMS**9)/(1563705*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (11713512359936*mcMS**9)/(521235*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (1464189044992*mcMS**9)/(173745*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (41256072896*mcMS**10)/(13365*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (660097166336*mcMS**10)/(40095*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (1155170041088*mcMS**10)/(120285*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (1320194332672*mcMS**10)/(40095*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (165024291584*mcMS**10)/(13365*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (31867463104*mcMS**11)/(8505*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (509879409664*mcMS**11)/(25515*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (127469852416*mcMS**11)/(10935*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (1019758819328*mcMS**11)/(25515*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (127469852416*mcMS**11)/(8505*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (6433846768*mcMS**12)/(1701*mbkin**16*(1 - mcMS/mbkin)**2) + 
            (102941548288*mcMS**12)/(5103*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (25735387072*mcMS**12)/(2187*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (205883096576*mcMS**12)/(5103*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (25735387072*mcMS**12)/(1701*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (27020210848*mcMS**13)/(8505*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (432323373568*mcMS**13)/(25515*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (108080843392*mcMS**13)/(10935*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (864646747136*mcMS**13)/(25515*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (108080843392*mcMS**13)/(8505*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (896774288*mcMS**14)/(405*mbkin**18*(1 - mcMS/mbkin)**2) + 
            (14348388608*mcMS**14)/(1215*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (25109680064*mcMS**14)/(3645*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (28696777216*mcMS**14)/(1215*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (3587097152*mcMS**14)/(405*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (73703643776*mcMS**15)/(57915*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (1179258300416*mcMS**15)/(173745*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (2063702025728*mcMS**15)/(521235*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (2358516600832*mcMS**15)/(173745*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (294814575104*mcMS**15)/(57915*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (34568184736*mcMS**16)/(57915*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (553090955776*mcMS**16)/(173745*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (967909172608*mcMS**16)/(521235*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (1106181911552*mcMS**16)/(173745*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (138272738944*mcMS**16)/(57915*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (7812195136*mcMS**17)/(34749*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (124995122176*mcMS**17)/(104247*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (218741463808*mcMS**17)/(312741*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (249990244352*mcMS**17)/(104247*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (31248780544*mcMS**17)/(34749*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (7339439504*mcMS**18)/(110565*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (117431032064*mcMS**18)/(331695*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (29357758016*mcMS**18)/(142155*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (234862064128*mcMS**18)/(331695*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (29357758016*mcMS**18)/(110565*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (18003292256*mcMS**19)/(1216215*mbkin**23*(1 - mcMS/mbkin)**2) - 
            (288052676096*mcMS**19)/(3648645*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (72013169024*mcMS**19)/(1563705*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (576105352192*mcMS**19)/(3648645*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (72013169024*mcMS**19)/(1216215*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (9695082136*mcMS**20)/(4135131*mbkin**24*(1 - mcMS/mbkin)**2) + 
            (155121314176*mcMS**20)/(12405393*mbkin**23*(1 - mcMS/mbkin)**2) + 
            (38780328544*mcMS**20)/(5316597*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (310242628352*mcMS**20)/(12405393*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (38780328544*mcMS**20)/(4135131*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (694454128*mcMS**21)/(2953665*mbkin**25*(1 - mcMS/mbkin)**2) - 
            (11111266048*mcMS**21)/(8860995*mbkin**24*(1 - mcMS/mbkin)**2) - 
            (19444715584*mcMS**21)/(26582985*mbkin**23*(1 - mcMS/mbkin)**2) + 
            (22222532096*mcMS**21)/(8860995*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (2777816512*mcMS**21)/(2953665*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (602792*mcMS**22)/(53703*mbkin**26*(1 - mcMS/mbkin)**2) + 
            (9644672*mcMS**22)/(161109*mbkin**25*(1 - mcMS/mbkin)**2) + 
            (16878176*mcMS**22)/(483327*mbkin**24*(1 - mcMS/mbkin)**2) - 
            (19289344*mcMS**22)/(161109*mbkin**23*(1 - mcMS/mbkin)**2) + 
            (2411168*mcMS**22)/(53703*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (56481088*mcMS)/(2380833*mbkin**5*(1 - mcMS/mbkin)) - 
            (903697408*mcMS)/(7142499*mbkin**4*(1 - mcMS/mbkin)) + 
            (1044900128*mcMS)/(3061071*mbkin**3*(1 - mcMS/mbkin)) + 
            (11239736512*mcMS)/(7142499*mbkin**2*(1 - mcMS/mbkin)) - 
            (422.44082590079324*mcMS)/(mbkin*(1 - mcMS/mbkin)) + 
            (1872102880*mcMS**2)/(7142499*mbkin**6*(1 - mcMS/mbkin)) + 
            (29953646080*mcMS**2)/(21427497*mbkin**5*(1 - mcMS/mbkin)) - 
            (34633903280*mcMS**2)/(9183213*mbkin**4*(1 - mcMS/mbkin)) - 
            (372548473120*mcMS**2)/(21427497*mbkin**3*(1 - mcMS/mbkin)) + 
            (4667.359847355476*mcMS**2)/(mbkin**2*(1 - mcMS/mbkin)) - 
            (429497312*mcMS**3)/(268515*mbkin**7*(1 - mcMS/mbkin)) - 
            (6871956992*mcMS**3)/(805545*mbkin**6*(1 - mcMS/mbkin)) + 
            (55619901904*mcMS**3)/(2416635*mbkin**5*(1 - mcMS/mbkin)) + 
            (85469965088*mcMS**3)/(805545*mbkin**4*(1 - mcMS/mbkin)) - 
            (28482.86432213551*mcMS**3)/(mbkin**3*(1 - mcMS/mbkin)) + 
            (48446596112*mcMS**4)/(6891885*mbkin**8*(1 - mcMS/mbkin)) + 
            (775145537792*mcMS**4)/(20675655*mbkin**7*(1 - mcMS/mbkin)) - 
            (896262028072*mcMS**4)/(8860995*mbkin**6*(1 - mcMS/mbkin)) - 
            (9640872626288*mcMS**4)/(20675655*mbkin**5*(1 - mcMS/mbkin)) + 
            (125174.82968621714*mcMS**4)/(mbkin**4*(1 - mcMS/mbkin)) - 
            (3218332144*mcMS**5)/(135135*mbkin**9*(1 - mcMS/mbkin)) - 
            (51493314304*mcMS**5)/(405405*mbkin**8*(1 - mcMS/mbkin)) + 
            (59539144664*mcMS**5)/(173745*mbkin**7*(1 - mcMS/mbkin)) + 
            (640448096656*mcMS**5)/(405405*mbkin**6*(1 - mcMS/mbkin)) - 
            (424086.8239833621*mcMS**5)/(mbkin**5*(1 - mcMS/mbkin)) + 
            (67081088*mcMS**6)/(1053*mbkin**10*(1 - mcMS/mbkin)) + 
            (1073297408*mcMS**6)/(3159*mbkin**9*(1 - mcMS/mbkin)) - 
            (8687000896*mcMS**6)/(9477*mbkin**8*(1 - mcMS/mbkin)) - 
            (13349136512*mcMS**6)/(3159*mbkin**7*(1 - mcMS/mbkin)) + 
            (1.1343928728981994e6*mcMS**6)/(mbkin**6*(1 - mcMS/mbkin)) - 
            (48050816*mcMS**7)/(351*mbkin**11*(1 - mcMS/mbkin)) - 
            (768813056*mcMS**7)/(1053*mbkin**10*(1 - mcMS/mbkin)) + 
            (6222580672*mcMS**7)/(3159*mbkin**9*(1 - mcMS/mbkin)) + 
            (9562112384*mcMS**7)/(1053*mbkin**8*(1 - mcMS/mbkin)) - 
            (2.4377289411589196e6*mcMS**7)/(mbkin**7*(1 - mcMS/mbkin)) + 
            (41683996096*mcMS**8)/(173745*mbkin**12*(1 - mcMS/mbkin)) + 
            (666943937536*mcMS**8)/(521235*mbkin**11*(1 - mcMS/mbkin)) - 
            (5398077494432*mcMS**8)/(1563705*mbkin**10*(1 - mcMS/mbkin)) - 
            (8295115223104*mcMS**8)/(521235*mbkin**9*(1 - mcMS/mbkin)) + 
            (4.272172633742393e6*mcMS**8)/(mbkin**8*(1 - mcMS/mbkin)) - 
            (926904896*mcMS**9)/(2673*mbkin**13*(1 - mcMS/mbkin)) - 
            (14830478336*mcMS**9)/(8019*mbkin**12*(1 - mcMS/mbkin)) + 
            (120034184032*mcMS**9)/(24057*mbkin**11*(1 - mcMS/mbkin)) + 
            (184454074304*mcMS**9)/(8019*mbkin**10*(1 - mcMS/mbkin)) - 
            (6.174872291693431e6*mcMS**9)/(mbkin**9*(1 - mcMS/mbkin)) + 
            (168587456*mcMS**10)/(405*mbkin**14*(1 - mcMS/mbkin)) + 
            (2697399296*mcMS**10)/(1215*mbkin**13*(1 - mcMS/mbkin)) - 
            (21832075552*mcMS**10)/(3645*mbkin**12*(1 - mcMS/mbkin)) - 
            (33548903744*mcMS**10)/(1215*mbkin**11*(1 - mcMS/mbkin)) + 
            (7.412453748823227e6*mcMS**10)/(mbkin**10*(1 - mcMS/mbkin)) - 
            (708244672*mcMS**11)/(1701*mbkin**15*(1 - mcMS/mbkin)) - 
            (11331914752*mcMS**11)/(5103*mbkin**14*(1 - mcMS/mbkin)) + 
            (13102526432*mcMS**11)/(2187*mbkin**13*(1 - mcMS/mbkin)) + 
            (140940689728*mcMS**11)/(5103*mbkin**12*(1 - mcMS/mbkin)) - 
            (7.414310421273456e6*mcMS**11)/(mbkin**11*(1 - mcMS/mbkin)) + 
            (590317408*mcMS**12)/(1701*mbkin**16*(1 - mcMS/mbkin)) + 
            (9445078528*mcMS**12)/(5103*mbkin**15*(1 - mcMS/mbkin)) - 
            (10920872048*mcMS**12)/(2187*mbkin**14*(1 - mcMS/mbkin)) - 
            (117473164192*mcMS**12)/(5103*mbkin**13*(1 - mcMS/mbkin)) + 
            (6.17978035420157e6*mcMS**12)/(mbkin**12*(1 - mcMS/mbkin)) - 
            (97319648*mcMS**13)/(405*mbkin**17*(1 - mcMS/mbkin)) - 
            (1557114368*mcMS**13)/(1215*mbkin**16*(1 - mcMS/mbkin)) + 
            (12602894416*mcMS**13)/(3645*mbkin**15*(1 - mcMS/mbkin)) + 
            (19366609952*mcMS**13)/(1215*mbkin**14*(1 - mcMS/mbkin)) - 
            (4.278950562322721e6*mcMS**13)/(mbkin**13*(1 - mcMS/mbkin)) + 
            (241010816*mcMS**14)/(1755*mbkin**18*(1 - mcMS/mbkin)) + 
            (3856173056*mcMS**14)/(5265*mbkin**17*(1 - mcMS/mbkin)) - 
            (31210900672*mcMS**14)/(15795*mbkin**16*(1 - mcMS/mbkin)) - 
            (47961152384*mcMS**14)/(5265*mbkin**15*(1 - mcMS/mbkin)) + 
            (2.445407134378434e6*mcMS**14)/(mbkin**14*(1 - mcMS/mbkin)) - 
            (25983502208*mcMS**15)/(405405*mbkin**19*(1 - mcMS/mbkin)) - 
            (415736035328*mcMS**15)/(1216215*mbkin**18*(1 - mcMS/mbkin)) + 
            (480694790848*mcMS**15)/(521235*mbkin**17*(1 - mcMS/mbkin)) + 
            (5170716939392*mcMS**15)/(1216215*mbkin**16*(1 - mcMS/mbkin)) - 
            (1.1413013981894134e6*mcMS**15)/(mbkin**15*(1 - mcMS/mbkin)) + 
            (216546752*mcMS**16)/(9009*mbkin**20*(1 - mcMS/mbkin)) + 
            (3464748032*mcMS**16)/(27027*mbkin**19*(1 - mcMS/mbkin)) - 
            (4006114912*mcMS**16)/(11583*mbkin**18*(1 - mcMS/mbkin)) - 
            (43092803648*mcMS**16)/(27027*mbkin**17*(1 - mcMS/mbkin)) + 
            (428022.7468324014*mcMS**16)/(mbkin**16*(1 - mcMS/mbkin)) - 
            (22334464*mcMS**17)/(3159*mbkin**21*(1 - mcMS/mbkin)) - 
            (357351424*mcMS**17)/(9477*mbkin**20*(1 - mcMS/mbkin)) + 
            (2892313088*mcMS**17)/(28431*mbkin**19*(1 - mcMS/mbkin)) + 
            (4444558336*mcMS**17)/(9477*mbkin**18*(1 - mcMS/mbkin)) - 
            (125897.6637429286*mcMS**17)/(mbkin**17*(1 - mcMS/mbkin)) + 
            (2953285408*mcMS**18)/(1879605*mbkin**22*(1 - mcMS/mbkin)) + 
            (47252566528*mcMS**18)/(5638815*mbkin**21*(1 - mcMS/mbkin)) - 
            (54635780048*mcMS**18)/(2416635*mbkin**20*(1 - mcMS/mbkin)) - 
            (587703796192*mcMS**18)/(5638815*mbkin**19*(1 - mcMS/mbkin)) + 
            (27978.897066866393*mcMS**18)/(mbkin**18*(1 - mcMS/mbkin)) - 
            (1709883424*mcMS**19)/(6891885*mbkin**23*(1 - mcMS/mbkin)) - 
            (27358134784*mcMS**19)/(20675655*mbkin**22*(1 - mcMS/mbkin)) + 
            (31632843344*mcMS**19)/(8860995*mbkin**21*(1 - mcMS/mbkin)) + 
            (340266801376*mcMS**19)/(20675655*mbkin**20*(1 - mcMS/mbkin)) - 
            (4417.944366776069*mcMS**19)/(mbkin**19*(1 - mcMS/mbkin)) + 
            (1392395248*mcMS**20)/(56119635*mbkin**24*(1 - mcMS/mbkin)) + 
            (22278323968*mcMS**20)/(168358905*mbkin**23*(1 - mcMS/mbkin)) - 
            (180315184616*mcMS**20)/(505076715*mbkin**22*(1 - mcMS/mbkin)) - 
            (277086654352*mcMS**20)/(168358905*mbkin**21*(1 - mcMS/mbkin)) + 
            (441.81398353634376*mcMS**20)/(mbkin**20*(1 - mcMS/mbkin)) - 
            (1205584*mcMS**21)/(1020357*mbkin**25*(1 - mcMS/mbkin)) - 
            (19289344*mcMS**21)/(3061071*mbkin**24*(1 - mcMS/mbkin)) + 
            (156123128*mcMS**21)/(9183213*mbkin**23*(1 - mcMS/mbkin)) + 
            (239911216*mcMS**21)/(3061071*mbkin**22*(1 - mcMS/mbkin)) - 
            (21.03958115779376*mcMS**21)/(mbkin**21*(1 - mcMS/mbkin)) + 
            (1376*mcMS)/(729*mbkin**6*(-1 + mcMS/mbkin)) + 
            (11008*mcMS)/(2187*mbkin**5*(-1 + mcMS/mbkin)) - 
            (2752*mcMS)/(729*mbkin**4*(-1 + mcMS/mbkin)) - 
            (1376*mcMS**2)/(729*mbkin**7*(-1 + mcMS/mbkin)) - 
            (11008*mcMS**2)/(2187*mbkin**6*(-1 + mcMS/mbkin)) + 
            (2752*mcMS**2)/(729*mbkin**5*(-1 + mcMS/mbkin)) - 
            (3520*mcMS**3)/(27*mbkin**8*(-1 + mcMS/mbkin)) - 
            (28160*mcMS**3)/(81*mbkin**7*(-1 + mcMS/mbkin)) + 
            (7040*mcMS**3)/(27*mbkin**6*(-1 + mcMS/mbkin)) + 
            (3520*mcMS**4)/(27*mbkin**9*(-1 + mcMS/mbkin)) + 
            (28160*mcMS**4)/(81*mbkin**8*(-1 + mcMS/mbkin)) - 
            (7040*mcMS**4)/(27*mbkin**7*(-1 + mcMS/mbkin)) - 
            (2912*mcMS**5)/(27*mbkin**10*(-1 + mcMS/mbkin)) - 
            (23296*mcMS**5)/(81*mbkin**9*(-1 + mcMS/mbkin)) + 
            (5824*mcMS**5)/(27*mbkin**8*(-1 + mcMS/mbkin)) + 
            (2912*mcMS**6)/(27*mbkin**11*(-1 + mcMS/mbkin)) + 
            (23296*mcMS**6)/(81*mbkin**10*(-1 + mcMS/mbkin)) - 
            (5824*mcMS**6)/(27*mbkin**9*(-1 + mcMS/mbkin)) - 
            (27173090*np.pi**2)/(1247103*mbkin**2) - (543461800*np.pi**2)/
            (11223927*mbkin) + (1515611872*mcMS*np.pi**2)/(8729721*mbkin**3) + 
            (3316144256*mcMS*np.pi**2)/(8729721*mbkin**2) + (134953869772*mcMS*np.pi**2)/
            (216729513*mbkin) - (8613416*mcMS**2*np.pi**2)/(11583*mbkin**4) - 
            (55822304*mcMS**2*np.pi**2)/(34749*mbkin**3) - (88793273432*mcMS**2*np.pi**2)/
            (44778825*mbkin**2) + (806281376*mcMS**3*np.pi**2)/(328185*mbkin**5) + 
            (15580744064*mcMS**3*np.pi**2)/(2953665*mbkin**4) + 
            (23695472788*mcMS**3*np.pi**2)/(5893965*mbkin**3) - (28515566*mcMS**4*np.pi**2)/
            (4095*mbkin**6) - (6062039944*mcMS**4*np.pi**2)/(405405*mbkin**5) - 
            (578116793966*mcMS**4*np.pi**2)/(88409475*mbkin**4) + 
            (88677568*mcMS**5*np.pi**2)/(5265*mbkin**7) + (38218240*mcMS**5*np.pi**2)/
            (1053*mbkin**6) + (869262045352*mcMS**5*np.pi**2)/(88409475*mbkin**5) - 
            (3997664*mcMS**6*np.pi**2)/(117*mbkin**8) - (233821184*mcMS**6*np.pi**2)/
            (3159*mbkin**7) - (24057438208*mcMS**6*np.pi**2)/(1804275*mbkin**6) + 
            (47435456*mcMS**7*np.pi**2)/(819*mbkin**9) + (310125056*mcMS**7*np.pi**2)/
            (2457*mbkin**8) + (5411795704*mcMS**7*np.pi**2)/(360855*mbkin**7) - 
            (854120324*mcMS**8*np.pi**2)/(10395*mbkin**10) - (3369673744*mcMS**8*np.pi**2)/
            (18711*mbkin**9) - (442248574*mcMS**8*np.pi**2)/(32805*mbkin**8) + 
            (13180544*mcMS**9*np.pi**2)/(135*mbkin**11) + (261243136*mcMS**9*np.pi**2)/
            (1215*mbkin**10) + (366145128484*mcMS**9*np.pi**2)/(37889775*mbkin**9) - 
            (13105904*mcMS**10*np.pi**2)/(135*mbkin**12) - (17379776*mcMS**10*np.pi**2)/
            (81*mbkin**11) - (1459481451368*mcMS**10*np.pi**2)/(265228425*mbkin**10) + 
            (71793536*mcMS**11*np.pi**2)/(891*mbkin**13) + (3340408576*mcMS**11*np.pi**2)/
            (18711*mbkin**12) + (7099875794876*mcMS**11*np.pi**2)/
            (2917512675*mbkin**11) - (16507364*mcMS**12*np.pi**2)/(297*mbkin**14) - 
            (329672912*mcMS**12*np.pi**2)/(2673*mbkin**13) - 
            (779524199374*mcMS**12*np.pi**2)/(972504225*mbkin**12) + 
            (13573696*mcMS**13*np.pi**2)/(429*mbkin**15) + (4069735936*mcMS**13*np.pi**2)/
            (57915*mbkin**14) + (91264*mcMS**13*np.pi**2)/(495*mbkin**13) - 
            (1192479328*mcMS**14*np.pi**2)/(81081*mbkin**16) - 
            (13247396608*mcMS**14*np.pi**2)/(405405*mbkin**15) - 
            (1525952*mcMS**14*np.pi**2)/(57915*mbkin**14) + (22495808*mcMS**15*np.pi**2)/
            (4095*mbkin**17) + (2969351680*mcMS**15*np.pi**2)/(243243*mbkin**16) + 
            (6784*mcMS**15*np.pi**2)/(3861*mbkin**15) - (20926*mcMS**16*np.pi**2)/
            (13*mbkin**18) - (418520*mcMS**16*np.pi**2)/(117*mbkin**17) + 
            (2127200*mcMS**17*np.pi**2)/(5967*mbkin**19) + (42544000*mcMS**17*np.pi**2)/
            (53703*mbkin**18) - (11719928*mcMS**18*np.pi**2)/(208845*mbkin**20) - 
            (46879712*mcMS**18*np.pi**2)/(375921*mbkin**19) + (684256*mcMS**19*np.pi**2)/
            (122265*mbkin**21) + (2737024*mcMS**19*np.pi**2)/(220077*mbkin**20) - 
            (150698*mcMS**20*np.pi**2)/(566865*mbkin**22) - (602792*mcMS**20*np.pi**2)/
            (1020357*mbkin**21) - (14120272*mcMS*np.pi**2)/(793611*mbkin**3*
            (1 - mcMS/mbkin)) - (112962176*mcMS*np.pi**2)/(2380833*mbkin**2*
            (1 - mcMS/mbkin)) - (7060136*mcMS*np.pi**2)/(793611*mbkin*
            (1 - mcMS/mbkin)) + (468025720*mcMS**2*np.pi**2)/(2380833*mbkin**4*
            (1 - mcMS/mbkin)) + (3744205760*mcMS**2*np.pi**2)/(7142499*mbkin**3*
            (1 - mcMS/mbkin)) + (234012860*mcMS**2*np.pi**2)/(2380833*mbkin**2*
            (1 - mcMS/mbkin)) - (107374328*mcMS**3*np.pi**2)/(89505*mbkin**5*
            (1 - mcMS/mbkin)) - (858994624*mcMS**3*np.pi**2)/(268515*mbkin**4*
            (1 - mcMS/mbkin)) - (53687164*mcMS**3*np.pi**2)/(89505*mbkin**3*
            (1 - mcMS/mbkin)) + (12111649028*mcMS**4*np.pi**2)/
            (2297295*mbkin**6*(1 - mcMS/mbkin)) + (96893192224*mcMS**4*np.pi**2)/
            (6891885*mbkin**5*(1 - mcMS/mbkin)) + (6055824514*mcMS**4*np.pi**2)/
            (2297295*mbkin**4*(1 - mcMS/mbkin)) - (804583036*mcMS**5*np.pi**2)/
            (45045*mbkin**7*(1 - mcMS/mbkin)) - (6436664288*mcMS**5*np.pi**2)/
            (135135*mbkin**6*(1 - mcMS/mbkin)) - (402291518*mcMS**5*np.pi**2)/
            (45045*mbkin**5*(1 - mcMS/mbkin)) + (16770272*mcMS**6*np.pi**2)/
            (351*mbkin**8*(1 - mcMS/mbkin)) + (134162176*mcMS**6*np.pi**2)/
            (1053*mbkin**7*(1 - mcMS/mbkin)) + (8385136*mcMS**6*np.pi**2)/
            (351*mbkin**6*(1 - mcMS/mbkin)) - (12012704*mcMS**7*np.pi**2)/
            (117*mbkin**9*(1 - mcMS/mbkin)) - (96101632*mcMS**7*np.pi**2)/
            (351*mbkin**8*(1 - mcMS/mbkin)) - (6006352*mcMS**7*np.pi**2)/
            (117*mbkin**7*(1 - mcMS/mbkin)) + (10420999024*mcMS**8*np.pi**2)/
            (57915*mbkin**10*(1 - mcMS/mbkin)) + (83367992192*mcMS**8*np.pi**2)/
            (173745*mbkin**9*(1 - mcMS/mbkin)) + (5210499512*mcMS**8*np.pi**2)/
            (57915*mbkin**8*(1 - mcMS/mbkin)) - (231726224*mcMS**9*np.pi**2)/
            (891*mbkin**11*(1 - mcMS/mbkin)) - (1853809792*mcMS**9*np.pi**2)/
            (2673*mbkin**10*(1 - mcMS/mbkin)) - (115863112*mcMS**9*np.pi**2)/
            (891*mbkin**9*(1 - mcMS/mbkin)) + (42146864*mcMS**10*np.pi**2)/
            (135*mbkin**12*(1 - mcMS/mbkin)) + (337174912*mcMS**10*np.pi**2)/
            (405*mbkin**11*(1 - mcMS/mbkin)) + (21073432*mcMS**10*np.pi**2)/
            (135*mbkin**10*(1 - mcMS/mbkin)) - (177061168*mcMS**11*np.pi**2)/
            (567*mbkin**13*(1 - mcMS/mbkin)) - (1416489344*mcMS**11*np.pi**2)/
            (1701*mbkin**12*(1 - mcMS/mbkin)) - (88530584*mcMS**11*np.pi**2)/
            (567*mbkin**11*(1 - mcMS/mbkin)) + (147579352*mcMS**12*np.pi**2)/
            (567*mbkin**14*(1 - mcMS/mbkin)) + (1180634816*mcMS**12*np.pi**2)/
            (1701*mbkin**13*(1 - mcMS/mbkin)) + (73789676*mcMS**12*np.pi**2)/
            (567*mbkin**12*(1 - mcMS/mbkin)) - (24329912*mcMS**13*np.pi**2)/
            (135*mbkin**15*(1 - mcMS/mbkin)) - (194639296*mcMS**13*np.pi**2)/
            (405*mbkin**14*(1 - mcMS/mbkin)) - (12164956*mcMS**13*np.pi**2)/
            (135*mbkin**13*(1 - mcMS/mbkin)) + (60252704*mcMS**14*np.pi**2)/
            (585*mbkin**16*(1 - mcMS/mbkin)) + (482021632*mcMS**14*np.pi**2)/
            (1755*mbkin**15*(1 - mcMS/mbkin)) + (30126352*mcMS**14*np.pi**2)/
            (585*mbkin**14*(1 - mcMS/mbkin)) - (6495875552*mcMS**15*np.pi**2)/
            (135135*mbkin**17*(1 - mcMS/mbkin)) - (51967004416*mcMS**15*np.pi**2)/
            (405405*mbkin**16*(1 - mcMS/mbkin)) - (3247937776*mcMS**15*np.pi**2)/
            (135135*mbkin**15*(1 - mcMS/mbkin)) + (54136688*mcMS**16*np.pi**2)/
            (3003*mbkin**18*(1 - mcMS/mbkin)) + (433093504*mcMS**16*np.pi**2)/
            (9009*mbkin**17*(1 - mcMS/mbkin)) + (27068344*mcMS**16*np.pi**2)/
            (3003*mbkin**16*(1 - mcMS/mbkin)) - (5583616*mcMS**17*np.pi**2)/
            (1053*mbkin**19*(1 - mcMS/mbkin)) - (44668928*mcMS**17*np.pi**2)/
            (3159*mbkin**18*(1 - mcMS/mbkin)) - (2791808*mcMS**17*np.pi**2)/
            (1053*mbkin**17*(1 - mcMS/mbkin)) + (738321352*mcMS**18*np.pi**2)/
            (626535*mbkin**20*(1 - mcMS/mbkin)) + (5906570816*mcMS**18*np.pi**2)/
            (1879605*mbkin**19*(1 - mcMS/mbkin)) + (369160676*mcMS**18*np.pi**2)/
            (626535*mbkin**18*(1 - mcMS/mbkin)) - (427470856*mcMS**19*np.pi**2)/
            (2297295*mbkin**21*(1 - mcMS/mbkin)) - (3419766848*mcMS**19*np.pi**2)/
            (6891885*mbkin**20*(1 - mcMS/mbkin)) - (213735428*mcMS**19*np.pi**2)/
            (2297295*mbkin**19*(1 - mcMS/mbkin)) + (348098812*mcMS**20*np.pi**2)/
            (18706545*mbkin**22*(1 - mcMS/mbkin)) + (2784790496*mcMS**20*np.pi**2)/
            (56119635*mbkin**21*(1 - mcMS/mbkin)) + (174049406*mcMS**20*np.pi**2)/
            (18706545*mbkin**20*(1 - mcMS/mbkin)) - (301396*mcMS**21*np.pi**2)/
            (340119*mbkin**23*(1 - mcMS/mbkin)) - (2411168*mcMS**21*np.pi**2)/
            (1020357*mbkin**22*(1 - mcMS/mbkin)) - (150698*mcMS**21*np.pi**2)/
            (340119*mbkin**21*(1 - mcMS/mbkin)) - (4656*mcMS*np.pi**4)/(385*mbkin) + 
            (352*mcMS**2*np.pi**4)/(9*mbkin**2) - (3536*mcMS**3*np.pi**4)/(45*mbkin**3) + 
            (4056*mcMS**4*np.pi**4)/(35*mbkin**4) - (416*mcMS**5*np.pi**4)/(3*mbkin**5) + 
            (2048*mcMS**6*np.pi**4)/(15*mbkin**6) - (736*mcMS**7*np.pi**4)/(7*mbkin**7) + 
            (2088*mcMS**8*np.pi**4)/(35*mbkin**8) - (1072*mcMS**9*np.pi**4)/(45*mbkin**9) + 
            (32*mcMS**10*np.pi**4)/(5*mbkin**10) - (3664*mcMS**11*np.pi**4)/
            (3465*mbkin**11) + (8*mcMS**12*np.pi**4)/(99*mbkin**12) + 
            (-304/(81*mbkin**3) + 13022602/(3357585*mbkin**2) + 
            52090408/(6043653*mbkin) + (141202720*mcMS)/(793611*mbkin**3) + 
            (2824054400*mcMS)/(7142499*mbkin**2) + (333324989499253*mcMS)/
                (547844046750*mbkin) - (160*mcMS**2)/(9*mbkin**5) - 
            (2070904*mcMS**2)/(1053*mbkin**4) - (41418080*mcMS**2)/
                (9477*mbkin**3) - (15478417721719*mcMS**2)/(3557428875*mbkin**2) + 
            (752032*mcMS**3)/(65*mbkin**5) + (3008128*mcMS**3)/(117*mbkin**4) + 
            (683837705723971*mcMS**3)/(42141849750*mbkin**3) + 
            (464*mcMS**4)/(27*mbkin**7) - (6606707542*mcMS**4)/(135135*mbkin**6) - 
            (26426830168*mcMS**4)/(243243*mbkin**5) - (561305364499738*mcMS**4)/
                (12642554925*mbkin**4) + (55416448*mcMS**5)/(351*mbkin**7) + 
            (1108328960*mcMS**5)/(3159*mbkin**6) + (70552906019*mcMS**5)/
                (727650*mbkin**5) + (4400*mcMS**6)/(81*mbkin**9) - 
            (139918240*mcMS**6)/(351*mbkin**8) - (2798364800*mcMS**6)/
                (3159*mbkin**7) - (45457380358813*mcMS**6)/(273648375*mbkin**6) + 
            (73025920*mcMS**7)/(91*mbkin**9) + (1460518400*mcMS**7)/
                (819*mbkin**8) + (119765484898127*mcMS**7)/(547296750*mbkin**7) - 
            (32*mcMS**8)/(3*mbkin**11) - (5833020164*mcMS**8)/(4455*mbkin**10) - 
            (23332080656*mcMS**8)/(8019*mbkin**9) - (47118708753404*mcMS**8)/
                (212837625*mbkin**8) + (709026368*mcMS**9)/(405*mbkin**11) + 
            (2836105472*mcMS**9)/(729*mbkin**10) + (6305059020913*mcMS**9)/
                (36486450*mbkin**9) - (260544944*mcMS**10)/(135*mbkin**12) - 
            (1042179776*mcMS**10)/(243*mbkin**11) - (7955554444931*mcMS**10)/
                (76621545*mbkin**10) + (10962228160*mcMS**11)/(6237*mbkin**13) + 
            (219244563200*mcMS**11)/(56133*mbkin**12) + 
            (181064392519811*mcMS**11)/(3831077250*mbkin**11) - 
            (3960380*mcMS**12)/(3*mbkin**14) - (79207600*mcMS**12)/
                (27*mbkin**13) - (332372802464726*mcMS**12)/(21070924875*
                mbkin**12) + (109805696*mcMS**13)/(135*mbkin**15) + 
            (439222784*mcMS**13)/(243*mbkin**14) + (5983215661469*mcMS**13)/
                (1641890250*mbkin**13) - (5001335648*mcMS**14)/(12285*mbkin**16) - 
            (20005342592*mcMS**14)/(22113*mbkin**15) - (194192359957*mcMS**14)/
                (372683025*mbkin**14) + (7341836416*mcMS**15)/(45045*mbkin**17) + 
            (29367345664*mcMS**15)/(81081*mbkin**16) + (3809167143809*mcMS**15)/
                (109568809350*mbkin**15) - (1987970*mcMS**16)/(39*mbkin**18) - 
            (39759400*mcMS**16)/(351*mbkin**17) + (214847200*mcMS**17)/
                (17901*mbkin**19) + (4296944000*mcMS**17)/(161109*mbkin**18) - 
            (1254032296*mcMS**18)/(626535*mbkin**20) - (5016129184*mcMS**18)/
                (1127763*mbkin**19) + (77320928*mcMS**19)/(366795*mbkin**21) + 
            (309283712*mcMS**19)/(660231*mbkin**20) - (1054886*mcMS**20)/
                (100035*mbkin**22) - (4219544*mcMS**20)/(180063*mbkin**21) - 
            (416*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**5) - 
            (3328*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**4) + 
            (16954*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**3) - 
            (2912*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(9*mbkin**9) - 
            (23296*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**8) - 
            (7006*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(9*mbkin**7) - 
            (294080*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**5) - 
            (2352640*(mcMS**2/mbkin**2)**(3/2))/(2187*mbkin**4) - 
            (685292*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**3) + 
            (376528*mcMS**2)/(1053*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (3012224*mcMS**2)/(3159*mbkin**3*(1 - mcMS/mbkin)**2) - 
            (753056*mcMS**2)/(1053*mbkin**2*(1 - mcMS/mbkin)**2) - 
            (429497312*mcMS**3)/(89505*mbkin**5*(1 - mcMS/mbkin)**2) - 
            (3435978496*mcMS**3)/(268515*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (858994624*mcMS**3)/(89505*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (234545329016*mcMS**4)/(6891885*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (1876362632128*mcMS**4)/(20675655*mbkin**5*(1 - mcMS/mbkin)**2) - 
            (469090658032*mcMS**4)/(6891885*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (75343294960*mcMS**5)/(459459*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (602746359680*mcMS**5)/(1378377*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (150686589920*mcMS**5)/(459459*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (79048041848*mcMS**6)/(135135*mbkin**8*(1 - mcMS/mbkin)**2) + 
            (632384334784*mcMS**6)/(405405*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (158096083696*mcMS**6)/(135135*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (566619520*mcMS**7)/(351*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (4532956160*mcMS**7)/(1053*mbkin**8*(1 - mcMS/mbkin)**2) + 
            (1133239040*mcMS**7)/(351*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (205356871136*mcMS**8)/(57915*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (1642854969088*mcMS**8)/(173745*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (410713742272*mcMS**8)/(57915*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (366047261248*mcMS**9)/(57915*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (2928378089984*mcMS**9)/(173745*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (732094522496*mcMS**9)/(57915*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (41256072896*mcMS**10)/(4455*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (330048583168*mcMS**10)/(13365*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (82512145792*mcMS**10)/(4455*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (31867463104*mcMS**11)/(2835*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (254939704832*mcMS**11)/(8505*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (63734926208*mcMS**11)/(2835*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (6433846768*mcMS**12)/(567*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (51470774144*mcMS**12)/(1701*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (12867693536*mcMS**12)/(567*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (27020210848*mcMS**13)/(2835*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (216161686784*mcMS**13)/(8505*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (54040421696*mcMS**13)/(2835*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (896774288*mcMS**14)/(135*mbkin**16*(1 - mcMS/mbkin)**2) + 
            (7174194304*mcMS**14)/(405*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (1793548576*mcMS**14)/(135*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (73703643776*mcMS**15)/(19305*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (589629150208*mcMS**15)/(57915*mbkin**16*(1 - mcMS/mbkin)**2) + 
            (147407287552*mcMS**15)/(19305*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (34568184736*mcMS**16)/(19305*mbkin**18*(1 - mcMS/mbkin)**2) + 
            (276545477888*mcMS**16)/(57915*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (69136369472*mcMS**16)/(19305*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (7812195136*mcMS**17)/(11583*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (62497561088*mcMS**17)/(34749*mbkin**18*(1 - mcMS/mbkin)**2) + 
            (15624390272*mcMS**17)/(11583*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (7339439504*mcMS**18)/(36855*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (58715516032*mcMS**18)/(110565*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (14678879008*mcMS**18)/(36855*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (18003292256*mcMS**19)/(405405*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (144026338048*mcMS**19)/(1216215*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (36006584512*mcMS**19)/(405405*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (9695082136*mcMS**20)/(1378377*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (77560657088*mcMS**20)/(4135131*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (19390164272*mcMS**20)/(1378377*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (694454128*mcMS**21)/(984555*mbkin**23*(1 - mcMS/mbkin)**2) - 
            (5555633024*mcMS**21)/(2953665*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (1388908256*mcMS**21)/(984555*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (602792*mcMS**22)/(17901*mbkin**24*(1 - mcMS/mbkin)**2) + 
            (4822336*mcMS**22)/(53703*mbkin**23*(1 - mcMS/mbkin)**2) - 
            (1205584*mcMS**22)/(17901*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (70601360*mcMS)/(2380833*mbkin**3*(1 - mcMS/mbkin)) - 
            (564810880*mcMS)/(7142499*mbkin**2*(1 - mcMS/mbkin)) + 
            (732489110*mcMS)/(2380833*mbkin*(1 - mcMS/mbkin)) + 
            (2340128600*mcMS**2)/(7142499*mbkin**4*(1 - mcMS/mbkin)) + 
            (18721028800*mcMS**2)/(21427497*mbkin**3*(1 - mcMS/mbkin)) - 
            (24278834225*mcMS**2)/(7142499*mbkin**2*(1 - mcMS/mbkin)) - 
            (107374328*mcMS**3)/(53703*mbkin**5*(1 - mcMS/mbkin)) - 
            (858994624*mcMS**3)/(161109*mbkin**4*(1 - mcMS/mbkin)) + 
            (1114008653*mcMS**3)/(53703*mbkin**3*(1 - mcMS/mbkin)) + 
            (12111649028*mcMS**4)/(1378377*mbkin**6*(1 - mcMS/mbkin)) + 
            (96893192224*mcMS**4)/(4135131*mbkin**5*(1 - mcMS/mbkin)) - 
            (251316717331*mcMS**4)/(2756754*mbkin**4*(1 - mcMS/mbkin)) - 
            (804583036*mcMS**5)/(27027*mbkin**7*(1 - mcMS/mbkin)) - 
            (6436664288*mcMS**5)/(81081*mbkin**6*(1 - mcMS/mbkin)) + 
            (16695097997*mcMS**5)/(54054*mbkin**5*(1 - mcMS/mbkin)) + 
            (83851360*mcMS**6)/(1053*mbkin**8*(1 - mcMS/mbkin)) + 
            (670810880*mcMS**6)/(3159*mbkin**7*(1 - mcMS/mbkin)) - 
            (869957860*mcMS**6)/(1053*mbkin**6*(1 - mcMS/mbkin)) - 
            (60063520*mcMS**7)/(351*mbkin**9*(1 - mcMS/mbkin)) - 
            (480508160*mcMS**7)/(1053*mbkin**8*(1 - mcMS/mbkin)) + 
            (623159020*mcMS**7)/(351*mbkin**7*(1 - mcMS/mbkin)) + 
            (10420999024*mcMS**8)/(34749*mbkin**10*(1 - mcMS/mbkin)) + 
            (83367992192*mcMS**8)/(104247*mbkin**9*(1 - mcMS/mbkin)) - 
            (108117864874*mcMS**8)/(34749*mbkin**8*(1 - mcMS/mbkin)) - 
            (1158631120*mcMS**9)/(2673*mbkin**11*(1 - mcMS/mbkin)) - 
            (9269048960*mcMS**9)/(8019*mbkin**10*(1 - mcMS/mbkin)) + 
            (12020797870*mcMS**9)/(2673*mbkin**9*(1 - mcMS/mbkin)) + 
            (42146864*mcMS**10)/(81*mbkin**12*(1 - mcMS/mbkin)) + 
            (337174912*mcMS**10)/(243*mbkin**11*(1 - mcMS/mbkin)) - 
            (437273714*mcMS**10)/(81*mbkin**10*(1 - mcMS/mbkin)) - 
            (885305840*mcMS**11)/(1701*mbkin**13*(1 - mcMS/mbkin)) - 
            (7082446720*mcMS**11)/(5103*mbkin**12*(1 - mcMS/mbkin)) + 
            (9185048090*mcMS**11)/(1701*mbkin**11*(1 - mcMS/mbkin)) + 
            (737896760*mcMS**12)/(1701*mbkin**14*(1 - mcMS/mbkin)) + 
            (5903174080*mcMS**12)/(5103*mbkin**13*(1 - mcMS/mbkin)) - 
            (7655678885*mcMS**12)/(1701*mbkin**12*(1 - mcMS/mbkin)) - 
            (24329912*mcMS**13)/(81*mbkin**15*(1 - mcMS/mbkin)) - 
            (194639296*mcMS**13)/(243*mbkin**14*(1 - mcMS/mbkin)) + 
            (252422837*mcMS**13)/(81*mbkin**13*(1 - mcMS/mbkin)) + 
            (60252704*mcMS**14)/(351*mbkin**16*(1 - mcMS/mbkin)) + 
            (482021632*mcMS**14)/(1053*mbkin**15*(1 - mcMS/mbkin)) - 
            (625121804*mcMS**14)/(351*mbkin**14*(1 - mcMS/mbkin)) - 
            (6495875552*mcMS**15)/(81081*mbkin**17*(1 - mcMS/mbkin)) - 
            (51967004416*mcMS**15)/(243243*mbkin**16*(1 - mcMS/mbkin)) + 
            (67394708852*mcMS**15)/(81081*mbkin**15*(1 - mcMS/mbkin)) + 
            (270683440*mcMS**16)/(9009*mbkin**18*(1 - mcMS/mbkin)) + 
            (2165467520*mcMS**16)/(27027*mbkin**17*(1 - mcMS/mbkin)) - 
            (2808340690*mcMS**16)/(9009*mbkin**16*(1 - mcMS/mbkin)) - 
            (27918080*mcMS**17)/(3159*mbkin**19*(1 - mcMS/mbkin)) - 
            (223344640*mcMS**17)/(9477*mbkin**18*(1 - mcMS/mbkin)) + 
            (289650080*mcMS**17)/(3159*mbkin**17*(1 - mcMS/mbkin)) + 
            (738321352*mcMS**18)/(375921*mbkin**20*(1 - mcMS/mbkin)) + 
            (5906570816*mcMS**18)/(1127763*mbkin**19*(1 - mcMS/mbkin)) - 
            (7660084027*mcMS**18)/(375921*mbkin**18*(1 - mcMS/mbkin)) - 
            (427470856*mcMS**19)/(1378377*mbkin**21*(1 - mcMS/mbkin)) - 
            (3419766848*mcMS**19)/(4135131*mbkin**20*(1 - mcMS/mbkin)) + 
            (4435010131*mcMS**19)/(1378377*mbkin**19*(1 - mcMS/mbkin)) + 
            (348098812*mcMS**20)/(11223927*mbkin**22*(1 - mcMS/mbkin)) + 
            (2784790496*mcMS**20)/(33671781*mbkin**21*(1 - mcMS/mbkin)) - 
            (7223050349*mcMS**20)/(22447854*mbkin**20*(1 - mcMS/mbkin)) - 
            (1506980*mcMS**21)/(1020357*mbkin**23*(1 - mcMS/mbkin)) - 
            (12055840*mcMS**21)/(3061071*mbkin**22*(1 - mcMS/mbkin)) + 
            (31269835*mcMS**21)/(2040714*mbkin**21*(1 - mcMS/mbkin)) - 
            (1936*mcMS)/(729*mbkin**6*(-1 + mcMS/mbkin)) - 
            (15488*mcMS)/(2187*mbkin**5*(-1 + mcMS/mbkin)) + 
            (5936*mcMS)/(729*mbkin**4*(-1 + mcMS/mbkin)) + (1936*mcMS**2)/
                (729*mbkin**7*(-1 + mcMS/mbkin)) + (15488*mcMS**2)/
                (2187*mbkin**6*(-1 + mcMS/mbkin)) - (5936*mcMS**2)/
                (729*mbkin**5*(-1 + mcMS/mbkin)) - (341600*mcMS**3)/
                (729*mbkin**8*(-1 + mcMS/mbkin)) - (2732800*mcMS**3)/
                (2187*mbkin**7*(-1 + mcMS/mbkin)) + (540640*mcMS**3)/
                (729*mbkin**6*(-1 + mcMS/mbkin)) + (341600*mcMS**4)/
                (729*mbkin**9*(-1 + mcMS/mbkin)) + (2732800*mcMS**4)/
                (2187*mbkin**8*(-1 + mcMS/mbkin)) - (540640*mcMS**4)/
                (729*mbkin**7*(-1 + mcMS/mbkin)) - (1456*mcMS**5)/
                (3*mbkin**10*(-1 + mcMS/mbkin)) - (11648*mcMS**5)/
                (9*mbkin**9*(-1 + mcMS/mbkin)) + (7280*mcMS**5)/
                (9*mbkin**8*(-1 + mcMS/mbkin)) + (1456*mcMS**6)/
                (3*mbkin**11*(-1 + mcMS/mbkin)) + (11648*mcMS**6)/
                (9*mbkin**10*(-1 + mcMS/mbkin)) - (7280*mcMS**6)/
                (9*mbkin**9*(-1 + mcMS/mbkin)) - (180704*mcMS*np.pi**2)/(9009*mbkin) + 
            (600176*mcMS**2*np.pi**2)/(3861*mbkin**2) - (308192*mcMS**3*np.pi**2)/
                (495*mbkin**3) + (2032544*mcMS**4*np.pi**2)/(1155*mbkin**4) - 
            (103328*mcMS**5*np.pi**2)/(27*mbkin**5) + (58064*mcMS**6*np.pi**2)/
                (9*mbkin**6) - (25184*mcMS**7*np.pi**2)/(3*mbkin**7) + 
            (531904*mcMS**8*np.pi**2)/(63*mbkin**8) - (98656*mcMS**9*np.pi**2)/
                (15*mbkin**9) + (35536*mcMS**10*np.pi**2)/(9*mbkin**10) - 
            (339424*mcMS**11*np.pi**2)/(189*mbkin**11) + (59296*mcMS**12*np.pi**2)/
                (99*mbkin**12) - (22816*mcMS**13*np.pi**2)/(165*mbkin**13) + 
            (381488*mcMS**14*np.pi**2)/(19305*mbkin**14) - (1696*mcMS**15*np.pi**2)/
                (1287*mbkin**15) - (172*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(243*mbkin**3) + 
            (364*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(9*mbkin**7) + 
            (440*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(9*mbkin**3))*
            np.log(mcMS**2/mbkin**2) + ((-3764*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**3) - 
            (9100*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) - 
            (103480*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) + 
            (94132*mcMS**2)/(351*mbkin**2*(1 - mcMS/mbkin)**2) - 
            (107374328*mcMS**3)/(29835*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (58636332254*mcMS**4)/(2297295*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (18835823740*mcMS**5)/(153153*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (19762010462*mcMS**6)/(45045*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (141654880*mcMS**7)/(117*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (51339217784*mcMS**8)/(19305*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (91511815312*mcMS**9)/(19305*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (10314018224*mcMS**10)/(1485*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (7966865776*mcMS**11)/(945*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (1608461692*mcMS**12)/(189*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (6755052712*mcMS**13)/(945*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (224193572*mcMS**14)/(45*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (18425910944*mcMS**15)/(6435*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (8642046184*mcMS**16)/(6435*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (1953048784*mcMS**17)/(3861*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (1834859876*mcMS**18)/(12285*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (4500823064*mcMS**19)/(135135*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (2423770534*mcMS**20)/(459459*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (173613532*mcMS**21)/(328185*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (150698*mcMS**22)/(5967*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (65306258*mcMS)/(793611*mbkin*(1 - mcMS/mbkin)) + 
            (2164618955*mcMS**2)/(2380833*mbkin**2*(1 - mcMS/mbkin)) - 
            (496606267*mcMS**3)/(89505*mbkin**3*(1 - mcMS/mbkin)) + 
            (112032753509*mcMS**4)/(4594590*mbkin**4*(1 - mcMS/mbkin)) - 
            (7442393083*mcMS**5)/(90090*mbkin**5*(1 - mcMS/mbkin)) + 
            (77562508*mcMS**6)/(351*mbkin**6*(1 - mcMS/mbkin)) - 
            (55558756*mcMS**7)/(117*mbkin**7*(1 - mcMS/mbkin)) + 
            (48197120486*mcMS**8)/(57915*mbkin**8*(1 - mcMS/mbkin)) - 
            (1071733786*mcMS**9)/(891*mbkin**9*(1 - mcMS/mbkin)) + 
            (194929246*mcMS**10)/(135*mbkin**10*(1 - mcMS/mbkin)) - 
            (818907902*mcMS**11)/(567*mbkin**11*(1 - mcMS/mbkin)) + 
            (682554503*mcMS**12)/(567*mbkin**12*(1 - mcMS/mbkin)) - 
            (112525843*mcMS**13)/(135*mbkin**13*(1 - mcMS/mbkin)) + 
            (278668756*mcMS**14)/(585*mbkin**14*(1 - mcMS/mbkin)) - 
            (30043424428*mcMS**15)/(135135*mbkin**15*(1 - mcMS/mbkin)) + 
            (250382182*mcMS**16)/(3003*mbkin**16*(1 - mcMS/mbkin)) - 
            (25824224*mcMS**17)/(1053*mbkin**17*(1 - mcMS/mbkin)) + 
            (3414736253*mcMS**18)/(626535*mbkin**18*(1 - mcMS/mbkin)) - 
            (1977052709*mcMS**19)/(2297295*mbkin**19*(1 - mcMS/mbkin)) + 
            (3219914011*mcMS**20)/(37413090*mbkin**20*(1 - mcMS/mbkin)) - 
            (2787913*mcMS**21)/(680238*mbkin**21*(1 - mcMS/mbkin)) - 
            (968*mcMS)/(243*mbkin**4*(-1 + mcMS/mbkin)) + (968*mcMS**2)/
                (243*mbkin**5*(-1 + mcMS/mbkin)) - (170800*mcMS**3)/
                (243*mbkin**6*(-1 + mcMS/mbkin)) + (170800*mcMS**4)/
                (243*mbkin**7*(-1 + mcMS/mbkin)) - (728*mcMS**5)/
                (mbkin**8*(-1 + mcMS/mbkin)) + (728*mcMS**6)/
                (mbkin**9*(-1 + mcMS/mbkin)))*np.log(mcMS**2/mbkin**2)**2)*
            np.log(1 - mcMS/mbkin) + (-612444982657/2917512675 - 
            39786064/(2606175*mbkin**2) - 159144256/(4691115*mbkin) + 
            (3580064*mcMS)/(405405*mbkin**3) + (114562048*mcMS)/
            (6081075*mbkin**2) + (163353094327534*mcMS)/(189638323875*mbkin) + 
            (888615968*mcMS**2)/(2606175*mbkin**4) + (1777231936*mcMS**2)/
            (2606175*mbkin**3) + (1432811618708*mcMS**2)/(2462835375*mbkin**2) - 
            (1987324448*mcMS**3)/(1403325*mbkin**5) - (31797191168*mcMS**3)/
            (12629925*mbkin**4) - (140992485674*mcMS**3)/(12629925*mbkin**3) + 
            (2262297616*mcMS**4)/(841995*mbkin**6) + (9049190464*mcMS**4)/
            (2525985*mbkin**5) + (435088432397*mcMS**4)/(12629925*mbkin**4) - 
            (4045856*mcMS**5)/(1575*mbkin**7) - (779808738796*mcMS**5)/
            (12629925*mbkin**5) - (1602644032*mcMS**6)/(164025*mbkin**7) + 
            (5025434831264*mcMS**6)/(63149625*mbkin**6) + (521545376*mcMS**7)/
            (127575*mbkin**9) + (8344726016*mcMS**7)/(382725*mbkin**8) - 
            (36286946545468*mcMS**7)/(442047375*mbkin**7) - (307974928*mcMS**8)/
            (42525*mbkin**10) - (1231899712*mcMS**8)/(42525*mbkin**9) + 
            (569054051471*mcMS**8)/(8037225*mbkin**8) + (5546336*mcMS**9)/
            (729*mbkin**11) + (177482752*mcMS**9)/(6561*mbkin**10) - 
            (1917905297986*mcMS**9)/(37889775*mbkin**9) - (141385696*mcMS**10)/
            (25515*mbkin**12) - (282771392*mcMS**10)/(15309*mbkin**11) + 
            (1100982598124*mcMS**10)/(37889775*mbkin**10) + (810886688*mcMS**11)/
            (280665*mbkin**13) + (12974187008*mcMS**11)/(1403325*mbkin**12) - 
            (26961815278186*mcMS**11)/(2083937625*mbkin**11) - 
            (1498143088*mcMS**12)/(1403325*mbkin**14) - (5992572352*mcMS**12)/
            (1804275*mbkin**13) + (425485476871*mcMS**12)/(99235125*mbkin**12) + 
            (2087832544*mcMS**13)/(7818525*mbkin**15) + (19088754688*mcMS**13)/
            (23455575*mbkin**14) - (596523584*mcMS**13)/(601425*mbkin**13) - 
            (16503104*mcMS**14)/(405405*mbkin**16) - (16503104*mcMS**14)/
            (135135*mbkin**15) + (8251552*mcMS**14)/(57915*mbkin**14) + 
            (17410912*mcMS**15)/(6081075*mbkin**17) + (278574592*mcMS**15)/
            (32837805*mbkin**16) - (34821824*mcMS**15)/(3648645*mbkin**15) + 
            (1470064*np.pi**2)/280665 - (1266080*mcMS*np.pi**2)/(56133*mbkin) + 
            (832*mcMS**2*np.pi**2)/(405*mbkin**2) + (4692448*mcMS**3*np.pi**2)/
            (25515*mbkin**3) - (4702672*mcMS**4*np.pi**2)/(8505*mbkin**4) + 
            (490816*mcMS**5*np.pi**2)/(567*mbkin**5) - (212992*mcMS**6*np.pi**2)/
            (243*mbkin**6) + (5329216*mcMS**7*np.pi**2)/(8505*mbkin**7) - 
            (938512*mcMS**8*np.pi**2)/(2835*mbkin**8) + (472288*mcMS**9*np.pi**2)/
            (3645*mbkin**9) - (904384*mcMS**10*np.pi**2)/(25515*mbkin**10) + 
            (37024*mcMS**11*np.pi**2)/(6237*mbkin**11) - (128624*mcMS**12*np.pi**2)/
            (280665*mbkin**12) + ((-1790032*mcMS)/(675675*mbkin) - 
            (222153992*mcMS**2)/(868725*mbkin**2) + (993662224*mcMS**3)/
                (467775*mbkin**3) - (2262297616*mcMS**4)/(280665*mbkin**4) + 
            (2022928*mcMS**5)/(105*mbkin**5) - (200330504*mcMS**6)/
                (6075*mbkin**6) + (260772688*mcMS**7)/(6075*mbkin**7) - 
            (615949856*mcMS**8)/(14175*mbkin**8) + (2773168*mcMS**9)/
                (81*mbkin**9) - (35346424*mcMS**10)/(1701*mbkin**10) + 
            (405443344*mcMS**11)/(42525*mbkin**11) - (1498143088*mcMS**12)/
                (467775*mbkin**12) + (149130896*mcMS**13)/(200475*mbkin**13) - 
            (2062888*mcMS**14)/(19305*mbkin**14) + (8705456*mcMS**15)/
                (1216215*mbkin**15))*np.log(mcMS**2/mbkin**2))*np.log(1 - mcMS/mbkin)**2 + 
        (23931088/2525985 - (14708576*mcMS)/(2525985*mbkin) - 
            (8950336*mcMS**2)/(32805*mbkin**2) + (33425824*mcMS**3)/
            (25515*mbkin**3) - (233018864*mcMS**4)/(76545*mbkin**4) + 
            (67270976*mcMS**5)/(15309*mbkin**5) - (47392768*mcMS**6)/
            (10935*mbkin**6) + (235510208*mcMS**7)/(76545*mbkin**7) - 
            (123588112*mcMS**8)/(76545*mbkin**8) + (20426656*mcMS**9)/
            (32805*mbkin**9) - (38385728*mcMS**10)/(229635*mbkin**10) + 
            (69725024*mcMS**11)/(2525985*mbkin**11) - (1775344*mcMS**12)/
            (841995*mbkin**12))*np.log(1 - mcMS/mbkin)**3 + 
        (-608/(243*mbkin**5) - 4864/(729*mbkin**4) + 1216/(243*mbkin**3) - 
            (320*mcMS**2)/(27*mbkin**7) - (2560*mcMS**2)/(81*mbkin**6) + 
            (640*mcMS**2)/(27*mbkin**5) + (928*mcMS**4)/(81*mbkin**9) + 
            (7424*mcMS**4)/(243*mbkin**8) - (1856*mcMS**4)/(81*mbkin**7) + 
            (8800*mcMS**6)/(243*mbkin**11) + (70400*mcMS**6)/(729*mbkin**10) - 
            (17600*mcMS**6)/(243*mbkin**9) - (64*mcMS**8)/(9*mbkin**13) - 
            (512*mcMS**8)/(27*mbkin**12) + (128*mcMS**8)/(9*mbkin**11) - 
            (3376*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**5) - 
            (27008*np.sqrt(0j + mcMS**2/mbkin**2))/(2187*mbkin**4) + 
            (6752*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**3) + 
            (5648*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**9) + 
            (45184*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**8) - 
            (11296*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**7) + 
            (49952*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**5) + 
            (399616*(mcMS**2/mbkin**2)**(3/2))/(2187*mbkin**4) - 
            (99904*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**3) + 
            (1376*mcMS)/(729*mbkin**6*(1 + mcMS/mbkin)) + 
            (11008*mcMS)/(2187*mbkin**5*(1 + mcMS/mbkin)) - 
            (2752*mcMS)/(729*mbkin**4*(1 + mcMS/mbkin)) + (1376*mcMS**2)/
            (729*mbkin**7*(1 + mcMS/mbkin)) + (11008*mcMS**2)/
            (2187*mbkin**6*(1 + mcMS/mbkin)) - (2752*mcMS**2)/
            (729*mbkin**5*(1 + mcMS/mbkin)) - (3520*mcMS**3)/
            (27*mbkin**8*(1 + mcMS/mbkin)) - (28160*mcMS**3)/
            (81*mbkin**7*(1 + mcMS/mbkin)) + (7040*mcMS**3)/
            (27*mbkin**6*(1 + mcMS/mbkin)) - (3520*mcMS**4)/
            (27*mbkin**9*(1 + mcMS/mbkin)) - (28160*mcMS**4)/
            (81*mbkin**8*(1 + mcMS/mbkin)) + (7040*mcMS**4)/
            (27*mbkin**7*(1 + mcMS/mbkin)) - (2912*mcMS**5)/
            (27*mbkin**10*(1 + mcMS/mbkin)) - (23296*mcMS**5)/
            (81*mbkin**9*(1 + mcMS/mbkin)) + (5824*mcMS**5)/
            (27*mbkin**8*(1 + mcMS/mbkin)) - (2912*mcMS**6)/
            (27*mbkin**11*(1 + mcMS/mbkin)) - (23296*mcMS**6)/
            (81*mbkin**10*(1 + mcMS/mbkin)) + (5824*mcMS**6)/
            (27*mbkin**9*(1 + mcMS/mbkin)))*np.log(1 + mcMS/mbkin) + 
        (62728/(729*mbkin**5) + 501824/(2187*mbkin**4) + 245971/(162*mbkin**3) - 
            23744/(729*mcMS**4) - 2968/(243*mbkin*mcMS**4) - 
            (30793*mbkin)/(1458*mcMS**4) + 359392/(10935*mbkin**3*mcMS**2) + 
            2875136/(32805*mbkin**2*mcMS**2) + 258313/(2187*mbkin*mcMS**2) - 
            (64*mcMS**2)/(27*mbkin**7) - (512*mcMS**2)/(81*mbkin**6) - 
            (1205510*mcMS**2)/(729*mbkin**5) - (196712*mcMS**4)/(2187*mbkin**9) - 
            (1573696*mcMS**4)/(6561*mbkin**8) - (2871017*mcMS**4)/(4374*mbkin**7) - 
            (241984*mcMS**6)/(729*mbkin**11) - (1935872*mcMS**6)/(2187*mbkin**10) + 
            (195403*mcMS**6)/(81*mbkin**9) + (353992*mcMS**8)/(1215*mbkin**13) + 
            (2831936*mcMS**8)/(3645*mbkin**12) - (2427967*mcMS**8)/
            (1458*mbkin**11) + (2000*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**5) + 
            (16000*np.sqrt(0j + mcMS**2/mbkin**2))/(2187*mbkin**4) - 
            (4000*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**3) + 
            (3088*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**9) + 
            (24704*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**8) - 
            (6176*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**7) + 
            (45088*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**5) + 
            (360704*(mcMS**2/mbkin**2)**(3/2))/(2187*mbkin**4) - 
            (90176*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**3) - 
            (13769*np.pi**2)/(243*mbkin**3) + (371*mbkin*np.pi**2)/(243*mcMS**4) - 
            (22462*np.pi**2)/(3645*mbkin*mcMS**2) + (13508*mcMS**2*np.pi**2)/
            (243*mbkin**5) + (20435*mcMS**4*np.pi**2)/(729*mbkin**7) - 
            (14242*mcMS**6*np.pi**2)/(243*mbkin**9) + (44089*mcMS**8*np.pi**2)/
            (1215*mbkin**11))*np.log(1 - mcMS**2/mbkin**2) + np.log(mcMS**2/mbkin**2)**2*
            (-4448/(243*mbkin**5) - 35584/(729*mbkin**4) - 92168/(729*mbkin**3) + 
            1/(9*mbkin**2) + 20/(81*mbkin) + (3689*mcMS**2)/(27*mbkin**5) - 
            (2176*mcMS**2)/(27*mbkin**4) - (4352*mcMS**2)/(27*mbkin**3) - 
            (500816321*mcMS**2)/(1587222*mbkin**2) - (383304088*mcMS**3)/
            (793611*mbkin**3) - (4112*mcMS**4)/(729*mbkin**9) - 
            (75664*mcMS**4)/(2187*mbkin**8) + (394621*mcMS**4)/(4374*mbkin**7) + 
            (12550*mcMS**4)/(9*mbkin**6) + (61592*mcMS**4)/(27*mbkin**5) + 
            (4961.843269314559*mcMS**4)/mbkin**4 - (34160890396*mcMS**5)/
            (3357585*mbkin**5) + (8128*mcMS**6)/(243*mbkin**11) + 
            (65024*mcMS**6)/(729*mbkin**10) + (28648*mcMS**6)/(243*mbkin**9) - 
            (26816*mcMS**6)/(81*mbkin**7) + (2959834964029*mcMS**6)/
            (87297210*mbkin**6) - (3136021172428*mcMS**7)/(43648605*mbkin**7) + 
            (2144*mcMS**8)/(81*mbkin**13) + (17152*mcMS**8)/(243*mbkin**12) - 
            (661411*mcMS**8)/(7290*mbkin**11) + (1405*mcMS**8)/(27*mbkin**10) + 
            (5620*mcMS**8)/(27*mbkin**9) + (7733547079451*mcMS**8)/
            (58198140*mbkin**8) - (2919038193364*mcMS**9)/(14549535*mbkin**9) + 
            (3569014609868*mcMS**10)/(14549535*mbkin**10) - 
            (1189994365324*mcMS**11)/(4849845*mbkin**11) + (1252064148332*mcMS**12)/
            (6235515*mbkin**12) - (834838011868*mcMS**13)/(6235515*mbkin**13) + 
            (449582744036*mcMS**14)/(6235515*mbkin**14) - (192697215868*mcMS**15)/
            (6235515*mbkin**15) + (449663179484*mcMS**16)/(43648605*mbkin**16) - 
            (112423410196*mcMS**17)/(43648605*mbkin**17) + (19840567804*mcMS**18)/
            (43648605*mbkin**18) - (314945252*mcMS**19)/(6235515*mbkin**19) + 
            (301396*mcMS**20)/(113373*mbkin**20) - (3820703038853*mcMS**2)/
            (20427547140*mbkin**2*(1 - mcMS/mbkin)**2) + 
            (1324623229305471443*mcMS**3)/(534793184125200*mbkin**3*
            (1 - mcMS/mbkin)**2) - (5514257196652051*mcMS**4)/
            (309307798800*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (2445534345846504631*mcMS**5)/(28147009690800*mbkin**5*
            (1 - mcMS/mbkin)**2) - (73546477272645373*mcMS**6)/
            (236529493200*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (70971441129844033*mcMS**7)/(82785322620*mbkin**7*(1 - mcMS/mbkin)**
                2) - (5292307230847823*mcMS**8)/(2815827300*mbkin**8*
            (1 - mcMS/mbkin)**2) + (1883705490346937*mcMS**9)/
            (563165460*mbkin**9*(1 - mcMS/mbkin)**2) - (3996921465647711*mcMS**10)/
            (816423300*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (34379222836281097*mcMS**11)/(5789183400*mbkin**11*(1 - mcMS/mbkin)**
                2) - (991151981540419*mcMS**12)/(165405240*mbkin**12*
            (1 - mcMS/mbkin)**2) + (2648132431926677*mcMS**13)/
            (526289400*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (223279079747087713*mcMS**14)/(63681017400*mbkin**14*
            (1 - mcMS/mbkin)**2) + (4765673910687463*mcMS**15)/
            (2365294932*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (55872539918685143*mcMS**16)/(59132373300*mbkin**16*
            (1 - mcMS/mbkin)**2) + (251792791028627*mcMS**17)/
            (707566860*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (2416054156353307*mcMS**18)/(22995922950*mbkin**18*(1 - mcMS/mbkin)**
                2) + (1436625073319083*mcMS**19)/(61322461200*mbkin**19*
            (1 - mcMS/mbkin)**2) - (20887342577910533*mcMS**20)/
            (5629401938160*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (198978903088724189*mcMS**21)/(534793184125200*mbkin**21*
            (1 - mcMS/mbkin)**2) - (24672660896281*mcMS**22)/
            (1389073205520*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (84769718158347181*mcMS)/(1477973890673280*mbkin*(1 - mcMS/mbkin)) - 
            (2446470123833473391*mcMS**2)/(4433921672019840*mbkin**2*
            (1 - mcMS/mbkin)) + (43176344676356419391*mcMS**3)/
            (12835036419004800*mbkin**3*(1 - mcMS/mbkin)) - 
            (4378795260555490441*mcMS**4)/(285223031533440*mbkin**4*
            (1 - mcMS/mbkin)) + (1825778755943637871*mcMS**5)/
            (34318279195200*mbkin**5*(1 - mcMS/mbkin)) - 
            (5409718168456526123*mcMS**6)/(37750107114720*mbkin**6*
            (1 - mcMS/mbkin)) + (2398306644621841*mcMS**7)/(7781922720*mbkin**7*
            (1 - mcMS/mbkin)) - (285450062438881289*mcMS**8)/
            (528712984800*mbkin**8*(1 - mcMS/mbkin)) + 
            (82990069185646877*mcMS**9)/(106368292800*mbkin**9*(1 - mcMS/mbkin)) - 
            (63406249636229341*mcMS**10)/(67688913600*mbkin**10*
            (1 - mcMS/mbkin)) + (494853577058200963*mcMS**11)/
            (527973526080*mbkin**11*(1 - mcMS/mbkin)) - 
            (412608449195796241*mcMS**12)/(527973526080*mbkin**12*
            (1 - mcMS/mbkin)) + (6615633306531721*mcMS**13)/
            (12221609400*mbkin**13*(1 - mcMS/mbkin)) - 
            (1192039271175325699*mcMS**14)/(3852051746400*mbkin**14*
            (1 - mcMS/mbkin)) + (1604043546702600131*mcMS**15)/
            (11102972680800*mbkin**15*(1 - mcMS/mbkin)) - 
            (4133080133491195*mcMS**16)/(76262842656*mbkin**16*(1 - mcMS/mbkin)) + 
            (267518569219007159*mcMS**17)/(16777825384320*mbkin**17*
            (1 - mcMS/mbkin)) - (65643525989913343*mcMS**18)/
            (18520976073600*mbkin**18*(1 - mcMS/mbkin)) + 
            (2394868598393790877*mcMS**19)/(4278345473001600*mbkin**19*
            (1 - mcMS/mbkin)) - (13653883931434631303*mcMS**20)/
            (243865691961091200*mbkin**20*(1 - mcMS/mbkin)) + 
            (844572094812157*mcMS**21)/(316708690858560*mbkin**21*
            (1 - mcMS/mbkin)) - (344*mcMS**2)/(243*mbkin**5*(-1 + mcMS/mbkin)) + 
            (880*mcMS**4)/(9*mbkin**7*(-1 + mcMS/mbkin)) + 
            (728*mcMS**6)/(9*mbkin**9*(-1 + mcMS/mbkin)) + 
            (344*mcMS**2)/(243*mbkin**5*(1 + mcMS/mbkin)) - 
            (880*mcMS**4)/(9*mbkin**7*(1 + mcMS/mbkin)) - (728*mcMS**6)/
            (9*mbkin**9*(1 + mcMS/mbkin)) - 7280/(81*mbkin*(mbkin**2 - mcMS**2)) + 
            (5888*mcMS**2)/(81*mbkin**3*(mbkin**2 - mcMS**2)) + 
            (7744*mcMS**4)/(81*mbkin**5*(mbkin**2 - mcMS**2)) + 
            (57776*mcMS**6)/(243*mbkin**7*(mbkin**2 - mcMS**2)) + 
            (1696*mcMS**8)/(81*mbkin**9*(mbkin**2 - mcMS**2)) + 
            (1408*mcMS**10)/(81*mbkin**11*(mbkin**2 - mcMS**2)) + 
            (556*np.pi**2)/(81*mbkin**3) + (154*mcMS**2*np.pi**2)/(81*mbkin**5) - 
            (514*mcMS**4*np.pi**2)/(243*mbkin**7) - (88*mcMS**4*np.pi**2)/(3*mbkin**6) - 
            (352*mcMS**4*np.pi**2)/(9*mbkin**5) + (196*mcMS**4*np.pi**2)/(3*mbkin**4) + 
            (508*mcMS**6*np.pi**2)/(81*mbkin**9) + (268*mcMS**8*np.pi**2)/(81*mbkin**11) + 
            ((3764*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**3) + 
            (9100*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (103480*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) - 
            (968*mcMS)/(243*mbkin**4*(1 + mcMS/mbkin)) - (968*mcMS**2)/
                (243*mbkin**5*(1 + mcMS/mbkin)) - (170800*mcMS**3)/
                (243*mbkin**6*(1 + mcMS/mbkin)) - (170800*mcMS**4)/
                (243*mbkin**7*(1 + mcMS/mbkin)) - (728*mcMS**5)/
                (mbkin**8*(1 + mcMS/mbkin)) - (728*mcMS**6)/
                (mbkin**9*(1 + mcMS/mbkin)))*np.log(1 + mcMS/mbkin) + 
            (1708/(243*mbkin**3) - (484*mcMS**2)/(27*mbkin**5) - 
            (302956*mcMS**4)/(729*mbkin**7) - (26600*mcMS**6)/(243*mbkin**9) - 
            (16544*mcMS**8)/(243*mbkin**11))*np.log(1 - mcMS**2/mbkin**2)) + 
        ((-13*mcMS**2)/(27*mbkin**2) - (481*mcMS**4)/(18*mbkin**4) + 
            (517*mcMS**6)/(9*mbkin**6) - (1633*mcMS**8)/(54*mbkin**8) + 
            (23*mcMS**4*np.log(mcMS**2/mbkin**2))/(9*mbkin**4))*np.log(mu0**2/mus**2)**3 + 
        (-204/mbkin**3 + 81/mbkin**2 + 180/mbkin + (192*mcMS**2)/mbkin**5 - 
            (432*mcMS**2)/mbkin**4 - (864*mcMS**2)/mbkin**3 + (144*mcMS**4)/mbkin**7 + 
            (324*mcMS**4)/mbkin**6 + (864*mcMS**4)/mbkin**5 - (192*mcMS**6)/mbkin**9 - 
            (288*mcMS**6)/mbkin**7 + (60*mcMS**8)/mbkin**11 + (27*mcMS**8)/mbkin**10 + 
            (108*mcMS**8)/mbkin**9 + (-144/mbkin**3 - (324*mcMS**4)/mbkin**6 - 
            (432*mcMS**4)/mbkin**5)*np.log(mcMS**2/mbkin**2))*np.log(2/mus)**2 + 
        (-649214328724507313/27096187995676800 - 3092/(243*mbkin**3) + 
            1/(9*mbkin**2) + 20/(81*mbkin) + (97800084676639637*mcMS)/
            (497685085634880*mbkin) - (3944684*mcMS)/(405405*(mbkin - mcMS)) + 
            (8704*mcMS**2)/(243*mbkin**5) - (2176*mcMS**2)/(27*mbkin**4) - 
            (4352*mcMS**2)/(27*mbkin**3) - (46805073050393*mcMS**2)/
            (42800574960*mbkin**2) + (2078096*mcMS**2)/(27027*mbkin*
            (mbkin - mcMS)) + (38612610883433682731*mcMS**3)/
            (19252554628507200*mbkin**3) - (3451012*mcMS**3)/
            (11583*mbkin**2*(mbkin - mcMS)) + (6640*mcMS**4)/(81*mbkin**7) + 
            (256*mcMS**4)/(9*mbkin**6) + (7664*mcMS**4)/(27*mbkin**5) - 
            (334623298906295687*mcMS**4)/(50333476152960*mbkin**4) + 
            (3544208*mcMS**4)/(4455*mbkin**3*(mbkin - mcMS)) + 
            (627014358573488279*mcMS**5)/(94375267786800*mbkin**5) - 
            (5843564*mcMS**5)/(3465*mbkin**4*(mbkin - mcMS)) - 
            (53632*mcMS**6)/(243*mbkin**9) - (26816*mcMS**6)/(81*mbkin**7) + 
            (5548892161367*mcMS**6)/(2403240840*mbkin**6) + (1188272*mcMS**6)/
            (405*mbkin**5*(mbkin - mcMS)) - (13481705292802667*mcMS**7)/
            (571971319920*mbkin**7) - (333868*mcMS**7)/(81*mbkin**6*
            (mbkin - mcMS)) + (28100*mcMS**8)/(243*mbkin**11) + 
            (1405*mcMS**8)/(27*mbkin**10) + (5620*mcMS**8)/(27*mbkin**9) + 
            (2061838534875439889*mcMS**8)/(29038543934400*mbkin**8) + 
            (289616*mcMS**8)/(63*mbkin**7*(mbkin - mcMS)) - 
            (14985440960897269*mcMS**9)/(113137184160*mbkin**9) - 
            (764612*mcMS**9)/(189*mbkin**8*(mbkin - mcMS)) + 
            (7426281032025892*mcMS**10)/(41247931725*mbkin**10) + 
            (1134544*mcMS**10)/(405*mbkin**9*(mbkin - mcMS)) - 
            (1665985504485284299*mcMS**11)/(8711563180320*mbkin**11) - 
            (204332*mcMS**11)/(135*mbkin**10*(mbkin - mcMS)) + 
            (2828644307790564073*mcMS**12)/(17423126360640*mbkin**12) + 
            (3903376*mcMS**12)/(6237*mbkin**11*(mbkin - mcMS)) - 
            (498341223538198393*mcMS**13)/(4494060370800*mbkin**13) - 
            (170476*mcMS**13)/(891*mbkin**12*(mbkin - mcMS)) + 
            (573303167239529591*mcMS**14)/(9437526778680*mbkin**14) + 
            (262384*mcMS**14)/(6435*mbkin**13*(mbkin - mcMS)) - 
            (276649915090469443*mcMS**15)/(10486140865200*mbkin**15) - 
            (2193556*mcMS**15)/(405405*mbkin**14*(mbkin - mcMS)) + 
            (12785968996764493*mcMS**16)/(1438099318656*mbkin**16) + 
            (19504*mcMS**16)/(57915*mbkin**15*(mbkin - mcMS)) - 
            (9696558349244617*mcMS**17)/(4321561083840*mbkin**17) + 
            (174697456199159021*mcMS**18)/(437558059738800*mbkin**18) - 
            (64106874851931161*mcMS**19)/(1434504070359360*mbkin**19) + 
            (81799475009197*mcMS**20)/(34550039002752*mbkin**20) - 
            (3820703038853*mcMS**2)/(20427547140*mbkin**2*(1 - mcMS/mbkin)**2) + 
            (1324623229305471443*mcMS**3)/(534793184125200*mbkin**3*
            (1 - mcMS/mbkin)**2) - (5514257196652051*mcMS**4)/
            (309307798800*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (2445534345846504631*mcMS**5)/(28147009690800*mbkin**5*
            (1 - mcMS/mbkin)**2) - (73546477272645373*mcMS**6)/
            (236529493200*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (70971441129844033*mcMS**7)/(82785322620*mbkin**7*(1 - mcMS/mbkin)**
                2) - (5292307230847823*mcMS**8)/(2815827300*mbkin**8*
            (1 - mcMS/mbkin)**2) + (1883705490346937*mcMS**9)/
            (563165460*mbkin**9*(1 - mcMS/mbkin)**2) - (3996921465647711*mcMS**10)/
            (816423300*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (34379222836281097*mcMS**11)/(5789183400*mbkin**11*(1 - mcMS/mbkin)**
                2) - (991151981540419*mcMS**12)/(165405240*mbkin**12*
            (1 - mcMS/mbkin)**2) + (2648132431926677*mcMS**13)/
            (526289400*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (223279079747087713*mcMS**14)/(63681017400*mbkin**14*
            (1 - mcMS/mbkin)**2) + (4765673910687463*mcMS**15)/
            (2365294932*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (55872539918685143*mcMS**16)/(59132373300*mbkin**16*
            (1 - mcMS/mbkin)**2) + (251792791028627*mcMS**17)/
            (707566860*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (2416054156353307*mcMS**18)/(22995922950*mbkin**18*(1 - mcMS/mbkin)**
                2) + (1436625073319083*mcMS**19)/(61322461200*mbkin**19*
            (1 - mcMS/mbkin)**2) - (20887342577910533*mcMS**20)/
            (5629401938160*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (198978903088724189*mcMS**21)/(534793184125200*mbkin**21*
            (1 - mcMS/mbkin)**2) - (24672660896281*mcMS**22)/
            (1389073205520*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (84769718158347181*mcMS)/(1477973890673280*mbkin*(1 - mcMS/mbkin)) - 
            (2446470123833473391*mcMS**2)/(4433921672019840*mbkin**2*
            (1 - mcMS/mbkin)) + (43176344676356419391*mcMS**3)/
            (12835036419004800*mbkin**3*(1 - mcMS/mbkin)) - 
            (4378795260555490441*mcMS**4)/(285223031533440*mbkin**4*
            (1 - mcMS/mbkin)) + (1825778755943637871*mcMS**5)/
            (34318279195200*mbkin**5*(1 - mcMS/mbkin)) - 
            (5409718168456526123*mcMS**6)/(37750107114720*mbkin**6*
            (1 - mcMS/mbkin)) + (2398306644621841*mcMS**7)/(7781922720*mbkin**7*
            (1 - mcMS/mbkin)) - (285450062438881289*mcMS**8)/
            (528712984800*mbkin**8*(1 - mcMS/mbkin)) + 
            (82990069185646877*mcMS**9)/(106368292800*mbkin**9*(1 - mcMS/mbkin)) - 
            (63406249636229341*mcMS**10)/(67688913600*mbkin**10*
            (1 - mcMS/mbkin)) + (494853577058200963*mcMS**11)/
            (527973526080*mbkin**11*(1 - mcMS/mbkin)) - 
            (412608449195796241*mcMS**12)/(527973526080*mbkin**12*
            (1 - mcMS/mbkin)) + (6615633306531721*mcMS**13)/
            (12221609400*mbkin**13*(1 - mcMS/mbkin)) - 
            (1192039271175325699*mcMS**14)/(3852051746400*mbkin**14*
            (1 - mcMS/mbkin)) + (1604043546702600131*mcMS**15)/
            (11102972680800*mbkin**15*(1 - mcMS/mbkin)) - 
            (4133080133491195*mcMS**16)/(76262842656*mbkin**16*(1 - mcMS/mbkin)) + 
            (267518569219007159*mcMS**17)/(16777825384320*mbkin**17*
            (1 - mcMS/mbkin)) - (65643525989913343*mcMS**18)/
            (18520976073600*mbkin**18*(1 - mcMS/mbkin)) + 
            (2394868598393790877*mcMS**19)/(4278345473001600*mbkin**19*
            (1 - mcMS/mbkin)) - (13653883931434631303*mcMS**20)/
            (243865691961091200*mbkin**20*(1 - mcMS/mbkin)) + 
            (844572094812157*mcMS**21)/(316708690858560*mbkin**21*
            (1 - mcMS/mbkin)) + (-16/(81*mbkin**3) + (1813*mcMS**2)/
                (9*mbkin**2) - (1660*mcMS**4)/(9*mbkin**6) - (6640*mcMS**4)/
                (27*mbkin**5) - (6923*mcMS**4)/(9*mbkin**4) - (5917*mcMS**6)/
                (3*mbkin**6) + (8833*mcMS**8)/(18*mbkin**8))*np.log(mcMS**2/mbkin**2) + 
            (3577*mcMS**4*np.log(mcMS**2/mbkin**2)**2)/(3*mbkin**4) + 
            (2093435917/104756652 - (776844436*mcMS)/(3357585*mbkin) + 
            (40836175*mcMS**2)/(34749*mbkin**2) - (11697739628*mcMS**3)/
                (2953665*mbkin**3) + (17282295109*mcMS**4)/(1621620*mbkin**4) - 
            (379688576*mcMS**5)/(15795*mbkin**5) + (710791064*mcMS**6)/
                (15795*mbkin**6) - (171522592*mcMS**7)/(2457*mbkin**7) + 
            (17059558777*mcMS**8)/(187110*mbkin**8) - (124023536*mcMS**9)/
                (1215*mbkin**9) + (40031806*mcMS**10)/(405*mbkin**10) - 
            (367940368*mcMS**11)/(4455*mbkin**11) + (314961001*mcMS**12)/
                (5346*mbkin**12) - (2045244112*mcMS**13)/(57915*mbkin**13) + 
            (1632457936*mcMS**14)/(93555*mbkin**14) - (241470128*mcMS**15)/
                (34749*mbkin**15) + (1014911*mcMS**16)/(468*mbkin**16) - 
            (27387700*mcMS**17)/(53703*mbkin**17) + (159684019*mcMS**18)/
                (1879605*mbkin**18) - (1967236*mcMS**19)/(220077*mbkin**19) + 
            (9117229*mcMS**20)/(20407140*mbkin**20) + (94132*mcMS**2)/
                (351*mbkin**2*(1 - mcMS/mbkin)**2) - (107374328*mcMS**3)/
                (29835*mbkin**3*(1 - mcMS/mbkin)**2) + (58636332254*mcMS**4)/
                (2297295*mbkin**4*(1 - mcMS/mbkin)**2) - (18835823740*mcMS**5)/
                (153153*mbkin**5*(1 - mcMS/mbkin)**2) + (19762010462*mcMS**6)/
                (45045*mbkin**6*(1 - mcMS/mbkin)**2) - (141654880*mcMS**7)/
                (117*mbkin**7*(1 - mcMS/mbkin)**2) + (51339217784*mcMS**8)/
                (19305*mbkin**8*(1 - mcMS/mbkin)**2) - (91511815312*mcMS**9)/
                (19305*mbkin**9*(1 - mcMS/mbkin)**2) + (10314018224*mcMS**10)/
                (1485*mbkin**10*(1 - mcMS/mbkin)**2) - (7966865776*mcMS**11)/
                (945*mbkin**11*(1 - mcMS/mbkin)**2) + (1608461692*mcMS**12)/
                (189*mbkin**12*(1 - mcMS/mbkin)**2) - (6755052712*mcMS**13)/
                (945*mbkin**13*(1 - mcMS/mbkin)**2) + (224193572*mcMS**14)/
                (45*mbkin**14*(1 - mcMS/mbkin)**2) - (18425910944*mcMS**15)/
                (6435*mbkin**15*(1 - mcMS/mbkin)**2) + (8642046184*mcMS**16)/
                (6435*mbkin**16*(1 - mcMS/mbkin)**2) - (1953048784*mcMS**17)/
                (3861*mbkin**17*(1 - mcMS/mbkin)**2) + (1834859876*mcMS**18)/
                (12285*mbkin**18*(1 - mcMS/mbkin)**2) - (4500823064*mcMS**19)/
                (135135*mbkin**19*(1 - mcMS/mbkin)**2) + (2423770534*mcMS**20)/
                (459459*mbkin**20*(1 - mcMS/mbkin)**2) - (173613532*mcMS**21)/
                (328185*mbkin**21*(1 - mcMS/mbkin)**2) + (150698*mcMS**22)/
                (5967*mbkin**22*(1 - mcMS/mbkin)**2) - (65306258*mcMS)/
                (793611*mbkin*(1 - mcMS/mbkin)) + (2164618955*mcMS**2)/
                (2380833*mbkin**2*(1 - mcMS/mbkin)) - (496606267*mcMS**3)/
                (89505*mbkin**3*(1 - mcMS/mbkin)) + (112032753509*mcMS**4)/
                (4594590*mbkin**4*(1 - mcMS/mbkin)) - (7442393083*mcMS**5)/
                (90090*mbkin**5*(1 - mcMS/mbkin)) + (77562508*mcMS**6)/
                (351*mbkin**6*(1 - mcMS/mbkin)) - (55558756*mcMS**7)/
                (117*mbkin**7*(1 - mcMS/mbkin)) + (48197120486*mcMS**8)/
                (57915*mbkin**8*(1 - mcMS/mbkin)) - (1071733786*mcMS**9)/
                (891*mbkin**9*(1 - mcMS/mbkin)) + (194929246*mcMS**10)/
                (135*mbkin**10*(1 - mcMS/mbkin)) - (818907902*mcMS**11)/
                (567*mbkin**11*(1 - mcMS/mbkin)) + (682554503*mcMS**12)/
                (567*mbkin**12*(1 - mcMS/mbkin)) - (112525843*mcMS**13)/
                (135*mbkin**13*(1 - mcMS/mbkin)) + (278668756*mcMS**14)/
                (585*mbkin**14*(1 - mcMS/mbkin)) - (30043424428*mcMS**15)/
                (135135*mbkin**15*(1 - mcMS/mbkin)) + (250382182*mcMS**16)/
                (3003*mbkin**16*(1 - mcMS/mbkin)) - (25824224*mcMS**17)/
                (1053*mbkin**17*(1 - mcMS/mbkin)) + (3414736253*mcMS**18)/
                (626535*mbkin**18*(1 - mcMS/mbkin)) - (1977052709*mcMS**19)/
                (2297295*mbkin**19*(1 - mcMS/mbkin)) + (3219914011*mcMS**20)/
                (37413090*mbkin**20*(1 - mcMS/mbkin)) - (2787913*mcMS**21)/
                (680238*mbkin**21*(1 - mcMS/mbkin)))*np.log(1 - mcMS/mbkin))*
            np.log(mus**2/mbkin**2)**2 + ((-1813*mcMS**2)/(27*mbkin**2) - 
            (7681*mcMS**4)/(18*mbkin**4) + (5917*mcMS**6)/(9*mbkin**6) - 
            (8833*mcMS**8)/(54*mbkin**8) - (3577*mcMS**4*np.log(mcMS**2/mbkin**2))/
            (9*mbkin**4))*np.log(mus**2/mbkin**2)**3 + 
        np.log(2)**2*(-612444982657/2917512675 - 39786064/(2606175*mbkin**2) - 
            159144256/(4691115*mbkin) + (3580064*mcMS)/(405405*mbkin**3) + 
            (114562048*mcMS)/(6081075*mbkin**2) + (163353094327534*mcMS)/
            (189638323875*mbkin) + (888615968*mcMS**2)/(2606175*mbkin**4) + 
            (1777231936*mcMS**2)/(2606175*mbkin**3) + (1432811618708*mcMS**2)/
            (2462835375*mbkin**2) - (1987324448*mcMS**3)/(1403325*mbkin**5) - 
            (31797191168*mcMS**3)/(12629925*mbkin**4) - (140992485674*mcMS**3)/
            (12629925*mbkin**3) + (2262297616*mcMS**4)/(841995*mbkin**6) + 
            (9049190464*mcMS**4)/(2525985*mbkin**5) + (435088432397*mcMS**4)/
            (12629925*mbkin**4) - (4045856*mcMS**5)/(1575*mbkin**7) - 
            (779808738796*mcMS**5)/(12629925*mbkin**5) - (1602644032*mcMS**6)/
            (164025*mbkin**7) + (5025434831264*mcMS**6)/(63149625*mbkin**6) + 
            (521545376*mcMS**7)/(127575*mbkin**9) + (8344726016*mcMS**7)/
            (382725*mbkin**8) - (36286946545468*mcMS**7)/(442047375*mbkin**7) - 
            (307974928*mcMS**8)/(42525*mbkin**10) - (1231899712*mcMS**8)/
            (42525*mbkin**9) + (569054051471*mcMS**8)/(8037225*mbkin**8) + 
            (5546336*mcMS**9)/(729*mbkin**11) + (177482752*mcMS**9)/
            (6561*mbkin**10) - (1917905297986*mcMS**9)/(37889775*mbkin**9) - 
            (141385696*mcMS**10)/(25515*mbkin**12) - (282771392*mcMS**10)/
            (15309*mbkin**11) + (1100982598124*mcMS**10)/(37889775*mbkin**10) + 
            (810886688*mcMS**11)/(280665*mbkin**13) + (12974187008*mcMS**11)/
            (1403325*mbkin**12) - (26961815278186*mcMS**11)/
            (2083937625*mbkin**11) - (1498143088*mcMS**12)/(1403325*mbkin**14) - 
            (5992572352*mcMS**12)/(1804275*mbkin**13) + (425485476871*mcMS**12)/
            (99235125*mbkin**12) + (2087832544*mcMS**13)/(7818525*mbkin**15) + 
            (19088754688*mcMS**13)/(23455575*mbkin**14) - (596523584*mcMS**13)/
            (601425*mbkin**13) - (16503104*mcMS**14)/(405405*mbkin**16) - 
            (16503104*mcMS**14)/(135135*mbkin**15) + (8251552*mcMS**14)/
            (57915*mbkin**14) + (17410912*mcMS**15)/(6081075*mbkin**17) + 
            (278574592*mcMS**15)/(32837805*mbkin**16) - (34821824*mcMS**15)/
            (3648645*mbkin**15) + (9872453363*np.pi**2)/35363790 - 
            (50654764162*mcMS*np.pi**2)/(17681895*mbkin) + (1745214652*mcMS**2*np.pi**2)/
            (127575*mbkin**2) - (322488358954*mcMS**3*np.pi**2)/(8037225*mbkin**3) + 
            (431082169079*mcMS**4*np.pi**2)/(5358150*mbkin**4) - 
            (11504598124*mcMS**5*np.pi**2)/(99225*mbkin**5) + 
            (47061573116*mcMS**6*np.pi**2)/(382725*mbkin**6) - 
            (258753888412*mcMS**7*np.pi**2)/(2679075*mbkin**7) + 
            (33155471029*mcMS**8*np.pi**2)/(595350*mbkin**8) - 
            (26373336058*mcMS**9*np.pi**2)/(1148175*mbkin**9) + 
            (51661091824*mcMS**10*np.pi**2)/(8037225*mbkin**10) - 
            (10749424454*mcMS**11*np.pi**2)/(9823275*mbkin**11) + 
            (3023674889*mcMS**12*np.pi**2)/(35363790*mbkin**12) - 
            (14120272*mcMS*np.pi**2)/(2380833*mbkin*(1 - mcMS/mbkin)) + 
            (468025720*mcMS**2*np.pi**2)/(7142499*mbkin**2*(1 - mcMS/mbkin)) - 
            (107374328*mcMS**3*np.pi**2)/(268515*mbkin**3*(1 - mcMS/mbkin)) + 
            (12111649028*mcMS**4*np.pi**2)/(6891885*mbkin**4*(1 - mcMS/mbkin)) - 
            (804583036*mcMS**5*np.pi**2)/(135135*mbkin**5*(1 - mcMS/mbkin)) + 
            (16770272*mcMS**6*np.pi**2)/(1053*mbkin**6*(1 - mcMS/mbkin)) - 
            (12012704*mcMS**7*np.pi**2)/(351*mbkin**7*(1 - mcMS/mbkin)) + 
            (10420999024*mcMS**8*np.pi**2)/(173745*mbkin**8*(1 - mcMS/mbkin)) - 
            (231726224*mcMS**9*np.pi**2)/(2673*mbkin**9*(1 - mcMS/mbkin)) + 
            (42146864*mcMS**10*np.pi**2)/(405*mbkin**10*(1 - mcMS/mbkin)) - 
            (177061168*mcMS**11*np.pi**2)/(1701*mbkin**11*(1 - mcMS/mbkin)) + 
            (147579352*mcMS**12*np.pi**2)/(1701*mbkin**12*(1 - mcMS/mbkin)) - 
            (24329912*mcMS**13*np.pi**2)/(405*mbkin**13*(1 - mcMS/mbkin)) + 
            (60252704*mcMS**14*np.pi**2)/(1755*mbkin**14*(1 - mcMS/mbkin)) - 
            (6495875552*mcMS**15*np.pi**2)/(405405*mbkin**15*(1 - mcMS/mbkin)) + 
            (54136688*mcMS**16*np.pi**2)/(9009*mbkin**16*(1 - mcMS/mbkin)) - 
            (5583616*mcMS**17*np.pi**2)/(3159*mbkin**17*(1 - mcMS/mbkin)) + 
            (738321352*mcMS**18*np.pi**2)/(1879605*mbkin**18*(1 - mcMS/mbkin)) - 
            (427470856*mcMS**19*np.pi**2)/(6891885*mbkin**19*(1 - mcMS/mbkin)) + 
            (348098812*mcMS**20*np.pi**2)/(56119635*mbkin**20*(1 - mcMS/mbkin)) - 
            (301396*mcMS**21*np.pi**2)/(1020357*mbkin**21*(1 - mcMS/mbkin)) + 
            ((-1790032*mcMS)/(675675*mbkin) - (222153992*mcMS**2)/
                (868725*mbkin**2) + (993662224*mcMS**3)/(467775*mbkin**3) - 
            (2262297616*mcMS**4)/(280665*mbkin**4) + (2022928*mcMS**5)/
                (105*mbkin**5) - (200330504*mcMS**6)/(6075*mbkin**6) + 
            (260772688*mcMS**7)/(6075*mbkin**7) - (615949856*mcMS**8)/
                (14175*mbkin**8) + (2773168*mcMS**9)/(81*mbkin**9) - 
            (35346424*mcMS**10)/(1701*mbkin**10) + (405443344*mcMS**11)/
                (42525*mbkin**11) - (1498143088*mcMS**12)/(467775*mbkin**12) + 
            (149130896*mcMS**13)/(200475*mbkin**13) - (2062888*mcMS**14)/
                (19305*mbkin**14) + (8705456*mcMS**15)/(1216215*mbkin**15) + 
            (256*mcMS**4*np.pi**2)/(27*mbkin**4))*np.log(mcMS**2/mbkin**2) + 
            (23931088/841995 - (14708576*mcMS)/(841995*mbkin) - 
            (8950336*mcMS**2)/(10935*mbkin**2) + (33425824*mcMS**3)/
                (8505*mbkin**3) - (233018864*mcMS**4)/(25515*mbkin**4) + 
            (67270976*mcMS**5)/(5103*mbkin**5) - (47392768*mcMS**6)/
                (3645*mbkin**6) + (235510208*mcMS**7)/(25515*mbkin**7) - 
            (123588112*mcMS**8)/(25515*mbkin**8) + (20426656*mcMS**9)/
                (10935*mbkin**9) - (38385728*mcMS**10)/(76545*mbkin**10) + 
            (69725024*mcMS**11)/(841995*mbkin**11) - (1775344*mcMS**12)/
                (280665*mbkin**12))*np.log(1 - mcMS/mbkin) + 
            ((1790032*mcMS)/(675675*mbkin) + (222153992*mcMS**2)/
                (868725*mbkin**2) - (993662224*mcMS**3)/(467775*mbkin**3) + 
            (2262297616*mcMS**4)/(280665*mbkin**4) - (2022928*mcMS**5)/
                (105*mbkin**5) + (200330504*mcMS**6)/(6075*mbkin**6) - 
            (260772688*mcMS**7)/(6075*mbkin**7) + (615949856*mcMS**8)/
                (14175*mbkin**8) - (2773168*mcMS**9)/(81*mbkin**9) + 
            (35346424*mcMS**10)/(1701*mbkin**10) - (405443344*mcMS**11)/
                (42525*mbkin**11) + (1498143088*mcMS**12)/(467775*mbkin**12) - 
            (149130896*mcMS**13)/(200475*mbkin**13) + (2062888*mcMS**14)/
                (19305*mbkin**14) - (8705456*mcMS**15)/(1216215*mbkin**15))*
            np.log(mu0**2/mus**2) + (-35409044/1658475 + (4925297456*mcMS)/
                (54729675*mbkin) + (757015612*mcMS**2)/(2606175*mbkin**2) - 
            (516183824*mcMS**3)/(168399*mbkin**3) + (1305304468*mcMS**4)/
                (120285*mbkin**4) - (1015014496*mcMS**5)/(42525*mbkin**5) + 
            (2086845388*mcMS**6)/(54675*mbkin**6) - (287782336*mcMS**7)/
                (6075*mbkin**7) + (18845788*mcMS**8)/(405*mbkin**8) - 
            (43741648*mcMS**9)/(1215*mbkin**9) + (1654009132*mcMS**10)/
                (76545*mbkin**10) - (179441648*mcMS**11)/(18225*mbkin**11) + 
            (13865219444*mcMS**12)/(4209975*mbkin**12) - (1193047168*mcMS**13)/
                (1563705*mbkin**13) + (44352092*mcMS**14)/(405405*mbkin**14) - 
            (400450976*mcMS**15)/(54729675*mbkin**15))*np.log(mus**2/mbkin**2)) + 
        np.log(mu0**2/mus**2)**2*(400/(27*mbkin**3) - (64*mcMS**2)/(81*mbkin**5) + 
            (16*mcMS**2)/(9*mbkin**4) + (32*mcMS**2)/(9*mbkin**3) + 
            (184687225*mcMS**2)/(1587222*mbkin**2) - (383304088*mcMS**3)/
            (793611*mbkin**3) + (736*mcMS**4)/(27*mbkin**7) - 
            (100*mcMS**4)/(3*mbkin**6) + (112*mcMS**4)/(3*mbkin**5) + 
            (30666059311*mcMS**4)/(15872220*mbkin**4) - (34160890396*mcMS**5)/
            (3357585*mbkin**5) - (3008*mcMS**6)/(27*mbkin**9) - 
            (1504*mcMS**6)/(9*mbkin**7) + (2753609854939*mcMS**6)/
            (87297210*mbkin**6) - (3136021172428*mcMS**7)/(43648605*mbkin**7) + 
            (5680*mcMS**8)/(81*mbkin**11) + (284*mcMS**8)/(9*mbkin**10) + 
            (1136*mcMS**8)/(9*mbkin**9) + (7761634148461*mcMS**8)/
            (58198140*mbkin**8) - (2919038193364*mcMS**9)/(14549535*mbkin**9) + 
            (3569014609868*mcMS**10)/(14549535*mbkin**10) - 
            (1189994365324*mcMS**11)/(4849845*mbkin**11) + (1252064148332*mcMS**12)/
            (6235515*mbkin**12) - (834838011868*mcMS**13)/(6235515*mbkin**13) + 
            (449582744036*mcMS**14)/(6235515*mbkin**14) - (192697215868*mcMS**15)/
            (6235515*mbkin**15) + (449663179484*mcMS**16)/(43648605*mbkin**16) - 
            (112423410196*mcMS**17)/(43648605*mbkin**17) + (19840567804*mcMS**18)/
            (43648605*mbkin**18) - (314945252*mcMS**19)/(6235515*mbkin**19) + 
            (301396*mcMS**20)/(113373*mbkin**20) - (3820703038853*mcMS**2)/
            (20427547140*mbkin**2*(1 - mcMS/mbkin)**2) + 
            (1324623229305471443*mcMS**3)/(534793184125200*mbkin**3*
            (1 - mcMS/mbkin)**2) - (5514257196652051*mcMS**4)/
            (309307798800*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (2445534345846504631*mcMS**5)/(28147009690800*mbkin**5*
            (1 - mcMS/mbkin)**2) - (73546477272645373*mcMS**6)/
            (236529493200*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (70971441129844033*mcMS**7)/(82785322620*mbkin**7*(1 - mcMS/mbkin)**
                2) - (5292307230847823*mcMS**8)/(2815827300*mbkin**8*
            (1 - mcMS/mbkin)**2) + (1883705490346937*mcMS**9)/
            (563165460*mbkin**9*(1 - mcMS/mbkin)**2) - (3996921465647711*mcMS**10)/
            (816423300*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (34379222836281097*mcMS**11)/(5789183400*mbkin**11*(1 - mcMS/mbkin)**
                2) - (991151981540419*mcMS**12)/(165405240*mbkin**12*
            (1 - mcMS/mbkin)**2) + (2648132431926677*mcMS**13)/
            (526289400*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (223279079747087713*mcMS**14)/(63681017400*mbkin**14*
            (1 - mcMS/mbkin)**2) + (4765673910687463*mcMS**15)/
            (2365294932*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (55872539918685143*mcMS**16)/(59132373300*mbkin**16*
            (1 - mcMS/mbkin)**2) + (251792791028627*mcMS**17)/
            (707566860*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (2416054156353307*mcMS**18)/(22995922950*mbkin**18*(1 - mcMS/mbkin)**
                2) + (1436625073319083*mcMS**19)/(61322461200*mbkin**19*
            (1 - mcMS/mbkin)**2) - (20887342577910533*mcMS**20)/
            (5629401938160*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (198978903088724189*mcMS**21)/(534793184125200*mbkin**21*
            (1 - mcMS/mbkin)**2) - (24672660896281*mcMS**22)/
            (1389073205520*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (2291073463739113*mcMS)/(113690299282560*mbkin*(1 - mcMS/mbkin)) + 
            (66120814157661443*mcMS**2)/(341070897847680*mbkin**2*
            (1 - mcMS/mbkin)) - (1166928234496119443*mcMS**3)/
            (987310493769600*mbkin**3*(1 - mcMS/mbkin)) + 
            (118345817852851093*mcMS**4)/(21940233194880*mbkin**4*
            (1 - mcMS/mbkin)) - (49345371782260483*mcMS**5)/
            (2639867630400*mbkin**5*(1 - mcMS/mbkin)) + 
            (146208599147473679*mcMS**6)/(2903854393440*mbkin**6*
            (1 - mcMS/mbkin)) - (64819098503293*mcMS**7)/(598609440*mbkin**7*
            (1 - mcMS/mbkin)) + (7714866552402197*mcMS**8)/(40670229600*mbkin**8*
            (1 - mcMS/mbkin)) - (29158672957119173*mcMS**9)/
            (106368292800*mbkin**9*(1 - mcMS/mbkin)) + 
            (22277871493810309*mcMS**10)/(67688913600*mbkin**10*
            (1 - mcMS/mbkin)) - (13374421001572999*mcMS**11)/
            (40613348160*mbkin**11*(1 - mcMS/mbkin)) + 
            (11151579707994493*mcMS**12)/(40613348160*mbkin**12*
            (1 - mcMS/mbkin)) - (178800900176533*mcMS**13)/(940123800*mbkin**13*
            (1 - mcMS/mbkin)) + (32217277599333127*mcMS**14)/
            (296311672800*mbkin**14*(1 - mcMS/mbkin)) - 
            (43352528289259463*mcMS**15)/(854074821600*mbkin**15*
            (1 - mcMS/mbkin)) + (111704868472735*mcMS**16)/(5866372512*mbkin**16*
            (1 - mcMS/mbkin)) - (7230231600513707*mcMS**17)/
            (1290601952640*mbkin**17*(1 - mcMS/mbkin)) + 
            (1774149351078739*mcMS**18)/(1424690467200*mbkin**18*
            (1 - mcMS/mbkin)) - (64726178334967321*mcMS**19)/
            (329103497923200*mbkin**19*(1 - mcMS/mbkin)) + 
            (369023890038773819*mcMS**20)/(18758899381622400*mbkin**20*
            (1 - mcMS/mbkin)) - (22826272832761*mcMS**21)/(24362206989120*
            mbkin**21*(1 - mcMS/mbkin)) + ((-4*mcMS**2)/(3*mbkin**2) - 
            (184*mcMS**4)/(3*mbkin**6) - (736*mcMS**4)/(9*mbkin**5) + 
            (2801*mcMS**4)/(18*mbkin**4) - (564*mcMS**6)/mbkin**6 + 
            (568*mcMS**8)/(3*mbkin**8))*np.log(mcMS**2/mbkin**2) + 
            (184*mcMS**4*np.log(mcMS**2/mbkin**2)**2)/mbkin**4 + 
            ((94132*mcMS**2)/(351*mbkin**2*(1 - mcMS/mbkin)**2) - 
            (107374328*mcMS**3)/(29835*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (58636332254*mcMS**4)/(2297295*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (18835823740*mcMS**5)/(153153*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (19762010462*mcMS**6)/(45045*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (141654880*mcMS**7)/(117*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (51339217784*mcMS**8)/(19305*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (91511815312*mcMS**9)/(19305*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (10314018224*mcMS**10)/(1485*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (7966865776*mcMS**11)/(945*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (1608461692*mcMS**12)/(189*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (6755052712*mcMS**13)/(945*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (224193572*mcMS**14)/(45*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (18425910944*mcMS**15)/(6435*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (8642046184*mcMS**16)/(6435*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (1953048784*mcMS**17)/(3861*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (1834859876*mcMS**18)/(12285*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (4500823064*mcMS**19)/(135135*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (2423770534*mcMS**20)/(459459*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (173613532*mcMS**21)/(328185*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (150698*mcMS**22)/(5967*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (1765034*mcMS)/(61047*mbkin*(1 - mcMS/mbkin)) - 
            (58503215*mcMS**2)/(183141*mbkin**2*(1 - mcMS/mbkin)) + 
            (13421791*mcMS**3)/(6885*mbkin**3*(1 - mcMS/mbkin)) - 
            (3027912257*mcMS**4)/(353430*mbkin**4*(1 - mcMS/mbkin)) + 
            (201145759*mcMS**5)/(6930*mbkin**5*(1 - mcMS/mbkin)) - 
            (2096284*mcMS**6)/(27*mbkin**6*(1 - mcMS/mbkin)) + 
            (1501588*mcMS**7)/(9*mbkin**7*(1 - mcMS/mbkin)) - 
            (1302624878*mcMS**8)/(4455*mbkin**8*(1 - mcMS/mbkin)) + 
            (376555114*mcMS**9)/(891*mbkin**9*(1 - mcMS/mbkin)) - 
            (68488654*mcMS**10)/(135*mbkin**10*(1 - mcMS/mbkin)) + 
            (287724398*mcMS**11)/(567*mbkin**11*(1 - mcMS/mbkin)) - 
            (239816447*mcMS**12)/(567*mbkin**12*(1 - mcMS/mbkin)) + 
            (39536107*mcMS**13)/(135*mbkin**13*(1 - mcMS/mbkin)) - 
            (7531588*mcMS**14)/(45*mbkin**14*(1 - mcMS/mbkin)) + 
            (811984444*mcMS**15)/(10395*mbkin**15*(1 - mcMS/mbkin)) - 
            (6767086*mcMS**16)/(231*mbkin**16*(1 - mcMS/mbkin)) + 
            (697952*mcMS**17)/(81*mbkin**17*(1 - mcMS/mbkin)) - 
            (92290169*mcMS**18)/(48195*mbkin**18*(1 - mcMS/mbkin)) + 
            (53433857*mcMS**19)/(176715*mbkin**19*(1 - mcMS/mbkin)) - 
            (87024703*mcMS**20)/(2877930*mbkin**20*(1 - mcMS/mbkin)) + 
            (75349*mcMS**21)/(52326*mbkin**21*(1 - mcMS/mbkin)))*
            np.log(1 - mcMS/mbkin) + ((4*mcMS**2)/(3*mbkin**2) - 
            (376*mcMS**4)/mbkin**4 + (564*mcMS**6)/mbkin**6 - 
            (568*mcMS**8)/(3*mbkin**8) - (184*mcMS**4*np.log(mcMS**2/mbkin**2))/
                mbkin**4)*np.log(mus**2/mbkin**2)) + np.log(2)*(270781.51583294815 + 
            338587652/(10072755*mbkin**4) + 4792317536/(30218265*mbkin**3) + 
            42887214162286888/(53086088130075*mbkin**2) + 143017147230146764/
            (95554958634135*mbkin) - (56481088*mcMS)/(340119*mbkin**5) - 
            (1807394816*mcMS)/(2380833*mbkin**4) - (310449803443917791*mcMS)/
            (53086088130075*mbkin**3) - (9521780371527925408*mcMS)/
            (796291321951125*mbkin**2) - (10767.745948118347*mcMS)/mbkin + 
            (376528*mcMS**2)/(3159*mbkin**6) + (3012224*mcMS**2)/(9477*mbkin**5) + 
            (236347688805556*mcMS**2)/(10672286625*mbkin**4) + 
            (80933419563736*mcMS**2)/(1524612375*mbkin**3) + 
            (32579.81524008369*mcMS**2)/mbkin**2 + (1504064*mcMS**3)/
            (663*mbkin**7) + (24065024*mcMS**3)/(1989*mbkin**6) - 
            (67140833722675667*mcMS**3)/(1074617168625*mbkin**5) - 
            (1834962771873232112*mcMS**3)/(9671554517625*mbkin**4) - 
            (62419.16538300422*mcMS**3)/mbkin**3 - (574496308*mcMS**4)/
            (36855*mbkin**8) - (4595970464*mcMS**4)/(57915*mbkin**7) + 
            (5451292868690296*mcMS**4)/(37927664775*mbkin**6) + 
            (67431499870061452*mcMS**4)/(113782994325*mbkin**5) + 
            (102968.6732429856*mcMS**4)/mbkin**4 + (64971008*mcMS**5)/
            (1053*mbkin**9) + (978386944*mcMS**5)/(3159*mbkin**8) - 
            (43027179107*mcMS**5)/(165375*mbkin**7) - (28651812118912*mcMS**5)/
            (18243225*mbkin**6) - (175330.469026736*mcMS**5)/mbkin**5 - 
            (183892544*mcMS**6)/(1053*mbkin**10) - (2750392832*mcMS**6)/
            (3159*mbkin**9) + (6251242037792*mcMS**6)/(18243225*mbkin**8) + 
            (25486894372646296*mcMS**6)/(7388506125*mbkin**7) + 
            (274814.73178371985*mcMS**6)/mbkin**6 + (103304960*mcMS**7)/
            (273*mbkin**11) + (170987520*mcMS**7)/(91*mbkin**10) - 
            (1528624525107313*mcMS**7)/(5746615875*mbkin**9) - 
            (107969763387927568*mcMS**7)/(17239847625*mbkin**8) - 
            (344959.21727236494*mcMS**7)/mbkin**7 - (1737495368*mcMS**8)/
            (2673*mbkin**12) - (25814216896*mcMS**8)/(8019*mbkin**11) - 
            (38836619987162*mcMS**8)/(638512875*mbkin**10) + 
            (54207343839673328*mcMS**8)/(5746615875*mbkin**9) + 
            (335759.41387939523*mcMS**8)/mbkin**8 + (1096984192*mcMS**9)/
            (1215*mbkin**13) + (16267472896*mcMS**9)/(3645*mbkin**12) + 
            (13635195701351*mcMS**9)/(23455575*mbkin**11) - 
            (17467877286240224*mcMS**9)/(1477701225*mbkin**10) - 
            (256666.41833722152*mcMS**9)/mbkin**9 - (415105504*mcMS**10)/
            (405*mbkin**14) - (2049031424*mcMS**10)/(405*mbkin**13) - 
            (6185635194938764*mcMS**10)/(5746615875*mbkin**12) + 
            (3272208865215496*mcMS**10)/(265228425*mbkin**11) + 
            (155910.77884184313*mcMS**10)/mbkin**10 + (17876864384*mcMS**11)/
            (18711*mbkin**15) + (37777524736*mcMS**11)/(8019*mbkin**14) + 
            (16492777551145283*mcMS**11)/(12642554925*mbkin**13) - 
            (677191509617510224*mcMS**11)/(63212774625*mbkin**12) - 
            (75364.08237712669*mcMS**11)/mbkin**11 - (6582040*mcMS**12)/
            (9*mbkin**16) - (97280320*mcMS**12)/(27*mbkin**15) - 
            (74839125943357106*mcMS**12)/(63212774625*mbkin**14) + 
            (626890258336895296*mcMS**12)/(81273567375*mbkin**13) + 
            (29205.97924280903*mcMS**12)/mbkin**12 + (37077248*mcMS**13)/
            (81*mbkin**17) + (182534144*mcMS**13)/(81*mbkin**16) + 
            (2435254597480993*mcMS**13)/(2910623625*mbkin**15) - 
            (439487617846053184*mcMS**13)/(96050579625*mbkin**14) - 
            (23348306427538*mcMS**13)/(2462835375*mbkin**13) - 
            (8556501952*mcMS**14)/(36855*mbkin**18) - (126298789376*mcMS**14)/
            (110565*mbkin**17) - (25479574691032376*mcMS**14)/
            (54784404675*mbkin**16) + (363735726365229992*mcMS**14)/
            (164353214025*mbkin**15) + (3175998276118*mcMS**14)/
            (1118049075*mbkin**14) + (164985088*mcMS**15)/(1755*mbkin**19) + 
            (187423059968*mcMS**15)/(405405*mbkin**18) + 
            (55426302030929857*mcMS**15)/(273922023375*mbkin**17) - 
            (254139874445295152*mcMS**15)/(295835785245*mbkin**16) - 
            (139299712876378*mcMS**15)/(164353214025*mbkin**15) - 
            (3473716*mcMS**16)/(117*mbkin**20) - (5691872*mcMS**16)/(39*mbkin**19) - 
            (71294882*mcMS**16)/(1053*mbkin**18) + (91697732*mcMS**16)/
            (351*mbkin**17) + (73241*mcMS**16)/(312*mbkin**16) + 
            (378641600*mcMS**17)/(53703*mbkin**21) + (5581772800*mcMS**17)/
            (161109*mbkin**20) + (2716434400*mcMS**17)/(161109*mbkin**19) - 
            (3220580800*mcMS**17)/(53703*mbkin**18) - (930650*mcMS**17)/
            (17901*mbkin**17) - (445357264*mcMS**18)/(375921*mbkin**22) - 
            (937594240*mcMS**18)/(161109*mbkin**21) - (9973658728*mcMS**18)/
            (3383289*mbkin**20) + (55107101456*mcMS**18)/(5638815*mbkin**19) + 
            (1464991*mcMS**18)/(179010*mbkin**18) + (138219712*mcMS**19)/
            (1100385*mbkin**23) + (678781952*mcMS**19)/(1100385*mbkin**22) + 
            (3201633824*mcMS**19)/(9903465*mbkin**21) - (23264704*mcMS**19)/
            (23085*mbkin**20) - (299362*mcMS**19)/(366795*mbkin**19) - 
            (32249372*mcMS**20)/(5101785*mbkin**24) - (475000096*mcMS**20)/
            (15305355*mbkin**23) - (1054886*mcMS**20)/(62985*mbkin**22) + 
            (9343276*mcMS**20)/(188955*mbkin**21) + (527443*mcMS**20)/
            (13604760*mbkin**20) + (376528*mcMS**2)/(3159*mbkin**6*
            (1 - mcMS/mbkin)**2) + (6024448*mcMS**2)/(9477*mbkin**5*
            (1 - mcMS/mbkin)**2) + (10542784*mcMS**2)/(28431*mbkin**4*
            (1 - mcMS/mbkin)**2) - (12048896*mcMS**2)/(9477*mbkin**3*
            (1 - mcMS/mbkin)**2) + (1506112*mcMS**2)/(3159*mbkin**2*
            (1 - mcMS/mbkin)**2) - (429497312*mcMS**3)/(268515*mbkin**7*
            (1 - mcMS/mbkin)**2) - (6871956992*mcMS**3)/(805545*mbkin**6*
            (1 - mcMS/mbkin)**2) - (12025924736*mcMS**3)/(2416635*mbkin**5*
            (1 - mcMS/mbkin)**2) + (13743913984*mcMS**3)/(805545*mbkin**4*
            (1 - mcMS/mbkin)**2) - (1717989248*mcMS**3)/(268515*mbkin**3*
            (1 - mcMS/mbkin)**2) + (234545329016*mcMS**4)/(20675655*mbkin**8*
            (1 - mcMS/mbkin)**2) + (3752725264256*mcMS**4)/(62026965*mbkin**7*
            (1 - mcMS/mbkin)**2) + (938181316064*mcMS**4)/(26582985*mbkin**6*
            (1 - mcMS/mbkin)**2) - (7505450528512*mcMS**4)/(62026965*mbkin**5*
            (1 - mcMS/mbkin)**2) + (938181316064*mcMS**4)/(20675655*mbkin**4*
            (1 - mcMS/mbkin)**2) - (75343294960*mcMS**5)/(1378377*mbkin**9*
            (1 - mcMS/mbkin)**2) - (1205492719360*mcMS**5)/(4135131*mbkin**8*
            (1 - mcMS/mbkin)**2) - (301373179840*mcMS**5)/(1772199*mbkin**7*
            (1 - mcMS/mbkin)**2) + (2410985438720*mcMS**5)/(4135131*mbkin**6*
            (1 - mcMS/mbkin)**2) - (301373179840*mcMS**5)/(1378377*mbkin**5*
            (1 - mcMS/mbkin)**2) + (79048041848*mcMS**6)/(405405*mbkin**10*
            (1 - mcMS/mbkin)**2) + (1264768669568*mcMS**6)/(1216215*mbkin**9*
            (1 - mcMS/mbkin)**2) + (316192167392*mcMS**6)/(521235*mbkin**8*
            (1 - mcMS/mbkin)**2) - (2529537339136*mcMS**6)/(1216215*mbkin**7*
            (1 - mcMS/mbkin)**2) + (316192167392*mcMS**6)/(405405*mbkin**6*
            (1 - mcMS/mbkin)**2) - (566619520*mcMS**7)/(1053*mbkin**11*
            (1 - mcMS/mbkin)**2) - (9065912320*mcMS**7)/(3159*mbkin**10*
            (1 - mcMS/mbkin)**2) - (15865346560*mcMS**7)/(9477*mbkin**9*
            (1 - mcMS/mbkin)**2) + (18131824640*mcMS**7)/(3159*mbkin**8*
            (1 - mcMS/mbkin)**2) - (2266478080*mcMS**7)/(1053*mbkin**7*
            (1 - mcMS/mbkin)**2) + (205356871136*mcMS**8)/(173745*mbkin**12*
            (1 - mcMS/mbkin)**2) + (3285709938176*mcMS**8)/(521235*mbkin**11*
            (1 - mcMS/mbkin)**2) + (5749992391808*mcMS**8)/(1563705*mbkin**10*
            (1 - mcMS/mbkin)**2) - (6571419876352*mcMS**8)/(521235*mbkin**9*
            (1 - mcMS/mbkin)**2) + (821427484544*mcMS**8)/(173745*mbkin**8*
            (1 - mcMS/mbkin)**2) - (366047261248*mcMS**9)/(173745*mbkin**13*
            (1 - mcMS/mbkin)**2) - (5856756179968*mcMS**9)/(521235*mbkin**12*
            (1 - mcMS/mbkin)**2) - (10249323314944*mcMS**9)/(1563705*mbkin**11*
            (1 - mcMS/mbkin)**2) + (11713512359936*mcMS**9)/
            (521235*mbkin**10*(1 - mcMS/mbkin)**2) - (1464189044992*mcMS**9)/
            (173745*mbkin**9*(1 - mcMS/mbkin)**2) + (41256072896*mcMS**10)/
            (13365*mbkin**14*(1 - mcMS/mbkin)**2) + (660097166336*mcMS**10)/
            (40095*mbkin**13*(1 - mcMS/mbkin)**2) + (1155170041088*mcMS**10)/
            (120285*mbkin**12*(1 - mcMS/mbkin)**2) - (1320194332672*mcMS**10)/
            (40095*mbkin**11*(1 - mcMS/mbkin)**2) + (165024291584*mcMS**10)/
            (13365*mbkin**10*(1 - mcMS/mbkin)**2) - (31867463104*mcMS**11)/
            (8505*mbkin**15*(1 - mcMS/mbkin)**2) - (509879409664*mcMS**11)/
            (25515*mbkin**14*(1 - mcMS/mbkin)**2) - (127469852416*mcMS**11)/
            (10935*mbkin**13*(1 - mcMS/mbkin)**2) + (1019758819328*mcMS**11)/
            (25515*mbkin**12*(1 - mcMS/mbkin)**2) - (127469852416*mcMS**11)/
            (8505*mbkin**11*(1 - mcMS/mbkin)**2) + (6433846768*mcMS**12)/
            (1701*mbkin**16*(1 - mcMS/mbkin)**2) + (102941548288*mcMS**12)/
            (5103*mbkin**15*(1 - mcMS/mbkin)**2) + (25735387072*mcMS**12)/
            (2187*mbkin**14*(1 - mcMS/mbkin)**2) - (205883096576*mcMS**12)/
            (5103*mbkin**13*(1 - mcMS/mbkin)**2) + (25735387072*mcMS**12)/
            (1701*mbkin**12*(1 - mcMS/mbkin)**2) - (27020210848*mcMS**13)/
            (8505*mbkin**17*(1 - mcMS/mbkin)**2) - (432323373568*mcMS**13)/
            (25515*mbkin**16*(1 - mcMS/mbkin)**2) - (108080843392*mcMS**13)/
            (10935*mbkin**15*(1 - mcMS/mbkin)**2) + (864646747136*mcMS**13)/
            (25515*mbkin**14*(1 - mcMS/mbkin)**2) - (108080843392*mcMS**13)/
            (8505*mbkin**13*(1 - mcMS/mbkin)**2) + (896774288*mcMS**14)/
            (405*mbkin**18*(1 - mcMS/mbkin)**2) + (14348388608*mcMS**14)/
            (1215*mbkin**17*(1 - mcMS/mbkin)**2) + (25109680064*mcMS**14)/
            (3645*mbkin**16*(1 - mcMS/mbkin)**2) - (28696777216*mcMS**14)/
            (1215*mbkin**15*(1 - mcMS/mbkin)**2) + (3587097152*mcMS**14)/
            (405*mbkin**14*(1 - mcMS/mbkin)**2) - (73703643776*mcMS**15)/
            (57915*mbkin**19*(1 - mcMS/mbkin)**2) - (1179258300416*mcMS**15)/
            (173745*mbkin**18*(1 - mcMS/mbkin)**2) - (2063702025728*mcMS**15)/
            (521235*mbkin**17*(1 - mcMS/mbkin)**2) + (2358516600832*mcMS**15)/
            (173745*mbkin**16*(1 - mcMS/mbkin)**2) - (294814575104*mcMS**15)/
            (57915*mbkin**15*(1 - mcMS/mbkin)**2) + (34568184736*mcMS**16)/
            (57915*mbkin**20*(1 - mcMS/mbkin)**2) + (553090955776*mcMS**16)/
            (173745*mbkin**19*(1 - mcMS/mbkin)**2) + (967909172608*mcMS**16)/
            (521235*mbkin**18*(1 - mcMS/mbkin)**2) - (1106181911552*mcMS**16)/
            (173745*mbkin**17*(1 - mcMS/mbkin)**2) + (138272738944*mcMS**16)/
            (57915*mbkin**16*(1 - mcMS/mbkin)**2) - (7812195136*mcMS**17)/
            (34749*mbkin**21*(1 - mcMS/mbkin)**2) - (124995122176*mcMS**17)/
            (104247*mbkin**20*(1 - mcMS/mbkin)**2) - (218741463808*mcMS**17)/
            (312741*mbkin**19*(1 - mcMS/mbkin)**2) + (249990244352*mcMS**17)/
            (104247*mbkin**18*(1 - mcMS/mbkin)**2) - (31248780544*mcMS**17)/
            (34749*mbkin**17*(1 - mcMS/mbkin)**2) + (7339439504*mcMS**18)/
            (110565*mbkin**22*(1 - mcMS/mbkin)**2) + (117431032064*mcMS**18)/
            (331695*mbkin**21*(1 - mcMS/mbkin)**2) + (29357758016*mcMS**18)/
            (142155*mbkin**20*(1 - mcMS/mbkin)**2) - (234862064128*mcMS**18)/
            (331695*mbkin**19*(1 - mcMS/mbkin)**2) + (29357758016*mcMS**18)/
            (110565*mbkin**18*(1 - mcMS/mbkin)**2) - (18003292256*mcMS**19)/
            (1216215*mbkin**23*(1 - mcMS/mbkin)**2) - (288052676096*mcMS**19)/
            (3648645*mbkin**22*(1 - mcMS/mbkin)**2) - (72013169024*mcMS**19)/
            (1563705*mbkin**21*(1 - mcMS/mbkin)**2) + (576105352192*mcMS**19)/
            (3648645*mbkin**20*(1 - mcMS/mbkin)**2) - (72013169024*mcMS**19)/
            (1216215*mbkin**19*(1 - mcMS/mbkin)**2) + (9695082136*mcMS**20)/
            (4135131*mbkin**24*(1 - mcMS/mbkin)**2) + (155121314176*mcMS**20)/
            (12405393*mbkin**23*(1 - mcMS/mbkin)**2) + (38780328544*mcMS**20)/
            (5316597*mbkin**22*(1 - mcMS/mbkin)**2) - (310242628352*mcMS**20)/
            (12405393*mbkin**21*(1 - mcMS/mbkin)**2) + (38780328544*mcMS**20)/
            (4135131*mbkin**20*(1 - mcMS/mbkin)**2) - (694454128*mcMS**21)/
            (2953665*mbkin**25*(1 - mcMS/mbkin)**2) - (11111266048*mcMS**21)/
            (8860995*mbkin**24*(1 - mcMS/mbkin)**2) - (19444715584*mcMS**21)/
            (26582985*mbkin**23*(1 - mcMS/mbkin)**2) + (22222532096*mcMS**21)/
            (8860995*mbkin**22*(1 - mcMS/mbkin)**2) - (2777816512*mcMS**21)/
            (2953665*mbkin**21*(1 - mcMS/mbkin)**2) + (602792*mcMS**22)/
            (53703*mbkin**26*(1 - mcMS/mbkin)**2) + (9644672*mcMS**22)/
            (161109*mbkin**25*(1 - mcMS/mbkin)**2) + (16878176*mcMS**22)/
            (483327*mbkin**24*(1 - mcMS/mbkin)**2) - (19289344*mcMS**22)/
            (161109*mbkin**23*(1 - mcMS/mbkin)**2) + (2411168*mcMS**22)/
            (53703*mbkin**22*(1 - mcMS/mbkin)**2) - (56481088*mcMS)/
            (2380833*mbkin**5*(1 - mcMS/mbkin)) - (903697408*mcMS)/
            (7142499*mbkin**4*(1 - mcMS/mbkin)) + (1044900128*mcMS)/
            (3061071*mbkin**3*(1 - mcMS/mbkin)) + (11239736512*mcMS)/
            (7142499*mbkin**2*(1 - mcMS/mbkin)) - (422.44082590079324*mcMS)/
            (mbkin*(1 - mcMS/mbkin)) + (1872102880*mcMS**2)/
            (7142499*mbkin**6*(1 - mcMS/mbkin)) + (29953646080*mcMS**2)/
            (21427497*mbkin**5*(1 - mcMS/mbkin)) - (34633903280*mcMS**2)/
            (9183213*mbkin**4*(1 - mcMS/mbkin)) - (372548473120*mcMS**2)/
            (21427497*mbkin**3*(1 - mcMS/mbkin)) + (4667.359847355476*mcMS**2)/
            (mbkin**2*(1 - mcMS/mbkin)) - (429497312*mcMS**3)/
            (268515*mbkin**7*(1 - mcMS/mbkin)) - (6871956992*mcMS**3)/
            (805545*mbkin**6*(1 - mcMS/mbkin)) + (55619901904*mcMS**3)/
            (2416635*mbkin**5*(1 - mcMS/mbkin)) + (85469965088*mcMS**3)/
            (805545*mbkin**4*(1 - mcMS/mbkin)) - (28482.86432213551*mcMS**3)/
            (mbkin**3*(1 - mcMS/mbkin)) + (48446596112*mcMS**4)/
            (6891885*mbkin**8*(1 - mcMS/mbkin)) + (775145537792*mcMS**4)/
            (20675655*mbkin**7*(1 - mcMS/mbkin)) - (896262028072*mcMS**4)/
            (8860995*mbkin**6*(1 - mcMS/mbkin)) - (9640872626288*mcMS**4)/
            (20675655*mbkin**5*(1 - mcMS/mbkin)) + (125174.82968621714*mcMS**4)/
            (mbkin**4*(1 - mcMS/mbkin)) - (3218332144*mcMS**5)/
            (135135*mbkin**9*(1 - mcMS/mbkin)) - (51493314304*mcMS**5)/
            (405405*mbkin**8*(1 - mcMS/mbkin)) + (59539144664*mcMS**5)/
            (173745*mbkin**7*(1 - mcMS/mbkin)) + (640448096656*mcMS**5)/
            (405405*mbkin**6*(1 - mcMS/mbkin)) - (424086.8239833621*mcMS**5)/
            (mbkin**5*(1 - mcMS/mbkin)) + (67081088*mcMS**6)/
            (1053*mbkin**10*(1 - mcMS/mbkin)) + (1073297408*mcMS**6)/
            (3159*mbkin**9*(1 - mcMS/mbkin)) - (8687000896*mcMS**6)/
            (9477*mbkin**8*(1 - mcMS/mbkin)) - (13349136512*mcMS**6)/
            (3159*mbkin**7*(1 - mcMS/mbkin)) + (1.1343928728981994e6*mcMS**6)/
            (mbkin**6*(1 - mcMS/mbkin)) - (48050816*mcMS**7)/
            (351*mbkin**11*(1 - mcMS/mbkin)) - (768813056*mcMS**7)/
            (1053*mbkin**10*(1 - mcMS/mbkin)) + (6222580672*mcMS**7)/
            (3159*mbkin**9*(1 - mcMS/mbkin)) + (9562112384*mcMS**7)/
            (1053*mbkin**8*(1 - mcMS/mbkin)) - (2.4377289411589196e6*mcMS**7)/
            (mbkin**7*(1 - mcMS/mbkin)) + (41683996096*mcMS**8)/
            (173745*mbkin**12*(1 - mcMS/mbkin)) + (666943937536*mcMS**8)/
            (521235*mbkin**11*(1 - mcMS/mbkin)) - (5398077494432*mcMS**8)/
            (1563705*mbkin**10*(1 - mcMS/mbkin)) - (8295115223104*mcMS**8)/
            (521235*mbkin**9*(1 - mcMS/mbkin)) + (4.272172633742393e6*mcMS**8)/
            (mbkin**8*(1 - mcMS/mbkin)) - (926904896*mcMS**9)/
            (2673*mbkin**13*(1 - mcMS/mbkin)) - (14830478336*mcMS**9)/
            (8019*mbkin**12*(1 - mcMS/mbkin)) + (120034184032*mcMS**9)/
            (24057*mbkin**11*(1 - mcMS/mbkin)) + (184454074304*mcMS**9)/
            (8019*mbkin**10*(1 - mcMS/mbkin)) - (6.174872291693431e6*mcMS**9)/
            (mbkin**9*(1 - mcMS/mbkin)) + (168587456*mcMS**10)/
            (405*mbkin**14*(1 - mcMS/mbkin)) + (2697399296*mcMS**10)/
            (1215*mbkin**13*(1 - mcMS/mbkin)) - (21832075552*mcMS**10)/
            (3645*mbkin**12*(1 - mcMS/mbkin)) - (33548903744*mcMS**10)/
            (1215*mbkin**11*(1 - mcMS/mbkin)) + (7.412453748823227e6*mcMS**10)/
            (mbkin**10*(1 - mcMS/mbkin)) - (708244672*mcMS**11)/
            (1701*mbkin**15*(1 - mcMS/mbkin)) - (11331914752*mcMS**11)/
            (5103*mbkin**14*(1 - mcMS/mbkin)) + (13102526432*mcMS**11)/
            (2187*mbkin**13*(1 - mcMS/mbkin)) + (140940689728*mcMS**11)/
            (5103*mbkin**12*(1 - mcMS/mbkin)) - (7.414310421273456e6*mcMS**11)/
            (mbkin**11*(1 - mcMS/mbkin)) + (590317408*mcMS**12)/
            (1701*mbkin**16*(1 - mcMS/mbkin)) + (9445078528*mcMS**12)/
            (5103*mbkin**15*(1 - mcMS/mbkin)) - (10920872048*mcMS**12)/
            (2187*mbkin**14*(1 - mcMS/mbkin)) - (117473164192*mcMS**12)/
            (5103*mbkin**13*(1 - mcMS/mbkin)) + (6.17978035420157e6*mcMS**12)/
            (mbkin**12*(1 - mcMS/mbkin)) - (97319648*mcMS**13)/
            (405*mbkin**17*(1 - mcMS/mbkin)) - (1557114368*mcMS**13)/
            (1215*mbkin**16*(1 - mcMS/mbkin)) + (12602894416*mcMS**13)/
            (3645*mbkin**15*(1 - mcMS/mbkin)) + (19366609952*mcMS**13)/
            (1215*mbkin**14*(1 - mcMS/mbkin)) - (4.278950562322721e6*mcMS**13)/
            (mbkin**13*(1 - mcMS/mbkin)) + (241010816*mcMS**14)/
            (1755*mbkin**18*(1 - mcMS/mbkin)) + (3856173056*mcMS**14)/
            (5265*mbkin**17*(1 - mcMS/mbkin)) - (31210900672*mcMS**14)/
            (15795*mbkin**16*(1 - mcMS/mbkin)) - (47961152384*mcMS**14)/
            (5265*mbkin**15*(1 - mcMS/mbkin)) + (2.445407134378434e6*mcMS**14)/
            (mbkin**14*(1 - mcMS/mbkin)) - (25983502208*mcMS**15)/
            (405405*mbkin**19*(1 - mcMS/mbkin)) - (415736035328*mcMS**15)/
            (1216215*mbkin**18*(1 - mcMS/mbkin)) + (480694790848*mcMS**15)/
            (521235*mbkin**17*(1 - mcMS/mbkin)) + (5170716939392*mcMS**15)/
            (1216215*mbkin**16*(1 - mcMS/mbkin)) - 
            (1.1413013981894134e6*mcMS**15)/(mbkin**15*(1 - mcMS/mbkin)) + 
            (216546752*mcMS**16)/(9009*mbkin**20*(1 - mcMS/mbkin)) + 
            (3464748032*mcMS**16)/(27027*mbkin**19*(1 - mcMS/mbkin)) - 
            (4006114912*mcMS**16)/(11583*mbkin**18*(1 - mcMS/mbkin)) - 
            (43092803648*mcMS**16)/(27027*mbkin**17*(1 - mcMS/mbkin)) + 
            (428022.7468324014*mcMS**16)/(mbkin**16*(1 - mcMS/mbkin)) - 
            (22334464*mcMS**17)/(3159*mbkin**21*(1 - mcMS/mbkin)) - 
            (357351424*mcMS**17)/(9477*mbkin**20*(1 - mcMS/mbkin)) + 
            (2892313088*mcMS**17)/(28431*mbkin**19*(1 - mcMS/mbkin)) + 
            (4444558336*mcMS**17)/(9477*mbkin**18*(1 - mcMS/mbkin)) - 
            (125897.6637429286*mcMS**17)/(mbkin**17*(1 - mcMS/mbkin)) + 
            (2953285408*mcMS**18)/(1879605*mbkin**22*(1 - mcMS/mbkin)) + 
            (47252566528*mcMS**18)/(5638815*mbkin**21*(1 - mcMS/mbkin)) - 
            (54635780048*mcMS**18)/(2416635*mbkin**20*(1 - mcMS/mbkin)) - 
            (587703796192*mcMS**18)/(5638815*mbkin**19*(1 - mcMS/mbkin)) + 
            (27978.897066866393*mcMS**18)/(mbkin**18*(1 - mcMS/mbkin)) - 
            (1709883424*mcMS**19)/(6891885*mbkin**23*(1 - mcMS/mbkin)) - 
            (27358134784*mcMS**19)/(20675655*mbkin**22*(1 - mcMS/mbkin)) + 
            (31632843344*mcMS**19)/(8860995*mbkin**21*(1 - mcMS/mbkin)) + 
            (340266801376*mcMS**19)/(20675655*mbkin**20*(1 - mcMS/mbkin)) - 
            (4417.944366776069*mcMS**19)/(mbkin**19*(1 - mcMS/mbkin)) + 
            (1392395248*mcMS**20)/(56119635*mbkin**24*(1 - mcMS/mbkin)) + 
            (22278323968*mcMS**20)/(168358905*mbkin**23*(1 - mcMS/mbkin)) - 
            (180315184616*mcMS**20)/(505076715*mbkin**22*(1 - mcMS/mbkin)) - 
            (277086654352*mcMS**20)/(168358905*mbkin**21*(1 - mcMS/mbkin)) + 
            (441.81398353634376*mcMS**20)/(mbkin**20*(1 - mcMS/mbkin)) - 
            (1205584*mcMS**21)/(1020357*mbkin**25*(1 - mcMS/mbkin)) - 
            (19289344*mcMS**21)/(3061071*mbkin**24*(1 - mcMS/mbkin)) + 
            (156123128*mcMS**21)/(9183213*mbkin**23*(1 - mcMS/mbkin)) + 
            (239911216*mcMS**21)/(3061071*mbkin**22*(1 - mcMS/mbkin)) - 
            (21.03958115779376*mcMS**21)/(mbkin**21*(1 - mcMS/mbkin)) - 
            (128*np.pi**2)/(81*mbkin**3) - (2639307106*np.pi**2)/(130945815*mbkin**2) - 
            (10557228424*np.pi**2)/(235702467*mbkin) + (4291805152*mcMS*np.pi**2)/
            (26189163*mbkin**3) + (141065516672*mcMS*np.pi**2)/(392837445*mbkin**2) - 
            (26002213093466807*mcMS*np.pi**2)/(84283699500*mbkin) + 
            (512*mcMS**2*np.pi**2)/(243*mbkin**5) - (867980128*mcMS**2*np.pi**2)/
            (1216215*mbkin**4) - (208991504*mcMS**2*np.pi**2)/(135135*mbkin**3) + 
            (7919159977847357*mcMS**2*np.pi**2)/(4925670750*mbkin**2) + 
            (2335139936*mcMS**3*np.pi**2)/(984555*mbkin**5) + (9080593024*mcMS**3*np.pi**2)/
            (1772199*mbkin**4) - (5402350449714527*mcMS**3*np.pi**2)/
            (1060913700*mbkin**3) + (256*mcMS**4*np.pi**2)/(81*mbkin**7) - 
            (251581118*mcMS**4*np.pi**2)/(36855*mbkin**6) - (17951993368*mcMS**4*np.pi**2)/
            (1216215*mbkin**5) + (428510251683179*mcMS**4*np.pi**2)/
            (39293100*mbkin**4) + (264191488*mcMS**5*np.pi**2)/(15795*mbkin**7) + 
            (38218240*mcMS**5*np.pi**2)/(1053*mbkin**6) - 
            (2939079895225459*mcMS**5*np.pi**2)/(176818950*mbkin**5) - 
            (512*mcMS**6*np.pi**2)/(81*mbkin**9) - (3997664*mcMS**6*np.pi**2)/
            (117*mbkin**8) - (4938654128*mcMS**6*np.pi**2)/(66339*mbkin**7) + 
            (388948027794511*mcMS**6*np.pi**2)/(21049875*mbkin**6) + 
            (713601856*mcMS**7*np.pi**2)/(12285*mbkin**9) + (4684996096*mcMS**7*np.pi**2)/
            (36855*mbkin**8) - (1481998443567913*mcMS**7*np.pi**2)/
            (98232750*mbkin**7) + (640*mcMS**8*np.pi**2)/(243*mbkin**11) - 
            (2571364252*mcMS**8*np.pi**2)/(31185*mbkin**10) - (3391281616*mcMS**8*np.pi**2)/
            (18711*mbkin**9) + (288531306169609*mcMS**8*np.pi**2)/
            (32148900*mbkin**8) + (277634368*mcMS**9*np.pi**2)/(2835*mbkin**11) + 
            (5513080064*mcMS**9*np.pi**2)/(25515*mbkin**10) - 
            (574609667222599*mcMS**9*np.pi**2)/(151559100*mbkin**9) - 
            (39403736*mcMS**10*np.pi**2)/(405*mbkin**12) - (52311376*mcMS**10*np.pi**2)/
            (243*mbkin**11) + (63557860033411*mcMS**10*np.pi**2)/
            (58939650*mbkin**10) + (1509705728*mcMS**11*np.pi**2)/(18711*mbkin**13) + 
            (16734706432*mcMS**11*np.pi**2)/(93555*mbkin**12) - 
            (10780604991879941*mcMS**11*np.pi**2)/(58350253500*mbkin**11) - 
            (247787932*mcMS**12*np.pi**2)/(4455*mbkin**14) - (4950062896*mcMS**12*np.pi**2)/
            (40095*mbkin**13) + (821592511777841*mcMS**12*np.pi**2)/
            (58350253500*mbkin**12) + (1833020416*mcMS**13*np.pi**2)/
            (57915*mbkin**15) + (17100205568*mcMS**13*np.pi**2)/(243243*mbkin**14) + 
            (921344*mcMS**13*np.pi**2)/(6237*mbkin**13) - (17889004304*mcMS**14*np.pi**2)/
            (1216215*mbkin**16) - (4416403664*mcMS**14*np.pi**2)/(135135*mbkin**15) - 
            (3670664*mcMS**14*np.pi**2)/(173745*mbkin**14) + (67488704*mcMS**15*np.pi**2)/
            (12285*mbkin**17) + (8908280320*mcMS**15*np.pi**2)/(729729*mbkin**16) + 
            (114304*mcMS**15*np.pi**2)/(81081*mbkin**15) - (20926*mcMS**16*np.pi**2)/
            (13*mbkin**18) - (418520*mcMS**16*np.pi**2)/(117*mbkin**17) + 
            (2127200*mcMS**17*np.pi**2)/(5967*mbkin**19) + (42544000*mcMS**17*np.pi**2)/
            (53703*mbkin**18) - (11719928*mcMS**18*np.pi**2)/(208845*mbkin**20) - 
            (46879712*mcMS**18*np.pi**2)/(375921*mbkin**19) + (684256*mcMS**19*np.pi**2)/
            (122265*mbkin**21) + (2737024*mcMS**19*np.pi**2)/(220077*mbkin**20) - 
            (150698*mcMS**20*np.pi**2)/(566865*mbkin**22) - (602792*mcMS**20*np.pi**2)/
            (1020357*mbkin**21) - (14120272*mcMS*np.pi**2)/(793611*mbkin**3*
            (1 - mcMS/mbkin)) - (112962176*mcMS*np.pi**2)/(2380833*mbkin**2*
            (1 - mcMS/mbkin)) - (2639567936425367*mcMS*np.pi**2)/
            (554240209002480*mbkin*(1 - mcMS/mbkin)) + (468025720*mcMS**2*np.pi**2)/
            (2380833*mbkin**4*(1 - mcMS/mbkin)) + (3744205760*mcMS**2*np.pi**2)/
            (7142499*mbkin**3*(1 - mcMS/mbkin)) + (97308544099303357*mcMS**2*
            np.pi**2)/(1662720627007440*mbkin**2*(1 - mcMS/mbkin)) - 
            (107374328*mcMS**3*np.pi**2)/(89505*mbkin**5*(1 - mcMS/mbkin)) - 
            (858994624*mcMS**3*np.pi**2)/(268515*mbkin**4*(1 - mcMS/mbkin)) - 
            (1720103377591543597*mcMS**3*np.pi**2)/(4813138657126800*mbkin**3*
            (1 - mcMS/mbkin)) + (12111649028*mcMS**4*np.pi**2)/
            (2297295*mbkin**6*(1 - mcMS/mbkin)) + (96893192224*mcMS**4*np.pi**2)/
            (6891885*mbkin**5*(1 - mcMS/mbkin)) + (32720872090422415*mcMS**4*
            np.pi**2)/(21391727365008*mbkin**4*(1 - mcMS/mbkin)) - 
            (804583036*mcMS**5*np.pi**2)/(45045*mbkin**7*(1 - mcMS/mbkin)) - 
            (6436664288*mcMS**5*np.pi**2)/(135135*mbkin**6*(1 - mcMS/mbkin)) - 
            (65589298818678797*mcMS**5*np.pi**2)/(12869354698200*mbkin**5*
            (1 - mcMS/mbkin)) + (16770272*mcMS**6*np.pi**2)/(351*mbkin**8*
            (1 - mcMS/mbkin)) + (134162176*mcMS**6*np.pi**2)/(1053*mbkin**7*
            (1 - mcMS/mbkin)) + (191974928813525041*mcMS**6*np.pi**2)/
            (14156290168020*mbkin**6*(1 - mcMS/mbkin)) - (12012704*mcMS**7*np.pi**2)/
            (117*mbkin**9*(1 - mcMS/mbkin)) - (96101632*mcMS**7*np.pi**2)/
            (351*mbkin**8*(1 - mcMS/mbkin)) - (84991693461827*mcMS**7*np.pi**2)/
            (2918221020*mbkin**7*(1 - mcMS/mbkin)) + (10420999024*mcMS**8*np.pi**2)/
            (57915*mbkin**10*(1 - mcMS/mbkin)) + (83367992192*mcMS**8*np.pi**2)/
            (173745*mbkin**9*(1 - mcMS/mbkin)) + (10122861686968843*mcMS**8*np.pi**2)/
            (198267369300*mbkin**8*(1 - mcMS/mbkin)) - (231726224*mcMS**9*np.pi**2)/
            (891*mbkin**11*(1 - mcMS/mbkin)) - (1853809792*mcMS**9*np.pi**2)/
            (2673*mbkin**10*(1 - mcMS/mbkin)) - (2943961782538279*mcMS**9*np.pi**2)/
            (39888109800*mbkin**9*(1 - mcMS/mbkin)) + (42146864*mcMS**10*np.pi**2)/
            (135*mbkin**12*(1 - mcMS/mbkin)) + (337174912*mcMS**10*np.pi**2)/
            (405*mbkin**11*(1 - mcMS/mbkin)) + (2248644571575527*mcMS**10*np.pi**2)/
            (25383342600*mbkin**10*(1 - mcMS/mbkin)) - (177061168*mcMS**11*np.pi**2)/
            (567*mbkin**13*(1 - mcMS/mbkin)) - (1416489344*mcMS**11*np.pi**2)/
            (1701*mbkin**12*(1 - mcMS/mbkin)) - (17539470929909561*mcMS**11*np.pi**2)/
            (197990072280*mbkin**11*(1 - mcMS/mbkin)) + (147579352*mcMS**12*np.pi**2)/
            (567*mbkin**14*(1 - mcMS/mbkin)) + (1180634816*mcMS**12*np.pi**2)/
            (1701*mbkin**13*(1 - mcMS/mbkin)) + (14614951658421347*mcMS**12*np.pi**2)/
            (197990072280*mbkin**12*(1 - mcMS/mbkin)) - (24329912*mcMS**13*np.pi**2)/
            (135*mbkin**15*(1 - mcMS/mbkin)) - (194639296*mcMS**13*np.pi**2)/
            (405*mbkin**14*(1 - mcMS/mbkin)) - (234186157046207*mcMS**13*np.pi**2)/
            (4583103525*mbkin**13*(1 - mcMS/mbkin)) + (60252704*mcMS**14*np.pi**2)/
            (585*mbkin**16*(1 - mcMS/mbkin)) + (482021632*mcMS**14*np.pi**2)/
            (1755*mbkin**15*(1 - mcMS/mbkin)) + (42172637038013753*mcMS**14*np.pi**2)/
            (1444519404900*mbkin**14*(1 - mcMS/mbkin)) - 
            (6495875552*mcMS**15*np.pi**2)/(135135*mbkin**17*(1 - mcMS/mbkin)) - 
            (51967004416*mcMS**15*np.pi**2)/(405405*mbkin**16*(1 - mcMS/mbkin)) - 
            (56718967980765817*mcMS**15*np.pi**2)/(4163614755300*mbkin**15*
            (1 - mcMS/mbkin)) + (54136688*mcMS**16*np.pi**2)/(3003*mbkin**18*
            (1 - mcMS/mbkin)) + (433093504*mcMS**16*np.pi**2)/(9009*mbkin**17*
            (1 - mcMS/mbkin)) + (146075958129473*mcMS**16*np.pi**2)/
            (28598565996*mbkin**16*(1 - mcMS/mbkin)) - (5583616*mcMS**17*np.pi**2)/
            (1053*mbkin**19*(1 - mcMS/mbkin)) - (44668928*mcMS**17*np.pi**2)/
            (3159*mbkin**18*(1 - mcMS/mbkin)) - (9450846437430613*mcMS**17*np.pi**2)/
            (6291684519120*mbkin**17*(1 - mcMS/mbkin)) + 
            (738321352*mcMS**18*np.pi**2)/(626535*mbkin**20*(1 - mcMS/mbkin)) + 
            (5906570816*mcMS**18*np.pi**2)/(1879605*mbkin**19*(1 - mcMS/mbkin)) + 
            (2318129640224621*mcMS**18*np.pi**2)/(6945366027600*mbkin**18*
            (1 - mcMS/mbkin)) - (427470856*mcMS**19*np.pi**2)/(2297295*mbkin**21*
            (1 - mcMS/mbkin)) - (3419766848*mcMS**19*np.pi**2)/(6891885*mbkin**20*
            (1 - mcMS/mbkin)) - (84541874005479719*mcMS**19*np.pi**2)/
            (1604379552375600*mbkin**19*(1 - mcMS/mbkin)) + 
            (348098812*mcMS**20*np.pi**2)/(18706545*mbkin**22*(1 - mcMS/mbkin)) + 
            (2784790496*mcMS**20*np.pi**2)/(56119635*mbkin**21*(1 - mcMS/mbkin)) + 
            (481841652534832741*mcMS**20*np.pi**2)/(91449634485409200*mbkin**20*
            (1 - mcMS/mbkin)) - (301396*mcMS**21*np.pi**2)/(340119*mbkin**23*
            (1 - mcMS/mbkin)) - (2411168*mcMS**21*np.pi**2)/(1020357*mbkin**22*
            (1 - mcMS/mbkin)) - (29795786977559*mcMS**21*np.pi**2)/
            (118765759071960*mbkin**21*(1 - mcMS/mbkin)) - 
            (4656*mcMS*np.pi**4)/(385*mbkin) + (352*mcMS**2*np.pi**4)/(9*mbkin**2) - 
            (3536*mcMS**3*np.pi**4)/(45*mbkin**3) + (4056*mcMS**4*np.pi**4)/(35*mbkin**4) - 
            (416*mcMS**5*np.pi**4)/(3*mbkin**5) + (2048*mcMS**6*np.pi**4)/(15*mbkin**6) - 
            (736*mcMS**7*np.pi**4)/(7*mbkin**7) + (2088*mcMS**8*np.pi**4)/(35*mbkin**8) - 
            (1072*mcMS**9*np.pi**4)/(45*mbkin**9) + (32*mcMS**10*np.pi**4)/(5*mbkin**10) - 
            (3664*mcMS**11*np.pi**4)/(3465*mbkin**11) + (8*mcMS**12*np.pi**4)/
            (99*mbkin**12) + (13022602/(3357585*mbkin**2) + 
            52090408/(6043653*mbkin) + (141202720*mcMS)/(793611*mbkin**3) + 
            (2824054400*mcMS)/(7142499*mbkin**2) + (333324989499253*mcMS)/
                (547844046750*mbkin) - (2070904*mcMS**2)/(1053*mbkin**4) - 
            (41418080*mcMS**2)/(9477*mbkin**3) - (15478417721719*mcMS**2)/
                (3557428875*mbkin**2) + (752032*mcMS**3)/(65*mbkin**5) + 
            (3008128*mcMS**3)/(117*mbkin**4) + (683837705723971*mcMS**3)/
                (42141849750*mbkin**3) - (6606707542*mcMS**4)/(135135*mbkin**6) - 
            (26426830168*mcMS**4)/(243243*mbkin**5) - (561305364499738*mcMS**4)/
                (12642554925*mbkin**4) + (55416448*mcMS**5)/(351*mbkin**7) + 
            (1108328960*mcMS**5)/(3159*mbkin**6) + (70552906019*mcMS**5)/
                (727650*mbkin**5) - (139918240*mcMS**6)/(351*mbkin**8) - 
            (2798364800*mcMS**6)/(3159*mbkin**7) - (45457380358813*mcMS**6)/
                (273648375*mbkin**6) + (73025920*mcMS**7)/(91*mbkin**9) + 
            (1460518400*mcMS**7)/(819*mbkin**8) + (119765484898127*mcMS**7)/
                (547296750*mbkin**7) - (5833020164*mcMS**8)/(4455*mbkin**10) - 
            (23332080656*mcMS**8)/(8019*mbkin**9) - (47118708753404*mcMS**8)/
                (212837625*mbkin**8) + (709026368*mcMS**9)/(405*mbkin**11) + 
            (2836105472*mcMS**9)/(729*mbkin**10) + (6305059020913*mcMS**9)/
                (36486450*mbkin**9) - (260544944*mcMS**10)/(135*mbkin**12) - 
            (1042179776*mcMS**10)/(243*mbkin**11) - (7955554444931*mcMS**10)/
                (76621545*mbkin**10) + (10962228160*mcMS**11)/(6237*mbkin**13) + 
            (219244563200*mcMS**11)/(56133*mbkin**12) + 
            (181064392519811*mcMS**11)/(3831077250*mbkin**11) - 
            (3960380*mcMS**12)/(3*mbkin**14) - (79207600*mcMS**12)/
                (27*mbkin**13) - (332372802464726*mcMS**12)/(21070924875*
                mbkin**12) + (109805696*mcMS**13)/(135*mbkin**15) + 
            (439222784*mcMS**13)/(243*mbkin**14) + (5983215661469*mcMS**13)/
                (1641890250*mbkin**13) - (5001335648*mcMS**14)/(12285*mbkin**16) - 
            (20005342592*mcMS**14)/(22113*mbkin**15) - (194192359957*mcMS**14)/
                (372683025*mbkin**14) + (7341836416*mcMS**15)/(45045*mbkin**17) + 
            (29367345664*mcMS**15)/(81081*mbkin**16) + (3809167143809*mcMS**15)/
                (109568809350*mbkin**15) - (1987970*mcMS**16)/(39*mbkin**18) - 
            (39759400*mcMS**16)/(351*mbkin**17) + (214847200*mcMS**17)/
                (17901*mbkin**19) + (4296944000*mcMS**17)/(161109*mbkin**18) - 
            (1254032296*mcMS**18)/(626535*mbkin**20) - (5016129184*mcMS**18)/
                (1127763*mbkin**19) + (77320928*mcMS**19)/(366795*mbkin**21) + 
            (309283712*mcMS**19)/(660231*mbkin**20) - (1054886*mcMS**20)/
                (100035*mbkin**22) - (4219544*mcMS**20)/(180063*mbkin**21) + 
            (376528*mcMS**2)/(1053*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (3012224*mcMS**2)/(3159*mbkin**3*(1 - mcMS/mbkin)**2) - 
            (753056*mcMS**2)/(1053*mbkin**2*(1 - mcMS/mbkin)**2) - 
            (429497312*mcMS**3)/(89505*mbkin**5*(1 - mcMS/mbkin)**2) - 
            (3435978496*mcMS**3)/(268515*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (858994624*mcMS**3)/(89505*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (234545329016*mcMS**4)/(6891885*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (1876362632128*mcMS**4)/(20675655*mbkin**5*(1 - mcMS/mbkin)**2) - 
            (469090658032*mcMS**4)/(6891885*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (75343294960*mcMS**5)/(459459*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (602746359680*mcMS**5)/(1378377*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (150686589920*mcMS**5)/(459459*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (79048041848*mcMS**6)/(135135*mbkin**8*(1 - mcMS/mbkin)**2) + 
            (632384334784*mcMS**6)/(405405*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (158096083696*mcMS**6)/(135135*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (566619520*mcMS**7)/(351*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (4532956160*mcMS**7)/(1053*mbkin**8*(1 - mcMS/mbkin)**2) + 
            (1133239040*mcMS**7)/(351*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (205356871136*mcMS**8)/(57915*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (1642854969088*mcMS**8)/(173745*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (410713742272*mcMS**8)/(57915*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (366047261248*mcMS**9)/(57915*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (2928378089984*mcMS**9)/(173745*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (732094522496*mcMS**9)/(57915*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (41256072896*mcMS**10)/(4455*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (330048583168*mcMS**10)/(13365*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (82512145792*mcMS**10)/(4455*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (31867463104*mcMS**11)/(2835*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (254939704832*mcMS**11)/(8505*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (63734926208*mcMS**11)/(2835*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (6433846768*mcMS**12)/(567*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (51470774144*mcMS**12)/(1701*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (12867693536*mcMS**12)/(567*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (27020210848*mcMS**13)/(2835*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (216161686784*mcMS**13)/(8505*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (54040421696*mcMS**13)/(2835*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (896774288*mcMS**14)/(135*mbkin**16*(1 - mcMS/mbkin)**2) + 
            (7174194304*mcMS**14)/(405*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (1793548576*mcMS**14)/(135*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (73703643776*mcMS**15)/(19305*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (589629150208*mcMS**15)/(57915*mbkin**16*(1 - mcMS/mbkin)**2) + 
            (147407287552*mcMS**15)/(19305*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (34568184736*mcMS**16)/(19305*mbkin**18*(1 - mcMS/mbkin)**2) + 
            (276545477888*mcMS**16)/(57915*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (69136369472*mcMS**16)/(19305*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (7812195136*mcMS**17)/(11583*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (62497561088*mcMS**17)/(34749*mbkin**18*(1 - mcMS/mbkin)**2) + 
            (15624390272*mcMS**17)/(11583*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (7339439504*mcMS**18)/(36855*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (58715516032*mcMS**18)/(110565*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (14678879008*mcMS**18)/(36855*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (18003292256*mcMS**19)/(405405*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (144026338048*mcMS**19)/(1216215*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (36006584512*mcMS**19)/(405405*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (9695082136*mcMS**20)/(1378377*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (77560657088*mcMS**20)/(4135131*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (19390164272*mcMS**20)/(1378377*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (694454128*mcMS**21)/(984555*mbkin**23*(1 - mcMS/mbkin)**2) - 
            (5555633024*mcMS**21)/(2953665*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (1388908256*mcMS**21)/(984555*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (602792*mcMS**22)/(17901*mbkin**24*(1 - mcMS/mbkin)**2) + 
            (4822336*mcMS**22)/(53703*mbkin**23*(1 - mcMS/mbkin)**2) - 
            (1205584*mcMS**22)/(17901*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (70601360*mcMS)/(2380833*mbkin**3*(1 - mcMS/mbkin)) - 
            (564810880*mcMS)/(7142499*mbkin**2*(1 - mcMS/mbkin)) + 
            (732489110*mcMS)/(2380833*mbkin*(1 - mcMS/mbkin)) + 
            (2340128600*mcMS**2)/(7142499*mbkin**4*(1 - mcMS/mbkin)) + 
            (18721028800*mcMS**2)/(21427497*mbkin**3*(1 - mcMS/mbkin)) - 
            (24278834225*mcMS**2)/(7142499*mbkin**2*(1 - mcMS/mbkin)) - 
            (107374328*mcMS**3)/(53703*mbkin**5*(1 - mcMS/mbkin)) - 
            (858994624*mcMS**3)/(161109*mbkin**4*(1 - mcMS/mbkin)) + 
            (1114008653*mcMS**3)/(53703*mbkin**3*(1 - mcMS/mbkin)) + 
            (12111649028*mcMS**4)/(1378377*mbkin**6*(1 - mcMS/mbkin)) + 
            (96893192224*mcMS**4)/(4135131*mbkin**5*(1 - mcMS/mbkin)) - 
            (251316717331*mcMS**4)/(2756754*mbkin**4*(1 - mcMS/mbkin)) - 
            (804583036*mcMS**5)/(27027*mbkin**7*(1 - mcMS/mbkin)) - 
            (6436664288*mcMS**5)/(81081*mbkin**6*(1 - mcMS/mbkin)) + 
            (16695097997*mcMS**5)/(54054*mbkin**5*(1 - mcMS/mbkin)) + 
            (83851360*mcMS**6)/(1053*mbkin**8*(1 - mcMS/mbkin)) + 
            (670810880*mcMS**6)/(3159*mbkin**7*(1 - mcMS/mbkin)) - 
            (869957860*mcMS**6)/(1053*mbkin**6*(1 - mcMS/mbkin)) - 
            (60063520*mcMS**7)/(351*mbkin**9*(1 - mcMS/mbkin)) - 
            (480508160*mcMS**7)/(1053*mbkin**8*(1 - mcMS/mbkin)) + 
            (623159020*mcMS**7)/(351*mbkin**7*(1 - mcMS/mbkin)) + 
            (10420999024*mcMS**8)/(34749*mbkin**10*(1 - mcMS/mbkin)) + 
            (83367992192*mcMS**8)/(104247*mbkin**9*(1 - mcMS/mbkin)) - 
            (108117864874*mcMS**8)/(34749*mbkin**8*(1 - mcMS/mbkin)) - 
            (1158631120*mcMS**9)/(2673*mbkin**11*(1 - mcMS/mbkin)) - 
            (9269048960*mcMS**9)/(8019*mbkin**10*(1 - mcMS/mbkin)) + 
            (12020797870*mcMS**9)/(2673*mbkin**9*(1 - mcMS/mbkin)) + 
            (42146864*mcMS**10)/(81*mbkin**12*(1 - mcMS/mbkin)) + 
            (337174912*mcMS**10)/(243*mbkin**11*(1 - mcMS/mbkin)) - 
            (437273714*mcMS**10)/(81*mbkin**10*(1 - mcMS/mbkin)) - 
            (885305840*mcMS**11)/(1701*mbkin**13*(1 - mcMS/mbkin)) - 
            (7082446720*mcMS**11)/(5103*mbkin**12*(1 - mcMS/mbkin)) + 
            (9185048090*mcMS**11)/(1701*mbkin**11*(1 - mcMS/mbkin)) + 
            (737896760*mcMS**12)/(1701*mbkin**14*(1 - mcMS/mbkin)) + 
            (5903174080*mcMS**12)/(5103*mbkin**13*(1 - mcMS/mbkin)) - 
            (7655678885*mcMS**12)/(1701*mbkin**12*(1 - mcMS/mbkin)) - 
            (24329912*mcMS**13)/(81*mbkin**15*(1 - mcMS/mbkin)) - 
            (194639296*mcMS**13)/(243*mbkin**14*(1 - mcMS/mbkin)) + 
            (252422837*mcMS**13)/(81*mbkin**13*(1 - mcMS/mbkin)) + 
            (60252704*mcMS**14)/(351*mbkin**16*(1 - mcMS/mbkin)) + 
            (482021632*mcMS**14)/(1053*mbkin**15*(1 - mcMS/mbkin)) - 
            (625121804*mcMS**14)/(351*mbkin**14*(1 - mcMS/mbkin)) - 
            (6495875552*mcMS**15)/(81081*mbkin**17*(1 - mcMS/mbkin)) - 
            (51967004416*mcMS**15)/(243243*mbkin**16*(1 - mcMS/mbkin)) + 
            (67394708852*mcMS**15)/(81081*mbkin**15*(1 - mcMS/mbkin)) + 
            (270683440*mcMS**16)/(9009*mbkin**18*(1 - mcMS/mbkin)) + 
            (2165467520*mcMS**16)/(27027*mbkin**17*(1 - mcMS/mbkin)) - 
            (2808340690*mcMS**16)/(9009*mbkin**16*(1 - mcMS/mbkin)) - 
            (27918080*mcMS**17)/(3159*mbkin**19*(1 - mcMS/mbkin)) - 
            (223344640*mcMS**17)/(9477*mbkin**18*(1 - mcMS/mbkin)) + 
            (289650080*mcMS**17)/(3159*mbkin**17*(1 - mcMS/mbkin)) + 
            (738321352*mcMS**18)/(375921*mbkin**20*(1 - mcMS/mbkin)) + 
            (5906570816*mcMS**18)/(1127763*mbkin**19*(1 - mcMS/mbkin)) - 
            (7660084027*mcMS**18)/(375921*mbkin**18*(1 - mcMS/mbkin)) - 
            (427470856*mcMS**19)/(1378377*mbkin**21*(1 - mcMS/mbkin)) - 
            (3419766848*mcMS**19)/(4135131*mbkin**20*(1 - mcMS/mbkin)) + 
            (4435010131*mcMS**19)/(1378377*mbkin**19*(1 - mcMS/mbkin)) + 
            (348098812*mcMS**20)/(11223927*mbkin**22*(1 - mcMS/mbkin)) + 
            (2784790496*mcMS**20)/(33671781*mbkin**21*(1 - mcMS/mbkin)) - 
            (7223050349*mcMS**20)/(22447854*mbkin**20*(1 - mcMS/mbkin)) - 
            (1506980*mcMS**21)/(1020357*mbkin**23*(1 - mcMS/mbkin)) - 
            (12055840*mcMS**21)/(3061071*mbkin**22*(1 - mcMS/mbkin)) + 
            (31269835*mcMS**21)/(2040714*mbkin**21*(1 - mcMS/mbkin)) - 
            (2315776*mcMS*np.pi**2)/(135135*mbkin) + (56914342*mcMS**2*np.pi**2)/
                (405405*mbkin**2) - (147040*mcMS**3*np.pi**2)/(297*mbkin**3) - 
            (64*mcMS**4*np.pi**2)/(9*mbkin**6) - (256*mcMS**4*np.pi**2)/(27*mbkin**5) + 
            (16212884*mcMS**4*np.pi**2)/(10395*mbkin**4) - (239168*mcMS**5*np.pi**2)/
                (81*mbkin**5) + (941978*mcMS**6*np.pi**2)/(189*mbkin**6) - 
            (298144*mcMS**7*np.pi**2)/(45*mbkin**7) + (1269020*mcMS**8*np.pi**2)/
                (189*mbkin**8) - (1650304*mcMS**9*np.pi**2)/(315*mbkin**9) + 
            (85102*mcMS**10*np.pi**2)/(27*mbkin**10) - (4070624*mcMS**11*np.pi**2)/
                (2835*mbkin**11) + (711968*mcMS**12*np.pi**2)/(1485*mbkin**12) - 
            (230336*mcMS**13*np.pi**2)/(2079*mbkin**13) + (917666*mcMS**14*np.pi**2)/
                (57915*mbkin**14) - (28576*mcMS**15*np.pi**2)/(27027*mbkin**15))*
            np.log(mcMS**2/mbkin**2) + ((94132*mcMS**2)/(351*mbkin**2*
                (1 - mcMS/mbkin)**2) - (107374328*mcMS**3)/(29835*mbkin**3*
                (1 - mcMS/mbkin)**2) + (58636332254*mcMS**4)/(2297295*mbkin**4*
                (1 - mcMS/mbkin)**2) - (18835823740*mcMS**5)/(153153*mbkin**5*
                (1 - mcMS/mbkin)**2) + (19762010462*mcMS**6)/(45045*mbkin**6*
                (1 - mcMS/mbkin)**2) - (141654880*mcMS**7)/(117*mbkin**7*
                (1 - mcMS/mbkin)**2) + (51339217784*mcMS**8)/(19305*mbkin**8*
                (1 - mcMS/mbkin)**2) - (91511815312*mcMS**9)/(19305*mbkin**9*
                (1 - mcMS/mbkin)**2) + (10314018224*mcMS**10)/(1485*mbkin**10*
                (1 - mcMS/mbkin)**2) - (7966865776*mcMS**11)/(945*mbkin**11*
                (1 - mcMS/mbkin)**2) + (1608461692*mcMS**12)/(189*mbkin**12*
                (1 - mcMS/mbkin)**2) - (6755052712*mcMS**13)/(945*mbkin**13*
                (1 - mcMS/mbkin)**2) + (224193572*mcMS**14)/(45*mbkin**14*
                (1 - mcMS/mbkin)**2) - (18425910944*mcMS**15)/(6435*mbkin**15*
                (1 - mcMS/mbkin)**2) + (8642046184*mcMS**16)/(6435*mbkin**16*
                (1 - mcMS/mbkin)**2) - (1953048784*mcMS**17)/(3861*mbkin**17*
                (1 - mcMS/mbkin)**2) + (1834859876*mcMS**18)/(12285*mbkin**18*
                (1 - mcMS/mbkin)**2) - (4500823064*mcMS**19)/(135135*mbkin**19*
                (1 - mcMS/mbkin)**2) + (2423770534*mcMS**20)/(459459*mbkin**20*
                (1 - mcMS/mbkin)**2) - (173613532*mcMS**21)/(328185*mbkin**21*
                (1 - mcMS/mbkin)**2) + (150698*mcMS**22)/(5967*mbkin**22*
                (1 - mcMS/mbkin)**2) - (65306258*mcMS)/(793611*mbkin*
                (1 - mcMS/mbkin)) + (2164618955*mcMS**2)/(2380833*mbkin**2*
                (1 - mcMS/mbkin)) - (496606267*mcMS**3)/(89505*mbkin**3*
                (1 - mcMS/mbkin)) + (112032753509*mcMS**4)/(4594590*mbkin**4*
                (1 - mcMS/mbkin)) - (7442393083*mcMS**5)/(90090*mbkin**5*
                (1 - mcMS/mbkin)) + (77562508*mcMS**6)/(351*mbkin**6*
                (1 - mcMS/mbkin)) - (55558756*mcMS**7)/(117*mbkin**7*
                (1 - mcMS/mbkin)) + (48197120486*mcMS**8)/(57915*mbkin**8*
                (1 - mcMS/mbkin)) - (1071733786*mcMS**9)/(891*mbkin**9*
                (1 - mcMS/mbkin)) + (194929246*mcMS**10)/(135*mbkin**10*
                (1 - mcMS/mbkin)) - (818907902*mcMS**11)/(567*mbkin**11*
                (1 - mcMS/mbkin)) + (682554503*mcMS**12)/(567*mbkin**12*
                (1 - mcMS/mbkin)) - (112525843*mcMS**13)/(135*mbkin**13*
                (1 - mcMS/mbkin)) + (278668756*mcMS**14)/(585*mbkin**14*
                (1 - mcMS/mbkin)) - (30043424428*mcMS**15)/(135135*mbkin**15*
                (1 - mcMS/mbkin)) + (250382182*mcMS**16)/(3003*mbkin**16*
                (1 - mcMS/mbkin)) - (25824224*mcMS**17)/(1053*mbkin**17*
                (1 - mcMS/mbkin)) + (3414736253*mcMS**18)/(626535*mbkin**18*
                (1 - mcMS/mbkin)) - (1977052709*mcMS**19)/(2297295*mbkin**19*
                (1 - mcMS/mbkin)) + (3219914011*mcMS**20)/(37413090*mbkin**20*
                (1 - mcMS/mbkin)) - (2787913*mcMS**21)/(680238*mbkin**21*
                (1 - mcMS/mbkin)) + (392*mcMS**4*np.pi**2)/(9*mbkin**4))*
            np.log(mcMS**2/mbkin**2)**2 + (-1224889965314/2917512675 - 
            79572128/(2606175*mbkin**2) - 318288512/(4691115*mbkin) + 
            (7160128*mcMS)/(405405*mbkin**3) + (229124096*mcMS)/
                (6081075*mbkin**2) + (326706188655068*mcMS)/(189638323875*mbkin) + 
            (1777231936*mcMS**2)/(2606175*mbkin**4) + (3554463872*mcMS**2)/
                (2606175*mbkin**3) + (2865623237416*mcMS**2)/(2462835375*mbkin**2) - 
            (3974648896*mcMS**3)/(1403325*mbkin**5) - (63594382336*mcMS**3)/
                (12629925*mbkin**4) - (281984971348*mcMS**3)/(12629925*mbkin**3) + 
            (4524595232*mcMS**4)/(841995*mbkin**6) + (18098380928*mcMS**4)/
                (2525985*mbkin**5) + (870176864794*mcMS**4)/(12629925*mbkin**4) - 
            (8091712*mcMS**5)/(1575*mbkin**7) - (1559617477592*mcMS**5)/
                (12629925*mbkin**5) - (3205288064*mcMS**6)/(164025*mbkin**7) + 
            (10050869662528*mcMS**6)/(63149625*mbkin**6) + (1043090752*mcMS**7)/
                (127575*mbkin**9) + (16689452032*mcMS**7)/(382725*mbkin**8) - 
            (72573893090936*mcMS**7)/(442047375*mbkin**7) - (615949856*mcMS**8)/
                (42525*mbkin**10) - (2463799424*mcMS**8)/(42525*mbkin**9) + 
            (1138108102942*mcMS**8)/(8037225*mbkin**8) + (11092672*mcMS**9)/
                (729*mbkin**11) + (354965504*mcMS**9)/(6561*mbkin**10) - 
            (3835810595972*mcMS**9)/(37889775*mbkin**9) - (282771392*mcMS**10)/
                (25515*mbkin**12) - (565542784*mcMS**10)/(15309*mbkin**11) + 
            (2201965196248*mcMS**10)/(37889775*mbkin**10) + (1621773376*mcMS**11)/
                (280665*mbkin**13) + (25948374016*mcMS**11)/(1403325*mbkin**12) - 
            (53923630556372*mcMS**11)/(2083937625*mbkin**11) - 
            (2996286176*mcMS**12)/(1403325*mbkin**14) - (11985144704*mcMS**12)/
                (1804275*mbkin**13) + (850970953742*mcMS**12)/(99235125*mbkin**12) + 
            (4175665088*mcMS**13)/(7818525*mbkin**15) + (38177509376*mcMS**13)/
                (23455575*mbkin**14) - (1193047168*mcMS**13)/(601425*mbkin**13) - 
            (33006208*mcMS**14)/(405405*mbkin**16) - (33006208*mcMS**14)/
                (135135*mbkin**15) + (16503104*mcMS**14)/(57915*mbkin**14) + 
            (34821824*mcMS**15)/(6081075*mbkin**17) + (557149184*mcMS**15)/
                (32837805*mbkin**16) - (69643648*mcMS**15)/(3648645*mbkin**15) + 
            (9621296*np.pi**2)/841995 - (225133984*mcMS*np.pi**2)/(4209975*mbkin) + 
            (707264*mcMS**2*np.pi**2)/(18225*mbkin**2) + (4244320*mcMS**3*np.pi**2)/
                (15309*mbkin**3) - (23972624*mcMS**4*np.pi**2)/(25515*mbkin**4) + 
            (12807232*mcMS**5*np.pi**2)/(8505*mbkin**5) - (27771392*mcMS**6*np.pi**2)/
                (18225*mbkin**6) + (137625664*mcMS**7*np.pi**2)/(127575*mbkin**7) - 
            (106640*mcMS**8*np.pi**2)/(189*mbkin**8) + (2403296*mcMS**9*np.pi**2)/
                (10935*mbkin**9) - (4601024*mcMS**10*np.pi**2)/(76545*mbkin**10) + 
            (14124512*mcMS**11*np.pi**2)/(1403325*mbkin**11) - 
            (3270896*mcMS**12*np.pi**2)/(4209975*mbkin**12) - (14120272*mcMS*np.pi**2)/
                (2380833*mbkin*(1 - mcMS/mbkin)) + (468025720*mcMS**2*np.pi**2)/
                (7142499*mbkin**2*(1 - mcMS/mbkin)) - (107374328*mcMS**3*np.pi**2)/
                (268515*mbkin**3*(1 - mcMS/mbkin)) + (12111649028*mcMS**4*np.pi**2)/
                (6891885*mbkin**4*(1 - mcMS/mbkin)) - (804583036*mcMS**5*np.pi**2)/
                (135135*mbkin**5*(1 - mcMS/mbkin)) + (16770272*mcMS**6*np.pi**2)/
                (1053*mbkin**6*(1 - mcMS/mbkin)) - (12012704*mcMS**7*np.pi**2)/
                (351*mbkin**7*(1 - mcMS/mbkin)) + (10420999024*mcMS**8*np.pi**2)/
                (173745*mbkin**8*(1 - mcMS/mbkin)) - (231726224*mcMS**9*np.pi**2)/
                (2673*mbkin**9*(1 - mcMS/mbkin)) + (42146864*mcMS**10*np.pi**2)/
                (405*mbkin**10*(1 - mcMS/mbkin)) - (177061168*mcMS**11*np.pi**2)/
                (1701*mbkin**11*(1 - mcMS/mbkin)) + (147579352*mcMS**12*np.pi**2)/
                (1701*mbkin**12*(1 - mcMS/mbkin)) - (24329912*mcMS**13*np.pi**2)/
                (405*mbkin**13*(1 - mcMS/mbkin)) + (60252704*mcMS**14*np.pi**2)/
                (1755*mbkin**14*(1 - mcMS/mbkin)) - (6495875552*mcMS**15*np.pi**2)/
                (405405*mbkin**15*(1 - mcMS/mbkin)) + (54136688*mcMS**16*np.pi**2)/
                (9009*mbkin**16*(1 - mcMS/mbkin)) - (5583616*mcMS**17*np.pi**2)/
                (3159*mbkin**17*(1 - mcMS/mbkin)) + (738321352*mcMS**18*np.pi**2)/
                (1879605*mbkin**18*(1 - mcMS/mbkin)) - (427470856*mcMS**19*np.pi**2)/
                (6891885*mbkin**19*(1 - mcMS/mbkin)) + (348098812*mcMS**20*np.pi**2)/
                (56119635*mbkin**20*(1 - mcMS/mbkin)) - (301396*mcMS**21*np.pi**2)/
                (1020357*mbkin**21*(1 - mcMS/mbkin)) + 
            ((-3580064*mcMS)/(675675*mbkin) - (444307984*mcMS**2)/
                (868725*mbkin**2) + (1987324448*mcMS**3)/(467775*mbkin**3) - 
                (4524595232*mcMS**4)/(280665*mbkin**4) + (4045856*mcMS**5)/
                (105*mbkin**5) - (400661008*mcMS**6)/(6075*mbkin**6) + 
                (521545376*mcMS**7)/(6075*mbkin**7) - (1231899712*mcMS**8)/
                (14175*mbkin**8) + (5546336*mcMS**9)/(81*mbkin**9) - 
                (70692848*mcMS**10)/(1701*mbkin**10) + (810886688*mcMS**11)/
                (42525*mbkin**11) - (2996286176*mcMS**12)/(467775*mbkin**12) + 
                (298261792*mcMS**13)/(200475*mbkin**13) - (4125776*mcMS**14)/
                (19305*mbkin**14) + (17410912*mcMS**15)/(1216215*mbkin**15))*
                np.log(mcMS**2/mbkin**2))*np.log(1 - mcMS/mbkin) + 
            (23931088/841995 - (14708576*mcMS)/(841995*mbkin) - 
            (8950336*mcMS**2)/(10935*mbkin**2) + (33425824*mcMS**3)/
                (8505*mbkin**3) - (233018864*mcMS**4)/(25515*mbkin**4) + 
            (67270976*mcMS**5)/(5103*mbkin**5) - (47392768*mcMS**6)/
                (3645*mbkin**6) + (235510208*mcMS**7)/(25515*mbkin**7) - 
            (123588112*mcMS**8)/(25515*mbkin**8) + (20426656*mcMS**9)/
                (10935*mbkin**9) - (38385728*mcMS**10)/(76545*mbkin**10) + 
            (69725024*mcMS**11)/(841995*mbkin**11) - (1775344*mcMS**12)/
                (280665*mbkin**12))*np.log(1 - mcMS/mbkin)**2 + 
            ((94132*mcMS**2)/(351*mbkin**2*(1 - mcMS/mbkin)**2) - 
            (107374328*mcMS**3)/(29835*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (58636332254*mcMS**4)/(2297295*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (18835823740*mcMS**5)/(153153*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (19762010462*mcMS**6)/(45045*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (141654880*mcMS**7)/(117*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (51339217784*mcMS**8)/(19305*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (91511815312*mcMS**9)/(19305*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (10314018224*mcMS**10)/(1485*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (7966865776*mcMS**11)/(945*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (1608461692*mcMS**12)/(189*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (6755052712*mcMS**13)/(945*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (224193572*mcMS**14)/(45*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (18425910944*mcMS**15)/(6435*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (8642046184*mcMS**16)/(6435*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (1953048784*mcMS**17)/(3861*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (1834859876*mcMS**18)/(12285*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (4500823064*mcMS**19)/(135135*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (2423770534*mcMS**20)/(459459*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (173613532*mcMS**21)/(328185*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (150698*mcMS**22)/(5967*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (1765034*mcMS)/(61047*mbkin*(1 - mcMS/mbkin)) - 
            (58503215*mcMS**2)/(183141*mbkin**2*(1 - mcMS/mbkin)) + 
            (13421791*mcMS**3)/(6885*mbkin**3*(1 - mcMS/mbkin)) - 
            (3027912257*mcMS**4)/(353430*mbkin**4*(1 - mcMS/mbkin)) + 
            (201145759*mcMS**5)/(6930*mbkin**5*(1 - mcMS/mbkin)) - 
            (2096284*mcMS**6)/(27*mbkin**6*(1 - mcMS/mbkin)) + 
            (1501588*mcMS**7)/(9*mbkin**7*(1 - mcMS/mbkin)) - 
            (1302624878*mcMS**8)/(4455*mbkin**8*(1 - mcMS/mbkin)) + 
            (376555114*mcMS**9)/(891*mbkin**9*(1 - mcMS/mbkin)) - 
            (68488654*mcMS**10)/(135*mbkin**10*(1 - mcMS/mbkin)) + 
            (287724398*mcMS**11)/(567*mbkin**11*(1 - mcMS/mbkin)) - 
            (239816447*mcMS**12)/(567*mbkin**12*(1 - mcMS/mbkin)) + 
            (39536107*mcMS**13)/(135*mbkin**13*(1 - mcMS/mbkin)) - 
            (7531588*mcMS**14)/(45*mbkin**14*(1 - mcMS/mbkin)) + 
            (811984444*mcMS**15)/(10395*mbkin**15*(1 - mcMS/mbkin)) - 
            (6767086*mcMS**16)/(231*mbkin**16*(1 - mcMS/mbkin)) + 
            (697952*mcMS**17)/(81*mbkin**17*(1 - mcMS/mbkin)) - 
            (92290169*mcMS**18)/(48195*mbkin**18*(1 - mcMS/mbkin)) + 
            (53433857*mcMS**19)/(176715*mbkin**19*(1 - mcMS/mbkin)) - 
            (87024703*mcMS**20)/(2877930*mbkin**20*(1 - mcMS/mbkin)) + 
            (75349*mcMS**21)/(52326*mbkin**21*(1 - mcMS/mbkin)))*
            np.log(mu0**2/mus**2)**2 + (-13022602/(124355*mbkin**2) - 
            52090408/(223839*mbkin) + (28240544*mcMS)/(29393*mbkin**3) + 
            (564810880*mcMS)/(264537*mbkin**2) - (188264*mcMS**2)/(39*mbkin**4) - 
            (3765280*mcMS**2)/(351*mbkin**3) + (20304864*mcMS**3)/
                (1105*mbkin**5) + (9024384*mcMS**3)/(221*mbkin**4) - 
            (287248154*mcMS**4)/(5005*mbkin**6) - (1148992616*mcMS**4)/
                (9009*mbkin**5) + (1910912*mcMS**5)/(13*mbkin**7) + 
            (38218240*mcMS**5)/(117*mbkin**6) - (3997664*mcMS**6)/(13*mbkin**8) - 
            (79953280*mcMS**6)/(117*mbkin**7) + (48090240*mcMS**7)/(91*mbkin**9) + 
            (106867200*mcMS**7)/(91*mbkin**8) - (124106812*mcMS**8)/
                (165*mbkin**10) - (496427248*mcMS**8)/(297*mbkin**9) + 
            (13377856*mcMS**9)/(15*mbkin**11) + (53511424*mcMS**9)/
                (27*mbkin**10) - (4416016*mcMS**10)/(5*mbkin**12) - 
            (17664064*mcMS**10)/(9*mbkin**11) + (168649664*mcMS**11)/
                (231*mbkin**13) + (3372993280*mcMS**11)/(2079*mbkin**12) - 
            (502020*mcMS**12)/mbkin**14 - (1115600*mcMS**12)/mbkin**13 + 
            (1426048*mcMS**13)/(5*mbkin**15) + (5704192*mcMS**13)/(9*mbkin**14) - 
            (60257056*mcMS**14)/(455*mbkin**16) - (241028224*mcMS**14)/
                (819*mbkin**15) + (247477632*mcMS**15)/(5005*mbkin**17) + 
            (329970176*mcMS**15)/(3003*mbkin**16) - (188334*mcMS**16)/
                (13*mbkin**18) - (418520*mcMS**16)/(13*mbkin**17) + 
            (2127200*mcMS**17)/(663*mbkin**19) + (42544000*mcMS**17)/
                (5967*mbkin**18) - (11719928*mcMS**18)/(23205*mbkin**20) - 
            (46879712*mcMS**18)/(41769*mbkin**19) + (684256*mcMS**19)/
                (13585*mbkin**21) + (2737024*mcMS**19)/(24453*mbkin**20) - 
            (150698*mcMS**20)/(62985*mbkin**22) - (602792*mcMS**20)/
                (113373*mbkin**21) - (14120272*mcMS)/(88179*mbkin**3*
                (1 - mcMS/mbkin)) - (112962176*mcMS)/(264537*mbkin**2*
                (1 - mcMS/mbkin)) + (468025720*mcMS**2)/(264537*mbkin**4*
                (1 - mcMS/mbkin)) + (3744205760*mcMS**2)/(793611*mbkin**3*
                (1 - mcMS/mbkin)) - (107374328*mcMS**3)/(9945*mbkin**5*
                (1 - mcMS/mbkin)) - (858994624*mcMS**3)/(29835*mbkin**4*
                (1 - mcMS/mbkin)) + (12111649028*mcMS**4)/(255255*mbkin**6*
                (1 - mcMS/mbkin)) + (96893192224*mcMS**4)/(765765*mbkin**5*
                (1 - mcMS/mbkin)) - (804583036*mcMS**5)/(5005*mbkin**7*
                (1 - mcMS/mbkin)) - (6436664288*mcMS**5)/(15015*mbkin**6*
                (1 - mcMS/mbkin)) + (16770272*mcMS**6)/(39*mbkin**8*
                (1 - mcMS/mbkin)) + (134162176*mcMS**6)/(117*mbkin**7*
                (1 - mcMS/mbkin)) - (12012704*mcMS**7)/(13*mbkin**9*
                (1 - mcMS/mbkin)) - (96101632*mcMS**7)/(39*mbkin**8*
                (1 - mcMS/mbkin)) + (10420999024*mcMS**8)/(6435*mbkin**10*
                (1 - mcMS/mbkin)) + (83367992192*mcMS**8)/(19305*mbkin**9*
                (1 - mcMS/mbkin)) - (231726224*mcMS**9)/(99*mbkin**11*
                (1 - mcMS/mbkin)) - (1853809792*mcMS**9)/(297*mbkin**10*
                (1 - mcMS/mbkin)) + (42146864*mcMS**10)/(15*mbkin**12*
                (1 - mcMS/mbkin)) + (337174912*mcMS**10)/(45*mbkin**11*
                (1 - mcMS/mbkin)) - (177061168*mcMS**11)/(63*mbkin**13*
                (1 - mcMS/mbkin)) - (1416489344*mcMS**11)/(189*mbkin**12*
                (1 - mcMS/mbkin)) + (147579352*mcMS**12)/(63*mbkin**14*
                (1 - mcMS/mbkin)) + (1180634816*mcMS**12)/(189*mbkin**13*
                (1 - mcMS/mbkin)) - (24329912*mcMS**13)/(15*mbkin**15*
                (1 - mcMS/mbkin)) - (194639296*mcMS**13)/(45*mbkin**14*
                (1 - mcMS/mbkin)) + (60252704*mcMS**14)/(65*mbkin**16*
                (1 - mcMS/mbkin)) + (482021632*mcMS**14)/(195*mbkin**15*
                (1 - mcMS/mbkin)) - (6495875552*mcMS**15)/(15015*mbkin**17*
                (1 - mcMS/mbkin)) - (51967004416*mcMS**15)/(45045*mbkin**16*
                (1 - mcMS/mbkin)) + (162410064*mcMS**16)/(1001*mbkin**18*
                (1 - mcMS/mbkin)) + (433093504*mcMS**16)/(1001*mbkin**17*
                (1 - mcMS/mbkin)) - (5583616*mcMS**17)/(117*mbkin**19*
                (1 - mcMS/mbkin)) - (44668928*mcMS**17)/(351*mbkin**18*
                (1 - mcMS/mbkin)) + (738321352*mcMS**18)/(69615*mbkin**20*
                (1 - mcMS/mbkin)) + (5906570816*mcMS**18)/(208845*mbkin**19*
                (1 - mcMS/mbkin)) - (427470856*mcMS**19)/(255255*mbkin**21*
                (1 - mcMS/mbkin)) - (3419766848*mcMS**19)/(765765*mbkin**20*
                (1 - mcMS/mbkin)) + (348098812*mcMS**20)/(2078505*mbkin**22*
                (1 - mcMS/mbkin)) + (2784790496*mcMS**20)/(6235515*mbkin**21*
                (1 - mcMS/mbkin)) - (301396*mcMS**21)/(37791*mbkin**23*
                (1 - mcMS/mbkin)) - (2411168*mcMS**21)/(113373*mbkin**22*
                (1 - mcMS/mbkin)))*np.log(2/mus) + 
            (4529650481847077/13411222264440 + 15778736/(405405*mbkin**2) + 
            63114944/(729729*mbkin) - (12148554800*mcMS)/(26189163*mbkin**3) - 
            (8777037184*mcMS)/(8729721*mbkin**2) - (2409611921485918004*mcMS)/
                (796291321951125*mbkin) + (100089736*mcMS**2)/(34749*mbkin**4) + 
            (646098304*mcMS**2)/(104247*mbkin**3) + (131181854232014*mcMS**2)/
                (10672286625*mbkin**2) - (13292598896*mcMS**3)/(984555*mbkin**5) - 
            (258245697152*mcMS**3)/(8860995*mbkin**4) - 
            (7211141123397511*mcMS**3)/(214923433725*mbkin**3) + 
            (21018966764*mcMS**4)/(405405*mbkin**6) + (19561361056*mcMS**4)/
                (173745*mbkin**5) + (3322103239097477*mcMS**4)/(43345902600*
                mbkin**4) - (2538967072*mcMS**5)/(15795*mbkin**7) - 
            (1108328960*mcMS**5)/(3159*mbkin**6) - (296155042786888*mcMS**5)/
                (1915538625*mbkin**5) + (139918240*mcMS**6)/(351*mbkin**8) + 
            (8277677888*mcMS**6)/(9477*mbkin**7) + (183584247858181*mcMS**6)/
                (703667250*mbkin**6) - (1956154784*mcMS**7)/(2457*mbkin**9) - 
            (12895944704*mcMS**7)/(7371*mbkin**8) - (20563912883023*mcMS**7)/
                (58046625*mbkin**7) + (24224006368*mcMS**8)/(18711*mbkin**10) + 
            (800142123712*mcMS**8)/(280665*mbkin**9) + (859851409549*mcMS**8)/
                (2211300*mbkin**8) - (700068352*mcMS**9)/(405*mbkin**11) - 
            (13893870848*mcMS**9)/(3645*mbkin**10) - (34893257460322*mcMS**9)/
                (98513415*mbkin**9) + (257056048*mcMS**10)/(135*mbkin**12) + 
            (1021246400*mcMS**10)/(243*mbkin**11) + (9520712957158453*mcMS**10)/
                (34479695250*mbkin**10) - (3602891840*mcMS**11)/(2079*mbkin**13) - 
            (214822247168*mcMS**11)/(56133*mbkin**12) - 
            (1708860475928401*mcMS**11)/(9030396375*mbkin**11) + 
            (1158984296*mcMS**12)/(891*mbkin**14) + (23041697408*mcMS**12)/
                (8019*mbkin**13) + (9753751392280913*mcMS**12)/(84283699500*
                mbkin**12) - (46381886432*mcMS**13)/(57915*mbkin**15) - 
            (307417939456*mcMS**13)/(173745*mbkin**14) - 
            (401875960070852*mcMS**13)/(6403371975*mbkin**13) + 
            (487143200864*mcMS**14)/(1216215*mbkin**16) + 
            (1076326757696*mcMS**14)/(1216215*mbkin**15) + 
            (1071889963712831*mcMS**14)/(36522936450*mbkin**14) - 
            (3093431392*mcMS**15)/(19305*mbkin**17) - (258362278912*mcMS**15)/
                (729729*mbkin**16) - (28039427185820671*mcMS**15)/
                (2465298210375*mbkin**15) + (5859280*mcMS**16)/(117*mbkin**18) + 
            (38838656*mcMS**16)/(351*mbkin**17) + (3274919*mcMS**16)/
                (936*mbkin**16) - (37226000*mcMS**17)/(3159*mbkin**19) - 
            (246755200*mcMS**17)/(9477*mbkin**18) - (43740550*mcMS**17)/
                (53703*mbkin**17) + (11719928*mcMS**18)/(5967*mbkin**20) + 
            (2719023296*mcMS**18)/(626535*mbkin**19) + (33694793*mcMS**18)/
                (250614*mbkin**18) - (2394896*mcMS**19)/(11583*mbkin**21) - 
            (79373696*mcMS**19)/(173745*mbkin**20) - (812554*mcMS**19)/
                (57915*mbkin**19) + (10548860*mcMS**20)/(1020357*mbkin**22) + 
            (69923872*mcMS**20)/(3061071*mbkin**21) + (2185121*mcMS**20)/
                (3139560*mbkin**20) - (376528*mcMS**2)/(1053*mbkin**4*
                (1 - mcMS/mbkin)**2) - (3012224*mcMS**2)/(3159*mbkin**3*
                (1 - mcMS/mbkin)**2) + (753056*mcMS**2)/(1053*mbkin**2*
                (1 - mcMS/mbkin)**2) + (429497312*mcMS**3)/(89505*mbkin**5*
                (1 - mcMS/mbkin)**2) + (3435978496*mcMS**3)/(268515*mbkin**4*
                (1 - mcMS/mbkin)**2) - (858994624*mcMS**3)/(89505*mbkin**3*
                (1 - mcMS/mbkin)**2) - (234545329016*mcMS**4)/(6891885*mbkin**6*
                (1 - mcMS/mbkin)**2) - (1876362632128*mcMS**4)/(20675655*mbkin**5*
                (1 - mcMS/mbkin)**2) + (469090658032*mcMS**4)/(6891885*mbkin**4*
                (1 - mcMS/mbkin)**2) + (75343294960*mcMS**5)/(459459*mbkin**7*
                (1 - mcMS/mbkin)**2) + (602746359680*mcMS**5)/(1378377*mbkin**6*
                (1 - mcMS/mbkin)**2) - (150686589920*mcMS**5)/(459459*mbkin**5*
                (1 - mcMS/mbkin)**2) - (79048041848*mcMS**6)/(135135*mbkin**8*
                (1 - mcMS/mbkin)**2) - (632384334784*mcMS**6)/(405405*mbkin**7*
                (1 - mcMS/mbkin)**2) + (158096083696*mcMS**6)/(135135*mbkin**6*
                (1 - mcMS/mbkin)**2) + (566619520*mcMS**7)/(351*mbkin**9*
                (1 - mcMS/mbkin)**2) + (4532956160*mcMS**7)/(1053*mbkin**8*
                (1 - mcMS/mbkin)**2) - (1133239040*mcMS**7)/(351*mbkin**7*
                (1 - mcMS/mbkin)**2) - (205356871136*mcMS**8)/(57915*mbkin**10*
                (1 - mcMS/mbkin)**2) - (1642854969088*mcMS**8)/(173745*mbkin**9*
                (1 - mcMS/mbkin)**2) + (410713742272*mcMS**8)/(57915*mbkin**8*
                (1 - mcMS/mbkin)**2) + (366047261248*mcMS**9)/(57915*mbkin**11*
                (1 - mcMS/mbkin)**2) + (2928378089984*mcMS**9)/(173745*mbkin**10*
                (1 - mcMS/mbkin)**2) - (732094522496*mcMS**9)/(57915*mbkin**9*
                (1 - mcMS/mbkin)**2) - (41256072896*mcMS**10)/(4455*mbkin**12*
                (1 - mcMS/mbkin)**2) - (330048583168*mcMS**10)/(13365*mbkin**11*
                (1 - mcMS/mbkin)**2) + (82512145792*mcMS**10)/(4455*mbkin**10*
                (1 - mcMS/mbkin)**2) + (31867463104*mcMS**11)/(2835*mbkin**13*
                (1 - mcMS/mbkin)**2) + (254939704832*mcMS**11)/(8505*mbkin**12*
                (1 - mcMS/mbkin)**2) - (63734926208*mcMS**11)/(2835*mbkin**11*
                (1 - mcMS/mbkin)**2) - (6433846768*mcMS**12)/(567*mbkin**14*
                (1 - mcMS/mbkin)**2) - (51470774144*mcMS**12)/(1701*mbkin**13*
                (1 - mcMS/mbkin)**2) + (12867693536*mcMS**12)/(567*mbkin**12*
                (1 - mcMS/mbkin)**2) + (27020210848*mcMS**13)/(2835*mbkin**15*
                (1 - mcMS/mbkin)**2) + (216161686784*mcMS**13)/(8505*mbkin**14*
                (1 - mcMS/mbkin)**2) - (54040421696*mcMS**13)/(2835*mbkin**13*
                (1 - mcMS/mbkin)**2) - (896774288*mcMS**14)/(135*mbkin**16*
                (1 - mcMS/mbkin)**2) - (7174194304*mcMS**14)/(405*mbkin**15*
                (1 - mcMS/mbkin)**2) + (1793548576*mcMS**14)/(135*mbkin**14*
                (1 - mcMS/mbkin)**2) + (73703643776*mcMS**15)/(19305*mbkin**17*
                (1 - mcMS/mbkin)**2) + (589629150208*mcMS**15)/(57915*mbkin**16*
                (1 - mcMS/mbkin)**2) - (147407287552*mcMS**15)/(19305*mbkin**15*
                (1 - mcMS/mbkin)**2) - (34568184736*mcMS**16)/(19305*mbkin**18*
                (1 - mcMS/mbkin)**2) - (276545477888*mcMS**16)/(57915*mbkin**17*
                (1 - mcMS/mbkin)**2) + (69136369472*mcMS**16)/(19305*mbkin**16*
                (1 - mcMS/mbkin)**2) + (7812195136*mcMS**17)/(11583*mbkin**19*
                (1 - mcMS/mbkin)**2) + (62497561088*mcMS**17)/(34749*mbkin**18*
                (1 - mcMS/mbkin)**2) - (15624390272*mcMS**17)/(11583*mbkin**17*
                (1 - mcMS/mbkin)**2) - (7339439504*mcMS**18)/(36855*mbkin**20*
                (1 - mcMS/mbkin)**2) - (58715516032*mcMS**18)/(110565*mbkin**19*
                (1 - mcMS/mbkin)**2) + (14678879008*mcMS**18)/(36855*mbkin**18*
                (1 - mcMS/mbkin)**2) + (18003292256*mcMS**19)/(405405*mbkin**21*
                (1 - mcMS/mbkin)**2) + (144026338048*mcMS**19)/(1216215*mbkin**20*
                (1 - mcMS/mbkin)**2) - (36006584512*mcMS**19)/(405405*mbkin**19*
                (1 - mcMS/mbkin)**2) - (9695082136*mcMS**20)/(1378377*mbkin**22*
                (1 - mcMS/mbkin)**2) - (77560657088*mcMS**20)/(4135131*mbkin**21*
                (1 - mcMS/mbkin)**2) + (19390164272*mcMS**20)/(1378377*mbkin**20*
                (1 - mcMS/mbkin)**2) + (694454128*mcMS**21)/(984555*mbkin**23*
                (1 - mcMS/mbkin)**2) + (5555633024*mcMS**21)/(2953665*mbkin**22*
                (1 - mcMS/mbkin)**2) - (1388908256*mcMS**21)/(984555*mbkin**21*
                (1 - mcMS/mbkin)**2) - (602792*mcMS**22)/(17901*mbkin**24*
                (1 - mcMS/mbkin)**2) - (4822336*mcMS**22)/(53703*mbkin**23*
                (1 - mcMS/mbkin)**2) + (1205584*mcMS**22)/(17901*mbkin**22*
                (1 - mcMS/mbkin)**2) + (70601360*mcMS)/(2380833*mbkin**3*
                (1 - mcMS/mbkin)) + (564810880*mcMS)/(7142499*mbkin**2*
                (1 - mcMS/mbkin)) - (732489110*mcMS)/(2380833*mbkin*
                (1 - mcMS/mbkin)) - (2340128600*mcMS**2)/(7142499*mbkin**4*
                (1 - mcMS/mbkin)) - (18721028800*mcMS**2)/(21427497*mbkin**3*
                (1 - mcMS/mbkin)) + (24278834225*mcMS**2)/(7142499*mbkin**2*
                (1 - mcMS/mbkin)) + (107374328*mcMS**3)/(53703*mbkin**5*
                (1 - mcMS/mbkin)) + (858994624*mcMS**3)/(161109*mbkin**4*
                (1 - mcMS/mbkin)) - (1114008653*mcMS**3)/(53703*mbkin**3*
                (1 - mcMS/mbkin)) - (12111649028*mcMS**4)/(1378377*mbkin**6*
                (1 - mcMS/mbkin)) - (96893192224*mcMS**4)/(4135131*mbkin**5*
                (1 - mcMS/mbkin)) + (251316717331*mcMS**4)/(2756754*mbkin**4*
                (1 - mcMS/mbkin)) + (804583036*mcMS**5)/(27027*mbkin**7*
                (1 - mcMS/mbkin)) + (6436664288*mcMS**5)/(81081*mbkin**6*
                (1 - mcMS/mbkin)) - (16695097997*mcMS**5)/(54054*mbkin**5*
                (1 - mcMS/mbkin)) - (83851360*mcMS**6)/(1053*mbkin**8*
                (1 - mcMS/mbkin)) - (670810880*mcMS**6)/(3159*mbkin**7*
                (1 - mcMS/mbkin)) + (869957860*mcMS**6)/(1053*mbkin**6*
                (1 - mcMS/mbkin)) + (60063520*mcMS**7)/(351*mbkin**9*
                (1 - mcMS/mbkin)) + (480508160*mcMS**7)/(1053*mbkin**8*
                (1 - mcMS/mbkin)) - (623159020*mcMS**7)/(351*mbkin**7*
                (1 - mcMS/mbkin)) - (10420999024*mcMS**8)/(34749*mbkin**10*
                (1 - mcMS/mbkin)) - (83367992192*mcMS**8)/(104247*mbkin**9*
                (1 - mcMS/mbkin)) + (108117864874*mcMS**8)/(34749*mbkin**8*
                (1 - mcMS/mbkin)) + (1158631120*mcMS**9)/(2673*mbkin**11*
                (1 - mcMS/mbkin)) + (9269048960*mcMS**9)/(8019*mbkin**10*
                (1 - mcMS/mbkin)) - (12020797870*mcMS**9)/(2673*mbkin**9*
                (1 - mcMS/mbkin)) - (42146864*mcMS**10)/(81*mbkin**12*
                (1 - mcMS/mbkin)) - (337174912*mcMS**10)/(243*mbkin**11*
                (1 - mcMS/mbkin)) + (437273714*mcMS**10)/(81*mbkin**10*
                (1 - mcMS/mbkin)) + (885305840*mcMS**11)/(1701*mbkin**13*
                (1 - mcMS/mbkin)) + (7082446720*mcMS**11)/(5103*mbkin**12*
                (1 - mcMS/mbkin)) - (9185048090*mcMS**11)/(1701*mbkin**11*
                (1 - mcMS/mbkin)) - (737896760*mcMS**12)/(1701*mbkin**14*
                (1 - mcMS/mbkin)) - (5903174080*mcMS**12)/(5103*mbkin**13*
                (1 - mcMS/mbkin)) + (7655678885*mcMS**12)/(1701*mbkin**12*
                (1 - mcMS/mbkin)) + (24329912*mcMS**13)/(81*mbkin**15*
                (1 - mcMS/mbkin)) + (194639296*mcMS**13)/(243*mbkin**14*
                (1 - mcMS/mbkin)) - (252422837*mcMS**13)/(81*mbkin**13*
                (1 - mcMS/mbkin)) - (60252704*mcMS**14)/(351*mbkin**16*
                (1 - mcMS/mbkin)) - (482021632*mcMS**14)/(1053*mbkin**15*
                (1 - mcMS/mbkin)) + (625121804*mcMS**14)/(351*mbkin**14*
                (1 - mcMS/mbkin)) + (6495875552*mcMS**15)/(81081*mbkin**17*
                (1 - mcMS/mbkin)) + (51967004416*mcMS**15)/(243243*mbkin**16*
                (1 - mcMS/mbkin)) - (67394708852*mcMS**15)/(81081*mbkin**15*
                (1 - mcMS/mbkin)) - (270683440*mcMS**16)/(9009*mbkin**18*
                (1 - mcMS/mbkin)) - (2165467520*mcMS**16)/(27027*mbkin**17*
                (1 - mcMS/mbkin)) + (2808340690*mcMS**16)/(9009*mbkin**16*
                (1 - mcMS/mbkin)) + (27918080*mcMS**17)/(3159*mbkin**19*
                (1 - mcMS/mbkin)) + (223344640*mcMS**17)/(9477*mbkin**18*
                (1 - mcMS/mbkin)) - (289650080*mcMS**17)/(3159*mbkin**17*
                (1 - mcMS/mbkin)) - (738321352*mcMS**18)/(375921*mbkin**20*
                (1 - mcMS/mbkin)) - (5906570816*mcMS**18)/(1127763*mbkin**19*
                (1 - mcMS/mbkin)) + (7660084027*mcMS**18)/(375921*mbkin**18*
                (1 - mcMS/mbkin)) + (427470856*mcMS**19)/(1378377*mbkin**21*
                (1 - mcMS/mbkin)) + (3419766848*mcMS**19)/(4135131*mbkin**20*
                (1 - mcMS/mbkin)) - (4435010131*mcMS**19)/(1378377*mbkin**19*
                (1 - mcMS/mbkin)) - (348098812*mcMS**20)/(11223927*mbkin**22*
                (1 - mcMS/mbkin)) - (2784790496*mcMS**20)/(33671781*mbkin**21*
                (1 - mcMS/mbkin)) + (7223050349*mcMS**20)/(22447854*mbkin**20*
                (1 - mcMS/mbkin)) + (1506980*mcMS**21)/(1020357*mbkin**23*
                (1 - mcMS/mbkin)) + (12055840*mcMS**21)/(3061071*mbkin**22*
                (1 - mcMS/mbkin)) - (31269835*mcMS**21)/(2040714*mbkin**21*
                (1 - mcMS/mbkin)) - (19503479*np.pi**2)/2432430 + 
            (295652*mcMS*np.pi**2)/(3861*mbkin) - (6412711*mcMS**2*np.pi**2)/
                (19305*mbkin**2) + (5424236*mcMS**3*np.pi**2)/(6237*mbkin**3) - 
            (120770513*mcMS**4*np.pi**2)/(62370*mbkin**4) + (6196072*mcMS**5*np.pi**2)/
                (1701*mbkin**5) - (48956219*mcMS**6*np.pi**2)/(8505*mbkin**6) + 
            (6930296*mcMS**7*np.pi**2)/(945*mbkin**7) - (8189821*mcMS**8*np.pi**2)/
                (1134*mbkin**8) + (1342636*mcMS**9*np.pi**2)/(243*mbkin**9) - 
            (3101377*mcMS**10*np.pi**2)/(945*mbkin**10) + (15420836*mcMS**11*np.pi**2)/
                (10395*mbkin**11) - (379667*mcMS**12*np.pi**2)/(770*mbkin**12) + 
            (9213440*mcMS**13*np.pi**2)/(81081*mbkin**13) - (19729819*mcMS**14*np.pi**2)/
                (1216215*mbkin**14) + (1314496*mcMS**15*np.pi**2)/(1216215*mbkin**15) + 
            ((748886504*mcMS)/(8729721*mbkin) - (7937476*mcMS**2)/
                (11583*mbkin**2) + (950785096*mcMS**3)/(328185*mbkin**3) - 
                (1198844138*mcMS**4)/(135135*mbkin**4) + (22613456*mcMS**5)/
                (1053*mbkin**5) - (14677064*mcMS**6)/(351*mbkin**6) + 
                (7772528*mcMS**7)/(117*mbkin**7) - (2746783208*mcMS**8)/
                (31185*mbkin**8) + (4479008*mcMS**9)/(45*mbkin**9) - 
                (872224*mcMS**10)/(9*mbkin**10) + (15355264*mcMS**11)/
                (189*mbkin**11) - (17248564*mcMS**12)/(297*mbkin**12) + 
                (51768368*mcMS**13)/(1485*mbkin**13) - (998628536*mcMS**14)/
                (57915*mbkin**14) + (185744752*mcMS**15)/(27027*mbkin**15) - 
                (83704*mcMS**16)/(39*mbkin**16) + (531800*mcMS**17)/
                (1053*mbkin**17) - (5859964*mcMS**18)/(69615*mbkin**18) + 
                (171064*mcMS**19)/(19305*mbkin**19) - (150698*mcMS**20)/
                (340119*mbkin**20) - (188264*mcMS**2)/(351*mbkin**2*
                (1 - mcMS/mbkin)**2) + (214748656*mcMS**3)/(29835*mbkin**3*
                (1 - mcMS/mbkin)**2) - (117272664508*mcMS**4)/(2297295*mbkin**
                    4*(1 - mcMS/mbkin)**2) + (37671647480*mcMS**5)/(153153*mbkin**
                    5*(1 - mcMS/mbkin)**2) - (39524020924*mcMS**6)/(45045*mbkin**
                    6*(1 - mcMS/mbkin)**2) + (283309760*mcMS**7)/(117*mbkin**7*
                (1 - mcMS/mbkin)**2) - (102678435568*mcMS**8)/(19305*mbkin**
                    8*(1 - mcMS/mbkin)**2) + (183023630624*mcMS**9)/
                (19305*mbkin**9*(1 - mcMS/mbkin)**2) - (20628036448*mcMS**10)/
                (1485*mbkin**10*(1 - mcMS/mbkin)**2) + (15933731552*mcMS**11)/
                (945*mbkin**11*(1 - mcMS/mbkin)**2) - (3216923384*mcMS**12)/
                (189*mbkin**12*(1 - mcMS/mbkin)**2) + (13510105424*mcMS**13)/
                (945*mbkin**13*(1 - mcMS/mbkin)**2) - (448387144*mcMS**14)/
                (45*mbkin**14*(1 - mcMS/mbkin)**2) + (36851821888*mcMS**15)/
                (6435*mbkin**15*(1 - mcMS/mbkin)**2) - (17284092368*mcMS**16)/
                (6435*mbkin**16*(1 - mcMS/mbkin)**2) + (3906097568*mcMS**17)/
                (3861*mbkin**17*(1 - mcMS/mbkin)**2) - (3669719752*mcMS**18)/
                (12285*mbkin**18*(1 - mcMS/mbkin)**2) + (9001646128*mcMS**19)/
                (135135*mbkin**19*(1 - mcMS/mbkin)**2) - (4847541068*mcMS**20)/
                (459459*mbkin**20*(1 - mcMS/mbkin)**2) + (347227064*mcMS**21)/
                (328185*mbkin**21*(1 - mcMS/mbkin)**2) - (301396*mcMS**22)/
                (5967*mbkin**22*(1 - mcMS/mbkin)**2) + (130612516*mcMS)/
                (793611*mbkin*(1 - mcMS/mbkin)) - (4329237910*mcMS**2)/
                (2380833*mbkin**2*(1 - mcMS/mbkin)) + (993212534*mcMS**3)/
                (89505*mbkin**3*(1 - mcMS/mbkin)) - (112032753509*mcMS**4)/
                (2297295*mbkin**4*(1 - mcMS/mbkin)) + (7442393083*mcMS**5)/
                (45045*mbkin**5*(1 - mcMS/mbkin)) - (155125016*mcMS**6)/
                (351*mbkin**6*(1 - mcMS/mbkin)) + (111117512*mcMS**7)/
                (117*mbkin**7*(1 - mcMS/mbkin)) - (96394240972*mcMS**8)/
                (57915*mbkin**8*(1 - mcMS/mbkin)) + (2143467572*mcMS**9)/
                (891*mbkin**9*(1 - mcMS/mbkin)) - (389858492*mcMS**10)/
                (135*mbkin**10*(1 - mcMS/mbkin)) + (1637815804*mcMS**11)/
                (567*mbkin**11*(1 - mcMS/mbkin)) - (1365109006*mcMS**12)/
                (567*mbkin**12*(1 - mcMS/mbkin)) + (225051686*mcMS**13)/
                (135*mbkin**13*(1 - mcMS/mbkin)) - (557337512*mcMS**14)/
                (585*mbkin**14*(1 - mcMS/mbkin)) + (60086848856*mcMS**15)/
                (135135*mbkin**15*(1 - mcMS/mbkin)) - (500764364*mcMS**16)/
                (3003*mbkin**16*(1 - mcMS/mbkin)) + (51648448*mcMS**17)/
                (1053*mbkin**17*(1 - mcMS/mbkin)) - (6829472506*mcMS**18)/
                (626535*mbkin**18*(1 - mcMS/mbkin)) + (3954105418*mcMS**19)/
                (2297295*mbkin**19*(1 - mcMS/mbkin)) - (3219914011*mcMS**20)/
                (18706545*mbkin**20*(1 - mcMS/mbkin)) + (2787913*mcMS**21)/
                (340119*mbkin**21*(1 - mcMS/mbkin)) - (392*mcMS**4*np.pi**2)/
                (9*mbkin**4))*np.log(mcMS**2/mbkin**2) + (-70818088/1658475 + 
                (9850594912*mcMS)/(54729675*mbkin) + (1514031224*mcMS**2)/
                (2606175*mbkin**2) - (1032367648*mcMS**3)/(168399*mbkin**3) + 
                (2610608936*mcMS**4)/(120285*mbkin**4) - (2030028992*mcMS**5)/
                (42525*mbkin**5) + (4173690776*mcMS**6)/(54675*mbkin**6) - 
                (575564672*mcMS**7)/(6075*mbkin**7) + (37691576*mcMS**8)/
                (405*mbkin**8) - (87483296*mcMS**9)/(1215*mbkin**9) + 
                (3308018264*mcMS**10)/(76545*mbkin**10) - (358883296*mcMS**11)/
                (18225*mbkin**11) + (27730438888*mcMS**12)/(4209975*mbkin**12) - 
                (2386094336*mcMS**13)/(1563705*mbkin**13) + (88704184*mcMS**14)/
                (405405*mbkin**14) - (800901952*mcMS**15)/(54729675*mbkin**15))*
                np.log(1 - mcMS/mbkin))*np.log(mus**2/mbkin**2) + 
            (2093435917/104756652 - (776844436*mcMS)/(3357585*mbkin) + 
            (40836175*mcMS**2)/(34749*mbkin**2) - (11697739628*mcMS**3)/
                (2953665*mbkin**3) + (17282295109*mcMS**4)/(1621620*mbkin**4) - 
            (379688576*mcMS**5)/(15795*mbkin**5) + (710791064*mcMS**6)/
                (15795*mbkin**6) - (171522592*mcMS**7)/(2457*mbkin**7) + 
            (17059558777*mcMS**8)/(187110*mbkin**8) - (124023536*mcMS**9)/
                (1215*mbkin**9) + (40031806*mcMS**10)/(405*mbkin**10) - 
            (367940368*mcMS**11)/(4455*mbkin**11) + (314961001*mcMS**12)/
                (5346*mbkin**12) - (2045244112*mcMS**13)/(57915*mbkin**13) + 
            (1632457936*mcMS**14)/(93555*mbkin**14) - (241470128*mcMS**15)/
                (34749*mbkin**15) + (1014911*mcMS**16)/(468*mbkin**16) - 
            (27387700*mcMS**17)/(53703*mbkin**17) + (159684019*mcMS**18)/
                (1879605*mbkin**18) - (1967236*mcMS**19)/(220077*mbkin**19) + 
            (9117229*mcMS**20)/(20407140*mbkin**20) + (94132*mcMS**2)/
                (351*mbkin**2*(1 - mcMS/mbkin)**2) - (107374328*mcMS**3)/
                (29835*mbkin**3*(1 - mcMS/mbkin)**2) + (58636332254*mcMS**4)/
                (2297295*mbkin**4*(1 - mcMS/mbkin)**2) - (18835823740*mcMS**5)/
                (153153*mbkin**5*(1 - mcMS/mbkin)**2) + (19762010462*mcMS**6)/
                (45045*mbkin**6*(1 - mcMS/mbkin)**2) - (141654880*mcMS**7)/
                (117*mbkin**7*(1 - mcMS/mbkin)**2) + (51339217784*mcMS**8)/
                (19305*mbkin**8*(1 - mcMS/mbkin)**2) - (91511815312*mcMS**9)/
                (19305*mbkin**9*(1 - mcMS/mbkin)**2) + (10314018224*mcMS**10)/
                (1485*mbkin**10*(1 - mcMS/mbkin)**2) - (7966865776*mcMS**11)/
                (945*mbkin**11*(1 - mcMS/mbkin)**2) + (1608461692*mcMS**12)/
                (189*mbkin**12*(1 - mcMS/mbkin)**2) - (6755052712*mcMS**13)/
                (945*mbkin**13*(1 - mcMS/mbkin)**2) + (224193572*mcMS**14)/
                (45*mbkin**14*(1 - mcMS/mbkin)**2) - (18425910944*mcMS**15)/
                (6435*mbkin**15*(1 - mcMS/mbkin)**2) + (8642046184*mcMS**16)/
                (6435*mbkin**16*(1 - mcMS/mbkin)**2) - (1953048784*mcMS**17)/
                (3861*mbkin**17*(1 - mcMS/mbkin)**2) + (1834859876*mcMS**18)/
                (12285*mbkin**18*(1 - mcMS/mbkin)**2) - (4500823064*mcMS**19)/
                (135135*mbkin**19*(1 - mcMS/mbkin)**2) + (2423770534*mcMS**20)/
                (459459*mbkin**20*(1 - mcMS/mbkin)**2) - (173613532*mcMS**21)/
                (328185*mbkin**21*(1 - mcMS/mbkin)**2) + (150698*mcMS**22)/
                (5967*mbkin**22*(1 - mcMS/mbkin)**2) - (65306258*mcMS)/
                (793611*mbkin*(1 - mcMS/mbkin)) + (2164618955*mcMS**2)/
                (2380833*mbkin**2*(1 - mcMS/mbkin)) - (496606267*mcMS**3)/
                (89505*mbkin**3*(1 - mcMS/mbkin)) + (112032753509*mcMS**4)/
                (4594590*mbkin**4*(1 - mcMS/mbkin)) - (7442393083*mcMS**5)/
                (90090*mbkin**5*(1 - mcMS/mbkin)) + (77562508*mcMS**6)/
                (351*mbkin**6*(1 - mcMS/mbkin)) - (55558756*mcMS**7)/
                (117*mbkin**7*(1 - mcMS/mbkin)) + (48197120486*mcMS**8)/
                (57915*mbkin**8*(1 - mcMS/mbkin)) - (1071733786*mcMS**9)/
                (891*mbkin**9*(1 - mcMS/mbkin)) + (194929246*mcMS**10)/
                (135*mbkin**10*(1 - mcMS/mbkin)) - (818907902*mcMS**11)/
                (567*mbkin**11*(1 - mcMS/mbkin)) + (682554503*mcMS**12)/
                (567*mbkin**12*(1 - mcMS/mbkin)) - (112525843*mcMS**13)/
                (135*mbkin**13*(1 - mcMS/mbkin)) + (278668756*mcMS**14)/
                (585*mbkin**14*(1 - mcMS/mbkin)) - (30043424428*mcMS**15)/
                (135135*mbkin**15*(1 - mcMS/mbkin)) + (250382182*mcMS**16)/
                (3003*mbkin**16*(1 - mcMS/mbkin)) - (25824224*mcMS**17)/
                (1053*mbkin**17*(1 - mcMS/mbkin)) + (3414736253*mcMS**18)/
                (626535*mbkin**18*(1 - mcMS/mbkin)) - (1977052709*mcMS**19)/
                (2297295*mbkin**19*(1 - mcMS/mbkin)) + (3219914011*mcMS**20)/
                (37413090*mbkin**20*(1 - mcMS/mbkin)) - (2787913*mcMS**21)/
                (680238*mbkin**21*(1 - mcMS/mbkin)))*np.log(mus**2/mbkin**2)**2 + 
            np.log(mu0**2/mus**2)*((-56481088*mcMS)/(264537*mbkin**3) - 
            (1129621760*mcMS)/(2380833*mbkin**2) - (333324989499253*mcMS)/
                (547844046750*mbkin) + (753056*mcMS**2)/(351*mbkin**4) + 
            (15061120*mcMS**2)/(3159*mbkin**3) + (15478417721719*mcMS**2)/
                (3557428875*mbkin**2) - (13536576*mcMS**3)/(1105*mbkin**5) - 
            (6016256*mcMS**3)/(221*mbkin**4) - (683837705723971*mcMS**3)/
                (42141849750*mbkin**3) + (2297985232*mcMS**4)/(45045*mbkin**6) + 
            (9191940928*mcMS**4)/(81081*mbkin**5) + (561305364499738*mcMS**4)/
                (12642554925*mbkin**4) - (19109120*mcMS**5)/(117*mbkin**7) - 
            (382182400*mcMS**5)/(1053*mbkin**6) - (70552906019*mcMS**5)/
                (727650*mbkin**5) + (15990656*mcMS**6)/(39*mbkin**8) + 
            (319813120*mcMS**6)/(351*mbkin**7) + (45457380358813*mcMS**6)/
                (273648375*mbkin**6) - (10686720*mcMS**7)/(13*mbkin**9) - 
            (71244800*mcMS**7)/(39*mbkin**8) - (119765484898127*mcMS**7)/
                (547296750*mbkin**7) + (1985708992*mcMS**8)/(1485*mbkin**10) + 
            (7942835968*mcMS**8)/(2673*mbkin**9) + (47118708753404*mcMS**8)/
                (212837625*mbkin**8) - (26755712*mcMS**9)/(15*mbkin**11) - 
            (107022848*mcMS**9)/(27*mbkin**10) - (6305059020913*mcMS**9)/
                (36486450*mbkin**9) + (17664064*mcMS**10)/(9*mbkin**12) + 
            (353281280*mcMS**10)/(81*mbkin**11) + (7955554444931*mcMS**10)/
                (76621545*mbkin**10) - (337299328*mcMS**11)/(189*mbkin**13) - 
            (6745986560*mcMS**11)/(1701*mbkin**12) - (181064392519811*mcMS**11)/
                (3831077250*mbkin**11) + (1338720*mcMS**12)/mbkin**14 + 
            (8924800*mcMS**12)/(3*mbkin**13) + (332372802464726*mcMS**12)/
                (21070924875*mbkin**12) - (37077248*mcMS**13)/(45*mbkin**15) - 
            (148308992*mcMS**13)/(81*mbkin**14) - (5983215661469*mcMS**13)/
                (1641890250*mbkin**13) + (241028224*mcMS**14)/(585*mbkin**16) + 
            (964112896*mcMS**14)/(1053*mbkin**15) + (194192359957*mcMS**14)/
                (372683025*mbkin**14) - (164985088*mcMS**15)/(1001*mbkin**17) - 
            (3299701760*mcMS**15)/(9009*mbkin**16) - (3809167143809*mcMS**15)/
                (109568809350*mbkin**15) + (669632*mcMS**16)/(13*mbkin**18) + 
            (13392640*mcMS**16)/(117*mbkin**17) - (4254400*mcMS**17)/
                (351*mbkin**19) - (85088000*mcMS**17)/(3159*mbkin**18) + 
            (46879712*mcMS**18)/(23205*mbkin**20) + (187518848*mcMS**18)/
                (41769*mbkin**19) - (1368512*mcMS**19)/(6435*mbkin**21) - 
            (5474048*mcMS**19)/(11583*mbkin**20) + (1205584*mcMS**20)/
                (113373*mbkin**22) + (24111680*mcMS**20)/(1020357*mbkin**21) - 
            (376528*mcMS**2)/(1053*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (3012224*mcMS**2)/(3159*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (753056*mcMS**2)/(1053*mbkin**2*(1 - mcMS/mbkin)**2) + 
            (429497312*mcMS**3)/(89505*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (3435978496*mcMS**3)/(268515*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (858994624*mcMS**3)/(89505*mbkin**3*(1 - mcMS/mbkin)**2) - 
            (234545329016*mcMS**4)/(6891885*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (1876362632128*mcMS**4)/(20675655*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (469090658032*mcMS**4)/(6891885*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (75343294960*mcMS**5)/(459459*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (602746359680*mcMS**5)/(1378377*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (150686589920*mcMS**5)/(459459*mbkin**5*(1 - mcMS/mbkin)**2) - 
            (79048041848*mcMS**6)/(135135*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (632384334784*mcMS**6)/(405405*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (158096083696*mcMS**6)/(135135*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (566619520*mcMS**7)/(351*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (4532956160*mcMS**7)/(1053*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (1133239040*mcMS**7)/(351*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (205356871136*mcMS**8)/(57915*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (1642854969088*mcMS**8)/(173745*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (410713742272*mcMS**8)/(57915*mbkin**8*(1 - mcMS/mbkin)**2) + 
            (366047261248*mcMS**9)/(57915*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (2928378089984*mcMS**9)/(173745*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (732094522496*mcMS**9)/(57915*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (41256072896*mcMS**10)/(4455*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (330048583168*mcMS**10)/(13365*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (82512145792*mcMS**10)/(4455*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (31867463104*mcMS**11)/(2835*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (254939704832*mcMS**11)/(8505*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (63734926208*mcMS**11)/(2835*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (6433846768*mcMS**12)/(567*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (51470774144*mcMS**12)/(1701*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (12867693536*mcMS**12)/(567*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (27020210848*mcMS**13)/(2835*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (216161686784*mcMS**13)/(8505*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (54040421696*mcMS**13)/(2835*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (896774288*mcMS**14)/(135*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (7174194304*mcMS**14)/(405*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (1793548576*mcMS**14)/(135*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (73703643776*mcMS**15)/(19305*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (589629150208*mcMS**15)/(57915*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (147407287552*mcMS**15)/(19305*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (34568184736*mcMS**16)/(19305*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (276545477888*mcMS**16)/(57915*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (69136369472*mcMS**16)/(19305*mbkin**16*(1 - mcMS/mbkin)**2) + 
            (7812195136*mcMS**17)/(11583*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (62497561088*mcMS**17)/(34749*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (15624390272*mcMS**17)/(11583*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (7339439504*mcMS**18)/(36855*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (58715516032*mcMS**18)/(110565*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (14678879008*mcMS**18)/(36855*mbkin**18*(1 - mcMS/mbkin)**2) + 
            (18003292256*mcMS**19)/(405405*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (144026338048*mcMS**19)/(1216215*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (36006584512*mcMS**19)/(405405*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (9695082136*mcMS**20)/(1378377*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (77560657088*mcMS**20)/(4135131*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (19390164272*mcMS**20)/(1378377*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (694454128*mcMS**21)/(984555*mbkin**23*(1 - mcMS/mbkin)**2) + 
            (5555633024*mcMS**21)/(2953665*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (1388908256*mcMS**21)/(984555*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (602792*mcMS**22)/(17901*mbkin**24*(1 - mcMS/mbkin)**2) - 
            (4822336*mcMS**22)/(53703*mbkin**23*(1 - mcMS/mbkin)**2) + 
            (1205584*mcMS**22)/(17901*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (28240544*mcMS)/(793611*mbkin**3*(1 - mcMS/mbkin)) + 
            (225924352*mcMS)/(2380833*mbkin**2*(1 - mcMS/mbkin)) - 
            (379482310*mcMS)/(2380833*mbkin*(1 - mcMS/mbkin)) - 
            (936051440*mcMS**2)/(2380833*mbkin**4*(1 - mcMS/mbkin)) - 
            (7488411520*mcMS**2)/(7142499*mbkin**3*(1 - mcMS/mbkin)) + 
            (12578191225*mcMS**2)/(7142499*mbkin**2*(1 - mcMS/mbkin)) + 
            (214748656*mcMS**3)/(89505*mbkin**5*(1 - mcMS/mbkin)) + 
            (1717989248*mcMS**3)/(268515*mbkin**4*(1 - mcMS/mbkin)) - 
            (577137013*mcMS**3)/(53703*mbkin**3*(1 - mcMS/mbkin)) - 
            (24223298056*mcMS**4)/(2297295*mbkin**6*(1 - mcMS/mbkin)) - 
            (193786384448*mcMS**4)/(6891885*mbkin**5*(1 - mcMS/mbkin)) + 
            (130200227051*mcMS**4)/(2756754*mbkin**4*(1 - mcMS/mbkin)) + 
            (1609166072*mcMS**5)/(45045*mbkin**7*(1 - mcMS/mbkin)) + 
            (12873328576*mcMS**5)/(135135*mbkin**6*(1 - mcMS/mbkin)) - 
            (8649267637*mcMS**5)/(54054*mbkin**5*(1 - mcMS/mbkin)) - 
            (33540544*mcMS**6)/(351*mbkin**8*(1 - mcMS/mbkin)) - 
            (268324352*mcMS**6)/(1053*mbkin**7*(1 - mcMS/mbkin)) + 
            (450701060*mcMS**6)/(1053*mbkin**6*(1 - mcMS/mbkin)) + 
            (24025408*mcMS**7)/(117*mbkin**9*(1 - mcMS/mbkin)) + 
            (192203264*mcMS**7)/(351*mbkin**8*(1 - mcMS/mbkin)) - 
            (322841420*mcMS**7)/(351*mbkin**7*(1 - mcMS/mbkin)) - 
            (20841998048*mcMS**8)/(57915*mbkin**10*(1 - mcMS/mbkin)) - 
            (166735984384*mcMS**8)/(173745*mbkin**9*(1 - mcMS/mbkin)) + 
            (56012869754*mcMS**8)/(34749*mbkin**8*(1 - mcMS/mbkin)) + 
            (463452448*mcMS**9)/(891*mbkin**11*(1 - mcMS/mbkin)) + 
            (3707619584*mcMS**9)/(2673*mbkin**10*(1 - mcMS/mbkin)) - 
            (6227642270*mcMS**9)/(2673*mbkin**9*(1 - mcMS/mbkin)) - 
            (84293728*mcMS**10)/(135*mbkin**12*(1 - mcMS/mbkin)) - 
            (674349824*mcMS**10)/(405*mbkin**11*(1 - mcMS/mbkin)) + 
            (226539394*mcMS**10)/(81*mbkin**10*(1 - mcMS/mbkin)) + 
            (354122336*mcMS**11)/(567*mbkin**13*(1 - mcMS/mbkin)) + 
            (2832978688*mcMS**11)/(1701*mbkin**12*(1 - mcMS/mbkin)) - 
            (4758518890*mcMS**11)/(1701*mbkin**11*(1 - mcMS/mbkin)) - 
            (295158704*mcMS**12)/(567*mbkin**14*(1 - mcMS/mbkin)) - 
            (2361269632*mcMS**12)/(1701*mbkin**13*(1 - mcMS/mbkin)) + 
            (3966195085*mcMS**12)/(1701*mbkin**12*(1 - mcMS/mbkin)) + 
            (48659824*mcMS**13)/(135*mbkin**15*(1 - mcMS/mbkin)) + 
            (389278592*mcMS**13)/(405*mbkin**14*(1 - mcMS/mbkin)) - 
            (130773277*mcMS**13)/(81*mbkin**13*(1 - mcMS/mbkin)) - 
            (120505408*mcMS**14)/(585*mbkin**16*(1 - mcMS/mbkin)) - 
            (964043264*mcMS**14)/(1755*mbkin**15*(1 - mcMS/mbkin)) + 
            (323858284*mcMS**14)/(351*mbkin**14*(1 - mcMS/mbkin)) + 
            (12991751104*mcMS**15)/(135135*mbkin**17*(1 - mcMS/mbkin)) + 
            (103934008832*mcMS**15)/(405405*mbkin**16*(1 - mcMS/mbkin)) - 
            (34915331092*mcMS**15)/(81081*mbkin**15*(1 - mcMS/mbkin)) - 
            (108273376*mcMS**16)/(3003*mbkin**18*(1 - mcMS/mbkin)) - 
            (866187008*mcMS**16)/(9009*mbkin**17*(1 - mcMS/mbkin)) + 
            (1454923490*mcMS**16)/(9009*mbkin**16*(1 - mcMS/mbkin)) + 
            (11167232*mcMS**17)/(1053*mbkin**19*(1 - mcMS/mbkin)) + 
            (89337856*mcMS**17)/(3159*mbkin**18*(1 - mcMS/mbkin)) - 
            (150059680*mcMS**17)/(3159*mbkin**17*(1 - mcMS/mbkin)) - 
            (1476642704*mcMS**18)/(626535*mbkin**20*(1 - mcMS/mbkin)) - 
            (11813141632*mcMS**18)/(1879605*mbkin**19*(1 - mcMS/mbkin)) + 
            (3968477267*mcMS**18)/(375921*mbkin**18*(1 - mcMS/mbkin)) + 
            (854941712*mcMS**19)/(2297295*mbkin**21*(1 - mcMS/mbkin)) + 
            (6839533696*mcMS**19)/(6891885*mbkin**20*(1 - mcMS/mbkin)) - 
            (2297655851*mcMS**19)/(1378377*mbkin**19*(1 - mcMS/mbkin)) - 
            (696197624*mcMS**20)/(18706545*mbkin**22*(1 - mcMS/mbkin)) - 
            (5569580992*mcMS**20)/(56119635*mbkin**21*(1 - mcMS/mbkin)) + 
            (3742062229*mcMS**20)/(22447854*mbkin**20*(1 - mcMS/mbkin)) + 
            (602792*mcMS**21)/(340119*mbkin**23*(1 - mcMS/mbkin)) + 
            (4822336*mcMS**21)/(1020357*mbkin**22*(1 - mcMS/mbkin)) - 
            (16200035*mcMS**21)/(2040714*mbkin**21*(1 - mcMS/mbkin)) + 
            (2315776*mcMS*np.pi**2)/(135135*mbkin) - (53911342*mcMS**2*np.pi**2)/
                (405405*mbkin**2) + (147040*mcMS**3*np.pi**2)/(297*mbkin**3) - 
            (14344864*mcMS**4*np.pi**2)/(10395*mbkin**4) + (239168*mcMS**5*np.pi**2)/
                (81*mbkin**5) - (946178*mcMS**6*np.pi**2)/(189*mbkin**6) + 
            (298144*mcMS**7*np.pi**2)/(45*mbkin**7) - (1268320*mcMS**8*np.pi**2)/
                (189*mbkin**8) + (1650304*mcMS**9*np.pi**2)/(315*mbkin**9) - 
            (85102*mcMS**10*np.pi**2)/(27*mbkin**10) + (4070624*mcMS**11*np.pi**2)/
                (2835*mbkin**11) - (711968*mcMS**12*np.pi**2)/(1485*mbkin**12) + 
            (230336*mcMS**13*np.pi**2)/(2079*mbkin**13) - (917666*mcMS**14*np.pi**2)/
                (57915*mbkin**14) + (28576*mcMS**15*np.pi**2)/(27027*mbkin**15) + 
            ((-188264*mcMS**2)/(351*mbkin**2*(1 - mcMS/mbkin)**2) + 
                (214748656*mcMS**3)/(29835*mbkin**3*(1 - mcMS/mbkin)**2) - 
                (117272664508*mcMS**4)/(2297295*mbkin**4*(1 - mcMS/mbkin)**2) + 
                (37671647480*mcMS**5)/(153153*mbkin**5*(1 - mcMS/mbkin)**2) - 
                (39524020924*mcMS**6)/(45045*mbkin**6*(1 - mcMS/mbkin)**2) + 
                (283309760*mcMS**7)/(117*mbkin**7*(1 - mcMS/mbkin)**2) - 
                (102678435568*mcMS**8)/(19305*mbkin**8*(1 - mcMS/mbkin)**2) + 
                (183023630624*mcMS**9)/(19305*mbkin**9*(1 - mcMS/mbkin)**2) - 
                (20628036448*mcMS**10)/(1485*mbkin**10*(1 - mcMS/mbkin)**2) + 
                (15933731552*mcMS**11)/(945*mbkin**11*(1 - mcMS/mbkin)**2) - 
                (3216923384*mcMS**12)/(189*mbkin**12*(1 - mcMS/mbkin)**2) + 
                (13510105424*mcMS**13)/(945*mbkin**13*(1 - mcMS/mbkin)**2) - 
                (448387144*mcMS**14)/(45*mbkin**14*(1 - mcMS/mbkin)**2) + 
                (36851821888*mcMS**15)/(6435*mbkin**15*(1 - mcMS/mbkin)**2) - 
                (17284092368*mcMS**16)/(6435*mbkin**16*(1 - mcMS/mbkin)**2) + 
                (3906097568*mcMS**17)/(3861*mbkin**17*(1 - mcMS/mbkin)**2) - 
                (3669719752*mcMS**18)/(12285*mbkin**18*(1 - mcMS/mbkin)**2) + 
                (9001646128*mcMS**19)/(135135*mbkin**19*(1 - mcMS/mbkin)**2) - 
                (4847541068*mcMS**20)/(459459*mbkin**20*(1 - mcMS/mbkin)**2) + 
                (347227064*mcMS**21)/(328185*mbkin**21*(1 - mcMS/mbkin)**2) - 
                (301396*mcMS**22)/(5967*mbkin**22*(1 - mcMS/mbkin)**2) + 
                (14120272*mcMS)/(264537*mbkin*(1 - mcMS/mbkin)) - 
                (468025720*mcMS**2)/(793611*mbkin**2*(1 - mcMS/mbkin)) + 
                (107374328*mcMS**3)/(29835*mbkin**3*(1 - mcMS/mbkin)) - 
                (12111649028*mcMS**4)/(765765*mbkin**4*(1 - mcMS/mbkin)) + 
                (804583036*mcMS**5)/(15015*mbkin**5*(1 - mcMS/mbkin)) - 
                (16770272*mcMS**6)/(117*mbkin**6*(1 - mcMS/mbkin)) + 
                (12012704*mcMS**7)/(39*mbkin**7*(1 - mcMS/mbkin)) - 
                (10420999024*mcMS**8)/(19305*mbkin**8*(1 - mcMS/mbkin)) + 
                (231726224*mcMS**9)/(297*mbkin**9*(1 - mcMS/mbkin)) - 
                (42146864*mcMS**10)/(45*mbkin**10*(1 - mcMS/mbkin)) + 
                (177061168*mcMS**11)/(189*mbkin**11*(1 - mcMS/mbkin)) - 
                (147579352*mcMS**12)/(189*mbkin**12*(1 - mcMS/mbkin)) + 
                (24329912*mcMS**13)/(45*mbkin**13*(1 - mcMS/mbkin)) - 
                (60252704*mcMS**14)/(195*mbkin**14*(1 - mcMS/mbkin)) + 
                (6495875552*mcMS**15)/(45045*mbkin**15*(1 - mcMS/mbkin)) - 
                (54136688*mcMS**16)/(1001*mbkin**16*(1 - mcMS/mbkin)) + 
                (5583616*mcMS**17)/(351*mbkin**17*(1 - mcMS/mbkin)) - 
                (738321352*mcMS**18)/(208845*mbkin**18*(1 - mcMS/mbkin)) + 
                (427470856*mcMS**19)/(765765*mbkin**19*(1 - mcMS/mbkin)) - 
                (348098812*mcMS**20)/(6235515*mbkin**20*(1 - mcMS/mbkin)) + 
                (301396*mcMS**21)/(113373*mbkin**21*(1 - mcMS/mbkin)) - 
                (64*mcMS**4*np.pi**2)/(3*mbkin**4))*np.log(mcMS**2/mbkin**2) + 
            ((3580064*mcMS)/(675675*mbkin) + (444307984*mcMS**2)/
                (868725*mbkin**2) - (1987324448*mcMS**3)/(467775*mbkin**3) + 
                (4524595232*mcMS**4)/(280665*mbkin**4) - (4045856*mcMS**5)/
                (105*mbkin**5) + (400661008*mcMS**6)/(6075*mbkin**6) - 
                (521545376*mcMS**7)/(6075*mbkin**7) + (1231899712*mcMS**8)/
                (14175*mbkin**8) - (5546336*mcMS**9)/(81*mbkin**9) + 
                (70692848*mcMS**10)/(1701*mbkin**10) - (810886688*mcMS**11)/
                (42525*mbkin**11) + (2996286176*mcMS**12)/(467775*mbkin**12) - 
                (298261792*mcMS**13)/(200475*mbkin**13) + (4125776*mcMS**14)/
                (19305*mbkin**14) - (17410912*mcMS**15)/(1216215*mbkin**15))*
                np.log(1 - mcMS/mbkin) + ((-748886504*mcMS)/(8729721*mbkin) + 
                (7937476*mcMS**2)/(11583*mbkin**2) - (950785096*mcMS**3)/
                (328185*mbkin**3) + (1198844138*mcMS**4)/(135135*mbkin**4) - 
                (22613456*mcMS**5)/(1053*mbkin**5) + (14677064*mcMS**6)/
                (351*mbkin**6) - (7772528*mcMS**7)/(117*mbkin**7) + 
                (2746783208*mcMS**8)/(31185*mbkin**8) - (4479008*mcMS**9)/
                (45*mbkin**9) + (872224*mcMS**10)/(9*mbkin**10) - 
                (15355264*mcMS**11)/(189*mbkin**11) + (17248564*mcMS**12)/
                (297*mbkin**12) - (51768368*mcMS**13)/(1485*mbkin**13) + 
                (998628536*mcMS**14)/(57915*mbkin**14) - (185744752*mcMS**15)/
                (27027*mbkin**15) + (83704*mcMS**16)/(39*mbkin**16) - 
                (531800*mcMS**17)/(1053*mbkin**17) + (5859964*mcMS**18)/
                (69615*mbkin**18) - (171064*mcMS**19)/(19305*mbkin**19) + 
                (150698*mcMS**20)/(340119*mbkin**20) + (188264*mcMS**2)/
                (351*mbkin**2*(1 - mcMS/mbkin)**2) - (214748656*mcMS**3)/
                (29835*mbkin**3*(1 - mcMS/mbkin)**2) + (117272664508*mcMS**4)/
                (2297295*mbkin**4*(1 - mcMS/mbkin)**2) - (37671647480*mcMS**5)/
                (153153*mbkin**5*(1 - mcMS/mbkin)**2) + (39524020924*mcMS**6)/
                (45045*mbkin**6*(1 - mcMS/mbkin)**2) - (283309760*mcMS**7)/
                (117*mbkin**7*(1 - mcMS/mbkin)**2) + (102678435568*mcMS**8)/
                (19305*mbkin**8*(1 - mcMS/mbkin)**2) - (183023630624*mcMS**9)/
                (19305*mbkin**9*(1 - mcMS/mbkin)**2) + (20628036448*mcMS**10)/
                (1485*mbkin**10*(1 - mcMS/mbkin)**2) - (15933731552*mcMS**11)/
                (945*mbkin**11*(1 - mcMS/mbkin)**2) + (3216923384*mcMS**12)/
                (189*mbkin**12*(1 - mcMS/mbkin)**2) - (13510105424*mcMS**13)/
                (945*mbkin**13*(1 - mcMS/mbkin)**2) + (448387144*mcMS**14)/
                (45*mbkin**14*(1 - mcMS/mbkin)**2) - (36851821888*mcMS**15)/
                (6435*mbkin**15*(1 - mcMS/mbkin)**2) + (17284092368*mcMS**16)/
                (6435*mbkin**16*(1 - mcMS/mbkin)**2) - (3906097568*mcMS**17)/
                (3861*mbkin**17*(1 - mcMS/mbkin)**2) + (3669719752*mcMS**18)/
                (12285*mbkin**18*(1 - mcMS/mbkin)**2) - (9001646128*mcMS**19)/
                (135135*mbkin**19*(1 - mcMS/mbkin)**2) + (4847541068*mcMS**20)/
                (459459*mbkin**20*(1 - mcMS/mbkin)**2) - (347227064*mcMS**21)/
                (328185*mbkin**21*(1 - mcMS/mbkin)**2) + (301396*mcMS**22)/
                (5967*mbkin**22*(1 - mcMS/mbkin)**2) - (14120272*mcMS)/
                (264537*mbkin*(1 - mcMS/mbkin)) + (468025720*mcMS**2)/
                (793611*mbkin**2*(1 - mcMS/mbkin)) - (107374328*mcMS**3)/
                (29835*mbkin**3*(1 - mcMS/mbkin)) + (12111649028*mcMS**4)/
                (765765*mbkin**4*(1 - mcMS/mbkin)) - (804583036*mcMS**5)/
                (15015*mbkin**5*(1 - mcMS/mbkin)) + (16770272*mcMS**6)/
                (117*mbkin**6*(1 - mcMS/mbkin)) - (12012704*mcMS**7)/
                (39*mbkin**7*(1 - mcMS/mbkin)) + (10420999024*mcMS**8)/
                (19305*mbkin**8*(1 - mcMS/mbkin)) - (231726224*mcMS**9)/
                (297*mbkin**9*(1 - mcMS/mbkin)) + (42146864*mcMS**10)/
                (45*mbkin**10*(1 - mcMS/mbkin)) - (177061168*mcMS**11)/
                (189*mbkin**11*(1 - mcMS/mbkin)) + (147579352*mcMS**12)/
                (189*mbkin**12*(1 - mcMS/mbkin)) - (24329912*mcMS**13)/
                (45*mbkin**13*(1 - mcMS/mbkin)) + (60252704*mcMS**14)/
                (195*mbkin**14*(1 - mcMS/mbkin)) - (6495875552*mcMS**15)/
                (45045*mbkin**15*(1 - mcMS/mbkin)) + (54136688*mcMS**16)/
                (1001*mbkin**16*(1 - mcMS/mbkin)) - (5583616*mcMS**17)/
                (351*mbkin**17*(1 - mcMS/mbkin)) + (738321352*mcMS**18)/
                (208845*mbkin**18*(1 - mcMS/mbkin)) - (427470856*mcMS**19)/
                (765765*mbkin**19*(1 - mcMS/mbkin)) + (348098812*mcMS**20)/
                (6235515*mbkin**20*(1 - mcMS/mbkin)) - (301396*mcMS**21)/
                (113373*mbkin**21*(1 - mcMS/mbkin)))*np.log(mus**2/mbkin**2))) + 
        (-14560/(243*mbkin**5) - 116480/(729*mbkin**4) - 44590/(81*mbkin**3) - 
            (21122*mcMS**2)/(243*mbkin**5) - (12320*mcMS**4)/(243*mbkin**9) - 
            (98560*mcMS**4)/(729*mbkin**8) + (162470*mcMS**4)/(243*mbkin**7) - 
            (6784*mcMS**6)/(243*mbkin**11) - (54272*mcMS**6)/(729*mbkin**10) + 
            (212*mcMS**6)/mbkin**9 + (896*mcMS**8)/(27*mbkin**13) + 
            (7168*mcMS**8)/(81*mbkin**12) - (15400*mcMS**8)/(81*mbkin**11) + 
            (1820*np.pi**2)/(81*mbkin**3) + (236*mcMS**2*np.pi**2)/(81*mbkin**5) - 
            (1540*mcMS**4*np.pi**2)/(81*mbkin**7) - (424*mcMS**6*np.pi**2)/(81*mbkin**9) + 
            (112*mcMS**8*np.pi**2)/(27*mbkin**11))*fp.polylog(2, 1 - mbkin**2/mcMS**2) + 
        (-1216/(243*mbkin**5) - 9728/(729*mbkin**4) - 3724/(81*mbkin**3) - 
            (7160*mcMS**2)/(27*mbkin**5) - (1856*mcMS**4)/(81*mbkin**9) - 
            (14848*mcMS**4)/(243*mbkin**8) + (24476*mcMS**4)/(81*mbkin**7) - 
            (35200*mcMS**6)/(243*mbkin**11) - (281600*mcMS**6)/(729*mbkin**10) + 
            (1100*mcMS**6)/mbkin**9 + (128*mcMS**8)/(3*mbkin**13) + 
            (1024*mcMS**8)/(9*mbkin**12) - (2200*mcMS**8)/(9*mbkin**11) - 
            (3376*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**5) - 
            (27008*np.sqrt(0j + mcMS**2/mbkin**2))/(2187*mbkin**4) - 
            (68786*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**3) - 
            (5648*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**9) - 
            (45184*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**8) + 
            (160262*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**7) - 
            (49952*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**5) - 
            (399616*(mcMS**2/mbkin**2)**(3/2))/(2187*mbkin**4) + 
            (405860*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) + 
            (152*np.pi**2)/(81*mbkin**3) + (80*mcMS**2*np.pi**2)/(9*mbkin**5) - 
            (232*mcMS**4*np.pi**2)/(27*mbkin**7) - (2200*mcMS**6*np.pi**2)/(81*mbkin**9) + 
            (16*mcMS**8*np.pi**2)/(3*mbkin**11) + (844*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/
            (243*mbkin**3) - (1412*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/
            (27*mbkin**7) - (12488*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(243*mbkin**3))*
            fp.polylog(2, -(mcMS/mbkin)) + (-1216/(243*mbkin**5) - 
            9728/(729*mbkin**4) - 3724/(81*mbkin**3) - (7160*mcMS**2)/
            (27*mbkin**5) - (1856*mcMS**4)/(81*mbkin**9) - (14848*mcMS**4)/
            (243*mbkin**8) + (24476*mcMS**4)/(81*mbkin**7) - 
            (35200*mcMS**6)/(243*mbkin**11) - (281600*mcMS**6)/(729*mbkin**10) + 
            (1100*mcMS**6)/mbkin**9 + (128*mcMS**8)/(3*mbkin**13) + 
            (1024*mcMS**8)/(9*mbkin**12) - (2200*mcMS**8)/(9*mbkin**11) - 
            (208*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**5) - 
            (1664*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**4) - 
            (4238*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**3) + 
            (11824*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**9) + 
            (94592*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**8) - 
            (335506*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**7) + 
            (140128*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**5) + 
            (1121024*(mcMS**2/mbkin**2)**(3/2))/(2187*mbkin**4) - 
            (1138540*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) + 
            (152*np.pi**2)/(81*mbkin**3) + (80*mcMS**2*np.pi**2)/(9*mbkin**5) - 
            (232*mcMS**4*np.pi**2)/(27*mbkin**7) - (2200*mcMS**6*np.pi**2)/(81*mbkin**9) + 
            (16*mcMS**8*np.pi**2)/(3*mbkin**11) + (52*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/
            (81*mbkin**3) + (2956*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/
            (27*mbkin**7) + (35032*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(243*mbkin**3))*
            fp.polylog(2, mcMS/mbkin) + (-2144/(243*mbkin**5) - 17152/(729*mbkin**4) - 
            6566/(81*mbkin**3) + (358*mcMS**2)/(3*mbkin**5) + 
            (544*mcMS**4)/(27*mbkin**9) + (4352*mcMS**4)/(81*mbkin**8) - 
            (7174*mcMS**4)/(27*mbkin**7) - (2752*mcMS**6)/(243*mbkin**11) - 
            (22016*mcMS**6)/(729*mbkin**10) + (86*mcMS**6)/mbkin**9 - 
            (1024*mcMS**8)/(27*mbkin**13) - (8192*mcMS**8)/(81*mbkin**12) + 
            (17600*mcMS**8)/(81*mbkin**11) + (1000*np.sqrt(0j + mcMS**2/mbkin**2))/
            (729*mbkin**5) + (8000*np.sqrt(0j + mcMS**2/mbkin**2))/(2187*mbkin**4) + 
            (20375*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**3) - 
            (1544*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**9) - 
            (12352*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**8) + 
            (43811*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**7) - 
            (22544*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**5) - 
            (180352*(mcMS**2/mbkin**2)**(3/2))/(2187*mbkin**4) + 
            (183170*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) + 
            (268*np.pi**2)/(81*mbkin**3) - (4*mcMS**2*np.pi**2)/mbkin**5 + 
            (68*mcMS**4*np.pi**2)/(9*mbkin**7) - (172*mcMS**6*np.pi**2)/(81*mbkin**9) - 
            (128*mcMS**8*np.pi**2)/(27*mbkin**11) - (250*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/
            (243*mbkin**3) - (386*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/
            (27*mbkin**7) - (5636*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(243*mbkin**3))*
            fp.polylog(2, mcMS**2/mbkin**2) + np.log(2/mus)*(320/(9*mbkin**5) + 
            1156/(27*mbkin**4) + 699641/(810*mbkin**3) - 8307098369529671/
            (12406679485200*mbkin**2) - 5783028354262871/(5583005768340*mbkin) + 
            371/(27*mbkin*mcMS**2) - (2649349310527609*mcMS)/
            (3421235858040*mbkin**3) - (2649349310527609*mcMS)/
            (1539556136118*mbkin**2) + (128*mcMS**2)/mbkin**6 + 
            (144094*mcMS**2)/(405*mbkin**5) + (7852842940487*mcMS**2)/
            (1134863730*mbkin**4) + (14375322959086*mcMS**2)/
            (1021377357*mbkin**3) - (371136550584956041*mcMS**3)/
            (29710732451400*mbkin**5) - (371136550584956041*mcMS**3)/
            (13369829603130*mbkin**4) + (256*mcMS**4)/(3*mbkin**9) + 
            (752*mcMS**4)/(9*mbkin**8) - (176627*mcMS**4)/(90*mbkin**7) + 
            (43696384832527241*mcMS**4)/(1165126762800*mbkin**6) + 
            (42996065972966921*mcMS**4)/(524307043260*mbkin**5) - 
            (45300689655370427*mcMS**5)/(436922536050*mbkin**7) - 
            (90601379310740854*mcMS**5)/(393230282445*mbkin**6) - 
            (2048*mcMS**6)/(9*mbkin**11) - (14656*mcMS**6)/(27*mbkin**10) + 
            (484477*mcMS**6)/(405*mbkin**9) + (40993337498401*mcMS**6)/
            (189143955*mbkin**8) + (165023140975844*mcMS**6)/
            (340459119*mbkin**7) - (328364278022567*mcMS**7)/
            (882671790*mbkin**9) - (656728556045134*mcMS**7)/
            (794404611*mbkin**8) + (320*mcMS**8)/(3*mbkin**13) + 
            (2596*mcMS**8)/(9*mbkin**12) - (467*mcMS**8)/mbkin**11 + 
            (3382991430320069*mcMS**8)/(6401795400*mbkin**10) + 
            (3380406385337549*mcMS**8)/(2880807930*mbkin**9) - 
            (546447454894529*mcMS**9)/(872972100*mbkin**11) - 
            (546447454894529*mcMS**9)/(392837445*mbkin**10) + 
            (315478243148254*mcMS**10)/(509233725*mbkin**12) + 
            (1261912972593016*mcMS**10)/(916620705*mbkin**11) - 
            (6882359619137911*mcMS**11)/(13443770340*mbkin**13) - 
            (6882359619137911*mcMS**11)/(6049696653*mbkin**12) + 
            (860250268694657*mcMS**12)/(2444321880*mbkin**14) + 
            (860250268694657*mcMS**12)/(1099944846*mbkin**13) - 
            (29086506506933*mcMS**13)/(145495350*mbkin**15) - 
            (58173013013866*mcMS**13)/(130945815*mbkin**14) + 
            (20276892020748997*mcMS**14)/(218461268025*mbkin**16) + 
            (81107568082995988*mcMS**14)/(393230282445*mbkin**15) - 
            (5046733464982007*mcMS**15)/(145640845350*mbkin**17) - 
            (10093466929964014*mcMS**15)/(131076760815*mbkin**16) + 
            (5929043471749*mcMS**16)/(584023440*mbkin**18) + 
            (5929043471749*mcMS**16)/(262810548*mbkin**17) - 
            (44981152207347*mcMS**17)/(20007227240*mbkin**19) - 
            (14993717402449*mcMS**17)/(3001084086*mbkin**18) + 
            (238966675564711*mcMS**18)/(675243919350*mbkin**20) + 
            (477933351129422*mcMS**18)/(607719527415*mbkin**19) - 
            (22978828361473*mcMS**19)/(651100249800*mbkin**21) - 
            (22978828361473*mcMS**19)/(292995112410*mbkin**20) + 
            (4916068298621*mcMS**20)/(2932487878320*mbkin**22) + 
            (4916068298621*mcMS**20)/(1319619545244*mbkin**21) + 
            (2291073463739113*mcMS)/(20527415148240*mbkin**3*(1 - mcMS/mbkin)) + 
            (2291073463739113*mcMS)/(7697780680590*mbkin**2*(1 - mcMS/mbkin)) - 
            (66120814157661443*mcMS**2)/(61582245444720*mbkin**4*
            (1 - mcMS/mbkin)) - (66120814157661443*mcMS**2)/
            (23093342041770*mbkin**3*(1 - mcMS/mbkin)) + 
            (1166928234496119443*mcMS**3)/(178264394708400*mbkin**5*
            (1 - mcMS/mbkin)) + (1166928234496119443*mcMS**3)/
            (66849148015650*mbkin**4*(1 - mcMS/mbkin)) - 
            (118345817852851093*mcMS**4)/(3961430993520*mbkin**6*
            (1 - mcMS/mbkin)) - (118345817852851093*mcMS**4)/
            (1485536622570*mbkin**5*(1 - mcMS/mbkin)) + 
            (49345371782260483*mcMS**5)/(476642766600*mbkin**7*(1 - mcMS/mbkin)) + 
            (49345371782260483*mcMS**5)/(178741037475*mbkin**6*(1 - mcMS/mbkin)) - 
            (146208599147473679*mcMS**6)/(524307043260*mbkin**8*
            (1 - mcMS/mbkin)) - (292417198294947358*mcMS**6)/
            (393230282445*mbkin**7*(1 - mcMS/mbkin)) + (64819098503293*mcMS**7)/
            (108082260*mbkin**9*(1 - mcMS/mbkin)) + (129638197006586*mcMS**7)/
            (81061695*mbkin**8*(1 - mcMS/mbkin)) - (7714866552402197*mcMS**8)/
            (7343235900*mbkin**10*(1 - mcMS/mbkin)) - (15429733104804394*mcMS**8)/
            (5507426925*mbkin**9*(1 - mcMS/mbkin)) + (2242974842855321*mcMS**9)/
            (1477337400*mbkin**11*(1 - mcMS/mbkin)) + (2242974842855321*mcMS**9)/
            (554001525*mbkin**10*(1 - mcMS/mbkin)) - (1713682422600793*mcMS**10)/
            (940123800*mbkin**12*(1 - mcMS/mbkin)) - (1713682422600793*mcMS**10)/
            (352546425*mbkin**11*(1 - mcMS/mbkin)) + (13374421001572999*mcMS**11)/
            (7332965640*mbkin**13*(1 - mcMS/mbkin)) + 
            (13374421001572999*mcMS**11)/(2749862115*mbkin**12*(1 - mcMS/mbkin)) - 
            (11151579707994493*mcMS**12)/(7332965640*mbkin**14*(1 - mcMS/mbkin)) - 
            (11151579707994493*mcMS**12)/(2749862115*mbkin**13*(1 - mcMS/mbkin)) + 
            (178800900176533*mcMS**13)/(169744575*mbkin**15*(1 - mcMS/mbkin)) + 
            (1430407201412264*mcMS**13)/(509233725*mbkin**14*(1 - mcMS/mbkin)) - 
            (32217277599333127*mcMS**14)/(53500718700*mbkin**16*
            (1 - mcMS/mbkin)) - (64434555198666254*mcMS**14)/
            (40125539025*mbkin**15*(1 - mcMS/mbkin)) + 
            (43352528289259463*mcMS**15)/(154207953900*mbkin**17*
            (1 - mcMS/mbkin)) + (86705056578518926*mcMS**15)/
            (115655965425*mbkin**16*(1 - mcMS/mbkin)) - 
            (111704868472735*mcMS**16)/(1059206148*mbkin**18*(1 - mcMS/mbkin)) - 
            (223409736945470*mcMS**16)/(794404611*mbkin**17*(1 - mcMS/mbkin)) + 
            (7230231600513707*mcMS**17)/(233025352560*mbkin**19*
            (1 - mcMS/mbkin)) + (7230231600513707*mcMS**17)/
            (87384507210*mbkin**18*(1 - mcMS/mbkin)) - 
            (1774149351078739*mcMS**18)/(257235778800*mbkin**20*
            (1 - mcMS/mbkin)) - (1774149351078739*mcMS**18)/
            (96463417050*mbkin**19*(1 - mcMS/mbkin)) + 
            (64726178334967321*mcMS**19)/(59421464902800*mbkin**21*
            (1 - mcMS/mbkin)) + (64726178334967321*mcMS**19)/
            (22283049338550*mbkin**20*(1 - mcMS/mbkin)) - 
            (369023890038773819*mcMS**20)/(3387023499459600*mbkin**22*
            (1 - mcMS/mbkin)) - (369023890038773819*mcMS**20)/
            (1270133812297350*mbkin**21*(1 - mcMS/mbkin)) + 
            (22826272832761*mcMS**21)/(4398731817480*mbkin**23*(1 - mcMS/mbkin)) + 
            (22826272832761*mcMS**21)/(1649524431555*mbkin**22*(1 - mcMS/mbkin)) - 
            (1396*np.pi**2)/(27*mbkin**3) + (18*np.pi**2)/mbkin**2 + (40*np.pi**2)/mbkin + 
            (42*mcMS**2*np.pi**2)/mbkin**5 - (96*mcMS**2*np.pi**2)/mbkin**4 - 
            (192*mcMS**2*np.pi**2)/mbkin**3 + (244*mcMS**4*np.pi**2)/(9*mbkin**7) + 
            (72*mcMS**4*np.pi**2)/mbkin**6 + (192*mcMS**4*np.pi**2)/mbkin**5 - 
            (172*mcMS**6*np.pi**2)/(9*mbkin**9) - (64*mcMS**6*np.pi**2)/mbkin**7 + 
            (148*mcMS**8*np.pi**2)/(9*mbkin**11) + (6*mcMS**8*np.pi**2)/mbkin**10 + 
            (24*mcMS**8*np.pi**2)/mbkin**9 + (86*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/
            (27*mbkin**3) - (182*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/mbkin**7 - 
            (220*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/mbkin**3 + 
            (556/(9*mbkin**3) + (154*mcMS**2)/(9*mbkin**5) - (514*mcMS**4)/
                (27*mbkin**7) - (264*mcMS**4)/mbkin**6 - (352*mcMS**4)/mbkin**5 + 
            (508*mcMS**6)/(9*mbkin**9) + (268*mcMS**8)/(9*mbkin**11))*
            np.log(mcMS**2/mbkin**2)**2 + (-13022602/(124355*mbkin**2) - 
            52090408/(223839*mbkin) + (28240544*mcMS)/(29393*mbkin**3) + 
            (564810880*mcMS)/(264537*mbkin**2) - (188264*mcMS**2)/(39*mbkin**4) - 
            (3765280*mcMS**2)/(351*mbkin**3) + (20304864*mcMS**3)/
                (1105*mbkin**5) + (9024384*mcMS**3)/(221*mbkin**4) - 
            (287248154*mcMS**4)/(5005*mbkin**6) - (1148992616*mcMS**4)/
                (9009*mbkin**5) + (1910912*mcMS**5)/(13*mbkin**7) + 
            (38218240*mcMS**5)/(117*mbkin**6) - (3997664*mcMS**6)/(13*mbkin**8) - 
            (79953280*mcMS**6)/(117*mbkin**7) + (48090240*mcMS**7)/(91*mbkin**9) + 
            (106867200*mcMS**7)/(91*mbkin**8) - (124106812*mcMS**8)/
                (165*mbkin**10) - (496427248*mcMS**8)/(297*mbkin**9) + 
            (13377856*mcMS**9)/(15*mbkin**11) + (53511424*mcMS**9)/
                (27*mbkin**10) - (4416016*mcMS**10)/(5*mbkin**12) - 
            (17664064*mcMS**10)/(9*mbkin**11) + (168649664*mcMS**11)/
                (231*mbkin**13) + (3372993280*mcMS**11)/(2079*mbkin**12) - 
            (502020*mcMS**12)/mbkin**14 - (1115600*mcMS**12)/mbkin**13 + 
            (1426048*mcMS**13)/(5*mbkin**15) + (5704192*mcMS**13)/(9*mbkin**14) - 
            (60257056*mcMS**14)/(455*mbkin**16) - (241028224*mcMS**14)/
                (819*mbkin**15) + (247477632*mcMS**15)/(5005*mbkin**17) + 
            (329970176*mcMS**15)/(3003*mbkin**16) - (188334*mcMS**16)/
                (13*mbkin**18) - (418520*mcMS**16)/(13*mbkin**17) + 
            (2127200*mcMS**17)/(663*mbkin**19) + (42544000*mcMS**17)/
                (5967*mbkin**18) - (11719928*mcMS**18)/(23205*mbkin**20) - 
            (46879712*mcMS**18)/(41769*mbkin**19) + (684256*mcMS**19)/
                (13585*mbkin**21) + (2737024*mcMS**19)/(24453*mbkin**20) - 
            (150698*mcMS**20)/(62985*mbkin**22) - (602792*mcMS**20)/
                (113373*mbkin**21) - (14120272*mcMS)/(88179*mbkin**3*
                (1 - mcMS/mbkin)) - (112962176*mcMS)/(264537*mbkin**2*
                (1 - mcMS/mbkin)) + (468025720*mcMS**2)/(264537*mbkin**4*
                (1 - mcMS/mbkin)) + (3744205760*mcMS**2)/(793611*mbkin**3*
                (1 - mcMS/mbkin)) - (107374328*mcMS**3)/(9945*mbkin**5*
                (1 - mcMS/mbkin)) - (858994624*mcMS**3)/(29835*mbkin**4*
                (1 - mcMS/mbkin)) + (12111649028*mcMS**4)/(255255*mbkin**6*
                (1 - mcMS/mbkin)) + (96893192224*mcMS**4)/(765765*mbkin**5*
                (1 - mcMS/mbkin)) - (804583036*mcMS**5)/(5005*mbkin**7*
                (1 - mcMS/mbkin)) - (6436664288*mcMS**5)/(15015*mbkin**6*
                (1 - mcMS/mbkin)) + (16770272*mcMS**6)/(39*mbkin**8*
                (1 - mcMS/mbkin)) + (134162176*mcMS**6)/(117*mbkin**7*
                (1 - mcMS/mbkin)) - (12012704*mcMS**7)/(13*mbkin**9*
                (1 - mcMS/mbkin)) - (96101632*mcMS**7)/(39*mbkin**8*
                (1 - mcMS/mbkin)) + (10420999024*mcMS**8)/(6435*mbkin**10*
                (1 - mcMS/mbkin)) + (83367992192*mcMS**8)/(19305*mbkin**9*
                (1 - mcMS/mbkin)) - (231726224*mcMS**9)/(99*mbkin**11*
                (1 - mcMS/mbkin)) - (1853809792*mcMS**9)/(297*mbkin**10*
                (1 - mcMS/mbkin)) + (42146864*mcMS**10)/(15*mbkin**12*
                (1 - mcMS/mbkin)) + (337174912*mcMS**10)/(45*mbkin**11*
                (1 - mcMS/mbkin)) - (177061168*mcMS**11)/(63*mbkin**13*
                (1 - mcMS/mbkin)) - (1416489344*mcMS**11)/(189*mbkin**12*
                (1 - mcMS/mbkin)) + (147579352*mcMS**12)/(63*mbkin**14*
                (1 - mcMS/mbkin)) + (1180634816*mcMS**12)/(189*mbkin**13*
                (1 - mcMS/mbkin)) - (24329912*mcMS**13)/(15*mbkin**15*
                (1 - mcMS/mbkin)) - (194639296*mcMS**13)/(45*mbkin**14*
                (1 - mcMS/mbkin)) + (60252704*mcMS**14)/(65*mbkin**16*
                (1 - mcMS/mbkin)) + (482021632*mcMS**14)/(195*mbkin**15*
                (1 - mcMS/mbkin)) - (6495875552*mcMS**15)/(15015*mbkin**17*
                (1 - mcMS/mbkin)) - (51967004416*mcMS**15)/(45045*mbkin**16*
                (1 - mcMS/mbkin)) + (162410064*mcMS**16)/(1001*mbkin**18*
                (1 - mcMS/mbkin)) + (433093504*mcMS**16)/(1001*mbkin**17*
                (1 - mcMS/mbkin)) - (5583616*mcMS**17)/(117*mbkin**19*
                (1 - mcMS/mbkin)) - (44668928*mcMS**17)/(351*mbkin**18*
                (1 - mcMS/mbkin)) + (738321352*mcMS**18)/(69615*mbkin**20*
                (1 - mcMS/mbkin)) + (5906570816*mcMS**18)/(208845*mbkin**19*
                (1 - mcMS/mbkin)) - (427470856*mcMS**19)/(255255*mbkin**21*
                (1 - mcMS/mbkin)) - (3419766848*mcMS**19)/(765765*mbkin**20*
                (1 - mcMS/mbkin)) + (348098812*mcMS**20)/(2078505*mbkin**22*
                (1 - mcMS/mbkin)) + (2784790496*mcMS**20)/(6235515*mbkin**21*
                (1 - mcMS/mbkin)) - (301396*mcMS**21)/(37791*mbkin**23*
                (1 - mcMS/mbkin)) - (2411168*mcMS**21)/(113373*mbkin**22*
                (1 - mcMS/mbkin)) + ((-172*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**3) + 
                (364*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/mbkin**7 + 
                (440*(mcMS**2/mbkin**2)**(3/2))/mbkin**3)*np.log(mcMS**2/mbkin**2))*
            np.log(1 - mcMS/mbkin) + (-13769/(27*mbkin**3) + (371*mbkin)/
                (27*mcMS**4) - 22462/(405*mbkin*mcMS**2) + (13508*mcMS**2)/
                (27*mbkin**5) + (20435*mcMS**4)/(81*mbkin**7) - (14242*mcMS**6)/
                (27*mbkin**9) + (44089*mcMS**8)/(135*mbkin**11))*
            np.log(1 - mcMS**2/mbkin**2) + np.log(mcMS**2/mbkin**2)*(256/(3*mbkin**5) + 
            2048/(9*mbkin**4) + 12422/(27*mbkin**3) - 6/mbkin**2 - 40/(3*mbkin) + 
            (3784*mcMS**2)/(9*mbkin**5) - (160*mcMS**2)/mbkin**4 - 
            (320*mcMS**2)/mbkin**3 - (48*mcMS**4)/mbkin**8 - (1171*mcMS**4)/
                (162*mbkin**7) + (2504*mcMS**4)/mbkin**6 + (12368*mcMS**4)/
                (3*mbkin**5) + (530*mcMS**6)/(3*mbkin**9) - (1088*mcMS**6)/
                (3*mbkin**7) + (15007*mcMS**8)/(270*mbkin**11) + 
            (46*mcMS**8)/mbkin**10 + (184*mcMS**8)/mbkin**9 - (32*np.pi**2)/mbkin**3 - 
            (72*mcMS**4*np.pi**2)/mbkin**6 - (96*mcMS**4*np.pi**2)/mbkin**5 + 
            ((172*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**3) - 
                (364*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/mbkin**7 - 
                (440*(mcMS**2/mbkin**2)**(3/2))/mbkin**3)*np.log(1 + mcMS/mbkin) + 
            (-1708/(9*mbkin**3) - (44*mcMS**2)/mbkin**5 - (13172*mcMS**4)/
                (27*mbkin**7) - (760*mcMS**6)/(9*mbkin**9) - (352*mcMS**8)/
                (9*mbkin**11))*np.log(1 - mcMS**2/mbkin**2)) + 
            (1820/(9*mbkin**3) + (236*mcMS**2)/(9*mbkin**5) - (1540*mcMS**4)/
                (9*mbkin**7) - (424*mcMS**6)/(9*mbkin**9) + (112*mcMS**8)/
                (3*mbkin**11))*fp.polylog(2, 1 - mbkin**2/mcMS**2) + 
            (152/(9*mbkin**3) + (80*mcMS**2)/mbkin**5 - (232*mcMS**4)/(3*mbkin**7) - 
            (2200*mcMS**6)/(9*mbkin**9) + (48*mcMS**8)/mbkin**11 + 
            (844*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**3) - 
            (1412*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(3*mbkin**7) - 
            (12488*(mcMS**2/mbkin**2)**(3/2))/(27*mbkin**3))*
            fp.polylog(2, -(mcMS/mbkin)) + (152/(9*mbkin**3) + 
            (80*mcMS**2)/mbkin**5 - (232*mcMS**4)/(3*mbkin**7) - 
            (2200*mcMS**6)/(9*mbkin**9) + (48*mcMS**8)/mbkin**11 + 
            (52*np.sqrt(0j + mcMS**2/mbkin**2))/(9*mbkin**3) + 
            (2956*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(3*mbkin**7) + 
            (35032*(mcMS**2/mbkin**2)**(3/2))/(27*mbkin**3))*
            fp.polylog(2, mcMS/mbkin) + (268/(9*mbkin**3) - (36*mcMS**2)/mbkin**5 + 
            (68*mcMS**4)/mbkin**7 - (172*mcMS**6)/(9*mbkin**9) - 
            (128*mcMS**8)/(3*mbkin**11) - (250*np.sqrt(0j + mcMS**2/mbkin**2))/
                (27*mbkin**3) - (386*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(3*mbkin**7) - 
            (5636*(mcMS**2/mbkin**2)**(3/2))/(27*mbkin**3))*
            fp.polylog(2, mcMS**2/mbkin**2)) + np.log(mcMS**2/mbkin**2)*
            (-256/(81*mbkin**7) - 4096/(243*mbkin**6) - 84272/(729*mbkin**5) - 
            663308/(2187*mbkin**4) - 181.172981917779/mbkin**3 + 
            8524215260520671/(334980346100400*mbkin**2) + 6000145245253871/
            (150741155745180*mbkin) - 4823/(729*mbkin*mcMS**2) - 
            (11097091471907069*mcMS)/(92373368167080*mbkin**3) - 
            (11097091471907069*mcMS)/(41568015675186*mbkin**2) - 
            (10072.277369115025*mcMS)/mbkin + (2464*mcMS**2)/(243*mbkin**7) + 
            (36992*mcMS**2)/(729*mbkin**6) - (9046066*mcMS**2)/(10935*mbkin**5) + 
            (64450970210629*mcMS**2)/(30641320710*mbkin**4) + 
            (121868924916314*mcMS**2)/(27577188639*mbkin**3) + 
            (127822.79805925587*mcMS**2)/mbkin**2 - (18941152960396819*mcMS**3)/
            (2483559678600*mbkin**5) - (365959514538213481*mcMS**3)/
            (21234435252030*mbkin**4) - (751346.8145151937*mcMS**3)/mbkin**3 - 
            (32*mcMS**4)/(9*mbkin**10) + (452*mcMS**4)/(2187*mbkin**9) + 
            (811024*mcMS**4)/(6561*mbkin**8) - (31065973*mcMS**4)/(43740*mbkin**7) + 
            (29144.502767014095*mcMS**4)/mbkin**6 + (67002.72792740198*mcMS**4)/
            mbkin**5 + (2.7571275579980034e6*mcMS**4)/mbkin**4 - 
            (1361058263765088943*mcMS**5)/(11796908473350*mbkin**7) - 
            (146638403344121642*mcMS**5)/(558800927685*mbkin**6) - 
            (6.976443096232966e6*mcMS**5)/mbkin**5 + (33568*mcMS**6)/
            (243*mbkin**11) + (297920*mcMS**6)/(729*mbkin**10) - 
            (1788182*mcMS**6)/(2187*mbkin**9) + (1526260315313219*mcMS**6)/
            (5106886785*mbkin**8) + (4496987686506164*mcMS**6)/
            (6565997295*mbkin**7) + (1.2876096969270526e7*mcMS**6)/mbkin**6 - 
            (51467289145739*mcMS**7)/(83621538*mbkin**9) - 
            (151247820892411318*mcMS**7)/(107244622485*mbkin**8) - 
            (1.7875794658439644e7*mcMS**7)/mbkin**7 + (9052*mcMS**8)/
            (135*mbkin**13) + (73796*mcMS**8)/(405*mbkin**12) - 
            (6850057*mcMS**8)/(14580*mbkin**11) + (177317130744439663*mcMS**8)/
            (172848475800*mbkin**10) + (36681142103833163*mcMS**8)/
            (15556362822*mbkin**9) + (1.8902317013163272e7*mcMS**8)/mbkin**8 - 
            (33029205593811157*mcMS**9)/(23570246700*mbkin**11) - 
            (6858046018668881*mcMS**9)/(2121322203*mbkin**10) - 
            (1.5239174522997856e7*mcMS**9)/mbkin**9 + (21690477917894546*mcMS**10)/
            (13749310575*mbkin**12) + (90359478398325128*mcMS**10)/
            (24748759035*mbkin**11) + (9.226033390994718e6*mcMS**10)/mbkin**10 - 
            (532343803520211863*mcMS**11)/(362981799180*mbkin**13) - 
            (2780470935305471923*mcMS**11)/(816709048155*mbkin**12) - 
            (4.00258832555982e6*mcMS**11)/mbkin**11 + (5705997898626787*mcMS**12)/
            (5076668520*mbkin**14) + (35323544788363849*mcMS**12)/
            (13499323110*mbkin**13) + (1.0657904615957544e6*mcMS**12)/mbkin**12 - 
            (2774635337862961*mcMS**13)/(3928374450*mbkin**15) - 
            (61366065765406*mcMS**13)/(37216179*mbkin**14) - 
            (22555.502580515895*mcMS**13)/mbkin**13 + 
            (2125863088351303871*mcMS**14)/(5898454236675*mbkin**16) + 
            (8957084938799491772*mcMS**14)/(10617217626015*mbkin**15) - 
            (136753.9567265611*mcMS**14)/mbkin**14 - (578233922940817183*mcMS**15)/
            (3932302824450*mbkin**17) - (1221278856677814014*mcMS**15)/
            (3539072542005*mbkin**16) + (78630.19954340154*mcMS**15)/mbkin**15 + 
            (133675069698713*mcMS**16)/(2853371664*mbkin**18) + 
            (74294729010461893*mcMS**16)/(674109055620*mbkin**17) - 
            (3597305435872*mcMS**16)/(130945815*mbkin**16) - 
            (18203849412067261*mcMS**17)/(1620585406440*mbkin**19) - 
            (32194885186226363*mcMS**17)/(1215439054830*mbkin**18) + 
            (899387281568*mcMS**17)/(130945815*mbkin**17) + 
            (34767181856381477*mcMS**18)/(18231585822450*mbkin**20) + 
            (73954207321324426*mcMS**18)/(16408427240205*mbkin**19) - 
            (158724542432*mcMS**18)/(130945815*mbkin**18) - 
            (3593079672637649*mcMS**19)/(17579706744600*mbkin**21) - 
            (65107596633353569*mcMS**19)/(134484756596190*mbkin**20) + 
            (2519562016*mcMS**19)/(18706545*mbkin**19) + (6920718196157*mcMS**20)/
            (665354392560*mbkin**22) + (7392400524989*mcMS**20)/
            (299409476652*mbkin**21) - (2411168*mcMS**20)/(340119*mbkin**20) - 
            (3820703038853*mcMS**2)/(15320660355*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (30565624310824*mcMS**2)/(45961981065*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (7641406077706*mcMS**2)/(15320660355*mbkin**2*(1 - mcMS/mbkin)**2) + 
            (1324623229305471443*mcMS**3)/(401094888093900*mbkin**5*
            (1 - mcMS/mbkin)**2) + (2649246458610942886*mcMS**3)/
            (300821166070425*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (1324623229305471443*mcMS**3)/(200547444046950*mbkin**3*
            (1 - mcMS/mbkin)**2) - (5514257196652051*mcMS**4)/
            (231980849100*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (11028514393304102*mcMS**4)/(173985636825*mbkin**5*(1 - mcMS/mbkin)**
                2) + (5514257196652051*mcMS**4)/(115990424550*mbkin**4*
            (1 - mcMS/mbkin)**2) + (2445534345846504631*mcMS**5)/
            (21110257268100*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (4891068691693009262*mcMS**5)/(15832692951075*mbkin**6*
            (1 - mcMS/mbkin)**2) - (2445534345846504631*mcMS**5)/
            (10555128634050*mbkin**5*(1 - mcMS/mbkin)**2) - 
            (73546477272645373*mcMS**6)/(177397119900*mbkin**8*(1 - mcMS/mbkin)**
                2) - (147092954545290746*mcMS**6)/(133047839925*mbkin**7*
            (1 - mcMS/mbkin)**2) + (73546477272645373*mcMS**6)/
            (88698559950*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (70971441129844033*mcMS**7)/(62088991965*mbkin**9*(1 - mcMS/mbkin)**
                2) + (567771529038752264*mcMS**7)/(186266975895*mbkin**8*
            (1 - mcMS/mbkin)**2) - (141942882259688066*mcMS**7)/
            (62088991965*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (5292307230847823*mcMS**8)/(2111870475*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (42338457846782584*mcMS**8)/(6335611425*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (10584614461695646*mcMS**8)/(2111870475*mbkin**8*(1 - mcMS/mbkin)**2) + 
            (1883705490346937*mcMS**9)/(422374095*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (15069643922775496*mcMS**9)/(1267122285*mbkin**10*(1 - mcMS/mbkin)**
                2) - (3767410980693874*mcMS**9)/(422374095*mbkin**9*
            (1 - mcMS/mbkin)**2) - (3996921465647711*mcMS**10)/
            (612317475*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (31975371725181688*mcMS**10)/(1836952425*mbkin**11*(1 - mcMS/mbkin)**
                2) + (7993842931295422*mcMS**10)/(612317475*mbkin**10*
            (1 - mcMS/mbkin)**2) + (34379222836281097*mcMS**11)/
            (4341887550*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (137516891345124388*mcMS**11)/(6512831325*mbkin**12*
            (1 - mcMS/mbkin)**2) - (34379222836281097*mcMS**11)/
            (2170943775*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (991151981540419*mcMS**12)/(124053930*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (3964607926161676*mcMS**12)/(186080895*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (991151981540419*mcMS**12)/(62026965*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (2648132431926677*mcMS**13)/(394717050*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (10592529727706708*mcMS**13)/(592075575*mbkin**14*(1 - mcMS/mbkin)**
                2) - (2648132431926677*mcMS**13)/(197358525*mbkin**13*
            (1 - mcMS/mbkin)**2) - (223279079747087713*mcMS**14)/
            (47760763050*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (893116318988350852*mcMS**14)/(71641144575*mbkin**15*
            (1 - mcMS/mbkin)**2) + (223279079747087713*mcMS**14)/
            (23880381525*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (4765673910687463*mcMS**15)/(1773971199*mbkin**17*(1 - mcMS/mbkin)**
                2) + (38125391285499704*mcMS**15)/(5321913597*mbkin**16*
            (1 - mcMS/mbkin)**2) - (9531347821374926*mcMS**15)/
            (1773971199*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (55872539918685143*mcMS**16)/(44349279975*mbkin**18*
            (1 - mcMS/mbkin)**2) - (446980319349481144*mcMS**16)/
            (133047839925*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (111745079837370286*mcMS**16)/(44349279975*mbkin**16*
            (1 - mcMS/mbkin)**2) + (251792791028627*mcMS**17)/
            (530675145*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (2014342328229016*mcMS**17)/(1592025435*mbkin**18*(1 - mcMS/mbkin)**
                2) - (503585582057254*mcMS**17)/(530675145*mbkin**17*
            (1 - mcMS/mbkin)**2) - (4832108312706614*mcMS**18)/
            (34493884425*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (38656866501652912*mcMS**18)/(103481653275*mbkin**19*
            (1 - mcMS/mbkin)**2) + (9664216625413228*mcMS**18)/
            (34493884425*mbkin**18*(1 - mcMS/mbkin)**2) + 
            (1436625073319083*mcMS**19)/(45991845900*mbkin**21*(1 - mcMS/mbkin)**
                2) + (2873250146638166*mcMS**19)/(34493884425*mbkin**20*
            (1 - mcMS/mbkin)**2) - (1436625073319083*mcMS**19)/
            (22995922950*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (20887342577910533*mcMS**20)/(4222051453620*mbkin**22*
            (1 - mcMS/mbkin)**2) - (41774685155821066*mcMS**20)/
            (3166538590215*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (20887342577910533*mcMS**20)/(2111025726810*mbkin**20*
            (1 - mcMS/mbkin)**2) + (198978903088724189*mcMS**21)/
            (401094888093900*mbkin**23*(1 - mcMS/mbkin)**2) + 
            (397957806177448378*mcMS**21)/(300821166070425*mbkin**22*
            (1 - mcMS/mbkin)**2) - (198978903088724189*mcMS**21)/
            (200547444046950*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (24672660896281*mcMS**22)/(1041804904140*mbkin**24*(1 - mcMS/mbkin)**
                2) - (49345321792562*mcMS**22)/(781353678105*mbkin**23*
            (1 - mcMS/mbkin)**2) + (24672660896281*mcMS**22)/
            (520902452070*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (2291073463739113*mcMS)/(110848041800496*mbkin**3*(1 - mcMS/mbkin)) + 
            (2291073463739113*mcMS)/(41568015675186*mbkin**2*(1 - mcMS/mbkin)) - 
            (190159097490346379*mcMS)/(886784334403968*mbkin*(1 - mcMS/mbkin)) - 
            (66120814157661443*mcMS**2)/(332544125401488*mbkin**4*
            (1 - mcMS/mbkin)) - (66120814157661443*mcMS**2)/
            (124704047025558*mbkin**3*(1 - mcMS/mbkin)) + 
            (5488027575085899769*mcMS**2)/(2660353003211904*mbkin**2*
            (1 - mcMS/mbkin)) + (1166928234496119443*mcMS**3)/
            (962627731425360*mbkin**5*(1 - mcMS/mbkin)) + 
            (1166928234496119443*mcMS**3)/(360985399284510*mbkin**4*
            (1 - mcMS/mbkin)) - (96855043463177913769*mcMS**3)/
            (7701021851402880*mbkin**3*(1 - mcMS/mbkin)) - 
            (118345817852851093*mcMS**4)/(21391727365008*mbkin**6*
            (1 - mcMS/mbkin)) - (118345817852851093*mcMS**4)/
            (8021897761878*mbkin**5*(1 - mcMS/mbkin)) + 
            (9822702881786640719*mcMS**4)/(171133818920064*mbkin**4*
            (1 - mcMS/mbkin)) + (49345371782260483*mcMS**5)/
            (2573870939640*mbkin**7*(1 - mcMS/mbkin)) + 
            (49345371782260483*mcMS**5)/(965201602365*mbkin**6*(1 - mcMS/mbkin)) - 
            (4095665857927620089*mcMS**5)/(20590967517120*mbkin**5*
            (1 - mcMS/mbkin)) - (146208599147473679*mcMS**6)/
            (2831258033604*mbkin**8*(1 - mcMS/mbkin)) - 
            (292417198294947358*mcMS**6)/(2123443525203*mbkin**7*
            (1 - mcMS/mbkin)) + (12135313729240315357*mcMS**6)/
            (22650064268832*mbkin**6*(1 - mcMS/mbkin)) + (64819098503293*mcMS**7)/
            (583644204*mbkin**9*(1 - mcMS/mbkin)) + (129638197006586*mcMS**7)/
            (437733153*mbkin**8*(1 - mcMS/mbkin)) - (5379985175773319*mcMS**7)/
            (4669153632*mbkin**7*(1 - mcMS/mbkin)) - (7714866552402197*mcMS**8)/
            (39653473860*mbkin**10*(1 - mcMS/mbkin)) - 
            (15429733104804394*mcMS**8)/(29740105395*mbkin**9*(1 - mcMS/mbkin)) + 
            (640333923849382351*mcMS**8)/(317227790880*mbkin**8*
            (1 - mcMS/mbkin)) + (2242974842855321*mcMS**9)/(7977621960*mbkin**11*
            (1 - mcMS/mbkin)) + (2242974842855321*mcMS**9)/(2991608235*mbkin**10*
            (1 - mcMS/mbkin)) - (186166911956991643*mcMS**9)/
            (63820975680*mbkin**9*(1 - mcMS/mbkin)) - (1713682422600793*mcMS**10)/
            (5076668520*mbkin**12*(1 - mcMS/mbkin)) - (1713682422600793*mcMS**10)/
            (1903750695*mbkin**11*(1 - mcMS/mbkin)) + 
            (142235641075865819*mcMS**10)/(40613348160*mbkin**10*
            (1 - mcMS/mbkin)) + (13374421001572999*mcMS**11)/
            (39598014456*mbkin**13*(1 - mcMS/mbkin)) + 
            (13374421001572999*mcMS**11)/(14849255421*mbkin**12*
            (1 - mcMS/mbkin)) - (1110076943130558917*mcMS**11)/
            (316784115648*mbkin**11*(1 - mcMS/mbkin)) - 
            (11151579707994493*mcMS**12)/(39598014456*mbkin**14*
            (1 - mcMS/mbkin)) - (11151579707994493*mcMS**12)/
            (14849255421*mbkin**13*(1 - mcMS/mbkin)) + 
            (925581115763542919*mcMS**12)/(316784115648*mbkin**12*
            (1 - mcMS/mbkin)) + (178800900176533*mcMS**13)/(916620705*mbkin**15*
            (1 - mcMS/mbkin)) + (1430407201412264*mcMS**13)/
            (2749862115*mbkin**14*(1 - mcMS/mbkin)) - 
            (14840474714652239*mcMS**13)/(7332965640*mbkin**13*(1 - mcMS/mbkin)) - 
            (32217277599333127*mcMS**14)/(288903880980*mbkin**16*
            (1 - mcMS/mbkin)) - (64434555198666254*mcMS**14)/
            (216677910735*mbkin**15*(1 - mcMS/mbkin)) + 
            (2674034040744649541*mcMS**14)/(2311231047840*mbkin**14*
            (1 - mcMS/mbkin)) + (43352528289259463*mcMS**15)/
            (832722951060*mbkin**17*(1 - mcMS/mbkin)) + 
            (86705056578518926*mcMS**15)/(624542213295*mbkin**16*
            (1 - mcMS/mbkin)) - (3598259848008535429*mcMS**15)/
            (6661783608480*mbkin**15*(1 - mcMS/mbkin)) - 
            (558524342363675*mcMS**16)/(28598565996*mbkin**18*(1 - mcMS/mbkin)) - 
            (1117048684727350*mcMS**16)/(21448924497*mbkin**17*(1 - mcMS/mbkin)) + 
            (46357520416185025*mcMS**16)/(228788527968*mbkin**16*
            (1 - mcMS/mbkin)) + (7230231600513707*mcMS**17)/
            (1258336903824*mbkin**19*(1 - mcMS/mbkin)) + 
            (7230231600513707*mcMS**17)/(471876338934*mbkin**18*
            (1 - mcMS/mbkin)) - (600109222842637681*mcMS**17)/
            (10066695230592*mbkin**17*(1 - mcMS/mbkin)) - 
            (1774149351078739*mcMS**18)/(1389073205520*mbkin**20*
            (1 - mcMS/mbkin)) - (1774149351078739*mcMS**18)/
            (520902452070*mbkin**19*(1 - mcMS/mbkin)) + 
            (147254396139535337*mcMS**18)/(11112585644160*mbkin**18*
            (1 - mcMS/mbkin)) + (64726178334967321*mcMS**19)/
            (320875910475120*mbkin**21*(1 - mcMS/mbkin)) + 
            (64726178334967321*mcMS**19)/(120328466428170*mbkin**20*
            (1 - mcMS/mbkin)) - (5372272801802287643*mcMS**19)/
            (2567007283800960*mbkin**19*(1 - mcMS/mbkin)) - 
            (369023890038773819*mcMS**20)/(18289926897081840*mbkin**22*
            (1 - mcMS/mbkin)) - (369023890038773819*mcMS**20)/
            (6858722586405690*mbkin**21*(1 - mcMS/mbkin)) + 
            (30628982873218226977*mcMS**20)/(146319415176654720*mbkin**20*
            (1 - mcMS/mbkin)) + (22826272832761*mcMS**21)/(23753151814392*
            mbkin**23*(1 - mcMS/mbkin)) + (22826272832761*mcMS**21)/
            (8907431930397*mbkin**22*(1 - mcMS/mbkin)) - 
            (1894580645119163*mcMS**21)/(190025214515136*mbkin**21*
            (1 - mcMS/mbkin)) - (688*mcMS**2)/(729*mbkin**7*(-1 + mcMS/mbkin)) - 
            (5504*mcMS**2)/(2187*mbkin**6*(-1 + mcMS/mbkin)) + 
            (1376*mcMS**2)/(729*mbkin**5*(-1 + mcMS/mbkin)) + 
            (1760*mcMS**4)/(27*mbkin**9*(-1 + mcMS/mbkin)) + 
            (14080*mcMS**4)/(81*mbkin**8*(-1 + mcMS/mbkin)) - 
            (3520*mcMS**4)/(27*mbkin**7*(-1 + mcMS/mbkin)) + 
            (1456*mcMS**6)/(27*mbkin**11*(-1 + mcMS/mbkin)) + 
            (11648*mcMS**6)/(81*mbkin**10*(-1 + mcMS/mbkin)) - 
            (2912*mcMS**6)/(27*mbkin**9*(-1 + mcMS/mbkin)) + 
            (688*mcMS**2)/(729*mbkin**7*(1 + mcMS/mbkin)) + 
            (5504*mcMS**2)/(2187*mbkin**6*(1 + mcMS/mbkin)) - 
            (1376*mcMS**2)/(729*mbkin**5*(1 + mcMS/mbkin)) - 
            (1760*mcMS**4)/(27*mbkin**9*(1 + mcMS/mbkin)) - 
            (14080*mcMS**4)/(81*mbkin**8*(1 + mcMS/mbkin)) + 
            (3520*mcMS**4)/(27*mbkin**7*(1 + mcMS/mbkin)) - 
            (1456*mcMS**6)/(27*mbkin**11*(1 + mcMS/mbkin)) - 
            (11648*mcMS**6)/(81*mbkin**10*(1 + mcMS/mbkin)) + 
            (2912*mcMS**6)/(27*mbkin**9*(1 + mcMS/mbkin)) - 
            14560/(243*mbkin**3*(mbkin**2 - mcMS**2)) - 
            116480/(729*mbkin**2*(mbkin**2 - mcMS**2)) + 
            526648/(3645*mbkin*(mbkin**2 - mcMS**2)) - (1484*mbkin)/
            (243*mcMS**2*(mbkin**2 - mcMS**2)) + (11776*mcMS**2)/
            (243*mbkin**5*(mbkin**2 - mcMS**2)) + (94208*mcMS**2)/
            (729*mbkin**4*(mbkin**2 - mcMS**2)) + (10508*mcMS**2)/
            (81*mbkin**3*(mbkin**2 - mcMS**2)) + (15488*mcMS**4)/
            (243*mbkin**7*(mbkin**2 - mcMS**2)) + (123904*mcMS**4)/
            (729*mbkin**6*(mbkin**2 - mcMS**2)) - (28336*mcMS**4)/
            (81*mbkin**5*(mbkin**2 - mcMS**2)) + (115552*mcMS**6)/
            (729*mbkin**9*(mbkin**2 - mcMS**2)) + (924416*mcMS**6)/
            (2187*mbkin**8*(mbkin**2 - mcMS**2)) - (312844*mcMS**6)/
            (729*mbkin**7*(mbkin**2 - mcMS**2)) + (3392*mcMS**8)/
            (243*mbkin**11*(mbkin**2 - mcMS**2)) + (27136*mcMS**8)/
            (729*mbkin**10*(mbkin**2 - mcMS**2)) + (5576*mcMS**8)/
            (27*mbkin**9*(mbkin**2 - mcMS**2)) + (2816*mcMS**10)/
            (243*mbkin**13*(mbkin**2 - mcMS**2)) + (22528*mcMS**10)/
            (729*mbkin**12*(mbkin**2 - mcMS**2)) - (22724*mcMS**10)/
            (135*mbkin**11*(mbkin**2 - mcMS**2)) + (256*np.pi**2)/(27*mbkin**5) + 
            (2048*np.pi**2)/(81*mbkin**4) + (42550*np.pi**2)/(729*mbkin**3) - 
            (2*np.pi**2)/(3*mbkin**2) - (40*np.pi**2)/(27*mbkin) + 
            (231412726775303*mcMS*np.pi**2)/(233746793280*mbkin) + 
            (418*mcMS**2*np.pi**2)/(9*mbkin**5) - (160*mcMS**2*np.pi**2)/(9*mbkin**4) - 
            (320*mcMS**2*np.pi**2)/(9*mbkin**3) - (19868355231013*mcMS**2*np.pi**2)/
            (1565268705*mbkin**2) + (972602406380027*mcMS**3*np.pi**2)/
            (12843230400*mbkin**3) - (16*mcMS**4*np.pi**2)/(3*mbkin**8) - 
            (7243*mcMS**4*np.pi**2)/(1458*mbkin**7) + (850*mcMS**4*np.pi**2)/(3*mbkin**6) + 
            (4184*mcMS**4*np.pi**2)/(9*mbkin**5) - (4701106426898551*mcMS**4*np.pi**2)/
            (16856739900*mbkin**4) + (38019171359233*mcMS**5*np.pi**2)/
            (53887680*mbkin**5) + (12190*mcMS**6*np.pi**2)/(243*mbkin**9) - 
            (1088*mcMS**6*np.pi**2)/(27*mbkin**7) - (433166649238*mcMS**6*np.pi**2)/
            (331695*mbkin**6) + (84997235817281*mcMS**7*np.pi**2)/
            (46702656*mbkin**7) + (9389*mcMS**8*np.pi**2)/(810*mbkin**11) + 
            (46*mcMS**8*np.pi**2)/(9*mbkin**10) + (184*mcMS**8*np.pi**2)/(9*mbkin**9) - 
            (165181234837501*mcMS**8*np.pi**2)/(85135050*mbkin**8) + 
            (206230056355249*mcMS**9*np.pi**2)/(129729600*mbkin**9) - 
            (519101215868*mcMS**10*np.pi**2)/(521235*mbkin**10) + 
            (460448316031807*mcMS**11*np.pi**2)/(980755776*mbkin**11) - 
            (25919209656889*mcMS**12*np.pi**2)/(160540380*mbkin**12) + 
            (1474778611823953*mcMS**13*np.pi**2)/(38529691200*mbkin**13) - 
            (1752616483195*mcMS**14*np.pi**2)/(313053741*mbkin**14) + 
            (12724616134177*mcMS**15*np.pi**2)/(33392399040*mbkin**15) + 
            (430*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(729*mbkin**3) - 
            (5278*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(27*mbkin**7) - 
            (3740*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(27*mbkin**3) - 
            (8*np.pi**4)/(3*mbkin**3) - (6*mcMS**4*np.pi**4)/mbkin**6 - 
            (8*mcMS**4*np.pi**4)/mbkin**5 - (37*mcMS**4*np.pi**4)/(162*mbkin**4) + 
            (-304/(81*mbkin**3) - (160*mcMS**2)/(9*mbkin**5) + 
            (464*mcMS**4)/(27*mbkin**7) + (4400*mcMS**6)/(81*mbkin**9) - 
            (32*mcMS**8)/(3*mbkin**11) + (416*np.sqrt(0j + mcMS**2/mbkin**2))/
                (243*mbkin**5) + (3328*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**4) - 
            (22954*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**3) + 
            (2912*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(9*mbkin**9) + 
            (23296*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**8) + 
            (17930*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (294080*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**5) + 
            (2352640*(mcMS**2/mbkin**2)**(3/2))/(2187*mbkin**4) + 
            (550028*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**3) - 
            (1936*mcMS)/(729*mbkin**6*(1 + mcMS/mbkin)) - (15488*mcMS)/
                (2187*mbkin**5*(1 + mcMS/mbkin)) + (5936*mcMS)/
                (729*mbkin**4*(1 + mcMS/mbkin)) - (1936*mcMS**2)/
                (729*mbkin**7*(1 + mcMS/mbkin)) - (15488*mcMS**2)/
                (2187*mbkin**6*(1 + mcMS/mbkin)) + (5936*mcMS**2)/
                (729*mbkin**5*(1 + mcMS/mbkin)) - (341600*mcMS**3)/
                (729*mbkin**8*(1 + mcMS/mbkin)) - (2732800*mcMS**3)/
                (2187*mbkin**7*(1 + mcMS/mbkin)) + (540640*mcMS**3)/
                (729*mbkin**6*(1 + mcMS/mbkin)) - (341600*mcMS**4)/
                (729*mbkin**9*(1 + mcMS/mbkin)) - (2732800*mcMS**4)/
                (2187*mbkin**8*(1 + mcMS/mbkin)) + (540640*mcMS**4)/
                (729*mbkin**7*(1 + mcMS/mbkin)) - (1456*mcMS**5)/
                (3*mbkin**10*(1 + mcMS/mbkin)) - (11648*mcMS**5)/
                (9*mbkin**9*(1 + mcMS/mbkin)) + (7280*mcMS**5)/
                (9*mbkin**8*(1 + mcMS/mbkin)) - (1456*mcMS**6)/(3*mbkin**11*
                (1 + mcMS/mbkin)) - (11648*mcMS**6)/(9*mbkin**10*
                (1 + mcMS/mbkin)) + (7280*mcMS**6)/(9*mbkin**9*(1 + mcMS/mbkin)) + 
            (172*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(243*mbkin**3) - 
            (364*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(9*mbkin**7) - 
            (440*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(9*mbkin**3))*
            np.log(1 + mcMS/mbkin) + (13664/(243*mbkin**5) + 109312/(729*mbkin**4) + 
            319247/(729*mbkin**3) - (9275*mbkin)/(729*mcMS**4) + 
            292006/(10935*mbkin*mcMS**2) + (252322*mcMS**2)/(729*mbkin**5) - 
            (105376*mcMS**4)/(729*mbkin**9) - (843008*mcMS**4)/(2187*mbkin**8) + 
            (4098655*mcMS**4)/(2187*mbkin**7) - (12160*mcMS**6)/(243*mbkin**11) - 
            (97280*mcMS**6)/(729*mbkin**10) - (242618*mcMS**6)/(729*mbkin**9) - 
            (2816*mcMS**8)/(81*mbkin**13) - (22528*mcMS**8)/(243*mbkin**12) + 
            (2803943*mcMS**8)/(3645*mbkin**11) + (1000*np.sqrt(0j + mcMS**2/mbkin**2))/
                (243*mbkin**3) + (1544*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (22544*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) - 
            (1708*np.pi**2)/(81*mbkin**3) - (44*mcMS**2*np.pi**2)/(9*mbkin**5) - 
            (13172*mcMS**4*np.pi**2)/(243*mbkin**7) - (760*mcMS**6*np.pi**2)/
                (81*mbkin**9) - (352*mcMS**8*np.pi**2)/(81*mbkin**11))*
            np.log(1 - mcMS**2/mbkin**2) + (-1820/(243*mbkin**3) + 
            (2596*mcMS**2)/(243*mbkin**5) - (35420*mcMS**4)/(243*mbkin**7) - 
            (14840*mcMS**6)/(243*mbkin**9) + (5264*mcMS**8)/(81*mbkin**11))*
            fp.polylog(2, 1 - mbkin**2/mcMS**2) + (-152/(243*mbkin**3) + 
            (880*mcMS**2)/(27*mbkin**5) - (5336*mcMS**4)/(81*mbkin**7) - 
            (77000*mcMS**6)/(243*mbkin**9) + (752*mcMS**8)/(9*mbkin**11) + 
            (4220*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**3) - 
            (40948*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**7) - 
            (212296*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**3))*
            fp.polylog(2, -(mcMS/mbkin)) + (-152/(243*mbkin**3) + 
            (880*mcMS**2)/(27*mbkin**5) - (5336*mcMS**4)/(81*mbkin**7) - 
            (77000*mcMS**6)/(243*mbkin**9) + (752*mcMS**8)/(9*mbkin**11) + 
            (260*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**3) + 
            (85724*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**7) + 
            (595544*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**3))*
            fp.polylog(2, mcMS/mbkin) + (-268/(243*mbkin**3) - 
            (44*mcMS**2)/(3*mbkin**5) + (1564*mcMS**4)/(27*mbkin**7) - 
            (6020*mcMS**6)/(243*mbkin**9) - (6016*mcMS**8)/(81*mbkin**11) - 
            (1250*np.sqrt(0j + mcMS**2/mbkin**2))/(729*mbkin**3) - 
            (11194*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**7) - 
            (95812*(mcMS**2/mbkin**2)**(3/2))/(729*mbkin**3))*
            fp.polylog(2, mcMS**2/mbkin**2)) + np.log(mus**2/mbkin**2)*
            (-262.23663785126166 - 4288/(243*mbkin**5) - 35708/(729*mbkin**4) - 
            12122/(243*mbkin**3) - 100963008379/(1352701350*mbkin**2) - 
            1633867767722/(10956880935*mbkin) + 1484/(243*mbkin*mcMS**2) + 
            (438979919676598877*mcMS)/(1219328459805456*mbkin**3) + 
            (254445926686789393*mcMS)/(326605837447890*mbkin**2) + 
            (21423.734593611563*mcMS)/mbkin - (15778736*mcMS)/
            (1216215*(mbkin - mcMS)) + (7889368*mcMS)/(1216215*mbkin**2*
            (mbkin - mcMS)) + (63114944*mcMS)/(3648645*mbkin*(mbkin - mcMS)) - 
            (640*mcMS**2)/(27*mbkin**6) - (1413496*mcMS**2)/(3645*mbkin**5) - 
            (30300442696187*mcMS**2)/(11111687730*mbkin**4) - 
            (8599509052862984*mcMS**2)/(1516745375145*mbkin**3) - 
            (188063.7215829625*mcMS**2)/mbkin**2 - (4156192*mcMS**2)/
            (81081*mbkin**3*(mbkin - mcMS)) - (33249536*mcMS**2)/
            (243243*mbkin**2*(mbkin - mcMS)) + (8312384*mcMS**2)/
            (81081*mbkin*(mbkin - mcMS)) + (4766090586234644881*mcMS**3)/
            (534793184125200*mbkin**5) + (3206405621051834413*mcMS**3)/
            (164084272402050*mbkin**4) + (955753.0757092843*mcMS**3)/mbkin**3 + 
            (6902024*mcMS**3)/(34749*mbkin**4*(mbkin - mcMS)) + 
            (55216192*mcMS**3)/(104247*mbkin**3*(mbkin - mcMS)) - 
            (13804048*mcMS**3)/(34749*mbkin**2*(mbkin - mcMS)) - 
            (2816*mcMS**4)/(81*mbkin**9) - (5680*mcMS**4)/(243*mbkin**8) + 
            (4422322*mcMS**4)/(3645*mbkin**7) - (1691978805531773411*mcMS**4)/
            (47187633893400*mbkin**6) - (460620994518133679*mcMS**4)/
            (5898454236675*mbkin**5) - (3.2499325005298792e6*mcMS**4)/mbkin**4 - 
            (7088416*mcMS**4)/(13365*mbkin**5*(mbkin - mcMS)) - 
            (56707328*mcMS**4)/(40095*mbkin**4*(mbkin - mcMS)) + 
            (14176832*mcMS**4)/(13365*mbkin**3*(mbkin - mcMS)) + 
            (8330148986242110343*mcMS**5)/(70781450840100*mbkin**7) + 
            (2793642635289043646*mcMS**5)/(10617217626015*mbkin**6) + 
            (7.850254525991183e6*mcMS**5)/mbkin**5 + (11687128*mcMS**5)/
            (10395*mbkin**6*(mbkin - mcMS)) + (93497024*mcMS**5)/
            (31185*mbkin**5*(mbkin - mcMS)) - (23374256*mcMS**5)/
            (10395*mbkin**4*(mbkin - mcMS)) + (34816*mcMS**6)/(243*mbkin**11) + 
            (249152*mcMS**6)/(729*mbkin**10) - (1359868*mcMS**6)/(1215*mbkin**9) - 
            (26834017108471*mcMS**6)/(89594505*mbkin**8) - 
            (31147020312235348*mcMS**6)/(45961981065*mbkin**7) - 
            (1.4071820865510551e7*mcMS**6)/mbkin**6 - (2376544*mcMS**6)/
            (1215*mbkin**7*(mbkin - mcMS)) - (19012352*mcMS**6)/
            (3645*mbkin**6*(mbkin - mcMS)) + (4753088*mcMS**6)/
            (1215*mbkin**5*(mbkin - mcMS)) + (87540297440995541*mcMS**7)/
            (142992829980*mbkin**9) + (29802698486179930*mcMS**7)/
            (21448924497*mbkin**8) + (1.916314416689955e7*mcMS**7)/mbkin**7 + 
            (667736*mcMS**7)/(243*mbkin**8*(mbkin - mcMS)) + 
            (5341888*mcMS**7)/(729*mbkin**7*(mbkin - mcMS)) - 
            (1335472*mcMS**7)/(243*mbkin**6*(mbkin - mcMS)) - 
            (7360*mcMS**8)/(81*mbkin**13) - (59708*mcMS**8)/(243*mbkin**12) + 
            (818756*mcMS**8)/(1215*mbkin**11) - (461633021890066603*mcMS**8)/
            (453727248975*mbkin**10) - (3160066041343928494*mcMS**8)/
            (1361181746925*mbkin**9) - (2.001554762180945e7*mcMS**8)/mbkin**8 - 
            (579232*mcMS**8)/(189*mbkin**9*(mbkin - mcMS)) - 
            (4633856*mcMS**8)/(567*mbkin**8*(mbkin - mcMS)) + 
            (1158464*mcMS**8)/(189*mbkin**7*(mbkin - mcMS)) + 
            (7271806428567029*mcMS**9)/(5237832600*mbkin**11) + 
            (56292741418339991*mcMS**9)/(17677685025*mbkin**10) + 
            (1.602731986410626e7*mcMS**9)/mbkin**9 + (1529224*mcMS**9)/
            (567*mbkin**10*(mbkin - mcMS)) + (12233792*mcMS**9)/
            (1701*mbkin**9*(mbkin - mcMS)) - (3058448*mcMS**9)/
            (567*mbkin**8*(mbkin - mcMS)) - (64439609814012478*mcMS**10)/
            (41247931725*mbkin**12) - (89048507291106604*mcMS**10)/
            (24748759035*mbkin**11) - (9.692907343497835e6*mcMS**10)/mbkin**10 - 
            (2269088*mcMS**10)/(1215*mbkin**11*(mbkin - mcMS)) - 
            (18152704*mcMS**10)/(3645*mbkin**10*(mbkin - mcMS)) + 
            (4538176*mcMS**10)/(1215*mbkin**9*(mbkin - mcMS)) + 
            (243233352943020871*mcMS**11)/(167530061160*mbkin**13) + 
            (548182094578026653*mcMS**11)/(163341809631*mbkin**12) + 
            (4.241506865286068e6*mcMS**11)/mbkin**11 + (408664*mcMS**11)/
            (405*mbkin**12*(mbkin - mcMS)) + (3269312*mcMS**11)/
            (1215*mbkin**11*(mbkin - mcMS)) - (817328*mcMS**11)/
            (405*mbkin**10*(mbkin - mcMS)) - (26916497538128519*mcMS**12)/
            (24198786612*mbkin**14) - (2106756722628178666*mcMS**12)/
            (816709048155*mbkin**13) - (1.1765895563018664e6*mcMS**12)/
            mbkin**12 - (7806752*mcMS**12)/(18711*mbkin**13*(mbkin - mcMS)) - 
            (62454016*mcMS**12)/(56133*mbkin**12*(mbkin - mcMS)) + 
            (15613504*mcMS**12)/(18711*mbkin**11*(mbkin - mcMS)) + 
            (67286517551264009*mcMS**13)/(96301293660*mbkin**15) + 
            (456513643805161814*mcMS**13)/(280878773175*mbkin**14) + 
            (71323.391447423*mcMS**13)/mbkin**13 + (340952*mcMS**13)/
            (2673*mbkin**14*(mbkin - mcMS)) + (2727616*mcMS**13)/
            (8019*mbkin**13*(mbkin - mcMS)) - (681904*mcMS**13)/
            (2673*mbkin**12*(mbkin - mcMS)) - (1261293020695272221*mcMS**14)/
            (3539072542005*mbkin**16) - (4012269145427881388*mcMS**14)/
            (4826008011825*mbkin**15) + (116404.32312467585*mcMS**14)/mbkin**14 - 
            (524768*mcMS**14)/(19305*mbkin**15*(mbkin - mcMS)) - 
            (4198144*mcMS**14)/(57915*mbkin**14*(mbkin - mcMS)) + 
            (1049536*mcMS**14)/(19305*mbkin**13*(mbkin - mcMS)) + 
            (87934389688061327*mcMS**15)/(604969665300*mbkin**17) + 
            (1203273721336601338*mcMS**15)/(3539072542005*mbkin**16) - 
            (71065.89600650764*mcMS**15)/mbkin**15 + (4387112*mcMS**15)/
            (1216215*mbkin**16*(mbkin - mcMS)) + (35096896*mcMS**15)/
            (3648645*mbkin**15*(mbkin - mcMS)) - (8774224*mcMS**15)/
            (1216215*mbkin**14*(mbkin - mcMS)) - (5201187629897093*mcMS**16)/
            (112351509270*mbkin**18) - (192594171980692*mcMS**16)/
            (1773971199*mbkin**17) + (361894073507460953*mcMS**16)/
            (14380993186560*mbkin**16) - (39008*mcMS**16)/(173745*mbkin**17*
            (mbkin - mcMS)) - (312064*mcMS**16)/(521235*mbkin**16*
            (mbkin - mcMS)) + (78016*mcMS**16)/(173745*mbkin**15*
            (mbkin - mcMS)) + (302251708171087*mcMS**17)/(27236729520*
            mbkin**19) + (1865303440131991*mcMS**17)/(71496414990*mbkin**18) - 
            (23453575261548899*mcMS**17)/(3704195214720*mbkin**17) - 
            (327054181915217*mcMS**18)/(173634150690*mbkin**20) - 
            (40463910021343156*mcMS**18)/(9115792911225*mbkin**19) + 
            (65550484012407871*mcMS**18)/(58341074631840*mbkin**18) + 
            (544280957164673*mcMS**19)/(2696436222480*mbkin**21) + 
            (16873686999478429*mcMS**19)/(35390725420050*mbkin**20) - 
            (94715351642409661*mcMS**19)/(755002142294400*mbkin**19) - 
            (34854172217977*mcMS**20)/(3393307402056*mbkin**22) - 
            (1082938390632629*mcMS**20)/(44537159651985*mbkin**21) + 
            (1938569020658903*mcMS**20)/(292346483869440*mbkin**20) + 
            (3820703038853*mcMS**2)/(15320660355*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (30565624310824*mcMS**2)/(45961981065*mbkin**3*(1 - mcMS/mbkin)**2) - 
            (7641406077706*mcMS**2)/(15320660355*mbkin**2*(1 - mcMS/mbkin)**2) - 
            (1324623229305471443*mcMS**3)/(401094888093900*mbkin**5*
            (1 - mcMS/mbkin)**2) - (2649246458610942886*mcMS**3)/
            (300821166070425*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (1324623229305471443*mcMS**3)/(200547444046950*mbkin**3*
            (1 - mcMS/mbkin)**2) + (5514257196652051*mcMS**4)/
            (231980849100*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (11028514393304102*mcMS**4)/(173985636825*mbkin**5*(1 - mcMS/mbkin)**
                2) - (5514257196652051*mcMS**4)/(115990424550*mbkin**4*
            (1 - mcMS/mbkin)**2) - (2445534345846504631*mcMS**5)/
            (21110257268100*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (4891068691693009262*mcMS**5)/(15832692951075*mbkin**6*
            (1 - mcMS/mbkin)**2) + (2445534345846504631*mcMS**5)/
            (10555128634050*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (73546477272645373*mcMS**6)/(177397119900*mbkin**8*(1 - mcMS/mbkin)**
                2) + (147092954545290746*mcMS**6)/(133047839925*mbkin**7*
            (1 - mcMS/mbkin)**2) - (73546477272645373*mcMS**6)/
            (88698559950*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (70971441129844033*mcMS**7)/(62088991965*mbkin**9*(1 - mcMS/mbkin)**
                2) - (567771529038752264*mcMS**7)/(186266975895*mbkin**8*
            (1 - mcMS/mbkin)**2) + (141942882259688066*mcMS**7)/
            (62088991965*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (5292307230847823*mcMS**8)/(2111870475*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (42338457846782584*mcMS**8)/(6335611425*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (10584614461695646*mcMS**8)/(2111870475*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (1883705490346937*mcMS**9)/(422374095*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (15069643922775496*mcMS**9)/(1267122285*mbkin**10*(1 - mcMS/mbkin)**
                2) + (3767410980693874*mcMS**9)/(422374095*mbkin**9*
            (1 - mcMS/mbkin)**2) + (3996921465647711*mcMS**10)/
            (612317475*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (31975371725181688*mcMS**10)/(1836952425*mbkin**11*(1 - mcMS/mbkin)**
                2) - (7993842931295422*mcMS**10)/(612317475*mbkin**10*
            (1 - mcMS/mbkin)**2) - (34379222836281097*mcMS**11)/
            (4341887550*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (137516891345124388*mcMS**11)/(6512831325*mbkin**12*
            (1 - mcMS/mbkin)**2) + (34379222836281097*mcMS**11)/
            (2170943775*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (991151981540419*mcMS**12)/(124053930*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (3964607926161676*mcMS**12)/(186080895*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (991151981540419*mcMS**12)/(62026965*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (2648132431926677*mcMS**13)/(394717050*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (10592529727706708*mcMS**13)/(592075575*mbkin**14*(1 - mcMS/mbkin)**
                2) + (2648132431926677*mcMS**13)/(197358525*mbkin**13*
            (1 - mcMS/mbkin)**2) + (223279079747087713*mcMS**14)/
            (47760763050*mbkin**16*(1 - mcMS/mbkin)**2) + 
            (893116318988350852*mcMS**14)/(71641144575*mbkin**15*
            (1 - mcMS/mbkin)**2) - (223279079747087713*mcMS**14)/
            (23880381525*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (4765673910687463*mcMS**15)/(1773971199*mbkin**17*(1 - mcMS/mbkin)**
                2) - (38125391285499704*mcMS**15)/(5321913597*mbkin**16*
            (1 - mcMS/mbkin)**2) + (9531347821374926*mcMS**15)/
            (1773971199*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (55872539918685143*mcMS**16)/(44349279975*mbkin**18*
            (1 - mcMS/mbkin)**2) + (446980319349481144*mcMS**16)/
            (133047839925*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (111745079837370286*mcMS**16)/(44349279975*mbkin**16*
            (1 - mcMS/mbkin)**2) - (251792791028627*mcMS**17)/
            (530675145*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (2014342328229016*mcMS**17)/(1592025435*mbkin**18*(1 - mcMS/mbkin)**
                2) + (503585582057254*mcMS**17)/(530675145*mbkin**17*
            (1 - mcMS/mbkin)**2) + (4832108312706614*mcMS**18)/
            (34493884425*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (38656866501652912*mcMS**18)/(103481653275*mbkin**19*
            (1 - mcMS/mbkin)**2) - (9664216625413228*mcMS**18)/
            (34493884425*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (1436625073319083*mcMS**19)/(45991845900*mbkin**21*(1 - mcMS/mbkin)**
                2) - (2873250146638166*mcMS**19)/(34493884425*mbkin**20*
            (1 - mcMS/mbkin)**2) + (1436625073319083*mcMS**19)/
            (22995922950*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (20887342577910533*mcMS**20)/(4222051453620*mbkin**22*
            (1 - mcMS/mbkin)**2) + (41774685155821066*mcMS**20)/
            (3166538590215*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (20887342577910533*mcMS**20)/(2111025726810*mbkin**20*
            (1 - mcMS/mbkin)**2) - (198978903088724189*mcMS**21)/
            (401094888093900*mbkin**23*(1 - mcMS/mbkin)**2) - 
            (397957806177448378*mcMS**21)/(300821166070425*mbkin**22*
            (1 - mcMS/mbkin)**2) + (198978903088724189*mcMS**21)/
            (200547444046950*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (24672660896281*mcMS**22)/(1041804904140*mbkin**24*(1 - mcMS/mbkin)**
                2) + (49345321792562*mcMS**22)/(781353678105*mbkin**23*
            (1 - mcMS/mbkin)**2) - (24672660896281*mcMS**22)/
            (520902452070*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (2291073463739113*mcMS)/(110848041800496*mbkin**3*(1 - mcMS/mbkin)) - 
            (2291073463739113*mcMS)/(41568015675186*mbkin**2*(1 - mcMS/mbkin)) + 
            (190159097490346379*mcMS)/(886784334403968*mbkin*(1 - mcMS/mbkin)) + 
            (66120814157661443*mcMS**2)/(332544125401488*mbkin**4*
            (1 - mcMS/mbkin)) + (66120814157661443*mcMS**2)/
            (124704047025558*mbkin**3*(1 - mcMS/mbkin)) - 
            (5488027575085899769*mcMS**2)/(2660353003211904*mbkin**2*
            (1 - mcMS/mbkin)) - (1166928234496119443*mcMS**3)/
            (962627731425360*mbkin**5*(1 - mcMS/mbkin)) - 
            (1166928234496119443*mcMS**3)/(360985399284510*mbkin**4*
            (1 - mcMS/mbkin)) + (96855043463177913769*mcMS**3)/
            (7701021851402880*mbkin**3*(1 - mcMS/mbkin)) + 
            (118345817852851093*mcMS**4)/(21391727365008*mbkin**6*
            (1 - mcMS/mbkin)) + (118345817852851093*mcMS**4)/
            (8021897761878*mbkin**5*(1 - mcMS/mbkin)) - 
            (9822702881786640719*mcMS**4)/(171133818920064*mbkin**4*
            (1 - mcMS/mbkin)) - (49345371782260483*mcMS**5)/
            (2573870939640*mbkin**7*(1 - mcMS/mbkin)) - 
            (49345371782260483*mcMS**5)/(965201602365*mbkin**6*(1 - mcMS/mbkin)) + 
            (4095665857927620089*mcMS**5)/(20590967517120*mbkin**5*
            (1 - mcMS/mbkin)) + (146208599147473679*mcMS**6)/
            (2831258033604*mbkin**8*(1 - mcMS/mbkin)) + 
            (292417198294947358*mcMS**6)/(2123443525203*mbkin**7*
            (1 - mcMS/mbkin)) - (12135313729240315357*mcMS**6)/
            (22650064268832*mbkin**6*(1 - mcMS/mbkin)) - (64819098503293*mcMS**7)/
            (583644204*mbkin**9*(1 - mcMS/mbkin)) - (129638197006586*mcMS**7)/
            (437733153*mbkin**8*(1 - mcMS/mbkin)) + (5379985175773319*mcMS**7)/
            (4669153632*mbkin**7*(1 - mcMS/mbkin)) + (7714866552402197*mcMS**8)/
            (39653473860*mbkin**10*(1 - mcMS/mbkin)) + 
            (15429733104804394*mcMS**8)/(29740105395*mbkin**9*(1 - mcMS/mbkin)) - 
            (640333923849382351*mcMS**8)/(317227790880*mbkin**8*
            (1 - mcMS/mbkin)) - (2242974842855321*mcMS**9)/(7977621960*mbkin**11*
            (1 - mcMS/mbkin)) - (2242974842855321*mcMS**9)/(2991608235*mbkin**10*
            (1 - mcMS/mbkin)) + (186166911956991643*mcMS**9)/
            (63820975680*mbkin**9*(1 - mcMS/mbkin)) + (1713682422600793*mcMS**10)/
            (5076668520*mbkin**12*(1 - mcMS/mbkin)) + (1713682422600793*mcMS**10)/
            (1903750695*mbkin**11*(1 - mcMS/mbkin)) - 
            (142235641075865819*mcMS**10)/(40613348160*mbkin**10*
            (1 - mcMS/mbkin)) - (13374421001572999*mcMS**11)/
            (39598014456*mbkin**13*(1 - mcMS/mbkin)) - 
            (13374421001572999*mcMS**11)/(14849255421*mbkin**12*
            (1 - mcMS/mbkin)) + (1110076943130558917*mcMS**11)/
            (316784115648*mbkin**11*(1 - mcMS/mbkin)) + 
            (11151579707994493*mcMS**12)/(39598014456*mbkin**14*
            (1 - mcMS/mbkin)) + (11151579707994493*mcMS**12)/
            (14849255421*mbkin**13*(1 - mcMS/mbkin)) - 
            (925581115763542919*mcMS**12)/(316784115648*mbkin**12*
            (1 - mcMS/mbkin)) - (178800900176533*mcMS**13)/(916620705*mbkin**15*
            (1 - mcMS/mbkin)) - (1430407201412264*mcMS**13)/
            (2749862115*mbkin**14*(1 - mcMS/mbkin)) + 
            (14840474714652239*mcMS**13)/(7332965640*mbkin**13*(1 - mcMS/mbkin)) + 
            (32217277599333127*mcMS**14)/(288903880980*mbkin**16*
            (1 - mcMS/mbkin)) + (64434555198666254*mcMS**14)/
            (216677910735*mbkin**15*(1 - mcMS/mbkin)) - 
            (2674034040744649541*mcMS**14)/(2311231047840*mbkin**14*
            (1 - mcMS/mbkin)) - (43352528289259463*mcMS**15)/
            (832722951060*mbkin**17*(1 - mcMS/mbkin)) - 
            (86705056578518926*mcMS**15)/(624542213295*mbkin**16*
            (1 - mcMS/mbkin)) + (3598259848008535429*mcMS**15)/
            (6661783608480*mbkin**15*(1 - mcMS/mbkin)) + 
            (558524342363675*mcMS**16)/(28598565996*mbkin**18*(1 - mcMS/mbkin)) + 
            (1117048684727350*mcMS**16)/(21448924497*mbkin**17*(1 - mcMS/mbkin)) - 
            (46357520416185025*mcMS**16)/(228788527968*mbkin**16*
            (1 - mcMS/mbkin)) - (7230231600513707*mcMS**17)/
            (1258336903824*mbkin**19*(1 - mcMS/mbkin)) - 
            (7230231600513707*mcMS**17)/(471876338934*mbkin**18*
            (1 - mcMS/mbkin)) + (600109222842637681*mcMS**17)/
            (10066695230592*mbkin**17*(1 - mcMS/mbkin)) + 
            (1774149351078739*mcMS**18)/(1389073205520*mbkin**20*
            (1 - mcMS/mbkin)) + (1774149351078739*mcMS**18)/
            (520902452070*mbkin**19*(1 - mcMS/mbkin)) - 
            (147254396139535337*mcMS**18)/(11112585644160*mbkin**18*
            (1 - mcMS/mbkin)) - (64726178334967321*mcMS**19)/
            (320875910475120*mbkin**21*(1 - mcMS/mbkin)) - 
            (64726178334967321*mcMS**19)/(120328466428170*mbkin**20*
            (1 - mcMS/mbkin)) + (5372272801802287643*mcMS**19)/
            (2567007283800960*mbkin**19*(1 - mcMS/mbkin)) + 
            (369023890038773819*mcMS**20)/(18289926897081840*mbkin**22*
            (1 - mcMS/mbkin)) + (369023890038773819*mcMS**20)/
            (6858722586405690*mbkin**21*(1 - mcMS/mbkin)) - 
            (30628982873218226977*mcMS**20)/(146319415176654720*mbkin**20*
            (1 - mcMS/mbkin)) - (22826272832761*mcMS**21)/(23753151814392*
            mbkin**23*(1 - mcMS/mbkin)) - (22826272832761*mcMS**21)/
            (8907431930397*mbkin**22*(1 - mcMS/mbkin)) + 
            (1894580645119163*mcMS**21)/(190025214515136*mbkin**21*
            (1 - mcMS/mbkin)) - 89848/(3645*mbkin*(mbkin**2 - mcMS**2)) + 
            (1484*mbkin)/(243*mcMS**2*(mbkin**2 - mcMS**2)) - 
            (55076*mcMS**2)/(243*mbkin**3*(mbkin**2 - mcMS**2)) + 
            (54032*mcMS**4)/(243*mbkin**5*(mbkin**2 - mcMS**2)) + 
            (81740*mcMS**6)/(729*mbkin**7*(mbkin**2 - mcMS**2)) - 
            (56968*mcMS**8)/(243*mbkin**9*(mbkin**2 - mcMS**2)) + 
            (176356*mcMS**10)/(1215*mbkin**11*(mbkin**2 - mcMS**2)) + 
            (440*np.pi**2)/(81*mbkin**3) + (2*np.pi**2)/(3*mbkin**2) + 
            (40*np.pi**2)/(27*mbkin) - (475536164252167*mcMS*np.pi**2)/
            (239059220400*mbkin) - (616*mcMS**2*np.pi**2)/(81*mbkin**5) + 
            (160*mcMS**2*np.pi**2)/(9*mbkin**4) + (320*mcMS**2*np.pi**2)/(9*mbkin**3) + 
            (144185877961357*mcMS**2*np.pi**2)/(7826343525*mbkin**2) - 
            (141923924754233*mcMS**3*np.pi**2)/(1481911200*mbkin**3) - 
            (704*mcMS**4*np.pi**2)/(81*mbkin**7) - (40*mcMS**4*np.pi**2)/(3*mbkin**6) - 
            (512*mcMS**4*np.pi**2)/(9*mbkin**5) + (9435512209148251*mcMS**4*np.pi**2)/
            (28897268400*mbkin**4) - (519407201064967*mcMS**5*np.pi**2)/
            (656756100*mbkin**5) - (368*mcMS**6*np.pi**2)/(81*mbkin**9) + 
            (1088*mcMS**6*np.pi**2)/(27*mbkin**7) + (233317981462754*mcMS**6*np.pi**2)/
            (164189025*mbkin**6) - (103028797584197*mcMS**7*np.pi**2)/
            (53071200*mbkin**7) - (152*mcMS**8*np.pi**2)/(9*mbkin**11) - 
            (46*mcMS**8*np.pi**2)/(9*mbkin**10) - (184*mcMS**8*np.pi**2)/(9*mbkin**9) + 
            (12514926772910899*mcMS**8*np.pi**2)/(6129723600*mbkin**8) - 
            (1451231083442867*mcMS**9*np.pi**2)/(875674800*mbkin**9) + 
            (48392827123139*mcMS**10*np.pi**2)/(46911150*mbkin**10) - 
            (3108958914635281*mcMS**11*np.pi**2)/(6421615200*mbkin**11) + 
            (145335055435249*mcMS**12*np.pi**2)/(875674800*mbkin**12) - 
            (210683940381223*mcMS**13*np.pi**2)/(5366635560*mbkin**13) + 
            (376814615761301*mcMS**14*np.pi**2)/(65741285610*mbkin**14) - 
            (58533527579083*mcMS**15*np.pi**2)/(150265795680*mbkin**15) - 
            (172*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(243*mbkin**3) + 
            (1820*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(9*mbkin**7) + 
            (440*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(3*mbkin**3) + 
            (32/(81*mbkin**3) - (616*mcMS**2)/(81*mbkin**5) - (1813*mcMS**2)/
                (9*mbkin**2) + (4112*mcMS**4)/(243*mbkin**7) + (3320*mcMS**4)/
                (9*mbkin**6) + (13280*mcMS**4)/(27*mbkin**5) + (50735*mcMS**4)/
                (18*mbkin**4) - (2032*mcMS**6)/(27*mbkin**9) + (5917*mcMS**6)/
                (3*mbkin**6) - (4288*mcMS**8)/(81*mbkin**11) - (8833*mcMS**8)/
                (18*mbkin**8))*np.log(mcMS**2/mbkin**2)**2 - 
            (3577*mcMS**4*np.log(mcMS**2/mbkin**2)**3)/(3*mbkin**4) + 
            (4529650481847077/13411222264440 + 304/(81*mbkin**3) + 
            15778736/(405405*mbkin**2) + 63114944/(729729*mbkin) - 
            (12148554800*mcMS)/(26189163*mbkin**3) - (8777037184*mcMS)/
                (8729721*mbkin**2) - (2409611921485918004*mcMS)/(796291321951125*
                mbkin) + (160*mcMS**2)/(9*mbkin**5) + (100089736*mcMS**2)/
                (34749*mbkin**4) + (646098304*mcMS**2)/(104247*mbkin**3) + 
            (131181854232014*mcMS**2)/(10672286625*mbkin**2) - 
            (13292598896*mcMS**3)/(984555*mbkin**5) - (258245697152*mcMS**3)/
                (8860995*mbkin**4) - (7211141123397511*mcMS**3)/(214923433725*
                mbkin**3) - (464*mcMS**4)/(27*mbkin**7) + (21018966764*mcMS**4)/
                (405405*mbkin**6) + (19561361056*mcMS**4)/(173745*mbkin**5) + 
            (3322103239097477*mcMS**4)/(43345902600*mbkin**4) - 
            (2538967072*mcMS**5)/(15795*mbkin**7) - (1108328960*mcMS**5)/
                (3159*mbkin**6) - (296155042786888*mcMS**5)/(1915538625*mbkin**5) - 
            (4400*mcMS**6)/(81*mbkin**9) + (139918240*mcMS**6)/(351*mbkin**8) + 
            (8277677888*mcMS**6)/(9477*mbkin**7) + (183584247858181*mcMS**6)/
                (703667250*mbkin**6) - (1956154784*mcMS**7)/(2457*mbkin**9) - 
            (12895944704*mcMS**7)/(7371*mbkin**8) - (20563912883023*mcMS**7)/
                (58046625*mbkin**7) + (32*mcMS**8)/(3*mbkin**11) + 
            (24224006368*mcMS**8)/(18711*mbkin**10) + (800142123712*mcMS**8)/
                (280665*mbkin**9) + (859851409549*mcMS**8)/(2211300*mbkin**8) - 
            (700068352*mcMS**9)/(405*mbkin**11) - (13893870848*mcMS**9)/
                (3645*mbkin**10) - (34893257460322*mcMS**9)/(98513415*mbkin**9) + 
            (257056048*mcMS**10)/(135*mbkin**12) + (1021246400*mcMS**10)/
                (243*mbkin**11) + (9520712957158453*mcMS**10)/(34479695250*
                mbkin**10) - (3602891840*mcMS**11)/(2079*mbkin**13) - 
            (214822247168*mcMS**11)/(56133*mbkin**12) - 
            (1708860475928401*mcMS**11)/(9030396375*mbkin**11) + 
            (1158984296*mcMS**12)/(891*mbkin**14) + (23041697408*mcMS**12)/
                (8019*mbkin**13) + (9753751392280913*mcMS**12)/(84283699500*
                mbkin**12) - (46381886432*mcMS**13)/(57915*mbkin**15) - 
            (307417939456*mcMS**13)/(173745*mbkin**14) - 
            (401875960070852*mcMS**13)/(6403371975*mbkin**13) + 
            (487143200864*mcMS**14)/(1216215*mbkin**16) + 
            (1076326757696*mcMS**14)/(1216215*mbkin**15) + 
            (1071889963712831*mcMS**14)/(36522936450*mbkin**14) - 
            (3093431392*mcMS**15)/(19305*mbkin**17) - (258362278912*mcMS**15)/
                (729729*mbkin**16) - (28039427185820671*mcMS**15)/
                (2465298210375*mbkin**15) + (5859280*mcMS**16)/(117*mbkin**18) + 
            (38838656*mcMS**16)/(351*mbkin**17) + (3274919*mcMS**16)/
                (936*mbkin**16) - (37226000*mcMS**17)/(3159*mbkin**19) - 
            (246755200*mcMS**17)/(9477*mbkin**18) - (43740550*mcMS**17)/
                (53703*mbkin**17) + (11719928*mcMS**18)/(5967*mbkin**20) + 
            (2719023296*mcMS**18)/(626535*mbkin**19) + (33694793*mcMS**18)/
                (250614*mbkin**18) - (2394896*mcMS**19)/(11583*mbkin**21) - 
            (79373696*mcMS**19)/(173745*mbkin**20) - (812554*mcMS**19)/
                (57915*mbkin**19) + (10548860*mcMS**20)/(1020357*mbkin**22) + 
            (69923872*mcMS**20)/(3061071*mbkin**21) + (2185121*mcMS**20)/
                (3139560*mbkin**20) + (104*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**3) + 
            (5912*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (70064*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) - 
            (376528*mcMS**2)/(1053*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (3012224*mcMS**2)/(3159*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (753056*mcMS**2)/(1053*mbkin**2*(1 - mcMS/mbkin)**2) + 
            (429497312*mcMS**3)/(89505*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (3435978496*mcMS**3)/(268515*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (858994624*mcMS**3)/(89505*mbkin**3*(1 - mcMS/mbkin)**2) - 
            (234545329016*mcMS**4)/(6891885*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (1876362632128*mcMS**4)/(20675655*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (469090658032*mcMS**4)/(6891885*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (75343294960*mcMS**5)/(459459*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (602746359680*mcMS**5)/(1378377*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (150686589920*mcMS**5)/(459459*mbkin**5*(1 - mcMS/mbkin)**2) - 
            (79048041848*mcMS**6)/(135135*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (632384334784*mcMS**6)/(405405*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (158096083696*mcMS**6)/(135135*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (566619520*mcMS**7)/(351*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (4532956160*mcMS**7)/(1053*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (1133239040*mcMS**7)/(351*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (205356871136*mcMS**8)/(57915*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (1642854969088*mcMS**8)/(173745*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (410713742272*mcMS**8)/(57915*mbkin**8*(1 - mcMS/mbkin)**2) + 
            (366047261248*mcMS**9)/(57915*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (2928378089984*mcMS**9)/(173745*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (732094522496*mcMS**9)/(57915*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (41256072896*mcMS**10)/(4455*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (330048583168*mcMS**10)/(13365*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (82512145792*mcMS**10)/(4455*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (31867463104*mcMS**11)/(2835*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (254939704832*mcMS**11)/(8505*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (63734926208*mcMS**11)/(2835*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (6433846768*mcMS**12)/(567*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (51470774144*mcMS**12)/(1701*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (12867693536*mcMS**12)/(567*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (27020210848*mcMS**13)/(2835*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (216161686784*mcMS**13)/(8505*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (54040421696*mcMS**13)/(2835*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (896774288*mcMS**14)/(135*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (7174194304*mcMS**14)/(405*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (1793548576*mcMS**14)/(135*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (73703643776*mcMS**15)/(19305*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (589629150208*mcMS**15)/(57915*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (147407287552*mcMS**15)/(19305*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (34568184736*mcMS**16)/(19305*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (276545477888*mcMS**16)/(57915*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (69136369472*mcMS**16)/(19305*mbkin**16*(1 - mcMS/mbkin)**2) + 
            (7812195136*mcMS**17)/(11583*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (62497561088*mcMS**17)/(34749*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (15624390272*mcMS**17)/(11583*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (7339439504*mcMS**18)/(36855*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (58715516032*mcMS**18)/(110565*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (14678879008*mcMS**18)/(36855*mbkin**18*(1 - mcMS/mbkin)**2) + 
            (18003292256*mcMS**19)/(405405*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (144026338048*mcMS**19)/(1216215*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (36006584512*mcMS**19)/(405405*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (9695082136*mcMS**20)/(1378377*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (77560657088*mcMS**20)/(4135131*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (19390164272*mcMS**20)/(1378377*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (694454128*mcMS**21)/(984555*mbkin**23*(1 - mcMS/mbkin)**2) + 
            (5555633024*mcMS**21)/(2953665*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (1388908256*mcMS**21)/(984555*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (602792*mcMS**22)/(17901*mbkin**24*(1 - mcMS/mbkin)**2) - 
            (4822336*mcMS**22)/(53703*mbkin**23*(1 - mcMS/mbkin)**2) + 
            (1205584*mcMS**22)/(17901*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (70601360*mcMS)/(2380833*mbkin**3*(1 - mcMS/mbkin)) + 
            (564810880*mcMS)/(7142499*mbkin**2*(1 - mcMS/mbkin)) - 
            (732489110*mcMS)/(2380833*mbkin*(1 - mcMS/mbkin)) - 
            (2340128600*mcMS**2)/(7142499*mbkin**4*(1 - mcMS/mbkin)) - 
            (18721028800*mcMS**2)/(21427497*mbkin**3*(1 - mcMS/mbkin)) + 
            (24278834225*mcMS**2)/(7142499*mbkin**2*(1 - mcMS/mbkin)) + 
            (107374328*mcMS**3)/(53703*mbkin**5*(1 - mcMS/mbkin)) + 
            (858994624*mcMS**3)/(161109*mbkin**4*(1 - mcMS/mbkin)) - 
            (1114008653*mcMS**3)/(53703*mbkin**3*(1 - mcMS/mbkin)) - 
            (12111649028*mcMS**4)/(1378377*mbkin**6*(1 - mcMS/mbkin)) - 
            (96893192224*mcMS**4)/(4135131*mbkin**5*(1 - mcMS/mbkin)) + 
            (251316717331*mcMS**4)/(2756754*mbkin**4*(1 - mcMS/mbkin)) + 
            (804583036*mcMS**5)/(27027*mbkin**7*(1 - mcMS/mbkin)) + 
            (6436664288*mcMS**5)/(81081*mbkin**6*(1 - mcMS/mbkin)) - 
            (16695097997*mcMS**5)/(54054*mbkin**5*(1 - mcMS/mbkin)) - 
            (83851360*mcMS**6)/(1053*mbkin**8*(1 - mcMS/mbkin)) - 
            (670810880*mcMS**6)/(3159*mbkin**7*(1 - mcMS/mbkin)) + 
            (869957860*mcMS**6)/(1053*mbkin**6*(1 - mcMS/mbkin)) + 
            (60063520*mcMS**7)/(351*mbkin**9*(1 - mcMS/mbkin)) + 
            (480508160*mcMS**7)/(1053*mbkin**8*(1 - mcMS/mbkin)) - 
            (623159020*mcMS**7)/(351*mbkin**7*(1 - mcMS/mbkin)) - 
            (10420999024*mcMS**8)/(34749*mbkin**10*(1 - mcMS/mbkin)) - 
            (83367992192*mcMS**8)/(104247*mbkin**9*(1 - mcMS/mbkin)) + 
            (108117864874*mcMS**8)/(34749*mbkin**8*(1 - mcMS/mbkin)) + 
            (1158631120*mcMS**9)/(2673*mbkin**11*(1 - mcMS/mbkin)) + 
            (9269048960*mcMS**9)/(8019*mbkin**10*(1 - mcMS/mbkin)) - 
            (12020797870*mcMS**9)/(2673*mbkin**9*(1 - mcMS/mbkin)) - 
            (42146864*mcMS**10)/(81*mbkin**12*(1 - mcMS/mbkin)) - 
            (337174912*mcMS**10)/(243*mbkin**11*(1 - mcMS/mbkin)) + 
            (437273714*mcMS**10)/(81*mbkin**10*(1 - mcMS/mbkin)) + 
            (885305840*mcMS**11)/(1701*mbkin**13*(1 - mcMS/mbkin)) + 
            (7082446720*mcMS**11)/(5103*mbkin**12*(1 - mcMS/mbkin)) - 
            (9185048090*mcMS**11)/(1701*mbkin**11*(1 - mcMS/mbkin)) - 
            (737896760*mcMS**12)/(1701*mbkin**14*(1 - mcMS/mbkin)) - 
            (5903174080*mcMS**12)/(5103*mbkin**13*(1 - mcMS/mbkin)) + 
            (7655678885*mcMS**12)/(1701*mbkin**12*(1 - mcMS/mbkin)) + 
            (24329912*mcMS**13)/(81*mbkin**15*(1 - mcMS/mbkin)) + 
            (194639296*mcMS**13)/(243*mbkin**14*(1 - mcMS/mbkin)) - 
            (252422837*mcMS**13)/(81*mbkin**13*(1 - mcMS/mbkin)) - 
            (60252704*mcMS**14)/(351*mbkin**16*(1 - mcMS/mbkin)) - 
            (482021632*mcMS**14)/(1053*mbkin**15*(1 - mcMS/mbkin)) + 
            (625121804*mcMS**14)/(351*mbkin**14*(1 - mcMS/mbkin)) + 
            (6495875552*mcMS**15)/(81081*mbkin**17*(1 - mcMS/mbkin)) + 
            (51967004416*mcMS**15)/(243243*mbkin**16*(1 - mcMS/mbkin)) - 
            (67394708852*mcMS**15)/(81081*mbkin**15*(1 - mcMS/mbkin)) - 
            (270683440*mcMS**16)/(9009*mbkin**18*(1 - mcMS/mbkin)) - 
            (2165467520*mcMS**16)/(27027*mbkin**17*(1 - mcMS/mbkin)) + 
            (2808340690*mcMS**16)/(9009*mbkin**16*(1 - mcMS/mbkin)) + 
            (27918080*mcMS**17)/(3159*mbkin**19*(1 - mcMS/mbkin)) + 
            (223344640*mcMS**17)/(9477*mbkin**18*(1 - mcMS/mbkin)) - 
            (289650080*mcMS**17)/(3159*mbkin**17*(1 - mcMS/mbkin)) - 
            (738321352*mcMS**18)/(375921*mbkin**20*(1 - mcMS/mbkin)) - 
            (5906570816*mcMS**18)/(1127763*mbkin**19*(1 - mcMS/mbkin)) + 
            (7660084027*mcMS**18)/(375921*mbkin**18*(1 - mcMS/mbkin)) + 
            (427470856*mcMS**19)/(1378377*mbkin**21*(1 - mcMS/mbkin)) + 
            (3419766848*mcMS**19)/(4135131*mbkin**20*(1 - mcMS/mbkin)) - 
            (4435010131*mcMS**19)/(1378377*mbkin**19*(1 - mcMS/mbkin)) - 
            (348098812*mcMS**20)/(11223927*mbkin**22*(1 - mcMS/mbkin)) - 
            (2784790496*mcMS**20)/(33671781*mbkin**21*(1 - mcMS/mbkin)) + 
            (7223050349*mcMS**20)/(22447854*mbkin**20*(1 - mcMS/mbkin)) + 
            (1506980*mcMS**21)/(1020357*mbkin**23*(1 - mcMS/mbkin)) + 
            (12055840*mcMS**21)/(3061071*mbkin**22*(1 - mcMS/mbkin)) - 
            (31269835*mcMS**21)/(2040714*mbkin**21*(1 - mcMS/mbkin)) - 
            (688*mcMS)/(243*mbkin**4*(-1 + mcMS/mbkin)) + (688*mcMS**2)/
                (243*mbkin**5*(-1 + mcMS/mbkin)) + (1760*mcMS**3)/
                (9*mbkin**6*(-1 + mcMS/mbkin)) - (1760*mcMS**4)/
                (9*mbkin**7*(-1 + mcMS/mbkin)) + (1456*mcMS**5)/
                (9*mbkin**8*(-1 + mcMS/mbkin)) - (1456*mcMS**6)/
                (9*mbkin**9*(-1 + mcMS/mbkin)) - (3849688*np.pi**2)/405405 + 
            (1087904*mcMS*np.pi**2)/(12285*mbkin) - (1472024*mcMS**2*np.pi**2)/
                (3861*mbkin**2) + (1623712*mcMS**3*np.pi**2)/(1485*mbkin**3) - 
            (1236872*mcMS**4*np.pi**2)/(495*mbkin**4) + (1940288*mcMS**5*np.pi**2)/
                (405*mbkin**5) - (3040664*mcMS**6*np.pi**2)/(405*mbkin**6) + 
            (195968*mcMS**7*np.pi**2)/(21*mbkin**7) - (8599144*mcMS**8*np.pi**2)/
                (945*mbkin**8) + (562336*mcMS**9*np.pi**2)/(81*mbkin**9) - 
            (555224*mcMS**10*np.pi**2)/(135*mbkin**10) + (2755744*mcMS**11*np.pi**2)/
                (1485*mbkin**11) - (616*mcMS**12*np.pi**2)/mbkin**12 + 
            (182528*mcMS**13*np.pi**2)/(1287*mbkin**13) - (8201992*mcMS**14*np.pi**2)/
                (405405*mbkin**14) + (78016*mcMS**15*np.pi**2)/(57915*mbkin**15) + 
            ((748886504*mcMS)/(8729721*mbkin) - (7937476*mcMS**2)/
                (11583*mbkin**2) + (950785096*mcMS**3)/(328185*mbkin**3) - 
                (1198844138*mcMS**4)/(135135*mbkin**4) + (22613456*mcMS**5)/
                (1053*mbkin**5) - (14677064*mcMS**6)/(351*mbkin**6) + 
                (7772528*mcMS**7)/(117*mbkin**7) - (2746783208*mcMS**8)/
                (31185*mbkin**8) + (4479008*mcMS**9)/(45*mbkin**9) - 
                (872224*mcMS**10)/(9*mbkin**10) + (15355264*mcMS**11)/
                (189*mbkin**11) - (17248564*mcMS**12)/(297*mbkin**12) + 
                (51768368*mcMS**13)/(1485*mbkin**13) - (998628536*mcMS**14)/
                (57915*mbkin**14) + (185744752*mcMS**15)/(27027*mbkin**15) - 
                (83704*mcMS**16)/(39*mbkin**16) + (531800*mcMS**17)/
                (1053*mbkin**17) - (5859964*mcMS**18)/(69615*mbkin**18) + 
                (171064*mcMS**19)/(19305*mbkin**19) - (150698*mcMS**20)/
                (340119*mbkin**20) + (1312*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**3) + 
                (2912*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(9*mbkin**7) + 
                (99520*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) - 
                (188264*mcMS**2)/(351*mbkin**2*(1 - mcMS/mbkin)**2) + 
                (214748656*mcMS**3)/(29835*mbkin**3*(1 - mcMS/mbkin)**2) - 
                (117272664508*mcMS**4)/(2297295*mbkin**4*(1 - mcMS/mbkin)**2) + 
                (37671647480*mcMS**5)/(153153*mbkin**5*(1 - mcMS/mbkin)**2) - 
                (39524020924*mcMS**6)/(45045*mbkin**6*(1 - mcMS/mbkin)**2) + 
                (283309760*mcMS**7)/(117*mbkin**7*(1 - mcMS/mbkin)**2) - 
                (102678435568*mcMS**8)/(19305*mbkin**8*(1 - mcMS/mbkin)**2) + 
                (183023630624*mcMS**9)/(19305*mbkin**9*(1 - mcMS/mbkin)**2) - 
                (20628036448*mcMS**10)/(1485*mbkin**10*(1 - mcMS/mbkin)**2) + 
                (15933731552*mcMS**11)/(945*mbkin**11*(1 - mcMS/mbkin)**2) - 
                (3216923384*mcMS**12)/(189*mbkin**12*(1 - mcMS/mbkin)**2) + 
                (13510105424*mcMS**13)/(945*mbkin**13*(1 - mcMS/mbkin)**2) - 
                (448387144*mcMS**14)/(45*mbkin**14*(1 - mcMS/mbkin)**2) + 
                (36851821888*mcMS**15)/(6435*mbkin**15*(1 - mcMS/mbkin)**2) - 
                (17284092368*mcMS**16)/(6435*mbkin**16*(1 - mcMS/mbkin)**2) + 
                (3906097568*mcMS**17)/(3861*mbkin**17*(1 - mcMS/mbkin)**2) - 
                (3669719752*mcMS**18)/(12285*mbkin**18*(1 - mcMS/mbkin)**2) + 
                (9001646128*mcMS**19)/(135135*mbkin**19*(1 - mcMS/mbkin)**2) - 
                (4847541068*mcMS**20)/(459459*mbkin**20*(1 - mcMS/mbkin)**2) + 
                (347227064*mcMS**21)/(328185*mbkin**21*(1 - mcMS/mbkin)**2) - 
                (301396*mcMS**22)/(5967*mbkin**22*(1 - mcMS/mbkin)**2) + 
                (130612516*mcMS)/(793611*mbkin*(1 - mcMS/mbkin)) - 
                (4329237910*mcMS**2)/(2380833*mbkin**2*(1 - mcMS/mbkin)) + 
                (993212534*mcMS**3)/(89505*mbkin**3*(1 - mcMS/mbkin)) - 
                (112032753509*mcMS**4)/(2297295*mbkin**4*(1 - mcMS/mbkin)) + 
                (7442393083*mcMS**5)/(45045*mbkin**5*(1 - mcMS/mbkin)) - 
                (155125016*mcMS**6)/(351*mbkin**6*(1 - mcMS/mbkin)) + 
                (111117512*mcMS**7)/(117*mbkin**7*(1 - mcMS/mbkin)) - 
                (96394240972*mcMS**8)/(57915*mbkin**8*(1 - mcMS/mbkin)) + 
                (2143467572*mcMS**9)/(891*mbkin**9*(1 - mcMS/mbkin)) - 
                (389858492*mcMS**10)/(135*mbkin**10*(1 - mcMS/mbkin)) + 
                (1637815804*mcMS**11)/(567*mbkin**11*(1 - mcMS/mbkin)) - 
                (1365109006*mcMS**12)/(567*mbkin**12*(1 - mcMS/mbkin)) + 
                (225051686*mcMS**13)/(135*mbkin**13*(1 - mcMS/mbkin)) - 
                (557337512*mcMS**14)/(585*mbkin**14*(1 - mcMS/mbkin)) + 
                (60086848856*mcMS**15)/(135135*mbkin**15*(1 - mcMS/mbkin)) - 
                (500764364*mcMS**16)/(3003*mbkin**16*(1 - mcMS/mbkin)) + 
                (51648448*mcMS**17)/(1053*mbkin**17*(1 - mcMS/mbkin)) - 
                (6829472506*mcMS**18)/(626535*mbkin**18*(1 - mcMS/mbkin)) + 
                (3954105418*mcMS**19)/(2297295*mbkin**19*(1 - mcMS/mbkin)) - 
                (3219914011*mcMS**20)/(18706545*mbkin**20*(1 - mcMS/mbkin)) + 
                (2787913*mcMS**21)/(340119*mbkin**21*(1 - mcMS/mbkin)) + 
                (968*mcMS)/(243*mbkin**4*(-1 + mcMS/mbkin)) - (968*mcMS**2)/
                (243*mbkin**5*(-1 + mcMS/mbkin)) + (170800*mcMS**3)/
                (243*mbkin**6*(-1 + mcMS/mbkin)) - (170800*mcMS**4)/
                (243*mbkin**7*(-1 + mcMS/mbkin)) + (728*mcMS**5)/(mbkin**8*(-1 + 
                    mcMS/mbkin)) - (728*mcMS**6)/(mbkin**9*(-1 + mcMS/mbkin)))*
                np.log(mcMS**2/mbkin**2))*np.log(1 - mcMS/mbkin) + 
            (-35409044/1658475 + (4925297456*mcMS)/(54729675*mbkin) + 
            (757015612*mcMS**2)/(2606175*mbkin**2) - (516183824*mcMS**3)/
                (168399*mbkin**3) + (1305304468*mcMS**4)/(120285*mbkin**4) - 
            (1015014496*mcMS**5)/(42525*mbkin**5) + (2086845388*mcMS**6)/
                (54675*mbkin**6) - (287782336*mcMS**7)/(6075*mbkin**7) + 
            (18845788*mcMS**8)/(405*mbkin**8) - (43741648*mcMS**9)/
                (1215*mbkin**9) + (1654009132*mcMS**10)/(76545*mbkin**10) - 
            (179441648*mcMS**11)/(18225*mbkin**11) + (13865219444*mcMS**12)/
                (4209975*mbkin**12) - (1193047168*mcMS**13)/(1563705*mbkin**13) + 
            (44352092*mcMS**14)/(405405*mbkin**14) - (400450976*mcMS**15)/
                (54729675*mbkin**15))*np.log(1 - mcMS/mbkin)**2 + 
            (304/(81*mbkin**3) + (160*mcMS**2)/(9*mbkin**5) - (464*mcMS**4)/
                (27*mbkin**7) - (4400*mcMS**6)/(81*mbkin**9) + 
            (32*mcMS**8)/(3*mbkin**11) + (1688*np.sqrt(0j + mcMS**2/mbkin**2))/
                (243*mbkin**3) - (2824*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) - 
            (24976*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) - 
            (688*mcMS)/(243*mbkin**4*(1 + mcMS/mbkin)) - (688*mcMS**2)/
                (243*mbkin**5*(1 + mcMS/mbkin)) + (1760*mcMS**3)/
                (9*mbkin**6*(1 + mcMS/mbkin)) + (1760*mcMS**4)/
                (9*mbkin**7*(1 + mcMS/mbkin)) + (1456*mcMS**5)/
                (9*mbkin**8*(1 + mcMS/mbkin)) + (1456*mcMS**6)/
                (9*mbkin**9*(1 + mcMS/mbkin)))*np.log(1 + mcMS/mbkin) + 
            (7904/(81*mbkin**3) + (2968*mbkin)/(243*mcMS**4) - 
            89848/(3645*mbkin*mcMS**2) - (53168*mcMS**2)/(243*mbkin**5) + 
            (16616*mcMS**4)/(729*mbkin**7) + (59320*mcMS**6)/(81*mbkin**9) - 
            (707344*mcMS**8)/(1215*mbkin**11) - (1000*np.sqrt(0j + mcMS**2/mbkin**2))/
                (243*mbkin**3) - (1544*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) - 
            (22544*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3))*
            np.log(1 - mcMS**2/mbkin**2) + np.log(mcMS**2/mbkin**2)*(256/(81*mbkin**5) + 
            2048/(243*mbkin**4) + 2992/(243*mbkin**3) - 2/(9*mbkin**2) - 
            40/(81*mbkin) - (287769071069817419*mcMS)/(4064428199351520*
                mbkin) + (3944684*mcMS)/(405405*(mbkin - mcMS)) - 
            (58832*mcMS**2)/(243*mbkin**5) + (4352*mcMS**2)/(27*mbkin**4) + 
            (8704*mcMS**2)/(27*mbkin**3) + (732617467696549*mcMS**2)/
                (674109055620*mbkin**2) - (2078096*mcMS**2)/(27027*mbkin*
                (mbkin - mcMS)) - (950435130379316129*mcMS**3)/(1069586368250400*
                mbkin**3) + (3451012*mcMS**3)/(11583*mbkin**2*(mbkin - mcMS)) + 
            (176*mcMS**4)/(9*mbkin**8) - (71396*mcMS**4)/(729*mbkin**7) - 
            (12806*mcMS**4)/(9*mbkin**6) - (69256*mcMS**4)/(27*mbkin**5) + 
            (530.2136065099671*mcMS**4)/mbkin**4 - (3544208*mcMS**4)/
                (4455*mbkin**3*(mbkin - mcMS)) + (597399054412007*mcMS**5)/
                (111029726808*mbkin**5) + (5843564*mcMS**5)/(3465*mbkin**4*
                (mbkin - mcMS)) - (40504*mcMS**6)/(243*mbkin**9) + 
            (53632*mcMS**6)/(81*mbkin**7) - (21916742419862*mcMS**6)/
                (567431865*mbkin**6) - (1188272*mcMS**6)/(405*mbkin**5*
                (mbkin - mcMS)) + (266828288220607*mcMS**7)/(2723672952*
                mbkin**7) + (333868*mcMS**7)/(81*mbkin**6*(mbkin - mcMS)) - 
            (212416*mcMS**8)/(1215*mbkin**11) - (2810*mcMS**8)/(27*mbkin**10) - 
            (11240*mcMS**8)/(27*mbkin**9) - (124724067810451511*mcMS**8)/
                (604969665300*mbkin**8) - (289616*mcMS**8)/(63*mbkin**7*
                (mbkin - mcMS)) + (10861877424689*mcMS**9)/(32432400*mbkin**9) + 
            (764612*mcMS**9)/(189*mbkin**8*(mbkin - mcMS)) - 
            (469354130863495*mcMS**10)/(1099944846*mbkin**10) - 
            (1134544*mcMS**10)/(405*mbkin**9*(mbkin - mcMS)) + 
            (11550816291717145*mcMS**11)/(26398676304*mbkin**11) + 
            (204332*mcMS**11)/(135*mbkin**10*(mbkin - mcMS)) - 
            (88019407268539741*mcMS**12)/(241987866120*mbkin**12) - 
            (3903376*mcMS**12)/(6237*mbkin**11*(mbkin - mcMS)) + 
            (42363677789859031*mcMS**13)/(172848475800*mbkin**13) + 
            (170476*mcMS**13)/(891*mbkin**12*(mbkin - mcMS)) - 
            (112063923742729709*mcMS**14)/(842636319525*mbkin**14) - 
            (262384*mcMS**14)/(6435*mbkin**13*(mbkin - mcMS)) + 
            (60126655391804087*mcMS**15)/(1048614086520*mbkin**15) + 
            (2193556*mcMS**15)/(405405*mbkin**14*(mbkin - mcMS)) - 
            (287746271552629*mcMS**16)/(14980201236*mbkin**16) - 
            (19504*mcMS**16)/(57915*mbkin**15*(mbkin - mcMS)) + 
            (613010854803901*mcMS**17)/(127104737760*mbkin**17) - 
            (3461383196851759*mcMS**18)/(4051463516100*mbkin**18) + 
            (5992954493052799*mcMS**19)/(62916845191200*mbkin**19) - 
            (398150205649799*mcMS**20)/(79177172714640*mbkin**20) + 
            (3820703038853*mcMS**2)/(10213773570*mbkin**2*(1 - mcMS/mbkin)**2) - 
            (1324623229305471443*mcMS**3)/(267396592062600*mbkin**3*
                (1 - mcMS/mbkin)**2) + (5514257196652051*mcMS**4)/
                (154653899400*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (2445534345846504631*mcMS**5)/(14073504845400*mbkin**5*
                (1 - mcMS/mbkin)**2) + (73546477272645373*mcMS**6)/
                (118264746600*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (70971441129844033*mcMS**7)/(41392661310*mbkin**7*(1 - mcMS/mbkin)**
                2) + (5292307230847823*mcMS**8)/(1407913650*mbkin**8*
                (1 - mcMS/mbkin)**2) - (1883705490346937*mcMS**9)/
                (281582730*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (3996921465647711*mcMS**10)/(408211650*mbkin**10*(1 - mcMS/mbkin)**
                2) - (34379222836281097*mcMS**11)/(2894591700*mbkin**11*
                (1 - mcMS/mbkin)**2) + (991151981540419*mcMS**12)/
                (82702620*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (2648132431926677*mcMS**13)/(263144700*mbkin**13*(1 - mcMS/mbkin)**
                2) + (223279079747087713*mcMS**14)/(31840508700*mbkin**14*
                (1 - mcMS/mbkin)**2) - (4765673910687463*mcMS**15)/
                (1182647466*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (55872539918685143*mcMS**16)/(29566186650*mbkin**16*
                (1 - mcMS/mbkin)**2) - (251792791028627*mcMS**17)/
                (353783430*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (2416054156353307*mcMS**18)/(11497961475*mbkin**18*(1 - mcMS/mbkin)**
                2) - (1436625073319083*mcMS**19)/(30661230600*mbkin**19*
                (1 - mcMS/mbkin)**2) + (20887342577910533*mcMS**20)/
                (2814700969080*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (198978903088724189*mcMS**21)/(267396592062600*mbkin**21*
                (1 - mcMS/mbkin)**2) + (24672660896281*mcMS**22)/
                (694536602760*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (84769718158347181*mcMS)/(738986945336640*mbkin*
                (1 - mcMS/mbkin)) + (2446470123833473391*mcMS**2)/
                (2216960836009920*mbkin**2*(1 - mcMS/mbkin)) - 
            (43176344676356419391*mcMS**3)/(6417518209502400*mbkin**3*
                (1 - mcMS/mbkin)) + (4378795260555490441*mcMS**4)/
                (142611515766720*mbkin**4*(1 - mcMS/mbkin)) - 
            (1825778755943637871*mcMS**5)/(17159139597600*mbkin**5*
                (1 - mcMS/mbkin)) + (5409718168456526123*mcMS**6)/
                (18875053557360*mbkin**6*(1 - mcMS/mbkin)) - 
            (2398306644621841*mcMS**7)/(3890961360*mbkin**7*(1 - mcMS/mbkin)) + 
            (285450062438881289*mcMS**8)/(264356492400*mbkin**8*
                (1 - mcMS/mbkin)) - (82990069185646877*mcMS**9)/
                (53184146400*mbkin**9*(1 - mcMS/mbkin)) + 
            (63406249636229341*mcMS**10)/(33844456800*mbkin**10*
                (1 - mcMS/mbkin)) - (494853577058200963*mcMS**11)/
                (263986763040*mbkin**11*(1 - mcMS/mbkin)) + 
            (412608449195796241*mcMS**12)/(263986763040*mbkin**12*
                (1 - mcMS/mbkin)) - (6615633306531721*mcMS**13)/
                (6110804700*mbkin**13*(1 - mcMS/mbkin)) + 
            (1192039271175325699*mcMS**14)/(1926025873200*mbkin**14*
                (1 - mcMS/mbkin)) - (1604043546702600131*mcMS**15)/
                (5551486340400*mbkin**15*(1 - mcMS/mbkin)) + 
            (4133080133491195*mcMS**16)/(38131421328*mbkin**16*
                (1 - mcMS/mbkin)) - (267518569219007159*mcMS**17)/
                (8388912692160*mbkin**17*(1 - mcMS/mbkin)) + 
            (65643525989913343*mcMS**18)/(9260488036800*mbkin**18*
                (1 - mcMS/mbkin)) - (2394868598393790877*mcMS**19)/
                (2139172736500800*mbkin**19*(1 - mcMS/mbkin)) + 
            (13653883931434631303*mcMS**20)/(121932845980545600*mbkin**20*
                (1 - mcMS/mbkin)) - (844572094812157*mcMS**21)/(158354345429280*
                mbkin**21*(1 - mcMS/mbkin)) + (344*mcMS**2)/(243*mbkin**5*
                (-1 + mcMS/mbkin)) - (880*mcMS**4)/(9*mbkin**7*
                (-1 + mcMS/mbkin)) - (728*mcMS**6)/(9*mbkin**9*
                (-1 + mcMS/mbkin)) - (344*mcMS**2)/(243*mbkin**5*
                (1 + mcMS/mbkin)) + (880*mcMS**4)/(9*mbkin**7*(1 + mcMS/mbkin)) + 
            (728*mcMS**6)/(9*mbkin**9*(1 + mcMS/mbkin)) + 
            7280/(81*mbkin*(mbkin**2 - mcMS**2)) - (5888*mcMS**2)/
                (81*mbkin**3*(mbkin**2 - mcMS**2)) - (7744*mcMS**4)/
                (81*mbkin**5*(mbkin**2 - mcMS**2)) - (57776*mcMS**6)/
                (243*mbkin**7*(mbkin**2 - mcMS**2)) - (1696*mcMS**8)/
                (81*mbkin**9*(mbkin**2 - mcMS**2)) - (1408*mcMS**10)/
                (81*mbkin**11*(mbkin**2 - mcMS**2)) - (32*np.pi**2)/(27*mbkin**3) + 
            (88*mcMS**4*np.pi**2)/(3*mbkin**6) + (352*mcMS**4*np.pi**2)/(9*mbkin**5) - 
            (196*mcMS**4*np.pi**2)/(3*mbkin**4) + ((-1312*np.sqrt(0j + mcMS**2/mbkin**2))/
                (243*mbkin**3) - (2912*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/
                (9*mbkin**7) - (99520*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) + 
                (968*mcMS)/(243*mbkin**4*(1 + mcMS/mbkin)) + (968*mcMS**2)/
                (243*mbkin**5*(1 + mcMS/mbkin)) + (170800*mcMS**3)/
                (243*mbkin**6*(1 + mcMS/mbkin)) + (170800*mcMS**4)/
                (243*mbkin**7*(1 + mcMS/mbkin)) + (728*mcMS**5)/(mbkin**8*(1 + 
                    mcMS/mbkin)) + (728*mcMS**6)/(mbkin**9*(1 + mcMS/mbkin)))*
                np.log(1 + mcMS/mbkin) + ((176*mcMS**2)/(9*mbkin**5) + 
                (105376*mcMS**4)/(243*mbkin**7) + (3040*mcMS**6)/(27*mbkin**9) + 
                (5632*mcMS**8)/(81*mbkin**11))*np.log(1 - mcMS**2/mbkin**2)) + 
            (440/(9*mbkin**3) + 6/mbkin**2 + 40/(3*mbkin) - (640*mcMS**2)/
                (9*mbkin**5) + (160*mcMS**2)/mbkin**4 + (320*mcMS**2)/mbkin**3 - 
            (352*mcMS**4)/(3*mbkin**7) - (120*mcMS**4)/mbkin**6 - 
            (512*mcMS**4)/mbkin**5 + (2176*mcMS**6)/(9*mbkin**9) + 
            (1088*mcMS**6)/(3*mbkin**7) - (920*mcMS**8)/(9*mbkin**11) - 
            (46*mcMS**8)/mbkin**10 - (184*mcMS**8)/mbkin**9 + 
            (-32/(3*mbkin**3) + (264*mcMS**4)/mbkin**6 + (352*mcMS**4)/mbkin**5)*
                np.log(mcMS**2/mbkin**2))*np.log(2/mus) + ((-944*mcMS**2)/(81*mbkin**5) + 
            (12320*mcMS**4)/(81*mbkin**7) + (1696*mcMS**6)/(27*mbkin**9) - 
            (1792*mcMS**8)/(27*mbkin**11))*fp.polylog(2, 1 - mbkin**2/mcMS**2) + 
            ((-320*mcMS**2)/(9*mbkin**5) + (1856*mcMS**4)/(27*mbkin**7) + 
            (8800*mcMS**6)/(27*mbkin**9) - (256*mcMS**8)/(3*mbkin**11) - 
            (1688*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**3) + 
            (14120*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (24976*(mcMS**2/mbkin**2)**(3/2))/(81*mbkin**3))*
            fp.polylog(2, -(mcMS/mbkin)) + ((-320*mcMS**2)/(9*mbkin**5) + 
            (1856*mcMS**4)/(27*mbkin**7) + (8800*mcMS**6)/(27*mbkin**9) - 
            (256*mcMS**8)/(3*mbkin**11) - (104*np.sqrt(0j + mcMS**2/mbkin**2))/
                (81*mbkin**3) - (29560*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) - 
            (70064*(mcMS**2/mbkin**2)**(3/2))/(81*mbkin**3))*
            fp.polylog(2, mcMS/mbkin) + ((16*mcMS**2)/mbkin**5 - 
            (544*mcMS**4)/(9*mbkin**7) + (688*mcMS**6)/(27*mbkin**9) + 
            (2048*mcMS**8)/(27*mbkin**11) + (500*np.sqrt(0j + mcMS**2/mbkin**2))/
                (243*mbkin**3) + (3860*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (11272*(mcMS**2/mbkin**2)**(3/2))/(81*mbkin**3))*
            fp.polylog(2, mcMS**2/mbkin**2)) + np.log(mu0**2/mus**2)*
            (-512/(27*mbkin**5) - 4096/(81*mbkin**4) - 13016/(243*mbkin**3) + 
            4/(3*mbkin**2) + 80/(27*mbkin) + 1484/(243*mbkin*mcMS**2) + 
            (2291073463739113*mcMS)/(15395561361180*mbkin**3) + 
            (2291073463739113*mcMS)/(6928002612531*mbkin**2) + 
            (10072.277369115025*mcMS)/mbkin - (256*mcMS**2)/(9*mbkin**6) - 
            (1502296*mcMS**2)/(3645*mbkin**5) - (11499343944346*mcMS**2)/
            (5106886785*mbkin**4) - (6204299942968*mcMS**2)/
            (1313199459*mbkin**3) - (127167.47832114284*mcMS**2)/mbkin**2 + 
            (40056351585142769*mcMS**3)/(4951788741900*mbkin**5) + 
            (366247127651921401*mcMS**3)/(20054744404695*mbkin**4) + 
            (751346.8145151937*mcMS**3)/mbkin**3 - (1024*mcMS**4)/(27*mbkin**9) - 
            (2144*mcMS**4)/(81*mbkin**8) + (4352842*mcMS**4)/(3645*mbkin**7) - 
            (15380074959618307*mcMS**4)/(436922536050*mbkin**6) - 
            (92618668773468466*mcMS**4)/(1179690847335*mbkin**5) - 
            (2.758209865486425e6*mcMS**4)/mbkin**4 + (46878631780681979*mcMS**5)/
            (393230282445*mbkin**7) + (958910347616350684*mcMS**5)/
            (3539072542005*mbkin**6) + (6.976443096232966e6*mcMS**5)/mbkin**5 + 
            (4096*mcMS**6)/(27*mbkin**11) + (29312*mcMS**6)/(81*mbkin**10) - 
            (1243228*mcMS**6)/(1215*mbkin**9) - (6146092756124*mcMS**6)/
            (20027007*mbkin**8) - (132894891494096*mcMS**6)/(189143955*mbkin**7) - 
            (1.2878774039595978e7*mcMS**6)/mbkin**6 + (1071181548897013*mcMS**7)/
            (1702295595*mbkin**9) + (7358641127268428*mcMS**7)/
            (5106886785*mbkin**8) + (1.7875794658439644e7*mcMS**7)/mbkin**7 - 
            (2560*mcMS**8)/(27*mbkin**13) - (20768*mcMS**8)/(81*mbkin**12) + 
            (759346*mcMS**8)/(1215*mbkin**11) - (836597821508002*mcMS**8)/
            (800224425*mbkin**10) - (1638553771861816*mcMS**8)/
            (682296615*mbkin**9) - (1.8901811575516436e7*mcMS**8)/mbkin**8 + 
            (207257117584603*mcMS**9)/(145495350*mbkin**11) + 
            (1290247316601442*mcMS**9)/(392837445*mbkin**10) + 
            (1.5239174522997856e7*mcMS**9)/mbkin**9 - (293412748813904*mcMS**10)/
            (183324141*mbkin**12) - (30540463790306048*mcMS**10)/
            (8249586345*mbkin**11) - (9.226033390994718e6*mcMS**10)/mbkin**10 + 
            (8170093380899239*mcMS**11)/(5499724230*mbkin**13) + 
            (85299476769732166*mcMS**11)/(24748759035*mbkin**12) + 
            (4.00258832555982e6*mcMS**11)/mbkin**11 - (5573248882267*mcMS**12)/
            (4901715*mbkin**14) - (21825569111970868*mcMS**12)/
            (8249586345*mbkin**13) - (1.0657904615957544e6*mcMS**12)/mbkin**12 + 
            (66755282008807*mcMS**13)/(93532725*mbkin**15) + 
            (280378536225116*mcMS**13)/(168358905*mbkin**14) + 
            (22555.502580515895*mcMS**13)/mbkin**13 - (14599591703211244*mcMS**14)/
            (40125539025*mbkin**16) - (12296860553581616*mcMS**14)/
            (14445194049*mbkin**15) + (136753.9567265611*mcMS**14)/mbkin**14 + 
            (19442688546859973*mcMS**15)/(131076760815*mbkin**17) + 
            (136819147067530892*mcMS**15)/(393230282445*mbkin**16) - 
            (78630.19954340154*mcMS**15)/mbkin**15 - (884283384020506*mcMS**16)/
            (18725251545*mbkin**18) - (2079388559452168*mcMS**16)/
            (18725251545*mbkin**17) + (3597305435872*mcMS**16)/
            (130945815*mbkin**16) + (28388224254937*mcMS**17)/
            (2508646140*mbkin**19) + (953523263154797*mcMS**17)/
            (35748207495*mbkin**18) - (899387281568*mcMS**17)/
            (130945815*mbkin**17) - (1944786029552566*mcMS**18)/
            (1012865879025*mbkin**20) - (2756745950831624*mcMS**18)/
            (607719527415*mbkin**19) + (158724542432*mcMS**18)/
            (130945815*mbkin**18) + (31719811412273*mcMS**19)/
            (154207953900*mbkin**21) + (114909187220173*mcMS**19)/
            (235938169467*mbkin**20) - (2519562016*mcMS**19)/
            (18706545*mbkin**19) - (34520063901721*mcMS**20)/
            (3299048863110*mbkin**22) - (73717644231026*mcMS**20)/
            (2969143976799*mbkin**21) + (2411168*mcMS**20)/(340119*mbkin**20) + 
            (3820703038853*mcMS**2)/(15320660355*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (30565624310824*mcMS**2)/(45961981065*mbkin**3*(1 - mcMS/mbkin)**2) - 
            (7641406077706*mcMS**2)/(15320660355*mbkin**2*(1 - mcMS/mbkin)**2) - 
            (1324623229305471443*mcMS**3)/(401094888093900*mbkin**5*
            (1 - mcMS/mbkin)**2) - (2649246458610942886*mcMS**3)/
            (300821166070425*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (1324623229305471443*mcMS**3)/(200547444046950*mbkin**3*
            (1 - mcMS/mbkin)**2) + (5514257196652051*mcMS**4)/
            (231980849100*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (11028514393304102*mcMS**4)/(173985636825*mbkin**5*(1 - mcMS/mbkin)**
                2) - (5514257196652051*mcMS**4)/(115990424550*mbkin**4*
            (1 - mcMS/mbkin)**2) - (2445534345846504631*mcMS**5)/
            (21110257268100*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (4891068691693009262*mcMS**5)/(15832692951075*mbkin**6*
            (1 - mcMS/mbkin)**2) + (2445534345846504631*mcMS**5)/
            (10555128634050*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (73546477272645373*mcMS**6)/(177397119900*mbkin**8*(1 - mcMS/mbkin)**
                2) + (147092954545290746*mcMS**6)/(133047839925*mbkin**7*
            (1 - mcMS/mbkin)**2) - (73546477272645373*mcMS**6)/
            (88698559950*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (70971441129844033*mcMS**7)/(62088991965*mbkin**9*(1 - mcMS/mbkin)**
                2) - (567771529038752264*mcMS**7)/(186266975895*mbkin**8*
            (1 - mcMS/mbkin)**2) + (141942882259688066*mcMS**7)/
            (62088991965*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (5292307230847823*mcMS**8)/(2111870475*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (42338457846782584*mcMS**8)/(6335611425*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (10584614461695646*mcMS**8)/(2111870475*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (1883705490346937*mcMS**9)/(422374095*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (15069643922775496*mcMS**9)/(1267122285*mbkin**10*(1 - mcMS/mbkin)**
                2) + (3767410980693874*mcMS**9)/(422374095*mbkin**9*
            (1 - mcMS/mbkin)**2) + (3996921465647711*mcMS**10)/
            (612317475*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (31975371725181688*mcMS**10)/(1836952425*mbkin**11*(1 - mcMS/mbkin)**
                2) - (7993842931295422*mcMS**10)/(612317475*mbkin**10*
            (1 - mcMS/mbkin)**2) - (34379222836281097*mcMS**11)/
            (4341887550*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (137516891345124388*mcMS**11)/(6512831325*mbkin**12*
            (1 - mcMS/mbkin)**2) + (34379222836281097*mcMS**11)/
            (2170943775*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (991151981540419*mcMS**12)/(124053930*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (3964607926161676*mcMS**12)/(186080895*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (991151981540419*mcMS**12)/(62026965*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (2648132431926677*mcMS**13)/(394717050*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (10592529727706708*mcMS**13)/(592075575*mbkin**14*(1 - mcMS/mbkin)**
                2) + (2648132431926677*mcMS**13)/(197358525*mbkin**13*
            (1 - mcMS/mbkin)**2) + (223279079747087713*mcMS**14)/
            (47760763050*mbkin**16*(1 - mcMS/mbkin)**2) + 
            (893116318988350852*mcMS**14)/(71641144575*mbkin**15*
            (1 - mcMS/mbkin)**2) - (223279079747087713*mcMS**14)/
            (23880381525*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (4765673910687463*mcMS**15)/(1773971199*mbkin**17*(1 - mcMS/mbkin)**
                2) - (38125391285499704*mcMS**15)/(5321913597*mbkin**16*
            (1 - mcMS/mbkin)**2) + (9531347821374926*mcMS**15)/
            (1773971199*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (55872539918685143*mcMS**16)/(44349279975*mbkin**18*
            (1 - mcMS/mbkin)**2) + (446980319349481144*mcMS**16)/
            (133047839925*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (111745079837370286*mcMS**16)/(44349279975*mbkin**16*
            (1 - mcMS/mbkin)**2) - (251792791028627*mcMS**17)/
            (530675145*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (2014342328229016*mcMS**17)/(1592025435*mbkin**18*(1 - mcMS/mbkin)**
                2) + (503585582057254*mcMS**17)/(530675145*mbkin**17*
            (1 - mcMS/mbkin)**2) + (4832108312706614*mcMS**18)/
            (34493884425*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (38656866501652912*mcMS**18)/(103481653275*mbkin**19*
            (1 - mcMS/mbkin)**2) - (9664216625413228*mcMS**18)/
            (34493884425*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (1436625073319083*mcMS**19)/(45991845900*mbkin**21*(1 - mcMS/mbkin)**
                2) - (2873250146638166*mcMS**19)/(34493884425*mbkin**20*
            (1 - mcMS/mbkin)**2) + (1436625073319083*mcMS**19)/
            (22995922950*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (20887342577910533*mcMS**20)/(4222051453620*mbkin**22*
            (1 - mcMS/mbkin)**2) + (41774685155821066*mcMS**20)/
            (3166538590215*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (20887342577910533*mcMS**20)/(2111025726810*mbkin**20*
            (1 - mcMS/mbkin)**2) - (198978903088724189*mcMS**21)/
            (401094888093900*mbkin**23*(1 - mcMS/mbkin)**2) - 
            (397957806177448378*mcMS**21)/(300821166070425*mbkin**22*
            (1 - mcMS/mbkin)**2) + (198978903088724189*mcMS**21)/
            (200547444046950*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (24672660896281*mcMS**22)/(1041804904140*mbkin**24*(1 - mcMS/mbkin)**
                2) + (49345321792562*mcMS**22)/(781353678105*mbkin**23*
            (1 - mcMS/mbkin)**2) - (24672660896281*mcMS**22)/
            (520902452070*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (2291073463739113*mcMS)/(92373368167080*mbkin**3*(1 - mcMS/mbkin)) - 
            (2291073463739113*mcMS)/(34640013062655*mbkin**2*(1 - mcMS/mbkin)) + 
            (98516158940781859*mcMS)/(886784334403968*mbkin*(1 - mcMS/mbkin)) + 
            (66120814157661443*mcMS**2)/(277120104501240*mbkin**4*
            (1 - mcMS/mbkin)) + (66120814157661443*mcMS**2)/
            (103920039187965*mbkin**3*(1 - mcMS/mbkin)) - 
            (2843195008779442049*mcMS**2)/(2660353003211904*mbkin**2*
            (1 - mcMS/mbkin)) - (1166928234496119443*mcMS**3)/
            (802189776187800*mbkin**5*(1 - mcMS/mbkin)) - 
            (1166928234496119443*mcMS**3)/(300821166070425*mbkin**4*
            (1 - mcMS/mbkin)) + (50177914083333136049*mcMS**3)/
            (7701021851402880*mbkin**3*(1 - mcMS/mbkin)) + 
            (118345817852851093*mcMS**4)/(17826439470840*mbkin**6*
            (1 - mcMS/mbkin)) + (118345817852851093*mcMS**4)/
            (6684914801565*mbkin**5*(1 - mcMS/mbkin)) - 
            (5088870167672596999*mcMS**4)/(171133818920064*mbkin**4*
            (1 - mcMS/mbkin)) - (49345371782260483*mcMS**5)/
            (2144892449700*mbkin**7*(1 - mcMS/mbkin)) - 
            (98690743564520966*mcMS**5)/(1608669337275*mbkin**6*
            (1 - mcMS/mbkin)) + (2121850986637200769*mcMS**5)/
            (20590967517120*mbkin**5*(1 - mcMS/mbkin)) + 
            (146208599147473679*mcMS**6)/(2359381694670*mbkin**8*
            (1 - mcMS/mbkin)) + (584834396589894716*mcMS**6)/
            (3539072542005*mbkin**7*(1 - mcMS/mbkin)) - 
            (6286969763341368197*mcMS**6)/(22650064268832*mbkin**6*
            (1 - mcMS/mbkin)) - (64819098503293*mcMS**7)/(486370170*mbkin**9*
            (1 - mcMS/mbkin)) - (259276394013172*mcMS**7)/(729555255*mbkin**8*
            (1 - mcMS/mbkin)) + (2787221235641599*mcMS**7)/(4669153632*mbkin**7*
            (1 - mcMS/mbkin)) + (7714866552402197*mcMS**8)/
            (33044561550*mbkin**10*(1 - mcMS/mbkin)) + 
            (30859466209608788*mcMS**8)/(49566842325*mbkin**9*(1 - mcMS/mbkin)) - 
            (331739261753294471*mcMS**8)/(317227790880*mbkin**8*
            (1 - mcMS/mbkin)) - (2242974842855321*mcMS**9)/(6648018300*mbkin**11*
            (1 - mcMS/mbkin)) - (4485949685710642*mcMS**9)/(4986013725*mbkin**10*
            (1 - mcMS/mbkin)) + (96447918242778803*mcMS**9)/
            (63820975680*mbkin**9*(1 - mcMS/mbkin)) + (1713682422600793*mcMS**10)/
            (4230557100*mbkin**12*(1 - mcMS/mbkin)) + (3427364845201586*mcMS**10)/
            (3172917825*mbkin**11*(1 - mcMS/mbkin)) - 
            (73688344171834099*mcMS**10)/(40613348160*mbkin**10*
            (1 - mcMS/mbkin)) - (13374421001572999*mcMS**11)/
            (32998345380*mbkin**13*(1 - mcMS/mbkin)) - 
            (26748842003145998*mcMS**11)/(24748759035*mbkin**12*
            (1 - mcMS/mbkin)) + (575100103067638957*mcMS**11)/
            (316784115648*mbkin**11*(1 - mcMS/mbkin)) + 
            (11151579707994493*mcMS**12)/(32998345380*mbkin**14*
            (1 - mcMS/mbkin)) + (22303159415988986*mcMS**12)/
            (24748759035*mbkin**13*(1 - mcMS/mbkin)) - 
            (479517927443763199*mcMS**12)/(316784115648*mbkin**12*
            (1 - mcMS/mbkin)) - (357601800353066*mcMS**13)/(1527701175*mbkin**15*
            (1 - mcMS/mbkin)) - (2860814402824528*mcMS**13)/
            (4583103525*mbkin**14*(1 - mcMS/mbkin)) + (7688438707590919*mcMS**13)/
            (7332965640*mbkin**13*(1 - mcMS/mbkin)) + 
            (32217277599333127*mcMS**14)/(240753234150*mbkin**16*
            (1 - mcMS/mbkin)) + (128869110397332508*mcMS**14)/
            (361129851225*mbkin**15*(1 - mcMS/mbkin)) - 
            (1385342936771324461*mcMS**14)/(2311231047840*mbkin**14*
            (1 - mcMS/mbkin)) - (43352528289259463*mcMS**15)/
            (693935792550*mbkin**17*(1 - mcMS/mbkin)) - 
            (173410113157037852*mcMS**15)/(1040903688825*mbkin**16*
            (1 - mcMS/mbkin)) + (1864158716438156909*mcMS**15)/
            (6661783608480*mbkin**15*(1 - mcMS/mbkin)) + 
            (111704868472735*mcMS**16)/(4766427666*mbkin**18*(1 - mcMS/mbkin)) + 
            (446819473890940*mcMS**16)/(7149641499*mbkin**17*(1 - mcMS/mbkin)) - 
            (24016546721638025*mcMS**16)/(228788527968*mbkin**16*
            (1 - mcMS/mbkin)) - (7230231600513707*mcMS**17)/
            (1048614086520*mbkin**19*(1 - mcMS/mbkin)) - 
            (7230231600513707*mcMS**17)/(393230282445*mbkin**18*
            (1 - mcMS/mbkin)) + (310899958822089401*mcMS**17)/
            (10066695230592*mbkin**17*(1 - mcMS/mbkin)) + 
            (1774149351078739*mcMS**18)/(1157561004600*mbkin**20*
            (1 - mcMS/mbkin)) + (1774149351078739*mcMS**18)/
            (434085376725*mbkin**19*(1 - mcMS/mbkin)) - 
            (76288422096385777*mcMS**18)/(11112585644160*mbkin**18*
            (1 - mcMS/mbkin)) - (64726178334967321*mcMS**19)/
            (267396592062600*mbkin**21*(1 - mcMS/mbkin)) - 
            (64726178334967321*mcMS**19)/(100273722023475*mbkin**20*
            (1 - mcMS/mbkin)) + (2783225668403594803*mcMS**19)/
            (2567007283800960*mbkin**19*(1 - mcMS/mbkin)) + 
            (369023890038773819*mcMS**20)/(15241605747568200*mbkin**22*
            (1 - mcMS/mbkin)) + (369023890038773819*mcMS**20)/
            (5715602155338075*mbkin**21*(1 - mcMS/mbkin)) - 
            (15868027271667274217*mcMS**20)/(146319415176654720*mbkin**20*
            (1 - mcMS/mbkin)) - (22826272832761*mcMS**21)/(19794293178660*
            mbkin**23*(1 - mcMS/mbkin)) - (45652545665522*mcMS**21)/
            (14845719883995*mbkin**22*(1 - mcMS/mbkin)) + 
            (981529731808723*mcMS**21)/(190025214515136*mbkin**21*
            (1 - mcMS/mbkin)) - 89848/(3645*mbkin*(mbkin**2 - mcMS**2)) + 
            (1484*mbkin)/(243*mcMS**2*(mbkin**2 - mcMS**2)) - 
            (55076*mcMS**2)/(243*mbkin**3*(mbkin**2 - mcMS**2)) + 
            (54032*mcMS**4)/(243*mbkin**5*(mbkin**2 - mcMS**2)) + 
            (81740*mcMS**6)/(729*mbkin**7*(mbkin**2 - mcMS**2)) - 
            (56968*mcMS**8)/(243*mbkin**9*(mbkin**2 - mcMS**2)) + 
            (176356*mcMS**10)/(1215*mbkin**11*(mbkin**2 - mcMS**2)) + 
            (64*np.pi**2)/(9*mbkin**3) - (231412726775303*mcMS*np.pi**2)/
            (233746793280*mbkin) - (248*mcMS**2*np.pi**2)/(27*mbkin**5) + 
            (64*mcMS**2*np.pi**2)/(3*mbkin**4) + (128*mcMS**2*np.pi**2)/(3*mbkin**3) + 
            (19885747105513*mcMS**2*np.pi**2)/(1565268705*mbkin**2) - 
            (972602406380027*mcMS**3*np.pi**2)/(12843230400*mbkin**3) - 
            (800*mcMS**4*np.pi**2)/(81*mbkin**7) - (16*mcMS**4*np.pi**2)/mbkin**6 - 
            (64*mcMS**4*np.pi**2)/mbkin**5 + (4690681865160961*mcMS**4*np.pi**2)/
            (16856739900*mbkin**4) - (38019171359233*mcMS**5*np.pi**2)/
            (53887680*mbkin**5) - (80*mcMS**6*np.pi**2)/(27*mbkin**9) + 
            (128*mcMS**6*np.pi**2)/(3*mbkin**7) + (433155592738*mcMS**6*np.pi**2)/
            (331695*mbkin**6) - (84997235817281*mcMS**7*np.pi**2)/
            (46702656*mbkin**7) - (1408*mcMS**8*np.pi**2)/(81*mbkin**11) - 
            (16*mcMS**8*np.pi**2)/(3*mbkin**10) - (64*mcMS**8*np.pi**2)/(3*mbkin**9) + 
            (165181707810001*mcMS**8*np.pi**2)/(85135050*mbkin**8) - 
            (206230056355249*mcMS**9*np.pi**2)/(129729600*mbkin**9) + 
            (519101215868*mcMS**10*np.pi**2)/(521235*mbkin**10) - 
            (460448316031807*mcMS**11*np.pi**2)/(980755776*mbkin**11) + 
            (25919209656889*mcMS**12*np.pi**2)/(160540380*mbkin**12) - 
            (1474778611823953*mcMS**13*np.pi**2)/(38529691200*mbkin**13) + 
            (1752616483195*mcMS**14*np.pi**2)/(313053741*mbkin**14) - 
            (12724616134177*mcMS**15*np.pi**2)/(33392399040*mbkin**15) - 
            (172*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(243*mbkin**3) + 
            (1820*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2)*np.pi**2)/(9*mbkin**7) + 
            (440*(mcMS**2/mbkin**2)**(3/2)*np.pi**2)/(3*mbkin**3) + 
            ((-616*mcMS**2)/(81*mbkin**5) - (196*mcMS**2)/(3*mbkin**2) + 
            (4112*mcMS**4)/(243*mbkin**7) + (736*mcMS**4)/(3*mbkin**6) + 
            (2944*mcMS**4)/(9*mbkin**5) + (1216*mcMS**4)/mbkin**4 - 
            (2032*mcMS**6)/(27*mbkin**9) + (1164*mcMS**6)/mbkin**6 - 
            (4288*mcMS**8)/(81*mbkin**11) - (968*mcMS**8)/(3*mbkin**8))*
            np.log(mcMS**2/mbkin**2)**2 - (584*mcMS**4*np.log(mcMS**2/mbkin**2)**3)/
            mbkin**4 + (304/(81*mbkin**3) - (56481088*mcMS)/(264537*mbkin**3) - 
            (1129621760*mcMS)/(2380833*mbkin**2) - (333324989499253*mcMS)/
                (547844046750*mbkin) + (160*mcMS**2)/(9*mbkin**5) + 
            (753056*mcMS**2)/(351*mbkin**4) + (15061120*mcMS**2)/(3159*mbkin**3) + 
            (15478417721719*mcMS**2)/(3557428875*mbkin**2) - (13536576*mcMS**3)/
                (1105*mbkin**5) - (6016256*mcMS**3)/(221*mbkin**4) - 
            (683837705723971*mcMS**3)/(42141849750*mbkin**3) - 
            (464*mcMS**4)/(27*mbkin**7) + (2297985232*mcMS**4)/(45045*mbkin**6) + 
            (9191940928*mcMS**4)/(81081*mbkin**5) + (561305364499738*mcMS**4)/
                (12642554925*mbkin**4) - (19109120*mcMS**5)/(117*mbkin**7) - 
            (382182400*mcMS**5)/(1053*mbkin**6) - (70552906019*mcMS**5)/
                (727650*mbkin**5) - (4400*mcMS**6)/(81*mbkin**9) + 
            (15990656*mcMS**6)/(39*mbkin**8) + (319813120*mcMS**6)/
                (351*mbkin**7) + (45457380358813*mcMS**6)/(273648375*mbkin**6) - 
            (10686720*mcMS**7)/(13*mbkin**9) - (71244800*mcMS**7)/(39*mbkin**8) - 
            (119765484898127*mcMS**7)/(547296750*mbkin**7) + 
            (32*mcMS**8)/(3*mbkin**11) + (1985708992*mcMS**8)/(1485*mbkin**10) + 
            (7942835968*mcMS**8)/(2673*mbkin**9) + (47118708753404*mcMS**8)/
                (212837625*mbkin**8) - (26755712*mcMS**9)/(15*mbkin**11) - 
            (107022848*mcMS**9)/(27*mbkin**10) - (6305059020913*mcMS**9)/
                (36486450*mbkin**9) + (17664064*mcMS**10)/(9*mbkin**12) + 
            (353281280*mcMS**10)/(81*mbkin**11) + (7955554444931*mcMS**10)/
                (76621545*mbkin**10) - (337299328*mcMS**11)/(189*mbkin**13) - 
            (6745986560*mcMS**11)/(1701*mbkin**12) - (181064392519811*mcMS**11)/
                (3831077250*mbkin**11) + (1338720*mcMS**12)/mbkin**14 + 
            (8924800*mcMS**12)/(3*mbkin**13) + (332372802464726*mcMS**12)/
                (21070924875*mbkin**12) - (37077248*mcMS**13)/(45*mbkin**15) - 
            (148308992*mcMS**13)/(81*mbkin**14) - (5983215661469*mcMS**13)/
                (1641890250*mbkin**13) + (241028224*mcMS**14)/(585*mbkin**16) + 
            (964112896*mcMS**14)/(1053*mbkin**15) + (194192359957*mcMS**14)/
                (372683025*mbkin**14) - (164985088*mcMS**15)/(1001*mbkin**17) - 
            (3299701760*mcMS**15)/(9009*mbkin**16) - (3809167143809*mcMS**15)/
                (109568809350*mbkin**15) + (669632*mcMS**16)/(13*mbkin**18) + 
            (13392640*mcMS**16)/(117*mbkin**17) - (4254400*mcMS**17)/
                (351*mbkin**19) - (85088000*mcMS**17)/(3159*mbkin**18) + 
            (46879712*mcMS**18)/(23205*mbkin**20) + (187518848*mcMS**18)/
                (41769*mbkin**19) - (1368512*mcMS**19)/(6435*mbkin**21) - 
            (5474048*mcMS**19)/(11583*mbkin**20) + (1205584*mcMS**20)/
                (113373*mbkin**22) + (24111680*mcMS**20)/(1020357*mbkin**21) + 
            (104*np.sqrt(0j + mcMS**2/mbkin**2))/(81*mbkin**3) + 
            (5912*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (70064*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) - 
            (376528*mcMS**2)/(1053*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (3012224*mcMS**2)/(3159*mbkin**3*(1 - mcMS/mbkin)**2) + 
            (753056*mcMS**2)/(1053*mbkin**2*(1 - mcMS/mbkin)**2) + 
            (429497312*mcMS**3)/(89505*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (3435978496*mcMS**3)/(268515*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (858994624*mcMS**3)/(89505*mbkin**3*(1 - mcMS/mbkin)**2) - 
            (234545329016*mcMS**4)/(6891885*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (1876362632128*mcMS**4)/(20675655*mbkin**5*(1 - mcMS/mbkin)**2) + 
            (469090658032*mcMS**4)/(6891885*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (75343294960*mcMS**5)/(459459*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (602746359680*mcMS**5)/(1378377*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (150686589920*mcMS**5)/(459459*mbkin**5*(1 - mcMS/mbkin)**2) - 
            (79048041848*mcMS**6)/(135135*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (632384334784*mcMS**6)/(405405*mbkin**7*(1 - mcMS/mbkin)**2) + 
            (158096083696*mcMS**6)/(135135*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (566619520*mcMS**7)/(351*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (4532956160*mcMS**7)/(1053*mbkin**8*(1 - mcMS/mbkin)**2) - 
            (1133239040*mcMS**7)/(351*mbkin**7*(1 - mcMS/mbkin)**2) - 
            (205356871136*mcMS**8)/(57915*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (1642854969088*mcMS**8)/(173745*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (410713742272*mcMS**8)/(57915*mbkin**8*(1 - mcMS/mbkin)**2) + 
            (366047261248*mcMS**9)/(57915*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (2928378089984*mcMS**9)/(173745*mbkin**10*(1 - mcMS/mbkin)**2) - 
            (732094522496*mcMS**9)/(57915*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (41256072896*mcMS**10)/(4455*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (330048583168*mcMS**10)/(13365*mbkin**11*(1 - mcMS/mbkin)**2) + 
            (82512145792*mcMS**10)/(4455*mbkin**10*(1 - mcMS/mbkin)**2) + 
            (31867463104*mcMS**11)/(2835*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (254939704832*mcMS**11)/(8505*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (63734926208*mcMS**11)/(2835*mbkin**11*(1 - mcMS/mbkin)**2) - 
            (6433846768*mcMS**12)/(567*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (51470774144*mcMS**12)/(1701*mbkin**13*(1 - mcMS/mbkin)**2) + 
            (12867693536*mcMS**12)/(567*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (27020210848*mcMS**13)/(2835*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (216161686784*mcMS**13)/(8505*mbkin**14*(1 - mcMS/mbkin)**2) - 
            (54040421696*mcMS**13)/(2835*mbkin**13*(1 - mcMS/mbkin)**2) - 
            (896774288*mcMS**14)/(135*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (7174194304*mcMS**14)/(405*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (1793548576*mcMS**14)/(135*mbkin**14*(1 - mcMS/mbkin)**2) + 
            (73703643776*mcMS**15)/(19305*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (589629150208*mcMS**15)/(57915*mbkin**16*(1 - mcMS/mbkin)**2) - 
            (147407287552*mcMS**15)/(19305*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (34568184736*mcMS**16)/(19305*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (276545477888*mcMS**16)/(57915*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (69136369472*mcMS**16)/(19305*mbkin**16*(1 - mcMS/mbkin)**2) + 
            (7812195136*mcMS**17)/(11583*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (62497561088*mcMS**17)/(34749*mbkin**18*(1 - mcMS/mbkin)**2) - 
            (15624390272*mcMS**17)/(11583*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (7339439504*mcMS**18)/(36855*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (58715516032*mcMS**18)/(110565*mbkin**19*(1 - mcMS/mbkin)**2) + 
            (14678879008*mcMS**18)/(36855*mbkin**18*(1 - mcMS/mbkin)**2) + 
            (18003292256*mcMS**19)/(405405*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (144026338048*mcMS**19)/(1216215*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (36006584512*mcMS**19)/(405405*mbkin**19*(1 - mcMS/mbkin)**2) - 
            (9695082136*mcMS**20)/(1378377*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (77560657088*mcMS**20)/(4135131*mbkin**21*(1 - mcMS/mbkin)**2) + 
            (19390164272*mcMS**20)/(1378377*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (694454128*mcMS**21)/(984555*mbkin**23*(1 - mcMS/mbkin)**2) + 
            (5555633024*mcMS**21)/(2953665*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (1388908256*mcMS**21)/(984555*mbkin**21*(1 - mcMS/mbkin)**2) - 
            (602792*mcMS**22)/(17901*mbkin**24*(1 - mcMS/mbkin)**2) - 
            (4822336*mcMS**22)/(53703*mbkin**23*(1 - mcMS/mbkin)**2) + 
            (1205584*mcMS**22)/(17901*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (28240544*mcMS)/(793611*mbkin**3*(1 - mcMS/mbkin)) + 
            (225924352*mcMS)/(2380833*mbkin**2*(1 - mcMS/mbkin)) - 
            (379482310*mcMS)/(2380833*mbkin*(1 - mcMS/mbkin)) - 
            (936051440*mcMS**2)/(2380833*mbkin**4*(1 - mcMS/mbkin)) - 
            (7488411520*mcMS**2)/(7142499*mbkin**3*(1 - mcMS/mbkin)) + 
            (12578191225*mcMS**2)/(7142499*mbkin**2*(1 - mcMS/mbkin)) + 
            (214748656*mcMS**3)/(89505*mbkin**5*(1 - mcMS/mbkin)) + 
            (1717989248*mcMS**3)/(268515*mbkin**4*(1 - mcMS/mbkin)) - 
            (577137013*mcMS**3)/(53703*mbkin**3*(1 - mcMS/mbkin)) - 
            (24223298056*mcMS**4)/(2297295*mbkin**6*(1 - mcMS/mbkin)) - 
            (193786384448*mcMS**4)/(6891885*mbkin**5*(1 - mcMS/mbkin)) + 
            (130200227051*mcMS**4)/(2756754*mbkin**4*(1 - mcMS/mbkin)) + 
            (1609166072*mcMS**5)/(45045*mbkin**7*(1 - mcMS/mbkin)) + 
            (12873328576*mcMS**5)/(135135*mbkin**6*(1 - mcMS/mbkin)) - 
            (8649267637*mcMS**5)/(54054*mbkin**5*(1 - mcMS/mbkin)) - 
            (33540544*mcMS**6)/(351*mbkin**8*(1 - mcMS/mbkin)) - 
            (268324352*mcMS**6)/(1053*mbkin**7*(1 - mcMS/mbkin)) + 
            (450701060*mcMS**6)/(1053*mbkin**6*(1 - mcMS/mbkin)) + 
            (24025408*mcMS**7)/(117*mbkin**9*(1 - mcMS/mbkin)) + 
            (192203264*mcMS**7)/(351*mbkin**8*(1 - mcMS/mbkin)) - 
            (322841420*mcMS**7)/(351*mbkin**7*(1 - mcMS/mbkin)) - 
            (20841998048*mcMS**8)/(57915*mbkin**10*(1 - mcMS/mbkin)) - 
            (166735984384*mcMS**8)/(173745*mbkin**9*(1 - mcMS/mbkin)) + 
            (56012869754*mcMS**8)/(34749*mbkin**8*(1 - mcMS/mbkin)) + 
            (463452448*mcMS**9)/(891*mbkin**11*(1 - mcMS/mbkin)) + 
            (3707619584*mcMS**9)/(2673*mbkin**10*(1 - mcMS/mbkin)) - 
            (6227642270*mcMS**9)/(2673*mbkin**9*(1 - mcMS/mbkin)) - 
            (84293728*mcMS**10)/(135*mbkin**12*(1 - mcMS/mbkin)) - 
            (674349824*mcMS**10)/(405*mbkin**11*(1 - mcMS/mbkin)) + 
            (226539394*mcMS**10)/(81*mbkin**10*(1 - mcMS/mbkin)) + 
            (354122336*mcMS**11)/(567*mbkin**13*(1 - mcMS/mbkin)) + 
            (2832978688*mcMS**11)/(1701*mbkin**12*(1 - mcMS/mbkin)) - 
            (4758518890*mcMS**11)/(1701*mbkin**11*(1 - mcMS/mbkin)) - 
            (295158704*mcMS**12)/(567*mbkin**14*(1 - mcMS/mbkin)) - 
            (2361269632*mcMS**12)/(1701*mbkin**13*(1 - mcMS/mbkin)) + 
            (3966195085*mcMS**12)/(1701*mbkin**12*(1 - mcMS/mbkin)) + 
            (48659824*mcMS**13)/(135*mbkin**15*(1 - mcMS/mbkin)) + 
            (389278592*mcMS**13)/(405*mbkin**14*(1 - mcMS/mbkin)) - 
            (130773277*mcMS**13)/(81*mbkin**13*(1 - mcMS/mbkin)) - 
            (120505408*mcMS**14)/(585*mbkin**16*(1 - mcMS/mbkin)) - 
            (964043264*mcMS**14)/(1755*mbkin**15*(1 - mcMS/mbkin)) + 
            (323858284*mcMS**14)/(351*mbkin**14*(1 - mcMS/mbkin)) + 
            (12991751104*mcMS**15)/(135135*mbkin**17*(1 - mcMS/mbkin)) + 
            (103934008832*mcMS**15)/(405405*mbkin**16*(1 - mcMS/mbkin)) - 
            (34915331092*mcMS**15)/(81081*mbkin**15*(1 - mcMS/mbkin)) - 
            (108273376*mcMS**16)/(3003*mbkin**18*(1 - mcMS/mbkin)) - 
            (866187008*mcMS**16)/(9009*mbkin**17*(1 - mcMS/mbkin)) + 
            (1454923490*mcMS**16)/(9009*mbkin**16*(1 - mcMS/mbkin)) + 
            (11167232*mcMS**17)/(1053*mbkin**19*(1 - mcMS/mbkin)) + 
            (89337856*mcMS**17)/(3159*mbkin**18*(1 - mcMS/mbkin)) - 
            (150059680*mcMS**17)/(3159*mbkin**17*(1 - mcMS/mbkin)) - 
            (1476642704*mcMS**18)/(626535*mbkin**20*(1 - mcMS/mbkin)) - 
            (11813141632*mcMS**18)/(1879605*mbkin**19*(1 - mcMS/mbkin)) + 
            (3968477267*mcMS**18)/(375921*mbkin**18*(1 - mcMS/mbkin)) + 
            (854941712*mcMS**19)/(2297295*mbkin**21*(1 - mcMS/mbkin)) + 
            (6839533696*mcMS**19)/(6891885*mbkin**20*(1 - mcMS/mbkin)) - 
            (2297655851*mcMS**19)/(1378377*mbkin**19*(1 - mcMS/mbkin)) - 
            (696197624*mcMS**20)/(18706545*mbkin**22*(1 - mcMS/mbkin)) - 
            (5569580992*mcMS**20)/(56119635*mbkin**21*(1 - mcMS/mbkin)) + 
            (3742062229*mcMS**20)/(22447854*mbkin**20*(1 - mcMS/mbkin)) + 
            (602792*mcMS**21)/(340119*mbkin**23*(1 - mcMS/mbkin)) + 
            (4822336*mcMS**21)/(1020357*mbkin**22*(1 - mcMS/mbkin)) - 
            (16200035*mcMS**21)/(2040714*mbkin**21*(1 - mcMS/mbkin)) - 
            (688*mcMS)/(243*mbkin**4*(-1 + mcMS/mbkin)) + (688*mcMS**2)/
                (243*mbkin**5*(-1 + mcMS/mbkin)) + (1760*mcMS**3)/
                (9*mbkin**6*(-1 + mcMS/mbkin)) - (1760*mcMS**4)/
                (9*mbkin**7*(-1 + mcMS/mbkin)) + (1456*mcMS**5)/
                (9*mbkin**8*(-1 + mcMS/mbkin)) - (1456*mcMS**6)/
                (9*mbkin**9*(-1 + mcMS/mbkin)) + (180704*mcMS*np.pi**2)/(9009*mbkin) - 
            (600176*mcMS**2*np.pi**2)/(3861*mbkin**2) + (308192*mcMS**3*np.pi**2)/
                (495*mbkin**3) - (2032544*mcMS**4*np.pi**2)/(1155*mbkin**4) + 
            (103328*mcMS**5*np.pi**2)/(27*mbkin**5) - (58064*mcMS**6*np.pi**2)/
                (9*mbkin**6) + (25184*mcMS**7*np.pi**2)/(3*mbkin**7) - 
            (531904*mcMS**8*np.pi**2)/(63*mbkin**8) + (98656*mcMS**9*np.pi**2)/
                (15*mbkin**9) - (35536*mcMS**10*np.pi**2)/(9*mbkin**10) + 
            (339424*mcMS**11*np.pi**2)/(189*mbkin**11) - (59296*mcMS**12*np.pi**2)/
                (99*mbkin**12) + (22816*mcMS**13*np.pi**2)/(165*mbkin**13) - 
            (381488*mcMS**14*np.pi**2)/(19305*mbkin**14) + (1696*mcMS**15*np.pi**2)/
                (1287*mbkin**15) + ((1312*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**3) + 
                (2912*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(9*mbkin**7) + 
                (99520*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) - 
                (188264*mcMS**2)/(351*mbkin**2*(1 - mcMS/mbkin)**2) + 
                (214748656*mcMS**3)/(29835*mbkin**3*(1 - mcMS/mbkin)**2) - 
                (117272664508*mcMS**4)/(2297295*mbkin**4*(1 - mcMS/mbkin)**2) + 
                (37671647480*mcMS**5)/(153153*mbkin**5*(1 - mcMS/mbkin)**2) - 
                (39524020924*mcMS**6)/(45045*mbkin**6*(1 - mcMS/mbkin)**2) + 
                (283309760*mcMS**7)/(117*mbkin**7*(1 - mcMS/mbkin)**2) - 
                (102678435568*mcMS**8)/(19305*mbkin**8*(1 - mcMS/mbkin)**2) + 
                (183023630624*mcMS**9)/(19305*mbkin**9*(1 - mcMS/mbkin)**2) - 
                (20628036448*mcMS**10)/(1485*mbkin**10*(1 - mcMS/mbkin)**2) + 
                (15933731552*mcMS**11)/(945*mbkin**11*(1 - mcMS/mbkin)**2) - 
                (3216923384*mcMS**12)/(189*mbkin**12*(1 - mcMS/mbkin)**2) + 
                (13510105424*mcMS**13)/(945*mbkin**13*(1 - mcMS/mbkin)**2) - 
                (448387144*mcMS**14)/(45*mbkin**14*(1 - mcMS/mbkin)**2) + 
                (36851821888*mcMS**15)/(6435*mbkin**15*(1 - mcMS/mbkin)**2) - 
                (17284092368*mcMS**16)/(6435*mbkin**16*(1 - mcMS/mbkin)**2) + 
                (3906097568*mcMS**17)/(3861*mbkin**17*(1 - mcMS/mbkin)**2) - 
                (3669719752*mcMS**18)/(12285*mbkin**18*(1 - mcMS/mbkin)**2) + 
                (9001646128*mcMS**19)/(135135*mbkin**19*(1 - mcMS/mbkin)**2) - 
                (4847541068*mcMS**20)/(459459*mbkin**20*(1 - mcMS/mbkin)**2) + 
                (347227064*mcMS**21)/(328185*mbkin**21*(1 - mcMS/mbkin)**2) - 
                (301396*mcMS**22)/(5967*mbkin**22*(1 - mcMS/mbkin)**2) + 
                (14120272*mcMS)/(264537*mbkin*(1 - mcMS/mbkin)) - 
                (468025720*mcMS**2)/(793611*mbkin**2*(1 - mcMS/mbkin)) + 
                (107374328*mcMS**3)/(29835*mbkin**3*(1 - mcMS/mbkin)) - 
                (12111649028*mcMS**4)/(765765*mbkin**4*(1 - mcMS/mbkin)) + 
                (804583036*mcMS**5)/(15015*mbkin**5*(1 - mcMS/mbkin)) - 
                (16770272*mcMS**6)/(117*mbkin**6*(1 - mcMS/mbkin)) + 
                (12012704*mcMS**7)/(39*mbkin**7*(1 - mcMS/mbkin)) - 
                (10420999024*mcMS**8)/(19305*mbkin**8*(1 - mcMS/mbkin)) + 
                (231726224*mcMS**9)/(297*mbkin**9*(1 - mcMS/mbkin)) - 
                (42146864*mcMS**10)/(45*mbkin**10*(1 - mcMS/mbkin)) + 
                (177061168*mcMS**11)/(189*mbkin**11*(1 - mcMS/mbkin)) - 
                (147579352*mcMS**12)/(189*mbkin**12*(1 - mcMS/mbkin)) + 
                (24329912*mcMS**13)/(45*mbkin**13*(1 - mcMS/mbkin)) - 
                (60252704*mcMS**14)/(195*mbkin**14*(1 - mcMS/mbkin)) + 
                (6495875552*mcMS**15)/(45045*mbkin**15*(1 - mcMS/mbkin)) - 
                (54136688*mcMS**16)/(1001*mbkin**16*(1 - mcMS/mbkin)) + 
                (5583616*mcMS**17)/(351*mbkin**17*(1 - mcMS/mbkin)) - 
                (738321352*mcMS**18)/(208845*mbkin**18*(1 - mcMS/mbkin)) + 
                (427470856*mcMS**19)/(765765*mbkin**19*(1 - mcMS/mbkin)) - 
                (348098812*mcMS**20)/(6235515*mbkin**20*(1 - mcMS/mbkin)) + 
                (301396*mcMS**21)/(113373*mbkin**21*(1 - mcMS/mbkin)) + 
                (968*mcMS)/(243*mbkin**4*(-1 + mcMS/mbkin)) - (968*mcMS**2)/
                (243*mbkin**5*(-1 + mcMS/mbkin)) + (170800*mcMS**3)/
                (243*mbkin**6*(-1 + mcMS/mbkin)) - (170800*mcMS**4)/
                (243*mbkin**7*(-1 + mcMS/mbkin)) + (728*mcMS**5)/(mbkin**8*(-1 + 
                    mcMS/mbkin)) - (728*mcMS**6)/(mbkin**9*(-1 + mcMS/mbkin)))*
                np.log(mcMS**2/mbkin**2))*np.log(1 - mcMS/mbkin) + 
            ((1790032*mcMS)/(675675*mbkin) + (222153992*mcMS**2)/
                (868725*mbkin**2) - (993662224*mcMS**3)/(467775*mbkin**3) + 
            (2262297616*mcMS**4)/(280665*mbkin**4) - (2022928*mcMS**5)/
                (105*mbkin**5) + (200330504*mcMS**6)/(6075*mbkin**6) - 
            (260772688*mcMS**7)/(6075*mbkin**7) + (615949856*mcMS**8)/
                (14175*mbkin**8) - (2773168*mcMS**9)/(81*mbkin**9) + 
            (35346424*mcMS**10)/(1701*mbkin**10) - (405443344*mcMS**11)/
                (42525*mbkin**11) + (1498143088*mcMS**12)/(467775*mbkin**12) - 
            (149130896*mcMS**13)/(200475*mbkin**13) + (2062888*mcMS**14)/
                (19305*mbkin**14) - (8705456*mcMS**15)/(1216215*mbkin**15))*
            np.log(1 - mcMS/mbkin)**2 + (304/(81*mbkin**3) + (160*mcMS**2)/
                (9*mbkin**5) - (464*mcMS**4)/(27*mbkin**7) - (4400*mcMS**6)/
                (81*mbkin**9) + (32*mcMS**8)/(3*mbkin**11) + 
            (1688*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**3) - 
            (2824*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) - 
            (24976*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) - 
            (688*mcMS)/(243*mbkin**4*(1 + mcMS/mbkin)) - (688*mcMS**2)/
                (243*mbkin**5*(1 + mcMS/mbkin)) + (1760*mcMS**3)/
                (9*mbkin**6*(1 + mcMS/mbkin)) + (1760*mcMS**4)/
                (9*mbkin**7*(1 + mcMS/mbkin)) + (1456*mcMS**5)/
                (9*mbkin**8*(1 + mcMS/mbkin)) + (1456*mcMS**6)/
                (9*mbkin**9*(1 + mcMS/mbkin)))*np.log(1 + mcMS/mbkin) + 
            (7904/(81*mbkin**3) + (2968*mbkin)/(243*mcMS**4) - 
            89848/(3645*mbkin*mcMS**2) - (53168*mcMS**2)/(243*mbkin**5) + 
            (16616*mcMS**4)/(729*mbkin**7) + (59320*mcMS**6)/(81*mbkin**9) - 
            (707344*mcMS**8)/(1215*mbkin**11) - (1000*np.sqrt(0j + mcMS**2/mbkin**2))/
                (243*mbkin**3) - (1544*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) - 
            (22544*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3))*
            np.log(1 - mcMS**2/mbkin**2) + np.log(mcMS**2/mbkin**2)*(-4064/(81*mbkin**3) - 
            (16624*mcMS**2)/(81*mbkin**5) + (704*mcMS**2)/(9*mbkin**4) + 
            (1408*mcMS**2)/(9*mbkin**3) + (28194016*mcMS**2)/(264537*mbkin**2) + 
            (766608176*mcMS**3)/(793611*mbkin**3) + (64*mcMS**4)/(3*mbkin**8) - 
            (27908*mcMS**4)/(729*mbkin**7) - (11512*mcMS**4)/(9*mbkin**6) - 
            (60064*mcMS**4)/(27*mbkin**5) - (4711.275998320473*mcMS**4)/mbkin**4 + 
            (68321780792*mcMS**5)/(3357585*mbkin**5) - (22376*mcMS**6)/
                (81*mbkin**9) + (4480*mcMS**6)/(9*mbkin**7) - (2844597796984*mcMS**6)/
                (43648605*mbkin**6) + (6272042344856*mcMS**7)/(43648605*mbkin**7) - 
            (157016*mcMS**8)/(1215*mbkin**11) - (752*mcMS**8)/(9*mbkin**10) - 
            (3008*mcMS**8)/(9*mbkin**9) - (11623406689684*mcMS**8)/
                (43648605*mbkin**8) + (5838076386728*mcMS**9)/(14549535*mbkin**9) - 
            (7138029219736*mcMS**10)/(14549535*mbkin**10) + 
            (2379988730648*mcMS**11)/(4849845*mbkin**11) - 
            (2504128296664*mcMS**12)/(6235515*mbkin**12) + 
            (1669676023736*mcMS**13)/(6235515*mbkin**13) - 
            (899165488072*mcMS**14)/(6235515*mbkin**14) + (385394431736*mcMS**15)/
                (6235515*mbkin**15) - (899326358968*mcMS**16)/(43648605*mbkin**16) + 
            (224846820392*mcMS**17)/(43648605*mbkin**17) - (39681135608*mcMS**18)/
                (43648605*mbkin**18) + (629890504*mcMS**19)/(6235515*mbkin**19) - 
            (602792*mcMS**20)/(113373*mbkin**20) + (3820703038853*mcMS**2)/
                (10213773570*mbkin**2*(1 - mcMS/mbkin)**2) - 
            (1324623229305471443*mcMS**3)/(267396592062600*mbkin**3*
                (1 - mcMS/mbkin)**2) + (5514257196652051*mcMS**4)/
                (154653899400*mbkin**4*(1 - mcMS/mbkin)**2) - 
            (2445534345846504631*mcMS**5)/(14073504845400*mbkin**5*
                (1 - mcMS/mbkin)**2) + (73546477272645373*mcMS**6)/
                (118264746600*mbkin**6*(1 - mcMS/mbkin)**2) - 
            (70971441129844033*mcMS**7)/(41392661310*mbkin**7*(1 - mcMS/mbkin)**
                2) + (5292307230847823*mcMS**8)/(1407913650*mbkin**8*
                (1 - mcMS/mbkin)**2) - (1883705490346937*mcMS**9)/
                (281582730*mbkin**9*(1 - mcMS/mbkin)**2) + 
            (3996921465647711*mcMS**10)/(408211650*mbkin**10*(1 - mcMS/mbkin)**
                2) - (34379222836281097*mcMS**11)/(2894591700*mbkin**11*
                (1 - mcMS/mbkin)**2) + (991151981540419*mcMS**12)/
                (82702620*mbkin**12*(1 - mcMS/mbkin)**2) - 
            (2648132431926677*mcMS**13)/(263144700*mbkin**13*(1 - mcMS/mbkin)**
                2) + (223279079747087713*mcMS**14)/(31840508700*mbkin**14*
                (1 - mcMS/mbkin)**2) - (4765673910687463*mcMS**15)/
                (1182647466*mbkin**15*(1 - mcMS/mbkin)**2) + 
            (55872539918685143*mcMS**16)/(29566186650*mbkin**16*
                (1 - mcMS/mbkin)**2) - (251792791028627*mcMS**17)/
                (353783430*mbkin**17*(1 - mcMS/mbkin)**2) + 
            (2416054156353307*mcMS**18)/(11497961475*mbkin**18*(1 - mcMS/mbkin)**
                2) - (1436625073319083*mcMS**19)/(30661230600*mbkin**19*
                (1 - mcMS/mbkin)**2) + (20887342577910533*mcMS**20)/
                (2814700969080*mbkin**20*(1 - mcMS/mbkin)**2) - 
            (198978903088724189*mcMS**21)/(267396592062600*mbkin**21*
                (1 - mcMS/mbkin)**2) + (24672660896281*mcMS**22)/
                (694536602760*mbkin**22*(1 - mcMS/mbkin)**2) - 
            (2291073463739113*mcMS)/(61582245444720*mbkin*(1 - mcMS/mbkin)) + 
            (66120814157661443*mcMS**2)/(184746736334160*mbkin**2*
                (1 - mcMS/mbkin)) - (1166928234496119443*mcMS**3)/
                (534793184125200*mbkin**3*(1 - mcMS/mbkin)) + 
            (118345817852851093*mcMS**4)/(11884292980560*mbkin**4*
                (1 - mcMS/mbkin)) - (49345371782260483*mcMS**5)/
                (1429928299800*mbkin**5*(1 - mcMS/mbkin)) + 
            (146208599147473679*mcMS**6)/(1572921129780*mbkin**6*
                (1 - mcMS/mbkin)) - (64819098503293*mcMS**7)/(324246780*mbkin**7*
                (1 - mcMS/mbkin)) + (7714866552402197*mcMS**8)/
                (22029707700*mbkin**8*(1 - mcMS/mbkin)) - 
            (2242974842855321*mcMS**9)/(4432012200*mbkin**9*(1 - mcMS/mbkin)) + 
            (1713682422600793*mcMS**10)/(2820371400*mbkin**10*
                (1 - mcMS/mbkin)) - (13374421001572999*mcMS**11)/
                (21998896920*mbkin**11*(1 - mcMS/mbkin)) + 
            (11151579707994493*mcMS**12)/(21998896920*mbkin**12*
                (1 - mcMS/mbkin)) - (178800900176533*mcMS**13)/
                (509233725*mbkin**13*(1 - mcMS/mbkin)) + 
            (32217277599333127*mcMS**14)/(160502156100*mbkin**14*
                (1 - mcMS/mbkin)) - (43352528289259463*mcMS**15)/
                (462623861700*mbkin**15*(1 - mcMS/mbkin)) + 
            (111704868472735*mcMS**16)/(3177618444*mbkin**16*(1 - mcMS/mbkin)) - 
            (7230231600513707*mcMS**17)/(699076057680*mbkin**17*
                (1 - mcMS/mbkin)) + (1774149351078739*mcMS**18)/
                (771707336400*mbkin**18*(1 - mcMS/mbkin)) - 
            (64726178334967321*mcMS**19)/(178264394708400*mbkin**19*
                (1 - mcMS/mbkin)) + (369023890038773819*mcMS**20)/
                (10161070498378800*mbkin**20*(1 - mcMS/mbkin)) - 
            (22826272832761*mcMS**21)/(13196195452440*mbkin**21*
                (1 - mcMS/mbkin)) + (344*mcMS**2)/(243*mbkin**5*
                (-1 + mcMS/mbkin)) - (880*mcMS**4)/(9*mbkin**7*
                (-1 + mcMS/mbkin)) - (728*mcMS**6)/(9*mbkin**9*
                (-1 + mcMS/mbkin)) - (344*mcMS**2)/(243*mbkin**5*
                (1 + mcMS/mbkin)) + (880*mcMS**4)/(9*mbkin**7*(1 + mcMS/mbkin)) + 
            (728*mcMS**6)/(9*mbkin**9*(1 + mcMS/mbkin)) + 
            7280/(81*mbkin*(mbkin**2 - mcMS**2)) - (5888*mcMS**2)/
                (81*mbkin**3*(mbkin**2 - mcMS**2)) - (7744*mcMS**4)/
                (81*mbkin**5*(mbkin**2 - mcMS**2)) - (57776*mcMS**6)/
                (243*mbkin**7*(mbkin**2 - mcMS**2)) - (1696*mcMS**8)/
                (81*mbkin**9*(mbkin**2 - mcMS**2)) - (1408*mcMS**10)/
                (81*mbkin**11*(mbkin**2 - mcMS**2)) + (32*mcMS**4*np.pi**2)/mbkin**6 + 
            (128*mcMS**4*np.pi**2)/(3*mbkin**5) - (32*mcMS**4*np.pi**2)/mbkin**4 + 
            ((-1312*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**3) - 
                (2912*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(9*mbkin**7) - 
                (99520*(mcMS**2/mbkin**2)**(3/2))/(243*mbkin**3) + 
                (968*mcMS)/(243*mbkin**4*(1 + mcMS/mbkin)) + (968*mcMS**2)/
                (243*mbkin**5*(1 + mcMS/mbkin)) + (170800*mcMS**3)/
                (243*mbkin**6*(1 + mcMS/mbkin)) + (170800*mcMS**4)/
                (243*mbkin**7*(1 + mcMS/mbkin)) + (728*mcMS**5)/(mbkin**8*(1 + 
                    mcMS/mbkin)) + (728*mcMS**6)/(mbkin**9*(1 + mcMS/mbkin)))*
                np.log(1 + mcMS/mbkin) + ((176*mcMS**2)/(9*mbkin**5) + 
                (105376*mcMS**4)/(243*mbkin**7) + (3040*mcMS**6)/(27*mbkin**9) + 
                (5632*mcMS**8)/(81*mbkin**11))*np.log(1 - mcMS**2/mbkin**2)) + 
            (64/mbkin**3 - (256*mcMS**2)/(3*mbkin**5) + (192*mcMS**2)/mbkin**4 + 
            (384*mcMS**2)/mbkin**3 - (128*mcMS**4)/mbkin**7 - 
            (144*mcMS**4)/mbkin**6 - (576*mcMS**4)/mbkin**5 + 
            (256*mcMS**6)/mbkin**9 + (384*mcMS**6)/mbkin**7 - 
            (320*mcMS**8)/(3*mbkin**11) - (48*mcMS**8)/mbkin**10 - 
            (192*mcMS**8)/mbkin**9 + ((288*mcMS**4)/mbkin**6 + (384*mcMS**4)/
                mbkin**5)*np.log(mcMS**2/mbkin**2))*np.log(2/mus) + 
            (64/(27*mbkin**3) + (287769071069817419*mcMS)/(4064428199351520*
                mbkin) - (3944684*mcMS)/(405405*(mbkin - mcMS)) + 
            (2816*mcMS**2)/(81*mbkin**5) - (704*mcMS**2)/(9*mbkin**4) - 
            (1408*mcMS**2)/(9*mbkin**3) - (379059751524889*mcMS**2)/
                (674109055620*mbkin**2) + (2078096*mcMS**2)/(27027*mbkin*
                (mbkin - mcMS)) + (950435130379316129*mcMS**3)/(1069586368250400*
                mbkin**3) - (3451012*mcMS**3)/(11583*mbkin**2*(mbkin - mcMS)) + 
            (2944*mcMS**4)/(27*mbkin**7) - (16*mcMS**4)/(3*mbkin**6) + 
            (320*mcMS**4)/mbkin**5 - (15374672075006119*mcMS**4)/
                (4494060370800*mbkin**4) + (3544208*mcMS**4)/(4455*mbkin**3*
                (mbkin - mcMS)) - (597399054412007*mcMS**5)/(111029726808*
                mbkin**5) - (5843564*mcMS**5)/(3465*mbkin**4*(mbkin - mcMS)) - 
            (8960*mcMS**6)/(27*mbkin**9) - (4480*mcMS**6)/(9*mbkin**7) + 
            (20418659248277*mcMS**6)/(567431865*mbkin**6) + (1188272*mcMS**6)/
                (405*mbkin**5*(mbkin - mcMS)) - (266828288220607*mcMS**7)/
                (2723672952*mbkin**7) - (333868*mcMS**7)/(81*mbkin**6*
                (mbkin - mcMS)) + (15040*mcMS**8)/(81*mbkin**11) + 
            (752*mcMS**8)/(9*mbkin**10) + (3008*mcMS**8)/(9*mbkin**9) + 
            (125044040747685461*mcMS**8)/(604969665300*mbkin**8) + 
            (289616*mcMS**8)/(63*mbkin**7*(mbkin - mcMS)) - 
            (10861877424689*mcMS**9)/(32432400*mbkin**9) - (764612*mcMS**9)/
                (189*mbkin**8*(mbkin - mcMS)) + (469354130863495*mcMS**10)/
                (1099944846*mbkin**10) + (1134544*mcMS**10)/(405*mbkin**9*
                (mbkin - mcMS)) - (11550816291717145*mcMS**11)/
                (26398676304*mbkin**11) - (204332*mcMS**11)/(135*mbkin**10*
                (mbkin - mcMS)) + (88019407268539741*mcMS**12)/
                (241987866120*mbkin**12) + (3903376*mcMS**12)/(6237*mbkin**11*
                (mbkin - mcMS)) - (42363677789859031*mcMS**13)/
                (172848475800*mbkin**13) - (170476*mcMS**13)/(891*mbkin**12*
                (mbkin - mcMS)) + (112063923742729709*mcMS**14)/
                (842636319525*mbkin**14) + (262384*mcMS**14)/(6435*mbkin**13*
                (mbkin - mcMS)) - (60126655391804087*mcMS**15)/
                (1048614086520*mbkin**15) - (2193556*mcMS**15)/(405405*mbkin**14*
                (mbkin - mcMS)) + (287746271552629*mcMS**16)/(14980201236*
                mbkin**16) + (19504*mcMS**16)/(57915*mbkin**15*(mbkin - mcMS)) - 
            (613010854803901*mcMS**17)/(127104737760*mbkin**17) + 
            (3461383196851759*mcMS**18)/(4051463516100*mbkin**18) - 
            (5992954493052799*mcMS**19)/(62916845191200*mbkin**19) + 
            (398150205649799*mcMS**20)/(79177172714640*mbkin**20) - 
            (3820703038853*mcMS**2)/(10213773570*mbkin**2*(1 - mcMS/mbkin)**2) + 
            (1324623229305471443*mcMS**3)/(267396592062600*mbkin**3*
                (1 - mcMS/mbkin)**2) - (5514257196652051*mcMS**4)/
                (154653899400*mbkin**4*(1 - mcMS/mbkin)**2) + 
            (2445534345846504631*mcMS**5)/(14073504845400*mbkin**5*
                (1 - mcMS/mbkin)**2) - (73546477272645373*mcMS**6)/
                (118264746600*mbkin**6*(1 - mcMS/mbkin)**2) + 
            (70971441129844033*mcMS**7)/(41392661310*mbkin**7*(1 - mcMS/mbkin)**
                2) - (5292307230847823*mcMS**8)/(1407913650*mbkin**8*
                (1 - mcMS/mbkin)**2) + (1883705490346937*mcMS**9)/
                (281582730*mbkin**9*(1 - mcMS/mbkin)**2) - 
            (3996921465647711*mcMS**10)/(408211650*mbkin**10*(1 - mcMS/mbkin)**
                2) + (34379222836281097*mcMS**11)/(2894591700*mbkin**11*
                (1 - mcMS/mbkin)**2) - (991151981540419*mcMS**12)/
                (82702620*mbkin**12*(1 - mcMS/mbkin)**2) + 
            (2648132431926677*mcMS**13)/(263144700*mbkin**13*(1 - mcMS/mbkin)**
                2) - (223279079747087713*mcMS**14)/(31840508700*mbkin**14*
                (1 - mcMS/mbkin)**2) + (4765673910687463*mcMS**15)/
                (1182647466*mbkin**15*(1 - mcMS/mbkin)**2) - 
            (55872539918685143*mcMS**16)/(29566186650*mbkin**16*
                (1 - mcMS/mbkin)**2) + (251792791028627*mcMS**17)/
                (353783430*mbkin**17*(1 - mcMS/mbkin)**2) - 
            (2416054156353307*mcMS**18)/(11497961475*mbkin**18*(1 - mcMS/mbkin)**
                2) + (1436625073319083*mcMS**19)/(30661230600*mbkin**19*
                (1 - mcMS/mbkin)**2) - (20887342577910533*mcMS**20)/
                (2814700969080*mbkin**20*(1 - mcMS/mbkin)**2) + 
            (198978903088724189*mcMS**21)/(267396592062600*mbkin**21*
                (1 - mcMS/mbkin)**2) - (24672660896281*mcMS**22)/
                (694536602760*mbkin**22*(1 - mcMS/mbkin)**2) + 
            (2291073463739113*mcMS)/(61582245444720*mbkin*(1 - mcMS/mbkin)) - 
            (66120814157661443*mcMS**2)/(184746736334160*mbkin**2*
                (1 - mcMS/mbkin)) + (1166928234496119443*mcMS**3)/
                (534793184125200*mbkin**3*(1 - mcMS/mbkin)) - 
            (118345817852851093*mcMS**4)/(11884292980560*mbkin**4*
                (1 - mcMS/mbkin)) + (49345371782260483*mcMS**5)/
                (1429928299800*mbkin**5*(1 - mcMS/mbkin)) - 
            (146208599147473679*mcMS**6)/(1572921129780*mbkin**6*
                (1 - mcMS/mbkin)) + (64819098503293*mcMS**7)/(324246780*mbkin**7*
                (1 - mcMS/mbkin)) - (7714866552402197*mcMS**8)/
                (22029707700*mbkin**8*(1 - mcMS/mbkin)) + 
            (2242974842855321*mcMS**9)/(4432012200*mbkin**9*(1 - mcMS/mbkin)) - 
            (1713682422600793*mcMS**10)/(2820371400*mbkin**10*
                (1 - mcMS/mbkin)) + (13374421001572999*mcMS**11)/
                (21998896920*mbkin**11*(1 - mcMS/mbkin)) - 
            (11151579707994493*mcMS**12)/(21998896920*mbkin**12*
                (1 - mcMS/mbkin)) + (178800900176533*mcMS**13)/
                (509233725*mbkin**13*(1 - mcMS/mbkin)) - 
            (32217277599333127*mcMS**14)/(160502156100*mbkin**14*
                (1 - mcMS/mbkin)) + (43352528289259463*mcMS**15)/
                (462623861700*mbkin**15*(1 - mcMS/mbkin)) - 
            (111704868472735*mcMS**16)/(3177618444*mbkin**16*(1 - mcMS/mbkin)) + 
            (7230231600513707*mcMS**17)/(699076057680*mbkin**17*
                (1 - mcMS/mbkin)) - (1774149351078739*mcMS**18)/
                (771707336400*mbkin**18*(1 - mcMS/mbkin)) + 
            (64726178334967321*mcMS**19)/(178264394708400*mbkin**19*
                (1 - mcMS/mbkin)) - (369023890038773819*mcMS**20)/
                (10161070498378800*mbkin**20*(1 - mcMS/mbkin)) + 
            (22826272832761*mcMS**21)/(13196195452440*mbkin**21*
                (1 - mcMS/mbkin)) + ((392*mcMS**2)/(3*mbkin**2) - 
                (736*mcMS**4)/(3*mbkin**6) - (2944*mcMS**4)/(9*mbkin**5) - 
                (440*mcMS**4)/mbkin**4 - (2328*mcMS**6)/mbkin**6 + 
                (1936*mcMS**8)/(3*mbkin**8))*np.log(mcMS**2/mbkin**2) + 
            (1168*mcMS**4*np.log(mcMS**2/mbkin**2)**2)/mbkin**4 + 
            ((-748886504*mcMS)/(8729721*mbkin) + (7937476*mcMS**2)/
                (11583*mbkin**2) - (950785096*mcMS**3)/(328185*mbkin**3) + 
                (1198844138*mcMS**4)/(135135*mbkin**4) - (22613456*mcMS**5)/
                (1053*mbkin**5) + (14677064*mcMS**6)/(351*mbkin**6) - 
                (7772528*mcMS**7)/(117*mbkin**7) + (2746783208*mcMS**8)/
                (31185*mbkin**8) - (4479008*mcMS**9)/(45*mbkin**9) + 
                (872224*mcMS**10)/(9*mbkin**10) - (15355264*mcMS**11)/
                (189*mbkin**11) + (17248564*mcMS**12)/(297*mbkin**12) - 
                (51768368*mcMS**13)/(1485*mbkin**13) + (998628536*mcMS**14)/
                (57915*mbkin**14) - (185744752*mcMS**15)/(27027*mbkin**15) + 
                (83704*mcMS**16)/(39*mbkin**16) - (531800*mcMS**17)/
                (1053*mbkin**17) + (5859964*mcMS**18)/(69615*mbkin**18) - 
                (171064*mcMS**19)/(19305*mbkin**19) + (150698*mcMS**20)/
                (340119*mbkin**20) + (188264*mcMS**2)/(351*mbkin**2*
                (1 - mcMS/mbkin)**2) - (214748656*mcMS**3)/(29835*mbkin**3*
                (1 - mcMS/mbkin)**2) + (117272664508*mcMS**4)/(2297295*mbkin**
                    4*(1 - mcMS/mbkin)**2) - (37671647480*mcMS**5)/(153153*mbkin**
                    5*(1 - mcMS/mbkin)**2) + (39524020924*mcMS**6)/(45045*mbkin**
                    6*(1 - mcMS/mbkin)**2) - (283309760*mcMS**7)/(117*mbkin**7*
                (1 - mcMS/mbkin)**2) + (102678435568*mcMS**8)/(19305*mbkin**
                    8*(1 - mcMS/mbkin)**2) - (183023630624*mcMS**9)/
                (19305*mbkin**9*(1 - mcMS/mbkin)**2) + (20628036448*mcMS**10)/
                (1485*mbkin**10*(1 - mcMS/mbkin)**2) - (15933731552*mcMS**11)/
                (945*mbkin**11*(1 - mcMS/mbkin)**2) + (3216923384*mcMS**12)/
                (189*mbkin**12*(1 - mcMS/mbkin)**2) - (13510105424*mcMS**13)/
                (945*mbkin**13*(1 - mcMS/mbkin)**2) + (448387144*mcMS**14)/
                (45*mbkin**14*(1 - mcMS/mbkin)**2) - (36851821888*mcMS**15)/
                (6435*mbkin**15*(1 - mcMS/mbkin)**2) + (17284092368*mcMS**16)/
                (6435*mbkin**16*(1 - mcMS/mbkin)**2) - (3906097568*mcMS**17)/
                (3861*mbkin**17*(1 - mcMS/mbkin)**2) + (3669719752*mcMS**18)/
                (12285*mbkin**18*(1 - mcMS/mbkin)**2) - (9001646128*mcMS**19)/
                (135135*mbkin**19*(1 - mcMS/mbkin)**2) + (4847541068*mcMS**20)/
                (459459*mbkin**20*(1 - mcMS/mbkin)**2) - (347227064*mcMS**21)/
                (328185*mbkin**21*(1 - mcMS/mbkin)**2) + (301396*mcMS**22)/
                (5967*mbkin**22*(1 - mcMS/mbkin)**2) - (14120272*mcMS)/
                (264537*mbkin*(1 - mcMS/mbkin)) + (468025720*mcMS**2)/
                (793611*mbkin**2*(1 - mcMS/mbkin)) - (107374328*mcMS**3)/
                (29835*mbkin**3*(1 - mcMS/mbkin)) + (12111649028*mcMS**4)/
                (765765*mbkin**4*(1 - mcMS/mbkin)) - (804583036*mcMS**5)/
                (15015*mbkin**5*(1 - mcMS/mbkin)) + (16770272*mcMS**6)/
                (117*mbkin**6*(1 - mcMS/mbkin)) - (12012704*mcMS**7)/
                (39*mbkin**7*(1 - mcMS/mbkin)) + (10420999024*mcMS**8)/
                (19305*mbkin**8*(1 - mcMS/mbkin)) - (231726224*mcMS**9)/
                (297*mbkin**9*(1 - mcMS/mbkin)) + (42146864*mcMS**10)/
                (45*mbkin**10*(1 - mcMS/mbkin)) - (177061168*mcMS**11)/
                (189*mbkin**11*(1 - mcMS/mbkin)) + (147579352*mcMS**12)/
                (189*mbkin**12*(1 - mcMS/mbkin)) - (24329912*mcMS**13)/
                (45*mbkin**13*(1 - mcMS/mbkin)) + (60252704*mcMS**14)/
                (195*mbkin**14*(1 - mcMS/mbkin)) - (6495875552*mcMS**15)/
                (45045*mbkin**15*(1 - mcMS/mbkin)) + (54136688*mcMS**16)/
                (1001*mbkin**16*(1 - mcMS/mbkin)) - (5583616*mcMS**17)/
                (351*mbkin**17*(1 - mcMS/mbkin)) + (738321352*mcMS**18)/
                (208845*mbkin**18*(1 - mcMS/mbkin)) - (427470856*mcMS**19)/
                (765765*mbkin**19*(1 - mcMS/mbkin)) + (348098812*mcMS**20)/
                (6235515*mbkin**20*(1 - mcMS/mbkin)) - (301396*mcMS**21)/
                (113373*mbkin**21*(1 - mcMS/mbkin)))*np.log(1 - mcMS/mbkin))*
            np.log(mus**2/mbkin**2) + ((-196*mcMS**2)/(3*mbkin**2) - 
            (776*mcMS**4)/mbkin**4 + (1164*mcMS**6)/mbkin**6 - 
            (968*mcMS**8)/(3*mbkin**8) - (584*mcMS**4*np.log(mcMS**2/mbkin**2))/
                mbkin**4)*np.log(mus**2/mbkin**2)**2 + ((-944*mcMS**2)/(81*mbkin**5) + 
            (12320*mcMS**4)/(81*mbkin**7) + (1696*mcMS**6)/(27*mbkin**9) - 
            (1792*mcMS**8)/(27*mbkin**11))*fp.polylog(2, 1 - mbkin**2/mcMS**2) + 
            ((-320*mcMS**2)/(9*mbkin**5) + (1856*mcMS**4)/(27*mbkin**7) + 
            (8800*mcMS**6)/(27*mbkin**9) - (256*mcMS**8)/(3*mbkin**11) - 
            (1688*np.sqrt(0j + mcMS**2/mbkin**2))/(243*mbkin**3) + 
            (14120*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (24976*(mcMS**2/mbkin**2)**(3/2))/(81*mbkin**3))*
            fp.polylog(2, -(mcMS/mbkin)) + ((-320*mcMS**2)/(9*mbkin**5) + 
            (1856*mcMS**4)/(27*mbkin**7) + (8800*mcMS**6)/(27*mbkin**9) - 
            (256*mcMS**8)/(3*mbkin**11) - (104*np.sqrt(0j + mcMS**2/mbkin**2))/
                (81*mbkin**3) - (29560*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) - 
            (70064*(mcMS**2/mbkin**2)**(3/2))/(81*mbkin**3))*
            fp.polylog(2, mcMS/mbkin) + ((16*mcMS**2)/mbkin**5 - 
            (544*mcMS**4)/(9*mbkin**7) + (688*mcMS**6)/(27*mbkin**9) + 
            (2048*mcMS**8)/(27*mbkin**11) + (500*np.sqrt(0j + mcMS**2/mbkin**2))/
                (243*mbkin**3) + (3860*mcMS**4*np.sqrt(0j + mcMS**2/mbkin**2))/(27*mbkin**7) + 
            (11272*(mcMS**2/mbkin**2)**(3/2))/(81*mbkin**3))*
            fp.polylog(2, mcMS**2/mbkin**2))))
    return res.real
