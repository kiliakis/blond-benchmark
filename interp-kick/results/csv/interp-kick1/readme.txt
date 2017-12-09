
Versions

v0: basic implementation, parallel, double, gcc/icc, vec/novec, notcm
v1: double, precalculate voltages, tcm/notcm, gcc/icc, vec/novec
v2: double, precalculate voltages + loop tiling, tcm/notcm, gcc/icc, vec/novec
v3: float, precalculate voltages + loop tiling, tcm/notcm, gcc/icc, vec/novec
v4: double, precalculate voltages + functor + loop tiling, tcm/notcm, gcc/icc, vec/novec
v5: double, precalculate voltages functor + loop tiling + AoS, tcm/notcm, gcc/icc, vec/novec
v6: float, precalculate voltages + loop tiling, tcm/notcm, gcc/icc, vec/novec


Plots

1) v0 vs v1 vs v2 vs v4 gcc, tcm, vec to show the max speedup
2) v4 gcc vs icc, vec, tcm to show that gcc does better than icc
3) v5 vs v6 gcc, vec, tcm to show float vs double
4) v4 gcc, tcm/notcm, vec/novec to show effect of tcm and vec