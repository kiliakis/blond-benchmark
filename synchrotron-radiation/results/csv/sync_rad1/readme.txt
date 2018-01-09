Versions

v0: std, double, icc/g++, vec, notcm
v1: boost, double, notcm, icc/g++, vec/novec 
v2: boost, double, tiled, notcm, icc/g++, vec/novec
v3: mkl, double, icc, vec/novec, tcm/notcm
v4: mkl, float, icc, vec, tcm
v5: boost, float, icc/g++, notcm, vec
v6: std, float, icc/g++, notcm, vec
v7: mkl, double, tiled, optimized, icc, vec/novec, tcm/notcm
v8: cuda double, host API
v9: cuda double, host API optimized
v10: cuda double, device API

plots

1: v0, v1, vec, icc/g++ (boost vs std)
2: v1, v2, vec/novec, tcm, icc/g++ (best implementation except mkl)
3: v2 vec best (g++) v3, v7 icc vec tcm (best implementation + optimization)
4: v7 vec/novec, tcm/notcm (to show effect of tcm and vec)
5: v5 g++ float vec, v1 g++ double vec (to show effect of float in boost)
6: v3 mkl double icc vec vs v4 mkl float  (to show effect of float in mkl)
7: all cuda versions
8: best cuda vs best cpu (v7 vec notcm vs v10) 