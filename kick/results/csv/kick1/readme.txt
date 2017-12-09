v0: std::sin, double, vec/novec, icc/g++
v1: same but float
v2: vdt::fast_sin, double, vec/novec, icc/g++ 
v3: same but float
# v4: vdt::fast_sin, double, vec/novec, icc/g++, L1 tiles
v5: vdt::fast_sin, double, vec/novec, icc/g++, L2 tiles
v6: std::sin, double, vec/novec, icc/g++, L2 tiles

Plots

1) v0 icc \w or \wo vec (to show vec effect)
2) v2 gcc \w or \wo vec (to show vec effect)
3) v0,v2 \w vec, icc/gcc to show that icc is better with std::sin and in general
4) v1 vs v0 on icc vec (floats vs doubles)
5) v6 vs v0 to show effect of L2 tiles