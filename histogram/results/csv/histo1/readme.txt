Versions

v0: naive, serial implementation, doubles
v1: parallel, global histogram atomics
v2: parallel, local histo, serial reduction
v3: parallel, local histo, parallel reduction
v4: parallel, local histo, parallel reduce, loop tiling
v5: parallel, local histo, parallel reduce, loop tiling, parallel allocation
# v6: parallel, local histo, parallel reduce, loop tiling, parallel allocation, heavier opts
v7: cuda, global histo (512X1024)
v8: cuda, shared histo (512X1024)
v9: cuda, shared histo, simplified condition (512X1024)

Plots

1) v0 vs v1 vs v2 (scalling improvement)
2) v2 vs v3 (with many slices) to show that serial reduction can be a problem
3) v3 vs v4 vs v5 to show that loop tiling and parallel allocation work
4) all gpu versions
5) best cpu vs best gpu v9 vs v5
