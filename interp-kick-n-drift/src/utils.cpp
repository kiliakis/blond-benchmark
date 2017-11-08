

#include "utils.h"

void linspace(const double start, const double end, const int n,
              double *__restrict__ out)
{
    const double step = (end - start) / (n - 1);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) out[i] = start + i * step;
}