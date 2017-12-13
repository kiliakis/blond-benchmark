#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <sstream>
#include "utils.h"
using namespace std;

void linspace(const double start, const double end, const int n,
              double *__restrict__ out)
{
    const double step = (end - start) / (n - 1);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) out[i] = start + i * step;
}


void linspace(const float start, const float end, const int n,
              float *__restrict__ out)
{
    const float step = (end - start) / (n - 1);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) out[i] = start + i * step;
}