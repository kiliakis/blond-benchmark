#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <sstream>
#include "utils.h"
#include <stdio.h>
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

// Kostis
size_t L1_cache_size (void)
{
    FILE * p = 0;
    p = fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "r");
    unsigned int i = 0;
    if (p) {
        fscanf(p, "%d", &i);
        fclose(p);
    }
    // i is in KB
    return i * 1024;
}


// Kostis
size_t L2_cache_size (void)
{
    FILE * p = 0;
    p = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
    unsigned int i = 0;
    if (p) {
        fscanf(p, "%d", &i);
        fclose(p);
    }
    // i is in KB
    return i * 1024;
}