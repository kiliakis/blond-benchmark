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

// void read_distribution(const string &file, const int n,
//                        vector<double> &dtV, vector<double> &dEV)
// {
//     dtV.clear(); dEV.clear();
//     ifstream source(file);
//     if (!source.good()) {
//         cout << "Error: file " << file << " does not exist\n";
//         source.close();
//         exit(-1);
//     }
//     string line;
//     int i = 0;
//     getline(source, line);
//     for (; getline(source, line) && i < n;){
//         istringstream in(line);
//         double dt, dE;
//         in >> dt >> dE;
//         dtV.push_back(dt);
//         dEV.push_back(dE);
//         i++;

//     }
    
//     int k = 0;
//     while (i < n) {
//         dtV.push_back(dtV[k]);
//         dEV.push_back(dEV[k]);
//         k++; i++;
//     }

//     source.close();
// }