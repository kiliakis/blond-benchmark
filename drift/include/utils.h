#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>

template <typename T>
void read_distribution(const std::string &file, const int n,
                       std::vector<T> &dtV, std::vector<T> &dEV)
{
    dtV.clear(); dEV.clear();
    std::ifstream source(file);
    if (!source.good()) {
        std::cout << "Error: file " << file << " does not exist\n";
        source.close();
        exit(-1);
    }
    std::string line;
    int i = 0;
    std::getline(source, line);
    for (; std::getline(source, line) && i < n;) {
        std::istringstream in(line);
        T dt, dE;
        in >> dt >> dE;
        dtV.push_back(dt);
        dEV.push_back(dE);
        i++;

    }

    int k = 0;
    while (i < n) {
        dtV.push_back(dtV[k]);
        dEV.push_back(dEV[k]);
        k++; i++;
    }

    source.close();

}

void linspace(const double start, const double end, const int n,
              double *__restrict__ out);

void linspace(const float start, const float end, const int n,
              float *__restrict__ out);


size_t L1_cache_size (void);

size_t L2_cache_size (void);

