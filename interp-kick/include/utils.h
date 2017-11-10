
#include <string>
#include <vector>


void read_distribution(const std::string &file, const int n,
                       std::vector<double> &dtV, std::vector<double> &dEV);

void linspace(const double start, const double end, const int n,
              double *__restrict__ out);