

void histogram_v0(const double * __restrict__ input,
                  double * __restrict__ output,
                  const double cut_left,
                  const double cut_right,
                  const int n_slices,
                  const int n_macroparticles);


void histogram_v6(const double *__restrict__ input,
                  double *__restrict__ output, const double cut_left,
                  const double cut_right, const int n_slices,
                  const int n_macroparticles);

void histogram_v9(const double *__restrict__ input,
                  double *__restrict__ output, const double cut_left,
                  const double cut_right, const int n_slices,
                  const int n_macroparticles);