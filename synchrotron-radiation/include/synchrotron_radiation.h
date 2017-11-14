

extern "C" void synchrotron_radiation(double * __restrict__ beam_dE,
                                      const double U0,
                                      const int n_macroparticles,
                                      const double tau_z,
                                      const int n_kicks);

extern "C" void synchrotron_radiation_full_v0(double * __restrict__ beam_dE,
    const double U0,
    const int n_macroparticles, const double sigma_dE,
    const double tau_z, const double energy,
    double * __restrict__ random_array,
    const int n_kicks);

extern "C" void synchrotron_radiation_full_v1(double * __restrict__ beam_dE,
    const double U0,
    const int n_macroparticles, const double sigma_dE,
    const double tau_z, const double energy,
    double * __restrict__ random_array,
    const int n_kicks)