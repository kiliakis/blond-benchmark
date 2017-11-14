
extern "C" void synchrotron_radiation_full(double * __restrict__ beam_dE,
    const double U0,
    const int n_macroparticles, const double sigma_dE,
    const double tau_z, const double energy,
    const int n_kicks);