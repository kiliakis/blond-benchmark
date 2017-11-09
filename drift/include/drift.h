
void drift_v0(double * __restrict__ beam_dt,
              const double * __restrict__ beam_dE,
              const char * __restrict__ solver,
              const double T0, const double length_ratio,
              const double alpha_order, const double eta_zero,
              const double eta_one, const double eta_two,
              const double beta, const double energy,
              const int n_macroparticles);