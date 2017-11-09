
void kick_v0(const double * __restrict__ beam_dt,
             double * __restrict__ beam_dE, const int n_rf,
             const double * __restrict__ voltage,
             const double * __restrict__ omega_RF,
             const double * __restrict__ phi_RF,
             const int n_macroparticles,
             const double acc_kick);