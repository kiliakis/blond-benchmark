
void linear_interp_kick_v0(
    double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    double * __restrict__ voltage_array,
    double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick);

void linear_interp_kick_v4(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick);

void linear_interp_kick_v5(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick);


void linear_interp_kick_v6(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick);