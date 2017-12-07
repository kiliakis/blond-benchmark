#pragma once

#ifndef FTYPE
#define FTYPE double
#endif

#ifndef HIST_T
#define HIST_T double
#endif

void histogram_v0(const FTYPE * __restrict__ input,
                  HIST_T * __restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,
                  const int n_macroparticles);

void histogram_v1(const FTYPE * __restrict__ input,
                  HIST_T * __restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,

                  const int n_macroparticles);
void histogram_v2(const FTYPE * __restrict__ input,
                  HIST_T * __restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,
                  const int n_macroparticles);

void histogram_v3(const FTYPE * __restrict__ input,
                  HIST_T * __restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,
                  const int n_macroparticles);

void histogram_v4(const FTYPE * __restrict__ input,
                  HIST_T * __restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,
                  const int n_macroparticles);

void histogram_v5(const FTYPE * __restrict__ input,
                  HIST_T * __restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,
                  const int n_macroparticles);


void histogram_v6(const FTYPE *__restrict__ input,
                  HIST_T *__restrict__ output, const FTYPE cut_left,
                  const FTYPE cut_right, const int n_slices,
                  const int n_macroparticles);

void histogram_v9(const FTYPE *__restrict__ input,
                  HIST_T *__restrict__ output, const FTYPE cut_left,
                  const FTYPE cut_right, const int n_slices,
                  const int n_macroparticles);
