#include <utility>
#include "fast_resonator.h"


void fast_resonator_v0(
    double *__restrict__ impedanceReal,
    double *__restrict__ impedanceImag,
    const double *__restrict__ frequencies,
    const double *__restrict__ shunt_impedances,
    const double *__restrict__ Q_values,
    const double *__restrict__ resonant_frequencies,
    const int n_resonators,
    const int n_frequencies)

{   /*
    This function takes as an input a list of resonators parameters and
    computes the impedance in an optimised way.

    Parameters
    ----------
    frequencies : float array
        array of frequency in Hz
    shunt_impedances : float array
        array of shunt impedances in Ohm
    Q_values : float array
        array of quality factors
    resonant_frequencies : float array
        array of resonant frequency in Hz
    n_resonators : int
        number of resonantors
    n_frequencies : int
        length of the array 'frequencies'

    Returns
    -------
    impedanceReal : float array
        real part of the impedance
    impedanceImag : float array
        imaginary part of the impedance
      */


    for (int res = 0; res < n_resonators; res++) {
        const double Qsquare = Q_values[res] * Q_values[res];
        const double invResFreq = 1. / resonant_frequencies[res];
        #pragma omp parallel for
        for (int freq = 1; freq < n_frequencies; freq++) {
            const double commonTerm = (frequencies[freq] * invResFreq
                                       - resonant_frequencies[res]
                                       / frequencies[freq]);
            const double commonTerm2 = shunt_impedances[res]
                                       / (1.0 + Qsquare * commonTerm * commonTerm);

            impedanceReal[freq] += commonTerm2;
            impedanceImag[freq] -= Q_values[res] * commonTerm * commonTerm2;
        }
    }
}

void fast_resonator_v1(
    double *__restrict__ impedanceReal,
    double *__restrict__ impedanceImag,
    const double *__restrict__ frequencies,
    const double *__restrict__ shunt_impedances,
    const double *__restrict__ Q_values,
    const double *__restrict__ resonant_frequencies,
    const int n_resonators,
    const int n_frequencies)

{   /*
    This function takes as an input a list of resonators parameters and
    computes the impedance in an optimised way.

    Parameters
    ----------
    frequencies : float array
        array of frequency in Hz
    shunt_impedances : float array
        array of shunt impedances in Ohm
    Q_values : float array
        array of quality factors
    resonant_frequencies : float array
        array of resonant frequency in Hz
    n_resonators : int
        number of resonantors
    n_frequencies : int
        length of the array 'frequencies'

    Returns
    -------
    impedanceReal : float array
        real part of the impedance
    impedanceImag : float array
        imaginary part of the impedance
      */


    for (int res = 0; res < n_resonators; res++) {
        const double Qsquare = Q_values[res] * Q_values[res];
        #pragma omp parallel for
        for (int freq = 1; freq < n_frequencies; freq++) {
            const double commonTerm = (frequencies[freq]
                                       / resonant_frequencies[res]
                                       - resonant_frequencies[res]
                                       / frequencies[freq]);
            const double commonTerm2 = shunt_impedances[res]
                                       / (1.0 + Qsquare * commonTerm * commonTerm);

            impedanceReal[freq] += commonTerm2;
            impedanceImag[freq] -= Q_values[res] * commonTerm * commonTerm2;
        }
    }
}

void fast_resonator_v2(
    double *__restrict__ impedanceReal,
    double *__restrict__ impedanceImag,
    const double *__restrict__ frequencies,
    const double *__restrict__ shunt_impedances,
    const double *__restrict__ Q_values,
    const double *__restrict__ resonant_frequencies,
    const int n_resonators,
    const int n_frequencies)

{   /*
    This function takes as an input a list of resonators parameters and
    computes the impedance in an optimised way.

    Parameters
    ----------
    frequencies : float array
        array of frequency in Hz
    shunt_impedances : float array
        array of shunt impedances in Ohm
    Q_values : float array
        array of quality factors
    resonant_frequencies : float array
        array of resonant frequency in Hz
    n_resonators : int
        number of resonantors
    n_frequencies : int
        length of the array 'frequencies'

    Returns
    -------
    impedanceReal : float array
        real part of the impedance
    impedanceImag : float array
        imaginary part of the impedance
      */


    for (int res = 0; res < n_resonators; res++) {
        // const double Qsquare = Q_values[res] * Q_values[res];
        #pragma omp parallel for
        for (int freq = 1; freq < n_frequencies; freq++) {
            const double commonTerm = (frequencies[freq]
                                       / resonant_frequencies[res]
                                       - resonant_frequencies[res]
                                       / frequencies[freq]);
            // const double commonTerm2 = shunt_impedances[res]
            //                            / (1.0 + Q_values[res] * Q_values[res] * commonTerm * commonTerm);

            impedanceReal[freq] += shunt_impedances[res]
                                   / (1.0 + Q_values[res] * Q_values[res] * commonTerm * commonTerm);
            impedanceImag[freq] -= Q_values[res] * commonTerm * shunt_impedances[res]
                                   / (1.0 + Q_values[res] * Q_values[res] * commonTerm * commonTerm);
        }
    }
}
