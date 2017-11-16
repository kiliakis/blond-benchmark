
void convolution_v0(const double *__restrict__ signal,
                    const int SignalLen,
                    const double *__restrict__ kernel,
                    const int KernelLen,
                    double *__restrict__ res);

void convolution_v1(const double *__restrict__ signal,
                    const int SignalLen,
                    const double *__restrict__ kernel,
                    const int KernelLen,
                    double *__restrict__ res,
                    const int resLen);

void convolution_v2(const double *__restrict__ signal,
                    const int SignalLen,
                    double *__restrict__ kernel,
                    const int KernelLen,
                    double *__restrict__ res);

void convolution_mkl_v0(const double * __restrict__ signal,
                        const int signalLen,
                        const double * __restrict__ kernel,
                        const int kernelLen,
                        double * __restrict__ result);

void convolution_mkl_v1(const double * __restrict__ signal,
                        const int signalLen,
                        const double * __restrict__ kernel,
                        const int kernelLen,
                        double * __restrict__ result,
                        const int threads = 1);