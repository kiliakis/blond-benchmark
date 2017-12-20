#include "cuda_utils.h"
#include <cufft.h>
#include <cuda.h>
#include <stdio.h>
#include <thrust/device_vector.h>

struct complexMultiplier
{
    double scale;
    complexMultiplier(double scale): scale(scale) {};

    __host__ __device__
    cufftDoubleComplex operator() (const cufftDoubleComplex &v1,
                                   const cufftDoubleComplex &v2) const
    {
        cufftDoubleComplex res;
        res.x = (v1.x * v2.x - v1.y * v2.y) * scale;
        res.y = (v1.x * v2.y + v1.y * v2.x) * scale;
        return res;
    }
};

/**
    Creates an FFT Plan if it has not been yet initialized

    @plan: Pointer to the plan that will be created/initialized
    @size: Size of the FFT for which this plan will be used
    @type: Type of the FFT
    @batch: Number of FFTs of the specified size that will be computed together.

*/
void create_plan(cufftHandle *plan, size_t size, cufftType type, int batch = 1)
{
    size_t workSize;
    int ret = cufftGetSize(*plan, &workSize);
    if (ret == CUFFT_INVALID_PLAN) {
        if (cufftPlan1d(plan, size, type, batch) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed");
        }
    }
}


/**
    Computes the FFT convolution of two complex signals

    @signal: The first signal. This is a pointer to host(CPU) memory
    @signalSize: The signal size
    @kernel: The second signal, also called kernel. This is a pointer to
             host(CPU) memory
    @kernelSize: The kernel size
    @result: Pointer to host(CPU) memory that contains the convolution result.
             Sufficient memory ((singalSize + kernelSize -1) * sizeof(cufftDoubleComplex))
             has to be allocated before calling the function.
    @d_in: Pointer to GPU memory used by the function. The size of the memory region
            has to be at least 2 * (signalSize + kernelSize - 1)
    @fwplan: An integer handle used to store the forward FFT plan.
    @bwplan: An integer handle used to store the backward FFT plan.
*/
void convolve_complex(cufftDoubleComplex * signal, int signalSize,
                      cufftDoubleComplex * kernel, int kernelSize,
                      cufftDoubleComplex * result,
                      cufftDoubleComplex * d_in,
                      cufftHandle *fwplan,
                      cufftHandle *bwplan)
{


    // timer timer, globalTimer;
    // globalTimer.restart();
    size_t real_size = signalSize + kernelSize - 1;

    // timer.restart();
    cudaMemset(d_in, 0, 2 * real_size * sizeof(cufftDoubleComplex));
    // timerMap["memset"].push_back(timer.elapsed());

    // timer.restart();
    cudaMemcpy(d_in, signal, signalSize * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in + real_size, kernel, kernelSize * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    // timerMap["memcpy"].push_back(timer.elapsed());

    // timer.restart();
    create_plan(fwplan, real_size, CUFFT_Z2Z, 2);
    create_plan(bwplan, real_size, CUFFT_Z2Z, 1);
    // timerMap["create_plans"].push_back(timer.elapsed());

    // timer.restart();
    cufftExecZ2Z(*fwplan, d_in, d_in, CUFFT_FORWARD);
    // timerMap["forward"].push_back(timer.elapsed());

    // timer.restart();
    thrust::device_ptr<cufftDoubleComplex> a(d_in);
    thrust::transform(a, a + real_size, a + real_size, a,
                      complexMultiplier(1.0 / real_size));
    // timerMap["multiply"].push_back(timer.elapsed());

    // timer.restart();
    cufftExecZ2Z(*bwplan, d_in, d_in, CUFFT_INVERSE);
    // timerMap["backward"].push_back(timer.elapsed());

    // timer.restart();
    cudaMemcpy(result, d_in, real_size * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    // timerMap["copy_back"].push_back(timer.elapsed());
    // timerMap["total_time"].push_back(globalTimer.elapsed());
}

/**
    Computes the FFT convolution of two real signals

    @signal: The first signal. This is a pointer to host(CPU) memory
    @signalSize: The signal size
    @kernel: The second signal, also called kernel. This is a pointer to
             host(CPU) memory
    @kernelSize: The kernel size
    @result: Pointer to host(CPU) memory where the convolution result will be copied.
             Sufficient memory ((signalSize + kernelSize - 1)*sizeof(double))
             has to be allocated before calling the function.
    @fwplan: An integer handle used to store the forward FFT plan.
    @bwplan: An integer handle used to store the backward FFT plan.
*/
void convolve_real(double * signal, int signalSize,
                   double * kernel, int kernelSize,
                   double * result,
                   cufftHandle *fwplan,
                   cufftHandle *bwplan)
{
    cufftDoubleComplex *d_out;
    double *d_in;

    size_t real_size = signalSize + kernelSize - 1;
    size_t complex_size = real_size / 2 + 1;

    cudaMalloc((void**)&d_in, 2 * real_size * sizeof(double));
    cudaMalloc((void**)&d_out, 2 * complex_size * sizeof(cufftDoubleComplex));

    cudaMemset(d_in, 0, 2 * real_size * sizeof(double));
    cudaMemcpy(d_in, signal, signalSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in + real_size, kernel, kernelSize * sizeof(double), cudaMemcpyHostToDevice);

    create_plan(fwplan, real_size, CUFFT_D2Z, 2);
    create_plan(bwplan, real_size, CUFFT_Z2D);

    cufftExecD2Z(*fwplan, d_in, d_out);

    thrust::device_ptr<cufftDoubleComplex> a(d_out);
    thrust::transform(a, a + complex_size, a + complex_size, a,
                      complexMultiplier(1.0 / real_size));

    cufftExecZ2D(*bwplan, d_out, d_in);

    cudaMemcpy(result, d_in, real_size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}


/**
    Computes the FFT convolution of two real signals

    @signal: The first signal. This is a pointer to GPU memory
    @signalSize: The signal size
    @kernel: The second signal, also called kernel. This is a pointer to
             GPU memory
    @kernelSize: The kernel size
    @result: Pointer to GPU memory where the convolution result will be copied.
             Sufficient memory ((signalSize + kernelSize - 1)*sizeof(double))
             has to be allocated before calling the function.
    @fwplan: An integer handle used to store the forward FFT plan.
    @bwplan: An integer handle used to store the backward FFT plan.
*/
void convolve_real_no_memcpy_v0(double * signal, int signalSize,
                                double * kernel, int kernelSize,
                                double * result)
{
    // auto globalTimer = get_time();
    // timer localTimer;
    // timer globalTimer;
    // double elapsed;

    // globalTimer.restart();
    cufftHandle fwplan;
    cufftHandle bwplan;
    // printf("I am in\n");
    cufftDoubleComplex *d_out;
    size_t real_size = signalSize + kernelSize - 1;
    size_t complex_size = real_size / 2 + 1;

    cudaMalloc((void**)&d_out, 2 * complex_size * sizeof(cufftDoubleComplex));
    // printf("After the allocation\n");

    // localTimer.restart();
    // auto localTimer = get_time();
    if (cufftPlan1d(&fwplan, real_size, CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
        // fprintf(stderr, "CUFFT error: Plan creation failed");
    }
    if (cufftPlan1d(&bwplan, real_size, CUFFT_Z2D, 1) != CUFFT_SUCCESS) {
        // fprintf(stderr, "CUFFT error: Plan creation failed");
    }
    // print_time_elapsed("Plan creation", localTimer);
    // printf("Plan creation: %.2lf\n", localTimer.elapsed());
    // create_plan(fwplan, real_size, CUFFT_D2Z);
    // create_plan(bwplan, real_size, CUFFT_Z2D);
    // double* temp = new double[real_size];

    // printf("outS\n");
    // cudaMemcpy(temp, singal, signalSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < complex_size; ++i)
    // {
    // printf("(%f, %f)\n", temp[i].x, temp[i].y);
    // }

    // printf("outK\n");
    // cudaMemcpy(temp, d_outK, complex_size * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < complex_size; ++i)
    // {
    // printf("(%f, %f)\n", temp[i].x, temp[i].y);
    // }
    // printf("After the plan creation\n");

    cufftExecD2Z(fwplan, signal, d_out);
    cufftExecD2Z(fwplan, kernel, d_out + complex_size);

    // printf("After the fwd ffts\n");

    thrust::device_ptr<cufftDoubleComplex> a(d_out);
    thrust::transform(a, a + complex_size, a + complex_size, a,
                      complexMultiplier(1.0 / real_size));

    // printf("After the multiplication\n");

    cufftExecZ2D(bwplan, d_out, result);
    cufftDestroy(fwplan);
    cufftDestroy(bwplan);
    // printf("After the bwd ffts\n");

    cudaFree(d_out);
    // print_time_elapsed("Total time", globalTimer);
    // printf("Total elapsed time: %.2lf\n", globalTimer.elapsed());
}


void convolve_real_no_memcpy_v1(double * signal, int signalSize,
                                double * kernel, int kernelSize,
                                double * result,
                                cufftHandle fwplan,
                                cufftHandle bwplan)
{
    // auto globalTimer = get_time();
    // timer localTimer;
    // timer globalTimer;
    // double elapsed;

    // globalTimer.restart();


    cufftDoubleComplex *d_out;
    size_t real_size = signalSize + kernelSize - 1;
    size_t complex_size = real_size / 2 + 1;

    // localTimer.restart();
    cudaMalloc((void**)&d_out, 2 * complex_size * sizeof(cufftDoubleComplex));
    // printf("After the allocation\n");
    // printf("Alloc time: %.2lf\n", localTimer.elapsed());

    cufftExecD2Z(fwplan, signal, d_out);
    cufftExecD2Z(fwplan, kernel, d_out + complex_size);

    // printf("After the fwd ffts\n");

    thrust::device_ptr<cufftDoubleComplex> a(d_out);
    thrust::transform(a, a + complex_size, a + complex_size, a,
                      complexMultiplier(1.0 / real_size));

    // printf("After the multiplication\n");

    cufftExecZ2D(bwplan, d_out, result);
    // cufftDestroy(fwplan);
    // cufftDestroy(bwplan);
    // printf("After the bwd ffts\n");

    cudaFree(d_out);
    // print_time_elapsed("Total time", globalTimer);
    // printf("Total elapsed time: %.2lf\n", globalTimer.elapsed());
}

void convolve_real_no_memcpy_v2(double * signal, int signalSize,
                                double * kernel, int kernelSize,
                                double * result,
                                cufftDoubleComplex * d_out,
                                cufftHandle fwplan,
                                cufftHandle bwplan)
{
    // timer localTimer;
    // timer globalTimer;
    // globalTimer.restart();

    size_t real_size = signalSize + kernelSize - 1;
    size_t complex_size = real_size / 2 + 1;

    cufftExecD2Z(fwplan, signal, d_out);
    cufftExecD2Z(fwplan, kernel, d_out + complex_size);

    thrust::device_ptr<cufftDoubleComplex> a(d_out);
    thrust::transform(a, a + complex_size, a + complex_size, a,
                      complexMultiplier(1.0 / real_size));

    cufftExecZ2D(bwplan, d_out, result);
    // printf("Total elapsed time: %.2lf\n", globalTimer.elapsed());
}