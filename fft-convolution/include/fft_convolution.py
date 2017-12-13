from scipy.signal import fftconvolve
import ctypes
import os



def convolution_v0(signal, kernel):
    return fftconvolve(signal, kernel, mode='full')


def convolution_v1(signal, kernel, result, threads):

    libcpp = ctypes.cdll.LoadLibrary(
        os.path.dirname(os.path.abspath(__file__)) + '/../src/libbase.so')
    libcpp.fft_convolution(signal.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(len(signal)),
                           kernel.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(len(kernel)),
                           result.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(threads))
    # libcpp.rfftfreq(1, signal.ctypes.data_as(ctypes.c_void_p), 1)
