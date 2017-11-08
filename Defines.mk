BENCH_DIRS := convolution \
			  histogram \
			  interp-kick-n-drift \
			  fft-convolution \
			  synchrotron-radiation

CC = g++
CFLAGS = -Wall -std=c++11 -g -O3
LDFLAGS = -fPIC -shared
INC_DIR = include
