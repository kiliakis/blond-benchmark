BENCH_DIRS := interp-kick-n-drift
			  # convolution \
			  # histogram \
			  # fft-convolution \
			  # synchrotron-radiation

CC = g++
CFLAGS = -std=c++11 -g -O3 -Wall -Wno-unknown-pragmas
LDFLAGS = 
INC_DIR = include

AR = ar
RANLIB = ranlib

LINKAGE = static
BASE = base
LIB_BASE = libbase


ifeq ($(LINKAGE), static)
	TARGET_LIB = $(LIB_BASE).a
	LIB_DEP = $(TARGET_LIB)
endif

ifeq ($(LINKAGE), dynamic)
	TARGET_LIB = $(LIB_BASE).so
	LIB_DEP = 
endif

