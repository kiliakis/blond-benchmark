HOME = \"/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/\"

BENCH_DIRS := interp-kick \
			  convolution \
			  histogram \
			  kick \
			  drift \
			  fft-convolution \
			  synchrotron-radiation

#CC = g++
CUCC = nvcc
CC = icc
ifeq ($(NOVEC),1)
	ifeq ($(CC),icc)
		OPTFLAGS = -O2 -no-vec
	else 
		OPTFLAGS = -O2 -fno-tree-vectorize
	endif
else
	OPTFLAGS = -Ofast -march=native
endif

ifeq ($(TCM),1)
	LIBS += -ltcmalloc
	CFLAGS += -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
endif


CFLAGS = -std=c++11 -g -fopenmp -DHOME=$(HOME) $(OPTFLAGS)
LDFLAGS = -L/afs/cern.ch/work/k/kiliakis/install/lib
INCDIRS = -I/afs/cern.ch/work/k/kiliakis/install/include

CUFLAGS = -std=c++11 -DHOME=$(HOME) -O3 -m64 -restrict -res-usage -gencode arch=compute_35,code=sm_35
CUDEBUG = -g -pg -lineinfo
CULDFLAGS = 
CULIBS = 

AR = ar
RANLIB = ranlib

LINKAGE = dynamic
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

