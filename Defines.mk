# HOME = \"/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/\"
HOME = \"/home/kiliakis/git/blond-benchmark/\"
INSTALL = /home/kiliakis/install/
# INSTALL = /afs/cern.ch/work/k/kiliakis/install/
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
LDFLAGS = -L$(INSTALL)/lib
INCDIRS = -I$(INSTALL)/include

# CUFLAGS = -std=c++11 -DHOME=$(HOME) -O3 -m64 -restrict -gencode arch=compute_35,code=sm_35
CUFLAGS = -std=c++11 -DHOME=$(HOME) -O3 -m64 -restrict -gencode arch=compute_60,code=sm_60
CUDEBUG = -g -pg -lineinfo -res-usage
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

