HOME = ../..

include $(HOME)/Defines.mk

SRCS := $(wildcard *.cpp)
OBJS := ${SRCS:.cpp=.o}

INCDIRS += -I../include
# MKL_LIBS = -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_rt

ifeq ($(LINKAGE), dynamic)
CFLAGS += -fPIC
LDFLAGS += -shared
endif

# CC=icc

.PHONY: default all clean

default: all

all: $(TARGET_LIB)
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])

$(LIB_BASE).a: $(OBJS)
	$(AR) cr $@ $(OBJS)
	$(RANLIB) $@

$(LIB_BASE).so: $(OBJS)
	$(CC) $(LDFLAGS) $(INCDIRS) -o $@ $(OBJS) $(MKL_LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCDIRS) -c $< -o $@

clean:
	rm -f $(TARGET_LIB) $(OBJS)
