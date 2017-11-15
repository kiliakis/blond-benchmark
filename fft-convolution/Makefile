HOME = ..

include $(HOME)/Defines.mk

BENCHES := $(wildcard benches/*.cpp)
OBJS := ${BENCHES:.cpp=.o}
PROGS := ${BENCHES:.cpp=}
LDFLAGS += -Lsrc -Wl,-rpath,src
LIBS += -l$(BASE) -lpapiprof
INCDIRS += -Iinclude
MKL_LIBS = -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_rt
CC=icc
# ifeq ($(LINKAGE), dynamic)
# CFLAGS += -fPIC
# endif

.PHONY: default all clean

default: all

all: $(TARGET_LIB) $(PROGS) $(OBJS)
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])

$(TARGET_LIB):
	$(MAKE) -C src --no-print-directory

benches/bench4: benches/bench4.o $(TARGET_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCDIRS) -o $@ $< $(LIBS) $(MKL_LIBS)

%: %.o $(TARGET_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCDIRS) -o $@ $< $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCDIRS) -c $< -o $@

clean:
	rm -f $(PROGS) $(OBJS)
	$(MAKE) -C src clean --no-print-directory
