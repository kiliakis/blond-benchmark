HOME = ..

include $(HOME)/Defines.mk

BENCHES := $(wildcard benches/*.cpp)
OBJS := ${BENCHES:.cpp=.o}
PROGS := ${BENCHES:.cpp=.exe}
LDFLAGS += -L$(realpath src) -Wl,-rpath,$(realpath src) -L/cvmfs/projects.cern.ch/intelsw/psxe/linux/x86_64/2018/vtune_amplifier/lib64
# LIBS += -l$(BASE) -Wl,-Bstatic -littnotify
LIBS += -l$(BASE)
# INCDIRS += -Iinclude -I/cvmfs/projects.cern.ch/intelsw/psxe/linux/x86_64/2018/vtune_amplifier/include/
INCDIRS += -Iinclude
PROGS_DIR = exe

# CC=icc
CUPROGS_DIR = exe_cuda
CUBENCHES := $(wildcard benches/*.cu)
CUPROGS := ${CUBENCHES:.cu=.cu.exe}
# CULDFLAGS += -L$(realpath src)
CULIBS += src/utils.cpp

.PHONY: default all clean

default: all

cuda: $(CUPROGS_DIR) $(CUPROGS) 
all: $(TARGET_LIB) $(PROGS_DIR) $(PROGS) $(OBJS)
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])

$(PROGS_DIR):
	mkdir -p $@

$(TARGET_LIB):
	$(MAKE) -C src --no-print-directory

%.cu.exe: %.cu
	$(CUCC) $(CUFLAGS) $(CULDFLAGS) $(CUDEBUG) $(INCDIRS) -o $(CUPROGS_DIR)/$(subst benches/,,$@) $< $(CULIBS)

$(CUPROGS_DIR):
	mkdir -p $@

%.exe: %.o $(TARGET_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCDIRS) -o $(PROGS_DIR)/$(subst benches/,,$@) $< $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCDIRS) -c $< -o $@

clean:
	rm -rf exe* $(OBJS)
	$(MAKE) -C src clean --no-print-directory
