HOME = ..

include $(HOME)/Defines.mk

PROGS_DIR = exe

BENCHES := $(wildcard benches/*.cpp)
OBJS := ${BENCHES:.cpp=.o}
PROGS := ${BENCHES:.cpp=.exe}
LDFLAGS += -L$(realpath src) -Wl,-rpath,$(realpath src)
LIBS += -l$(BASE)
INCDIRS += -Iinclude


CUPROGS_DIR = exe_cuda
CUBENCHES := $(wildcard benches/*.cu)
CUPROGS := ${CUBENCHES:.cu=.cu.exe}


.PHONY: default all clean

default: all

cuda: $(CUPROGS_DIR) $(CUPROGS) 
all: $(TARGET_LIB) $(PROGS_DIR) $(PROGS) $(OBJS)
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])


%.cu.exe: %.cu
	$(CUCC) $(CUFLAGS) $(CULDFLAGS) $(CUDEBUG) $(INCDIRS) -o $(CUPROGS_DIR)/$(subst benches/,,$@) $< $(CULIBS)

$(CUPROGS_DIR):
	mkdir -p $@

$(PROGS_DIR):
	mkdir -p $@

$(TARGET_LIB):
	$(MAKE) -C src --no-print-directory

%.exe: %.o $(TARGET_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCDIRS) -o $(PROGS_DIR)/$(subst benches/,,$@) $< $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCDIRS) -c $< -o $@

clean:
	rm -rf exe* $(OBJS)
	$(MAKE) -C src clean --no-print-directory
