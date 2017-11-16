HOME = ..

include $(HOME)/Defines.mk

BENCHES := $(wildcard benches/*.cpp)
OBJS := ${BENCHES:.cpp=.o}
PROGS := ${BENCHES:.cpp=}
LDFLAGS += -L$(realpath src) -Wl,-rpath,$(realpath src)
LIBS += -l$(BASE) -lpapiprof
INCDIRS += -Iinclude
CC=icc

.PHONY: default all clean

default: all

all: $(TARGET_LIB) $(PROGS) $(OBJS)
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])

$(TARGET_LIB):
	$(MAKE) -C src --no-print-directory

%: %.o $(TARGET_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCDIRS) -o $@ $< $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCDIRS) -c $< -o $@

clean:
	rm -f $(PROGS) $(OBJS)
	$(MAKE) -C src clean --no-print-directory
