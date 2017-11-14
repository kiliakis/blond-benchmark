HOME = ..

include $(HOME)/Defines.mk

BENCHES := $(wildcard benches/*.cpp)
OBJS := ${BENCHES:.cpp=.o}
PROGS := ${BENCHES:.cpp=}
LDFLAGS += -Lsrc -Wl,-rpath,src
LIBS += -l$(BASE) -lpapiprof
INCDIRS += -Iinclude

.PHONY: default all clean

default: all

all: base $(PROGS) $(OBJS)
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])

base:
	$(MAKE) -C src --no-print-directory

%: %.o base
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCDIRS) -o $@ $< $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCDIRS) -c $< -o $@ -I$(INC_DIR)

clean:
	rm -f $(PROGS) $(OBJS)
	$(MAKE) -C src clean --no-print-directory
