HOME = ..

include $(HOME)/Defines.mk

BENCHES := $(wildcard benches/*.cpp)
OBJS := ${BENCHES:.cpp=.o}
PROGS := ${BENCHES:.cpp=}
LIBS += -Lsrc -l$(BASE)
.PHONY: default all clean

default: all

all: $(PROGS) $(LIB) $(OBJS)
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])

$(LIB):
	$(MAKE) -C src

%: %.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INC_DIR)

clean:
	rm $(PROGS) $(OBJS)
	$(MAKE) -C src clean --no-print-directory
