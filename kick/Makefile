HOME = ..

include $(HOME)/Defines.mk

PROGS_DIR = exe
BENCHES := $(wildcard benches/*.cpp)
OBJS := ${BENCHES:.cpp=.o}
PROGS := ${BENCHES:.cpp=}
LDFLAGS += -L$(realpath src) -Wl,-rpath,$(realpath src)
LIBS += -l$(BASE)
INCDIRS += -Iinclude

.PHONY: default all clean

default: all

all: $(TARGET_LIB) $(PROGS_DIR) $(PROGS) $(OBJS)
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])

$(PROGS_DIR):
	mkdir -p $@

$(TARGET_LIB):
	$(MAKE) -C src --no-print-directory

%: %.o $(TARGET_LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCDIRS) -o $(PROGS_DIR)/$(subst benches/,,$@) $< $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCDIRS) -c $< -o $@

clean:
	rm -rf $(PROGS_DIR) $(OBJS)
	$(MAKE) -C src clean --no-print-directory
