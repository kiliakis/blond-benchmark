HOME = ../..

include $(HOME)/Defines.mk

SRCS := $(wildcard *.cpp)
OBJS := ${SRCS:.cpp=.o}

INCDIRS += -I../include
LIBS = -lm
# CC = icc
ifeq ($(LINKAGE), dynamic)
CFLAGS += -fPIC
LDFLAGS += -shared
LIB_BASE = libbase
TARGET_LIB = libbase.so
endif

.PHONY: default all clean

default: all

all: $(TARGET_LIB)
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])

$(LIB_BASE).a: $(OBJS) 
	$(AR) cr $@ $(OBJS)
	$(RANLIB) $@

$(LIB_BASE).so: $(OBJS) 
	$(CC) $(LDFLAGS) $(INCDIRS) -o $@ $(OBJS) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCDIRS) -c $< -o $@

clean:
	rm -f $(TARGET_LIB) $(OBJS)
