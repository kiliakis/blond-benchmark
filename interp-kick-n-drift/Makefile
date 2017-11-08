# PROJECT_DIR=$(shell pwd)
# CC=g++
# CFLAGS=-Wall -std=c++11 -g -O3
# LDFLAGS=-fPIC -shared
# INCLUDE=-I$(DIR)/include
# LIB_SOURCE=$(DIR)/src/PAPIProf.cpp $(DIR)/src/PAPIProfMultiThread.cpp
# LIB=$(DIR)/lib/libpapiprof.so
# EXAMPLES=$(wildcard $(DIR)/examples/*.cpp)
# EXECUTABLES=$(patsubst %.cpp, %, $(EXAMPLES))

# all: $(LIB) $(EXECUTABLES)

# %: %.cpp
# 	$(CC) $(CFLAGS) $(INCLUDE) -L$(DIR)/lib -o $@ $^ -lpapiprof

# $(LIB): $(LIB_SOURCE)
# 	$(CC) $(CFLAGS) $(INCLUDE) $(LDFLAGS) $^ -o $@ -lpapi
HOME = ..

include $(HOME)/Defines.mk

BENCHES := $(wildcard src/*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(BENCHES))
PROGS := $(patsubst %.cpp, %, $(BENCHES))

.PHONY: default all clean

default: all

all:  $(PROGS) $(OBJS) 
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])

%: %.o
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INC_DIR)

clean:
	rm -f $(PROGS) $(OBJS)
