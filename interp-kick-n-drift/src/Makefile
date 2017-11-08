HOME = ../..

include $(HOME)/Defines.mk

SRCS := $(wildcard *.cpp)
OBJS := ${SRCS:.cpp=.o}

ifeq ($(LINKAGE), dynamic)
CFLAGS += -fPIC
endif

.PHONY: default all clean

default: all

all: $(TARGET_LIB)
# $(info $$OBJS is [${OBJS}]) $(info $$PROGS is [${PROGS}])

$(LIB_BASE).a: $(OBJS) 
	$(AR) cr $@ $(OBJS)
	$(RANLIB) $@

$(LIB_BASE).so: $(OBJS) 
	$(CC) --shared -o $@ $(OBJS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ -I../$(INC_DIR)

clean:
	rm $(TARGET_LIB) $(OBJS)
