# This Makefile requires GNU make.

include Defines.mk

.PHONY: default all benches clean

default: all

all: benches

benches:
	@$(foreach BENCH, $(BENCH_DIRS), \
		$(MAKE) -C $(BENCH);)
 
clean:
	@$(foreach BENCH, $(BENCH_DIRS), \
		$(MAKE) -C $(BENCH) clean;)
