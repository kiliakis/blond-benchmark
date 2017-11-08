# This Makefile requires GNU make.

include Defines.mk

.PHONY: default all benches clean

default: all

all: benches

benches:
	@$(foreach BENCH, $(BENCH_DIRS), \
		$(MAKE) -C $(BENCH)/src; \
		$(MAKE) -C $(BENCH);)
 
clean:
	@$(foreach BENCH, $(BENCH_DIRS), \
		$(MAKE) -C $(BENCH) clean --no-print-directory;)
