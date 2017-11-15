# This Makefile requires GNU make.

include Defines.mk

.PHONY: default all benches clean

default: all

all: benches

benches:
	@$(foreach BENCH, $(BENCH_DIRS), \
		$(MAKE) -C $(BENCH) --no-print-directory;)
 
clean:
	@$(foreach BENCH, $(BENCH_DIRS), \
		$(MAKE) -C $(BENCH) clean --no-print-directory;)
