# kernel/cuda/rwkv_v8/Makefile -- build and check RWKV-8's kernel. THE KERNEL IS NOT OURS.
#
#     make             build upstream's extension for THIS node (an alias for `upstream`)
#     make upstream    the same thing, named for what it is
#     make check       does it reproduce the recurrence, and how much faster?
#     make unittest    does it reproduce the recurrence. Correctness only, no timing
#     make env         what would happen: python, nvcc, GPU, where the build lands
#     make clean       remove the built extension
#     make help        this list
#
# THIS FAMILY HAS NO CUBIN AND NO .cu OF OURS, which is why this Makefile is shorter than its
# four siblings and why `universal`, `all-arch` and `ARCH` are absent rather than stubbed.
# The other families build with
#
#     nvcc -cubin -arch=sm_XX  ->  build/kernel/cuda/<family>/*.cubin  ->  cuModuleLoad
#
# ahead of time, and that is the right shape for a kernel written here: no host compiler on
# the compute node, no PyTorch ABI, no rebuild when torch moves. RWKV-8's kernel is BlinkDL's,
# vendored unmodified at upstream/cuda/, and it is torch C++ end to end -- at::Tensor in the
# signature, TORCH_LIBRARY registration, the grid computed inside their host function. There
# is no extern "C" __global__ for cuModuleLoad to find. Building it as a cubin would mean
# writing a launcher into their file, over their names, and the comparison would rest on our
# code again -- which is the exact thing vendoring was done to stop.
#
# SO UPSTREAM'S OWN LOADER RUNS: torch.utils.cpp_extension.load, from
# kernel/cuda/rwkv_v8/upstream/op_x070_fp32.py, which is their line for line. That needs nvcc
# AND a C++ compiler AND torch on the node at RUN time, and pays a one-off compile of a minute
# or two. `make upstream` is how a job script pays it on a login node instead of on the first
# training step. See kernel/cuda/upstream_extension.py for the whole argument.
#
# A MISSING KERNEL IS NOT AN ERROR, which is this repository's rule and is why `unittest` here
# exits 0 when nvcc is absent: check.py reports and returns 0 rather than failing a stage that
# a login node was never going to be able to run.

SHELL := /bin/sh

# DERIVED FROM THIS FILE'S OWN LOCATION, NOT THE WORKING DIRECTORY, so every target works from
# anywhere -- the same rule the other four family Makefiles follow, and for the same reason:
# the scripts that call them do not cd first.
SRCDIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT   := $(abspath $(SRCDIR)/../../..)

# MODULES ARE LOADED BY YOU, NOT BY THIS MAKEFILE -- identical to the other families. Empty by
# default so every target runs in the environment you prepared.
LOAD_MODULES ?=

NVCC    ?= nvcc
PYTHON  ?= python3

MODINIT := $(firstword $(wildcard $(LMOD_PKG)/init/sh /usr/share/lmod/lmod/init/sh \
                                  /opt/software/lmod/lmod/init/sh /etc/profile.d/lmod.sh))
ifneq ($(strip $(LOAD_MODULES)),)
ENV := { [ -n "$(MODINIT)" ] && . $(MODINIT); } >/dev/null 2>&1; $(LOAD_MODULES) >/dev/null;
else
ENV :=
endif

help:
	@echo "make            build upstream's RWKV-8 extension for this node"
	@echo "make upstream   the same, named for what it is"
	@echo "make check      verify against the reference recurrence, and time both"
	@echo "make unittest   verify against the reference recurrence. Correctness only"
	@echo "make env        show python, nvcc, GPU and where the build lands"
	@echo "make clean      remove the built extension"
	@echo
	@echo "There is no cubin here: RWKV-8's kernel is upstream's and builds upstream's way."

env:
	@echo "LOAD_MODULES: $(if $(strip $(LOAD_MODULES)),$(LOAD_MODULES),(none -- modules are yours to load))"
	@echo "lmod init:    $(if $(MODINIT),$(MODINIT),(none found))"
	@echo
	@echo "GPU:          $$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null \
	                        || echo 'nvidia-smi not found -- login node?')"
	@$(ENV) echo "nvcc:         $$(command -v $(NVCC) 2>/dev/null || echo 'NOT FOUND')"
	@$(ENV) echo "python:       $$(command -v $(PYTHON) 2>/dev/null || echo 'NOT FOUND')"
	@echo
	@echo "sources:      $(SRCDIR)/upstream/cuda/wkv7_cuda.cu, wkv7_op.cpp   (BlinkDL's)"
	@echo "loader:       $(SRCDIR)/upstream/op_x070_fp32.py                  (BlinkDL's)"
	@echo "output:       $(ROOT)/build/kernel/cuda/rwkv_v8/upstream/"
	@echo "              libwind_backstepping_fp32.so, built by nvcc, inside the project"

# BUILT BY IMPORTING UPSTREAM'S MODULE, because that import IS the build -- their load() call
# is at their module scope. Nothing is written into build/kernel/cuda/: torch owns the output
# directory for an extension it compiled, and pretending otherwise would mean two places to
# clean.
upstream:
	@cd $(ROOT) && $(ENV) $(PYTHON) -c \
	 "import sys; from kernel.cuda.rwkv_v8 import rwkv_v8_upstream as k; \
	  m = k.build(); print('  ' + k.describe()); sys.exit(0)"

build: upstream

check:
	@cd $(ROOT) && $(ENV) $(PYTHON) $(SRCDIR)/check.py

# THE CORRECTNESS HALF ON ITS OWN, which is the half a job script can gate on -- see
# kernel/cuda/rwkv_v7/Makefile for the argument. It drives check.py, it does not reimplement
# it: main(timing=False) runs the same cases over the same shapes against the same tolerance
# and returns the same status.
unittest:
	@cd $(ROOT) && $(ENV) $(PYTHON) -c \
	 "import sys; from kernel.cuda.rwkv_v8.check import main; sys.exit(main(timing=False))"

# TORCH OWNS THE BUILD DIRECTORY, so this removes the one subdirectory it made for this
# extension and nothing else. It is deliberately not `rm -rf` on the whole cache: other
# projects share it.
clean:
	rm -f $(ROOT)/build/kernel/cuda/rwkv_v8/upstream/libwind_backstepping_fp32.so \
	      $(ROOT)/build/kernel/cuda/rwkv_v8/upstream/libwind_backstepping_fp32.so.built-with

.DEFAULT_GOAL := build
.PHONY: help env upstream build check unittest clean
