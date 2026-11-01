# kernel/cuda/mamba_s6/Makefile -- build and check OUR cuda kernels. One command, nothing to paste.
#
#     make             build every kernel for THIS GPU  -> build/kernel/cuda/mamba_s6/*.cubin
#     make universal   ONE fatbin each: A100 + L40S + PTX for newer
#     make check       do they reproduce the PyTorch reference, and how much faster?
#     make unittest    do they reproduce the PyTorch reference. Correctness only, no timing.
#     make env         what would happen: modules, GPU, -arch, nvcc, output path
#     make all-arch    one cubin per architecture, for a mixed cluster
#     make clean       remove them
#     make help        this list
#
# THE KERNELS, all ours. This project ships no third-party CUDA in its hot path.
#
#     mamba_s6.cu   the BASELINE this project is measured against, written here so the
#                    comparison depends on no external package. See the file header.
#
# THE ARCHITECTURE IS DETECTED, so nothing has to be typed and no placeholder can be pasted:
# nvidia-smi reports compute_cap as e.g. "8.9", and stripping the dot gives -arch=sm_89.

SHELL := /bin/sh

# FIRST: OUT and the source paths below are immediate (:=) assignments that use these.
#
# BOTH ARE DERIVED FROM THIS FILE'S OWN LOCATION, NOT FROM THE WORKING DIRECTORY, so every
# target works from anywhere:
#
#     cd kernel/cuda/mamba_s6 && make universal
#     make -f kernel/cuda/mamba_s6/Makefile universal          # from the repository root
#     make -C kernel/cuda/mamba_s6 universal                   # equivalent
#
# which matters because stage5_pretrain.sh and other scripts call it without cd-ing first.
SRCDIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT   := $(abspath $(SRCDIR)/../../..)

# =============================================================================================
# MODULES ARE LOADED BY YOU, NOT BY THIS MAKEFILE. The lines below are recorded here only so
# that one can be pasted.
#
#   MeluXina          A100, sm_80
#       module load foss/2025a CUDA/12.8.0
#       module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl        # for `make check`
#
#   Alliance Canada   L40S, sm_89
#       module --force purge
#       module load StdEnv/2023 gcc/12.3 cuda/12.9
#
# LOAD_MODULES IS EMPTY BY DEFAULT, so every target runs in the environment you prepared.
LOAD_MODULES ?=
# =============================================================================================

SRCS    := mamba_s6
NVCC    ?= nvcc
PYTHON  ?= python3
OUT     := $(ROOT)/build/kernel/cuda/mamba_s6
FLAGS   := -cubin -O3 --use_fast_math

# ARCHS -- the architectures `make universal` puts in ONE file: the two clusters this project
# runs on, plus PTX for anything newer, which the driver JIT-compiles on first load.
#     89  L40S      80  A100      70 V100  75 T4  86 A10/3090  90 H100
include $(dir $(lastword $(MAKEFILE_LIST)))../arch.mk


# nvidia-smi needs no modules, so the architecture is known before anything is loaded, but the
# answer is only usable if THIS nvcc can emit it -- see NATIVE_ARCH in ../arch.mk, which reads
# the card and then intersects with the toolkit. `make ARCH=sm_80` still overrides.
ARCH    ?= $(NATIVE_ARCH)

# arch.mk IS A SOURCE OF EVERY CUBIN, not just a setting. It decides ARCH and GENCODE, so
# editing it changes which architectures nvcc is asked to emit and the cubins on disk stop
# answering the question that was asked -- which is why it is a prerequisite of the rules
# below alongside the .cu. Our .cu files include CUDA's own headers and none of this
# repository's, so those two are the whole prerequisite set.
ARCHMK  := $(abspath $(SRCDIR)/../arch.mk)

# `module` is a shell FUNCTION and a recipe gets a fresh non-interactive shell where it may not
# be defined; sourcing Lmod's init first is what makes it exist.
MODINIT := $(firstword $(wildcard $(LMOD_PKG)/init/sh /usr/share/lmod/lmod/init/sh \
                                  /opt/software/lmod/lmod/init/sh /etc/profile.d/lmod.sh))
ifneq ($(strip $(LOAD_MODULES)),)
ENV := { [ -n "$(MODINIT)" ] && . $(MODINIT); } >/dev/null 2>&1; $(LOAD_MODULES) >/dev/null;
else
ENV :=
endif

define need_nvcc
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
	  echo "nvcc not found -- load your CUDA module first (see the top of this Makefile)."; \
	  exit 1; fi
endef

define require_arch
	@if [ -z "$(ARCH)" ]; then \
	  echo "GPU architecture not detected -- nvidia-smi found no device."; \
	  echo "Are you on a login node? Get one, or name it: make ARCH=sm_80"; exit 1; fi
endef

help:
	@echo "make            build all $(words $(SRCS)) cubins for THIS GPU ($(if $(ARCH),$(ARCH),not detected))"
	@echo "make universal  one fatbin each: $(ARCH_SUMMARY)"
	@echo "make check      verify against the PyTorch reference, and time both"
	@echo "make unittest   verify against the PyTorch reference. Correctness only, no timing"
	@echo "make quicktest  the SMALLEST shape, correctness only: the precision error per kernel"
	@echo "                and a pass/fail. No speed table, no timings.json"
	@echo "make env        show modules, GPU, -arch, nvcc, output path"
	@echo "make all-arch   one cubin per architecture"
	@echo "make upstream   build the VENDORED upstream kernel, upstream's own way"
	@echo "make clean      remove them"

env:
	@echo "LOAD_MODULES: $(if $(strip $(LOAD_MODULES)),$(LOAD_MODULES),(none -- modules are yours to load; see the top of this Makefile))"
	@echo "lmod init:    $(if $(MODINIT),$(MODINIT),(none found))"
	@echo
	@echo "GPU:          $$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null \
	                        || echo 'nvidia-smi not found -- login node?')"
	@echo "arch:         $(if $(ARCH),$(ARCH) (detected),not detected -- pass ARCH=sm_89)"
	@echo
	@$(ENV) echo "nvcc:         $$(command -v $(NVCC) 2>/dev/null || echo 'NOT FOUND')"
	@$(ENV) $(NVCC) --version 2>/dev/null | tail -1 || true
	@echo
	@echo "kernels:      $(SRCS)"
	@echo "output:       $(OUT)/"

# TWO WAYS TO FILL ONE OUTPUT PATH, SO CURRENCY IS RECORDED PER MODE IN A STAMP. `build` and
# `universal` both write $(OUT)/<kernel>.cubin -- one with -arch=$(ARCH), SASS for this card
# only, the other with $(GENCODE), every architecture plus PTX -- so the cubin's own timestamp
# cannot say which of the two produced it, and a single-architecture cubin must never let
# `make universal` report it has nothing to do. Each recipe touches its own stamp after nvcc
# succeeds and deletes the other mode's, because it has just overwritten the file that stamp
# attested to. `make clean` removes both stamps, so `make clean universal` always compiles.
#
# THIS IS WHAT THE STAMPS FIX. `universal` had no prerequisite that existed on disk, so it
# always re-ran nvcc and make could never report "nothing to be done". Every launcher worked
# around that with its own `[ -f <kernel>.cubin ]`, a test that cannot distinguish a cubin
# compiled from the current .cu from one compiled before the .cu gained new kernel
# instantiations. In mira-matrix that shipped a RuntimeError naming a kernel absent from the
# cubin; the mechanism was the same in every family, so the same rule is used in all of them.
build: $(SRCS:%=$(OUT)/%.build.stamp)
	@echo "now run: make check"

$(OUT)/%.build.stamp: $(SRCDIR)/%.cu $(ARCHMK)
	$(require_arch)
	@set -e; $(ENV) \
	 if ! command -v $(NVCC) >/dev/null 2>&1; then \
	   echo "nvcc not found -- load your CUDA module first."; exit 1; fi; \
	 mkdir -p $(OUT); \
	 set -x; \
	 $(NVCC) $(FLAGS) -arch=$(ARCH) -o $(OUT)/$*.cubin $<; \
	 set +x; \
	 echo "built $(OUT)/$*.cubin for $(ARCH)"
	@rm -f $(OUT)/$*.universal.stamp
	@touch $@

# ONE FILE PER KERNEL THAT RUNS ON EVERY CARD. A cubin holds SASS for a single architecture and
# cuModuleLoad refuses it on any other (209). A fatbin holds several, plus PTX the driver
# JIT-compiles for an architecture that was not built for.
universal: $(SRCS:%=$(OUT)/%.universal.stamp)
	@echo "now run: make check"

# universal-<kernel> IS KEPT AS A NAME, because it is what `universal` expanded to before the
# stamps existed and it costs one rule to leave that spelling working. It now asks for the
# stamp rather than compiling, so it obeys the same currency test as everything else.
universal-%: $(OUT)/%.universal.stamp
	@:

$(OUT)/%.universal.stamp: $(SRCDIR)/%.cu $(ARCHMK)
	@set -e; $(ENV) \
	 if ! command -v $(NVCC) >/dev/null 2>&1; then \
	   echo "nvcc not found -- load your CUDA module first."; exit 1; fi; \
	 mkdir -p $(OUT); \
	 set -x; \
	 $(NVCC) -fatbin -O3 --use_fast_math \
	   $(GENCODE) \
	   -o $(OUT)/$*.cubin $(SRCDIR)/$*.cu; \
	 set +x; \
	 echo "built $(OUT)/$*.cubin   $(ARCH_SUMMARY)"
	@rm -f $(OUT)/$*.build.stamp
	@touch $@

check:
	@cd $(ROOT) && $(PYTHON) $(SRCDIR)/check.py

# THE CORRECTNESS HALF ON ITS OWN, WHICH IS THE HALF A JOB SCRIPT CAN GATE ON. `make check`
# answers two questions -- do they agree, and how much faster -- and only the first has an
# answer that is a property of the KERNEL. The second is a property of the card and of how busy
# it is, so a job script cannot fail on it.
#
# IT DRIVES check.py, IT DOES NOT REIMPLEMENT IT. main(timing=False) runs the same cases over
# the same SHAPES against the same tolerance and returns the same status -- 0 when the kernel
# agrees in value and in every gradient, 1 when it does not. A second copy of that comparison
# living in a Makefile would be a second thing to keep right, and kernel/cuda/checker.py's
# header is the record of what happens when a check exists in two places.
unittest:
	@cd $(ROOT) && $(PYTHON) -c "import sys; from kernel.cuda.mamba_s6.check import main; sys.exit(main(timing=False))"

# `make quicktest` -- THE QUICK ONE, AND IT IS NOT `unittest` WITH A SHORTER NAME.
#
#     "in trainers, we only do quick cuda test, write make -f Makefile quicktest
#      only show the cuda kernel's precision error comparing to."           -- 2026-08-16
#
# THREE TARGETS NOW, AND EACH ANSWERS A DIFFERENT QUESTION:
#
#   check      every case over all four SHAPES, then the speed table, and it WRITES
#              build/kernel/cuda/mamba_s6/timings.json as the baseline for the next `vs prev`
#   unittest   the same four shapes, correctness only. The whole battery, no clock
#   quicktest  ONE SHAPE -- the smallest of SHAPES, chosen by size so reordering them cannot
#              change which -- correctness only. Seconds, which is what makes it affordable at
#              the head of a script that is about to load the cubin
#
# ONLY THE PRECISION ERROR IS PRINTED, which is the whole of what was asked for: the absolute
# error and the ULP count for the value and for every gradient, then PASS or FAILED. NO SPEED
# TABLE AND NO timings.json -- smoke() passes timing=False, and checker.run() writes the
# baseline file only on the timing half.
#
# IT DRIVES check.py, IT DOES NOT REIMPLEMENT IT. smoke() calls the same main() `unittest`
# calls, which calls the same checker.run() `check` calls, against the same tolerance, and
# returns the same status -- non-zero on any disagreement, which is what a job script gates on.
# A second copy of that comparison living in a Makefile would be a second thing to keep right,
# and kernel/cuda/checker.py's header is the record of what that costs.
quicktest:
	@cd $(ROOT) && $(PYTHON) -c "import sys; from kernel.cuda.mamba_s6.check import smoke; sys.exit(smoke())"

all-arch:
	@set -e; $(ENV) mkdir -p $(OUT); \
	 for f in $(SRCS); do \
	   for a in sm_70 sm_75 sm_80 sm_86 sm_89 sm_90; do \
	     set -x; \
	     $(NVCC) $(FLAGS) -arch=$$a -o $(OUT)/$$f.$$a.cubin $(SRCDIR)/$$f.cu \
	       && { set +x; echo "built $(OUT)/$$f.$$a.cubin"; } \
	       || { set +x; true; }; \
	   done; \
	 done


# =============================================================================================
# THE VENDORED UPSTREAM KERNEL, WHICH IS NOT BUILT BY ANY TARGET ABOVE.
#
# upstream/ holds Tri Dao and Albert Gu's own CUDA, byte for byte, with its LICENSE and a PROVENANCE naming
# the archive it came from. It does NOT build into a cubin and it never will: it is torch C++
# end to end -- at::Tensor in the signature, the grid computed inside their host function, a
# TORCH_LIBRARY registration -- so there is no extern "C" __global__ for cuModuleLoad to find.
# Making one would mean writing a launcher into their file, over their names, and the whole
# reason the source was vendored is that the comparison must not rest on our code.
#
# SO UPSTREAM'S OWN LOADER RUNS: torch.utils.cpp_extension.load, from the module in upstream/
# that is their line for line. That needs nvcc AND a C++ compiler AND torch on the node at RUN
# time, and it happens on the first forward if it has not happened already. This target is how
# a job script pays that compile once, up front, instead of inside the first training step.
#
# NOT AN ERROR WHEN IT CANNOT BE DONE -- it prints why and exits 0, this repository's rule.
# See kernel/cuda/upstream_extension.py for the whole argument.
upstream:
	@cd $(ROOT) && $(ENV) $(PYTHON) -c \
	 "import sys; from kernel.cuda.mamba_s6 import mamba_s6_upstream as k; \
	  k.build(); print('  ' + k.describe()); sys.exit(0)"

clean:
	rm -f $(OUT)/*.cubin $(OUT)/*.stamp

.DEFAULT_GOAL := build
.PHONY: help env build universal check quicktest unittest all-arch upstream clean
