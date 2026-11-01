SHELL := /bin/sh
SRCDIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT   := $(abspath $(SRCDIR)/../../..)
OUT    := $(ROOT)/build/kernel/cuda/mira_dspv1
PYTHON ?= python3

# MIRA CUDA v7, LIKE v2, IS BUILT FOR EVERY CUDA GPU FROM TURING ON, AS FLASHATTENTION SHIPS ONE IMAGE PER
# ARCHITECTURE GENERATION RATHER THAN ONE PER CARD. A cubin of sm_X.y runs on every X.z with z >= y,
# so these are the generations, and the cards they serve:
#
#     75   T4, Quadro RTX 6000                         Turing
#     80   A100, A30                                   Ampere (data centre)
#     86   A10, A40, RTX 3090, RTX A6000; also Orin    Ampere (GA10x)
#     89   L4, L40S, RTX 4090, RTX 6000 Ada            Ada
#     90   H100, H200, GH200                           Hopper
#     100  B200, GB200                                 Blackwell (data centre)
#     101  Jetson Thor under CUDA 12.8 / 12.9          (named 110 from CUDA 13.0)
#     103  B300, GB300                                 Blackwell Ultra
#     110  Jetson Thor under CUDA 13
#     120  RTX 5090, RTX PRO 6000 Blackwell            Blackwell (workstation, consumer)
#     121  DGX Spark (GB10)
#
# plus PTX of the newest of them, which the driver compiles on first load for any later GPU.
# arch.mk keeps only the ones this nvcc can emit, so an older toolkit still builds what it knows.
ARCHS ?= 75 80 86 89 90 100 101 103 110 120 121
PTX   ?= 121

# THE NEWEST TOOLKIT ON THE MACHINE IS THE ONE THAT BUILDS v2, because only a new nvcc emits the
# Blackwell images: of every nvcc found, the one whose highest compute_ number is largest. An NVCC set
# by the caller is obeyed, as arch.mk obeys it.
ifeq ($(origin NVCC),undefined)
_v2_nvcc_candidates := $(strip \
    $(wildcard $(CUDA_HOME)/bin/nvcc) \
    $(wildcard $(CUDA_PATH)/bin/nvcc) \
    $(wildcard /usr/local/cuda-*/bin/nvcc) \
    $(wildcard /usr/local/cuda/bin/nvcc) \
    $(shell command -v nvcc 2>/dev/null))
_v2_nvcc_newest := $(shell for n in $(_v2_nvcc_candidates); do \
    top=$$($$n --list-gpu-arch 2>/dev/null | sed -n 's/^compute_//p' | sort -n | tail -n 1); \
    [ -n "$$top" ] && echo "$$top $$n"; done | sort -n | tail -n 1 | cut -d' ' -f2)
ifneq ($(strip $(_v2_nvcc_newest)),)
NVCC := $(_v2_nvcc_newest)
endif
endif

include $(dir $(lastword $(MAKEFILE_LIST)))../arch.mk
ARCH ?= $(NATIVE_ARCH)

# THE PTX IS THE NEWEST ARCHITECTURE THIS nvcc LISTS, BY NUMBER. arch.mk's fallback takes the last
# of `sort`, which orders as text ("90" after "121"), so v2 picks its own.
V2_PTX := $(if $(filter $(PTX),$(_nvcc_archs)),$(PTX),$(if $(strip $(_nvcc_archs)),$(shell \
    printf '%s\n' $(_nvcc_archs) | sort -n | tail -n 1),$(PTX)))
V2_GENCODE := $(foreach a,$(BUILD_ARCHS),-gencode=arch=compute_$(a)$(,)code=sm_$(a)) \
              -gencode=arch=compute_$(V2_PTX)$(,)code=compute_$(V2_PTX)

# WHAT MAKES THE CUBIN OUT OF DATE, LISTED RATHER THAN ASSUMED. mira_iir.cu includes CUDA's own
# headers -- cuda_bf16.h, cuda_fp16.h, cuda_runtime.h -- and the .cuh files of this directory, so the
# prerequisite set is the .cu, those .cuh files, arch.mk and this Makefile. arch.mk is on the list
# because it decides ARCH and GENCODE: editing it changes which architectures nvcc is asked to
# emit, and a cubin built before that edit no longer answers the question that was asked.
DEPS := $(SRCDIR)/mira_iir.cu $(wildcard $(SRCDIR)/*.cuh) $(abspath $(SRCDIR)/../arch.mk) $(SRCDIR)/Makefile

# TWO WAYS TO FILL ONE OUTPUT PATH, SO CURRENCY IS RECORDED PER MODE IN A STAMP. `build` emits
# -arch=$(ARCH), SASS for one architecture; `universal` emits $(GENCODE), every architecture in
# arch.mk plus PTX. Both write mira_iir.cubin, so the cubin's own timestamp cannot say which
# of the two produced it, and a cubin holding only sm_89 must never let `make universal` report
# it has nothing to do. Each recipe deletes the other mode's stamp, because it has just
# overwritten the file that stamp attested to.
#
# THIS IS WHAT THE STAMPS FIX. `universal` was .PHONY with no prerequisites, so it always re-ran
# nvcc and make could never report "nothing to be done". Every launcher worked around that with
# its own `[ -f mira_iir.cubin ]`, a test that cannot distinguish a cubin compiled from the
# current .cu from one compiled before mira_iir.cu gained new instantiations. That is what
# shipped this, in production:
#
#     RuntimeError: the mira-dsp kernel declined ... K=1:
#     mira_matrix_scan_p1_k1_m16_l4_g128_f32 not found in mira_iir.cubin -- the cubin is stale
#
# `make clean universal` STILL ALWAYS COMPILES, because clean removes the stamps along with the
# cubin, and the stamp is the only record that a compile happened.
.PHONY: build universal check quicktest unittest clean
build: $(OUT)/mira_iir.build.stamp

$(OUT)/mira_iir.build.stamp: $(DEPS)
	@mkdir -p $(OUT)
	$(NVCC) -fatbin -O3 --use_fast_math -arch=$(ARCH) -o $(OUT)/mira_iir.cubin $<
	@rm -f $(OUT)/mira_iir.universal.stamp
	@touch $@

universal: $(OUT)/mira_iir.universal.stamp

$(OUT)/mira_iir.universal.stamp: $(DEPS)
	@mkdir -p $(OUT)
	$(NVCC) -fatbin -O3 --use_fast_math \
	  $(V2_GENCODE) \
	  -o $(OUT)/mira_iir.cubin $(SRCDIR)/mira_iir.cu
	@rm -f $(OUT)/mira_iir.build.stamp
	@touch $@
	@echo "built $(OUT)/mira_iir.cubin: SASS $(foreach a,$(BUILD_ARCHS),sm_$(a)) + PTX compute_$(V2_PTX)"

check:
	@cd $(ROOT) && $(PYTHON) $(SRCDIR)/check.py

quicktest:
	@cd $(ROOT) && $(PYTHON) -c "import importlib,sys; sys.exit(importlib.import_module('kernel.cuda.mira_dspv1.check').smoke())"

unittest: check

clean:
	rm -f $(OUT)/*.cubin $(OUT)/*.stamp
