# Which -gencode targets `make universal` emits, decided by THIS toolkit rather than by a list.
#
#     include $(dir $(lastword $(MAKEFILE_LIST)))../arch.mk
#
# Every kernel family includes this and uses $(GENCODE); none of them names an architecture.
#
# THE TOOLKIT ANSWERS, BECAUSE A FIXED LIST IS WRONG SOMEWHERE. compute_89 exists only from CUDA
# 11.8 and compute_90 only from 12.0, so naming them unconditionally makes
#
#     nvcc fatal : Unsupported gpu architecture 'compute_89'
#
# the first thing an older toolkit prints, and the build stops before a single kernel compiles.
# `nvcc --list-gpu-arch` is the toolkit's own statement of what it can emit, and the requested
# set is intersected with it, so one tree builds on every machine this project runs on.
#
# 86 IS IN THE LIST FOR ADA'S SAKE, NOT FOR AMPERE'S. A cubin built for compute capability X.y
# runs on any device of capability X.z with z >= y, and Ada is 8.9, so on a toolkit too old to
# know compute_89 the sm_86 image is what an RTX 6000 Ada or an L40S actually executes. Without
# it such a build carries no SASS those cards can run and every load pays a JIT from PTX.
#
#     75  Quadro RTX 6000, T4         Turing, and PTX compute_90 cannot reach it
#     80  A100                        the cluster
#     86  A10, RTX 3090               and the image Ada falls back to on an older toolkit
#     89  L40S, RTX 6000 Ada          native Ada, CUDA 11.8 and later
#     90  H100                        PTX only, so anything newer still loads
#
# Override from the command line to build for one card: `make universal ARCHS=89 PTX=89`.
ARCHS ?= 75 80 86 89
PTX   ?= 90

# THE TOOLKIT IS SEARCHED, NOT TAKEN FROM PATH, AND THE PROFILE SAYS WHY.
#
# Every mira-dsp cubin in outputs/plots/profiler/runtime_throughput_mira-dsp.json carries
#
#     binary_compute_capability: 86        while    device.compute_capability: [8, 9]
#
# on an RTX 6000 Ada. The intersection below did its job: PATH's nvcc was the distribution's
# 11.5, which does not list compute_89, so 89 was dropped from BUILD_ARCHS and the card ran the
# sm_86 fallback image. Ada raises FP32 throughput over Ampere and this family's inner loop is a
# chain of FP32 FMA, so the scan was being scheduled for the wrong machine -- see _arch_note in
# ../driver.py, which exists to report exactly this and did.
#
# kernel/cuda/upstream_extension.py had the identical defect and was fixed on 2026-09-14 by
# searching the node's toolkits instead of PATH alone. THAT FIX NEVER REACHED THIS FILE, which is
# the Makefile path and builds every cubin this repository owns. Fixing the class, not the
# instance: the same candidate list, in the same order.
#
# NVCC SET IN THE ENVIRONMENT IS OBEYED AND NOT SEARCHED AROUND. Someone who names a toolkit
# wants that toolkit; silently building with another would be this defect in the other direction.
ifeq ($(origin NVCC),undefined)
_nvcc_search := $(strip \
    $(wildcard $(CUDA_HOME)/bin/nvcc) \
    $(wildcard $(CUDA_PATH)/bin/nvcc) \
    $(lastword $(sort $(wildcard /usr/local/cuda-*/bin/nvcc))) \
    $(wildcard /usr/local/cuda/bin/nvcc) \
    $(shell command -v nvcc 2>/dev/null))
# The first candidate that can emit the newest architecture asked for; else the first that runs;
# else the bare name, so the compile reports a missing nvcc in its own words.
_nvcc_best := $(firstword $(foreach n,$(_nvcc_search), \
    $(if $(filter $(lastword $(sort $(ARCHS))),$(shell $(n) --list-gpu-arch 2>/dev/null | sed -n 's/^compute_//p')),$(n))))
NVCC := $(firstword $(_nvcc_best) $(_nvcc_search) nvcc)
endif

, := ,

# A toolkit that cannot be queried at all -- nvcc absent from PATH, as on a login node before
# the module is loaded -- leaves the requested set alone. The compile is what reports that,
# with its own message, rather than this file guessing.
_nvcc_archs := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | sed -n 's/^compute_//p')
ifeq ($(strip $(_nvcc_archs)),)
BUILD_ARCHS := $(strip $(ARCHS))
BUILD_PTX   := $(PTX)
else
BUILD_ARCHS := $(strip $(foreach a,$(ARCHS),$(if $(filter $(a),$(_nvcc_archs)),$(a))))
BUILD_PTX   := $(if $(filter $(PTX),$(_nvcc_archs)),$(PTX),$(lastword $(sort $(_nvcc_archs))))
endif

ifeq ($(strip $(BUILD_ARCHS)),)
$(error nvcc supports none of ARCHS=$(ARCHS); it lists compute_: $(_nvcc_archs))
endif

GENCODE := $(foreach a,$(BUILD_ARCHS),-gencode=arch=compute_$(a)$(,)code=sm_$(a)) \
           -gencode=arch=compute_$(BUILD_PTX)$(,)code=compute_$(BUILD_PTX)
ARCH_SUMMARY := SASS $(foreach a,$(BUILD_ARCHS),sm_$(a)) + PTX compute_$(BUILD_PTX)

# THE DEFAULT `make` TARGET BUILDS FOR THE CARD IN THIS MACHINE, and NATIVE_ARCH is what it
# names. Every family sets `ARCH ?= $(NATIVE_ARCH)`, so `make ARCH=sm_80` still overrides.
#
# THE SAME INTERSECTION APPLIES HERE AS ABOVE, AND IT WAS MISSING. Each family used to read
# nvidia-smi directly and pass the answer to nvcc unchecked, which on a machine whose toolkit is
# older than its card is a fatal error before anything compiles:
#
#     nvcc fatal   : Value 'sm_89' is not defined for option 'gpu-architecture'
#
# -- an RTX 6000 Ada, which is sm_89, in front of the nvcc 11.5 that Ubuntu 22.04 installs from
# apt. `universal` was already safe because it went through BUILD_ARCHS; the default target was
# not, so the two disagreed about which architectures exist.
#
# WHEN nvcc KNOWS THE CARD, THAT IS THE ANSWER, and every architecture nvcc lists counts here --
# a Turing sm_75 or a Volta sm_70 is built natively even though neither is on the ARCHS line,
# because ARCHS governs which images `universal` SHIPS, not which cards may be compiled for.
#
# THE FALLBACK DRAWS FROM BUILD_ARCHS INSTEAD, and only from the card's own major version. SASS
# is compatible from one minor version UP to the next within a major version, so the sm_86 image
# runs on Ada's 8.9 -- which is the reason 86 is on the ARCHS line at all, as the comment above
# says. Restricting the fallback to that curated list is deliberate: the highest architecture an
# old toolkit merely LISTS below 8.9 is sm_87, Jetson Orin, an integrated part nobody here
# builds for. BUILD_ARCHS gives sm_86, a discrete desktop image known to run on these cards.
#
# NOTHING IS PRINTED IF NO SUCH ARCHITECTURE EXISTS -- a toolkit with no member of the card's
# major version, say nvcc 11.5 in front of an H100, where no cubin it can build would load at
# all. ARCH is then empty, and each family's require_arch says so and stops.
NATIVE_ARCH := $(shell \
  card=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d '. '); \
  [ -n "$$card" ] || exit 0; \
  known="$(_nvcc_archs)"; \
  if [ -z "$$known" ]; then echo "sm_$$card"; exit 0; fi; \
  for a in $$known; do [ "$$a" = "$$card" ] && { echo "sm_$$card"; exit 0; }; done; \
  best=""; \
  for a in $(BUILD_ARCHS); do \
    if [ "$$a" -le "$$card" ] && [ $$(( a / 10 )) -eq $$(( card / 10 )) ]; then \
      if [ -z "$$best" ] || [ "$$a" -gt "$$best" ]; then best=$$a; fi; \
    fi; \
  done; \
  [ -n "$$best" ] && echo "sm_$$best")
