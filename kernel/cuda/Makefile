# kernel/cuda/Makefile -- every kernel family, one command. NOTHING IS BUILT HERE.
#
#     make unittest   correctness for EVERY family on the FAMILIES line below
#     make help       this list
#
# THIS FILE DISPATCHES AND OWNS NOTHING. Each family's Makefile is the authority on how its
# kernels are built and checked -- the architectures, the module lines to paste, the output
# path, the ncu invocation -- and every one of those targets stays exactly where it is. No
# target is moved here, renamed here or copied here. What was missing was a way to ask every
# family the ONE question that has a right answer, in one command, from a job script:
#
#     make -C kernel/cuda unittest                 all of them
#     make -C kernel/cuda/mira_dspv2 unittest  one of them, unchanged
#
# A family added under kernel/cuda/ needs one word on the FAMILIES line below and no other
# change, because the recipe knows nothing about any family except its directory name.

SHELL := /bin/sh

# DERIVED FROM THIS FILE'S OWN LOCATION, NOT FROM THE WORKING DIRECTORY, so the target works
# from anywhere -- the same rule every family Makefile follows, and for the same reason: the
# scripts that call it do not cd first.
SRCDIR   := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
# rwkv_v8 IS THE ONLY FAMILY HERE WITH NO .cu OF OURS. Its kernel is BlinkDL's, vendored at
# rwkv_v8/upstream/cuda/ and built by their own cpp_extension.load; its Makefile therefore has
# `upstream` where the others have `universal`, and no cubin at all. It is on this line because
# the ONE question this file asks -- does the kernel agree with its reference -- has an answer
# for it too, and because the recipe below knows nothing about any family except its directory
# name. See kernel/cuda/upstream_extension.py for why upstream's loader runs rather than ours.
#
# MIRA-LATTICE LEADS THIS LINE, AND WAS ABSENT FROM IT. This repository's own kernel -- the one
# every claim rests on -- was the one family `make unittest` never checked, so a lattice kernel
# that disagreed with its reference would have been reported as "PASS: every kernel agrees".
#
# MIRA-HIGH IS COMMENTED OUT, NOT REMOVED, and it had no directory under kernel/cuda/ at all, so
# the loop below died entering it and the families after it went unmeasured. Restore by putting
# the word back when the directory returns.
#
# MIRA-NEWTON AND MIRA-SDPA ARE COMMENTED OUT, NOT REMOVED. Neither is an architecture this
# repository trains any more, so a disagreement between either kernel and its reference decides
# nothing and only costs queue time on the card `make unittest` has to reach.
# kernel/cuda/mira-sdpa/ and kernel/cuda/mira-newton/ are untouched on disk and each still builds
# and checks by `make -C kernel/cuda/<family>`. Restore by putting the two words back on the
# FAMILIES line.
FAMILIES := mira_dspv2 mamba_s6 rwkv_v6 rwkv_v7 rwkv_v8
#   with mira-sdpa and mira-newton:
#   FAMILIES := mira-vector mira-sdpa mira-newton mamba_s6 rwkv_v6 rwkv_v7 rwkv_v8
#   with mira-high, once kernel/cuda/mira-high/ exists again:
#   FAMILIES := mira-vector mira-sdpa mira-newton mira-high mamba_s6 rwkv_v6 rwkv_v7 rwkv_v8

help:
	@echo "make unittest   correctness for every family: $(FAMILIES)"
	@echo
	@echo "Everything else is per family and stays there:"
	@echo "  make -C kernel/cuda/<family> help"

# EVERY FAMILY IS RUN AND THEN THE VERDICT -- not the first failure and stop. Halting at the
# first leaves the rest unmeasured and the reader with no idea whether they were fine, and on a
# card that costs queue time to reach that is the wrong trade: the reason to run this at all is
# to come back with the whole answer. `|| bad=` catches a family's non-zero exit so the loop
# continues, the summary is the last thing printed, and THE STATUS IS NON-ZERO IF ANY FAMILY
# DISAGREED -- which is what makes this usable in a job script under `set -e`.
unittest:
	@bad=""; \
	 for f in $(FAMILIES); do \
	   echo "=== $$f ==="; \
	   $(MAKE) --no-print-directory -C $(SRCDIR)/$$f unittest || bad="$$bad $$f"; \
	   echo; \
	 done; \
	 if [ -n "$$bad" ]; then \
	   echo "FAILED:$$bad -- see the MISMATCH lines above. DO NOT USE those kernels."; \
	   exit 1; \
	 fi; \
	 echo "PASS: every kernel in $(FAMILIES) agrees with its reference."

.DEFAULT_GOAL := help
.PHONY: help unittest
