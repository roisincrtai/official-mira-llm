SHELL := /bin/sh
SRCDIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT   := $(abspath $(SRCDIR)/../../..)
OUT    := $(ROOT)/build/kernel/metal/mira_dspv2
PYTHON ?= python3
METAL ?= xcrun -sdk macosx metal
METALLIB ?= xcrun -sdk macosx metallib
CLANG ?= xcrun -sdk macosx clang++
ORDERS := 1 2 4 8
MEMORIES := 8 16 32 64
VALUES := float half bfloat
DEPS := $(SRCDIR)/mira_iir.metal $(wildcard $(SRCDIR)/*.h) $(SRCDIR)/mira_metal.mm $(SRCDIR)/Makefile

.PHONY: build universal check quicktest unittest clean
build: $(OUT)/mira_iir.build.stamp

universal: build

$(OUT)/mira_iir.build.stamp: $(DEPS)
	@rm -rf $(OUT)/air
	@mkdir -p $(OUT)/air
	@for k in $(ORDERS); do for m in $(MEMORIES); do for v in $(VALUES); do \
	    $(METAL) -std=metal3.1 -O2 -DMIRA_ORDER=$$k -DMIRA_MEMORY=$$m -DMIRA_VALUE=$$v \
	        -I$(SRCDIR) -c $(SRCDIR)/mira_iir.metal -o $(OUT)/air/mira_iir_k$${k}_m$${m}_$$v.air || exit 1; \
	done; done; done
	$(METALLIB) $(OUT)/air/*.air -o $(OUT)/mira_iir.metallib
	@include=$$($(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I' + path for path in include_paths()))") && \
	library=$$($(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(library_paths()[0])") && \
	python_include=$$($(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])") && \
	echo "$(CLANG) $(SRCDIR)/mira_metal.mm -o $(OUT)/mira_metal.so" && \
	$(CLANG) -std=c++17 -O2 -fPIC -shared -DTORCH_EXTENSION_NAME=mira_metal \
	    -DTORCH_API_INCLUDE_EXTENSION_H $$include -I$$python_include $(SRCDIR)/mira_metal.mm \
	    -L$$library -Wl,-rpath,$$library -lc10 -ltorch -ltorch_cpu -ltorch_python \
	    -undefined dynamic_lookup -framework Foundation -framework Metal \
	    -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	    -o $(OUT)/mira_metal.so
	@touch $@
	@echo "built $(OUT)/mira_iir.metallib and $(OUT)/mira_metal.so"

check: build
	@cd $(ROOT) && $(PYTHON) $(SRCDIR)/check.py

quicktest: build
	@cd $(ROOT) && $(PYTHON) $(SRCDIR)/check.py --quick

unittest: check

clean:
	rm -rf $(OUT)
