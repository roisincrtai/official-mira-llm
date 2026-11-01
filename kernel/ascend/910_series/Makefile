PYTHON ?= python3
ROOT := $(abspath ../../..)
SOC_VERSION ?= $(shell $(PYTHON) -c 'import torch_npu; print(torch_npu.npu.get_device_name())')
REBUILD ?= 0
BUILD := $(ROOT)/build/kernel/ascend/910_series/$(SOC_VERSION)

.PHONY: all check
all:
	cmake -S . -B $(BUILD) -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=$(SOC_VERSION) -DPython3_EXECUTABLE=$(PYTHON)
	cmake --build $(BUILD) --parallel 2 $(if $(filter 1,$(REBUILD)),--clean-first,)

check: all
	$(PYTHON) check.py
