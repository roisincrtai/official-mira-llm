# tokenizer/cpp/Makefile -- build/tokenizer/cpp/libbpe.so from the sources here, standard C++17 only.
#
#     make -f tokenizer/cpp/Makefile           build it when a .cpp or .hpp here is newer
#     make -f tokenizer/cpp/Makefile LIB=path  build it at `path` (the yaml key libbpe)
#     make -f tokenizer/cpp/Makefile clean     remove it
#
# The sources are every .cpp here (bpe.hpp lists them, one concern each), compiled together into
# the one library.

SOURCE_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(SOURCE_DIR)/../..)
LIB ?= $(ROOT)/build/tokenizer/cpp/libbpe.so
SOURCES := $(sort $(wildcard $(SOURCE_DIR)/*.cpp))
HEADERS := $(sort $(wildcard $(SOURCE_DIR)/*.hpp))

CXXFLAGS ?= -O3
CXXFLAGS += -std=c++17 -fPIC -pthread -Wall -Wextra
ifeq ($(shell uname -s),Darwin)
SHARED := -dynamiclib
else
SHARED := -shared
endif

# Written under a temporary name and renamed, so a process loading the library never reads a
# half-written file.
$(LIB): $(SOURCES) $(HEADERS)
	@mkdir -p $(dir $(LIB))
	@echo "[tokenizer] building $(LIB)"
	$(CXX) $(CXXFLAGS) $(SHARED) -o $@.tmp.$$$$ $(SOURCES) && mv -f $@.tmp.$$$$ $@

clean:
	rm -f $(LIB)

.PHONY: clean
