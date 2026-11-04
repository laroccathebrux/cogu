# The medula firmware on the computer: the generated headers, the native tests, and the shared library the
# simulator loads (sim/brain/medula_c.py). The ESP32 build is esp32/ (PlatformIO).
#     make -C firmware/medula            # headers + library + tests
#     make -C firmware/medula test

# relative paths: make cannot handle the spaces in the repo path
ROOT := ../..
OUT := $(ROOT)/build/medula
CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror -fno-fast-math
INC := -Isrc -Igen
CORE := src/mlp.cpp src/protocol.cpp src/controllers.cpp src/medula.cpp src/imu.cpp
HEADERS := $(wildcard src/*.h) gen/cogu_policies.h gen/cogu_params.h
UNAME := $(shell uname)
LIB := $(OUT)/libcogu_medula.$(if $(filter Darwin,$(UNAME)),dylib,so)

all: $(LIB) test

gen/cogu_policies.h: tools/export.py
	cd $(ROOT) && uv run python firmware/medula/tools/export.py

gen/cogu_params.h: gen/cogu_policies.h

$(LIB): $(CORE) src/sil.cpp $(HEADERS)
	@mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $(INC) -shared -fPIC -o $@ $(CORE) src/sil.cpp

$(OUT)/test_native: $(CORE) test/test_native.cpp $(HEADERS)
	@mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $(CORE) test/test_native.cpp

test: $(OUT)/test_native
	$(OUT)/test_native

lib: $(LIB)

.PHONY: all test lib
