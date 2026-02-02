# Dendrite - GPU-Accelerated Battery Simulation Kernels
# Makefile for building the library and examples

# Compiler settings
NVCC := nvcc
CC := gcc
AR := ar

# CUDA architecture (RTX 3090 = SM 86)
CUDA_ARCH := -arch=sm_86

# Compiler flags
NVCC_FLAGS := $(CUDA_ARCH) -O3 -Xcompiler -fPIC -Iinclude
NVCC_FLAGS += --use_fast_math
NVCC_FLAGS += -Xptxas -v  # Show register usage

# Debug flags (uncomment for debugging)
# NVCC_FLAGS += -G -g -lineinfo

# Directories
SRC_DIR := csrc
INC_DIR := include
BUILD_DIR := build
LIB_DIR := lib
BIN_DIR := bin
EXAMPLE_DIR := examples
BENCH_DIR := benchmarks

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Library name
LIB_NAME := libdendrite
STATIC_LIB := $(LIB_DIR)/$(LIB_NAME).a
SHARED_LIB := $(LIB_DIR)/$(LIB_NAME).so

# Default target
all: dirs $(STATIC_LIB) $(SHARED_LIB)

# Create directories
dirs:
	@mkdir -p $(BUILD_DIR) $(LIB_DIR) $(BIN_DIR)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(INC_DIR)/dendrite.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Static library
$(STATIC_LIB): $(OBJS)
	$(AR) rcs $@ $^
	@echo "Built static library: $@"

# Shared library
$(SHARED_LIB): $(OBJS)
	$(NVCC) $(CUDA_ARCH) -shared -o $@ $^
	@echo "Built shared library: $@"

# Examples (statically linked - no LD_LIBRARY_PATH needed)
examples: dirs $(STATIC_LIB)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/simple_diffusion $(EXAMPLE_DIR)/simple_diffusion.cu $(STATIC_LIB) -lcudart
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/battery_particle $(EXAMPLE_DIR)/battery_particle.cu $(STATIC_LIB) -lcudart
	@echo "Built examples in $(BIN_DIR)/"

# Benchmarks (statically linked)
benchmarks: dirs $(STATIC_LIB)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/benchmark $(BENCH_DIR)/benchmark.cu $(STATIC_LIB) -lcudart
	@echo "Built benchmark in $(BIN_DIR)/"

# Run benchmarks
bench: benchmarks
	./$(BIN_DIR)/benchmark

# Install (to /usr/local by default)
PREFIX ?= /usr/local
install: $(STATIC_LIB) $(SHARED_LIB)
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install -m 644 $(STATIC_LIB) $(PREFIX)/lib/
	install -m 755 $(SHARED_LIB) $(PREFIX)/lib/
	install -m 644 $(INC_DIR)/dendrite.h $(PREFIX)/include/
	ldconfig || true
	@echo "Installed to $(PREFIX)"

# Uninstall
uninstall:
	rm -f $(PREFIX)/lib/$(LIB_NAME).a
	rm -f $(PREFIX)/lib/$(LIB_NAME).so
	rm -f $(PREFIX)/include/dendrite.h

# Clean
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR) $(BIN_DIR)

# Show PTX (for optimization analysis)
ptx: $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) --ptx -o $(BUILD_DIR)/diffusion.ptx $(SRC_DIR)/diffusion.cu
	$(NVCC) $(NVCC_FLAGS) --ptx -o $(BUILD_DIR)/electrochemistry.ptx $(SRC_DIR)/electrochemistry.cu
	$(NVCC) $(NVCC_FLAGS) --ptx -o $(BUILD_DIR)/thermal.ptx $(SRC_DIR)/thermal.cu
	@echo "PTX files generated in $(BUILD_DIR)/"

# Show SASS (actual GPU assembly)
sass: $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) --cubin -o $(BUILD_DIR)/diffusion.cubin $(SRC_DIR)/diffusion.cu
	cuobjdump --dump-sass $(BUILD_DIR)/diffusion.cubin > $(BUILD_DIR)/diffusion.sass
	@echo "SASS files generated in $(BUILD_DIR)/"

.PHONY: all dirs examples benchmarks bench install uninstall clean ptx sass
