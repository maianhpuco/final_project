# Simple Makefile for K-Means Image Segmentation
# This project uses CMake for building

.PHONY: all build clean help run-cpu run-omp run-benchmark

# Default target: build the project
all: build

# Build the project using CMake
build:
	@echo "Building project..."
	@if ! command -v cmake >/dev/null 2>&1; then \
		echo "CMake not found. Attempting to load module..."; \
		if [ -f /usr/share/lmod/lmod/init/bash ]; then \
			. /usr/share/lmod/lmod/init/bash && module load cmake 2>/dev/null || true; \
		elif [ -f /usr/share/lmod/lmod/init/sh ]; then \
			. /usr/share/lmod/lmod/init/sh && module load cmake 2>/dev/null || true; \
		fi; \
		if ! command -v cmake >/dev/null 2>&1; then \
			echo ""; \
			echo "Error: CMake not found. Please run:"; \
			echo "  module load cmake"; \
			echo "  module load cudatoolkit  (if using CUDA)"; \
			echo "  make build"; \
			echo ""; \
			exit 1; \
		fi; \
	fi
	@echo "Checking for CUDA..."
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "CUDA found: $$(nvcc --version | head -1)"; \
	else \
		echo "CUDA not found - GPU version will not be available"; \
		echo "  (To enable CUDA: module load cudatoolkit)"; \
	fi
	@mkdir -p build
	@cd build && cmake .. && make
	@echo "Build complete! Executable: build/kmeans_segmentation"

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf build
	@echo "Clean complete!"

# Show help message
help:
	@echo "K-Means Image Segmentation - Makefile"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - CMake (will try to load module automatically)"
	@echo "  - C++ compiler with OpenMP support"
	@echo "  - CUDA Toolkit (optional, for GPU version)"
	@echo ""
	@echo "If CMake is not found, run: module load cmake"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build the project (default)"
	@echo "  make build    - Build the project"
	@echo "  make clean    - Remove build files"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Example Usage (after building):"
	@echo ""
	@echo "  # Run single-threaded version"
	@echo "  ./build/kmeans_segmentation image.png 5 rgb --version cpu"
	@echo ""
	@echo "  # Run OpenMP version with 4 threads"
	@echo "  ./build/kmeans_segmentation image.png 5 rgb --version omp --threads 4"
	@echo ""
	@echo "  # Benchmark both versions"
	@echo "  ./build/kmeans_segmentation image.png 5 rgb --version benchmark"
	@echo ""
	@echo "  # With Lab color space"
	@echo "  ./build/kmeans_segmentation image.png 8 lab --version omp"
	@echo ""
	@echo "  # Custom output file"
	@echo "  ./build/kmeans_segmentation image.png 5 rgb --version omp --output result.png"

# Example run targets (adjust image path as needed)
run-cpu:
	@./build/kmeans_segmentation cat1.png 5 rgb --version cpu

run-omp:
	@./build/kmeans_segmentation cat1.png 5 rgb --version omp --threads 4

run-benchmark:
	@./build/kmeans_segmentation cat1.png 5 rgb --version benchmark

