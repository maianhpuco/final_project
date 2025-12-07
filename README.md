# K-Means Image Segmentation

A project implementing K-means clustering for image segmentation with multiple parallelization approaches:
- Single-threaded CPU
- OpenMP multi-threaded CPU  
- CUDA GPU (global memory)
- CUDA GPU (shared memory)

## Quick Start

### 1. Load Required Modules

```bash
module load cmake
module load cudatoolkit  # If using CUDA
export LD_LIBRARY_PATH="/usr/lib64/nagios/plugins/python3/lib:$LD_LIBRARY_PATH"
```

### 2. Build the Project

```bash
cd /project/hnguyen2/mvu9/folder_04_ma/final_project
make clean
make build
```

Or manually:
```bash
mkdir build
cd build
cmake ..
make
```

### 3. Run All Versions

**Quick method:**
```bash
./run_all_versions_scene.sh
```

Or using Makefile:
```bash
make run-all
```

**Individual commands:**
```bash
# CPU version
./build/kmeans_segmentation scene.png 5 rgb --version cpu --output output_scene_cpu.png

# OpenMP version
./build/kmeans_segmentation scene.png 5 rgb --version omp --threads 4 --output output_scene_omp.png

# CUDA GPU version (global memory)
./build/kmeans_segmentation scene.png 5 rgb --version cuda --output output_scene_gpu.png

# CUDA GPU version (shared memory)
./build/kmeans_segmentation scene.png 5 rgb --version cuda_shared --output output_scene_gpu_shared.png

# Benchmark all versions
./build/kmeans_segmentation scene.png 5 rgb --version benchmark
```

## Usage

```bash
./build/kmeans_segmentation <image_path> <k> <color_space> [options]
```

**Parameters:**
- `image_path`: Path to input image (PNG, JPG, etc.)
- `k`: Number of clusters (3-100)
- `color_space`: `rgb` or `lab`

**Options:**
- `--version <cpu|omp|cuda|cuda_shared|benchmark>`: Choose implementation
- `--output <path>`: Output file path (default: `output_segmented.png`)
- `--threads <n>`: Number of OpenMP threads (default: auto)

## Examples

```bash
# Different color spaces
./build/kmeans_segmentation scene.png 5 rgb --version omp
./build/kmeans_segmentation scene.png 5 lab --version cuda_shared

# Different number of clusters
./build/kmeans_segmentation scene.png 3 rgb --version cuda_shared
./build/kmeans_segmentation scene.png 8 rgb --version cuda_shared
```

## Documentation

For detailed information about the project, implementation details, and results, see **report.md** or the generated PDF: **report.pdf**.
