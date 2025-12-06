# CUDA K-Means Image Segmentation Studio

A GPU course final project that implements K-means clustering for image segmentation with three versions:
- (A) Single-threaded CPU
- (B) OpenMP multi-threaded CPU  
- (C) CUDA GPU

## Current Status

✅ **Phase 1: Single-threaded CPU implementation** (COMPLETE)  
✅ **Phase 2: OpenMP multi-threaded CPU implementation** (COMPLETE)  
✅ **Phase 3: CUDA GPU implementation** (COMPLETE)

## Building

### Prerequisites
- C++17 compiler (GCC, Clang, or MSVC)
- CMake 3.15+
- OpenMP (required for multi-threaded version)
- CUDA Toolkit (optional, for GPU version)
- OpenCV (optional, for image I/O) OR stb_image (included)

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

Or with OpenCV:
```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/opencv ..
make
```

## Usage

```bash
./kmeans_segmentation <image_path> <k> <color_space> [options]
```

### Parameters
- `image_path`: Path to input image (PNG, JPG, etc.)
- `k`: Number of clusters (3-12)
- `color_space`: `rgb` or `lab`

### Options
- `--version <cpu|omp|cuda|benchmark>`: Choose implementation
  - `cpu`: Single-threaded CPU (default)
  - `omp`: OpenMP multi-threaded CPU
  - `cuda`: CUDA GPU version (requires CUDA Toolkit)
  - `benchmark`: Compare all available versions with performance metrics
- `--output <path>`: Output file path (default: `output_segmented.png`)
- `--threads <n>`: Number of OpenMP threads (default: auto, uses OMP_NUM_THREADS)

### Examples

```bash
# Run single-threaded CPU version
./kmeans_segmentation test_image.png 5 rgb --version cpu

# Run OpenMP multi-threaded version with 4 threads
./kmeans_segmentation test_image.png 5 rgb --version omp --threads 4

# Run CUDA GPU version
./kmeans_segmentation test_image.png 5 rgb --version cuda

# Benchmark all versions
./kmeans_segmentation test_image.png 5 rgb --version benchmark

# Use Lab color space
./kmeans_segmentation test_image.png 8 lab --version cuda
```

## Project Structure

```
.
├── CMakeLists.txt          # Build configuration
├── CMakeLists.txt          # Build configuration
├── include/                # Header files
│   ├── image_loader.h
│   ├── kmeans_cpu.h
│   ├── kmeans_omp.h
│   ├── kmeans_cuda.h
│   └── timer.h
├── src/                    # Source files
│   ├── main.cpp
│   ├── kmeans_cpu.cpp
│   ├── kmeans_omp.cpp
│   ├── kmeans_cuda.cu
│   ├── image_loader.cpp
│   └── timer.cpp
└── lib/                    # Third-party libraries
    └── stb/                # stb_image (if not using OpenCV)
```

## Next Steps

- [x] Implement OpenMP version (multi-threaded CPU) ✅
- [x] Implement CUDA version (GPU) ✅
- [ ] Add UI for real-time visualization
- [ ] Generate error vs. iteration plots
- [ ] Create speedup charts for different image sizes

## Performance

### OpenMP Version
The OpenMP version parallelizes the following operations:
- **Assignment step**: Each pixel is assigned to its nearest centroid in parallel
- **Update step**: Centroids are recomputed using thread-local accumulators
- **Error computation**: Uses OpenMP reduction for thread-safe summation
- **Image generation**: Final segmented image creation is parallelized

### CUDA Version
The CUDA version uses GPU acceleration with:
- **Assignment kernel**: One thread per pixel for parallel assignment
- **Update kernel**: Atomic operations for centroid accumulation
- **Error kernel**: Atomic reduction for error computation
- **Image generation kernel**: Parallel segmented image creation
- **Memory management**: Efficient GPU memory allocation and transfers

Use `--version benchmark` to compare performance between all available versions (CPU, OpenMP, and CUDA if available).

## Notes

- The CPU implementation uses single-threaded execution
- The OpenMP implementation uses multi-threaded parallel execution
- The CUDA implementation uses GPU acceleration (requires CUDA-capable GPU)
- Supports both RGB and Lab color spaces
- Lab color space provides better perceptual clustering
- Random initialization of centroids from image pixels
- All implementations produce identical results (within numerical precision)
- CUDA version automatically detects if CUDA is available during compilation




make clean 
makd 
./build/kmeans_segmentation cat1.png 5 rgb --version benchmark 

# Test single-threaded version
./build/kmeans_segmentation cat1.png 5 rgb --version cpu

# Test OpenMP version
./build/kmeans_segmentation cat1.png 5 rgb --version omp --threads 4

# Benchmark both versions
./build/kmeans_segmentation cat1.png 5 rgb --version benchmark 