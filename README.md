# CUDA K-Means Image Segmentation Studio

A GPU course final project that implements K-means clustering for image segmentation with three versions:
- (A) Single-threaded CPU
- (B) OpenMP multi-threaded CPU  
- (C) CUDA GPU

## Current Status

✅ **Phase 1: Single-threaded CPU implementation** (COMPLETE)

## Building

### Prerequisites
- C++17 compiler (GCC, Clang, or MSVC)
- CMake 3.15+
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
./kmeans_segmentation <image_path> <k> <color_space> [output_path]
```

### Parameters
- `image_path`: Path to input image (PNG, JPG, etc.)
- `k`: Number of clusters (3-12)
- `color_space`: `rgb` or `lab`
- `output_path`: Optional output file (default: `output_segmented.png`)

### Example

```bash
./kmeans_segmentation test_image.png 5 rgb output.png
./kmeans_segmentation test_image.png 8 lab
```

## Project Structure

```
.
├── CMakeLists.txt          # Build configuration
├── include/                # Header files
│   ├── image_loader.h
│   ├── kmeans_cpu.h
│   └── timer.h
├── src/                    # Source files
│   ├── main.cpp
│   ├── kmeans_cpu.cpp
│   ├── image_loader.cpp
│   └── timer.cpp
└── lib/                    # Third-party libraries
    └── stb/                # stb_image (if not using OpenCV)
```

## Next Steps

- [ ] Implement OpenMP version (multi-threaded CPU)
- [ ] Implement CUDA version (GPU)
- [ ] Add UI for real-time visualization
- [ ] Generate error vs. iteration plots
- [ ] Create speedup charts for different image sizes

## Notes

- The CPU implementation uses single-threaded execution
- Supports both RGB and Lab color spaces
- Lab color space provides better perceptual clustering
- Random initialization of centroids from image pixels

