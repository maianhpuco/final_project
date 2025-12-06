#!/bin/bash

# Script to run all three versions and save with different output names

IMAGE="cat1.png"
K=5
COLOR_SPACE="rgb"

echo "Running all three versions of K-means segmentation..."
echo "Input image: $IMAGE"
echo "K=$K, Color space=$COLOR_SPACE"
echo ""

# Run CPU version
echo "=== Running CPU single-threaded version ==="
./build/kmeans_segmentation "$IMAGE" $K "$COLOR_SPACE" --version cpu --output output_cat1_cpu.png
echo ""

# Run OpenMP version
echo "=== Running OpenMP multi-threaded version ==="
./build/kmeans_segmentation "$IMAGE" $K "$COLOR_SPACE" --version omp --threads 4 --output output_cat1_omp.png
echo ""

# Run CUDA/GPU version
echo "=== Running CUDA GPU version ==="
./build/kmeans_segmentation "$IMAGE" $K "$COLOR_SPACE" --version cuda --output output_cat1_gpu.png
echo ""

echo "=== All versions completed! ==="
echo "Output files:"
echo "  - output_cat1_cpu.png (CPU single-threaded)"
echo "  - output_cat1_omp.png (OpenMP multi-threaded)"
echo "  - output_cat1_gpu.png (CUDA GPU)"

