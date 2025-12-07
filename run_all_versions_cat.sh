#!/bin/bash

# Script to run all versions and save with different output names

IMAGE="cat1.png"
K=5
COLOR_SPACE="rgb"

echo "running all versions of k-means segmentation..."
echo "input image: $IMAGE"
echo "k=$K, color space=$COLOR_SPACE"
echo ""

# Run CPU version
echo "=== running cpu single-threaded version ==="
./build/kmeans_segmentation "$IMAGE" $K "$COLOR_SPACE" --version cpu --output output_cat1_cpu.png
echo ""

# Run OpenMP version
echo "=== running openmp multi-threaded version ==="
./build/kmeans_segmentation "$IMAGE" $K "$COLOR_SPACE" --version omp --threads 4 --output output_cat1_omp.png
echo ""

# Run CUDA/GPU version (global memory)
echo "=== running cuda gpu version (global memory) ==="
./build/kmeans_segmentation "$IMAGE" $K "$COLOR_SPACE" --version cuda --output output_cat1_gpu.png
echo ""

# Run CUDA/GPU version (shared memory)
echo "=== running cuda gpu version (shared memory) ==="
./build/kmeans_segmentation "$IMAGE" $K "$COLOR_SPACE" --version cuda_shared --output output_cat1_gpu_shared.png
echo ""

echo "=== all versions completed! ==="
echo "output files:"
echo "  - output_cat1_cpu.png (cpu single-threaded)"
echo "  - output_cat1_omp.png (openmp multi-threaded)"
echo "  - output_cat1_gpu.png (cuda gpu - global memory)"
echo "  - output_cat1_gpu_shared.png (cuda gpu - shared memory)"

