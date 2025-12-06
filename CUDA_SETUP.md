# CUDA Setup Instructions

## Step-by-Step Guide to Use CUDA Version

### Step 1: Load CUDA Toolkit Module
```bash
module load cudatoolkit
```

Verify CUDA is loaded:
```bash
which nvcc
nvcc --version
```

### Step 2: Clean Previous Build
```bash
cd /project/hnguyen2/mvu9/folder_04_ma/final_project
make clean
```

### Step 3: Rebuild with CUDA Support
```bash
make build
```

Look for this message in the output:

### Step 4: Verify CUDA Version is Available
```bash
./build/kmeans_segmentation --help
# or
./build/kmeans_segmentation cat1.png 5 rgb --version cuda --help
```

### Step 5: Test CUDA Version
```bash
# Run CUDA version
./build/kmeans_segmentation cat1.png 5 rgb --version cuda --output output_cuda.png
```

### Step 6: Benchmark All Versions (including CUDA)
```bash
./build/kmeans_segmentation cat1.png 5 rgb --version benchmark --output output_benchmark.png
```

This will compare CPU, OpenMP, and CUDA versions and show speedup!

## Troubleshooting

If CUDA version is not available:
1. Make sure CUDA module is loaded: `module list | grep cuda`
2. Check nvcc is available: `which nvcc`
3. Rebuild: `make clean && make build`
4. Check build output for CUDA messages

## Expected Output

When CUDA works, you should see:
- "Using CUDA GPU" or similar message
- Faster execution times compared to CPU/OpenMP
- Speedup numbers in benchmark mode

