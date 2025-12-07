#include "kmeans_cuda_shared.h"
#include "timer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <iostream>
#include <cstring>

// Maximum centroids that can fit in shared memory (similar to MAX_SHARED_SPHERES)
#define MAX_SHARED_CENTROIDS 64

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// RGB to Lab conversion helper functions (device) - static to avoid duplicate symbols
static __device__ float f_device_shared(float t) {
    return (t > 0.008856f) ? powf(t, 1.0f/3.0f) : (7.787f * t + 16.0f/116.0f);
}

static __device__ float f_inv_device_shared(float t) {
    return (t > 0.206897f) ? powf(t, 3.0f) : ((t - 16.0f/116.0f) / 7.787f);
}

static __device__ void rgbToLab_device_shared(float r, float g, float b, float& l, float& a, float& b_out) {
    r /= 255.0f;
    g /= 255.0f;
    b /= 255.0f;
    
    float x = 0.412453f * r + 0.357580f * g + 0.180423f * b;
    float y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
    float z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
    
    x /= 0.95047f;
    z /= 1.08883f;
    
    float fx = f_device_shared(x);
    float fy = f_device_shared(y);
    float fz = f_device_shared(z);
    
    l = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    b_out = 200.0f * (fy - fz);
}

static __device__ void labToRgb_device_shared(float l, float a, float b_in, float& r, float& g, float& blue) {
    float fy = (l + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b_in / 200.0f;
    
    float x = 0.95047f * f_inv_device_shared(fx);
    float y = f_inv_device_shared(fy);
    float z = 1.08883f * f_inv_device_shared(fz);
    
    r = 3.240479f * x - 1.537150f * y - 0.498535f * z;
    g = -0.969256f * x + 1.875992f * y + 0.041556f * z;
    blue = 0.055648f * x - 0.204043f * y + 1.057311f * z;
    
    r = fmaxf(0.0f, fminf(1.0f, r)) * 255.0f;
    g = fmaxf(0.0f, fminf(1.0f, g)) * 255.0f;
    blue = fmaxf(0.0f, fminf(1.0f, blue)) * 255.0f;
}

static __device__ float colorDistance_device_shared(const float* color1, const float* color2) {
    float dx = color1[0] - color2[0];
    float dy = color1[1] - color2[1];
    float dz = color1[2] - color2[2];
    return dx * dx + dy * dy + dz * dz;
}

// Kernel: Assign pixels using SHARED MEMORY for centroids (optimized)
__global__ void assignPixelsSharedKernel(
    const unsigned char* imageData,
    int width,
    int height,
    int channels,
    const float* globalCentroids,  // Centroids in global memory
    int k,
    int* assignments,
    int colorSpace
) {
    // =========================================================================
    // SHARED MEMORY: Load centroids cooperatively into shared memory
    // This reduces global memory access latency significantly
    // =========================================================================
    extern __shared__ float sharedCentroids[];  // k * 3 floats
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    int centroidsToLoad = (k < MAX_SHARED_CENTROIDS) ? k : MAX_SHARED_CENTROIDS;
    int totalFloats = centroidsToLoad * 3;
    
    // Cooperative loading: each thread loads some centroids
    for (int i = tid; i < totalFloats; i += blockSize) {
        sharedCentroids[i] = globalCentroids[i];
    }
    
    // Synchronize to ensure all centroids are loaded before assignment
    __syncthreads();
    
    // =========================================================================
    // Calculate pixel coordinates
    // =========================================================================
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int pixelOffset = (y * width + x) * channels;
    
    float pixelColor[3];
    
    if (colorSpace == 0) {  // RGB
        pixelColor[0] = imageData[pixelOffset];
        pixelColor[1] = imageData[pixelOffset + 1];
        pixelColor[2] = imageData[pixelOffset + 2];
    } else {  // LAB
        rgbToLab_device_shared(
            imageData[pixelOffset],
            imageData[pixelOffset + 1],
            imageData[pixelOffset + 2],
            pixelColor[0], pixelColor[1], pixelColor[2]
        );
    }
    
    // =========================================================================
    // Find nearest centroid using SHARED MEMORY (fast access)
    // =========================================================================
    float minDist = 1e30f;
    int bestCentroid = 0;
    
    for (int i = 0; i < centroidsToLoad; i++) {
        float dist = colorDistance_device_shared(pixelColor, &sharedCentroids[i * 3]);
        if (dist < minDist) {
            minDist = dist;
            bestCentroid = i;
        }
    }
    
    // If k > MAX_SHARED_CENTROIDS, check remaining centroids in global memory
    if (k > MAX_SHARED_CENTROIDS) {
        for (int i = MAX_SHARED_CENTROIDS; i < k; i++) {
            float dist = colorDistance_device_shared(pixelColor, &globalCentroids[i * 3]);
            if (dist < minDist) {
                minDist = dist;
                bestCentroid = i;
            }
        }
    }
    
    assignments[idx] = bestCentroid;
}

// Kernel: Update centroids (renamed to avoid duplicate symbols)
__global__ void updateCentroidsSharedKernel(
    const unsigned char* imageData,
    int width,
    int height,
    int channels,
    const int* assignments,
    float* newCentroids,
    int* counts,
    int k,
    int colorSpace
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPixels = width * height;
    
    if (idx >= numPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    int pixelOffset = (y * width + x) * channels;
    int cluster = assignments[idx];
    
    float pixelColor[3];
    
    if (colorSpace == 0) {  // RGB
        pixelColor[0] = imageData[pixelOffset];
        pixelColor[1] = imageData[pixelOffset + 1];
        pixelColor[2] = imageData[pixelOffset + 2];
    } else {  // LAB
        rgbToLab_device_shared(
            imageData[pixelOffset],
            imageData[pixelOffset + 1],
            imageData[pixelOffset + 2],
            pixelColor[0], pixelColor[1], pixelColor[2]
        );
    }
    
    atomicAdd(&newCentroids[cluster * 3 + 0], pixelColor[0]);
    atomicAdd(&newCentroids[cluster * 3 + 1], pixelColor[1]);
    atomicAdd(&newCentroids[cluster * 3 + 2], pixelColor[2]);
    atomicAdd(&counts[cluster], 1);
}

// Helper function for atomic add on double (CUDA compatibility)
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Kernel: Compute error using shared memory for centroids
__global__ void computeErrorSharedKernel(
    const unsigned char* imageData,
    int width,
    int height,
    int channels,
    const float* globalCentroids,
    const int* assignments,
    int k,
    int colorSpace,
    double* error
) {
    // Load centroids into shared memory
    extern __shared__ float sharedCentroids[];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    int centroidsToLoad = (k < MAX_SHARED_CENTROIDS) ? k : MAX_SHARED_CENTROIDS;
    int totalFloats = centroidsToLoad * 3;
    
    for (int i = tid; i < totalFloats; i += blockSize) {
        sharedCentroids[i] = globalCentroids[i];
    }
    __syncthreads();
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int pixelOffset = (y * width + x) * channels;
    int cluster = assignments[idx];
    
    float pixelColor[3];
    
    if (colorSpace == 0) {  // RGB
        pixelColor[0] = imageData[pixelOffset];
        pixelColor[1] = imageData[pixelOffset + 1];
        pixelColor[2] = imageData[pixelOffset + 2];
    } else {  // LAB
        rgbToLab_device_shared(
            imageData[pixelOffset],
            imageData[pixelOffset + 1],
            imageData[pixelOffset + 2],
            pixelColor[0], pixelColor[1], pixelColor[2]
        );
    }
    
    float dist;
    if (cluster < centroidsToLoad) {
        dist = colorDistance_device_shared(pixelColor, &sharedCentroids[cluster * 3]);
    } else {
        dist = colorDistance_device_shared(pixelColor, &globalCentroids[cluster * 3]);
    }
    
    atomicAddDouble(error, (double)dist);
}

// Kernel: Generate segmented image (renamed to avoid duplicate symbols)
__global__ void generateSegmentedImageSharedKernel(
    const int* assignments,
    const float* centroids,
    int width,
    int height,
    int channels,
    int k,
    int colorSpace,
    unsigned char* outputData,
    const unsigned char* inputData
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPixels = width * height;
    
    if (idx >= numPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    int pixelOffset = (y * width + x) * channels;
    int cluster = assignments[idx];
    
    if (colorSpace == 0) {  // RGB
        outputData[pixelOffset + 0] = (unsigned char)lrintf(centroids[cluster * 3 + 0]);
        outputData[pixelOffset + 1] = (unsigned char)lrintf(centroids[cluster * 3 + 1]);
        outputData[pixelOffset + 2] = (unsigned char)lrintf(centroids[cluster * 3 + 2]);
    } else {  // LAB
        float r, g, b;
        labToRgb_device_shared(
            centroids[cluster * 3 + 0],
            centroids[cluster * 3 + 1],
            centroids[cluster * 3 + 2],
            r, g, b
        );
        outputData[pixelOffset + 0] = (unsigned char)lrintf(r);
        outputData[pixelOffset + 1] = (unsigned char)lrintf(g);
        outputData[pixelOffset + 2] = (unsigned char)lrintf(b);
    }
    
    if (channels == 4) {
        outputData[pixelOffset + 3] = inputData[pixelOffset + 3];
    }
}

// Initialize centroids (same as before)
std::vector<std::vector<float>> KMeansCUDAShared::initializeCentroids(
    const Image& image,
    int k,
    ColorSpace colorSpace
) {
    std::vector<std::vector<float>> centroids(k, std::vector<float>(3));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, image.width * image.height - 1);
    
    auto f = [](float t) {
        return (t > 0.008856f) ? std::pow(t, 1.0f/3.0f) : (7.787f * t + 16.0f/116.0f);
    };
    
    for (int i = 0; i < k; i++) {
        int pixelIdx = dis(gen);
        int x = pixelIdx % image.width;
        int y = pixelIdx / image.width;
        const unsigned char* pixel = image.getPixel(x, y);
        
        if (colorSpace == ColorSpace::RGB) {
            centroids[i][0] = pixel[0];
            centroids[i][1] = pixel[1];
            centroids[i][2] = pixel[2];
        } else {
            float r = pixel[0] / 255.0f;
            float g = pixel[1] / 255.0f;
            float b = pixel[2] / 255.0f;
            
            float x_val = 0.412453f * r + 0.357580f * g + 0.180423f * b;
            float y_val = 0.212671f * r + 0.715160f * g + 0.072169f * b;
            float z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
            
            x_val /= 0.95047f;
            z /= 1.08883f;
            
            float fx = f(x_val);
            float fy = f(y_val);
            float fz = f(z);
            
            centroids[i][0] = 116.0f * fy - 16.0f;
            centroids[i][1] = 500.0f * (fx - fy);
            centroids[i][2] = 200.0f * (fy - fz);
        }
    }
    
    return centroids;
}

KMeansResult KMeansCUDAShared::segment(
    const Image& image,
    int k,
    ColorSpace colorSpace,
    int maxIterations,
    float threshold
) {
    Timer timer;
    KMeansResult result;
    
    int numPixels = image.width * image.height;
    int imageSize = numPixels * image.channels;
    
    // Allocate device memory
    unsigned char* d_imageData;
    int* d_assignments;
    float* d_centroids;
    float* d_newCentroids;
    int* d_counts;
    double* d_error;
    unsigned char* d_outputData;
    
    CUDA_CHECK(cudaMalloc(&d_imageData, imageSize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_assignments, numPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_centroids, k * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_newCentroids, k * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_error, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_outputData, imageSize * sizeof(unsigned char)));
    
    // Copy image to device
    CUDA_CHECK(cudaMemcpy(d_imageData, image.data.data(), 
                          imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    // Initialize centroids on CPU
    std::vector<std::vector<float>> centroids = initializeCentroids(image, k, colorSpace);
    
    // Copy centroids to device
    std::vector<float> centroidsFlat(k * 3);
    for (int i = 0; i < k; i++) {
        centroidsFlat[i * 3 + 0] = centroids[i][0];
        centroidsFlat[i * 3 + 1] = centroids[i][1];
        centroidsFlat[i * 3 + 2] = centroids[i][2];
    }
    CUDA_CHECK(cudaMemcpy(d_centroids, centroidsFlat.data(), 
                          k * 3 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Setup kernel launch parameters - use 16x16 blocks like the reference
    dim3 blockSize(16, 16);
    dim3 gridSize((image.width + blockSize.x - 1) / blockSize.x,
                  (image.height + blockSize.y - 1) / blockSize.y);
    int colorSpaceInt = (colorSpace == ColorSpace::RGB) ? 0 : 1;
    
    // Calculate shared memory size for centroids
    int centroidsToLoad = (k < MAX_SHARED_CENTROIDS) ? k : MAX_SHARED_CENTROIDS;
    size_t sharedMemSize = centroidsToLoad * 3 * sizeof(float);
    
    // Setup 1D grid parameters for update and image generation kernels
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    
    std::vector<double> errors;
    int actualIterations = 0;
    
    // Main K-means loop
    for (int iter = 0; iter < maxIterations; iter++) {
        actualIterations++;
        // Assignment step - using shared memory kernel
        assignPixelsSharedKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_imageData, image.width, image.height, image.channels,
            d_centroids, k, d_assignments, colorSpaceInt
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Update step (can start immediately after assignment kernel launches)
        CUDA_CHECK(cudaMemset(d_newCentroids, 0, k * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, k * sizeof(int)));
        
        // Use 1D grid for update kernel (simpler)
        updateCentroidsSharedKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_imageData, image.width, image.height, image.channels,
            d_assignments, d_newCentroids, d_counts, k, colorSpaceInt
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Synchronize only once after both kernels complete
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Average centroids on CPU
        std::vector<float> h_newCentroids(k * 3);
        std::vector<int> h_counts(k);
        CUDA_CHECK(cudaMemcpy(h_newCentroids.data(), d_newCentroids, 
                              k * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, 
                              k * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Average and check convergence
        float maxDelta = 0.0f;
        for (int i = 0; i < k; i++) {
            if (h_counts[i] > 0) {
                h_newCentroids[i * 3 + 0] /= h_counts[i];
                h_newCentroids[i * 3 + 1] /= h_counts[i];
                h_newCentroids[i * 3 + 2] /= h_counts[i];
            }
            
            float dx = centroidsFlat[i * 3 + 0] - h_newCentroids[i * 3 + 0];
            float dy = centroidsFlat[i * 3 + 1] - h_newCentroids[i * 3 + 1];
            float dz = centroidsFlat[i * 3 + 2] - h_newCentroids[i * 3 + 2];
            float delta = dx * dx + dy * dy + dz * dz;
            maxDelta = std::max(maxDelta, delta);
        }
        
        centroidsFlat = h_newCentroids;
        CUDA_CHECK(cudaMemcpy(d_centroids, centroidsFlat.data(), 
                              k * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        if (maxDelta < threshold) {
            std::cout << "Converged at iteration " << iter + 1 << std::endl;
            break;
        }
    }
    
    // Compute final error only once at the end (for reporting)
    double h_error = 0.0;
    CUDA_CHECK(cudaMemset(d_error, 0, sizeof(double)));
    computeErrorSharedKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_imageData, image.width, image.height, image.channels,
        d_centroids, d_assignments, k, colorSpaceInt, d_error
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_error, d_error, sizeof(double), cudaMemcpyDeviceToHost));
    errors.push_back(h_error);
    
    // Generate segmented image
    generateSegmentedImageSharedKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_assignments, d_centroids, image.width, image.height, image.channels,
        k, colorSpaceInt, d_outputData, d_imageData
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    result.segmentedImage.width = image.width;
    result.segmentedImage.height = image.height;
    result.segmentedImage.channels = image.channels;
    result.segmentedImage.data.resize(imageSize);
    CUDA_CHECK(cudaMemcpy(result.segmentedImage.data.data(), d_outputData, 
                          imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    std::vector<int> h_assignments(numPixels);
    CUDA_CHECK(cudaMemcpy(h_assignments.data(), d_assignments, 
                          numPixels * sizeof(int), cudaMemcpyDeviceToHost));
    
    result.centroids.resize(k);
    for (int i = 0; i < k; i++) {
        result.centroids[i].resize(3);
        result.centroids[i][0] = centroidsFlat[i * 3 + 0];
        result.centroids[i][1] = centroidsFlat[i * 3 + 1];
        result.centroids[i][2] = centroidsFlat[i * 3 + 2];
    }
    result.assignments = h_assignments;
    result.iterations = actualIterations;
    result.finalError = errors.empty() ? 0.0 : errors.back();
    result.runtimeMs = timer.elapsedMs();
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_imageData));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_newCentroids));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_error));
    CUDA_CHECK(cudaFree(d_outputData));
    
    return result;
}

