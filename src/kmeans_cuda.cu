#include "kmeans_cuda.h"
#include "timer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <iostream>
#include <cstring>

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

// RGB to Lab conversion helper functions (device)
__device__ float f_device(float t) {
    return (t > 0.008856f) ? powf(t, 1.0f/3.0f) : (7.787f * t + 16.0f/116.0f);
}

__device__ float f_inv_device(float t) {
    return (t > 0.206897f) ? powf(t, 3.0f) : ((t - 16.0f/116.0f) / 7.787f);
}

__device__ void rgbToLab_device(float r, float g, float b, float& l, float& a, float& b_out) {
    // Normalize RGB to 0-1
    r /= 255.0f;
    g /= 255.0f;
    b /= 255.0f;
    
    // Convert to XYZ
    float x = 0.412453f * r + 0.357580f * g + 0.180423f * b;
    float y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
    float z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
    
    // Normalize by D65 white point
    x /= 0.95047f;
    z /= 1.08883f;
    
    // Convert to Lab
    float fx = f_device(x);
    float fy = f_device(y);
    float fz = f_device(z);
    
    l = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    b_out = 200.0f * (fy - fz);
}

__device__ void labToRgb_device(float l, float a, float b_in, float& r, float& g, float& blue) {
    // Convert Lab to XYZ
    float fy = (l + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b_in / 200.0f;
    
    float x = 0.95047f * f_inv_device(fx);
    float y = f_inv_device(fy);
    float z = 1.08883f * f_inv_device(fz);
    
    // Convert XYZ to RGB
    r = 3.240479f * x - 1.537150f * y - 0.498535f * z;
    g = -0.969256f * x + 1.875992f * y + 0.041556f * z;
    blue = 0.055648f * x - 0.204043f * y + 1.057311f * z;
    
    // Clamp and convert to 0-255
    r = fmaxf(0.0f, fminf(1.0f, r)) * 255.0f;
    g = fmaxf(0.0f, fminf(1.0f, g)) * 255.0f;
    blue = fmaxf(0.0f, fminf(1.0f, blue)) * 255.0f;
}

__device__ float colorDistance_device(const float* color1, const float* color2) {
    float dx = color1[0] - color2[0];
    float dy = color1[1] - color2[1];
    float dz = color1[2] - color2[2];
    return dx * dx + dy * dy + dz * dz;  // Squared Euclidean distance
}

// Kernel: Assign each pixel to nearest centroid
__global__ void assignPixelsKernel(
    const unsigned char* imageData,
    int width,
    int height,
    int channels,
    const float* centroids,
    int k,
    int* assignments,
    int colorSpace  // 0 = RGB, 1 = LAB
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPixels = width * height;
    
    if (idx >= numPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    int pixelOffset = (y * width + x) * channels;
    
    float pixelColor[3];
    
    if (colorSpace == 0) {  // RGB
        pixelColor[0] = imageData[pixelOffset];
        pixelColor[1] = imageData[pixelOffset + 1];
        pixelColor[2] = imageData[pixelOffset + 2];
    } else {  // LAB
        rgbToLab_device(
            imageData[pixelOffset],
            imageData[pixelOffset + 1],
            imageData[pixelOffset + 2],
            pixelColor[0], pixelColor[1], pixelColor[2]
        );
    }
    
    // Find nearest centroid
    float minDist = 1e30f;
    int bestCentroid = 0;
    
    for (int i = 0; i < k; i++) {
        float dist = colorDistance_device(pixelColor, &centroids[i * 3]);
        if (dist < minDist) {
            minDist = dist;
            bestCentroid = i;
        }
    }
    
    assignments[idx] = bestCentroid;
}

// Kernel: Update centroids using reduction
__global__ void updateCentroidsKernel(
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
        rgbToLab_device(
            imageData[pixelOffset],
            imageData[pixelOffset + 1],
            imageData[pixelOffset + 2],
            pixelColor[0], pixelColor[1], pixelColor[2]
        );
    }
    
    // Use atomic operations to accumulate
    atomicAdd(&newCentroids[cluster * 3 + 0], pixelColor[0]);
    atomicAdd(&newCentroids[cluster * 3 + 1], pixelColor[1]);
    atomicAdd(&newCentroids[cluster * 3 + 2], pixelColor[2]);
    atomicAdd(&counts[cluster], 1);
}

// Kernel: Compute error
__global__ void computeErrorKernel(
    const unsigned char* imageData,
    int width,
    int height,
    int channels,
    const float* centroids,
    const int* assignments,
    int k,
    int colorSpace,
    double* error
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
        rgbToLab_device(
            imageData[pixelOffset],
            imageData[pixelOffset + 1],
            imageData[pixelOffset + 2],
            pixelColor[0], pixelColor[1], pixelColor[2]
        );
    }
    
    float dist = colorDistance_device(pixelColor, &centroids[cluster * 3]);
    atomicAdd(error, (double)dist);
}

// Kernel: Generate segmented image
__global__ void generateSegmentedImageKernel(
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
        labToRgb_device(
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
        outputData[pixelOffset + 3] = inputData[pixelOffset + 3];  // Preserve alpha
    }
}

// Initialize centroids (same as CPU version)
std::vector<std::vector<float>> KMeansCUDA::initializeCentroids(
    const Image& image,
    int k,
    ColorSpace colorSpace
) {
    std::vector<std::vector<float>> centroids(k, std::vector<float>(3));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, image.width * image.height - 1);
    
    // RGB to Lab conversion helper (CPU version)
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
        } else { // LAB
            float r = pixel[0] / 255.0f;
            float g = pixel[1] / 255.0f;
            float b = pixel[2] / 255.0f;
            
            float x = 0.412453f * r + 0.357580f * g + 0.180423f * b;
            float y_val = 0.212671f * r + 0.715160f * g + 0.072169f * b;
            float z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
            
            x /= 0.95047f;
            z /= 1.08883f;
            
            float fx = f(x);
            float fy = f(y_val);
            float fz = f(z);
            
            centroids[i][0] = 116.0f * fy - 16.0f;
            centroids[i][1] = 500.0f * (fx - fy);
            centroids[i][2] = 200.0f * (fy - fz);
        }
    }
    
    return centroids;
}

KMeansResult KMeansCUDA::segment(
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
    
    // Setup kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    int colorSpaceInt = (colorSpace == ColorSpace::RGB) ? 0 : 1;
    
    std::vector<double> errors;
    
    // Main K-means loop
    for (int iter = 0; iter < maxIterations; iter++) {
        // Assignment step
        assignPixelsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_imageData, image.width, image.height, image.channels,
            d_centroids, k, d_assignments, colorSpaceInt
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Compute error
        double h_error = 0.0;
        CUDA_CHECK(cudaMemset(d_error, 0, sizeof(double)));
        computeErrorKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_imageData, image.width, image.height, image.channels,
            d_centroids, d_assignments, k, colorSpaceInt, d_error
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_error, d_error, sizeof(double), cudaMemcpyDeviceToHost));
        errors.push_back(h_error);
        
        // Update step
        CUDA_CHECK(cudaMemset(d_newCentroids, 0, k * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, k * sizeof(int)));
        
        updateCentroidsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_imageData, image.width, image.height, image.channels,
            d_assignments, d_newCentroids, d_counts, k, colorSpaceInt
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Average centroids on device
        // Copy back to CPU for averaging (simpler than doing it on GPU)
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
            
            // Compute delta
            float dx = centroidsFlat[i * 3 + 0] - h_newCentroids[i * 3 + 0];
            float dy = centroidsFlat[i * 3 + 1] - h_newCentroids[i * 3 + 1];
            float dz = centroidsFlat[i * 3 + 2] - h_newCentroids[i * 3 + 2];
            float delta = dx * dx + dy * dy + dz * dz;
            maxDelta = std::max(maxDelta, delta);
        }
        
        // Update centroids
        centroidsFlat = h_newCentroids;
        CUDA_CHECK(cudaMemcpy(d_centroids, centroidsFlat.data(), 
                              k * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        if (maxDelta < threshold) {
            std::cout << "Converged at iteration " << iter + 1 << std::endl;
            break;
        }
    }
    
    // Generate segmented image
    generateSegmentedImageKernel<<<blocksPerGrid, threadsPerBlock>>>(
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
    
    // Copy assignments back (for debugging/verification)
    std::vector<int> h_assignments(numPixels);
    CUDA_CHECK(cudaMemcpy(h_assignments.data(), d_assignments, 
                          numPixels * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Convert centroids back to vector of vectors
    result.centroids.resize(k);
    for (int i = 0; i < k; i++) {
        result.centroids[i].resize(3);
        result.centroids[i][0] = centroidsFlat[i * 3 + 0];
        result.centroids[i][1] = centroidsFlat[i * 3 + 1];
        result.centroids[i][2] = centroidsFlat[i * 3 + 2];
    }
    result.assignments = h_assignments;
    result.iterations = errors.size();
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

