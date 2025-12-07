#ifndef KMEANS_CUDA_SHARED_H
#define KMEANS_CUDA_SHARED_H

#include "image_loader.h"
#include "kmeans_cpu.h"  // Reuse ColorSpace and KMeansResult

class KMeansCUDAShared {
public:
    // Run K-means clustering on image using CUDA with shared memory optimization
    static KMeansResult segment(
        const Image& image,
        int k,
        ColorSpace colorSpace,
        int maxIterations = 100,
        float threshold = 0.01f
    );
    
private:
    // Initialize centroids randomly (on CPU, same as CPU version)
    static std::vector<std::vector<float>> initializeCentroids(
        const Image& image,
        int k,
        ColorSpace colorSpace
    );
};

#endif // KMEANS_CUDA_SHARED_H

