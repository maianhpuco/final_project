#ifndef KMEANS_OMP_H
#define KMEANS_OMP_H

#include "image_loader.h"
#include "kmeans_cpu.h"  // Reuse ColorSpace and KMeansResult
#include <vector>

class KMeansOMP {
public:
    // Run K-means clustering on image using OpenMP
    static KMeansResult segment(
        const Image& image,
        int k,
        ColorSpace colorSpace,
        int maxIterations = 100,
        float threshold = 0.01f,
        int numThreads = 0  // 0 = use default (OMP_NUM_THREADS)
    );
    
private:
    // Convert RGB to Lab color space (reuse from KMeansCPU)
    static void rgbToLab(float r, float g, float b, float& l, float& a, float& b_out);
    static void labToRgb(float l, float a, float b_in, float& r, float& g, float& blue);
    
    // Initialize centroids randomly
    static std::vector<std::vector<float>> initializeCentroids(
        const Image& image,
        int k,
        ColorSpace colorSpace
    );
    
    // Assignment step: assign each pixel to nearest centroid (parallelized)
    static void assignPixels(
        const Image& image,
        const std::vector<std::vector<float>>& centroids,
        std::vector<int>& assignments,
        ColorSpace colorSpace
    );
    
    // Update step: recompute centroids based on assignments (parallelized)
    static std::vector<std::vector<float>> updateCentroids(
        const Image& image,
        const std::vector<int>& assignments,
        int k,
        ColorSpace colorSpace
    );
    
    // Compute total error (sum of squared distances) (parallelized)
    static double computeError(
        const Image& image,
        const std::vector<std::vector<float>>& centroids,
        const std::vector<int>& assignments,
        ColorSpace colorSpace
    );
    
    // Compute distance between two colors
    static float colorDistance(
        const float* color1,
        const float* color2,
        ColorSpace colorSpace
    );
};

#endif // KMEANS_OMP_H

