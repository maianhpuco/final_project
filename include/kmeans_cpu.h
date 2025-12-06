#ifndef KMEANS_CPU_H
#define KMEANS_CPU_H

#include "image_loader.h"
#include <vector>

enum class ColorSpace {
    RGB,
    LAB
};

struct KMeansResult {
    Image segmentedImage;
    std::vector<std::vector<float>> centroids;  // K centroids, each with 3 components
    std::vector<int> assignments;  // Assignment for each pixel
    int iterations;
    double finalError;
    double runtimeMs;
};

class KMeansCPU {
public:
    // Run K-means clustering on image
    static KMeansResult segment(
        const Image& image,
        int k,
        ColorSpace colorSpace,
        int maxIterations = 100,
        float threshold = 0.01f
    );
    
private:
    // Convert RGB to Lab color space
    static void rgbToLab(float r, float g, float b, float& l, float& a, float& b_out);
    static void labToRgb(float l, float a, float b_in, float& r, float& g, float& blue);
    
    // Initialize centroids randomly
    static std::vector<std::vector<float>> initializeCentroids(
        const Image& image,
        int k,
        ColorSpace colorSpace
    );
    
    // Assignment step: assign each pixel to nearest centroid
    static void assignPixels(
        const Image& image,
        const std::vector<std::vector<float>>& centroids,
        std::vector<int>& assignments,
        ColorSpace colorSpace
    );
    
    // Update step: recompute centroids based on assignments
    static std::vector<std::vector<float>> updateCentroids(
        const Image& image,
        const std::vector<int>& assignments,
        int k,
        ColorSpace colorSpace
    );
    
    // Compute total error (sum of squared distances)
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

#endif // KMEANS_CPU_H

