#include "kmeans_cpu.h"
#include "timer.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <iostream>

// RGB to Lab conversion helper functions
static float f(float t) {
    return (t > 0.008856f) ? std::pow(t, 1.0f/3.0f) : (7.787f * t + 16.0f/116.0f);
}

static float f_inv(float t) {
    return (t > 0.206897f) ? std::pow(t, 3.0f) : ((t - 16.0f/116.0f) / 7.787f);
}

void KMeansCPU::rgbToLab(float r, float g, float b, float& l, float& a, float& b_out) {
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
    float fx = f(x);
    float fy = f(y);
    float fz = f(z);
    
    l = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    b_out = 200.0f * (fy - fz);
}

void KMeansCPU::labToRgb(float l, float a, float b_in, float& r, float& g, float& blue) {
    // Convert Lab to XYZ
    float fy = (l + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b_in / 200.0f;
    
    float x = 0.95047f * f_inv(fx);
    float y = f_inv(fy);
    float z = 1.08883f * f_inv(fz);
    
    // Convert XYZ to RGB
    r = 3.240479f * x - 1.537150f * y - 0.498535f * z;
    g = -0.969256f * x + 1.875992f * y + 0.041556f * z;
    blue = 0.055648f * x - 0.204043f * y + 1.057311f * z;
    
    // Clamp and convert to 0-255
    r = std::max(0.0f, std::min(1.0f, r)) * 255.0f;
    g = std::max(0.0f, std::min(1.0f, g)) * 255.0f;
    blue = std::max(0.0f, std::min(1.0f, blue)) * 255.0f;
}

std::vector<std::vector<float>> KMeansCPU::initializeCentroids(
    const Image& image,
    int k,
    ColorSpace colorSpace
) {
    std::vector<std::vector<float>> centroids(k, std::vector<float>(3));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, image.width * image.height - 1);
    
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
            rgbToLab(pixel[0], pixel[1], pixel[2],
                     centroids[i][0], centroids[i][1], centroids[i][2]);
        }
    }
    
    return centroids;
}

float KMeansCPU::colorDistance(
    const float* color1,
    const float* color2,
    ColorSpace /* colorSpace */
) {
    float dx = color1[0] - color2[0];
    float dy = color1[1] - color2[1];
    float dz = color1[2] - color2[2];
    return dx * dx + dy * dy + dz * dz;  // Squared Euclidean distance
}

void KMeansCPU::assignPixels(
    const Image& image,
    const std::vector<std::vector<float>>& centroids,
    std::vector<int>& assignments,
    ColorSpace colorSpace
) {
    int numPixels = image.width * image.height;
    assignments.resize(numPixels);
    
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            const unsigned char* pixel = image.getPixel(x, y);
            float pixelColor[3];
            
            if (colorSpace == ColorSpace::RGB) {
                pixelColor[0] = pixel[0];
                pixelColor[1] = pixel[1];
                pixelColor[2] = pixel[2];
            } else { // LAB
                rgbToLab(pixel[0], pixel[1], pixel[2],
                         pixelColor[0], pixelColor[1], pixelColor[2]);
            }
            
            // Find nearest centroid
            float minDist = std::numeric_limits<float>::max();
            int bestCentroid = 0;
            
            for (size_t k = 0; k < centroids.size(); k++) {
                float dist = colorDistance(pixelColor, centroids[k].data(), colorSpace);
                if (dist < minDist) {
                    minDist = dist;
                    bestCentroid = k;
                }
            }
            
            assignments[y * image.width + x] = bestCentroid;
        }
    }
}

std::vector<std::vector<float>> KMeansCPU::updateCentroids(
    const Image& image,
    const std::vector<int>& assignments,
    int k,
    ColorSpace colorSpace
) {
    std::vector<std::vector<float>> newCentroids(k, std::vector<float>(3, 0.0f));
    std::vector<int> counts(k, 0);
    
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            int idx = y * image.width + x;
            int cluster = assignments[idx];
            const unsigned char* pixel = image.getPixel(x, y);
            
            float pixelColor[3];
            if (colorSpace == ColorSpace::RGB) {
                pixelColor[0] = pixel[0];
                pixelColor[1] = pixel[1];
                pixelColor[2] = pixel[2];
            } else { // LAB
                rgbToLab(pixel[0], pixel[1], pixel[2],
                         pixelColor[0], pixelColor[1], pixelColor[2]);
            }
            
            newCentroids[cluster][0] += pixelColor[0];
            newCentroids[cluster][1] += pixelColor[1];
            newCentroids[cluster][2] += pixelColor[2];
            counts[cluster]++;
        }
    }
    
    // Average the centroids
    for (int i = 0; i < k; i++) {
        if (counts[i] > 0) {
            newCentroids[i][0] /= counts[i];
            newCentroids[i][1] /= counts[i];
            newCentroids[i][2] /= counts[i];
        }
    }
    
    return newCentroids;
}

double KMeansCPU::computeError(
    const Image& image,
    const std::vector<std::vector<float>>& centroids,
    const std::vector<int>& assignments,
    ColorSpace colorSpace
) {
    double totalError = 0.0;
    
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            int idx = y * image.width + x;
            int cluster = assignments[idx];
            const unsigned char* pixel = image.getPixel(x, y);
            
            float pixelColor[3];
            if (colorSpace == ColorSpace::RGB) {
                pixelColor[0] = pixel[0];
                pixelColor[1] = pixel[1];
                pixelColor[2] = pixel[2];
            } else { // LAB
                rgbToLab(pixel[0], pixel[1], pixel[2],
                         pixelColor[0], pixelColor[1], pixelColor[2]);
            }
            
            float dist = colorDistance(pixelColor, centroids[cluster].data(), colorSpace);
            totalError += dist;
        }
    }
    
    return totalError;
}

KMeansResult KMeansCPU::segment(
    const Image& image,
    int k,
    ColorSpace colorSpace,
    int maxIterations,
    float threshold
) {
    Timer timer;
    KMeansResult result;
    
    // Initialize centroids
    std::vector<std::vector<float>> centroids = initializeCentroids(image, k, colorSpace);
    std::vector<int> assignments;
    
    std::vector<double> errors;  // Track error per iteration
    
    // Main K-means loop
    for (int iter = 0; iter < maxIterations; iter++) {
        // Assignment step
        assignPixels(image, centroids, assignments, colorSpace);
        
        // Compute error
        double error = computeError(image, centroids, assignments, colorSpace);
        errors.push_back(error);
        
        // Update step
        std::vector<std::vector<float>> newCentroids = updateCentroids(
            image, assignments, k, colorSpace
        );
        
        // Check convergence
        float maxDelta = 0.0f;
        for (int i = 0; i < k; i++) {
            float delta = colorDistance(centroids[i].data(), newCentroids[i].data(), colorSpace);
            maxDelta = std::max(maxDelta, delta);
        }
        
        centroids = newCentroids;
        
        if (maxDelta < threshold) {
            std::cout << "Converged at iteration " << iter + 1 << std::endl;
            break;
        }
    }
    
    result.runtimeMs = timer.elapsedMs();
    result.iterations = errors.size();
    result.finalError = errors.empty() ? 0.0 : errors.back();
    result.centroids = centroids;
    result.assignments = assignments;
    
    // Create segmented image
    result.segmentedImage.width = image.width;
    result.segmentedImage.height = image.height;
    result.segmentedImage.channels = image.channels;
    result.segmentedImage.data.resize(image.width * image.height * image.channels);
    
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            int idx = y * image.width + x;
            int cluster = assignments[idx];
            unsigned char* outPixel = result.segmentedImage.getPixel(x, y);
            
            if (colorSpace == ColorSpace::RGB) {
                outPixel[0] = (unsigned char)std::round(centroids[cluster][0]);
                outPixel[1] = (unsigned char)std::round(centroids[cluster][1]);
                outPixel[2] = (unsigned char)std::round(centroids[cluster][2]);
            } else { // LAB
                float r, g, b;
                labToRgb(centroids[cluster][0], centroids[cluster][1], centroids[cluster][2],
                         r, g, b);
                outPixel[0] = (unsigned char)std::round(r);
                outPixel[1] = (unsigned char)std::round(g);
                outPixel[2] = (unsigned char)std::round(b);
            }
            
            if (image.channels == 4) {
                outPixel[3] = image.getPixel(x, y)[3];  // Preserve alpha
            }
        }
    }
    
    return result;
}

