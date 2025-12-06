#include <iostream>
#include <string>
#include <vector>
#include "image_loader.h"
#include "kmeans_cpu.h"
#include "timer.h"

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <image_path> <k> <color_space> [output_path]\n";
    std::cout << "  k: number of clusters (3-12)\n";
    std::cout << "  color_space: 'rgb' or 'lab'\n";
    std::cout << "  output_path: optional output file (default: output_segmented.png)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string imagePath = argv[1];
    int k = std::stoi(argv[2]);
    std::string colorSpaceStr = argv[3];
    std::string outputPath = (argc > 4) ? argv[4] : "output_segmented.png";
    
    if (k < 3 || k > 12) {
        std::cerr << "Error: k must be between 3 and 12" << std::endl;
        return 1;
    }
    
    ColorSpace colorSpace;
    if (colorSpaceStr == "rgb" || colorSpaceStr == "RGB") {
        colorSpace = ColorSpace::RGB;
    } else if (colorSpaceStr == "lab" || colorSpaceStr == "LAB") {
        colorSpace = ColorSpace::LAB;
    } else {
        std::cerr << "Error: color_space must be 'rgb' or 'lab'" << std::endl;
        return 1;
    }
    
    std::cout << "Loading image: " << imagePath << std::endl;
    Image image = ImageLoader::loadImage(imagePath);
    
    if (image.data.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }
    
    std::cout << "Image loaded: " << image.width << "x" << image.height 
              << " (" << image.channels << " channels)" << std::endl;
    std::cout << "Running K-means with k=" << k 
              << ", color space=" << colorSpaceStr << std::endl;
    
    // Run K-means
    Timer totalTimer;
    KMeansResult result = KMeansCPU::segment(image, k, colorSpace);
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Final error: " << result.finalError << std::endl;
    std::cout << "Runtime (CPU single-thread): " << result.runtimeMs << " ms" << std::endl;
    std::cout << "Total time: " << totalTimer.elapsedMs() << " ms" << std::endl;
    
    // Save segmented image
    std::cout << "\nSaving segmented image to: " << outputPath << std::endl;
    if (ImageLoader::saveImage(result.segmentedImage, outputPath)) {
        std::cout << "Successfully saved segmented image!" << std::endl;
    } else {
        std::cerr << "Failed to save segmented image" << std::endl;
        return 1;
    }
    
    // Print centroids
    std::cout << "\nFinal centroids:" << std::endl;
    for (size_t i = 0; i < result.centroids.size(); i++) {
        if (colorSpace == ColorSpace::RGB) {
            std::cout << "  Cluster " << i << ": RGB(" 
                      << (int)result.centroids[i][0] << ", "
                      << (int)result.centroids[i][1] << ", "
                      << (int)result.centroids[i][2] << ")" << std::endl;
        } else {
            std::cout << "  Cluster " << i << ": Lab(" 
                      << result.centroids[i][0] << ", "
                      << result.centroids[i][1] << ", "
                      << result.centroids[i][2] << ")" << std::endl;
        }
    }
    
    return 0;
}

