#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "image_loader.h"
#include "kmeans_cpu.h"
#include "kmeans_omp.h"
#include "timer.h"

#ifndef CUDA_AVAILABLE
#define CUDA_AVAILABLE 0
#endif

#if CUDA_AVAILABLE
#include "kmeans_cuda.h"
#include "kmeans_cuda_shared.h"
#endif

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <image_path> <k> <color_space> [options]\n";
    std::cout << "  k: number of clusters (3-100)\n";
    std::cout << "  color_space: 'rgb' or 'lab'\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --version <cpu|omp";
    if (CUDA_AVAILABLE) {
        std::cout << "|cuda|cuda_shared";
    }
    std::cout << "|benchmark>  : Choose implementation (default: cpu)\n";
    std::cout << "  --output <path>                : Output file (default: output_segmented.png)\n";
    std::cout << "  --threads <n>                  : Number of OpenMP threads (default: auto)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << programName << " image.png 5 rgb --version cpu\n";
    std::cout << "  " << programName << " image.png 5 rgb --version omp --threads 4\n";
    if (CUDA_AVAILABLE) {
        std::cout << "  " << programName << " image.png 5 rgb --version cuda\n";
        std::cout << "  " << programName << " image.png 5 rgb --version cuda_shared\n";
    }
    std::cout << "  " << programName << " image.png 5 rgb --version benchmark\n";
}

bool compareResults(const KMeansResult& result1, const KMeansResult& result2, double tolerance = 1e-3) {
    // Compare centroids (order may differ, so we need to match them)
    if (result1.centroids.size() != result2.centroids.size()) {
        return false;
    }
    
    // Compare final error (should be very similar)
    if (std::abs(result1.finalError - result2.finalError) > tolerance * std::max(result1.finalError, result2.finalError)) {
        return false;
    }
    
    // Compare iterations (should be same)
    if (result1.iterations != result2.iterations) {
        return false;
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string imagePath = argv[1];
    int k = std::stoi(argv[2]);
    std::string colorSpaceStr = argv[3];
    std::string version = "cpu";
    std::string outputPath = "output_segmented.png";
    int numThreads = 0;  // 0 = auto
    
    // Parse optional arguments
    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--version" && i + 1 < argc) {
            version = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        }
    }
    
    if (k < 3 || k > 100) {
        std::cerr << "Error: k must be between 3 and 100" << std::endl;
        return 1;
    }
    
    // Validate version
    if ((version == "cuda" || version == "cuda_shared") && !CUDA_AVAILABLE) {
        std::cerr << "Error: CUDA version is not available (CUDA not found during compilation)" << std::endl;
        return 1;
    }
    
    if (version != "cpu" && version != "omp" && version != "benchmark" && 
        version != "cuda" && version != "cuda_shared") {
        std::cerr << "Error: version must be one of: cpu, omp";
        if (CUDA_AVAILABLE) {
            std::cerr << ", cuda, cuda_shared";
        }
        std::cerr << ", benchmark" << std::endl;
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
    
    if (version == "benchmark") {
        // Benchmark all available versions
        std::cout << "\n=== BENCHMARKING MODE ===" << std::endl;
        std::cout << "Comparing all available implementations\n" << std::endl;
        
        const int numRuns = 3;  // Run each version multiple times for average
        std::vector<double> cpuTimes, ompTimes;
        
        // Run CPU version
        std::cout << "Running CPU single-threaded version (" << numRuns << " runs)..." << std::endl;
        KMeansResult cpuResult;
        for (int run = 0; run < numRuns; run++) {
            Timer runTimer;
            cpuResult = KMeansCPU::segment(image, k, colorSpace);
            double time = runTimer.elapsedMs();
            cpuTimes.push_back(time);
            std::cout << "  Run " << (run + 1) << ": " << std::fixed << std::setprecision(2) 
                      << time << " ms" << std::endl;
        }
        
        // Run OpenMP version
        std::cout << "\nRunning OpenMP multi-threaded version (" << numRuns << " runs)..." << std::endl;
        KMeansResult ompResult;
        for (int run = 0; run < numRuns; run++) {
            Timer runTimer;
            ompResult = KMeansOMP::segment(image, k, colorSpace, 100, 0.01f, numThreads);
            double time = runTimer.elapsedMs();
            ompTimes.push_back(time);
            std::cout << "  Run " << (run + 1) << ": " << std::fixed << std::setprecision(2) 
                      << time << " ms" << std::endl;
        }
        
        // Calculate statistics
        double cpuAvg = 0.0, ompAvg = 0.0;
        for (double t : cpuTimes) cpuAvg += t;
        for (double t : ompTimes) ompAvg += t;
        cpuAvg /= numRuns;
        ompAvg /= numRuns;
        
        double cpuMin = *std::min_element(cpuTimes.begin(), cpuTimes.end());
        double cpuMax = *std::max_element(cpuTimes.begin(), cpuTimes.end());
        double ompMin = *std::min_element(ompTimes.begin(), ompTimes.end());
        double ompMax = *std::max_element(ompTimes.begin(), ompTimes.end());
        
        double speedup = cpuAvg / ompAvg;
        
        // Print results
        std::cout << "\n=== BENCHMARK RESULTS ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nCPU Single-threaded:" << std::endl;
        std::cout << "  Average: " << cpuAvg << " ms" << std::endl;
        std::cout << "  Min:     " << cpuMin << " ms" << std::endl;
        std::cout << "  Max:     " << cpuMax << " ms" << std::endl;
        std::cout << "  Iterations: " << cpuResult.iterations << std::endl;
        std::cout << "  Final error: " << std::scientific << std::setprecision(6) 
                  << cpuResult.finalError << std::endl;
        
        std::cout << "\nOpenMP Multi-threaded:" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Average: " << ompAvg << " ms" << std::endl;
        std::cout << "  Min:     " << ompMin << " ms" << std::endl;
        std::cout << "  Max:     " << ompMax << " ms" << std::endl;
        std::cout << "  Iterations: " << ompResult.iterations << std::endl;
        std::cout << "  Final error: " << std::scientific << std::setprecision(6) 
                  << ompResult.finalError << std::endl;
        
        std::cout << "\nPerformance:" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Speedup: " << speedup << "x" << std::endl;
        std::cout << "  Improvement: " << ((cpuAvg - ompAvg) / cpuAvg * 100.0) << "% faster" << std::endl;
        
        // Verify results match
        std::cout << "\nVerification:" << std::endl;
        if (compareResults(cpuResult, ompResult)) {
            std::cout << "  ✓ Results match (within tolerance)" << std::endl;
        } else {
            std::cout << "  ⚠ Results differ (may be due to different initialization)" << std::endl;
            std::cout << "    CPU error: " << std::scientific << std::setprecision(6) 
                      << cpuResult.finalError << std::endl;
            std::cout << "    OMP error: " << ompResult.finalError << std::endl;
        }
        
        // Run CUDA version if available
        std::vector<double> cudaTimes;
        KMeansResult cudaResult;
        std::vector<double> cudaSharedTimes;
        KMeansResult cudaSharedResult;
#if CUDA_AVAILABLE
        if (true) {
            std::cout << "\nRunning CUDA GPU version (Global Memory) (" << numRuns << " runs)..." << std::endl;
            for (int run = 0; run < numRuns; run++) {
                Timer runTimer;
                cudaResult = KMeansCUDA::segment(image, k, colorSpace);
                double time = runTimer.elapsedMs();
                cudaTimes.push_back(time);
                std::cout << "  Run " << (run + 1) << ": " << std::fixed << std::setprecision(2) 
                          << time << " ms" << std::endl;
            }
            
            double cudaAvg = 0.0;
            for (double t : cudaTimes) cudaAvg += t;
            cudaAvg /= numRuns;
            double cudaMin = *std::min_element(cudaTimes.begin(), cudaTimes.end());
            double cudaMax = *std::max_element(cudaTimes.begin(), cudaTimes.end());
            
            std::cout << "\nCUDA GPU (Global Memory):" << std::endl;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  Average: " << cudaAvg << " ms" << std::endl;
            std::cout << "  Min:     " << cudaMin << " ms" << std::endl;
            std::cout << "  Max:     " << cudaMax << " ms" << std::endl;
            std::cout << "  Iterations: " << cudaResult.iterations << std::endl;
            std::cout << "  Final error: " << std::scientific << std::setprecision(6) 
                      << cudaResult.finalError << std::endl;
            
            double speedupCuda = cpuAvg / cudaAvg;
            std::cout << "\n  Speedup vs CPU: " << speedupCuda << "x" << std::endl;
            std::cout << "  Speedup vs OMP: " << (ompAvg / cudaAvg) << "x" << std::endl;
            
            // Verify CUDA results
            if (compareResults(cpuResult, cudaResult)) {
                std::cout << "  ✓ CUDA results match CPU (within tolerance)" << std::endl;
            } else {
                std::cout << "  ⚠ CUDA results differ from CPU" << std::endl;
            }
            
            // Run CUDA Shared Memory version
            std::cout << "\nRunning CUDA GPU version (Shared Memory) (" << numRuns << " runs)..." << std::endl;
            for (int run = 0; run < numRuns; run++) {
                Timer runTimer;
                cudaSharedResult = KMeansCUDAShared::segment(image, k, colorSpace);
                double time = runTimer.elapsedMs();
                cudaSharedTimes.push_back(time);
                std::cout << "  Run " << (run + 1) << ": " << std::fixed << std::setprecision(2) 
                          << time << " ms" << std::endl;
            }
            
            double cudaSharedAvg = 0.0;
            for (double t : cudaSharedTimes) cudaSharedAvg += t;
            cudaSharedAvg /= numRuns;
            double cudaSharedMin = *std::min_element(cudaSharedTimes.begin(), cudaSharedTimes.end());
            double cudaSharedMax = *std::max_element(cudaSharedTimes.begin(), cudaSharedTimes.end());
            
            std::cout << "\nCUDA GPU (Shared Memory):" << std::endl;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  Average: " << cudaSharedAvg << " ms" << std::endl;
            std::cout << "  Min:     " << cudaSharedMin << " ms" << std::endl;
            std::cout << "  Max:     " << cudaSharedMax << " ms" << std::endl;
            std::cout << "  Iterations: " << cudaSharedResult.iterations << std::endl;
            std::cout << "  Final error: " << std::scientific << std::setprecision(6) 
                      << cudaSharedResult.finalError << std::endl;
            
            double speedupCudaShared = cpuAvg / cudaSharedAvg;
            std::cout << "\n  Speedup vs CPU: " << speedupCudaShared << "x" << std::endl;
            std::cout << "  Speedup vs OMP: " << (ompAvg / cudaSharedAvg) << "x" << std::endl;
            std::cout << "  Speedup vs CUDA Global: " << (cudaAvg / cudaSharedAvg) << "x" << std::endl;
            
            // Verify CUDA Shared results
            if (compareResults(cpuResult, cudaSharedResult)) {
                std::cout << "  ✓ CUDA Shared results match CPU (within tolerance)" << std::endl;
            } else {
                std::cout << "  ⚠ CUDA Shared results differ from CPU" << std::endl;
            }
        }
#else
        // CUDA not available
        if (false) {
            // This block never executes, but keeps code structure
        }
#endif
        
        // Save the best result (CUDA Shared if available, then CUDA Global, then OMP)
#if CUDA_AVAILABLE
        KMeansResult& bestResult = (!cudaSharedTimes.empty()) ? cudaSharedResult : 
                                   (!cudaTimes.empty()) ? cudaResult : ompResult;
        std::string bestVersion = (!cudaSharedTimes.empty()) ? "CUDA Shared" : 
                                  (!cudaTimes.empty()) ? "CUDA Global" : "OpenMP";
#else
        KMeansResult& bestResult = ompResult;
        std::string bestVersion = "OpenMP";
#endif
        std::cout << "\nSaving segmented image (from " << bestVersion << ") to: " << outputPath << std::endl;
        if (ImageLoader::saveImage(bestResult.segmentedImage, outputPath)) {
            std::cout << "Successfully saved segmented image!" << std::endl;
        } else {
            std::cerr << "Failed to save segmented image" << std::endl;
            return 1;
        }
        
    } else if (version == "cpu") {
        // Run CPU version
        Timer totalTimer;
        KMeansResult result = KMeansCPU::segment(image, k, colorSpace);
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Final error: " << result.finalError << std::endl;
        std::cout << "Runtime (CPU single-thread): " << std::fixed << std::setprecision(2) 
                  << result.runtimeMs << " ms" << std::endl;
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
                          << std::fixed << std::setprecision(2)
                          << result.centroids[i][0] << ", "
                          << result.centroids[i][1] << ", "
                          << result.centroids[i][2] << ")" << std::endl;
            }
        }
        
    } else if (version == "omp") {
        // Run OpenMP version
        Timer totalTimer;
        KMeansResult result = KMeansOMP::segment(image, k, colorSpace, 100, 0.01f, numThreads);
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Final error: " << result.finalError << std::endl;
        std::cout << "Runtime (OpenMP multi-thread): " << std::fixed << std::setprecision(2) 
                  << result.runtimeMs << " ms" << std::endl;
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
                          << std::fixed << std::setprecision(2)
                          << result.centroids[i][0] << ", "
                          << result.centroids[i][1] << ", "
                          << result.centroids[i][2] << ")" << std::endl;
            }
        }
#if CUDA_AVAILABLE
    } else if (version == "cuda") {
        // Run CUDA version (global memory)
        Timer totalTimer;
        KMeansResult result = KMeansCUDA::segment(image, k, colorSpace);
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Final error: " << result.finalError << std::endl;
        std::cout << "Runtime (CUDA GPU - Global Memory): " << std::fixed << std::setprecision(2) 
                  << result.runtimeMs << " ms" << std::endl;
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
                          << std::fixed << std::setprecision(2)
                          << result.centroids[i][0] << ", "
                          << result.centroids[i][1] << ", "
                          << result.centroids[i][2] << ")" << std::endl;
            }
        }
    } else if (version == "cuda_shared") {
        // Run CUDA version (shared memory)
        Timer totalTimer;
        KMeansResult result = KMeansCUDAShared::segment(image, k, colorSpace);
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Final error: " << result.finalError << std::endl;
        std::cout << "Runtime (CUDA GPU - Shared Memory): " << std::fixed << std::setprecision(2) 
                  << result.runtimeMs << " ms" << std::endl;
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
                          << std::fixed << std::setprecision(2)
                          << result.centroids[i][0] << ", "
                          << result.centroids[i][1] << ", "
                          << result.centroids[i][2] << ")" << std::endl;
            }
        }
#endif
    }
    
    return 0;
}
