cmake_minimum_required(VERSION 3.15)
project(KMeansImageSegmentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for CUDA before trying to enable it
find_program(CUDA_NVCC_EXECUTABLE nvcc PATHS
    ENV CUDA_PATH
    ENV CUDA_HOME
    ENV CUDA_ROOT
    /usr/local/cuda
    /opt/cuda
    PATH_SUFFIXES bin bin64
    NO_DEFAULT_PATH
)

if(CUDA_NVCC_EXECUTABLE)
    enable_language(CUDA)
    message(STATUS "CUDA compiler found: ${CUDA_NVCC_EXECUTABLE}")
else()
    message(STATUS "CUDA compiler not found - GPU version will not be available")
endif()

# Add executable
add_executable(kmeans_segmentation
    src/main.cpp
    src/kmeans_cpu.cpp
    src/kmeans_omp.cpp
    src/image_loader.cpp
    src/timer.cpp
)

# Add CUDA source if CUDA compiler is available
if(CMAKE_CUDA_COMPILER)
    target_sources(kmeans_segmentation PRIVATE src/kmeans_cuda.cu src/kmeans_cuda_shared.cu)
    set_property(TARGET kmeans_segmentation PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(kmeans_segmentation PRIVATE CUDA_AVAILABLE=1)
    message(STATUS "CUDA found - GPU version enabled (global and shared memory)")
else()
    target_compile_definitions(kmeans_segmentation PRIVATE CUDA_AVAILABLE=0)
    message(STATUS "CUDA not found - GPU version will not be available")
endif()

# Include directories
target_include_directories(kmeans_segmentation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/stb
)

# Link libraries (for image I/O)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    target_link_libraries(kmeans_segmentation ${OpenCV_LIBS})
    target_compile_definitions(kmeans_segmentation PRIVATE USE_OPENCV)
    message(STATUS "OpenCV found - using for image I/O")
else()
    message(STATUS "OpenCV not found - using stb_image")
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kmeans_segmentation PRIVATE -Wall -Wextra -O3)
endif()

# CUDA compiler flags
if(CMAKE_CUDA_COMPILER)
    set_property(TARGET kmeans_segmentation PROPERTY CUDA_ARCHITECTURES native)
    target_compile_options(kmeans_segmentation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
endif()

# For OpenMP (required for multi-threaded version)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kmeans_segmentation PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - multi-threaded version enabled")
else()
    message(WARNING "OpenMP not found - multi-threaded version will not work")
endif()

