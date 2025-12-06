cmake_minimum_required(VERSION 3.15)
project(KMeansImageSegmentation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(kmeans_segmentation
    src/main.cpp
    src/kmeans_cpu.cpp
    src/image_loader.cpp
    src/timer.cpp
)

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

# For OpenMP (will be used later)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kmeans_segmentation PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found")
endif()

