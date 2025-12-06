#include "image_loader.h"
#include <iostream>
#include <fstream>
#include <cstring>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#else
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

Image ImageLoader::loadImage(const std::string& filepath) {
    Image img;
    
#ifdef USE_OPENCV
    cv::Mat cvImg = cv::imread(filepath, cv::IMREAD_COLOR);
    if (cvImg.empty()) {
        std::cerr << "Error: Could not load image " << filepath << std::endl;
        return img;
    }
    
    img.width = cvImg.cols;
    img.height = cvImg.rows;
    img.channels = 3;
    img.data.resize(img.width * img.height * img.channels);
    
    cv::Mat rgbImg;
    cv::cvtColor(cvImg, rgbImg, cv::COLOR_BGR2RGB);
    std::memcpy(img.data.data(), rgbImg.data, img.data.size());
#else
    int channels;
    unsigned char* data = stbi_load(filepath.c_str(), &img.width, &img.height, &channels, 0);
    
    if (!data) {
        std::cerr << "Error: Could not load image " << filepath << std::endl;
        return img;
    }
    
    img.channels = channels;
    if (channels == 1) {
        // Convert grayscale to RGB
        img.channels = 3;
        img.data.resize(img.width * img.height * 3);
        for (int i = 0; i < img.width * img.height; i++) {
            img.data[i * 3] = data[i];
            img.data[i * 3 + 1] = data[i];
            img.data[i * 3 + 2] = data[i];
        }
        stbi_image_free(data);
    } else {
        img.data.assign(data, data + img.width * img.height * channels);
        stbi_image_free(data);
    }
#endif
    
    return img;
}

bool ImageLoader::saveImage(const Image& image, const std::string& filepath) {
#ifdef USE_OPENCV
    cv::Mat cvImg(image.height, image.width, CV_8UC3, (void*)image.data.data());
    cv::Mat bgrImg;
    cv::cvtColor(cvImg, bgrImg, cv::COLOR_RGB2BGR);
    return cv::imwrite(filepath, bgrImg);
#else
    int success = stbi_write_png(
        filepath.c_str(),
        image.width,
        image.height,
        image.channels,
        image.data.data(),
        image.width * image.channels
    );
    return success != 0;
#endif
}

