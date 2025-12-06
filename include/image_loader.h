#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <string>
#include <vector>

struct Image {
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data;  // Raw pixel data
    
    // Get pixel at (x, y) - returns pointer to RGB/RGBA values
    unsigned char* getPixel(int x, int y) {
        return &data[(y * width + x) * channels];
    }
    
    const unsigned char* getPixel(int x, int y) const {
        return &data[(y * width + x) * channels];
    }
};

class ImageLoader {
public:
    static Image loadImage(const std::string& filepath);
    static bool saveImage(const Image& image, const std::string& filepath);
};

#endif // IMAGE_LOADER_H

