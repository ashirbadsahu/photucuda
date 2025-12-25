#pragma once
#include <string>

struct Image {
    int width;
    int height;
    int channels;
    float *data;
};


Image load_image_grayscale(const std::string& path);
void save_image_grayscale(const std::string& path, const Image& img);
void free_image(Image& img);
