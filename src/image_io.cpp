#include "image_io.hpp"
#include <stdexcept>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

Image load_image_grayscale(const std::string& path){
    Image img{};
    int w, h, c;
    unsigned char* raw = stbi_load(
        path.c_str(),
        &w,
        &h,
        &c,
        3
    );

    if (!raw)
        throw std::runtime_error("Failed to load image 🥀");
    

    img.width = w;
    img.height = h;
    img.channels = 1;

    int size = w * h;
    img.data = new float[size];

    for(int i = 0; i < size; i++){
        float r = raw[3*i + 0];
        float g = raw[3*i + 1];
        float b = raw[3*i + 2];

        img.data[i] = (0.299f*r + 0.587f*g + 0.144f*b) / 255.0f;
    }

    stbi_image_free(raw);
    return img;
}

void save_image_grayscale(const std::string& path, const Image& img){
    int size = img.width * img.height;
    unsigned char *out = new unsigned char[size];

    for (int i = 0; i < size; i++){
        float v = std::clamp(img.data[i], 0.0f, 1.0f);
        out[i] = static_cast<unsigned char>(v * 255.0f);
    }

    stbi_write_png(
        path.c_str(),
        img.width,
        img.height,
        1,
        out,
        img.width * img.channels
    );

    delete[] out;
}

void free_image(Image& img){
    delete[] img.data;
    img.data = nullptr;
}