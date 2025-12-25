#pragma once
#include "image_io.hpp"

enum class FilterType{
    GAUSSIAN
};

void apply_filter(const Image& input, Image& output, FilterType filter);
