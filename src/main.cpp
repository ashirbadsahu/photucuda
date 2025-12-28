#include "convolution.hpp"
#include "image_io.hpp"
#include <iostream>
#include <string.h>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: ./photocuda <image> <filter>\n lil bro do better next "
                 "time.👍";
    return 1;
  }

  std::string input_path = argv[1];
  std::string filter_str = argv[2];

  FilterType filter;
  if (filter_str == "gb") {
    filter = FilterType::GAUSSIAN;
  } else if(filter_str == "hp"){
    filter = FilterType::HIGHPASS;
  } else{
    std::cerr<< "Sybau lil bro 🥀. Unsupported filter";
  }

  Image input = load_image_grayscale(input_path);
  Image output;
  output.width = input.width;
  output.height = input.height;
  output.channels = 1;
  output.data = new float[input.width * input.height];

  apply_filter(input, output, filter);

  std::string output_name = "output_gray_" + filter_str + ".png";
  save_image_grayscale(output_name.c_str(), output);

  free_image(input);
  free_image(output);
}
