#include "convolution.hpp"
#include "image_io.hpp"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: ./photocuda <image> <filter>\n lil bro do better next "
                 "time.👍";
    return 1;
  }

  std::string input_path = argv[1];
  std::string filter_str = argv[2];

  if (filter_str != "gb") {
    std::cerr << "Sybau lil bro🥀 Unsupported filter.";
    return 1;
  }

  Image input = load_image_grayscale(input_path);
  Image output;
  output.width = input.width;
  output.height = input.height;
  output.channels = 1;
  output.data = new float[input.width * input.height * 3];

  apply_filter(input, output, FilterType::GAUSSIAN);
  save_image_grayscale("output_gray.png", output);

  free_image(input);
  free_image(output);
}
