
#include <iostream>
#include <vector>

#include "tiny_cnn/tiny_cnn.h"

using tiny_cnn::activation::tan_h;
using tiny_cnn::activation::identity;

tiny_cnn::vec_t max(tiny_cnn::vec_t vec) {
  double max = 0;
  for (auto v : vec) {
    if (v > max) {
      max = v;
    }
  }
  return {max};
}

tiny_cnn::vec_t avg(tiny_cnn::vec_t vec) {
  double tot = 0;
  for (auto v : vec) {
      tot += v;
  }
  return {tot / vec.size()};
}

void construct_nn() {
  // specify loss-function and optimization-algorithm
  tiny_cnn::network<tiny_cnn::mse, tiny_cnn::gradient_descent> nn;

  std::vector<tiny_cnn::vec_t> ins;
  int n = 100;
  for (double i = 0; i < n; i++) {
    double a = (double)rand() / RAND_MAX;
    double b = (double)rand() / RAND_MAX;
    double c = (double)rand() / RAND_MAX;
    ins.push_back({a, b, c});
  }

  std::vector<tiny_cnn::vec_t> outs;
  for (int i = 0; i < ins.size(); i++) {
    //outs.push_back(max(ins[i]));
    outs.push_back(avg(ins[i]));
  }

  bool biasing = false;
  nn << tiny_cnn::fully_connected_layer<tan_h>(3, 3, biasing)
     << tiny_cnn::fully_connected_layer<tan_h>(3, 3, biasing)
     << tiny_cnn::fully_connected_layer<tan_h>(3, 1, biasing);

  std::cout << "indim=" << nn.in_dim() << "\n";
  std::cout << "outdim=" << nn.out_dim() << "\n";
  std::cout << "depth=" << nn.depth() << "\n";

  std::cout << "layer0 props:\n";
  std::cout << "    insize=" << nn[0]->in_size() << "\n";
  std::cout << "    indim=" << nn[0]->in_dim() << "\n";
  std::cout << "    outsize=" << nn[0]->out_size() << "\n";
  std::cout << "    outdim=" << nn[0]->out_dim() << "\n";

  nn.train(ins, outs);

  double toterr = 0;
  for (int i = 0; i < ins.size(); i++) {
    tiny_cnn::vec_t out = nn.predict(ins[i]);
    double error = std::abs(out[0] - outs[i][0])/1;
    toterr += error;
    std::cout << "f["
      << ins[i][0] << ", "
      << ins[i][1] << ", "
      << ins[i][2]
      << "] = " << out[0] << ", error= " << error << "\n";
  }
  std::cout << "avgerr = " << toterr / ins.size() << "\n";

  std::cout << "weights:\n" << nn << "\n";

  nn[0]->output_to_image().write("layer0.bmp");
  nn[1]->output_to_image().write("layer1.bmp");
  //nn[2]->output_to_image().write("layer2.bmp");
}

int main(int argc, char* argv[]) {
  construct_nn();
  return 0;
}

