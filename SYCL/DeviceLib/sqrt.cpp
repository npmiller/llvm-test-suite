// REQUIRES: cuda || hip

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -fsycl-fp32-prec-sqrt -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <iostream>

#include <vector>

#include <CL/sycl.hpp>

int main() {
  sycl::queue queue;

  std::vector<float> input = {
      8.4018772840e-01f, 3.9438292384e-01f, 7.8309923410e-01f,
      7.9844003916e-01f, 9.1164737940e-01f, 1.9755136967e-01f,
      3.3522275090e-01f, 7.6822960377e-01f, 2.7777472138e-01f,
      5.5396997929e-01f, 4.7739705443e-01f, 6.2887090445e-01f,
      3.6478447914e-01f, 5.1340091228e-01f, 9.5222973824e-01f,
      9.1619509459e-01f};

  std::vector<float> expected = {
      9.1661757231e-01f, 6.2799912691e-01f, 8.8492894173e-01f,
      8.9355474710e-01f, 9.5480227470e-01f, 4.4446751475e-01f,
      5.7898426056e-01f, 8.7648707628e-01f, 5.2704340219e-01f,
      7.4429160357e-01f, 6.9093924761e-01f, 7.9301381111e-01f,
      6.0397392511e-01f, 7.1652001143e-01f, 9.7582256794e-01f,
      9.5718079805e-01f};
  std::vector<float> output(input.size(), 0.0f);

  {
    sycl::buffer<float> inputB(input.data(), input.size());
    sycl::buffer<float> outputB(output.data(), output.size());

    queue
        .submit([&](sycl::handler &cgh) {
          auto inputAcc = inputB.get_access<sycl::access_mode::read>(cgh);
          auto outputAcc = outputB.get_access<sycl::access_mode::write>(cgh);
          cgh.parallel_for(sycl::range<1>(input.size()), [=](sycl::item<1> i) {
            outputAcc[i] = sycl::sqrt(inputAcc[i]);
          });
        })
        .wait_and_throw();
  }

  float diff = 0.0f;
  for (int i = 0; i < input.size(); ++i) {
    diff += fabs(expected[i] - output[i]);
  }

  if (diff > 0.0f) {
    std::cerr << "Incorrectly rounded sqrt, total diff: " << diff << std::endl;
    return -1;
  }

  std::cout << "Pass" << std::endl;
  return 0;
}
