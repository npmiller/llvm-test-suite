#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;

bool isSupportedDevice(device D) {
  std::string PlatformName =
      D.get_platform().get_info<sycl::info::platform::name>();
  if (PlatformName.find("CUDA") != std::string::npos)
    return true;

  if (PlatformName.find("Level-Zero") != std::string::npos)
    return true;

  if (PlatformName.find("OpenCL") != std::string::npos) {
    return false;
  }

  return false;
}
