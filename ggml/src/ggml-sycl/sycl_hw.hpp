#ifndef SYCL_HW_HPP
#define SYCL_HW_HPP

#include <algorithm>
#include <stdio.h>
#include <vector>
#include <map>

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;
using gpu_arch = sycl::ext::oneapi::experimental::architecture;

struct sycl_hw_info {
  syclex::architecture arch;
  const char* arch_name;
  int32_t device_id;
  std::string name;
};

sycl_hw_info get_device_hw_info(sycl::device *device_ptr);

#endif // SYCL_HW_HPP
