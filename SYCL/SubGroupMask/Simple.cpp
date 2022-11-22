// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// REQUIRES: gpu
// GroupNonUniformBallot capability is supported on Intel GPU only
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: ze_debug-1,ze_debug4

//==---------- Simple.cpp - sub-group mask basic test ----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int global_size = 256;
constexpr int local_size = 128;

struct Results {
  bool Masks;
  bool OrAll;
  bool AndNone;
  bool NotAndAny;
  bool Any;
  bool XorAll;
  bool Not;
  bool FindHigh;
  bool FindLow;
  bool ShiftLeft;
  bool ShiftRight;
  bool Count;
  bool Flip;
  bool FlipId;
  bool Set;
  bool Reset;
  bool SetId;
  bool ResetId;
  bool ResetHigh;
  bool ResetLow;

  size_t SGSize;
  int EvenMask[local_size];
  int OddMask[local_size];
};

int main() {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  queue Queue;

  try {
    nd_range<1> NdRange(global_size, local_size);
    Results R;
    {
      buffer resbuf(&R, range<1>(1));
      Queue
          .submit([&](handler &cgh) {
            auto resacc = resbuf.get_access<access::mode::read_write>(cgh);

            // Let the device pick the sub-group size, assuming an even
            // sub-group size of at least 4 threads.
            cgh.parallel_for<class sub_group_mask_test>(
                NdRange, [=](nd_item<1> NdItem) {
                  size_t GID = NdItem.get_global_linear_id();
                  auto SG = NdItem.get_sub_group();

                  auto evenmask = ext::oneapi::group_ballot(
                      NdItem.get_sub_group(), !(SG.get_local_id() % 2));
                  auto oddmask = ext::oneapi::group_ballot(
                      NdItem.get_sub_group(), SG.get_local_id() % 2);

                  // Check results in the first thread
                  if (GID == 0) {
                    resacc[0].SGSize = SG.get_local_range()[0];

                    int res = 1;
                    for (size_t i = 0; i < SG.get_local_range()[0]; i++) {
                      res &= evenmask[i] == !(i % 2);
                      res &= oddmask[i] == i % 2;

                      resacc[0].EvenMask[i] = evenmask[i];
                      resacc[0].OddMask[i] = oddmask[i];
                    }
                    resacc[0].Masks = res;

                    resacc[0].OrAll = (evenmask | oddmask).all();
                    resacc[0].AndNone = (evenmask & oddmask).none();
                    resacc[0].NotAndAny = !(evenmask & oddmask).any();
                    resacc[0].Any = evenmask.any();
                    resacc[0].XorAll = (evenmask ^ oddmask).all();
                    resacc[0].Not = ~evenmask == oddmask;
                    resacc[0].FindHigh =
                        evenmask.find_high() == (SG.get_local_range()[0] - 2);
                    resacc[0].FindLow = evenmask.find_low() == 0;
                    resacc[0].ShiftLeft = (evenmask << 2).find_low() == 2;
                    resacc[0].ShiftRight = (evenmask >> 2).find_high() ==
                                           (SG.get_local_range()[0] - 4);
                    resacc[0].Count =
                        (evenmask.count() == (SG.get_local_range()[0] / 2));

                    evenmask.flip();
                    resacc[0].Flip = evenmask == oddmask;

                    evenmask.flip(0);
                    resacc[0].FlipId = evenmask.find_low() == 0;

                    evenmask.set();
                    resacc[0].Set = evenmask.all();

                    evenmask.reset();
                    resacc[0].Reset = evenmask.none();

                    evenmask.set(1);
                    resacc[0].SetId = evenmask.find_low() == 1;

                    evenmask.set(2);
                    evenmask.reset(1);
                    resacc[0].ResetId = evenmask.find_low() == 2;

                    evenmask.set();
                    evenmask.reset_high();
                    resacc[0].ResetHigh =
                        evenmask.find_high() == (SG.get_local_range()[0] - 2);

                    evenmask.reset_low();
                    resacc[0].ResetLow = evenmask.find_low() == 1;
                  }
                });
          })
          .wait_and_throw();
    }

    bool errors = false;
    if (!R.Masks) {
      std::cout << "Incorrect masks from group_ballot operations" << std::endl;
      errors = true;
    }

    if (!R.OrAll) {
      std::cout << "Incorrect results from operator| and .all()" << std::endl;
      errors = true;
    }

    if (!R.AndNone) {
      std::cout << "Incorrect results from operator& and .none()" << std::endl;
      errors = true;
    }

    if (!R.NotAndAny) {
      std::cout << "Incorrect results from !, operator& and .any()"
                << std::endl;
      errors = true;
    }

    if (!R.Any) {
      std::cout << "Incorrect results from .any()" << std::endl;
      errors = true;
    }

    if (!R.XorAll) {
      std::cout << "Incorrect results from operator^ and .all()" << std::endl;
      errors = true;
    }

    if (!R.Not) {
      std::cout << "Incorrect results from operator~" << std::endl;
      errors = true;
    }

    if (!R.FindHigh) {
      std::cout << "Incorrect results from .find_high()" << std::endl;
      errors = true;
    }

    if (!R.FindLow) {
      std::cout << "Incorrect results from .find_low()" << std::endl;
      errors = true;
    }

    if (!R.ShiftLeft) {
      std::cout << "Incorrect results from operator<< and .find_low()"
                << std::endl;
      errors = true;
    }

    if (!R.ShiftRight) {
      std::cout << "Incorrect results from operator>> and .find_high()"
                << std::endl;
      errors = true;
    }

    if (!R.Count) {
      std::cout << "Incorrect results from .count()" << std::endl;
      errors = true;
    }

    if (!R.Flip) {
      std::cout << "Incorrect results from .flip()" << std::endl;
      errors = true;
    }

    if (!R.FlipId) {
      std::cout << "Incorrect results from .flip(id<1>)" << std::endl;
      errors = true;
    }

    if (!R.Set) {
      std::cout << "Incorrect results from .set()" << std::endl;
      errors = true;
    }

    if (!R.Reset) {
      std::cout << "Incorrect results from .reset()" << std::endl;
      errors = true;
    }

    if (!R.SetId) {
      std::cout << "Incorrect results from .set(id<1>)" << std::endl;
      errors = true;
    }

    if (!R.ResetId) {
      std::cout << "Incorrect results from .reset(id<1>)" << std::endl;
      errors = true;
    }

    if (!R.ResetHigh) {
      std::cout << "Incorrect results from .reset_high()" << std::endl;
      errors = true;
    }

    if (!R.ResetLow) {
      std::cout << "Incorrect results from .reset_low()" << std::endl;
      errors = true;
    }

    if (errors) {
      for (int i = 0; i < R.SGSize; ++i) {
        std::cout << "EvenMask[" << i << "] = " << R.EvenMask[i] << ", OddMask["
                  << i << "] = " << R.OddMask[i] << std::endl;
      }
      exit(1);
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }

  std::cout << "Test passed." << std::endl;
#else
  std::cout << "Test skipped due to missing extension." << std::endl;
#endif
  return 0;
}
