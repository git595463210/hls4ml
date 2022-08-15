//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_DROPOUT_H_
#define NNET_DROPOUT_H_

#include "ap_fixed.h"
#include "nnet_common.h"
#include <cmath>
#include <stdint.h>
#include <random>

namespace nnet {

struct dropout_config
{
    // IO size
    static const unsigned n_in = 10;

    // Internal info
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef ap_fixed<18,8> table_t;
};

int bernouli_distribution(float p, std::default_random_engine generator) {
  int res = ((double)generator() / (double)generator.max()) < p ? 1 : 0;
  return res;
}

// *************************************************
//       Bayesian Dropout
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  dropout(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    #pragma HLS PIPELINE

  static std::default_random_engine generator(CONFIG_T::seed);
  float keep_rate = 1 - CONFIG_T::drop_rate;
  for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
    float temp = data[ii] * nnet::bernouli_distribution(keep_rate, generator);
    res[ii] = temp * keep_rate;
  }
}
}

#endif
