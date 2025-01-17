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

#ifndef NNET_RANDOMDROPOUT_H_
#define NNET_RANDOMDROPOUT_H_

#include "ap_fixed.h"
#include "nnet_common.h"
#include <cmath>
#include <random>
#include <stdint.h>


namespace nnet {

struct random_dropout_config
{
    // IO size
    static const unsigned n_in = 10;
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
};

// *************************************************
//       Bayesian RandomDropout
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void random_dropout(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_in],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_in]

)
{
    #pragma HLS PIPELINE

  static std::default_random_engine generator(0);
  data_T keep_rate = 1 - CONFIG_T::drop_rate;
  data_T max = generator.max(); 
  
  bool random_array[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_in];
    RandomNumberLoop: for (int i = 0; i < CONFIG_T::n_in; i++) {
                        for (int j = 0; j < CONFIG_T::in_height; j++) {
                          for (int k = 0; k < CONFIG_T::in_width; k++) {
                            random_array[k + j*CONFIG_T::in_width + i*CONFIG_T::in_height*CONFIG_T::in_width] = ((data_T)generator() / max) < keep_rate;
        }
      }
    }
    
  for (int ii = 0; ii < CONFIG_T::n_in; ii++) { 
    for (int jj = 0; jj < CONFIG_T::in_height; jj++){
      for (int kk = 0; kk < CONFIG_T::in_width; kk++){
        data_T zero = {};
        data_T temp = random_array[kk + jj*CONFIG_T::in_width + ii*CONFIG_T::in_height*CONFIG_T::in_width] ? data[kk + jj*CONFIG_T::in_width + ii*CONFIG_T::in_height*CONFIG_T::in_width] : zero;
        res[kk + jj*CONFIG_T::in_width + ii*CONFIG_T::in_height*CONFIG_T::in_width] = temp * keep_rate;
    	}
    }
  }
  
}
}

#endif
