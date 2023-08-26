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

#ifndef NNET_BLOCKDROPOUT_STREAM_H_
#define NNET_BLOCKDROPOUT_STREAM_H_

#include <cmath>
#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_types.h"
#include "nnet_stream.h"
#include "nnet_block_dropout.h"

namespace nnet {

// *************************************************
//       Bayesian BlockDropout
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void block_dropout(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream) {

    typename data_T::value_type data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=res complete

    DataPrepare: for(int i_in = 0; i_in < CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_in / data_T::size; i_in++) {
        if (CONFIG_T::n_in / data_T::size > 1) {
            #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
        DataPack: for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }  
    }






    static std::minstd_rand generator(0);
    float keep_rate = 1 - CONFIG_T::drop_rate/(CONFIG_T::block_size * CONFIG_T::block_size);
    float max = generator.max();
    BlockDropoutLoop: for (int i = 0; i < CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
        
        
        for (int j = 0; j < CONFIG_T::in_height; j++){
          for (int k = 0; k < CONFIG_T::in_width; k++) {
            for (int l = 0; l < CONFIG_T::block_size; l++){
              for (int m = 0; m < CONFIG_T::block_size; m++){
                typename data_T::value_type zero = {};
                typename data_T::value_type temp =((float)generator() / max) < keep_rate? data[k + j*CONFIG_T::in_height + i*CONFIG_T::in_height*CONFIG_T::in_width + m + l*CONFIG_T::in_width] : zero;
                res[k + j*CONFIG_T::in_height + i*CONFIG_T::in_height*CONFIG_T::in_width + m + l*CONFIG_T::in_width] = temp * (typename data_T::value_type)keep_rate;
            
            
                  }
                }
              }
            }
          
          
    }

















    ResWrite: for(unsigned i_out = 0; i_out < CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_in / res_T::size; i_out++) {
        if (CONFIG_T::n_in / res_T::size > 1) {
            #pragma HLS PIPELINE
        }
        res_T res_pack;
        #pragma HLS DATA_PACK variable=res_pack
        ResPack: for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}
}

#endif
