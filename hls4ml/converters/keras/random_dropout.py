import math
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler
from hls4ml.converters.utils import parse_data_format, compute_padding_1d, compute_padding_2d

@keras_handler('RandomDropout')
def parse_random_dropout_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('RandomDropout' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)
    if layer['data_format'] != 'channels_last':
        raise Exception('Only channels_last data format supported for RandomDropout layer.')
    
    layer['drop_rate'] = keras_layer['config']['drop_rate']
    layer['seed'] = keras_layer['config']['seed']
    #
    (
        layer['in_height'],
        layer['in_width'],
        layer['n_in']
    ) = parse_data_format(input_shapes[0], layer['data_format'])
    #
    
    return layer, [shape for shape in input_shapes[0]]
