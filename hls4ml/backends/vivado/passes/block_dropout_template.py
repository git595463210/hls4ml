from hls4ml.backends.backend import get_backend
from hls4ml.model.layers import BlockDropout
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

# Bayesian Dropout template
block_dropout_config_template = """
struct config{index} : nnet::dropout_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_in = {n_in};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static constexpr float drop_rate = {drop_rate};
    static constexpr float block_size = {block_size};
    std::default_random_engine eng = std::default_random_engine();
}};\n"""

# isBayes must be set to True in config!
block_dropout_function_template = 'nnet::block_dropout<{input_t}, {output_t}, {config}>({input}, {output});'

block_dropout_include_list = ['nnet_utils/nnet_block_dropout.h', 'nnet_utils/nnet_block_dropout_stream.h']

class BlockDropoutConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(BlockDropout)
        self.template = block_dropout_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)

class BlockDropoutFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(BlockDropout, include_header=block_dropout_include_list)
        self.template = block_dropout_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)






