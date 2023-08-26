
from hls4ml.backends.backend import get_backend
from hls4ml.model.layers import RandomDropout
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

# Bayesian Dropout template

random_dropout_config_template = """
struct config{index} : nnet::random_dropout_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_in = {n_in};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static constexpr float drop_rate = {drop_rate};
    std::default_random_engine eng = std::default_random_engine();
}};\n"""

# isBayes must be set to True in config!
random_dropout_function_template = 'nnet::random_dropout<{input_t}, {output_t}, {config}>({input}, {output});'

random_dropout_include_list = ['nnet_utils/nnet_random_dropout.h', 'nnet_utils/nnet_random_dropout_stream.h']

class RandomDropoutConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(RandomDropout)
        self.template = random_dropout_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)

class RandomDropoutFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(RandomDropout, include_header=random_dropout_include_list)
        self.template = random_dropout_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)






