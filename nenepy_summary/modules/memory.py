import numpy as np
import torch

from .abstract_module import _AbstractModule
from .output import _Output


class _Memory(_AbstractModule):
    _total_n_param_repr = "Total Params"
    _total_weight_param_repr = "Total Weight Params"
    _total_bias_param_repr = "Total Bias Params"

    _trainable_params_repr = "Total Trainable Params"
    _non_trainable_params_repr = "Total Non-Trainable Params"
    _trainable_weight_params_repr = "Trainable Weight Params"

    _non_trainable_weight_params_repr = "Non-Trainable Weight Params"
    _trainable_bias_params_repr = "Trainable Bias Params"
    _non_trainable_bias_params_repr = "Non-Trainable Bias Params"

    _total_param_mb_repr = "Total Params (MB)"
    _total_size_repr = "Total Size (MB)"

    _all_repr = [
        _total_n_param_repr, _total_param_mb_repr, _trainable_params_repr, _non_trainable_params_repr,
        _trainable_weight_params_repr, _non_trainable_weight_params_repr, _trainable_bias_params_repr, _non_trainable_bias_params_repr, _total_size_repr
    ]

    _max_name_length = 0
    _max_value_length = 0

    _trainable_n_weights = 0
    _non_trainable_n_weights = 0
    _trainable_n_biases = 0
    _non_trainable_n_biases = 0

    _total_n_params = 0
    _total_trainable_n_params = 0
    _total_non_trainable_n_params = 0
    _total_n_weights = 0
    _total_n_biases = 0

    _total_size = 0
    _total_param_memory_size = 0
    _torch_default_memory_size = 1382 * (1024 ** 2)

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def adjust(cls, modules):
        output_tensors = _Output.get_all_tensors([module.output for module in modules])
        input_tensors = _Output.get_all_tensors([module.input for module in modules])
        tensors = list(set(input_tensors + output_tensors))

        params = np.sum([cls._get_n_params(module.parameter) for module in modules], axis=0)
        trainable_n_weights, non_trainable_n_weights, trainable_n_biases, non_trainable_n_biases = params

        cls._trainable_n_weights = trainable_n_weights
        cls._non_trainable_n_weights = non_trainable_n_weights
        cls._trainable_n_biases = trainable_n_biases
        cls._non_trainable_n_biases = non_trainable_n_biases

        cls._total_n_params = trainable_n_weights + non_trainable_n_weights + trainable_n_biases + non_trainable_n_biases
        cls._total_trainable_n_params = trainable_n_weights + trainable_n_biases
        cls._total_non_trainable_n_params = non_trainable_n_weights + non_trainable_n_biases
        cls._total_n_weights = trainable_n_weights + non_trainable_n_weights
        cls._total_n_biases = trainable_n_biases + non_trainable_n_biases

        cls._total_param_memory_size = cls._get_param_memory_size([module.parameter for module in modules])
        cls._total_size = cls.byte_to_mb(torch.cuda.memory_reserved() + cls._torch_default_memory_size)

        cls._max_name_length = cls._get_max_text_length(cls._all_repr)
        cls._max_value_length = cls._get_max_text_length([f"{cls._total_size:,.2f}", f"{cls._total_n_params}:,"])

    @classmethod
    def to_print_format(cls):
        texts = [
            "",
            cls._value_to_text(cls._total_n_param_repr, cls._total_n_params),
            cls._value_to_text(cls._total_weight_param_repr, cls._total_n_weights),
            cls._value_to_text(cls._total_bias_param_repr, cls._total_n_biases),

            "",
            cls._value_to_text(cls._trainable_params_repr, cls._total_trainable_n_params),
            cls._value_to_text(cls._trainable_weight_params_repr, cls._trainable_n_weights),
            cls._value_to_text(cls._trainable_bias_params_repr, cls._trainable_n_biases),
            "",
            cls._value_to_text(cls._non_trainable_params_repr, cls._total_non_trainable_n_params),
            cls._value_to_text(cls._non_trainable_weight_params_repr, cls._non_trainable_n_weights),
            cls._value_to_text(cls._non_trainable_bias_params_repr, cls._non_trainable_n_biases),
            "",
            "",
            cls._memory_size_to_text(cls._total_param_mb_repr, cls.byte_to_mb(cls._total_param_memory_size)),
            "",
            cls._memory_size_to_text(cls._total_size_repr, cls._total_size),
        ]

        return "\n".join(texts)

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================
    @classmethod
    def _value_to_text(cls, text, value):
        return f"{text:>{cls._max_name_length}}: {value:>{cls._max_value_length},}"

    @classmethod
    def _memory_size_to_text(cls, text, value):
        return f"{text:>{cls._max_name_length}}: {value:>{cls._max_value_length},.2f}"

    @staticmethod
    def _get_memory_size(tensors):
        def func(tensor):
            # if tensor.requires_grad:
            #     return tensor.element_size() * tensor.nelement() * 2
            return tensor.element_size() * tensor.nelement()

        return sum([func(tensor) for tensor in tensors])

    @staticmethod
    def byte_to_mb(value):
        return value / (1024 ** 2)

    @staticmethod
    def _get_n_params(parameter):
        trainable_weight = 0
        non_trainable_weight = 0
        trainable_bias = 0
        non_trainable_bias = 0

        if parameter.has_weight:
            if parameter.weight_requires_grad and parameter.is_train:
                trainable_weight = parameter.n_weight_params
            else:
                non_trainable_weight = parameter.n_weight_params

        if parameter.has_bias:
            if parameter.bias_requires_grad and parameter.is_train:
                trainable_bias = parameter.n_bias_params
            else:
                non_trainable_bias = parameter.n_bias_params

        return trainable_weight, non_trainable_weight, trainable_bias, non_trainable_bias

    @staticmethod
    def _get_param_memory_size(parameters):
        params = []
        for parameter in parameters:
            params += parameter.params

        params = set(params)
        memory_size = 0
        for param in params:
            memory_size += param.element_size() * param.nelement()

        return memory_size
