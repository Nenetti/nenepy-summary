import numpy as np
import torch
from torch import nn

from .abstract_module import _AbstractModule


class _Parameter(_AbstractModule):

    def __init__(self, module):
        """

        Args:
            module (nn.Module):
        """
        super(_Parameter, self).__init__()
        self.module = module
        self.params = module.state_dict(keep_vars=True).values()
        self.is_train = module.training
        self.has_weight = self._has_weight(module)
        self.has_bias = self._has_bias(module)
        self.weight, self.n_weight_params, self.weight_requires_grad = self._analyze_weight(module)
        self.bias, self.n_bias_params, self.bias_requires_grad = self._analyze_bias(module)

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def to_formatted_text(self):
        weight_format = self._weight_str()
        bias_format = self._bias_str()
        train_format = self._to_bool_format(self.is_train)
        requires_grad_weight_format = self._to_bool_format(self.weight_requires_grad)
        requires_grad_bias_format = self._to_bool_format(self.bias_requires_grad)

        print_format = [weight_format, bias_format, train_format, requires_grad_weight_format, requires_grad_bias_format]
        return print_format

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    def _weight_str(self):
        if self.has_weight:
            return f"{self.n_weight_params:,}"
        else:
            return "-"

    def _bias_str(self):
        if self.has_bias:
            return f"{self.n_bias_params:,}"
        else:
            return "-"

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def adjust(cls, parameters):
        cls.n_max_length = cls._get_max_text_length([parameter.to_formatted_text() for parameter in parameters])

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================
    @classmethod
    def _analyze_weight(cls, module):
        if cls._has_weight(module):
            weight_tensor = module.weight
            n_params = cls._calc_n_params(weight_tensor)
            requires_grad = module.weight.requires_grad
            return weight_tensor, n_params, requires_grad

        return None, 0, False

    @classmethod
    def _analyze_bias(cls, module):
        if cls._has_bias(module):
            bias_tensor = module.bias
            n_params = cls._calc_n_params(bias_tensor)
            requires_grad = module.bias.requires_grad
            return bias_tensor, n_params, requires_grad

        return None, 0, False

    @classmethod
    def _calc_max_weight_length(cls, parameter):
        """
        Args:
            parameter (_Parameter):

        Returns:
            int

        """
        return len(parameter._weight_str())

    @classmethod
    def _calc_max_bias_length(cls, parameter):
        """

        Args:
            parameter (_Parameter):

        Returns:
            int

        """
        return len(parameter._bias_str())

    @staticmethod
    def _has_weight(module):
        if hasattr(module, "weight") and (module.weight is not None) and isinstance(module.weight, torch.Tensor):
            return True

        return False

    @staticmethod
    def _has_bias(module):
        if hasattr(module, "bias") and (module.bias is not None) and isinstance(module.bias, torch.Tensor):
            return True

        return False

    @staticmethod
    def _calc_n_params(tensor):
        return np.prod(tensor.size())

    @staticmethod
    def _to_bool_format(is_train):
        """

        Args:
            is_train (bool):

        Returns:

        """
        if is_train:
            return "✓"
        else:
            return "-"
