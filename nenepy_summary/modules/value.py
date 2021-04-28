import itertools

import numpy as np
import torch

from .abstract_module import _AbstractModule


class _Value(_AbstractModule):

    def __init__(self, value):
        super(_Value, self).__init__()
        self.value = value
        self.is_tensor = self._is_tensor(value)
        self.sizes = self._to_shape_list(value)
        self.text = self._to_text(value)
        self.type = self._to_type(value)

    def to_adjusted_text(self, each_dim_size):
        """

        Args:
            each_dim_size (list[int]):

        Returns:

        """
        if self.is_tensor:
            return self._shape_to_text(self.sizes, each_dim_size)
        else:
            return self.text

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def calc_max_n_dims(cls, tensors):
        return max([len(cls._tensor_to_str(tensor)) for tensor in tensors])

    @classmethod
    def calc_max_each_dim_size(cls, tensors, max_n_dims):
        def func(tensor):
            each_size = np.zeros(shape=max_n_dims, dtype=np.int32)
            shapes = cls._tensor_to_str(tensor)
            for i, value_str in enumerate(shapes):
                each_size[i] = len(value_str)

            return each_size

        return np.max(np.stack([func(tensor) for tensor in tensors], axis=0), axis=0)

    @classmethod
    def _to_text(cls, value):
        if isinstance(value, torch.Tensor):
            return str(cls._tensor_to_str(value))
        elif cls._is_built_in_type(value):
            return f"<'{cls._to_type(value)}' {str(value)}>"
        elif value is None:
            return str(None)
        else:
            return f"{cls._to_type(value)}()"

    @classmethod
    def _to_shape_list(cls, value):
        if cls._is_tensor(value):
            return cls._tensor_to_str(value)
        return None

    @staticmethod
    def _shape_to_text(sizes, each_dim_size):
        texts = [f"{size:>{each_dim_size[i]}}" for i, size in enumerate(sizes)]
        text = ", ".join(texts)
        return f"[{text}]"

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @classmethod
    def get_all_tensors(cls, values):
        def recursive(value):
            if cls._is_iterable(value):
                if isinstance(value, dict):
                    return itertools.chain.from_iterable([recursive(v) for v in value.values()])
                else:
                    return itertools.chain.from_iterable([recursive(v) for v in value])
            elif isinstance(value, _Value):
                return [value.value] if value.is_tensor else []
            else:
                raise TypeError()

        return recursive(values)
