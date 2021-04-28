from enum import Enum, auto

import torch


class _Align(Enum):
    Center = auto()
    Left = auto()
    Right = auto()


class _AbstractModule:
    indent_space = 5
    n_max_length = 0

    def __init__(self):
        self.adjusted_texts = None

    @classmethod
    def set_n_max_length(cls, print_format):
        if cls.n_max_length < len(print_format):
            cls.n_max_length = len(print_format)

    @staticmethod
    def to_replace(text, char=" "):
        return char * len(text)

    @staticmethod
    def to_empty(text):
        return " " * len(text)

    @staticmethod
    def generate_empty(length):
        return " " * length

    @staticmethod
    def _is_iterable(value):
        if isinstance(value, (tuple, list, set, dict)):
            return True
        return False

    @staticmethod
    def _get_max_dict_key_length(value_dict):
        return max([len(key) for key in value_dict.keys()], default=0)

    @staticmethod
    def _get_max_text_length(texts):
        return max([len(text) for text in texts], default=0)

    @staticmethod
    def _to_adjust_length(texts, adjustment_length=None):
        if adjustment_length is None:
            adjustment_length = max([len(text) for text in texts], default=0)
        return [f"{text:>{adjustment_length}}: " for text in texts]

    @staticmethod
    def _to_type(value):
        return str(value.__class__.__name__)

    @staticmethod
    def _is_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        return False

    @staticmethod
    def _tensor_to_str(value):
        size = list(value.shape)
        if len(size) == 0:
            size = [1]
        return list(map(str, size))

    @staticmethod
    def _is_built_in_type(value):
        if isinstance(value, (bool, int, float, complex, str)):
            return True

        return False

    @classmethod
    def get_max_dict_key_length(cls, outputs):
        def func(dict):
            return max([len(key) for key in dict.keys()], default=0)

        return max([func(output.values) if isinstance(output.values, dict) else 0 for output in outputs], default=0)

    @staticmethod
    def _replace(text, char=" "):
        return char * len(text)

    @staticmethod
    def _empty(text=" "):
        return " " * len(text)

    @staticmethod
    def _fill(char, length):
        return char * length

    @staticmethod
    def _align(text, length, mode):
        """
        Args:
            text (str):
            length:
            mode:

        Returns:

        """
        if mode == _Align.Left:
            return text.ljust(length)
        elif mode == _Align.Right:
            return text.rjust(length)
        elif mode == _Align.Center:
            return text.center(length)
        else:
            raise TypeError()

    @staticmethod
    def _to_empty_text(length):
        return " " * length

    @staticmethod
    def _init_list(init_value, size):
        return [init_value] * size
