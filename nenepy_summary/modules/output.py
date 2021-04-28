import itertools

from .abstract_module import _AbstractModule
from .value import _Value


class _Output(_AbstractModule):
    _max_key_length = 0
    _max_each_dim_size = 0

    def __init__(self, values):
        super(_Output, self).__init__()
        self.values = self._analyze_values(values)

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def to_formatted_text(self):
        return self._iterable_to_text_formats(self.values, self._max_key_length, True)

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def adjust(cls, outputs):
        tensors = cls.get_all_tensors(outputs)
        max_n_dims = _Value.calc_max_n_dims(tensors)
        cls._max_each_dim_size = _Value.calc_max_each_dim_size(tensors, max_n_dims)
        cls._max_key_length = cls.get_max_dict_key_length(outputs)
        cls.n_max_length = cls._get_max_text_length([output.to_formatted_text() for output in outputs])

    @classmethod
    def get_all_tensors(cls, outputs):
        return list(itertools.chain.from_iterable([_Value.get_all_tensors(output.values) for output in outputs]))

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================
    @classmethod
    def _analyze_values(cls, values):
        def recursive(value):
            if cls._is_iterable(value):
                if isinstance(value, dict):
                    return dict((str(key), recursive(v)) for key, v in value.items())
                else:
                    return [recursive(v) for v in value]
            else:
                return _Value(value)
                # if not isinstance(value, torch.Tensor) and hasattr(value, "__dict__"):
                #     return {f"{cls._to_type(value)}()": dict((key, recursive(v)) for key, v in value.__dict__.items())}
                # else:
                #     return Value(value)

        if not isinstance(values, dict):
            values = {"": values}
        return recursive(values)

    @classmethod
    def _iterable_to_text_formats(cls, value, key_length=None, is_root=False):
        if len(value) == 0:
            return []
        if isinstance(value, dict):
            return cls._dict_to_text(value, key_length, is_root)
        else:
            return cls._list_to_text(value)

    @classmethod
    def _list_to_text(cls, value_list):
        value_type = f"<{cls._to_type(value_list)}>"
        key_length = len(value_type)
        texts = []
        for value in value_list:
            if cls._is_iterable(value):
                texts += cls._iterable_to_text_formats(value)
            else:
                texts += [value.to_adjusted_text(cls._max_each_dim_size)]

        brackets = cls._get_list_brackets(len(texts))
        max_length = cls._get_max_text_length(texts)

        for i, (text, bracket) in enumerate(zip(texts, brackets)):
            bracket_top, bracket_bottom = bracket
            if i == 0:
                type_format = value_type
            else:
                type_format = f"{'':>{key_length}}"

            texts[i] = f"{type_format}{bracket_top} {text:<{max_length}} {bracket_bottom}"

        return texts

    @classmethod
    def _dict_to_text(cls, value_dict, key_length=None, is_root=False):

        if key_length is None:
            key_length = cls._get_max_dict_key_length(value_dict)

        adjusted_keys = cls._to_adjust_length(value_dict.keys(), key_length)

        texts = []
        for key, value in zip(adjusted_keys, value_dict.values()):
            if is_root and len(value_dict) == 1:
                key = f"{cls.to_empty(key)}"

            if cls._is_iterable(value):
                formatted_values = cls._iterable_to_text_formats(value)
                child_texts = [""] * len(formatted_values)
                for i, formatted_value in enumerate(formatted_values):
                    if i == 0:
                        child_texts[i] = f"{key}{formatted_value}"
                    else:
                        child_texts[i] = f"{cls.to_empty(key)}{formatted_value}"
                texts += child_texts
            else:
                value_format = value.to_adjusted_text(cls._max_each_dim_size)
                text = f"{key}{value_format}"
                texts.append(text)
        return texts

    @classmethod
    def _get_list_brackets(cls, size):
        def get_list_bracket(index, size):
            if size <= 1:
                return ["[", "]"]
            if index == 0:
                return ["┌", "┐"]
            elif index == size - 1:
                return ["└", "┘"]
            else:
                return ["│", "│"]

        return [get_list_bracket(i, size) for i in range(size)]
