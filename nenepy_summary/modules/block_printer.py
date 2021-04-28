import numpy as np

from .abstract_module import _AbstractModule, _Align
from .input import _Input
from .memory import _Memory
from .network_architecture import _NetworkArchitecture
from .output import _Output
from .parameter import _Parameter
from .time import _Time


class _BlockPrinter(_AbstractModule):
    architecture_length = 0
    input_length = 0
    output_length = 0
    parameter_length = 0
    time_length = 0
    weight_length = 0
    bias_length = 0
    train_length = 0
    requires_grad_length = 0
    requires_grad_weight_length = 0
    requires_grad_bias_length = 0

    network_architecture_header = "Network Architecture"
    input_header = "Input"
    output_header = "Output"
    parameter_header = "Parameter"
    weight_header = "Weight"
    bias_header = "Bias"
    train_header = "Train"
    requires_grad_header = "Requires Grad"
    requires_grad_weight_header = "Weight"
    requires_grad_bias_header = "Bias"
    time_header = "Time (ms)"

    def __init__(self, module):
        super(_BlockPrinter, self).__init__()
        self.module = module

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def to_formatted_texts(self, before_block_bottom):

        formatted_architecture = self.module.network_architecture.to_formatted_text()
        formatted_inputs = self.module.input.to_formatted_text()
        formatted_outputs = self.module.output.to_formatted_text()
        formatted_time = self.module.time.to_formatted_text()
        formatted_parameters = self.module.parameter.to_formatted_text()

        n_lines = max([len(formatted_inputs), len(formatted_outputs)])
        n_inputs = len(formatted_inputs)
        n_outputs = len(formatted_outputs)

        needs_top_space = ((n_lines > 1 or self.module.has_children()) and (not before_block_bottom))
        needs_bottom_space = (n_lines > 1)

        architectures = [formatted_architecture, *self._init_list(self.module.network_architecture.to_bottom_formant(), n_lines - 1)]
        parameters = [self._to_parameter_format(*formatted_parameters), *self._init_list(self._to_parameter_format("", "", "", "", ""), n_lines - 1)]
        times = [formatted_time, *self._init_list("", n_lines - 1)]
        inputs = formatted_inputs + self._init_list("", n_lines - n_inputs)
        outputs = formatted_outputs + self._init_list("", n_lines - n_outputs)

        if needs_top_space:
            n_lines += 1
            architectures.insert(0, self.module.network_architecture.to_top_format())
            parameters.insert(0, self._to_parameter_format("", "", "", "", ""))
            times.insert(0, "")
            inputs.insert(0, "")
            outputs.insert(0, "")

        if needs_bottom_space:
            n_lines += 1
            architectures.append(self.module.network_architecture.to_bottom_formant())
            parameters.append(self._to_parameter_format("", "", "", "", ""))
            times.append("")
            inputs.append("")
            outputs.append("")
        #
        # if n_lines > 1:
        #     n_lines += 1
        #     architectures = self._init_list(self.module.network_architecture.to_bottom_formant(), n_lines)
        #     architectures[0] = self.module.network_architecture.to_top_format()
        #     architectures[1] = formatted_architecture
        #     parameters = self._init_list(self._to_parameter_format("", "", "", "", ""), n_lines)
        #     parameters[1] = self._to_parameter_format(*formatted_parameters)
        #     times = self._init_list("", n_lines)
        #     times[1] = formatted_time
        #
        #     inputs = self._init_list("", n_lines)
        #     inputs[1:n_inputs + 1] = formatted_inputs
        #     outputs = self._init_list("", n_lines)
        #     outputs[1:n_outputs + 1] = formatted_outputs
        #
        # if self.module.has_children():
        #     n_lines += 1
        #     architectures = self._init_list(self.module.network_architecture.to_bottom_formant(), n_lines)
        #     architectures[0] = self.module.network_architecture.to_top_format()
        #     architectures[1] = formatted_architecture
        #     parameters = self._init_list(self._to_parameter_format("", "", "", "", ""), n_lines)
        #     parameters[1] = self._to_parameter_format(*formatted_parameters)
        #     times = self._init_list("", n_lines)
        #     times[1] = formatted_time
        #
        #     inputs = self._init_list("", n_lines)
        #     inputs[1:n_inputs] = formatted_inputs
        #     outputs = self._init_list("", n_lines)
        #     outputs[1:n_outputs] = formatted_outputs

        # architectures = [formatted_architecture] + self._init_list(self.module.network_architecture.to_bottom_formant(), (n_lines - 1))
        # parameters = [self._to_parameter_format(*formatted_parameters)] + self._init_list(self._to_parameter_format("", "", "", "", ""), (n_lines - 1))
        # times = [formatted_time] + self._init_list("", (n_lines - 1))
        # inputs = formatted_inputs + self._init_list("", (n_lines - n_inputs))
        # outputs = formatted_outputs + self._init_list("", (n_lines - n_outputs))

        lines = [""] * n_lines
        for i in range(n_lines):
            architecture_format = self._align(architectures[i], self.architecture_length, _Align.Left)
            input_format = self._align(inputs[i], self.input_length, _Align.Left)
            output_format = self._align(outputs[i], self.output_length, _Align.Left)
            time_format = self._align(times[i], self.time_length, _Align.Right)
            parameter_format = parameters[i]

            lines[i] = f"{architecture_format} │ {input_format} │ {output_format} │ {parameter_format} │ {time_format} │"

        return lines, needs_bottom_space

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def adjust(cls, modules):
        """
        Args:
            modules (list[Module]):

        Returns:

        """
        _NetworkArchitecture.adjust([module.network_architecture for module in modules])
        _Input.adjust([module.input for module in modules])
        _Output.adjust([module.output for module in modules])
        _Parameter.adjust([module.parameter for module in modules])
        _Time.adjust([module.time for module in modules])
        _Memory.adjust(modules)

        cls.architecture_length = cls._calc_max_architecture_length(modules)
        cls.input_length = cls._calc_max_input_length(modules)
        cls.output_length = cls._calc_max_output_length(modules)
        cls.time_length = cls._calc_max_time_length(modules)

        parameter_lengths = cls._calc_max_each_parameter_length(modules)
        cls.weight_length = parameter_lengths[0]
        cls.bias_length = parameter_lengths[1]
        cls.train_length = parameter_lengths[2]
        cls.requires_grad_weight_length = parameter_lengths[3]
        cls.requires_grad_bias_length = parameter_lengths[4]
        cls.requires_grad_length = cls._calc_max_requires_grad_length()
        cls.parameter_length = cls._calc_max_parameter_length()

    @classmethod
    def to_header_text(cls, reverse=False):

        architecture_format = cls._align(cls.network_architecture_header, cls.architecture_length, _Align.Center)
        input_format = cls._align(cls.input_header, cls.input_length, _Align.Center)
        output_format = cls._align(cls.output_header, cls.output_length, _Align.Center)
        time_format = cls._align(cls.time_header, cls.time_length, _Align.Center)

        empty_architecture_format = cls._fill(" ", cls.architecture_length)
        empty_input_format = cls._fill(" ", cls.input_length)
        empty_output_format = cls._fill(" ", cls.output_length)
        empty_time_format = cls._fill(" ", cls.time_length)

        bar_architecture_format = cls._fill(cls.architecture_length, "=")
        bar_input_format = cls._fill(cls.input_length, "=")
        bar_output_format = cls._fill(cls.output_length, "=")
        bar_parameter_format = cls._fill(cls.parameter_length, "=")
        bar_time_format = cls._fill(cls.time_length, "=")

        parameter_headers = cls._parameter_headers()

        bar = f"{bar_architecture_format}=│={bar_input_format}=│={bar_output_format}=│={bar_parameter_format}=│={bar_time_format}=│"

        line1 = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │ {parameter_headers[0]} │ {empty_time_format} │"
        line2 = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │ {parameter_headers[1]} │ {time_format} │"
        line3 = f"{architecture_format} │ {input_format} │ {output_format} │ {parameter_headers[2]} │ {empty_time_format} │"
        line4 = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │ {parameter_headers[3]} │ {empty_time_format} │"
        line5 = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │ {parameter_headers[4]} │ {empty_time_format} │"

        lines = [bar, line1, line2, line3, line4, line5, bar]

        return "\n".join(reversed(lines) if reverse else lines)

    @staticmethod
    def to_footer_text():
        return _Memory.to_print_format()

    @classmethod
    def _parameter_headers(cls):
        line1 = cls._align(cls.parameter_header, cls.parameter_length, _Align.Center)
        line2 = cls._fill("-", cls.parameter_length)
        line3 = cls._to_parameter_header_format("", "", "", cls.requires_grad_header)
        line4 = cls._to_parameter_header_format(cls.weight_header, cls.bias_header, cls.train_header, cls._fill('-', cls.requires_grad_length))
        line5 = cls._to_parameter_header_format("", "", "", cls._to_requires_grad_format(cls.requires_grad_weight_header, cls.requires_grad_bias_header, True))

        return [line1, line2, line3, line4, line5]

    @classmethod
    def _to_parameter_header_format(cls, weight, bias, train, requires_grad):
        weight = cls._align(weight, cls.weight_length, _Align.Center)
        bias = cls._align(bias, cls.bias_length, _Align.Center)
        train = cls._align(train, cls.train_length, _Align.Center)
        requires_grad = cls._align(requires_grad, cls.requires_grad_length, _Align.Center)
        return f"{weight} │ {bias} │ {train} │ {requires_grad}"

    @classmethod
    def _to_parameter_format(cls, weight, bias, train, requires_grad_weight, requires_grad_bias):
        weight = cls._align(weight, cls.weight_length, _Align.Right)
        bias = cls._align(bias, cls.bias_length, _Align.Right)
        train = cls._align(train, cls.train_length, _Align.Center)
        requires_grad_weight = cls._align(requires_grad_weight, cls.requires_grad_weight_length, _Align.Center)
        requires_grad_bias = cls._align(requires_grad_bias, cls.requires_grad_bias_length, _Align.Center)
        requires_grad = cls._align(cls._to_requires_grad_format(requires_grad_weight, requires_grad_bias), cls.requires_grad_length, _Align.Center)

        return f"{weight} │ {bias} │ {train} │ {requires_grad}"

    @classmethod
    def _to_requires_grad_format(cls, requires_grad_weight, requires_grad_bias, is_header=False):
        if is_header:
            return f"{requires_grad_weight} / {requires_grad_bias}"
        else:
            return f"{requires_grad_weight}   {requires_grad_bias}"

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================
    @classmethod
    def _calc_max_architecture_length(cls, modules):
        length = max([len(module.network_architecture.to_formatted_text()) for module in modules], default=0)
        return max([length, len(cls.network_architecture_header)])

    @classmethod
    def _calc_max_input_length(cls, modules):
        length = max([max([len(text) for text in module.input.to_formatted_text()], default=0) for module in modules], default=0)
        return max([length, len(cls.input_header)])

    @classmethod
    def _calc_max_output_length(cls, modules):
        length = max([max([len(text) for text in module.output.to_formatted_text()], default=0) for module in modules], default=0)
        return max([length, len(cls.output_header)])

    @classmethod
    def _calc_max_each_parameter_length(cls, modules):
        headers = [
            cls.weight_header, cls.bias_header, cls.train_header, cls.requires_grad_weight_header, cls.requires_grad_bias_header
        ]
        headers = [len(header) for header in headers]
        length = np.max(np.stack([[len(text) for text in module.parameter.to_formatted_text()] for module in modules], axis=0), axis=0)
        return np.max(np.stack([headers, length], axis=0), axis=0)

    @classmethod
    def _calc_max_time_length(cls, modules):
        length = max([len(module.time.to_formatted_text()) for module in modules], default=0)
        return max(length, len(cls.time_header))

    @classmethod
    def _calc_max_parameter_length(cls):
        return len(cls._to_parameter_format("", "", "", "", ""))

    @classmethod
    def _calc_max_requires_grad_length(cls):
        return max([len(cls._to_requires_grad_format("", "")), len(cls.requires_grad_header)])
