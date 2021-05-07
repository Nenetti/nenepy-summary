import sys
import time
from time import sleep

import torch
import torch.nn as nn

from .modules import _BlockPrinter
from .modules import _Module


class TorchSummary:

    def __init__(self,
                 model,
                 batch_size=2,
                 is_train=True,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 is_print=True,
                 display_delay_time=0,
                 is_exit=False,
                 forward_function=None,
                 ):
        """

        Args:
            model (nn.Module):
            batch_size (int):
            is_train (bool):
            device (str):
            is_print (bool):
            display_delay_time (float):
            is_exit (bool):

        """
        self.model = model.to(device)
        self.batch_size = batch_size
        self.display_delay_time = display_delay_time
        self.device = device
        self.is_print = is_print
        self.hooks = []
        self.modules = []
        self.modules_dict = dict()
        self.roots = []
        self.ordered_modules = []
        self.is_exit = is_exit
        self.forward_function = None
        if forward_function is not None:
            if isinstance(forward_function, str):
                self.forward_function = forward_function
            else:
                self.forward_function = forward_function.__name__
        if is_train:
            self.model.train()
        else:
            self.model.eval()

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward_size(self, *input_size, **kwargs):
        if not isinstance(input_size[0], (tuple, list, dict, set)):
            input_size = [input_size]

        x = [torch.randn(self.batch_size, *in_size).to(self.device) for in_size in input_size]

        return self._forward(x, **kwargs)

    def forward_tensor(self, input_tensor, **kwargs):
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.to(self.device)
            input_tensor = [input_tensor]

        elif not hasattr(input_tensor, "__iter__"):
            input_tensor = [input_tensor]

        return self._forward(input_tensor, **kwargs)

    def forward_dict(self, input_dict):
        input_dict = dict([(key, value.to(self.device) if isinstance(value, torch.Tensor) else value) for key, value in input_dict.items()])
        return self._forward_dict(input_dict)

    def __call__(self, input_tensor, **kwargs):
        return self.forward_tensor(input_tensor, **kwargs)

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    def _forward_pre_hook(self):
        self.model.apply(self._register_hook)

    def _forward_hook(self):
        sleep(self.display_delay_time)
        if self.is_print:
            self._print_network()

        self._remove()

        if self.is_exit:
            sys.exit()

    def _forward(self, x, **kwargs):
        self._forward_pre_hook()
        if self.forward_function is None:
            out = self.model(*x, **kwargs)
        else:
            out = getattr(self.model, self.forward_function)(*x, **kwargs)
        self._forward_hook()
        return out

    def _forward_dict(self, kwargs):
        self._forward_pre_hook()
        if self.forward_function is None:
            out = self.model(**kwargs)
        else:
            out = getattr(self.model, self.forward_function)(**kwargs)
        self._forward_hook()
        return out

    def _print_network(self):
        printers = [_BlockPrinter(module) for module in self.ordered_modules]
        _BlockPrinter.adjust(self.ordered_modules)

        print(_BlockPrinter.to_header_text())

        needs_bottom_space = 1
        for printer in printers:
            print_formats, needs_bottom_space = printer.to_formatted_texts(needs_bottom_space)
            for print_format in print_formats:
                print(print_format)

        print(_BlockPrinter.to_header_text(reverse=True))
        print(_BlockPrinter.to_footer_text())

        print()
        print(_Module.to_summary_text(self.ordered_modules))
        print()

    def _remove(self):
        for h in self.hooks:
            h.remove()
        del self.hooks
        del self.modules
        del self.roots
        del self.ordered_modules

    # ==================================================================================================
    #
    #   Hook
    #
    # ==================================================================================================
    def _register_hook(self, module):
        """

        Args:
            module (nn.Module):

        """
        # if (isinstance(module, nn.Sequential)) or (isinstance(module, nn.ModuleList)):
        #     return

        self.hooks.append(module.register_forward_pre_hook(self._pre_hook))
        self.hooks.append(module.register_forward_hook(self._hook))

    def _pre_hook(self, module, module_in):
        """

        Args:
            module (nn.Module):
            module_in:

        """

        module_id = len(self.modules_dict) + 1
        is_duplicated = False
        if module in self.modules_dict:
            module_id = self.modules_dict[module]
            is_duplicated = True
        else:
            self.modules_dict[module] = module_id

        summary_module = _Module(module, module_id, is_duplicated)
        if len(self.modules) == 0:
            summary_module.is_root = True
            self.roots.append(summary_module)

        self.ordered_modules.append(summary_module)
        self.modules.append((summary_module, time.time()))

    def _hook(self, module, module_in, module_out):
        """

        Args:
            module (nn.Module):
            module_in:
            module_out:

        """
        summary_module, start_time = self.modules.pop(-1)
        summary_module.processing_time = time.time() - start_time

        summary_module.init_in_out(module_in, module_out)

        if len(summary_module.child_modules) > 0:
            summary_module.child_modules[-1].is_last_module_in_sequential = True

        if len(self.modules) > 0:
            parent_block = self.modules[-1][0]
            parent_block.child_modules.append(summary_module)
            summary_module.parent_module = parent_block
