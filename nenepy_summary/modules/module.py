from collections import Counter

from torch import nn

from .abstract_module import _AbstractModule, _Align
from .input import _Input
from .network_architecture import _NetworkArchitecture
from .output import _Output
from .parameter import _Parameter
from .time import _Time


class _Module(_AbstractModule):

    def __init__(self, module, module_id, is_duplicated):
        """

        Args:
            module (nn.Module):
        """
        super(_Module, self).__init__()
        self.module = module
        self.module_id = module_id
        self.is_duplicated = is_duplicated
        self.input = None
        self.output = None
        self.parameter = None
        self.network_architecture = None
        self.time = None

        self.parent_module = None
        self.child_modules = []
        self.processing_time = 0
        self.is_root = False
        self.is_last_module_in_sequential = False

    def init_in_out(self, module_in, module_out):
        self.input = _Input(self.module, module_in)
        self.output = _Output(module_out)
        self.parameter = _Parameter(self.module)
        self.network_architecture = _NetworkArchitecture(self)
        self.time = _Time(self.processing_time)

    def has_children(self):
        return len(self.child_modules) > 0

    @property
    def module_name(self):
        module_id = f"(*{self.module_id})" if self.is_duplicated else f"({self.module_id})"
        return f"{module_id} {self.module.__class__.__name__}"

    @classmethod
    def to_summary_text(cls, modules):
        types = [(cls._to_type(module.module), cls.to_module_class_path(module.module)) for module in modules]
        counter = Counter(types)
        class_length = max([len(t[0]) for t in types], default=0)
        count_length = len(str(counter.most_common()[0][1]))
        classes = counter.most_common()
        texts = [""] * len(classes)
        for i, (key, count) in enumerate(classes):
            texts[i] = f"{cls._align(key[0], class_length, _Align.Right)}: {cls._align(str(count), count_length, _Align.Right)}   ({key[1]})"

        return "\n".join(texts)

    @staticmethod
    def to_module_class_path(module):
        text = str(type(module))
        return text[8:len(text) - 2]

    @staticmethod
    def module_path_to_name(module_path):
        text = module_path.split(".")[-1]
        return text
