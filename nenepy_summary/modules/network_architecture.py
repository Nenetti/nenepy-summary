from .abstract_module import _AbstractModule


class _NetworkArchitecture(_AbstractModule):

    def __init__(self, module):
        """

        Args:
            module (Module):

        """
        super(_NetworkArchitecture, self).__init__()
        self.module = module

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================
    @classmethod
    def adjust(cls, architectures):
        cls.n_max_length = cls._get_max_text_length([architecture.to_formatted_text() for architecture in architectures])

    def to_formatted_text(self):
        return self._to_text_format(self.module)

    def to_bottom_formant(self):
        def recursive(m):
            if m is not None and m.parent_module is not None:
                parent_format = recursive(m.parent_module)
                self_format = self._to_connection_format(m)
                return f"{parent_format}{self_format}"
            else:
                return ""

        if len(self.module.child_modules) > 0:
            return recursive(self.module) + self._to_connect_format()
        else:
            return recursive(self.module)

    def to_top_format(self):
        def recursive(m):
            if m is not None and m.parent_module is not None:
                parent_format = recursive(m.parent_module)
                self_format = self._to_connection_format(m)
                return f"{parent_format}{self_format}"
            else:
                return ""

        if self.module.parent_module is not None:
            return recursive(self.module.parent_module) + self._to_connect_format()
        else:
            return self._to_unconnect_format()

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def _to_text_format(cls, module):
        parent_directory_format = cls._to_parent_formant(module)
        self_directory_format = cls._to_directory_format(module)
        return f"{parent_directory_format}{self_directory_format}{module.module_name}"

    @classmethod
    def _to_parent_formant(cls, module):
        """

        Args:
            module (Module):

        Returns:

        """

        def recursive(m):
            if m is not None and m.parent_module is not None:
                parent_format = recursive(m.parent_module)
                self_format = cls._to_connection_format(m)
                return f"{parent_format}{self_format}"
            else:
                return ""

        return recursive(module.parent_module)

    @classmethod
    def _to_connection_format(cls, module):
        if module.is_last_module_in_sequential:
            return cls._to_unconnect_format()
        else:
            return cls._to_connect_format()

    @classmethod
    def _to_connect_format(cls):
        return f"{'│ ':>{cls.indent_space}}"

    @classmethod
    def _to_unconnect_format(cls):
        return f"{'':>{cls.indent_space}}"

    @classmethod
    def _to_directory_format(cls, module):
        if module.is_root:
            return ""
        else:
            directory_type = cls._to_directory_type(module)
            return f"{directory_type:>{cls.indent_space}}"

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @staticmethod
    def _to_directory_type(module):
        if module.is_root:
            return ""
        if module.is_last_module_in_sequential:
            return "└ "
        else:
            return "├ "
