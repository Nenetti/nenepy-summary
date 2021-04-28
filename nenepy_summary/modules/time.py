from .abstract_module import _AbstractModule


class _Time(_AbstractModule):
    max_length = 0

    def __init__(self, time):
        """

        Args:

        """
        super(_Time, self).__init__()
        self.time = time

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================
    def to_formatted_text(self):
        time_str = self._to_time_str(self.time)
        time_format = f"{time_str:>{self.max_length}}"
        return time_format

    @classmethod
    def adjust(cls, times):
        cls.max_length = max([cls._get_max_text_length(cls._to_time_str(time.time)) for time in times])

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @staticmethod
    def _to_time_str(time):
        t = time * 1000
        if int(t) > 0:
            return f"{t:.2f}"
        else:
            return f"{t:.2f}"[1:]
