from collections import OrderedDict

# Implementing cache data structure as an ordered-limited dictionary
class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        # Frequency of access counter (count,[list of timestamps])
        self.freq_table = {}

        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    # This is a FIFO replacement
    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                out = self.popitem(last=False)
                del self.freq_table[out[0]]