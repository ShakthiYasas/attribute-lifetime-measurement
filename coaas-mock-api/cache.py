from collections import OrderedDict

# Implementing a simple fixed sized in-memory cache
class Cache:
    def __init__(self, size):
        self.cache_spec = LimitedSizeDict(size_limit = size)
    
    # Insert/Update to cache by key
    def save(self, cacheitems) -> None:
        for key, value in cacheitems.items():
            self.cache_spec[key] = value

    # Evicts an item from cache
    def evict(self, key) -> None:
        self.cache_spec.move_to_end(key,last=False)
        self.cache_spec.popitem(last=False)
    
    # Read from cache using key
    def get_values(self) -> dict:
        return self.cache_spec

# Implementing cache data structure as an ordered-limited dictionary
class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
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
                self.popitem(last=False)
