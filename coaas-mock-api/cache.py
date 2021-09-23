from collections import OrderedDict
from datetime import datetime

# Implementing a simple fixed sized in-memory cache
class Cache:
    def __init__(self, size):
        self.cache_spec = LimitedSizeDict(size_limit = size)
    
    # Insert/Update to cache by key
    def save(self, cacheitems) -> None:
        for key, value in cacheitems.items():
            if(key not in self.freq_table):
                self.cache_spec.freq_table[key] = (0,[])
            self.cache_spec[key] = value

    # Evicts an item from cache
    def evict(self, key) -> None:
        self.cache_spec.move_to_end(key,last=False)
        self.cache_spec.popitem(last=False)
        del self.cache_spec.freq_table[key]
    
    # Read the entire cache
    def get_values(self) -> dict:
        return self.cache_spec

    # Read from cache using key
    def get_value_by_key(self,key):
        if(key in self.cache_spec):
            stat = self.cache_spec.freq_table[key]
            stat[0]=+1
            stat[1].append(datetime.now())
            return self.cache_spec[key]
        else:
            return None
            
    # Retrive frequency of access statistics for currently cached items
    def get_statistics(self):
        return self.cache_spec.freq_table

# Implementing cache data structure as an ordered-limited dictionary
class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        # Frequency of access counter (count,[list of timestamps])
        self.freq_table = []

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
