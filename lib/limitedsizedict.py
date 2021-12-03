from collections import OrderedDict
from lib.exceptions import OutOfCacheMemory

# Implementing cache data structure as an ordered-limited dictionary
class LimitedSizeDict(OrderedDict):
    def __init__(self, enable_eviction=False, *args, **kwds,):
        # Frequency of access counter (count,[list of timestamps])
        self.freq_table = {}
        self.__enable_auto_eviction = enable_eviction

        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        if(self.is_full()):
            if(self.__enable_auto_eviction):
                self._check_size_limit()
            else:
                raise OutOfCacheMemory
        else:
            OrderedDict.__setitem__(self, key, value)
        
    # This is a FIFO replacement
    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                out = self.popitem(last=False)
                del self.freq_table[out[0]]

    # Return whether cache is full
    def is_full(self):
        if(self.size_limit == None):
            return False
        else:
            return len(self) >= self.size_limit

    # Expand the dictionary by the same factor
    def expand_dictionary(self, size):
        if(self.size_limit != None):
            self.size_limit += size 
            print('Expanded Cache Memory and new size: '+str(self.size_limit))

    # Return the size of the cache
    def get_size(self):
        return self.size_limit
