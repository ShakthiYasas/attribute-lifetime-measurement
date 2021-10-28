import threading

from cache.eviction.evictor import Evictor
from cache.inmemcache import InMemoryCache

class RandomEvictor(Evictor):
    def __init__(self, parent_cache):
        self.__cache = parent_cache
        print('Random evictor statrted!')

    def evict():
        pass

    # Evicts an attribute from cache
    def evict_attribute(self, entityid, attribute) -> None:
        self.__entityhash.move_to_end(entityid,last=False)
        self.__entityhash.popitem(last=False)
        del self.__entityhash.freq_table[entityid]