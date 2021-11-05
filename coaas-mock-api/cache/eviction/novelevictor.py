from cache.eviction.evictor import Evictor
from cache.inmemcache import InMemoryCache

class NovelEvictor(Evictor):
    def __init__(self, cache:InMemoryCache):
        self.__cache = cache

    def select_for_evict(self):

        self.__cache.get_cachedlifetime()
        return [(0,0)]