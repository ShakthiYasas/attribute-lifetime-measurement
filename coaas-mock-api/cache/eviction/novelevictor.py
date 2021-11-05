from cache.eviction.evictor import Evictor

class NovelEvictor(Evictor):
    def __init__(self, cache):
        self.__cache = cache

    def select_for_evict(self):

        self.__cache.get_cachedlifetime()
        return [(0,0)]