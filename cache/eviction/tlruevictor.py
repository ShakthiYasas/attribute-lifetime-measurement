from cache.eviction.evictor import Evictor

class TLRUEvictor(Evictor):
    def __init__(self, parent_cache, threshold = 1.0):
        self.__cache = parent_cache