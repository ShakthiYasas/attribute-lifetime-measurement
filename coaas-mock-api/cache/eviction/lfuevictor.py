from cache.eviction.evictor import Evictor
from cache.inmemcache import InMemoryCache

class LFUEvictor(Evictor):
    def __init__(self, cache:InMemoryCache):
        pass