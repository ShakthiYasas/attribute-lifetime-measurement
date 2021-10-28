from cache.eviction.evictor import Evictor
from cache.inmemcache import InMemoryCache

class RandomEvictor(Evictor):
    def __init__(self, parent_cache:InMemoryCache):
        self.__cache = parent_cache

    def evict(self):
        entityhash = self.__cache.get_values()
        current_occupancy = len(entityhash.freq_table.items())/entityhash.size_limit
        if(current_occupancy>0):
            # Select what to evict
            # Commenting out for now until eviction cost model is designed
            # self.__cache.evict_attribute(entityid, attribute)
            print('Evicted!')