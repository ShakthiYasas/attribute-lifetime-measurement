from inmemcache import InMemoryCache
# from cloudcache import CloudCache
from configurations.cacheconfig import CacheConfiguration

# Instantiate a cache memory instance according to configuration
class CacheFactory:
    # Class varaible
    cache = None
    configuration = None

    def __init__(self, configuration = None):
        if(configuration != None):
            self.configuration = configuration

    # Retruns the singleton instance of a cache
    def get_cache_memory(self) -> Cache:
        if(self.cache == None):
            if(self.configuration != None and isinstance(self.configuration,CacheConfiguration)):
                return InMemoryCache(self.configuration)
            # if(self.configuration != None and isinstance(self.configuration,CloudCacheConfiguration)):
            #    return Cache(self.configuration)
            else:
                raise ValueError('Invalid configuration.')
        else:
            return self.cache