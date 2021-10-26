from cacheagent import CacheAgent
from cloudcache import CloudCache
from inmemcache import InMemoryCache

from configurations.cacheconfig import CacheConfiguration

# Instantiate a cache memory instance according to configuration
class CacheFactory:
    # Class varaible
    __cache = None

    def __init__(self, configuration = None):
        if(configuration != None):
            self.__configuration = configuration

    # Retruns the singleton instance of a cache
    def get_cache_memory(self) -> CacheAgent:
        if(self.cache == None):
            if(self.__configuration != None and isinstance(self.__configuration, CacheConfiguration) 
                and CacheConfiguration.type == 'in-memory'):
                self.__cache = InMemoryCache(self.__configuration)
            if(self.__configuration != None and isinstance(self.__configuration, CacheConfiguration)
                and CacheConfiguration.type == 'cloud'):
                self.__cache = CloudCache(self.__configuration)
            else:
                raise ValueError('Invalid configuration.')
        
        return self.__cache