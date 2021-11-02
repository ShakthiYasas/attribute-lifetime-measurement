from cacheagent import CacheAgent
from cloudcache import CloudCache
from inmemcache import InMemoryCache

from configurations.cacheconfig import CacheConfiguration

# Instantiate a cache memory instance according to configuration
class CacheFactory:
    # Class varaible
    __cache = None

    def __init__(self, configuration, registry=None):
        self.__configuration = configuration
        self.__registry = registry

    # Retruns the singleton instance of a cache
    def get_cache_memory(self, db) -> CacheAgent:
        if(self.cache == None):
            if(self.__configuration != None and isinstance(self.__configuration, CacheConfiguration) 
                and CacheConfiguration.type == 'in-memory'):
                print('Initializing a local in-memory cache instance.')
                self.__cache = InMemoryCache(self.__configuration, db, self.__registry )
            if(self.__configuration != None and isinstance(self.__configuration, CacheConfiguration)
                and CacheConfiguration.type == 'cloud'):
                print('Initializing a cloud based cache instance.')
                self.__cache = CloudCache(self.__configuration, db)
            else:
                raise ValueError('Invalid configuration.')
        
        return self.__cache