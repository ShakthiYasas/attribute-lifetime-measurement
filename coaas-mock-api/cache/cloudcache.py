from datetime import datetime

from cache.cacheagent import CacheAgent
from lib.limitedsizedict import LimitedSizeDict

# Implementing a simple fixed sized in-memory cache
class CloudCache(CacheAgent):
    def __init__(self, config, db):
        self.window = config.window_size

    # Evicts an entity from cache
    def evict(self, entityid) -> None: pass
    
    # Insert/Update to cache by key
    def save(self, entityid, cacheitems) -> None: pass

    # Read the entire cache
    def get_values(self) -> dict: pass

    # Read from cache using key
    def get_value_by_key(self,entityid,attribute): pass
   
    # Check if the entity is cached
    def _is_cached(self,entityid,attribute): pass
    
    # Retrive frequency of access statistics for currently cached entities
    def get_statistics(self): pass

    # Retrive frequency of access statistics for a context attribute
    def get_statistics(self, entityid, attribute): pass