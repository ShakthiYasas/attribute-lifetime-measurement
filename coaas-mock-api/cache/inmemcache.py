from datetime import datetime

from cache.cacheagent import CacheAgent
from lib.limitedsizedict import LimitedSizeDict

# Implementing a simple fixed sized in-memory cache
class InMemoryCache(CacheAgent):
    def __init__(self, size):
        self.__entityhash = LimitedSizeDict(size_limit = size)
    
    # Insert/Update to cache by key
    def save(self, entityid, cacheitems) -> None:
        if(entityid in self.__entityhash):
            for key, value in cacheitems.items():
                if(key not in self.__entityhash[entityid].freq_table):
                    self.__entityhash[entityid].freq_table[key] = (0,[])
                self.__entityhash[entityid][key] = value
        else:
            self.__entityhash[entityid] = LimitedSizeDict()
            for key, value in cacheitems.items():
                self.__entityhash[entityid].freq_table[key] = (0,[])
                self.__entityhash[entityid][key] = value

    # Evicts an entity from cache
    def evict(self, entityid) -> None:
        self.__entityhash.move_to_end(entityid,last=False)
        self.__entityhash.popitem(last=False)
        del self.__entityhash.freq_table[entityid]
    
    # Read the entire cache
    def get_values(self) -> dict:
        return self.__entityhash

    # Check if the entity is cached
    def _is_cached(self,entityid,attribute):
        return entityid in self.__entityhash and attribute in self.__entityhash[entityid]        

    # Read from cache using key
    def get_value_by_key(self,entityid,attribute):
        # Check if both the entity and the the attribute are already cached
        if(self._is_cached(entityid,attribute)):
            # Updating frequency table for context attributes
            att_stat = list(self.__entityhash[entityid].freq_table[attribute])
            att_stat[0]=+1
            att_stat[1].append(datetime.now())
            self.__entityhash[entityid].freq_table[attribute] = tuple(att_stat)
            
            # Updating frequency table for entities
            ent_stat = list(self.__entityhash.freq_table[entityid])
            ent_stat[0]=+1
            ent_stat[1].append(datetime.now())
            self.__entityhash.freq_table[entityid] = tuple(ent_stat)

            return self.__entityhash[entityid][attribute]
        else:
            return None
            
    # Retrive frequency of access statistics for currently cached entities
    def get_statistics(self):
        return self.__entityhash.freq_table

    # Retrive frequency of access statistics for a context attribute
    def get_statistics(self, entityid, attribute):
        return self.__entityhash[entityid].freq_table[attribute]


