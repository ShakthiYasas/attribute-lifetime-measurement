import time
import threading
from datetime import datetime

from cache.cacheagent import CacheAgent
from lib.limitedsizedict import LimitedSizeDict
from lib.fifoqueue import FIFOQueue

# Implementing a simple fixed sized in-memory cache
class InMemoryCache(CacheAgent):
    def __init__(self, config):      
        # Data structure of the cache
        self.__entityhash = LimitedSizeDict(size_limit = config.cache_size)

        # Statistical configurations
        self.window = config.window_size
        self.__hitrate_trend = FIFOQueue(round(self.window/5000)) 
        self.__localstats = []
        
        # Initializing background thread to calculate current hit rate.
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True               
        thread.start() 

    def run(self):
        while True:
            self.calculate_hitrate()
            # Hit rate is calculated each 5 seconds
            time.sleep(5)

    async def calculate_hitrate(self):
        local = self.__localstats.copy()
        self.__localstats.clear()
        self.__hitrate_trend.push(sum(local)/len(local))
    
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
    def __is_cached(self,entityid,attribute):
        res = entityid in self.__entityhash and attribute in self.__entityhash[entityid]
        if(res):
            self.__localstats.append(1)
        else:
            self.__localstats.append(0)
        return res

    # Check if the all the attributes requested for the entity is cached
    def are_all_atts_cached(self,entityid,attributes):
        for attribute in attributes:
            if(not(entityid in self.__entityhash and attribute in self.__entityhash[entityid])):
                return False
        return True

    # Read from cache using key
    def get_value_by_key(self,entityid,attribute):
        # Check if both the entity and the the attribute are already cached
        if(self.__is_cached(entityid,attribute)):
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

    # Get all attribute values for an entity
    def get_values_for_entity(self,entityid,attr_list):
        output = {}
        for att in attr_list:
            output[att] = self.get_value_by_key(entityid,att)
        return output

    # Retrive frequency of access statistics for currently cached entities
    def get_statistics(self):
        return self.__entityhash.freq_table

    # Retrive frequency of access statistics for a context attribute
    def get_statistics(self, entityid, attribute):
        return self.__entityhash[entityid].freq_table[attribute]

    # Returns the trend of the hit rate with in the moving window
    def get_hitrate_trend(self):
        return self.__hitrate_trend

