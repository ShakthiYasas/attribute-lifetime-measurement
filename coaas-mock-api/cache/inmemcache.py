import time
import threading
from datetime import datetime

from cache.cacheagent import CacheAgent
from lib.limitedsizedict import LimitedSizeDict
from lib.fifoqueue import FIFOQueue, FIFOQueue_2
from eviction.evictorfactory import EvictorFactory

# Implementing a simple fixed sized in-memory cache
class InMemoryCache(CacheAgent):
    def __init__(self, config, db):      
        # Data structure of the cache
        self.__entityhash = LimitedSizeDict(size_limit = config.cache_size)

        # Statistical configurations
        self.window = config.window_size
        self.__hitrate_trend = FIFOQueue(round(self.window/5000)) 
        self.__localstats = []

        # Statistical DB
        self.__db = db

        # Inialize Eviction Algorithm
        self.__evictor = EvictorFactory(config.eviction_algo, self).getevictor()
        
        # Initializing background thread to calculate current hit rate.
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True               
        thread.start() 

    def run(self):
        while True:
            # Hit rate is calculated each 5 seconds
            self.calculate_hitrate()
            # Items are evicted every 5 seconds as well 
            for ent,att in self.__evictor.select_for_evict():
                self.evict_attribute(ent, att)
            time.sleep(5)

    def calculate_hitrate(self):
        local = self.__localstats.copy()
        self.__localstats.clear()
        self.__hitrate_trend.push(sum(local)/len(local))
    
    # Insert/Update to cache by key
    def save(self, entityid, cacheitems) -> None:
        now = datetime.datetime.now()
        if(entityid in self.__entityhash):
            for att_name, values in cacheitems.items():
                if(att_name not in self.__entityhash[entityid].freq_table):
                    self.__entityhash[entityid].freq_table[att_name] = (FIFOQueue_2(100).push(now),now)
                self.__entityhash[entityid][att_name] = values
        else:
            self.__entityhash[entityid] = LimitedSizeDict()
            self.__entityhash.freq_table[entityid] = (FIFOQueue_2(100).push(now),now)

            for att_name, values in cacheitems.items():
                self.__entityhash[entityid].freq_table[att_name] = (FIFOQueue_2().push(now),now)
                self.__entityhash[entityid][att_name] = values

    # Evicts an entity from cache
    def evict(self, entityid) -> None:
        now = datetime.now()
        attribute_lifetimes = []
        
        for key,value in self.__entityhash[entityid].freq_table.items():
            attribute_lifetimes.append({
                'entityid': entityid,
                'attribute': key,
                'c_lifetime': now - value[2]
            })

        # Push cached lifetime entity to Statistical DB
        self.__db.insert_one('entity-cached-lifetime',{
            'entityid': entityid,
            'c_lifetime': now - self.__entityhash.freq_table[entityid][2]
        })

        # Push cached lifetimes of all the attributes to Statistical DB
        self.__db.insert_many('attribute-cached-lifetime', attribute_lifetimes)

        del self.__entityhash[entityid]
        del self.__entityhash.freq_table[entityid]

    # Evicts an attribute of an entity from cache
    def evict_attribute(self, entityid, attribute) -> None:
        now = datetime.now()
        # Push cached lifetime of attribute to Statistical DB
        att_meta = self.__entityhash[entityid].freq_table[attribute]
        self.__db.insert_one('attribute-cached-lifetime',{
                'entityid': entityid,
                'attribute': attribute,
                'c_lifetime': now - att_meta[2]
            })

        del self.__entityhash[entityid][attribute]
        del self.__entityhash[entityid].freq_table[attribute]

    # Get all attributes cached for an entity
    def get_attributes_of_entity(self,entityid) -> LimitedSizeDict:
        return list(self.__entityhash[entityid].keys())

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
            att_stat[0].push(datetime.now())
            self.__entityhash[entityid].freq_table[attribute] = tuple(att_stat)
            
            # Updating frequency table for entities
            ent_stat = list(self.__entityhash.freq_table[entityid])
            ent_stat[0].push(datetime.now())
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
    def get_statistics_all(self)->dict:
        return self.__entityhash.freq_table

    # Retrive frequency of access of all attribute of an entity
    def get_statistics_entity(self, entityid)->dict:
        return self.__entityhash[entityid].freq_table

    # Retrive frequency of access statistics for a context attribute
    def get_statistics(self, entityid, attribute):
        return self.__entityhash[entityid].freq_table[attribute]

    # Returns the trend of the hit rate with in the moving window
    def get_hitrate_trend(self):
        return self.__hitrate_trend

