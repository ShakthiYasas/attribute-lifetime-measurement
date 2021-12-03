import statistics
import time
import _thread
import threading
from datetime import datetime, timedelta

from lib.fifoqueue import FIFOQueue_2
from cache.cacheagent import CacheAgent
from lib.exceptions import OutOfCacheMemory
from lib.event import post_event_with_params
from lib.limitedsizedict import LimitedSizeDict
from cache.eviction.evictorfactory import EvictorFactory

# Implementing a simple fixed sized in-memory cache
class InMemoryCache(CacheAgent):
    def __init__(self, config, db, registry):      
        # Data structure of the cache
        self.__cache_size = config.cache_size
        self.is_auto_evicting = config.isevict
        self.is_limited_cache = config.limited_cache
        self.__entityhash = LimitedSizeDict(size_limit = self.__cache_size)
        self.__cache_write_lock = threading.Lock()

        # Access to SQLite instance
        self.__registry = registry

        # Statistical configurations
        self.window = config.window_size
        self.__hitrate_trend = FIFOQueue_2(100) 
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
            # Hit rate is calculated in each window
            self.calculate_hitrate()

            # Items are evicted every each window
            if(self.__evictor and self.__entityhash.is_full()):
                items_to_evict = self.__evictor.select_for_evict()
                if(isinstance(items_to_evict,list)):
                    if(all(isinstance(item, tuple) for item in items_to_evict)):
                        for ent,att in items_to_evict:
                            self.__evict_attribute(ent, att)
                            if(not len(self.__entityhash[ent])):
                                self.__evict(ent)
                    else:
                        for ent in items_to_evict:
                            self.__evict(ent)
                else:
                    self.__evict(items_to_evict)

            _thread.start_new_thread(self.__reset_provider_recency, ())

            time.sleep(self.window/1000)
    
    def __reset_provider_recency(self):
        threads = []
        copy_of_hashtable = self.__entityhash.copy()
        for entityid in copy_of_hashtable.keys():
            copy_of_attributes = copy_of_hashtable[entityid].copy().keys()
            for attr in copy_of_attributes:
                en_re_th = threading.Thread(target=self.__update_cache_providers(entityid, attr))
                en_re_th.start()
                threads.append(en_re_th)
            
            for t in threads:
                t.join()

    # Removes all the recently not used provider's cached data
    def __update_cache_providers(self, entityid, attr):
        is_att_evicted = False
        copy_of_cached_items = self.__entityhash[entityid][attr].copy()
        for cache_item in copy_of_cached_items:
            if(not cache_item[3] and self.is_auto_evicting):
                self.__cache_write_lock.acquire()
                self.__entityhash[entityid][attr].remove(cache_item)
                self.__cache_write_lock.release()

                if(not self.__entityhash[entityid][attr]):
                    self.__evict_attribute(entityid, attr)
                    is_att_evicted = True

        if(not is_att_evicted):
            self.__cache_write_lock.acquire() 
            self.__entityhash[entityid][attr] = [(cache_item[0], cache_item[1], cache_item[2], False) for cache_item in self.__entityhash[entityid][attr]]
            self.__cache_write_lock.release()

    def calculate_hitrate(self):
        local = self.__localstats.copy()
        self.__localstats.clear()
        self.__hitrate_trend.push((sum(local)/len(local) if local else 0,len(local)))
    
    def getdb(self):
        return self.__db
    
    def get_cachedlifetime(self, entityid, attribute):
        in_db = self.__registry.get_cached_life(entityid, attribute)
        if(in_db):
            return in_db
        else:
            cached_time = self.__entityhash[entityid].freq_table[attribute][1]
            avg_cached_lt = 0
            cached_lt_res = self.__db.read_all_with_limit('attribute-cached-lifetime',{
                    'entity': entityid,
                    'attribute': attribute
                },10)
            if(cached_lt_res):
                avg_lts = list(map(lambda x: x['c_lifetime'], cached_lt_res))
                avg_cached_lt = round(statistics.mean(avg_lts),3)
            exp_time = cached_time + timedelta(seconds=avg_cached_lt)
            return (exp_time, cached_time)

    def get_expired_lifetimes(self):
        return self.__registry.get_expired_cached_lifetimes()
    
    def get_ret_latency(self):
        return self.__registry.get_ret_latency()

    # Insert/Update to cache by key
    def save(self, entityid, cacheitems) -> None:
        recency_bit = True
        if(entityid in self.__entityhash):
            now = datetime.now()
            for att_name, values in cacheitems.items():
                if(att_name in self.__entityhash[entityid].freq_table):
                    for prodid, res, ret_time in values:
                        current = [item for item in self.__entityhash[entityid][att_name] if item[0] == prodid]
                        new_record = (prodid, res, ret_time, recency_bit)
                        if(current):
                            self.__entityhash[entityid][att_name].remove(current[0])
                        self.__entityhash[entityid][att_name].append(new_record)
                else:
                    que = FIFOQueue_2(100)
                    que.push(now)
                    self.__entityhash[entityid].freq_table[att_name] = (que,now)
                    self.__entityhash[entityid][att_name] = [list(tup)+[recency_bit] for tup in values]   
        else:
            result = self.__create_new_entity_cache(entityid, cacheitems)
            if(result == -1):
                # Check if any entities can be removed from cache
                if(self.__evictor != None):
                    entity_ids = self.__evictor.select_entity_to_evict(is_limited=self.is_limited_cache)
                    if(entity_ids):
                        # If there can be, then remove them and replace by the new ones 
                        for ent_id in entity_ids:
                            self.__evict(ent_id)
                    else:
                        # If not, then call the expansion routine and then save the item
                        self.__entityhash.expand_dictionary(self.__cache_size)
                else:
                    # If not, then call the expansion routine and then save the item
                    self.__entityhash.expand_dictionary(self.__cache_size)

                _ = self.__create_new_entity_cache(entityid, cacheitems) 
                      
    def __create_new_entity_cache(self, entityid, cacheitems):
        try:
            recency_bit = True
            now = datetime.now()
            self.__entityhash[entityid] = LimitedSizeDict()
            que = FIFOQueue_2(100)
            que.push(now)
            self.__entityhash.freq_table[entityid] = (que,now)

            for att_name, values in cacheitems.items():
                que_1 = FIFOQueue_2(100)
                que_1.push(now)
                self.__entityhash[entityid].freq_table[att_name] = (que_1,now)
                self.__entityhash[entityid][att_name] = [list(tup)+[recency_bit] for tup in values]
            return 0
        except OutOfCacheMemory:
            print('Out of cache memory')
            return -1

    # Add to cached lifetime
    def addcachedlifetime(self, action, cachedlife):
        self.__registry.add_cached_life(action[0], action[1], cachedlife)

    # Update cached lifetime
    def updatecachedlifetime(self, action, cachedlife):
        self.__registry.update_cached_life(action[0], action[1], cachedlife)
    
    def removeentitycachedlifetime(self, entityid):
        self.__registry.remove_cached_life_entity(entityid)
    
    def removecachedlifetime(self,entity, attribute):
        self.__registry.remove_cached_life(entity, attribute)

    # Evicts an entity from cache
    def __evict(self, entityid) -> None:
        now = datetime.now()
        attribute_lifetimes = []
        
        if(self.__entityhash[entityid].freq_table.items()):
            for key,value in self.__entityhash[entityid].freq_table.items():
                cachedtime = None
                if(len(value)>2):
                    cachedtime = value[2]
                else:
                    cachedtime = value[1]
                attribute_lifetimes.append({
                    'entityid': entityid,
                    'attribute': key,
                    'c_lifetime': (now - cachedtime).total_seconds()
                })
                self.__registry.remove_cached_life(entityid, key)
            
            # Push cached lifetimes of all the attributes to Statistical DB
            self.__db.insert_many('attribute-cached-lifetime', attribute_lifetimes)

        # Push cached lifetime entity to Statistical DB
        self.__db.insert_one('entity-cached-lifetime',{
            'entityid': entityid,
            'c_lifetime': (now - self.__entityhash.freq_table[entityid][1]).total_seconds()
        })

        del self.__entityhash[entityid]
        del self.__entityhash.freq_table[entityid]

        # Publish this evition to stats in Adaptive
        post_event_with_params("subscribed_evictions", (entityid, None))

    # Evicts an attribute of an entity from cache
    def __evict_attribute(self, entityid, attribute) -> None:
        now = datetime.now()
        # Push cached lifetime of attribute to Statistical DB
        att_meta = self.__entityhash[entityid].freq_table[attribute]
        self.__db.insert_one('attribute-cached-lifetime',{
                'entityid': entityid,
                'attribute': attribute,
                'c_lifetime': (now - att_meta[1]).total_seconds()
            })

        self.__registry.remove_cached_life(entityid, attribute)
        del self.__entityhash[entityid][attribute]
        del self.__entityhash[entityid].freq_table[attribute]

        # Publish this evition to stats in Adaptive
        post_event_with_params("subscribed_evictions", (entityid, attribute))

    # Get all attributes cached for an entity
    def get_attributes_of_entity(self,entityid) -> LimitedSizeDict:
        return list(self.__entityhash[entityid].keys())

    # Check if the entity is cached
    def is_cached(self,entityid,attribute):
        res = entityid in self.__entityhash.freq_table and attribute in self.__entityhash[entityid].freq_table
        if(res):
            self.__localstats.append(1)
        else:
            self.__localstats.append(0)
        return res

    # Check if the all the attributes requested for the entity is cached
    def are_all_atts_cached(self,entityid,attributes):
        uncached = set()
        for attribute in attributes:
            if(not(entityid in self.__entityhash and attribute in self.__entityhash[entityid])):
                uncached.add(attribute)

        if(len(uncached)>0): 
            return False, uncached
        return True, []

    # Read from cache using key
    def get_value_by_key(self,entityid,attribute):
        # Check if both the entity and the the attribute are already cached
        if(self.is_cached(entityid,attribute)):
            # Updating frequency table for context attributes
            att_stat = list(self.__entityhash[entityid].freq_table[attribute])
            att_stat[0].push(datetime.now())
            self.__entityhash[entityid].freq_table[attribute] = tuple(att_stat)
            
            # Updating frequency table for entities
            ent_stat = list(self.__entityhash.freq_table[entityid])
            ent_stat[0].push(datetime.now())
            self.__entityhash.freq_table[entityid] = tuple(ent_stat)

            self.__entityhash[entityid][attribute] = [(data[0], data[1], data[2], True) for data in self.__entityhash[entityid][attribute]]
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
    def get_statistics_entity(self, entityid):
        if(entityid in self.__entityhash):
            return self.__entityhash[entityid].freq_table
        else: return None

    # Retrive frequency of access statistics for a context attribute
    def get_statistics(self, entityid, attribute):
        if(entityid in self.__entityhash and attribute in self.__entityhash[entityid].freq_table):
            return self.__entityhash[entityid].freq_table[attribute]
        else: return None

    # Returns the trend of the hit rate with in the moving window
    def get_hitrate_trend(self):
        return self.__hitrate_trend

    def get_providers_for_entity(self, entityid):
        return self.__registry.get_providers_for_entity(entityid)
    
    def get_providers_for_attribute(self, entityid, attrlist):
        return self.__registry.get_context_producers(entityid, attrlist)
    
    def reevaluate_for_eviction(self, entity, attribute):
        return self.caller_strategy.reevaluate_for_eviction(entity, attribute)

    def get_longest_cache_lifetime_for_entity(self, entityid):
        in_db = self.__registry.get_longest_cache_lifetime_for_entity(entityid)
        if(in_db):
            return in_db
        else:
            max_cached = None
            max_exp = None
            for attribute in self.__entityhash[entityid].freq_table.keys():
                cached_time = self.__entityhash[entityid].freq_table[attribute][1]
                avg_cached_lt = 0
                cached_lt_res = self.__db.read_all_with_limit('attribute-cached-lifetime',{
                        'entity': entityid,
                        'attribute': attribute
                    },10)
                if(cached_lt_res):
                    avg_lts = list(map(lambda x: x['c_lifetime'], cached_lt_res))
                    avg_cached_lt = round(statistics.mean(avg_lts),3)
            
                exp_time = cached_time + timedelta(seconds=avg_cached_lt)
                if(max_exp == None or exp_time > max_exp):
                    max_cached = cached_time
                    max_exp = exp_time


            return (exp_time, max_cached)
