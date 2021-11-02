from cache.eviction.evictor import Evictor
from cache.inmemcache import InMemoryCache

class LFUEvictor(Evictor):
    def __init__(self, cache:InMemoryCache):
        self.__cache = cache
    
    # How many attributes and entities to evict?
    # When to evict?
    def select_for_evict(self):
        cached_access_ratio = list(map(lambda entityid,stat: (entityid, stat[0]/(stat[1][0]-stat[1][-1]).total_seconds()), 
                    self.__cache.get_statistics()))
        sorted_c_ar = sorted(cached_access_ratio, key=lambda tup: tup[1])
        least_frequently_used_entity = sorted_c_ar[-1][0]
   
        att_access_ratio = list(map(lambda att,stat: (att, stat[0]/(stat[1][0]-stat[1][-1]).total_seconds()), 
                    self.__cache.get_statistics(least_frequently_used_entity)))
        att_c_ar = sorted(att_access_ratio, key=lambda tup: tup[1])
        least_frequently_used_att = att_c_ar[-1][0]

        return [(least_frequently_used_entity,least_frequently_used_att)]