from cache.eviction.evictor import Evictor

# How many attributes and entities to evict?
# When to evict?
class LFUEvictor(Evictor):
    def __init__(self, cache):
        self.__cache = cache
    
    def select_for_evict(self):
        cached_access_ratio = list(map(lambda entityid,stat: 
            (entityid, stat[0].get_queue_size()/(stat[0].get_head()-stat[0].get_last()).total_seconds()), 
            self.__cache.get_statistics_all().items()))
        sorted_c_ar = sorted(cached_access_ratio, key=lambda tup: tup[1])
        least_frequently_used_entity = sorted_c_ar[-1][0]
   
        att_access_ratio = list(map(lambda att,stat: 
            (att, stat[0].get_queue_size()/(stat[0].get_head()-stat[0].get_last()).total_seconds()), 
            self.__cache.get_statistics_entity(least_frequently_used_entity).items()))
        att_c_ar = sorted(att_access_ratio, key=lambda tup: tup[1])
        least_frequently_used_att = att_c_ar[-1][0]

        return [(least_frequently_used_entity,least_frequently_used_att)]