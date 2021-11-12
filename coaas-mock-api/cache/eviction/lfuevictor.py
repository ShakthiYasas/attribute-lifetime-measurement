from cache.eviction.evictor import Evictor

class LFUEvictor(Evictor):
    def __init__(self, cache, threshold = 1.0):
        self.__cache = cache
        self.__threshold = threshold
    
    def select_for_evict(self):
        cached_access_ratio = list(map(lambda entityid,stat: 
            (entityid, stat[0].get_queue_size()/(stat[0].get_head()-stat[0].get_last()).total_seconds()), 
            self.__cache.get_statistics_all().items()))
        sorted_c_ar = [entity for entity, 
            access in sorted(cached_access_ratio, key=lambda tup: tup[1]) if access < self.__threshold]

        eviction_list = []
        if(len(sorted_c_ar)>0):
            for ent in sorted_c_ar:
                att_access_ratio = list(map(lambda stat: 
                    (stat[0], stat[1][0].get_queue_size()/(stat[1][0].get_head()-stat[1][0].get_last()).total_seconds()), 
                    self.__cache.get_statistics_entity(ent).items()))
                att_c_ar = [att for att, access in sorted(att_access_ratio, key=lambda tup: tup[1]) if access < self.__threshold]

                if(len(att_c_ar) > 0):
                    eviction_list = eviction_list + [(ent, att) for att in att_c_ar]
        
        return eviction_list