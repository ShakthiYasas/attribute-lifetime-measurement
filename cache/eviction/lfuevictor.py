from cache.eviction.evictor import Evictor

class LFUEvictor(Evictor):
    def __init__(self, cache, threshold = 1.0):
        self.__cache = cache
        self.__threshold = threshold
    
    # Select the entity, attributes suitable for eviction
    def select_for_evict(self):
        mandatory, sorted_c_ar = self.select_entity_to_evict(internal = True)
        eviction_list = []
        
        if(mandatory):
            for entityid in mandatory:
                eviction_list += [(entityid, att) for att in self.__cache.get_statistics_entity(entityid).keys()]
        
        if(len(sorted_c_ar)>0):
            for ent in sorted_c_ar:
                att_access_ratio = []
                for stat in self.__cache.get_statistics_entity(ent).items():
                    time_diff = (stat[1][0].get_last()-stat[1][0].get_head()).total_seconds()
                    if(time_diff > 0):
                        att_access_ratio.append((stat[0], stat[1][0].get_queue_size()/time_diff))
                    else:
                        att_access_ratio.append((stat[0],0))

                att_c_ar = [att for att, access in sorted(att_access_ratio, key=lambda tup: tup[1]) if access < self.__threshold]

                if(len(att_c_ar) > 0):
                    eviction_list += [(ent, att) for att in att_c_ar]
        
        return eviction_list

    # Select the entitites suitable for eviction
    def select_entity_to_evict(self, internal=False, is_limited=False):
        cached_access_ratio = []
        for stat in self.__cache.get_statistics_all().items():
            time_diff = (stat[1][0].get_last()-stat[1][0].get_head()).total_seconds()
            if(time_diff > 0):
                cached_access_ratio.append((stat[0], stat[1][0].get_queue_size()/time_diff))
            else:
                cached_access_ratio.append((stat[0],0))

        if(internal):
            mandatory = []
            selective = []
            for entity, access in sorted(cached_access_ratio, key=lambda tup: tup[1]):
                if(access >= self.__threshold):
                    continue
                elif(access < 0.01):
                    mandatory.append(entity)
                else:
                    selective.append(entity)
            
            return mandatory, selective
        else:
            sorted_entities = [entity_access_pair for entity_access_pair in sorted(cached_access_ratio, key=lambda tup: tup[1])]
            if(is_limited):
                return [sorted_entities[0][0]]

            selected = [entity for entity, access in sorted_entities if access < self.__threshold]    
            return selected