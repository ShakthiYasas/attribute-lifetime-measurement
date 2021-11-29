import statistics
from cache.eviction.evictor import Evictor
from datetime import datetime, timedelta

class NovelEvictor(Evictor):
    def __init__(self, cache, threshold = 1.0):
        self.__cache = cache
        self.__threshold = threshold

    def select_for_evict(self):
        eviction_list = []
        expired = self.__cache.get_expired_lifetimes()
        if(len(expired)>0):
            for entity, att in expired:
                entity_stats = self.__cache.get_statistics_entity(entity)
                try:
                    if(entity_stats != None):
                        if(att in entity_stats):
                            att_stats = entity_stats[att]
                            access_rate = 0
                            time_diff = (att_stats[0].get_last()-att_stats[0].get_head()).total_seconds()
                            if(time_diff > 0):
                                access_rate = att_stats[0].get_queue_size()/time_diff

                            if(access_rate < self.__threshold):
                                # The item is least frequently used
                                eviction_list.append((entity,att))
                            else:
                                # The item has been used frequently
                                # Reevaluate the items
                                action, (est_c_lifetime, est_delay) = self.__cache.reevaluate_for_eviction(entity, att)
                                if(action == (0,0)):
                                    eviction_list.append((entity,att))
                                else:
                                    wait_time = datetime.now()+timedelta(seconds=est_c_lifetime)
                                    self.__cache.updatecachedlifetime(action[0], action[1], wait_time)
                        else:
                            self.__cache.removecachedlifetime(entity, att)
                    else:
                        self.__cache.removeentitycachedlifetime(entity)
                except Exception:
                    print('An error occured when selecting for eviction in Novel Evictor!')

        return eviction_list
    
    def select_entity_to_evict(self, internal=False, is_limited=False):
        # Value = Delay + Popularity + Remaining Cache Life
        # The item with the least delay, least popularity and the least remaining time cache will be evicetd.
        entities = self.__cache.get_statistics_all().items()

        # rel_popularity = "number of accesses" within the time that the hit rate trend was recorded
        rel_popularities = []
        for stat in entities:
            total_time = (stat[1][0].get_last()-stat[1][0].get_head()).total_seconds()
            if(total_time > 0):
                rel_popularities.append((stat[0], stat[1][0].get_queue_size()/total_time))
            else:
                rel_popularities.append((stat[0], 0))

        most_popular = sorted(rel_popularities, key=lambda item: item[1])[-1][1]
        rel_popularities = dict([(entity, access/most_popular) for entity, access in rel_popularities])

        calculated_value = {}
        ent_delays = {}
        providers_dict = {}
        for entityid, stats in entities:
            # Remaining Cached Life
            cached_lt_res = self.__cache.getdb().read_all_with_limit('entity-cached-lifetime',{
                        'entity': entityid,
                    },10)
            remaining_life = 0
            if(cached_lt_res):
                avg_lts = statistics.mean(list(map(lambda x: x['c_lifetime'], cached_lt_res)))
                rem_l = avg_lts - (datetime.now()-stats[1]).total_seconds()
                # Remaining cached life as a ratio of total cached lifetime
                # This could be a negative value as well.
                remaining_life = rem_l/avg_lts 
            else:
                remaining_life = 0
                try:
                    exp_time, cached_time = self.__cache.get_longest_cache_lifetime_for_entity(entityid)
                    rem_lf = (exp_time - datetime.now()).total_seconds()
                    cache_lf = (exp_time - cached_time).total_seconds()
                    remaining_life = rem_lf/cache_lf
                except Exception:
                    print("Error occured in fecthing longest lifetime for entity: " + str(entityid))

            # Delay
            providers = self.__cache.get_providers_for_entity(entityid)
            delay_list = []
            for prodid in providers:
                if(prodid in providers_dict):
                    continue
                else:
                    cached_res = self.__cache.getdb().read_all_with_limit('responsetimes',{
                            'context_producer': prodid
                        }, 10)
                        
                    mean_rt = 9999
                    if(cached_res):
                        mean_rt = statistics.mean(list(map(lambda x: x['avg_response_time'], cached_res)))

                    delay_list.append(mean_rt)
                    providers_dict[prodid] = mean_rt
            ent_delays[entityid] = statistics.mean(delay_list)

            calculated_value[entityid] = rel_popularities[entityid] + remaining_life

        most_delaying = sorted(ent_delays.items(), key=lambda item: item[1])[-1]
        for entityid, value in calculated_value.items():
            calculated_value[entityid] = value + (ent_delays[entityid]/most_delaying[1])

        if(internal):
            mandatory = []
            selective = []
            for entity, value in sorted(calculated_value.items(), key=lambda item: item[1]):
                if(value >= self.__threshold):
                    continue
                elif(value <= 0.01):
                    mandatory.append(entity)
                else:
                    selective.append(entity)
            
            return mandatory, selective
        else:
            sorted_entities = [ent_val_pair for ent_val_pair in sorted(calculated_value.items(), key=lambda item: item[1])]
            if(is_limited):
                return [sorted_entities[0][0]]
            
            return [entity for entity, value in sorted_entities if value < self.__threshold]