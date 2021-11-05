from datetime import datetime
import statistics

from cache.eviction.evictor import Evictor

class LVFEvictor(Evictor):
    def __init__(self, parent_cache):
        self.__cache = parent_cache
    
    def select_for_evict(self):
        # Value = Delay + Popularity + Remaining Cache Life
        # The item with the least delay, least popularity and the least remaining time cache will be evicetd.
        entities = self.__cache.get_statistics_all().items()
        y = self.__cache.get_hitrate_trend().getlist()
        
        totalreqs = sum(j for i, j in y)  

        rel_popularities = list(map(lambda entityid,stat: 
            (entityid, (stat[0].get_queue_size()*((len(y)*(self.__cache.window/1000))/(stat[0].get_head()-stat[0].get_last()).total_seconds()))/totalreqs), 
            entities))

        calculated_value = {}
        ent_delays = {}
        providers_dict = {}
        for entityid, stats in entities:
            # Poularity
            cached_lt_res = self.__db.read_all_with_limit('entity-cached-lifetime',{
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

            # Delay
            providers = self.__cache.get_providers_for_entity(entityid)
            delay_list = []
            for prodid in providers:
                if(prodid in providers_dict):
                    continue
                else:
                    cached_res = self.__cache.getdb.read_all_with_limit('responsetimes',{
                            'context_producer': prodid
                        }, 10)
                    mean_rt = statistics.mean(list(map(lambda x: x['avg_response_time'], cached_res)))
                    delay_list.append(mean_rt)
                    providers_dict[prodid] = mean_rt
            ent_delays[entityid] = statistics.mean(delay_list)

            calculated_value[entityid] = rel_popularities[entityid] + remaining_life

        most_delaying = sorted(ent_delays.items(), key=lambda item: item[1])[-1]
        for entityid, value in calculated_value.items():
            calculated_value[entityid] = value + (ent_delays[entityid]/most_delaying[1])

        return sorted(calculated_value.items(), key=lambda item: item[1])[0][0]

        