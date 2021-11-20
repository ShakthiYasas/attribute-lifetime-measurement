from datetime import datetime
import statistics

from cache.eviction.evictor import Evictor

class LVFEvictor(Evictor):
    def __init__(self, parent_cache, threshold = 1.0):
        self.__cache = parent_cache
        self.__threshold = threshold
    
    # Select the entity, attributes suitable for eviction
    def select_for_evict(self):
        # Value = Delay + Popularity + Remaining Cache Life
        # The item with the least delay, least popularity and the least remaining time cache will be evicetd.
        eviction_list = []
        mandatory, selective = self.select_entity_to_evict(internal=True)

        for entityid in mandatory:
            eviction_list += [(entityid, att) for att in self.parent_cache.__entityhash[entityid].keys()]

        now = datetime.now()
        cur_ret_latency = self.__cache.registry.get_ret_latency()

        for ent in selective:
            for att, stat in self.__cache.get_statistics_entity(ent).items():
                # Popularity
                att_access_ratio = (stat[0].get_queue_size()*(self.__cache.window/1000))/((stat[0].get_head()-stat[0].get_last()).total_seconds())
                # Remaing Cached Lifetime
                (remaining, cached) = self.__cache.get_cachedlifetime(ent, att)
                relative_remaning = (now - remaining)/(remaining - cached)
                # Delay 
                providers = self.__cache.get_providers_for_attribute(ent, att)
                avg_delays = []
                for prodid in providers.keys():
                    cached_res = self.__cache.getdb.read_all_with_limit('responsetimes',{
                                'context_producer': prodid
                            }, 10)
                    avg_delays.append(statistics.mean(list(map(lambda x: x['avg_response_time'], cached_res))))
                rel_ret_latency = 1
                avg_att_latency = statistics.mean(avg_delays)
                if(avg_att_latency <=  cur_ret_latency and cur_ret_latency != 0):
                    rel_ret_latency = avg_att_latency/cur_ret_latency
                else:
                    rel_ret_latency = (avg_att_latency-cur_ret_latency)/cur_ret_latency

                value = att_access_ratio + relative_remaning + rel_ret_latency
                if(value < self.__threshold):
                    eviction_list.append(ent, att)

        return eviction_list

    # Select the entitites suitable for eviction
    def select_entity_to_evict(self, internal=False):
        # Value = Delay + Popularity + Remaining Cache Life
        # The item with the least delay, least popularity and the least remaining time cache will be evicetd.
        entities = self.__cache.get_statistics_all().items()
        y = self.__cache.get_hitrate_trend().getlist()
        
        totalreqs = sum([j[1] for j in y])  

        # rel_popularity = "number of accesses" within the time that the hit rate trend was recorded
        rel_popularities = list(map(lambda stat: 
            (stat[0], (stat[1][0].get_queue_size()*((len(y)*(self.__cache.window/1000))/(stat[1][0].get_head()-stat[1][0].get_last()).total_seconds()))/totalreqs), 
            entities))

        most_popular = sorted(rel_popularities, key=lambda item: item[1])[-1][1]
        rel_popularities = dict([(entity, access/most_popular) for entity, access in rel_popularities])

        calculated_value = {}
        ent_delays = {}
        providers_dict = {}
        for entityid, stats in entities:
            # Remaining Cached Life
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
            return [entity for entity, value in sorted(calculated_value.items(), 
                        key=lambda item: item[1]) if value < self.__threshold]