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
                att_stats = entity_stats[att]
                access_rate = att_stats.get_queue_size()/(att_stats.get_head()-att_stats.get_last()).total_seconds()
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
                        self.__cache.updatecachedlifetime(action[0], action[1], datetime.now()+timedelta(seconds=est_c_lifetime))

        return eviction_list