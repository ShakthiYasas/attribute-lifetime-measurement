import random

from agent import ExplorationAgent

# Takes __observed (in adaptive) as input 
class RandomAgent(ExplorationAgent):
    # Random agent looks at the observed (un-cached) context list 
    # and decide to cache one of the items randomly.

    @staticmethod
    def __choose_action(obs_list_len):
        return random.randint(0, obs_list_len - 1)

    def choose_action(self, observation:dict):
        rand_ent_idx = RandomAgent.__choose_action(len(observation))
        rand_ent = list(observation.items())[rand_ent_idx]

        rand_att_idx = RandomAgent.__choose_action(len(rand_ent['attributes']))
        rand_att = list(rand_ent['attributes'].items())[rand_att_idx]

        # Returns (entityid, attribute)
        return (rand_ent[0], rand_att[0])

# Takes __observed (in adaptive) as input 
class MRUAgent(ExplorationAgent):
    # Most Recently Used agent looks at the observed (un-cached) context list 
    # and decide to cache one of the items that is the 
    # MOST RECENTLY ACCESSED among the observed.

    @staticmethod
    def __choose_action(observation:dict):
        ent_obs = sorted(list(map(lambda x,y: (x,y['req_ts'][-1]), 
                        observation.items())), key=lambda tup: tup[1])
        att_obs = sorted(list(map(lambda x,y: (x,y[-1]), 
                        observation[ent_obs[-1][0]]['attributes'].items())), key=lambda tup: tup[1])
        return (ent_obs[-1][0], att_obs[-1][0])

    def choose_action(self, observation):
        # Returns (entityid, attribute)
        return MRUAgent.__choose_action(observation)

# Takes __attribute_access_trend (in adaptive) as input 
class MFUAgent(ExplorationAgent):
    # Most Frequently Used agent looks at the observed (un-cached) context list 
    # and decide to cache one of the items that is the 
    # MOST FREQUENTLY ACCESSED among the observed.
    # This overrides the evalution of other parameters (i.e. cost) and purely caches on populrity only

    @staticmethod
    def __choose_action(observation:dict):
        glo= []
        for entity,attrs in observation.items():
            local = sorted(list(map(lambda x,y: (x,y[-1]), attrs)), key=lambda tup: tup[1])
            glo.append((entity,local[-1][0],local[-1][1]))

        sorted_glo = sorted(glo, key=lambda tup: tup[2])
        return (sorted_glo[-1][0], sorted_glo[-1][1])

    def choose_action(self, observation:dict):
        # Returns (entityid, attribute)
        return MFUAgent.__choose_action(observation)
         