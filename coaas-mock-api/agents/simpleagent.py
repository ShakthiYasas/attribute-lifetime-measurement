import numpy as np

from agents.agent import Agent
from agents.exploreagent import RandomAgent, MRUAgent, MFUAgent

class SimpleAgent(Agent):
    def __init__(self, config, caller):
        # General Configurations
        self.__caller = caller
        self.__window = config.window
        self.__gamma = config.discount_rate
        
        # Extrapolation ranges 
        self.__short = config.short
        self.__mid = config.mid
        self.__long = config.long

        # e-Greedy Exploration
        self.__epsilons = config.e_greedy_init
        if (config.e_greedy_init is None) or (config.e_greedy_decrement is None):
            self.__epsilons = self.epsilons_min

        self.discount_max = config.e_greedy_max
        self.discount_increment = config.e_greedy_increment
        self.discount_decrement = config.e_greedy_decrement

        # Selecting the algorithem to execute during the exploration phase
        self.__explore_mentor = None
        if config.explore_mentor.lower() == 'mru':
            self.__explore_mentor = MRUAgent()
        elif config.explore_mentor.lower() == 'mfu':
            self.__explore_mentor = MFUAgent()
        else:
            self.__explore_mentor = RandomAgent()

    # Decide whether to cache or not cache
    def choose_action(self, observation, skipRandom=False): 
        entityid = observation['entityid']
        attribute = observation['attribute']
        observation = observation['features']

        # Cost of caching this item
        # Currently set to 0 becasue in memory caches don't cost now. 
        # However, when costing for cloud caches, this need to be retrived.
        cost_of_caching = 0

        # When the cached lifetime is not available, currently the mid is used as default. 
        # But this should be imporved to use the collaboraive filter to find the most similar.
        t_for_discounting = self.__mid

        if(observation[12]>0):
            cached_life_units = (observation[12]*1000)/self.__window
            t_for_discounting = self.__closest_point(cached_life_units)

        cur_rr_size = self.__caller.rr_trend_size
        cur_sla = self.__caller.get_most_expensive_sla()
        cur_rr_exp = self.__caller.req_rate_extrapolation  

        disearning_sequence = []
        if(t_for_discounting >= self.__short):
            disearning_sequence += self.caclulcate_for_range('short', observation, cur_sla, cur_rr_exp, cur_rr_size)
        if(t_for_discounting >= self.__mid):
            disearning_sequence += self.caclulcate_for_range('mid', observation, cur_sla, cur_rr_exp, cur_rr_size)
        if(t_for_discounting == self.__long):
            disearning_sequence += self.caclulcate_for_range('long', observation, cur_sla, cur_rr_exp, cur_rr_size)

        npv = sum(disearning_sequence) - cost_of_caching

        if(npv<=0):
            random_value = np.random.uniform()
            if((random_value < self.__epsilons) and not skipRandom):
                if(isinstance(self.__explore_mentor,MFUAgent)):
                    # Should the cached lifetime of these random items be calculated
                    return (self.__explore_mentor.choose_action(self.__caller.get_attribute_access_trend()),((self.__mid*self.__window)/1000,0))
                else:
                    return (self.__explore_mentor.choose_action(self.__caller.get_observed()),((self.__mid*self.__window)/1000,0))        
            
            caching_delay = -1
            # Item is not cached, but could be cached later
            if(disearning_sequence[0] < disearning_sequence[-1]):
                for i in range(len(disearning_sequence)):
                    if(i == 0):
                        continue
                    elif(disearning_sequence[i]>=0):
                        caching_delay = i-1

            return ((0,0),(0,caching_delay))
        else:
            # Here the action is a (entityid,attribute) to cache
            estimated_lifetime = 0
            if(disearning_sequence[0] > disearning_sequence[-1]):
                # The item could be cached
                # So, calculate the estimated cached lifetime
                es_delta = self.cached_life_when_delta(disearning_sequence)
                es_zero = 0
                if(disearning_sequence[-1]<0):
                    es_zero = self.cached_life_when_zero(disearning_sequence)
                estimated_lifetime = max(es_delta, es_zero)
                
            elif(disearning_sequence[-1] > 0):
                estimated_lifetime = (t_for_discounting*self.__window)/1000

            return ((entityid, attribute), (estimated_lifetime, 0))

    def cached_life_when_zero(self, disearning_sequence):
        for i in range(len(disearning_sequence)):
            if(i == 0):
                continue
            elif(disearning_sequence[i]<=0):
                return ((i-1)*self.__window)/1000


    def cached_life_when_delta(self, disearning_sequence):
        est_life = 0
        for i in range(len(disearning_sequence)):
            if(i == 0): continue
            else:
                delta = disearning_sequence[i-1] - disearning_sequence[i]
                if(delta < 0.01): est_life = ((i+1)*self.__window)/1000
        if(est_life == 0): return (len(disearning_sequence)*self.__window)/1000
        else: return est_life

    def caclulcate_for_range(self, rang, observation, cur_sla, cur_rr_exp, cur_rr_size):
        disearning_list = []

        if(rang == 'short'):
            count = int(self.__short)
            step = observation[7]/count

            curr_hr = step
            for i in range(1,count+1):
                expected_access = (cur_rr_exp[cur_rr_size-1+i]*observation[1]*self.__window)/1000
                earning = curr_hr*cur_sla[1]
                del_pen = (1-curr_hr)*cur_sla[2]
                ret_cost = (1-curr_hr)*observation[14]
            
                total_earning = expected_access*(earning - del_pen - ret_cost)
                disearning_list.append(total_earning/((1+self.__gamma)**i))
                curr_hr += step

        elif(rang == 'mid'):
            count = self.__mid - self.__short
            step = (observation[9]-observation[7])/count

            curr_hr = observation[7] + step
            for i in range(self.__mid+1,count+1):
                expected_access = (cur_rr_exp[cur_rr_size-1+self.__short+i]*observation[3]*self.__window)/1000
                earning = curr_hr*cur_sla[1]
                del_pen = (1-curr_hr)*cur_sla[2]
                ret_cost = (1-curr_hr)*observation[14]
                
                total_earning = expected_access*(earning - del_pen - ret_cost)
                disearning_list.append(total_earning/((1+self.__gamma)**i))
                curr_hr += step

        else:
            count = self.__long - self.__mid
            step = (observation[9]-observation[11])/count
            
            curr_hr = observation[9] + step
            for i in range(self.__mid+1,count+1):
                expected_access = (cur_rr_exp[cur_rr_size-1+self.__mid+i]*observation[3]*self.__window)/1000
                earning = curr_hr*cur_sla[1]
                del_pen = (1-curr_hr)*cur_sla[2]
                ret_cost = (1-curr_hr)*observation[14]
                
                total_earning = expected_access*(earning - del_pen - ret_cost)
                disearning_list.append(total_earning/((1+self.__gamma)**i))
                curr_hr += step

        return disearning_list

    # Select the closet range boundary for the current state 
    def __closest_point(self, lifeunits):
        s_dis = abs(self.__short - lifeunits)
        m_dis = abs(self.__mid - lifeunits)
        l_dis = abs(self.__long - lifeunits)

        closest = self.__short
        if(m_dis<s_dis or m_dis==s_dis):
            closest = self.__mid
            if(l_dis<m_dis or l_dis==m_dis):
                closest = self.__long

        return closest
    
    # Modify Discount Rate
    def modify_dicount_rate(self, increment=True):
        if(increment):
            self.__gamma = self.discount_max if self.__gamma + self.discount_increment > self.discount_max else self.__gamma + self.discount_increment
        else:
            self.__gamma = self.epsilons_min if self.__gamma - self.discount_decrement < self.epsilons_min else self.__gamma - self.discount_increment

    # Get current discunt rate
    def get_discount_rate(self):
        return self.__gamma