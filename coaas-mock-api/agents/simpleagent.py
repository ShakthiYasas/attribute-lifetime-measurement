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
            self.__epsilons = list(self.epsilons_min)

        # Selecting the algorithem to execute during the exploration phase
        self.__explore_mentor = None
        if config.explore_mentor.lower() == 'mru':
            self.__explore_mentor = MRUAgent()
        elif config.explore_mentor.lower() == 'mfu':
            self.__explore_mentor = MFUAgent()
        else:
            self.__explore_mentor = RandomAgent()

    # Decide whether to cache or not cache
    def choose_action(self, observation): 
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

        npv = 0
        sequence = [0,0,0]
        if(t_for_discounting >= self.__short):
            dv = self.caclulcate_for_range('short', observation, cur_sla, cur_rr_exp, cur_rr_size)
            sequence[0] = 1 if dv > 0 else 0
            npv += dv
        if(t_for_discounting >= self.__mid):
            dv = self.caclulcate_for_range('mid', observation, cur_sla, cur_rr_exp, cur_rr_size)
            sequence[1] = 1 if dv > 0 else 0
            npv += dv
        if(t_for_discounting == self.__long):
            dv = self.caclulcate_for_range('long', observation, cur_sla, cur_rr_exp, cur_rr_size)
            sequence[2] = 1 if dv > 0 else 0
            npv += dv

        npv -= cost_of_caching
        sequence = tuple(sequence)

        estimated_lifetime = 0
        if(npv>0): 
            if(sequence == (1,0,0)):
                estimated_lifetime = (self.__short*self.__window)/1000
            elif(sequence == (1,1,0)):
                estimated_lifetime = (self.__mid*self.__window)/1000
            elif(any(seq == sequence for seq in [(0,1,1),(1,0,1),(1,1,1)])):
                estimated_lifetime = max((self.__long*self.__window)/1000, 
                        observation[12] if observation[12] > 0 else (self.__mid*self.__window)/1000)
            else:
                estimated_lifetime = observation[12] if observation[12] > 0 else (self.__mid*self.__window)/1000

        if(npv<=0):
            random_value = np.random.uniform()
            if(random_value < self.__epsilons):
                if(isinstance(self.__explore_mentor,MFUAgent)):
                    # Should the cached lifetime of these random items be calculated
                    return (self.__explore_mentor.choose_action(self.__caller.get_attribute_access_trend()),(self.__mid*self.__window)/1000)
                else:
                    return (self.__explore_mentor.choose_action(self.__caller.get_observed()),(self.__mid*self.__window)/1000)        
            return ((0,0),0)
        else:
            # Here the action is a (entityid,attribute) to cache
            return ((entityid, attribute), estimated_lifetime)

    def caclulcate_for_range(self, rang, observation, cur_sla, cur_rr_exp, cur_rr_size):
        if(rang == 'short'):
            count = int(self.__short)

            expected_access = (cur_rr_exp[cur_rr_size-1+self.__short]*observation[1]*self.__window)/1000
            earning = observation[7]*cur_sla[1]
            del_pen = (1-observation[7])*cur_sla[2]
            ret_cost = (1-observation[7])*observation[14]
            
            total_earning = expected_access*(earning - del_pen - ret_cost)
            total_dis_earning = 0
            for i in range(1,count+1):
                total_dis_earning += total_earning/((1+self.__gamma)**i)

            return total_dis_earning

        elif(rang == 'mid'):
            count = self.__mid - self.__short

            expected_access = (cur_rr_exp[cur_rr_size-1+self.__mid]*observation[3]*self.__window)/1000
            earning = observation[9]*cur_sla[1]
            del_pen = (1-observation[9])*cur_sla[2]
            ret_cost = (1-observation[9])*observation[14]
            
            total_earning = expected_access*(earning - del_pen - ret_cost)
            total_dis_earning = 0
            for i in range(self.__mid+1,count+1):
                total_dis_earning += total_earning/((1+self.__gamma)^i)

            return total_dis_earning

        else:
            count = self.__long - self.__mid

            expected_access = (cur_rr_exp[cur_rr_size-1+self.__long]*observation[3]*self.__window)/1000
            earning = observation[11]*cur_sla[1]
            del_pen = observation[11]*cur_sla[2]
            ret_cost = (1-observation[11])*observation[14]
            
            total_earning = expected_access*(earning - del_pen - ret_cost)
            total_dis_earning = 0
            for i in range(self.__mid+1,count+1):
                total_dis_earning += total_earning/((1+self.__gamma)^i)

            return total_dis_earning

    def __closest_point(self, lifeunits):
        s_dis = abs(self.__short-lifeunits)
        m_dis = abs(self.__mid-lifeunits)
        l_dis = abs(self.__long-lifeunits)

        closest = self.__short
        if(m_dis<s_dis or m_dis==s_dis):
            closest = self.__mid
            if(l_dis<m_dis or l_dis==m_dis):
                closest = self.__long

        return closest