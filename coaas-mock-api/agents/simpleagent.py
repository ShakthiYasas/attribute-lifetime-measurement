from agent import Agent

class SimpleAgent(Agent):
    def __init__(self, config, caller):
        # General Configurations
        self.__caller = caller
        self.__window = config.window
        self.__gamma = config.discounting_factor
        
        # Extrapolation ranges 
        self.__short = config.short
        self.__mid = config.mid
        self.__long = config.long

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

        total_discounted_earning = 0
        if(t_for_discounting >= self.__short):
            total_discounted_earning += self.caclulcate_for_range('short', observation, cur_sla, cur_rr_exp, cur_rr_size)
        if(t_for_discounting >= self.__mid):
            total_discounted_earning += self.caclulcate_for_range('mid', observation, cur_sla, cur_rr_exp, cur_rr_size)
        if(t_for_discounting == self.__long):
            total_discounted_earning += self.caclulcate_for_range('long', observation, cur_sla, cur_rr_exp, cur_rr_size)

        total_discounted_earning -= cost_of_caching

        if(total_discounted_earning<=0):
            return (0,0)
        else:
            return (entityid, attribute)

    def caclulcate_for_range(self, range, observation, cur_sla, cur_rr_exp, cur_rr_size):
        if(range == 'short'):
            count = self.__short

            expected_access = cur_rr_exp[cur_rr_size-1+self.__short]*observation[1]
            earning = observation[7]*cur_sla[1]
            del_pen = observation[7]*cur_sla[2]
            ret_cost = (1-observation[7])*observation[14]
            
            total_earning = expected_access*(earning - del_pen - ret_cost)
            total_dis_earning = 0
            for i in range(1,count+1):
                total_dis_earning += total_earning/((1+self.__gamma)^i)

            return total_dis_earning

        elif(range == 'mid'):
            count = self.__mid - self.__short

            expected_access = cur_rr_exp[cur_rr_size-1+self.__mid]*observation[3]
            earning = observation[9]*cur_sla[1]
            del_pen = observation[9]*cur_sla[2]
            ret_cost = (1-observation[9])*observation[14]
            
            total_earning = expected_access*(earning - del_pen - ret_cost)
            total_dis_earning = 0
            for i in range(self.__mid+1,count+1):
                total_dis_earning += total_earning/((1+self.__gamma)^i)

            return total_dis_earning

        else:
            count = self.__long - self.__mid

            expected_access = cur_rr_exp[cur_rr_size-1+self.__long]*observation[3]
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