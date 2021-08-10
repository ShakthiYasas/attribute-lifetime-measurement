from math import dist, trunc
import numpy as np
from scipy.stats import skewnorm
from matplotlib.pyplot import plot

from error import ValueConfigurationError
from configuration import CarParkConfiguration

class Distribution(object):
    """"Base class of all distributions"""
    pass

class NormalDistribution(Distribution):
    def __init__(self, configuration, index):
        self.time_step = []
        self.occupancy = []
        self.sample_size = 0
        self.variation = 0

        self.seletced_periods = configuration.selected_periods

        if(isinstance(configuration,CarParkConfiguration)):
            print('Starting to create a skewed normal distribution.')
            linspace = np.linspace(start=0, stop=configuration.sample_size, num=configuration.no_of_refreshes)
            init_dist = skewnorm.pdf(linspace, a=configuration.skew[index], scale=configuration.standard_deviation[index])
        
            distributions = normalizing_distribution(init_dist, configuration.sample_size)
            self.time_step = distributions[0]
            self.occupancy = distributions[1]

            if(len(configuration.selected_periods) > 1 or (not(configuration.selected_periods[0][0] == 1 and configuration.selected_periods[0][1] == configuration.no_of_refreshes))):
                res = trim_distribution(configuration.selected_periods,self.occupancy)
                self.time_step = res[0]
                self.occupancy = res[1]
            
            print('Random skewed distribution created.')
        else:
            raise ValueConfigurationError()

    def get_occupancy_level(self, step):
        closest_step = min(self.time_step, key=lambda x:abs(x-step))
        idx = self.time_step.index(closest_step)

        if(closest_step>step):
            if(self.variation>0):
                return self.occupancy[idx-1]
            else:
                return self.sample_size - self.occupancy[idx-1]
        else:
            if(self.variation>0):
                return self.occupancy[idx]
            else:
                return self.sample_size - self.occupancy[idx]

class RandomDistribution(Distribution):
    def __init__(self, configuration, index):

        print('Starting to create a random normal distribution.')

        self.time_step = []
        self.occupancy = []

        number_of_data = (configuration.sample_size)**2
        init_dist = np.random.normal(0, configuration.standard_deviation[index], number_of_data)
        sorted_array = np.sort(init_dist)

        min_value = sorted_array[0]
        max_value = sorted_array[-1]
        diff = max_value - min_value
        step_size = diff/configuration.no_of_refreshes

        current_frequency = 0
        current_index = 0
        try:
            for snap in range(1,configuration.no_of_refreshes+1):
                current_upbound = min_value + (snap*step_size)
                for idx in range(current_index,number_of_data):
                    if(sorted_array[idx]<=current_upbound):
                        current_frequency += 1
                    else:
                        self.time_step.append(snap)
                        self.occupancy.append(current_frequency)
                        current_frequency = 0
                        current_index = idx
                        break
            
            if(len(configuration.selected_periods) > 1 or (not(configuration.selected_periods[0][0] == 1 and configuration.selected_periods[0][1] == configuration.no_of_refreshes))):
                res = trim_distribution(configuration.selected_periods, self.occupancy)
                self.time_step = res[0]
                self.occupancy = res[1]

        except(Exception):
            print('End of distribution')

        self.occupancy = list(map(lambda x: configuration.sample_size if x > configuration.sample_size else x, self.occupancy))
        plot(self.time_step, self.occupancy)

    def get_occupancy_level(self, step):
        closest_step = min(self.time_step, key=lambda x:abs(x-step))
        idx = self.time_step.index(closest_step)
        if(closest_step>step):
            return self.occupancy[idx-1]
        else:
            return self.occupancy[idx]

class SuperImposedDistribution(Distribution):
        def __init__(self, configuration, index):  
            self.time_step = []
            self.occupancy = []

            random = RandomDistribution(configuration, index)
            norm = NormalDistribution(configuration, index)

            res = super_impose(random, norm)
            self.time_step = res[0]
            self.occupancy = res[1]

#Static Method
def super_impose(dist1, dist2):
    dists = []
    dists.extend(list(map(lambda x: (x, dist1.occupancy[dist1.time_step.index(x)]), dist1.time_step)))
    dists.extend(list(map(lambda x: (x, dist2.occupancy[dist2.time_step.index(x)]), dist2.time_step)))

    dists = sorted(dists, key=lambda x: x[0])

    d = {}
    for k,v in dists:
        if(k in d):
            d[k] = (d[k][0]+1, ((d[k][0]*d[k][1])+v)/(d[k][0]+1))
        else:
            d[k] = (1,v)

    return list(map(lambda x : x[0], d.items())), list(map(lambda x : round(x[1][1]), d.items()))

# Static Method
def normalizing_distribution(init_dist, sample_size):
    constant = sample_size/sum(init_dist)
    dist = map(lambda x: x * constant, init_dist)

    time_step = 1
    cum_occupancy = 0
    last_occupancy = 0

    time_step_list = [0]
    occupancy_list = [last_occupancy]

    for data in list(dist):
        cum_occupancy = cum_occupancy + data
        if(abs(cum_occupancy - last_occupancy) >= 1):
            last_occupancy = trunc(cum_occupancy)
            time_step_list.append(time_step)
            occupancy_list.append(last_occupancy)
        time_step = time_step + 1

    return (time_step_list, occupancy_list)

def trim_distribution(selections, dist):
    occupancy = []
    for sel in selections:
        occupancy.extend(dist[sel[0]:sel[1]+1])
    time_step = range(1,len(occupancy)+1)
    return (occupancy,time_step)
