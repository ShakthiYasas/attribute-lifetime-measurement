from math import trunc
import numpy as np
from scipy.stats import skewnorm
from matplotlib.pyplot import plot

from error import ValueConfigurationError
from configuration import CarParkConfiguration

class Distribution:
    """"Base class of all distributions"""
    pass

class NormalDistribution(Distribution):
    time_step = []
    occupancy = []
    sample_size = 0
    variation = 0

    def __init__(self, configuration):
        if(isinstance(configuration,CarParkConfiguration)):
            print('Starting to create a skewed normal distribution.')
            linspace = np.linspace(start=0, stop=configuration.sample_size, num=configuration.no_of_refreshes)
            init_dist = skewnorm.pdf(linspace, a=configuration.skew, scale=configuration.standard_deviation)

            distributions = normalizing_distribution(init_dist, configuration.sample_size)
            self.time_step = distributions[0]
            self.occupancy = distributions[1]
            
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
    time_step = []
    occupancy = []

    def __init__(self, configuration):
        print('Starting to create a random normal distribution.')
        number_of_data = (configuration.sample_size)**2
        print(number_of_data)
        init_dist = np.random.normal(0, configuration.standard_deviation, number_of_data)
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

# Static Method
def normalizing_distribution(init_dist, sample_size):
    constant = sample_size/sum(init_dist)
    dist = map(lambda x: x * constant, init_dist)

    time_step = 1
    cum_occupancy = 0
    last_occupancy = 0

    time_step_list = [0]
    occupancy_list = [last_occupancy]

    for data in dist:
        cum_occupancy = cum_occupancy + data
        if(abs(cum_occupancy - last_occupancy) >= 1):
            last_occupancy = trunc(cum_occupancy)
            time_step_list.append(time_step)
            occupancy_list.append(last_occupancy)
            time_step = time_step + 1

    return (time_step_list, occupancy_list)