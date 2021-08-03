from math import trunc
import numpy as np
from scipy.stats import skewnorm

from error import ValueConfigurationError
from configuration import CarParkConfiguration

class NormalDistribution():
    time_step = []
    occupancy = []
    sample_size = 0

    def __init__(self, configuration):
        if(isinstance(configuration,CarParkConfiguration)):
            print('Starting to create a normal distribution.')
            linspace = np.linspace(start=0, stop=configuration.sample_size, num=configuration.no_of_refreshes)
            init_dist = skewnorm.pdf(linspace, a=configuration.skew, scale=configuration.standard_deviation)

            constant = configuration.sample_size/sum(init_dist)
            dist = map(lambda x: x * constant, init_dist)

            time_step = 1
            cum_occupancy = 0
            last_occupancy = 0
            self.time_step.append(0)
            self.occupancy.append(last_occupancy)
            self.sample_size = configuration.sample_size

            for data in dist:
                cum_occupancy = cum_occupancy + data
                if(abs(cum_occupancy - last_occupancy) >= 1):
                    last_occupancy = trunc(cum_occupancy)
                    self.time_step.append(time_step)
                    self.occupancy.append(last_occupancy)
                time_step = time_step + 1
            
            print('Random distribution created.')
        else:
            raise ValueConfigurationError()

    def get_occupancy_level(self, step):
        closest_step = min(self.time_step, key=lambda x:abs(x-step))
        idx = self.time_step.index(closest_step)

        if(closest_step>step):
            return self.sample_size - self.occupancy[idx-1]
        else:
            return self.sample_size - self.occupancy[idx]

