from math import trunc
import numpy as np
from scipy.stats import skewnorm

from Error import ValueConfigurationError
from Configuration import CarParkConfiguration

class NormalDistribution():
    occupancy = []
    def __init__(self, configuration):
        if(isinstance(configuration,CarParkConfiguration)):
            print('Starting to create a normal distribution.')
            linspace = np.linspace(0, configuration.sample_size, configuration.no_of_refreshes)
            dist = skewnorm.pdf(linspace, configuration.skew, scale=configuration.standard_deviation)
            
            time_step = 1
            cum_occupancy = 0
            last_occupancy = 0
            self.occupancy.append((0, last_occupancy))

            for data in dist:
                cum_occupancy = cum_occupancy + data
                if(abs(cum_occupancy - last_occupancy) >= 1):
                    last_occupancy = trunc(cum_occupancy)
                    self.occupancy.append((time_step*configuration.sampling_rate, last_occupancy))
                time_step = time_step + 1
            
            print('Random distribution created.')
        else:
            raise ValueConfigurationError()

    def get_occupancy_level(self, step):
        idx = (np.abs(self.occupancy[0] - step)).argmin()
        if(self.occupancy[idx][1]>step):
            return self.occupancy[idx-1][1]
        else:
            return self.occupancy[idx][1]

