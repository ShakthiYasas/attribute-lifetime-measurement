import math
import Constants as const

class Configuration:
    """Base class for Configuartions"""
    pass

# Configuration of the Car Park
class CarParkConfiguration(Configuration):
    sample_size = 100 # total available (constrait).
    total_time = 600000 # In miliseconds. 10 minutes.
    skew = 0 # median skewness 
    standard_deviation = 2 # Using the emperical rule of 95% representation
    sampling_rate = 1000 # sampling every 1000ms = 1s
    no_of_refreshes = 0 # number of times the car park has refreshed data during the total time period

    def __init__(self, sample_size = None, standard_deviation = None, total_time = None, skew = None, sampling_rate = None):
        
        if(sample_size != None and sample_size > 0):
            self.sample_size = sample_size
        if(standard_deviation != None and standard_deviation > 0):
            self.standard_deviation = standard_deviation
        if(total_time != None and total_time > 60000):
            self.total_time = total_time
        if(skew != None):
            self.skew = skew
        if(sampling_rate != None and sampling_rate > 0 and sampling_rate <= self.total_time):
            self.sampling_rate = sampling_rate
        
        self.no_of_refreshes =  self.total_time/self.sampling_rate

        print('Configuration:\n\tSample Size = '+self.sample_size+'\n\tStandard Distruntion = '+self.standard_deviation
        +'\n\tTotal Time = '+int(math.ceil(self.total_time/60000))+const.mins
        +'\n\tSkew = '+self.skew
        +'\n\tSampling Interval = '+self.sampling_rate)

