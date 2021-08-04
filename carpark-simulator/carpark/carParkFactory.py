from carpark.carPark import CarPark
from configuration import CarParkConfiguration

class CarParkFactory:
    carpark = None
    configuration = None

    def __init__(self, configuration = None):
        if(configuration != None):
            self.configuration = configuration

    def get_carpark(self) -> CarPark:
        if(self.carpark == None):
            if(self.configuration != None and isinstance(self.configuration,CarParkConfiguration)):
                return CarPark(self.configuration)
            else:
                return CarPark()
        else:
            return self.carpark







        