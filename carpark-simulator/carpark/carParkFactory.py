from carpark.carPark import CarPark
from configuration import CarParkConfiguration

# Produces a car park instance according to configuration
class CarParkFactory:
    # Class varaible
    carpark = None
    configuration = None

    def __init__(self, configuration = None):
        if(configuration != None):
            self.configuration = configuration

    # Retruns the singleton instance of the car park
    def get_carpark(self) -> CarPark:
        if(self.carpark == None):
            if(self.configuration != None and isinstance(self.configuration,CarParkConfiguration)):
                return CarPark(self.configuration)
            else:
                return CarPark()
        else:
            return self.carpark







        