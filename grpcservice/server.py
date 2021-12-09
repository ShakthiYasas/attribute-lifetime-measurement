import time
import configparser
from pathlib import Path
from concurrent import futures
from cache.cachefactory import CacheFactory
from cache.configurations.cacheconfig import CacheConfiguration

import grpc
import grpcservice.services_pb2 as pb2
import grpcservice.services_pb2_grpc as pb2_grpc
from lib.mongoclient import MongoClient
from lib.sqlliteclient import SQLLiteClient

MAX_WORKERS = 20

# Global variables
config = configparser.ConfigParser()
config.read('config.ini')
default_config = config['DEFAULT']

# Connecting to DB instances
__service_registry = SQLLiteClient(default_config['SQLDBName'])
__db = MongoClient(default_config['ConnectionString'], default_config['DBName'])

class Listener(pb2_grpc.CacheServiceServicer):
    def __init__(self): 
        __cache_fac = CacheFactory(CacheConfiguration(default_config), __service_registry)
        self.__cache_memory = __cache_fac.get_cache_memory(__db)

    def save(self, request, context):
        self.__cache_memory.save(request.entityid, request.cacheitems)

    def addcachedlifetime(self, request, context):
        self.__cache_memory.addcachedlifetime(request.action, request.cacheLife)

    def get_hitrate_trend(self, request, context): 
        return self.__cache_memory.get_hitrate_trend()

    def updatecachedlifetime(self, request, context):
        self.__cache_memory.updatecachedlifetime(request.action, request.cacheLife)

    def is_cached(self, request, context):
        response = self.__cache_memory.is_cached(request.entityId, request.attribute) 
        return pb2.BoolType(status = True if response else False)

    def get_statistics_all(self, request, context): 
        return pb2.FrequencyTable(attributes = self.__cache_memory.get_statistics_all())

    def get_last_hitrate(self, request, context):
        return self.__cache_memory.get_last_hitrate(request.count)

    def get_statistics(self, request, context): 
        response = self.__cache_memory.get_statistics(request.entityId, request.attribute)
        return pb2.Statistic(datelist = response[0], cachedTime = response[1])
    
    def removeentitycachedlifetime(self, request, context): 
        self.__cache_memory.removeentitycachedlifetime(request.count)

    def removecachedlifetime(self, request, context): 
        self.__cache_memory.removecachedlifetime(request.entityId, request.attribute)

    def get_statistics_entity(self, request, context): 
        return pb2.FrequencyTable(attributes = self.__cache_memory.get_statistics_entity(request.count))

    def get_attributes_of_entity(self, request, context): 
        return self.__cache_memory.get_attributes_of_entity(request.count)

    def get_value_by_key(self, request, context): 
        values = []
        response = self.__cache_memory.get_value_by_key(request.entityId, request.attribute)
        for prodid, response_val, cachedtime, recencybit in response:
            values.append(pb2.CachedItem(
                prodid = prodid,
                response = response_val,
                cachedTime = cachedtime,
                recencybit = recencybit
            ))

        return values

    def get_values_for_entity(self, request, context): 
        response = self.__cache_memory.get_values_for_entity(request.entityId, request.attribute)
        return pb2.CachedRecords(attributes = response)

    def are_all_atts_cached(self, request, context): 
        response = self.__cache_memory.are_all_atts_cached(request.entityId, request.attributes)
        return pb2.CachedAttributes(isCached = response[0], attributeList = response[1])

# Starting Server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers = MAX_WORKERS))
    pb2_grpc.add_CacheServiceServicer_to_server(Listener(), server)
    server.add_insecure_port("[::]:8040")
    server.start()

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Stopping GRPC Server!")
        server.stop(0)

if __name__ == "__main__":
    serve()