import grpc
import grpcservice.services_pb2 as pb2
import grpcservice.services_pb2_grpc as pb2_grpc

class GRPCClient:
    caller_strategy = None

    def __int__(self):
        self.__channel = grpc.insecure_channel("localhost:8040")

    def run(self, func, args=None):
        with pb2_grpc.CacheServiceStub(self.__channel) as stub:
            if(func == 'save'):
                stub.save(pb2.CacheItem(entityid = args[0], cacheitems = args[1]))  # Remain
            if(func == 'get_statistics_all'):
                response = stub.get_statistics_all(pb2.Empty())
                return response.attributes
            if(func == 'are_all_atts_cached'):
                response = stub.are_all_atts_cached(pb2.EntityAttributeList(entityId = args[0], attributes = args[1]))
                return (response.isCached, response.attributeList)
            if(func == 'get_value_by_key'):
                return stub.get_value_by_key(pb2.EntityAttributePair(entityId = args[0], attribute = args[1]))
            if(func == 'get_values_for_entity'):
                response = stub.get_values_for_entity(pb2.EntityAttributeList(entityId = args[0], attributes = args[1]))
                return response.attributes
            if(func == 'addcachedlifetime'):
                entity_att_pair = pb2.EntityAttributePair(entityId = args[0][0], attribute = args[0][1])
                stub.addcachedlifetime(pb2.CachedLife(action = entity_att_pair, cacheLife = args[1]))
            if(func == 'get_statistics'):
                return stub.get_statistics(pb2.EntityAttributePair(entityId = args[0], attribute = args[1]))
            if(func == 'get_attributes_of_entity'):
                return stub.get_attributes_of_entity(pb2.CacheResponse(count = args[0]))
            if(func == 'get_statistics_entity'):
                response = stub.get_statistics_entity(pb2.CacheResponse(count = args[0]))
                return response.attributes
            if(func == 'get_last_hitrate'):
                return stub.get_last_hitrate(pb2.CacheResponse(count = args[0]))
            if(func == 'get_hitrate_trend'):
                return stub.get_hitrate_trend(pb2.Empty())

    def __close_connection(self):
        self.__channel.unsubscribe(self.__close)
        exit()

    def __close(self):
        self.__channel.close()