import sys, os
import traceback
sys.path.append(os.path.abspath(os.path.join('..')))

import json
import grpc
import grpcservice.services_pb2 as pb2
import grpcservice.services_pb2_grpc as pb2_grpc

class GRPCClient:
    caller_strategy = None
    def __init__(self):
        self.__channel = grpc.insecure_channel("localhost:8040")

    def run(self, func, args=None):
        stub = pb2_grpc.CacheServiceStub(self.__channel) 
        try:
            if(func == 'save'):
                items = json.dumps(args[1], default=str)
                stub.save(pb2.CacheItem(entityid = args[0], cacheitems = items))
            if(func == 'get_statistics_all'):
                response = stub.get_statistics_all(pb2.Empty())
                json_res = json.loads(response.string)
                return {int(k): v for k, v in json_res.items()}
            if(func == 'are_all_atts_cached'):
                response = stub.are_all_atts_cached(pb2.EntityAttributeList(entityId = args[0], attributes = pb2.ListOfString(string = args[1])))
                return (response.isCached, set(response.attributeList.string))
            if(func == 'get_value_by_key'):
                response = stub.get_value_by_key(pb2.EntityAttributePair(entityId = args[0], attribute = args[1]))
                return [(res.prodid, res.response, res.cachedTime, res.recencybit) for res in response.values]
            if(func == 'get_values_for_entity'):
                response = stub.get_values_for_entity(pb2.EntityAttributeList(entityId = args[0], attributes = pb2.ListOfString(string = args[1])))
                return json.loads(response.string)
            if(func == 'addcachedlifetime'):  
                stub.addcachedlifetime(pb2.CachedLife(entityId = args[0][0], attribute = args[0][1], cacheLife = args[1].strftime("%Y-%m-%d %H:%M:%S")))
            if(func == 'get_statistics'):
                response = stub.get_statistics(pb2.EntityAttributePair(entityId = args[0], attribute = args[1]))
                if(response.isAvailable == False):
                    return None
                return (response.datelist, response.cachedTime)
            if(func == 'get_attributes_of_entity'):
                return stub.get_attributes_of_entity(pb2.CacheResponse(count = args[0]))
            if(func == 'get_statistics_entity'):
                response = stub.get_statistics_entity(pb2.CacheResponse(count = args[0]))
                return json.loads(response.string)
            if(func == 'get_last_hitrate'):
                res = stub.get_last_hitrate(pb2.CacheResponse(count = args[0]))
                return [(x.hitrate, x.count) for x in res.hitrate]
            if(func == 'get_hitrate_trend'):
                return stub.get_hitrate_trend(pb2.Empty())
        except Exception:
            print('An error occured : ' + traceback.format_exc())

    def __close_connection(self):
        self.__channel.unsubscribe(self.__close)
        exit()

    def __close(self):
        self.__channel.close()