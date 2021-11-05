from queue import Queue

class FIFOQueue():
    def __init__(self,size):
        self.__queue = Queue(maxsize = size)
    
    def push(self, val)->None:
        if(self.__queue.full()):
            self.__queue.get()
        self.__queue.put_nowait(val)

    def getlist(self):
        return list(self.__queue)
    
    def getsize(self):
        return self.__queue.qsize()


class FIFOQueue_2():
    def __init__(self,size):
        self.__size = size
        self.__queue = list()

    def push(self, val)->None:
        if(len(self.__queue)==self.__size):
            self.__queue.pop(0)
        self.__queue.append(val)

    def get_last(self):
        return self.__queue[-1]
    
    def get_head(self):
        return self.__queue[0]

    def get_last_range(self, length):
        return self.__queue[len(self.__queue)-length : len(self.__queue)]
    
    def get_last_position(self, length):
        idx = len(self.__queue)-length
        return self.__queue[0 if idx < 0 else idx]
    
    def remove(self,val):
        self.__queue.remove(val)

    def getlist(self):
        return self.__queue

    def get_queue_size(self):
        return len(self.__queue)