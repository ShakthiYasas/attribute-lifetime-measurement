from queue import Queue

class FIFOQueue():
    def __init__(self,size):
        self.__queue = Queue(maxsize = size)
    
    def push(self, val)->None:
        if(self.__queue.full()):
            self.__queue.get()
        self.__queue.put_nowait(val)
