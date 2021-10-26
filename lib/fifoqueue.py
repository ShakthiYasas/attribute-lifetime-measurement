from queue import Queue

class FIFOQueue():
    def __init__(self,size):
        self.__queue = Queue(maxsize = size)
    
    def push(self, val)->None:
        if(self.__queue.full()):
            self.__queue.get()
        self.__queue.put_nowait(val)


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
    
    def remove(self,val):
        self.__queue.remove(val)