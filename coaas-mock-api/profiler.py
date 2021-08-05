import time
import datetime
import threading
from event import post_event

class Profiler:
    db = None
    mean = []
    lookup = {}
    interval = 1
    last_time = datetime.now()
    threadpool = []

    def __init__(self, attributes, db, window):
        index = 0
        self.db = db   
        self.window = window
        self.mean = [0] * len(attributes)
        self.most_recently_used = [[]] * len(attributes)
        
        for att in attributes:
            self.lookup[att] = index
            index+=1
        
        for att in attributes:
            th = MyThread(self.lookup[att], att, self)
            self.threadpool.append(th)
            th.start()

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True               
        thread.start() 

    def run(self):
        while True:
            self.clear_expired()
            time.sleep(self.interval)

    def clear_expired(self) -> None:
        curr_time = datetime.now()
        for row in self.most_recently_used:
            for stamp in row:
                if(len(stamp) != 0):
                    diff = ((curr_time - stamp[1]).microseconds)/1000
                    if(diff >= self.window):
                        row.remove(stamp)

    def reactive_push(self, response) -> None:
        curr_time = datetime.now()
        duration = ((self.last_time - curr_time).microseconds)/1000

        for key,value in response.items():
            idx = self.lookup[key]
            lst_vals = self.most_recently_used[idx]
            lst_vals.append((value, curr_time, duration))

            count = 0
            total_sum = 0
            local_sum = 0
            curr_val = None
            for item in lst_vals:
                if(curr_val == None):
                    curr_val = item[0]
                    local_sum = item[2]
                else:
                    if(item[0] != curr_val):
                        count += 1
                        total_sum += local_sum
                        curr_val = item[0]
                        local_sum = item[2]
                    else:
                        local_sum += item[2]
            
            mean = total_sum/count
            self.mean[idx] = mean    
        
        self.last_time = curr_time

    def get_details(self) -> dict:
        return {
            'mean_lifetimes': self.mean
            }


class MyThread (threading.Thread):
    def __init__(self, thread_id, name, caller):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.caller = caller
        self.name = name

    def run(self):
        self.refresh()

    def refresh(self):
        while True:
            post_event("need_to_refresh", self.name)
            means = getattr(self.caller,'mean')
            time.sleep(means[self.thread_id]/1000)
            
            

        