import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import time
from datetime import datetime
import threading
from lib.event import post_event

class Profiler:
    db = None
    mean = []
    lookup = {}
    interval = 1
    threadpool = []
    last_time = datetime.now()

    def __init__(self, attributes, db, window, caller_name, session = None):
        index = 0
        self.db = db   
        self.window = window
        self.session = session
        self.caller_name = caller_name
        self.mean = [0] * len(attributes)
        self.most_recently_used = [[]] * len(attributes)
        
        for att in attributes:
            self.lookup[att] = index
            index+=1
        
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True               
        thread.start() 

    def run(self):
        while True:
            self.clear_expired()
            time.sleep(self.interval)

    def auto_cache_refresh_for_greedy(self, attributes) -> None:
        for att in attributes:
            th = MyThread(self.lookup[att], att, self)
            self.threadpool.append(th)
            th.start()

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

        for key,value in response.items():
            idx = self.lookup[key]
            lst_vals = self.most_recently_used[idx]
            duration = 0 
            if not lst_vals:
                duration = ((self.last_time - curr_time).microseconds)/1000
            else:
                duration = ((lst_vals[-1][1] - curr_time).microseconds)/1000
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

            total_sum += local_sum
            count += 1
            mean = total_sum/count             

            self.mean[idx] = mean
            self.db.insert_one(key+'-lifetime',{'session': self.session, 'strategy': self.caller_name, 'lifetime:':mean, 'time': curr_time})    
        
        self.last_time = curr_time

    def get_details(self) -> dict:
        return {
            'mean_lifetimes': self.mean,
            'most_recent': self.most_recently_used
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
            