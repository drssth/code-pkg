import sys
import os
import os.path
import pandas as pd
import datetime
import time
import queue
import logging



class Timer(object):
    def __init__(self, maxsize=100):
        self.start = datetime.datetime.now()
        self.q = queue.Queue(maxsize=maxsize)
        self.interval = 0
        self.current = self.start
        self.last = self.start
        self.q.put(self.start)
        
        
    def time_now(self):
        self.last = self.current
        self.current = datetime.datetime.now()
        self.interval = self.current - self.last
        self.q.put(self.current)
        return 'current time = {}; interval = {}'.format(self.current, self.interval)





def test_timer():
    t = Timer()
    time.sleep(1)
    print(t.time_now())
    