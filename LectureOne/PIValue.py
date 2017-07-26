#coding = "utf-8"
#author = "Rachel"
from __future__ import division
import time
import random
def PIvaue():
    for i in range(2,10):
        counter = 0
        startTime = time.clock()
        for j in range(10**i):
            x = random.uniform(-1,1)
            y = random.uniform(-1,1)
            print j
            if(x**2+y**2 < 1):
                counter +=1
        endTime = time.clock()
        pi = 4*(counter/10**i)
        print("number:",i)
        print("PI:",pi)
        print("time:",startTime-endTime)

PIvaue()





