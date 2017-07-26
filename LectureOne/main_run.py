#-*-coding:utf-8-*-
#author = "Rachel"
"""
项目名称： 使用python实现蒙特卡洛模拟的期权估值
项目参考：《python金融大数据分析》第三章
"""
from __future__ import division,print_function
from mentcaro import bsm_call_value
from time import time
from math import exp,sqrt,log
from random import gauss,seed
import numpy as np
def run_main():
    """
    主函数
    :return:
    """
    S0 = 100
    K = 105
    T= 1.
    r = 0.05
    sigma = 0.2
    init_value = bsm_call_value(S0,K,T,r,sigma)
    print("BSM方法的期权估值：",init_value)
    M = 50    # 子时段个数
    dt = T / M    # 子时段时间间隔
    I = 250000  #迭代次数
    #方法1：纯python,内置函数和标准库
    t0 = time()
    S = []
    for i in range(I):
        path = []
        for t in range(M+1):
            if t ==0:
                path.append(S0)
            else:
                z = gauss(0.,1.)
                S_t = path[t-1]*exp((r -0.5*sigma**2)*dt + sigma*sqrt(dt)*z)
                path.append(S_t)
        S.append(path)
    C_0 = exp(-r*T)*sum([max(path[-1]-K,0)for path in S])/I

    duration = time() - t0
    print ("使用纯python实现期权估值的模型：",C_0)
    print ("耗时{}秒".format(duration))

    #方法2. 向量化numpy,使用numpy功能实现更加紧凑高效的版本
    t1 = time()
    S = np.zeros((M+1,I))
    S[0]=S0
    for t in range(1,M+1):
        z = np.random.standard_normal(I)
        S[t] = S[t-1]*np.exp((r - 0.5*sigma**2)*dt+sigma*sqrt(dt)*z)
    C_0 = exp(-r*T)*np.sum(np.maximum(S[-1]-K,0))/I

    duration2 = time()-t1
    print ("使用numpy实现期权估值的模拟：",C_0)
    print("耗时{}秒".format(duration2))

if __name__ == '__main__':
    run_main()






