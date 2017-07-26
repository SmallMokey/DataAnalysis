#-*- coding:utf-8 -*-
#author = "Rachel"
from __future__ import division,print_function
from math import log,sqrt,exp
from scipy import stats
def bsm_call_value(S0,K,T,r,sigma):
    """
    根据BSM公式计算期权估值
    :param S0: 初始标的物价格，t = 0
    :param K: 期全行权价格
    :param T: 期权到期日
    :param r: 固定无风险短期利率
    :param sigma: 标的物固定波动率
    :return:
    """
    S0 =float(S0)
    d1 = (log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = (log(S0/K)+(r-0.5*sigma**2)*T)/(sigma*sqrt(T))
    value = S0*stats.norm.cdf(d1,0.0,1.0)-K*exp(-r*T)*stats.norm.cdf(d2,0.0,1.0)
    return value





