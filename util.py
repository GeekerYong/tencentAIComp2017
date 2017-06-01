# -*- coding: utf-8 -*-
"""
Created on Wed May 10 08:51:22 2017

@author: fay
"""

import numpy as np
import scipy as sp
import pandas as pd

##结果评估函数
def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll
  

##读取数据
def loadData(path):
    data = pd.read_csv(path)
    return data
    
##写入结果
##def writeResult(result):
    
    