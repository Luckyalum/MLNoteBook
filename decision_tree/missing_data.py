#-*- coding=utf-8 -*-
import re
import util
import operator
from math import log

def calD(dataSet,feature,w):
    D = []
    sub_w = []
    for i,example in enumerate(dataSet):
        if example[feature] != "":
            D.append(example)
            sub_w.append(w[i])
    return D,sub_w

def calRho(sub_w,w):
    return sum(sub_w)/sum(w)

def calP(subDataSet,k,sub_w):
    D_kw = []
    for i,example in enumerate(subDataSet):
        if example[-1] == k:
            D_kw.append(sub_w[i])
    return sum(D_kw)/sum(sub_w)

def calR(subDataSet,feature,v,sub_w):
    D_vw = []
    for i,example in enumerate(subDataSet):
        if example[feature] == v:
            D_vw.append(sub_w[i])
    return sum(D_vw)/sum(sub_w)


#信息熵推广形式
def calEntPromotion(subDataSet,sub_w):
    labelCounts = set(example[-1] for example in subDataSet)
    Ent = 0.0
    for k in labelCounts:
        p = calP(subDataSet,k,sub_w)
        Ent += - p*log(p,2)
    return Ent

#信息增益推广形式
def calGainPromotion(dataSet, feature):
    w = [1.0]*len(dataSet)
    D,sub_w = calD(dataSet,feature,w)
    rho = calRho(sub_w,w)
    baseEnt = calEntPromotion(D,sub_w)
    featList = [example[feature] for example in D]
    uniqueVals = set(featList)
    newEntropy = 0.0
    for value in uniqueVals:
        subDataSet = []
        s_w = []
        for i,example in enumerate(D):
            if example[feature] == value:
                subDataSet.append(example)
                s_w.append(sub_w[i])
        r = calR(subDataSet,feature,value,sub_w)
        newEntropy += r*calEntPromotion(subDataSet,s_w)
    Gain = rho*(baseEnt - newEntropy)
    return Gain

