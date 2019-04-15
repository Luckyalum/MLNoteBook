#-*- coding=utf-8 -*-
from math import log
import operator
import itertools

#建立一个简单的数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],\
               [1, 1, 'yes'],\
               [1, 0, 'no'],\
               [0, 1, 'no'],\
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#判断数据是否是连续型数据
def isContinuity(data):
    try:
        float(data)
        return True
    except:
        return False

#获取每个属性的取值集合，连续属性返回["是","否"]
def getVals(dataSet):
    reFeat = []
    numFeatures = len(dataSet[0])-1
    for i in range(numFeatures): 
        if(isContinuity(dataSet[0][i])):
            feat = ["是","否"]
        else:
            feat = list(set([example[i] for example in dataSet]))
        reFeat.append(feat)
    return reFeat

#划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet 

#计算出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), \
                        key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#计算信息熵
def calEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Ent = 0.0 
    for key in labelCounts:
        if labelCounts[key] == 0:
            p_current = 0
        else:
            p_current = float(labelCounts[key])/numEntries
        Ent += -p_current*log(p_current,2) 
    return Ent

#计算信息增益
def calGain(dataSet, feature, baseEnt):
    featList = [example[feature] for example in dataSet]
    uniqueVals = set(featList)
    newEntropy = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, feature, value)
        p_current = float(len(subDataSet)/len(dataSet))
        newEntropy += p_current*calEnt(subDataSet)
    Gain = baseEnt - newEntropy
    return Gain

#计算信息增益率
def calGainRatio(dataSet, feature, baseEnt):
    featList = [example[feature] for example in dataSet]
    uniqueVals = set(featList)
    #理论上IV初值为0，但是为了避免除零的问题，所以给一个足够小的值
    IV = 0.000000001
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, feature, value)
        p_current = float(len(subDataSet)/len(dataSet))
        IV += -p_current*log(p_current, 2)
    Gain = calGain(dataSet, feature, baseEnt)/IV
    return Gain

#计算基尼值
def calGini(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    Gini = 0
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    pList = [0 if labelCounts[key]==0 else float(labelCounts[key])/numEntries \
                for key in labelCounts]
    #获得p的两两组合的list
    comList = itertools.combinations(pList,2)
    for item in comList:
        Gini += item[0]*item[1]
    return Gini

#计算基尼指数
def calGiniIndex(dataSet, feature):
    featList = [example[feature] for example in dataSet]
    uniqueVals = set(featList)
    GiniIndex = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, feature, value)
        p_current = float(len(subDataSet)/len(dataSet))
        GiniIndex += p_current*calGini(subDataSet)
    return GiniIndex 

#根据信息增益选择最好的特征
def chooseBestFeatureByGain(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEnt = calEnt(dataSet)
    bestGain = 0.0 
    bestFeature = -1
    for i in range(numFeatures):
        Gain = calGain(dataSet, i, baseEnt)
        if Gain > bestGain:
            bestGain = Gain
            bestFeature = i
    return bestFeature

#根据信息增益率选择最好的特征
def chooseBestFeatureByGainRatio(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEnt = calEnt(dataSet)
    bestGainRatio = 0.0 
    bestFeature = -1
    for i in range(numFeatures):
        GainRatio = calGainRatio(dataSet, i, baseEnt)
        if GainRatio > bestGainRatio:
            bestGainRatio = GainRatio
            bestFeature = i
    return bestFeature

#根据基尼指数选择最好的特征
def chooseBestFeatureByGiniIndex(dataSet):
    numFeatures = len(dataSet[0])-1
    bestGiniIndex = calGiniIndex(dataSet, 0) 
    bestFeature = -1
    for i in range(1,numFeatures):
        GiniIndex = calGiniIndex(dataSet, i)
        if GiniIndex < bestGiniIndex:
            bestGiniIndex = GiniIndex
            bestFeature = i
    return bestFeature


