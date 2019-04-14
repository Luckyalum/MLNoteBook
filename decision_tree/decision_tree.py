#-*- coding=utf-8 -*-
from math import log
import itertools
import operator
import random
import re
import util


#建立树
def createTree(dataSet, labels, features, divide_by="gain"):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，
    # 也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包
    # 含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return util.majorityCnt(classList)
    # 选择划分特征依据
    divide_func = {"gain":util.chooseBestFeatureByGain, 
                "gainRatio":util.chooseBestFeatureByGainRatio,
                "giniIndex":util.chooseBestFeatureByGiniIndex}
    bestFeat = divide_func[divide_by](dataSet)
    #bestFeat = chooseBestFeatureToSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #书上原句有错
    #原句：del(labels[bestFeat])
    #这样会修改labels，后面运行会报'no surfacing' is not in list错误
    #修改如下：
    subLabels = labels[:]
    del(subLabels[bestFeat])
    subFeatures = features[:]
    uniqueVals = subFeatures[bestFeat]
    del(subFeatures[bestFeat])
    for value in uniqueVals:
        subSet = util.splitDataSet(dataSet, bestFeat, value)
        if(subSet == []):
            myTree[bestFeatLabel][value] =  util.majorityCnt(classList)
        else:
            myTree[bestFeatLabel][value] =  createTree(subSet, subLabels, subFeatures)
    return myTree

#分类器 
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]        
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

#获得训练集和验证集
def getTest(dataSet, axisList=[]):
    if len(axisList) == 0:
        return distillation(dataSet)
    trainSet = dataSet[:]
    testSet = [] 
    for each in axisList:
        example = dataSet[each-1]
        testSet.append(example)
        trainSet.remove(example)        
    return trainSet, testSet

#留出法
def distillation(dataSet):
    predictSetSize = int(0.3*len(dataSet))
    trainSetSize = len(dataSet) - predictSetSize
    trainSet = dataSet[:]
    predictSet = [] 
    for each in range(predictSetSize):
        randomIndex = random.randint(0,trainSetSize)
        example = trainSet[randomIndex]
        predictSet.append(example)
        trainSet.remove(example)
        trainSetSize -= 1
    return trainSet, predictSet

#使用验证集验证预测结果
def testing(myTree,data_test,labels):  
    error=0.0  
    for i in range(len(data_test)):  
        if classify(myTree,labels,data_test[i])!=data_test[i][-1]:  
            error+=1  
    return float(error) 



#当前树的预测值为投票所得major
def testingMajor(major,data_test):  
    error=0.0  
    for i in range(len(data_test)):  
        if major!=data_test[i][-1]:  
            error+=1  
    return float(error)  



