#-*- coding=utf-8 -*-
import util
#对连续变量划分数据集  
#划分出小于value的数据样本和大于value的数据样本集  
def splitContinuousDataSet(dataSet,axis,value):  
    retDataSetLeft=[]  
    retDataSetRight=[] 
    for featVec in dataSet:  
        reducedFeatVec=featVec[:axis]  
        reducedFeatVec.extend(featVec[axis+1:])  
        if float(featVec[axis])>value:  
            retDataSetRight.append(reducedFeatVec)  
        else:
            retDataSetLeft.append(reducedFeatVec)  
    return retDataSetLeft, retDataSetRight
 
#增加可以处理连续数据的部分 
def chooseBestFeatureToSplit(dataSet,labels):  
    numFeatures=len(dataSet[0])-1  
    baseEntropy=util.calEnt(dataSet)  
    bestInfoGain=0.0  
    bestFeature=-1  
    bestSplitDict={}  
    for i in range(numFeatures):  
        featList=[example[i] for example in dataSet]  
        #对连续型特征进行处理  
        if util.isContinuity(featList[0]):  
            #产生n-1个候选划分点  
            sortfeatList=sorted(list(map(float,featList)))  
            splitList=[]  
            for j in range(len(sortfeatList)-1):  
                splitList.append((sortfeatList[j]+sortfeatList[j+1])/2.0)  
              
            bestSplitEntropy=10000  
            slen=len(splitList)  
            #求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点  
            for j in range(slen):  
                value=splitList[j]  
                newEntropy=0.0  
                subDataSetLeft, subDataSetRight=splitContinuousDataSet(dataSet,i,value)  
                prob0=len(subDataSetLeft)/float(len(dataSet))  
                newEntropy+=prob0*util.calEnt(subDataSetLeft)  
                prob1=len(subDataSetRight)/float(len(dataSet))  
                newEntropy+=prob1*util.calEnt(subDataSetRight)  
                if newEntropy<bestSplitEntropy:  
                    bestSplitEntropy=newEntropy  
                    bestSplit=j  
            #用字典记录当前特征的最佳划分点  
            bestSplitDict[labels[i]]=splitList[bestSplit]  
            infoGain=baseEntropy-bestSplitEntropy  
        #对离散型特征进行处理  
        else:  
            uniqueVals=set(featList)  
            newEntropy=0.0  
            #计算该特征下每种划分的信息熵  
            for value in uniqueVals:  
                subDataSet=util.splitDataSet(dataSet,i,value)  
                prob=len(subDataSet)/float(len(dataSet))  
                newEntropy+=prob*util.calEnt(subDataSet)  
            infoGain=baseEntropy-newEntropy  
        if infoGain>bestInfoGain:  
            bestInfoGain=infoGain  
            bestFeature=i  
    #若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理  
    #即是否小于等于bestSplitValue  
    if util.isContinuity(dataSet[0][bestFeature]):        
        bestSplitValue=bestSplitDict[labels[bestFeature]]          
        labels[bestFeature]=labels[bestFeature]+'<='+str(bestSplitValue)  
        for i in range(len(dataSet)):  
            if float(dataSet[i][bestFeature])<=float(bestSplitValue):  
                dataSet[i][bestFeature]='是'
            else:  
                dataSet[i][bestFeature]='否'  
    return bestFeature  

#建立树
def createTree(dataSet, labels, vals):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return util.majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet,labels)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    subLabels = labels[:]
    del(subLabels[bestFeat])
    subVals = vals[:]
    uniqueVals = subVals[bestFeat]
    del(subVals[bestFeat])
    for value in uniqueVals:
        subSet = util.splitDataSet(dataSet, bestFeat, value)
        if(subSet == []):
            myTree[bestFeatLabel][value] =  util.majorityCnt(classList)
        else:
            myTree[bestFeatLabel][value] =  createTree(subSet, subLabels, subVals)
    return myTree


