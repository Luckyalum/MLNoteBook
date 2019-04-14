#-*- coding=utf-8 -*-
import re
import util

#连续数据+缺失数据
#由于在Tree中，连续值特征的名称以及改为了  feature<=value的形式  
#因此对于这类特征，需要利用正则表达式进行分割，获得特征名以及分割阈值  
def classify(inputTree,featLabels,testVec):  
    firstStr=list(inputTree.keys())[0]  
    if '<=' in firstStr:  
        featvalue=float(re.compile("(<=.+)").search(firstStr).group()[2:])  
        featkey=re.compile("(.+<=)").search(firstStr).group()[:-2]  
        secondDict=inputTree[firstStr]  
        featIndex=featLabels.index(featkey)  
        if testVec[featIndex]<=featvalue:  
            judge=1  
        else:  
            judge=0  
        for key in secondDict.keys():  
            if judge==int(key):  
                if type(secondDict[key]).__name__=='dict':  
                    classLabel=classify(secondDict[key],featLabels,testVec)  
                else:  
                    classLabel=secondDict[key]  
    else:  
        secondDict=inputTree[firstStr]  
        featIndex=featLabels.index(firstStr)  
        for key in secondDict.keys():  
            if testVec[featIndex]==key:  
                if type(secondDict[key]).__name__=='dict':  
                    classLabel=classify(secondDict[key],featLabels,testVec)  
                else:  
                    classLabel=secondDict[key]  
    return classLabel  