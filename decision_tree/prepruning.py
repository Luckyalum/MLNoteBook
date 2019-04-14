import util

#建立树-预剪枝
def createTreeWithPrePruning(dataSet, testSet, labels, divide_by="gain"):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    temp_labels=labels[:]
    divide_func = {"gain":chooseBestFeatureByGain, 
                "gainRatio":chooseBestFeatureByGainRatio,
                "giniIndex":chooseBestFeatureByGiniIndex}
    bestFeat = divide_func[divide_by](dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    subLabels = labels[:]
    del(subLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)       
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
                                        (dataSet, bestFeat, value), \
                                        subLabels,divide_by)
    #判断当前分支是否能带来泛化性能的提升
    if testing(myTree,testSet,temp_labels)<\
        testingMajor(majorityCnt(classList),testSet):  
        return myTree  
    return majorityCnt(classList) 