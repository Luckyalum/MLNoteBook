#-*- coding=utf-8 -*-

#存储树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

#读取树
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)