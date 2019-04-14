#-*- coding=utf-8 -*-
import decision_tree
import treePlotter
import imp
import util
#隐形眼镜数据集
fr=open('./lenses.txt')
lenses=[inst.strip().split('  ') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree = decision_tree.createTree(lenses, lensesLabels)
treePlotter.createPlot(lensesTree)

#使用信息增益划分
imp.reload(decision_tree)
fr=open('./watermelon2.0.txt','r',encoding = 'utf8')
dateSet=[inst.strip().split(',') for inst in fr.readlines()]
Labels=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
features = util.getVal(dateSet)
watermelonTree = decision_tree.createTree(dateSet, Labels, features)
treePlotter.createPlot(watermelonTree)



#处理连续值
import continuity
import treePlotter
import util
fr=open('./watermelon3.0.txt','r',encoding = 'utf8')
dateSet=[inst.strip().split(',') for inst in fr.readlines()]
Labels=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感','密度','糖率']
vals = util.getVals(dateSet)
watermelonTree = continuity.createTree(dateSet, Labels, vals)
treePlotter.createPlot(watermelonTree)

#预剪枝
imp.reload(decision_tree)
fr=open('./watermelon2.0.txt','r',encoding = 'utf8')
watermelon=[inst.strip().split(',') for inst in fr.readlines()]
watermelonLabels=['色泽', '根蒂', '敲声', '纹理', '脐部', 
            '触感']
# 将原数据集划分为训练集和验证集
# 可以使用留出法或指定哪些作为验证集
data, test = decision_tree.getTest(watermelon,[4,5,8,9,11,12,13])
#data, test = decision_tree.getTest(watermelon)
watermelonTree = decision_tree.createTree(data, watermelonLabels,divide_by="gainRatio")
treePlotter.createPlot(watermelonTree)
watermelonTree = decision_tree.createTreeWithPrePruning(data, test, watermelonLabels)
treePlotter.createPlot(watermelonTree)