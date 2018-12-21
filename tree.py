#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
@version: 1.0
@author: huangwan
@license: peopleNet
@contact: wan_huang@people2000.net
@software: PyCharm
@file: tree.py
@time: 2018/6/8 11:14
"""
#决策树示例代码
from math import log
import operator
import treePlotter
import pickle

'''创建树'''
#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#按照给定的特征划分数据集
#输入参数：待划分的数据集，特征索引，特征值
#返回参数：特征值为指定值的数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式，在当前特征集中使用最好的特征进行划分
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    #按照特征划分数据集，划分之后信息增益最大的特征返回。信息增益表示数据无序度的减少。
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

#如果数据集已经处理了所有的属性，但是类标签仍然不是唯一的。采用多数表决的方式来确定该叶子节点的分类。
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#创建树,labels表示特征的名称（用于生成树节点-非叶子节点），dataset表示数据集，包含具体的特征内容和数据标签。
def createTree(dataSet,labels):
    print "tree create =========="
    classList = [example[-1] for example in dataSet]
    print "classList = ",classList
    #类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return  classList[0]
    #遍历完所有特征时(当前特征个数为1)，返回出现次数最多的类别
    print "dataset[0]= ",dataSet[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    print "cur bestFeat = ",bestFeat,"cur bestFeatLabel = ",bestFeatLabel
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    print "cur featValues = ",featValues
    uniqueVals = set(featValues)

    for val in uniqueVals:
        subLabels = labels[:]
        print "for val = ",val,"cur subLabels = ",subLabels
        print "cur tree = ",myTree
        myTree[bestFeatLabel][val] = createTree(splitDataSet(dataSet,bestFeat,val),subLabels)

    return myTree

'''测试算法'''
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

#持久化树结构
def storeTree(inputTree,filename):
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)



def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def mainTest():
    print "tree test..."
    dataSet , labels = createDataSet()
    # print calcShannonEnt(dataSet)
    # print splitDataSet(dataSet,0,1)
    # print chooseBestFeatureToSplit(dataSet)
    mytree = createTree(dataSet,labels)
    print "mytree = ",mytree
    # print "labels = ",labels

    # myTreeTest = treePlotter.retrieveTree(0)
    # print myTreeTest,labels,type(myTreeTest)
    # print classify(myTreeTest,labels,[1,0])

    storeTree(mytree,'tree.txt')
    print grabTree('tree.txt')

#使用隐形眼镜数据集进行测试
def testLenses():
    fr = open('../file/tree/lenses.txt')
    lenses = [line.strip().split("\t") for line in fr.readlines()]
    labels = ['age','prescript','astigmatic','tearRate']
    lensesTree = createTree(lenses,labels)
    treePlotter.createPlot(lensesTree)

if __name__ == '__main__':
    # mainTest()
    testLenses()
