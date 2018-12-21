#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
@version: 1.0
@author: huangwan
@license: peopleNet
@contact: wan_huang@people2000.net
@software: PyCharm
@file: kNN.py
@time: 2018/6/6 15:42
"""
#knn一般采用欧式距离或者曼哈顿距离，因此一般都需要将距离归一化。
# K < 20并通常选取奇数，通过交叉验证来选择最合适的K值。
#投票原则除了少数服从多数，可以采用距离的倒数作为权重，以权重和的最高值作为分类标准。


import operator
import numpy as np

#读取数据文件并格式化成待处理的数据格式
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLaberVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFormLine = line.split("\t")
        returnMat[index,:] = listFormLine[0:3]
        classLaberVector.append(int(listFormLine[-1]))
        index += 1
    # print returnMat
    # print classLaberVector
    return returnMat,classLaberVector

#使用欧式距离，存在数据量纲问题，需要提前将数据归一化。(cur - min) / (max -min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals #1x3

    normalDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    #tile函数是将矩阵在行和列上分别拷贝m份和1份，相当于将1*3的矩阵拷贝成为1000*3的矩阵
    normalDataSet = dataSet -np.tile(minVals,(m,1))
    normalDataSet = normalDataSet / np.tile(ranges,(m,1))
    return normalDataSet, ranges, minVals

#欧式距离,输入点和原始数据集，输出该点到数据集中每个点的欧式距离
def euclideanDistance(inX, dataSet):
    dataSetSize = dataSet.shape[0]
    #拷贝m行，与原始数据集中每一行做差
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    #求每一个数的平方，对列求和，然后开方，是欧式距离的计算方式
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    return distances

#曼哈顿距离

#兰氏距离，输入点和原始数据集，输出该点到数据集中每个点的兰氏距离
def canberraDistance(inX, dataSet):
    dataSetSize = dataSet.shape[0]
    #拷贝m行，与原始数据集中每一行做差
    diffMat = abs(np.tile(inX,(dataSetSize,1)) - dataSet)
    sumMat = np.tile(inX,(dataSetSize,1)) + dataSet
    dMat = diffMat / sumMat
    sqDistances = dMat.sum(axis=1)
    distances = sqDistances / float(dataSet.shape[1])
    return distances


#knn算法核心
#输入参数：inX是待分类的向量，dataSet是已知数据集，labels为标签向量，k为选择的最近邻居的个数
#输出参数：inX的分类结果
def classify0(inX, dataSet, labels, k ,type=1):
    #距离公式选择
    if type == 1:
        distances = euclideanDistance(inX,dataSet)
    elif type == 2:
        distances = canberraDistance(inX,dataSet)
    #argsort返回的是数组从小到大的索引值
    sortedDistIndicies = distances.argsort()
    #获取前K个邻居的分类以及对应分类的个数
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    #对字典排序，返回分类最多的那个结果
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    # print "sortedClassCount ",sortedClassCount
    return sortedClassCount[0][0]


#使用约会数据测试knn分类器效果，以错误率作为衡量标准
def datingClassTest(filename,type = 1):
    hoRatio = 0.10 #确定测试数据的比例，剩余的数据作为训练数据
    datingDataMat , datingLabels = file2matrix(filename)
    if type == 1:
        normMat, ranges,minVals = autoNorm(datingDataMat)
        m = normMat.shape[0]
    elif type == 2:
        m = datingDataMat.shape[0]

    numTestVecs = int(m*hoRatio)
    errorCount = 0.0

    #遍历头100条数据，将其与剩余的900个点分别参与计算并得到预估的分类结果，与实际结果比较
    for i in range(numTestVecs):
        if type == 1:
            classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:]
                                         ,datingLabels[numTestVecs:m],3)
        elif type == 2:
            classifierResult = classify0(datingDataMat[i,:],datingDataMat[numTestVecs:m,:]
                                         ,datingLabels[numTestVecs:m],3,type)

        # print "the classifier came back with:%d ,the real answer is: %d"%(classifierResult,datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "errorCount = ",errorCount, "numTestVecs = ",numTestVecs,"m = ",m
    print "the total error rate is :%f"%(errorCount / float(numTestVecs))


#使用算法，输入一个新的点，看看属于哪个分类；type=1表示使用欧式距离，type=2表示使用兰氏距离
def classifyPerson(filename,inX,type=1):
    if type == 1:
        #欧式距离，需要将新输入的点进行归一化之后计算
        datingDataMat , datingLabels = file2matrix(filename)
        normMat, ranges,minVals = autoNorm(datingDataMat)
        inArr = np.array(inX)
        classifyRlt = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
        print "classifyRlt = ",classifyRlt
    elif type == 2:
        #兰氏距离，不需要归一化
        datingDataMat , datingLabels = file2matrix(filename)
        inArr = np.array(inX)
        classifyRlt = classify0(inArr, datingDataMat, datingLabels, 3,type)
        print "classifyRlt = ",classifyRlt


def mainProcess(filename):
    datingClassTest(filename,2)
    inX = [10,10000,0.5]
    classifyPerson(filename,inX,1)
    # datingDateMat,datingLabels = file2matrix(filename)
    # picture(datingDateMat,datingLabels)
    # autoNorm(datingDateMat)

if __name__ == '__main__':
    # mainProcess("../file/knn/datingTestSet2.txt")
    print canberraDistance(np.array([1.0,2.0,3.0]),np.matrix([[4.0,5.0,6.0],[4.0,5.0,6.0]]))
    print euclideanDistance(np.array([1.0,2.0,3.0]),np.matrix([[4.0,5.0,6.0],[4.0,5.0,6.0]]))
