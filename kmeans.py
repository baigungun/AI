#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
@version: 1.0
@author: huangwan
@license: peopleNet
@contact: wan_huang@people2000.net
@software: PyCharm
@file: kmeans.py
@time: 2017/11/28 16:09
"""

from numpy import *
import random

class KmeansTest():
    def __init__(self):
        pass

    #读取文件，获取矩阵
    def loadData(self,file):
        dataSet = []
        with open(file) as f:
            for line in f:
                if line.strip() == "":
                    continue
                lineArr = line.strip().split("\t",1)
                linedata = lineArr[1].split("\t")
                data = map(float,linedata)
                dataSet.append(data)
        return dataSet

    #欧式距离
    def euclDistance(self,vector1,vector2):
        return sqrt(sum(power(vector1-vector2,2)))

    #初始化质心
    def initCentroids(self,dataSet,k):
        numSamples,dim = dataSet.shape
        centroids = zeros((k,dim))
        for i in range(k):
            index = int(random.uniform(0,numSamples))
            centroids[i,:] = dataSet[index,:]
        return centroids

    #k-means聚类
    def kmeans(self,dataSet,k):
        numSamples = dataSet.shape[0]
        #第一列存该条样本属于哪一类，第二列存该样本和他质心的误差
        clusterAssment = mat(zeros((numSamples,2)))
        clusterChanged = True

        #step1 : init centroids
        centroids = self.initCentroids(dataSet,k)
        while clusterChanged:
            print "=================== begin ============"
            clusterChanged = False
            #for each sample
            for i in xrange(numSamples):
                minDist = 1000000.0
                minIndex = 0
                #for each centroid
                # print "======== each sample  ======",dataSet[i,:]
                #step2: find the centroid who is closest
                # print "==== cal distance begin =="
                for j in range(k):
                    distance = self.euclDistance(centroids[j,:],dataSet[i,:])
                    if distance < minDist:
                        minDist = distance
                        minIndex = j
                print "mindis = ",minDist," min idex = ",minIndex," org cluster = ",clusterAssment[i,0]
                # print "==== cal distance end =="
                # step3: update its cluster
                if clusterAssment[i,0] != minIndex:
                    clusterChanged = True
                    clusterAssment[i,:] = minIndex,minDist**2

            #step4:update centroids
            for j in range(k):
                print "dddd ", clusterAssment[:,0].A,j,nonzero(clusterAssment[:,0].A == j),nonzero(clusterAssment[:,0].A == j)[0]
                pointsInCluster = dataSet[nonzero(clusterAssment[:,0].A == j)[0]]
                centroids[j,:] = mean(pointsInCluster,axis=0)
            print "=================== end ============"
        print 'ok ! cluster finish'
        return centroids, clusterAssment


    def mainProcess(self,file):
        print 'step 1 : load data ...'
        dataSet = mat(self.loadData(file))

        print 'step 2: clustering ...'
        k = 4
        centroids,clusterAssment = self.kmeans(dataSet,k)
        # print centroids


if __name__ == '__main__':
    file = "../file/1"
    k = KmeansTest()
    k.mainProcess(file)
