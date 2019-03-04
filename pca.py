
#https://blog.csdn.net/bbbeoy/article/details/80301262
#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
@version: 1.0
@author: huangwan
@license: peopleNet
@contact: wan_huang@people2000.net
@software: PyCharm
@file: pcaTest.py
@time: 2018/5/22 14:54
"""
import numpy as np

class PCATest():
    def __init__(self):
        pass

    #29 features
    def setData(self,file):
        dataMat = []
        with open(file) as file:
            for line in file:
                if line.strip() == "":
                    continue
                infos = line.strip().split("\t")
                list = map(float,infos[1:len(infos)-1])
                dataMat.append(list)
        # print dataMat
        return np.array(dataMat,dtype='float64')



    '''通过方差（特征值）的百分比来计算将数据降到多少维是比较合适的，
       函数传入的参数是特征值和百分比percentage，返回需要降到的维度数num'''
    def eigValPct(self,eigVals,percent):
        sortArray = np.sort(eigVals)
        sortArray = sortArray[-1::-1] #降序排列
        arraySum = sum(sortArray)
        tmpSum = 0
        num = 0
        for i in sortArray:
            tmpSum += i
            num += 1
            if tmpSum >= arraySum * percent:
                return num

    '''percentRatio=1认为取全量特征，否则只取部分特征进行投影.percentRatio为方差百分比'''
    def getScore(self,mat,percentRatio=1):
        p,n = np.shape(mat)
        print "p, n",p,n
        #对每一列求均值
        mean = np.mean(mat,0)
        print "mean = ",len(mean),mean
        #每个数据减去自己所在列的均值
        meanRemovedMat = mat - mean
        #计算协方差
        cov = np.cov(meanRemovedMat,rowvar=0)
        # print "cov = ",cov
        # print "shape cov = ",np.shape(cov)
        # cov2 = np.dot(mat.T, mat) / (p - 1)
        # print "cov2 = ",cov2
        # print "shape cov2 ",np.shape(cov2)

        #计算特征值和特征向量
        eigVals,eigVects = np.linalg.eig(np.mat(cov))
        # print eigVals,eigVects
        eiglen = self.eigValPct(eigVals,percentRatio)

        #对特征进行裁剪，取前k个特征
        eigValIndice = np.argsort(eigVals)
        # print "eigValIndice = ",eigValIndice
        n_eigValIndice = eigValIndice[-1:-(eiglen+1):-1]
        # n_eigVect = eigVects[:,n_eigValIndice]
        print eigVals,n_eigValIndice

        scoreDict = {}
        #计算每条数据在所有向量上的投影的和，作为异常得分
        for i in range(p):
            datai = np.mat(mat[i])
            # print "mat[i] = ",datai,np.shape(datai)
            sumi = 0.0
            for j in n_eigValIndice:
                eigVal = eigVals[j]
                eigVect = eigVects[j]
                # print " ..... loop  eigVect = ",eigVal,eigVect
                dij = np.square(np.dot(eigVect,datai.T)) / eigVal
                # print "dij = ",dij
                sumi = sumi + dij
            # print "sumi = ",np.array(sumi)[0][0]
            datasumi = np.array(sumi)[0][0]
            scoreDict[i] = datasumi
        return scoreDict

    '''阈值算法'''
    def getScoreThd(self,scoreDict):
        scoreList = []
        for key in scoreDict.keys():
            scoreList.append(scoreDict.get(key))
        mean = np.mean(scoreList)
        var = np.var(scoreList)
        print "score mean = ",mean,"score var = ",var
        rlt = mean + 3*var
        return rlt


    '''根据阈值给数据打标签'''
    def setDataLabel(self,scoreDict,scoreThd):
        labelDict = {}
        keys = scoreDict.keys()
        for key in keys:
            val = scoreDict.get(key,0.0)
            if float(val) > float(scoreThd):
                labelDict[key] = "1"
            else:
                labelDict[key] = "0"
        return labelDict


    '''对比实际结果和预测结果，计算准确率'''
    def getAUC(self,file,labelDict):

        linenum = 0
        realLabelDict = {}
        with open(file) as file:
            for line in file:
                lined = line.strip()
                if lined == "":
                    continue
                tokens = lined.split("\t")
                realLabel = tokens[-1].replace("\"","")
                realLabelDict[linenum] = realLabel
                linenum += 1
        # print "realLabelDict = ",realLabelDict
        # print "labelDict = ",labelDict

        self.calRatio(realLabelDict,labelDict)

    def calRatio(self,realDict,calDict):
        tp = 0 #yes - yes
        fn = 0 #yes - no
        fp = 0 #no - yes
        tn = 0 #no -no

        keys = realDict.keys()
        num = len(keys)

        for key in keys:
            realLabel = realDict.get(key)
            calLabel = calDict.get(key)
            if realLabel == "0" and calLabel == "0":
                tp += 1
            elif realLabel == "0" and calLabel == "1":
                fn += 1
            elif realLabel == "1" and calLabel == "0":
                fp += 1
            elif realLabel == "1" and calLabel == "1":
                tn += 1
        print "tp,tn,fp,fn = ",tp,tn,fp,fn
        #正确率
        accuracy = (tp + tn) / float(num)
        #精度
        precision = tp /float(tp + fp)
        #召回率
        recall = tp / float(tp + fn)
        print "accuracy, precision, recall",accuracy,precision,recall

    def mainProcess(self,file):
        #读取文档，将数据转换成矩阵
        mat = self.setData(file)
        #获取每行样本的得分
        scoreDict = self.getScore(mat,0.999)
        # print "scoreDict = ",scoreDict,len(scoreDict)
        #取异常值的阈值
        scoreThd = self.getScoreThd(scoreDict)
        print "scoreThd = ",scoreThd
        #根据结果，给数据打标签，对比实际标签计算准确率
        labelDict = self.setDataLabel(scoreDict,scoreThd)
        # print "labelDict = ",labelDict
        self.getAUC(file,labelDict)




if __name__ == '__main__':
    p = PCATest()
    p.mainProcess("../file/credit.org")



