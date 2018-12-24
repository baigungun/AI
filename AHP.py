#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: huangwan
@license: peopleNet
@contact: wan_huang@people2000.net
@software: PyCharm
@file: ahp_test.py
@time: 2017/4/17 11:12
"""
import numpy as np

#层次分析法计算权重，和概率无关

class AHPTest():
    def __init__(self):
        self.n = 0
        self.RI = [0,0,0.58,0.90,1.12,1.24,1.32,1.41,1.45,1.49,1.51]

    #根据经验生成对比矩阵
    def set_vec_data(self):
        i = np.array([[1,3,5],[1/3.0,1,3],[1/5.0,1/3.0,1]],dtype=float)
        # i = np.array([
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [3,3,1,3,5,1,3,5,5,3,3,5,3,3,3,3,3,5,5,5],
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [1/3.0,1/3.0,1/5.0,1/3.0,1,1/5.0,1/3.0,1,1,1/3.0,1/3.0,1,1/3.0,1/3.0,1/3.0,1/3.0,1/3.0,1,1,1],
        #     [3,3,1,3,5,1,3,5,5,3,3,5,3,3,3,3,3,5,5,5],
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [1/3.0,1/3.0,1/5.0,1/3.0,1,1/5.0,1/3.0,1,1,1/3.0,1/3.0,1,1/3.0,1/3.0,1/3.0,1/3.0,1/3.0,1,1,1],
        #     [1/3.0,1/3.0,1/5.0,1/3.0,1,1/5.0,1/3.0,1,1,1/3.0,1/3.0,1,1/3.0,1/3.0,1/3.0,1/3.0,1/3.0,1,1,1],
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [1/3.0,1/3.0,1/5.0,1/3.0,1,1/5.0,1/3.0,1,1,1/3.0,1/3.0,1,1/3.0,1/3.0,1/3.0,1/3.0,1/3.0,1,1,1],
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [1,1,1/3.0,1,3,1/3.0,1,3,3,1,1,3,1,1,1,1,1,3,3,3],
        #     [1/3.0,1/3.0,1/5.0,1/3.0,1,1/5.0,1/3.0,1,1,1/3.0,1/3.0,1,1/3.0,1/3.0,1/3.0,1/3.0,1/3.0,1,1,1],
        #     [1/3.0,1/3.0,1/5.0,1/3.0,1,1/5.0,1/3.0,1,1,1/3.0,1/3.0,1,1/3.0,1/3.0,1/3.0,1/3.0,1/3.0,1,1,1],
        #     [1/3.0,1/3.0,1/5.0,1/3.0,1,1/5.0,1/3.0,1,1,1/3.0,1/3.0,1,1/3.0,1/3.0,1/3.0,1/3.0,1/3.0,1,1,1]
        #
        # ],dtype=float)

        #for test
        # i = np.array([
        #     [1,7,4,5,3],
        #     [1/7.0,1,1/3.0,1/4.0,1/2.0],
        #     [1/4.0,3,1,2/3.0,2/3.0],
        #     [1/5.0,4,3/2.0,1,1],
        #     [1/3.0,2,3/2.0,1,1]
        # ],dtype=float)


        self.n = np.shape(i)[0]
        return i

    #矩阵的最大特征值和特征向量
    def get_vector(self,vec):
        a,b = np.linalg.eig(vec)
        # print a,b
        max_a = a.max()
        idx = np.argmax(a)
        vec_b = b[:,idx]
        # print max_a, vec_b
        return max_a,vec_b

    #传入特征向量，归一化得到权重
    def get_weight(self,vec):
        v_sum = vec.sum()
        return map(lambda x:x/v_sum,vec)

    #传入原始矩阵，计算得到权重
    def get_weight2(self,vec):
        vT= vec#vec.T
        # print "vt = ",vT
        colSumV = vT.sum(axis=0)
        # print "colSumV=",colSumV
        colNum = vT.shape[1]
        rowNum = vT.shape[0]
        #列循环
        tmpV = np.zeros((rowNum,colNum))
        for i in range(colNum):
            colV = vT[:,i]
            tmpColV = map(lambda x:x/colSumV[i],colV)
            tmpV[:,i] = tmpColV

        # print "tmpV = ",tmpV

        rowSumV = tmpV.sum(axis=1)
        sum = rowSumV.sum()
        w = map(lambda x:x/sum,rowSumV)
        # print "w=",w
        return w



    #检查矩阵的一致性，传入最大特征值
    def check_cr(self,max_f):
        num = self.n
        if num > 11:
            ri = self.RI[-1]
        else:
            ri = self.RI[num-1]
        print "shape=",self.n, "ri=",ri
        cr0 = (max_f - num) / (num -1) / ri
        print "cr0=",cr0
        if cr0 <= 0.1:
            return True
        else:
            return False

    def mainProcess(self):
        v = self.set_vec_data()
        a,b = self.get_vector(v)
        print "最大特征值：",a
        print "最大特征值对应的特征向量：",b
        flag = self.check_cr(a)
        if flag:
            # w0 = self.get_weight(b)
            w0 = self.get_weight2(v)
            print "维度权重值：",w0
        else:
            print "初始矩阵一致性不够，请重新构造矩阵"


if __name__ == '__main__':
    ahp = AHPTest()
    vec = ahp.mainProcess()
