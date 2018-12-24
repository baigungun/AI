#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
@version: 1.0
@author: huangwan
@license: peopleNet
@contact: wan_huang@people2000.net
@software: PyCharm
@file: evaluation_metrics.py
@time: 2017/11/2 14:42
"""
'''
重点关注离线实验部分
'''

import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# 预测准确度
class Accuracy():
    def __init__(self):
        pass

    def getData(self,file):
        with open(file) as file:
            for line in file:
                if line.strip() == "":
                    continue

    '''衡量回归模型效果'''
    #用户u,物品i,rui是用户对物品的实际评分，pui是算法预测出来的评分
    #均方根误差
    def RMSE(self, records):
        return math.sqrt(sum([(rui - pui) * (rui - pui) for u, i, rui, pui in records])) \
               / float(len(records))
    #平均绝对误差
    def MAE(self, records):
        return sum([abs(rui - pui) for u, i, rui, pui in records]) \
               / float(len(records))

    '''topN 推荐场景，最常用'''
    #精确率和召回率
    # def PrecisionRecall(self,test,N):
    #     hit = 0
    #     n_recall = 0
    #     n_precision = 0
    #     for user, items in test.items():
    #         rank = Recommend(user,N)
    #         hit = len(rank & items)
    #         n_recall += len(items)
    #         n_precision += N
    #     return [hit / (1.0 *n_recall), hit / (1.0*n_precision)]

    '''衡量分类模型效果'''

    # y_true（实际结果）, y_pred（预测结果）
    # TP = (y_pred==1)*(y_true==1)
    # FP = (y_pred==1)*(y_true==0)
    # FN = (y_pred==0)*(y_true==1)
    # TN = (y_pred==0)*(y_true==0)
    # TP + FP = y_pred==1
    # TP + FN = y_true==1

    #分别计算精准率，召回率和F1得分
    def precision_score(self,y_true,y_pred):
        return ((y_true==1)*(y_pred==1)).sum() / (y_pred==1).sum()

    #计算auc
    def aucScore(self):
        y_true = np.array([0,0,1,1])
        y_scores = np.array([0.1,0.4,0.35,0.8])
        print roc_auc_score(y_true,y_scores)

    #画图roc曲线
    def rocPoint(self):
        y = np.array([1,1,2,2]) #实际值
        scores = np.array([0.1,0.4,0.35,0.8]) #预测值
        fpr,tpr,thresholds = metrics.roc_curve(y,scores,pos_label=2)#pos_label=2，表示值为2的实际值为正样本
        print fpr
        print tpr
        print thresholds

if __name__ == '__main__':
    ac = Accuracy()
    records = ac.getData("../file/records")
    # ac.RMSE(records)
    # ac.aucScore()
    # ac.rocPoint()
    y_true = np.array([0,0,1,1])
    y_scores = np.array([0.1,0.4,0.35,0.8])
    # print ac.precision_score(y_true,y_scores)
    print True * True, True*False,False*True,False*False
