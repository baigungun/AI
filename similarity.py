#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
@version: 1.0
@author: huangwan
@license: peopleNet
@contact: wan_huang@people2000.net
@software: PyCharm
@file: similarity.py
@time: 2017/4/14 16:11
"""

class SimCal():

    def cos_v(self,v1,v2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a,b in zip(v1,v2):
            dot_product += a*b
            normA += a**2
            normB += b**2
        if normA == 0.0 or normB==0.0:
            return None
        else:
            return dot_product / ((normA*normB)**0.5)

    def pearson(self,p,q):
        #只计算两者共同有的
        cnt = 0
        same = 0
        for i in p:
            cnt += 1
            if i in q:
                same +=1
        # print "cnt=",cnt,"same=",same
        n = same
        #分别求p，q的和
        sumx = sum([p[i] for i in range(n)])
        sumy = sum([q[i] for i in range(n)])
        #分别求出p，q的平方和
        sumxsq = sum([p[i]**2 for i in range(n)])
        sumysq = sum([q[i]**2 for i in range(n)])
        #求出p，q的乘积和
        sumxy = sum([p[i]*q[i] for i in range(n)])
        # print sumxy
        #求出pearson相关系数
        up = sumxy - sumx*sumy/n
        down = ((sumxsq - pow(sumx,2)/n)*(sumysq - pow(sumy,2)/n))**.5
        #若down为零则不能计算，return 0
        if down == 0 :return 0
        r = up/down
        return r

if __name__ == '__main__':
    # v1=(0.44,0.05,0.196,0.196,0.158)
    # v2=(0.44,0,0.196,0.196,0)
    v1 = [1,3,2,2,4,3]
    v2 = [1,3,2,2,3,3]
    r = SimCal()
    print "余弦相识度=", r.cos_v(v1,v2)
    print "皮尔逊相似度=",r.pearson(v1,v2)
