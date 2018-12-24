# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
This module provide function of LR，auc
Date:    2016/09/20 17:23:16

"""
import re
import os
import sys

import numpy as ny
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


G_AUC_THD_SEGMENT = 100

def bias(ori_file):
    m_r_file_obj = open(ori_file, "r")
    m_bias = 0
    try:
        m_bias = m_r_file_obj.readlines()[4].strip()
        m_bias = re.split(",", m_bias)[0]
        m_bias = re.split(" ", m_bias)[1]
    except:
        print "m_bias is null"
    return float(m_bias)

def sigmoid(inX):
    return 1.0/(1+ny.exp(-inX))

def weight(ori_file, dim_num):
    m_r_file_obj = open(ori_file, "r")
    weight_line_list = range(7, 7 + dim_num)
    weight_mat = []
    num = 0
    for line in m_r_file_obj.readlines():
        line = line.strip()
        num += 1
        if num in weight_line_list:
            weight = float(line)
            weight_mat.append(weight)
    return weight_mat

def loaddata(ori_file, dim_num, bias):
    """ deal with data"""
    m_r_file_obj = open(ori_file, "r")
    t_data_Mat = []
    line_label = []
    for line in m_r_file_obj.readlines():
        line = line.strip()
        if bias > 0:
            t_region_array = re.split("\t", line)
            t_region_label = int(t_region_array[0])
            t_region_array[0] = 1.0
            t_filLine = map(float, t_region_array[0 : dim_num+1])
            t_data_Mat.append(t_filLine)
            line_label.append(t_region_label)
        else:
            t_region_array = re.split("\t", line)
            t_region_label = int(t_region_array[0])
            t_filLine = map(float,t_region_array[1 : dim_num+1])
            t_data_Mat.append(t_filLine)
            line_label.append(t_region_label)
    return t_data_Mat,line_label

def predict_classify(inX, weights):
    m_predict_mat = []
    inX = ny.array(inX)
    for row in range(len(inX)):
        predict = sigmoid(inX[row]*weights)
        m_predict_mat.append(predict)  
    return m_predict_mat  

def cal_auc(t_true_y,predict_result):
    fpr, tpr, thresholds = roc_curve(t_true_y, predict_result, pos_label=1)
    auc_value = roc_auc_score(t_true_y, predict_result)
    return auc_value

def classify_label_by_thd(score_mat, thd):
    rlt_mat = []
    m_class_value = 0
    for idx in range(len(score_mat)):
        m_score = score_mat[idx]
        if m_score >= thd:
            m_class_value = 1
        else:
            m_class_value = 0
        rlt_mat.append(m_class_value)
    return rlt_mat

def check_auc_threshold(w_auc_file, auc_score_mat, label_mat, seg_num):
    if seg_num <= 0:
         return -1

    m_w_file_obj = open(w_auc_file, "w")
    m_min_score = 1.0
    m_max_score = 0.0
    m_fit_thd = 0.0
    m_fit_auc = 0.0
    for idx in range(len(auc_score_mat)):
        m_score = auc_score_mat[idx]
        if m_min_score > m_score:
            m_min_score = m_score
        if m_max_score < m_score:
            m_max_score = m_score
    m_step_width = (m_max_score - m_min_score) / seg_num
    
    for idx in range(1, seg_num):
        cur_auc_thd = m_min_score + idx * m_step_width
        m_predict_mat = classify_label_by_thd(auc_score_mat, cur_auc_thd)
        cur_auc_value = cal_auc(m_predict_mat, label_mat)
        w_str = "%s\t%f\t%f\n" % ("list", cur_auc_thd, cur_auc_value)
        m_w_file_obj.write(w_str)
        if cur_auc_value > m_fit_auc:
            m_fit_thd = cur_auc_thd
            m_fit_auc = cur_auc_value
    w_str = "%s\t%f\t%f\n" % ("fit", m_fit_thd, m_fit_auc)
    m_w_file_obj.write(w_str)
    if m_w_file_obj:
        m_w_file_obj.close()
    return 0
        
if __name__ == '__main__': 
    if len(sys.argv) < 5:
        print "Example:python func.py model_file test_file auc_rlt dim_num"
        sys.exit(-1)
    #get args & check valid
    m_r_model_file = sys.argv[1]
    m_r_predict_file = sys.argv[2]
    m_w_auc_file = sys.argv[3]
    g_feature_dim_num = int(sys.argv[4])
    if os.path.getsize(m_r_model_file) == 0:
        print "%s is null!" % (m_r_model_file)
        sys.exit(-2)
    if os.path.getsize(m_r_predict_file) == 0:
        print "%s is null!" % (m_r_predict_file)
        sys.exit(-3)
    
    #load data and weight
    m_bias = bias(m_r_model_file)
    t_data_mat, t_label_mat = loaddata(m_r_predict_file, g_feature_dim_num, m_bias)
    t_weight_mat = ny.mat(weight(m_r_model_file, g_feature_dim_num))
    if m_bias > 0:
        t_weights_mat = ny.hstack((ny.mat(m_bias),t_weight_mat))
    t_weight_mat = t_weight_mat.T
 
    #predict
    t_predict_mat = predict_classify(t_data_mat, t_weight_mat)
    
    #cal different AUC by thd segment
    check_auc_threshold(m_w_auc_file, t_predict_mat, t_label_mat, G_AUC_THD_SEGMENT)
    
    sys.exit(0)
