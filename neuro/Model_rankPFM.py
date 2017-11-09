#! /usr/bin/env python
# coding: UTF-8
from Model_pFM import PFM_Model
from Model_BPR import BPR_model



def save1D(data, des):
    res = open(des, 'w')
    tmp = ''
    for e in range(len(data)):
        tmp += str(data[e]) + ' '
    res.write(tmp[0:-1] + '\n')
    res.close();
    print 'finish saving ' + des;
        
    
def save2D(data, des):
    res = open(des, 'w')
    row, col = data.shape
    for r in range(row):
        tmp = ''
        for c in range(col):
            tmp += str(data[r, c]) + ' '
        res.write(tmp[0:-1] + '\n')
    res.close();
    print 'finish saving ' + des;      


def load2D(srcFile, delimiter=' '):
    fin = open(srcFile, 'r')
    res = []
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            vs = line.strip().split(delimiter)
            rs = []
            for v in vs:
                rs.append(float(v))
            res.append(rs)
    fin.close()
    return res;


def load1D(srcFile, delimiter=' '):
    fin = open(srcFile, 'r')
    res = []
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            vs = line.strip().split(delimiter)
            for v in vs:
                res.append(float(v))
    fin.close()
    return res;


###################################################################3
class RankPairFM_Model(PFM_Model, BPR_model):
    def __init__(self, K, uf_num, vf_num, user_set, item_set):
        PFM_Model.__init__(self, K, uf_num, vf_num)
        BPR_model.__init__(self, K, user_set, item_set)
        
    
    def initModel(self, sigma=1): 
        PFM_Model.initModel(self, sigma)
        
        
    
    def saveModel(self, model_dir, symbol):
        PFM_Model.saveModel(self, model_dir, symbol)
        BPR_model.saveModel(self, model_dir, symbol)
    
    
    def loadModel(self, model_dir, symbol):
        PFM_Model.loadModel(self, model_dir, symbol)
        BPR_model.loadModel(self, model_dir, symbol)
        
    
    
