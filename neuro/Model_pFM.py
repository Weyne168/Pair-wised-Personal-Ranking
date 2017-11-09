#! /usr/bin/env python
# coding: UTF-8
import numpy
import copy


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
class PFM_Model:
    Z = 10
    W0 = 0
    
    def __init__(self, K, uf_num, vf_num):
        self.Z = K
        self.uf_num = uf_num 
        self.vf_num = vf_num
        
    
    def initModel(self, sigma=1): 
        self.userW = numpy.zeros(self.uf_num)
        self.userV = numpy.matrix(numpy.random.normal(0, sigma, (self.uf_num, self.Z))) 
        self.itemW = numpy.zeros(self.vf_num)
        self.itemV = numpy.matrix(numpy.random.normal(0, sigma, (self.vf_num, self.Z)))
        
    
    def setModel(self, W0, uW, vW, userV, itemV):
        self.W0 = W0
        self.userW = copy.deepcopy(uW)
        self.itemW = copy.deepcopy(vW)
        self.itemV = copy.deepcopy(userV)
        self.userV = copy.deepcopy(itemV)
        
    
    def saveModel(self, model_dir, symbol):
        W = []
        W.append(self.W0)
        W.extend(self.userW)
        save1D(W, model_dir + '/userW_' + symbol)
        save1D(self.itemW, model_dir + '/itemW_' + symbol)
        save2D(self.userV, model_dir + '/userV_' + symbol)
        save2D(self.itemV, model_dir + '/itemV_' + symbol)
    
    
    def loadModel(self, model_dir, symbol):
        tW = load1D(model_dir + '/userW_' + symbol, delimiter=' ')
        self.W0 = tW[0]
        self.userW = tW[1:]
        self.itemW = load1D(model_dir + '/itemW_' + symbol, delimiter=' ')
        self.itemV = numpy.matrix(load2D(model_dir + '/itemV_' + symbol, delimiter=' '))
        self.userV = numpy.matrix(load2D(model_dir + '/userV_' + symbol, delimiter=' '))
        
        
if __name__ == "__main__": 
    pass
    

