#! /usr/bin/env python
# coding: UTF-8
import random
import numpy
from numpy import linalg as LA
from Model_BPR import BPR_model
from scipy.sparse import csr_matrix


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


#######################################################
class CA_BPR_Model(BPR_model):
    def __init__(self, K, user_set, item_set, uf_num, vf_num):
        BPR_model.__init__(self, K, user_set, item_set)
        self.user_num = len(user_set)
        self.item_num = len(item_set)
        
        r0 = 6.0 / numpy.sqrt(self.Z)
        r2 = -6.0 / numpy.sqrt(self.Z)
        # initialize the latent vectors
        self.uLatent = numpy.matrix([[random.uniform(r2, r0)  for col in range(self.Z)] for row in range(self.user_num)])
        for rw in range(self.user_num):
            s = LA.norm(self.uLatent[rw, :], 1)
            for z in range(self.Z):
                self.uLatent[rw, z] /= s 
        
        self.vLatent = numpy.matrix([[random.uniform(r2, r0)  for col in range(self.Z)] for row in range(self.item_num)])
        for rw in range(self.item_num):
            s = LA.norm(self.vLatent[rw, :], 1)
            for z in range(self.Z):
                self.vLatent[rw, z] /= s
                
        
        self.userMatrix = None
        self.itemMatrix = None
        
        if(uf_num > 0):       
            self.userMatrix = numpy.matrix([[random.uniform(r2, r0)  for col in range(self.Z)] for row in range(uf_num)])
            for rw in range(self.Z):
                s = LA.norm(self.userMatrix[:, rw], 1)
                for c in range(uf_num):
                    self.userMatrix[c, rw] /= s
            self.userMatrix = csr_matrix(self.userMatrix)
        
        if(vf_num > 0):
            self.itemMatrix = numpy.matrix([[random.uniform(r2, r0)  for col in range(self.Z)] for row in range(vf_num)])
            for rw in range(self.Z):
                s2 = LA.norm(self.itemMatrix[:, rw], 1)
                for c in range(vf_num):
                    self.itemMatrix[c, rw] /= s2  
            self.itemMatrix = csr_matrix(self.itemMatrix)
                
    
    def saveModel(self, output, symbol):
        BPR_model.saveModel(self, output, symbol)
        self.save2D(self.userMatrix.toarray(), output + '/userMat_' + symbol)
        self.save2D(self.itemMatrix.toarray(), output + '/itemMat_' + symbol)
    
    
    def save2D(self, data, des):
        if(data == None):
            return;
        res = open(des, 'w')
        for r in range(len(data)):
            tmp = ''
            for c in range(len(data[r])):
                tmp += str(data[r][c]) + ' '
            res.write(tmp[0:-1] + '\n')
        res.close();
        print 'finish saving ' + des;  
        
    
    def loadMats(self, des, symbol):
        if(self.userMatrix != None):
            self.userMatrix = csr_matrix(numpy.matrix(load2D(des + '/userMat_' + symbol, delimiter=' ')))
        if(self.itemMatrix != None):
            self.itemMatrix = csr_matrix(numpy.matrix(load2D(des + '/itemMat_' + symbol, delimiter=' ')))
    
    
    def loadModel(self, des, symbol):
        BPR_model.loadModel(des, symbol)
        self.loadMats(des, symbol)
              
