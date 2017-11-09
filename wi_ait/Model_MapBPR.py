#! /usr/bin/env python
# coding: UTF-8
import random
import numpy
from numpy import linalg as LA
from Model_BPR import BPR_model
from Model_WI_CABpr import CA_BPR_Model
from Model_WI_CABpr import load2D



class MAP_BPR_Model(BPR_model, CA_BPR_Model):
    def __init__(self, K, user_set, item_set, vf_num):
        BPR_model.__init__(self, K, user_set, item_set)
        self.itemMatrix = None
        r0 = 6.0 / numpy.sqrt(self.Z)
        r2 = -6.0 / numpy.sqrt(self.Z)
        
        if(vf_num > 0):
            self.itemMatrix = numpy.matrix([[random.uniform(r2, r0)  for col in range(self.Z)] for row in range(vf_num)])
            for rw in range(self.Z):
                s2 = LA.norm(self.itemMatrix[:, rw], 1)
                for c in range(vf_num):
                    self.itemMatrix[c, rw] /= s2  


    def saveModel(self, output, symbol):
        BPR_model.saveModel(self, output, symbol)
        self.save2D(numpy.squeeze(numpy.asarray(self.itemMatrix)), output + '/itemMat_' + symbol)
        
   
    def loadMats(self, des, symbol):
        if(self.itemMatrix != None):
            self.itemMatrix = numpy.matrix(load2D(des + '/itemMat_' + symbol, delimiter=' '))
    
    
    def loadModel(self, des, symbol):
        BPR_model.loadModel(self, des, symbol)
        self.loadMats(des, symbol)
