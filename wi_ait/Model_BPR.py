#! /usr/bin/env python
# coding: UTF-8
import random
import numpy
from utils import load_L


#######################################################################################33
class BPR_model:
    def __init__(self, K, user_set, item_set):
        self.Z = K
        self.user_set = user_set
        self.item_set = item_set
        self.uLatent = numpy.matrix([[random.random()  for col in range(self.Z)] for row in range(len(user_set))])
        self.vLatent = numpy.matrix([[random.random()  for col in range(self.Z)] for row in range(len(item_set))])
    
    
    def saveRes(self, e_dict, data, des):
        res = open(des, 'w')
        for e in e_dict:
            tmp = e + '\t'
            indx = e_dict[e]
            for z in range(self.Z):
                tmp += str(data[indx, z]) + ","
            res.write(tmp[0:-1] + '\n')
        res.close();
        print 'finish saving ' + des;
        
    
    def saveModel(self, output, symbol):
        self.saveRes(self.user_set, self.uLatent, output + '/userLatent_' + symbol)
        self.saveRes(self.item_set, self.vLatent, output + '/itemLatent_' + symbol)
        
    
    def loadModel(self, des, symbol):
        uLatent = load_L(des + '/userLatent_' + symbol);
        for u in uLatent:
            if(self.user_set.has_key(u) == False):
                continue
            for z in range(self.Z):
                self.uLatent[self.user_set[u], z] = uLatent[u][z]     
        
        vLatent = load_L(des + '/itemLatent_' + symbol);
        for v in vLatent:
            if(self.item_set.has_key(v) == False):
                continue
            for z in range(self.Z):
                self.vLatent[self.item_set[v], z] = vLatent[v][z]
    
