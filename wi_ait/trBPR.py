#! /usr/bin/env python
# coding: UTF-8
import sys
import os
import string
import numpy
from sampler import sampler
from utils import log
from utils import sigmoid
from data import data
from Model_BPR import BPR_model
from infer import Tmeasure 
from time import clock

#####################################################################################################
class BPRSvd:
    Z = 0
    lambdaU = 1
    lambdaVi = 1
    lambdaVj = 1
    lrate = 0.001  # learning rates of user's projection matrixes
    iterTime = 100
    minConv = 0.00001
    saveIter = 1
    obj_val = 0
    
    
    def __init__(self, debug=False):
        self.params = {}
        self.params['output'] = sys.path[0]
        for param in sys.argv[1:]:
            k, v = param.split('=')
            self.params[k] = v
        
        if(self.params.has_key('config')):
            self.config(self.params['config'])
        
        for param in sys.argv[1:]:
            k, v = param.split('=')
            self.params[k] = v
        
        self.initParms()
        self.debug = debug
    
    
    def config(self, confFile):
        if(os.path.exists(confFile) == 0):
            print 'There is not configuration file!'
            exit(0)     
        conf = open(confFile, 'r')
        while 1:
            line = conf.readline()
            if not line:
                break
            line = line.strip()
            if(len(line) < 1):
                continue
            k, v = line.split('=')
            self.params[k] = v
        conf.close();  
        
    
    def initParms(self):            
        if(self.params.has_key('k')):
            self.Z = string.atoi(self.params['k'])
        if(self.params.has_key('lr')):
            self.lrate = float(self.params['lr'])
        if(self.params.has_key('itn')):
            self.iterTime = string.atoi(self.params['itn'])
        if(self.params.has_key('its')):
            self.saveIter = string.atoi(self.params['its'])
        if(self.params.has_key('report')):
            self.report = string.atoi(self.params['report'])
        if(self.params.has_key('l2')):
            vs = self.params['l2'].split(',')
            self.lambdaU = float(vs[0])
            self.lambdaVi = float(vs[1])
            self.lambdaVj = float(vs[2])
        if(self.params.has_key('minC')):  
            self.minConv = float(self.params['minC']) 
        if(self.params.has_key('itt')):
            self.iterTest = string.atoi(self.params['itt'])    
      
    
    def prepare(self):
        dataset = data()
        self.graph = dataset.loadGraph(self.params['graph'], string.atoi(self.params['dim']))
        self.drawer = sampler(self.graph, self.Z)
        self.model = BPR_model(self.Z, self.drawer.user_set, self.drawer.item_set)
        
          
    def reloadModel(self, model_symbol='final', niter=-1):
        self.model.loadModel(self.params['output'],model_symbol)
        if(niter > 0):
            self.iterTime = niter
    
    
    def calObj(self, u, i, j):
        return -1 * numpy.log(self.calPairLoss(u, i, j));
    
    
    def calPairLoss(self, u, i, j):
        uvec = numpy.squeeze(numpy.asarray(self.model.uLatent[u, :]))
        ivec = numpy.squeeze(numpy.asarray(self.model.vLatent[i, :]))
        jvec = numpy.squeeze(numpy.asarray(self.model.vLatent[j, :]))
        x1 = numpy.dot(uvec, ivec)
        x2 = numpy.dot(uvec, jvec)
        return sigmoid(x1 - x2)
    
    
    def training(self, samples):
        
        for s in samples:
            u = self.model.user_set[s[0]]
            vi = self.model.item_set[s[1]]
            vj = self.model.item_set[s[2]]
            f = self.calPairLoss(u, vi, vj) - 1 
            for z in range(0, self.Z):
                self.model.uLatent[u, z] -= self.lrate * (f * (self.model.vLatent[vi, z] - self.model.vLatent[vj, z]) + self.lambdaU * self.model.uLatent[u, z])
                self.model.vLatent[vi, z] -= self.lrate * (f * self.model.uLatent[u, z] + self.lambdaVi * self.model.vLatent[vi, z])
                self.model.vLatent[vj, z] -= self.lrate * (-1.0 * f * self.model.uLatent[u, z] + self.lambdaVj * self.model.vLatent[vj, z])
        print 'finish a turn!'   
          
     
    def calNLL(self, samples):
        nll_val = 0
        for s in samples:
            u = self.model.user_set[s[0]]
            vi = self.model.item_set[s[1]]
            vj = self.model.item_set[s[2]]
            nll_val += self.calObj(u, vi, vj)
        return nll_val;
    
    
    def isConv(self, samples):
        res = self.calNLL(samples)
        if (numpy.abs(res - self.obj_val) < self.minConv):
            self.obj_val = res
            return True
        else:
            self.obj_val = res
            return False;
            
    
    def train(self, itera=1):
        self.time_comsuming = 0
        while itera < self.iterTime :
            t0 = clock()
            samples = self.drawer.drawRandomly()
            self.training(samples)
            t1 = clock()
            self.time_comsuming += t1 - t0
          
            if(itera % self.saveIter == 0):
                self.model.saveModel(self.params['output'], str(itera))
                #===============================================================
                # if(self.isConv(samples)):
                #     break 
                # log(self.params['output'] + '/log/logs.txt', str(itera) + '--' + str(self.obj_val))
                #===============================================================
                
            if(itera % self.iterTest == 0 and self.params.has_key('test_set')):
                info = Tmeasure(self.model, self.params['test_set'], self.params['test_pairs'] , 3, 'bpr')
                log(self.params['output'] + '/log/log2.txt', str(itera) + '----time:' +str(self.time_comsuming)+'\t' + info)
                self.isConv(samples)
                # log(self.params['output'] + '/log/logs2.txt', str(itera) + '--' + str(self.obj_val))
           
            itera += 1 
        
        self.model.saveModel(self.params['output'], 'final')
        log(self.params['output'] + '/log/logs.txt', str(itera) + '--' + str(self.obj_val))
        print "training is done !";
              


if __name__ == "__main__":  
    model_symbol = '0'
    itera = 0
    bpr = BPRSvd()
    bpr.prepare()
    
    if(bpr.params.has_key('m')):
        model_symbol = bpr.params['m']
    if(bpr.params.has_key('cit')):
        itera = string.atoi(bpr.params['cit'])
    if(bpr.params.has_key('m')):
        bpr.reloadModel(model_symbol, bpr.iterTime)
    print 'start to train...'
    bpr.train(itera)
