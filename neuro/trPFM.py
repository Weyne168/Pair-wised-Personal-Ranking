#! /usr/bin/env python
# coding: UTF-8
import os
import string
import numpy
import sys
import utils
import copy
from sampler import sampler
from data import data
from Model_pFM import PFM_Model
from infer import Tmeasure 


########################################################################
class trFM:
    l2_0 = 0
    l2_W = 0
    l2_V = 0
    lrate = 0.001
    iterTime = 10
    saveIter = 2
    report = 1
    minConv = 0.00001
    
    
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
     
    
    def initParms(self):
        if(self.params.has_key('l2i')):
            vs = self.params['l2i'].split(',')
            self.l2_0 = float(vs[0])
            self.l2_W = float(vs[1])
            self.l2_V = float(vs[2])
        if(self.params.has_key('l2j')):
            vs = self.params['l2j'].split(',')
            self.l2_0j = float(vs[0])
            self.l2_Wj = float(vs[1])
            self.l2_Vj = float(vs[2])
        if(self.params.has_key('lr')):
            self.lrate = float(self.params['lr'])
        if(self.params.has_key('itn')):
            self.iterTime = string.atoi(self.params['itn'])
        if(self.params.has_key('its')):
            self.saveIter = string.atoi(self.params['its'])
        if(self.params.has_key('report')):
            self.report = string.atoi(self.params['report'])
        if(self.params.has_key('minC')):
            self.minConv = float(self.params['minC'])
        if(self.params.has_key('task')):
            self.task = self.params['task']
        if(self.params.has_key('itt')):
            self.iterTest = string.atoi(self.params['itt'])
        
    
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
       
    
    def prepare(self):
        dataset = data()
        self.userX, uf_num, self.uIDx = dataset.loadFeature(self.params['userX'])
        self.itemX, vf_num, self.vIDx = dataset.loadFeature(self.params['itemX'])
        self.graph = dataset.loadGraph(self.params['graph'], string.atoi(self.params['dim']))
        self.drawer = sampler(self.graph, self.vIDx, vf_num, self.uIDx, 100, self.userX, self.itemX)
        self.model = PFM_Model(string.atoi(self.params['k']), uf_num, vf_num)
        self.len_ux = len(self.userX)
        self.len_vx = len(self.itemX)
      
   
    def initModel(self):
        self.prepare()
        self.model.initModel(self.params['sigma'])
        
    
    def reloadModel(self, model_symbol='final', niter=-1):
        self.prepare()
        self.model.loadModel(self.params['output'], model_symbol)
        if(niter > 0):
            self.iterTime = niter
    
    
    def calFM_obj(self, u, v, y, model):
        predict = utils.fmObj(self.userX[u], self.itemX[v], model)
        if(self.task == 'class'):
            x = y * predict
            e = -1 * numpy.log(utils.sigmoid(x))
        else:
            e = (predict - y) * (predict - y)
        return e       
            
    
    def trainFM(self, u, v, y, r=None):
        if(r == None):
            predict = utils.fmObj(self.userX[u], self.itemX[v], self.model)
        else:
            predict = r
        
        if(self.task == 'class'):
            x = y * predict
            e = y * (utils.sigmoid(x) - 1)
        else:
            e = predict - y 
           
        if(self.debug):
            print e, self.checkFM(self.model, 'W0', u, v, y, i=-1, j=-1)
        self.model.W0 -= self.lrate * (e + self.l2_0 * self.model.W0)
        
        for f in self.userX[u].indx: 
            if(self.debug):
                print e * self.userX[u].values[self.userX[u].indx[f]], self.checkFM(self.model, 'userW', u, v, y, f)
            self.model.userW[f] -= self.lrate * (e * self.userX[u].values[self.userX[u].indx[f]] + self.l2_W * self.model.userW[f])

        for f in self.itemX[v].indx: 
            if(self.debug):
                print e * self.itemX[v].values[self.itemX[v].indx[f]], self.checkFM(self.model, 'itemW', u, v, y, f)
            self.model.itemW[f] -= self.lrate * (e * self.itemX[v].values[self.itemX[v].indx[f]] + self.l2_W * self.model.itemW[f])
      
        for z in range(0, self.model.Z):
            s = 0
           
            for f in self.userX[u].indx:
                s += self.model.userV[f, z] * self.userX[u].values[self.userX[u].indx[f]]
              
            for f in self.itemX[v].indx:
                s += self.model.itemV[f, z] * self.itemX[v].values[self.itemX[v].indx[f]]
            
            for f in self.userX[u].indx:
                grav = self.userX[u].values[self.userX[u].indx[f]] * s \
                    - self.userX[u].values[self.userX[u].indx[f]] * self.userX[u].values[self.userX[u].indx[f]] * self.model.userV[f, z]
                    
                if(self.debug):
                    print e * grav, self.checkFM(self.model, 'userV', u, v, y, f, z)
                self.model.userV[f, z] -= self.lrate * (e * grav + self.l2_V * self.model.userV[f, z])  
            
            for f in self.itemX[v].indx:
                grav = self.itemX[v].values[self.itemX[v].indx[f]] * s \
                    - self.itemX[v].values[self.itemX[v].indx[f]] * self.itemX[v].values[self.itemX[v].indx[f]] * self.model.itemV[f, z]
                    
                if(self.debug):
                    print e * grav, self.checkFM(self.model, 'itemV', u, v, y, f, z)
                self.model.itemV[f, z] -= self.lrate * (e * grav + self.l2_V * self.model.itemV[f, z])    
     
     
    def training(self, samples):
        count = 0
        for s in samples:
            u = self.drawer.user_set[s[0]]
            vi = self.drawer.item_set[s[1]]
            vj = self.drawer.item_set[s[2]]
            
            if(u < self.len_ux and vi < self.len_vx):
                self.trainFM(u, vi, 1)
            if(u < self.len_ux and vj < self.len_vx):
                self.trainFM(u, vj, -1) 
            
            count += 1
            if(count % self.report == 0):
                print 'training...', count, 'samples have been trained!'
        print 'finish a turn!' 
            
    
    
    def train(self, itera=1):
        self.nloss = 0
        while itera < self.iterTime :
            samples = self.drawer.drawRandomly()
            # samples = self.drawer.drawAdv(self.model)
            if(itera % self.iterTest == 0 and self.params.has_key('test_set')):
                info = Tmeasure(self.model, self.params['test_set'], self.params['test_pairs'], \
                                 3, self.params['test_userX'], self.params['test_itemX'], string.atoi(self.params['typ']))
                utils.log(self.params['output'] + '/log/log2.txt', str(itera) + '----' + info)
                # self.nloss = self.calNLL(samples)
                # utils.log(self.params['output'] + '/log/logs2.txt', str(itera) + '----NLL:' + str(self.nloss) + '\n')
            
            self.training(samples)
            
            if(itera % self.saveIter == 0):
                #===============================================================
                # is_conv = self.isConv(samples)
                # if(is_conv):
                #     break
                #===============================================================
                self.model.saveModel(self.params['output'], str(itera))
                # utils.log(self.params['output'] + '/log/logs.txt', str(itera) + '----NLL:' + str(self.nloss) + '\n')
            itera += 1
            
        self.model.saveModel(self.params['output'], 'final')
        print "training is done !";
    
    
    def init_check(self, model, param, i=-1, j=-1):
        mm1 = copy.deepcopy(model)
        p1 = getattr(mm1, param)
        p2 = copy.deepcopy(p1)
        if(i >= 0 and j >= 0 and i < len(p1)):
            p1[i][j] += 0.00001
            p2[i][j] -= 0.00001
        elif(i >= 0 and j == -1 and i < len(p1)):
            p1[i] += 0.00001
            p2[i] -= 0.00001
        else:
            p1 += 0.00001
            p2 -= 0.00001
        return p1, p2, mm1;
    
    
    def checkFM(self, model, param, u, v, y, i=-1, j=-1):
        p1, p2, mm1 = self.init_check(model, param, i, j)  
        setattr(mm1, param, p1)
        r11 = self.calFM_obj(u, v, y, mm1)
        setattr(mm1, param, p2)
        r12 = self.calFM_obj(u, v, y, mm1)
        return (r11 - r12) / 0.00002;
    
    
    def calNLL(self, samples):
        nll_val = 0
        count = 1
        print len(samples)
        for s in samples:
            if(count % self.report == 0):
                print count, 'samples have been calculated!'
            count += 1 
            u = self.uIDx[s[0]]
            vi = self.vIDx[s[1]]
            vj = self.vIDx[s[2]]
          
            nll_val += self.calFM_obj(u, vi, 1, self.model)
            nll_val += self.calFM_obj(u, vj, -1, self.model)
        return nll_val;
    
    
    def isConv(self, samples):
        res = self.calNLL(samples)
        delt = numpy.abs(res - self.nloss)
        self.nloss = res
        if (delt < self.minConv):
            return True
        else:
            return False;
        
    
if __name__ == "__main__":  
    # pfm = trFM_Bpr(True)
    pfm = trFM()
    pfm.initModel()
    model_symbol = '0'
    niter = -1
    itera = 0
    if(pfm.params.has_key('m')):
        model_symbol = pfm.params['m']
    if(pfm.params.has_key('rn')):
        niter = string.atoi(pfm.params['rn'])
    if(pfm.params.has_key('bn')):
        itera = string.atoi(pfm.params['bn'])
    if(pfm.params.has_key('m')):
        pfm.reloadModel(model_symbol, niter)
    pfm.train(itera)        
