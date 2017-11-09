#! /usr/bin/env python
# coding: UTF-8
import string
from sampler import sampler
from utils import log
from trBPR import BPRSvd
from Model_MapBPR import MAP_BPR_Model
from infer import Tmeasure 
from data import data
from time import clock

class atrrBPR(BPRSvd):    
    
    def __init__(self, debug=False):
        BPRSvd.__init__(self, debug)
        
    
    def initParms(self):            
        BPRSvd.initParms(self)
        if(self.params.has_key('lrW')):
            self.lrate_W = float(self.params['lrW'])
        if(self.params.has_key('l2w')):
            self.lambdaIw = float(self.params['l2w'])
        
      
    def prepare(self):
        dataset = data()
        self.graph = dataset.loadGraph(self.params['graph'], string.atoi(self.params['dim']))
        self.drawer = sampler(self.graph, self.Z)
        itemX, vf_num, self.vIDx = dataset.loadFeature(self.params['itemX'])
        self.itemX = dataset.constructSparseMat(itemX, vf_num)
        self.model = MAP_BPR_Model(string.atoi(self.params['k']), self.drawer.user_set, self.drawer.item_set, vf_num)
    
    
    def reloadModel(self, model_symbol='final', niter=-1):
        self.model.loadModel(self.params['output'], model_symbol)
        if(niter > 0):
            self.iterTime = niter
    
    def training(self, samples):

        for s in samples: 
            u = self.model.user_set[s[0]]
            vi = self.model.item_set[s[1]]
            vj = self.model.item_set[s[2]]
            f = self.calPairLoss(u, vi, vj) - 1 
            for z in range(0, self.Z):
                if(self.vIDx.has_key(s[1]) and self.vIDx.has_key(s[2])):
                    vii = self.vIDx[s[1]]
                    vjj = self.vIDx[s[2]]
                    self.model.itemMatrix[:, z] -= self.lrate_W * (f * self.model.uLatent[u, z] * (self.itemX.getrow(vii) - self.itemX.getrow(vjj)).T \
                                                            + self.lambdaIw * self.model.itemMatrix[:, z])
                elif(self.vIDx.has_key(s[1])):
                    vii = self.vIDx[s[1]]
                    self.model.itemMatrix[:, z] -= self.lrate_W * (f * self.model.uLatent[u, z] * self.itemX.getrow(vii).T \
                                                            + self.lambdaIw * self.model.itemMatrix[:, z])
                elif(self.vIDx.has_key(s[2])):
                    vjj = self.vIDx[s[2]]
                    self.model.itemMatrix[:, z] -= self.lrate_W * (f * self.model.uLatent[u, z] * (-1.0 * self.itemX.getrow(vjj)).T \
                                                            + self.lambdaIw * self.model.itemMatrix[:, z])
                    
                self.model.uLatent[u, z] -= self.lrate * (f * (self.model.vLatent[vi, z] - self.model.vLatent[vj, z]) \
                                                            + self.lambdaU * self.model.uLatent[u, z])
                self.model.vLatent[vi, z] -= self.lrate * (f * self.model.uLatent[u, z] + self.lambdaVi * self.model.vLatent[vi, z])
                self.model.vLatent[vj, z] -= self.lrate * (-1.0 * f * self.model.uLatent[u, z] + self.lambdaVj * self.model.vLatent[vj, z])
            
        print 'finish a turn!'
        
    def train(self, itera=1):
        self.time_comsuming = 0
        while itera < self.iterTime :
            t0 = clock()
            samples = self.drawer.drawRandomly()
            self.training(samples)
            self.time_comsuming += clock() - t0
            if(itera % self.saveIter == 0):
                self.model.saveModel(self.params['output'], str(itera))
                #===============================================================
                # if(self.isConv(samples)):
                #     break 
                # log(self.params['output'] + '/log/logs.txt', str(itera) + '--' + str(self.obj_val))
                #===============================================================
                
            if(itera % self.iterTest == 0 and self.params.has_key('test_set')):
                info = Tmeasure(self.model, self.params['test_set'], self.params['test_pairs'] , 3, 'atbpr', None, self.params['itemX'])
                log(self.params['output'] + '/log/log2.txt', str(itera) + '----time:' +str(self.time_comsuming)+'\t'  + info)
                #===============================================================
                # self.isConv(samples)
                # log(self.params['output'] + '/log/logs2.txt', str(itera) + '--' + str(self.obj_val))
                #===============================================================   
           
            itera += 1 
        
        self.model.saveModel(self.params['output'], 'final')
        log(self.params['output'] + '/log/logs.txt', str(itera) + '--' + str(self.obj_val))
        print "training is done !";


if __name__ == "__main__":  
    model_symbol = '0'
    itera = 0
    abpr = atrrBPR()
    abpr.prepare()
    #abpr.params['output']='I:/dataset/ml-100k/cleanedData/res/atbpr/v10'
    
    if(abpr.params.has_key('m')):
        model_symbol = abpr.params['m']
    
    if(abpr.params.has_key('cit')):
        itera = string.atoi(abpr.params['cit'])
    if(abpr.params.has_key('m')):
        abpr.reloadModel(model_symbol, abpr.iterTime)
    print 'start to train...'
    abpr.train(itera)
