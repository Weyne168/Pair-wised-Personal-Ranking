#! /usr/bin/env python
# coding: UTF-8
import string
from sampler import sampler
from utils import log
from data import data
from Model_BPR import BPR_model
from trBPR import BPRSvd
from infer import Tmeasure 


class Rendel_BPRSvd(BPRSvd):
    def __init__(self, debug=False):
        BPRSvd.__init__(self, debug)
             
    def prepare(self):
        dataset = data()
        self.graph = dataset.loadGraph(self.params['graph'], string.atoi(self.params['dim']))
        self.drawer = sampler(self.graph, self.Z, lambda_setting=string.atoi(self.params['seed_num']))
        self.model = BPR_model(self.Z, self.drawer.user_set, self.drawer.item_set)
    
    
    def train(self, itera=1):
        while itera < self.iterTime :
            u_cat_vars, item_sorts = self.drawer.preDraw(self.model.uLatent, self.model.vLatent)
            samples = self.drawer.drawRendle(self.model.uLatent, u_cat_vars, item_sorts)
            self.training(samples)
            
            if(itera % self.saveIter == 0):
                self.model.saveModel(self.params['output'], str(itera))
                if(self.isConv(samples)):
                    break 
                log(self.params['output'] + '/log/logs.txt', str(itera) + '--' + str(self.obj_val))
            
            if(itera % self.iterTest == 0 and self.params.has_key('test')):
                self.model.saveModel(self.params['output'], '0')
                info = Tmeasure(self.model.Z, self.params['test'], 3, self.params['output'], str(0))
                log(self.params['output'] + '/log/log2.txt', str(itera) + '--' + info)
                #self.isConv(samples)
                #log(self.params['output'] + '/log/logs2.txt', str(itera) + '--' + str(self.obj_val))
            
            itera += 1 
        
        self.model.saveModel(self.params['output'], 'final')
        log(self.params['output'] + '/log/logs.txt', str(itera) + '--' + str(self.obj_val))
        print "training is done !";
              


if __name__ == "__main__":  
    model_symbol = '0'
    itera = 0
    bpr = Rendel_BPRSvd()
    bpr.prepare()
    if(bpr.params.has_key('m')):
        model_symbol = bpr.params['m']
    if(bpr.params.has_key('cit')):
        itera = string.atoi(bpr.params['cit'])
    if(bpr.params.has_key('m')):
        bpr.reloadModel(model_symbol, bpr.iterTime)
    print 'start to train...'
    bpr.train(itera=0)
