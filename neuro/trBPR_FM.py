#! /usr/bin/env python
# coding: UTF-8
import numpy
import utils
import string
from infer import Tmeasure 
from trPFM import trFM
from sampler import sampler
from data import data
from Model_rankPFM import RankPairFM_Model
from time import clock


########################################################################
class tr_BprFM(trFM):
    updIter = 1
    iterTest = 1
    
    def __init__(self, debug=False):
        trFM.__init__(self, debug)
    
    
    def prepare(self):
        dataset = data()
        self.userX, uf_num, self.uIDx = dataset.loadFeature(self.params['userX'])
        self.itemX, vf_num, self.vIDx = dataset.loadFeature(self.params['itemX'])
        self.graph = dataset.loadGraph(self.params['graph'], string.atoi(self.params['dim']))
        self.drawer = sampler(self.graph, self.vIDx, vf_num, self.uIDx, float(self.params['sigma']), self.userX, self.itemX)
        self.model = RankPairFM_Model(string.atoi(self.params['k']), uf_num, vf_num, self.drawer.user_set, self.drawer.item_set)
        self.len_ux = len(self.userX)
        self.len_vx = len(self.itemX)
    
    
    def initParms(self): 
        trFM.initParms(self)  
        if(self.params.has_key('upd')):
            self.updIter = string.atoi(self.params['upd'])
        if(self.params.has_key('itt')):
            self.iterTest = string.atoi(self.params['itt'])
        if(self.params.has_key('l2j')):
            vs = self.params['l2j'].split(',')
            self.lambdaU = float(vs[0])
            self.lambdaVi = float(vs[1])
            self.lambdaVj = float(vs[2])
            self.lambdaTp = float(vs[1])
            self.lambdaTn = float(vs[0])
            self.lambdaTp = 0
            self.lambdaTn = 0
            if(len(vs) > 3):
                self.lambdaTp = float(vs[3])
                self.lambdaTn = float(vs[4])
        if(self.params.has_key('lr0')):
            self.lrate0 = float(self.params['lr0'])
      
        
    def calBPR_obj(self, u, vi, vj, model):
        uvec = numpy.squeeze(numpy.asarray(self.model.uLatent[u, :]))
        ivec = numpy.squeeze(numpy.asarray(self.model.vLatent[vi, :]))
        jvec = numpy.squeeze(numpy.asarray(self.model.vLatent[vj, :]))
        bpr_predict_p = numpy.dot(uvec, ivec)
        bpr_predict_n = numpy.dot(uvec, jvec)
        return -1 * numpy.log(utils.sigmoid(bpr_predict_p - bpr_predict_n));
    
    
    def training(self, samples):
     
        for s in samples:
            u = self.drawer.user_set[s[0]]
            vi = self.drawer.item_set[s[1]]
            vj = self.drawer.item_set[s[2]]
            
            uvec = numpy.squeeze(numpy.asarray(self.model.uLatent[u, :]))
            ivec = numpy.squeeze(numpy.asarray(self.model.vLatent[vi, :]))
            jvec = numpy.squeeze(numpy.asarray(self.model.vLatent[vj, :]))
            bpr_predict_p = numpy.dot(uvec, ivec)
            bpr_predict_n = numpy.dot(uvec, jvec)
            
            
            if(u < self.len_ux and vi < self.len_vx):
                fm_predict_p = utils.fmObj(self.userX[u], self.itemX[vi], self.model)
                self.trainFM(u, vi, bpr_predict_p, fm_predict_p)
                
            else:
                fm_predict_p = bpr_predict_p
            
            if(u < self.len_ux and vj < self.len_vx):
                fm_predict_n = utils.fmObj(self.userX[u], self.itemX[vj], self.model)
                self.trainFM(u, vj, bpr_predict_n, fm_predict_n)
                
            else:
                fm_predict_n = bpr_predict_n
            
            err_p = bpr_predict_p - fm_predict_p
            err_n = bpr_predict_n - fm_predict_n
            
            f = utils.sigmoid(bpr_predict_p - bpr_predict_n) - 1
            
            fmw = fm_predict_p - fm_predict_n
            if(fmw > 0):
                f *= (1 + utils.sigmoid(fmw))
            else:
                f *= (1 - utils.sigmoid(fmw))
                
            for z in range(0, self.model.Z):
                self.model.uLatent[u, z] -= self.lrate0 * (f * (self.model.vLatent[vi, z] - self.model.vLatent[vj, z]) \
                                                          + self.lambdaU * self.model.uLatent[u, z]\
                                                          + self.lambdaTp * err_p * self.model.vLatent[vi, z]\
                                                          + self.lambdaTn * err_n * self.model.vLatent[vj, z])
                self.model.vLatent[vi, z] -= self.lrate0 * (f * self.model.uLatent[u, z] \
                                                           + self.lambdaVi * self.model.vLatent[vi, z]\
                                                           + self.lambdaTp * err_p * self.model.uLatent[u, z])
                self.model.vLatent[vj, z] -= self.lrate0 * (-1.0 * f * self.model.uLatent[u, z]\
                                                             + self.lambdaVj * self.model.vLatent[vj, z]\
                                                            + self.lambdaTn * err_n * self.model.uLatent[u, z])  
           
        print 'finish a turn!'
    

    def calNLL(self, samples):
        nll_val2 = 0
        count = 1
        print len(samples)
        for s in samples:
            if(count % self.report == 0):
                print count, 'samples have been calculated!'
            count += 1 
            u = self.uIDx[s[0]]
            vi = self.vIDx[s[1]]
            vj = self.vIDx[s[2]]
            nll_val2 += self.calBPR_obj(u, vi, vj, self.model)
            # nll_val += self.calFM_obj(u, vi, 1, self.model)
            # nll_val += self.calFM_obj(u, vj, -1, self.model)
        return nll_val2  # , nll_val + nll_val2
    
    
    def isConv(self, samples):
        res = self.calNLL(samples)
        delt = numpy.abs(res - self.nloss)
        self.nloss = res
        if (delt < self.minConv):
            return True
        else:
            return False;
            
    
    def train(self, itera=1):
        self.nloss = 0
        self.time_comsuming = 0
        while itera < self.iterTime :
            # samples = self.drawer.drawRandomly()
            t0 = clock()
            if(itera % self.updIter == 0):
                upd = True
            else:
                upd = False
            
            if(self.params.has_key('sampler')):
                samples = self.drawer.drawAdv(self.model, upd)
            else:
                samples = self.drawer.drawRandomly()
                # samples = self.drawer.drawAdv(self.model, upd)
            self.training(samples)
            self.time_comsuming += clock() - t0
            
            if(itera % self.iterTest == 0 and self.params.has_key('test_set')):
                info = Tmeasure(self.model, self.params['test_set'], self.params['test_pairs'], \
                                 3, self.params['test_userX'], self.params['test_itemX'], string.atoi(self.params['typ']))
                utils.log(self.params['output'] + '/log/log2.txt', str(itera) + '----time:' + str(self.time_comsuming) + '\t' + info + '\n')
                # utils.log(self.params['output'] + '/log/logs2.txt', str(itera) + '----NLL:' + str(self.calNLL(samples)) + '\n')            
            
            if(itera % self.saveIter == 0):
                # is_conv = self.isConv(samples)
                # utils.log(self.params['output'] + '/log/logs.txt', str(itera) + '----NLL:' + str(self.nloss) + '\n')
                # if(is_conv):
                #    break
                self.model.saveModel(self.params['output'], str(itera))
          
            itera += 1  
            print 'iteration:', itera
        self.model.saveModel(self.params['output'], 'final')
        print "training is done !";
      

if __name__ == "__main__":  
    # ad_fm_bpr = ad_FM_Bpr_Model(True)
    ad_fm_bpr = tr_BprFM()
    ad_fm_bpr.initModel()
    ad_fm_bpr.params['output'] = 'I:/dataset/ml-100k/cleanedData/res/our/v10'
    ad_fm_bpr.params['sampler'] = 'ada'
    itera = 0
    model_symbol = '0'
    if(ad_fm_bpr.params.has_key('symbol')):
        if(ad_fm_bpr.params.has_key('cit')):
            itera = string.atoi(ad_fm_bpr.params['cit'])
        ad_fm_bpr.reloadModel(ad_fm_bpr.params['symbol'], ad_fm_bpr.iterTime)
    ad_fm_bpr.train(itera)
