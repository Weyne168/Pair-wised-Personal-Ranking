#! /usr/bin/env python
# coding: UTF-8
import numpy
import utils
import string
from infer import Tmeasure 
from trPFM import trFM
from time import clock 


########################################################################
class rank_FM_Bpr_Model(trFM):
    updIter = 1
    iterTest = 1
    
    def __init__(self, debug=False):
        trFM.__init__(self, debug)
    
    
    def initParms(self): 
        trFM.initParms(self)  
        if(self.params.has_key('upd')):
            self.updIter = string.atoi(self.params['upd'])
        if(self.params.has_key('itt')):
            self.iterTest = string.atoi(self.params['itt'])
      
    
    def calPairLoss(self, u, vi, vj, model):
        vp = utils.fmObj(u, vi, model)
        vn = utils.fmObj(u, vj, model)
        x_uij = vp - vn
        return utils.sigmoid(x_uij);
    
    
    def dim_sub(self, dim, x1, x2):
        res = 0
        if(x1.indx.has_key(dim)):
            res = x1.values[x1.indx[dim]]
        if(x2.indx.has_key(dim)):
            res -= x2.values[x2.indx[dim]]
        return res;
            
        
    def calBPR_obj(self, u, vi, vj, model):
        return -1 * numpy.log(self.calPairLoss(self.userX[u], self.itemX[vi], self.itemX[vj], model));
    
   
    def checkBPR(self, model, param, u, vi, vj, i=-1, j=-1):
        p1, p2, mm1 = self.init_check(model, param, i, j)
        setattr(mm1, param, p1)
        r11 = self.calBPR_obj(u, vi, vj, mm1)
        setattr(mm1, param, p2)
        r12 = self.calBPR_obj(u, vi, vj, mm1)
        return (r11 - r12) / 0.00002;


    def training(self, samples):
        for s in samples:
            if(self.uIDx.has_key(s[0])==False or self.vIDx.has_key(s[1])==False or self.vIDx.has_key(s[2])==False):
                continue
            u = self.uIDx[s[0]]
            vi = self.vIDx[s[1]]
            vj = self.vIDx[s[2]]
           
            predict_p = utils.fmObj(self.userX[u], self.itemX[vi], self.model)
            predict_n = utils.fmObj(self.userX[u], self.itemX[vj], self.model)
            
            e = utils.sigmoid(predict_p - predict_n) - 1
            for i in range(self.model.vf_num):
                grav = self.dim_sub(i, self.itemX[vi], self.itemX[vj])
                if(grav != 0):
                    self.model.itemW[i] -= self.lrate * (e * grav + self.l2_W * self.model.itemW[i])
                 
            for z in range(0, self.model.Z):
                su = 0
                for f in self.userX[u].indx:
                    su += self.model.userV[f, z] * self.userX[u].values[self.userX[u].indx[f]]
                
                si = su
                for f in self.itemX[vi].indx:
                    si += self.model.itemV[f, z] * self.itemX[vi].values[self.itemX[vi].indx[f]]
 
                sj = su
                for f in self.itemX[vj].indx:
                    sj += self.model.itemV[f, z] * self.itemX[vj].values[self.itemX[vj].indx[f]]
                
                for i in range(self.model.vf_num):
                    grav = 0
                    if(self.itemX[vi].indx.has_key(i)):
                        grav = self.itemX[vi].values[self.itemX[vi].indx[i]] * si \
                        - self.itemX[vi].values[self.itemX[vi].indx[i]] * self.itemX[vi].values[self.itemX[vi].indx[i]] * self.model.itemV[i, z]
                    
                    if(self.itemX[vj].indx.has_key(i)): 
                        grav -= self.itemX[vj].values[self.itemX[vj].indx[i]] * sj \
                        - self.itemX[vj].values[self.itemX[vj].indx[i]] * self.itemX[vj].values[self.itemX[vj].indx[i]] * self.model.itemV[i, z]
                    if(grav != 0):
                        self.model.itemV[i, z] -= self.lrate * (e * grav + self.l2_V * self.model.itemV[i, z])    
                       
                for i in self.userX[u].indx: 
                    grav = self.userX[u].values[self.userX[u].indx[i]] * (si - sj)
                    self.model.userV[i, z] -= self.lrate * (e * grav + self.l2_V * self.model.userV[i, z])                 
           
        print 'finish a turn!'
    

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
            nll_val += self.calBPR_obj(u, vi, vj, self.model)
        return nll_val;
    
    
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
                utils.log(self.params['output'] + '/log/log2.txt', str(itera) + '----time:' +str(self.time_comsuming)+'\t' +info)
                # utils.log(self.params['output'] + '/log/logs2.txt', str(itera) + '----NLL:' + str(self.calNLL(samples)) + '\n')
            
            if(itera % self.saveIter == 0):
                #===============================================================
                # is_conv = self.isConv(samples)
                # utils.log(self.params['output'] + '/log/logs.txt', str(itera) + '----NLL:' + str(self.nloss) + '\n')
                # if(is_conv):
                #     break
                #===============================================================
                self.model.saveModel(self.params['output'], str(itera))
            
            itera += 1 
            print 'iteration:', itera 
        self.model.saveModel(self.params['output'], 'final')
        print "training is done !";
      

if __name__ == "__main__":  
    # ad_fm_bpr = ad_FM_Bpr_Model(True)
    ad_fm_bpr = rank_FM_Bpr_Model()
    ad_fm_bpr.initModel()
    # ad_fm_bpr.reloadModel(model_symbol='27', niter=-1)
    ad_fm_bpr.train(itera=0)
