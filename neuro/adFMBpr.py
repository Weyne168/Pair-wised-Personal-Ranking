#! /usr/bin/env python
# coding: UTF-8
import numpy
import utils
import string
import copy
from infer import Tmeasure 
from trPFM import trFM
# from time import clock


########################################################################
class ad_FM_Bpr_Model(trFM):
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
    
    
    def sgn(self, x, y):  
        if(x >= y):
            return True
        return False
            
        
    def calBPR_obj(self, u, vi, vj, model):
        return -1 * numpy.log(self.calPairLoss(self.userX[u], self.itemX[vi], self.itemX[vj], model));
    
   
    def checkBPR(self, model, param, u, vi, vj, i=-1, j=-1):
        p1, p2, mm1 = self.init_check(model, param, i, j)
        setattr(mm1, param, p1)
        r11 = self.calBPR_obj(u, vi, vj, mm1)
        setattr(mm1, param, p2)
        r12 = self.calBPR_obj(u, vi, vj, mm1)
        return (r11 - r12) / 0.00002;

    
    def training0(self, samples):
        count = 0
        for s in samples:
            u = self.uIDx[s[0]]
            vi = self.vIDx[s[1]]
            vj = self.vIDx[s[2]]
            model = copy.deepcopy(self.model)
            predict_p = utils.fmObj(self.userX[u], self.itemX[vi], model)
            predict_n = utils.fmObj(self.userX[u], self.itemX[vj], model)
           
            e = utils.sigmoid(predict_p - predict_n) - 1
            for i in range(model.vf_num):
                grav = self.dim_sub(i, self.itemX[vi], self.itemX[vj])
                if(grav != 0):
                    model.itemW[i] -= self.lrate * (e * grav + self.l2_W * model.itemW[i])
                 
            for z in range(0, self.model.Z):
                su = 0
                for f in self.userX[u].indx:
                    su += model.userV[f, z] * self.userX[u].values[self.userX[u].indx[f]]
                
                si = su
                for f in self.itemX[vi].indx:
                    si += model.itemV[f, z] * self.itemX[vi].values[self.itemX[vi].indx[f]]
 
                sj = su
                for f in self.itemX[vj].indx:
                    sj += model.itemV[f, z] * self.itemX[vj].values[self.itemX[vj].indx[f]]
                
                for i in range(model.vf_num):
                    grav = 0
                    if(self.itemX[vi].indx.has_key(i)):
                        grav = self.itemX[vi].values[self.itemX[vi].indx[i]] * si \
                        - self.itemX[vi].values[self.itemX[vi].indx[i]] * self.itemX[vi].values[self.itemX[vi].indx[i]] * model.itemV[i, z]
                    
                    if(self.itemX[vj].indx.has_key(i)): 
                        grav -= self.itemX[vj].values[self.itemX[vj].indx[i]] * sj \
                        - self.itemX[vj].values[self.itemX[vj].indx[i]] * self.itemX[vj].values[self.itemX[vj].indx[i]] * model.itemV[i, z]
                    if(grav != 0):
                        model.itemV[i, z] -= self.lrate * (e * grav + self.l2_V * model.itemV[i, z])    
                       
                for i in self.userX[u].indx: 
                    grav = self.userX[u].values[self.userX[u].indx[i]] * (si - sj)
                    model.userV[i, z] -= self.lrate * (e * grav + self.l2_V * model.userV[i, z]) 
            
            sn = self.sgn(predict_p, predict_n) 
            if(sn == False):
                self.trainFM(u, vi, utils.fmObj(self.userX[u], self.itemX[vi], model), predict_p) 
            else:
                self.model = copy.deepcopy(model)               
            count += 1
            if(count % self.report == 0):
                print 'training...', count, 'samples have been trained!'
        print 'finish a turn!'
    
    
    def training(self, samples):
        count = 0
        for s in samples:
            u = self.uIDx[s[0]]
            vi = self.vIDx[s[1]]
            vj = self.vIDx[s[2]]
            
            predict_p = utils.fmObj(self.userX[u], self.itemX[vi], self.model)
            predict_n = utils.fmObj(self.userX[u], self.itemX[vj], self.model)
            
            e = utils.sigmoid(predict_p - predict_n) - 1
            self.model.W0 -= self.lrate * (e + self.l2_0 * self.model.W0)
            self.model.W0 -= self.lrate * (-1 * e + self.l2_0j * self.model.W0)
            
            for i in self.itemX[vi].indx:
                grav = self.itemX[vi].values[self.itemX[vi].indx[i]]
                self.model.itemW[i] -= self.lrate * (e * grav + self.l2_W * self.model.itemW[i])
            
            for i in self.itemX[vj].indx:
                grav = self.itemX[vj].values[self.itemX[vj].indx[i]]
                self.model.itemW[i] -= self.lrate * (-1 * e * grav + self.l2_Wj * self.model.itemW[i])
            
            for i in self.userX[u].indx: 
                grav = self.userX[u].values[self.userX[u].indx[i]]
                self.model.userW[i] -= self.lrate * (e * grav + self.l2_W * self.model.userW[i])
                grav = self.userX[u].values[self.userX[u].indx[i]]
                self.model.userW[i] -= self.lrate * (-1 * e * grav + self.l2_Wj * self.model.userW[i])
                  
            for z in range(0, self.model.Z):
                su = 0
                for f in self.userX[u].indx:
                    su += self.model.userV[f, z] * self.userX[u].values[self.userX[u].indx[f]]
                
                si = su
                for f in self.itemX[vi].indx:
                    si += self.model.itemV[f, z] * self.itemX[vi].values[self.itemX[vi].indx[f]]
                    
                for i in self.itemX[vi].indx:
                    grav = self.itemX[vi].values[self.itemX[vi].indx[i]] * si \
                        - self.itemX[vi].values[self.itemX[vi].indx[i]] * self.itemX[vi].values[self.itemX[vi].indx[i]] * self.model.itemV[i, z]
                   
                    self.model.itemV[i, z] -= self.lrate * (e * grav + self.l2_V * self.model.itemV[i, z])   
                
                sj = su
                for f in self.itemX[vj].indx:
                    sj += self.model.itemV[f, z] * self.itemX[vj].values[self.itemX[vj].indx[f]]
                
                for i in self.itemX[vj].indx:
                    grav = self.itemX[vj].values[self.itemX[vj].indx[i]] * sj \
                        - self.itemX[vj].values[self.itemX[vj].indx[i]] * self.itemX[vj].values[self.itemX[vj].indx[i]] * self.model.itemV[i, z]
                    
                    self.model.itemV[i, z] -= self.lrate * (-1 * e * grav + self.l2_Vj * self.model.itemV[i, z])  
                     
                for i in self.userX[u].indx: 
                    grav = self.userX[u].values[self.userX[u].indx[i]] * si
                    self.model.userV[i, z] -= self.lrate * (e * grav + self.l2_V * self.model.userV[i, z])     
                    
                    grav = self.userX[u].values[self.userX[u].indx[i]] * si
                    self.model.userV[i, z] -= self.lrate * (-1 * e * grav + self.l2_Vj * self.model.userV[i, z])                 
        
            count += 1
            if(count % self.report == 0):
                print 'training...', count, 'samples have been trained!'
        print 'finish a turn!'
    

    def training2(self, samples):
        count = 0
        for s in samples:
            u = self.uIDx[s[0]]
            vi = self.vIDx[s[1]]
            vj = self.vIDx[s[2]]
          
            predict_p = utils.fmObj(self.userX[u], self.itemX[vi], self.model)
            predict_n = utils.fmObj(self.userX[u], self.itemX[vj], self.model)
            
            if(self.task == 'class'):
                x = 1 * predict_p
                ei = 1 * (utils.sigmoid(x) - 1) 
                x = -1 * predict_n
                ej = -1 * (utils.sigmoid(x) - 1) 
            else:
                ei = predict_p - 1
                ej = predict_n + 1
            sn = self.sgn(predict_p, predict_n) 
            if(sn):    
                self.model.W0 -= self.lrate * (ei + ej + self.l2_0 * self.model.W0)
                for i in self.userX[u].indx:
                    grav = self.userX[u].values[self.userX[u].indx[i]] * (ei + ej)
                    self.model.userW[i] -= self.lrate * (grav + self.l2_W * self.model.userW[i])
                    
            e = utils.sigmoid(predict_p - predict_n) - 1
            for i in range(self.model.vf_num):
                grav = self.dim_sub(i, self.itemX[vi], self.itemX[vj]) * e
                if(self.itemX[vi].indx.has_key(i) and sn):
                    grav += ei * self.itemX[vi].values[self.itemX[vi].indx[i]] 
                if(self.itemX[vj].indx.has_key(i) and sn):
                    grav += ej * self.itemX[vj].values[self.itemX[vj].indx[i]]    
                if(grav != 0):
                    self.model.itemW[i] -= self.lrate * (grav + self.l2_W * self.model.itemW[i])
                 
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
                    g1 = g2 = 0
                    if(self.itemX[vi].indx.has_key(i)):
                        g1 = self.itemX[vi].values[self.itemX[vi].indx[i]] * si \
                            - self.itemX[vi].values[self.itemX[vi].indx[i]] * self.itemX[vi].values[self.itemX[vi].indx[i]] * self.model.itemV[i, z]
                        
                    if(self.itemX[vj].indx.has_key(i)): 
                        g2 = self.itemX[vj].values[self.itemX[vj].indx[i]] * sj \
                            - self.itemX[vj].values[self.itemX[vj].indx[i]] * self.itemX[vj].values[self.itemX[vj].indx[i]] * self.model.itemV[i, z]
                    grav = (g1 - g2) * e
                    
                    if(g1 != 0 and sn):
                        grav += g1 * ei
                    if(g2 != 0 and sn) :
                        grav += g2 * ej
                        
                    if(grav != 0): 
                        self.model.itemV[i, z] -= self.lrate * (grav + self.l2_V * self.model.itemV[i, z])    
                       
                for i in self.userX[u].indx: 
                    grav = self.userX[u].values[self.userX[u].indx[i]] * (si - sj) * e
                    if(sn):
                        g1 = self.userX[u].values[self.userX[u].indx[i]] * si \
                            - self.userX[u].values[self.userX[u].indx[i]] * self.userX[u].values[self.userX[u].indx[i]] * self.model.userV[i, z]
                        g2 = self.userX[u].values[self.userX[u].indx[i]] * sj \
                            - self.userX[u].values[self.userX[u].indx[i]] * self.userX[u].values[self.userX[u].indx[i]] * self.model.userV[i, z]
                        grav += g1 * ei + g2 * ej
                         
                    self.model.userV[i, z] -= self.lrate * (grav + self.l2_V * self.model.userV[i, z])                 
            count += 1
            if(count % self.report == 0):
                print 'training...', count, 'samples have been trained!'
        print 'finish a turn!'
        
           
    def training3(self, samples):
        count = 0
        for s in samples:
            u = self.uIDx[s[0]]
            vi = self.vIDx[s[1]]
            vj = self.vIDx[s[2]]
          
            predict_p = utils.fmObj(self.userX[u], self.itemX[vi], self.model)
            predict_n = utils.fmObj(self.userX[u], self.itemX[vj], self.model)
            
            sn = self.sgn(predict_p, predict_n) 
            if(sn == False):
                if(self.task == 'class'):
                    x = 1 * predict_p
                    ei = 1 * (utils.sigmoid(x) - 1) 
                    x = -1 * predict_n
                    ej = -1 * (utils.sigmoid(x) - 1) 
                else:
                    ei = predict_p - 1.0
                    ej = predict_n - 0
                
                grav = ei + ej
                self.model.W0 -= self.lrate * (grav + self.l2_0 * self.model.W0)
                
                for i in self.userX[u].indx:
                    grav = self.userX[u].values[self.userX[u].indx[i]] * (ei + ej)
                    self.model.userW[i] -= self.lrate * (grav + self.l2_W * self.model.userW[i])
                    
            e = utils.sigmoid(predict_p - predict_n) - 1
            for i in range(self.model.vf_num):
                grav = 0
                if(sn == False):
                    if(self.itemX[vi].indx.has_key(i)):
                        grav += ei * self.itemX[vi].values[self.itemX[vi].indx[i]] 
                    if(self.itemX[vj].indx.has_key(i)):
                        grav += ej * self.itemX[vj].values[self.itemX[vj].indx[i]] 
                else:
                    grav = self.dim_sub(i, self.itemX[vi], self.itemX[vj]) * e
                if(grav != 0):
                    self.model.itemW[i] -= self.lrate * (grav + self.l2_W * self.model.itemW[i])
                 
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
                    g1 = g2 = grav = 0
                    if(self.itemX[vi].indx.has_key(i)):
                        g1 = self.itemX[vi].values[self.itemX[vi].indx[i]] * si \
                            - self.itemX[vi].values[self.itemX[vi].indx[i]] * self.itemX[vi].values[self.itemX[vi].indx[i]] * self.model.itemV[i, z]    
                    if(self.itemX[vj].indx.has_key(i)): 
                        g2 = self.itemX[vj].values[self.itemX[vj].indx[i]] * sj \
                            - self.itemX[vj].values[self.itemX[vj].indx[i]] * self.itemX[vj].values[self.itemX[vj].indx[i]] * self.model.itemV[i, z]
                     
                    if(sn == False):
                        if(g1 != 0):
                            grav = g1 * ei
                        if(g2 != 0) :
                            grav += g2 * ej
                    else:
                        grav = (g1 - g2) * e
                    if(grav != 0):
                        self.model.itemV[i, z] -= self.lrate * (grav + self.l2_V * self.model.itemV[i, z])    
                       
                for i in self.userX[u].indx: 
                    grav = 0
                    if(sn == False):
                        g1 = self.userX[u].values[self.userX[u].indx[i]] * si \
                            - self.userX[u].values[self.userX[u].indx[i]] * self.userX[u].values[self.userX[u].indx[i]] * self.model.userV[i, z]
                        g2 = self.userX[u].values[self.userX[u].indx[i]] * sj \
                            - self.userX[u].values[self.userX[u].indx[i]] * self.userX[u].values[self.userX[u].indx[i]] * self.model.userV[i, z]
                        grav += (g1 * ei + g2 * ej)
                    else:
                        grav = self.userX[u].values[self.userX[u].indx[i]] * (si - sj) * e
                    self.model.userV[i, z] -= self.lrate * (grav + self.l2_V * self.model.userV[i, z])                 
            count += 1
            if(count % self.report == 0):
                print 'training...', count, 'samples have been trained!'
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
        while itera < self.iterTime :
            samples = self.drawer.drawRandomly()
            if(itera % self.updIter == 0):
                upd = True
            else:
                upd = False
            # samples = self.drawer.drawAdv(self.model, upd)
            # self.training0(samples)
            if(itera % self.iterTest == 0 and self.params.has_key('test_set')):
                info = Tmeasure(self.model, self.params['test_set'], self.params['test_pairs'], \
                                 3, self.params['test_userX'], self.params['test_itemX'])
                utils.log(self.params['output'] + '/log/log2.txt', str(itera) + '----' + info)
                # utils.log(self.params['output'] + '/log/logs2.txt', str(itera) + '----NLL:' + str(self.calNLL(samples)) + '\n')
            
            if(itera % self.saveIter == 0):
                is_conv = self.isConv(samples)
                utils.log(self.params['output'] + '/log/logs.txt', str(itera) + '----NLL:' + str(self.nloss) + '\n')
                if(is_conv):
                    break
                self.model.saveModel(self.params['output'], str(itera))
            
            self.training0(samples)
            itera += 1  
            print 'iteration:', itera
        self.model.saveModel(self.params['output'], 'final')
        print "training is done !";
      

if __name__ == "__main__":  
    # ad_fm_bpr = ad_FM_Bpr_Model(True)
    ad_fm_bpr = ad_FM_Bpr_Model()
    ad_fm_bpr.initModel()
    # ad_fm_bpr.reloadModel(model_symbol='27', niter=-1)
    ad_fm_bpr.train(itera=0)
