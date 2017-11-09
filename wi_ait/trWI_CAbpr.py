#! /usr/bin/env python
# coding: UTF-8
import string 
import NewMa as SGD
from sampler import  sampler_CA_BPR 
from utils import log
from data import data
from trBPR import BPRSvd
from Model_WI_CABpr import CA_BPR_Model
from infer import Tmeasure 


def CAL_MAT(para):
    return SGD.calMat3(para);


def SGD_BPR(para):
    return SGD.SGD_BPR(para);


#######################################################################
class CA_BPR_Training(BPRSvd):
    thred = 0.5
   
    def __init__(self, debug=False):
        BPRSvd.__init__(self, debug)
        
    
    def initParms(self):            
        BPRSvd.initParms(self)
        if(self.params.has_key('lW')):
            vs = self.params['lW'].split(',')
            self.lambdaUw = float(vs[0])
            self.lambdaIw = float(vs[1])
        else:
            self.lambdaUw = 0.001
            self.lambdaIw = 0.001
    
    def prepare(self):
        dataset = data()
        
        uf_num = 0
        self.userX = None
        self.uIDx = {}
        vf_num = 0
        self.itemX = None
        self.vIDx = {}
        
        self.graph = dataset.loadGraph(self.params['graph'], string.atoi(self.params['dim']))
        feature, f_num, fvIDx = dataset.loadFeature2(self.params['feature'])
        self.drawer = sampler_CA_BPR(self.graph, feature, fvIDx, self.Z, \
                                     lambda_setting=string.atoi(self.params['seed_num']), \
                                     thred=float(self.params['thred']))
        
        if(self.params.has_key('userX')):
            userX, uf_num, self.uIDx = dataset.loadFeature3(self.params['userX'],self.drawer.user_set)
            self.userX = dataset.constructSparseMat(userX, uf_num)
            # self.userX = numpy.matrix(self.userX)
        if(self.params.has_key('itemX')):
            itemX, vf_num, self.vIDx = dataset.loadFeature3(self.params['itemX'],self.drawer.item_set)
            # self.itemX = numpy.matrix(self.itemX)
            self.itemX = dataset.constructSparseMat(itemX, vf_num)
        
        #feature, f_num, fvIDx = dataset.loadFeature2(self.params['feature'])
        #feature = dataset.constructSparseMat(feature, f_num).todense()
        self.model = CA_BPR_Model(string.atoi(self.params['k']), self.drawer.user_set, self.drawer.item_set, uf_num, vf_num)
    
    
    def trainBPR(self, samples):
        features = (self.userX, self.itemX, self.model.userMatrix, self.model.itemMatrix, self.model.uLatent, self.model.vLatent)
        lambdaW = (self.lambdaUw, self.lambdaIw, self.lambdaU, self.lambdaVi, self.lambdaVj)
        nodes = (self.uIDx, self.vIDx, self.drawer.user_set, self.drawer.item_set)
        res = []
        args_bpr = (samples, nodes, features, lambdaW, self.lrate, self.Z)
        res.append(SGD_BPR(args_bpr))
    
        tt = res[0]
        ul = tt[0]
        vl = tt[1]
        return ul, vl;
        
    
    def train(self, itera=1):
        u_cat_vars, v_cat_vars, item_sorts = self.drawer.preDraw(self.model.uLatent, self.model.vLatent)
        while itera < self.iterTime :
            samples = self.drawer.draw_CA_BPR(self.model.uLatent, self.model.vLatent, u_cat_vars, v_cat_vars, item_sorts)
            # samples = sampler.drawRandomly(self.truvInd, self.truNodes, self.trvNodes)
            if(self.userX != None):
                paras = (self.userX, self.model.uLatent, self.lambdaUw)
                self.model.userMatrix = CAL_MAT(paras)
            if(self.itemX != None):
                paras = (self.itemX, self.model.vLatent, self.lambdaIw)
                self.model.itemMatrix = CAL_MAT(paras)
            
            paras = self.trainBPR(samples)  
            z, orders, u_var, v_var = self.drawer.resort_cate(paras[0], paras[1])
            if(orders != None):
                item_sorts[z] = orders
                u_cat_vars[z] = u_var
                v_cat_vars[z] = v_var  
            self.model.uLatent = paras[0]
            self.model.vLatent = paras[1]
           
            if(itera % self.saveIter == 0):
                print 'iteration:',itera
                self.model.saveModel(self.params['output'], str(itera))
               
                #if(self.isConv(samples)):
                #    break
                #log(self.params['output'] + '/log/logs.txt', str(itera) + '--' + str(self.obj_val))
                
            if(itera % self.iterTest == 0 and self.params.has_key('test')):
                self.model.saveModel(self.params['output'], '0')
                info = Tmeasure(self.model.Z, self.params['test'], 3, \
                                 self.params['output'], str(0), \
                                 'acbpr', \
                                 self.params['test_userX'], self.params['test_itemX'])
                self.isConv(samples)
                log(self.output + '/log/log2.txt', str(itera) + ':\n' + info)
                #log(self.params['output'] + '/log/logs2.txt', str(itera) + '--' + str(self.obj_val))
            itera += 1 
        
        self.model.saveModel(self.params['output'], str(itera))
        #log(self.params['output'] + '/log/logs.txt', str(itera) + '--' + str(self.obj_val))  
        print 'finish!' ;  
    


if __name__ == "__main__":
    itera = 0
    model_symbol = '0'
    train = CA_BPR_Training() 
    train.prepare()
   
    if(train.params.has_key('m')):
        model_symbol=train.params['m']
    if(train.params.has_key('cit')):
        itera = string.atoi(train.params['cit'])
    if(train.params.has_key('m')):   
        train.reloadModel(train.params['m'], train.iterTime)
    print 'start to train...'
    train.train(itera)
