#! /usr/bin/env python
# coding: UTF-8
import os
import numpy
import sys
import string
import copy
import random
import utils
from Model_pFM import PFM_Model
from data import data
from sklearn.metrics import auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_squared_error


###########################################################
class Infer:
    def __init__(self, gen=True):
        self.params = {}
        self.params['output'] = sys.path[0]
        if(gen):
            if(len(sys.argv) > 2):
                for param in sys.argv[1:]:
                    k, v = param.split('=')
                    self.params[k] = v
        
            if(self.params.has_key('config')):
                self.config(self.params['config'])
    
    
    def config(self, confFile):
        if(os.path.exists(confFile) == 0):
            print 'There is not configuration file!'
            exit(0)     
        conf = open(confFile, 'r')
        while 1:
            line = conf.readline()
            if not line:
                break
            k, v = line.strip().split('=')
            self.params[k] = v
        conf.close();  
    
        
    def setParms(self, k, test_set, test_pairs, dim, ux, vx):
        self.params['k'] = k
        self.params['test_set'] = test_set
        self.params['test_pairs'] = test_pairs
        self.params['dim'] = dim
        self.params['userX'] = ux
        self.params['itemX'] = vx
    
   
    def prepare(self):
        dataset = data()
        self.userX, uf_num, self.uIDx = dataset.loadFeature(self.params['userX'])
        self.itemX, vf_num, self.vIDx = dataset.loadFeature(self.params['itemX'])
        self.model = PFM_Model(string.atoi(self.params['k']), uf_num, vf_num)
        self.model.loadModel(self.params['model_dir'], self.params['model_name'])   
        self.test_set = dataset.loadGraph(self.params['test_set'], string.atoi(self.params['dim']), False)
        self.test_pairs = dataset.loadGraph(self.params['test_pairs'], string.atoi(self.params['dim']), False)
        
    
    def prepare2(self, model):
        self.model = copy.deepcopy(model)
        dataset = data()
        self.userX, uf_num, self.uIDx = dataset.loadFeature(self.params['userX'])
        self.itemX, vf_num, self.vIDx = dataset.loadFeature(self.params['itemX'])
        self.test_set = dataset.loadGraph(self.params['test_set'], self.params['dim'], False,5)
        self.test_pairs = dataset.loadGraph(self.params['test_pairs'], self.params['dim'], False)
        
         
    def predict2(self):
        res = {}
        count = 0
        for u in self.test_pairs:
            res[u] = {}
            for v in self.test_pairs[u]:
                if(self.model.user_set.has_key(u) and self.model.item_set.has_key(v)):
                    res[u][v] = numpy.squeeze(numpy.asarray(self.model.uLatent[self.model.user_set[u], :] \
                                                            * self.model.vLatent[self.model.item_set[v], :].T))
                elif(self.vIDx.has_key(v) and self.uIDx.has_key(u)):
                    res[u][v] = utils.fmObj(self.userX[self.uIDx[u]], self.itemX[self.vIDx[v]], self.model)
                else:
                    res[u][v] = random.uniform(-1.0, 5.0)
                    # res[u][v] = random.uniform()
                count += 1  
        print count, 'instances have been predicted !'
        return res; 
    
    def predict3(self):
        res = {}
        count = 0
        for u in self.test_pairs:
            res[u] = {}
            for v in self.test_pairs[u]:
                if(self.vIDx.has_key(v) and self.uIDx.has_key(u)):
                    res[u][v] = utils.fmObj(self.userX[self.uIDx[u]], self.itemX[self.vIDx[v]], self.model)
                elif(self.model.user_set.has_key(u) and self.model.item_set.has_key(v)):
                    res[u][v] = numpy.squeeze(numpy.asarray(self.model.uLatent[self.model.user_set[u], :] \
                                                            * self.model.vLatent[self.model.item_set[v], :].T))
                else:
                    res[u][v] = random.uniform(-1.0, 5.0)
                    # res[u][v] = random.uniform()
                count += 1  
        print count, 'instances have been predicted !'
        return res; 
    
    def predict4(self):
        res = {}
        count = 0
        for u in self.test_pairs:
            res[u] = {}
            for v in self.test_pairs[u]:
                if(self.vIDx.has_key(v) and self.uIDx.has_key(u)):
                    res[u][v] = utils.fmObj(self.userX[self.uIDx[u]], self.itemX[self.vIDx[v]], self.model)
                else:
                    res[u][v] = random.uniform(-1.0, 5.0)
                    # res[u][v] = random.uniform()
                count += 1  
        print count, 'instances have been predicted !'
        return res;
    
    
    def predict(self):
        res = {}
        count = 0
        for u in self.test_pairs:
            res[u] = {}
            for v in self.test_pairs[u]:
                if(self.vIDx.has_key(v) and self.uIDx.has_key(u)):
                    res[u][v] = utils.fmObj(self.userX[self.uIDx[u]], self.itemX[self.vIDx[v]], self.model)
                else:
                    res[u][v] = random.uniform(-1.0, 5.0)
                    # res[u][v] = random.uniform()
                count += 1  
        print count, 'instances have been predicted !'
        return res; 
    

################################################################################################3
class eveluate:
    def __init__(self, test_set, test_pairs, res):
        self.test_set = test_set
        self.test_pairs = test_pairs
        self.res = res
        self.y = []
        self.pred = []
        for u in test_pairs:
            for v in test_pairs[u]:
                self.pred.append(res[u][v])
                self.y.append(test_pairs[u][v])   
    

    def MAP(self, rks):
        map_val = calMAP(self.test_set, rks)
        print 'MAP:' , map_val;
        return map_val; 
    
    
    def NDCG(self, n, rks):
        ndcg_val = dcg(self.test_set, rks, n)
        print 'NDCG@' + str(n) , ndcg_val;
        return ndcg_val; 
    
    
    def AUC(self):
        auc = 0
        for u in self.test_set:
            pair_num = 0
            loss = 0
            for v in self.test_set[u]:
                for vj in self.test_pairs[u]:
                    if(self.test_pairs[u][vj] != 1):
                        pair_num += 1
                        loss += utils.sigmoid(self.res[u][v] - self.res[u][vj])
            auc += loss / pair_num
        
        auc /= len(self.test_set)
        # auc = auc_score(numpy.array(self.y), numpy.array(res));
        print 'AUC:' , auc;
        return auc;
    
    
    def PR_curve(self):
        precision, recall, thresholds = precision_recall_curve(numpy.array(self.y), numpy.array(self.pred))
        return precision, recall, thresholds;
    
    
    def RMSE(self):
        rmse = numpy.sqrt(mean_squared_error(numpy.array(self.y), numpy.array(self.pred)))
        print 'RMSE:' , rmse;
        return rmse;
    
    
    def Precision_Recall(self, thresh=[0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1, 2.8, 3.6, 4.5, 5.4, 6.2, 6.9, 7.5, 8.0, 8.4, 8.7, 8.9, 9]):
        recall_num = numpy.array([0.001  for col in range(len(thresh))])
        hits = numpy.array([0.001  for col in range(len(thresh))])
        for u in self.test_pairs:
            for v in self.test_pairs[u]:
                for i in range(len(thresh)):
                    if(self.res[u][v] > thresh[i]):
                        recall_num[i] += 1.0
                        if(self.test_set.has_key(u)):
                            if(self.test_set[u].has_key(v)):
                                hits[i] += 1.0
                   
        ground_num = 0      
        for u in self.test_set:
            for v in self.test_set[u]:
                ground_num += 1.0
        # precision = precision_score(numpy.array(self.y), numpy.array(self.pred), pos_label=1)
        print 'Precision:' , hits / recall_num;
        print 'Recall:' , hits / ground_num;
        return hits / recall_num, hits / ground_num;  


########################################################################################
def rankRes(simData, topK=-1):
    ranks = {}
    for u in simData.keys():
        vs = [] 
        sim = []
        for v in simData[u].keys():
            if(simData[u][v] < 0):
                continue
            sim.append(simData[u][v])
            vs.append(v)
        at = numpy.argsort(sim)
        ranks[u] = {}
        n = len(at) - 1 
        i = 1 
            
        if(topK < 0 or topK - 1 > n):
            topK = len(at)
        rank = []                    
        while n >= 0 and i < topK:
            rank.append(vs[at[n]])
            n -= 1
            i += 1
        ranks[u] = rank
    return ranks;
         

def calMAP(grks, rks):
    s = 0
    for u in rks.keys():
        if(grks.has_key(u) == False):
            continue
        
        i = 0.0
        rank_score = 0
        for j in range(len(rks[u])):
            if(grks[u].has_key(rks[u][j])):
                if(grks[u][rks[u][j]] > 0):
                    i += 1.0
                    rank_score += i / (j + 1)
        if(i > 0): 
            # rank_score /= len(rks[u])  
            rank_score /= len(grks[u])   
        s += rank_score  
    return s / len(grks);


# for implicit feedback, there is only two kinds of rates, i.e., 0 and 1#
def dcg(grks, rks, n):
    s = 0
    for u in rks.keys():
        if(grks.has_key(u) == False):
            continue

        dcg_score = 0
        gains = []
        for j in range(len(rks[u][0:n])):
            if(grks[u].has_key(rks[u][j])):
                if(grks[u][rks[u][j]] > 0):
                    gains.append(numpy.power(2, grks[u][rks[u][j]]) - 1)
                    # gains.append(numpy.power(2, 1) - 1)
                    dcg_score += gains[len(gains) - 1] / numpy.log2 (j + 1 + 1)   
        
        if(len(gains) < 1):
            continue
        at = numpy.argsort(gains)
        max_dcg_score = 0
        for j in range(0, len(gains)):
            max_dcg_score += gains[at[j]] / numpy.log2(len(gains) + 1 - j) 
        
        s += dcg_score / max_dcg_score 
    return s / len(grks);

        
def Tmeasure(model, test_set, test_pairs, dim, ux, vx, t, ndcg_num=[3, 5, 10, 20]):
    inf = Infer(False)
    inf.setParms(model.Z, test_set, test_pairs, dim, ux, vx)
    inf.prepare2(model)
    
    if(t == 1):
        res = inf.predict()
    elif(t == 3):
        res = inf.predict3() 
    elif(t == 4):
        res = inf.predict4() 
    else:
        res = inf.predict2()
    
    ranks = rankRes(res)
    elt = eveluate(inf.test_set, inf.test_pairs, res)
    
    ndcg_res = []
    for i in range(len(ndcg_num)):
        ndcg_res.append(elt.NDCG(ndcg_num[i], ranks))
    
    ap = elt.MAP(ranks)
    auc = elt.AUC()
    # rmse = elt.RMSE()
    prec, recall = elt.Precision_Recall()
    # info = 'MAP:' + str(ap) + '\tAUC:' + str(auc) + '\tPercision:' + str(per) + '\tRMSE:' + str(rmse) + '\n'
    info = 'MAP:' + str(ap) + '\tAUC:' + str(auc) + '\tPercision:' + str(prec) + '\tRecall:' + str(recall)\
        + '\tNDCG:' + str(ndcg_res) + '\n'
    return info
    

def saveRes(data, des):
    fout = open(des, 'w')
    for u in data:
        for v in data[u]:
            fout.write(u + '\t' + v + str(res[u][v]) + '\n')
    fout.close();


if __name__ == "__main__":
    inf = Infer()
    inf.prepare()
    res = inf.predict()
    saveRes(res, 'output')
