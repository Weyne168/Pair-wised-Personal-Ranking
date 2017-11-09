#! /usr/bin/env python
# coding: UTF-8
import random
import numpy as npy
from utils import sigmoid
import copy
from data import data
import sys


def innerVecs(v1, v2):
    return npy.squeeze(npy.asarray(v1 * v2.T));

def d_expr(x):  # 1/exp(x)
    if(npy.abs(x) > 36):
        if(x > 0):
            return 0
    return 1.0 / npy.exp(x)
        

###########################################################################33
class sampler:
   
    def __init__(self, graph, node_set, feature_num=0, user_set=None, lambda_setting=None, fuX=None, fvX=None):
        self.graph = graph
        self.ufID = user_set
        self.vfID = node_set
        self.item_set = copy.deepcopy(node_set)
        self.user_set = copy.deepcopy(user_set)
        self.node_set = []
        
        for u in graph:
            if(self.user_set.has_key(u) == False):
                self.user_set[u] = len(self.user_set)
            for v in graph[u]:
                if(self.item_set.has_key(v) == False):
                    self.item_set[v] = len(self.item_set)
            
        for k in self.item_set:
            self.node_set.append(k)
        
        self.fnum = feature_num
        self.node_num = len(self.node_set)
        
        if(lambda_setting != None):
            self.rank = self.getRankingDistribution(lambda_setting)
        if(fvX != None):
            self.sortedNodes = self.sortNodes(fvX)
        self.fvX = fvX
        self.fuX = fuX
        
    
    def sortNodes(self, fX):
        orders = []
        for f in range(self.fnum):
            fvs = []
            for n in self.node_set:
                if(self.vfID.has_key(n)):
                    if(fX[self.vfID[n]].indx.has_key(f)):
                        fvs.append(fX[self.vfID[n]].values[fX[self.vfID[n]].indx[f]]) 
                        continue
                fvs.append(0)
                      
            ods = npy.argsort(-npy.array(fvs))
            order = []
            for i in range(len(ods)):
                order.append(self.node_set[ods[i]]) 
            orders.append(order)
        return orders;
    
    
    def getRankingDistribution(self, lambda_setting):
        p = []
        # p.append(npy.exp(-1.0 / lambda_setting))
        p.append(d_expr(1.0 / lambda_setting))
        for k in range(1, len(self.node_set)):
            p.append(d_expr(1.0 * (k + 1) / lambda_setting) + p[k - 1])
        return p;
     
     
    def calFactoralDistribution(self, model, u, v):
        fws = []
        fs = []
        for f in self.fvX[v].indx:
            s = model.itemW[f]
            for fu in self.fuX[u].indx:
                s += innerVecs(model.userV[fu, :], model.itemV[f, :]) * self.fuX[u].values[self.fuX[u].indx[fu]]    
            s *= self.fvX[v].values[self.fvX[v].indx[f]] 
            if(len(fws) == 0):
                fws.append(sigmoid(s)) 
            else:   
                fws.append(fws[len(fws) - 1] + sigmoid(s))
            fs.append(f)
        return fws, fs;

    
    def getCate(self, vec):
        q = npy.random.uniform() * vec[len(vec) - 1]
        for f in range(len(vec)):
            if(vec[f] > q):
                break
        return f;
    
    
    def getRank(self, pos):
        q = npy.random.uniform() * self.rank[pos - 1]
        for p in range(pos):
            if(self.rank[p] > q):
                break
        return p;
     
    
    def get_adv_sample(self, cates_w, cates, u, vp):
        if(cates != None):
            c = cates[self.getCate(cates_w)]
        else:
            c = self.getCate(cates_w)
        r = self.getRank(len(self.sortedNodes[c]))
        if(self.fvX[vp].indx.has_key(c)):
            if(self.fvX[vp].values[self.fvX[vp].indx[c]] < 0):
                # return self.node_set[self.sortedNodes[c][self.node_num - r]]
                return self.sortedNodes[c][len(self.vfID) - r]
        # return self.node_set[self.sortedNodes[c][r]] 
        return self.sortedNodes[c][r]
       
      
    def adv_sample_init(self, model):
        us = {}
        for u in self.ufID:
            s = 0
            for fu in self.fuX[self.ufID[u]].indx:
                s += model.userV[fu, :] * self.fuX[self.ufID[u]].values[self.fuX[self.ufID[u]].indx[fu]]
            us[u] = s
        
        res = {}
        for u in self.graph:
            if(self.ufID.has_key(u) == False):
                continue
            res[u] = {}
            for v in self.graph[u]:
                fws = []
                fs = []
                if(self.vfID.has_key(v) == False):
                    continue
                for f in self.fvX[self.vfID[v]].indx:
                    s = innerVecs(us[u], model.itemV[f, :]) * self.fvX[self.vfID[v]].values[self.fvX[self.vfID[v]].indx[f]]
                    s += model.itemW[f] * self.fvX[self.vfID[v]].values[self.fvX[self.vfID[v]].indx[f]]
                    
                    if(len(fws) == 0):
                        fws.append(sigmoid(s)) 
                    else:   
                        fws.append(fws[len(fws) - 1] + sigmoid(s))
                    fs.append(f)
                res[u][v] = (fws, fs)
        return res;
        
        
    def drawAdv(self, model, upd=True, num=-1):
        print 'sampling...'
        res = []
        count = 0
        if(upd):
            self.factDistribution = self.adv_sample_init(model)
        for u in self.graph:
            for v in self.graph[u]:
                if(self.user_set[u] >= len(self.ufID) or self.item_set[v] >= len(self.vfID)):
                    vj = random.sample(self.node_set, 1)[0]
                else:
                    vj = self.get_adv_sample(self.factDistribution[u][v][0], self.factDistribution[u][v][1], self.user_set[u], self.item_set[v])
                search_time = 100
                while(self.graph[u].has_key(vj) and search_time > 0):
                    if(self.user_set[u] >= len(self.ufID) or self.item_set[v] >= len(self.vfID)):
                        vj = random.sample(self.node_set, 1)[0]
                    else:
                        vj = self.get_adv_sample(self.factDistribution[u][v][0], self.factDistribution[u][v][1], self.user_set[u], self.item_set[v])
                    search_time -= 1
                if(search_time == 0):
                    break
                res.append((u, v, vj))
                count += 1
                if(count == num):
                    random.shuffle(res)
                    return res
        random.shuffle(res)
        return res;  
    
    
    def drawRandomly(self, num=-1):
        print 'sampling...'
        res = []
        count = 0
        for u in self.graph:
            for v in self.graph[u]:
                vj = random.sample(self.node_set, 1)
                search_time = 100 
                while(self.graph[u].has_key(vj[0]) and search_time > 0):
                    vj = random.sample(self.node_set, 1) 
                    search_time -= 1
                if(search_time == 0):
                    break
                res.append((u, v, vj[0]))
                count += 1
                if(count == num):
                    random.shuffle(res)
                    return res
        random.shuffle(res)
        return res; 
    
if __name__ == "__main__":
    dataset = data()
    g1 = dataset.loadGraph(sys.argv[1], 3)
    g2 = dataset.loadGraph(sys.argv[2], 3)
    g = {}
    nds = []
    nd_ = {}
   
    
    for u in g1:
        g[u] = {}
        for v in g1[u]:
            if(nd_.has_key(v)):
                continue
            nd_[v] = 0
            nds.append(v)
            g[u][v] = 1
    
    for u in g2:
        if(g.has_key(u) == False):
            g[u] = {}
        for v in g2[u]:
            if(nd_.has_key(v)):
                continue
            nd_[v] = 0
            nds.append(v)
            g[u][v] = 1
    
    fout = open(sys.argv[3], 'w')
    for u in g1:
        for v in g1[u]:
            vj = random.sample(nds, 1)
          
            search_time = 100 
            while(g[u].has_key(vj[0]) and search_time > 0):
                vj = random.sample(nds, 1) 
                search_time -= 1
                if(search_time == 0):
                    break
           
            if(npy.random.uniform() < float(sys.argv[4])):
                fout.write(u + '\t' + vj[0] + '\t0\n')
    fout.close()
                
                
            
            
            
    
      
      
        
