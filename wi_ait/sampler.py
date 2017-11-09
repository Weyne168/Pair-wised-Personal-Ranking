#! /usr/bin/env python
# coding: UTF-8
import random
import numpy as npy
from numpy import linalg as LA


###########################################################
class sampler:
    def __init__(self, graph, K, lambda_setting=None):
        self.graph = graph
        self.Z = K
        self.user_set = {}
        self.item_set = {}
        self.node_set = []
        for u in graph:
            self.user_set[u] = len(self.user_set)
            for v in graph[u]:
                if(self.item_set.has_key(v) == False):
                    self.item_set[v] = len(self.item_set)
                    self.node_set.append(v)
        self.node_num = len(self.node_set)
        
        if(lambda_setting != None):
            self.rank = self.getRankingDistribution(lambda_setting)
                    
    
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
    
    
    def getRankingDistribution(self, lambda_setting):
        p = []
        p.append(npy.exp(-1.0 / lambda_setting))
        for k in range(1, self.node_num):
            p.append(npy.exp(-1.0 * (k + 1) / lambda_setting) + p[k - 1])
        return p;
    
    
    def getPos(self, pos):
        q = npy.random.uniform() * self.rank[pos - 1]
        for p in range(pos):
            if(self.rank[p] > q):
                break
        return p;
    

    def preDraw(self, uL, vL):
        item_sorts = []
        #=======================================================================
        # u_cate_mean = []
        # v_cate_mean = []
        #=======================================================================
        u_cate_var = []
        # v_cate_var = []
        for z in range(self.Z):
            #===================================================================
            # u_cate_mean.append(npy.mean(uL[:, z]))
            # v_cate_mean.append(npy.mean(vL[:, z]))
            #===================================================================
            u_cate_var.append(npy.std(uL[:, z]))
            # v_cate_var.append(npy.std(vL[:, z]))
            item_sorts.append(npy.squeeze(npy.asarray(npy.argsort(-1 * vL[:, z].T))))   
        return u_cate_var, item_sorts

    
    def getCate(self, u_vec, cat_vars):
        p = []
        p.append(npy.abs(u_vec[0]) * cat_vars[0])
        for k in range(1, self.Z):
            p.append(p[k - 1] + npy.abs(u_vec[k]) * cat_vars[k])
        q = npy.random.uniform() * p[self.Z - 1]
        for f in range(self.Z):
            if(p[f] > q):
                break
        return f;
    

    def drawRendle(self, uL, u_cat_vars, item_sorts):
        res = []
        for u in self.graph:
            for v in self.graph[u]:
                cate = self.getCate(npy.squeeze(npy.asarray(uL[self.user_set[u], :])), u_cat_vars)
                search_time = 100
                while 1 :
                    r = self.getPos(self.node_num)
                    if(uL[self.user_set[u], cate] < 0):
                        r = self.node_num - r - 1
                    vj = self.node_set[item_sorts[cate][r]]
                    if(self.graph[u].has_key(vj) == False or search_time == 0):
                        break
                    search_time -= 1
                if(search_time == 0):
                    break
            res.append((u, v, vj))
        random.shuffle(res)
        return res;


#######################################################################
class sampler_CA_BPR(sampler):
    def __init__(self, graph, feature, feature_item_set, K, lambda_setting=100, thred=0.5):
        sampler.__init__(self, graph, K, lambda_setting)
       
        self.thred = thred
        feature_ori = []
       
        for i in range(len(self.node_set)):
            if(feature_item_set.has_key(self.node_set[i])):
                v = feature_item_set[self.node_set[i]]
                f = feature[v]
                s = LA.norm(f, 1)
                f /= s
            else:
                f = [1.0 / K for col in range(K)]
            feature_ori.append(f)
       
        self.item_feature_init = npy.matrix(feature_ori) 
        self.uSum = len(self.user_set)
        self.vSum = len(self.item_set)
      
        
             
    def preDraw(self, uL, vL):
        item_sorts = []
        u_cate_var = []
        v_cate_var = []
        for z in range(self.Z):
            u_cate_var.append(npy.std(uL[:, z]))
            v_cate_var.append(npy.std(vL[:, z]))
            item_sorts.append(npy.squeeze(npy.asarray( npy.argsort(-1*npy.squeeze(npy.asarray(self.item_feature_init[:, z].T)))))) 
        return u_cate_var, v_cate_var, item_sorts
        
    
    def getCate(self, u_vec, v_vec):
        p = []
        p.append(u_vec[0] * v_vec[0])
        for k in range(1, self.Z):
            p.append(p[k - 1] + u_vec[k] * v_vec[k])
        
        q = npy.random.uniform() * p[self.Z - 1]
        for f in range(self.Z):
            if(p[f] > q):
                break
        return f;
    
    
    def resort_cate(self, uL, vL):
        cate = self.getCate([1  for col in range(self.Z)], self.cate_count)
        v_var_new = npy.std(vL[:, cate])
        u_var_new = npy.std(uL[:, cate])
        
        d = npy.squeeze(npy.asarray(npy.abs(self.item_feature_init[:, cate].T * vL[:, cate])\
                                     / (npy.sqrt(self.item_feature_init[:, cate].T * self.item_feature_init[:, cate])\
                                     * npy.sqrt(vL[:, cate].T * vL[:, cate]))))
        if(d < self.thred):
            self.item_feature_init[:, cate] = vL[:, cate]
            return cate, npy.squeeze(npy.asarray(npy.argsort(-1 * vL[:, cate].T))), u_var_new, v_var_new
        else:
            return cate, None, u_var_new, v_var_new
    
    
    def draw_CA_BPR(self, uL, vL, u_cat_vars, v_cat_vars, item_sorts):
        self.cate_count = [0  for col in range(self.Z)]
        uNorm = [[0 for col in range(self.Z)] for row in range(self.uSum)]
        vNorm = [[0 for col in range(self.Z)] for row in range(self.vSum)]
        
        for u in self.user_set: 
            indx = self.user_set[u]
            for c in range(self.Z):
                uNorm[indx][c] = npy.abs(uL[indx, c]) * u_cat_vars[c]
                      
        for v in self.item_set:
            indx = self.item_set[v]
            for c in range(self.Z):
                vNorm[indx][c] = npy.abs(vL[indx, c]) * v_cat_vars[c]
       
        res = []
        for u in self.graph:
            for v in self.graph[u]:
                cate = self.getCate(uNorm[self.user_set[u]], vNorm[self.item_set[v]])
                self.cate_count[cate] += 1
                while 1 :
                    r = self.getPos(self.vSum)
                    if(uL[self.user_set[u], cate] < 0):
                        r = self.vSum - r - 1
                    vj = self.node_set[item_sorts[cate][r]]  # the N.O. of vj
                    if(self.graph[u].has_key(vj) == False):
                        break
                res.append((u, v, vj))
        random.shuffle(res)
        return res;

if __name__ == "__main__":
    print 'dddddddddddddddd'
