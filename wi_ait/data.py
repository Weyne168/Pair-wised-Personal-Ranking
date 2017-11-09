#!/usr/bin/env python
# coding: UTF-8
import string
import numpy as npy
from scipy.sparse import csr_matrix


def createGraph(vs, m, l=True):
    if(len(vs) == 2):
        if(l):
            m[vs[0]] = vs[1]  
        else:
            m[vs[0]] = float(vs[1])
        return m
    else:
        if(m.has_key(vs[0]) == False):
            m[vs[0]] = {}
        m[vs[0]] = createGraph(vs[1:], m[vs[0]], l)
        return m
    


class feature:
    def __init__(self):
        self.indx = {}  # feature number
        self.values = []  # feature value
    
    def add(self, fid, v):
        self.indx[fid] = len(self.indx)
        self.values.append(v)


class data:
    def __init__(self):
        pass
    
    
    def loadGraph(self, graphFile, dim=2, label=True):
        graph = {}
        fin = open(graphFile, 'r')
        while 1:
            lines = fin.readlines(100000)
            if not lines:
                break
            for line in lines:
                line = line.strip()
                vs = line.split('\t')[0:dim]
                self.grah = createGraph(vs, graph, label)
        fin.close()
        return graph
                
    
    def loadFeature(self, featureFile):
        fs = []
        idx2id = {}
        
        fin = open(featureFile, 'r')
        max_num = string.atoi(fin.readline().strip())
        while 1:
            lines = fin.readlines(100000)
            if not lines:
                break
            for line in lines:
                line = line.strip()
                idx, vs = line.split('\t')
                
                if(idx2id.has_key(idx) == False):
                    idx2id[idx] = len(idx2id)
                kvs = vs.split(',')
                x = feature()
                for kv in kvs:
                    k, v = kv.split(':')
                    k = string.atoi(k)
                    x.add(k, float(v))
                fs.append(x)
        fin.close()
        return fs, max_num, idx2id;
    
    def loadFeature2(self, featureFile):
        fs = []
        idx2id = {}
        fin = open(featureFile, 'r')
        max_num = string.atoi(fin.readline().strip())
        while 1:
            lines = fin.readlines(100000)
            if not lines:
                break
            for line in lines:
                line = line.strip()
                idx, vs = line.split('\t')
                
                if(idx2id.has_key(idx) == False):
                    idx2id[idx] = len(idx2id)
                kvs = vs.split(',')
                x=[]
                for v in kvs:
                    x.append(float(v))
                fs.append(npy.array(x))
        fin.close()
        return fs, max_num, idx2id;
    
    
    def loadFeature3(self, featureFile, idx_dict=None):
        fs = []
        idx2id = {}
        
        if(idx_dict!=None):
            fs=[None  for col in range(len(idx_dict))]
        
        fin = open(featureFile, 'r')
        max_num = string.atoi(fin.readline().strip())
        while 1:
            lines = fin.readlines(100000)
            if not lines:
                break
            for line in lines:
                line = line.strip()
                idx, vs = line.split('\t')
                if(idx_dict!=None):
                    if(idx_dict.has_key(idx)==False):
                        continue
                if(idx2id.has_key(idx) == False):
                    idx2id[idx] = len(idx2id)
                kvs = vs.split(',')
                x = feature()
                for kv in kvs:
                    k, v = kv.split(':')
                    k = string.atoi(k)
                    x.add(k, float(v))
                fs[idx_dict[idx]]=x
        fin.close()
        return fs, max_num, idx2id;

    def constructSparseMat(self, fX, feature_num):
        rows = []
        cols = []
        vals = []
        for r in range(len(fX)):
            if(fX[r]==None):
                continue
            for c in fX[r].indx:
                rows.append(r)
                cols.append(c)
                vals.append(fX[r].values[fX[r].indx[c]])
        return csr_matrix((vals, (rows, cols)), shape=(len(fX), feature_num));


