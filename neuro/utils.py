#! /usr/bin/env python
# coding: UTF-8
import random 
import numpy
from numpy import linalg as LA
import sys
import string


def sigmoid(x):
    if (numpy.abs(x) > 36):
        if(x > 0):
            return 1.0
        else:
            return 0.0
    return 1.0 / (1.0 + numpy.exp(-1 * x));
    

def fmObj(uX, vX, model):
    res = model.W0
    for f in uX.indx:
        res += uX.values[uX.indx[f]] * model.userW[f]
    for f in vX.indx:
        res += vX.values[vX.indx[f]] * model.itemW[f]
        
    for k in range(model.Z):
        ts = 0
        ts2 = 0
        for f in uX.indx:
            t = uX.values[uX.indx[f]] * model.userV[f, k]
            ts += t
            ts2 += t * t
        for f in vX.indx:
            t = vX.values[vX.indx[f]] * model.itemV[f, k]
            ts += t
            ts2 += t * t
        res += (ts * ts - ts2)
    return 0.5 * res    


def cosin(vec1, vec2):
    res = numpy.dot(vec1, vec2)
    t = LA.norm(vec1, 2)
       
    if(t == 0):
        return 0
    res /= t
        
    t = LA.norm(vec2, 2)
    if(t == 0):
        return 0
    res /= t
    return res;


def log(logFile, text):
    fout = open(logFile , 'a')
    fout.write(text)
    fout.close() ;


def createTestPairsSet(totalFile, testFile, ave_num, output):
        uvInd = {}
        vs = {}
        fin = open(totalFile, 'r')
        while 1:
            line = fin.readline()
            if not line:
                break
            uv = line.strip().split('\t')
            if(uvInd.has_key(uv[0]) == False):
                uvInd[uv[0]] = {}
            uvInd[uv[0]][uv[1]] = 1
            vs[uv[1]] = 1
        fin.close()
        
        vNodes = []
        for v in vs:
            vNodes.append(v) 
        random.shuffle(vNodes) 
        print 'item:', len(vNodes), 'user:', len(uvInd)
       
        
        testInd = {}
       
        fin = open(testFile, 'r')
        while 1:
            line = fin.readline()
            if not line:
                break
            uv = line.strip().split('\t')
            if(testInd.has_key(uv[0]) == False):
                testInd[uv[0]] = {}
            testInd[uv[0]][uv[1]] = 1
        fin.close()
        
        un_observed = {}
        for u in testInd:
            un_observed[u] = {}
            exNum = 0
            for v in vs:
                if(uvInd[u].has_key(v)):
                    continue
                un_observed[u][v] = -1
                if(ave_num < exNum):
                    break
                exNum += 1
        
        total = 0                             
        fout = open(output, 'w')
        for u in testInd:
            for v in testInd[u]:
                fout.write(u + '\t' + v + '\t' + str(testInd[u][v]) + '\n')
                total += 1
        for u in un_observed:
            for v in un_observed[u]:
                fout.write(u + '\t' + v + '\t' + str(un_observed[u][v]) + '\n')
                total += 1
        fout.close()   
        print 'pairNum:', total, 'finish getting test pairs!'; 
        

def samplePairs(srcFile, num, rate, output):
    fin = open(srcFile, 'r')
    fout = open(output, 'w')
    while 1:
        line = fin.readline()
        if not line:
            break
        if(num < 0):
            break
        if(rate < numpy.random.uniform()):
            continue
        fout.write(line)
        num -= 1
    fin.close() 
    fout.close() 
  

def getAverage():
    res = {}
    fs_lab = []
    fn = 0
    for f in sys.argv[1:]:
        fs_lab.append(fn)
        
        fin = open(f, 'r')
        while 1:
            line = fin.readline()
            if not line:
                break
            if(line.find('--') == -1):
                continue
            vs = line.split('\t')
            lab = vs[0].split('-')
            n = string.atoi(lab[0])
            if(res.has_key(n) == False):
                res[n] = {}
            auc = vs[1].split(':')
            res[n][fn] = float(auc[1])
        fn += 1    
        fin.close()
        
    x = 0
    while 1:
        if(x > 995):
            break
        if(res.has_key(x) == False):
            x += 5
            continue
        tmp = str(x) + ','
        s = 0
        k=0
        for fn in fs_lab:
            if(res[x].has_key(fn) == False):
                tmp += ','
            else:
                tmp += str(res[x][fn]) + ','
                s += res[x][fn]
                k+=1
        tmp+=str(s/k)
        x+=5
        print tmp
   
    
if __name__ == "__main__":
    #getAverage()
    #exit(0)
    srcFile = 'I:/dataset/tag-genome/tag_relevance.dat'
    output = 'I:/dataset/tag-genome/test/demo.set'
    # filterdata(srcFile, 0.2, output)
    
    #============================================================================
    #  srcFile = 'I:/dataset/citation-network1/data/refGraph.txt'
    #  samplePairsByItem(srcFile, 100, 'output')
    # 
    #  srcFile = 'I:/dataset/citation-network1/data/refGraph.txt'
    #  output = 'I:/dataset/citation-network1/new_test/demo.set'
    #  samplePairs(srcFile, 10000, 0.2, output)
    #============================================================================
      
    srcFile = 'I:/dataset/ml-100k/cleanedData/dataset.user50/u.data.sample_user50'
    output = 'II:/dataset/ml-100k/cleanedData/dataset.user50/u.data.sample_user50'
    # splitDataSet(srcFile, 0.2, output)
     
    srcFile = 'I:/dataset/delicious/data/test'
    output = 'I:/dataset/delicious/data/test.pair.100'
    
    totalFile = 'I:/dataset/ml-100k/cleanedData/dataset.user50/u.data.sample_user50'
    testFile = 'I:/dataset/ml-100k/cleanedData/dataset.user50/train10/test'
    output = 'I:/dataset/ml-100k/cleanedData/dataset.user50/train10/test.pairs.txt'

    createTestPairsSet(totalFile, testFile, 10000, output)
    # getId('I:/dataset/ml-100k/cleanedData/fm/total','I:/dataset/ml-100k/cleanedData/fm')
    # count('I:/dataset/citation-network1/test/ref_train.rand.txt','I:/dataset/citation-network1/test/ref_test.rand.txt')
    # count('I:/dataset/ml-100k/u1.base','I:/dataset/ml-100k/u1.test')
    # filter_cold('I:/dataset/citation-network1/test/ref_train.rand.txt','I:/dataset/citation-network1/test/ref_test_pairs_rand.50.txt','I:/dataset/citation-network1/test/ref_test_pairs_rand.50.cold.txt')
    # map2Id('I:/dataset/ml-100k/cleanedData/fm/id2u','I:/dataset/ml-100k/cleanedData/fm/id2v','I:/dataset/ml-100k/cleanedData/fm/test.pairs.txt','I:/dataset/ml-100k/cleanedData/fm/')
