#! /usr/bin/env python

import random 
import numpy
from numpy import linalg as LA
import string


def calF(uNode, iNode, jNode):
    tmp = numpy.dot(uNode, iNode)
    tmp2 = numpy.dot(uNode, jNode)
    tmp -= tmp2
    tmp = numpy.exp(-1 * tmp)
    tmp2 = 1 + tmp
    return -1 * tmp / tmp2;
    
    
def calObj(uNode, iNode, jNode):
    tmp = numpy.dot(uNode, iNode)
    tmp2 = numpy.dot(uNode, jNode)
    tmp -= tmp2
    tmp = numpy.exp(-1 * tmp)
    tmp2 = 1 + tmp
    return numpy.log(1.0 / tmp2);


def calNLL(uLatent, vLatent, samples):
    nll_val = 0
    for s in samples:
        nll_val += calObj(numpy.squeeze(numpy.asarray(uLatent[s[0], :])), numpy.squeeze(numpy.asarray(vLatent[s[1], :])), numpy.squeeze(numpy.asarray(vLatent[s[2], :])))   
    return -1 * nll_val;


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


def sigmoid(x):
    if (numpy.abs(x) > 36):
        if(x > 0):
            return 1.0
        else:
            return 0.0
    return 1.0 / (1.0 + numpy.exp(-1 * x));
  

def log(logFile, text):
    fout = open(logFile , 'a')
    fout.write(text + '\n')
    fout.close() ;

def load_L(srcFile, opt=0):
    fin = open(srcFile, 'r')
    res = {} 
    while 1:
        lines = fin.readlines(1000000)
        if not lines:
            break
        for line in lines:
            if(opt != 0):
                opt = 0
                continue
            line = line.strip()
            tmp = line.split('\t')
            vs = tmp[1].split(',')
            t = []
            for c in range(len(vs)):
                t.append(float(vs[c]))
            res[tmp[0]] = numpy.array(t)
    fin.close()
    return res;


def load_W(srcFile):
    fin = open(srcFile, 'r')
    i = 0
    res = []
    while 1:
        lines = fin.readlines(1000000)
        if not lines:
            break
        for line in lines:
            if(i == 0):
                i += 1
                continue
            line = line.strip()
            vs = line.split(',')
            tmp = []
            for c in range(len(vs)):
                tmp.append(float(vs[c]))
            res.append(tmp)
    fin.close()
    return numpy.array(res);


def loadGraph(tieFile):
    entity = {}
    entity2 = {}
    uNodes = {}
    vNodes = {}
    
    uNodes2 = []
    vNodes2 = []
    unum = 0
    vnum = 0
    fin = open(tieFile, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        line = line.strip()
        tmp = line.split('\t')
        if(len(tmp) == 2):
            tmp.append('1')    
        if(entity.has_key(tmp[0]) == False):
            entity[tmp[0]] = {}
        entity[tmp[0]][tmp[1]] = 0
            
        if(entity2.has_key(tmp[1]) == False):
            entity2[tmp[1]] = {}
        entity2[tmp[1]][tmp[0]] = 0
        
        if(uNodes.has_key(tmp[0]) == False):
            uNodes[tmp[0]] = unum 
            unum += 1
            uNodes2.append(tmp[0])
        if(vNodes.has_key(tmp[1]) == False):
            vNodes[tmp[1]] = vnum 
            vnum += 1
            vNodes2.append(tmp[1])
                  
    fin.close()
    return entity, uNodes, vNodes, uNodes2, vNodes2, entity2;

def loadTestPairs(pairFile):
    relations = {}
    fin = open(pairFile, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        tmp = line.strip().split('\t')  
        if(relations.has_key(tmp[0]) == False):
            relations[tmp[0]] = {}        
        relations[tmp[0]][tmp[1]] = 1    
    fin.close() 
    return relations; 

def createPairsSet(srcFile, ave_num, output):
        uvInd = {}
        vs = {}
        fin = open(srcFile, 'r')
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
       
        un_observed = {}
        for u in uvInd:
            un_observed[u] = {}
            for v in uvInd[u]:
                exNum = 0
                while exNum < ave_num:
                    v = random.sample(vNodes, 1)
                    if(uvInd[u].has_key(v[0])):
                        continue
                    un_observed[u][v[0]] = 0
                    exNum += 1
        
        total = 0                             
        fout = open(output, 'w')
        for u in uvInd:
            for v in uvInd[u]:
                fout.write(u + '\t' + v + '\t' + str(uvInd[u][v]) + '\n')
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
 

def samplePairsByItem(srcFile, setting, output): 
    fin = open(srcFile, 'r')
    vuInd = {}
    uvInd = {}
    vs = {}
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        if(uvInd.has_key(uv[0]) == False):
            uvInd[uv[0]] = {}
        uvInd[uv[0]][uv[1]] = 1
        
        if(vuInd.has_key(uv[1]) == False):
            vuInd[uv[1]] = {}
            vs[uv[1]] = 0
        vuInd[uv[1]][uv[0]] = 1
        vs[uv[1]] += 1
    fin.close() 
    
    vset = {}
    upopular = {}
    vpopular = {}
    for v in vs:
        if(vs[v] > setting):
            vpopular[v] = 1
            for u in vuInd[v]:
                upopular[u] = 1
                for i in uvInd[u]:
                    vset[i] = 1
    print 'upop:', len(upopular)
    
    n = {}
    for v in vset:
        for u in vuInd[v]:
            upopular[u] = 1 
            k = True
            for v2 in uvInd[u]:
                if(vpopular.has_key(v2)):
                    k = False
                    break
            if(k):
                n[u] = 1       
    print 'upop:', len(upopular), len(n)
    

                    
def samplePairsByUser(srcFile, setting, output): 
    fin = open(srcFile, 'r')
    item_userInd = {}
    user_itemInd = {}
    users = []
    users_d = {}
    items = []
    items_d = {}
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        if(user_itemInd.has_key(uv[0]) == False):
            user_itemInd[uv[0]] = {}
            users.append(uv[0])
            users_d[uv[0]] = 0
        user_itemInd[uv[0]][uv[1]] = uv[2]
        users_d[uv[0]] += 1
        
        if(item_userInd.has_key(uv[1]) == False):
            item_userInd[uv[1]] = {}
            items.append(uv[1])
            items_d[uv[1]] = 0
        item_userInd[uv[1]][uv[0]] = uv[2]
        items_d[uv[1]] += 1
    fin.close() 
    
    random.shuffle(users)
    random.shuffle(items)
    
    fout = open(output + '_user' + str(setting), 'w')
    c_v = {}
    for i in range(setting):
        for v in user_itemInd[users[i]]:
            fout.write(users[i] + '\t' + v + '\t' + user_itemInd[users[i]][v] + '\n')
            c_v[v] = 0
    fout.close()
    print 'item_num:', len(c_v)
    
    fout = open(output + '_item' + str(setting), 'w')
    c_u = {}
    for i in range(setting):
        for u in item_userInd[items[i]]:
            fout.write(u + '\t' + items[i] + '\t'+item_userInd[items[i]][u] + '\n')
            c_u[u] = 0
    fout.close()
    print 'user_num:', len(c_u)
        
    
       
     

def splitDataSet(srcFile, rate, output):
    test = open(output + '/test', 'w')
    train = open(output + '/train', 'w')   
    fin = open(srcFile, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        if(rate < numpy.random.uniform()):
            test.write(line)
        else:
            train.write(line)
    fin.close()
    test.close()
    train.close()
    

def filterdata(srcFile, num, output):
    fin = open(srcFile, 'r')
    n = 0
    mov = {}
    tag = {}
    fout = open(output, 'w')
    while 1:
        line = fin.readline()
        if not line:
            break
        uvw = line.strip().split('\t')
        if(string.atof(uvw[2]) > num):
            mov[uvw[0]] = 1
            if(tag.has_key(uvw[1]) == False):
                tag[uvw[1]] = 0
            tag[uvw[1]] += 1
            n += 1
            fout.write(uvw[0] + '\t' + uvw[1] + '\n')
    fin.close() 
    fout.close()
    print n, len(mov), len(tag)
    

def count(f1, f2):
    vs = {}
    us = {}
    vs2 = {}
    us2 = {}
    fin = open(f1, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        us[uv[0]] = 1
        vs[uv[1]] = 1
    fin.close()
    
    fin = open(f2, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        us2[uv[0]] = 1
        vs2[uv[1]] = 1
    fin.close()
    print 'us:', len(us), 'vs:', len(vs), 'us2:', len(us2), 'vs2:', len(vs2)
    
    cu = 0
    for u in us:
        if(us2.has_key(u)):
            cu += 1
    cv = 0
    for v in vs:
        if(vs2.has_key(v)):
            cv += 1
        
    print 'cu:', cu, 'cv:', cv         
 
 
def filter_cold(f1, f2, output):
    # uvs={}
    vs = {}
    fin = open(f1, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        vs[uv[1]] = 1
    fin.close()
    
    
    
    fout = open(output, 'w')
    fin = open(f2, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        
        if(vs.has_key(uv[1])):
            continue
        fout.write(line)
    fin.close()
    fout.close()
    

def map2Id(uidF, vidF, scrFile, output):
    uid = {}
    vid = {}
    
    fin = open(uidF, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        uid[uv[1]] = uv[0]
    fin.close()
    
    fin = open(vidF, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        vid[uv[1]] = uv[0]
    fin.close()
    
    fin = open(scrFile, 'r')
    fout = open(output + '/pairs.id', 'w')
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        fout.write(uid[uv[0]] + ' ' + vid[uv[1]] + ' ' + uv[2] + '\n')
    fin.close()
    fout.close()


def getId(scrFile, output):
    uid = {}
    vid = {}
    
    fin = open(scrFile, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        if(uid.has_key(uv[0]) == False):
            uid[uv[0]] = len(uid)
        if(vid.has_key(uv[1]) == False):
            vid[uv[1]] = len(vid)  
    fin.close()

    fout = open(output + '/id2u', 'w')
    for u in uid:
        fout.write(str(uid[u]) + '\t' + u + '\n')
    fout.close()
    
    fout = open(output + '/id2v', 'w')
    for v in vid:
        fout.write(str(vid[v]) + '\t' + v + '\n')
    fout.close()
    

if __name__ == "__main__":
    srcFile = 'I:/dataset/tag-genome/tag_relevance.dat'
    output = 'I:/dataset/tag-genome/test/demo.set'
    # filterdata(srcFile, 0.2, output)
    
    
     
    srcFile = 'I:/dataset/ml-100k/u.data'
    output = 'I:/dataset/ml-100k/u.data.sample'
    srcFile = 'H:/study_data/xusong/dblp/reg_rating_dataset.txt'
    output = 'H:/study_data/xusong/dblp/dataset.sample'
    #samplePairsByUser(srcFile, 100, output)
      
    srcFile = 'I:/dataset/ml-100k/cleanedData/dataset.user50/u.data.sample_user50'
    output = 'I:/dataset/ml-100k/cleanedData/dataset.user50/train70'
    
    splitDataSet(srcFile, 0.7, output)
     
    srcFile = 'I:/dataset/delicious/data/test'
    output = 'I:/dataset/delicious/data/test.pair.100'
    
    srcFile = 'I:/dataset/ml-100k/cleanedData/u1.test'
    output = 'I:/dataset/ml-100k/cleanedData/test_pairs.200.txt'
    # createPairsSet(srcFile, 200, output)
    # getId('I:/dataset/ml-100k/cleanedData/fm/total','I:/dataset/ml-100k/cleanedData/fm')
    # count('I:/dataset/citation-network1/test/ref_train.rand.txt','I:/dataset/citation-network1/test/ref_test.rand.txt')
    # count('I:/dataset/ml-100k/u1.base','I:/dataset/ml-100k/u1.test')
    # filter_cold('I:/dataset/citation-network1/test/ref_train.rand.txt','I:/dataset/citation-network1/test/ref_test_pairs_rand.50.txt','I:/dataset/citation-network1/test/ref_test_pairs_rand.50.cold.txt')
    # map2Id('I:/dataset/ml-100k/cleanedData/fm/id2u','I:/dataset/ml-100k/cleanedData/fm/id2v','I:/dataset/ml-100k/cleanedData/fm/test.pairs.txt','I:/dataset/ml-100k/cleanedData/fm/')
