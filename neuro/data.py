#!/usr/bin/env python
# coding: UTF-8
import string


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
    

def transform(fea):
    fs = {}
    fin = open(fea, 'r')
    flen = string.atoi(fin.readline().strip())
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            k, vs = line.split('\t' , 1)
            fs[k] = []
            vals = vs.split(' ')
            if(len(vals) != flen):
                print 'error feature!'
                exit(0)
            no = 0
            for v in vals:
                if(float(v) != 0):
                    fs[k].append(str(no) + ':' + v)   
                no += 1    
    fin.close()
    return fs, flen;


def transform2(fea):
    fs = {}
    fin = open(fea, 'r')
    flen = string.atoi(fin.readline().strip())
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            k, vs = line.split('\t' , 1)
            fs[k] = []
            vals = vs.split(' ')
            s = 0
            tmp = []
         
            for v in vals:
                s += float(v)
                tmp.append(float(v))
            for no in range(len(tmp)):
                if(tmp[no] / s > 0.001):
                    fs[k].append(str(no) + ':' + str(tmp[no] / s))   
    fin.close()
    return fs, flen;

def transformRelation(graph, fea):
    u2v = {}
    v2u = {}
    uids = {}
    vids = {}
    fin = open(graph, 'r')
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            kv = line.split('\t')
            if(uids.has_key(kv[0]) == False):
                uids[kv[0]] = str(len(uids))
            if(vids.has_key(kv[1]) == False):
                vids[kv[1]] = str(len(vids))
    fin.close()
    
    fin = open(fea, 'r')
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            kv = line.split('\t')
            if(u2v.has_key(kv[0]) == False):
                u2v[kv[0]] = []
            if(v2u.has_key(kv[1]) == False):
                v2u[kv[1]] = []
            u2v[kv[0]].append(vids[kv[1]] + ':' + kv[2])
            v2u[kv[1]].append(uids[kv[0]] + ':' + kv[2]) 
    fin.close()
    return u2v, v2u, len(uids), len(vids);


def assembleFeature(fea, fea2):
    fs = {}
    fin = open(fea, 'r')
    flen = string.atoi(fin.readline().strip())
    
    fin2 = open(fea2, 'r')
    flen2 = string.atoi(fin2.readline().strip())
    
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            k, vs = line.split('\t' , 1)
            fs[k] = ['0' for col in range(flen + flen2)]
            vals = vs.split(',')
            if(len(vals) != flen):
                print 'error feature!'
                exit(0)
            for i in range(len(vals)):
                fs[k][i] = vals[i]       
    fin.close()
    
    while 1:
        lines = fin2.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            k, vs = line.split('\t' , 1)
            vals = vs.split(',')
            if(len(vals) != flen2):
                print 'error feature!'
                exit(0)
            if(fs.has_key(k) == False):
                fs[k] = ['0' for col in range(flen + flen2)]
            for i in range(len(vals)):
                fs[k][i + flen] = vals[i]         
    fin2.close()
    return fs, flen + flen2;


def assembleSparseFeature(fea, fea2):
    fs = {}
    fin = open(fea, 'r')
    flen = string.atoi(fin.readline().strip())
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            k, vs = line.split('\t' , 1)
            fs[k] = []
            for v in vs.split(','):
                fs[k].append(v)       
    fin.close()
    
    fin = open(fea2, 'r')
    flen2 = string.atoi(fin.readline().strip())
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            k, vs = line.split('\t' , 1)
            if(fs.has_key(k) == False):
                fs[k] = []
            for v in vs.split(','):
                n, val = v.split(':')
                fs[k].append(str(string.atoi(n) + flen) + ':' + val)       
    fin.close()
    return fs, flen2 + flen;


def saveFeature(res, flen, output):
    fout = open(output, 'w')
    fout.write(str(flen) + '\n')
    for i in res:
        tmp = i + '\t'
        for v in res[i]:
            tmp += v + ','
        fout.write(tmp[0:-1] + '\n')
    fout.close()

        
    
###########################################################
class feature:
    def __init__(self):
        self.indx = {}
        self.values = []
    
    def add(self, fid, v):
        self.indx[fid] = len(self.indx)
        self.values.append(v)



############################################################
class data:
    def loadGraph(self, graphFile, dim=2, label=True, rat=-10):
        graph = {}
        fin = open(graphFile, 'r')
        while 1:
            lines = fin.readlines(100000)
            if not lines:
                break
            for line in lines:
                line = line.strip()
                vs = line.split('\t')[0:dim]
                if(string.atoi(vs[2])>=rat):
                    createGraph(vs, graph, label)
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

######################################################################   
def relFeature():  
    userR, itemR, ulen, vlen = transformRelation('I:/dataset/ml-100k/u.data', 'I:/dataset/ml-100k/u1.base')
    saveFeature(userR, vlen, 'I:/dataset/ml-100k/cleanedData/user.u1.base.rel')
    saveFeature(itemR, ulen, 'I:/dataset/ml-100k/cleanedData/item.u1.base.rel')  
########################################################################


def assembleFes():
    res, flen = assembleFeature('I:/dataset/ml-100k/cleanedData/user/user.occus', 'I:/dataset/ml-100k/cleanedData/user/user.genders')
    saveFeature(res, flen, 'I:/dataset/ml-100k/cleanedData/tmp')
    res, flen = assembleFeature('I:/dataset/ml-100k/cleanedData/tmp', 'I:/dataset/ml-100k/cleanedData/user/user.ages')
    saveFeature(res, flen, 'I:/dataset/ml-100k/cleanedData/tmp2')

def trans():
    res, flen = transform('I:/dataset/ml-100k/cleanedData/tmp2')
    saveFeature(res, flen, 'I:/dataset/ml-100k/cleanedData/tmp3')

def mkFeature():
    res, flen = assembleSparseFeature('I:/dataset/ml-100k/cleanedData/tmp3', 'I:/dataset/ml-100k/cleanedData/user.u1.base.rel')
    saveFeature(res, flen, 'I:/dataset/ml-100k/cleanedData/user.feature.base')

###################################################################################
def assembleFes_item():
    res, flen = assembleFeature('I:/dataset/ml-100k/cleanedData/item/item.genres', 'I:/dataset/ml-100k/cleanedData/item/item.years')
    saveFeature(res, flen, 'I:/dataset/ml-100k/cleanedData/tmp')
    #===========================================================================
    # res, flen = assembleFeature('I:/dataset/ml-100k/cleanedData/tmp', 'I:/dataset/ml-100k/cleanedData/item/titles/itemTitle3')
    # saveFeature(res, flen, 'I:/dataset/ml-100k/cleanedData/tmp2')
    #===========================================================================

def trans_item():
    res, flen = transform('I:/dataset/ml-100k/cleanedData/tmp')
    saveFeature(res, flen, 'I:/dataset/ml-100k/cleanedData/tmp2')

def mkFeature_item():
    res, flen = assembleSparseFeature('I:/dataset/ml-100k/cleanedData/tmp2', 'I:/dataset/ml-100k/cleanedData/item/titles/itemTitle.org')
    saveFeature(res, flen, 'I:/dataset/ml-100k/cleanedData/tmp3')
    res, flen = assembleSparseFeature('I:/dataset/ml-100k/cleanedData/tmp3', 'I:/dataset/ml-100k/cleanedData/item.u1.base.rel')
    saveFeature(res, flen, 'I:/dataset/ml-100k/cleanedData/item.feature.base')

###########################################################################33
def addId(ids, titles):
    fin = open(ids, 'r')
    ids = []
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            ids.append(line)
    fin.close()
    fm = 0
    fin = open(titles, 'r')
    fout = open('I:/dataset/ml-100k/cleanedData/item/titles/itemTitle.org', 'w')
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        i = 0
        for line in lines:
            vs = line.strip().split(' ')
            tmp = ids[i] + '\t'
            i += 1
            for v in vs[1:]:
                tmp += v + ','
                k, n = v.split(':')
                if(string.atoi(k) > fm):
                    fm = string.atoi(k)
            fout.write(tmp[0:-1] + '\n')
    print fm            
    fout.close()        
    fin.close()


def proTittle(titles):
    fin = open(titles, 'r')
    fout = open('I:/dataset/ml-100k/cleanedData/item/titles/itemTitle2', 'w')
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            vs = line.strip().split(' ')
            tmp = ''
            
            for v in vs[1:]:
                k, t = v.split(':')
                for j in range(string.atoi(t)):
                    tmp += k + ' '
            fout.write(tmp[0:-1] + '\n')          
    fout.close()        
    fin.close()

def addCol(col1, col2, output):
    fin = open(col1, 'r')
    ids = []
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            ids.append(line)
    fin.close()
    
    fin = open(col2, 'r')
    fout = open(output, 'w')
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        i = 0
        for line in lines:
            tmp = ids[i] + '\t' + line.strip()
            i += 1
            fout.write(tmp[0:-1] + '\n')           
    fout.close()        
    fin.close()
  
  
def getDBLPFeature(srcFile, output):
    fin = open(srcFile, 'r')
    fout = open(output, 'w')
    fmax = 0
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            vs = line.split(' ')
            tmp = vs[0] + '\t'
            flen = 0.01
            fkv = {}
            for v in vs[1:]:
                ks = v.split(':')
                flen += string.atof(ks[1])
                
                if(fmax < string.atoi(ks[0])):
                    fmax = string.atoi(ks[0])
                
                if(fkv.has_key(ks[0]) == False):   
                    fkv[ks[0]] = 0
                fkv[ks[0]] += string.atof(ks[1]) / flen
            
            for k in fkv:
                tmp += k + ':' + str(fkv[k]) + ','
            fout.write(tmp[:-1] + '\n')   
    fin.close()
    fout.close()
    print fmax
      
  
def getDBLPFeature2(srcFile, output):
    fin = open(srcFile, 'r')
    fout_id = open(output + '_id', 'w')
    fout_w = open(output + '_w', 'w')
    while 1:
        lines = fin.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip()
            vs = line.split(' ')
            tmp = str(len(vs) - 1) + ' '
            
            for v in vs[1:]:
                tmp += v + ' '
            fout_w.write(tmp[:-1] + '\n') 
            fout_id.write(vs[0] + '\n')   
    fin.close()
    fout_w.close()
    
  
   
if __name__ == "__main__":
    col1 = 'H:/study_data/xusong/dblp/dataset/author.feature.X1_id'
    col2 = 'H:/study_data/xusong/dblp/dataset/res/020.gamma'
    output = 'H:/study_data/xusong/dblp/dataset/author.feature.lda.100'
    
    #addCol(col1, col2, output)
    
    res, flen = transform2(output)
    saveFeature(res, flen, 'H:/study_data/xusong/dblp/dataset/author.feature.X.lda100')
    
    srcFile = 'H:/study_data/xusong/dblp/ap_pub.dat'
    output = 'H:/study_data/xusong/dblp/pub.feature.X1'
    # getDBLPFeature(srcFile, output)
    # addId('I:/dataset/ml-100k/cleanedData/item/titles/item.title.id','I:/dataset/ml-100k/cleanedData/item/titles/item.titles')
    # getDBLPFeature2(srcFile, output)
    exit(0)
    #===========================================================================
    relFeature()
    assembleFes()
    trans()
    mkFeature()
    #===========================================================================
    # exit(0)
    assembleFes_item()
    trans_item()
    mkFeature_item()
    
