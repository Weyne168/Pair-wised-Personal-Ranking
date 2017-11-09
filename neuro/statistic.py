#! /usr/bin/env python
# coding: UTF-8
import string


def countRates(srcFile, pos, output):
    uvInd = {}
    fin = open(srcFile, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        if(uvInd.has_key(uv[pos]) == False and len(uv)>1):
            uvInd[uv[pos]] = 0
        uvInd[uv[pos]] += 1
    fin.close()
    
    print len(uvInd)
    vuInd = {}
    for e in uvInd:
        if(vuInd.has_key(uvInd[e]) == False):
            vuInd[uvInd[e]] = 0
        vuInd[uvInd[e]] += 1
    
    print len(vuInd)
    fout = open(output, 'w')
    for d in vuInd:
        fout.write(str(vuInd[d]) + ',' + str(d) + ',1\n')
    fout.close()
   

def countRates2(srcFile, pos, output):
    uvInd = {}
    fin = open(srcFile, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        uv = line.strip().split('\t')
        if(uvInd.has_key(uv[pos]) == False and len(uv)>1):
            uvInd[uv[pos]] = 0
        uvInd[uv[pos]] += string.atoi(uv[2])
    fin.close()
    
    print len(uvInd)
    vuInd = {}
    for e in uvInd:
        if(vuInd.has_key(uvInd[e]) == False):
            vuInd[uvInd[e]] = 0
        vuInd[uvInd[e]] += 1
    
    print len(vuInd)
    fout = open(output, 'w')
    for d in vuInd:
        fout.write(str(vuInd[d]) + ',' + str(d) + ',1\n')
    fout.close()
   
        
if __name__ == "__main__":
    srcFile = 'I:/dataset/ml-100k/u.data'
    output = 'I:/dataset/ml-100k/cleanedData/uNum-degree.csv'
    #countRates(srcFile, 0, output)   
    
    srcFile = 'H:/study_data/xusong/dblp/ap_rating.dat'
    output = 'H:/study_data/xusong/dblp/pNum-degree.csv'
    countRates2(srcFile, 1, output)   
    
    
    srcFile = 'I:/dataset/citation-network1/test/refG'
    output = 'I:/dataset/citation-network1/test/refG_p.csv'
    #countRates(srcFile, 1, output) 
    
    