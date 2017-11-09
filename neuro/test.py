#! /usr/bin/env python
# coding: UTF-8
import numpy as npy

def getTimeAUC(datafile):
    fin = open(datafile, 'r')
   
    m=0
    k=0
    while 1:
        line = fin.readline()
        if not line:
            break
        if(line.find('----') != -1):
            vs = line.split('\t')
            v=float(vs[0].split(':')[1])
            m+=v
            k+=1    
    fin.close()
    print m/k  


if __name__ == "__main__":  
    getTimeAUC('I:/dataset/ml-100k/cleanedData/res/30_log2_bpr.txt')
    a=100
    b=20>a
    print b