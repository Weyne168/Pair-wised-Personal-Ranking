#! /usr/bin/env python
# coding: UTF-8
import copy
import numpy as npy
from utils import sigmoid
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from scipy import sparse


def calPairLoss(uNode, iNode, jNode):
    uvec = npy.squeeze(npy.asarray(uNode))
    ivec = npy.squeeze(npy.asarray(iNode))
    jvec = npy.squeeze(npy.asarray(jNode))
    x1 = npy.dot(uvec, ivec)
    x2 = npy.dot(uvec, jvec)
    return sigmoid(x1 - x2)


def calMat3(para):
    A = para[0]
    Y = csr_matrix(para[1])
    M = inv(A.T * A + para[2] * sparse.eye(A.shape[1])) * A.T * Y 
    return M;
   
    
def SGD_BPR(para):
    samples = para[0]
    
    f_user_set = para[1][0]
    f_item_set = para[1][1]
    user_set = para[1][2]
    item_set = para[1][3]
   
    userFeatures = para[2][0]
    itemFeatures = para[2][1]
    
    userMatrix = para[2][2]  # k*f
    itemMatrix = para[2][3]
    
    uLatent = copy.deepcopy(para[2][4])
    vLatent = copy.deepcopy(para[2][5])
    
    lambdaUw = para[3][0]
    lambdaIw = para[3][1]
    lambdaU = para[3][2]
    lambdaVi = para[3][3]
    lambdaVj = para[3][4]
   
    lrate = para[4]
    Z = para[5]
   
    usum = {}
    vsum = {}
    for s in samples:
        if(f_user_set.has_key(s[0]) and user_set.has_key(s[0]) and usum.has_key(s[0]) == False):
            tmp = userFeatures.getrow(user_set[s[0]]) * userMatrix
            tmp = lambdaUw * (uLatent[user_set[s[0]], :] - tmp)
            usum[s[0]] = npy.squeeze(npy.asarray(tmp))
                   
        if(f_item_set.has_key(s[1]) and item_set.has_key(s[1]) and vsum.has_key(s[1]) == False):
            tmp = itemFeatures.getrow(item_set[s[1]]) * itemMatrix 
            tmp = lambdaIw * (vLatent[item_set[s[1]], :] - tmp)      
            vsum[s[1]] = npy.squeeze(npy.asarray(tmp))
                
        if(f_item_set.has_key(s[2]) and item_set.has_key(s[2]) and vsum.has_key(s[2]) == False):
            tmp = itemFeatures.getrow(item_set[s[2]]) * itemMatrix
            tmp = vLatent[item_set[s[2]], :] - tmp    
            vsum[s[2]] = npy.squeeze(npy.asarray(tmp))
        
        u = user_set[s[0]]
        vi = item_set[s[1]]
        vj = item_set[s[2]]
        f = calPairLoss(uLatent[u, :], vLatent[vi, :], vLatent[vj, :]) - 1
        for z in range(0, Z):
            grav = f * (vLatent[vi, z] - vLatent[vj, z]) + lambdaU * uLatent[u, z] 
            if(usum.has_key(s[0])):
                grav += usum[s[0]][z]
            uLatent[u, z] = uLatent[u, z] - lrate * grav
            
            grav = f * uLatent[u, z] + lambdaVi * vLatent[vi, z]
            if(vsum.has_key(s[1])):
                grav += vsum[s[1]][z]
            vLatent[vi, z] = vLatent[vi, z] + lrate * grav
            
            grav = -1.0 * f * uLatent[u, z] + lambdaVj * vLatent[vi, z]
            if(vsum.has_key(s[2])):
                grav += vsum[s[2]][z]
            vLatent[vj, z] = vLatent[vj, z] + lrate * grav
    return uLatent, vLatent;   


if __name__ == "__main__": 
    row = npy.array([0, 3, 0])
    col = npy.array([0, 1, 5])
    val = npy.array([2, 5.1, 2.2])
    E = npy.eye(6)
    E = npy.power(npy.matrix([[1, 1, 2, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),2)
    A = csr_matrix((val, (row, col)), shape=(6, 6))
    # print inv(A+sparse.eye(A.shape[1])).toarray()
    
    print A.getrow(0).nonzero()
    exit(0)
    print npy.power(A.getrow(0).toarray(),2)
    rc = npy.squeeze(npy.asarray(A.getrow(0) * npy.zeros(6).T ))
    print rc
    # print rc
    exit(0)
    ss = []
    B = npy.squeeze(npy.asarray(A.toarray()))
    print B[1][1]
    C = [1.0  for col in range(6)]
    ss.append(B)
    ss.append(C)
    # print ss[1][3]
    
    E = csr_matrix(npy.eye(6)).T
    print E
    
    # print A.T.toarray()
