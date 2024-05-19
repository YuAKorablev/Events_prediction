# взято у AndyLamperski и адаптировано под numba
# https://github.com/AndyLamperski/lemkelcp?ysclid=lr9md22auy264370024


import numpy as np
from numba import njit

def Lemke(M,q,maxIter = 10000):

    def step():
        #q = T[:,-1]
        a = T[:,-2]
        ind = np.nan
        minRatio = np.inf
        for i in range(n):
            if a[i] > 0:
                newRatio = q[i] / a[i]
                if newRatio < minRatio:
                    ind = i
                    minRatio = newRatio
                    
        if minRatio < np.inf:

            a = T[ind,-2]
            T[ind] /= a
            for i in range(n):
                if i != ind:
                    b = T[i,-2]
                    T[i] -= b * T[ind]
                    
            pivot(ind)
            return True
        else:
            return False
            
    def pivot(pos):
        v,ind = Tind[:,pos]
        if v == W:
            ppos = zPos[ind]
        elif v == Z:
            ppos = wPos[ind]
        else:
            ppos = None
        if ppos is not None:
            swapColumns(pos,ppos)
            swapColumns(pos,-2)
            return True
        else:
            swapColumns(pos,-2)
            return False
            
    def swapColumns(i,j):        
        v,ind = Tind[:,i]
        if v == W:
            wPos[ind] = j % (2*n+2)
        elif v == Z:
            zPos[ind] = j % (2*n+2)
            
        v,ind = Tind[:,j]
        if v == W:
            wPos[ind] = i % (2*n+2)
        elif v == Z:
            zPos[ind] = i % (2*n+2)
        
        Tind_i = Tind[:,i].copy()
        Tind[:,i] = Tind[:,j]
        Tind[:,j] = Tind_i
        
        T_i = T[:,i].copy()
        T[:,i] = T[:,j]
        T[:,j] = T_i  

        
    n = len(q)
    T = np.hstack((np.eye(n),-M,-np.ones((n,1)),q.reshape((n,1))))
    wPos = np.arange(n)
    zPos = np.arange(n,2*n)
    W = 0
    Z = 1
    Y = 2
    Q = 3
    TbInd = np.vstack((W*np.ones(n,dtype=int), 
                       np.arange(n,dtype=int)))
    TnbInd = np.vstack((Z*np.ones(n,dtype=int),
                        np.arange(n,dtype=int)))
    DriveInd = np.array([[Y],[0]])
    QInd = np.array([[Q],[0]])
    Tind = np.hstack((TbInd,TnbInd,DriveInd,QInd))

    q = T[:,-1]
    minQ = np.min(q)
    if minQ < 0:
        ind = np.argmin(q)
        a = T[ind,-2]
        T[ind] /= a
        for i in range(n):
            if i != ind:
                b = T[i,-2]
                T[i] -= b * T[ind]
        pivot(ind)
    else:
        return np.zeros(n),0,'Solution Found'
    
    
    for k in range(maxIter):
        stepVal = step()

        
        if Tind[0,-2] == Y:
            # Solution Found
            z = np.zeros(n)
            q = T[:,-1]
            for i in range(n):
                if Tind[0,i] == Z:
                    z[Tind[1,i]] = q[i]

            return z,0,'Solution Found'
        elif not stepVal:
            return np.array([np.nan]),1,'Secondary ray found'
        
    return np.array([np.nan]),2,'Max Iterations Exceeded'

@njit(cache=True, fastmath=False)
def Lemke_njit(M,q,maxIter = 10000):

    n = len(q) 
    T = np.hstack((np.eye(n),-M,-np.ones((n,1),dtype=np.int64),q.reshape((n,1))))
    wPos = np.arange(n,dtype=np.int64)
    zPos = np.arange(n,2*n,dtype=np.int64)
    W = 0  
    Z = 1 
    Y = 2 
    Q = 3 
    TbInd = np.vstack((W*np.ones(n,dtype=np.int64), 
                       np.arange(n,dtype=np.int64)))
    TnbInd = np.vstack((Z*np.ones(n,dtype=np.int64),
                        np.arange(n,dtype=np.int64)))
    DriveInd = np.array([[Y],[0]],dtype=np.int64)
    QInd = np.array([[Q],[0]],dtype=np.int64)
    Tind = np.hstack((TbInd,TnbInd,DriveInd,QInd))
    #q = T[:,-1]

    v: np.int64
    ind : np.int64
    ind2 : np.int64
    # v = Tind[0,0]
    # ind2 = Tind[1,0]
    
    minQ = np.min(q)
    if minQ < 0:
        ind = np.argmin(q)
        a = T[ind,-2]
        T[ind] /= a
        for i in range(n):
            if i != ind:
                b = T[i,-2]
                T[i] -= b * T[ind]

        v = Tind[0,ind]
        ind2 = Tind[1,ind]
        if v == W:
            ppos = zPos[ind2]
        elif v == Z:
            ppos = wPos[ind2]

        if v == W or v == Z:
                    
            v = Tind[0,ind]
            ind2 = Tind[1,ind]
            if v == W:
                wPos[ind2] = ppos % (2*n+2)
            elif v == Z:
                zPos[ind2] = ppos % (2*n+2)
                
            v = Tind[0, ppos]
            ind2 = Tind[1, ppos]
            if v == W:
                wPos[ind2] = ind % (2*n+2)
            elif v == Z:
                zPos[ind2] = ind % (2*n+2)
            
            Tind_ind = Tind[:,ind].copy()
            Tind[:,ind] = Tind[:,ppos]
            Tind[:,ppos] = Tind_ind
            
            T_ind = T[:,ind].copy()
            T[:,ind] = T[:,ppos]
            T[:,ppos] = T_ind  
       
        v = Tind[0,ind]
        ind2 = Tind[1,ind]
        if v == W:
            wPos[ind2] = (-2) % (2*n+2)
        elif v == Z:
            zPos[ind2] = (-2) % (2*n+2)
            
        v = Tind[0,-2]
        ind2 = Tind[1,-2]
        if v == W:
            wPos[ind2] = ind % (2*n+2)
        elif v == Z:
            zPos[ind2] = ind % (2*n+2)
        
        Tind_ind = Tind[:,ind].copy()
        Tind[:,ind] = Tind[:,-2]
        Tind[:,-2] = Tind_ind
        
        T_ind = T[:,ind].copy()
        T[:,ind] = T[:,-2]
        T[:,-2] = T_ind  
    

    else:
        return np.zeros(n),0,'Solution Found'

    
    
    for k in range(maxIter):
        #stepVal = step()
        #def step():
        q = T[:,-1]
        a2 = T[:,-2]
        ind : np.int32
        ind = 0
        minRatio = np.inf
        for i in range(n):
            if a2[i] > 0:
                newRatio = q[i] / a2[i]
                if newRatio < minRatio:
                    ind = i
                    minRatio = newRatio
        #ind = int(ind)            
        if minRatio < np.inf:
            a = T[int(ind),-2]
            T[int(ind)] /= a
            for i in range(n):
                if i != ind:
                    b = T[i,-2]
                    T[i] -= b * T[ind]
            
            v = Tind[0,ind]
            ind2 = Tind[1,ind]
            if v == W:
                ppos = zPos[ind2]
            elif v == Z:
                ppos = wPos[ind2]

            if v == W or v == Z:
                #swapColumns(ind,ppos)
                #def swapColumns(i=ind,j=ppos):        
                v = Tind[0,ind]
                ind2 = Tind[1,ind]
                if v == W:
                    wPos[ind2] = ppos % (2*n+2)
                elif v == Z:
                    zPos[ind2] = ppos % (2*n+2)
                    
                v = Tind[0,ppos]
                ind2 = Tind[1,ppos]
                if v == W:
                    wPos[ind2] = ind % (2*n+2)
                elif v == Z:
                    zPos[ind2] = ind % (2*n+2)
                
                Tind_ind = Tind[:,ind].copy()
                Tind[:,ind] = Tind[:,ppos]
                Tind[:,ppos] = Tind_ind
                
                T_ind = T[:,ind].copy()
                T[:,ind] = T[:,ppos]
                T[:,ppos] = T_ind 
                
            #swapColumns(ind,-2)
            #def swapColumns(i=ind,j=-2):        
            v = Tind[0,ind]
            ind2 = Tind[1,ind]
            if v == W:
                wPos[ind2] = (-2) % (2*n+2)
            elif v == Z:
                zPos[ind2] = (-2) % (2*n+2)
                
            v  = Tind[0,-2]
            ind2 = Tind[1,-2]
            if v == W:
                wPos[ind2] = ind % (2*n+2)
            elif v == Z:
                zPos[ind2] = ind % (2*n+2)
            
            Tind_ind = Tind[:,ind].copy()
            Tind[:,ind] = Tind[:,-2]
            Tind[:,-2] = Tind_ind
            
            T_ind = T[:,ind].copy()
            T[:,ind] = T[:,-2]
            T[:,-2] = T_ind 
            
            stepVal = True
        else:
            stepVal = False
        
        if Tind[0,-2] == Y:
            # Solution Found
            z = np.zeros(n)
            q = T[:,-1]
            for i in range(n):
                if Tind[0,i] == Z:
                    z[Tind[1,i]] = q[i]

            return z,0,'Solution Found'
        elif not stepVal:
            return np.array([np.nan]),1,'Secondary ray found'       
        
    return np.array([np.nan]),2,'Max Iterations Exceeded'