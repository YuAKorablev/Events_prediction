import numpy as np
import cvxopt
from functools import lru_cache
from numba import njit,types
from MyLemke2 import *

@lru_cache(maxsize=None)
def int_spline(t, Y, s=None, m=None, W=None, alpha=10**5, x=None, info=False ):
  # t - array of observation coordinates (beginning of integrals) 
  # Y - array of observation, where Y[i] is integral of unknow function from t[i] to t[i+1]
  # s - spline knots
  # m - number of spline knots
  # W - weights of observation
  # alpha - smoothing parameter
  # x - coordinates in which spline will be calculated
    if s is None:
        s = t
    if m is None:
        m = len(s)
    if W is None:
        W = np.ones(len(Y))
    if x is None:
        x = np.append( np.arange(t[0],t[len(t)-1],1) , t[len(t)-1])
    
    n = len(t) #number of observation coordinates   
    # in case s is not defined
    if m != len(s): #its possible when m was defined, but s wasn't
        s = np.linspace(t[0],t[n-1],m)
  
    h = np.zeros(m-1) #array of distance between knots
    h[0:(m-1)]=s[1:m]-s[0:(m-1)]

    #Matrix Q
    Q = np.zeros( (m, m-2) )
    for i in range(m-2):
        Q[i,i]=1/h[i];
        Q[i+1,i]=-1/h[i]-1/h[i+1];
        Q[i+2,i]=1/h[i+1]

    #Matrix R
    R=np.zeros((m-2,m-2))
    for i in range(m-2):
        R[i,i]=(h[i]+h[i+1])/3
        if (i<m-2 -1):
            R[i+1,i]=h[i+1]/6
            R[i,i+1]=h[i+1]/6

    #Matrix K calculation
    inv_R = np.linalg.inv(R)
    t_Q = np.transpose(Q)
    K = Q @ inv_R @ t_Q

  #Filling in V and P matrices
    V=np.zeros((n-1,m))
    P=np.zeros((n-1,m))
    k=0
    while( (s[k]<=t[0]) and (s[k+1]<t[0])): #find first k, that s[k+1]>t[0]
        k=k+1
        
    for i in range(n-1):
        #finding L, it can be 0
        for L in range(m-k-1 +1):
            if t[i+1] <= s[k+L+1]:
                break;
        
        V[i,k]=(s[k+1]-t[i])**2/h[k]/2  
        P[i,k]=h[k]**3/24-(t[i]-s[k])**2*(s[k+1]-t[i]+h[k])**2/h[k]/24

        l=1;
        while l<=L:
            V[i,k+l]=(h[k+l-1]+h[k+l])/2
            P[i,k+l]=(h[k+l-1]**3+h[k+l]**3)/24
            l=l+1;

        V[i,k+1]=V[i,k+1]-(t[i]-s[k])**2/h[k]/2
        P[i,k+1]=P[i,k+1]+(t[i]-s[k])**2*((t[i]-s[k])**2-2*h[k]**2)/h[k]/24
        V[i,k+L]=V[i,k+L]-(s[k+L+1]-t[i+1])**2/h[k+L]/2
        P[i,k+L]=P[i,k+L]+(s[k+L+1]-t[i+1])**2*((s[k+L+1]-t[i+1])**2-2*h[k+L]**2)/h[k+L]/24    
        V[i,k+L+1]=(t[i+1]-s[k+L])**2/h[k+L]/2
        P[i,k+L+1]=h[k+L]**3/24-(s[k+L+1]-t[i+1])**2*(t[i+1]-s[k+L]+h[k+L])**2/h[k+L]/24
        k=k+L

    P=P[:,1:(m-1)] #don't need first and last column

    #Matrix C calculation
  
    C = V - P @ inv_R @ t_Q

    #Calculation of g and gamma
    t_C = np.transpose(C)
    W = np.diag(W)                            # Weight matrix
    A = t_C @ W @ C + alpha * K
    g = np.linalg.solve(A , t_C @ W @ Y )
    gamma = inv_R @ t_Q @ g
    #After that spline is completely defined via g and gamma

      #============== Calculating and returning spline values in x coordinates  ================
    
    #Second derivative on the edges was zero
    g2=np.hstack(([0], gamma))
    g2=np.hstack((g2, [0]))

    #x=seq(t[1],t[n],1) by default
    y=np.zeros(len(x)) 

    k=0 #index of interval 
    for j in range(len(x)):
        while x[j]<t[n-1] and x[j]>s[k]+h[k]:
            k=k+1;
        y[j] = ((x[j]-s[k])*g[k+1]+(s[k+1]-x[j])*g[k])/h[k] - 1/6*(x[j]-s[k])*(s[k+1]-x[j])*(g2[k+1]*(1+(x[j]-s[k])/h[k])+g2[k]*(1+(s[k+1]-x[j])/h[k]) )

    if info:
        sigma_y = 0
        Y_ = C @ g
        for i in range(n-1):
            sigma_y = sigma_y + W[i,i]*((Y[i]-Y_[i])/Y_[i])**2
        sigma_y = np.sqrt( sigma_y / (n-1) )
    
        #result=list(x=x,y=y,g=g,gamma=g2,s=s,h=h, Y_=Y_, sigma_y=sigma_y)
        #return (result)
        return {x:x, y:y, g:g, gamma:g2, s:s, h:h, Y_:Y_, sigma_y:sigma_y}
    else: 
        return y


def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))



@lru_cache(maxsize=None)
def monotone_int_spline(t, Y, s=None, m=None, W=None, alpha=10**5, x=None, info=False ):
    import numpy as np
  # t - array of observation coordinates (beginning of integrals) 
  # Y - array of observation, where Y[i] is integral of unknow function from t[i] to t[i+1]
  # s - spline knots
  # m - number of spline knots
  # W - weights of observation
  # alpha - smoothing parameter
  # x - coordinates in which spline will be calculated
    if s is None:
        s = t
    if m is None:
        m = len(s)
    if W is None:
        W = np.ones(len(Y))
    if x is None:
        x = np.append( np.arange(t[0],t[len(t)-1],1) , t[len(t)-1])
    
    n = len(t) #number of observation coordinates   
    # in case s is not defined
    if m != len(s): #its possible when m was defined, but s wasn't
        s = np.linspace(t[0],t[n-1],m)
  
    h = np.zeros(m-1) #array of distance between knots
    h[0:(m-1)]=s[1:m]-s[0:(m-1)]

    #Matrix Q
    Q = np.zeros( (m, m-2) )
    for i in range(m-2):
        Q[i,i]=1/h[i];
        Q[i+1,i]=-1/h[i]-1/h[i+1];
        Q[i+2,i]=1/h[i+1]

    #Matrix R
    R=np.zeros((m-2,m-2))
    for i in range(m-2):
        R[i,i]=(h[i]+h[i+1])/3
        if (i<m-2 -1):
            R[i+1,i]=h[i+1]/6
            R[i,i+1]=h[i+1]/6

    #Matrix K calculation
    inv_R = np.linalg.inv(R)
    t_Q = np.transpose(Q)
    K = Q @ inv_R @ t_Q

  #Filling in V and P matrices
    V=np.zeros((n-1,m))
    P=np.zeros((n-1,m))
    k=0
    while( (s[k]<=t[0]) and (s[k+1]<t[0])): #find first k, that s[k+1]>t[0]
        k=k+1
        
    for i in range(n-1):
        #finding L, it can be 0
        for L in range(m-k-1 +1):
            if t[i+1] <= s[k+L+1]:
                break;
        
        V[i,k]=(s[k+1]-t[i])**2/h[k]/2  
        P[i,k]=h[k]**3/24-(t[i]-s[k])**2*(s[k+1]-t[i]+h[k])**2/h[k]/24

        l=1;
        while l<=L:
            V[i,k+l]=(h[k+l-1]+h[k+l])/2
            P[i,k+l]=(h[k+l-1]**3+h[k+l]**3)/24
            l=l+1;

        V[i,k+1]=V[i,k+1]-(t[i]-s[k])**2/h[k]/2
        P[i,k+1]=P[i,k+1]+(t[i]-s[k])**2*((t[i]-s[k])**2-2*h[k]**2)/h[k]/24
        V[i,k+L]=V[i,k+L]-(s[k+L+1]-t[i+1])**2/h[k+L]/2
        P[i,k+L]=P[i,k+L]+(s[k+L+1]-t[i+1])**2*((s[k+L+1]-t[i+1])**2-2*h[k+L]**2)/h[k+L]/24    
        V[i,k+L+1]=(t[i+1]-s[k+L])**2/h[k+L]/2
        P[i,k+L+1]=h[k+L]**3/24-(s[k+L+1]-t[i+1])**2*(t[i+1]-s[k+L]+h[k+L])**2/h[k+L]/24
        k=k+L

    P=P[:,1:(m-1)] #don't need first and last column

    #Matrix C calculation
  
    C = V - P @ inv_R @ t_Q

    #Calculation of g and gamma
    t_C = np.transpose(C)
    W = np.diag(W)                            # Weight matrix
    A = t_C @ W @ C + alpha * K
    #g = np.linalg.solve(A , t_C @ W @ Y )
    g = cvxopt_solve_qp(A, 
                        - t_C @ W @ Y, 
                        -np.eye(m), 
                        np.zeros(m))
    if g is None:
        print("g is None")
        print("alpha = ", alpha)
        return None
        
    gamma = inv_R @ t_Q @ g
    #After that spline is completely defined via g and gamma

      #============== Calculating and returning spline values in x coordinates  ================
    
    #Second derivative on the edges was zero
    g2=np.hstack(([0], gamma))
    g2=np.hstack((g2, [0]))

    #x=seq(t[1],t[n],1) by default
    y=np.zeros(len(x)) 

    k=0 #index of interval 
    for j in range(len(x)):
        while x[j]<t[n-1] and x[j]>s[k]+h[k]:
            k=k+1;
        y[j] = ((x[j]-s[k])*g[k+1]+(s[k+1]-x[j])*g[k])/h[k] - 1/6*(x[j]-s[k])*(s[k+1]-x[j])*(g2[k+1]*(1+(x[j]-s[k])/h[k])+g2[k]*(1+(s[k+1]-x[j])/h[k]) )

    if info:
        sigma_y = 0
        Y_ = C @ g
        for i in range(n-1):
            sigma_y = sigma_y + W[i,i]*((Y[i]-Y_[i])/Y_[i])**2
        sigma_y = np.sqrt( sigma_y / (n-1) )
    
        #result=list(x=x,y=y,g=g,gamma=g2,s=s,h=h, Y_=Y_, sigma_y=sigma_y)
        #return (result)
        return {x:x, y:y, g:g, gamma:g2, s:s, h:h, Y_:Y_, sigma_y:sigma_y}
    else: 
        return y


#@lru_cache(maxsize=None)
def monotone_int_spline_Lemke(t, Y, s=None, m=None, W=None, alpha=10**5, x=None, info=False ):
  # t - array of observation coordinates (beginning of integrals) 
  # Y - array of observation, where Y[i] is integral of unknow function from t[i] to t[i+1]
  # s - spline knots
  # m - number of spline knots
  # W - weights of observation
  # alpha - smoothing parameter
  # x - coordinates in which spline will be calculated
    if s is None:
        s = t
    if m is None:
        m = len(s)
    if W is None:
        W = np.ones(len(Y))
    if x is None:
        x = np.append( np.arange(t[0],t[len(t)-1],1) , t[len(t)-1])
    
    n = len(t) #number of observation coordinates   
    # in case s is not defined
    if m != len(s): #its possible when m was defined, but s wasn't
        s = np.linspace(t[0],t[n-1],m)
  
    h = np.zeros(m-1) #array of distance between knots
    h[0:(m-1)]=s[1:m]-s[0:(m-1)]

    #Matrix Q
    Q = np.zeros( (m, m-2) )
    for i in range(m-2):
        Q[i,i]=1/h[i];
        Q[i+1,i]=-1/h[i]-1/h[i+1];
        Q[i+2,i]=1/h[i+1]

    #Matrix R
    R=np.zeros((m-2,m-2))
    for i in range(m-2):
        R[i,i]=(h[i]+h[i+1])/3
        if (i<m-2 -1):
            R[i+1,i]=h[i+1]/6
            R[i,i+1]=h[i+1]/6

    #Matrix K calculation
    inv_R = np.linalg.inv(R)
    t_Q = np.transpose(Q)
    K = Q @ inv_R @ t_Q

  #Filling in V and P matrices
    V=np.zeros((n-1,m))
    P=np.zeros((n-1,m))
    k=0
    while( (s[k]<=t[0]) and (s[k+1]<t[0])): #find first k, that s[k+1]>t[0]
        k=k+1
        
    for i in range(n-1):
        #finding L, it can be 0
        for L in range(m-k-1 +1):
            if t[i+1] <= s[k+L+1]:
                break;
        
        V[i,k]=(s[k+1]-t[i])**2/h[k]/2  
        P[i,k]=h[k]**3/24-(t[i]-s[k])**2*(s[k+1]-t[i]+h[k])**2/h[k]/24

        l=1;
        while l<=L:
            V[i,k+l]=(h[k+l-1]+h[k+l])/2
            P[i,k+l]=(h[k+l-1]**3+h[k+l]**3)/24
            l=l+1;

        V[i,k+1]=V[i,k+1]-(t[i]-s[k])**2/h[k]/2
        P[i,k+1]=P[i,k+1]+(t[i]-s[k])**2*((t[i]-s[k])**2-2*h[k]**2)/h[k]/24
        V[i,k+L]=V[i,k+L]-(s[k+L+1]-t[i+1])**2/h[k+L]/2
        P[i,k+L]=P[i,k+L]+(s[k+L+1]-t[i+1])**2*((s[k+L+1]-t[i+1])**2-2*h[k+L]**2)/h[k+L]/24    
        V[i,k+L+1]=(t[i+1]-s[k+L])**2/h[k+L]/2
        P[i,k+L+1]=h[k+L]**3/24-(s[k+L+1]-t[i+1])**2*(t[i+1]-s[k+L]+h[k+L])**2/h[k+L]/24
        k=k+L

    P=P[:,1:(m-1)] #don't need first and last column

    #Matrix C calculation
  
    C = V - P @ inv_R @ t_Q

    #Calculation of g and gamma
    t_C = np.transpose(C)
    W = np.diag(W)                            # Weight matrix
    A = t_C @ W @ C + alpha * K
    #g = np.linalg.solve(A , t_C @ W @ Y )
    #g = cvxopt_solve_qp(A, - t_C @ W @ Y, -np.eye(m), np.zeros(m))
    g, exit_code, exit_string = Lemke_njit(A, -t_C @ W @ Y, maxIter = 10000)
    
    if g[0] == np.nan:
        print("g is np.nan")
        print("alpha = ", alpha)
        return np.array([np.nan])
        
    gamma = inv_R @ t_Q @ g
    #After that spline is completely defined via g and gamma

      #============== Calculating and returning spline values in x coordinates  ================
    
    #Second derivative on the edges was zero
    g2=np.hstack(([0], gamma))
    g2=np.hstack((g2, [0]))

    #x=seq(t[1],t[n],1) by default
    y=np.zeros(len(x)) 

    k=0 #index of interval 
    for j in range(len(x)):
        while x[j]<t[n-1] and x[j]>s[k]+h[k]:
            k=k+1;
        y[j] = ((x[j]-s[k])*g[k+1]+(s[k+1]-x[j])*g[k])/h[k] - 1/6*(x[j]-s[k])*(s[k+1]-x[j])*(g2[k+1]*(1+(x[j]-s[k])/h[k])+g2[k]*(1+(s[k+1]-x[j])/h[k]) )

    if info:
        sigma_y = 0
        Y_ = C @ g
        for i in range(n-1):
            sigma_y = sigma_y + W[i,i]*((Y[i]-Y_[i])/Y_[i])**2
        sigma_y = np.sqrt( sigma_y / (n-1) )
    
        #result=list(x=x,y=y,g=g,gamma=g2,s=s,h=h, Y_=Y_, sigma_y=sigma_y)
        #return (result)
        return {x:x, y:y, g:g, gamma:g2, s:s, h:h, Y_:Y_, sigma_y:sigma_y}
    else: 
        return y


#@lru_cache(maxsize=None)
@njit(types.Array(types.float64, 1, "C")(types.Array(types.int32, 1, "C"), types.Array(types.float64, 1, "C"), types.int64, types.float64), cache=True, fastmath=False)
#@njit("float64[:](int32[:],float64[:],int64,float64)", cache=True, fastmath=False)
#@njit(cache=True, fastmath=False)
def monotone_int_spline_Lemke_njit(t, Y, m, alpha):

    x = np.append( np.arange(t[0],t[len(t)-1],1) , t[len(t)-1])
    
    n = len(t) #number of observation coordinates   
    s = np.linspace(t[0],t[n-1],m)
  
    h = np.zeros(m-1) #array of distance between knots
    h[0:(m-1)]=s[1:m]-s[0:(m-1)]

    #Matrix Q
    Q = np.zeros( (m, m-2) )
    for i in range(m-2):
        Q[i,i]=1/h[i];
        Q[i+1,i]=-1/h[i]-1/h[i+1];
        Q[i+2,i]=1/h[i+1]

    #Matrix R
    R=np.zeros((m-2,m-2))
    for i in range(m-2):
        R[i,i]=(h[i]+h[i+1])/3
        if (i<m-2 -1):
            R[i+1,i]=h[i+1]/6
            R[i,i+1]=h[i+1]/6

    #Matrix K calculation
    inv_R = np.linalg.inv(R)
    t_Q = np.transpose(Q)
    K = Q @ inv_R @ t_Q

  #Filling in V and P matrices
    V=np.zeros((n-1,m))
    P=np.zeros((n-1,m))
    k=0
    while( (s[k]<=t[0]) and (s[k+1]<t[0])): #find first k, that s[k+1]>t[0]
        k=k+1
        
    for i in range(n-1):
        #finding L, it can be 0
        for L in range(m-k-1 +1):
            if t[i+1] <= s[k+L+1]:
                break;
        
        V[i,k]=(s[k+1]-t[i])**2/h[k]/2  
        P[i,k]=h[k]**3/24-(t[i]-s[k])**2*(s[k+1]-t[i]+h[k])**2/h[k]/24

        l=1;
        while l<=L:
            V[i,k+l]=(h[k+l-1]+h[k+l])/2
            P[i,k+l]=(h[k+l-1]**3+h[k+l]**3)/24
            l=l+1;

        V[i,k+1]=V[i,k+1]-(t[i]-s[k])**2/h[k]/2
        P[i,k+1]=P[i,k+1]+(t[i]-s[k])**2*((t[i]-s[k])**2-2*h[k]**2)/h[k]/24
        V[i,k+L]=V[i,k+L]-(s[k+L+1]-t[i+1])**2/h[k+L]/2
        P[i,k+L]=P[i,k+L]+(s[k+L+1]-t[i+1])**2*((s[k+L+1]-t[i+1])**2-2*h[k+L]**2)/h[k+L]/24    
        V[i,k+L+1]=(t[i+1]-s[k+L])**2/h[k+L]/2
        P[i,k+L+1]=h[k+L]**3/24-(s[k+L+1]-t[i+1])**2*(t[i+1]-s[k+L]+h[k+L])**2/h[k+L]/24
        k=k+L

    #P=P[:,1:(m-1)] #don't need first and last column
    P = np.ascontiguousarray(P[:,1:(m-1)])
    
    #Matrix C calculation
  
    C = V - P @ inv_R @ t_Q

    #Calculation of g and gamma
    t_C = np.transpose(C)
    A = t_C @ C + alpha * K
    #g = np.linalg.solve(A , t_C @ W @ Y )
    #g = cvxopt_solve_qp(A, - t_C @ W @ Y, -np.eye(m), np.zeros(m))
    q = -t_C @ Y #np.array(Y) 
    g, exit_code, exit_string = Lemke_njit(A, q, maxIter = 10000)
    
    if g[0] == np.nan:
        #print("g is None")
        #print("alpha = ", alpha)
        return np.array([np.nan])
        
    gamma = inv_R @ t_Q @ g
    #After that spline is completely defined via g and gamma

      #============== Calculating and returning spline values in x coordinates  ================
    
    #Second derivative on the edges was zero
    #g2=np.hstack(([0], gamma))
    #g2=np.hstack((g2, [0]))
    g2 = np.append([0],gamma)
    g2 = np.append(g2,0)

    y=np.zeros(len(x)) 

    k=0 #index of interval 
    for j in range(len(x)):
        while x[j]<t[n-1] and x[j]>s[k]+h[k]:
            k=k+1;
        y[j] = ((x[j]-s[k])*g[k+1]+(s[k+1]-x[j])*g[k])/h[k] - 1/6*(x[j]-s[k])*(s[k+1]-x[j])*(g2[k+1]*(1+(x[j]-s[k])/h[k])+g2[k]*(1+(s[k+1]-x[j])/h[k]) )


    return y