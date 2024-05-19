import numpy as np
from numba import njit,types

def Quinn_Fernandes_extrapolation(xx, yy, Npast, Nfut, Nharm = 10, FreqTOL = 0.000001 , MaxIterations = 10000):
    #---- Initialization
    N = Npast#max(Npast,Nfut+1);
    av = sum(yy)/Npast;
    xm = np.ones(N)*av
    ym = np.ones(Nfut)*av

#+------------------------------------------------------------------+
#|      Quinn and Fernandes algorithm for finding frequency         |
#+------------------------------------------------------------------+
    def Freq(x,n):
        z=np.zeros(n);
        alpha=0.0;
        beta=2.0;
        z[0]=x[0]-xm[0];
        iterations = 0
        while (abs(alpha-beta)>FreqTOL) and (iterations < MaxIterations ):
            iterations += 1
            alpha=beta;
            z[1]=x[1]-xm[1]+alpha*z[0];
            num=z[0]*z[1]; #
            den=z[0]*z[0]; #
            for i in range(2,n):
                z[i]=x[i]-xm[i]+alpha*z[i-1]-z[i-2];
                num=num+z[i-1]*(z[i]+z[i-2]);
                den=den+z[i-1]*z[i-1];
            beta=num/den;
        if beta>2:
            print("!!! beta = " , beta, "!!!") 
            beta = 2
        w=np.arccos(beta/2.0);
        r=TrigFit(x,n,w);
        return(np.append([w],r))

    def TrigFit(x,n,w):
        Sc =0.0;
        Ss =0.0;
        Scc=0.0;
        Sss=0.0;
        Scs=0.0;
        Sx =0.0;
        Sxc=0.0;
        Sxs=0.0;
        for i in range(n):
            c=np.cos(w*(i+1));
            s=np.sin(w*(i+1));
            dx=x[i]-xm[i];
            Sc=Sc+c;
            Ss=Ss+s;
            Scc=Scc+c*c;
            Sss=Sss+s*s;
            Scs=Scs+c*s;
            Sx=Sx+dx;
            Sxc=Sxc+dx*c;
            Sxs=Sxs+dx*s;
        Sc=Sc/n;
        Ss=Ss/n;
        Scc=Scc/n;
        Sss=Sss/n;
        Scs=Scs/n;
        Sx=Sx/n;
        Sxc=Sxc/n;
        Sxs=Sxs/n;
        if(w==0.0):
            m=Sx;
            a=0.0;
            b=0.0;
        else:
        # calculating a, b, and m
            den=(Scs-Sc*Ss)**2-(Scc-Sc*Sc)*(Sss-Ss*Ss);
            a=((Sxs-Sx*Ss)*(Scs-Sc*Ss)-(Sxc-Sx*Sc)*(Sss-Ss*Ss))/den;
            b=((Sxc-Sx*Sc)*(Scs-Sc*Ss)-(Sxs-Sx*Ss)*(Scc-Sc*Sc))/den;
            m=Sx-a*Sc-b*Ss;
        return([m,a,b])

    #--- fit trigonometric model and calculate predictions
    for harm in range(Nharm): 
        r=Freq(yy,Npast);
        w=r[0];
        m=r[1];
        a=r[2];
        b=r[3];
        for i in range(N):
            xm[i]=xm[i]+m+a*np.cos(w*i)+b*np.sin(w*i);
        for j in range(Nfut):
            ym[j]=ym[j]+m+a*np.cos(w*(N+j))+b*np.sin(w*(N+j));
    return xm,ym


#@njit( types.UniTuple( types.Array(types.float64, 1, "C") , 2) (types.Array(types.int32, 1, "C"), types.Array(types.float64, 1, "C"), types.int64, types.int64, types.int64, types.float64, types.int64), cache=True, fastmath=False)

@njit(cache=True, fastmath=False)
def Quinn_Fernandes_extrapolation_njit(xx, yy, Npast, Nfut, Nharm = 10, FreqTOL = 0.000001 , MaxIterations = 10000):
    #---- Initialization
    N = Npast#max(Npast,Nfut+1);
    av = np.sum(yy)/Npast;
    xm = np.ones(N)*av
    ym = np.ones(Nfut)*av

#+------------------------------------------------------------------+
#|      Quinn and Fernandes algorithm for finding frequency         |
#+------------------------------------------------------------------+

    #--- fit trigonometric model and calculate predictions
    for harm in range(Nharm): 

        #======================================================================
        #r=Freq(yy,Npast);
        #def Freq(x = yy, n = Npast):
        z=np.zeros(Npast);
        alpha=0.0;
        beta=2.0;
        z[0]=yy[0]-xm[0];
        iterations = 0
        while (abs(alpha-beta)>FreqTOL) and (iterations < MaxIterations ):
            iterations += 1
            alpha=beta;
            z[1]=yy[1]-xm[1]+alpha*z[0];
            num=z[0]*z[1]; #
            den=z[0]*z[0]; #
            for i in range(2,Npast):
                z[i]=yy[i]-xm[i]+alpha*z[i-1]-z[i-2];
                num=num+z[i-1]*(z[i]+z[i-2]);
                den=den+z[i-1]*z[i-1];
            beta=num/den;
        if beta>2:
            #print("!!! beta = " , beta, "!!!") 
            beta = 2
        w=np.arccos(beta/2.0);
        # =========================================================
        #r=TrigFit(x,n,w);
        #def TrigFit(x,n,w):
        Sc =0.0;
        Ss =0.0;
        Scc=0.0;
        Sss=0.0;
        Scs=0.0;
        Sx =0.0;
        Sxc=0.0;
        Sxs=0.0;
        for i in range(Npast):
            c=np.cos(w*(i+1));
            s=np.sin(w*(i+1));
            dx=yy[i]-xm[i];
            Sc=Sc+c;
            Ss=Ss+s;
            Scc=Scc+c*c;
            Sss=Sss+s*s;
            Scs=Scs+c*s;
            Sx=Sx+dx;
            Sxc=Sxc+dx*c;
            Sxs=Sxs+dx*s;
        Sc=Sc/Npast;
        Ss=Ss/Npast;
        Scc=Scc/Npast;
        Sss=Sss/Npast;
        Scs=Scs/Npast;
        Sx=Sx/Npast;
        Sxc=Sxc/Npast;
        Sxs=Sxs/Npast;
        if(w==0.0):
            m=Sx;
            a=0.0;
            b=0.0;
        else:
        # calculating a, b, and m
            den=(Scs-Sc*Ss)**2-(Scc-Sc*Sc)*(Sss-Ss*Ss);
            a=((Sxs-Sx*Ss)*(Scs-Sc*Ss)-(Sxc-Sx*Sc)*(Sss-Ss*Ss))/den;
            b=((Sxc-Sx*Sc)*(Scs-Sc*Ss)-(Sxs-Sx*Ss)*(Scc-Sc*Sc))/den;
            m=Sx-a*Sc-b*Ss;

        # ===================================================
        for i in range(N):
            xm[i]=xm[i]+m+a*np.cos(w*i)+b*np.sin(w*i);
        for j in range(Nfut):
            ym[j]=ym[j]+m+a*np.cos(w*(N+j))+b*np.sin(w*(N+j));
    return xm,ym