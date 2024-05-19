import numpy as np
import cvxopt
import matplotlib.pyplot as plt
from int_spline import *
from Quinn_Fernandes_extrapolation import *
from numba import njit

#@njit(cache=True)
def combine_index(TTT,YYY,i):
    TTT = TTT.copy()
    YYY = YYY.copy()
    YYY[i-1] += YYY[i]
    YYY = np.delete(YYY,i)
    TTT = np.delete(TTT,i)
    return TTT,YYY

#@njit(cache=True)
def Combine_same_dates(TTT,YYY, n_days = 0):
    TTT = TTT.copy()
    YYY = YYY.copy()    
    i = 0
    while i<len(TTT)-1:
        if (TTT[i+1]-TTT[i]).days <= n_days:
            TTT,YYY = combine_index(TTT,YYY,i+1)
        else:
            i +=1
    return TTT,YYY

#@njit(cache=True)
def Check_first(TTT,YYY, n_means = 5):
    n = len(TTT)
    if n >= 3:        
        dt = [(TTT[i+1]-TTT[i]).days for i in range(n-1) ]
        mean = np.mean(dt[1:])
        if (dt[0]> mean*n_means):
          return TTT[1:],YYY[1:]
    return TTT,YYY


def get_next_event(t, Y, n_predict = 200, alpha = 10**5, n1 = 0, n2 = 0, draw_plots = False, print_info = False):
    
    if t[-1] - t[0] - n1 - n2 - 1 < 0 :
        return np.nan, np.nan    
    n=len(t)

    Yn=Y[-1] # Объем последней покупки запоминаем в отдельную переменную
    Y=Y[0:(len(Y)-1)].copy() # Last value not used
    
    m=round(3*n)
    x=np.append( np.arange(t[0],t[n-1],1) , t[n-1])
    #W = np.ones(n-1) 
    y = monotone_int_spline_Lemke_njit(t, Y, m, alpha)
    #y = monotone_int_spline_Lemke(t, Y, m=m, alpha=alpha)
    if y[0] == np.nan:
        print("monotone_int_spline_Lemke - failed and returned None")
        return np.nan, np.nan
        
    # n1 = math.floor((t[1]-t[0]))     # number of points that will be ignored on left sides
    # n2 = math.floor((t[n-1]-t[n-2])) # number of points that will be ignored on right sides
    #n_predict = 200             # number of points to extrapolate
    yy = y[(0+n1):(len(y)-n2)]
    xx = x[(0+n1):(len(y)-n2)]
    Npast=len(yy) ;     # Past values, to which trigonometric series is fitted
    Nfut    =n_predict+n2;      # Predicted future values
    Nharm   =10;      # Harmonics in model
    FreqTOL =0.000001; # Tolerance of frequency calculations
    MaxIterations = 10000
    
    y_past, y_fut =  Quinn_Fernandes_extrapolation_njit(xx, yy, Npast, Nfut, Nharm, FreqTOL , MaxIterations  )
    x3 = np.arange(0+n1+x[0], n1+x[0]+len(y_past)-1+len(y_fut)+1, 1 )
    
    if (draw_plots):
        x2 = np.array([])
        y2 = np.array([])
        for i in range(n-1):
            x2 = np.append(x2,[ t[i], t[i+1] ])
            y2 = np.append(y2, [ Y[i]/(t[i+1]-t[i]), Y[i]/(t[i+1]-t[i]) ] )
        plt.plot(x3,np.append(y_past, y_fut),color="red",label='extrapolated', linestyle='--')# type='l',ann=FALSE, xaxt="n", col="red",lty=2,ylim=range(c(y,xm,ym,y2)))
        plt.plot(x,y,color="red", label='restored')#,type="l",lwd="1",lty=1, xaxt="n",ylim=range(c(y,y2)),xlim=range(c(x,x2)))
        #axis(1, at=t,labels=MyData[[2]],las=2)
        plt.plot(x2,y2,color="grey", label='avg')#,type="l",lwd="2",lty=1)
       
    # Расчет времени, когда закончится последний запас Yn
    
    #print("Yn = ", Yn)
    Tn=t[-1]   
    if n2>=1:
        Storage=Yn-y_fut[n2-1]/2          # корректировка на f(tn)/2, так как часть от покупки Yn должна была отложиться в страховые запасы
    else:
        Storage=Yn-y[-1]/2                # тут y[-1] и y[Tn] одно и то же
        
    for i in range(n2,len(y_fut)):
        Storage -= y_fut[i]
        if Storage <= 0:
            break
    Delta_t = i+1 -n2 # тут вычитается n2 так как i считается не от 1, а от n2        

    # определение максимального запаса из данных
    max_storage = 0
    for i in range(len(Y)):
        max_storage += Y[i]-y[t[i]-t[0]]/2  # коррекция на f(ti)/2 так как предполагается, что в момент покупки запас опускается ниже критического уровня 
                                            # и часть берется из страховых запасов, а новая наблюдемая покупка Y компенсирует эту величину, 
                                            # тем самым максимальный запас меньше чем наблюдаемый объем покупки
    max_storage += Yn-y_fut[n2-1]/2 # последнего наблюдения Yn не было в массиве Y
    max_storage = max_storage / n
    
    if Storage <= 0:       
        T_future = Tn + Delta_t     
        Y_future = max_storage - Storage # будущая покупка должна восполнить страховые запасы, которые уменьшились на величину Storage       
        
        if (draw_plots):
            plt.scatter(T_future,min(y), color="green")
             
        if print_info:
            print("Delta_t = " , Delta_t, ", Storage = ", Storage)
            print("T_future = ", T_future)
            print("Y_future = " , Y_future )
            mean_dt = np.mean(np.array(t[1:n])-np.array(t[0:(n-1)]))
            print("mean_dt = ", mean_dt)

        return T_future, Y_future
    else:
        print("Yn still positive!!!")
        if (draw_plots):
            plt.scatter(Tn + Delta_t,min(y), color="red")
        return np.nan, np.nan



@njit(cache=True)
def get_next_event_njit(t, Y, n_predict = 200, alpha = 10**5, n1 = 0, n2 = 0, draw_plots = False, print_info = False):

    if t[-1] - t[0] - n1 - n2 - 1 < 0 :
        return np.nan, np.nan        
    n=len(t)

    Yn=Y[-1] # Объем последней покупки запоминаем в отдельную переменную
    Y=Y[0:(n-1)].copy() # Last value not used
    
    m=round(3*n)
    x=np.append( np.arange(t[0],t[n-1],1) , t[n-1]) 
    y = monotone_int_spline_Lemke_njit(t, Y, m, alpha)
    if y[0] == np.nan:
        #print("monotone_int_spline_Lemke - failed and returned None")
        return np.nan, np.nan
        
    yy = y[(0+n1):(len(y)-n2)]
    xx = x[(0+n1):(len(y)-n2)]
    Npast = len(yy) ;     # Past values, to which trigonometric series is fitted
    Nfut    = n_predict+n2;      # Predicted future values
    Nharm   = 10;      # Harmonics in model
    FreqTOL = 0.000001; # Tolerance of frequency calculations
    MaxIterations = 10000
    
    y_past, y_fut =  Quinn_Fernandes_extrapolation_njit(xx, yy, Npast, Nfut, Nharm, FreqTOL , MaxIterations  )
    x3 = np.arange(0+n1+x[0], n1+x[0]+len(y_past)-1+len(y_fut)+1, 1 )
    
    # if (draw_plots):
    #     x2 = np.array([])
    #     y2 = np.array([])
    #     for i in range(n-1):
    #         x2 = np.append(x2,[ t[i], t[i+1] ])
    #         y2 = np.append(y2, [ Y[i]/(t[i+1]-t[i]), Y[i]/(t[i+1]-t[i]) ] )
    #     plt.plot(x3,np.append(y_past, y_fut),color="red",linestyle='--')# type='l',ann=FALSE, xaxt="n", col="red",lty=2,ylim=range(c(y,xm,ym,y2)))
    #     plt.plot(x,y,color="red")#,type="l",lwd="1",lty=1, xaxt="n",ylim=range(c(y,y2)),xlim=range(c(x,x2)))
    #     #axis(1, at=t,labels=MyData[[2]],las=2)
    #     plt.plot(x2,y2,color="grey")#,type="l",lwd="2",lty=1)
       
    # Расчет времени, когда закончится последний запас Yn
    
    #print("Yn = ", Yn)
    Tn=t[-1]   
    if n2>=1:
        Storage=Yn-y_fut[n2-1]/2          # корректировка на f(tn)/2, так как часть от покупки Yn должна была отложиться в страховые запасы
    else:
        Storage=Yn-y[-1]/2                # тут y[-1] и y[Tn] одно и то же
        
    for i in range(n2,len(y_fut)):
        Storage -= y_fut[i]
        if Storage <= 0:
            break
    Delta_t = i+1 -n2 # тут вычитается n2 так как i считается не от 1, а от n2        

    # определение максимального запаса из данных
    max_storage = 0
    for i in range(len(Y)):
        max_storage += Y[i]-y[t[i]-t[0]]/2  # коррекция на f(ti)/2 так как предполагается, что в момент покупки запас опускается ниже критического уровня 
                                            # и часть берется из страховых запасов, а новая наблюдемая покупка Y компенсирует эту величину, 
                                            # тем самым максимальный запас меньше чем наблюдаемый объем покупки
    max_storage += Yn-y_fut[n2-1]/2 # последнего наблюдения Yn не было в массиве Y
    max_storage = max_storage / n
    
    if Storage <= 0:       
        T_future = Tn + Delta_t     
        Y_future = max_storage - Storage # будущая покупка должна восполнить страховые запасы, которые уменьшились на величину Storage       
        
        # if (draw_plots):
        #     plt.scatter(T_future,min(y), color="green")
             
        # if print_info:
        #     print("Delta_t = " , Delta_t, ", Storage = ", Storage)
        #     print("T_future = ", T_future)
        #     print("Y_future = " , Y_future )
        #     mean_dt = np.mean(np.array(t[1:n])-np.array(t[0:(n-1)]))
        #     print("mean_dt = ", mean_dt)

        return T_future, Y_future
    else:
        # print("Yn still positive!!!")
        # if (draw_plots):
        #     plt.scatter(Tn + Delta_t,min(y), color="red")
        return np.nan, np.nan

def  Calculate_score(T_future, Y_future, T_future_real, T_pred_real, Y_future_real, print_info = False):
    if (T_future == np.nan or Y_future == np.nan):
        return np.inf
    
    Error_t = T_future - T_future_real
    Error_t_rel = Error_t / (T_future_real- T_pred_real)
    
    Error_Y = Y_future - Y_future_real        
    Error_Y_rel = Error_Y / Y_future_real

    if (print_info):
        print("Error_t = ", Error_t )
        print("T_future_real - T_pred_real = " , T_future_real - T_pred_real)
        print("Error_t_rel = ", Error_t_rel )
        print("Error_Y = ", Error_Y )
        print("Error_Y_rel = ", Error_Y_rel ) 

    return(abs(Error_t_rel)+0.1*abs(Error_Y_rel))


@njit(cache=True)
def  Calculate_score_njit(T_future, Y_future, T_future_real, T_pred_real, Y_future_real, print_info = False):
    if (T_future == np.nan or Y_future == np.nan):
        return np.inf
    
    Error_t = T_future - T_future_real
    Error_t_rel = Error_t / (T_future_real- T_pred_real)
    
    Error_Y = Y_future - Y_future_real        
    Error_Y_rel = Error_Y / Y_future_real

    return(abs(Error_t_rel)+0.1*abs(Error_Y_rel))


def Predict_and_score_parameters(X , *args):
    alpha = X[0]
    n1 = round(X[1])
    n2 = round(X[2])
    if alpha < 0 or n1<0 or n2<0:
        return np.inf
    t = args[0]
    Y = args[1]
    T_future_real = args[2]
    T_pred_real = t[-1]
    Y_future_real = args[3]
    n_predict = args[4]

    T_future, Y_future = get_next_event(t, Y, n_predict, alpha, n1, n2, draw_plots = False, print_info = False)
    Score = Calculate_score(T_future, Y_future, T_future_real, T_pred_real, Y_future_real, print_info = False)
    return Score

def Predict_and_score_parameters2(X , *args):
    alpha = X[0]
    n1 = round(X[1])
    n2 = round(X[2])
    if alpha < 0 or n1<0 or n2<0:
        return np.inf
    
    t = args[0]
    Y = args[1]
    n=len(t)
    n_predict = args[2]
    
    t_test = t[-2] 
    Y_test = Y[-2]
    t_train = t[0:(n-2)] # срез и так не включает последний элемент, то есть индексы будут от 0 до n-3
    Y_train = Y[0:(n-2)]

    if t_train[-1]-t_train[0] - n1 - n2 -1 < 0:
        return np.inf
    print(t_train[-1]-t_train[0] - n1 - n2 -1)
    
    T_future, Y_future = get_next_event(t_train, Y_train, n_predict, alpha, n1, n2, draw_plots = False, print_info = False)
    Score1 = Calculate_score(T_future, Y_future, t_test, t_train[-1], Y_test, print_info = False)
    if Score1 == np.inf:
        return np.inf

    t_test = t[-1] 
    Y_test = Y[-1]
    t_train = t[0:(n-1)] # срез и так не включает последний элемент, то есть индексы будут от 0 до n-2
    Y_train = Y[0:(n-1)]
   
    T_future, Y_future = get_next_event(t_train, Y_train, n_predict, alpha, n1, n2, draw_plots = False, print_info = False)
    Score2 = Calculate_score(T_future, Y_future, t_test, t_train[-1], Y_test, print_info = False)
    
    return (Score1+Score2)/2

@njit(cache=True, parallel = True)
def Predict_and_score_parameters2_njit(X , *args):
    alpha = X[0]
    n1 = round(X[1])
    n2 = round(X[2])
    if alpha < 0 or n1<0 or n2<0:
        return np.inf 
    
    t = args[0]
    Y = args[1]
    n=len(t)
    n_predict = args[2]
    
    t_test = t[-2] 
    Y_test = Y[-2]
    t_train = t[0:(n-2)] # срез и так не включает последний элемент, то есть индексы будут от 0 до n-3
    Y_train = Y[0:(n-2)]

    if t_train[-1]-t_train[0] - n1 - n2 -1 < 0:
        return np.inf 
    
    T_future, Y_future = get_next_event_njit(t_train, Y_train, n_predict, alpha, n1, n2, draw_plots = False, print_info = False)
    Score1 = Calculate_score_njit(T_future, Y_future, t_test, t_train[-1], Y_test, print_info = False)
    if Score1 == np.inf:
        return np.inf

    t_test = t[-1] 
    Y_test = Y[-1]
    t_train = t[0:(n-1)] # срез и так не включает последний элемент, то есть индексы будут от 0 до n-2
    Y_train = Y[0:(n-1)]
   
    T_future, Y_future = get_next_event_njit(t_train, Y_train, n_predict, alpha, n1, n2, draw_plots = False, print_info = False)
    Score2 = Calculate_score_njit(T_future, Y_future, t_test, t_train[-1], Y_test, print_info = False)
    
    return (Score1+Score2)/2