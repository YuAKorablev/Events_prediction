import numpy as np
#import math
#import scipy
import cvxopt
import numba
from numba import njit,prange#,types


def optNM2(X, objective, L_Bounds = None, R_Bounds = None, is_int = None, is_global_bounds=False, max_iter=10000, abs_tol=0.000001 , rel_tol=0.000001, *args):
    import numpy as np
    cvxopt.solvers.options['show_progress'] = False
    # print("X=",X)
    # print("objective=",objective)
    # print("args=",args)
    # print("L_Bounds=",L_Bounds)
    # print("R_Bounds=",R_Bounds)
    # print("is_int=",is_int)
    # print("is_global_bounds=",is_global_bounds)
    # print("max_iter=",max_iter)
    # print("abs_tol=",abs_tol)
    # print("rel_tol=",rel_tol)
    n = len(X)
    if L_Bounds is None: L_Bounds = np.repeat(-np.inf, n )
    if R_Bounds is None: R_Bounds = np.repeat(np.inf, n )
    #if is_int is None: is_int = np.repeat(False, n )
    dif =  R_Bounds - L_Bounds
    dif[dif == np.inf] = X[dif == np.inf] # сделать относительным от значения, если границы не заданы,
    dif[dif == 0]  = 0.0001                  # если нулевое то брать шаг относительно минимальной границы

    dif = dif * 0.05
    if is_int is not None:
        #print(np.abs(np.round(dif[is_int])))
        dif[is_int] = np.maximum(1,np.abs(np.round(dif[is_int])) )*np.sign(dif[is_int])
        sum_is_int = sum(is_int)
        not_is_int =  ~is_int
    
    # строится симплекс
    Points = np.tile(X, (n,1) ) + np.eye(n)*dif   
    Points = np.vstack( (X, Points))  #добавляется стартовая вершина
    num=n+1          #всего вершин
  
    iterations = 0
    Values = np.apply_along_axis(objective, 1, Points, *args)
    Total_calls = num
    break_message = ""
    converged = True
    indexes = np.arange(0,n)
  
    while True:
        ord = np.argsort(Values)       #сортировка вершин по значениям
        Points = Points[ord]
        Values = Values[ord]

        Contracted = None # чтобы не было ошибки при выводе info, когда сжатие еще ни разу не делалось
        Expanded = None
        
        iterations = iterations + 1
        # условия остановки, последние два условия - это выход за границы ячейки
        if Values[0] == np.inf : 
            break_message = "Выполнилось условие остановки Values[0] == np.inf"
            break
        if iterations >= max_iter : 
            break_message = "Выполнилось условие остановки iterations >= max_iter"
            break
        if abs(Values[-1] - Values[0]) < abs_tol : 
            break_message = "Выполнилось условие остановки abs(Values[-1] - Values[0]) < abs_tol"
            break
        if abs((Values[-1] - Values[0]) / Values[0]) < rel_tol : 
            break_message = "Выполнилось условие остановки abs((Values[-1] - Values[0]) / Values[0]) < rel_tol"
            break        
        # if is_int is not None:
        #     if sum(abs(Points[-1][not_is_int] - Points[0][not_is_int])) < abs_tol and sum(abs(Points[-1][is_int] - Points[0][is_int])) <= sum_is_int: 
        #         print("Выполнилось условие остановки sum(abs(Points[-1][not_is_int] - Points[0][not_is_int])) < abs_tol and sum(abs(Points[-1][is_int] - Points[0][is_int])) <= sum_is_int")
        #         break
        # else: 
        if is_int is None:
            if sum(abs(Points[-1] - Points[0])) < abs_tol : 
                print("Выполнилось условие остановки sum(abs(Points[-1] - Points[0])) < abs_tol")
                break
                
        if sum(Points[0] > R_Bounds) > 0 : 
            break_message = "Выполнилось условие остановки sum(Points[0] > R_Bounds) > 0"
            converged = False
            break
        if sum(Points[0] < L_Bounds) > 0 : 
            break_message = "Выполнилось условие остановки sum(Points[0] < L_Bounds) > 0"
            converged = False
            break

        

        # Уменьшение размерности симплекса, когда все кроме одной вершины лежат на одной грани
        if is_global_bounds:
            for i in range(len(indexes)):
                if sum(Points[:,indexes[i]] == L_Bounds[indexes[i]]) >= n or sum(Points[:,indexes[i]] == R_Bounds[indexes[i]]) >= n: # суммирование по num=n+1 точкам
                    #print("!!! Уменьшаем размерность симплекса !!!")
                    indexes = np.delete(indexes,i)
                    Points = np.delete(Points,n,0)
                    Values = np.delete(Values,n,0)
                    n -= 1
                    num -= 1
                    break  # this for
                    
        
        
        
        # Центр противолежащей плоскости, тут n=num-1
        Centr = np.apply_along_axis(np.mean, 0, Points[0:n])

        DX = Centr - Points[-1]
        
        DX2 = 2*DX # 1*DX
        if is_int is not None:
            DX2[is_int] = np.maximum(1,np.abs(np.round(DX2[is_int])) )*np.sign(DX2[is_int])
        #print("before Reflected, DX2 = ",DX2)
        #Reflected = Points[-1] + DX2 # тут DX2=2*DX откладывается от худшей точки, ранее было #Centr + DX2
        Reflected = Points[-1].copy()
        Reflected[indexes] += DX2[indexes]
        #print("before end iteration Reflected = ", Reflected)
        Reflected_Value = objective(Reflected, *args)
        Total_calls += 1

        
        if (Reflected_Value < Values[0]):
            DX2 = 3*DX
            if is_int is not None:
                DX2[is_int] = np.maximum(1,np.abs(np.round(DX2[is_int])) )*np.sign(DX2[is_int])
            #Expanded =  Points[-1] + DX2 #Centr + DX2 #2*DX
            Expanded =  Points[-1].copy() 
            Expanded[indexes] += DX2[indexes]
            Expanded_Value = objective(Expanded, *args)
            Total_calls += 1
            if (Expanded_Value < Reflected_Value):              
                Points[-1] = Expanded
                Values[-1] = Expanded_Value
            else: #"Expanded_Value < Reflected_Value"
                Points[-1] = Reflected
                Values[-1] = Reflected_Value
              
        else: #"Reflected_Value < Values[0]"
            if (Reflected_Value < Values[-2]):
                Points[-1] = Reflected
                Values[-1] = Reflected_Value
            else: #"Reflected_Value < Values[-2]"
                if (Reflected_Value < Values[-1]):
                    Points[-1] = Reflected
                    Values[-1] = Reflected_Value
                else:
                    DX2 = 1.5*DX
                    if is_int is not None:
                        DX2[is_int] = np.maximum(1,np.abs(np.round(DX2[is_int])) )*np.sign(DX2[is_int])
                    #Contracted = Points[-1] + DX2  #Centr + DX2 #0.5*DX
                    Contracted =  Points[-1].copy() 
                    Contracted[indexes] += DX2[indexes]
                    Contracted_Value = objective(Contracted, *args)
                    Total_calls += 1
                    if (Contracted_Value < Values[-1]):
                        Points[-1] = Contracted
                        Values[-1] = Contracted_Value
                    else: #"Contracted_Value < Values[num]"  

                        DX2 =  0.5*(Points - Points[0])
                        if is_int is not None:
                            for i in range(len(Points)):
                                DX2[i][is_int] = np.maximum(1,np.abs(np.round(DX2[i][is_int])) )*np.sign(DX2[i][is_int])
                        Points_old = Points.copy()
                        Points = DX2 + Points[0]   #0.5*(Points - Points[0]) + Points[0]  
                        if np.array_equal(Points_old, Points):
                            break_message = "Выполнилось условие остановки np.array_equal(Points_old, Points)"
                            break
                        Values[1:num] = np.apply_along_axis(objective, 1, Points[1:num], *args)
                        Total_calls += n            
    
        # if iterations >=1:
        #     print("=== iterations = ", iterations, "===")
        #     print("indexes = ", indexes)
        #     print("Points = ")
        #     print(Points)
        #     print("Values = ", Values)
        #     print("Centr = ", Centr)
        #     print("DX = ", DX)
        #     print("DX2 = ", DX2)
        #     print("Reflected = ", Reflected)
        #     print("Reflected_Value = ", Reflected_Value)
        #     if Expanded is not None:
        #         print("Expanded = ", Expanded)
        #         print("Expanded_Value = ", Expanded_Value)
        #     if Contracted is not None:
        #         print("Contracted = ", Contracted)
        #         print("Contracted_Value = ", Contracted_Value)                
        #     #if iterations >=10: break

    #end while true
    return Points[0], Values[0], converged, iterations, Total_calls, break_message

@njit(cache=True, fastmath=False, parallel = True)
def optNM2_njit(X, objective, L_Bounds = None, R_Bounds = None, is_int = None, is_global_bounds=False, max_iter=10000, abs_tol=0.000001 , rel_tol=0.000001, *args):
    n = len(X)
    lenX = n
    if L_Bounds is None: L_Bounds = np.repeat(-np.inf, n )
    if R_Bounds is None: R_Bounds = np.repeat(np.inf, n )
    #if is_int is None: is_int = np.repeat(False, n )
    dif =  R_Bounds - L_Bounds
    dif[dif == np.inf] = X[dif == np.inf] # сделать относительным от значения, если границы не заданы,
    dif[dif == 0]  = 0.0001                  # если нулевое то брать шаг относительно минимальной границы

    dif = dif * 0.05
    if is_int is not None:
        dif[is_int] = np.maximum(1,np.abs(np.round(dif[is_int])) )*np.sign(dif[is_int])
        sum_is_int = sum(is_int)
        not_is_int =  ~is_int
    
    # строится симплекс
    #Points = np.tile(X, (n,1) ) + np.eye(n)*dif 
    #Points = np.vstack( (X, Points))  #добавляется стартовая вершина
   
    Points = np.vstack( (   X.reshape((1, n)), 
                             np.ascontiguousarray(X.repeat(n).reshape((-1, n)).T) + np.eye(n)*dif) )
    num=n+1          #всего вершин
  
    iterations = 0
    #Values = np.apply_along_axis(objective, 1, Points, *args)
    Values = np.zeros(num)
    for i in prange(num):
        Values[i] = objective(Points[i], *args)
    
    Total_calls = num
    break_message = ""
    converged = True
    indexes = np.arange(0,n)
  
    while True:
        ord = np.argsort(Values)       #сортировка вершин по значениям
        Points = Points[ord]
        Values = Values[ord]

        Contracted = None # чтобы не было ошибки при выводе info, когда сжатие еще ни разу не делалось
        Expanded = None
        
        iterations = iterations + 1
        
        # условия остановки, последние два условия - это выход за границы ячейки
        if Values[0] == np.inf : 
            break_message = "Выполнилось условие остановки Values[0] == np.inf"
            break
        if iterations >= max_iter : 
            break_message = "Выполнилось условие остановки iterations >= max_iter"
            break
        if np.abs(Values[-1] - Values[0]) < abs_tol : 
            break_message = "Выполнилось условие остановки abs(Values[-1] - Values[0]) < abs_tol"
            break
        if np.abs((Values[-1] - Values[0]) / Values[0]) < rel_tol : 
            break_message = "Выполнилось условие остановки abs((Values[-1] - Values[0]) / Values[0]) < rel_tol"
            break        
        # if is_int is not None:
        #     if sum(abs(Points[-1][not_is_int] - Points[0][not_is_int])) < abs_tol and sum(abs(Points[-1][is_int] - Points[0][is_int])) <= sum_is_int: 
        #         print("Выполнилось условие остановки sum(abs(Points[-1][not_is_int] - Points[0][not_is_int])) < abs_tol and sum(abs(Points[-1][is_int] - Points[0][is_int])) <= sum_is_int")
        #         break
        # else: 
        if is_int is None:
            if sum(np.abs(Points[-1] - Points[0])) < abs_tol : 
                print("Выполнилось условие остановки sum(abs(Points[-1] - Points[0])) < abs_tol")
                break
                
        if sum(Points[0] > R_Bounds) > 0 : 
            break_message = "Выполнилось условие остановки sum(Points[0] > R_Bounds) > 0"
            converged = False
            break
        if sum(Points[0] < L_Bounds) > 0 : 
            break_message = "Выполнилось условие остановки sum(Points[0] < L_Bounds) > 0"
            converged = False
            break

        

        # Уменьшение размерности симплекса, когда все кроме одной вершины лежат на одной грани
        if is_global_bounds:
            for i in range(len(indexes)):
                indexes_i = indexes[i]
                if sum(Points[:,indexes_i] == L_Bounds[indexes_i]) >= n or sum(Points[:,indexes_i] == R_Bounds[indexes_i]) >= n: # суммирование по num=n+1 точкам
                    #print("!!! Уменьшаем размерность симплекса !!!")
                    indexes = np.delete(indexes,i)
                    Points = Points[0:-1] #np.delete(Points,n,0)
                    Values = Values[0:-1] #np.delete(Values,n,0)
                    n -= 1
                    num -= 1
                    break  # this for
                    
        
                
        # Центр противолежащей плоскости, тут n=num-1
        #Centr = np.mean(Points[0:n],0) #np.apply_along_axis(np.mean, 0, Points[0:n])
        Centr = np.zeros(lenX)
        sub_array = np.ascontiguousarray(Points[0:n])
        
        for i in prange(lenX):
            Centr[i] = np.mean( sub_array[:,i])
                    
        DX = np.ascontiguousarray(Centr) - np.ascontiguousarray(Points[-1])
        
        # if iterations==8:
        #     return "OK",n,num,indexes,Points, Values, sub_array, "ok2", Centr, DX
        
        DX2 = 2*DX # 1*DX
        if is_int is not None:
            DX2[is_int] = np.maximum(1,np.abs(np.round(DX2[is_int])) )*np.sign(DX2[is_int])
        #print("before Reflected, DX2 = ",DX2)
        #Reflected = Points[-1] + DX2 # тут DX2=2*DX откладывается от худшей точки, ранее было #Centr + DX2
        Reflected = Points[-1].copy()
        Reflected[indexes] += DX2[indexes]
        #print("before end iteration Reflected = ", Reflected)
        Reflected_Value = objective(Reflected, *args)
        Total_calls += 1

        
        if (Reflected_Value < Values[0]):
            DX2 = 3*DX
            if is_int is not None:
                DX2[is_int] = np.maximum(1,np.abs(np.round(DX2[is_int])) )*np.sign(DX2[is_int])
            #Expanded =  Points[-1] + DX2 #Centr + DX2 #2*DX
            Expanded =  Points[-1].copy() 
            Expanded[indexes] += DX2[indexes]
            Expanded_Value = objective(Expanded, *args)
            Total_calls += 1
            if (Expanded_Value < Reflected_Value):              
                Points[-1] = Expanded
                Values[-1] = Expanded_Value
            else: #"Expanded_Value < Reflected_Value"
                Points[-1] = Reflected
                Values[-1] = Reflected_Value
              
        else: #"Reflected_Value < Values[0]"
            if (Reflected_Value < Values[-2]):
                Points[-1] = Reflected
                Values[-1] = Reflected_Value
            else: #"Reflected_Value < Values[-2]"
                if (Reflected_Value < Values[-1]):
                    Points[-1] = Reflected
                    Values[-1] = Reflected_Value
                else:
                    DX2 = 1.5*DX
                    if is_int is not None:
                        DX2[is_int] = np.maximum(1,np.abs(np.round(DX2[is_int])) )*np.sign(DX2[is_int])
                    #Contracted = Points[-1] + DX2  #Centr + DX2 #0.5*DX
                    Contracted =  Points[-1].copy() 
                    Contracted[indexes] += DX2[indexes]
                    Contracted_Value = objective(Contracted, *args)
                    Total_calls += 1
                    if (Contracted_Value < Values[-1]):
                        Points[-1] = Contracted
                        Values[-1] = Contracted_Value
                    else: #"Contracted_Value < Values[num]"  

                        DX2 =  0.5*(Points - Points[0])
                        if is_int is not None:
                            for i in range(len(Points)):
                                DX2[i][is_int] = np.maximum(1,np.abs(np.round(DX2[i][is_int])) )*np.sign(DX2[i][is_int])
                        Points_old = Points.copy()
                        Points = DX2 + Points[0]   #0.5*(Points - Points[0]) + Points[0]  
                        if np.array_equal(Points_old, Points):
                            break_message = "Выполнилось условие остановки np.array_equal(Points_old, Points)"
                            break
                        #Values[1:num] = np.apply_along_axis(objective, 1, Points[1:num], *args)
                        for i in prange(1,num):
                            Values[i] = objective(Points[i], *args)

                        
                        Total_calls += n            


        #if iterations==7:
           # return "Ok"#Points, Values, Centr, DX, Reflected, Expanded, Contracted
    #end while true
    #return "OK",n,num,indexes,Points, Values, sub_array, "ok2", Centr, DX
    return Points[0], Values[0], converged, iterations, Total_calls, break_message




def my_grid_search(objective, *args,  each_axis_knots, is_int = None, num_results = 1, overlap = 1, max_iter=10000, abs_tol=0.000001 , rel_tol=0.000001):
    d = len(each_axis_knots)
    X = np.zeros(d)
    L_Bounds = np.zeros(d)
    R_Bounds = np.zeros(d)
    num_knots = np.zeros(d)
    for k in range(d):
        num_knots[k] = len(each_axis_knots[k])
    index_knot = np.zeros(d, dtype = np.int32)
    results = []
    Total_iterations = 0
    Total_calls = 0
    
    while True:
        is_global_bounds=False
        
        for k in range(d):
            min(index_knot[k]+overlap, num_knots[k]-1 )
            L_Bounds[k] = each_axis_knots[k][index_knot[k]]
            R_Bounds[k] = each_axis_knots[k][index_knot[k]+1]
            if L_Bounds[k] == -np.inf:
                if R_Bounds[k] == np.inf:
                    X[k] = 0
                else:
                    if index_knot[k]+2 < num_knots[k]:
                        X[k] = R_Bounds[k] - ( each_axis_knots[k][index_knot[k]+2] - each_axis_knots[k][index_knot[k]+1])
                    else:
                        X[k] = R_Bounds[k]
            else:
                if R_Bounds[k] == np.inf:
                    if index_knot[k]-1 >=0:
                        X[k] = L_Bounds[k] + ( each_axis_knots[k][index_knot[k]] - each_axis_knots[k][index_knot[k]-1])
                    else:
                        X[k] = L_Bounds[k]
                else:
                    X[k] = (R_Bounds[k]+L_Bounds[k])/2

            if is_int is not None and is_int[k]:
                X[k] = round(X[k])

            if index_knot[k] == 0 or index_knot[k]+2 >= num_knots[k]:
                is_global_bounds = True
                
            R_Bounds[k] = each_axis_knots[k][ int(min(index_knot[k]+overlap, num_knots[k]-1 ) ) ]
        #end for k

        # print("=============================")
        # print("X = ", X)
        # print("L_Bounds = ", L_Bounds)
        # print("R_Bounds = ", R_Bounds)
        # print("is_global_bounds = ",is_global_bounds)
        
        r = optNM2(X ,objective,  
                   L_Bounds, 
                   R_Bounds, 
                   is_int,
                   is_global_bounds,
                   max_iter,
                   abs_tol, 
                   rel_tol, 
                   *args)
        
        # print("OK")
        
        Total_iterations += r[3]
        Total_calls += r[4]
        
        if r[2]: # converged = True
            if len(results) < num_results:
                results.append(r)
            else:
                # находится худший результат
                last_result = results[0][1]
                last_result_index = 0
                for i in range(1,num_results):
                    if results[i][1] > last_result:
                        last_result = results[i][1]
                        last_result_index = i
                if r[1]<last_result:
                    results[last_result_index] = r
    
        index_knot[-1] += 1
        for i in range(1,d+1):
            if index_knot[-i] < num_knots[-i]-1: 
                break
            else:
                if i >= d:
                    ord = np.argsort([r[1] for r in results])
                    return [results[i] for i in ord] , Total_iterations, Total_calls
                else:
                    index_knot[-i] = 0
                    index_knot[-i-1] += 1
                     

    # end while true

def my_grid_search_njit(objective, *args,  each_axis_knots, is_int = None, num_results = 1, overlap = 1, max_iter=10000, abs_tol=0.000001 , rel_tol=0.000001):
    d = len(each_axis_knots)
    X = np.zeros(d, dtype=np.float64)
    L_Bounds = np.zeros(d, dtype=np.float64)
    R_Bounds = np.zeros(d, dtype=np.float64)
    num_knots = np.zeros(d)
    for k in range(d):
        num_knots[k] = len(each_axis_knots[k])
    index_knot = np.zeros(d, dtype = np.int32)
    results = []
    Total_iterations = 0
    Total_calls = 0
    
    grid_iteration = 0
    
    while True:
        grid_iteration = grid_iteration + 1
        is_global_bounds=False
        
        for k in range(d):
            min(index_knot[k]+overlap, num_knots[k]-1 )
            L_Bounds[k] = each_axis_knots[k][index_knot[k]]
            R_Bounds[k] = each_axis_knots[k][index_knot[k]+1]
            if L_Bounds[k] == -np.inf:
                if R_Bounds[k] == np.inf:
                    X[k] = 0
                else:
                    if index_knot[k]+2 < num_knots[k]:
                        X[k] = R_Bounds[k] - ( each_axis_knots[k][index_knot[k]+2] - each_axis_knots[k][index_knot[k]+1])
                    else:
                        X[k] = R_Bounds[k]
            else:
                if R_Bounds[k] == np.inf:
                    if index_knot[k]-1 >=0:
                        X[k] = L_Bounds[k] + ( each_axis_knots[k][index_knot[k]] - each_axis_knots[k][index_knot[k]-1])
                    else:
                        X[k] = L_Bounds[k]
                else:
                    X[k] = (R_Bounds[k]+L_Bounds[k])/2

            if is_int is not None and is_int[k]:
                X[k] = round(X[k])

            if index_knot[k] == 0 or index_knot[k]+2 >= num_knots[k]:
                is_global_bounds = True
                
            R_Bounds[k] = each_axis_knots[k][ int(min(index_knot[k]+overlap, num_knots[k]-1 ) ) ]
        #end for k

        if grid_iteration == 4:
            r = optNM2_njit(X ,objective,  
                       L_Bounds, 
                       R_Bounds, 
                       is_int,
                       is_global_bounds,
                       max_iter,
                       abs_tol, 
                       rel_tol, 
                       *args)
            return X, L_Bounds, R_Bounds, r
        
        # Total_iterations += r[3]
        # Total_calls += r[4]
        
        # if r[2]: # converged = True
        #     if len(results) < num_results:
        #         results.append(r)
        #     else:
        #         # находится худший результат
        #         last_result = results[0][1]
        #         last_result_index = 0
        #         for i in range(1,num_results):
        #             if results[i][1] > last_result:
        #                 last_result = results[i][1]
        #                 last_result_index = i
        #         if r[1]<last_result:
        #             results[last_result_index] = r
    
        index_knot[-1] += 1
        for i in range(1,d+1):
            if index_knot[-i] < num_knots[-i]-1: 
                break
            else:
                if i >= d:
                    ord = np.argsort([r[1] for r in results])
                    return [results[i] for i in ord] , Total_iterations, Total_calls
                else:
                    index_knot[-i] = 0
                    index_knot[-i-1] += 1                     
    # end while true




#X, objective, *args,  L_Bounds = None, R_Bounds = None, is_int = None, is_global_bounds=False, max_iter=10000, abs_tol=0.000001 , rel_tol=0.000001
def my_grid_search_parallel(objective, *args,  each_axis_knots, is_int = None, num_results = 1, overlap = 1, max_iter=10000, abs_tol=0.000001 , 
                                rel_tol=0.000001):
    import multiprocess as mp
    d = len(each_axis_knots)
    X = np.zeros(d)
    L_Bounds = np.zeros(d)
    R_Bounds = np.zeros(d)
    num_knots = np.zeros(d)
    for k in range(d):
        num_knots[k] = len(each_axis_knots[k])
    index_knot = np.zeros(d, dtype = np.int32)
    results = []
    args_list = []
    Total_iterations = 0
    Total_calls = 0
    
    while True:
        is_global_bounds=False
        
        for k in range(d):
            min(index_knot[k]+overlap, num_knots[k]-1 )
            L_Bounds[k] = each_axis_knots[k][index_knot[k]]
            R_Bounds[k] = each_axis_knots[k][index_knot[k]+1]
            if L_Bounds[k] == -np.inf:
                if R_Bounds[k] == np.inf:
                    X[k] = 0
                else:
                    if index_knot[k]+2 < num_knots[k]:
                        X[k] = R_Bounds[k] - ( each_axis_knots[k][index_knot[k]+2] - each_axis_knots[k][index_knot[k]+1])
                    else:
                        X[k] = R_Bounds[k]
            else:
                if R_Bounds[k] == np.inf:
                    if index_knot[k]-1 >=0:
                        X[k] = L_Bounds[k] + ( each_axis_knots[k][index_knot[k]] - each_axis_knots[k][index_knot[k]-1])
                    else:
                        X[k] = L_Bounds[k]
                else:
                    X[k] = (R_Bounds[k]+L_Bounds[k])/2

            if is_int is not None and is_int[k]:
                X[k] = round(X[k])

            if index_knot[k] == 0 or index_knot[k]+2 >= num_knots[k]:
                is_global_bounds = True
                
            R_Bounds[k] = each_axis_knots[k][ int(min(index_knot[k]+overlap, num_knots[k]-1 ) ) ]
        #end for k

        # print("=============================")
        # print("X = ", X)
        # print("L_Bounds = ", L_Bounds)
        # print("R_Bounds = ", R_Bounds)
        # print("is_global_bounds = ",is_global_bounds)
        
        # r = optNM2(X ,objective, *args, 
        #            L_Bounds = L_Bounds, 
        #            R_Bounds = R_Bounds, 
        #            is_int = is_int,
        #            is_global_bounds = is_global_bounds,
        #            max_iter = max_iter,
        #            abs_tol = abs_tol, 
        #            rel_tol = rel_tol)

        args_list.append( (X.copy() ,
                           objective, 
                           L_Bounds.copy(),
                           R_Bounds.copy(),
                           is_int.copy(),
                           is_global_bounds,
                           max_iter,
                           abs_tol,
                           rel_tol,
                           *args  ) )
        
        if len(args_list) >= 1000: 
            # for i in args_list:
            #     print(i)
            with mp.Pool(mp.cpu_count() ) as p:
                res = list(p.starmap(optNM2, args_list ))
            args_list = []
            
            for r in res:
                Total_iterations += r[3]
                Total_calls += r[4]
                
                if r[2]: # converged = True
                    if len(results) < num_results:
                        results.append(r)
                    else:
                        # находится худший результат
                        last_result = results[0][1]
                        last_result_index = 0
                        for i in range(1,num_results):
                            if results[i][1] > last_result:
                                last_result = results[i][1]
                                last_result_index = i
                        if r[1]<last_result:
                            results[last_result_index] = r
    
        index_knot[-1] += 1
        for i in range(1,d+1):
            if index_knot[-i] < num_knots[-i]-1: 
                break
            else:
                if i >= d:
                    if len(args_list)>0:
                        with mp.Pool(mp.cpu_count()) as p:
                            res = list(p.starmap(optNM2, args_list))
                        args_list = []
                    
                        for r in res:
                            Total_iterations += r[3]
                            Total_calls += r[4]
                            
                            if r[2]: # converged = True
                                if len(results) < num_results:
                                    results.append(r)
                                else:
                                    # находится худший результат
                                    last_result = results[0][1]
                                    last_result_index = 0
                                    for i in range(1,num_results):
                                        if results[i][1] > last_result:
                                            last_result = results[i][1]
                                            last_result_index = i
                                    if r[1]<last_result:
                                        results[last_result_index] = r                    
                    
                    ord = np.argsort([r[1] for r in results])
                    return [results[i] for i in ord] , Total_iterations , Total_calls
                else:
                    index_knot[-i] = 0
                    index_knot[-i-1] += 1
    # end while true