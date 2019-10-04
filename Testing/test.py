import numpy as np
from scipy.optimize import minimize

def function(coef):


    def objective(y, coef):
        print 5
        return y[0]*y[3]*(y[0]+y[1]+y[2])+y[2]

def constraint1(x):
    return x[0]*x[1]*x[2]*x[3]-25.0

def constraint2(x):
    sum_eq = 40.0
    for i in range(4):
        sum_eq = sum_eq - x[i]**2
    return sum_eq

n = 4
x0 = np.zeros(n)
x0[0] = 1.0
x0[1] = 5.0
x0[2] = 5.0
x0[3] = 1.0

# show initial objective

# optimize
b = (1.0,5.0)
bnds = (b, b, b, b)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
cons = ([con1,con2])
solution = minimize(objective,x0,args=([ 5,4,6 ]),  method='SLSQP', \
                    bounds=bnds,constraints=cons)
return solution

# initial guesses
# n = 4
# x0 = np.zeros(n)
# x0[0] = 1.0
# x0[1] = 5.0
# x0[2] = 5.0
# x0[3] = 1.0

# # show initial objective
# print('Initial Objective: ' + str(objective(x0)))

# # optimize
# b = (1.0,5.0)
# bnds = (b, b, b, b)
# con1 = {'type': 'ineq', 'fun': constraint1}
# con2 = {'type': 'eq', 'fun': constraint2}
# cons = ([con1,con2])
# solution = minimize(objective,x0,method='SLSQP',\
#                     bounds=bnds,constraints=cons)
# x = solution.x

x = function(5)
print x
# show final objective
#print('Final Objective: ' + str(objective(x)))

# print solution
# print('Solution')
# print('x1 = ' + str(x[0]))
# print('x2 = ' + str(x[1]))
# print('x3 = ' + str(x[2]))
# print('x4 = ' + str(x[3]))