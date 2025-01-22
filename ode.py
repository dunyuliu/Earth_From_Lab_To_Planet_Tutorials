#
# ODE routines
#
# all routines take t,x,fx,n,hs,par and update x to the new timestep
# dx/dt = f(t,x) 
# t:     time
# x[n]:  starting values of variables
# fx[n]: functions which take (x, par) as argument
# n:     dimension of problem
# h:    timestep
# par:   parameters

import numpy as np


#  Euler method 
def euler(t, y, f, neq, h,par):
    # The function will take in y at the current time and update and return the y one step size h further according to function fx.
    # neq: number of equations in the reduced order ODE system. In this case, neq == 1.
    yout=np.zeros(neq)
    for i in range(neq):            # Loop over all the equations in the system.
        # NOTE: in python, index starts from 0. Therefore, for neq==1, i will start from 0. 
        # In this case, there is only i==0 in the loop. 
        # If you want to check value in i, try print it out and uncomment the following line:
        # print(i)
        yout[i] = y[i] + h * f[i](t,y,par) 
        # Please fill out the above missing parts ?? to make this function work!
    return yout


#
# midpoint method 
#


def midpoint(t, y, f, neq, h,par):
    # The function will take in y at the current time and update and return the y one step size h further according to function fx.

    k1 = np.zeros(neq) 
    k2 = np.zeros(neq)
    yout = np.zeros(neq)
    
    for i in range(neq):            # Loop over all the equations in the system.
        k1[i] = f[i](t,y,par)*h     
    for i in range(neq):   
        k2[i] = f[i](t+h/2,y+k1/2,par)*h
    yout=y+k2
    
    return yout


#
# 4th order Runge Kutta 
#

def runge_kutta(t, y, f, neq, h,par):
    # The function will take in x at the current time and update and return the x one step size h further according to function fx.
    k1 = np.zeros(neq) 
    k2 = np.zeros(neq)
    k3 = np.zeros(neq)
    k4 = np.zeros(neq)
    for i in range(neq):   # Loop over all the equations in the system.
        k1[i]=f[i](t    , y     ,par)*h 
    for i in range(neq): 
        k2[i]=f[i](t+h/2, y+k1/2,par)*h
    for i in range(neq): 
        k3[i]=f[i](t+h/2, y+k2/2,par)*h
    for i in range(neq): 
        k4[i]=f[i](t+h  , y+k3,  par)*h
   
    yout = y + (k1 + 2*k2 + 2*k3 + k4)/6
    return yout


