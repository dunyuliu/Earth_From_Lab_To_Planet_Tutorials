#
# ODE routines
#
# all routines take t,x,fx,n,hs,par and update x to the new timestep
# dx/dt = f(t,x) 
# t:     time
# x[n]:  starting values of variables
# fx[n]: functions which take (x, par) as argument
# n:     dimension of problem
# hs:    timestep
# par:   parameters

#
# Euler method 
#
def euler(t, x, fx, n, hs, par):
    for i in range(n):
        xout[i] = x[i] + hs * fx[i](t,x,par)
    return xout

#
# midpoint method 
#
def midpoint(t, x, fx, n, hs, par):
    ind = range(n)
    k1 = [] 
    for i in ind:
        k1.append(fx[i](t,x,par)*hs)
        
    for i in ind:               # euler step but df/dt evaluated at the midpoint
        xout[i] = xout[i] + hs * fx[i](t+hs/2,x+k1/2,par)

    return xout


#
# 4th order Runge Kutta 
#
def runge_kutta(t, x, fx, n, hs, par):
    ind = range(n)
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    xk = []
    for i in ind:               # k1 = h * f(t,y)
        k1.append(fx[i](t,x,par)*hs)
    for i in ind:
        xk.append(x[i] + k1[i]/2)
    for i in ind:               # k2 = h * f(t+h/2,x+k1/2)
        k2.append(fx[i](t+hs/2,xk,par)*hs)
    for i in ind:
        xk[i] = x[i] + k2[i]/2
    for i in ind:               # k3 = h * f(t+h/2,x+k2/2)
        k3.append(fx[i](t+hs/2,xk,par)*hs)
    for i in ind:
        xk[i] = x[i] + k3[i]
    for i in ind:               # k4 = h * f(t+h,x+k3)
        k4.append(fx[i](t,xk,par)*hs)
    for i in ind:               # dx = k1/6 + k2/3 + k3/3 + k4/6
        xout[i] = x[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6
    return xout

