# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:13:58 2021

@author: xiao208
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:54:28 2019

@author: dalan

list of unit and default parameters:
    time: ms
    voltage: mV
    current: uA/cm^2
    capacitance: uf/cm^2
    conducrance: mS/cm^2

    Cm = 1 uF/cm^2
    Vna = 115mV
    Vk = -12mV
    Vl = 10.6mV
    gna_bar = 120mS/cm^2
    gk_bar = 36mS/cm^2
    gl_bar = 0.3mS/cm^2
    (rest potential is 0mV)
    gamma_K = 20pS = 20e-9 mS
    gamma_Na = 20pS = 20e-9 mS
    D_K = 18 channels/um^2  = 18e8 channels/cm^2
    D_Na = 60 channels/um^2 = 60e8 channels/cm^2
    area = 200 um^2 = 2e-6 cm^2
    
"""


import numpy as np
import scipy as sci
from scipy.stats import norm
from numpy.random import RandomState

        
def euler(odes, start, stop, step, initial_values,**kwargs):
    """
    solve a 1 order ODE with euler method. 
    For dy/dt = f(y, t), solve as y[n+1] = y[n] + h*f(t_n, y_n)

    Parameters
    ----------
    odes : list of python function
        ODE(interval, value) = f(t, y), return f
    start : numerical
        Time point of start (in ms)
    stop : numerical
        Time point of stop (in ms)
    step : numerical
        interval size (in ms)
    initial_values : list of numerical values
        the initial value of y (y_0)

    Returns: y, t; lists (note: y will be list of lists)
    -------
    """
    y = [initial_values]
    N = len(initial_values) # number of orders
    if len(odes) != N:
        raise ValueError('The %d ODEs doesn\'t match with %d initial values' % (len(odes),len(initial_values)))
    t = [start]
    dt = step
    # k_all = []
    for t_c in np.arange(start, stop, step):
        y_c = y[-1]
        y_n = []
        k_n = []
        for ode, i in zip(odes, range(N)):
            k = ode(t_c,dt, y_c, **kwargs)
            y_n.append(y_c[i] + k)
            k_n.append(k)
        t_n = t_c + dt
        y.append(y_n)
        t.append(t_n)
        # k_all.append(k_n)
    y_transpose = []
    for i in range(N):
        y_transpose.append([x[i] for x in y]) 
#        k_transpose.append([x[i] for x in k_all])
    return y_transpose, t
    
def rk(odes, start, stop, step, initial_values, **kwargs):
    """
    solve a 1 order ODE with euler method. 
    For dy/dt = f(y, t), solve with 4 order RK method

    Parameters
    ----------
    odes : list of python functions
        ODE(interval, value) = f(t, y), return f
    start : numerical
        Time point of start (in ms)
    stop : numerical
        Time point of stop (in ms)
    step : numerical
        interval size (in ms)
    initial_values : list
        the initial value of y (y_0)

    Returns: y, t; lists
    -------
    """
    y = [initial_values]
    t = [start]
    # k_all = []
    N = len(initial_values)
    if len(odes) != N:
        raise ValueError('The %d ODEs doesn\'t match with %d initial values' % (len(odes),len(initial_values)))
    dt = step
    for t_c in np.arange(start, stop, step):
        y_c = y[-1]
        
        k1 = []
        y_c1 = []
        for ode, i in zip(odes, range(N)):
            k1.append(ode(t_c, dt,y_c, **kwargs))
            y_c1.append(y_c[i] + k1[i]/2)

        k2 = []   
        y_c2 = []
        for ode, i in zip(odes, range(N)):
            k2.append(ode(t_c+dt/2, dt,y_c1, **kwargs))
            y_c2.append(y_c[i] + k2[i]/2)
            
        k3 = []
        y_c3 = []
        for ode, i in zip(odes, range(N)):
            k3.append(ode(t_c+dt/2, dt,y_c2, **kwargs))
            y_c3.append(y_c[i]+k3[i]/2)
        
        k4 = []
        y_n = []
        for ode, i in zip(odes, range(N)):
            k4.append(ode(t_c+dt, dt,y_c3, **kwargs))
            y_n.append(y_c[i]+1/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]))
        
        t_n = t_c + dt # next value fo t
        y.append(y_n)
        t.append(t_n)
        
    y_transpose = []
    for i in range(N):
        y_transpose.append([x[i] for x in y]) 
    return y_transpose, t 

# hh equations
def vmp_hh(t_c, dt,values, Cm = 1, Vna = 115, Vk = -12, Vl = 10.6, gamma_K = 20e-9,gamma_Na = 20e-9,D_Na = 60e8, D_K = 18e8, gl_bar = 0.3,NNa = 12000, NK = 3600, **kwargs):
    """
    f(t_c, values) = dVm/dt

    Parameters
    ----------
    t_c : numerical
        current time point
    values : list [vm, n, mh]
        list of variables in the f function
        n: ndarray, [n0,n1,n2,n3,n4], each represent fraction of channels under the said state
        mh: ndarray, [m0h0, m1h0, m2h0, m3h0, m0h1,m1h1,m2h1,m3h1]
    ip : python function
        I_inj = ip(t)
    gp : python function
        i_syn = gp(t)
    Returns
    -------
    f(t_c, values)

    """
    vm = values[0]
    n = values[1]
    mh = values[2]
    try:
        stim_i = kwargs['stim_i']
        if stim_i:
            i_inj = ip_hh(t_c, **kwargs) 
        else:
            i_inj = 10
    except KeyError:
        i_inj = 10

    try:
        stim_g = kwargs['stim_g']
        if stim_g:
            Vg = kwargs['Vg']
            i_syn = gp_hh(t_c, **kwargs)*(vm-Vg)
        else:
            i_syn = 0
    except KeyError:
        i_syn = 0
     
    # i_inj = 10
    # f = 1/Cm*(i_inj - i_syn - gk_bar*n**4*(vm-Vk) - gna_bar*m**3*h*(vm-Vna) - gl_bar*(vm-Vl))
    f = dt*1/Cm*(i_inj - i_syn - gamma_K*n[-1]*D_K*(vm-Vk) - gamma_Na*mh[-1]*D_Na*(vm-Vna) - gl_bar*(vm-Vl))
    return f

def np_hh(t_c, dt,values,NK = 3600, **kwargs):
    """
    f(t_c, values) = dn/dt

    Parameters
    ----------
    t_c : numerical
        current time point
    values : list
        [vm, n, mh]

    Returns
    -------
    f: ndarray, [dn0, dn1, dn2, dn3, dn4]

    """
    vm = values[0]
    # n = np.concatenate((np.asarray([crop(1-np.sum(values[1]/NK), 0, 1)]),values[1]/NK))
    n = values[1]
    mh = values[2]
    if vm == 10:
        vm = vm+0.0001
    alpha_n = 0.01*(10-vm)/(np.exp((10-vm)/10)-1)
    beta_n = 0.125*np.exp(-vm/80)
    rand_norm = norm
    rand_norm.random_state=RandomState(seed=None)
    Z = rand_norm.rvs(0,1, size = 4) # unit gaussian random variables
    R0 = Z[0]*np.sqrt(np.abs(4*alpha_n*n[0]+beta_n*n[1])/(NK*dt))
    R1 = Z[1]*np.sqrt(np.abs(3*alpha_n*n[1]+2*beta_n*n[2])/(NK*dt))
    R2 = Z[2]*np.sqrt(np.abs(2*alpha_n*n[2]+3*beta_n*n[3])/(NK*dt))
    R3 = Z[3]*np.sqrt(np.abs(alpha_n*n[3]+4*beta_n*n[4])/(NK*dt))
    dn0 = dt*((-4*alpha_n*n[0]+beta_n*n[1])+R0)
    dn1 = dt*((4*alpha_n*n[0]-beta_n*n[1]-3*alpha_n*n[1]+2*beta_n*n[2])-R0+R1)
    dn2 = dt*((3*alpha_n*n[1]-2*beta_n*n[2]-2*alpha_n*n[2]+3*beta_n*n[3])-R1+R2)
    dn3 = dt*((2*alpha_n*n[2]-3*beta_n*n[3]-alpha_n*n[3]+4*beta_n*n[4])-R2+R3)
    dn4 = dt*((alpha_n*n[3]-4*beta_n*n[4])-R3)
    # dn4 = (0-dn1-dn2-dn3-dn0)
    f = np.asarray([dn0,dn1,dn2,dn3, dn4])
    return f

def mhp_hh(t_c, dt,values,NNa = 12000, **kwargs):
    """
    f(t_c, values) = dm/dt

    Parameters
    ----------
    t_c : numerical
        current time point
    values : list
        [vm, n, m, h]

    Returns
    -------
    f

    """
    vm = values[0]
    n = values[1]
    # mh = np.concatenate((np.asarray([crop(1-np.sum(values[2]/NNa), 0, 1)]),values[2]/NNa))
    mh = values[2]
    # print(vm)
    if vm == 25:
        vm = vm+0.0001
    alpha_m = 0.1*(25-vm)/(np.exp((25-vm)/10)-1)
    beta_m = 4*np.exp(-vm/18)
    alpha_h = 0.07*np.exp(-vm/20)
    beta_h = 1/(np.exp((30-vm)/10)+1)
    rand_norm = norm
    rand_norm.random_state=RandomState(seed=None)
    Z = rand_norm.rvs(0,1, size = 10) # unit independent gaussian random variables
    R0 = Z[0]*np.sqrt(np.abs(3*alpha_m*mh[0]+beta_m*mh[1])/(NNa*dt))
    R1 = Z[1]*np.sqrt(np.abs(2*alpha_m*mh[1]+2*beta_m*mh[2])/(NNa*dt))
    R2 = Z[2]*np.sqrt(np.abs(alpha_m*mh[2]+3*beta_m*mh[3])/(NNa*dt))
    R3 = Z[3]*np.sqrt(np.abs(3*alpha_m*mh[4]+beta_m*mh[5])/(NNa*dt))
    R4 = Z[4]*np.sqrt(np.abs(2*alpha_m*mh[5]+2*beta_m*mh[6])/(NNa*dt))
    R5 = Z[5]*np.sqrt(np.abs(alpha_m*mh[6]+3*beta_m*mh[7])/(NNa*dt))
    R6 = Z[6]*np.sqrt(np.abs(alpha_h*mh[0]+beta_h*mh[4])/(NNa*dt))
    R7 = Z[7]*np.sqrt(np.abs(alpha_h*mh[1]+beta_h*mh[5])/(NNa*dt))
    R8 = Z[8]*np.sqrt(np.abs(alpha_h*mh[2]+beta_h*mh[6])/(NNa*dt))
    R9 = Z[9]*np.sqrt(np.abs(alpha_h*mh[3]+beta_h*mh[7])/(NNa*dt))
    d0 = dt*(((-3*alpha_m-alpha_h)*mh[0]+beta_m*mh[1]+beta_h*mh[4])+R0+R6)
    d1 = dt*((3*alpha_m*mh[0]+(-beta_m-2*alpha_m-alpha_h)*mh[1]+2*beta_m*mh[2]+beta_h*mh[5])-R0+R1+R7)
    d2 = dt*((2*alpha_m*mh[1]+(-2*beta_m-alpha_m-alpha_h)*mh[2]+3*beta_m*mh[3]+beta_h*mh[6])-R1+R2+R8)
    d3 = dt*((alpha_m*mh[2]+(-3*beta_m-alpha_h)*mh[3]+beta_h*mh[7])-R2+R9)
    d4 = dt*((alpha_h*mh[0]+(-beta_h-3*alpha_m)*mh[4]+beta_m*mh[5])+R3-R6)
    d5 = dt*((alpha_h*mh[1]+3*alpha_m*mh[4]+(-2*alpha_m-beta_m-beta_h)*mh[5]+2*beta_m*mh[6])-R3+R4-R7)
    d6 = dt*((alpha_h*mh[2]+2*alpha_m*mh[5]+(-alpha_m-2*beta_m-beta_h)*mh[6]+3*beta_m*mh[7])-R4+R5-R8)
    d7 = dt*((alpha_h*mh[3]+alpha_m*mh[6]+(-3*beta_m-beta_h)*mh[7])-R5-R9)
    # d7 = (0-d1-d2-d3-d4-d5-d6-d0)
    # print(R9)
    # print((alpha_m*mh[2]+(-3*beta_m-alpha_h)*mh[3]+beta_h*mh[7]))
    f = np.asarray([d0,d1,d2,d3,d4,d5,d6, d7])
    return f 

def ip_hh(t_c, dt = 0.01, **kwargs):
    """
    inject current with a specific waveform

    Parameters
    ----------
    tc : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    try:
        stim_wave = kwargs['i_waveform']
    except():
        return 0
    idx = int(np.floor(t_c/dt))
    return stim_wave[idx]

# def ip_hh(t_c, **kwargs):
#     """
#     step current injection
#     params needed: stim_i(bool), stim_start, stim_step, stim_amp
#     """
#     try:
#         stim_start = kwargs['stim_start']
#     except():
#         stim_start = 0
#     try:
#         stim_step = kwargs['stim_step']
#     except():
#         stim_step = 0
#     try:
#         stim_amp = kwargs['stim_amp']
#     except():
#         stim_amp = 0
#     if t_c>=stim_start and t_c<=stim_start+stim_step:
#         i = stim_amp
#     else:
#         i = 0
#     return i
        
def gp_hh(t_c, **kwargs):
    """
    Alpha synapses
    params needed: stim_g(bool), stim_start, tau, g_max, Vg

    """
    try:
        stim_start = kwargs['stim_start']
    except():
        stim_start = 0
    try:
        tau = kwargs['tau']
    except():
        tau = 10
    try:
        g_max = kwargs['g_max']
    except():
        g_max = 0
    if t_c>=stim_start:
        g = g_max*(t_c-stim_start)/tau*np.exp(-(t_c-stim_start-tau)/tau)
    else:
        g = 0
    return g
    

def round_up(x):
    if x>1:
        y = 1
    elif x<0:
        y = 0
    else:
        y = x
    return y
            
def nonNeg(x):
    if x>=0:
        y = x
    else:
        y = 0
    return y
    
def choose_random_binom(n,p):
    scipy_randomGen = binom
    scipy_randomGen.random_state=RandomState(seed=562765)
    a = np.random.choice(scipy_randomGen.rvs(n,p,size = 100),1,replace=False)
    return a[0]

def ceil(x):
    if x>=0:
        y = np.ceil(x)
    else:
        y = -(np.ceil(-x))
    return y

def crop(x, a, b):
    # make a<=x<=b
    if x>b:
        y = b
    elif x<a:
        y = a
    else:
        y = x
    return y