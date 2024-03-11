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
    Vl = 10.613mV
    gna_bar = 120mS/cm^2
    gk_bar = 36mS/cm^2
    gl_bar = 0.3mS/cm^2
    (rest potential is 0mV)
    
"""


import numpy as np
import scipy as sci

        
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
    for t_c in np.arange(start, stop, step):
        y_c = y[-1]
        y_n = []
        for ode, i in zip(odes, range(N)):
            k = dt*ode(t_c, y_c, **kwargs)
            y_n.append(y_c[i] + k)
        t_n = t_c + dt
        y.append(y_n)
        t.append(t_n)
    y_transpose = []
    for i in range(N):
        y_transpose.append([x[i] for x in y]) 
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
        y_c = np.asarray(y[-1]) # current value of y
        
        k1 = []
        for ode, i in zip(odes, range(N)):
            k1.append(dt*ode(t_c, list(y_c), **kwargs))

        k2 = []   
        for ode, i in zip(odes, range(N)):
            k2.append(dt*ode(t_c+dt/2, list(y_c + np.asarray(k1)/2), **kwargs))
            
        k3 = []
        for ode, i in zip(odes, range(N)):
            k3.append(dt*ode(t_c+dt/2, list(y_c + np.asarray(k2)/2), **kwargs))
        
        k4 = []
        for ode, i in zip(odes, range(N)):
            k4.append(dt*ode(t_c+dt, list(y_c + np.asarray(k3)), **kwargs))
        
        y_n = y_c + 1/6*((np.asarray(k1) + 2*np.asarray(k2) + 2*np.asarray(k3) + np.asarray(k4)))
        t_n = t_c + dt # next value fo t
        y.append(list(y_n))
        t.append(t_n)
        # k_all.append([k1,k2,k3,k4])
    
    y_transpose = []
    for i in range(N):
        y_transpose.append([x[i] for x in y]) 
    return y_transpose, t 

# hh equations
def vmp_hh(t_c, values, Cm = 1, Vna = 115, Vk = -12, Vl = 10.6, gna_bar = 120, gk_bar = 36, gl_bar = 0.3, **kwargs):
    """
    f(t_c, values) = dVm/dt

    Parameters
    ----------
    t_c : numerical
        current time point
    values : list [vm, n, m, h]
        list of variables in the f function
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
    m = values[2]
    h = values[3]
    try:
        stim_i = kwargs['stim_i']
        if stim_i:
            i_inj = ip_hh(t_c, **kwargs) 
        else:
            i_inj = 10
            # print('no i stim given')
    except KeyError:
        i_inj = 10
        # print('no i stim given')

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
    f = 1/Cm*(i_inj - i_syn - gk_bar*n**4*(vm-Vk) - gna_bar*m**3*h*(vm-Vna) - gl_bar*(vm-Vl))
    return f

def np_hh(t_c, values, **kwargs):
    """
    f(t_c, values) = dn/dt

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
    m = values[2]
    h = values[3]
    alpha_n = 0.01*(10-vm)/(np.exp((10-vm)/10)-1)
    beta_n = 0.125*np.exp(-vm/80)
    f = alpha_n*(1-n) - beta_n*(n)
    return f

def mp_hh(t_c, values, **kwargs):
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
    m = values[2]
    h = values[3]
    alpha_m = 0.1*(25-vm)/(np.exp((25-vm)/10)-1)
    beta_m = 4*np.exp(-vm/18)
    f = alpha_m*(1-m) - beta_m*(m)
    return f
    
def hp_hh(t_c, values, **kwargs):
    """
    f(t_c, values) = dh/dt

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
    m = values[2]
    h = values[3]
    alpha_h = 0.07*np.exp(-vm/20)
    beta_h = 1/(np.exp((30-vm)/10)+1)
    f = alpha_h*(1-h) - beta_h*(h)
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
        i_bias = kwargs['i']
    except KeyError:
        i_bias = 0
    try:
        stim_start = kwargs['stim_start']
    except KeyError:
        stim_start = 0
    try:
        stim_wave = kwargs['i_waveform']
    except KeyError:
        return i_bias
    if (t_c>=stim_start) and (t_c<stim_start+len(stim_wave)*dt):
        idx = int(np.floor((t_c-stim_start)/dt))
        return stim_wave[idx] + i_bias
    else:
        return i_bias

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
    
        
            
        
    