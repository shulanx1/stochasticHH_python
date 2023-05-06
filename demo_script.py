# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:25:10 2019

@author: Shulan

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
"""
import sys
sys.path.insert(1, 'E:\\Code\\stochastichh_python')

import numpy as np
import matplotlib.pyplot as plt
import deterministic_HH
import stocastic_HH
import scipy.io as sio
from scipy.stats import norm
from numpy.random import RandomState

#%% additional functions
def plot_deterministic(Vnmh, t, Vna = -115, Vk = 12, gna_bar = 120, gk_bar = 36):
    """
    plot Vm and the Na and K current

    Parameters
    ----------
    Vnmh : list of lists
        the y_rk solved with RK method
    t : list
        list of time series
    Vna : TYPE, optional
        DESCRIPTION. The default is 115.
    Vk : TYPE, optional
        DESCRIPTION. The default is -12.
    gna_bar : TYPE, optional
        DESCRIPTION. The default is 120.
    gk_bar : TYPE, optional
        DESCRIPTION. The default is 36.

    Returns
    -------
    None.

    """
    vm = np.asarray(Vnmh[0])
    n = np.asarray(Vnmh[1])
    m = np.asarray(Vnmh[2])
    h = np.asarray(Vnmh[3])
    i_na = gna_bar*m**3*h*(vm-Vna)
    i_k = gk_bar*n**4*(vm-Vk)
    g_na = gna_bar*m**3*h
    g_k = gk_bar*n**4
    fig = plt.figure(figsize = [8, 10])
    plt.subplot(3,1,1)
    plt.plot(t, vm)
    plt.xticks([])
    plt.ylabel('Vm (mV)')
    plt.subplot(3,1,2)
    plt.plot(t, i_na)
    plt.plot(t, i_k)
    plt.legend(['ina','ik'])
    plt.xticks([])
    plt.ylabel('i (uA/cm^2)')
    plt.subplot(3,1,3)
    plt.plot(t, g_na)
    plt.plot(t, g_k)
    plt.legend(['g_na', 'g_k'])
    plt.xlabel('t (ms)')
    plt.ylabel('g (mS/cm^2)')
    
def plot_stocastic(Vnmh, t,  Vna = 115, Vk = -12, gamma_K = 20e-9,gamma_Na = 20e-9,D_Na = 60e8, D_K = 18e8, NNa = 12000, NK = 3600):
    """
    plot Vm and the Na and K current

    Parameters
    ----------
    Vnmh : list of lists
        the y_rk solved with RK method
    t : list
        list of time series
    Vna : TYPE, optional
        DESCRIPTION. The default is 115.
    Vk : TYPE, optional
        DESCRIPTION. The default is -12.
    gna_bar : TYPE, optional
        DESCRIPTION. The default is 120.
    gk_bar : TYPE, optional
        DESCRIPTION. The default is 36.

    Returns
    -------
    None.

    """
    vm = np.asarray(Vnmh[0])
    n = np.transpose(np.asarray(Vnmh[1]))
    mh = np.transpose(np.asarray(Vnmh[2]))
    i_na = gamma_Na/(NNa/D_Na)*mh[-1,:]*(vm-Vna)
    i_k = gamma_K/(NK/D_K)*n[-1,:]*(vm-Vk)
    g_na = gamma_Na/(NNa/D_Na)*mh[-1,:]
    g_k = gamma_K/(NK/D_K)*n[-1,:]
    fig = plt.figure(figsize = [8, 10])
    plt.subplot(3,1,1)
    plt.plot(t, vm)
    plt.xticks([])
    plt.ylabel('Vm (mV)')
    plt.subplot(3,1,2)
    plt.plot(t, i_na)
    plt.plot(t, i_k)
    plt.legend(['ina','ik'])
    plt.xticks([])
    plt.ylabel('i (uA/cm^2)')
    plt.subplot(3,1,3)
    plt.plot(t, g_na)
    plt.plot(t, g_k)
    plt.legend(['g_na', 'g_k'])
    plt.xlabel('t (ms)')
    plt.ylabel('g (mS/cm^2)')



#%% deterministic model, step current I = 10uA/cm^2
i = 0
y_rk, t = deterministic_HH.euler([deterministic_HH.vmp_hh,
                                    deterministic_HH.np_hh, 
                                    deterministic_HH.mp_hh,
                                    deterministic_HH.hp_hh], 
                                    start = 0, stop = 250, step = 0.01, initial_values = [0, 0, 0, 0])
plot_deterministic(y_rk, t)

#%% stocastic mode, step current I = 10
NNa = 12000
NK = 3600
i = 0
y_rk, t = stocastic_HH.euler([stocastic_HH.vmp_hh,
									stocastic_HH.np_hh, 
									stocastic_HH.mhp_hh],
									start = 0, stop = 250, step = 0.01,
									initial_values = [0.0, np.asarray([NK,0,0,0,0]), np.asarray([NNa,0,0,0,0,0,0,0])])
plot_stocastic(y_rk, t)


