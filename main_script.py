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
import os
sys.path.insert(1, os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import deterministic_HH
import stochastic_HH
import stochastic_HH_matrix

import stochastic_HH2_matrix
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
    
# def plot_stocastic(Vnmh, t, Vna = 115, Vk = -12,gamma_K = 20e-9,gamma_Na = 20e-9,D_Na = 60e8, D_K = 18e8,NNa = 12000, NK = 3600):
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
    
def plot_stimulation(Vnmh, t, **kwargs):
    vm = np.asarray(Vnmh[0])
    try:
        stim_i = kwargs['stim_i']
        i = []
        for t_c in t:
            i.append(deterministic_HH.ip_hh(t_c, **kwargs))
        plt.figure(figsize = [8, 3])
        plt.plot(t, i)
        plt.xlabel('t (ms)')
        plt.ylabel('stim_i (uA/cm^2)')
    except KeyError:
        pass
    try:
        stil_g = kwargs['stim_g']
        i = []
        g = []
        for t_c, idx in zip(t, range(len(vm))):
            Vg = kwargs['Vg']
            g.append(deterministic_HH.gp_hh(t_c, **kwargs))
            i.append(deterministic_HH.gp_hh(t_c, **kwargs)*(vm[idx]-Vg))
        fig = plt.figure(figsize = [8, 3])
        ax1 = fig.subplots()
        color = 'tab:red'
        ax1.plot(t, i, color = color)
        ax1.set_xlabel('t (ms)')
        ax1.set_ylabel('stim_i (uA/cm^2)', color = color)
        ax1.tick_params(axis = 'y', labelcolor = color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.plot(t, g, color = color)
        ax2.set_ylabel('stim_g (mS/cm^2)')
        ax2.tick_params(axis = 'y', labelcolor = color)
    except KeyError:
        pass

def noisy_input(mu = 10, sigma = 7, tau = 1, idx = 0, fld = os.getcwd()):
    rand_norm = norm
    rand_norm.random_state=RandomState(seed=None)
    wn = rand_norm.rvs(mu,sigma, size = [7, int(5e4)])
    t = np.arange(0,(5e4)*1e-5, 1e-5)
    wn_conv = []
    wn_new = []
    for i in range(len(tau)):
        conv_f = t*np.exp(-t/(tau[i]/1e3))
        conv_f = conv_f/np.sum(conv_f) #normalize
        wn_conv.append(np.convolve(wn[i,:], conv_f))
        wn_new.append(mu + (wn_conv[i][int(2e4):int(5e4)]-np.mean(wn_conv[i][int(2e4):int(5e4)]))*sigma/np.std(wn_conv[i][int(2e4):int(5e4)]))

    noise_i = np.asarray(wn_new)

    vm_all = []
    n_all = []
    mh_all = []
    # h_all = []
    for i in range(noise_i.shape[0]):
        vm = []
        n = []
        mh = []
        # h = []
        IStim_Params = {
            'stim_i': True,
            'i_waveform' : noise_i[i,:],
            }
        for j in range(10):
            y_rk, t = stochastic_HH.euler([stochastic_HH.vmp_hh,
                                            stochastic_HH.np_hh,
                                            stochastic_HH.mhp_hh],
                                            start = 0, stop = 250, step = 0.01,
                                            initial_values = [0.0, np.asarray([3600,0,0,0,0]), np.asarray([12000,0,0,0,0,0,0,0])],
                                            **IStim_Params)
            vm.append(np.asarray(y_rk[0]))
            n.append(np.transpose(np.asarray(y_rk[1])))
            mh.append(np.transpose(np.asarray(y_rk[2])))
        vm_all.append(vm)
        n_all.append(n)
        mh_all.append(mh)
        # h_all.append(h)
    data = {'vm':vm_all,
            'n': n_all,
            'mh': mh_all,
            # 'h': h_all,
            't':t,
            'sigma': sigma,
            'mu': mu,
            'tau': tau,
            'noise_i': noise_i}
    result_folder = os.path.join(fld, "result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    sio.savemat(os.path.join(result_folder, 'stochastic_matrix2state_noise_mu%d_sigma%d_%d.mat' %(mu, sigma, idx)),data)

def waveform_input(I, idx = 0, fld = os.getcwd()):
    vm_all = []
    n_all = []
    mh_all = []
    for i in range(I.shape[0]):
        vm = []
        n = []
        mh = []

        IStim_Params = {
            'stim_i': True,
            'i_waveform' : I[i,:],
            'stim_start': 50,
            }
        for j in range(10):
            y_rk, t = stochastic_HH.euler([stochastic_HH.vmp_hh,
                                            stochastic_HH.np_hh,
                                            stochastic_HH.mhp_hh],
                                            start = 0, stop = 250, step = 0.01,
                                            initial_values = [0.0, np.asarray([3600,0,0,0,0]), np.asarray([12000,0,0,0,0,0,0,0])],
                                            **IStim_Params)
            vm.append(np.asarray(y_rk[0]))
            n.append(np.transpose(np.asarray(y_rk[1])))
            mh.append(np.transpose(np.asarray(y_rk[2])))
        vm_all.append(vm)
        n_all.append(n)
        mh_all.append(mh)

    data = {'vm':vm_all,
            'n': n_all,
            'mh': mh_all,
            't':t,
            'I': I}
    result_folder = os.path.join(fld, "result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    sio.savemat(os.path.join(result_folder, 'waveform_response_%d.mat'%idx),data)

#%% deterministic, step I = 10uA/cm^2
IStim_Params = {
    'stim_i': True,
    'i': 10,
    }
y_rk, t = deterministic_HH.euler([deterministic_HH.vmp_hh,
                                    deterministic_HH.np_hh, 
                                    deterministic_HH.mp_hh,
                                    deterministic_HH.hp_hh], 
                                    start = 0, stop = 250, step = 0.01, initial_values = [0, 0, 0, 0], **IStim_Params)
plot_deterministic(y_rk, t)
data = {'Vnmh':y_rk,
        't':t,
        'sigma':0,
        'mu':10}
fld = os.getcwd()
result_folder = os.path.join(fld, "result")
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
sio.savemat(os.path.join(result_folder, 'deterministic_noise_mu%d_sigma%d_%d.mat.mat'%(10, 0, i)),data)
#%% stocastic, I = 10
for j in range(6):
    vm = []
    n = []
    mh = []
    for i in range(10):
        IStim_Params = {
            'stim_i': True,
            'i': 10,
        }
        y_rk, t = stochastic_HH.euler([stochastic_HH.vmp_hh,
                                            stochastic_HH.np_hh, 
                                            stochastic_HH.mhp_hh],
                                            start = 0, stop = 250, step = 0.01,
                                            initial_values = [0.0, np.asarray([3600,0,0,0,0]), np.asarray([12000,0,0,0,0,0,0,0])], **IStim_Params)
        vm.append(np.asarray(y_rk[0]))
        n.append(np.transpose(np.asarray(y_rk[1])))
        mh.append(np.transpose(np.asarray(y_rk[2])))
        plot_stocastic(y_rk, t)
    data = {'vm':[vm],
            'n': [n],
            'mh': [mh],
            't':t}
    sio.savemat(os.path.join(result_folder, 'E:\\Code\\stochastic HH model\\result\\stochastic_noise_mu%d_sigma%d_%d.mat'%(10, 0, j)),data)

#%% noisy inputs
sigma = 7
mu = 10
tau = [1,3,5,10,15,20,25]

for j in range(6):
    noisy_input(mu, sigma, tau, j)

#%% waveform inputs
data = sio.loadmat(os.path.join(os.getcwd(), 'input', 'NMDAwaveform.mat'))
I = data["noise_i"]/6*6.7
waveform_input(I)

    
    