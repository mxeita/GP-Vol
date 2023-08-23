#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:43:17 2023

@author: mohamedeita
"""
import numpy as np
import prob
import matplotlib.pyplot as plt


def simulator(theta, T:int):
    
    
    f_T=np.zeros(T+1)
    v_T=np.zeros(T)
    x_T=np.zeros(T)
    
    # gp=prob.GaussianProcess(theta)
    
    # f_0_mean=gp.mean[0]
    # f_0_var=gp.Sigma[0][0]
    
    f_T[0]=np.random.normal(loc=0,scale=np.sqrt(theta.gamma))
    #gp=prob.GaussianProcess(theta)
   
    for t in range (T):
        
        if t==T-1:
            print('\x1b[2K','Generating:','100 % DONE !')
        else:
            print('\x1b[2K','Generating:',int(t/(T-1)*100),'%', end='')

        v_T[t]=f_T[t]+np.random.normal(loc=0,scale=theta.sigma_e)
        x_T[t]=np.random.normal(loc=0,scale=np.exp(v_T[t]/2))
        #gp.update(x_T[t],v_T[t])
        
        f_T[t+1]=prob.f_t_rvs(f_T[:t+1], v_T[:t+1], x_T[:t+1], theta)
   
    return {'x_T':x_T, 'v_T':v_T, 'f_T':f_T}

# theta=prob.Theta()
# theta.set_params(0.4, 0.3 , 0.2, 0.5, 0.05)
# simulation=simulator(theta, T=150)


# plt.plot(simulation['x_T'])
# plt.scatter(np.exp(simulation['v_T']/2),simulation['x_T'])
# plt.plot(simulation['v_T'],0.4*simulation['v_T'])
# plt.show()