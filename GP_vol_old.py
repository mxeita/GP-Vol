#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:43:17 2023

@author: mohamedeita
"""
import numpy as np



def simulator(theta, T:int):
    
    lag_v=len(theta.alpha)
    lag_x=len(theta.beta)
    
    lag=max(len(theta.alpha),len(theta.beta))
    v_T_seed=np.random.normal(loc=0,scale=theta.sigma_e,size=lag)
    x_T_seed=np.random.normal(loc=0,scale=np.sqrt(np.exp(v_T_seed)),size=lag)
    
    f_T=np.zeros(T)
    v_T=np.zeros(T)
    x_T=np.zeros(T)
    
    f_T[0]=np.random.normal(loc=np.dot(v_T_seed[-lag_v:],theta.alpha)+
                            np.dot(x_T_seed[-lag_x],theta.beta)
                            , scale=np.sqrt(theta.gamma))
   
    for t in range (T):
        v_T[t]=f_T[t]+np.random.normal(loc=0,scale=theta.sigma_e)
        x_T[t]=np.random.normal(loc=0,scale=np.sqrt(np.exp(v_T[t])))
        
        sigma=cov_mat(x_T[:t+2],v_T[:t+2],ker,lamda,gamma)
        sigma_2_2_inv=np.linalg.inv(sigma[:t+1,:t+1])
        sigma_1_2=sigma[t+1,:t+1]
        
        mu_t_1=a*v_T[t+1]+b*x_T[t+1]+sigma_1_2@sigma_2_2_inv@(f_T[:t+1]-a*v_T[:t+1]+b*x_T[:t+1])
        sigma_t_1=1/np.linalg.inv(sigma)[t+1,t+1]
        f_T[t+1]=np.random.normal(mu_t_1,sigma_t_1)
        print(t)    
    return x_T, v_T, f_T 