#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:41:53 2023

@author: mohamedeita
"""

import numpy as np
import pandas as pd
import prob
import matplotlib.pyplot as plt
from multiprocessing import Pool


def fit(x_T, num_particles=800, shrink=0.98, _lag_x=1, _lag_v=1, perf=False, macro=None):
    
    
    #Create and randomly set the values of theta partices
    
    theta_particles=[prob.Theta() for i in range(num_particles)]
    for particle in theta_particles:
        particle.set_random(_lag_v, _lag_x)
    theta_particles=np.array(theta_particles)
    
    # for i in range(num_particles):
    #     theta_particles[i].alpha=np.random.normal(0,sigma_prior,lag_v)**2
    #     theta_particles[i].beta=np.random.normal(0,sigma_prior,lag_x)**2
    #     theta_particles[i].sigma_e=np.absolute(np.random.normal(1,sigma_prior/100))
    #     theta_particles[i].gamma=np.absolute(np.random.normal(1,sigma_prior))
    #     theta_particles[i].lamda=np.random.uniform(low=0.1)
    
    #Intitalise the weights and zero vectors
    
    weights=np.array([1/num_particles]*num_particles)
    weights_g=np.zeros(num_particles)
    mu=np.zeros(num_particles)
    v_particles=np.zeros((num_particles,len(x_T)))
    pred_var=np.zeros(len(x_T))
    #gp_particles=[prob.GaussianProcess(theta_particles[i]) for i in range(num_particles)]
    #gp_particles=np.array(gp_particles)
    
    # t_ls=np.zeros((num_particles,7))
    # t_sum=np.zeros((7,len(x_T)))
    # t_alpha_1=np.zeros(200)
    if perf:
        theta_perf=np.zeros( ( len(theta_particles[0]),len(x_T) ) )
        
    
    for t in range(len(x_T)):
        #mu_i_dbg=pd.DataFrame(columns=['x_T','v_T','v_mu','theta'])
        
        if t==len(x_T)-1:
            print('\x1b[2K', 'Fitting:','100 % DONE !')
        else:
            print('\x1b[2K','Fitting:',int(t/(len(x_T)-1)*100),'%', end='')
    
        
        theta_mean=np.dot(weights,theta_particles)
        theta_var=prob.get_empirical_var(theta_particles, weights)
        #print(np.linalg.eigvals(theta_var))
        
        
        # Shrink theta #
        theta_particles=shrink*theta_particles+(1-shrink)*theta_mean
        
        #Compute new mu#
        
        
        # mu_inputs=[(v_particles[i,:t], x_T[:t], theta_particles[i]) for i in range(num_particles)]
        
        # with Pool() as pool:
        #     for i, mu in enumerate(pool.starmap(get_mu, mu_inputs)):
        #     # report the value to show progress
        #         weights_g[i]=weights[i]*prob.x_t_pdf(x_T[t], mu)
                
        for i in range(num_particles):
            mu[i]=prob.v_t_full(v_particles[i,:t], x_T[:t], theta_particles[i],mu=True, macro=macro)
            #mu_i_dbg.loc[i]=[x_T[:t],v_particles[i,:t],mu[i],theta_particles[i].to_list()]
        pred_var[t]=np.exp(np.dot(weights,mu))
       
        #filename='mu_'+str(t)+'.csv
        #directory='/Users/mohamedeita/Documents/GP-vol/files/'+filename
        #mu_i_dbg.to_csv(directory)
                
  
        #Compute g-weights#


        for i in range(num_particles):
            # print(f'weight{i}: ',weights[i])
            # print(f'mu {i}: ', mu[i])
            # print( f'prob {i}: ',prob.x_t_pdf(x_T[t],mu[i]))
            weights_g[i]=weights[i]*prob.x_t_pdf(x_T[t], mu[i])
        
        if np.all(weights_g==0):
            weights_g=np.array([1/num_particles]*num_particles)
        else:
            weights_g/=sum(weights_g)

        
        #Resample auxillary indices and propagate
        indices=np.random.choice(range(num_particles),num_particles,replace=True,p=weights_g)
        v_particles=v_particles[indices]
        theta_particles=theta_particles[indices]
        #gp_particles=gp_particles[indices]
        weights_g=weights_g[indices]
        weights_g=weights_g/sum(weights_g)
        
        
        #print('weights_g',weights_g)
        
        #weights_g=weights_g/sum(weights_g)
        
        #Add jitter to theta
        for i in range(num_particles):
            
            theta_particles[i]=prob.jitter(theta_particles[i], var=theta_var,shrink=shrink)

            v_particles[i,t]=prob.v_t_full(v_particles[i,:t], x_T[:t], theta_particles[i], macro=macro)
            #gp_particles[i].update(x_T[t],v_particles[i,t])
            weights[i]=prob.x_t_pdf(x_T[t], v_particles[i,t])/weights_g[i]
        
        if np.all(weights==0):
            weights=np.array([1/num_particles]*num_particles)
        else:
            weights/=sum(weights) 
       
        
        if perf:
            theta_perf[:,t]=np.dot(weights,theta_particles).to_list()

        
        # for i in range(num_particles):
        #     t_ls[i]=theta_particles[i].to_list()
        #     t_sum[:,t]=np.mean(t_ls,axis=0)
        #print('weights',weights, sum(weights))
        # t_alpha_1[t]=np.dot(weights,theta_particles).gamma
        #probs=[prob.x_t_pdf(x_T[t], v_particles[i,t]) for i in range(num_particles)]
        # plt.scatter(t,np.w)
        # plt.title(f'v_t at {t}')
        # plt.show()
    if perf:
        return theta_perf
    return v_particles, theta_particles, weights, pred_var

# plt.plot(np.dot(w,v)[:100])
# plt.plot(simulation['v_T'][:100])
# plt.show()






def pred_log_lik(_data, _v_particles, _theta_particles, _weights):
    ans=0
    num_particles=len(_weights)
    num_time_points=len(_v_particles[0,:])
    for i in range(num_particles):
        v_t=prob.v_t_full(_v_particles[i,:], _data[:num_time_points], _theta_particles[i], mu=True)
        p=prob.x_t_pdf(_data[num_time_points], v_t)
        ans+=_weights[i]*p
    return np.log(ans)







def update(_data, _v_particles, _theta_particles, _weights, shrink=0.98):
    
    num_particles=len(_weights)
    num_time_points=len(_v_particles[0])
    t=num_time_points
    mu=np.zeros(num_particles)
    weights_g=np.zeros(num_particles)
    
    v_particles = np.hstack( ( _v_particles,np.zeros( (num_particles, 1) ) ) )
    theta_particles=np.copy(_theta_particles)
    weights=np.copy(_weights)
    if np.all(weights==0):
        weights=np.array([1/num_particles]*num_particles)
    
    theta_mean=np.dot(weights,theta_particles)
    theta_var=prob.get_empirical_var(theta_particles, weights)
    
    # Shrink theta #
    theta_particles=shrink*theta_particles+(1-shrink)*theta_mean
    
    #Compute new mu#
    for i in range(num_particles):
        mu[i]=prob.v_t_full(v_particles[i,:t], _data[:t], theta_particles[i],mu=True)    
  
    #Compute g-weights#


    for i in range(num_particles):
        weights_g[i]=weights[i]*prob.x_t_pdf(_data[t-1], mu[i])
    if np.all(weights_g==0):
        weights_g=np.array([1/num_particles]*num_particles)
       
    weights_g=weights_g/sum(weights_g)

    
    
    #Resample auxillary indices and propagate
    
    indices=np.random.choice(range(num_particles),num_particles,replace=True,p=weights_g)
    v_particles=v_particles[indices]
    theta_particles=theta_particles[indices]

    weights_g=weights_g[indices]
    weights_g=weights_g/sum(weights_g)
    
    #Add jitter to theta
    
    for i in range(num_particles):
        
        theta_particles[i]=prob.jitter(theta_particles[i], var=theta_var,shrink=shrink)
        v_particles[i,t]=prob.v_t_full(v_particles[i,:t], _data[:t], theta_particles[i])
        weights[i]=prob.x_t_pdf(_data[t-1], v_particles[i,t])/weights_g[i]
    
    if np.all(weights==0):
        weights=np.array([1/num_particles]*num_particles)
    weights=weights/sum(weights)
    
    return v_particles, theta_particles, weights 