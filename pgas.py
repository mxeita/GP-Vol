#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:42:41 2023

@author: mohamedeita
"""
import numpy as np
import prob


def prob_AS(v_ref, v_obs, x_T, theta,max_trunc=20):
    ans=1
    trunc=min(max_trunc, len(v_ref))
    for k in range(trunc):
        _v=v_ref[k]
        v_known=np.append(v_obs,v_ref[:k])
        t=k+len(v_obs)
        ans*=prob.v_t_full(v_known, x_T[:t], theta, pdf=True , v=_v)*10
    return ans


def markov_kernel(x_T,v_T, theta, num_particles=25):
    #Creating arrays for the particles and weights
    num_timepoints=len(v_T)
    v_particles=np.zeros((num_particles,num_timepoints))
    weights=np.zeros(num_particles)
    weights_AS=np.zeros(num_particles)
    
    # Initialising the values of the particles and arrays at time 0
    v_particles[range(num_particles-1),0]=np.random.normal(0,theta.sigma_e,size=num_particles-1)
    v_particles[-1,0]=v_T[0]
    weights=np.array([prob.x_t_pdf(x_T[0], v) for v in v_particles[:,0]])
    weights=weights/sum(weights)
    
    for t in range(1,len(x_T)):
        
        if t==len(x_T)-1:
            print('\x1b[2K', 'Markov kernel:','100 % DONE !')
        else:
            print('\x1b[2K','Markov kernel:',int(t/(len(x_T)-1)*100),'%', end='')
        
        # Resampling particles 1,...,N-1
        indices=np.random.choice(range(num_particles),size=num_particles-1,
                                 replace=True, p=weights)
        v_particles[range(num_particles-1),:]=v_particles[indices,:]
        
        # Propgating particlea 1...N-1
        for i in range(num_particles-1):
            v_particles[i,t]=prob.v_t_full(v_particles[i,:t],x_T[:t],theta)
        v_particles[-1,t]=v_T[t]
        
    
        # Ancestor sampling for particle N
        for i in range(num_particles):
            weights_AS[i]=weights[i]*prob_AS(v_T[t:],v_particles[i,:t],x_T, theta)
        
        if np.all(weights_AS==0):
            weights_AS=np.array([1/num_particles]*num_particles)
        else:
            weights_AS/=sum(weights_AS)
                
        ancestor_index=np.random.choice(range(num_particles),size=1, p=weights_AS)
        v_particles[-1,:t]=v_particles[ancestor_index,:t]
        
        for i in range(num_particles):
            weights[i]=prob.x_t_pdf(x_T[t], v_particles[i,t])
        weights/=sum(weights)
    
    ans_index=int(np.random.choice(range(num_particles),size=1, p=weights))
    return v_particles[ans_index,:]

def fit(_x_T, _v_T, lag_v=1, lag_x=1, num_iterations=100, shrink=0.95, perf=False):
    
    x_T=_x_T
    v_T=_v_T
    
    num_theta_partcls=600
    theta_particles=[prob.Theta() for i in range(num_theta_partcls)]
    for particle in theta_particles:
        particle.set_random(1, 1)
    theta_particles=np.array(theta_particles)
    
    theta_weights=np.array([1/num_theta_partcls]*num_theta_partcls)
    if perf:
        theta_arr=np.zeros((5,num_iterations))
    
    for i in range(num_iterations):
        
        #Get empirical mean and var of theta
        
        theta_mean=np.dot(theta_weights,theta_particles)
        theta_var=prob.get_empirical_var(theta_particles, theta_weights)
        #print(np.linalg.eigvals(theta_var))
        
        
        # Shrink theta #
        theta_particles=shrink*theta_particles+(1-shrink)*theta_mean
        
        v_T=markov_kernel(x_T, v_T, theta_mean)
        
        # Importance sampling theta
        # theta_particles=[prob.Theta() for i in range(num_theta_partcls)]
        # for particle in theta_particles:
        #     particle.set_random(lag_v,lag_x)
        # theta_particles=np.array(theta_particles)
        
        for k in range(num_theta_partcls):
           
            theta_particles[i]=prob.jitter(theta_particles[i], var=theta_var,shrink=shrink)
            theta_particles[i].sigma_e=np.abs(theta_particles[i].sigma_e)
            theta_particles[i].gamma=np.abs(theta_particles[i].gamma)
            theta_weights[i]=prob_AS(v_T[1:], v_T[:1], x_T, theta_particles[i])
        
        
        # for i in range(num_theta_partcls):
        #     print('Particle: ', i)
        #     theta_weights[i]=prob_AS(v_T[1:], v_T[:1], x_T, theta_particles[i])
        theta_weights/=sum(theta_weights)
        indices=np.random.choice(range(num_theta_partcls),size=num_theta_partcls, p=theta_weights)
        theta_particles=theta_particles[indices]
        
        print('Interation no.', i)
        print('theta_estimate:',theta_mean)
        #print('theta_var', theta_var)
        if perf:
            theta_arr[:,i]=theta_mean.to_list()
    if perf:
        return theta_arr
    return v_T, np.dot(theta_weights,theta_particles)
