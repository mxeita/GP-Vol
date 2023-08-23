#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:50:26 2023

@author: mohamedeita
"""
import numpy as np



# def generate_GP_Vol_sample (num_points:int=100, 
#                             a_param:float=0, b_param:float=0,
#                             sigma_e:float=1, krn_scale_para:float=1,
#                             krn_var_param:float=1):
#     log_var=[np.random.normal(loc=0,scale=sigma_e)]
#     returns=[np.random.normal(loc=0,scale=np.exp(log_var[0]))]
#     log_var_Sigma=np.array([krn_var_param])
    
#     for t in range(num_points):
#         mu_log_var=a_param*log_var[t]+b_param*returns[t]+
                              
#     return 5

def kernel(r_1:float,v_1:float,r_2:float,v_2:float,
           krn_scale_param:float=1, krn_var_param:float=1)->float:
    return krn_var_param*np.exp(-np.linalg.norm([r_2-r_1,v_2-v_1])**2
                  /(2*krn_scale_param**2))

# class CovMat(object):
    
#     def __init__ (self, T:int=500, krn_scale_param:float=1, krn_var_param:float=1):
#         self.x_T=np.zeros(T)
#         self.v_T=np.zeros(T)
#         self.Sigma=np.zeros((T,T))
#         self.t=0
        
#         self.krn_var_param=krn_var_param
#         self.krn_scale_param=krn_scale_param
   
#     def update(self,r_t,v_t):
#         self.t+=1
#         self.r_T[self.t]=r_t
#         self.v_T[self.t]
#         for k in range(self.t):
#             self.Sigma[self.t,k]=kernel(self.r_T[k], self.v_T[k], r_t, v_t)
#             self.Sigma[k,self.t]=kernel(self.r_T[k], self.v_T[k], r_t, v_t)
#         self.Sigma[self.t,self.t]=self.krn_var_param
#     def get_Sigma(self):
#         return self.Sigma[:self.t+1,:self.t+1]


# def mean_conditional_Gauss(x_T,v_T,theta,ker):

#     sigma=cov_mat(x_T,v_T,ker,theta,gamma)
#     sigma_2_2_inv=np.linalg.inv(sigma[:t+1,:t+1])
#     sigma_1_2=sigma[t+1,:t+1]
    
#     mu_t_1=a*v_T[t+1]+b*x_T[t+1]+sigma_1_2@sigma_2_2_inv@(f_T[:t+1]-a*v_T[:t+1]+b*x_T[:t+1])

class GaussianProcess():
    def __init__ (self, _x_t, _v_t, _theta, _macro=None):
        
        
        self.lag_x=len(self.theta.beta)
        self.lag_v=len(self.theta.alpha)
        m=max(self.lag_v,self.lag_x)
        
        self.v_t=np.concatenate((np.random.normal(m),_v_t))
       
        self.x_t=np.concatenate((np.random.normal(m),_x_t))
        
        self.theta=_theta
        self.macro=_macro
        
    def kernel(self,t_1,t_2):
        
        dist_sq=(self.x_t[range(t_2-self.lag_x,t_2)]-self.x_t[range(t_2-self.lag_x,t_2)])**2
        +(self.v_t[range(t_2-self.lag_v,t_2)]-self.v_t[range(t_2-self.lag_v,t_2)])**2
        if self.macro!=None:
            dist_sq=(self.macro[t_2]-self.macro[t_1])**2
        return self.theta.gamma*np.exp(dist_sq
                      /(2*self.theta.lamda**2))
    
    def get_mean (self):
        
        mean=np.zeros(len(self.x_t))
        
        for t in range(len(self.x_t)):
            mean[t]= np.dot(self.theta.alpha, self.x_t[range(t-self.lag_x,t)])
            
            mean[t]+=np.dot(self.theta.beta, self.v_t[range(t-self.lag_v,t)])
        
        return mean
    def get_covariance(self):
        n=len(self.x_t)+1
        cov=np.zeros(n,n)
        for i in range(n):
            for j in range(n):
                cov[i][j]=self.kernel(i,j)
    

def cov_mat(x_t,v_t,ker,krn_scale_param:float=1, krn_var_param:float=1):
    N=len(x_t)
    ans=np.zeros((N,N))
    for i in range (N):
        for j in range (N):
            ans[i][j]=ker(x_t[i],v_t[i],x_t[j],v_t[j],krn_scale_param, krn_var_param)
    return ans        
        



# def RACPF(x_t,N=200,shrink=0.95, T=200):
#     weights=np.zeros(N)
#     theta=np.zero(N,5)
#     v_T_hat=np.zero(N,T)
#     for i in range(N):
#         theta[i]=np.random.normal(0,1,5)
#         weights[i]=1/N
#     for t in range(T):
#         theta_avg=weights.T@theta
#         theta=shrink*theta+(1-shrink)*np.array([theta_avg]*N)

# def gaussian_pdf(x,mean,var):
#     if len(x)==1:
#         return 1/np.sqrt(2*np.pi*variance) * np.exp(-(x-mean)**2/(2*variance))
#     return 1/np.sqrt((2*np.pi)**len(x)*np.linalg.det(var))*np.exp(-1*np.dot(x-mean,np.linalg.inv(var)@(x-mean)))

class Prob_V_1_V():
    
    def get_params(v_t, x_t, theta):
        gp=GaussianProcess(x_t,v_t,theta)
    
        mu=gp.get_mean()
        n=len(mu)
        mu, mu_t=mu[range(n-1)],mu[-1]
    
        Sigma=gp.get_covariance()
        
        Sigma_2=Sigma[range(n-1),range(n-1)]
        Sigma_2_inv=np.linalg.inv(Sigma_2)
        Sigma_1=Sigma[-1,range(n-1)]
        mat_A=np.linalg.inv(1/(theta.sigma_e**2)*np.identity(n-1)+Sigma_2)
    
        return {'mean':mu_t+ Sigma_1 @ Sigma_2_inv @ mat_A @ (Sigma_2_inv@mu+1/(theta.sigma_e**2)*v_t),
                'var':Sigma_1@Sigma_2_inv@mat_A@Sigma_2_inv@Sigma_1+np.linalg.inv(Sigma)[-1,-1]+theta.sigma_e**2}
        
    def __init__(self, v_t, x_t, theta):
        gp=GaussianProcess(x_t,v_t,theta)
    
        mu=gp.get_mean()
        n=len(mu)
        mu, mu_t=mu[range(n-1)],mu[-1]
    
        Sigma=gp.get_covariance()
        Sigma_2=Sigma[range(n-1),range(n-1)]
        Sigma_2_inv=np.linalg.inv(Sigma_2)
        Sigma_1=Sigma[-1,range(n-1)]
        mat_A=np.linalg.inv(1/(theta.sigma_e**2)*np.identity(n-1)+Sigma_2)
    
        self.mean=get_params(v_t, x_t, theta)['mean']
    
        self.var=get_params(v_t, x_t, theta)['var']
        
        self.x_t=x_t
        self.v_t=v_t
        self.theta=theta
    
        
        
    def propagate(self):
        return np.normal(loc=self.mean,scale=self.var)
    
    def pdf(self, v_T,x_T,t):
        ans=1
        
        for i in range(t,len(x_T)):
            params=get_params(v_T[:t],x_T[:t], theta)
            mean=params['mean']
            var=params['var']
            ans=ans*gaussian_pdf(v_T[t], mean, var)
        return ans
            

def prob_x_t_1_v_t(x_t,v_t):
    var=np.exp(v_t)
    Sigma_inv=np.diag(1/var)
    
    
    
    return 1/np.sqrt(np.prod(var))*np.exp(-x_t@Sigma_inv@x_t)
    
        

def pgas_markov_kernel(x_T,v_T, theta, num_particles):
    num_timepoints=len(v_T)
    particles=np.zeros((num_particles,num_timepoints))
    weights=np.zeros(num_particles)
    weights_AS=np.zeros(num_particles)
    
    particles[range(num_particles-1),0]=np.random.normal(0,theta.sigma_e,size=num_particles-1)
    particles[-1,0]=v_T[0]
    weights=np.array([np.exp(-(x_T[0]**2)/(2*np.exp(v)) for v in particles[:,0])])
    weights=weights/sum(weights)
    
    for t in range(1,len(x_T)):
        indices=np.random.choice(range(num_particles-1),size=num_particles-1,
                                 replace=True, p=weights)
        particles[range(num_particles-1),:]=particles[indices,:]
        for i in range(num_particles-1):
            particles[i,t]=propgate(particles[i,range(t)],x_T[range(t)],theta)
        particles[i,-1]=x_T[t]
        weights_AS[i]=weights[i]*prob_x_t_1_v_t(x_T[t:], v_T[t:])
        
        
        
        
        
    
    



      