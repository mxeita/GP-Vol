#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 13:36:18 2023

@author: mohamedeita
"""

import numpy as np
import math
from scipy.stats import multivariate_normal as mvn

class Theta():
    
    def __init__ (self):
        self.alpha=0
        self.beta=0
        self.sigma_e=0
        self.gamma=0
        self.lamda=0
        
        self.lag_v=0
        self.lag_x=0
    
    def set_params (self, _alpha, _beta, _sigma_e, _gamma, _lamda):
        
        self.alpha=_alpha
        self.beta=_beta
        self.sigma_e=_sigma_e
        self.gamma=_gamma
        self.lamda=_lamda
        
        self.lag_x=1 if type(self.beta)in [int , float,np.float_] else len(self.beta)
        self.lag_v=1 if type(self.alpha)in [int , float,np.float_] else len(self.alpha)  
    
    
    def __add__(self,other):
        alpha=self.alpha+other.alpha
        beta=self.beta+other.beta
        sigma_e=self.sigma_e+other.sigma_e
        gamma=self.gamma+other.gamma
        lamda=self.lamda+other.lamda
        
        ans=Theta()
        ans.set_params(alpha,beta,sigma_e,gamma,lamda)
       
        return ans
    
    def __sub__(self,other):
        alpha=self.alpha-other.alpha
        beta=self.beta-other.beta
        sigma_e=self.sigma_e-other.sigma_e
        gamma=self.gamma-other.gamma
        lamda=self.lamda-other.lamda
        
        ans=Theta()
        ans.set_params(alpha,beta,sigma_e,gamma,lamda)
        
        return ans
    
    def __mul__(self,scalar):
        alpha=self.alpha*scalar
        beta=self.beta*scalar
        sigma_e=self.sigma_e*scalar
        gamma=self.gamma*scalar
        lamda=self.lamda*scalar
        
        ans=Theta()
        ans.set_params(alpha,beta,sigma_e,gamma,lamda)
        
        return ans
    
    def __rmul__(self,scalar):
        alpha=self.alpha*scalar
        beta=self.beta*scalar
        sigma_e=self.sigma_e*scalar
        gamma=self.gamma*scalar
        lamda=self.lamda*scalar
        
        ans=Theta()
        ans.set_params(alpha,beta,sigma_e,gamma,lamda)
        
        return ans
    
    def __truediv__(self,scalar):
        alpha=self.alpha/scalar
        beta=self.beta/scalar
        sigma_e=self.sigma_e/scalar
        gamma=self.gamma/scalar
        lamda=self.lamda/scalar
        
        ans=Theta()
        ans.set_params(alpha,beta,sigma_e,gamma,lamda)
        
        return ans
    
    def to_list(self):
        alpha=[self.alpha] if type(self.alpha) in [int,float] else self.alpha
        beta=[self.beta] if type(self.beta) in [int,float] else self.beta
        
        arr=np.concatenate((alpha,beta,[self.sigma_e],[self.gamma],[self.lamda]))
        return list(arr)
    
    def __len__(self):
       return  self.lag_v+self.lag_x+3
   
    def __str__(self):
        ls=self.to_list()
        arr=np.array(ls)
        return arr.__str__()
    
    def set_random(self, _lag_v, _lag_x):
        
        self.alpha=np.random.uniform(low=-5, high=5, size=_lag_v)
        self.beta=np.random.uniform(low=-5, high=5, size=_lag_x)
        self.sigma_e=10**np.random.uniform(low=-4, high=1)
        self.gamma=10**np.random.uniform(low=0, high=1)
        self.lamda=10**np.random.uniform(low=-2, high=1)
        
        # self.sigma_e=0.2
        # self.gamma=0.5
        # self.lamda=0.05
        
        
        self.lag_v=_lag_v
        self.lag_x=_lag_x
   
def x_t_pdf(x_t, v_t):
    
    # n=1 if type(x_t) in [int,float,np.float64] else len(x_t)
    
    # covar=np.exp(v_t) if type(v_t) in [int,float,np.float64] else np.diag(np.exp(v_t))

    
    # if type(x_t) not in [np.ndarray,list]:
    #     x_t=np.array([x_t])
    # if type(v_t) not in [np.ndarray,list]:
    #     v_t=np.array([v_t])
    
    if np.any(v_t > 20.0)  or np.any(v_t< -20.0) or np.any(math.isnan(v_t)):
        return 0
    
    try:
        n=len(x_t)
    except:
        n=1
        
    _mean=np.zeros(n)
    _cov=np.exp(v_t)
 
    return mvn.pdf(x_t, mean=_mean, cov=_cov)
    
    

def x_t_rvs(v_t):

    # n=1 if type(v_t) in [int,float,np.float64] else len(v_t)
    
    # covar=np.exp(v_t) if type(v_t) in [int,float,np.float64] else np.diag(np.exp(v_t))
    
    try:
        n=len(v_t)
    except:
        n=1
        
    _loc=np.zeros(n)
    _scale=np.exp(v_t/2)
 
    return np.random.normal(loc=_loc, scale=_scale)


def invert_mat(mat):
    try:
        len(mat)
    except:
        return np.array([[1/mat]])
    mat_inv=np.array([[1/mat[0][0]]])
    print(mat_inv)
    for i in range(1,len(mat)):
        mat_inv_1=mat_inv
        mat_t=mat[i,:i]
        e=mat_inv_1@mat_t
    
        g=mat[i,i]-np.dot(e,mat_t)
        g=1/g
        
        new_sub_mat=mat_inv_1+g*np.tensordot(e, e,axes=0)
        new_row=-g*e
        new_col=np.append(new_row,g)
        new_col=np.array([new_col]).T
        
        mat_inv=np.hstack((np.vstack((new_sub_mat,new_row)),new_col))
        
    return mat_inv



class GaussianProcess():
    def __init__ (self, _theta, _macro=None):
        
        self.theta=_theta
        if np.any(_macro):
            self.macro=[_macro,_macro] if type(_macro)in [int , np.float_] else np.concatenate(([np.mean(_macro)],_macro))
        else:
            self.macro=_macro
        
        self.lag_x=1 if type(self.theta.beta)in [int , float,np.float_] else len(self.theta.beta)
        self.lag_v=1 if type(self.theta.alpha)in [int , float,np.float_] else len(self.theta.alpha)
        
        x_seed=np.zeros(self.lag_x)
        v_seed=np.zeros(self.lag_v)
        
        self.x_t=x_seed
        self.v_t=v_seed
        
        self.mean=np.array([0])
        self.Sigma=np.array([[self.theta.gamma]])
        
        self.Sigma_inv=np.linalg.inv(self.Sigma)
        # self.Sigma_1,self.Sigma_t=self.Sigma[:-1,:][:,:-1], self.Sigma[-1,:][:-1]
        # self.Sigma_1_inv=1/self.Sigma_1 if len(self.Sigma)==1 else np.linalg.inv(self.Sigma_1)
        # self.Sigma_inv_11=1/(np.linalg.inv(self.Sigma)[-1,-1])
        # self.Lambda=(1/(self.theta.sigma_e**2)+self.Sigma_1_inv)**(-1) if len(self.Sigma_1_inv)==1 else np.linalg.inv(1/(self.theta.sigma_e**2)*np.identity(len(self.Sigma_1_inv))+self.Sigma_1_inv)

        
    
    def set_params(self,_x_t,_v_t):
        
        # x_t=[_x_t] if type(_x_t)in [int , float, np.float_] else _x_t
        
        self.x_t=np.append(self.x_t,_x_t)
        
        
        # v_t=[_v_t] if type(_v_t)in [int , float, np.float_] else _v_t
        
        self.v_t=np.append(self.v_t,_v_t)
        
        self.find_mean()
        self.find_covariance()
        
    # def kernel(self,t_1,t_2):
        
    #     x_1=self.x_t[range(t_1,t_1+self.theta.lag_x)]
    #     x_2=self.x_t[range(t_2,t_2+self.theta.lag_x)]
    #     v_1=self.v_t[range(t_1,t_1+self.theta.lag_v)]
    #     v_2=self.v_t[range(t_2,t_2+self.theta.lag_v)]
        
        
    #     d_x=sum((x_2-x_1)**2)
    #     d_v=sum((v_2-v_1)**2)                
    #     dist_sq=d_x+d_v
        
    #     if self.macro!=None:
    #         dist_sq+=(self.macro[t_2]-self.macro[t_1])**2
            
    #     ans=self.theta.gamma*np.exp(-1*dist_sq
    #                   /(2*self.theta.lamda**2))        
    #     return ans
    
    def find_mean (self):
        
        n=len(self.x_t)-self.lag_x+1
        
        mean=np.zeros(n)

              
        alpha_reversed=np.flip(self.theta.alpha)
        beta_reversed=np.flip(self.theta.beta)
        
        for t in range(n):
            
            mean[t]= np.dot(alpha_reversed, self.v_t[t:][:self.lag_v])
            
            mean[t]+=np.dot( beta_reversed, self.x_t[t:][:self.lag_x])
        
        self.mean=mean
        self.mu_1,self.mu_t=self.mean[:-1],self.mean[-1]
        
        
    def find_covariance(self):
        
        # n=len(self.x_t)-self.lag_x+1
        
        # cov=np.zeros((n,n))
        
        # for t_i in range(n):
        #     for t_j in range(t_i,n):
        #         cov[t_i][t_j]=self.kernel(t_i,t_j)
        #         cov[t_j][t_i]=self.kernel(t_i,t_j)
        
        x=self.x_t
        v=self.v_t
        
        lag_x=self.lag_x
        lag_v=self.lag_v
        
        n_mat=len(x)-lag_x+1
        
        D_x=np.zeros((n_mat,n_mat,lag_x))
        D_v=np.zeros((n_mat,n_mat,lag_v))
        
        
        for i in range(lag_x):
            M=np.tile(x[range(i,i+n_mat)],(n_mat,1))
            D_x[:,:,i]=(M-M.T)**2
        D_x=np.sum(D_x, axis=2)
        
        self.D_x=D_x
         
        for i in range(lag_v):
            M=np.tile(v[range(i,i+n_mat)],(n_mat,1))
            D_v[:,:,i]=(M-M.T)**2
        D_v=np.sum(D_v,axis=2)
        
        self.D_v=D_v
        
        D=D_x+D_v
        self.D=D
        
        if np.any(self.macro):
            D+=(np.tile(self.macro[:n_mat],(n_mat,1))-np.tile(self.macro[:n_mat],(n_mat,1)).T)**2
        
        g=self.theta.gamma
        l=self.theta.lamda
        
        
        cov= g*np.exp(-1*D/(2*l**2))
        
        self.Sigma=cov
        # if  np.any(np.linalg.eigvals(self.Sigma)<0):
        #     print(np.linalg.eigvals(self.D))
        #     print('min Sigma: ', np.min(self.Sigma))
        
        
        
    def get_params(self, Lamda=False):
        
            
        mu_1,mu_t=self.mean[:-1],self.mean[-1]
        
        
        
        Sigma=self.Sigma
        #print(Sigma)
        Sigma_1,Sigma_t=Sigma[:-1,:][:,:-1], Sigma[-1,:][:-1]
        Sigma_1_inv=1/Sigma_1 if len(Sigma)==1 else np.linalg.pinv(Sigma_1)
        g=np.linalg.pinv(Sigma)
        #print(g)
        Sigma_inv_11=1/(g[-1,-1])
        
        # if not(np.all(np.linalg.eigvals(Sigma) >= 0.0)):
        #        print (np.linalg.eigvals(Sigma))
        
            
        if Lamda:
            Lambda=(1/(self.theta.sigma_e**2)+Sigma_1_inv)**(-1) if len(Sigma_1_inv)==1 else np.linalg.inv(1/(self.theta.sigma_e**2)*np.identity(len(Sigma_1_inv))+Sigma_1_inv)
            
            return [mu_1, mu_t,Sigma_1_inv, Sigma_t,
                    Sigma_inv_11,Lambda]
        return [mu_1, mu_t,Sigma_1_inv, Sigma_t,
                Sigma_inv_11]
    
    def update(self,x,v):
        self.x_t=np.append(self.x_t,x)
        self.v_t=np.append(self.v_t,v)
        
        
        self.find_mean()
        
        self.mu_1=self.mean[:-1]
        self.mu_t=self.mean[-1]
        
        

        self.Sigma_1_inv=self.Sigma_inv
        
        
        t_f=len(self.Sigma_1_inv)
        
        self.Sigma_t=np.array([self.kernel(t_j,t_f) for t_j in range(t_f)])
        
        
        #Updating inverse according to S-Morrison
        
        e=self.Sigma_1_inv@self.Sigma_t
        
    
        g=self.theta.gamma-np.dot(e,self.Sigma_t)
        #print(g)
        g=1/g
        
        new_sub_mat=self.Sigma_1_inv+g*np.tensordot(e, e,axes=0)
        #print(new_sub_mat)
        new_row=-g*e
        #print(new_row)
        new_col=np.append(new_row,g)
        new_col=np.array([new_col]).T
        #print(new_col)
        self.Sigma_inv=np.hstack((np.vstack((new_sub_mat,new_row)),new_col))
        
        # self.Sigma_inv_11=1/g
        
        # self.Lambda=(1/(self.theta.sigma_e**2)+self.Sigma_1_inv)**(-1) if len(self.Sigma_1_inv)==1 else np.linalg.inv(1/(self.theta.sigma_e**2)*np.identity(len(self.Sigma_1_inv))+self.Sigma_1_inv)

    def printout(self):
       
        print('x_t: ', self.x_t,'\n',
              'v_t: ',self.v_t,'\n',
              'mu_1: ',self.mu_1,'\n',
              'mu_t: ', self.mu_t,'\n',
              'Sigma_egval: ',np.linalg.eigvals(self.Sigma),'\n',
              'Sigma_inv_egval: ',np.linalg.eigvals(self.Sigma_inv)
              )
              
        #print(self.Sigma_1_inv)
        
        t_f=len(self.x_t)-self.lag_x
        
        self.Sigma_t=np.array([self.kernel(t_j,t_f) for t_j in range(t_f)])
        #print(self.Sigma_t)
        
        #Updating inverse according to S-Morrison
        
        e=self.Sigma_1_inv@self.Sigma_t
    
        g=self.theta.gamma-np.dot(e,self.Sigma_t)
        g=1/g
        
        new_sub_mat=self.Sigma_1_inv+g*np.tensordot(e, e,axes=0)
        #print(new_sub_mat)
        new_row=-g*e
        #print(new_row)
        new_col=np.append(new_row,g)
        new_col=np.array([new_col]).T
        #print(new_col)
        self.Sigma_inv=np.hstack((np.vstack((new_sub_mat,new_row)),new_col))
        
        self.Sigma_inv_11=g
        
        self.Lambda=(1/(self.theta.sigma_e**2)+self.Sigma_1_inv)**(-1) if len(self.Sigma_1_inv)==1 else np.linalg.inv(1/(self.theta.sigma_e**2)*np.identity(len(self.Sigma_1_inv))+self.Sigma_1_inv)

    
            
def f_t_rvs(f_t, v_t, x_t, theta, macro=None):
    

    gp=GaussianProcess(theta, _macro=macro)

    gp.set_params(x_t, v_t)

    
    # pm=pm=gp.get_params()

    
    # mu_f=pm[1]+np.dot(pm[3],np.dot(pm[2],(f_t-pm[0])))
    # var_f=pm[4]
    e_val,e_vec=np.linalg.eigh(gp.Sigma)
    Sigma_inv=e_vec@np.diag(1/e_val)@e_vec.T
    
    _loc=gp.mu_t-(Sigma_inv[-1,:][:-1]@(f_t-gp.mu_1))/Sigma_inv[-1,-1]
    _scale=np.sqrt(1/Sigma_inv[-1,-1])
    
    return np.random.normal(loc=_loc,scale=_scale)

def v_t_full(v_t,x_t,theta, mu=False, pdf=False, v=0, macro=None):

    gp=GaussianProcess(theta, _macro=macro)
        
    gp=GaussianProcess(theta)
    gp.set_params(x_t, v_t)
    
    #pm=gp.get_params(Lamda=True)
    
    # pm=[gp.mu_1, gp.mu_t,gp.Sigma_1_inv, gp.Sigma_t,
            #gp.Sigma_inv_11,gp.Lambda]
            
    
    
    #e_val,e_vec=np.linalg.eigh(gp.Sigma)
    #Sigma_inv=e_vec@np.diag(1/e_val)@e_vec.T
    Sigma_inv=np.linalg.inv(gp.Sigma)
    
    
    #e_val_d,e_vec_d=np.linalg.eigh(Sigma_inv[:-1,:][:,:-1]+np.identity(len(Sigma_inv)-1)/(theta.sigma_e**2))
    #Delta=e_vec_d@np.diag(1/e_val_d)@e_vec_d.T
    Delta=np.linalg.inv(Sigma_inv[:-1,:][:,:-1]+np.identity(len(Sigma_inv)-1)/(theta.sigma_e**2))
    
    # print(Delta.shape,Sigma_inv.shape,gp.mu_1.shape)
    
    mu_f_vec=Delta@Sigma_inv[:-1,:][:,:-1]@gp.mu_1+(Delta@v_t)/(theta.sigma_e**2)

    if len(v_t)>0:
        #mu_v=pm[1]+ pm[3] @ pm[2] @ pm[5] @ (pm[2]@pm[0]+v_t/(theta.sigma_e**2))
        mu_v=gp.mu_t-(Sigma_inv[-1,:-1]@(mu_f_vec-gp.mu_1))/Sigma_inv[-1,-1]
        if mu_v<-10 :
            mu_v=-5
        if mu_v>10:
            mu_v=5

        #var_v=np.absolute(pm[3]@pm[2]@pm[5]@pm[2]@pm[3]+pm[4]+theta.sigma_e**2)
        var_v=(Sigma_inv[-1,:-1]@Delta@Sigma_inv[:-1,-1])/Sigma_inv[-1,-1]**2+1/Sigma_inv[-1,-1]+theta.sigma_e**2
    else:
        mu_v=0
        var_v=theta.sigma_e**2
    if mu:
        return mu_v
    if pdf:

        return mvn.pdf(v,mean=mu_v,cov=np.abs(var_v))
    v_ans=np.random.normal(loc=mu_v,scale=np.sqrt(np.abs(var_v)))
    # v_ans= min(v_ans,20.0) if v_ans>0.0 else max(v_ans,-20.0)                       
    return v_ans

def get_empirical_var(theta_particles, weights):
    N=len(theta_particles)
    n=len(theta_particles[0])
    partcl_lst=np.zeros((N,n))
    for i in range(N):
        partcl_lst[i]=theta_particles[i].to_list()
    # partcl_cent=partcl_lst-np.mean(partcl_lst,axis=0)
    
    
    return np.cov(partcl_lst.T, ddof=0, aweights=weights)
 

def jitter(theta, var, shrink):
    _mean=theta.to_list()
    _cov=(1-shrink**2)*var
    vec=mvn.rvs(mean=_mean,cov=_cov)
    
    ans=Theta()
    ans.set_params(vec[:theta.lag_v],vec[theta.lag_v:][:-3],vec[-3],vec[-2],vec[-1])
    
    return ans
