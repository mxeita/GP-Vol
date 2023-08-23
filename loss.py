#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 21:14:49 2023

@author: eita
"""
import numpy as np
import prob as pr
from scipy.stats import multivariate_normal as mvn
from statsmodels.tsa.stattools import acovf as acf

def AD(true_var, pred_var):
    return np.abs(np.sqrt(true_var)-np.sqrt(pred_var))

def MAD(true_var, pred_var):
    return np.mean(np.abs(np.sqrt(true_var)-np.sqrt(pred_var)))

def MLAE(true_var, pred_var):
    return np.mean(np.log(np.abs(true_var-pred_var)))

def LAE(true_var, pred_var):
    return np.log(np.abs(true_var-pred_var))

def QLIKE(true_var, pred_var):
    return np.mean(true_var/pred_var+np.log(pred_var))

def qLIKE(true_var, pred_var):
    return true_var/pred_var+np.log(pred_var)

def HMSE (true_var, pred_var):
    return np.mean((true_var/pred_var-1)**2)

def HSE (true_var, pred_var):
    return (true_var/pred_var-1)**2

def p_log_lik (true_ret, pred_var):
    ans=0
    for k, ret in enumerate(true_ret):
        p=mvn.pdf(ret, mean=0, cov=pred_var[k])
        ans+=-np.log(p) if p>0 else 1e-100
        
    return ans/len(true_ret)

def dm_test(tru_val, pred_1, pred_2, func=AD):
    diff=func(tru_val, pred_1) - func(tru_val, pred_2)
    d_bar=np.mean(diff)
    ACF=acf(diff)
    var=ACF[0]
    sum_ACF=sum(ACF[1:int(len(diff)/3)])
    d_var=(var+2*sum_ACF)/len(diff)
    if d_var<=0:
        return 0
    z_score=d_bar/np.sqrt(d_var)
    print(z_score)
    #return 2*mvn.cdf(-np.abs(z_score),mean=0,cov=1)
    return z_score
    
    