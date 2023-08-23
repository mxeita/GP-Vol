#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:54:02 2023

@author: mohamedeita
"""

import rapcf
import pgas
import prob
import GP_vol
import yfinance as yfin
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import datetime as dt
import loss
from scipy import interpolate

### Estimating theta for synthetic data ###

## Generating data with lag=1 ##

# Setting the theta values for the four GP-Vol models that will 
# generate simulated data

thetas={}

theta_1=prob.Theta()
theta_1.set_params(0.35, 0.15, 0.5, 2, 0.3) # GP-Vol (1,1)
thetas['theta_1']=theta_1

theta_2=prob.Theta()
theta_2.set_params([0.35,0.1], 0.15, 0.5, 2, 0.3) # GP-Vol (2,1)
thetas['theta_2']=theta_2

theta_3=prob.Theta()
theta_3.set_params(0.35, [0.15,0.07], 0.5, 2, 0.3) # GP-Vol (1,2)
thetas['theta_3']=theta_3

theta_4=prob.Theta()
theta_4.set_params([0.35,0.1], [0.15,0.07], 0.5, 2, 0.3) # GP-Vol (2,2)
thetas['theta_4']=theta_4

# Generating the synthetic data

x_T_syn={}

for key in thetas.keys():
    x_T_syn[key]=x_T_syn_1=GP_vol.simulator(theta=thetas[key], T=150)['x_T']
    plt.plot(x_T_syn[key])



#Estimating thetas using RAPCF

def get_theta_series(theta, num_timepoints=100, num_series=20, num_particels=20
                     ,pgas=False):
    num_params=len(theta)
    theta_series=np.zeros((num_params,num_timepoints, num_series))

    for t in range(num_series):
        print ('theta_out:',t)
    # filename='theta_1_'+str(t)+'.csv'
    # directory='/Users/eita/Documents/GP-vol/files/'+filename
    
        x_T_syn=GP_vol.simulator(theta=theta, T=num_timepoints)['x_T']
        if pgas:
            for k in range(num_timepoints):
            theta_series[:,k,t]=pgas.fit(x_T_syn,
        else:
            theta_series[:,:,t]=rapcf.fit(_x_T_syn, perf=True,num_particles=num_particels,
                                      _lag_v=theta.lag_v, _lag_x=theta.lag_x)
    
    # theta_out=pd.DataFrame(theta_out)
    # theta_out.to_csv(directory)
    
        if t%5==4:
            print('mean=',np.mean(theta_series[:,:,t], axis=1))
            print('std=',np.sqrt(np.var(theta_series[:,:,t], axis=1)))
    return theta_series

t_ser=get_theta_series(theta_1,num_particels=2000)

t_ser_2=get_theta_series(theta_2,num_particels=3000, num_series=10)

t_ser_3=get_theta_series(theta_3,num_particels=3000, num_series=10)

t_ser_4=get_theta_series(theta_4,num_particels=4000, num_series=10)

t_ser_6=get_theta_series(theta_1, num_particels=800, num_series=10)


fig, axs= plt.subplots(3,2, constrained_layout = True)
# fig.tight_layout(pad=2.5)
fig.suptitle('Convergence of RAPCF in GP-Vol(1,2)')

for i, ax in enumerate(axs.flat):
    if i==len(theta_3.to_list()):
        break
    tru=theta_3.to_list()[i]
    offset=[1,1,1,1,6,4]
    ax.set_ylim(bottom=tru-offset[i],top=tru+offset[i])
    ax.plot(np.mean(t_ser_3[i,:,:], axis=1), label='mean')
    ax.plot(np.quantile(t_ser_3[i,:,:],0.9,axis=1), label='90 % q')
    ax.plot(np.quantile(t_ser_3[i, :,:],0.1,axis=1), label='10 % q')
    ax.plot([tru]*100, label='true')
    ax.tick_params(axis='x', labelsize=6)
    ax.legend(loc='lower left', prop={'size':3})
    ax.set_title(f'Parameter {i}', fontsize=8)

plt.savefig('theta_3.png', dpi=800)


#Estimating thetas using PGAS

v_T_ref=np.random.uniform(low=0.01, high=3, size=len(x_T_syn['theta_1']))

thata_hat_pgas=pgas.fit(x_T_syn['theta_1'], v_T_ref)
    



## Generating data with lag=2##

theta_2=prob.Theta()
theta_2.set_params([0.3,0.15],[0.2,0.1],0.5,3,0.05)

x_T_syn_2=GP_vol.simulator(theta=theta_2, T=100, )['x_T']

theta_2_ls=[]

for t in range(20):
    print ('theta_out:',t)
    filename='theta_1_'+str(t)+'.csv'
    directory='/Users/eita/Documents/GP-vol/files/'+filename
    
    x_T_syn_1=GP_vol.simulator(theta=theta_1, T=100)['x_T']
    theta_out=rapcf.fit(x_T_syn_1, perf=True,num_particles=2000)
    theta_ls.append(theta_out)
    
    theta_out=pd.DataFrame(theta_out)
    theta_out.to_csv(directory)
    
    if t%5==4:
        print('mean=',np.mean([arr[:,-1] for arr in theta_ls], axis=0))
        print('std=',np.sqrt(np.var([arr[:,-1] for arr in theta_ls], axis=0))

#Estimating theta_1 using RAPCF

theta_2_estim=rapcf.fit(x_T_syn_2, perf=True,num_particles=1600)

#Estimating theta_1 using PGAS

##############################################################################



####    REAL DATA ####
#RACPF(data=x_T,num_particles=200, shrink=0.95 )

### Estimating theta for real data ###


# Loading the data for the tickers into a datframe



data_days=210
end_dt=dt.date.today()
start_dt=end_dt-dt.timedelta(days=data_days)

tickers=["AAPL", "MSFT", "AMZN","NVDA","GOOGL", "META",
         "TSLA", "JPM", "JNJ","XOM", "V", "AVGO","PG","LLY","HD"]

macro_tick="^IRX"



date_index=[start_dt+dt.timedelta(days=k) for k in range(data_days)]
returns_df=pd.DataFrame(index=date_index,columns=tickers+['macro'])



t=yfin.Ticker(macro_tick)
macro_data=t.history(start=start_dt , end=end_dt)['Close']
macro_data=100*(macro_data-np.mean(macro_data))/np.sqrt(np.var(macro_data))
macro_data.index=[ind.date() for ind in macro_data.index]



returns_df['macro']=macro_data

for tick in tickers:
    t=yfin.Ticker(tick)
    data=t.history(start=start_dt-dt.timedelta(days=1),end=end_dt)['Close']
    data.index=[ind.date() for ind in data.index]
    returns_df[tick]=100*data.pct_change()


not_all_NA=[not np.all(returns_df.iloc[k].isna()) for k in range(len(returns_df))]
returns_df=returns_df[not_all_NA]
returns_df.isna().any()

# Creating a dataframe of realised volatilities


start_dt_rvol=end_dt-dt.timedelta(days=59)
end_dt_rvol=end_dt

rvol_df=pd.DataFrame(index=[start_dt_rvol+dt.timedelta(days=k)for k in range(60)],
                     columns=tickers)
for tick in tickers:
    t=yfin.Ticker(tick)
    rvol_data=t.history(start=start_dt_rvol,end=end_dt_rvol, interval="5m")['Close']
    rvol_data.index=[ind.date() for ind in rvol_data.index]
    
    dates=[ind for ind in np.unique(rvol_data.index)]
    t_df=pd.DataFrame(index=dates, columns=[tick])
    
    for day in np.unique(rvol_data.index):
        rvol_day=rvol_data[rvol_data.index==day]
        returns_day=100*rvol_day.pct_change()
        t_df[t_df.index==day]=len(rvol_day)*np.var(returns_day)
    rvol_df[tick]=t_df



# Checking for Na or null values in the realised volatilities dataframe

not_all_NA=[not np.all(rvol_df.iloc[k].isna()) for k in range(len(rvol_df))]
rvol_df=rvol_df[not_all_NA]
rvol_df.isna().any()
rvol_df.isnull().any()



loss_func_names=['MAD', 'MLAE', 'QLIKE', 'HMSE', 'p_log_lik']
loss_func=[loss.MAD, loss.MLAE, loss.QLIKE, loss.HMSE, loss.p_log_lik]


#### COMPARING PGAS vs RAPCF ####

v,t,w,p = rapcf.fit(returns_df['JPM'],num_particles=2000)
pred_var_RAPCF=p[-len(rvol_df):]

pred_var_PGAS=[]

for i in range(-len(rvol_df),0):
    v,t=pgas.fit(returns_df['JPM'][:i],
                 np.random.uniform(-2,2,size=len(returns_df['JPM'][:i]))
                 ,num_iterations=50)
    v_pred=prob.v_t_full(v, returns_df['JPM'][:i], t, mu=True)
    pred_var_PGAS.append(v_pred)
   
pred_var_PGAS=np.exp(np.array(pred_var_PGAS))


p_val_dm=loss.dm_test(np.array(rvol_df['JPM'].values[[0,2]], dtype=np.float64),
                      pred_var_RAPCF[[0,2]], pred_var_PGAS[[0,2]],func=p_log_lik)
print(p_val_dm)
#### COMPARING DIFFERENT GP-Vol MODELS ###

### Comparing RAPCF with lag=1, lag=2, lag=1+ macro and lag=2 + macro ###


## Model 1 GP-Vol(1,1)
pred_vol_df=pd.DataFrame(index=rvol_df.index, columns=tickers)
perform_df=pd.DataFrame(index=tickers,columns=loss_func_names)

for tick in tickers:
    print(tick)
    v,t,w,pred_var=rapcf.fit(returns_df[tick], num_particles=2000)
    pred_vol_df[tick]=pred_var[-len(rvol_df):]

    for i, f in enumerate(loss_func):
        tru_val=np.array(rvol_df[tick], dtype=np.float64)
        prd_val=np.array(pred_vol_df[tick],dtype=np.float64)
    
        if loss_func_names[i]=='p_log_lik':
            tru_val=np.array(returns_df[tick][-len(rvol_df):],dtype=np.float64)
       
        loss=f(tru_val,prd_val)

        perform_df[loss_func_names[i]][perform_df.index==tick]=loss
    print(perform_df)


#Model 2 GP-Vol(1,1) with macro
pred_vol2_df=pd.DataFrame(index=rvol_df.index, columns=tickers)
perform2_df=pd.DataFrame(index=tickers,columns=loss_func_names)

for tick in tickers:
    print(tick)
    v,t,w,pred_var=rapcf.fit(returns_df[tick], num_particles=2000, macro=returns_df['macro'])
    pred_vol2_df[tick]=pred_var[-len(rvol_df):]

    for i, f in enumerate(loss_func):
        tru_val=np.array(rvol_df[tick], dtype=np.float64)
        prd_val=np.array(pred_vol2_df[tick],dtype=np.float64)
        
        if loss_func_names[i]=='p_log_lik':
            tru_val=np.array(returns_df[tick][-len(rvol_df):],dtype=np.float64)
        
        loss=f(tru_val,prd_val)
      
        perform2_df[loss_func_names[i]][perform2_df.index==tick]=loss
    print(perform2_df)

stats_df=pd.DataFrame(index=tickers,columns=loss_func_names[:-1])
for tick in tickers:
    tru_val=np.array(rvol_df[tick],dtype=np.float64)
    print(tru_val)
    pred_1=np.array(pred_vol_df[tick],dtype=np.float64)
    print(pred_1)
    pred_2=np.array(pred_vol2_df[tick],dtype=np.float64)
    print(pred_2)
    for i, f in enumerate([loss.AD,loss.LAE,loss.qLIKE,loss.HSE]):
        stats_df[loss_func_names[i]][stats_df.index==tick]=loss.dm_test(tru_val,
                                                                        pred_1,
                                                                        pred_2,
                                                                        func=f)
                                                                    
    print (stats_df)
    
    
    
# Model 3 GP-Vol(2,2)
pred_vol3_df=pd.DataFrame(index=rvol_df.index, columns=tickers)
perform3_df=pd.DataFrame(index=tickers,columns=loss_func_names)

for tick in tickers:
    print(tick)
    v,t,w,pred_var=rapcf.fit(returns_df[tick], num_particles=2000, _lag_x=2,_lag_v=2)
    pred_vol3_df[tick]=pred_var[-len(rvol_df):]

    for i, f in enumerate(loss_func):
        tru_val=np.array(rvol_df[tick], dtype=np.float64)
        prd_val=np.array(pred_vol3_df[tick],dtype=np.float64)
        
        if loss_func_names[i]=='p_log_lik':
            tru_val=np.array(returns_df[tick][-len(rvol_df):],dtype=np.float64)
        
        loss=f(tru_val,prd_val)
     
        perform3_df[loss_func_names[i]][perform3_df.index==tick]=loss
    print(perform3_df)
    
stats2_df=pd.DataFrame(index=tickers,columns=loss_func_names[:-1])
for tick in tickers:
    tru_val=np.array(rvol_df[tick],dtype=np.float64)
    print(tru_val)
    pred_1=np.array(pred_vol_df[tick],dtype=np.float64)
    print(pred_1)
    pred_2=np.array(pred_vol3_df[tick],dtype=np.float64)
    print(pred_2)
    for i, f in enumerate([loss.AD,loss.LAE,loss.qLIKE,loss.HSE]):
        stats2_df[loss_func_names[i]][stats2_df.index==tick]=loss.dm_test(tru_val,
                                                                        pred_1,
                                                                        pred_2,
                                                                        func=f)
                                                                    
    print (stats2_df)
    
gp_pll_df=pd.DataFrame(index=tickers)
gp_pll_df['GP-Vol(1,1)']=-1*perform_df[perform_df.columns[-1]]
gp_pll_df['GP-Vol(1,1)+']=-1*perform2_df[perform2_df.columns[-1]]
gp_pll_df['GP-Vol(2,2)']=-1*perform3_df[perform3_df.columns[-1]]
print(gp_pll_df)
#### COMPARING GP-VOL VS GARCH VS EGARCH ####


## GARCH ##

pred_vol_garch_df=pd.DataFrame(index=rvol_df.index, columns=tickers)
perform_garch_df=pd.DataFrame(index=tickers,columns=loss_func_names)

for tick in tickers:
    pred_gar=[]
    for t in range(len(rvol_df)):
        print(tick,'at',t)
        am=arch_model(returns[tick][:(t-len(rvol_df))])
        res=am.fit(disp='off')
        forecast=res.forecast(reindex=False, method='simulation')
        # forecast.variance.index+=dt.timedelta(days=1)
        pred_gar.append(forecast.variance.iloc[0,0])
    pred_vol_garch_df[tick]=pred_gar
    
    for i, f in enumerate(loss_func):
        tru_val=np.array(rvol_df[tick], dtype=np.float64)
        prd_val=np.array(pred_vol_garch_df[tick],dtype=np.float64)
        
        if loss_func_names[i]=='p_log_lik':
            tru_val=np.array(returns_df[tick][-len(rvol_df):],dtype=np.float64)
            
        lss=f(tru_val,prd_val)
        print(lss)
        perform_garch_df[loss_func_names[i]][perform_garch_df.index==tick]=lss

stats_gar_df=pd.DataFrame(index=tickers,columns=loss_func_names[:-1])
for tick in tickers:
    tru_val=np.array(rvol_df[tick],dtype=np.float64)
    print(tru_val)
    pred_1=np.array(pred_vol_df[tick],dtype=np.float64)
    print(pred_1)
    pred_2=np.array(pred_gar,dtype=np.float64)
    print(pred_2)
    for i, f in enumerate([loss.AD,loss.LAE,loss.qLIKE,loss.HSE]):
        stats_gar_df[loss_func_names[i]][stats_gar_df.index==tick]=loss.dm_test(tru_val,
                                                                        pred_1,
                                                                        pred_2,
                                                                        func=f)

## TARCH ##

pred_vol_tgarch_df=pd.DataFrame(index=rvol_df.index, columns=tickers)
perform_tgarch_df=pd.DataFrame(index=tickers,columns=loss_func_names)

for tick in tickers:
    pred_tar=[]
    for t in range(len(rvol_df)):
        print(tick,'at',t)
        am=arch_model(returns[tick][:(t-len(rvol_df))],p=1, o=1, q=1, power=1.0)
        res=am.fit(disp='off')
        forecast=res.forecast(reindex=False, method='simulation')
        # forecast.variance.index+=dt.timedelta(days=1)
        pred_tar.append(forecast.variance.iloc[0,0])
    pred_vol_tgarch_df[tick]=pred_tar
    
    for i, f in enumerate(loss_func):
        tru_val=np.array(rvol_df[tick], dtype=np.float64)
        prd_val=np.array(pred_vol_tgarch_df[tick],dtype=np.float64)
        if loss_func_names[i]=='p_log_lik':
            tru_val=np.array(returns_df[tick][-len(rvol_df):],dtype=np.float64)
            
        lss=f(tru_val,prd_val)
        print(lss)
        perform_tgarch_df[loss_func_names[i]][perform_tgarch_df.index==tick]=lss

stats_tar_df=pd.DataFrame(index=tickers,columns=loss_func_names[:-1])
for tick in tickers:
    tru_val=np.array(rvol_df[tick],dtype=np.float64)
    print(tru_val)
    pred_1=np.array(pred_vol_df[tick],dtype=np.float64)
    print(pred_1)
    pred_2=np.array(pred_tar,dtype=np.float64)
    print(pred_2)
    for i, f in enumerate([loss.AD,loss.LAE,loss.qLIKE,loss.HSE]):
        stats_tar_df[loss_func_names[i]][stats_tar_df.index==tick]=loss.dm_test(tru_val,
                                                                        pred_1,
                                                                        pred_2,
                                                                        func=f)

## GJR-GARCH

pred_vol_gjr_df=pd.DataFrame(index=rvol_df.index, columns=tickers)
perform_gjr_df=pd.DataFrame(index=tickers,columns=loss_func_names)

for tick in tickers:
    pred_gjr=[]
    for t in range(len(rvol_df)):
        print(tick,'at',t)
        am=arch_model(returns[tick][:(t-len(rvol_df))],p=1, o=1, q=1)
        res=am.fit(disp='off')
        forecast=res.forecast(reindex=False, method='simulation')
        # forecast.variance.index+=dt.timedelta(days=1)
        pred_gjr.append(forecast.variance.iloc[0,0])
    pred_vol_gjr_df[tick]=pred_gjr
    
    for i, f in enumerate(loss_func):
        tru_val=np.array(rvol_df[tick], dtype=np.float64)
        prd_val=np.array(pred_vol_gjr_df[tick],dtype=np.float64)
        if loss_func_names[i]=='p_log_lik':
            tru_val=np.array(returns_df[tick][-len(rvol_df):],dtype=np.float64)
            
        lss=f(tru_val,prd_val)
        print(lss)
        perform_gjr_df[loss_func_names[i]][perform_garch_df.index==tick]=lss

stats_gjr_df=pd.DataFrame(index=tickers,columns=loss_func_names[:-1])
for tick in tickers:
    tru_val=np.array(rvol_df[tick],dtype=np.float64)
    print(tru_val)
    pred_1=np.array(pred_vol_df[tick],dtype=np.float64)
    print(pred_1)
    pred_2=np.array(pred_gjr,dtype=np.float64)
    print(pred_2)
    for i, f in enumerate([loss.AD,loss.LAE,loss.qLIKE,loss.HSE]):
        stats_gjr_df[loss_func_names[i]][stats_gjr_df.index==tick]=loss.dm_test(tru_val,
                                                                        pred_1,
                                                                        pred_2,
                                                                        func=f)
    
gp_gar_ppl_df=pd.DataFrame(index=tickers)
gp_gar_ppl_df['GP-Vol(1,1)']=-1*perform_df[perform_df.columns[-1]]
gp_gar_ppl_df['GARCH(1,1)']=-1*perform_garch_df[perform_df.columns[-1]]
gp_gar_ppl_df['TARCH']=-1*perform_tgarch_df[perform_df.columns[-1]]
gp_gar_ppl_df['GJR-GARCH']=-1*perform_gjr_df[perform_df.columns[-1]]


### VOLATILITY FUNCTION ###

v_prtcl, t_prtcl, wght, dmp = rapcf.fit(returns_df['META'].values,
                                        num_particles=2000)
## Plotting v_t vs  v_t-1##
v_t_1=np.dot(wght, v_prtcl)
x_t_1=np.copy(returns_df['META'].values)
x_t_1[-1]=0
the=np.dot(wght,t_prtcl)
v_out=[]
for v_arg in np.arange(-7,7,0.1):
    v_t_1[-1]=v_arg
    print(v_t_1[-1])
    v_out.append(prob.v_t_full(v_t_1[-70:], x_t_1[-70:], 
                               the, mu=True))
bspline = interpolate.make_interp_spline(np.arange(-7,7,0.1), v_out, k=1)
v_out = bspline(np.arange(-7,7,0.001))
plt.plot(np.arange(-7,7,0.001),v_out)
plt.title('Plotting $v_t$ against  $v_{t-1}$')
plt.xlabel('$v_{t-1}$')
plt.ylabel('$v_t$')
plt.show()

## Plotting v_t vs  x_t-1##
the=np.dot(wght,t_prtcl)
v_out=[]
v_t_1[-1]=2
for x_arg in np.arange(-30,30,1):
    x_t_1[-1]=x_arg
    v,t,w=rapcf.update(x_t_1, v_prtcl, t_prtcl, wght)
    print(x_t_1[-1])
    #v_out.append(prob.v_t_full(v_t_1[-10:], x_t_1[-10:], 
                               #the, mu=True))
    v_out.append(np.dot(w,v)[-1])
bspline = interpolate.make_interp_spline(np.arange(-30,30,1), v_out)
v_out = bspline(np.arange(-30,30,0.0001))
plt.plot(np.arange(-30,30,0.0001),v_out)
plt.title('Plotting $v_t$ against  $x_{t-1}$')
plt.xlabel('$x_{t-1}$')
plt.ylabel('$v_t$')
plt.show()
    
rapcf_p={}
for tick in tickers:
    v,t,w=rapcf.fit(returns[tick].values[:100])
    pred_log_lik=[]
    for k in range(1,50):
        print(tick,'at',k)
        p=rapcf.pred_log_lik(returns[tick].values,v,t,w)
        pred_log_lik.append(p)
        v,t,w=rapcf.update(returns[tick].values,v,t,w)
    rapcf_p[tick]=np.mean(pred_log_lik)
    print(rapcf_p)
#### COMPARING PGAS VS RAPCF ####

