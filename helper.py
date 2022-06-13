## Helper functions
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import pickle
from datetime import datetime,timedelta


def plot_flare(flare,aia_times,aia_data,window):
    lams = ['193','171','304','1600','131','94']
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']

    start = flare['mean_start_time']-timedelta(minutes=window)
    end = flare['mean_end_time']+timedelta(minutes=window)
    fig,ax = plt.subplots(6,1,figsize=(10,10))
    for j in range(len(lams)):
        lam = lams[j]
        color = colors[j]   

        sum_intensity = np.array(aia_data[j])
        times = np.array(aia_times[j])
        sum_intensity = sum_intensity[np.logical_and(times>np.datetime64(start),times<np.datetime64(end))]
        times = times[np.logical_and(times>np.datetime64(start),times<np.datetime64(end))]
        ax[j].axvspan(flare['goes_start_time'],flare['goes_end_time'],alpha=0.3,color='blue')
        if len(times) > 0:
            ax[j].plot(times,sum_intensity,'-',color=color,label=lam, alpha=1)
            ax[j].set_ylim(0.9*np.min(sum_intensity),1.1*np.max(sum_intensity))
        
        if pd.notna(flare[lam+'_peak_time']): 
            ax[j].plot(flare[lam+'_peak_time'],flare[lam+'_magnitude'],'.k',label='_')
            ax[j].plot(flare[lam+'_start_time'],flare[lam+'_magnitude']-flare[lam+'_prominence'],'xk',label='_')
            ax[j].axvline(flare[lam+'_end_time'],color='k',ls='--',label='_')
        ax[j].set_xlim(start,end)
        ax[j].legend()
        ax[j].grid()
    ax[0].set_title('SHARP '+str(flare['SHARP'])+': GOES flare '+str(int(flare['goes_flare_ind']))+' '+str(flare['CMX'])+str(flare['CMX_VALUE']/10)+' predicted '+str(flare['ert_pred_CMX'])+str(flare['ert_pred_intensity'])[:4])

def plot_flare_goes(flare,aia_times,aia_data,window):
    lams = ['193','171','304','1600','131','94']
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']

    start = flare['mean_start_time']-timedelta(minutes=window)
    end = flare['mean_end_time']+timedelta(minutes=window)
    fig,ax = plt.subplots(6,1,figsize=(10,10))
    for j in range(len(lams)):
        lam = lams[j]
        color = colors[j]   

        sum_intensity = np.array(aia_data[j])
        times = np.array(aia_times[j])
        sum_intensity = sum_intensity[np.logical_and(times>np.datetime64(start),times<np.datetime64(end))]
        times = times[np.logical_and(times>np.datetime64(start),times<np.datetime64(end))]

        ax[j].axvspan(flare['mean_start_time'],flare['mean_end_time'],alpha=0.3,color='blue')
        if len(times) > 0 and len(sum_intensity):
            ax[j].plot(times,sum_intensity,'-',color=color,label=lam, alpha=1)
            ax[j].set_ylim(0.9*np.min(sum_intensity),1.1*np.max(sum_intensity))
        ax[j].set_xlim(start,end)
        ax[j].legend()
        ax[j].grid()
    ax[0].set_title('SHARP '+str(flare['SHARP'])+': GOES flare '+str(int(flare['flare_ind']))+' '+str(flare['CMX'])+str(flare['CMX_VALUE']/10))
    print('start:',flare['mean_start_time'],', end:',flare['mean_end_time'])