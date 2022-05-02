import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import h5py
import os
import cv2
from sklearn.decomposition import PCA
from pathlib import Path
import aia_sharps_movies
from data_analysis import analyzeFlare, alignPeaks
import scipy.signal as ss
import scipy.ndimage as sn
import sunpy.map
from aiapy.calibrate import normalize_exposure, register, update_pointing, correct_degradation
from aiapy.calibrate.util import get_correction_table
from astropy.io import fits

flare_catalog = pd.read_csv('goes_catalog_with_noaa_ar2.csv',na_values=' ')
flare_catalog = flare_catalog.dropna(subset=['SHARP'])
flares_MX = flare_catalog[np.logical_or(flare_catalog['CMX']=='M',flare_catalog['CMX']=='X')]
flares_MX = flares_MX[pd.to_numeric(flares_MX['SHARP'])>=20]
flares_MX.SHARP = pd.to_numeric(flares_MX['SHARP']).astype('int64')

print(len(flares_MX))
bhs = pd.read_csv('/home/kiva6588/Code/sharps_badheaders.csv')
bhs = [x[0] for x in bhs.values]
flares_MX = flares_MX[~flares_MX.SHARP.isin(bhs)]
print(len(flares_MX))
outofrange = pd.read_csv('/home/kiva6588/Code/flares_with_limbs2.csv')
outofrange = [x[0] for x in outofrange.values]
flares_MX = flares_MX.drop(outofrange,errors='ignore')
print(len(flares_MX))
nes_noaa = pd.read_csv('/home/kiva6588/Code/nonempty_sharps_with_noaa_ar.csv')
flares_MX = flares_MX[flares_MX.SHARP.isin(nes_noaa.HARPNUM)]
print(len(flares_MX))
flares_MX = flares_MX.reset_index().rename(columns={'index':'flare_ind'})
print(flares_MX.head)

lams = ['193','171','304','1600','131','94']
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
root_dir = '/srv/data/sdo_sharps/hdf5/aia_'
fits_dir = '/srv/data/sdo_sharps/aia_temp'
sos = ss.butter(3,0.15,output='sos')

peak_data = np.zeros((len(flares_MX),len(lams),4))
inds_nofiles = []
corr_table = get_correction_table()

for i in range(len(flares_MX)):
    flare = flares_MX.iloc[i]
    sharpd = 'sharp_'+str(flare.SHARP)

    flare_date = str(flare.DATE)
    flare_start = str(flare.START_TIME).zfill(4)
    flare_end = str(flare.END_TIME).zfill(4)
    flare_start = datetime.strptime(flare_date+flare_start,'%y%m%d%H%M')
    flare_end = datetime.strptime(flare_date+flare_end,'%y%m%d%H%M')
    if flare_end < flare_start:
        flare_end = flare_end + timedelta(days=1)
        
    # take window around flare 
    last_flare_start = str(flare_catalog.loc[flare['flare_ind']-1].DATE)+str(flare_catalog.loc[flare['flare_ind']-1].START_TIME).zfill(4)
    last_flare_end = str(flare_catalog.loc[flare['flare_ind']-1].DATE)+str(flare_catalog.loc[flare['flare_ind']-1].END_TIME).zfill(4)
    last_flare_start = datetime.strptime(last_flare_start,'%y%m%d%H%M')
    last_flare_end = datetime.strptime(last_flare_end,'%y%m%d%H%M')
    if last_flare_end < last_flare_start:
        last_flare_end = last_flare_end + timedelta(days=1)
    window_start = max(flare_start-timedelta(hours=1),last_flare_end)
    
    next_flare_start = str(flare_catalog.loc[flare['flare_ind']+1].DATE)+str(flare_catalog.loc[flare['flare_ind']+1].START_TIME).zfill(4)
    next_flare_start = datetime.strptime(next_flare_start,'%y%m%d%H%M')
    window_end = min(flare_end+timedelta(hours=2),next_flare_start)

    plt.figure()
    plt.axvspan(flare_start,flare_end,alpha=0.4,color='blue')
    n_nofiles = 0
    for j in range(len(lams)):
        lam = lams[j]
        color = colors[j]

        if not os.path.exists(root_dir + lam+os.sep+sharpd):
            print('Directory does not exist '+ root_dir + lam+os.sep+sharpd)
            n_nofiles +=1
            continue

        file_list = sorted(os.listdir(root_dir + lam+os.sep+sharpd),key=lambda x:x.split('.')[3])
        file_list = [file for file in file_list if file.split('.')[-2]==str(lam) and file.split('.')[1].split('_')[-1]=='60s']
        flare_files = [file for file in file_list if datetime.strptime(file.split('.')[3].strip('_TAI'),'%Y%m%d_%H%M%S')>=window_start and datetime.strptime(file.split('.')[3].strip('_TAI'),'%Y%m%d_%H%M%S')<=window_end]

        if len(flare_files) == 0:
            print('O files for flare ',flare['flare_ind'], flare_start, lam)
            n_nofiles += 1
            continue

        sum_intensity = []
        times = []
        for file in flare_files:
            try:
                file = root_dir + lam + os.sep+sharpd + os.sep+file
                fits_file = file.replace(root_dir+lam,fits_dir).replace('.hdf5','.fits')
                img_map = sunpy.map.Map(fits_file)
                img_map = normalize_exposure(img_map)
                img_map = correct_degradation(img_map,correction_table=corr_table)
                img_data = np.array(img_map.data)
                # img_data = np.float64(np.array(h5py.File(root_dir+ lam+os.sep+sharpd+os.sep+file,'r')['aia']))
                img_data[img_data<1] = 1
                sum_intensity.append(np.sum(img_data[:]))
                times.append(datetime.strptime(file.split('.')[3].strip('_TAI'),'%Y%m%d_%H%M%S'))
            except OSError:
                continue
        times = np.array(times)
        sum_intensity = np.array(sum_intensity)
        # plot 
        plt.plot(times,sum_intensity,'--',color=color,label='_',alpha=0.5)
        sum_intensity = sn.maximum_filter1d(sum_intensity,3)
        if len(sum_intensity) > 11:
            sum_intensity = ss.sosfiltfilt(sos,sum_intensity)
        if lam == '171' or lam == '193':
            # height = 0.15*np.abs(np.median(sum_intensity))+np.min(sum_intensity)
            height = 0.0*np.abs(np.median(sum_intensity))+np.median(sum_intensity)
            prominence=0.1*(np.max(sum_intensity)-np.median(sum_intensity))
        else:
            height = 0.05*np.abs(np.median(sum_intensity))+np.min(sum_intensity)
            prominence=0.35*(np.max(sum_intensity)-np.median(sum_intensity))
        peaks, props = ss.find_peaks(sum_intensity,height=height,prominence=prominence,width=4,rel_height=0.9)
        width_heights = props['width_heights']
        left = np.round(props['left_ips']).astype(int)
        right = np.round(props['right_ips']).astype(int)
        # filter out certain peaks
        inds_early = np.array(times)[peaks]<flare_start-timedelta(minutes=20)
        peaks = peaks[~inds_early]
        width_heights = width_heights[~inds_early]
        left = left[~inds_early]
        right = right[~inds_early]
        plt.plot(times,sum_intensity,label=lam,color=color)
        plt.plot(np.array(times)[peaks],np.array(sum_intensity)[peaks],'x',color='black',label='_')

        # save data - start, end and peak time
        if len(left) > 0:
            ind_max = np.argmax(sum_intensity[peaks])
            if (times[left[ind_max]]<times[left[0]] and times[right[ind_max]]>times[right[0]]) or times[peaks[ind_max]]<flare_end:
                ind_peak = ind_max
            else:
                ind_peak = 0
            plt.hlines(width_heights[ind_peak],times[left[ind_peak]],times[right[ind_peak]],color='tab:cyan')
            peak_data[i,j,0] = datetime.strftime((np.array(times)[left[ind_peak]]),'%Y%m%d_%H%M%S')
            peak_data[i,j,1] = datetime.strftime((np.array(times)[right[ind_peak]]),'%Y%m%d_%H%M%S')
            peak_data[i,j,2] = datetime.strftime((np.array(times)[peaks[ind_peak]]),'%Y%m%d_%H%M%S')
            peak_data[i,j,3] = sum_intensity[peaks[ind_peak]]

        else:
            peak_data[i,j,:] = np.NaN

    if n_nofiles == len(lams):
        inds_nofiles.append(i)
    else:
        plt.legend()
        plt.grid()
        plt.title('flare_'+str(flare['flare_ind'])+'_'+sharpd+'_'+datetime.strftime(flare_start,'%Y%m%d_%H%M')+'_'+flare['CMX']+str(flare['CMX_VALUE']))
        plt.savefig('aia_figures/flare_'+str(flare['flare_ind'])+'_'+sharpd+'_'+datetime.strftime(flare_start,'%Y%m%d_%H%M')+'_'+flare['CMX']+str(flare['CMX_VALUE'])+'_timeseries_corr.png')

for j in range(len(lams)):
    flares_MX[lams[j]+'_start'] = peak_data[:,j,0]
    flares_MX[lams[j]+'_end'] = peak_data[:,j,1]
    flares_MX[lams[j]+'_peak'] = peak_data[:,j,2]
    flares_MX[lams[j]+'_magnitude'] = peak_data[:,j,3]

flares_MX = flares_MX.drop(inds_nofiles,errors='ignore')
flares_MX.to_csv('flares_MX_catalog_3.csv')
print('Done, ', len(flares_MX) ,' flares')