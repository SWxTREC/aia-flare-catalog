import h5py
import os
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
import csv
from datetime import datetime,timedelta
import kernels
from joblib import Parallel, delayed

def add_size(flare_catalog,slot):
    lams = ['193','171','304','1600','131','94']
    root_dir = '/srv/data/sdo_sharps/hdf5_processed/aia_'

    flare_sizes = np.zeros((len(flare_catalog),len(lams)))
    flare_intensities = np.zeros((len(flare_catalog),len(lams)))
    cutout_sizes = np.zeros((len(flare_catalog),2))
    for i in flare_catalog.index:
        flare = flare_catalog.iloc[i]
        sharpd = 'sharp_' + str(int(flare.SHARP))
        if (flare['goes_flare_ind'].isnull() and flare['ert_pred_CMX'] == 'C') or (flare['CMX'] == 'B' or flare['CMX'] == 'C'):
            continue

        for k in range(len(lams)):
            lam = lams[k]
            file_list = sorted(os.listdir(root_dir + lam + os.sep + sharpd),key=lambda x:x.split('.')[3])
            file_list = [file for file in file_list if file.split('.')[-2]==lam and file.split('.')[1].split('_')[-1]=='60s']


        print('Done flare ',i)             

    return flare_catalog

def main():
    flare_catalog = pd.read_csv('../flare_catalogs/aia_flares_catalog_verified.csv')
    lams = ['193','171','304','1600','131','94']
    flare_catalog['aia_min_start_time'] = pd.to_datetime(flare_catalog['aia_min_start_time'])
    flare_catalog['aia_max_end_time'] = pd.to_datetime(flare_catalog['aia_max_end_time'])

    for lam in lams:
        flare_catalog[lam+'_start_time'] = pd.to_datetime(flare_catalog[lam+'_start_time'])
        flare_catalog[lam+'_end_time'] = pd.to_datetime(flare_catalog[lam+'_end_time'])
        flare_catalog[lam+'_peak_time'] = pd.to_datetime(flare_catalog[lam+'_peak_time'])

    # num_cores = 9
    # slots = np.linspace(0,flare_catalog.shape[0],num_cores).astype(int);
    # print(slots)
    # Parallel(n_jobs=num_cores)(delayed(add_size)(flare_catalog.iloc[slots[i]:slots[i+1]],i)for i in np.arange(slots.shape[0]-1))

    # df = pd.DataFrame
    # for i in np.arange(slots.shape[0]-1):
    #     df = pd.concat([df,pd.read_csv('aia_flares_catalog_2_updated_slot'+str(i)+'.csv',ignore_index=True)])
    df = add_size(flare_catalog,0)
    df.to_csv('../flare_catalogs/aia_flares_catalog_7_verified.csv',index=False)
    
if __name__=="__main__":
	main()
