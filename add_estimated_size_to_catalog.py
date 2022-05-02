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
    root_dir = '/srv/data/sdo_sharps/hdf5/aia_'

    flare_sizes = np.zeros((len(flare_catalog),len(lams)))
    flare_intensities = np.zeros((len(flare_catalog),len(lams)))
    cutout_sizes = np.zeros((len(flare_catalog),2))
    for i in flare_catalog.index:
        flare = flare_catalog.iloc[i]
        sharpd = 'sharp_' + str(int(flare.SHARP))
        ## Use the first listed wavelength to find the flare in the SHARP cutouts based on the area with the max intensity
        for k in range(len(lams)):
            lam = lams[k]
            if True:
            # if pd.isnull(flare[lam+'_peak_time']):
                file_list = sorted(os.listdir(root_dir + lam + os.sep + sharpd),key=lambda x:x.split('.')[3])
                file_list = [file for file in file_list if file.split('.')[-2]==lam and file.split('.')[1].split('_')[-1]=='60s']
                flare_files = [file for file in file_list if datetime.strptime(file.split('.')[3].strip('_TAI'),'%Y%m%d_%H%M%S')>=flare['aia_min_start_time'] and datetime.strptime(file.split('.')[3].strip('_TAI'),'%Y%m%d_%H%M%S')<=flare['aia_max_end_time']]
                if len(flare_files) == 0:
                    continue
                maxintensity = 0
                maxsumintensity = 0
                maxflaresize = 0
                ind_peak = 0
                for j in range(len(flare_files)):
                    try:
                        img_data = np.array(h5py.File(root_dir+lam+os.sep+sharpd+os.sep+flare_files[j],'r')['aia'])
                        img_max = np.max(img_data[:])
                        img_sum = np.sum(img_data[:])
                        # G_img = kernels.convolve(img_data,G)
                        flaresize = int(np.sum(img_data[:] >= img_max*0.85))
                        if img_sum > maxsumintensity:
                            maxsumintensity = img_sum
                        if flaresize > maxflaresize and img_max > 8000:
                            maxflaresize = flaresize
                            if k == 0 or filepath == '':
                                filepath = root_dir+ '_' + str(lams[k])+os.sep+sharpd+os.sep+flare_files[j]
                            Ny,Nx = np.shape(img_data)  
                    except (OSError, KeyError) as e:
                        # couldn't read a file fosr some reason
                        continue
            else:
                file = root_dir+lam +os.sep+sharpd+os.sep + 'aia.sharp_cea_60s.' + str(flare.SHARP)+'.'+flare[lam+'_peak_time'].strftime('%Y%m%d_%H%M%S') +'_TAI.' + lam + '.hdf5'
                img_data = np.array(h5py.File(file,'r')['aia'])
                img_max = np.max(img_data[:])
                maxflaresize = int(np.sum(img_data[:] >= img_max*0.85))*(img_max>8000)
                Ny,Nx = np.shape(img_data)   

            flare_sizes[i,k] = maxflaresize
            flare_intensities[i,k] = maxsumintensity
            cutout_sizes[i,:] = [Nx,Ny]
        print('Done flare ',i)

    for k in range(len(lams)):
        flare_catalog[lams[k]+'_est_size']=flare_sizes[:,k]
        flare_catalog[lams[k]+'_sum_intensity']=flare_intensities[:,k] 
    flare_catalog['Nx'] = cutout_sizes[:,0]
    flare_catalog['Ny'] = cutout_sizes[:,1]

    # flare_catalog.to_csv('aia_flares_catalog_2_updated_slot'+str(slot)+'.csv',index=False)               

    return flare_catalog

def main():
    flare_catalog = pd.read_csv('aia_flares_catalog_7.csv')
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
    df.to_csv('aia_flares_catalog_7_updated.csv',index=False)
    
if __name__=="__main__":
	main()
