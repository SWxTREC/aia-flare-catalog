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
import pickle
import kink
import astropy.units as u
from multiprocessing import Pool

def generate_series(sharpd, lams, corr_table):
    fits_dir = '/srv/data/sdo_sharps/aia_temp'

    if os.path.exists('aia_timeseries/'+sharpd+'_aia_sum_intensity'):
        return

    print('Working on '+sharpd)
    aia_data = []
    aia_times = []
    file_list = os.listdir(fits_dir+os.sep+sharpd)
    file_list = [file for file in file_list if file.split('.')[-1]=='fits']
    for lam in lams:
        lam_intensity = []
        lam_times = []
        lam_list = [file for file in file_list if file.split('.')[1].split('_')[-1]=='60s' and file.split('.')[-2]==lam]
        lam_list = sorted(lam_list,key=lambda x:x.split('.')[3])  
        print(len(lam_list),' files for AIA ',lam)
        for file in lam_list:
            try:
                fits_file = fits_dir + os.sep + sharpd + os.sep + file
                img_map = sunpy.map.Map(fits_file)
                img_map.data[img_map.data<1] = 1
                if img_map.exposure_time < 0.1*u.s:
                    continue
                img_map = normalize_exposure(img_map)
                img_map = correct_degradation(img_map,correction_table=corr_table)
                img_data = np.array(img_map.data)
                lam_intensity.append(np.sum(img_data[:]))
                lam_times.append(datetime.strptime(file.split('.')[3].strip('_TAI'),'%Y%m%d_%H%M%S'))
            except OSError:
                continue
        aia_data.append(lam_intensity)
        aia_times.append(lam_times)
        print('Done AIA '+lam)
    
    with open('aia_timeseries/'+sharpd+'_aia_sum_intensity', "wb") as fp:   #Pickling
        pickle.dump(aia_data, fp)
    with open('aia_timeseries/'+sharpd+'_aia_times', "wb") as fp:   #Pickling
        pickle.dump(aia_times, fp)
    return

if __name__ == '__main__':
    lams = ['193','171','304','1600','131','94']
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']

    fits_dir = '/srv/data/sdo_sharps/aia_temp'

    corr_table = get_correction_table()

    dirs = os.listdir(fits_dir)
    args = [(dir,lams,corr_table) for dir in dirs]

    with Pool(8) as p:
        p.starmap(generate_series,args)

    