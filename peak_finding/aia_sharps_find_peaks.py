import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import os
import pickle
from multiprocessing import Pool
import tools.peak2 as peak

if __name__ == '__main__':
    # load GOES catalog and eliminate irrelevant entries
    flare_catalog = pd.read_csv('../flare_catalogs/goes_catalog_with_noaa_ar2.csv',na_values=' ')
    flare_catalog = flare_catalog.dropna(subset=['SHARP'])
    flares = flare_catalog[pd.to_numeric(flare_catalog['SHARP'])>=20]
    flares.SHARP = pd.to_numeric(flares['SHARP']).astype('int64')
    
    bhs = pd.read_csv('../sharps_badheaders.csv')
    bhs = [x[0] for x in bhs.values]
    flares = flares[~flares.SHARP.isin(bhs)]

    outofrange = pd.read_csv('../flares_with_limbs2.csv')
    outofrange = [x[0] for x in outofrange.values]
    flares = flares.drop(outofrange,errors='ignore')

    nes_noaa = pd.read_csv('../nonempty_sharps_with_noaa_ar.csv')
    flares = flares[flares.SHARP.isin(nes_noaa.HARPNUM)]
    flares = flares.reset_index().rename(columns={'index':'flare_ind'})

    lams = ['193','171','304','1600','131','94']
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']

    # directory of AIA fits files -- just to get list of downloaded SHARPs
    fits_dir = '/srv/data/sdo_sharps/aia_temp'

    dirs = os.listdir(fits_dir)
    dirs = sorted(dirs)

    df = pd.DataFrame()

    for sharpd in dirs:
        print('Working on ' + sharpd)
        flares_sharp = flares[flares['SHARP']==int(sharpd.split('_')[-1])]

        with open('aia_timeseries/'+sharpd+'_aia_sum_intensity', "rb") as fp:   #Pickling
            aia_data = pickle.load(fp)
        with open('aia_timeseries/'+sharpd+'_aia_times', "rb") as fp:   #Pickling
            aia_times = pickle.load(fp)

        fig,ax = plt.subplots(6,1,figsize=(21,10))

        for i in range(len(flares_sharp)):
            flare = flares_sharp.iloc[i]
            if flare['CMX'] == 'B' or (flare['CMX'] == 'C' and flare['CMX_VALUE'] <10):
                continue
            flare_date = str(flare.DATE)
            flare_start = str(flare.START_TIME).zfill(4)
            flare_end = str(flare.END_TIME).zfill(4)
            flare_start = datetime.strptime(flare_date+flare_start,'%y%m%d%H%M')
            flare_end = datetime.strptime(flare_date+flare_end,'%y%m%d%H%M')

            if flare_end < flare_start:
                flare_end = flare_end + timedelta(days=1)
            ax[0].axvspan(flare_start,flare_end,alpha=0.3,color='blue')
            ax[1].axvspan(flare_start,flare_end,alpha=0.3,color='blue')
            ax[2].axvspan(flare_start,flare_end,alpha=0.3,color='blue')
            ax[3].axvspan(flare_start,flare_end,alpha=0.3,color='blue')
            ax[4].axvspan(flare_start,flare_end,alpha=0.3,color='blue')
            ax[5].axvspan(flare_start,flare_end,alpha=0.3,color='blue')

        peak_data = peak.generate_peak_data(aia_data,aia_times,lams, flares_sharp)

        for j in range(len(lams)):
            lam = lams[j]
            color = colors[j]   
            sum_intensity = np.array(aia_data[j])
        
            times = np.array(aia_times[j])
            ax[j].plot(times,sum_intensity,'-',color=color,label=lam, alpha=1)
            ax[j].plot(times,np.median(sum_intensity)+0*sum_intensity,'--',color=color,label='_',alpha=0.5)
            if len(peak_data)>0:
                ax[j].plot(peak_data[lam+'_peak_time'],peak_data[lam+'_magnitude'],'x',color='black',label='_')
                ax[j].plot(peak_data[lam+'_start_time'],peak_data[lam+'_magnitude']-peak_data[lam+'_prominence'],'.',color='black',label='_')
            ax[j].legend()
            ax[j].grid()

        plt.title(sharpd)
        plt.savefig('../aia_timeseries/'+sharpd+'_aia_timeseries.png')
        plt.close()

        if len(peak_data)>0:
            peak_data['SHARP'] = int(sharpd.split('_')[-1])
            df = pd.concat([df,peak_data],ignore_index=True)

    cols = df.columns.tolist()
    print(cols)
    cols = cols[-1:]+cols[-7:-1]+cols[:-7]
    print(cols)
    df = df[cols]
    df.to_csv('../flares_catalog/aia_flares_catalog.csv',index=False)
    


    