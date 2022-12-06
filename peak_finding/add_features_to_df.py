import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from datetime import datetime,timedelta
import os
from astropy.io import fits
import h5py


def add_cnn_pred(flare_dir,df):
    cnn_pred = NaN*np.zeros(len(df))

    flares = np.load(flare_dir+'/test_testindices.npy')
    y_pred = np.load(flare_dir+'test_pred.npy')

    for i in range(len(flares)):
        if len(np.where(df['flare_ind']==flares[i])[0]) > 0:
            print(np.where(df['flare_ind']==flares[i])[0][0])
            ind = np.where(df['flare_ind']==flares[i])[0][0]
            cnn_pred[ind] = y_pred[i]

    flares = np.load(flare_dir+'/train_testindices.npy')
    y_pred = np.load(flare_dir+'/train_pred.npy')
            
    for i in range(len(flares)):
        if len(np.where(df['flare_ind']==flares[i])[0]) > 0:
            ind = np.where(df['flare_ind']==flares[i])[0][0]
            cnn_pred[ind] = y_pred[i]

    flares = np.load(flare_dir+'/valid_testindices.npy')
    y_pred = np.load(flare_dir+'/valid_pred.npy')
            
    for i in range(len(flares)):
        if len(np.where(df['flare_ind']==flares[i])[0]) > 0:
            ind = np.where(df['flare_ind']==flares[i])[0][0]
            cnn_pred[ind] = y_pred[i]

    df['cnn_pred2'] = cnn_pred

    return df

def add_NxNy(df):
    Nxs = np.zeros(len(df))
    Nys = np.zeros(len(df))
    for i in df.index:
        flare = df.iloc[i]
        img_data = np.array(h5py.File(flare.filename,'r')['aia'])
        Ny,Nx = np.shape(img_data)   
        Nxs[i] = Nx
        Nys[i] = Ny
    df['Nx'] = Nxs
    df['Ny'] = Nys
    return df


def add_location(header_dir,df):
    # lons = NaN*np.zeros(len(df))
    # lats = NaN*np.zeros(len(df))
    df['aia_min_start_time'] = pd.to_datetime(df['aia_min_start_time'])
    df['aia_max_end_time'] = pd.to_datetime(df['aia_max_end_time'])
    params = ['LAT_FWT','LON_FWT','AREA_ACR','USFLUXL','MEANGAM','MEANGBT','MEANGBZ','MEANGBH','MEANJZD','TOTUSJZ','MEANALP','MEANJZH','ABSNJZH','SAVNCPP','MEANPOT','TOTPOT','MEANSHR','SHRGT45','R_VALUE','NACR','SIZE_ACR','SIZE'] #SHARP params
    sharp_data = NaN*np.zeros((len(params),len(df)))
    for i in df.index:
        flare = df.iloc[i]
        # sharpno = flare.sharp
        # flare_start = datetime.strptime(flare['aia_min_start_time'],'%Y%m%d_%H%M%S')
        # flare_end = datetime.strptime(flare['aia_max_end_time'],'%Y%m%d_%H%M%S')

        sharpno = flare.SHARP
        flare_start = flare['aia_min_start_time']
        flare_end = flare['aia_max_end_time']

        if (flare_end-flare_start).total_seconds() < 12*60:
            flare_start = flare_end - timedelta(minutes=13)
        print('Flare ',i)
        try:
            file_list = os.listdir(header_dir + os.sep + 'sharp_' + str(int(sharpno)))
        except FileNotFoundError:
            file_list = []
            
        file_list = [file for file in file_list if len(file.split('.'))>3]
        flare_files = [file for file in file_list if datetime.strptime(file.split('.')[3].strip('_TAI'),'%Y%m%d_%H%M%S')>=flare_start and datetime.strptime(file.split('.')[3].strip('_TAI'),'%Y%m%d_%H%M%S')<=flare_end]

        flare_files2 = []
        for k in range(6):
            flare_files1 = [file for file in flare_files if file.split('.')[-1]==str(k+1)]
            if len(flare_files1) > 0:
                flare_files2 = flare_files1 
                break
        if len(flare_files2) > 0:
            flare_files = flare_files2

        flare_files = sorted(flare_files,key=lambda x:x.split('.')[3]) # sorted in chronological order

        # get location of flare - currently using center of SHARP but really would like to use center of flare
        if len(flare_files) > 0:
            header = fits.getheader(header_dir+os.sep+'sharp_'+str(int(sharpno))+os.sep+flare_files[0],1)
            print(flare_files[0])
            # print(repr(header))
            # break
            try:
                for k in range(len(params)):
                    sharp_data[k,i] = header[params[k]]
                # lons[i] = header['CRVAL1']
                # lats[i] = header['CRVAL2']
            except KeyError:
                print(flare_files[0])
                print('Key error for flare ', i)
        else: # use header from closest file
            flare_files = sorted(file_list,key=lambda x:abs((datetime.strptime(x.split('.')[3].strip('_TAI'),'%Y%m%d_%H%M%S')-flare_start).total_seconds()))
            print(len(flare_files))
            flare_files2 = []
            for k in range(6):
                flare_files1 = [file for file in flare_files if file.split('.')[-1]==str(k+1)]
                if len(flare_files1) > 0:
                    flare_files2 = flare_files1 
                    break
            if len(flare_files2) > 0:
                flare_files = flare_files2
            if len(flare_files) > 0:
                header = fits.getheader(header_dir+os.sep+'sharp_'+str(int(sharpno))+os.sep+flare_files[0],1)
                print(flare_files[0])
                # print(repr(header))
                # break
                try:
                    for k in range(len(params)):
                        sharp_data[k,i] = header[params[k]]
                    # lons[i] = header['CRVAL1']
                    # lats[i] = header['CRVAL2']
                except KeyError:
                    print(flare_files[0])
                    print('Key error for flare ', i)
            
    for k in range(len(params)):
        df[params[k]] = sharp_data[k,:]      

    return df

def main():

    df = pd.read_csv('../flare_catalogs/aia_flares_catalog_720s_3_updated.csv')
    df = add_location('/srv/data/sdo_sharps/hmi_temp',df)
    # df = add_NxNy(df)
    # df = add_cnn_pred(outdir,df)
    df.to_csv('../flare_catalogs/aia_flares_catalog_720s_updated.csv',index=False)


if __name__== "__main__":
    main()