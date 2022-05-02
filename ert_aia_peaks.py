import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load
import h5py
import time
import gzip, pickle, pickletools
import glob,os
import re
import random
from sklearn.model_selection import GridSearchCV
import json
from xgboost import XGBRegressor


def predict(ralphie,X,y):
    y_ralph = ralphie.predict(X)

    MSE = mean_squared_error(y*3-6,y_ralph*3-6)
    MAE = mean_absolute_error(y*3-6,y_ralph*3-6)
    r2 = r2_score(y*3-6,y_ralph*3-6)
    print('RMSE: ',np.sqrt(MSE))
    print('MAE: ',MAE)
    print('R2: ', r2)
    y_ralph_MX = np.squeeze((y_ralph[y>=(1/3.)])*3-6)
    y_true_MX = (y[y>=(1/3.)])*3-6
    l2err = np.sqrt(np.sum((y_true_MX-y_ralph_MX)**2)/np.sum(y_true_MX**2))*100
    print("Percent L2 error on MX (in log of intensity): {:.2f}".format(l2err))
    return y_ralph, [np.sqrt(MSE),MAE,r2,l2err]

def generateTrainValidData(df,config,cols,label_col):
    seed = config['seed']

    df['flare'] =(((np.log10(df['goes_magnitude']))+6)/3.)    

    flares_C = df['goes_flare_ind'][np.logical_and(df['CMX']=='C',df['CMX_VALUE']>=10)].unique().astype('int64')
    flares_M = df['goes_flare_ind'][df['CMX']=='M'].unique().astype('int64')
    flares_X = df['goes_flare_ind'][df['CMX']=='X'].unique().astype('int64')
    
    random.seed(seed)
    random.shuffle(flares_C)
    random.shuffle(flares_M)
    random.shuffle(flares_X)
    split_train_C = int(np.round(0.8*flares_C.shape[0])) 
    split_valid_C = int(np.round(1.0*flares_C.shape[0]))
    split_train_M = int(np.round(0.8*flares_M.shape[0])) 
    split_valid_M = int(np.round(1.0*flares_M.shape[0]))
    split_train_X = int(np.round(0.8*flares_X.shape[0])) 
    split_valid_X = int(np.round(1.0*flares_X.shape[0]))
    train_flares_C = flares_C[:split_train_C]
    valid_flares_C = flares_C[split_train_C:split_valid_C]
    train_flares_M = flares_M[:split_train_M]
    valid_flares_M = flares_M[split_train_M:split_valid_M]
    train_flares_X = flares_X[:split_train_X]
    valid_flares_X = flares_X[split_train_X:split_valid_X]
    # train_flares = train_flares_C.append(train_flares_M).append(train_flares_X)
    # valid_flares = valid_flares_C.append(valid_flares_M).append(valid_flares_X)
    train_flares = np.concatenate([train_flares_C,train_flares_M,train_flares_X])
    valid_flares = np.concatenate([valid_flares_C,valid_flares_M,valid_flares_X])
    df_train = df[df['goes_flare_ind'].isin(train_flares)] 
    df_valid = df[df['goes_flare_ind'].isin(valid_flares)]
    df_test = df[df['goes_flare_ind'].isnull()].sample(frac=1).reset_index(drop=True)

    y_train = df_train[label_col]
    y_valid = df_valid[label_col]
    y_test = df_test[label_col]
    print('Features: ',cols)
    data = df_train[cols].to_numpy()
    scaler = StandardScaler()
    scaler.fit(data)

    data = scaler.transform(data)
    df_new = pd.DataFrame(data, columns=[x for x in cols])
    df_new.insert(loc=0, column='SHARP', value=df_train['SHARP'].values)
    df_new.insert(loc=1, column='goes_flare_ind', value=df_train['goes_flare_ind'].values)
    df_train = df_new

    data = df_valid[cols].to_numpy()
    data = scaler.transform(data)
    df_new = pd.DataFrame(data, columns=[x for x in cols])
    df_new.insert(loc=0, column='SHARP', value=df_valid['SHARP'].values)
    df_new.insert(loc=1, column='goes_flare_ind', value=df_valid['goes_flare_ind'].values)
    df_valid = df_new

    data = df_test[cols].to_numpy()
    data = scaler.transform(data)
    df_new = pd.DataFrame(data, columns=[x for x in cols])
    df_new.insert(loc=0, column='SHARP', value=df_test['SHARP'].values)
    df_new.insert(loc=1, column='flare_time', value=df_test['aia_min_start_time'].values)
    df_test = df_new

    df_train = df_train.replace([np.inf, -np.inf, np.nan], 0)
    # df_train = df_train.dropna()
    df_valid = df_valid.replace([np.inf, -np.inf, np.nan], 0)
    # df_valid = df_valid.dropna()
    df_test = df_test.replace([np.inf, -np.inf , np.nan], 0)
    # df_test = df_test.dropna()

    print('Number of C/M/X flares in training set: ',len(train_flares_C),'/',len(train_flares_M),'/',len(train_flares_X))
    print('Number of C/M/X flares in valid set: ',len(valid_flares_C),'/',len(valid_flares_M),'/',len(valid_flares_X))

    trainX, trainY = np.array(df_train), np.array(y_train)
    validX, validY = np.array(df_valid), np.array(y_valid)
    testX, testY = np.array(df_test), np.array(y_test)

    return [trainX, trainY, validX, validY, testX, testY]

def main():
    # Load the configuration parameters for the experiment
    with open('config_regression.json', 'r') as jsonfile:
        config = json.load(jsonfile)  
    outdir = config["output_dir"]

    impurity = 0.00004
    df = pd.read_csv(config['labels_file'])
    label_col = 'flare'
    feature_cols = ['LAT_FWT','LON_FWT','AREA_ACR','USFLUXL','MEANGAM','MEANGBT','MEANGBZ','MEANGBH','MEANJZD','TOTUSJZ','MEANALP','MEANJZH','ABSNJZH','SAVNCPP','MEANPOT','TOTPOT','MEANSHR','SHRGT45','R_VALUE','NACR','SIZE_ACR','SIZE']
    lams = ['193','171','304','1600','131','94']
    # feature_cols = []
    # for lam in lams:
    #     feature_cols.append(lam+'_magnitude')
    #     df.rename(columns = {lam+'_prominence':lam+'_rel_magnitude'},inplace=True)
    #     feature_cols.append(lam+'_rel_magnitude')
    #     feature_cols.append(lam+'_est_size')
    #     df[lam+'_duration'] = (pd.to_datetime(df[lam+'_end_time'])-pd.to_datetime(df[lam+'_start_time'])).dt.total_seconds()
    #     feature_cols.append(lam+'_duration')
    # feature_cols.append('Nx')
    # feature_cols.append('Ny')

    X_train,y_train,X_valid,y_valid,X_test,y_test = generateTrainValidData(df,config,feature_cols,label_col)

    # Use train and validation data for k-fold cross validation
    X = np.append(X_train,X_valid,axis=0)
    y = np.append(y_train,y_valid,axis=0)

    param_grid = {'min_impurity_decrease':(0.000025,0.00005,0.0001,0.0002,0.0004,0.0008,0.001,0.002)}
    # param_grid = {'n_estimators':(25,50,100,200)}
    # param_grid = {'max_depth':(3,5),'gamma':(0,1,5,10),'min_child_weight':(1,5,10)}
    # param_grid = {'n_neighbors':(1,3,5,8),'p':(1,2)}

    # ralphie = svm.SVR(kernel='rbf',degree=4)
    # model = KNeighborsRegressor()
    model = ExtraTreesRegressor(100,bootstrap=True,criterion='mae',random_state=1,min_impurity_decrease=impurity)
    # model = XGBRegressor(objective = 'reg:squarederror',eval_metric='rmse',seed=1)
    # clf = GridSearchCV(model,param_grid,cv=5,verbose=2,scoring='neg_mean_squared_error')
    # clf.fit(X_train[:,2:],y_train)
    model.fit(X_train[:,2:],y_train)
    # print(clf.cv_results_)

    plt.rc('font',size=16)
    plt.rc('axes',titlesize=16)
    plt.rc('axes',labelsize=16)
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    plt.rc('legend',fontsize=16)
    plt.rc('figure',titlesize=20)
    
    # plt.figure()
    # plt.plot((np.array(param_grid['n_estimators'])),clf.cv_results_['split0_test_score'])
    # plt.plot((np.array(param_grid['n_estimators'])),clf.cv_results_['split1_test_score'])
    # plt.plot((np.array(param_grid['n_estimators'])),clf.cv_results_['split2_test_score'])
    # plt.plot((np.array(param_grid['n_estimators'])),clf.cv_results_['split3_test_score'])
    # plt.plot((np.array(param_grid['n_estimators'])),clf.cv_results_['split4_test_score'])
    # plt.plot((np.array(param_grid['n_estimators'])),clf.cv_results_['mean_test_score'],'k')
    # plt.legend(['Split 0','Split 1','Split 2','Split 3','Split 4','Mean'],fontsize=16,loc='upper right')
    # plt.xlabel('Number of trees',fontsize=16)
    # plt.ylabel('Training score (Negative MSE)',fontsize=16)
    # plt.tight_layout()
    # plt.grid(True)
    # plt.plot(np.log10(np.array(param_grid['min_impurity_decrease'])),clf.cv_results_['split0_test_score'])
    # plt.plot(np.log10(np.array(param_grid['min_impurity_decrease'])),clf.cv_results_['split1_test_score'])
    # plt.plot(np.log10(np.array(param_grid['min_impurity_decrease'])),clf.cv_results_['split2_test_score'])
    # plt.plot(np.log10(np.array(param_grid['min_impurity_decrease'])),clf.cv_results_['split3_test_score'])
    # plt.plot(np.log10(np.array(param_grid['min_impurity_decrease'])),clf.cv_results_['split4_test_score'])
    # plt.plot(np.log10(np.array(param_grid['min_impurity_decrease'])),clf.cv_results_['mean_test_score'],'k')
    # plt.legend(['Split 0','Split 1','Split 2','Split 3','Split 4','Mean'],fontsize=16,loc='upper right')
    # plt.xlabel('Log Impurity',fontsize=16)
    # plt.ylabel('Training score (Negative MSE)',fontsize=16)
    # plt.tight_layout()
    # plt.grid(True)
    # plt.savefig(outdir+'ert_negMSEvsntrees_cvresults_hmiandaia.png')

    # model = clf.best_estimator_

    print('Results on training set')
    y_train_pred, trainerrs = predict(model,X_train[:,2:],y_train)

    print('Results on validation set')
    y_valid_pred, validerrs = predict(model,X_valid[:,2:],y_valid)

    ind_outliers = np.logical_and(y_valid >0.3,abs(y_valid_pred-y_valid)>0.25)
    print('Outlier validation results:')
    print(X_valid[ind_outliers,:2])

    ind_outliers = np.logical_and(y_train >0.3,abs(y_train_pred-y_train)>0.25)
    print('Outlier training results:')
    print(X_train[ind_outliers,:2])

    print('Results on test set')
    y_test_pred = model.predict(X_test[:,2:])

    plt.figure() 
    plt.plot(y_train*3-6,y_train_pred*3-6,'.')
    plt.plot(y_valid*3-6,y_valid_pred*3-6,'.')
    plt.plot(y_train*3-6,y_train*3-6,'k')
    plt.xticks(ticks=np.log10([1e-6,3e-6,5e-6,1e-5,3e-5,5e-5,1e-4,3e-4,5e-4,1e-3]),labels=['C1','C3','C5','M1','M3','M5','X1','X3','X5','X10'])
    plt.yticks(ticks=np.log10([1e-6,3e-6,5e-6,1e-5,3e-5,5e-5,1e-4,3e-4,5e-4,1e-3]),labels=['C1','C3','C5','M1','M3','M5','X1','X3','X5','X10'])
    plt.xlabel('True (log intensity)',fontsize=16)
    plt.ylabel('Pred (log intensity)',fontsize=16)
    plt.legend(['Training','Test'],fontsize=16)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(outdir+'ert_predvstrue_trainvalid_hmi_7.png',dpi=300,bbox_inches='tight')

    # plot test data
    plt.figure()
    plt.plot(X_test[:,-3],y_test_pred*3-6,'.')
    plt.yticks(ticks=np.log10([1e-6,5e-6,1e-5,3e-5,5e-5,1e-4,3e-4,5e-4,1e-3]),labels=['C1','C5','M1','M3','M5','X1','X3','X5','X10'])
    plt.xlabel('AIA 131 prominence')
    plt.ylabel('Predicted intensity')
    plt.grid(True)
    plt.savefig(outdir+'ert_predvstrue_test.png')

    print('Feature importances ', model.feature_importances_)

    return model,feature_cols,trainerrs,validerrs


if __name__=="__main__":
	main()