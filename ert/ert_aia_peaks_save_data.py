import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load
import random
from sklearn.model_selection import GridSearchCV
import json


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
    train_flares = np.concatenate([train_flares_C,train_flares_M,train_flares_X])
    valid_flares = np.concatenate([valid_flares_C,valid_flares_M,valid_flares_X])
    df_train = df[df['goes_flare_ind'].isin(train_flares)] 
    df_valid = df[df['goes_flare_ind'].isin(valid_flares)]
    df_test = df[df['goes_flare_ind'].isnull()].sample(frac=1)

    y_train = df_train[label_col]
    y_valid = df_valid[label_col]
    y_test = df_test[label_col]
    print('Features: ',cols)
    data = df_train[cols].to_numpy()
    scaler = StandardScaler()
    scaler.fit(data)

    data = scaler.transform(data)
    df_new = pd.DataFrame(data, columns=[x for x in cols])
    df_new.insert(loc=0, column='index', value=df_train.index)
    df_new.insert(loc=1, column='SHARP', value=df_train['SHARP'].values)
    df_new.insert(loc=2, column='goes_flare_ind', value=df_train['goes_flare_ind'].values)
    df_train = df_new

    data = df_valid[cols].to_numpy()
    data = scaler.transform(data)
    df_new = pd.DataFrame(data, columns=[x for x in cols])
    df_new.insert(loc=0, column='index', value=df_valid.index)
    df_new.insert(loc=1, column='SHARP', value=df_valid['SHARP'].values)
    df_new.insert(loc=2, column='goes_flare_ind', value=df_valid['goes_flare_ind'].values)
    df_valid = df_new

    data = df_test[cols].to_numpy()
    data = scaler.transform(data)
    df_new = pd.DataFrame(data, columns=[x for x in cols])
    df_new.insert(loc=0, column='index', value=df_test.index)
    df_new.insert(loc=1, column='SHARP', value=df_test['SHARP'].values)
    df_new.insert(loc=2, column='flare_time', value=df_test['aia_max_start_time'].values)
    # df_new.insert(loc=0, column="filename", value=df_test["filename"].values)
    df_test = df_new

    df_train = df_train.replace([np.inf, -np.inf, np.nan], 0)
    # df_train = df_train.dropna()
    df_valid = df_valid.replace([np.inf, -np.inf, np.nan], 0)
    # df_valid = df_valid.dropna()
    df_test = df_test.replace([np.inf, -np.inf , np.nan], 0)
    # df_test = df_test.dropna()

    print('Number of C/M/X flares in training set: ',len(train_flares_C),'/',len(train_flares_M),'/',len(train_flares_X))
    print('Number of C/M/X flares in valid set: ',len(valid_flares_C),'/',len(valid_flares_M),'/',len(valid_flares_X))
    print(len(df_test))

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
    for lam in lams:
        feature_cols.append(lam+'_magnitude')
        feature_cols.append(lam+'_prominence')
        feature_cols.append(lam+'_est_size')
        df[lam+'_duration'] = (pd.to_datetime(df[lam+'_end_time'])-pd.to_datetime(df[lam+'_start_time'])).dt.total_seconds()
        feature_cols.append(lam+'_duration')
    # feature_cols.append('Nx')
    # feature_cols.append('Ny')

    X_train,y_train,X_valid,y_valid,X_test,y_test = generateTrainValidData(df,config,feature_cols,label_col)

    # Use train and validation data for k-fold cross validation
    X = np.append(X_train,X_valid,axis=0)
    y = np.append(y_train,y_valid,axis=0)

    param_grid = {'min_impurity_decrease':(0.000025,0.00005,0.0001,0.0002,0.0004,0.0008,0.001,0.002)}
    # param_grid = {'n_neighbors':(1,3,5,8),'p':(1,2)}

    # ralphie = svm.SVR(kernel='rbf',degree=4)
    # model = KNeighborsRegressor()
    model = ExtraTreesRegressor(100,bootstrap=True,criterion='mae',random_state=1,min_impurity_decrease=impurity)
    model.fit(X[:,3:],y)
    # print(clf.cv_results_)

    print('Results on training set')
    y_train_pred, trainerrs = predict(model,X[:,3:],y)

    print('Results on test set')
    y_test_pred = model.predict(X_test[:,3:])
    print('Number of unlabelled >=M1 flares:', np.sum(10**(y_test_pred*3-6)>=1e-5))

    plt.figure()
    plt.plot(y*3-6,y_train_pred*3-6,'.')
    plt.plot(y_train*3-6,y_train*3-6,'k')
    plt.xticks(ticks=np.log10([1e-6,5e-6,1e-5,2e-5,3e-5,4e-5,5e-5,7e-5,1e-4,2e-4,3e-4,4e-4,5e-4,7e-4,1e-3]),labels=['C1','C5','M1','M2','M3','M4','M5','M7','X1','X2','X3','X4','X5','X7','X10'])
    plt.yticks(ticks=np.log10([1e-6,5e-6,1e-5,2e-5,3e-5,4e-5,5e-5,7e-5,1e-4,2e-4,3e-4,4e-4,5e-4,7e-4,1e-3]),labels=['C1','C5','M1','M2','M3','M4','M5','M7','X1','X2','X3','X4','X5','X7','X10'])
    plt.xlabel('True (log intensity)')
    plt.ylabel('Pred (log intensity)')
    plt.legend(['Training'],fontsize=12)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(outdir+'ert_predvstrue_train.png')

    # plot test data
    plt.figure()
    plt.plot(X_test[:,-3],y_test_pred*3-6,'.')
    plt.yticks(ticks=np.log10([1e-6,5e-6,1e-5,2e-5,3e-5,4e-5,5e-5,7e-5,1e-4,2e-4,3e-4,4e-4,5e-4,7e-4,1e-3]),labels=['C1','C5','M1','M2','M3','M4','M5','M7','X1','X2','X3','X4','X5','X7','X10'])
    plt.xlabel('AIA 131 prominence')
    plt.ylabel('Predicted intensity')
    plt.grid(True)
    plt.savefig(outdir+'ert_predvstrue_test.png')
    
    print('Feature importances ', model.feature_importances_)

    y_pred = np.nan*np.zeros(len(df))
    cmx = [np.nan]*len(df)

    # save prediction
    for i in range(len(y_test_pred)):
        if len(np.where(df.index==X_test[i,0])[0]) > 0:
            ind = np.where(df.index==X_test[i,0])[0][0]
            y_pred[ind] = 10**(y_test_pred[i]*3-6)
            if y_pred[ind]>=1e-4:
                cmx[ind] = 'X'
            elif y_pred[ind]>=1e-5:
                cmx[ind] = 'M'
            elif y_pred[ind]>=1e-6:
                cmx[ind] = 'C'

    for i in range(len(y_train_pred)):
        if len(np.where(df.index==X[i,0])[0]) > 0:
            ind = np.where(df.index==X[i,0])[0][0]
            y_pred[ind] = 10**(y_train_pred[i]*3-6) 
            if y_pred[ind]>=1e-4:
                cmx[ind] = 'X'
            elif y_pred[ind]>=1e-5:
                cmx[ind] = 'M'
            elif y_pred[ind]>=1e-6:
                cmx[ind] = 'C'

    df['ert_pred_intensity'] = y_pred
    df['ert_pred_CMX'] = cmx
    df.to_csv('../flare_catalogs/aia_flares_catalog_verified_pred_2.csv',index=False)

    return 


if __name__=="__main__":
	main()