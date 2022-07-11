import json
from re import L
import numpy as np
import pandas as pd
import ert_aia_peaks
import os

features = 'hmi' # can be either 'hmi','hmiandaia' or 'aia'
seeds = np.arange(1001,11001,1000)
data = {'seed':seeds,'RMSE_train':[],'MAE_train':[],'R2_train':[],'L2err_train':[],'RMSE_test':[],'MAE_test':[],'R2_test':[],'L2err_test':[]}
for seed in seeds:

    # first update config file
    with open('config_regression.json', 'r') as jsonfile:
        config = json.load(jsonfile)

    print('Running test with seed ', seed)
    config['features'] = features
    config["seed"] = int(seed)
    config["labels_file"] = 'aia_flares_catalog_verified.csv'
    config["output_dir"] = 'ert_results_'+features+'/seed'+str(seed)+'/'

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    with open('config_regression.json', 'w') as outfile:
        json.dump(config, outfile)

    # then run code
    model, features, trainerrs, testerrs = ert_aia_peaks.main()
    data['RMSE_train'].append(trainerrs[0])
    data['MAE_train'].append(trainerrs[1])
    data['R2_train'].append(trainerrs[2])
    data['L2err_train'].append(trainerrs[3])
    data['RMSE_test'].append(testerrs[0])
    data['MAE_test'].append(testerrs[1])
    data['R2_test'].append(testerrs[2])
    data['L2err_test'].append(testerrs[3])
    for feature,importance in zip(features,model.feature_importances_):
        if feature in data.keys():
            data[feature].append(importance)
        else:
            data[feature] = [importance]

df = pd.DataFrame(data)
df.to_csv(config["output_dir"]+'ert_results_'+features+'_verified.csv',index=False)

    

