#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import import_ipynb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import InterpolatedUnivariateSpline
import shap

from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

RF = 1
GB = 2
SV = 3

MSE = 0
MAE = 1



'''
Divide the data into training and testing sets
It is key that the testing set contains the latter years, as we are goint to predict
into the future
input: dataframe
ouput: train and test dataframes
'''
def sampling_data(data):
    train = data.reset_index(drop=True)

    test = data[data['Year'] == 2014]
    test = test.reset_index(drop=True)
    
    return train, test



"""
Input:the results of sampling_data, and the label of the target
Output: returns the test and train sets with the label columns removed, and the label
columns themselves.
"""

def split_data(train, test, target):
    # training split
    x_train = train.drop([target], axis = 1)
    y_train = train[target] 
    
    # testing split
    x_test = test.drop([target], axis = 1)
    y_test = test[target] 
    
    return x_train, y_train, x_test, y_test




# create random forest
"""
Input: XY train/test, and an estimator parameter for random forest

Output: array feature importance , trained regressor, and the prditions on the test set
"""
def create_random_forest(x_train, y_train, x_test, est):
    rf = RandomForestRegressor(n_estimators=est, oob_score = True)
    rf.fit(x_train, y_train)
    preds = rf.predict(x_test)
    rf_feature_importance = rf.feature_importances_
    
    return rf_feature_importance, rf, preds





# creat gb
"""
Input: XY train/test, and an estimator parameter for random forest

Output: array feature importance , trained regressor, and the predictions on the test set
"""
def create_gradient_boost(x_train, y_train, x_test, est):
    gbc = GradientBoostingClassifier(n_estimators = est)
    gbc.fit(x_train, y_train)
    preds = gbc.predict(x_test)
    gbc_feature_importance = gbc.feature_importances_
    
    return gbc_feature_importance, gcb, preds
    
    
# estimator for svm is the C value 
    
def create_svm_regreession(x_train, y_train, x_test, est):
    svm = SVR(c = est, kernel='rbf', gamma = 'auto')
    svm.fit(x_train,y_train)
    preds = svm.predict(x_test)
    
    return svm, preds    
    
    
'''
Input: x_train, y_train, x_test, the appropriate datasets which are retrieved from split_data
        est, the estimator for the model
        sel, a selector which will decide which model will be run
output: an instance of the chosen models modeling.create function
'''
def run_model(x_train, y_train, x_test, est, sel):
    if sel == RF:
        return create_random_forest(x_train, y_train, x_test, est)
    elif sel == GB:
        return create_gradient_boost(x_train, y_train, x_test, est)
    elif sel == SV:
        return create_svm_regreession(x_train, y_train, x_test)
        
    
    
    

def xValSVR(dataset, target, k, cs, error_metric):
    
    kfold = KFold(n_splits = k)
    count = 0
    
    err_dict = {}
    for c in cs:
        err_dict[c] = []
    
    X = dataset.drop(target, axis=1)
    Y = dataset[target]
    
    for train_index, val_index in kfold.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        Y_train, Y_val = Y.iloc[train_index], Y.iloc[val_index]
        
        count += 1
        for c in cs:
            model = SVR(C=c, kernel = 'rbf', gamma='auto')
            model.fit(X_train, Y_train)
            preds = model.predict(X_val)
            if error_metric == MSE:
                err_c_k = mean_squared_error(Y_val, preds)
            if error_metric == MAE:
                err_c_k = mean_absolute_error
            err_dict[c].append(err_c_k)
            print('c = ', c, 'predicted on the ', count, 'th fold!!')
            

    return err_dict    
    
    
    
    
    
    
    
    
    
    
"""
Input: takes in an array of feature importance (retrieved from 
modeling.run_model/create function)and the dataset x_train for 
the purposes of index labeling

Output: the function has no output, but graphs the feature importance 

"""
def graph_feature_importance(feature_mi, x_train):
    feat_importances = pd.Series(feature_mi, index=x_train.columns).sort_values()
    plt.figure(figsize=(10, 10))
    plt.title("Feature Importance", fontsize = 14)
    plt.xlabel('feature', fontsize=12)
    plt.ylabel('importance', fontsize=12)
    feat_importances[::-1][:20].plot(kind="bar")
    plt.show()
    
    
def spline_extrapolate_missing_years(merged_df, target):
    TRAINING_YEARS = [2003, 2005, 2007, 2009, 2011, 2012, 2013]
    TEST_YEAR = 2014
    
    set_ids = set(merged_df['UNITID'])
    y_pred = pd.DataFrame(set_ids, columns=['UNITID'])
    y_pred[target] = np.nan
    
    merged_df = merged_df[merged_df['Year'].isin(TRAINING_YEARS)]
    
    for unit_id in set_ids:
        entry = merged_df.loc[merged_df['UNITID'] == unit_id].sort_values(by=['Year'])
        x = entry['Year']
        y = entry['MD_EARN_WNE_P6']
        #print(entry[['MD_EARN_WNE_P6', 'Year']])
        
        spl = InterpolatedUnivariateSpline(x, y)

        spl_val = spl(TEST_YEAR)
        y_pred.loc[y_pred["UNITID"] == unit_id, "MD_EARN_WNE_P6"] = spl_val
    return y_pred

'''
Input: takes fitted tree model and dataset that it was trained on

Output: Shap summary of the features that most affect the SHAP value, that is, 
the impact on model output.
'''
def shap_summary_plot_for_Trees(fitted_tree, x_train):
    explainer = shap.TreeExplainer(fitted_tree)
    shap_values = explainer.shap_values(x_train, approximate=True)
    shap.summary_plot(shap_values, x_train)
     
    
