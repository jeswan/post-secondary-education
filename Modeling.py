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
get_ipython().run_line_magic('matplotlib', 'inline')

RF = 1
GB = 2


def sampling_data(data):
    data_exists = [2003, 2005, 2007, 2009, 2011, 2012]
    
    relevant_years_df = data[data['Year'].isin(data_exists)]
    train = relevant_years_df.reset_index(drop=True)

    test = data[data['Year'] == 2013]
    test = test.reset_index(drop=True)
    
    return train, test





def split_data(train, test, target):
    # training split
    x_train = train.drop([target], axis = 1)
    y_train = train[target] 
    
    # testing split
    x_test = test.drop([target], axis = 1)
    y_test = test[target] 
    
    return x_train, y_train, x_test, y_test



# create random forest

def create_random_forest(x_train, y_train, x_test, est):
    rf = RandomForestRegressor(n_estimators=est, oob_score = True)
    rf.fit(x_train, y_train)
    rf_feature_importance = rf.feature_importances_
    
    return rf_feature_importance

# creat random forest

def create_gradient_boost(x_train, y_train, x_test, est):
    gbc = GradientBoostingClassifier(n_estimators = est)
    gbc.fit(x_train, y_train)
    gbc_feature_importance = gbc.feature_importances_
    
    return gbc_feature_importance
    

def run_model(x_train, y_train, x_test, est, sel):
    if sel == RF:
        return create_random_forest(x_train, y_train, x_test, est)
    elif sel == GB:
        return create_gradient_boost(x_train, y_train, x_test, est)
        
    

# graphing feature importance

def graph_feature_importance(x_train, y_train, x_test, est, model):
    feature_mi = run_model(x_train, y_train, x_test, est, name)
    feature_mi_dict = dict(zip(x_train.columns.values, feature_mi))
    feat_importances = pd.Series(feature_mi, index=x_train.columns)
    plt.figure(figsize=(10, 10))
    plt.title("Feature Importance", fontsize = 14)
    plt.xlabel('importance', fontsize=12)
    plt.ylabel('feature', fontsize=12)
    feat_importances.sort_values().plot(kind="bar")
    plt.show()





# testing different esitmators

estimators = []

auc_oob = []
auc_test = []

for n in estimators:
    oob = create_random_forest(train_df, test_df, 'MD_EARN_WNE_P6', n, 2)
    proba = create_random_forest(train_df, test_df,'MD_EARN_WNE_P6' , n, 4)
    auc_oob.append(roc_auc_score(train_df['MD_EARN_WNE_P6'], oob[:,1]))
    auc_test.append(roc_auc_score(test_df['MD_EARN_WNE_P6'], proba[:,1]))

