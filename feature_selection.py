import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

F_REGRESSION = 0
MUTUAL_REGRESSION = 1

def selectFeatures(merged_df_no_id, target, n_features_to_select, sel):
    x = merged_df_no_id.drop([target], axis=1)
    y = merged_df_no_id[target]
    
    if sel == F_REGRESSION:
        feature_cols =  fRegression(x, y, n_features_to_select)
    elif sel == MUTUAL_REGRESSION:
        feature_cols = mutualInfoRegression(x, y, n_features_to_select)
        
    # Create new dataframe with only desired columns
    return merged_df_no_id.iloc[:,cols]
    
def fRegression(x, y, n_features_to_select):
    selector = SelectKBest(f_regression, n_features_to_select)
    
    return postSelectKBest(x, y, selector)

def mutualInfoRegression(x, y, n_features_to_select):
    selector = SelectKBest(mutual_info_regression, n_features_to_select)
    
    return postSelectKBest(x, y, selector)

def postSelectKBest(x, y, selector):
    selector.fit(x, y)
    # Get columns to keep
    cols = selector.get_support(indices=True)
   
    return cols