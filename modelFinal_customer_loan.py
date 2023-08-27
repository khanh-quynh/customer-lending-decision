# general setup
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import accuracy_score, roc_auc_score

# load dataset
df=pd.read_csv('df_to_train2.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.head(5)


'''           TRAIN TEST SPLIT           '''

data_train, data_test = train_test_split(df,train_size=0.8,random_state=149)
data_X_train = data_train.drop(columns=['TARGET_LOAN_SQL'])
data_X_test  = data_test.drop(columns=['TARGET_LOAN_SQL'])
data_y_train = data_train['TARGET_LOAN_SQL']
data_y_test  = data_test['TARGET_LOAN_SQL']



'''         FIT AND PREDICT      '''
model = RandomForestClassifier(max_depth=25,
                               criterion='gini',
                               min_samples_split=200,
                               class_weight={0:0.1, 1:0.9},
                              random_state=8)

model.fit(data_X_train, data_y_train)
y_hat = model.predict_proba(data_X_test)[:,1]
auc = roc_auc_score(data_y_test,y_hat)
print(auc)

model2 = GradientBoostingClassifier(learning_rate = 0.1, n_estimators= 80,
                                  min_samples_leaf=60,
                                  max_features='sqrt',
                                  subsample=0.8, 
                                  random_state=8,
                                  max_depth= 10, min_samples_split=400)

model2.fit(data_X_train, data_y_train)
y_hat2 = model2.predict_proba(data_X_test)[:,1]
auc = roc_auc_score(data_y_test,y_hat2)
print(auc)



'''         CROSS VALIDATION        '''
data = data_train
X = data_X_train
y = data_y_train

# set parameters
random_state=8
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

param = {
    'learning_rate' : 0.1, 
    'n_estimators': 80,
    'min_samples_leaf':60,
    'max_features':'sqrt',
    'subsample':0.8, 
    'max_depth': 10,
    'min_samples_split':400,
    'random_state':8}
                            
# store results of folds
model_list=[]
feature_imp_list=[]


start_time = datetime.now()

for i, (train_index, test_index) in enumerate(kfold.split(data)):
    #loop through each fold
    model = GradientBoostingClassifier(**param)
    
    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
    y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
    
    print(f'======================== Fold {i+1}/{n_splits} ========================')
    
    # fit model and predict
    model.fit(X_train, y_train)
    y_hat = model.predict_proba(X_test)[:,1]
    
    # calculate AUC score
    auc_score = roc_auc_score(y_test,y_hat)
    print(f'AUC score: {auc_score}')
    
    # store folds
    feature_imp_list.append(model.feature_importances_)
    model_list.append(model)
    
train_time = datetime.now()-start_time
print(f'TOTAL TRAIN TIME: {train_time}')


def mean_y_hat(X, model_list):
    total_y_hat = np.zeros(X.shape[0])
    # loop through each model and predict
    for model in model_list:
        y_hat = model.predict_proba(X)[:,1]
        total_y_hat += y_hat
    # find avg of all y predicted from 5 models
    mean_y_hat = total_y_hat/len(model_list)
    return mean_y_hat

mean_y_hat_train = mean_y_hat(data_X_train, model_list)
print(f'Train AUC score: {roc_auc_score(data_y_train, mean_y_hat_train)}')
mean_y_hat_test = mean_y_hat(data_X_test, model_list)
print(f'Test AUC score: {roc_auc_score(data_y_test, mean_y_hat_test)}')

df_valid = data_test.copy()
df_valid['y_pred'] = mean_y_hat_test

df_valid

df_valid.to_csv('df_valid.csv')


