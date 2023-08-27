# general setup
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score, roc_auc_score

# load dataset
df = pd.read_csv('df_to_train2.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.sample(5)



# ================== TRAIN TEST SPLIT ==================

data_train, data_test = train_test_split(df, random_state=149, train_size=0.8)

data_X_train = data_train.drop(columns=['TARGET_LOAN_SQL'])
data_X_test  = data_test.drop(columns=['TARGET_LOAN_SQL'])
data_y_train = data_train['TARGET_LOAN_SQL']
data_y_test  = data_test['TARGET_LOAN_SQL']



# ================== Logistic Regression ==================
model = LogisticRegression(random_state=149)
model.fit(data_X_train, data_y_train)
    
y_pred_train = model.predict(data_X_train)
y_pred_test = model.predict(data_X_test)
    
auc_train = roc_auc_score(data_y_train, y_pred_train)
auc_test = roc_auc_score(data_y_test, y_pred_test)
    
print(f'AUC train for {model.__class__.__name__}: {auc_train}')
print(f'AUC test for {model.__class__.__name__}: {auc_test}')



# ================== Random Forest Classifier ==================
model = RandomForestClassifier(max_depth=10,
                               class_weight={0:0.1, 1:0.9},
                              random_state=149)

model.fit(data_X_train, data_y_train)
y_pred_train = model.predict(data_X_train)
y_pred_test = model.predict(data_X_test)
    
auc_train = roc_auc_score(data_y_train, y_pred_train)
auc_test = roc_auc_score(data_y_test, y_pred_test)
    
print(f'AUC train for {model.__class__.__name__}: {auc_train}')
print(f'AUC test for {model.__class__.__name__}: {auc_test}')


model = RandomForestClassifier(max_depth=10,
                               min_samples_split=400,
                               class_weight={0:0.1, 1:0.9},
                              random_state=149)

model.fit(data_X_train, data_y_train)
y_pred_train = model.predict(data_X_train)
y_pred_test = model.predict(data_X_test)
    
auc_train = roc_auc_score(data_y_train, y_pred_train)
auc_test = roc_auc_score(data_y_test, y_pred_test)
    
print(f'AUC train for {model.__class__.__name__}: {auc_train}')
print(f'AUC test for {model.__class__.__name__}: {auc_test}')


model = RandomForestClassifier(max_depth=25,
                               criterion='gini',
                               min_samples_split=200,
                               class_weight={0:0.1, 1:0.9},
                              random_state=8)

model.fit(data_X_train, data_y_train)

y_hat = model.predict(data_X_test)

auc = roc_auc_score(data_y_test,y_hat)
print(auc)



# ================== Decision Tree Classifier ==================
dtc = DecisionTreeClassifier(criterion='gini',     # gini faster than entropy
                             splitter='random',    # random>best to avoid overfitting
                             min_samples_split=200, #node has to have at least 250 elements to continue splitting (avoid overfitting)
                             random_state=149,
                             class_weight='balanced')

dtc.fit(data_X_train, data_y_train)
    
y_pred_train = dtc.predict(data_X_train)
y_pred_test = dtc.predict(data_X_test)
    
auc_train = roc_auc_score(data_y_train, y_pred_train)
auc_test = roc_auc_score(data_y_test, y_pred_test)
    
print(f'AUC train for {model.__class__.__name__}: {auc_train}')
print(f'AUC test for {model.__class__.__name__}: {auc_test}')



# ================== KNeighborsClassifier ==================
knn = KNeighborsClassifier()

#gbc = GradientBoostingClassifier()
knn.fit(data_X_train, data_y_train)
    
y_pred_train = knn.predict(data_X_train)
y_pred_test = knn.predict(data_X_test)
    
auc_train = roc_auc_score(data_y_train, y_pred_train)
auc_test = roc_auc_score(data_y_test, y_pred_test)
    
print(f'AUC train for {model.__class__.__name__}: {auc_train}')
print(f'AUC test for {model.__class__.__name__}: {auc_test}')



# ================== Gradient Boosting Classifier ==================
gbc = GradientBoostingClassifier(learning_rate=0.2, 
                                 max_depth=10, 
                                 random_state=149)

gbc.fit(data_X_train, data_y_train)
y_pred_train = gbc.predict(data_X_train)
y_pred_test = gbc.predict(data_X_test)
    
auc_train = roc_auc_score(data_y_train, y_pred_train)
auc_test = roc_auc_score(data_y_test, y_pred_test)
    
print(f'AUC train for {model.__class__.__name__}: {auc_train}')
print(f'AUC test for {model.__class__.__name__}: {auc_test}')

gbc = GradientBoostingClassifier(learning_rate=0.1, 
                                 max_depth=5, 
                                 random_state=149)

gbc.fit(data_X_train, data_y_train)
    
y_pred_train = gbc.predict(data_X_train)
y_pred_test = gbc.predict(data_X_test)
    
auc_train = roc_auc_score(data_y_train, y_pred_train)
auc_test = roc_auc_score(data_y_test, y_pred_test)
    
print(f'AUC train for {model.__class__.__name__}: {auc_train}')
print(f'AUC test for {model.__class__.__name__}: {auc_test}')



    ### tune hyperparameter

# default learning rate 0.1
# default n_estimators = 100
'''   learning_rate + n_estimators   '''

grid = {'learning_rate':[0.01,0.05,0.1,0.2],
       'n_estimators':range(30,81,10)}

model = GradientBoostingClassifier(min_samples_split=500,
                                  min_samples_leaf=60,
                                  max_depth=10,
                                  max_features='sqrt',
                                  subsample=0.8, 
                                  random_state=149)

grid_search = GridSearchCV(estimator=model,
                          param_grid=grid,
                          cv=3,
                           scoring='roc_auc',
                           return_train_score=True)

grid_search.fit(data_X_train, data_y_train)

grid_search.best_score_, grid_search.best_params_, grid_search.best_estimator_


''' max_depth + min_samples_split '''

grid = {'min_samples_split':range(300,1100,100),
       'max_depth':range(4,11,1)}

model = GradientBoostingClassifier(learning_rate = 0.1, n_estimators= 80,
                                  min_samples_leaf=60,
                                  max_features='sqrt',
                                  subsample=0.8, 
                                  random_state=149)

grid_search = GridSearchCV(estimator=model,
                          param_grid=grid,
                          cv=3,
                           scoring='roc_auc',
                           return_train_score=True)

grid_search.fit(data_X_train, data_y_train)

grid_search.best_score_, grid_search.best_params_, grid_search.best_estimator_



''' min_samples_leaf + subsample '''

grid = {'subsample':[0.4,0.5,0.6,0.7,0.8]}

model = GradientBoostingClassifier(learning_rate = 0.1, n_estimators= 80,
                                  max_depth=10,
                                  min_samples_split=400,
                                  random_state=149)

grid_search = GridSearchCV(estimator=model,
                          param_grid=grid,
                          cv=3,
                           scoring='roc_auc',
                           return_train_score=True)

grid_search.fit(data_X_train, data_y_train)

yhat = grid_search.predict(data_X_test)
roc_auc_score(data_y_test, yhat)

grid_search.best_params_


#final model testing
model = GradientBoostingClassifier(learning_rate = 0.1, n_estimators= 80,
                                  min_samples_leaf=60,
                                  max_features='sqrt',
                                  subsample=0.8, 
                                  random_state=8,
                                  max_depth= 10, min_samples_split=400)


model.fit(data_X_train, data_y_train)

y_hat = model.predict_proba(data_X_test)[:,1]

auc = roc_auc_score(data_y_test,y_hat)
print(auc)

