# general setup
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn import metrics
import statsmodels.api as sm

# load dataset
df = pd.read_csv('df_feed_new.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.describe()
df['TARGET_LOAN_SQL'].value_counts().plot(kind='bar')

pos_df = df[df['TARGET_LOAN_SQL']==1].sample(n=30000)
neg_df = df[df['TARGET_LOAN_SQL']==0].sample(n=300000)

df_model = pd.concat([pos_df, neg_df], ignore_index = True)
df_model.head()



'''================== TRAIN TEST SPLIT =================='''
X = df_model.drop(columns=['TARGET_LOAN_SQL','REF_DT','CUSTOMER_ID'], axis=1)
y = df_model['TARGET_LOAN_SQL']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=149)



'''================== MODEL =================='''
model_rf = RandomForestClassifier(random_state=149)



'''================== Boruta  Feature Selection =================='''
# create shadow df
np.random.seed(149)

X_shadow = X.apply(np.random.permutation)
X_shadow.columns = ['shadow' + col for col in X.columns]

# merge original and shadow df to use Boruta feature selection
X_boruta = pd.concat([X, X_shadow], axis=1)

# fit model
boruta_fit = model_rf.fit(X_boruta, y)

# feature importance
featr_imp_X = boruta_fit.feature_importances_[:len(X.columns)] 
featr_imp_shadow = boruta_fit.feature_importances_[len(X.columns):]

# compute shadow threshold
shadow_thres = np.round(featr_imp_shadow.mean(),6)

# create a df of features > than shadow's threshold
boruta_imp = boruta_fit.feature_importances_
boruta_imp_df = pd.DataFrame({'feature':X_boruta.columns.to_list(), 'importance':boruta_imp})

boruta_imp_df

top_boruta_df = boruta_imp_df.sort_values(by=['importance'], ascending=False).head(60)
top_boruta_df

boruta_feature_list = top_boruta_df['feature'].to_list()
boruta_feature_list2 = []
shadow=0

for ft in boruta_feature_list:
    if ft[0:6] != 'shadow':
        boruta_feature_list2.append(ft)
    else:
        shadow+=1
        
len(boruta_feature_list2), shadow

auc_listB = []
no_feature_listB = []

for i in range(2,33):

    feature_list = boruta_feature_list2[0:i]
    #print(feature_list)
    X = df_model[feature_list]
    y = df_model['TARGET_LOAN_SQL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
    
    model2 = RandomForestClassifier(random_state=149, n_jobs=-1)
    model2.fit(X_train, y_train)
    
    y_pred2 = model2.predict(X_test)

    aucB = metrics.roc_auc_score(y_test, y_pred2)
    auc_listB.append(aucB)

    no_feature_listB.append(len(feature_list))
    
plt.plot(no_feature_listB, auc_listB)
plt.ylabel('AUC Score')
plt.xlabel('Number of Features')

try_boruta = boruta_feature_list2[0:25]



'''================== Filter method: P_value =================='''
# new X, y for OLS
X_ols = df_model.drop(columns=['TARGET_LOAN_SQL','REF_DT','CUSTOMER_ID'], axis=1)
y_ols = df_model['TARGET_LOAN_SQL']

# add constant
X_ols = sm.add_constant(X_ols)

# fit and get result
model_ols = sm.OLS(y_ols, X_ols)
result = model_ols.fit()

# create a df with pvalues of features
ols_pval=pd.DataFrame(result.pvalues)
ols_pval.rename(columns={0:'p_value'},inplace=True)

# drop all features with pvalues > 0.01
feature_drop_ols = ols_pval[ols_pval['p_value'] > 0.01].index.to_list()



''' ================== Wrapper: Recursive Feature Elimination =================='''
# Run RFECV (cross-validation) to identify the optimal number of features first
rfe_cv = RFECV(estimator=model,
               cv=3,
               step=1)
rfe_cv.fit(X_train, y_train)
print(f'Optimal number of output: {rfe_cv.n_features}')

rfe_method = RFE(model_rf, 
                 n_features_to_select = 1,
                step=1)
rfe_method.fit(X_train, y_train)

# Predicted y_value
y_pred = rfe_method.predict(X_test)

imp = rfe_method.ranking_.tolist()
feature = X.columns.to_list()
imp_df = pd.DataFrame({'feature':feature, 'importance':imp}).sort_values(by=['importance'])
imp_df.head(20)

# Plot ROC, AUC

auc_list = []
no_feature_list = []

for i in range(2,51):
    feature_list = imp_df['feature'].to_list()[0:i]
    
    X = df_model[feature_list]
    y = df_model['TARGET_LOAN_SQL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
    
    model1 = RandomForestClassifier(random_state=149, n_jobs=-1)
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    
    auc = metrics.roc_auc_score(y_test, y_pred)
    auc_list.append(auc)
    no_feature_list.append(len(feature_list))
    
plt.plot(no_feature_list, auc_list)
plt.ylabel('AUC Score')
plt.xlabel('Number of Features')

x = pd.DataFrame({'number of feature':no_feature_list, 'AUC':auc_list}).sort_values(by=['number of feature'])
x

# get list of features from rfe
test = x[20:]
plt.plot(test['number of feature'],test['AUC'])
rfe_feature_list = imp_df['feature'].to_list()[0:24]
rfe_feature_list

q = list((set(rfe_feature_list) & set(boruta_feature_list)).difference(set(feature_drop_ols)))
len(q)



''' ================== FINAL MERGE =================='''
q= (rfe_feature_list) + (try_boruta)
final_feature = list(set(q)-set(feature_drop_ols))
len(final_feature)

x=rfe_feature_list+try_boruta
xx = pd.DataFrame(x, columns=['feature']).sort_values('feature')

x=rfe_feature_list+try_boruta

#y.remove_duplicates()
x = [*set(x)]
xx = pd.DataFrame(x, columns=['feature'])
xx.shape

final_feature.append('TARGET_LOAN_SQL')

df_to_train = pd.read_csv('df_cleaned.csv')
df_to_train.drop(columns=['Unnamed: 0'], inplace=True)

df_to_train.sample(5)

df_to_train = df_to_train[final_feature]
df_to_train

df_to_train.to_csv('df_to_train2.csv')
