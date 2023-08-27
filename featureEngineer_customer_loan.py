# general setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# load dataset
df=pd.read_csv('lending_potential_customer.csv')
df.tail(10)

df.shape

# exploratory data analysis: binary
text_df = df.select_dtypes(include='object')
text_df.head(5)

check_unique_df = pd.DataFrame(df.nunique())
check_unique_df.rename(columns={0: 'num_unique'}, inplace=True)
check_unique_df.sample(5)

binary_feature_df = check_unique_df[check_unique_df['num_unique'] ==2]
binary_feature_df

# exploratory data analysis: categorize


''' income '''
df.groupby('INCOME').count()
df['INCOME_D10'] = np.where(df['INCOME'] == 'D10', 1, 0)
df['INCOME_T10D30'] = np.where(df['INCOME'] == 'T10D30', 1, 0)
df['INCOME_T30D45'] = np.where(df['INCOME'] == 'T30D45', 1, 0)
df['INCOME_T45'] = np.where(df['INCOME'] == 'T45', 1, 0)

df.drop(columns=['INCOME'], inplace=True)
df.sample(6)


''' age '''
df.groupby('AGE').count()
new_age_list = []
for i in df['AGE']:
    if i <=18:
        new_age_list.append(18)
    elif i >= 70:
        new_age_list.append(70)
    else:
        new_age_list.append(i)
        
df['AGE'] = new_age_list

age_median = df['AGE'].median()

df['AGE'].fillna(age_median, inplace=True)


''' education '''
df.groupby('EDUCATION').count()

df['EDUCATION_PHOTHONG'] = np.where(df['EDUCATION'] == 'PHOTHONG',1,0)
df['EDUCATION_DH_CD'] = np.where(df['EDUCATION'] == 'DH_CD',1,0)
df['EDUCATION_TDH'] = np.where(df['EDUCATION'] == 'TDH',1,0)

df.drop(columns=['EDUCATION'], inplace=True)


''' gender '''
df.groupby('GENDER').count()
gender_df = pd.get_dummies(df['GENDER'])
df['GENDER'] = gender_df['MALE']
df.groupby('GENDER').count()
binary_feature_df.loc['GENDER']=[2]


''' region_acb '''
df.groupby('REGION_ACB').count()
df['REGION_TPHCM'] = np.where(df['REGION_ACB'] == 'TPHCM', 1,0)
df['REGION_HANOI'] = np.where(df['REGION_ACB'] == 'HANOI', 1,0)
df['REGION_DONGNAMBO'] = np.where(df['REGION_ACB'] == 'DONGNAMBO', 1,0)
#df.drop(columns=['REGION_ACB'], inplace=True)


''' customer type '''
df.groupby('CUST_TYPE').count()
df['CUST_AFF'] = np.where(df['CUST_TYPE'] == 'AFF', 1,0)
df['CUST_ENT'] = np.where(df['CUST_TYPE'] == 'ENT', 1,0)
df['CUST_RET'] = np.where(df['CUST_TYPE'] == 'RET', 1,0)
df['CUST_RURAL'] = np.where(df['CUST_TYPE'] == 'RURAL', 1,0)
df['CUST_UM'] = np.where(df['CUST_TYPE'] == 'UM', 1,0)
df['CUST_UP'] = np.where(df['CUST_TYPE'] == 'UP', 1,0)
df.drop(columns=['CUST_TYPE'], inplace=True)



# Missing values
null_df = pd.DataFrame(df.isnull().sum(), columns = ['num_null'])
length = len(df)
null_df.sample(5)

null_drop_list = null_df[null_df['num_null']/length > 0.85].index.to_list()
null_drop_list

df.drop(null_drop_list, axis=1, inplace=True)

null_df2 = pd.DataFrame(df.isnull().sum())
null_df2[null_df2[0]!=0]

df['AVG_GROWTH_AMT_DEPOSIT'].fillna(0.0, inplace=True)
df['CASA_STATUS'].fillna(0, inplace=True)
df['CASA_STATUS'].replace('ACT',1,inplace=True)

''' SP_CHINH_SUDUNG '''
df.drop(columns=['SP_CHINH_SUDUNG'], inplace=True)
df.drop(columns=['SP_CHINH_EVER'], inplace=True)
df.isnull().sum()



# Outliers
avg_tot_list = []

for col in df.columns.to_list():
    if col[:3] in ['AVG', 'TOT']:
        avg_tot_list.append(col)
        
avg_tot_list[:5]

num_df = df.select_dtypes(exclude='object')
num_df.head(10)

set(num_df.columns.to_list()) - set(avg_tot_list) - set(binary_feature_df.index.to_list())

df['SOLUONG_LOAIVAY'].unique()

avg_tot_list.append('PERS_LOAN_OUTSTANDING_BAL')
avg_tot_list.append('SEC_LOAN_OUTSTANDING_BAL')
avg_tot_list.append('CC_LIMIT')

outlier_df = df[avg_tot_list]
outlier_df.head(5)

# loop through each feature in outlier dataframe
for i in outlier_df.columns:
    # calculate upper and lower bound
    lower = np.percentile(df[i],1)
    upper = np.percentile(df[df[i]>0][i],99)
    adjusted_list = []
    # rescale lower 1 percentile and 99 percentile values
    for j in (df[i]):
        if j <= lower:
            adjusted_list.append(lower)
        elif j >= upper:
            adjusted_list.append(upper)
        else:
            adjusted_list.append(j)
    df[i] = adjusted_list

original_outlier = outlier_df.describe()
original_outlier




# WOE-IV for feature selection
def woe_iv(dfCont, dfAll, target):
    
    # empty dataframe
    all_bindf = []
    
    # loop through different attributes
    for featr in dfCont.columns.to_list():
        
        #data binning
        X = dfAll[[featr, target]]
        bins, thres = pd.qcut(X[featr], q = 4, retbins = True, duplicates = 'drop')
        
        X['bins'] = bins
        bin_df = pd.pivot_table(X,
                         index = ['bins'],
                         columns = [target],
                         aggfunc = {target:np.size})
        
        # calculate positive and negative
        bin_df.columns = ['#neg', '#pos']
        bin_df['ratio'] = bin_df['#pos']/bin_df['#neg']
        bin_df['%pos'] = bin_df['#pos']/(bin_df['#pos'].sum())
        bin_df['%neg'] = bin_df['#neg']/(bin_df['#neg'].sum())
        
        # calculate WOE and IV
        bin_df['WOE'] = np.log(bin_df['%pos']/bin_df['%neg'])
        bin_df['IV'] = ((bin_df['%pos'] - bin_df['%neg'])*bin_df['WOE']).sum()
        
        # assign attribute
        bin_df['feature'] = featr
        
        # concat all woe, iv results
        all_bindf.append(bin_df)
        
    final_df = pd.concat(all_bindf)
        
    return final_df

iv_proc_df = woe_iv(outlier_df, df, 'TARGET_LOAN_SQL')
iv_proc_df

iv_proc_df = iv_proc_df[['IV', 'feature']].drop_duplicates().reset_index().drop(columns=['bins'])
iv_df = iv_proc_df.sort_values(by=['IV'], ascending=False)

iv_proc_df = iv_proc_df[['IV', 'feature']].drop_duplicates().reset_index().drop(columns=['bins'])
iv_df = iv_proc_df.sort_values(by=['IV'], ascending=False)

iv_df2 = iv_df.copy()
iv_df2.rename(columns={'feature':'feature2'}, inplace=True)



# Correlation (filter method)
df.corr().abs()

x_list = outlier_df.columns.to_list()
len(x_list)

def corr_iv(df, list, thres):
    #create correlation df with 1.o's diagonally
    corr_df = df[list].corr().abs()
    
    #correlation df: 2 columns of variables
    corr_lst = []
    for c in corr_df.columns:
        for r in corr_df.index:
            if c != r and corr_df[c][r] > thres:
                corr_lst.append({"feature":c, "feature2":r, "corr": corr_df[c][r]})
    corr_df2 = pd.DataFrame(corr_lst)               
    
    #merge IV into correlation df
    corr_iv = corr_df2.merge(iv_df, on='feature').merge(iv_df2, on='feature2')
    
    #add "drop" column
    corr_iv['drop'] = [0]*len(corr_iv)
    for i in range(len(corr_iv)):
        if corr_iv['IV_x'][i] < corr_iv['IV_y'][i]:
            corr_iv['drop'][i] = corr_iv['feature'][i]
        else:
            corr_iv['drop'][i] = corr_iv['feature2'][i]
            
    #list of variables to be dropped
    drop_list = corr_iv['drop'].to_list()
    drop_list=[*set(drop_list)]
    
    return drop_list

drop_list = corr_iv(df, x_list, 0.75)
len(drop_list)
new_df = df.drop(columns=drop_list)
new_df.head()

# wrap up
new_df.to_csv('df_feed_new.csv')


