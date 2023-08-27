# general setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, roc_auc_score, PrecisionRecallDisplay

# load dataset
df=pd.read_csv('df_valid.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.sample(5)

# function to create confusion matrix table
def confusion_matrix_table(target, predicted, df):
    precision = []
    recall = []
    pred_0 = []
    pred_1 = []
    tn_lst =[]
    fn_lst = []
    fp_lst = []
    tp_lst = []
    f1 = []
    for thres in np.round(np.arange(0.1,1.0,0.1),2):
        # iterate through different thresholds
        binary_y_pred = []
        for y in df[predicted]:
            if y >= thres:
                binary_y_pred.append(1)
            else:
                binary_y_pred.append(0)
        pred_1.append(sum(binary_y_pred))
        pred_0.append(len(df) - sum(binary_y_pred))

        # confusion matrix values
        cm = confusion_matrix(df[target], binary_y_pred)
        tn,fp,fn,tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        tn_lst.append(tn), fn_lst.append(fn), fp_lst.append(fp), tp_lst.append(tp)

        # calculate precision, recall
        precision.append(tp/(tp+fp))
        recall.append(tp/(tp+fn))
        f1.append(f1_score(df[target], binary_y_pred))

    # create confusion matrix df
    cm_df = pd.DataFrame({'thres':np.round(np.arange(0.1,1.0,0.1),2),
                          'pred_0':pred_0, 'pred_1':pred_1,
                          'true_neg':tn_lst,'false_neg':fn_lst,'true_pos':tp_lst,'false_pos':fp_lst,
                          'precision':precision,'recall':recall, 'f1':f1}) 
    cm_df 
    return cm_df

cm_df = confusion_matrix_table('TARGET_LOAN_SQL','y_pred', df)
cm_df

binary_y_pred = []
for y in df['y_pred']:
    if y >= 0.6:
        binary_y_pred.append(1)
    else:
        binary_y_pred.append(0)

df_test = df.copy()
df_test['y_pred'] = binary_y_pred
cm = confusion_matrix(df_test['TARGET_LOAN_SQL'], df_test['y_pred'])
tn, fp, fn, tp = cm.ravel()
ConfusionMatrixDisplay(cm).plot()



# PR Curve
precision_recall = PrecisionRecallDisplay(cm_df['precision'],cm_df['recall'])
precision_recall.plot()



# function to create bins for decile list
def create_decile(predicted, df):
    bins, thres = pd.qcut(df[predicted],q=10, retbins=True)
    thres_list = sorted(thres.tolist(), reverse=True)
    max_prob = thres_list[0:10]
    min_prob = thres_list[1:11]
    decile_list = []
    for i in range(len(thres_list)-1):
        decile_list.append([min_prob[i], max_prob[i]])
    return decile_list
decile_list = create_decile('y_pred', df)



# function to create KS dataframe 
def create_KS_df(df, target, decile_list):
    percent_event = []
    percent_nonevent = []
    total_event = df[target].sum()
    total_nonevent = len(df) - total_event
    for i in decile_list:
        # subset dataframe within 1 decile
        df_test = df[(df.y_pred >= i[0]) & (df.y_pred <= i[1])]
        # count 1s and 0s observations
        event = df_test[target].sum()
        nonevent = len(df_test)-event
        # calculate %event and %nonevent observation
        percent_event.append(np.round(event*100/total_event,3))
        percent_nonevent.append(np.round(nonevent*100/total_nonevent,3))
    cml_event = []
    cml_nonevent = []
    ks = []
    for i in range(10):
        cml_event.append(np.round(sum(percent_event[0:i+1]),2))
        cml_nonevent.append(np.round(sum(percent_nonevent[0:i+1]),2))
        ks.append(np.round(sum(percent_event[0:i+1]),2) - np.round(sum(percent_nonevent[0:i+1]),2))
    ks_df = pd.DataFrame({
    'min_prob':min_prob,
    'max_prob':max_prob,
    '%event':percent_event,
    '%nonevent':percent_nonevent,
    '%cml_event':cml_event,
    '%cml_nonevent':cml_nonevent,
    'KS score':ks})
    
    return ks_df
create_KS_df(df, 'TARGET_LOAN_SQL', decile_list)



# Gini Coefficient
auc=roc_auc_score(df['TARGET_LOAN_SQL'], df['y_pred'])
fpr, tpr, thres = roc_curve(df['TARGET_LOAN_SQL'], df['y_pred'])

fig, ax = plt.subplots()
ax.plot([0,1],[0,1], '--b')
ax.plot(fpr, tpr, label=f'ROC (AUC={np.round(auc,4)})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

leg = ax.legend(loc='lower right')

gini = 2*auc-1
(auc-0.5)/0.5
