# Customer Loan Decision Engine
## Summer Internship 2023
## Quynh K Nguyen, June-July 2023

### A Random Forest Classifier model that predicts lending churn rate of Asia Commercial Bank's business customer
### Files included:
- featureEngineer_customer_loan.py: deal with dummy variables, missing values, outliers + WOE-IV and correlation filter method included
- featureSelection_customer_loan.py: recursive feature selection with cross validation (auc, roc as metrics) and boruta feature selection
- modelTest_customer_loan.py: find the optimal model among Logistic Regression, Random Forest Classifier, Decision Tree Classifier, K Neighbors Classifier, and GradientBoostingClassifier
- modelFinal_customer_loan.py: implemented GradientBoostingClassifier with cross validation
- modelEval_customer_loan.py: use confusion matrix, Kolmogorov-Smirnov score, and Gini coefficient for evaluation
