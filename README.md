# Supervised Learning. Classification.

Summary: this project is an introduction to classification problems and related ML algorithms.

## Goal

The goal of this task is to get a deep understanding of the base models for classification (mainly Logistic Regression, NB and KNN). 


## Task

1. Download data from Don’tGetKicked competition. <br/><br/>
2. Design train/validation/test split.
Use “PurchDate” field for splitting, test must be later in time than validation, the same goes for validation and train: train.PurchDate < valid.PurchDate < test.PurchDate.
Use the first 33% of dates for train, last 33% of dates for test, and middle 33% for validation set.
*Don’t use the test dataset until the end!* <br/><br/>
3. Use LabelEncoder or OneHotEncoder from sklearn to preprocess categorical variables. Be careful with data leakage (fit Encoder on train and apply on validation & test). Consider another encoding approach if you meet new categorical values in valid & test (unseen in the training dataset), for example: https://contrib.scikit-learn.org/category_encoders/count.html <br/><br/>
4. Train LogisticRegression, GaussianNB, KNN from sklearn on the training dataset and check the quality of your algorithms on the validation dataset. The dependent variable (IsBadBuy) is binary. Don’t forget to normalize your datasets before training models.
<br/><br/>You must receive at least **0.15 Gini score** (the best of all four). Which algorithm performs better? Why?
<br/><br/>
5. Implement Gini score calculation. You can use 2\*ROC AUC - 1 approach, so you need to implement ROC AUC calculation. Check if your metric approximately equals `abs(2\*sklearn.metrcs.roc_auc_score - 1)`.
<br/><br/>
6. Implement your own versions of LogisticRegression, KNN and NaiveBayes classifiers. For LogisticRegression compute gradients with respect to the loss and use stochastic gradient descent.
Are you able to reproduce results from step 4?
<br/><br/>Guide for this task:
Your model must be represented by class with *fit*, *predict* (*predict_proba* with 0.5 *threshold*), *predict_proba* methods.
For LR moder compute gradient of loss with respect to parameters **w** and parameter **b** in fit function. Use a simple SGD approach for estimating optimal values of parameters.<br/><br/>
7. Try to create non-linear features, for example:
<br/><br/>
fractions: feature1/feature2
groupby features: `df[‘categorical_feature’].map(df.groupby(‘categorical_feature’)[‘continious_feature’].mean())`
<br/><br/>
Add new features into your pipeline, repeat step 4. Did you manage to increase your Gini score (you should!)?
<br/><br/>
8. Detect the best features for the problem using coefficients of the Logistic model. Try to eliminate useless features by hand and by using L1 regularization. Which approach is better in terms of Gini score?
<br/><br/>
9. *Try to apply non-linear variants of SVM, use the RAPIDS library if you have access to GPU. In other cases, use sklearn SVC with a non-linear kernel. If the training process needs too much time or memory try to subsample training data. Are you able to receive a better Gini score (on valid dataset) with this approach?
<br/><br/>
10. Select your best model (algorithm + feature set) and tweak its hyperparameters to increase Gini score on the validation dataset. Which hyperparameters are the most impactful?
<br/><br/>
11. Check Gini scores on all three datasets for your best model: train Gini, valid Gini, test Gini. Can you see any drop in performance when comparing valid quality vs test quality? Is your model overfitted or not? Explain.
<br/><br/>
12. Implement calculation or Recall, Precision, F1 score and AUC PR metrics.
<br/>Compare your algorithms on the test dataset using AUC PR metric.
<br/><br/>
13. Which hard label metric do you prefer for the task of detecting “lemon” cars?

## Bonus part

* Bonus exercises marked with * 
* Derivation: Using MLE in this context means minimizing NLL.
