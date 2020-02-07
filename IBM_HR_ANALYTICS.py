#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#importing the dataset 
df=pd.read_csv("employee.csv")
df.head()
#Removing The Unwanted Column
df.pop('EmployeeNumber')
df.pop('Over18')
df.pop('StandardHours')
df.pop('EmployeeCount')

df.shape
df.info()
X=df.drop('Attrition',axis=1)
y=df['Attrition']
#Conversion Of Target Variable into Binary Variable
from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y=lb.fit_transform(y)
y.shape

t=df.select_dtypes(['object'])
#Conversion into dummy variables
BusinessTravel_ind=pd.get_dummies(df['BusinessTravel'],prefix='BusinessTravel')
Department_ind=pd.get_dummies(df['Department'],prefix='Department')
EducationField_ind=pd.get_dummies(df['EducationField'],prefix='EducationField')
Gender_ind=pd.get_dummies(df['Gender'],prefix='Gender')
JobRole_ind=pd.get_dummies(df['JobRole'],prefix='JobRole')
MaritialStatus_ind=pd.get_dummies(df['MaritalStatus'],prefix='MaritalStatus')
OverTime_ind=pd.get_dummies(df['OverTime'],prefix='OverTime')
OverTime_ind.head()
df["BusinessTravel"].unique()
df1=pd.concat([BusinessTravel_ind,Department_ind,EducationField_ind,Gender_ind,JobRole_ind,MaritialStatus_ind,OverTime_ind],sort=False)

df1=pd.concat([BusinessTravel_ind,Department_ind,EducationField_ind,Gender_ind,JobRole_ind,MaritialStatus_ind,OverTime_ind,df.select_dtypes(['int64'])],axis=1)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df1,y,test_size=0.25,random_state=0)
#Checking the base line accuracy using Dummy Classifier
from sklearn.dummy import DummyClassifier

dummy_classifier=DummyClassifier(strategy='most_frequent')

dummy_classifier.fit(X_train,y_train)
y_pred_dummy=dummy_classifier.predict(X_test)
#
print(accuracy_score(y_test,y_pred_dummy))

from sklearn.tree import DecisionTreeClassifier
classifier1=DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=0)


classifier1.fit(X_train,y_train)
y_pred_train=classifier1.predict(X_train)
print("accuracy of training data  is",accuracy_score(y_train,y_pred_train))
y_pred_test=classifier1.predict(X_test)
print("accuracy of test data  is",accuracy_score(y_test,y_pred_test))

parameters_dt=[{'criterion':['entropy'],'max_depth':[1,2,3,4,5,6]},
               {'criterion':['gini'],'max_depth':[1,2,3,4,5,6]}]

from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(estimator=classifier1,
                         param_grid=parameters_dt,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1)
grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
print(best_parameters)
#Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_test)
#Classification Report
print("classification report is given as")
print(classification_report(y_test,y_pred_test))

#Evaluation of Decision Tree Classifier
print("accuracy of Decision Tree Classifier is ",accuracy_score(y_test,y_pred_test))
print("F1 score is given as ",f1_score(y_test,y_pred_test))
print("Jaccard Scores is given as ",jaccard_score(y_test,y_pred_test))

print("Confusion Matrix is ")
print(confusion_matrix(y_test,y_pred_test))





from sklearn.ensemble import RandomForestClassifier
classifier2=RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=6,random_state=0)
classifier2.fit(X_train,y_train)
y_pred_train2=classifier2.predict(X_train)
print("accuracy of training data  is",accuracy_score(y_train,y_pred_train2))
y_pred_test2=classifier2.predict(X_test)
print("accuracy of test data  is",accuracy_score(y_test,y_pred_test2))

parameters_rf=[{'n_estimators':[100,500,1000,1500,2000],'criterion':['entropy'],'max_depth':[1,2,3,4,5,6]},
               {'n_estimators':[100,500,1000,1500,2000],'criterion':['gini'],'max_depth':[1,2,3,4,5,6]}]
grid_search_rf=GridSearchCV(estimator=classifier2,
                         param_grid=parameters_rf,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1)
grid_search_rf.fit(X_train,y_train)
best_parameters_rf=grid_search_rf.best_params_
print(best_parameters_rf)

#Classification Report
print("classification report is given as")
print(classification_report(y_test,y_pred_test2))

#Evaluation of Random Forest Classifier
print("accuracy of  RandomForestClassifier is ",accuracy_score(y_test,y_pred_test2))
print("F1 score is given as ",f1_score(y_test,y_pred_test2))
print("Jaccard Scores is given as ",jaccard_score(y_test,y_pred_test2))

print("Confusion Matrix is ")
print(confusion_matrix(y_test,y_pred_test))




#Using of Gradient Boosted Trees
from sklearn.ensemble import GradientBoostingClassifier
classifier3=GradientBoostingClassifier(n_estimators=1500,learning_rate=0.05,max_depth=1,random_state=0)

classifier3.fit(X_train,y_train)

y_pred_train3=classifier3.predict(X_train)
print("accuracy of training data  is",accuracy_score(y_train,y_pred_train3))
y_pred_test3=classifier3.predict(X_test)
print("accuracy of test data  is",accuracy_score(y_test,y_pred_test3))

parameters_gbt=[{'n_estimators':[100,500,1000,1500,2000],'learning_rate':[0.01,0.05,0.1,0.5,1,1.5,2],'max_depth':[1,2,3,4,5,6]}]
grid_search_gbt=GridSearchCV(estimator=classifier3,               
                            param_grid=parameters_gbt,
                            scoring='accuracy',
                            cv=10,
                            n_jobs=-1)
grid_search_gbt.fit(X_train,y_train)
best_parameters_gbt=grid_search_gbt.best_params_
print(best_parameters_gbt)

print("classification report is given as")
print(classification_report(y_test,y_pred_test3))

#Evaluation of Gradient Boosting Classifier
print("accuracy of  RandomForestClassifier is ",accuracy_score(y_test,y_pred_test3))
print("F1 score is given as ",f1_score(y_test,y_pred_test3))
print("Jaccard Scores is given as ",jaccard_score(y_test,y_pred_test3))

print("Confusion Matrix is ")
print(confusion_matrix(y_test,y_pred_test3))


#USING XGBOOST CLASSIFIER
from xgboost import XGBClassifier
classifier4=XGBClassifier(n_estimators=1500,learning_rate=0.05,max_depth=1,min_child_weight=7)
classifier4.fit(X_train,y_train)
y_pred_train4=classifier4.predict(X_train)
print("accuracy of training data  is",accuracy_score(y_train,y_pred_train4))
y_pred_test4=classifier4.predict(X_test)
print("accuracy of test data  is",accuracy_score(y_test,y_pred_test4))

parameters_xgb=[{'n_estimators':[100,500,1000,1500,2000,5000],'learning_rate':[0.01,0.05,0.1,0.5,1,1.5,2],'max_depth':[1,2,3,4,5,6],'min_child_weight':[1,3,5,7,9]}]
grid_search_xgb=GridSearchCV(estimator=classifier4,               
                            param_grid=parameters_xgb,
                            scoring='accuracy',
                            cv=10,
                            n_jobs=-1)
grid_search_xgb.fit(X_train,y_train)
best_parameters_xgb=grid_search_xgb.best_params_
print(best_parameters_xgb)

print("classification report is given as")
print(classification_report(y_test,y_pred_test3))

#Evaluation of XGBoost Classifier
print("accuracy of  XGBOOST classifier is ",accuracy_score(y_test,y_pred_test4))
print("F1 score is given as ",f1_score(y_test,y_pred_test4))
print("Jaccard Scores is given as ",jaccard_score(y_test,y_pred_test4))

print("Confusion Matrix is ")
print(confusion_matrix(y_test,y_pred_test4))

#AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
classifier5=AdaBoostClassifier(n_estimators=100,learning_rate=0.5,random_state=0)
y_train=y_train.ravel()
classifier5.fit(X_train,y_train)
y_pred_train5=classifier5.predict(X_train)
y_pred_test5=classifier5.predict(X_test)
print("accuracy on the training data is ",accuracy_score(y_train,y_pred_train5))
print("accuracy on the test data is ",accuracy_score(y_test,y_pred_test5))
#Parameter tuning for adaboost classifier

parameters_ada=[{'n_estimators':[100,500,1000,1500,2000,5000],'learning_rate':[0.01,0.05,0.1,0.5,1,1.5,2]}]
grid_search_ada=GridSearchCV(estimator=classifier5,               
                            param_grid=parameters_ada,
                            scoring='accuracy',
                            cv=10,
                            n_jobs=-1)
grid_search_ada.fit(X_train,y_train)
best_parameters_ada=grid_search_ada.best_params_
print(best_parameters_ada)

print("classification report is given as")
print(classification_report(y_test,y_pred_test5))

#Evaluation of AdaBoost Classifier
print("accuracy of  AdaBoost is ",accuracy_score(y_test,y_pred_test5))
print("F1 score is given as ",f1_score(y_test,y_pred_test5))
print("Jaccard Scores is given as ",jaccard_score(y_test,y_pred_test5))

print("Confusion Matrix is ")
print(confusion_matrix(y_test,y_pred_test5))

#Random forest with AdaBoost
classifier6=AdaBoostClassifier(RandomForestClassifier())
classifier6.fit(X_train,y_train)
y_pred_train6=classifier6.predict(X_train)
y_pred_test6=classifier6.predict(X_test)
print("accuracy on the training data is ",accuracy_score(y_train,y_pred_train6))
print("accuracy on the test data is ",accuracy_score(y_test,y_pred_test6))

print("classification report is given as")
print(classification_report(y_test,y_pred_test6))

#Evaluation of Random Forest Classifier
print("accuracy of  AdaBoostClassifier is ",accuracy_score(y_test,y_pred_test6))
print("F1 score is given as ",f1_score(y_test,y_pred_test6))
print("Jaccard Scores is given as ",jaccard_score(y_test,y_pred_test6))

print("Confusion Matrix is ")
print(confusion_matrix(y_test,y_pred_test6))

#Using of ExtraTree Classifier
from sklearn.ensemble import ExtraTreesClassifier
classifier7=ExtraTreesClassifier(n_estimators=2000,criterion='gini',max_depth=6,random_state=0)
classifier7.fit(X_train,y_train)
y_pred_train7=classifier7.predict(X_train)
y_pred_test7=classifier7.predict(X_test)
parameters_ETC=[{'n_estimators':[100,500,1000,1500,2000],'criterion':['entropy'],'max_depth':[1,2,3,4,5,6]},
               {'n_estimators':[100,500,1000,1500,2000],'criterion':['gini'],'max_depth':[1,2,3,4,5,6]}]
grid_search_ETC=GridSearchCV(estimator=classifier7,               
                            param_grid=parameters_ETC,
                            scoring='accuracy',
                            cv=10,
                            n_jobs=-1)
grid_search_ETC.fit(X_train,y_train)
best_parameters_ETC=grid_search_ETC.best_params_
print(best_parameters_ETC)

print("classification report is given as")
print(classification_report(y_test,y_pred_test7))

#Evaluation of Random Forest Classifier
print("accuracy of  extratreeClassifier is ",accuracy_score(y_test,y_pred_test7))
print("F1 score is given as ",f1_score(y_test,y_pred_test7))
print("Jaccard Scores is given as ",jaccard_score(y_test,y_pred_test7))

print("Confusion Matrix is ")
print(confusion_matrix(y_test,y_pred_test7))