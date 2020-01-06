#implementation of family of trees on iris flower dataset
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
#Getting The dataset
from sklearn.datasets import load_iris
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['class']=iris.target
#Setting the target and output
X=df.drop('class',axis=1)
y=df['class']
#Splitting The Data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#Getting the accuracy base line using dummy classifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
clf=DummyClassifier(strategy='uniform',random_state=0)
clf.fit(X_train,y_train)
y_pred1=clf.predict(X_test)
print("accuracy of the dummy classifier is ",accuracy_score(y_test,y_pred1))
#Useing DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
classifier1=DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=0)
classifier1.fit(X_train,y_train)
y_pred2=classifier1.predict(X_test)
print("accuracy score using Decision Tree is ",accuracy_score(y_test,y_pred2))
#Checking The Importance Of Each Feature
importance=classifier1.feature_importances_
print(importance)

#Using OF RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier2=RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=2,random_state=0)
classifier2.fit(X_train,y_train)
y_pred3=classifier2.predict(X_test)
print("accuracy score using Random Forest can be calculated as ",accuracy_score(y_test,y_pred3))
#Selection of Suitable Parameters for Random Forest Algorithm
from sklearn.model_selection import GridSearchCV
parameters_rf=[{'n_estimators':[100,150,200,500,1000],'criterion':['entropy'],'max_depth':[1,2,3,4,5,6]},
               {'n_estimators':[100,150,200,500,1000],'criterion':['gini'],'max_depth':[1,2,3,4,5,6]}] 

grid_search_rf=GridSearchCV(estimator=classifier2,
                            param_grid=parameters_rf,
                            scoring='accuracy',
                            cv=10,
                            n_jobs=-1)
grid_search_rf.fit(X_train,y_train)
best_parameters=grid_search_rf.best_params_

print(best_parameters)
#Using of XGBOOST Classifier
from xgboost import XGBClassifier
classifier3=XGBClassifier(learning_rate=0.8,max_depth=3,gamma=0.4,min_child_weight=3)
classifier3.fit(X_train,y_train)
y_pred4=classifier3.predict(X_test)
print("Accuracy Score of XGBClassifier is ",accuracy_score(y_test,y_pred4))

#XGBOOST HYPERPARAMETER TUNING
parameters_xgb=[{'learning_rate':[0.01,0.1,0.2,0.5,0.75,0.8],'max_depth':[0,1,2,3,4,5],'gamma':[0.1,0.2,0.3,0.4],'min_child_weight':[1,3,5,7]}]

grid_search_xgb=GridSearchCV(estimator=classifier3,
                             param_grid=parameters_xgb,
                             scoring='accuracy',
                             cv=10,
                             n_jobs=-1)
grid_search_xgb.fit(X_train,y_train)
best_parameters=grid_search_xgb.best_params_

#Using of Gradient Boosted trees
from sklearn.ensemble import GradientBoostingClassifier
classifier4=GradientBoostingClassifier(learning_rate=0.01,max_depth=5,random_state=0)
classifier4.fit(X_train,y_train)
y_pred5=classifier4.predict(X_test)

print("accuracy score for Gradient Boosted Tree is ",accuracy_score(y_test,y_pred5))
#Feature Scaling of the data
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test) 

#Using of Support Vector Classifier

from sklearn.svm import SVC
classifier5=SVC(kernel='linear',C=0.5,gamma=0.1)
classifier5.fit(X_train,y_train)
y_pred6=classifier5.predict(X_test)
print("accuracy score using SVC Classifier is ",accuracy_score(y_test,y_pred6))
#HyperParameter Tuning for SVC
parameters_svm=[{'kernel':['rbf'],'C':[0.01,0.1,0.5,1,10,100],'gamma':[0.1,0.2,0.3,0.4]},
                {'kernel':['linear'],'C':[0.01,0.1,0.5,10,100],'gamma':[0.1,0.2,0.3,0,4]}]
grid_search_svm=GridSearchCV(estimator=classifier5,
                             param_grid=parameters_svm,
                             scoring='accuracy',
                             cv=10,
                             n_jobs=-1)
grid_search_svm.fit(X_train,y_train)
best_parameters=grid_search_svm.best_params_
best_score=grid_search_svm.best_score_
print(best_parameters)
#Using Of Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier6=GaussianNB()
classifier6.fit(X_train,y_train)
y_pred7=classifier6.predict(X_test)

print("accuracy score of Naive Bayes Classifier is ",accuracy_score(y_test,y_pred7))