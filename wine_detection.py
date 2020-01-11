#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#Importing the csv file
df=pd.read_csv("winequality-red.csv")
df.head()
df.info()


bins=[2,6.5,8]#(2-6.5) will be bad and(6.5-8) will be good
labels=['bad','good']
#Conversion into Dummy Variables
df['quality']=pd.cut(df['quality'],bins=bins,labels=labels)
df['quality']=pd.factorize(df['quality'])[0]

X=df.iloc[:,:-1]
y=df.iloc[:,11]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Implementation of Decison Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier1=DecisionTreeClassifier(criterion='entropy',max_depth=6,max_features=6,random_state=0)
classifier1.fit(X_train,y_train)
y_pred1=classifier1.predict(X_test)
print("Accuracy Score of Decision Tree Classifier is ",accuracy_score(y_test,y_pred1))

#Hyperparameter Tuning using GridSearch to get the best accuracy
from sklearn.model_selection import GridSearchCV
parameters_dt=[{'criterion':['entropy'],'max_depth':[1,2,3,4,5,6],'max_features':[2,3,4,5,6,7,8]},
               {'criterion':['gini'],'max_depth':[1,2,3,4,5,6],'max_features':[2,3,4,5,6,7,8]}]
grid_search_dt=GridSearchCV(estimator=classifier1,
                            param_grid=parameters_dt,
                            scoring='accuracy',
                            cv=10,
                            n_jobs=-1)
grid_search_dt.fit(X_train,y_train)
best_parameters_dt=grid_search_dt.best_params_
print(best_parameters_dt)

#Evaluation of DecisionTreeClassifier
print(classifier1.feature_importances_)
print("Confusion Matrix can be given as ")
print(confusion_matrix(y_test,y_pred1))
#Checking the R2_score
#print("R2 score of the model is ",r2_score(y_test,y_pred1))
print("Negative value shows the model doesn't follow a linear trend")
#Classification Report 
print("Classification report is given as ")
print(classification_report(y_test,y_pred1))
#F1-score
print("F1-score of the model is ",f1_score(y_test,y_pred1))



#Using of RandomForest Algorithm
from sklearn.ensemble import RandomForestClassifier
classifier2=RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=1000,random_state=0)
classifier2.fit(X_train,y_train)
y_pred2=classifier2.predict(X_test)
print("accuracy score of RandomForestClassifer ",accuracy_score(y_test,y_pred2))

#Creating a dictionary of parameters for Random forest Classifier
parameters_rf=[{'n_estimators':[100,200,500,1000],'criterion':['entropy'],'max_depth':[3,4,5,6,7,8]},
               {'n_estimators':[100,200,500,1000],'criterion':['gini'],'max_depth':[3,4,5,6,7,8]}]
            

#HyperParameter Tuning of Random Forest Algorithm
grid_search_rf=GridSearchCV(estimator=classifier2,
                            param_grid=parameters_rf,
                            scoring='accuracy',
                            cv=10,
                            n_jobs=-1)
grid_search_rf.fit(X_train,y_train)
best_parameters=grid_search_rf.best_params_
print(best_parameters)
                            
print("Confusion Matrix can be given as ")
print(confusion_matrix(y_test,y_pred2))
#Checking the R2_score
print("R2 score of the model is ",r2_score(y_test,y_pred2))
print("Negative value shows the model doesn't follow a linear trend")
#Classification Report 
print("Classification report is given as ")
print(classification_report(y_test,y_pred2))
#F1-score
print("F1-score of the model is ",f1_score(y_test,y_pred2))                      
                            
                         

#Implementation of Xgboost Classifier
from xgboost import XGBClassifier
classifier3=XGBClassifier(learning_rate=0.5,gamma=0.5,min_child_weight=5,max_depth=6)
classifier3.fit(X_train,y_train)
y_pred3=classifier3.predict(X_test)
print("accuracy score of the classifier is ",accuracy_score(y_test,y_pred3))
#Hyperparameter tuning for XGBClassifier
parameters_xgb=[{'learning_rate':[0.01,0.03,0.1,0.3,0.5,0.8],'gamma':[0.01,0.02,0.1,0.2,0.5,0.8],'min_child_weight':[3,4,5,6,7,8],'max_depth':[3,4,5,6,7,8]}]                                              
grid_search_xgb=GridSearchCV(estimator=classifier3,
                             param_grid=parameters_xgb,
                             scoring='accuracy',
                             cv=10,
                             n_jobs=-1)
grid_search_xgb.fit(X_train,y_train)
best_parameters_xgb=grid_search_xgb.best_params_
print(best_parameters_xgb)                


print("Confusion Matrix can be given as ")
print(confusion_matrix(y_test,y_pred3))
#Checking the R2_score
print("R2 score of the model is ",r2_score(y_test,y_pred3))
print("Negative value shows the model doesn't follow a linear trend")
#Classification Report 
print("Classification report is given as ")
print(classification_report(y_test,y_pred3))
#F1-score
print("F1-score of the model is ",f1_score(y_test,y_pred3))     

#Implementation of Gradient Boosted Trees
from sklearn.ensemble import GradientBoostingClassifier
classifier4=GradientBoostingClassifier(n_estimators=500,max_depth=7,learning_rate=0.01,random_state=0)
classifier4.fit(X_train,y_train)
y_pred4=classifier4.predict(X_test)
print("Accuracy of the classifier is ",accuracy_score(y_test,y_pred4))
#Hyperparametertuning for gradientboosting classifier
parameters_gbt=[{'n_estimators':[100,200,500,1000,1500],'max_depth':[3,4,5,6,7,8],'learning_rate':[0.01,0.1,0.2,0.5,0.6,0.8]}]
grid_search_gbt=GridSearchCV(estimator=classifier4,
                             param_grid=parameters_gbt,
                             scoring='accuracy',
                             cv=10,
                             n_jobs=-1)

grid_search_gbt.fit(X_train,y_train)

best_parameters_gbt=grid_search_gbt.best_params_
print("Best parameters of gradient boosted trees are")
print(best_parameters_gbt)

print("Confusion Matrix can be given as ")
print(confusion_matrix(y_test,y_pred4))
#Checking the R2_score
print("R2 score of the model is ",r2_score(y_test,y_pred4))
print("Negative value shows the model doesn't follow a linear trend")
#Classification Report 
print("Classification report is given as ")
print(classification_report(y_test,y_pred4))
#F1-score
print("F1-score of the model is ",f1_score(y_test,y_pred4))   
  
#Useing of adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
classifier_ada=AdaBoostClassifier(n_estimators=1000,learning_rate=1,random_state=0)
classifier_ada.fit(X_train,y_train)
y_pred_ada=classifier_ada.predict(X_test)
print("accuracy of adaboost classifier is ",accuracy_score(y_test,y_pred_ada))

#Hyperparameter tuning of Adaboost Classifier
parameters_ada=[{'n_estimators':[100,200,250,1000,10000],'learning_rate':[0.1,0.2,0.5,0.8,1,2]}]

grid_search_ada=GridSearchCV(estimator=classifier_ada,
                             param_grid=parameters_ada,
                             scoring='accuracy',
                             cv=10,
                             n_jobs=-1)
grid_search_ada.fit(X_train,y_train)
best_parameters=grid_search_ada.best_params_
print("best parameters of adaboost classifier are ")
print(best_parameters)
#Evaluation of AdaBoost Classifier
print("Confusion Matrix can be given as ")
print(confusion_matrix(y_test,y_pred_ada))
#Checking the R2_score
print("R2 score of the model is ",r2_score(y_test,y_pred_ada))
print("Negative value will  show that the model doesn't follow a linear trend")
#Classification Report 
print("Classification report is given as ")
print(classification_report(y_test,y_pred_ada))
#F1-score
print("F1-score of the model is ",f1_score(y_test,y_pred_ada))     

#standard scaling of the data
from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
X_train=Sc_X.fit_transform(X_train)
X_test=Sc_X.transform(X_test)
from sklearn.svm import SVC
classifier5=SVC(kernel='rbf',C=10,gamma=0.1)
classifier5.fit(X_train,y_train)
y_pred5=classifier5.predict(X_test)
print("Accuracy score of this classifier is ",accuracy_score(y_test,y_pred5))

#Hyperparameter tuning of SVM
parameters_svm=[{'kernel':['rbf'],'C':[0.1,0.5,1,10,100,1000],'gamma':[0.01,0.1,0.2,0.5,1,2]},
                {'kernel':['linear'],'C':[0.1,0.5,10,100,1000],'gamma':[0.01,0.1,0.2,0.5,1,2]}]

grid_search_svm=GridSearchCV(estimator=classifier5,
                             param_grid=parameters_svm,
                             scoring='accuracy',
                             cv=10,
                             n_jobs=-1)
grid_search_svm.fit(X_train,y_train)
best_parameters_svm=grid_search_svm.best_params_
#Plotting the confusion matrix
print("Confusion Matrix can be given as ")
print(confusion_matrix(y_test,y_pred5))
#Checking the R2_score
print("R2 score of the model is ",r2_score(y_test,y_pred5))
print("Negative value shows the model doesn't follow a linear trend")
#Classification Report 
print("Classification report is given as ")
print(classification_report(y_test,y_pred5))
#F1-score
print("F1-score of the model is ",f1_score(y_test,y_pred5)) 


from sklearn.naive_bayes import GaussianNB
classifier6=GaussianNB()
classifier6.fit(X_train,y_train)
y_pred6=classifier6.predict(X_test)
#Evaluation of Naive Bayes
print("Confusion Matrix can be given as ")
print(confusion_matrix(y_test,y_pred6))
#Checking the R2_score
print("R2 score of the model is ",r2_score(y_test,y_pred6))
print("Negative value shows the model doesn't follow a linear trend")
#Classification Report 
print("Classification report is given as ")
print(classification_report(y_test,y_pred6))
#F1-score
print("F1-score of the model is ",f1_score(y_test,y_pred6)) 

#Using of KNearestNeighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier7=KNeighborsClassifier(metric='minkowski',n_neighbors=4,p=2)
classifier7.fit(X_train,y_train)
y_pred7=classifier7.predict(X_test)

print("Accuracy score of KNeighborsClassifier is ",accuracy_score(y_test,y_pred7))

parameters_knn=[{'metric':['minkowski'],'n_neighbors':[2,3,4,5,6,7]},
                {'metric':['manhattan'],'n_neighbors':[2,3,4,5,6,7]}]

grid_search_knn=GridSearchCV(estimator=classifier7,
                             param_grid=parameters_knn,                             
                             scoring='accuracy',
                             cv=10,
                             n_jobs=-1)
grid_search_knn.fit(X_train,y_train)  
best_parameters=grid_search_knn.best_params_
print("best parameters using knn algorithm were found to be")
print(best_parameters)

#Evaluation of KNN classifier
print("Confusion Matrix can be given as ")
print(confusion_matrix(y_test,y_pred7))
#Checking the R2_score
print("R2 score of the model is ",r2_score(y_test,y_pred7))
print("Negative value shows the model doesn't follow a linear trend")
#Classification Report 
print("Classification report is given as ")
print(classification_report(y_test,y_pred7))
#F1-score
print("F1-score of the model is ",f1_score(y_test,y_pred7))

#Using of Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier8=LogisticRegression()
classifier8.fit(X_train,y_train)
y_pred8=classifier8.predict(X_test)
print("accuracy score using knn classifier was found to be ",accuracy_score(y_test,y_pred8))
#Evaluation of KNNClassifier
print("Confusion Matrix can be given as ")
print(confusion_matrix(y_test,y_pred8))
#Checking the R2_score
#print("R2 score of the model is ",r2_score(y_test,y_pred1))
print("Negative value shows the model doesn't follow a linear trend")
#Classification Report 
print("Classification report is given as ")
print(classification_report(y_test,y_pred8))
#F1-score
print("F1-score of the model is ",f1_score(y_test,y_pred8))


