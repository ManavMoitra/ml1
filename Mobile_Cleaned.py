#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,mean_squared_error
from tqdm import tqdm_notebook
    
#Creating a Class Sigmoid Neuron
class sigmoidneuron:
    
    def __init__(self):
        
        self.w=None
        self.b=None
    def perceptron(self,x):
        
        return np.dot(x,self.w.T)+self.b
    def sigmoid(self,x):
        
        return 1.0/(1.0+np.exp(-x))
    def grad_w(self,x,y):
        
        y_pred=self.sigmoid(self.perceptron(x))
        return (y_pred-y)*y_pred*(1-y_pred)*x    
    def grad_b(self,x,y):
        
        y_pred=self.sigmoid(self.perceptron(x))
        return (y_pred-y)*y_pred*(1-y_pred)
    def fit(self,X,Y,epochs=1,learning_rate=1,initialise=True,display_loss=False):
        if initialise:
            
            self.w=np.random.randn(1,X.shape[1])
            self.b=0
        if display_loss:
            
            loss=[]
                
        for i in tqdm_notebook(range(epochs),total=epochs,unit='epoch'):
            dw=0
            db=0
            for x,y in zip(X,Y):
                dw+=self.grad_w(x,y)
                db+=self.grad_b(x,y)
            self.w=self.w-learning_rate*dw
            self.b=self.b-learning_rate*db
            if display_loss:
                Y_pred=self.sigmoid(self.perceptron(X))
                loss.append(mean_squared_error(Y_pred,Y))
                
        if display_loss:
            #print(loss.values())
            
            plt.plot(np.array(loss))
            plt.xlabel("epochs")
            plt.ylabel("mean squared error")
            plt.show()
            
    def predict(self,X):
        Y=[]
        for x in X:
            result=self.sigmoid(self.perceptron(x))
            Y.append(result)
        return(np.array(Y))   
            
        
#Importing of the dataset            
df=pd.read_csv("mobile_cleaned1.csv")
df.head(10)

#Seperating the Target Variable and Features
X=df.drop("Rating",axis=1)
y=df['Rating'].values
threshold=4.2
df['class']=(df['Rating']>threshold).astype(np.int)

df['class'].value_counts(normalize=True)

y_binarised=df['class'].values    
#Creating train and test dataset for training of models and predictions
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,stratify=y_binarised,random_state=0)
#Scaling of feature variables
sc_X=StandardScaler()
X_scaled_train=sc_X.fit_transform(X_train)
X_scaled_test=sc_X.transform(X_test)
#calling of MinMaxScaler Class
mms=MinMaxScaler()
#Scaling of the Output in range(0,1)
y_scaled_train=mms.fit_transform(y_train.reshape(-1,1))
y_scaled_test=mms.transform(y_test.reshape(-1,1))
#Standard Scaling of the threshold    
scaled_threshold=list(mms.transform(np.array([threshold]).reshape(-1,1)))[0][0]
print(scaled_threshold)
y_binarised_train=(y_scaled_train>scaled_threshold).astype(np.int).ravel()
y_binarised_test=(y_scaled_test>scaled_threshold).astype(np.int).ravel()
#Fitting The Class Sigmoid Neuron
sn=sigmoidneuron()
sn.fit(X_scaled_train,y_scaled_train,epochs=5000,learning_rate=0.01,initialise=True,display_loss=True)
#Making Predicitions for checking of Accuracy
y_pred_train=sn.predict(X_scaled_train)
y_pred_test=sn.predict(X_scaled_test)

y_binarised_pred_train=(y_pred_train>scaled_threshold).astype(np.int).ravel()
y_binarised_pred_test=(y_pred_test>scaled_threshold).astype(np.int).ravel()
#Evaluation of accuracy on training and test data
print("accuracy on training data is   ",accuracy_score(y_binarised_pred_train,y_binarised_train))
print("accuracy on test data is    ",accuracy_score(y_binarised_pred_test,y_binarised_test))







