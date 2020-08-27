import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\503107711\Desktop\Project for GIT\Diabetes\Diabetes.csv')

#Checking the shape of the dataset and basic information.
print(data.info())
print(data.shape)

#Creating a copy of the dataset
data_copy=data.copy()

#Replacing the 0 values with NaN where 0 means no data available
data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(data_copy.isnull().sum())
#Replacing the NaN with mean
data_copy['Glucose']=data_copy['Glucose'].fillna(data_copy['Glucose'].mean())
data_copy['BloodPressure']=data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean())
data_copy['SkinThickness']=data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].mean())
data_copy['Insulin']=data_copy['Insulin'].fillna(data_copy['Insulin'].mean())
data_copy['BMI']=data_copy['BMI'].fillna(data_copy['BMI'].mean())

#Checking if the dataset is free of null values
print(data_copy.isnull().sum())

#Splitting the dataset into Dependent and independent variables
X=data_copy.iloc[:,:-1]
Y=data.iloc[:,-1]

#Using StandardScaler to Standardize the data
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_scaled=ss.fit_transform(X)

#Splitting the data into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.3,random_state=1234)

#Creating multiple models and evaluating the models
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
from sklearn.svm import SVC
svc=SVC()
from sklearn.metrics import accuracy_score
lr.fit(x_train,y_train)
y_predict=lr.predict(x_test)
ac_sc=accuracy_score(y_test, y_predict)
print(ac_sc)

dt.fit(x_train,y_train)
y_predict1=dt.predict(x_test)
ac_sc1=accuracy_score(y_test, y_predict1)
print(ac_sc1)

rf.fit(x_train,y_train)
y_predict_rf=rf.predict(x_test)
ac_score_rf=accuracy_score(y_test,y_predict_rf) 
print(ac_score_rf)

#As the Random forest is giving best accuracy so creating a pickle file of Random forest classifier.
import pickle
filename = r'C:\Users\503107711\Desktop\Project for GIT\Diabetes\diabetes-prediction-rfc-model.pkl'
pickle.dump(rf, open(filename, 'wb'))
