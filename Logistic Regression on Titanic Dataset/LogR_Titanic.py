import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib qt5
import math

## Data Collection

data = pd.read_csv('titanic.csv')
data.head(10)
print("# of passengers in original data : "+str(len(data.index)))
      
## Analysis Data

sns.countplot(x="Survived",data=data)   
sns.countplot(x="Survived",hue="Sex",data=data)   
sns.countplot(x="Survived",hue="Pclass",data=data)
data["Age"].plot.hist()
data["Fare"].plot.hist(bins=20,figsize=(10,5))
data.info()
sns.countplot(x="SibSp",data=data)

## Data Wrangling

data.isnull()
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False)
sns.boxplot(x="Pclass",y="Age",data=data)

data.head()
data.drop("Cabin",axis=1,inplace=True)
data.head()
data.dropna(inplace=True)
sns.heatmap(data.isnull(), yticklabels=False)
data.isnull().sum()

data.head(2)

sex=pd.get_dummies(data["Sex"],drop_first=True)
sex.head(5)

embark=pd.get_dummies(data["Embarked"],drop_first=True)
embark.head()

pcl=pd.get_dummies(data["Pclass"],drop_first=True)
pcl.head()

data=pd.concat([data,sex,embark,pcl],axis=1)
data.head()

data.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)
data.head()
data.drop('Pclass',axis=1,inplace=True)
data.head()

##Train & Test Data

X=data.drop("Survived",axis=1)
y=data["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)

predict = log.predict(X_test)

from sklearn.metrics import classification_report
classification_report(y_test,predict)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predict)

## Accuracy Check

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predict)























