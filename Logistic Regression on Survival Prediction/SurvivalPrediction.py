import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib qt4

data = pd.read_csv('logistic case study life survival prediction.csv')
data.head()
data.columns
data.isnull().sum()
data['age'].plot.hist()
data['age'].fillna(data['age'].mean(), inplace=True)
data.pop('name')
data.info()

gen = pd.get_dummies(data['age'],drop_first=True)
gen.head()

emb=pd.get_dummies(data["embarked"],drop_first=True)
emb.head(25)

data=pd.concat((data,gen,emb),axis=1)
data.shape

data=data.drop(["sex","embarked"],axis=1)
data.shape

y=data["survived"]
x=data.drop(["survived"],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train,y_train)
log.score(x_train,y_train)

y_pred = log.predict(x_test)
y_pred

y_pred_pro=log.predict_proba(x_test)
y_pred_pro

from sklearn.metrics import accuracy_score
print("Model Accuracy is:",accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix
print("confusion matrix :")
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))















































