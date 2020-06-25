import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_openml
dataset = fetch_openml('mnist_784',version = 1)

X = dataset.data
y = dataset.target
y = y.astype(np.int32)

%matplotlib qt5

some_digit = X[69696]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image)
plt.show()

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 30)
dtf.fit(X,y)

dtf.score(X,y)

y_pred = dtf.predict(X)
y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)


some_digit = X[32645]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image)
plt.show()

y_pred_test = dtf.predict(X[[11111,3936,7654,42042,7777], 0:784])
y_pred_test

for i in range(25):
    image = X[i]
    image = image.reshape(28,28)
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
plt.show()
