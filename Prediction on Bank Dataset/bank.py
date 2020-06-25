import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('bank-additional-full.csv', sep = ';', na_values = 'unknown')
miss_cond = np.array(dataset.isnull().sum() != 0)
miss_col_ind = np.where(miss_cond)[0]
miss_cols = list(dataset.columns[miss_col_ind])

missing_column_freq = []

for i in miss_col_ind:
    missing_column_freq.append(dataset.iloc[:, i].value_counts())

dataset_miss = dataset[miss_cols]
dataset_miss.isnull().sum()    

for i in range(6):
    missing_column_freq[i] = missing_column_freq[i].append(pd.Series({'null' : dataset_miss.isnull().sum()[i]}))

for i in range(6):
    plt.figure(figsize = (12, 12))
    sns.barplot(missing_column_freq[i].index, missing_column_freq[i].values)
    plt.savefig("figure {}".format(i))
plt.show()

from sklearn.impute import SimpleImputer
sim = SimpleImputer(strategy = "most_frequent")
sim.fit(dataset_miss)
sim.statistics_
dataset_miss = pd.DataFrame(sim.transform(dataset_miss))

dataset_miss.isnull().sum()

dataset1 = dataset.copy()
dataset1.isnull().sum()

dataset1[miss_cols] = dataset_miss
dataset1.isnull().sum()

plt.hist(dataset1['age'], bins = 100)
plt.show()

# Will be discarded
plt.hist(dataset1['duration'], bins = 100)
plt.show()

plt.hist(dataset1['campaign'], bins = 100)
plt.show()

# Will be discarded
plt.hist(dataset1['pdays'], bins = 100)
plt.show()

plt.hist(dataset1['previous'], bins = 100)
plt.show()

plt.hist(dataset1['emp.var.rate'], bins = 100)
plt.show()

plt.hist(dataset1['cons.price.idx'], bins = 100)
plt.show()

plt.hist(dataset1['cons.conf.idx'], bins = 100)
plt.show()

plt.hist(dataset1['euribor3m'], bins = 100)
plt.show()

plt.hist(dataset1['nr.employed'], bins = 100)
plt.show()

pd.plotting.scatter_matrix(dataset1)

corr_mat = dataset1.corr()
corr_mat_sp = dataset1.corr('spearman')

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

dataset2 = dataset1.copy()
dataset2 = dataset2.drop(['duration', 'pdays'], axis = 1)
dataset2 = pd.get_dummies(dataset2)

X = dataset2.iloc[:, :-2]
y = dataset1.iloc[:, -1]
y = lab.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

log_reg.score(X_train, y_train)
log_reg.score(X_valid, y_valid)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

nb.score(X_train, y_train)
nb.score(X_valid, y_valid)

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X_train, y_train)

dtf.score(X_train, y_train)
dtf.score(X_valid, y_valid)

from sklearn.model_selection import cross_val_score, cross_val_predict
cross_val_score(log_reg, X_train, y_train, cv = 5)
cross_val_score(nb, X_train, y_train, cv = 5)
cross_val_score(dtf, X_train, y_train, cv = 5)

y_pred_log = cross_val_predict(log_reg, X_train, y_train, cv = 5)
y_pred_nb = cross_val_predict(nb, X_train, y_train, cv = 5)
y_pred_dtf = cross_val_predict(dtf, X_train, y_train, cv = 5)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

precision_score(y_train, y_pred_log)
recall_score(y_train, y_pred_log)
f1_score(y_train, y_pred_log)

precision_score(y_train, y_pred_nb)
recall_score(y_train, y_pred_nb)
f1_score(y_train, y_pred_nb)

precision_score(y_train, y_pred_dtf)
recall_score(y_train, y_pred_dtf)
f1_score(y_train, y_pred_dtf)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_log)
metrics.auc(fpr, tpr)

fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_nb)
metrics.auc(fpr, tpr)

fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_dtf)
metrics.auc(fpr, tpr)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 7)
rf.fit(X_train, y_train)

rf.score(X_train, y_train)
rf.score(X_valid, y_valid)

y_pred_rf = cross_val_predict(rf, X_train, y_train, cv = 5)

precision_score(y_train, y_pred_rf)
recall_score(y_train, y_pred_rf)
f1_score(y_train, y_pred_rf)

fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_rf)
metrics.auc(fpr, tpr)

cm_log = confusion_matrix(y_train, y_pred_log)
cm_nb = confusion_matrix(y_train, y_pred_nb)
cm_dtf = confusion_matrix(y_train, y_pred_dtf)
cm_rf = confusion_matrix(y_train, y_pred_rf)

metrics.roc_auc_score(y_train, y_pred_log)
metrics.roc_auc_score(y_train, y_pred_nb)
metrics.roc_auc_score(y_train, y_pred_dtf)
metrics.roc_auc_score(y_train, y_pred_rf)



















































































































