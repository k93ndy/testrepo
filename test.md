

```python
"""test"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, Normalizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import log_loss, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

#load raw data
print("data loading...")
df = pd.read_csv("C:/Users/di.sun/Box Sync/di.sun/work/2017/data analystics/data/10-kyusyu.csv")
print("data loaded")

#
# X = df.iloc[:, 1:-2].values
# y = df.loc[:, 'riyo_flg'].values
# print(y)
# print(np.count_nonzero(y))
# y = np.invert(y.astype(np.bool)).astype(np.int)
# print(y)
# print(np.count_nonzero(y))

#split data into train set and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#select sample
def select_sample(data):
    y_0_num = data[data['riyo_flg']==0].shape[0]
    y_1_num = 1 * y_0_num
    y_0 = data[data['riyo_flg']==0]
    y_1 = data[data['riyo_flg']==1]
    shuffled_indices = np.random.permutation(data[data['riyo_flg']==1].shape[0])
    y_1_indices = shuffled_indices[:y_1_num]
    y_1 = data[data['riyo_flg']==1].iloc[y_1_indices]
    return pd.concat([y_0, y_1]).reset_index(drop=True)
    
#stratified split
# df = select_sample(df)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
for train_index, test_index in split.split(df, df.loc[:, 'riyo_flg']):
    train = df.loc[train_index]
    test = df.loc[test_index]
X_train_raw = train.iloc[:, 3:].values
y_train = train.loc[:, 'riyo_flg'].values
X_test_raw = test.iloc[:, 3:].values
y_test = test.loc[:, 'riyo_flg'].values
y_train = np.invert(y_train.astype(np.bool)).astype(np.int)
y_test = np.invert(y_test.astype(np.bool)).astype(np.int)
print(X_train_raw.shape, X_test_raw.shape)
print(y_train.shape, y_test.shape)
```


```python
df.describe()
```


```python
#standardization
std_scaler = StandardScaler()
# norm1 = Normalizer()
# norm2 = Normalizer()
#X = std_scaler.fit_transform(X)
# X_train_scaled = std_scaler.fit_transform(X_train_raw[:, 14:])
# X_test_scaled = std_scaler.transform(X_test_raw[:, 14:])
X_train_scaled = std_scaler.fit_transform(np.c_[X_train_raw[:, 22:26], X_train_raw[:, 34:]])
X_test_scaled = std_scaler.transform(np.c_[X_test_raw[:, 22:26], X_test_raw[:, 34:]])
# X_train_scaled = std_scaler.fit_transform(np.c_[X_train_raw[:, :12], X_train_raw[:, 12:]])
# X_test_scaled = std_scaler.transform(np.c_[X_test_raw[:, :12], X_test_raw[:, 12:]])
#std first 2col and norm rest
# first2column_scaled = std_scaler.fit_transform(X_train_raw[:,:2])
# X_train_scaled = np.c_[norm1.fit_transform(X_train_raw[:,:12]),norm2.fit_transform(X_train_raw[:,12:])]
# X_test_scaled = np.c_[norm1.transform(X_test_raw[:,:12]),norm2.transform(X_test_raw[:,12:])]

# X_train_scaled = np.c_[norm1.fit_transform(X_train_raw[:,10:14]),norm2.fit_transform(X_train_raw[:,22:])]
# X_test_scaled = np.c_[norm1.fit_transform(X_test_raw[:,10:14]),norm2.fit_transform(X_test_raw[:,22:])]
print(X_train_scaled.shape, X_test_scaled.shape)
```


```python
#use knn classifier 
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

#predict
y_train_pred = knn.predict(X_train_scaled)
y_test_pred = knn.predict(X_test_scaled)
# y_test_pred = forest.predict(X_test)

#evaluate
print("cross entropy on train set:", log_loss(y_train, y_train_pred))
print("cross entropy on test set:", log_loss(y_test, y_test_pred))

print("confusion matrix on train set:\n", confusion_matrix(y_train, y_train_pred))
print("confusion matrix on test set:\n", confusion_matrix(y_test, y_test_pred))

print("precision on train set:\n", precision_score(y_train, y_train_pred))
print("precision on test set:\n", precision_score(y_test, y_test_pred))

print("recall on train set:\n", recall_score(y_train, y_train_pred))
print("recall on test set:\n", recall_score(y_test, y_test_pred))

# tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
# print(tn, fp, fn, tp)

print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))

```


```python
#use random forest for training
forest = RandomForestClassifier()
forest.fit(X_train_scaled, y_train)

#predict
y_train_pred = forest.predict(X_train_scaled)
y_test_pred = forest.predict(X_test_scaled)
# y_test_pred = forest.predict(X_test)

#evaluate
print("cross entropy on train set:", log_loss(y_train, y_train_pred))
print("cross entropy on test set:", log_loss(y_test, y_test_pred))

# print("confusion matrix on train set:\n", confusion_matrix(y_train, y_train_pred))
# print("confusion matrix on test set:\n", confusion_matrix(y_test, y_test_pred))

# print("precision on train set:\n", precision_score(y_train, y_train_pred))
# print("precision on test set:\n", precision_score(y_test, y_test_pred))

# print("recall on train set:\n", recall_score(y_train, y_train_pred))
# print("recall on test set:\n", recall_score(y_test, y_test_pred))

# tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
# print(tn, fp, fn, tp)

print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))
```


```python
print(forest.feature_importances_)
```


```python
#use SGD as classifier
sgd = SGDClassifier(max_iter=1000, tol=1e-3)
sgd.fit(X_train_scaled, y_train)

#predict
y_train_pred = sgd.predict(X_train_scaled)
y_test_pred = sgd.predict(X_test_scaled)
# y_test_pred = forest.predict(X_test)

#evaluate
print("cross entropy on train set:", log_loss(y_train, y_train_pred))
print("cross entropy on test set:", log_loss(y_test, y_test_pred))

print("confusion matrix on train set:\n", confusion_matrix(y_train, y_train_pred))
print("confusion matrix on test set:\n", confusion_matrix(y_test, y_test_pred))

print("precision on train set:\n", precision_score(y_train, y_train_pred))
print("precision on test set:\n", precision_score(y_test, y_test_pred))

print("recall on train set:\n", recall_score(y_train, y_train_pred))
print("recall on test set:\n", recall_score(y_test, y_test_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
print(tn, fp, fn, tp)

print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))
```


```python
print(sgd.loss_function_)
print(sgd.intercept_)
print(sgd.coef_)
```


```python
sgd.predict(std_scaler.transform([[4,3,2,1,4,3,2,1]]))
```


```python
svc_clf = SVC(class_weight={1: 5})
svc_clf.fit(X_train_scaled, y_train)

#predict
y_train_pred = svc_clf.predict(X_train_scaled)
y_test_pred = svc_clf.predict(X_test_scaled)
# y_test_pred = forest.predict(X_test)

#evaluate
print("cross entropy on train set:", log_loss(y_train, y_train_pred))
print("cross entropy on test set:", log_loss(y_test, y_test_pred))

print("confusion matrix on train set:\n", confusion_matrix(y_train, y_train_pred))
print("confusion matrix on test set:\n", confusion_matrix(y_test, y_test_pred))

print("precision on train set:\n", precision_score(y_train, y_train_pred))
print("precision on test set:\n", precision_score(y_test, y_test_pred))

print("recall on train set:\n", recall_score(y_train, y_train_pred))
print("recall on test set:\n", recall_score(y_test, y_test_pred))

# tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
# print(tn, fp, fn, tp)

print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))
```


```python
print(svc_clf.intercept_)
print(svc_clf.coef_)
```


```python
svc_clf.predict(std_scaler.transform([[0,1,0,0,0,1,0,1]]))
```


```python
svc_clf = LinearSVC(C=100)
svc_clf.fit(X_train_scaled, y_train)

#predict
y_train_pred = svc_clf.predict(X_train_scaled)
y_test_pred = svc_clf.predict(X_test_scaled)
# y_test_pred = forest.predict(X_test)

#evaluate
print("cross entropy on train set:", log_loss(y_train, y_train_pred))
print("cross entropy on test set:", log_loss(y_test, y_test_pred))

print("confusion matrix on train set:\n", confusion_matrix(y_train, y_train_pred))
print("confusion matrix on test set:\n", confusion_matrix(y_test, y_test_pred))

print("precision on train set:\n", precision_score(y_train, y_train_pred))
print("precision on test set:\n", precision_score(y_test, y_test_pred))

print("recall on train set:\n", recall_score(y_train, y_train_pred))
print("recall on test set:\n", recall_score(y_test, y_test_pred))

# tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
# print(tn, fp, fn, tp)

print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))
```


```python

```
