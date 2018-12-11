

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, Imputer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
```


```python
pd.options.display.max_columns = 10
pd.options.display.max_rows = 100
DATA_PATH = "dataset\Iris.csv"
df = pd.read_csv(DATA_PATH)
```


```python
#cleaning data
df = df.drop(axis=1, columns="Id")
df.dropna(axis=0, how="any")
```


```python
df = df.replace(
        ["Iris-versicolor", "Iris-virginica", "Iris-setosa"], 
        range(3)
    )
```


```python
split = StratifiedShuffleSplit(test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["Species"]):
    train = df.loc[train_index]
    test = df.loc[test_index]
x_train = train.drop(columns="Species", axis=1).values
y_train = train["Species"].values
x_test = test.drop(columns="Species", axis=1).values
y_test = test["Species"].values
```


```python
#standardization
#std_scaler = StandardScaler()
#x_train = std_scaler.fit_transform(x_train)
#x_test = std_scaler.transform(x_test)
```


```python
pipe = Pipeline([
        #('imputer', Imputer(strategy="median")),
        #('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
x_train = pipe.fit_transform(x_train)
x_test = pipe.transform(x_test)
```


```python
cross_val_score(SGDClassifier(max_iter=1000, tol=1e-3), x_train, y_train, cv=5)
```




    array([0.95833333, 0.875     , 0.91666667, 0.91666667, 0.95833333])




```python
cross_val_score(RandomForestClassifier(n_estimators=20), x_train, y_train, cv=5)
```




    array([0.95833333, 0.95833333, 0.91666667, 0.95833333, 1.        ])




```python
cross_val_score(SVC(gamma="scale", kernel="rbf", class_weight="balanced"), x_train, y_train, cv=5)
```




    array([0.95833333, 0.95833333, 0.91666667, 0.95833333, 1.        ])




```python
cross_val_score(SVC(gamma="scale", kernel="linear"), x_train, y_train, cv=5)
```




    array([0.95833333, 0.95833333, 1.        , 0.91666667, 1.        ])




```python
cross_val_score(SVC(gamma="scale", kernel="poly"), x_train, y_train, cv=5)
```




    array([1.        , 0.875     , 0.875     , 0.91666667, 0.91666667])




```python
cross_val_score(SVC(gamma="scale", kernel="sigmoid"), x_train, y_train, cv=5)
```




    array([0.91666667, 0.875     , 0.91666667, 0.875     , 0.91666667])




```python
cross_val_score(LinearSVC(), x_train, y_train, cv=5)
```




    array([0.95833333, 0.83333333, 0.91666667, 0.91666667, 1.        ])




```python
cross_val_score(KNeighborsClassifier(), x_train, y_train, cv=5)
```




    array([0.95833333, 0.95833333, 0.91666667, 0.91666667, 1.        ])


