

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, Imputer
```


```python
pd.options.display.max_columns = 10
pd.options.display.max_rows = 100
DATA_PATH = "dataset\Iris.csv"
df = pd.read_csv(DATA_PATH)
df = df.drop(axis=1, columns="Id")
```


```python
print(df.describe())
print(df.info())
print(df.columns)
print(df["Species"].value_counts(dropna=False))
print(df.head())
```


```python
test = df.replace(["Iris-versicolor", "Iris-virginica", "Iris-setosa"], [1, 2, 3])
test.plot(kind="scatter", x="SepalLengthCm", y="PetalLengthCm", alpha=0.5,
        c="Species", cmap=plt.get_cmap("jet"), colorbar=True)
```


```python
print(test["Species"].value_counts())
```


```python
columns = df.columns
scatter_matrix(df[columns], figsize=(12, 8))
```


```python
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])
print(le.classes_)
```


```python
df.corr()["Species"].sort_values(ascending=False)
```


```python
split = StratifiedShuffleSplit(test_size=0.2, random_state=42)
for train_index, test_index in split.split(test, test["Species"]):
    train_set = test.loc[train_index]
    test_set = test.loc[test_index]
```


```python
print(train_set["Species"].value_counts())
```


```python
#cleaning data
df.dropna(axis=0, how="any")
#standardization
std_scaler = StandardScaler()
test = df.values[:, :-1]
test = std_scaler.fit_transform(test)
test = pd.DataFrame(test, index=df.index, columns=df.columns[:-1])
test["Species"] = df["Species"]
columns = test.columns
scatter_matrix(test[columns], figsize=(12, 8))
```


```python
test.info()
```


```python
test.head()
```


```python
le = LabelEncoder()
ohe = OneHotEncoder()
labels = le.fit_transform(test["Species"])
#reconvert
#[le.classes_[x] for x in labels]
#ohe.fit_transform(labels.reshape(-1, 1))
test["Species"] = labels
test.head()
```


```python
split = StratifiedShuffleSplit(test_size=0.2, random_state=42)
for train_index, test_index in split.split(test, test["Species"]):
    train_set = test.loc[train_index]
    test_set = test.loc[test_index]
train_x = train_set[:, :-1].values
test_x = test_set[:, :-1].values
```


```python

```
