
## Feature Scaling and Normalization - Lab

## Introduction
In this lab, you'll practice your feature scaling and normalization skills!

## Objectives
You will be able to:
* Implement min-max scaling, mean-normalization, log normalization and unit vector normalization in python
* Identify appropriate normalization and scaling techniques for given dataset

## Back to our Boston Housing data

Let's import our Boston Housing data. Remember we categorized two variables and deleted the "NOX" (nitride oxide concentration) variable because it was highly correlated with two other features.


```python
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()

boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)

# first, create bins for based on the values observed. 5 values will result in 4 bins
bins = [0, 3, 4 , 5, 24]
bins_rad = pd.cut(boston_features['RAD'], bins)
bins_rad = bins_rad.cat.as_unordered()

# first, create bins for based on the values observed. 5 values will result in 4 bins
bins = [0, 250, 300, 360, 460, 712]
bins_tax = pd.cut(boston_features['TAX'], bins)
bins_tax = bins_tax.cat.as_unordered()

tax_dummy = pd.get_dummies(bins_tax, prefix="TAX")
rad_dummy = pd.get_dummies(bins_rad, prefix="RAD")
boston_features = boston_features.drop(["RAD","TAX"], axis=1)
boston_features = pd.concat([boston_features, rad_dummy, tax_dummy], axis=1)
boston_features = boston_features.drop("NOX",axis=1)
```

## Look at the histograms for the continuous variables


```python
%matplotlib inline
```


```python
boston_features.drop(['RAD_(0, 3]', 'RAD_(3, 4]', 'RAD_(4, 5]', 'RAD_(5, 24]',
       'TAX_(0, 250]', 'TAX_(250, 300]', 'TAX_(300, 360]', 'TAX_(360, 460]',
       'TAX_(460, 712]'], axis=1).hist(figsize=(8,8));
```


![png](index_files/index_8_0.png)


## Perform log transformations for the variables where it makes sense


```python
#log for skewed data
import numpy as np
# boston_features_log = np.log(boston_features.drop(['RAD_(0, 3]', 'RAD_(3, 4]', 'RAD_(4, 5]', 'RAD_(5, 24]',
#        'TAX_(0, 250]', 'TAX_(250, 300]', 'TAX_(300, 360]', 'TAX_(360, 460]',
#        'TAX_(460, 712]'], axis=1))
# boston_features_log.replace('-inf', 0)#.hist()
data_log= pd.DataFrame([])
data_log["AGE"] = np.log(boston_features["AGE"])
data_log["B"] = np.log(boston_features["B"])
data_log["CRIM"] = np.log(boston_features["CRIM"])
data_log["DIS"] = np.log(boston_features["DIS"])
data_log["INDUS"] = np.log(boston_features["INDUS"])
data_log["LSTAT"] = np.log(boston_features["LSTAT"])
data_log["PTRATIO"] = np.log(boston_features["PTRATIO"])
data_log.hist(figsize  = [6, 6]);
```


![png](index_files/index_10_0.png)


Analyze the results in terms of how they improved the normality performance. What is the problem with the "ZN" variable?  


```python
data_log["ZN"] = np.log(boston_features["ZN"])
boston_features["ZN"].describe()
```

    /anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:1: RuntimeWarning: divide by zero encountered in log
      """Entry point for launching an IPython kernel.





    count    506.000000
    mean      11.363636
    std       23.322453
    min        0.000000
    25%        0.000000
    50%        0.000000
    75%       12.500000
    max      100.000000
    Name: ZN, dtype: float64



"ZN" has a lot of zeros (more than 50%!). Remember that this variable denoted: "proportion of residential land zoned for lots over 25,000 sq.ft.". It might have made sense to categorize this variable to "over 25,000 feet or not (binary variable 1/0). Now you have a zero-inflated variable which is cumbersome to work with.

## Try different types of transformations on the continuous variables

Store your final features in a dataframe `features_final`


```python

```

## Summary
Great! You've now transformed your final data using feature scaling and normalization, and stored them in the `features_final` dataframe.
