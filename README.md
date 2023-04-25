# Ex-07-Feature-Selection
# AIM

To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation

Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM

## STEP 1

Read the given Data

## STEP 2

Clean the Data Set using Data Cleaning Process

## STEP 3

Apply Feature selection techniques to all the features of the data set

## STEP 4

Save the data to the file


# CODE

Hemasonica.P
212222230048

```
from sklearn.datasets import load_boston
boston_data=load_boston()
import pandas as pd
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV
boston.head(10)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from math import sqrt

cv = KFold(n_splits=10, random_state=None, shuffle=False)
classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))

boston.var()

X = X.drop(columns = ['NOX','CHAS'])
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))

# Filter Features by Correlation
import seaborn as sn
import matplotlib.pyplot as plt
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sn.heatmap(boston.corr(), ax=ax)
plt.show()
abs(boston.corr()["MEDV"])
abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>0.5].drop('MEDV')).index.tolist()
vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
for val in vals:
    features = abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>val].drop('MEDV')).index.tolist()
    
    X = boston.drop(columns='MEDV')
    X=X[features]
    
    print(features)

    y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
    print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
    print("R_squared: " + str(round(r2_score(y,y_pred),2)))

# Feature Selection Using a Wrapper

boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
boston['RAD'] = boston['RAD'].astype('category')
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(classifier_pipeline, 
           k_features=1, 
           forward=False, 
           scoring='neg_mean_squared_error',
           cv=cv)

X = boston.drop(columns='MEDV')
sfs1.fit(X,y)
sfs1.subsets_

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']].corr()

boston['RM*LSTAT']=boston['RM']*boston['LSTAT']

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

sn.pairplot(boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']])

boston = boston.drop(boston[boston['MEDV']==boston['MEDV'].max()].index.tolist())

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT','RM*LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston['LSTAT_2']=boston['LSTAT']**2

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

```

# OUPUT

![image](https://user-images.githubusercontent.com/118361409/234186008-80ed598b-c827-42be-8b43-f61150a31fd0.png)

![image](https://user-images.githubusercontent.com/118361409/234186059-0f81b57c-402e-43bb-9ecd-20e937f1b4e4.png)

![image](https://user-images.githubusercontent.com/118361409/234186095-657c9564-5679-40b2-8781-1ab06fdda43e.png)

![image](https://user-images.githubusercontent.com/118361409/234186198-ade66032-4e36-4b82-9b3d-d856ab509568.png)

![image](https://user-images.githubusercontent.com/118361409/234186597-408711f1-8a1f-4b1e-b010-3cb53b52809d.png)

![image](https://user-images.githubusercontent.com/118361409/234186650-32405636-40b4-4a49-b983-f7f759fc3964.png)

![image](https://user-images.githubusercontent.com/118361409/234186726-01c5a5e3-b3bb-44ba-abcd-87551c4fcb0e.png)

![image](https://user-images.githubusercontent.com/118361409/234186776-93bded25-0069-4eee-bc02-22ad2c974ed5.png)

![image](https://user-images.githubusercontent.com/118361409/234186824-e0967cfa-41a5-4606-b738-b41e84f70517.png)

![image](https://user-images.githubusercontent.com/118361409/234186856-c02329d6-30c5-4aff-b4a5-a5c5ff77936a.png)

![image](https://user-images.githubusercontent.com/118361409/234186881-231dfaed-9c96-45c5-98d5-aa5df9d611fb.png)

![image](https://user-images.githubusercontent.com/118361409/234186917-1fef7457-9186-44eb-90c9-65f62002ee80.png)

![image](https://user-images.githubusercontent.com/118361409/234186973-baf47f96-af47-40ce-bc6b-e0feb72d8978.png)

![image](https://user-images.githubusercontent.com/118361409/234187038-c8e4fb7d-433f-411b-987b-14272d5e9802.png)

![image](https://user-images.githubusercontent.com/118361409/234187094-d4f9e6ec-e148-471d-b8ad-fc4d90227fa3.png)

![image](https://user-images.githubusercontent.com/118361409/234187127-8e0e6544-6285-421a-8f2c-691005b618db.png)

 # RESULT

The various feature selection techniques has been performed on a dataset and saved the data to a file.
